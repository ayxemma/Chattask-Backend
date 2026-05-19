import json
import logging
import re
from typing import Any, Optional

import httpx

from app.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_PARSE_MODEL,
    OPENAI_TRANSCRIBE_MODEL,
)
from app.llm_clients.parse_normalization import (
    coerce_alert_style,
    coerce_optional_float,
    coerce_optional_str,
    normalize_action_type,
    parse_response_from_llm_dict,
)
from app.models.parse_models import CommandInterpretResponse, TaskTargetResolveResponse

logger = logging.getLogger(__name__)

# Backward-compatible aliases for tests.
_normalize_action_type = normalize_action_type
_parse_response_from_llm_dict = parse_response_from_llm_dict


TASK_TARGET_RESOLVER_PROMPT = """You are selecting which existing task the user wants to edit/delete/reschedule.
Choose from candidates only. Do not invent tasks.

Return exactly one JSON object with:
- "resolution": "resolved" | "needs_confirmation" | "ambiguous" | "no_match"
- "selected_id": candidate id or null
- "confidence": number from 0.0 to 1.0
- "reason": short explanation
- "candidates": array of candidate ids for ambiguous choices, or null

Rules:
- If user says "this", "it", "这个", "這個", "它" and active_task_id is provided, prefer the active task.
- If user names a task title, choose the candidate with the closest semantic/title match.
- Understand Chinese speech recognition variants and near-homophones.
- Understand partial references: "接阿里" may refer to "去接阿瑞".
- Use target_time and candidate scheduled_at/recurrence_label when provided.
- Use recurrence info if the user mentions repeat/weekly/weekday language.
- Never return "resolved" unless reasonably confident.
- Use "needs_confirmation" for a likely candidate with medium confidence.
- Use "ambiguous" when multiple candidates are plausible; return the best 2-3 ids in candidates.
- Use "no_match" if no provided candidate fits.
- selected_id and candidates must only contain ids from the provided candidates.

Respond with valid JSON only."""


COMMAND_INTERPRETER_PROMPT = """You are ChatTask's one-shot command interpreter.
Interpret the user's command, the intended edit target, edit/create fields, confidence, and the short assistant response.
Return exactly one JSON object matching the schema below. No markdown.

Schema:
{
  "action_type": "createReminder" | "createEvent" | "rescheduleTask" | "renameTask" | "appendToTask" | "deleteTask" | "updateRecurrence" | "updateAlertStyle" | "unknown",
  "confidence": 0.0,
  "requires_confirmation": true,
  "confirmation_kind": "none" | "confirm_action" | "choose_candidate" | "clarify",
  "assistant_message": "...",
  "target": {
    "resolution": "active_task" | "candidate" | "ambiguous" | "none",
    "selected_task_id": null,
    "selected_task_title": null,
    "candidate_ids": null,
    "reason": "..."
  },
  "create": {
    "title": null,
    "notes": null,
    "scheduled_at": null,
    "end_at": null,
    "has_specific_time": null,
    "recurrence_type": "none",
    "recurrence_weekdays": null,
    "recurrence_end_at": null,
    "alert_style": null
  },
  "edit": {
    "new_title": null,
    "new_scheduled_at": null,
    "append_text": null,
    "new_recurrence_type": null,
    "new_recurrence_weekdays": null,
    "alert_style": null,
    "apply_scope": "single"
  }
}

Rules:
- Use create.* only for createReminder/createEvent.
- Use target.* and edit.* only for edit actions.
- Do not mix target title and new title. new_title is only for explicit rename/title changes.
- "改成 + time", "改到 + time", "换到 + time" means rescheduleTask, NOT renameTask.
  Examples: "把这个改成四点半", "把给艾瑞喂水果酸奶的任务改成四点半", "改到下午四点半", "换到明天九点".
  Set edit.new_scheduled_at and keep edit.new_title null.
- Rename only when user explicitly says 改名, 名字改成, 标题改成, rename, or change the name/title to.
- If user says this/it/这个/这个任务/它 and active_task is present, use target.resolution="active_task" and active_task.id.
- If user names a task, choose from candidate_tasks only. Understand Chinese ASR variants, partial references, and semantic matches:
  "接阿里" may refer to "去接阿瑞"; "水果酸奶" may refer to "给艾瑞喂芒果酸奶".
- If uncertain but one likely candidate exists, use requires_confirmation=true, confirmation_kind="confirm_action".
- If multiple candidates are plausible, use target.resolution="ambiguous", confirmation_kind="choose_candidate", and return top 2-3 candidate_ids.
- Never invent a task ID. IDs must come from active_task or candidate_tasks.
- If required fields are missing or intent is unclear, action_type="unknown", confirmation_kind="clarify".
- Delete should usually require confirmation unless confidence is very high and active_task was explicitly referenced.
- For weekly recurrence creates, set create.recurrence_type="weekly" and ISO weekdays Monday=1...Sunday=7.
- For alert style, use only "silent", "default", or "important". Map 静音/silent/no sound/不要声音提醒 to "silent"; normal/default/普通提醒 to "default"; important/loud/alarm-like/重要提醒/明显一点/声音大一点 to "important".
- For create commands, put alert style in create.alert_style when specified.
- For existing task alert edits such as "把这个任务改成重要提醒", use action_type="updateAlertStyle" and put the value in edit.alert_style.
- Do not call Important a true alarm or imply it can bypass Silent Mode or Focus.
- Always resolve relative times from Current time and Timezone. Use ISO 8601 datetimes with timezone offset.

Respond with valid JSON only."""


def _auth_headers() -> dict[str, str]:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")
    return {"Authorization": f"Bearer {OPENAI_API_KEY}"}


def _coerce_resolution(raw: Optional[Any]) -> str:
    if raw is None:
        return "no_match"
    s = str(raw).strip().lower()
    mapping = {
        "resolved": "resolved",
        "needsconfirmation": "needs_confirmation",
        "needs_confirmation": "needs_confirmation",
        "confirm": "needs_confirmation",
        "confirmation": "needs_confirmation",
        "ambiguous": "ambiguous",
        "disambiguate": "ambiguous",
        "no_match": "no_match",
        "nomatch": "no_match",
        "none": "no_match",
    }
    return mapping.get(re.sub(r"[\s-]+", "_", s), mapping.get(re.sub(r"[_\s-]+", "", s), "no_match"))


def _parse_task_target_resolution(data: dict[str, Any], allowed_ids: set[str]) -> TaskTargetResolveResponse:
    resolution = _coerce_resolution(data.get("resolution"))
    selected_id = coerce_optional_str(data.get("selected_id"))
    if selected_id not in allowed_ids:
        selected_id = None

    candidate_ids: list[str] = []
    raw_candidates = data.get("candidates")
    if isinstance(raw_candidates, list):
        for item in raw_candidates:
            cid = coerce_optional_str(item)
            if cid in allowed_ids and cid not in candidate_ids:
                candidate_ids.append(cid)

    confidence = coerce_optional_float(data.get("confidence")) or 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = coerce_optional_str(data.get("reason"))

    if resolution in {"resolved", "needs_confirmation"} and not selected_id:
        resolution = "no_match"
    if resolution == "ambiguous" and not candidate_ids:
        if selected_id:
            resolution = "needs_confirmation"
        else:
            resolution = "no_match"

    return TaskTargetResolveResponse(
        resolution=resolution,
        selected_id=selected_id,
        confidence=confidence,
        reason=reason,
        candidates=candidate_ids or None,
    )


def _normalize_interpret_action(raw: Optional[Any]) -> str:
    if raw is None:
        return "unknown"
    collapsed = re.sub(r"[_\s-]+", "", str(raw).strip()).lower()
    mapping = {
        "createreminder": "createReminder",
        "reminder": "createReminder",
        "createevent": "createEvent",
        "calendarevent": "createEvent",
        "rescheduletask": "rescheduleTask",
        "renametask": "renameTask",
        "updatetasktitle": "renameTask",
        "appendtotask": "appendToTask",
        "deletetask": "deleteTask",
        "updaterecurrence": "updateRecurrence",
        "updatealertstyle": "updateAlertStyle",
        "editalertstyle": "updateAlertStyle",
        "changealertstyle": "updateAlertStyle",
        "setalertstyle": "updateAlertStyle",
        "unknown": "unknown",
    }
    return mapping.get(collapsed, "unknown")


def _sanitize_interpret_response(data: dict[str, Any], allowed_ids: set[str], active_id: Optional[str]) -> CommandInterpretResponse:
    response = CommandInterpretResponse.model_validate(data)
    response.action_type = _normalize_interpret_action(response.action_type)
    response.confidence = max(0.0, min(1.0, response.confidence or 0.0))

    if response.target:
        if response.target.selected_task_id not in allowed_ids and response.target.selected_task_id != active_id:
            response.target.selected_task_id = None
            response.target.selected_task_title = None
            if response.target.resolution in {"active_task", "candidate"}:
                response.target.resolution = "none"
        if response.target.candidate_ids:
            response.target.candidate_ids = [
                cid for cid in response.target.candidate_ids
                if cid in allowed_ids or cid == active_id
            ][:3]

    if response.action_type == "renameTask" and response.edit and response.edit.new_scheduled_at and not response.edit.new_title:
        logger.warning("interpretWarnings renameTask has new_scheduled_at but no new_title; converting to rescheduleTask")
        response.action_type = "rescheduleTask"
    if response.action_type == "rescheduleTask" and response.edit:
        response.edit.new_title = None

    if response.create:
        response.create.alert_style = coerce_alert_style(response.create.alert_style)
    if response.edit:
        response.edit.alert_style = coerce_alert_style(response.edit.alert_style)

    if response.confirmation_kind is None:
        response.confirmation_kind = "none" if not response.requires_confirmation else "confirm_action"
    return response


async def transcribe_audio(file_bytes: bytes, filename: str, content_type: str) -> str:
    """Send audio bytes to OpenAI's audio transcription endpoint and return the transcript text."""
    headers = _auth_headers()

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{OPENAI_BASE_URL}/audio/transcriptions",
            headers=headers,
            files={"file": (filename, file_bytes, content_type)},
            data={"model": OPENAI_TRANSCRIBE_MODEL},
        )

    if response.status_code != 200:
        logger.error("OpenAI transcription error %s: %s", response.status_code, response.text)
        response.raise_for_status()

    result = response.json()
    return result.get("text", "")


async def resolve_task_target(
    *,
    user_text: str,
    action_type: str,
    target_title: Optional[str],
    target_time: Optional[str],
    candidates: list[dict[str, Any]],
    active_task_id: Optional[str],
    timezone: str,
    locale: Optional[str] = None,
) -> TaskTargetResolveResponse:
    """Ask the model to choose an edit target from a provided candidate list only."""
    headers = {**_auth_headers(), "Content-Type": "application/json"}
    allowed_ids = {str(candidate.get("id")) for candidate in candidates if candidate.get("id")}
    if not allowed_ids:
        return TaskTargetResolveResponse(resolution="no_match", confidence=0.0, reason="no candidates")

    user_payload = {
        "user_text": user_text,
        "action_type": action_type,
        "target_title": target_title,
        "target_time": target_time,
        "candidates": candidates,
        "active_task_id": active_task_id,
        "timezone": timezone,
        "locale": locale,
    }

    payload = {
        "model": OPENAI_PARSE_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": TASK_TARGET_RESOLVER_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )

    if response.status_code != 200:
        logger.error("OpenAI task target resolver error %s: %s", response.status_code, response.text)
        response.raise_for_status()

    raw_content = response.json()["choices"][0]["message"]["content"]
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        logger.error("Failed to decode task target resolver JSON: %s\nRaw: %s", e, raw_content)
        raise ValueError(f"Model returned invalid JSON: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("Model returned JSON that is not an object")

    return _parse_task_target_resolution(data, allowed_ids)


async def interpret_command(
    *,
    text: str,
    now: str,
    timezone: str,
    locale: Optional[str],
    active_task: Optional[dict[str, Any]],
    candidate_tasks: list[dict[str, Any]],
    request_id: Optional[str] = None,
) -> CommandInterpretResponse:
    headers = {**_auth_headers(), "Content-Type": "application/json"}
    active_id = coerce_optional_str((active_task or {}).get("id"))
    allowed_ids = {str(candidate.get("id")) for candidate in candidate_tasks if candidate.get("id")}
    if active_id:
        allowed_ids.add(active_id)
    logger.info(
        "interpretRequest request_id=%s text=%s candidate_count=%s active_task_present=%s",
        request_id,
        text,
        len(candidate_tasks),
        bool(active_task),
    )
    user_payload = {
        "text": text,
        "current_time": now,
        "timezone": timezone,
        "locale": locale,
        "active_task": active_task,
        "candidate_tasks": candidate_tasks,
    }
    payload = {
        "model": OPENAI_PARSE_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": COMMAND_INTERPRETER_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
    if response.status_code != 200:
        logger.error("OpenAI command interpretation error %s: %s", response.status_code, response.text)
        response.raise_for_status()
    raw_content = response.json()["choices"][0]["message"]["content"]
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        logger.error("Failed to decode command interpretation JSON: %s\nRaw: %s", e, raw_content)
        raise ValueError(f"Model returned invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("Model returned JSON that is not an object")
    result = _sanitize_interpret_response(data, allowed_ids, active_id)
    logger.info(
        "interpretResult action_type=%s confidence=%s requires_confirmation=%s target_resolution=%s selected_id=%s",
        result.action_type,
        result.confidence,
        result.requires_confirmation,
        result.target.resolution if result.target else None,
        result.target.selected_task_id if result.target else None,
    )
    return result
