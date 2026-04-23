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
from app.models.parse_models import ParseResponse

logger = logging.getLogger(__name__)

# Canonical parse instructions for the ChatTask iOS client. Output keys use snake_case to match `LLMTaskParseResponse`.
PARSE_SYSTEM_PROMPT = """You are a task parsing assistant for a productivity app. Given natural language (typed or transcribed speech), emit exactly one JSON object — no markdown, no explanation, no text outside the JSON.

## Output schema (all keys required; use null where not applicable)
- "action_type": string — one of the allowed values below (use these exact spellings).
- "title": string or null — short, human-readable title; for edit commands may be empty or a short summary.
- "notes": string or null — extra detail not in the title.
- "scheduled_at": string or null — ISO 8601 datetime with timezone offset when a start/reminder/fire time applies; null if not applicable.
- "end_at": string or null — ISO 8601 end instant for calendar-style blocks when the user gives a duration or explicit end; null otherwise.
- "has_specific_time": boolean or null — true if the user expressed a concrete clock time or relative delay (e.g. "at 3pm", "in 20 minutes"); false for date-only phrasing (e.g. "tomorrow" with no time, all-day style); null only if there is no date/time at all.
- "confidence": number or null — 0.0–1.0 parse confidence; null if unsure.
- "language_code": string or null — ISO 639-1 (or best guess) for the input text.
- "target_time": string or null — for edit commands only: ISO 8601 instant identifying which existing item to change (usually the task's current scheduled time the user refers to).
- "new_scheduled_at": string or null — for rescheduleTask only: ISO 8601 new scheduled instant.
- "append_text": string or null — for appendToTask only: text to add to notes (no need to repeat existing content).
- "new_title": string or null — for updateTaskTitle only: the new task title.

## Allowed action_type values (exact strings)
- "reminder": Time-based reminder / todo with a notify time. Use scheduled_at as the reminder fire time. Prefer this for simple to-dos, alarms, "remind me to…", and relative delays ("in 5 minutes…") when the user is not describing a calendar meeting/block.
- "calendarEvent": Calendar entry with a definite time window. Use scheduled_at as start; set end_at when the user gives an end time or a duration you can convert to an end. Prefer for meetings, appointments, "block from X to Y", events with location/attendees flavor.
- "unknown": Intent is ambiguous or unsupported; set minimal fields and low/null confidence.
- "deleteTask": User wants to remove/cancel an existing item. Set target_time to the referenced schedule instant if inferable; title may briefly restate what to delete.
- "rescheduleTask": User moves an existing item to a new time. Set target_time (old) and new_scheduled_at (new). Both should include timezone offsets consistent with the provided timezone.
- "appendToTask": User adds a note to an existing item. Set target_time if inferable and append_text to the new fragment only.
- "updateTaskTitle": User renames the task. Set new_title to the full new title; target_time may be the active task's scheduled instant when disambiguating.

## Follow-up vs new task (when "Active task context" appears in the user message)
- The client may send a snapshot of the task the user just created or edited. The user may follow up with short commands: "delete that", "make it tomorrow", "also add …", "change the title to …", "30 minutes instead", etc. Prefer resolving these against that active task when the wording clearly refers to it (pronouns, "that", "this task", incremental edits).
- If the message is clearly a **new standalone** task (different subject and intent, e.g. first message was "remind me to cook dinner at 6" and the next is "buy milk tomorrow"), emit a **create** action (reminder or calendarEvent) with no edit action_type — do not bind to the previous task.
- When using an edit action and the active task has a scheduled time, set target_time to that instant (with offset) if the user did not specify another time anchor.

## Time rules
- Use the "Current time" and "Timezone" from the user message to resolve relative phrases ("today", "tomorrow", "in 2 hours", "next Friday").
- Always express scheduled_at, end_at, target_time, and new_scheduled_at with explicit timezone offsets (never zone-less local strings).
- If no time is given and the phrase is purely informational, scheduled_at may be null and action_type may be unknown or reminder without a time depending on intent.
- Speech transcripts may drop small words; tolerate ASR errors but do not invent specific times.

## Title and notes
- For short task phrases, keep the full wording in "title"; use null for "notes" when a single short phrase is enough.

## ASR / transcription
- Input may be speech: connectors like "in", "after", or "后" may be missing. Tolerate imperfect grammar; infer only missing glue words when intent is clear. Do not invent times when the phrase is genuinely ambiguous.

## Duration-first phrases (relative to Current time)
- If the text begins with a duration in minutes or hours, then states a task, treat as a relative-time "reminder" from Current time: set scheduled_at ≈ now + that duration, has_specific_time true, and put the task in title.

Examples (scheduled_at relative to Current time):
- "20 minutes, take a walk" → reminder; title reflects the walk; scheduled_at ≈ now+20m.
- "five minutes call John" → reminder; title for calling John; scheduled_at ≈ now+5m.
- "二十分钟去散步" → reminder; title e.g. 去散步; scheduled_at ≈ now+20m.
- "两小时接孩子" → reminder; title e.g. 接孩子; scheduled_at ≈ now+2h.

## Reminder vs calendarEvent
- If the user describes something that sounds like a timed to-do or nudge, use reminder.
- If they describe a scheduled block, meeting, or explicit start/end window, use calendarEvent.

Respond with valid JSON only matching the schema."""


def _auth_headers() -> dict[str, str]:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")
    return {"Authorization": f"Bearer {OPENAI_API_KEY}"}


def _normalize_action_type(raw: Optional[Any]) -> str:
    """Map model output to iOS ActionType raw strings (camelCase)."""
    if raw is None:
        return "unknown"
    s = str(raw).strip()
    if not s:
        return "unknown"
    collapsed = re.sub(r"[_\s]+", "", s).lower()
    # Keys are underscore/space-insensitive forms; values match Swift `ActionType` rawValues.
    to_canonical = {
        "reminder": "reminder",
        "unknown": "unknown",
        "calendarevent": "calendarEvent",
        "event": "calendarEvent",
        "task": "reminder",
        "todo": "reminder",
        "deletetask": "deleteTask",
        "rescheduletask": "rescheduleTask",
        "appendtotask": "appendToTask",
        "updatetasktitle": "updateTaskTitle",
        "renametask": "updateTaskTitle",
        "edittitle": "updateTaskTitle",
    }
    return to_canonical.get(collapsed, "unknown")


def _coerce_optional_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        t = v.strip()
        return t if t else None
    return str(v)


def _coerce_optional_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return None


def _coerce_optional_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_response_from_llm_dict(data: dict[str, Any]) -> ParseResponse:
    """Build ParseResponse from raw JSON object; tolerate minor type drift from the model."""
    return ParseResponse(
        action_type=_normalize_action_type(data.get("action_type")),
        title=_coerce_optional_str(data.get("title")),
        notes=_coerce_optional_str(data.get("notes")),
        scheduled_at=_coerce_optional_str(data.get("scheduled_at")),
        end_at=_coerce_optional_str(data.get("end_at")),
        has_specific_time=_coerce_optional_bool(data.get("has_specific_time")),
        language_code=_coerce_optional_str(data.get("language_code")),
        confidence=_coerce_optional_float(data.get("confidence")),
        target_time=_coerce_optional_str(data.get("target_time")),
        new_scheduled_at=_coerce_optional_str(data.get("new_scheduled_at")),
        append_text=_coerce_optional_str(data.get("append_text")),
        new_title=_coerce_optional_str(data.get("new_title")),
    )


async def transcribe_audio(file_bytes: bytes, filename: str, content_type: str) -> str:
    """
    Send audio bytes to OpenAI's audio transcription endpoint and return the transcript text.
    """
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


def _truncate_notes(s: Optional[str], max_len: int = 1200) -> Optional[str]:
    if s is None:
        return None
    t = s.strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


async def parse_task_text(
    *,
    text: str,
    now: str,
    timezone: str,
    locale: Optional[str] = None,
    parse_instructions: Optional[str] = None,
    source: Optional[str] = None,
    last_active_task_id: Optional[str] = None,
    active_task_title: Optional[str] = None,
    active_task_scheduled_at: Optional[str] = None,
    active_task_notes: Optional[str] = None,
) -> ParseResponse:
    """
    Send task text to OpenAI chat completions and return structured ParseResponse (iOS contract).
    """
    headers = {**_auth_headers(), "Content-Type": "application/json"}

    system_content = PARSE_SYSTEM_PROMPT
    if parse_instructions and parse_instructions.strip():
        # Client hints are additive; the block above remains authoritative.
        system_content = (
            f"{PARSE_SYSTEM_PROMPT}\n\n---\nClient parsing hints (follow when consistent with rules above):\n"
            f"{parse_instructions.strip()}"
        )

    user_lines = [
        f'Parse this task:\n\nText: "{text}"',
        f"Current time (ISO 8601): {now}",
        f"Timezone (IANA): {timezone}",
    ]
    if locale:
        user_lines.append(f"Locale: {locale}")
    if source:
        user_lines.append(f"Input source: {source}")
    if last_active_task_id and str(last_active_task_id).strip():
        user_lines.append("")
        user_lines.append("Active task context (for follow-up commands; user may refer to this task without naming it):")
        user_lines.append(f"- task_id: {last_active_task_id.strip()}")
        if active_task_title:
            user_lines.append(f'- title: "{active_task_title.strip()}"')
        if active_task_scheduled_at:
            user_lines.append(f"- scheduled_at (ISO 8601): {active_task_scheduled_at.strip()}")
        notes_ctx = _truncate_notes(active_task_notes)
        if notes_ctx:
            user_lines.append(f"- notes (may be truncated): {notes_ctx}")
    user_message = "\n".join(user_lines)

    payload = {
        "model": OPENAI_PARSE_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_message},
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
        logger.error("OpenAI parse error %s: %s", response.status_code, response.text)
        response.raise_for_status()

    raw_content = response.json()["choices"][0]["message"]["content"]

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        logger.error("Failed to decode OpenAI JSON response: %s\nRaw: %s", e, raw_content)
        raise ValueError(f"Model returned invalid JSON: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("Model returned JSON that is not an object")

    return _parse_response_from_llm_dict(data)
