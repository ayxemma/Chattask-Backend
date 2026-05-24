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
from app.models.parse_models import CommandInterpretAction, CommandInterpretResponse, ParseResponse, TaskTargetResolveResponse

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
- "recurrence": object or null — weekly recurrence metadata for recurring creates. Shape: {"frequency":"weekly","weekdays":[1..7],"time":"HH:mm","timezone":"IANA zone","start_date":"YYYY-MM-DD or ISO datetime","end_date":null or date}. Use ISO weekdays where Monday=1 and Sunday=7.
- "alert_style": "normal" | "important" | null — user-facing reminder alert type.
- "target_time": string or null — for edit commands only: ISO 8601 instant identifying which existing item to change (usually the task's current scheduled time the user refers to).
- "new_scheduled_at": string or null — for rescheduleTask only: ISO 8601 new scheduled instant.
- "append_text": string or null — for appendToTask only: text to add to notes (no need to repeat existing content).
- "new_title": string or null — for updateTaskTitle only: the new task title.
- "target_reference_type": string or null — for edit/follow-up commands: "task_id", "recent_task", "time", or "title".
- "target_task_id": string or null — exact active task_id when target_reference_type is "task_id" or "recent_task".
- "recurrence_update": object or null — for updateRecurrence only. Shape: {"operation":"set_weekdays|add_weekdays|remove_weekdays|set_time|clear_recurrence","weekdays":[1..7] or null,"time":"HH:mm" or null,"timezone":"IANA zone" or null,"start_date":null,"end_date":null}.

## Allowed action_type values (exact strings)
- "reminder": Time-based reminder / todo with a notify time. Use scheduled_at as the reminder fire time. Prefer this for simple to-dos, alarms, "remind me to…", and relative delays ("in 5 minutes…") when the user is not describing a calendar meeting/block.
- "calendarEvent": Calendar entry with a definite time window. Use scheduled_at as start; set end_at when the user gives an end time or a duration you can convert to an end. Prefer for meetings, appointments, "block from X to Y", events with location/attendees flavor.
- "unknown": Intent is ambiguous or unsupported; set minimal fields and low/null confidence.
- "deleteTask": User wants to remove/cancel an existing item. Set target_time to the referenced schedule instant if inferable; title may briefly restate what to delete.
- "rescheduleTask": User moves an existing item to a new time. Set target_time (old) and new_scheduled_at (new). Both should include timezone offsets consistent with the provided timezone.
- "appendToTask": User adds a note to an existing item. Set target_time if inferable and append_text to the new fragment only.
- "updateTaskTitle": User renames the task. Set new_title to the full new title; target_time may be the active task's scheduled instant when disambiguating.
- "updateRecurrence": User changes recurrence for an existing item. Set recurrence_update and target_reference_type/target_task_id when using active context.
- "updateAlertStyle": User changes an existing task's reminder alert style. Set alert_style and target_reference_type/target_task_id when using active context.

## Conversation context
- If last_active_task_id is provided, use it only for clear follow-up edits. Do not force all new messages onto that task.
- If the user requests a separate new task/reminder, return a create action (reminder or calendarEvent) and ignore the active task context.
- For edit actions resolved to the active task, set target_reference_type to "recent_task" and target_task_id to the active task_id.
- When using an edit action and the active task has a scheduled time, set target_time to that instant (with offset) if the user did not specify another time anchor.
- If active recurrence context is provided, use it to interpret recurrence follow-ups such as removing a weekday.

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

## Recurrence
- Support weekly recurrence phrases including Chinese weekday ranges. Example: "每周一到周五 11:50 提醒我去接阿瑞" means recurrence.frequency="weekly", recurrence.weekdays=[1,2,3,4,5], recurrence.time="11:50", and title="去接阿瑞".
- For recurring creates, also set scheduled_at to the next occurrence datetime so older clients can still create a one-off reminder.
- Do not place recurrence words such as "每周一到周五" in the title.
- For active recurring-task follow-up edits:
  - "把周四去掉" means action_type="updateRecurrence", target_reference_type="recent_task", target_task_id=last_active_task_id, recurrence_update.operation="remove_weekdays", recurrence_update.weekdays=[4].
  - "改成周一到周三" means action_type="updateRecurrence", target_reference_type="recent_task", target_task_id=last_active_task_id, recurrence_update.operation="set_weekdays", recurrence_update.weekdays=[1,2,3].
  - "不要周五提醒了" means action_type="updateRecurrence", target_reference_type="recent_task", target_task_id=last_active_task_id, recurrence_update.operation="remove_weekdays", recurrence_update.weekdays=[5].

## Alert style
- Support only two product-facing reminder alert types: "normal", "important".
- Do not describe this as a true alarm, Critical Alert, or something that bypasses Silent Mode or Focus.
- Map normal / default / silent / 普通提醒 / 静音 / no sound to alert_style="normal".
- Map important / loud / 重要提醒 / 明显一点 / 声音大一点 to alert_style="important".
- For creates, set alert_style when the user specifies one; otherwise null.
- For existing task edits such as "把这个任务改成重要提醒", use action_type="updateAlertStyle" and set alert_style.

## Reminder vs calendarEvent
- If the user describes something that sounds like a timed to-do or nudge, use reminder.
- If they describe a scheduled block, meeting, or explicit start/end window, use calendarEvent.

Respond with valid JSON only matching the schema."""


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
Interpret the user's command, intended edit target, edit/create fields, confidence, and short assistant response.
Return exactly one JSON object matching the schema below. No markdown.

Preserve the existing single-action contract:
- For a single command, fill the top-level fields exactly as before. You may omit actions or return actions with one matching item.
- For multiple independent actions in one sentence, return actions with every action in execution order. Also mirror the first action into the top-level fields for backward compatibility.
- Do not split by keywords mechanically. Decide semantically whether the user means separate actions or one edit.
- Do not discard trailing details after an edit. If the user edits the active task and then adds a related detail about what they will do/bring/take/remember, return a second appendToTask action unless they explicitly ask for a separate reminder/task.
- Do not copy an edit time onto a separate new reminder unless the user explicitly gives that time for the reminder. "Change this to tomorrow at 11 and remind me to buy vegetables" creates an untimed buy-vegetables reminder.
- If a trailing bare phrase after an edit could be either a note or a new reminder, ask for clarification instead of guessing.
- If the multi-action plan is uncertain, set requires_confirmation=true on the uncertain action(s) or use a clarifying assistant_message.
- If you cannot produce a valid multi-action plan, return the best single-action interpretation so the original parser can handle fallback safely.

Schema:
{
  "actions": [
    {
      "action_type": "createReminder" | "createEvent" | "rescheduleTask" | "renameTask" | "appendToTask" | "deleteTask" | "updateRecurrence" | "updateAlertStyle" | "unknown",
      "confidence": 0.0,
      "requires_confirmation": false,
      "confirmation_kind": "none" | "confirm_action" | "choose_candidate" | "clarify",
      "assistant_message": "...",
      "target": {"resolution": "active_task" | "candidate" | "ambiguous" | "none", "selected_task_id": null, "selected_task_title": null, "candidate_ids": null, "reason": "..."},
      "create": {"title": null, "notes": null, "scheduled_at": null, "end_at": null, "has_specific_time": null, "recurrence_type": "none", "recurrence_weekdays": null, "recurrence_end_at": null, "alert_style": null},
      "edit": {"new_title": null, "new_scheduled_at": null, "append_text": null, "new_recurrence_type": null, "new_recurrence_weekdays": null, "alert_style": null, "apply_scope": "single"}
    }
  ],
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
- Use actions only for execution plans. One action per separate create/edit/delete operation.
- Examples:
  "Remind me tomorrow at 11 to call mom and also remind me to buy tomatoes." => two createReminder actions.
  "Change this to tomorrow at 11 and add a note to buy vegetables." => reschedule active task, then appendToTask on the same task.
  "Change this to tomorrow at 11 and remind me to buy vegetables." => reschedule active task, then createReminder for buy vegetables with scheduled_at null.
  "Change this to tomorrow at 11 and buy vegetables." => requires_confirmation true, confirmation_kind "clarify", ask whether to add it as a note or create a separate reminder.
  "Change it to 11:30pm and I'll bring my cup to my room." => reschedule active task, then appendToTask with "bring my cup to my room".
  "Change it to 11:30pm and remember to bring my cup to my room." => reschedule active task, then appendToTask with "bring my cup to my room".
  "把这个改到明天十一点，备注里加上买菜西红柿土豆" => reschedule active task, then appendToTask.
  "把这个改到明天十一点，买菜西红柿土豆也提醒我" => reschedule active task, then createReminder with scheduled_at null unless a separate time is given.
  "把这个改到明天十一点，买菜" => requires_confirmation true, confirmation_kind "clarify", ask whether to add it as a note or create a separate reminder.
  "九点给艾瑞做早餐，十一点给艾瑞接回家。" => two createReminder actions, one at 09:00 and one at 11:00.
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
- For alert style, use only "normal" or "important". Map silent/default/普通/静音/no sound to "normal"; important/loud/重要提醒/明显一点/声音大一点 to "important".
- For create commands, put alert style in create.alert_style when specified.
- For existing task alert edits such as "把这个任务改成重要提醒", use action_type="updateAlertStyle" and put the value in edit.alert_style.
- Do not call Important a true alarm or imply it can bypass Silent Mode or Focus.
- Always resolve relative times from Current time and Timezone. Use ISO 8601 datetimes with timezone offset.

Respond with valid JSON only."""


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
        "updaterecurrence": "updateRecurrence",
        "editrecurrence": "updateRecurrence",
        "changerecurrence": "updateRecurrence",
        "updatealertstyle": "updateAlertStyle",
        "editalertstyle": "updateAlertStyle",
        "changealertstyle": "updateAlertStyle",
        "setalertstyle": "updateAlertStyle",
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


def _coerce_alert_style(v: Any) -> Optional[str]:
    raw = _coerce_optional_str(v)
    if not raw:
        return None
    collapsed = re.sub(r"[_\s-]+", "", raw).lower()
    mapping = {
        "silent": "normal",
        "quiet": "normal",
        "nosound": "normal",
        "mute": "normal",
        "muted": "normal",
        "静音": "normal",
        "不要声音": "normal",
        "default": "normal",
        "normal": "normal",
        "standard": "normal",
        "普通": "normal",
        "普通提醒": "normal",
        "sound": "normal",
        "soundonly": "normal",
        "vibration": "normal",
        "vibrate": "normal",
        "vibrationonly": "normal",
        "vibrateonly": "normal",
        "important": "important",
        "loud": "important",
        "strong": "important",
        "alarmlike": "important",
        "soundandvibration": "important",
        "soundvibration": "important",
        "重要": "important",
        "重要提醒": "important",
        "明显一点": "important",
        "声音大一点": "important",
    }
    return mapping.get(collapsed)


def _sanitize_weekdays(v: Any) -> Optional[list[int]]:
    if not isinstance(v, list):
        return None
    days: list[int] = []
    for item in v:
        try:
            day = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= day <= 7 and day not in days:
            days.append(day)
    return sorted(days) if days else None


def _coerce_recurrence(v: Any) -> Optional[dict[str, Any]]:
    if not isinstance(v, dict):
        return None
    recurrence = {
        "frequency": _coerce_optional_str(v.get("frequency")),
        "weekdays": _sanitize_weekdays(v.get("weekdays")),
        "time": _coerce_optional_str(v.get("time")),
        "timezone": _coerce_optional_str(v.get("timezone")),
        "start_date": _coerce_optional_str(v.get("start_date")),
        "end_date": _coerce_optional_str(v.get("end_date")),
    }
    return recurrence if any(value is not None for value in recurrence.values()) else None


def _coerce_recurrence_update(v: Any) -> Optional[dict[str, Any]]:
    if not isinstance(v, dict):
        return None
    update = {
        "operation": _coerce_optional_str(v.get("operation")),
        "weekdays": _sanitize_weekdays(v.get("weekdays")),
        "time": _coerce_optional_str(v.get("time")),
        "timezone": _coerce_optional_str(v.get("timezone")),
        "start_date": _coerce_optional_str(v.get("start_date")),
        "end_date": _coerce_optional_str(v.get("end_date")),
    }
    return update if any(value is not None for value in update.values()) else None


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
        recurrence=_coerce_recurrence(data.get("recurrence")),
        alert_style=_coerce_alert_style(data.get("alert_style")),
        target_time=_coerce_optional_str(data.get("target_time")),
        new_scheduled_at=_coerce_optional_str(data.get("new_scheduled_at")),
        append_text=_coerce_optional_str(data.get("append_text")),
        new_title=_coerce_optional_str(data.get("new_title")),
        target_reference_type=_coerce_optional_str(data.get("target_reference_type")),
        target_task_id=_coerce_optional_str(data.get("target_task_id")),
        recurrence_update=_coerce_recurrence_update(data.get("recurrence_update")),
    )


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
    selected_id = _coerce_optional_str(data.get("selected_id"))
    if selected_id not in allowed_ids:
        selected_id = None

    candidate_ids: list[str] = []
    raw_candidates = data.get("candidates")
    if isinstance(raw_candidates, list):
        for item in raw_candidates:
            cid = _coerce_optional_str(item)
            if cid in allowed_ids and cid not in candidate_ids:
                candidate_ids.append(cid)

    confidence = _coerce_optional_float(data.get("confidence")) or 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = _coerce_optional_str(data.get("reason"))

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


def _sanitize_interpret_action(
    action: CommandInterpretAction,
    allowed_ids: set[str],
    active_id: Optional[str],
) -> CommandInterpretAction:
    action.action_type = _normalize_interpret_action(action.action_type)
    action.confidence = max(0.0, min(1.0, action.confidence or 0.0))

    if action.target:
        if action.target.selected_task_id not in allowed_ids and action.target.selected_task_id != active_id:
            action.target.selected_task_id = None
            action.target.selected_task_title = None
            if action.target.resolution in {"active_task", "candidate"}:
                action.target.resolution = "none"
        if action.target.candidate_ids:
            action.target.candidate_ids = [
                cid for cid in action.target.candidate_ids
                if cid in allowed_ids or cid == active_id
            ][:3]

    if action.action_type == "renameTask" and action.edit and action.edit.new_scheduled_at and not action.edit.new_title:
        logger.warning("interpretWarnings renameTask has new_scheduled_at but no new_title; converting to rescheduleTask")
        action.action_type = "rescheduleTask"
    if action.action_type == "rescheduleTask" and action.edit:
        action.edit.new_title = None

    if action.create:
        action.create.alert_style = _coerce_alert_style(action.create.alert_style)
    if action.edit:
        action.edit.alert_style = _coerce_alert_style(action.edit.alert_style)

    if action.confirmation_kind is None:
        action.confirmation_kind = "none" if not action.requires_confirmation else "confirm_action"
    return action


def _expand_compound_interpret_action(action: CommandInterpretAction) -> list[CommandInterpretAction]:
    """Split model outputs that packed two edit fields into one action.

    This keeps semantic interpretation in the LLM while making the structured
    response executable. For example, a reschedule action that also has
    edit.append_text should execute as reschedule + append note.
    """
    if (
        action.action_type == "rescheduleTask"
        and action.edit
        and action.edit.new_scheduled_at
        and action.edit.append_text
    ):
        append_edit = action.edit.model_copy()
        append_edit.new_scheduled_at = None
        reschedule_edit = action.edit.model_copy()
        reschedule_edit.append_text = None
        action.edit = reschedule_edit
        return [
            action,
            CommandInterpretAction(
                action_type="appendToTask",
                confidence=action.confidence,
                requires_confirmation=action.requires_confirmation,
                confirmation_kind=action.confirmation_kind,
                assistant_message=action.assistant_message,
                target=action.target.model_copy() if action.target else None,
                edit=append_edit,
            ),
        ]
    return [action]


def _sanitize_interpret_response(data: dict[str, Any], allowed_ids: set[str], active_id: Optional[str]) -> CommandInterpretResponse:
    response = CommandInterpretResponse.model_validate(data)
    _sanitize_interpret_action(response, allowed_ids, active_id)

    if response.actions:
        expanded_actions: list[CommandInterpretAction] = []
        for action in response.actions:
            if action.action_type is None:
                continue
            sanitized = _sanitize_interpret_action(action, allowed_ids, active_id)
            expanded_actions.extend(_expand_compound_interpret_action(sanitized))
        response.actions = expanded_actions
        if response.actions:
            first = response.actions[0]
            response.action_type = first.action_type
            response.confidence = first.confidence
            response.requires_confirmation = any(action.requires_confirmation for action in response.actions)
            response.confirmation_kind = (
                "clarify"
                if any(action.requires_confirmation and action.confirmation_kind == "clarify" for action in response.actions)
                else first.confirmation_kind
            )
            response.target = first.target
            response.create = first.create
            response.edit = first.edit
    else:
        expanded_actions = _expand_compound_interpret_action(response)
        if len(expanded_actions) > 1:
            response.actions = expanded_actions
            response.requires_confirmation = any(action.requires_confirmation for action in expanded_actions)
    return response


async def transcribe_audio(file_bytes: bytes, filename: str, content_type: str) -> tuple[str, int]:
    """
    Send audio bytes to OpenAI's audio transcription endpoint and return (transcript, retry_count).
    """
    headers = _auth_headers()
    retries = 0

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{OPENAI_BASE_URL}/audio/transcriptions",
            headers=headers,
            files={"file": (filename, file_bytes, content_type)},
            data={"model": OPENAI_TRANSCRIBE_MODEL},
        )

    if response.status_code != 200:
        logger.error("OpenAI transcription error %s: %s model=%s", response.status_code, response.text, OPENAI_TRANSCRIBE_MODEL)
        response.raise_for_status()

    result = response.json()
    return result.get("text", ""), retries


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
    active_task_recurrence: Optional[dict[str, Any]] = None,
    request_id: Optional[str] = None,
    command_session_id: Optional[str] = None,
) -> ParseResponse:
    """
    Send task text to OpenAI chat completions and return structured ParseResponse (iOS contract).
    """
    import time

    route_t0 = time.perf_counter()
    headers = {**_auth_headers(), "Content-Type": "application/json"}

    system_content = PARSE_SYSTEM_PROMPT
    has_client_hints = bool(parse_instructions and parse_instructions.strip())
    if has_client_hints:
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
        user_lines.append('- target_reference_type to use for active-task follow-ups: "recent_task"')
        user_lines.append(f"- target_task_id to use for active-task follow-ups: {last_active_task_id.strip()}")
        if active_task_title:
            user_lines.append(f'- title: "{active_task_title.strip()}"')
        if active_task_scheduled_at:
            user_lines.append(f"- scheduled_at (ISO 8601): {active_task_scheduled_at.strip()}")
        if active_task_recurrence:
            user_lines.append(f"- recurrence: {json.dumps(active_task_recurrence, ensure_ascii=False)}")
        notes_ctx = _truncate_notes(active_task_notes)
        if notes_ctx:
            user_lines.append(f"- notes (may be truncated): {notes_ctx}")
    user_message = "\n".join(user_lines)

    prompt_build_ms = (time.perf_counter() - route_t0) * 1000
    prompt_chars = len(system_content) + len(user_message)
    from app.util.request_timing import estimate_tokens_from_chars

    logger.info(
        "parse promptBuilt request_id=%s command_session_id=%s promptBuildMs=%.1f promptChars=%s promptTokenEstimate=%s hasClientHints=%s sendsActiveTaskContext=%s",
        request_id,
        command_session_id,
        prompt_build_ms,
        prompt_chars,
        estimate_tokens_from_chars(prompt_chars),
        has_client_hints,
        bool(last_active_task_id),
    )

    payload = {
        "model": OPENAI_PARSE_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_message},
        ],
        "response_format": {"type": "json_object"},
    }

    openai_t0 = time.perf_counter()
    logger.info("parse openAILLMStart request_id=%s model=%s", request_id, OPENAI_PARSE_MODEL)
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
    openai_ms = (time.perf_counter() - openai_t0) * 1000

    if response.status_code != 200:
        logger.error("OpenAI parse error %s: %s model=%s", response.status_code, response.text, OPENAI_PARSE_MODEL)
        response.raise_for_status()

    raw_content = response.json()["choices"][0]["message"]["content"]

    json_t0 = time.perf_counter()
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        logger.error("Failed to decode OpenAI JSON response: %s\nRaw: %s", e, raw_content)
        raise ValueError(f"Model returned invalid JSON: {e}") from e
    json_parse_ms = (time.perf_counter() - json_t0) * 1000

    if not isinstance(data, dict):
        raise ValueError("Model returned JSON that is not an object")

    result = _parse_response_from_llm_dict(data)
    total_ms = (time.perf_counter() - route_t0) * 1000
    logger.info(
        "parse totalInterpretMs=%.1f openAILLMMs=%.1f jsonParseMs=%.1f request_id=%s command_session_id=%s model=%s",
        total_ms,
        openai_ms,
        json_parse_ms,
        request_id,
        command_session_id,
        OPENAI_PARSE_MODEL,
    )
    return result


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
    command_session_id: Optional[str] = None,
) -> CommandInterpretResponse:
    import time
    from app.util.request_timing import estimate_tokens_from_chars

    route_t0 = time.perf_counter()
    headers = {**_auth_headers(), "Content-Type": "application/json"}
    active_id = _coerce_optional_str((active_task or {}).get("id"))
    allowed_ids = {str(candidate.get("id")) for candidate in candidate_tasks if candidate.get("id")}
    if active_id:
        allowed_ids.add(active_id)
    logger.info(
        "interpretRequest request_id=%s command_session_id=%s text=%s candidateTaskCount=%s active_task_present=%s",
        request_id,
        command_session_id,
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
    user_json = json.dumps(user_payload, ensure_ascii=False)
    prompt_chars = len(COMMAND_INTERPRETER_PROMPT) + len(user_json)
    prompt_build_ms = (time.perf_counter() - route_t0) * 1000
    logger.info(
        "interpret promptBuilt request_id=%s promptBuildMs=%.1f promptChars=%s promptTokenEstimate=%s candidateTaskCount=%s sendsFullHistory=false",
        request_id,
        prompt_build_ms,
        prompt_chars,
        estimate_tokens_from_chars(prompt_chars),
        len(candidate_tasks),
    )
    payload = {
        "model": OPENAI_PARSE_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": COMMAND_INTERPRETER_PROMPT},
            {"role": "user", "content": user_json},
        ],
        "response_format": {"type": "json_object"},
    }
    openai_t0 = time.perf_counter()
    logger.info("interpret openAILLMStart request_id=%s model=%s", request_id, OPENAI_PARSE_MODEL)
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )
    openai_ms = (time.perf_counter() - openai_t0) * 1000
    if response.status_code != 200:
        logger.error("OpenAI command interpretation error %s: %s model=%s", response.status_code, response.text, OPENAI_PARSE_MODEL)
        response.raise_for_status()
    raw_content = response.json()["choices"][0]["message"]["content"]
    json_t0 = time.perf_counter()
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        logger.error("Failed to decode command interpretation JSON: %s\nRaw: %s", e, raw_content)
        raise ValueError(f"Model returned invalid JSON: {e}") from e
    json_parse_ms = (time.perf_counter() - json_t0) * 1000
    if not isinstance(data, dict):
        raise ValueError("Model returned JSON that is not an object")
    result = _sanitize_interpret_response(data, allowed_ids, active_id)
    actions_count = len(result.actions or [])
    total_ms = (time.perf_counter() - route_t0) * 1000
    logger.info(
        "interpret totalInterpretMs=%.1f openAILLMMs=%.1f jsonParseMs=%.1f request_id=%s command_session_id=%s model=%s",
        total_ms,
        openai_ms,
        json_parse_ms,
        request_id,
        command_session_id,
        OPENAI_PARSE_MODEL,
    )
    logger.info(
        "interpretResult action_type=%s confidence=%s requires_confirmation=%s target_resolution=%s selected_id=%s actions_count=%s",
        result.action_type,
        result.confidence,
        result.requires_confirmation,
        result.target.resolution if result.target else None,
        result.target.selected_task_id if result.target else None,
        actions_count,
    )
    logger.info("interpretActionsCount count=%s", actions_count)
    for idx, action in enumerate(result.actions or []):
        logger.info(
            "action[%s] type=%s confidence=%s targetResolution=%s",
            idx,
            action.action_type,
            action.confidence,
            action.target.resolution if action.target else None,
        )
    return result
