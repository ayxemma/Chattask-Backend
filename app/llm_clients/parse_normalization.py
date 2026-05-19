"""Normalize raw LLM JSON into the iOS-facing ParseResponse contract."""

import re
from typing import Any, Optional

from app.models.parse_models import ParseResponse


def normalize_action_type(raw: Optional[Any]) -> str:
    """Map model output to iOS ActionType raw strings (camelCase)."""
    if raw is None:
        return "unknown"
    s = str(raw).strip()
    if not s:
        return "unknown"
    collapsed = re.sub(r"[_\s]+", "", s).lower()
    to_canonical = {
        "reminder": "reminder",
        "unknown": "unknown",
        "calendarevent": "calendarEvent",
        "event": "calendarEvent",
        "task": "reminder",
        "todo": "reminder",
        "createtask": "reminder",
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


def normalize_intent(raw: Optional[Any]) -> Optional[str]:
    """Map simplified experimental intent strings to action_type values."""
    if raw is None:
        return None
    collapsed = re.sub(r"[_\s-]+", "", str(raw).strip()).lower()
    mapping = {
        "createtask": "reminder",
        "createreminder": "reminder",
        "createevent": "calendarEvent",
        "createcalendarevent": "calendarEvent",
        "deletetask": "deleteTask",
        "rescheduletask": "rescheduleTask",
        "appendtotask": "appendToTask",
        "updatetitle": "updateTaskTitle",
        "renametask": "updateTaskTitle",
        "updaterecurrence": "updateRecurrence",
        "updatealertstyle": "updateAlertStyle",
        "unknown": "unknown",
    }
    return mapping.get(collapsed)


def coerce_optional_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        t = v.strip()
        return t if t else None
    return str(v)


def coerce_optional_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return None


def coerce_optional_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def coerce_alert_style(v: Any) -> Optional[str]:
    raw = coerce_optional_str(v)
    if not raw:
        return None
    collapsed = re.sub(r"[_\s-]+", "", raw).lower()
    mapping = {
        "silent": "silent",
        "quiet": "silent",
        "nosound": "silent",
        "mute": "silent",
        "muted": "silent",
        "静音": "silent",
        "不要声音": "silent",
        "default": "default",
        "normal": "default",
        "standard": "default",
        "普通": "default",
        "普通提醒": "default",
        "important": "important",
        "loud": "important",
        "strong": "important",
        "alarmlike": "important",
        "重要": "important",
        "重要提醒": "important",
        "明显一点": "important",
        "声音大一点": "important",
    }
    return mapping.get(collapsed)


def sanitize_weekdays(v: Any) -> Optional[list[int]]:
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


def coerce_recurrence(v: Any) -> Optional[dict[str, Any]]:
    if not isinstance(v, dict):
        return None
    recurrence = {
        "frequency": coerce_optional_str(v.get("frequency")),
        "weekdays": sanitize_weekdays(v.get("weekdays")),
        "time": coerce_optional_str(v.get("time")),
        "timezone": coerce_optional_str(v.get("timezone")),
        "start_date": coerce_optional_str(v.get("start_date")),
        "end_date": coerce_optional_str(v.get("end_date")),
    }
    return recurrence if any(value is not None for value in recurrence.values()) else None


def coerce_recurrence_update(v: Any) -> Optional[dict[str, Any]]:
    if not isinstance(v, dict):
        return None
    update = {
        "operation": coerce_optional_str(v.get("operation")),
        "weekdays": sanitize_weekdays(v.get("weekdays")),
        "time": coerce_optional_str(v.get("time")),
        "timezone": coerce_optional_str(v.get("timezone")),
        "start_date": coerce_optional_str(v.get("start_date")),
        "end_date": coerce_optional_str(v.get("end_date")),
    }
    return update if any(value is not None for value in update.values()) else None


def parse_response_from_llm_dict(data: dict[str, Any]) -> ParseResponse:
    """Build ParseResponse from raw JSON object; tolerate minor type drift from the model."""
    action_raw = data.get("action_type")
    if action_raw is None and data.get("intent") is not None:
        action_raw = normalize_intent(data.get("intent"))

    scheduled_at = coerce_optional_str(data.get("scheduled_at"))
    if scheduled_at is None:
        scheduled_at = coerce_optional_str(data.get("datetime"))

    has_specific_time = coerce_optional_bool(data.get("has_specific_time"))
    if has_specific_time is None:
        has_specific_time = coerce_optional_bool(data.get("needs_time"))

    return ParseResponse(
        action_type=normalize_action_type(action_raw),
        title=coerce_optional_str(data.get("title")),
        notes=coerce_optional_str(data.get("notes")),
        scheduled_at=scheduled_at,
        end_at=coerce_optional_str(data.get("end_at")),
        has_specific_time=has_specific_time,
        language_code=coerce_optional_str(data.get("language_code")),
        confidence=coerce_optional_float(data.get("confidence")),
        recurrence=coerce_recurrence(data.get("recurrence")),
        alert_style=coerce_alert_style(data.get("alert_style")),
        target_time=coerce_optional_str(data.get("target_time")),
        new_scheduled_at=coerce_optional_str(data.get("new_scheduled_at")),
        append_text=coerce_optional_str(data.get("append_text")),
        new_title=coerce_optional_str(data.get("new_title")),
        target_reference_type=coerce_optional_str(data.get("target_reference_type")),
        target_task_id=coerce_optional_str(data.get("target_task_id")),
        recurrence_update=coerce_recurrence_update(data.get("recurrence_update")),
    )
