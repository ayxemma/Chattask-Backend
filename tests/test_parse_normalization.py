"""Contract tests for LLM parse normalization (action types + coercion)."""

from app.services.openai_service import _normalize_action_type, _parse_response_from_llm_dict


def test_normalize_action_type_maps_legacy_and_variants():
    assert _normalize_action_type(None) == "unknown"
    assert _normalize_action_type("") == "unknown"
    assert _normalize_action_type("reminder") == "reminder"
    assert _normalize_action_type("unknown") == "unknown"
    assert _normalize_action_type("calendarEvent") == "calendarEvent"
    assert _normalize_action_type("event") == "calendarEvent"
    assert _normalize_action_type("calendar_event") == "calendarEvent"
    assert _normalize_action_type("task") == "reminder"
    assert _normalize_action_type("delete_task") == "deleteTask"
    assert _normalize_action_type("rescheduleTask") == "rescheduleTask"
    assert _normalize_action_type("append_to_task") == "appendToTask"
    assert _normalize_action_type("update_task_title") == "updateTaskTitle"
    assert _normalize_action_type("rename_task") == "updateTaskTitle"


def test_parse_response_from_llm_dict_normalizes_and_coerces():
    r = _parse_response_from_llm_dict(
        {
            "action_type": "event",
            "title": "  x  ",
            "confidence": "0.75",
            "has_specific_time": 1,
            "scheduled_at": "2026-04-16T18:15:00-04:00",
            "target_time": None,
        }
    )
    assert r.action_type == "calendarEvent"
    assert r.title == "x"
    assert r.confidence == 0.75
    assert r.has_specific_time is True
    assert r.scheduled_at == "2026-04-16T18:15:00-04:00"


def test_parse_response_includes_new_title():
    r = _parse_response_from_llm_dict({"action_type": "updateTaskTitle", "new_title": "  Buy eggs  "})
    assert r.action_type == "updateTaskTitle"
    assert r.new_title == "Buy eggs"
