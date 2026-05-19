"""Tests for StructuredTaskParseOutput validation."""

from app.models.llm_parse_output import StructuredTaskParseOutput


def test_validates_simplified_experimental_schema():
    output = StructuredTaskParseOutput.from_llm_dict(
        {
            "intent": "create_task",
            "title": "Call John",
            "datetime": "2026-05-20T15:00:00-04:00",
            "needs_time": True,
            "recurrence": None,
            "confidence": 0.92,
        }
    )
    response = output.to_parse_response()
    assert response.action_type == "reminder"
    assert response.title == "Call John"
    assert response.scheduled_at == "2026-05-20T15:00:00-04:00"
    assert response.has_specific_time is True
    assert response.confidence == 0.92


def test_validates_full_production_schema():
    output = StructuredTaskParseOutput.from_llm_dict(
        {
            "action_type": "reminder",
            "title": "Walk the dog",
            "scheduled_at": "2026-05-20T08:00:00-04:00",
            "has_specific_time": True,
            "confidence": 0.85,
        }
    )
    response = output.to_parse_response()
    assert response.action_type == "reminder"
    assert response.title == "Walk the dog"
