"""Tests for POST /parse."""
from unittest.mock import AsyncMock, patch

from starlette.testclient import TestClient

from app.models.parse_models import ParseResponse

VALID_PAYLOAD = {
    "text": "pick up Ari at 6:15",
    "now": "2026-04-16T17:00:00-04:00",
    "timezone": "America/New_York",
}

MOCK_PARSE_RESPONSE = ParseResponse(
    action_type="reminder",
    title="Pick up Ari",
    notes=None,
    scheduled_at="2026-04-16T18:15:00-04:00",
    confidence=0.95,
    language_code="en",
)


# ---------------------------------------------------------------------------
# Happy-path
# ---------------------------------------------------------------------------

def test_parse_returns_structured_response(client: TestClient):
    """Mocks the OpenAI service layer and verifies the full response shape."""
    with patch(
        "app.routes.parse.parse_task_text",
        new_callable=AsyncMock,
        return_value=MOCK_PARSE_RESPONSE,
    ):
        response = client.post("/parse", json=VALID_PAYLOAD)

    assert response.status_code == 200
    data = response.json()
    assert data["action_type"] == "reminder"
    assert data["title"] == "Pick up Ari"
    assert data["scheduled_at"] == "2026-04-16T18:15:00-04:00"
    assert data["confidence"] == 0.95
    assert data["language_code"] == "en"
    assert "notes" in data  # field present even when null


def test_parse_calls_service_with_correct_args(client: TestClient):
    """Verifies the route forwards request fields to the service unchanged."""
    mock_fn = AsyncMock(return_value=MOCK_PARSE_RESPONSE)
    with patch("app.routes.parse.parse_task_text", new=mock_fn):
        client.post("/parse", json=VALID_PAYLOAD)

    mock_fn.assert_awaited_once_with(
        text=VALID_PAYLOAD["text"],
        now=VALID_PAYLOAD["now"],
        timezone=VALID_PAYLOAD["timezone"],
        locale=None,
        parse_instructions=None,
        source=None,
        last_active_task_id=None,
        active_task_title=None,
        active_task_scheduled_at=None,
        active_task_notes=None,
        active_task_recurrence=None,
    )


def test_parse_returns_full_response_when_service_populates_all_fields(client: TestClient):
    """HTTP JSON exposes the full iOS-facing parse contract when values are set."""
    full = ParseResponse(
        action_type="rescheduleTask",
        title="Move meeting",
        notes="optional",
        scheduled_at="2026-04-16T10:00:00-04:00",
        end_at="2026-04-16T11:00:00-04:00",
        has_specific_time=True,
        language_code="en",
        confidence=0.88,
        recurrence={
            "frequency": "weekly",
            "weekdays": [1, 2, 3, 4, 5],
            "time": "11:50",
            "timezone": "America/New_York",
        },
        target_time="2026-04-16T09:00:00-04:00",
        new_scheduled_at="2026-04-16T15:00:00-04:00",
        append_text=None,
        recurrence_update={
            "operation": "remove_weekdays",
            "weekdays": [4],
        },
    )
    with patch("app.routes.parse.parse_task_text", new_callable=AsyncMock, return_value=full):
        response = client.post("/parse", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert data["action_type"] == "rescheduleTask"
    assert data["title"] == "Move meeting"
    assert data["scheduled_at"] == "2026-04-16T10:00:00-04:00"
    assert data["end_at"] == "2026-04-16T11:00:00-04:00"
    assert data["has_specific_time"] is True
    assert data["target_time"] == "2026-04-16T09:00:00-04:00"
    assert data["new_scheduled_at"] == "2026-04-16T15:00:00-04:00"
    assert data["language_code"] == "en"
    assert data["confidence"] == 0.88
    assert data["recurrence"]["frequency"] == "weekly"
    assert data["recurrence"]["weekdays"] == [1, 2, 3, 4, 5]
    assert data["recurrence_update"]["operation"] == "remove_weekdays"
    assert data["recurrence_update"]["weekdays"] == [4]


def test_parse_forwards_optional_client_context(client: TestClient):
    """Optional JSON fields from iOS are passed into the OpenAI layer."""
    payload = {
        **VALID_PAYLOAD,
        "locale": "en-US",
        "parse_instructions": "Prefer reminders for short phrases.",
        "request_id": "trace-1",
        "source": "voice",
    }
    mock_fn = AsyncMock(return_value=MOCK_PARSE_RESPONSE)
    with patch("app.routes.parse.parse_task_text", mock_fn):
        response = client.post("/parse", json=payload)
    assert response.status_code == 200
    mock_fn.assert_awaited_once_with(
        text=VALID_PAYLOAD["text"],
        now=VALID_PAYLOAD["now"],
        timezone=VALID_PAYLOAD["timezone"],
        locale="en-US",
        parse_instructions="Prefer reminders for short phrases.",
        source="voice",
        last_active_task_id=None,
        active_task_title=None,
        active_task_scheduled_at=None,
        active_task_notes=None,
        active_task_recurrence=None,
    )


def test_parse_forwards_active_task_snapshot(client: TestClient):
    payload = {
        **VALID_PAYLOAD,
        "last_active_task_id": "550e8400-e29b-41d4-a716-446655440000",
        "active_task_title": "Cook dinner",
        "active_task_scheduled_at": "2026-04-16T18:00:00-04:00",
        "active_task_notes": "salt",
        "active_task_recurrence": {
            "frequency": "weekly",
            "weekdays": [1, 2, 3, 4, 5],
            "time": "11:50",
            "timezone": "America/New_York",
        },
    }
    mock_fn = AsyncMock(return_value=MOCK_PARSE_RESPONSE)
    with patch("app.routes.parse.parse_task_text", mock_fn):
        response = client.post("/parse", json=payload)
    assert response.status_code == 200
    mock_fn.assert_awaited_once_with(
        text=VALID_PAYLOAD["text"],
        now=VALID_PAYLOAD["now"],
        timezone=VALID_PAYLOAD["timezone"],
        locale=None,
        parse_instructions=None,
        source=None,
        last_active_task_id="550e8400-e29b-41d4-a716-446655440000",
        active_task_title="Cook dinner",
        active_task_scheduled_at="2026-04-16T18:00:00-04:00",
        active_task_notes="salt",
        active_task_recurrence={
            "frequency": "weekly",
            "weekdays": [1, 2, 3, 4, 5],
            "time": "11:50",
            "timezone": "America/New_York",
        },
    )


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_parse_rejects_empty_text(client: TestClient):
    payload = {**VALID_PAYLOAD, "text": "   "}
    with patch("app.routes.parse.parse_task_text", new_callable=AsyncMock):
        response = client.post("/parse", json=payload)
    assert response.status_code == 400
    assert "text" in response.json()["detail"].lower()


def test_parse_rejects_missing_text_field(client: TestClient):
    payload = {"now": VALID_PAYLOAD["now"], "timezone": VALID_PAYLOAD["timezone"]}
    response = client.post("/parse", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_parse_rejects_missing_now_field(client: TestClient):
    payload = {"text": "buy milk", "timezone": "America/New_York"}
    response = client.post("/parse", json=payload)
    assert response.status_code == 422


def test_parse_rejects_missing_timezone_field(client: TestClient):
    payload = {"text": "buy milk", "now": VALID_PAYLOAD["now"]}
    response = client.post("/parse", json=payload)
    assert response.status_code == 422


def test_parse_rejects_non_json_body(client: TestClient):
    response = client.post("/parse", content=b"not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Service / upstream errors
# ---------------------------------------------------------------------------

def test_parse_returns_422_on_invalid_model_json(client: TestClient):
    """If the service raises ValueError (bad JSON from model), the route returns 422."""
    with patch(
        "app.routes.parse.parse_task_text",
        new_callable=AsyncMock,
        side_effect=ValueError("Model returned invalid JSON"),
    ):
        response = client.post("/parse", json=VALID_PAYLOAD)
    assert response.status_code == 422


def test_parse_returns_502_on_upstream_error(client: TestClient):
    """Generic upstream exception surfaces as 502."""
    with patch(
        "app.routes.parse.parse_task_text",
        new_callable=AsyncMock,
        side_effect=Exception("network timeout"),
    ):
        response = client.post("/parse", json=VALID_PAYLOAD)
    assert response.status_code == 502


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------

def test_parse_returns_500_when_api_key_missing(client: TestClient):
    with patch("app.routes.parse.OPENAI_API_KEY", None):
        response = client.post("/parse", json=VALID_PAYLOAD)
    assert response.status_code == 500
    assert "API key" in response.json()["detail"]
