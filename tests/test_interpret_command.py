from unittest.mock import AsyncMock, patch

from starlette.testclient import TestClient

from app.models.parse_models import (
    CommandInterpretAction,
    CommandInterpretCreate,
    CommandInterpretEdit,
    CommandInterpretResponse,
    CommandInterpretTarget,
)


VALID_PAYLOAD = {
    "text": "把给艾瑞喂水果酸奶的任务改成四点半",
    "now": "2026-05-17T12:00:00-04:00",
    "timezone": "America/New_York",
    "locale": "zh",
    "active_task": None,
    "candidate_tasks": [
        {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "给艾瑞喂芒果酸奶",
            "scheduled_at": "2026-05-17T18:00:00-04:00",
            "is_recurring": False,
        }
    ],
    "request_id": "trace-interpret-1",
}


def test_interpret_command_forwards_payload(client: TestClient):
    result = CommandInterpretResponse(
        action_type="rescheduleTask",
        confidence=0.72,
        requires_confirmation=True,
        confirmation_kind="confirm_action",
        assistant_message="Did you mean 给艾瑞喂芒果酸奶?",
        target=CommandInterpretTarget(
            resolution="candidate",
            selected_task_id="550e8400-e29b-41d4-a716-446655440000",
            selected_task_title="给艾瑞喂芒果酸奶",
        ),
        edit=CommandInterpretEdit(new_scheduled_at="2026-05-17T16:30:00-04:00"),
    )
    mock_fn = AsyncMock(return_value=result)
    with patch("app.routes.interpret_command.interpret_command", mock_fn):
        response = client.post("/interpret-command", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert response.json()["action_type"] == "rescheduleTask"
    mock_fn.assert_awaited_once_with(
        text=VALID_PAYLOAD["text"],
        now=VALID_PAYLOAD["now"],
        timezone=VALID_PAYLOAD["timezone"],
        locale="zh",
        active_task=None,
        candidate_tasks=VALID_PAYLOAD["candidate_tasks"],
        request_id="trace-interpret-1",
    )


def test_interpret_command_create_response_shape(client: TestClient):
    result = CommandInterpretResponse(
        action_type="createReminder",
        confidence=0.95,
        requires_confirmation=False,
        confirmation_kind="none",
        assistant_message="Added.",
        create=CommandInterpretCreate(
            title="喝水",
            scheduled_at="2026-05-17T12:10:00-04:00",
            has_specific_time=True,
            recurrence_type="none",
        ),
    )
    with patch("app.routes.interpret_command.interpret_command", new_callable=AsyncMock, return_value=result):
        response = client.post("/interpret-command", json={**VALID_PAYLOAD, "text": "提醒我十分钟后喝水"})
    assert response.status_code == 200
    data = response.json()
    assert data["action_type"] == "createReminder"
    assert data["create"]["title"] == "喝水"


def test_interpret_command_multi_action_response_shape(client: TestClient):
    active_id = "550e8400-e29b-41d4-a716-446655440000"
    result = CommandInterpretResponse(
        assistant_message="Done — I moved it and added the reminder.",
        actions=[
            CommandInterpretAction(
                action_type="rescheduleTask",
                confidence=0.95,
                requires_confirmation=False,
                confirmation_kind="none",
                target=CommandInterpretTarget(
                    resolution="active_task",
                    selected_task_id=active_id,
                ),
                edit=CommandInterpretEdit(new_scheduled_at="2026-05-18T11:00:00-04:00"),
            ),
            CommandInterpretAction(
                action_type="createReminder",
                confidence=0.94,
                requires_confirmation=False,
                confirmation_kind="none",
                create=CommandInterpretCreate(
                    title="buy vegetables, tomatoes, potatoes",
                    scheduled_at="2026-05-17T12:10:00-04:00",
                    has_specific_time=True,
                ),
            ),
        ],
    )
    with patch("app.routes.interpret_command.interpret_command", new_callable=AsyncMock, return_value=result):
        response = client.post(
            "/interpret-command",
            json={
                **VALID_PAYLOAD,
                "text": "Actually change it to tomorrow at 11 and also remind me to buy vegetables, tomatoes, potatoes.",
                "active_task": VALID_PAYLOAD["candidate_tasks"][0],
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert len(data["actions"]) == 2
    assert data["actions"][0]["action_type"] == "rescheduleTask"
    assert data["actions"][1]["create"]["title"] == "buy vegetables, tomatoes, potatoes"


def test_interpret_command_rejects_empty_text(client: TestClient):
    response = client.post("/interpret-command", json={**VALID_PAYLOAD, "text": "   "})
    assert response.status_code == 400
