from unittest.mock import AsyncMock, patch

from starlette.testclient import TestClient

from app.models.parse_models import TaskTargetResolveResponse


VALID_PAYLOAD = {
    "user_text": "把接阿里那个任务改到12点",
    "action_type": "rescheduleTask",
    "target_title": "接阿里",
    "target_time": None,
    "timezone": "America/New_York",
    "locale": "zh-Hans",
    "candidates": [
        {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "去接阿瑞",
            "scheduled_at": "2026-05-15T11:50:00-04:00",
            "is_recurring": True,
            "recurrence_label": "Mon-Fri 11:50",
        }
    ],
}


def test_resolve_task_target_returns_service_response(client: TestClient):
    result = TaskTargetResolveResponse(
        resolution="needs_confirmation",
        selected_id="550e8400-e29b-41d4-a716-446655440000",
        confidence=0.72,
        reason="Chinese near-homophone title match",
    )
    with patch("app.routes.resolve_task_target.resolve_task_target", new_callable=AsyncMock, return_value=result):
        response = client.post("/resolve-task-target", json=VALID_PAYLOAD)

    assert response.status_code == 200
    data = response.json()
    assert data["resolution"] == "needs_confirmation"
    assert data["selected_id"] == "550e8400-e29b-41d4-a716-446655440000"
    assert data["confidence"] == 0.72


def test_resolve_task_target_forwards_payload(client: TestClient):
    mock_fn = AsyncMock(return_value=TaskTargetResolveResponse(resolution="no_match", confidence=0.0))
    with patch("app.routes.resolve_task_target.resolve_task_target", mock_fn):
        response = client.post("/resolve-task-target", json=VALID_PAYLOAD)

    assert response.status_code == 200
    mock_fn.assert_awaited_once_with(
        user_text=VALID_PAYLOAD["user_text"],
        action_type=VALID_PAYLOAD["action_type"],
        target_title=VALID_PAYLOAD["target_title"],
        target_time=None,
        candidates=VALID_PAYLOAD["candidates"],
        active_task_id=None,
        timezone=VALID_PAYLOAD["timezone"],
        locale=VALID_PAYLOAD["locale"],
    )


def test_resolve_task_target_empty_candidates_short_circuits(client: TestClient):
    payload = {**VALID_PAYLOAD, "candidates": []}
    with patch("app.routes.resolve_task_target.resolve_task_target", new_callable=AsyncMock) as mock_fn:
        response = client.post("/resolve-task-target", json=payload)

    assert response.status_code == 200
    assert response.json()["resolution"] == "no_match"
    mock_fn.assert_not_called()
