from unittest.mock import AsyncMock, patch

from starlette.testclient import TestClient

from app.models.parse_models import (
    CommandInterpretAction,
    CommandInterpretCreate,
    CommandInterpretEdit,
    CommandInterpretResponse,
    CommandInterpretTarget,
)
from app.capabilities.validation import infer_input_language, normalize_reminder_offset_minutes
from app.services.openai_service import _looks_like_multi_timed_create, _sanitize_interpret_response


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
        command_session_id=None,
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
    result = CommandInterpretResponse(
        action_type="rescheduleTask",
        confidence=0.93,
        requires_confirmation=False,
        confirmation_kind="none",
        assistant_message="Done — moved it and added the reminder.",
        target=CommandInterpretTarget(
            resolution="active_task",
            selected_task_id="550e8400-e29b-41d4-a716-446655440000",
        ),
        edit=CommandInterpretEdit(new_scheduled_at="2026-05-18T11:00:00-04:00"),
        actions=[
            CommandInterpretAction(
                action_type="rescheduleTask",
                confidence=0.93,
                requires_confirmation=False,
                confirmation_kind="none",
                target=CommandInterpretTarget(
                    resolution="active_task",
                    selected_task_id="550e8400-e29b-41d4-a716-446655440000",
                ),
                edit=CommandInterpretEdit(new_scheduled_at="2026-05-18T11:00:00-04:00"),
            ),
            CommandInterpretAction(
                action_type="createReminder",
                confidence=0.91,
                requires_confirmation=False,
                confirmation_kind="none",
                create=CommandInterpretCreate(title="buy tomatoes"),
            ),
        ],
    )
    with patch("app.routes.interpret_command.interpret_command", new_callable=AsyncMock, return_value=result):
        response = client.post(
            "/interpret-command",
            json={
                **VALID_PAYLOAD,
                "text": "Change this to tomorrow at 11 and remind me to buy tomatoes.",
                "active_task": VALID_PAYLOAD["candidate_tasks"][0],
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert data["action_type"] == "rescheduleTask"
    assert len(data["actions"]) == 2
    assert data["actions"][0]["action_type"] == "rescheduleTask"
    assert data["actions"][1]["action_type"] == "createReminder"
    assert data["actions"][1]["create"]["title"] == "buy tomatoes"


def test_interpret_command_splits_compound_reschedule_append_action():
    result = _sanitize_interpret_response(
        {
            "actions": [
                {
                    "action_type": "rescheduleTask",
                    "confidence": 0.94,
                    "requires_confirmation": False,
                    "confirmation_kind": "none",
                    "target": {
                        "resolution": "active_task",
                        "selected_task_id": "550e8400-e29b-41d4-a716-446655440000",
                    },
                    "edit": {
                        "new_scheduled_at": "2026-05-18T11:00:00-04:00",
                        "append_text": "buy vegetables",
                    },
                }
            ]
        },
        allowed_ids={"550e8400-e29b-41d4-a716-446655440000"},
        active_id="550e8400-e29b-41d4-a716-446655440000",
    )
    assert result.actions is not None
    assert len(result.actions) == 2
    assert result.actions[0].action_type == "rescheduleTask"
    assert result.actions[0].edit.new_scheduled_at == "2026-05-18T11:00:00-04:00"
    assert result.actions[0].edit.append_text is None
    assert result.actions[1].action_type == "appendToTask"
    assert result.actions[1].edit.append_text == "buy vegetables"
    assert result.actions[1].target.selected_task_id == "550e8400-e29b-41d4-a716-446655440000"


def test_interpret_command_promotes_multi_action_clarification_kind():
    result = _sanitize_interpret_response(
        {
            "actions": [
                {
                    "action_type": "rescheduleTask",
                    "confidence": 0.9,
                    "requires_confirmation": False,
                    "confirmation_kind": "none",
                    "target": {
                        "resolution": "active_task",
                        "selected_task_id": "550e8400-e29b-41d4-a716-446655440000",
                    },
                    "edit": {"new_scheduled_at": "2026-05-18T11:00:00-04:00"},
                },
                {
                    "action_type": "createReminder",
                    "confidence": 0.55,
                    "requires_confirmation": True,
                    "confirmation_kind": "clarify",
                    "assistant_message": "Should I add it as a note or create a reminder?",
                    "create": {"title": "buy vegetables"},
                },
            ]
        },
        allowed_ids={"550e8400-e29b-41d4-a716-446655440000"},
        active_id="550e8400-e29b-41d4-a716-446655440000",
    )
    assert result.requires_confirmation is True
    assert result.confirmation_kind == "clarify"


def test_interpret_command_rejects_empty_text(client: TestClient):
    response = client.post("/interpret-command", json={**VALID_PAYLOAD, "text": "   "})
    assert response.status_code == 400


def test_sanitize_promotes_top_level_single_create_to_actions():
    result = _sanitize_interpret_response(
        {
            "action_type": "createReminder",
            "confidence": 0.95,
            "requires_confirmation": False,
            "confirmation_kind": "none",
            "create": {
                "title": "喝水",
                "scheduled_at": "2026-05-17T12:10:00-04:00",
                "has_specific_time": True,
            },
        },
        allowed_ids=set(),
        active_id=None,
    )
    assert result.actions is not None
    assert len(result.actions) == 1
    assert result.actions[0].action_type == "createReminder"
    assert result.actions[0].create.title == "喝水"


def test_sanitize_preserves_dual_create_actions():
    result = _sanitize_interpret_response(
        {
            "action_type": "createReminder",
            "confidence": 0.95,
            "requires_confirmation": False,
            "confirmation_kind": "none",
            "create": {
                "title": "给艾瑞做早餐",
                "scheduled_at": "2026-05-26T08:00:00-04:00",
                "has_specific_time": True,
            },
            "actions": [
                {
                    "action_type": "createReminder",
                    "confidence": 0.95,
                    "requires_confirmation": False,
                    "confirmation_kind": "none",
                    "create": {
                        "title": "给艾瑞做早餐",
                        "scheduled_at": "2026-05-26T08:00:00-04:00",
                        "has_specific_time": True,
                    },
                },
                {
                    "action_type": "createReminder",
                    "confidence": 0.93,
                    "requires_confirmation": False,
                    "confirmation_kind": "none",
                    "create": {
                        "title": "给艾瑞接回家",
                        "scheduled_at": "2026-05-26T11:50:00-04:00",
                        "has_specific_time": True,
                    },
                },
            ],
        },
        allowed_ids=set(),
        active_id=None,
        user_text="提醒我周二早晨八点给艾瑞做早餐,十一点五十给艾瑞接回来。",
    )
    assert result.actions is not None
    assert len(result.actions) == 2
    assert result.actions[0].create.title == "给艾瑞做早餐"
    assert result.actions[1].create.title == "给艾瑞接回家"


def test_looks_like_multi_timed_create_heuristics():
    assert _looks_like_multi_timed_create("提醒我六点做晚餐,七点陪艾瑞玩儿。")
    assert _looks_like_multi_timed_create("提醒我周二早晨八点给艾瑞做早餐,十一点五十给艾瑞接回来。")
    assert not _looks_like_multi_timed_create("把这个改成四点半,并把酸奶和水果带进去。")
    assert not _looks_like_multi_timed_create("提醒我十分钟后喝水。")


def test_sanitize_logs_warning_when_multi_create_likely_dropped(caplog):
    import logging

    caplog.set_level(logging.WARNING)
    _sanitize_interpret_response(
        {
            "action_type": "createReminder",
            "confidence": 0.95,
            "requires_confirmation": False,
            "confirmation_kind": "none",
            "create": {
                "title": "给艾瑞做早餐",
                "scheduled_at": "2026-05-26T08:00:00-04:00",
                "has_specific_time": True,
            },
        },
        allowed_ids=set(),
        active_id=None,
        user_text="提醒我周二早晨八点给艾瑞做早餐,十一点五十给艾瑞接回来。",
    )
    assert any(
        "multiCreateExpectedButSingleActionReturned" in record.message
        for record in caplog.records
    )


def test_infer_input_language_from_chinese_text():
    assert infer_input_language("把九点的任务改成十点。", "en") == "zh"


def test_normalize_reminder_offset_minutes_snaps_to_catalog():
    assert normalize_reminder_offset_minutes(10) == 10
    assert normalize_reminder_offset_minutes(12) == 10
    assert normalize_reminder_offset_minutes(7) == 5


def test_enforce_catalog_composition_remind_before_is_single_action():
    from app.capabilities.validation import enforce_catalog_composition
    from app.models.parse_models import CommandInterpretAction, CommandInterpretEdit, CommandInterpretTarget

    actions = [
        CommandInterpretAction(
            action_type="rescheduleTask",
            edit=CommandInterpretEdit(new_scheduled_at="2026-05-25T18:10:00-04:00", reminder_offset_minutes=10),
            target=CommandInterpretTarget(resolution="active_task", selected_task_id="abc"),
        ),
        CommandInterpretAction(
            action_type="appendToTask",
            edit=CommandInterpretEdit(append_text="有面试"),
            target=CommandInterpretTarget(resolution="active_task", selected_task_id="abc"),
        ),
    ]
    result = enforce_catalog_composition(actions)
    assert len(result) == 1
    assert result[0].action_type == "rescheduleTask"
