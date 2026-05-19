"""Tests for task_parser_service provider selection and fallback."""

from unittest.mock import AsyncMock, patch

import pytest

from app.models.parse_models import ParseResponse
from app.services import task_parser_service

MOCK_RESPONSE = ParseResponse(
    action_type="reminder",
    title="Call John",
    scheduled_at="2026-05-20T15:00:00-04:00",
    confidence=0.9,
)


@pytest.fixture(autouse=True)
def _reset_clients():
    task_parser_service.reset_parser_clients_for_tests()
    yield
    task_parser_service.reset_parser_clients_for_tests()


@pytest.mark.asyncio
async def test_default_uses_openai_client():
    with patch("app.services.task_parser_service.USE_SGLANG_PARSER", False):
        with patch("app.services.task_parser_service.OpenAIClient") as mock_cls:
            mock_cls.return_value.parse_task = AsyncMock(return_value=MOCK_RESPONSE)
            result = await task_parser_service.parse_task_text(
                text="call John at 3pm",
                now="2026-05-19T10:00:00-04:00",
                timezone="America/New_York",
            )
    assert result.title == "Call John"
    mock_cls.return_value.parse_task.assert_awaited_once()


@pytest.mark.asyncio
async def test_sglang_enabled_uses_sglang_client():
    with patch("app.services.task_parser_service.USE_SGLANG_PARSER", True):
        with patch("app.services.task_parser_service.SGLangClient") as sglang_cls:
            with patch("app.services.task_parser_service.OpenAIClient"):
                sglang_cls.return_value.parse_task = AsyncMock(return_value=MOCK_RESPONSE)
                result = await task_parser_service.parse_task_text(
                    text="call John at 3pm",
                    now="2026-05-19T10:00:00-04:00",
                    timezone="America/New_York",
                )
    assert result.title == "Call John"
    sglang_cls.return_value.parse_task.assert_awaited_once()


@pytest.mark.asyncio
async def test_sglang_failure_falls_back_to_openai():
    with patch("app.services.task_parser_service.USE_SGLANG_PARSER", True):
        with patch("app.services.task_parser_service.SGLangClient") as sglang_cls:
            with patch("app.services.task_parser_service.OpenAIClient") as openai_cls:
                sglang_cls.return_value.parse_task = AsyncMock(side_effect=RuntimeError("sglang down"))
                openai_cls.return_value.parse_task = AsyncMock(return_value=MOCK_RESPONSE)
                result = await task_parser_service.parse_task_text(
                    text="call John at 3pm",
                    now="2026-05-19T10:00:00-04:00",
                    timezone="America/New_York",
                )
    assert result.title == "Call John"
    sglang_cls.return_value.parse_task.assert_awaited_once()
    openai_cls.return_value.parse_task.assert_awaited_once()


def test_select_parser_client_defaults_to_openai():
    with patch("app.services.task_parser_service.USE_SGLANG_PARSER", False):
        with patch("app.services.task_parser_service.OpenAIClient") as mock_cls:
            client = task_parser_service.select_parser_client()
    assert client is mock_cls.return_value


def test_select_parser_client_uses_sglang_when_enabled():
    with patch("app.services.task_parser_service.USE_SGLANG_PARSER", True):
        with patch("app.services.task_parser_service.SGLangClient") as mock_cls:
            with patch("app.services.task_parser_service.OpenAIClient"):
                client = task_parser_service.select_parser_client()
    assert client is mock_cls.return_value
