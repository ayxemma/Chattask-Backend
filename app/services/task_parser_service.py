"""Task parsing orchestration with provider selection and fallback."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from app.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_PARSE_MODEL,
    SGLANG_API_KEY,
    SGLANG_BASE_URL,
    SGLANG_MODEL,
    USE_SGLANG_PARSER,
)
from app.llm_clients.base import BaseLLMClient, ParseTaskContext
from app.llm_clients.openai_client import OpenAIClient
from app.llm_clients.sglang_client import SGLangClient
from app.models.parse_models import ParseResponse

logger = logging.getLogger(__name__)

_openai_client: Optional[OpenAIClient] = None
_sglang_client: Optional[SGLangClient] = None


def _get_openai_client() -> OpenAIClient:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIClient(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            model=OPENAI_PARSE_MODEL,
        )
    return _openai_client


def _get_sglang_client() -> SGLangClient:
    global _sglang_client
    if _sglang_client is None:
        _sglang_client = SGLangClient(
            base_url=SGLANG_BASE_URL or "",
            api_key=SGLANG_API_KEY,
            model=SGLANG_MODEL or "",
        )
    return _sglang_client


def reset_parser_clients_for_tests() -> None:
    """Clear cached clients so tests can reconfigure providers."""
    global _openai_client, _sglang_client
    _openai_client = None
    _sglang_client = None


def select_parser_client() -> BaseLLMClient:
    """Return the configured primary parser provider."""
    if USE_SGLANG_PARSER:
        return _get_sglang_client()
    return _get_openai_client()


def _log_parse_result(
    *,
    provider: str,
    latency_ms: float,
    success: bool,
    fallback_from: Optional[str] = None,
) -> None:
    logger.info(
        "taskParse provider=%s latency_ms=%.1f success=%s fallback_from=%s",
        provider,
        latency_ms,
        success,
        fallback_from or "none",
    )


async def _parse_with_client(client: BaseLLMClient, context: ParseTaskContext) -> ParseResponse:
    started = time.perf_counter()
    try:
        result = await client.parse_task(context)
    except Exception:
        latency_ms = (time.perf_counter() - started) * 1000
        _log_parse_result(provider=client.provider_name, latency_ms=latency_ms, success=False)
        raise
    latency_ms = (time.perf_counter() - started) * 1000
    _log_parse_result(provider=client.provider_name, latency_ms=latency_ms, success=True)
    return result


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
) -> ParseResponse:
    """
    Parse task text using the configured LLM provider.

    When USE_SGLANG_PARSER=true, SGLang is tried first; failures fall back to OpenAI
    without failing the HTTP request.
    """
    context = ParseTaskContext(
        text=text,
        now=now,
        timezone=timezone,
        locale=locale,
        parse_instructions=parse_instructions,
        source=source,
        last_active_task_id=last_active_task_id,
        active_task_title=active_task_title,
        active_task_scheduled_at=active_task_scheduled_at,
        active_task_notes=active_task_notes,
        active_task_recurrence=active_task_recurrence,
    )

    if USE_SGLANG_PARSER:
        sglang_client = _get_sglang_client()
        try:
            return await _parse_with_client(sglang_client, context)
        except Exception as exc:
            logger.warning(
                "SGLang parser failed (%s); falling back to OpenAI parser",
                exc,
                exc_info=True,
            )
            openai_client = _get_openai_client()
            started = time.perf_counter()
            try:
                result = await openai_client.parse_task(context)
            except Exception:
                latency_ms = (time.perf_counter() - started) * 1000
                _log_parse_result(
                    provider=openai_client.provider_name,
                    latency_ms=latency_ms,
                    success=False,
                    fallback_from=sglang_client.provider_name,
                )
                raise
            latency_ms = (time.perf_counter() - started) * 1000
            _log_parse_result(
                provider=openai_client.provider_name,
                latency_ms=latency_ms,
                success=True,
                fallback_from=sglang_client.provider_name,
            )
            return result

    return await _parse_with_client(_get_openai_client(), context)
