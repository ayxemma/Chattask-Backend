"""Request timing helpers and cold-start detection for latency audit logs."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
COMMAND_SESSION_ID_HEADER = "X-Command-Session-ID"

_PROCESS_START = time.perf_counter()
_REQUEST_COUNT = 0


@dataclass
class RequestCorrelation:
    request_id: Optional[str] = None
    command_session_id: Optional[str] = None
    likely_cold_start: bool = False
    process_uptime_ms: float = 0.0


@dataclass
class TimingSpan:
    label: str
    correlation: RequestCorrelation
    started_at: float = field(default_factory=time.perf_counter)
    openai_ms: Optional[float] = None
    retry_count: int = 0

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000

    def log(self, **fields: object) -> None:
        parts = [
            f"{self.label}",
            f"request_id={self.correlation.request_id}",
            f"command_session_id={self.correlation.command_session_id}",
            f"totalMs={self.elapsed_ms():.1f}",
            f"likelyColdStart={self.correlation.likely_cold_start}",
            f"processUptimeMs={self.correlation.process_uptime_ms:.1f}",
        ]
        if self.openai_ms is not None:
            parts.append(f"openAIMs={self.openai_ms:.1f}")
        if self.retry_count:
            parts.append(f"retries={self.retry_count}")
        for key, value in fields.items():
            parts.append(f"{key}={value}")
        logger.info(" ".join(parts))


def correlation_from_headers(headers: dict) -> RequestCorrelation:
    global _REQUEST_COUNT
    _REQUEST_COUNT += 1
    uptime_ms = (time.perf_counter() - _PROCESS_START) * 1000
    likely_cold = _REQUEST_COUNT == 1 and uptime_ms < 30_000
    return RequestCorrelation(
        request_id=headers.get(REQUEST_ID_HEADER) or headers.get("x-request-id"),
        command_session_id=headers.get(COMMAND_SESSION_ID_HEADER) or headers.get("x-command-session-id"),
        likely_cold_start=likely_cold,
        process_uptime_ms=uptime_ms,
    )


def estimate_tokens_from_chars(char_count: int) -> int:
    """Rough token estimate (~4 chars per token for mixed CJK/Latin)."""
    return max(1, char_count // 4)
