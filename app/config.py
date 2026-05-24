from __future__ import annotations

import logging
import os
import sys
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _strip_or_none(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    s = val.strip()
    return s if s else None


OPENAI_API_KEY: Optional[str] = _strip_or_none(os.getenv("OPENAI_API_KEY"))
OPENAI_BASE_URL: str = (
    _strip_or_none(os.getenv("OPENAI_BASE_URL")) or "https://api.openai.com/v1"
).rstrip("/")
OPENAI_TRANSCRIBE_MODEL: str = (
    _strip_or_none(os.getenv("OPENAI_TRANSCRIBE_MODEL")) or "gpt-4o-mini-transcribe"
)
OPENAI_PARSE_MODEL: str = (
    _strip_or_none(os.getenv("OPENAI_PARSE_MODEL")) or "gpt-4o-mini"
)
OPENAI_INTERPRET_MODEL: str = (
    _strip_or_none(os.getenv("OPENAI_INTERPRET_MODEL"))
    or _strip_or_none(os.getenv("OPENAI_PARSE_MODEL"))
    or "gpt-4o-mini"
)
OPENAI_INTERPRET_MAX_TOKENS: int = int(
    _strip_or_none(os.getenv("OPENAI_INTERPRET_MAX_TOKENS")) or "900"
)


def cors_allow_origins() -> List[str]:
    """Origins allowed for browser CORS. Empty list means no cross-origin browser access."""
    raw = os.getenv("CORS_ORIGIN", "").strip()
    if not raw:
        return []
    return [o.strip() for o in raw.split(",") if o.strip()]


def listen_port() -> int:
    raw = os.getenv("PORT", "3000").strip()
    try:
        return int(raw)
    except ValueError:
        logger.error("PORT must be an integer, got: %r", raw)
        sys.exit(1)


def validate_config() -> None:
    if not OPENAI_API_KEY:
        logger.error(
            "Missing required environment variable OPENAI_API_KEY. "
            "Set it in your hosting provider's environment (or .env for local development)."
        )
        sys.exit(1)
