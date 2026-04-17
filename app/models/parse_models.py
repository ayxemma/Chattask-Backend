from pydantic import BaseModel
from typing import Optional


class ParseRequest(BaseModel):
    text: str
    now: str  # ISO 8601 datetime string with timezone offset, e.g. "2026-04-16T17:00:00-04:00"
    timezone: str  # IANA timezone name, e.g. "America/New_York"


class ParseResponse(BaseModel):
    action_type: str           # e.g. "reminder", "task", "event"
    title: str
    notes: Optional[str]
    scheduled_at: Optional[str]  # ISO 8601 datetime string or null
    confidence: float
    language_code: str          # e.g. "en", "es", "zh"
