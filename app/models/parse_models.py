from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ParseRequest(BaseModel):
    """POST /parse body — aligned with iOS `LLMTaskParserService` payload."""

    model_config = ConfigDict(extra="ignore")

    text: str
    now: str  # ISO 8601 with offset, e.g. "2026-04-16T17:00:00-04:00"
    timezone: str  # IANA, e.g. "America/New_York"
    locale: Optional[str] = None  # BCP 47 / Apple identifier from client (parsing + language_code hints)
    parse_instructions: Optional[str] = None  # Short client hints; merged into system prompt on server (not a full prompt)
    request_id: Optional[str] = None  # Correlation id only (optional JSON field `request_id` from app)

    # Optional input modality for the model (typed vs voice); safe telemetry-style context only.
    source: Optional[str] = Field(
        default=None,
        description='e.g. "typed" | "voice" — may appear in user message for disambiguation',
    )

    # Chat follow-up: most recently created/edited task while the sheet is open (client snapshot).
    last_active_task_id: Optional[str] = None
    active_task_title: Optional[str] = None
    active_task_scheduled_at: Optional[str] = None
    active_task_notes: Optional[str] = None


class ParseResponse(BaseModel):
    """JSON returned to iOS — field names match `LLMTaskParseResponse` CodingKeys (snake_case)."""

    model_config = ConfigDict(extra="ignore")

    action_type: Optional[str] = None
    title: Optional[str] = None
    notes: Optional[str] = None
    scheduled_at: Optional[str] = None
    end_at: Optional[str] = None
    has_specific_time: Optional[bool] = None
    language_code: Optional[str] = None
    confidence: Optional[float] = None
    target_time: Optional[str] = None
    new_scheduled_at: Optional[str] = None
    append_text: Optional[str] = None
    new_title: Optional[str] = None
    target_reference_type: Optional[str] = None
    target_task_id: Optional[str] = None
