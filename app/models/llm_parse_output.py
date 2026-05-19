"""Pydantic models for validating structured LLM parse JSON."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.llm_clients.parse_normalization import parse_response_from_llm_dict
from app.models.parse_models import ParseResponse, RecurrenceSpec, RecurrenceUpdate


class StructuredTaskParseOutput(BaseModel):
    """
    Validated LLM JSON for task parsing.

    Supports the full ChatTask production schema and a simplified experimental schema:

        {
          "intent": "create_task",
          "title": "...",
          "datetime": "...",
          "needs_time": false,
          "recurrence": null,
          "confidence": 0.0
        }
    """

    model_config = ConfigDict(extra="ignore")

    # Full production schema
    action_type: Optional[str] = None
    title: Optional[str] = None
    notes: Optional[str] = None
    scheduled_at: Optional[str] = None
    end_at: Optional[str] = None
    has_specific_time: Optional[bool] = None
    language_code: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    recurrence: Optional[RecurrenceSpec] = None
    alert_style: Optional[str] = None
    target_time: Optional[str] = None
    new_scheduled_at: Optional[str] = None
    append_text: Optional[str] = None
    new_title: Optional[str] = None
    target_reference_type: Optional[str] = None
    target_task_id: Optional[str] = None
    recurrence_update: Optional[RecurrenceUpdate] = None

    # Simplified experimental schema (SGLang-friendly)
    intent: Optional[str] = None
    datetime: Optional[str] = None
    needs_time: Optional[bool] = None

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, parsed))

    @model_validator(mode="after")
    def _require_action_or_intent(self) -> "StructuredTaskParseOutput":
        if not self.action_type and not self.intent:
            # Allow empty for unknown/ambiguous parses; normalization maps to "unknown".
            return self
        return self

    def to_parse_response(self) -> ParseResponse:
        payload = self.model_dump(exclude_none=True)
        if self.recurrence is not None:
            payload["recurrence"] = self.recurrence.model_dump(exclude_none=True)
        if self.recurrence_update is not None:
            payload["recurrence_update"] = self.recurrence_update.model_dump(exclude_none=True)
        return parse_response_from_llm_dict(payload)

    @classmethod
    def from_llm_dict(cls, data: dict[str, Any]) -> "StructuredTaskParseOutput":
        if not isinstance(data, dict):
            raise ValueError("Model returned JSON that is not an object")
        return cls.model_validate(data)
