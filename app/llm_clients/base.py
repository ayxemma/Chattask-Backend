"""Base types for LLM task parsing providers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from app.llm_clients.prompts import PARSE_SYSTEM_PROMPT
from app.models.llm_parse_output import StructuredTaskParseOutput
from app.models.parse_models import ParseResponse


@dataclass(frozen=True)
class ParseTaskContext:
    text: str
    now: str
    timezone: str
    locale: Optional[str] = None
    parse_instructions: Optional[str] = None
    source: Optional[str] = None
    last_active_task_id: Optional[str] = None
    active_task_title: Optional[str] = None
    active_task_scheduled_at: Optional[str] = None
    active_task_notes: Optional[str] = None
    active_task_recurrence: Optional[dict[str, Any]] = None


def truncate_notes(s: Optional[str], max_len: int = 1200) -> Optional[str]:
    if s is None:
        return None
    t = s.strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def build_parse_system_content(parse_instructions: Optional[str]) -> str:
    system_content = PARSE_SYSTEM_PROMPT
    if parse_instructions and parse_instructions.strip():
        system_content = (
            f"{PARSE_SYSTEM_PROMPT}\n\n---\nClient parsing hints (follow when consistent with rules above):\n"
            f"{parse_instructions.strip()}"
        )
    return system_content


def build_parse_user_message(context: ParseTaskContext) -> str:
    user_lines = [
        f'Parse this task:\n\nText: "{context.text}"',
        f"Current time (ISO 8601): {context.now}",
        f"Timezone (IANA): {context.timezone}",
    ]
    if context.locale:
        user_lines.append(f"Locale: {context.locale}")
    if context.source:
        user_lines.append(f"Input source: {context.source}")
    if context.last_active_task_id and str(context.last_active_task_id).strip():
        user_lines.append("")
        user_lines.append("Active task context (for follow-up commands; user may refer to this task without naming it):")
        user_lines.append(f"- task_id: {context.last_active_task_id.strip()}")
        user_lines.append('- target_reference_type to use for active-task follow-ups: "recent_task"')
        user_lines.append(f"- target_task_id to use for active-task follow-ups: {context.last_active_task_id.strip()}")
        if context.active_task_title:
            user_lines.append(f'- title: "{context.active_task_title.strip()}"')
        if context.active_task_scheduled_at:
            user_lines.append(f"- scheduled_at (ISO 8601): {context.active_task_scheduled_at.strip()}")
        if context.active_task_recurrence:
            user_lines.append(
                f"- recurrence: {json.dumps(context.active_task_recurrence, ensure_ascii=False)}"
            )
        notes_ctx = truncate_notes(context.active_task_notes)
        if notes_ctx:
            user_lines.append(f"- notes (may be truncated): {notes_ctx}")
    return "\n".join(user_lines)


def parse_llm_json_content(raw_content: str) -> ParseResponse:
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned invalid JSON: {e}") from e
    validated = StructuredTaskParseOutput.from_llm_dict(data)
    return validated.to_parse_response()


class BaseLLMClient(ABC):
    """Provider interface for structured task parsing."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short provider label used in logs (e.g. 'openai', 'sglang')."""

    @abstractmethod
    async def parse_task(self, context: ParseTaskContext) -> ParseResponse:
        """Parse natural language into a structured ParseResponse."""
