"""Structural validation helpers for interpret actions (not semantic repair)."""
from __future__ import annotations

import re
from typing import Optional

from app.capabilities.prompt import load_capability_catalog
from app.models.parse_models import CommandInterpretAction


def infer_input_language(text: str, locale: Optional[str]) -> str:
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    if re.search(r"[\u3040-\u30ff\u31f0-\u31ff]", text):
        return "ja"
    if locale:
        base = locale.split("_")[0].split("-")[0].lower()
        if base in {"zh", "ja", "ko"}:
            return base
    return "en"


def normalize_reminder_offset_minutes(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if value < 0:
        return None
    allowed = [int(v) for v in load_capability_catalog()["reminder_offset_minutes"]]
    if value in allowed:
        return value
    return min(allowed, key=lambda option: abs(option - value))


def drop_invalid_actions(actions: list[CommandInterpretAction]) -> list[CommandInterpretAction]:
    """Remove actions missing required fields from the capability catalog."""
    kept: list[CommandInterpretAction] = []
    for action in actions:
        if action.action_type == "appendToTask":
            append_text = action.edit.append_text if action.edit else None
            if not append_text or not append_text.strip():
                continue
        if action.action_type in {"createReminder", "createEvent"}:
            title = action.create.title if action.create else None
            if not title or not title.strip():
                continue
        kept.append(action)
    return kept or actions


def enforce_catalog_composition(actions: list[CommandInterpretAction]) -> list[CommandInterpretAction]:
    """Apply composition rules from interpreter_capabilities.json."""
    if len(actions) <= 1:
        return actions
    first = actions[0]
    if (
        first.action_type == "rescheduleTask"
        and first.edit
        and first.edit.reminder_offset_minutes
    ):
        disallowed = {"createReminder", "createEvent", "appendToTask"}
        if all(action.action_type in disallowed for action in actions[1:]):
            return [first]
    return actions
