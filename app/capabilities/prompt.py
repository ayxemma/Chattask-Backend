"""Build the interpret-command system prompt from the capability catalog."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_CATALOG_PATH = Path(__file__).with_name("interpreter_capabilities.json")


@lru_cache(maxsize=1)
def load_capability_catalog() -> dict[str, Any]:
    with _CATALOG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_interpreter_system_prompt() -> str:
    catalog = load_capability_catalog()
    action_line = "; ".join(
        f"{item['type']}({','.join(item['fields'])})"
        for item in catalog["actions"]
    )
    offsets = ",".join(str(v) for v in catalog["reminder_offset_minutes"])
    styles = "|".join(catalog["alert_styles"])
    rules = "\n".join(f"- {rule}" for rule in catalog["rules"])
    examples = "\n".join(f"- {example}" for example in catalog["examples"])
    schema = (
        '{"actions":[{"action_type":"...","confidence":0,"requires_confirmation":false,'
        '"confirmation_kind":"none|confirm_action|choose_candidate|clarify","assistant_message":"...",'
        '"target":{"resolution":"active_task|candidate|ambiguous|none","selected_task_id":null,'
        '"candidate_ids":null},"create":{"title":null,"scheduled_at":null,"reminder_offset_minutes":null,'
        '"alert_style":null,"recurrence_type":"none","recurrence_weekdays":null},'
        '"edit":{"new_scheduled_at":null,"new_title":null,"append_text":null,'
        '"reminder_offset_minutes":null,"alert_style":null,"new_recurrence_type":null,'
        '"new_recurrence_weekdays":null}}],"action_type":"...","confidence":0,'
        '"requires_confirmation":false,"confirmation_kind":"none","target":{},"create":{},"edit":{}}'
    )
    return (
        "ChatTask command interpreter. Valid JSON only.\n\n"
        f"Actions: {action_line}\n"
        f"reminder_offset_minutes: {offsets}. alert_style: {styles}. Weekdays Mon=1..Sun=7.\n\n"
        f"Schema:\n{schema}\n\n"
        f"Rules:\n{rules}\n\n"
        f"Examples:\n{examples}"
    )


def catalog_version() -> int:
    return int(load_capability_catalog().get("version", 0))
