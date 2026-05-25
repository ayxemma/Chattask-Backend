from app.capabilities.prompt import build_interpreter_system_prompt, catalog_version, load_capability_catalog


def test_capability_catalog_loads():
    catalog = load_capability_catalog()
    assert catalog["version"] >= 1
    action_types = {item["type"] for item in catalog["actions"]}
    assert "createReminder" in action_types
    assert "rescheduleTask" in action_types
    assert "updateAlertStyle" in action_types


def test_interpreter_prompt_is_compact():
    prompt = build_interpreter_system_prompt()
    assert len(prompt) < 2800
    assert "createReminder" in prompt
    assert "reminder_offset_minutes" in prompt
    assert catalog_version() >= 1
