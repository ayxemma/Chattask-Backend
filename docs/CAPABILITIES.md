# ChatTask voice command capabilities

Source of truth for `/interpret-command`. The LLM maps user speech to these actions; the iOS app executes them.

**Machine-readable catalog:** [`app/capabilities/interpreter_capabilities.json`](../app/capabilities/interpreter_capabilities.json)  
**System prompt:** built at runtime by [`app/capabilities/prompt.py`](../app/capabilities/prompt.py) (keep prompt short for latency)

## Supported actions

| App feature | `action_type` | Primary fields |
|---|---|---|
| Create reminder | `createReminder` | `create.title`, `create.scheduled_at`, `create.reminder_offset_minutes`, `create.alert_style`, recurrence |
| Create calendar event | `createEvent` | `create.title`, `create.scheduled_at`, `create.end_at` |
| Change time | `rescheduleTask` | `target`, `edit.new_scheduled_at`, `edit.reminder_offset_minutes` |
| Change title | `renameTask` | `target`, `edit.new_title` |
| Add note | `appendToTask` | `target`, `edit.append_text` |
| Delete task | `deleteTask` | `target` |
| Change repeat schedule | `updateRecurrence` | `target`, `edit.new_recurrence_*` |
| Mark important / normal alert | `updateAlertStyle` | `target`, `edit.alert_style` |

### Reminder lead time (`reminder_offset_minutes`)

Minutes **before** `scheduled_at` to fire the notification. Allowed values: **0, 5, 10, 15, 30, 60**.

- `0` = at task time  
- Phrases like “提前十分钟提醒我” set this on **reschedule** or **create**, not a second task.

### Alert style

- `normal` — default / silent / 普通提醒  
- `important` — louder / 重要提醒  

## Multi-action composition

| User intent | Expected `actions[]` |
|---|---|
| Two timed reminders in one sentence | 2× `createReminder` |
| Reschedule + add detail to same task | `rescheduleTask`, `appendToTask` |
| Reschedule + remind X min before | 1× `rescheduleTask` with `reminder_offset_minutes` |
| Reschedule + separate new reminder | `rescheduleTask`, `createReminder` (only if user clearly wants a new task) |

When ambiguous (note vs new reminder), return `confirmation_kind=clarify`.

## Backend validation (structural only)

The server **does not** guess user intent from keywords. It only:

- Validates task IDs against `candidate_tasks` / `active_task`
- Normalizes `action_type` spelling and `alert_style` enums
- Snaps `reminder_offset_minutes` to the nearest allowed value
- Expands one action that combines `new_scheduled_at` + `append_text` into two executable actions
- Drops actions missing required fields
- Enforces catalog composition (e.g. remind-before reschedule cannot also append/create)
- Promotes a lone top-level action into `actions[]`

Semantic parsing is the LLM’s job, guarded by [`scripts/verify_changes.py`](../scripts/verify_changes.py).

## Adding a new app feature (checklist)

1. Implement UI + executor in **ChatTask iOS**.
2. Add action/fields to **`interpreter_capabilities.json`**.
3. Add 1–2 **examples** in the JSON `examples` array (prompt stays small).
4. Extend **`CommandInterpretEdit` / `Create`** models if new fields are needed.
5. Add **unit test** in `tests/test_interpret_command.py` for response shape.
6. Add **smoke case** in `scripts/verify_changes.py`.
7. Run `python3 scripts/verify_changes.py --live` before merging.

Do **not** add phrase-matching repair logic in `openai_service.py` unless it is pure structural normalization (like compound field expansion).

## Regression smoke scenarios

See [`.cursor/rules/backend-change-verification.mdc`](../.cursor/rules/backend-change-verification.mdc).
