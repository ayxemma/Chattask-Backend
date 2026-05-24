#!/usr/bin/env python3
"""Verify backend changes: unit tests + optional live interpret regressions.

Usage:
  python3 scripts/verify_changes.py          # pytest only
  python3 scripts/verify_changes.py --live   # pytest + live OpenAI smoke (needs .env key)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SmokeCase:
    name: str
    text: str
    now: str
    timezone: str
    locale: str
    active_task: dict | None
    candidate_tasks: list[dict]
    expect_actions: int
    expect_action_types: list[str] | None = None
    expect_title_contains: str | None = None
    expect_title_not_english: bool = False
    expect_no_clarify: bool = True
    expect_new_hour: int | None = None  # local hour in new_scheduled_at when set


SMOKE_CASES = [
    SmokeCase(
        name="chinese-create-title",
        text="晚上九点的时候提醒我用吸尘器吸一下屋子。",
        now="2026-05-24T16:00:00-04:00",
        timezone="America/New_York",
        locale="en",
        active_task=None,
        candidate_tasks=[],
        expect_actions=1,
        expect_action_types=["createReminder"],
        expect_title_not_english=True,
    ),
    SmokeCase(
        name="reschedule-plus-append",
        text="把四点半去叫醒Ery改成四点五十去叫醒Ery,并且给Ery带点水进去。",
        now="2026-05-24T16:00:00-04:00",
        timezone="America/New_York",
        locale="zh",
        active_task={
            "id": "BB8045FA-75F1-4194-AE4A-6671C748D1BD",
            "title": "叫醒阿瑞",
            "scheduled_at": "2026-05-24T20:30:00+00:00",
        },
        candidate_tasks=[],
        expect_actions=2,
        expect_action_types=["rescheduleTask", "appendToTask"],
    ),
    SmokeCase(
        name="dual-timed-create",
        text="提醒我周二早晨八点给艾瑞做早餐,十一点五十给艾瑞接回来。",
        now="2026-05-24T16:00:00-04:00",
        timezone="America/New_York",
        locale="zh",
        active_task=None,
        candidate_tasks=[],
        expect_actions=2,
        expect_action_types=["createReminder", "createReminder"],
    ),
    SmokeCase(
        name="reschedule-by-hour",
        text="把九点的任务改成十点。",
        now="2026-05-24T16:00:00-04:00",
        timezone="America/New_York",
        locale="en",
        active_task=None,
        candidate_tasks=[
            {
                "id": "D0E14557-E97C-40D2-9DB5-4F9D631CA45C",
                "title": "用吸尘器吸一下屋子",
                "scheduled_at": "2026-05-24T21:00:00-04:00",
                "is_recurring": False,
            }
        ],
        expect_actions=1,
        expect_action_types=["rescheduleTask"],
        expect_new_hour=22,
    ),
    SmokeCase(
        name="simple-relative-create",
        text="十分钟后喝水",
        now="2026-05-24T16:00:00-04:00",
        timezone="America/New_York",
        locale="zh",
        active_task=None,
        candidate_tasks=[],
        expect_actions=1,
        expect_action_types=["createReminder"],
    ),
]


def run_pytest() -> int:
    print("==> pytest tests/")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=ROOT,
    )
    return result.returncode


def _has_live_key() -> bool:
    key = os.getenv("OPENAI_API_KEY", "")
    return bool(key) and not key.startswith("test-fake")


def _is_mostly_english(s: str) -> bool:
    ascii_letters = sum(1 for c in s if c.isascii() and c.isalpha())
    return ascii_letters > len(s) * 0.5


async def run_live_smoke() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    if not _has_live_key():
        print("SKIP live smoke: no real OPENAI_API_KEY after loading .env")
        return 0

    sys.path.insert(0, str(ROOT))
    from app.services.openai_service import interpret_command

    failures = 0
    print("==> live interpret smoke")
    for case in SMOKE_CASES:
        result = await interpret_command(
            text=case.text,
            now=case.now,
            timezone=case.timezone,
            locale=case.locale,
            active_task=case.active_task,
            candidate_tasks=case.candidate_tasks,
            request_id=f"smoke-{case.name}",
        )
        actions = result.actions or []
        ok = True
        reasons: list[str] = []

        if len(actions) != case.expect_actions:
            ok = False
            reasons.append(f"actions={len(actions)} want {case.expect_actions}")

        if case.expect_action_types:
            types = [a.action_type for a in actions]
            if types != case.expect_action_types:
                ok = False
                reasons.append(f"types={types} want {case.expect_action_types}")

        if case.expect_no_clarify and (
            result.confirmation_kind == "clarify"
            or any(a.confirmation_kind == "clarify" for a in actions)
        ):
            ok = False
            reasons.append("unexpected clarify")

        title = (result.create.title if result.create else None) or (
            actions[0].create.title if actions and actions[0].create else None
        )
        if case.expect_title_not_english and title and _is_mostly_english(title):
            ok = False
            reasons.append(f"title translated to English: {title!r}")

        if case.expect_new_hour is not None and result.edit and result.edit.new_scheduled_at:
            from datetime import datetime

            dt = datetime.fromisoformat(result.edit.new_scheduled_at)
            if dt.hour != case.expect_new_hour:
                ok = False
                reasons.append(
                    f"new_scheduled_at hour={dt.hour} want {case.expect_new_hour}"
                )

        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {case.name}" + (f" — {'; '.join(reasons)}" if reasons else ""))
        if not ok:
            failures += 1

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify ChatTask backend changes")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live OpenAI interpret smoke tests (requires .env key)",
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip unit tests (not recommended)",
    )
    args = parser.parse_args()

    if not args.skip_pytest:
        code = run_pytest()
        if code != 0:
            return code

    if args.live:
        failures = asyncio.run(run_live_smoke())
        if failures:
            print(f"\n{failures} live smoke case(s) failed")
            return 1

    print("\nAll verification checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
