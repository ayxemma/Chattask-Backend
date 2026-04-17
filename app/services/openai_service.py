import json
import logging
from typing import Optional

import httpx

from app.config import OPENAI_API_KEY
from app.models.parse_models import ParseResponse

logger = logging.getLogger(__name__)

OPENAI_BASE_URL = "https://api.openai.com/v1"

PARSE_SYSTEM_PROMPT = """You are a task parsing assistant. Given a natural language task description, extract structured information and respond with a single JSON object — no explanation, no markdown, no extra text.

The JSON must have exactly these fields:
- "action_type": one of "reminder", "task", or "event"
- "title": short, clean title for the task (capitalize properly)
- "notes": any extra detail not captured in the title, or null
- "scheduled_at": ISO 8601 datetime with timezone offset if a time was mentioned, otherwise null
- "confidence": float between 0.0 and 1.0 representing parse confidence
- "language_code": ISO 639-1 language code of the input text (e.g. "en", "es", "zh")

Rules:
- Use the "now" field provided to resolve relative times (e.g. "at 6:15" → same day if in the future)
- Use the "timezone" field to produce the correct UTC offset in "scheduled_at"
- If no time is specified, set "scheduled_at" to null
- Keep "title" concise and human-readable
- Respond with valid JSON only"""


def _auth_headers() -> dict[str, str]:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")
    return {"Authorization": f"Bearer {OPENAI_API_KEY}"}


async def transcribe_audio(file_bytes: bytes, filename: str, content_type: str) -> str:
    """
    Send audio bytes to OpenAI's audio transcription endpoint and return the transcript text.
    """
    headers = _auth_headers()

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{OPENAI_BASE_URL}/audio/transcriptions",
            headers=headers,
            files={"file": (filename, file_bytes, content_type)},
            data={"model": "gpt-4o-mini-transcribe"},
        )

    if response.status_code != 200:
        logger.error("OpenAI transcription error %s: %s", response.status_code, response.text)
        response.raise_for_status()

    result = response.json()
    return result.get("text", "")


async def parse_task_text(text: str, now: str, timezone: str) -> ParseResponse:
    """
    Send task text to OpenAI chat completions and return a structured ParseResponse.
    """
    headers = {**_auth_headers(), "Content-Type": "application/json"}

    user_message = f'Parse this task:\n\nText: "{text}"\nCurrent time (ISO 8601): {now}\nTimezone: {timezone}'

    payload = {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": PARSE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
        )

    if response.status_code != 200:
        logger.error("OpenAI parse error %s: %s", response.status_code, response.text)
        response.raise_for_status()

    raw_content = response.json()["choices"][0]["message"]["content"]

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        logger.error("Failed to decode OpenAI JSON response: %s\nRaw: %s", e, raw_content)
        raise ValueError(f"Model returned invalid JSON: {e}") from e

    return ParseResponse(**data)
