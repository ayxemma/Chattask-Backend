"""OpenAI chat-completions client for task parsing."""

from typing import Optional

from openai import AsyncOpenAI

from app.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_PARSE_MODEL
from app.llm_clients.base import (
    BaseLLMClient,
    ParseTaskContext,
    build_parse_system_content,
    build_parse_user_message,
    parse_llm_json_content,
)
from app.models.parse_models import ParseResponse


class OpenAIClient(BaseLLMClient):
    """Production parser backed by OpenAI (or any OpenAI-compatible endpoint via OPENAI_BASE_URL)."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = OPENAI_API_KEY,
        base_url: str = OPENAI_BASE_URL,
        model: str = OPENAI_PARSE_MODEL,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @property
    def provider_name(self) -> str:
        return "openai"

    async def parse_task(self, context: ParseTaskContext) -> ParseResponse:
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": build_parse_system_content(context.parse_instructions)},
                {"role": "user", "content": build_parse_user_message(context)},
            ],
            response_format={"type": "json_object"},
        )
        raw_content = response.choices[0].message.content or ""
        return parse_llm_json_content(raw_content)
