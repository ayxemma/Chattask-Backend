"""SGLang OpenAI-compatible client for experimental task parsing."""

from typing import Optional

from openai import AsyncOpenAI

from app.config import SGLANG_API_KEY, SGLANG_BASE_URL, SGLANG_MODEL
from app.llm_clients.base import (
    BaseLLMClient,
    ParseTaskContext,
    build_parse_system_content,
    build_parse_user_message,
    parse_llm_json_content,
)
from app.models.parse_models import ParseResponse


class SGLangClient(BaseLLMClient):
    """
    Experimental parser backed by an external SGLang server exposing the OpenAI API.

    Example server setup (not run by this repo):

        client = OpenAI(api_key=..., base_url="http://sglang-server/v1")
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str] = SGLANG_API_KEY,
        model: str = SGLANG_MODEL,
    ) -> None:
        if not base_url:
            raise ValueError("SGLANG_BASE_URL is not set")
        if not model:
            raise ValueError("SGLANG_MODEL is not set")
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key or "not-needed",
            base_url=base_url.rstrip("/"),
        )

    @property
    def provider_name(self) -> str:
        return "sglang"

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
