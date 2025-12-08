import os
from collections.abc import Iterable
from dataclasses import dataclass

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
)

from .utils import retry_loop


@dataclass(slots=True)
class LLMConfig:
    api_key: str | None = None
    base_url: str | None = None
    timeout: int = 30
    max_retries: int = 3

    @staticmethod
    def load_from_env() -> "LLMConfig":
        return LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )


def create_llm_client(cfg: LLMConfig | None = None) -> OpenAI:
    cfg = cfg or LLMConfig.load_from_env()

    return OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
    )


client = create_llm_client()


def call_llm(
    model: str,
    messages: list[ChatCompletionMessageParam],
    tools: Iterable[ChatCompletionToolUnionParam] | None = None,
) -> ChatCompletionMessage:
    def _call():
        return client.chat.completions.create(
            model=model, messages=messages, tools=tools or []
        )

    res = retry_loop(_call)
    return res.choices[0].message
