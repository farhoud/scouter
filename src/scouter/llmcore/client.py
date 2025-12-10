import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypedDict

from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
)

from .utils import retry_loop


class ChatCompletionOptions(TypedDict, total=False):
    """Options for ChatCompletion API calls.

    Attributes:
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 to 2.0).
        top_p: Nucleus sampling parameter.
        frequency_penalty: Frequency penalty (-2.0 to 2.0).
        presence_penalty: Presence penalty (-2.0 to 2.0).
        stop: List of stop sequences.
    """

    max_tokens: int
    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    stop: list[str]


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
    options: ChatCompletionOptions | None = None,
) -> ChatCompletion:
    """Call the LLM with the given parameters.

    Args:
        model: The model to use.
        messages: List of messages.
        tools: Optional tools.
        options: Optional ChatCompletion options like max_tokens, temperature, etc.
    """

    def _call():
        kwargs = options or {}
        return client.chat.completions.create(
            model=model, messages=messages, tools=tools or [], **kwargs
        )

    return retry_loop(_call)
