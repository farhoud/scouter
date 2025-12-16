import logging
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

logger = logging.getLogger(__name__)


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
        provider = os.getenv("LLM_PROVIDER", "openai")
        if provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
        else:
            # Default to openai for backward compatibility
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")

        return LLMConfig(
            api_key=api_key,
            base_url=base_url,
        )


def create_llm_client(cfg: LLMConfig | None = None) -> OpenAI:
    cfg = cfg or LLMConfig.load_from_env()
    logger.debug(
        "Creating LLM client with timeout=%d, max_retries=%d",
        cfg.timeout,
        cfg.max_retries,
    )

    client = OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
    )
    logger.info("LLM client created successfully")
    return client


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
    tools_count = sum(1 for _ in tools) if tools else 0
    logger.debug(
        "Calling LLM with model=%s, message_count=%d, tools_count=%d",
        model,
        len(messages),
        tools_count,
    )

    def _call():
        kwargs = options or {}
        return client.chat.completions.create(
            model=model, messages=messages, tools=tools or [], **kwargs
        )

    result = retry_loop(_call)
    logger.debug("LLM call completed successfully")
    return result
