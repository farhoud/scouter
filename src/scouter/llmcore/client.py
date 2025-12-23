import json
import logging
from collections.abc import Iterable
from functools import lru_cache
from typing import TypedDict

from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
)
from pydantic import BaseModel

from scouter.config import config

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
        response_format: Response format specification.
        api_key: Client API key
        base_url: Base Url of Client
    """

    max_tokens: int
    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    stop: list[str]
    response_format: dict
    api_key: str | None
    base_url: str | None


@lru_cache(maxsize=1)
def get_llm_client() -> OpenAI:
    """Get a singleton LLM client."""
    return OpenAI(
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        timeout=config.llm.timeout,
        max_retries=config.llm.max_retries,
    )


@lru_cache(maxsize=32)  # Cache up to 32 different user clients
def get_user_llm_client(api_key: str, base_url: str | None = None) -> OpenAI:
    """Get an LLM client for a specific user API key."""
    return OpenAI(
        api_key=api_key,
        base_url=base_url or config.llm.base_url,
        timeout=config.llm.timeout,
        max_retries=config.llm.max_retries,
    )


client = get_llm_client()


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
        api_key: Optional user API key to use instead of global.
    """
    tools_count = sum(1 for _ in tools) if tools else 0
    logger.debug(
        "Calling LLM with model=%s, message_count=%d, tools_count=%d",
        model,
        len(messages),
        tools_count,
    )

    client = (
        get_user_llm_client(options.get("api_key"), options.get("base_url"))
        if options and options.get("api_key")
        else get_llm_client()
    )

    def _call():
        kwargs = options or {}
        return client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools or [],
            **kwargs,  # type: ignore[arg-type]
        )

    result = retry_loop(_call)
    logger.debug("LLM call completed successfully")
    return result


def structured_call_llm(
    model: str,
    messages: list[ChatCompletionMessageParam],
    output_model: type[BaseModel],
    tools: Iterable[ChatCompletionToolUnionParam] | None = None,
    options: ChatCompletionOptions | None = None,
) -> BaseModel:
    """
    Call the LLM with structured output, returning a validated Pydantic model.

    Args:
        model: The model to use.
        messages: List of messages.
        output_model: Pydantic model class for the expected output.
        tools: Optional tools.
        options: Optional ChatCompletion options.
        api_key: Optional user API key to use instead of global.

    Returns:
        An instance of output_model with validated data.

    Raises:
        ValueError: If JSON parsing or model validation fails.
    """
    schema = output_model.model_json_schema()
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "schema": schema,
            "strict": True,
        },
    }

    kwargs = options or {}
    kwargs["response_format"] = response_format  # type: ignore[assignment]

    tools_count = sum(1 for _ in tools) if tools else 0
    logger.debug(
        "Calling LLM with structured output: model=%s, message_count=%d, tools_count=%d, output_model=%s",
        model,
        len(messages),
        tools_count,
        output_model.__name__,
    )

    client = (
        get_user_llm_client(options.get("api_key"))
        if options and options.get("api_key")
        else get_llm_client()
    )

    def _call():
        return client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools or [],
            **kwargs,  # type: ignore[arg-type]
        )

    completion = retry_loop(_call)
    content = completion.choices[0].message.content
    if not content:
        msg = "LLM returned empty content for structured output"
        raise ValueError(msg)

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        msg = f"Failed to parse LLM response as JSON: {e}"
        logger.exception(msg)
        raise ValueError(msg) from e

    try:
        result = output_model(**data)
    except Exception as e:
        msg = f"Failed to validate LLM response against {output_model.__name__}: {e}"
        logger.exception(msg)
        raise ValueError(msg) from e

    logger.debug(
        "Structured LLM call completed successfully, returned %s", output_model.__name__
    )
    return result
