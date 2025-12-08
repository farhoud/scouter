import random
import time
from typing import cast

from openai import APIError, RateLimitError
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

ERROR_MAX_RETRY = "max retries exceeded"


def retry_loop(func, max_retries=5, base_delay=1):
    last_exception: BaseException | None = None

    for attempt in range(max_retries):
        try:
            return func()
        except (RateLimitError, APIError) as e:  # noqa: PERF203
            last_exception = e
            if attempt == max_retries - 1:
                break

            sleep_time = base_delay * (2**attempt) + random.uniform(0, 0.5)  # noqa: S311
            time.sleep(sleep_time)

    # If we reach here, all retries failed
    raise last_exception or RuntimeError(ERROR_MAX_RETRY)


def as_param(msg: ChatCompletionMessage) -> ChatCompletionMessageParam:
    role = msg.role

    # ---------------- USER ----------------
    if role == "user":
        return cast(
            "ChatCompletionUserMessageParam",
            {
                "role": "user",
                "content": msg.content or "",
            },
        )

    # ---------------- SYSTEM ----------------
    if role == "system":
        return cast(
            "ChatCompletionSystemMessageParam",
            {
                "role": "system",
                "content": msg.content or "",
            },
        )

    # ---------------- TOOL ----------------
    if role == "tool":
        # Narrow type: Pyright understands msg.tool_call_id here
        return cast(
            "ChatCompletionToolMessageParam",
            {
                "role": "tool",
                "content": msg.content or "",
                "tool_call_id": msg.tool_call_id,  # valid only for tool messages
            },
        )

    # ---------------- ASSISTANT ----------------
    assistant: ChatCompletionAssistantMessageParam = {
        "role": "assistant",
    }

    if msg.content is not None:
        assistant["content"] = msg.content

    if msg.tool_calls:
        assistant["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]

    return assistant
