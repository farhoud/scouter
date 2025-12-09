import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolUnionParam,
)

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
    )

from .client import call_llm
from .tools import run_tool


def create_agent(
    model: str = "gpt-5.1-mini",
    tools: Iterable[ChatCompletionToolUnionParam] | None = None,
):
    """
    Returns a function that acts as a functional agent with its own tools and model.
    """

    def agent(
        messages: list[ChatCompletionMessageParam], max_steps: int = 5
    ) -> ChatCompletionMessage:
        steps = 0
        while steps < max_steps:
            msg: ChatCompletionMessage = call_llm(model, messages, tools)
            messages.append(cast("ChatCompletionMessageParam", msg))

            # Handle tool calls
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tc = cast("ChatCompletionMessageToolCall", tc)
                    args = json.loads(tc.function.arguments)
                    output = run_tool(tc.function.name, args)
                    messages.append(
                        cast(
                            "ChatCompletionMessageParam",
                            ChatCompletionToolMessageParam(
                                role="tool", content=output, tool_call_id=tc.id
                            ),
                        )
                    )
            else:
                break
            steps += 1
        return cast("ChatCompletionMessage", messages[-1])

    return agent
