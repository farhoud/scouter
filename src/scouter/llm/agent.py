import json
from collections.abc import Iterable

from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolUnionParam,
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
            messages.append(msg)

            # Handle tool calls
            if getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    output = run_tool(tc.function.name, args)
                    messages.append(
                        ChatCompletionMessage(
                            role="tool", content=output, tool_call_id=tc.id
                        )
                    )
            else:
                break
            steps += 1
        return messages[-1]

    return agent
