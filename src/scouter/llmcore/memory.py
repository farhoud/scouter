"""Memory functions for configurable agent context."""

from collections.abc import Callable
from typing import TYPE_CHECKING

from .types import ChatCompletionMessageParam

if TYPE_CHECKING:
    from .agent import AgentRuntime


def full_history_memory(run: "AgentRuntime") -> list[ChatCompletionMessageParam]:
    """Memory that includes all conversation history."""
    messages: list[ChatCompletionMessageParam] = []
    for flow in run.flows:
        for step in flow.steps:
            messages.extend(step.messages)
    return messages


MemoryFunction = Callable[["AgentRuntime"], list[ChatCompletionMessageParam]]
