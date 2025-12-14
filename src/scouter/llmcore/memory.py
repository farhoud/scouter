"""Memory functions for configurable agent context."""

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import AgentRun


def full_history_memory(run) -> list:
    """Memory that includes all conversation history."""
    messages = []
    for flow in run.flows:
        messages.extend(step.message for step in flow["steps"])
    return messages


def summarized_memory(run, max_steps: int = 10) -> list:
    """Memory that keeps recent steps."""
    all_messages = []
    for flow in run.flows:
        all_messages.extend([step.message for step in flow["steps"]])

    if len(all_messages) <= max_steps:
        return all_messages

    # Placeholder: keep last N
    return all_messages[-max_steps:]


def vector_memory(_run) -> list:
    """Placeholder for vector-based memory."""
    # TODO: Implement
    return []


MemoryFunction = Callable[["AgentRun"], list]
