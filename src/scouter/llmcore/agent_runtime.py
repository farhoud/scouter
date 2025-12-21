from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel

from .exceptions import InvalidRunStateError
from .flow import Flow, InputStep, LLMStep, ToolStep
from .memory import MemoryFunction, full_history_memory
from .types import (
    ChatCompletion,
    ChatCompletionMessageParam,
    InstructionType,
)

if TYPE_CHECKING:
    from collections.abc import Callable


logger = logging.getLogger(__name__)


class AgentRuntimeSerializer(Protocol):
    """Protocol for AgentRuntime serialization/deserialization."""

    def serialize(self, agent_runtime: AgentRuntime) -> dict[str, Any]:
        """Serialize an AgentRuntime to a dictionary."""
        ...

    def deserialize(self, data: dict[str, Any]) -> AgentRuntime:
        """Deserialize an AgentRuntime from a dictionary."""
        ...


class DefaultAgentRuntimeSerializer:
    """Default implementation of AgentRuntime serialization."""

    def serialize(self, agent_runtime: AgentRuntime) -> dict[str, Any]:
        """Serialize an AgentRuntime to a dictionary."""
        return serialize_agent_runtime(agent_runtime)

    def deserialize(self, data: dict[str, Any]) -> AgentRuntime:
        """Deserialize an AgentRuntime from a dictionary."""
        return deserialize_agent_runtime(data)


# Global serializer instance
agent_runtime_serializer = DefaultAgentRuntimeSerializer()


def memory_persistence(run: AgentRuntime) -> None:
    """Default persistence function: memory-only (no-op)."""


def memory_trace(data: dict[str, Any]) -> None:
    """Default trace function: memory-only (no-op)."""


@dataclass
class AgentConfig:
    """Configuration for agent creation."""

    name: str = "default"
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int | None = None
    instructions: InstructionType = None
    tools: list[str] | None = None  # Tool names
    memory_function: MemoryFunction = full_history_memory
    continue_condition: Callable[[AgentRuntime], bool] | None = None
    persistence_function: Callable[[AgentRuntime], None] = memory_persistence
    tracing_enabled: bool = False
    trace_function: Callable[[dict[str, Any]], None] = memory_trace
    api_key: str | None = None
    track_usage: bool = True


@dataclass
class AgentRuntime:
    config: AgentConfig
    continue_condition: Callable[[AgentRuntime], bool] = field(
        default_factory=lambda: default_continue_condition_factory()
    )
    flows: list[Flow] = field(default_factory=list)
    memory_function: MemoryFunction = field(default=full_history_memory)

    def add_flow(self, flow: Flow) -> None:
        """Add a flow to the run."""
        self.flows.append(flow)

    def get_context(self) -> list[ChatCompletionMessageParam]:
        """Get configurable memory context instead of flat history."""
        return self.memory_function(self)

    @property
    def total_usage(
        self,
    ) -> dict:  # Simplified, can make proper ChatCompletionUsage later
        if not self.config.track_usage:
            return {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
        total = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
        for flow in self.flows:
            for step in flow.steps:
                if (
                    isinstance(step, LLMStep)
                    and isinstance(step.completion, ChatCompletion)
                    and step.completion.usage
                ):
                    usage = step.completion.usage
                    total["completion_tokens"] += usage.completion_tokens or 0
                    total["prompt_tokens"] += usage.prompt_tokens or 0
                    total["total_tokens"] += usage.total_tokens or 0
        return total

    @property
    def last_output(self) -> str:
        if not self.flows:
            msg = "No flows in run"
            logger.error("Attempted to get last output from empty run")
            raise InvalidRunStateError(msg)
        last_flow = self.flows[-1]
        if not last_flow.steps:
            return ""
        last_step = last_flow.steps[-1]
        if isinstance(last_step, LLMStep):
            if isinstance(last_step.completion, ChatCompletion):
                content = last_step.completion.choices[0].message.content
                return content if content else ""
            # Structured output
            return last_step.completion.model_dump_json()
        if isinstance(last_step, ToolStep):
            return str(last_step.messages)
        return ""

    def save(self) -> None:
        """Save the agent run using the configured persistence function."""
        self.config.persistence_function(self)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the agent run to a dictionary."""
        return agent_runtime_serializer.serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentRuntime:
        """Deserialize an agent run from a dictionary."""
        return agent_runtime_serializer.deserialize(data)

    @property
    def tool_executions(self) -> list[ToolStep]:
        executions = []
        for flow in self.flows:
            executions.extend(
                [step for step in flow.steps if isinstance(step, ToolStep)]
            )
        return executions


def default_continue_condition_factory(
    max_steps: int | None = None,
) -> Callable[[AgentRuntime], bool]:
    def condition(run: AgentRuntime) -> bool:
        if max_steps is not None:
            llm_count = sum(
                1
                for flow in run.flows
                for step in flow.steps
                if isinstance(step, LLMStep)
            )
            if llm_count >= max_steps:
                return False
        # Find the last step across all flows
        all_steps = [step for flow in run.flows for step in flow.steps]
        if not all_steps:
            return True  # No steps yet
        last_step = all_steps[-1]
        return isinstance(last_step, ToolStep)

    return condition


def serialize_agent_runtime(agent_run: AgentRuntime) -> dict[str, Any]:
    """Serialize an AgentRuntime to a dictionary.

    Args:
        agent_run: The AgentRuntime instance to serialize.

    Returns:
        A dictionary representation of the AgentRuntime.
    """
    # Serialize flows
    flows_data = []
    for flow in agent_run.flows:
        steps_data = []
        for step in flow.steps:
            step_data = {
                "type": type(step).__name__,
                "id": getattr(step, "id", None),
            }
            if isinstance(step, LLMStep):
                if isinstance(step.completion, BaseModel):
                    step_data["completion"] = {
                        "type": "BaseModel",
                        "class": step.completion.__class__.__name__,
                        "data": step.completion.model_dump(),
                    }
                else:
                    # Assume ChatCompletion or similar
                    step_data["completion"] = {
                        "type": "ChatCompletion",
                        "model": getattr(step.completion, "model", None),
                        "choices": [
                            {
                                "message": {
                                    "role": getattr(choice.message, "role", None),
                                    "content": getattr(choice.message, "content", None),
                                    "tool_calls": getattr(
                                        choice.message, "tool_calls", None
                                    ),
                                }
                            }
                            for choice in getattr(step.completion, "choices", [])
                        ],
                        "usage": getattr(step.completion, "usage", None),
                    }
            elif isinstance(step, ToolStep):
                step_data["calls"] = [
                    {
                        "tool_call_id": call.tool_call_id,
                        "tool_name": call.tool_name,
                        "args": call.args,
                        "output": call.output,
                        "execution_time": call.execution_time,
                        "success": call.success,
                    }
                    for call in step.calls
                ]
            elif isinstance(step, InputStep):
                step_data["input"] = step.input

            steps_data.append(step_data)

        flow_data = {
            "id": flow.id,
            "status": flow.status,
            "metadata": flow.metadata,
            "steps": steps_data,
        }
        flows_data.append(flow_data)

    # Serialize memory (current context)
    memory_data = agent_run.memory_function(agent_run)

    return {
        "flows": flows_data,
        "memory": memory_data,
        "continue_condition_name": getattr(
            agent_run.continue_condition, "__name__", "unknown"
        ),
    }


def deserialize_agent_runtime(data: dict[str, Any]) -> AgentRuntime:
    """Deserialize an AgentRuntime from a dictionary.

    Args:
        data: The dictionary representation of the AgentRuntime.

    Returns:
        The reconstructed AgentRuntime instance.
    """
    # This is a simplified deserialization - in practice, you'd need to reconstruct
    # the full objects. For now, return a basic AgentRuntime.

    # Reconstruct flows (simplified)
    flows = []
    for flow_data in data.get("flows", []):
        flow = Flow(id=flow_data["id"])
        flow.status = flow_data["status"]
        flow.metadata = flow_data["metadata"]
        # Steps reconstruction would be more complex
        flows.append(flow)

    # Use a basic config for deserialization
    config = AgentConfig()
    return AgentRuntime(
        config=config,
        flows=flows,
        memory_function=full_history_memory,
    )
