"""Flow dataclass for grouping steps in agent runs."""

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, cast
from uuid import uuid4

from pydantic import BaseModel

from .types import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
)


@dataclass
class InputStep:
    input: list[ChatCompletionMessageParam]

    @property
    def messages(self) -> list[ChatCompletionMessageParam]:
        return self.input


@dataclass
class LLMStep:
    completion: ChatCompletion | BaseModel

    @property
    def messages(self) -> list[ChatCompletionMessageParam]:
        if isinstance(self.completion, ChatCompletion):
            return [
                cast("ChatCompletionMessageParam", self.completion.choices[0].message)
            ]
        # For structured output, create a message with the JSON content
        content = self.completion.model_dump_json()
        return [{"role": "assistant", "content": content}]  # type: ignore[return-value]


@dataclass
class ToolCall:
    tool_call_id: str
    tool_name: str
    args: dict
    output: str
    execution_time: float
    success: bool
    error_message: str | None

    @property
    def message(self) -> ChatCompletionToolMessageParam:
        return ChatCompletionToolMessageParam(
            role="tool", content=self.output, tool_call_id=self.tool_call_id
        )


@dataclass
class ToolStep:
    calls: list[ToolCall]

    @property
    def messages(self) -> list[ChatCompletionToolMessageParam]:
        return [item.message for item in self.calls]


Step = LLMStep | ToolStep | InputStep


@dataclass
class Flow:
    id: str = field(default_factory=lambda: str(uuid4()))
    agent_id: str = "default"
    steps: list[Step] = field(default_factory=list)
    status: str = "pending"
    metadata: dict = field(default_factory=dict)
    parent_flow_id: str | None = None

    def add_step(self, step: "Step") -> None:
        """Add a step to the flow."""
        self.steps.append(step)

    def mark_running(self) -> None:
        self.status = "running"
        self.metadata["start_time"] = time.time()

    def mark_completed(self) -> None:
        self.status = "completed"
        self.metadata["end_time"] = time.time()

    def mark_failed(self, error: str) -> None:
        self.status = "failed"
        self.metadata["error"] = error
        self.metadata["end_time"] = time.time()


@dataclass
class State:
    id: str = field(default_factory=lambda: str(uuid4()))
    flows: list[Flow] = field(default_factory=list)


class StateStore(Protocol):
    """Protocol for AgentRuntime serialization/deserialization."""

    def store(self, state: State) -> bool:
        """Serialize State"""
        ...

    def load(self, query: dict[str, Any]) -> State:
        """Deserialize an AgentRuntime from a dictionary."""
        ...


class DefaultStateStore:
    """Default implementation of AgentRuntime serialization."""

    def store(self, state: State) -> bool:
        """Serialize an AgentRuntime to a dictionary."""
        return True

    def load(self, query: dict[str, Any]) -> State:
        """Deserialize an AgentRuntime from a dictionary."""
        return State()
