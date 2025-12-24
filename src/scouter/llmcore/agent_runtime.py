from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .exceptions import InvalidRunStateError
from .state import (
    DefaultStateStore,
    Flow,
    LLMStep,
    State,
    StateStore,
    ToolStep,
)
from .types import (
    ChatCompletion,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel


logger = logging.getLogger(__name__)


def memory_trace(data: dict[str, Any]) -> None:
    """Default trace function: memory-only (no-op)."""


@dataclass
class Options:
    persistence: StateStore = field(default_factory=(lambda: DefaultStateStore()))
    tracing_enabled: bool = False
    trace_function: Callable[[dict[str, Any]], None] = memory_trace
    api_key: str | None = None
    track_usage: bool = True
    output_model: type[BaseModel] | None = None


@dataclass
class Runtime:
    options: Options
    state: State = field(default_factory=(lambda: State()))

    def add_flow(self, flow: Flow) -> None:
        """Add a flow to the run."""
        self.state.flows.append(flow)

    @property
    def total_usage(
        self,
    ) -> dict:  # Simplified, can make proper ChatCompletionUsage later
        total = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
        for flow in self.state.flows:
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
        if not self.state:
            msg = "No flows in run"
            logger.error("Attempted to get last output from empty run")
            raise InvalidRunStateError(msg)
        last_flow = self.state.flows[-1]
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

    @property
    def tool_executions(self) -> list[ToolStep]:
        executions = []
        for flow in self.state.flows:
            executions.extend(
                [step for step in flow.steps if isinstance(step, ToolStep)]
            )
        return executions


def default_continue_condition_factory(
    max_steps: int | None = None,
) -> Callable[[Runtime], bool]:
    def condition(run: Runtime) -> bool:
        if max_steps is not None:
            llm_count = sum(
                1
                for flow in run.state.flows
                for step in flow.steps
                if isinstance(step, LLMStep)
            )
            if llm_count >= max_steps:
                return False
        # Find the last step across all flows
        all_steps = [step for flow in run.state.flows for step in flow.steps]
        if not all_steps:
            return True  # No steps yet
        last_step = all_steps[-1]
        return isinstance(last_step, ToolStep)

    return condition
