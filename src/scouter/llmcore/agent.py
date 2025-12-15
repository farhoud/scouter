from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING, cast

from .client import ChatCompletionOptions, call_llm
from .exceptions import InvalidRunStateError
from .flow import Flow, LLMStep, ToolCall, ToolStep
from .memory import MemoryFunction, full_history_memory
from .tools import run_tool

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .types import (
        ChatCompletion,
        ChatCompletionMessageParam,
        ChatCompletionMessageToolCall,
        ChatCompletionToolUnionParam,
    )


logger = logging.getLogger(__name__)


@dataclass
class AgentRun:
    continue_condition: Callable[[AgentRun], bool] = field(
        default_factory=lambda: default_continue_condition_factory()
    )
    flows: list[Flow] = field(default_factory=list)
    memory_function: MemoryFunction = field(default=full_history_memory)
    agents: dict[str, Callable[[], AgentRun]] = field(
        default_factory=dict
    )  # For multi-agent: factory functions

    def add_flow(self, flow: Flow) -> None:
        """Add a flow to the run."""
        self.flows.append(flow)

    def get_context(self) -> list[ChatCompletionMessageParam]:
        """Get configurable memory context instead of flat history."""
        return self.memory_function(self)

    def run_sub_agent(self, agent_id: str) -> Flow:
        """Run a sub-agent within this run, returning its flow."""
        if agent_id not in self.agents:
            msg = f"Agent {agent_id} not registered"
            raise ValueError(msg)
        flow = Flow(id=f"{agent_id}_{len(self.flows)}", agent_id=agent_id)
        flow.mark_running()
        self.add_flow(flow)
        # TODO: Integrate with run_agent for actual execution
        # For now, placeholder: assume sub_run executes and adds steps to flow
        flow.mark_completed()
        return flow

    @property
    def total_usage(
        self,
    ) -> dict:  # Simplified, can make proper ChatCompletionUsage later
        total = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
        for flow in self.flows:
            for step in flow.steps:
                if isinstance(step, LLMStep) and step.completion.usage:
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
            content = last_step.completion.choices[0].message.content
            return content if content else ""
        if isinstance(last_step, ToolStep):
            return str(last_step.messages)
        return ""

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
) -> Callable[[AgentRun], bool]:
    def condition(run: AgentRun) -> bool:
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


async def run_flow(
    run: AgentRun,
    model: str = "gpt-4o-mini",
    tools: Iterable[ChatCompletionToolUnionParam] | None = None,
    options: ChatCompletionOptions | None = None,
    agent_id: str = "default",
):
    logger.info(
        "Starting agent run with model=%s, initial_flows=%d", model, len(run.flows)
    )
    current_flow = Flow(id=f"{agent_id}_main", agent_id=agent_id)
    current_flow.mark_running()
    run.add_flow(current_flow)

    while run.continue_condition(run):
        context = run.get_context()
        completion: ChatCompletion = call_llm(model, context, tools, options)
        msg = completion.choices[0].message
        step = LLMStep(completion=completion)
        current_flow.add_step(step)

        # Handle tool calls
        if msg.tool_calls:
            logger.debug("Processing %d tool calls", len(msg.tool_calls))
            tool_calls = [
                cast("ChatCompletionMessageToolCall", tc) for tc in msg.tool_calls
            ]

            # Prepare async tasks for parallel execution
            async def execute_single_tool(tc: ChatCompletionMessageToolCall):
                args = json.loads(tc.function.arguments)
                logger.debug(
                    "Executing tool '%s' with args: %s", tc.function.name, args
                )
                start = time()
                output = ""
                success = False
                error = None
                try:
                    output = await run_tool(tc.function.name, args)
                    success = True
                    logger.debug("Tool '%s' executed successfully", tc.function.name)
                except Exception as e:  # noqa: BLE001
                    error = str(e)
                    logger.warning(
                        "Tool '%s' execution failed: %s", tc.function.name, str(e)
                    )
                end = time()
                return ToolCall(
                    tool_call_id=tc.id,
                    tool_name=tc.function.name,
                    args=args,
                    output=output,
                    execution_time=end - start,
                    success=success,
                    error_message=error,
                )

            # Execute all tools concurrently
            tasks = [execute_single_tool(tc) for tc in tool_calls]
            tool_steps = await asyncio.gather(*tasks, return_exceptions=True)
            success: list[ToolCall] = []
            # Add steps to current flow
            for i, result in enumerate(tool_steps):
                if isinstance(result, Exception):
                    # Handle unexpected errors in gather
                    tc = tool_calls[i]
                    logger.error(
                        "Unexpected error in tool execution for '%s': %s",
                        tc.function.name,
                        result,
                    )
                elif isinstance(result, ToolCall):
                    success.append(result)
            current_flow.add_step(ToolStep(calls=success))
    current_flow.mark_completed()
    logger.info("Agent run completed with %d total flows", len(run.flows))
