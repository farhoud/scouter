from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING, cast

from .client import ChatCompletionOptions, call_llm
from .exceptions import InvalidRunStateError
from .flow import Flow, InputStep, LLMStep, ToolCall, ToolStep
from .memory import MemoryFunction, full_history_memory
from .messages import create_instruction
from .tools import lookup_tool, run_tool

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .types import (
        ChatCompletion,
        ChatCompletionMessageParam,
        ChatCompletionMessageToolCall,
        ChatCompletionToolUnionParam,
    )


logger = logging.getLogger(__name__)

# Constants
TUPLE_INSTRUCTION_LENGTH = 2


# Type for flexible instruction specification
InstructionType = (
    str  # Just system prompt
    | tuple[str, str]  # (system_prompt, user_prompt)
    | list["ChatCompletionMessageParam"]  # Full message list
    | None  # No instructions
)


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
    continue_condition: Callable[[AgentRun], bool] | None = None


@dataclass
class AgentRun:
    continue_condition: Callable[[AgentRun], bool] = field(
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


def _process_instructions(
    instructions: InstructionType,
) -> list[ChatCompletionMessageParam]:
    """Convert instruction specification to message list."""
    if instructions is None:
        return []
    if isinstance(instructions, str):
        # Just system prompt
        return [{"role": "system", "content": instructions}]
    if (
        isinstance(instructions, tuple)
        and len(instructions) == TUPLE_INSTRUCTION_LENGTH
    ):
        return create_instruction(instructions[0], instructions[1])
    if isinstance(instructions, list):
        # Full message list
        return instructions

    msg = f"Invalid instruction format: {type(instructions)}"
    raise ValueError(msg)


def create_agent(config: AgentConfig) -> AgentRun:
    """Create an agent from configuration."""
    # Start with default continue condition if none specified
    continue_cond = config.continue_condition
    if continue_cond is None:
        continue_cond = default_continue_condition_factory()

    return AgentRun(
        memory_function=config.memory_function, continue_condition=continue_cond
    )


async def run_agent(
    agent: AgentRun,
    config: AgentConfig,
    messages: list[ChatCompletionMessageParam] | None = None,
    **options,
) -> AgentRun:
    """Run an agent with configuration."""
    input_messages = messages or []

    # Get tools from registry
    tools = None
    if config.tools:
        tools = [lookup_tool(name).openai_tool_spec() for name in config.tools]

    # Process instructions and combine with input messages
    instruction_messages = _process_instructions(config.instructions)
    all_messages = instruction_messages + input_messages

    # Add initial messages as InputStep to the agent
    initial_flow = Flow(id="initial", agent_id=config.name)
    initial_flow.add_step(InputStep(input=all_messages))
    agent.add_flow(initial_flow)

    # Build options dict, only including max_tokens if set
    flow_options = {"temperature": config.temperature, **options}
    if config.max_tokens is not None:
        flow_options["max_tokens"] = config.max_tokens  # type: ignore[assignment]

    await run_flow(
        agent,
        model=config.model,
        tools=tools,
        options=ChatCompletionOptions(**flow_options),
    )

    return agent
