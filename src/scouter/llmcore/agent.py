from __future__ import annotations

import asyncio
import json
import logging
from time import time
from typing import TYPE_CHECKING, cast

from .agent_runtime import AgentConfig, AgentRun, default_continue_condition_factory
from .client import ChatCompletionOptions, call_llm, structured_call_llm
from .flow import Flow, InputStep, LLMStep, ToolCall, ToolStep
from .messages import create_instruction
from .tools import lookup_tool, run_tool
from .types import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionToolUnionParam,
    InstructionType,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pydantic import BaseModel


logger = logging.getLogger(__name__)

# Constants
TUPLE_INSTRUCTION_LENGTH = 2


async def run_flow(  # noqa: PLR0913
    run: AgentRun,
    model: str = "gpt-4o-mini",
    tools: Iterable[ChatCompletionToolUnionParam] | None = None,
    options: ChatCompletionOptions | None = None,
    agent_id: str = "default",
    output_model: type[BaseModel] | None = None,
):
    logger.info(
        "Starting agent run with model=%s, initial_flows=%d", model, len(run.flows)
    )
    current_flow = Flow(id=f"{agent_id}_main", agent_id=agent_id)
    current_flow.mark_running()
    run.add_flow(current_flow)

    while run.continue_condition(run):
        context = run.get_context()
        if output_model:
            completion = structured_call_llm(
                model, context, output_model, tools, options
            )
        else:
            completion = call_llm(model, context, tools, options)
        step = LLMStep(completion=completion)
        current_flow.add_step(step)

        # Handle tool calls
        if (
            isinstance(completion, ChatCompletion)
            and completion.choices[0].message.tool_calls
        ):
            msg = completion.choices[0].message
            assert msg.tool_calls is not None
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
        config=config,
        memory_function=config.memory_function,
        continue_condition=continue_cond,
    )


async def run_agent(
    agent: AgentRun,
    config: AgentConfig,
    messages: list[ChatCompletionMessageParam] | None = None,
    output_model: type[BaseModel] | None = None,
    **options,
) -> AgentRun:
    """Run an agent with configuration."""
    agent.config = config  # Attach config for persistence/tracing
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
        output_model=output_model,
    )

    return agent
