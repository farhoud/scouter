from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING, cast

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolUnionParam,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
    )

from .client import ChatCompletionOptions, call_llm
from .exceptions import InvalidRunStateError
from .tools import run_tool

logger = logging.getLogger(__name__)


@dataclass
class InputStep:
    message: ChatCompletionMessageParam


@dataclass
class LLMStep:
    completion: ChatCompletion

    @property
    def message(self) -> ChatCompletionMessageParam:
        return cast("ChatCompletionMessageParam", self.completion.choices[0].message)


@dataclass
class ToolStep:
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


Step = InputStep | LLMStep | ToolStep


@dataclass
class AgentRun:
    continue_condition: Callable[[AgentRun], bool] = field(
        default_factory=lambda: default_continue_condition_factory()
    )
    steps: list[Step] = field(default_factory=list)

    def add_step(self, step: Step) -> None:
        """Add a step to the run."""
        self.steps.append(step)

    @property
    def conversation_history(self) -> list[ChatCompletionMessageParam]:
        return [step.message for step in self.steps]

    @property
    def total_usage(
        self,
    ) -> dict:  # Simplified, can make proper ChatCompletionUsage later
        total = {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
        for step in self.steps:
            if isinstance(step, LLMStep) and step.completion.usage:
                usage = step.completion.usage
                total["completion_tokens"] += usage.completion_tokens or 0
                total["prompt_tokens"] += usage.prompt_tokens or 0
                total["total_tokens"] += usage.total_tokens or 0
        return total

    @property
    def last_output(self) -> str:
        if not self.steps:
            msg = "No steps in run"
            logger.error("Attempted to get last output from empty run")
            raise InvalidRunStateError(msg)
        last_step = self.steps[-1]
        if isinstance(last_step, LLMStep):
            content = last_step.message.get("content")
            return content if isinstance(content, str) else ""
        if isinstance(last_step, ToolStep):
            return last_step.output
        return ""

    @property
    def tool_executions(self) -> list[ToolStep]:
        return [step for step in self.steps if isinstance(step, ToolStep)]


def default_continue_condition_factory(
    max_steps: int | None = None,
) -> Callable[[AgentRun], bool]:
    def condition(run: AgentRun) -> bool:
        if max_steps is not None:
            llm_count = sum(1 for step in run.steps if isinstance(step, LLMStep))
            if llm_count >= max_steps:
                return False
        # Filter out InputStep to find the last meaningful step
        non_input_steps = [
            step for step in run.steps if not isinstance(step, InputStep)
        ]
        if not non_input_steps:
            return True  # Only InputSteps present, initial state
        last_non_input = non_input_steps[-1]
        return isinstance(last_non_input, ToolStep)

    return condition


def run_agent(
    run: AgentRun,
    model: str = "gpt-4o-mini",
    tools: Iterable[ChatCompletionToolUnionParam] | None = None,
    options: ChatCompletionOptions | None = None,
):
    logger.info(
        "Starting agent run with model=%s, initial_steps=%d", model, len(run.steps)
    )
    while run.continue_condition(run):
        completion: ChatCompletion = call_llm(
            model, run.conversation_history, tools, options
        )
        msg = completion.choices[0].message
        run.add_step(LLMStep(completion))

        # Handle tool calls
        if msg.tool_calls:
            logger.debug("Processing %d tool calls", len(msg.tool_calls))
            for tc in msg.tool_calls:
                tc = cast("ChatCompletionMessageToolCall", tc)
                args = json.loads(tc.function.arguments)
                logger.debug(
                    "Executing tool '%s' with args: %s", tc.function.name, args
                )
                start = time()
                try:
                    output = run_tool(tc.function.name, args)
                    success = True
                    error = None
                    logger.debug("Tool '%s' executed successfully", tc.function.name)
                except Exception as e:  # noqa: BLE001
                    output = ""
                    success = False
                    error = str(e)
                    logger.warning(
                        "Tool '%s' execution failed: %s", tc.function.name, str(e)
                    )
                end = time()
                run.add_step(
                    ToolStep(
                        tc.id,
                        tc.function.name,
                        args,
                        output,
                        end - start,
                        success,
                        error,
                    )
                )
    logger.info("Agent run completed with %d total steps", len(run.steps))
