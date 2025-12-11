import logging

from .agent import InputStep, Step
from .types import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

logger = logging.getLogger(__name__)


def create_instruction(
    steps: list[Step], system: str | None = None, prompt: str | None = None
) -> None:
    """Add system and user messages to the steps list as InputStep instances."""
    logger.debug(
        "Creating instruction with system=%s, prompt=%s", bool(system), bool(prompt)
    )
    if system:
        steps.append(
            InputStep(
                message=ChatCompletionSystemMessageParam(role="system", content=system)
            )
        )
        logger.debug("Added system message to steps")
    if prompt:
        steps.append(
            InputStep(
                message=ChatCompletionUserMessageParam(role="user", content=prompt)
            )
        )
        logger.debug("Added user message to steps")
