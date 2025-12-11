import logging

from .types import Prompt

logger = logging.getLogger(__name__)


def resolve_prompt(prompt: Prompt, *args, **kwargs) -> str:
    """Resolve a Prompt to a string, executing callables with optional args if necessary."""
    if isinstance(prompt, str):
        logger.debug("Resolved string prompt directly")
        return prompt
    logger.debug("Executing callable prompt")
    result = prompt(*args, **kwargs)
    logger.debug("Callable prompt executed successfully")
    return result
