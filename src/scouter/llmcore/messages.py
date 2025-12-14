"""Message creation utilities."""

from openai.types.chat import ChatCompletionMessageParam


def create_instruction(
    system_prompt: str, user_prompt: str
) -> list[ChatCompletionMessageParam]:
    """Create instruction messages."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})
    return messages
