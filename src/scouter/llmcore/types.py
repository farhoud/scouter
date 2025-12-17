from collections.abc import Callable

# Re-export OpenAI types
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionToolUnionParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

# Custom types
Prompt = str | Callable[[], str]

# Type for flexible instruction specification
InstructionType = (
    str  # Just system prompt
    | tuple[str, str]  # (system_prompt, user_prompt)
    | list["ChatCompletionMessageParam"]  # Full message list
    | None  # No instructions
)

__all__ = [
    "ChatCompletion",
    "ChatCompletionAssistantMessageParam",
    "ChatCompletionMessage",
    "ChatCompletionMessageParam",
    "ChatCompletionMessageToolCall",
    "ChatCompletionMessageToolCall",
    "ChatCompletionSystemMessageParam",
    "ChatCompletionToolMessageParam",
    "ChatCompletionToolParam",
    "ChatCompletionToolUnionParam",
    "ChatCompletionUserMessageParam",
    "InstructionType",
    "Prompt",
]
