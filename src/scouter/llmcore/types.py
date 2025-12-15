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
    "Prompt",
]
