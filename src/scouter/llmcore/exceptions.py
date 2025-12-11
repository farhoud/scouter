class LLMError(Exception):
    """Base exception for LLM related errors."""


class ToolExecutionError(LLMError):
    """Raised when a tool fails to execute."""


class AgentError(LLMError):
    """Raised when agent operations fail."""


class MaxRetriesExceededError(LLMError):
    """Raised when maximum retry attempts are exceeded."""


class InvalidRunStateError(LLMError):
    """Raised when an agent run is in an invalid state."""


class InvalidToolDefinitionError(LLMError):
    """Raised when a tool is defined incorrectly."""
