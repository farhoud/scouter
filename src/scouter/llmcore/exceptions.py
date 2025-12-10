class LLMError(Exception):
    """Base exception for LLM related errors."""


class ToolExecutionError(LLMError):
    """Raised when a tool fails to execute."""


class AgentError(LLMError):
    """Raised when agent operations fail."""
