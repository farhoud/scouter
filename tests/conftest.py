"""Shared pytest fixtures for tests."""

import pytest
from pydantic import BaseModel

from scouter.llmcore.agent import AgentConfig


class TestOutput(BaseModel):
    answer: str


@pytest.fixture
def mock_openai_client(monkeypatch):
    """Fixture to mock the global OpenAI client."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "Mock response"
    mock_completion.choices[0].message.tool_calls = None
    mock_completion.usage = MagicMock()
    mock_completion.usage.completion_tokens = 10
    mock_completion.usage.prompt_tokens = 5
    mock_completion.usage.total_tokens = 15
    mock_client.chat.completions.create.return_value = mock_completion

    monkeypatch.setattr("scouter.llmcore.client.client", mock_client)
    return mock_client


@pytest.fixture
def mock_structured_response(monkeypatch):
    """Fixture for structured output response."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = '{"answer": "test", "confidence": 0.9}'
    mock_completion.choices[0].message.tool_calls = None
    mock_client.chat.completions.create.return_value = mock_completion

    monkeypatch.setattr("scouter.llmcore.client.client", mock_client)
    return mock_client


@pytest.fixture
def sample_agent_config():
    """Fixture for a sample agent config."""
    return AgentConfig(name="test_agent", model="gpt-4", tools=[])


@pytest.fixture
def sample_agent_config_with_tools():
    """Fixture for a sample agent config with tools."""
    return AgentConfig(name="test_agent", model="gpt-4", tools=["test_tool"])


@pytest.fixture
def mock_tool_call_response(monkeypatch):
    """Fixture for tool call response."""
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = None
    mock_completion.choices[0].message.tool_calls = [
        {"function": {"name": "test_tool", "arguments": '{"arg": "value"}'}}
    ]
    mock_client.chat.completions.create.return_value = mock_completion

    monkeypatch.setattr("scouter.llmcore.client.client", mock_client)
    return mock_client


@pytest.fixture
def mock_tool_registry(monkeypatch):
    """Fixture to register a mock tool."""
    from pydantic import BaseModel

    from scouter.llmcore.tools import Tool, register_tool

    class MockInput(BaseModel):
        args: dict

    def mock_handler(inputs: MockInput) -> str:
        return "tool result"

    tool = Tool(
        name="test_tool",
        description="Test tool",
        handler=mock_handler,
    )
    register_tool(tool)
