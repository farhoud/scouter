"""Tests for llmcore client functions with mocked OpenAI client."""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from scouter.llmcore.client import call_llm, structured_call_llm


class TestOutput(BaseModel):
    answer: str
    confidence: float


@pytest.fixture
def mock_structured_response(monkeypatch):
    """Fixture for structured output response."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = '{"answer": "test", "confidence": 0.9}'
    mock_completion.choices[0].message.tool_calls = None
    mock_client.chat.completions.create.return_value = mock_completion

    monkeypatch.setattr("scouter.llmcore.client.client", mock_client)
    return mock_client


def test_call_llm_basic(mock_openai_client):
    """Test basic call_llm functionality."""
    messages = [{"role": "user", "content": "Hello"}]  # type: ignore[list-item]
    result = call_llm("gpt-4", messages)  # type: ignore[arg-type]

    assert result.choices[0].message.content == "Mock response"
    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args
    assert call_args[1]["model"] == "gpt-4"
    assert call_args[1]["messages"] == messages
    assert call_args[1]["tools"] == []


def test_call_llm_with_tools(mock_openai_client):
    """Test call_llm with tools."""
    messages = [{"role": "user", "content": "Hello"}]  # type: ignore[list-item]
    tools = [{"type": "function", "function": {"name": "test", "description": "test"}}]  # type: ignore[list-item]
    call_llm("gpt-4", messages, tools)  # type: ignore[arg-type]

    mock_openai_client.chat.completions.create.assert_called_once()
    call_args = mock_openai_client.chat.completions.create.call_args
    assert call_args[1]["tools"] == tools


def test_structured_call_llm_success(mock_structured_response):
    """Test structured_call_llm with valid JSON response."""
    messages = [{"role": "user", "content": "Hello"}]  # type: ignore[list-item]
    result = structured_call_llm("gpt-4", messages, TestOutput)  # type: ignore[arg-type]

    assert isinstance(result, TestOutput)
    assert result.answer == "test"
    assert result.confidence == 0.9
    mock_structured_response.chat.completions.create.assert_called_once()


def test_structured_call_llm_invalid_json(monkeypatch):
    """Test structured_call_llm with invalid JSON."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "invalid json"
    mock_client.chat.completions.create.return_value = mock_completion

    monkeypatch.setattr("scouter.llmcore.client.client", mock_client)
    messages = [{"role": "user", "content": "Hello"}]  # type: ignore[list-item]
    with pytest.raises(ValueError, match="Failed to parse LLM response as JSON"):
        structured_call_llm("gpt-4", messages, TestOutput)  # type: ignore[arg-type]


def test_structured_call_llm_validation_error(monkeypatch):
    """Test structured_call_llm with JSON that fails validation."""
    mock_client = MagicMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = '{"invalid": "data"}'
    mock_client.chat.completions.create.return_value = mock_completion

    monkeypatch.setattr("scouter.llmcore.client.client", mock_client)
    messages = [{"role": "user", "content": "Hello"}]  # type: ignore[list-item]
    with pytest.raises(ValueError, match="Failed to validate LLM response"):
        structured_call_llm("gpt-4", messages, TestOutput)  # type: ignore[arg-type]</content>
