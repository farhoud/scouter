"""Tests for llmcore agent functionality."""

import pytest

from scouter.llmcore.agent import AgentRun, run_agent
from scouter.llmcore.flow import Flow


def test_flow_status():
    flow = Flow(id="test")
    flow.mark_running()
    assert flow.status == "running"
    flow.mark_completed()
    assert flow.status == "completed"


@pytest.mark.asyncio
async def test_agent_creation(mock_openai_client, sample_agent_config):
    """Test basic agent creation and run."""
    agent = AgentRun()
    messages = [{"role": "user", "content": "Hello"}]  # type: ignore[list-item]

    result = await run_agent(agent, sample_agent_config, messages)  # type: ignore[arg-type]

    assert isinstance(result, AgentRun)
    assert len(result.flows) > 0
    assert any(flow.status == "completed" for flow in result.flows)


@pytest.mark.asyncio
async def test_agent_with_tools(
    mock_openai_client, sample_agent_config, mock_tool_registry
):
    """Test agent with tools configured."""
    agent = AgentRun()
    messages = [{"role": "user", "content": "Hello"}]  # type: ignore[list-item]

    result = await run_agent(agent, sample_agent_config, messages)  # type: ignore[arg-type]

    assert isinstance(result, AgentRun)


@pytest.mark.asyncio
async def test_agent_tool_call_execution(
    mock_tool_call_response, sample_agent_config, mock_tool_registry
):
    """Test agent executing tool calls."""
    agent = AgentRun()
    messages = [{"role": "user", "content": "Hello"}]  # type: ignore[list-item]

    result = await run_agent(agent, sample_agent_config, messages)  # type: ignore[arg-type]

    assert isinstance(result, AgentRun)


# TODO: Add structured output test when mocking is fixed
