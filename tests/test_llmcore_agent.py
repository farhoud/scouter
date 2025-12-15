"""Tests for llmcore agent functionality."""

from src.scouter.llmcore.flow import Flow


def test_flow_status():
    flow = Flow(id="test")
    flow.mark_running()
    assert flow.status == "running"
    flow.mark_completed()
    assert flow.status == "completed"
