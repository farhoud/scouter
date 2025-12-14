"""Tests for llmcore agent functionality."""

from src.scouter.llmcore import (
    create_flow,
    mark_flow_completed,
    mark_flow_running,
)


def test_flow_status():
    flow = create_flow(flow_id="test")
    mark_flow_running(flow)
    assert flow["status"] == "running"
    mark_flow_completed(flow)
    assert flow["status"] == "completed"
