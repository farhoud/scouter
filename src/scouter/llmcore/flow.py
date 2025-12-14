"""Flow as dict for grouping steps in agent runs."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Step


def create_flow(
    flow_id: str, agent_id: str = "default", parent_flow_id: str | None = None
) -> dict:
    """Create a flow dict."""
    return {
        "id": flow_id,
        "agent_id": agent_id,
        "steps": [],
        "status": "pending",
        "metadata": {},
        "parent_flow_id": parent_flow_id,
    }


def add_step_to_flow(flow: dict, step: "Step") -> None:
    """Add a step to the flow."""
    flow["steps"].append(step)


def mark_flow_running(flow: dict) -> None:
    flow["status"] = "running"
    flow["metadata"]["start_time"] = __import__("time").time()


def mark_flow_completed(flow: dict) -> None:
    flow["status"] = "completed"
    flow["metadata"]["end_time"] = __import__("time").time()


def mark_flow_failed(flow: dict, error: str) -> None:
    flow["status"] = "failed"
    flow["metadata"]["error"] = error
    flow["metadata"]["end_time"] = __import__("time").time()
