"""Database models and abstractions for Scouter.

This module defines data models and operations for storing agent-related data
in Neo4j, such as agent runs, flows, and steps.
"""

import json
from typing import Any

import neo4j
from ..llmcore.flow import LLMStep, ToolStep, InputStep
from ..llmcore.agent import AgentRun

# Flow is now dict


class AgentRunRepository:
    """Repository for managing AgentRun persistence in Neo4j."""

    def __init__(self, driver: neo4j.Driver):
        self.driver = driver

    def save_run(self, run: AgentRun, run_id: str) -> None:
        """Save an AgentRun to Neo4j."""
        with self.driver.session() as session:
            # Create run node
            session.run(
                """
                MERGE (r:AgentRun {id: $id})
                SET r.total_usage = $total_usage,
                    r.memory_strategy = $memory_strategy
                """,
                id=run_id,
                total_usage=json.dumps(run.total_usage),
                memory_strategy=type(run.memory_function).__name__,
            )

            # Create flow nodes and relationships
            for flow in run.flows:
                session.run(
                    """
                    MATCH (r:AgentRun {id: $run_id})
                    MERGE (f:Flow {id: $flow_id})
                    SET f.agent_id = $agent_id,
                        f.status = $status,
                        f.metadata = $metadata,
                        f.parent_flow_id = $parent_flow_id
                    CREATE (r)-[:HAS_FLOW]->(f)
                    """,
                    run_id=run_id,
                    flow_id=flow.id,
                    agent_id=flow.agent_id,
                    status=flow.status,
                    metadata=json.dumps(flow.metadata),
                    parent_flow_id=flow.parent_flow_id,
                )

                # Create step nodes for each flow
                for i, step in enumerate(flow.steps):
                    step_data = self._serialize_step(step)
                    session.run(
                        """
                        MATCH (f:Flow {id: $flow_id})
                        CREATE (s:AgentStep {index: $index, type: $type, data: $data})
                        CREATE (f)-[:HAS_STEP]->(s)
                        """,
                        flow_id=flow.id,
                        index=i,
                        type=type(step).__name__,
                        data=json.dumps(step_data),
                    )

    def load_run(self, run_id: str) -> AgentRun | None:
        """Load an AgentRun from Neo4j."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (r:AgentRun {id: $id})-[:HAS_FLOW]->(f:Flow)-[:HAS_STEP]->(s:AgentStep)
                RETURN r, f, collect(s) as steps ORDER BY f.id, s.index
                """,
                id=run_id,
            )
            records = list(result)
            if not records:
                return None

            records[0]["r"]
            flows = {}
            for record in records:
                flow_node = record["f"]
                step_nodes = record["steps"]

                flow_id = flow_node["id"]
                if flow_id not in flows:
                    flows[flow_id] = {
                        "id": flow_id,
                        "agent_id": flow_node["agent_id"],
                        "status": flow_node["status"],
                        "metadata": json.loads(flow_node["metadata"]),
                        "parent_flow_id": flow_node["parent_flow_id"],
                        "steps": [],
                    }

                # Reconstruct steps
                for step_node in step_nodes:
                    step = self._deserialize_step(step_node["type"], step_node["data"])
                    flows[flow_id]["steps"].append(step)

            return AgentRun(flows=list(flows.values()))
            # TODO: Restore memory_function from run_node

    def _serialize_step(self, step) -> dict[str, Any]:
        """Serialize a step to a dict."""
        if isinstance(step, LLMStep):
            return {
                "completion": step.completion.model_dump(),
            }
        if isinstance(step, ToolStep):
            return {
                "calls": [call.__dict__ for call in step.calls],
            }
        if isinstance(step, InputStep):
            return {
                "input": step.input,
            }
        return {"data": str(step)}

    def _deserialize_step(self, step_type: str, data: str):
        """Deserialize a step from JSON."""
        from ..llmcore.types import ChatCompletion

        data_dict = json.loads(data)
        if step_type == "LLMStep":
            return LLMStep(completion=ChatCompletion(**data_dict["completion"]))
        if step_type == "ToolStep":
            from ..llmcore.flow import ToolCall

            calls = [ToolCall(**call) for call in data_dict["calls"]]
            return ToolStep(calls=calls)
        if step_type == "InputStep":
            return InputStep(input=data_dict["input"])
        msg = f"Unknown step type: {step_type}"
        raise ValueError(msg)
