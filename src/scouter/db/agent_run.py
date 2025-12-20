"""Agent run persistence for Scouter.

This module provides functions for persisting and loading agent runs
in Neo4j with proper separation of config, runtime, and trace data.
"""

import json
import uuid
from typing import Any

import neo4j
from scouter.llmcore.agent import AgentRun
from scouter.llmcore.agent_runtime import AgentConfig
from scouter.llmcore.flow import InputStep, LLMStep, ToolCall, ToolStep
from scouter.llmcore.types import ChatCompletion

# Flow is now dict


def _serialize_step(step) -> dict[str, Any]:
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


def _deserialize_step(step_type: str, data: str):
    """Deserialize a step from JSON."""
    data_dict = json.loads(data)
    if step_type == "LLMStep":
        return LLMStep(completion=ChatCompletion(**data_dict["completion"]))
    if step_type == "ToolStep":
        calls = [ToolCall(**call) for call in data_dict["calls"]]
        return ToolStep(calls=calls)
    if step_type == "InputStep":
        return InputStep(input=data_dict["input"])
    msg = f"Unknown step type: {step_type}"
    raise ValueError(msg)


def persist_agent_run(driver: neo4j.Driver, run: AgentRun, run_id: str) -> None:
    """Persist an AgentRun to Neo4j with separate config and runtime data."""
    with driver.session() as session:
        # Create config node
        config_id = f"{run_id}_config"
        session.run(
            """
            MERGE (c:AgentConfig {id: $config_id})
            SET c.api_key = $api_key,
                c.model = $model,
                c.provider = $provider,
                c.track_usage = $track_usage
            """,
            config_id=config_id,
            api_key=run.config.api_key,
            model=run.config.model,
            provider=run.config.provider,
            track_usage=run.config.track_usage,
        )

        # Create run node (runtime data only)
        session.run(
            """
            MERGE (r:AgentRun {id: $id})
            SET r.total_usage = $total_usage,
                r.memory_strategy = $memory_strategy
            WITH r
            MATCH (c:AgentConfig {id: $config_id})
            MERGE (r)-[:USES_CONFIG]->(c)
            """,
            id=run_id,
            total_usage=json.dumps(run.total_usage),
            memory_strategy=type(run.memory_function).__name__,
            config_id=config_id,
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
                step_data = _serialize_step(step)
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


def load_agent_run(driver: neo4j.Driver, run_id: str) -> AgentRun | None:
    """Load an AgentRun from Neo4j."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (r:AgentRun {id: $id})-[:USES_CONFIG]->(c:AgentConfig),
                  (r)-[:HAS_FLOW]->(f:Flow)-[:HAS_STEP]->(s:AgentStep)
            RETURN r, c, f, collect(s) as steps ORDER BY f.id, s.index
            """,
            id=run_id,
        )
        records = list(result)
        if not records:
            return None

        config_node = records[0]["c"]
        config = AgentConfig(
            api_key=config_node.get("api_key"),
            model=config_node.get("model", "gpt-4o-mini"),
            provider=config_node.get("provider", "openai"),
            track_usage=config_node.get("track_usage", True),
        )
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
                step = _deserialize_step(step_node["type"], step_node["data"])
                flows[flow_id]["steps"].append(step)

        return AgentRun(config=config, flows=list(flows.values()))
        # TODO: Restore memory_function from run_node


def persist_trace(driver: neo4j.Driver, run_id: str, data: dict) -> None:
    """Persist trace data for an agent run."""
    span_id = str(uuid.uuid4())
    data["span_id"] = span_id

    with driver.session() as session:
        # Create Trace node linked to AgentRun
        session.run(
            """
            MATCH (r:AgentRun {id: $run_id})
            CREATE (t:Trace {
                span_id: $span_id,
                operation: $operation,
                start_time: $start_time,
                end_time: $end_time,
                attributes: $attributes
            })
            CREATE (r)-[:HAS_TRACE]->(t)
            """,
            run_id=run_id,
            span_id=span_id,
            operation=data.get("operation"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            attributes=data.get("attributes", {}),
        )
