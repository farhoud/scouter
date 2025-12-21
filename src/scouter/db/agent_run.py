"""Agent run persistence for Scouter.

This module provides functions for persisting and loading agent runs
in Neo4j with proper separation of config, runtime, and trace data.
"""

import json
import uuid
from datetime import datetime, timezone

import neo4j
from scouter.llmcore.agent_runtime import AgentRuntime, agent_runtime_serializer


def persist_agent_runtime(driver: neo4j.Driver, run: AgentRuntime, run_id: str) -> None:
    """Persist an AgentRuntime to Neo4j using dict serialization."""
    with driver.session() as session:
        # Serialize the runtime to dict
        data = agent_runtime_serializer.serialize(run)

        # Save as JSON in a single node
        session.run(
            """
            MERGE (r:AgentRuntime {id: $id})
            SET r.data = $data, r.updated_at = $timestamp
            """,
            id=run_id,
            data=json.dumps(data),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


def load_agent_runtime(driver: neo4j.Driver, run_id: str) -> AgentRuntime | None:
    """Load an AgentRuntime from Neo4j."""
    with driver.session() as session:
        result = session.run(
            "MATCH (r:AgentRuntime {id: $id}) RETURN r.data as data",
            id=run_id,
        )
        record = result.single()
        if not record:
            return None

        data = json.loads(record["data"])
        return agent_runtime_serializer.deserialize(data)


def persist_trace(driver: neo4j.Driver, run_id: str, data: dict) -> None:
    """Persist trace data for an agent run."""
    span_id = str(uuid.uuid4())
    data["span_id"] = span_id

    with driver.session() as session:
        # Create Trace node linked to AgentRuntime
        session.run(
            """
            MATCH (r:AgentRuntime {id: $run_id})
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
