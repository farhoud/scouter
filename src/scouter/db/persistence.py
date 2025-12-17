"""Persistence utilities for AgentRun using Neo4j."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from scouter.llmcore.agent_runtime import deserialize_agent_run, serialize_agent_run

from .neo4j import get_neo4j_driver

if TYPE_CHECKING:
    from scouter.llmcore.agent import AgentRun


def neo4j_persistence(run: AgentRun) -> None:
    """Persist an AgentRun to Neo4j.

    Args:
        run: The AgentRun to persist.
    """
    driver = get_neo4j_driver()
    data = serialize_agent_run(run)
    agent_id = run.config.name  # Use config name as ID

    with driver.session() as session:
        # Create or update AgentRun node
        session.run(
            """
            MERGE (a:AgentRun {id: $id})
            SET a.data = $data, a.updated_at = $timestamp
            """,
            id=agent_id,
            data=data,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


def neo4j_trace_function(data: dict) -> None:
    """Persist trace data to Neo4j.

    Args:
        data: Trace data dictionary.
    """
    driver = get_neo4j_driver()
    span_id = str(uuid.uuid4())
    data["span_id"] = span_id

    with driver.session() as session:
        # Create Trace node
        session.run(
            """
            CREATE (t:Trace {
                span_id: $span_id,
                operation: $operation,
                start_time: $start_time,
                end_time: $end_time,
                attributes: $attributes
            })
            """,
            span_id=span_id,
            operation=data.get("operation"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            attributes=data.get("attributes", {}),
        )


def load_agent_run_from_neo4j(agent_id: str) -> AgentRun:
    """Load an AgentRun from Neo4j.

    Args:
        agent_id: The ID of the agent run.

    Returns:
        The loaded AgentRun.
    """
    driver = get_neo4j_driver()

    with driver.session() as session:
        result = session.run(
            "MATCH (a:AgentRun {id: $id}) RETURN a.data as data",
            id=agent_id,
        )
        record = result.single()
        if not record:
            msg = f"AgentRun with id {agent_id} not found"
            raise ValueError(msg)

        data = record["data"]
        return deserialize_agent_run(data)
