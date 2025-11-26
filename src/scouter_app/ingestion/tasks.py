import os
from celery import Celery
from neo4j import GraphDatabase

app = Celery(
    "scouter_app.ingestion.tasks",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
)


@app.task
def process_document_task(document_data: dict):
    """
    Long-running task to process and ingest document into the knowledge graph.
    """
    # Simulate processing time
    import time

    time.sleep(5)  # Simulate heavy computation

    # Connect to Neo4j
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        # Create a node for the document
        session.run(
            "CREATE (d:Document {title: $title, content: $content, metadata: $metadata})",
            title=document_data["title"],
            content=document_data["content"],
            metadata=document_data["metadata"],
        )

    driver.close()
    return {"status": "ingested", "title": document_data["title"]}
