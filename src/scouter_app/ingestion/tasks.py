import os
import time

from celery import Celery

from scouter_app.ingestion.service import IngestionService

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
    time.sleep(5)  # Simulate heavy computation

    service = IngestionService()
    result = service.ingest_document(
        title=document_data["title"],
        content=document_data["content"],
        metadata=document_data["metadata"],
    )
    service.close()
    return result
