"""Celery tasks for asynchronous document processing."""

import asyncio
import concurrent.futures
import os
from typing import Any

from celery import Celery

from scouter.ingestion.service import IngestionService

app = Celery(
    "scouter.ingestion.tasks",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
)


@app.task
def process_document_task(task_data: dict[str, Any]) -> dict[str, Any]:
    """Long-running task to process PDF or text into the knowledge graph.

    Args:
        task_data: Dictionary containing file_path, text, and metadata.

    Returns:
        Dictionary with processing result.
    """

    async def run_async() -> dict[str, Any]:
        service = IngestionService()
        try:
            file_path = task_data.get("file_path")
            text = task_data.get("text")
            metadata = task_data.get("metadata", {})
            return await service.process_document(
                file_path=file_path,
                text=text,
                metadata=metadata,
            )
        finally:
            service.close()

    # Run the async function in the event loop
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If there's already a running loop, use ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, run_async())
            return future.result()
    else:
        return asyncio.run(run_async())
