import asyncio
import os

from celery import Celery

from scouter_app.ingestion.service import IngestionService

app = Celery(
    "scouter_app.ingestion.tasks",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
)


@app.task
def process_document_task(task_data: dict):
    """
    Long-running task to process PDF or text into the knowledge graph.
    """

    async def run_async():
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
        # If there's already a running loop, use asyncio.create_task or similar
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, run_async())
            return future.result()
    else:
        return asyncio.run(run_async())
