from fastapi import APIRouter

from scouter_app.ingestion.tasks import process_document_task
from scouter_app.shared.domain_models import DocumentIngestRequest, IngestResponse

router = APIRouter()


@router.post("/v1/ingest", response_model=IngestResponse, status_code=202)
async def ingest_document(request: DocumentIngestRequest):
    """
    Ingest a document into the knowledge graph asynchronously.
    """
    task = process_document_task.delay(request.dict())
    return IngestResponse(task_id=task.id, status="accepted")
