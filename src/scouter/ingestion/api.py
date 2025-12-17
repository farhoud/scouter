"""API endpoints for document ingestion."""

import json
import tempfile

from fastapi import APIRouter, Form, UploadFile

from scouter.config import config
from scouter.ingestion.tasks import process_document_task
from scouter.shared.domain_models import IngestResponse

router = APIRouter()


@router.post("/v1/ingest", response_model=IngestResponse, status_code=202)
async def ingest_document(
    file: UploadFile | None = None,
    text: str | None = Form(None),
    metadata: str = Form("{}"),
) -> IngestResponse:
    """Ingest a PDF file or raw text into the knowledge graph asynchronously.

    Provide either 'file' (PDF) or 'text', not both.

    Args:
        file: PDF file to ingest.
        text: Raw text content to ingest.
        metadata: JSON string containing metadata.

    Returns:
        IngestResponse with task ID and status.

    Raises:
        ValueError: If input validation fails.
    """
    # Parse metadata
    try:
        metadata_dict = json.loads(metadata)
    except json.JSONDecodeError:
        metadata_dict = {}

    # Validate input
    if (file is None and text is None) or (file is not None and text is not None):
        msg = "Exactly one of 'file' or 'text' must be provided"
        raise ValueError(msg)

    task_data = {"metadata": metadata_dict}

    if file is not None:
        # Save file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            task_data["file_path"] = temp_file.name
    else:
        task_data["text"] = text

    cfg = config.llm
    task = process_document_task.apply_async(args=[task_data])
    return IngestResponse(task_id=task.id, status="accepted", env=cfg.env)
