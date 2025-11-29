import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from scouter_app.config.llm import get_client_config
from scouter_app.ingestion.tasks import process_document_task
from scouter_app.shared.domain_models import IngestResponse

router = APIRouter()


@router.post("/v1/ingest", response_model=IngestResponse, status_code=202)
async def ingest_document(
    file: UploadFile = File(None), text: str = Form(None), metadata: str = Form("{}")
):
    """
    Ingest a PDF file or raw text into the knowledge graph asynchronously.
    Provide either 'file' (PDF) or 'text', not both.
    """
    # Parse metadata
    try:
        metadata_dict = json.loads(metadata)
    except json.JSONDecodeError:
        metadata_dict = {}

    # Validate input
    if (file is None and text is None) or (file is not None and text is not None):
        raise ValueError("Exactly one of 'file' or 'text' must be provided")

    task_data = {"metadata": metadata_dict}

    if file is not None:
        # Save file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            task_data["file_path"] = temp_file.name
    else:
        task_data["text"] = text

    config = get_client_config()
    task = process_document_task.delay(task_data)
    return IngestResponse(task_id=task.id, status="accepted", env=config.env)
