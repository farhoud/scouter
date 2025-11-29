from typing import Optional

from fastapi import UploadFile
from pydantic import BaseModel, Field, model_validator


class DocumentIngestRequest(BaseModel):
    file: Optional[UploadFile] = Field(
        None, description="PDF file to ingest (mutually exclusive with text)"
    )
    text: Optional[str] = Field(
        None, description="Raw text content to ingest (mutually exclusive with file)"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata for the document",
    )

    @model_validator(mode="after")
    def validate_input(self):
        if self.file is not None and self.text is not None:
            raise ValueError(
                "Exactly one of 'file' or 'text' must be provided, not both"
            )
        if self.file is None and self.text is None:
            raise ValueError("Exactly one of 'file' or 'text' must be provided")
        if self.file is not None:
            filename = self.file.filename
            if not filename or not filename.lower().endswith(".pdf"):
                raise ValueError("Only PDF files are supported")
        return self


class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query string")
    limit: int = Field(default=10, description="Maximum number of results to return")


class SearchResult(BaseModel):
    content: str = Field(..., description="Retrieved content snippet")
    score: float = Field(..., description="Relevance score of the result")
    node_id: str = Field(..., description="ID of the graph node")


class IngestResponse(BaseModel):
    task_id: str = Field(..., description="Celery task ID for tracking ingestion")
    status: str = Field(..., description="Status of the ingestion request")
    env: str = Field(..., description="Current environment")
