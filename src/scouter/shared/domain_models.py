from fastapi import UploadFile
from pydantic import BaseModel, Field, model_validator


class DocumentIngestRequest(BaseModel):
    file: UploadFile | None = Field(
        None,
        description="PDF file to ingest (mutually exclusive with text)",
    )
    text: str | None = Field(
        None,
        description="Raw text content to ingest (mutually exclusive with file)",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata for the document",
    )

    @model_validator(mode="after")
    def validate_input(self):
        if self.file is not None and self.text is not None:
            msg = "Exactly one of 'file' or 'text' must be provided, not both"
            raise ValueError(
                msg,
            )
        if self.file is None and self.text is None:
            msg = "Exactly one of 'file' or 'text' must be provided"
            raise ValueError(msg)
        if self.file is not None:
            filename = self.file.filename
            if not filename or not filename.lower().endswith(".pdf"):
                msg = "Only PDF files are supported"
                raise ValueError(msg)
        return self


class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query string")
    limit: int = Field(default=10, description="Maximum number of results to return")


class SearchResult(BaseModel):
    content: str = Field(..., description="Retrieved content snippet")
    score: float = Field(..., description="Relevance score of the result")
    node_id: str = Field(..., description="ID of the graph node")


class VectorSearchResult(BaseModel):
    node_id: str = Field(description="Unique node identifier")
    score: float = Field(description="Similarity score")
    content: str = Field(description="Retrieved content")
    metadata: dict | None = Field(default=None, description="Additional metadata")


class IngestResponse(BaseModel):
    task_id: str = Field(..., description="Celery task ID for tracking ingestion")
    status: str = Field(..., description="Status of the ingestion request")
    env: str = Field(..., description="Current environment")
