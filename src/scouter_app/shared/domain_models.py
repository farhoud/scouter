from pydantic import BaseModel, Field


class DocumentIngestRequest(BaseModel):
    content: str = Field(..., description="The text content of the document to ingest")
    title: str = Field(..., description="Title of the document")
    metadata: dict = Field(
        default_factory=dict, description="Additional metadata for the document"
    )


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
