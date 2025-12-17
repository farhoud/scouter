import ast

from neo4j_graphrag.retrievers import VectorRetriever
from pydantic import BaseModel, Field

from scouter.db import get_neo4j_driver, get_neo4j_embedder
from scouter.llmcore import tool
from scouter.shared.domain_models import VectorSearchResult


class SemanticSearchParams(BaseModel):
    query_text: str = Field(description="exact user query")
    top_k: int = Field(default=10, description="Number of results to return (1-20)")
    filters: dict | None = Field(default=None, description="Optional filters")
    effective_search_ratio: float = Field(
        default=1.0, description="Search pool multiplier for better accuracy"
    )


class SearchResults(BaseModel):
    results: list[VectorSearchResult]


@tool("semantic_search")
def semantic_search(params: SemanticSearchParams) -> SearchResults:
    """Find relevant information based on cosine similarity search."""
    # Cast to the expected parameter type
    search_params = SemanticSearchParams(**params.model_dump())

    retriever = VectorRetriever(
        driver=get_neo4j_driver(),
        index_name="chunkEmbedding",
        embedder=get_neo4j_embedder(),
    )

    raw_results = retriever.search(**search_params.model_dump())
    items = raw_results.items
    results = [
        VectorSearchResult(
            node_id=data.get("id", "unknown"),
            score=data.get("score", 0.0),
            content=data.get("text", ""),
            metadata=data,
        )
        for result in items
        for data in [ast.literal_eval(result.content)]
    ]
    return SearchResults(results=results)
