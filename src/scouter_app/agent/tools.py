import ast

from neo4j_graphrag.retrievers import VectorRetriever
from pydantic import BaseModel, Field

from scouter_app.config.llm import get_neo4j_driver, get_neo4j_embedder
from scouter_app.shared.domain_models import VectorSearchResult


class SemanticSearchParams(BaseModel):
    query_text: str = Field(description="exact user query")
    top_k: int = Field(default=10, description="Number of results to return (1-20)")
    filters: dict | None = Field(default=None, description="Optional filters")
    effective_search_ratio: float = Field(
        default=1.0, description="Search pool multiplier for better accuracy"
    )


def _get_vector_search_tool():
    retriever = VectorRetriever(
        driver=get_neo4j_driver(),
        index_name="chunkEmbedding",
        embedder=get_neo4j_embedder(),
    )

    def semantic_search(**kwargs):
        params = SemanticSearchParams(**kwargs)
        raw_results = retriever.search(**params.model_dump())
        items = raw_results.items
        return [
            VectorSearchResult(
                node_id=data.get("id", "unknown"),
                score=data.get("score", 0.0),
                content=data.get("text", ""),
                metadata=data,
            )
            for result in items
            for data in [ast.literal_eval(result.content)]
        ]

    return {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "find relative information based on cosine similarity",
            "parameters": SemanticSearchParams.model_json_schema(),
        },
        "callable": semantic_search,
    }


def get_tools():
    return [_get_vector_search_tool()]
