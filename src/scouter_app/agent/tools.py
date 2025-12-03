from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.tool import Tool

from scouter_app.config.llm import get_neo4j_driver, get_neo4j_embedder


def _get_vector_search_tool() -> Tool:
    retriever = VectorRetriever(
        driver=get_neo4j_driver(),
        index_name="chunkEmbedding",
        embedder=get_neo4j_embedder(),
    )
    return retriever.convert_to_tool(
        name="semantic_search",
        description="find relative information base of cosine simlarity",
        parameter_descriptions={
            "query_text": "exact user query",
            "top_k": "Number of movie recommendations to return (1-20)",
            "filters": "Optional filters for genre, year, rating, etc.",
            "effective_search_ratio": "Search pool multiplier for better accuracy",
        },
    )


def get_tools():
    return [_get_vector_search_tool()]
