from fastmcp import FastMCP
from .service import SearchService

app = FastMCP("Scouter Agent")

service = SearchService()


@app.tool()
def search_knowledge_graph(query: str, limit: int = 10) -> str:
    """
    Search the knowledge graph for information related to the query.

    This tool allows LLMs to retrieve relevant documents and knowledge from the Scouter knowledge graph.
    It performs a full-text search on document titles and contents, returning the most relevant snippets.

    Args:
        query: The search query string to find relevant information
        limit: Maximum number of results to return (default: 10)

    Returns:
        A formatted string containing search results with content snippets and relevance scores
    """
    results = service.search(query, limit)
    if not results:
        return "No results found for the query."
    return "\n".join([f"- {r.content} (score: {r.score})" for r in results])
