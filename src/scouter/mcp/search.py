from scouter.agents.search import search_knowledge_graph
from scouter.mcp import app


@app.tool()
async def search_knowledge_graph_tool(query: str, hints: str = "") -> str:
    """Search the knowledge graph for information related to the query using semantic search.

    This tool allows LLMs to retrieve relevant documents and knowledge from the Scouter knowledge graph.
    It performs vector-based semantic search, returning the most relevant results with analysis.

    Args:
        query: The search query string to find relevant information
        hints: Optional hints to guide the search agent

    Returns:
        A response string containing search results and analysis

    """
    return await search_knowledge_graph(query, hints)
