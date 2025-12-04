from fastmcp import FastMCP

from .agent import search_agent

app = FastMCP("Scouter Agent")


@app.tool()
def search_knowledge_graph(query: str, hints: str = "") -> str:
    """Search the knowledge graph for information related to the query using semantic search.

    This tool allows LLMs to retrieve relevant documents and knowledge from the Scouter knowledge graph.
    It performs vector-based semantic search, returning the most relevant results.

    Args:
        query: The search query string to find relevant information
        hints: Optional hints to guide the search agent

    Returns:
        A response string containing search results and analysis

    """
    return search_agent(query, hints)
