from scouter.llmcore import AgentConfig, AgentRun, create_agent, run_agent
from scouter.llmcore.types import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


def get_search_agent() -> tuple[AgentRun, AgentConfig]:
    """Get a pre-configured search agent for knowledge graph queries."""
    config = AgentConfig(
        name="search",
        provider="openai",  # Use OpenAI for search (more deterministic)
        model="gpt-4o-mini",
        temperature=0.0,  # Deterministic for search
        instructions=(
            "You are a search agent specialized in retrieving information from a knowledge graph. "
            "Use the semantic_search tool to find relevant information based on the user's query. "
            "Analyze the search results and provide a comprehensive answer.",
            "What information are you looking for?",
        ),
        tools=["semantic_search"],
        max_tokens=1000,  # Allow longer responses for search results
    )
    agent = create_agent(config)
    return agent, config


async def search_knowledge_graph(query: str, hints: str = "") -> str:
    """Search the knowledge graph for information related to the query.

    Args:
        query: The search query string
        hints: Optional hints to guide the search

    Returns:
        A response string containing search results and analysis
    """
    agent, config = get_search_agent()

    # Prepare messages with hints if provided
    base_instructions = (
        "You are a search agent specialized in retrieving information from a knowledge graph. "
        "Use the semantic_search tool to find relevant information based on the user's query. "
        "Analyze the search results and provide a comprehensive answer."
    )

    system_content = base_instructions
    if hints:
        system_content += f"\n\nAdditional hints: {hints}"

    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=system_content),
        ChatCompletionUserMessageParam(role="user", content=query),
    ]

    # Run the agent
    result_agent = await run_agent(agent, config, messages)

    # Extract the final response
    return result_agent.last_output
