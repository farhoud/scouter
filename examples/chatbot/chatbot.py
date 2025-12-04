"""CLI chatbot with RAG using Scouter + OpenRouter and MCP tools."""

from scouter_app.agent.agent import handle_tool_calls
from scouter_app.agent.tools import get_tools
from scouter_app.config.llm import (
    DEFAULT_MODEL,
    call_with_rate_limit,
    get_chatbot_client,
)

# Get LLM client
llm = get_chatbot_client()


def chat_with_rag(query: str) -> str:
    """Single message chatbot with RAG using Scouter + OpenRouter and MCP tools."""
    tools = get_tools()
    openai_tools = [tool["function"] for tool in tools]
    messages = [
        {
            "role": "system",
            "content": "You are a helpful chatbot with access to a knowledge graph. Use the available tools to search for information before answering if it helps provide accurate responses.",
        },
        {"role": "user", "content": query},
    ]

    # Call LLM with tools
    response = call_with_rate_limit(
        llm,
        model=DEFAULT_MODEL,
        messages=messages,  # type: ignore[arg-type]
        tools=openai_tools,
        tool_choice="auto",
        max_tokens=200,
    )

    # Handle tool calls
    final_response, _ = handle_tool_calls(response, tools, llm, messages)

    print(final_response)  # noqa: T201

    return final_response


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        query = sys.argv[1]
        print(chat_with_rag(query))  # noqa: T201
    else:
        print("Usage: python chatbot.py 'your query'")  # noqa: T201
