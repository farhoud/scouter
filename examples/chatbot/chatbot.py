"""CLI chatbot with RAG using Scouter + OpenRouter and MCP tools."""

import json

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
        if response.choices[0].message.tool_calls:  # type: ignore
            messages.append(response.choices[0].message.model_dump())  # type: ignore[arg-type]
            for tool_call in response.choices[0].message.tool_calls:  # type: ignore
                tool_name = tool_call.function.name  # type: ignore[attr-defined]
                args = json.loads(tool_call.function.arguments)  # type: ignore[attr-defined]
                for tool in tools:
                    if tool["function"]["name"] == tool_name:
                        result = tool["callable"](**args)
                        # Append tool result
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps([r.model_dump() for r in result]) if isinstance(result, list) else json.dumps(result)
                        })
                        break
            
            # Follow-up response
            response = call_with_rate_limit(
                llm,
                model=DEFAULT_MODEL,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=200,
            )

        print(response.choices[0].message.content)  # noqa: T201  # type: ignore[attr]
        messages.append(response.choices[0].message.model_dump())  # type: ignore[arg-type]
        for tool_call in response.choices[0].message.tool_calls:
            tool_name = tool_call.function.name  # type: ignore[attr-defined]
            args = json.loads(tool_call.function.arguments)  # type: ignore[attr-defined]
            for tool in tools:
                if tool["function"]["name"] == tool_name:
                    result = tool["callable"](**args)
                    # Append tool result
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps([r.model_dump() for r in result])
                            if isinstance(result, list)
                            else json.dumps(result),
                        }
                    )
                    break

        # Follow-up response
        response = llm.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=200,
        )

    return response.choices[0].message.content or ""


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        query = sys.argv[1]
        print(chat_with_rag(query))  # noqa: T201
    else:
        print("Usage: python chatbot.py 'your query'")  # noqa: T201
