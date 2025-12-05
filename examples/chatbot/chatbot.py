"""CLI chatbot with RAG using Scouter + OpenRouter and MCP tools."""

import asyncio
import json

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from scouter_app.config.llm import (
    DEFAULT_MODEL,
    call_with_rate_limit,
    get_chatbot_client,
)

# Get LLM client
llm = get_chatbot_client()


async def chat_with_rag(query: str) -> str:
    """Single message chatbot with RAG using Scouter + OpenRouter and MCP tools."""
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "src.scouter_app.agent.mcp"],
        env=None,
    )

    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        mcp_tools = await session.list_tools()

        print(mcp_tools)

        # Convert MCP tools to OpenAI format
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in mcp_tools.tools
        ]

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
        if response.choices[0].message.tool_calls:  # type: ignore[attr-defined]
            for tool_call in response.choices[0].message.tool_calls:  # type: ignore[attr-defined]
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                result = await session.call_tool(tool_name, tool_args)
                # Add to messages
                messages.append(  # type: ignore
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]}  # type: ignore
                )
                messages.append(  # type: ignore
                    {
                        "role": "tool",
                        "content": str(result),
                        "tool_call_id": tool_call.id,
                    }  # type: ignore
                )

            # Call LLM again with updated messages
            final_response = call_with_rate_limit(
                llm,
                model=DEFAULT_MODEL,
                messages=messages,
                max_tokens=200,
            )
            final_content = final_response.choices[0].message.content  # type: ignore[attr-defined]
        else:
            final_content = response.choices[0].message.content  # type: ignore[attr-defined]

        print(final_content)  # noqa: T201

        return final_content


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        query = sys.argv[1]
        asyncio.run(chat_with_rag(query))
    else:
        print("Usage: python chatbot.py 'your query'")  # noqa: T201
