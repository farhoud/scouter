"""CLI chatbot with RAG using Scouter + OpenRouter and MCP tools."""

import asyncio
import json
from typing import TYPE_CHECKING, Any, cast

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from scouter.config import config
from scouter.llmcore import call_llm

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageToolCall


async def chat_with_rag(query: str) -> str | None:
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

        # Convert MCP tools to OpenAI format
        openai_tools: list[dict[str, Any]] = [
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
        response = call_llm(
            config.llm.model,
            messages,  # type: ignore[arg-type]
            openai_tools,  # type: ignore[arg-type]
            {"temperature": 0.9, "max_tokens": 200, "tool_choice": "auto"},  # type: ignore[arg-type]
        )

        # Handle tool calls
        if response.choices[0].message.tool_calls:  # type: ignore[attr-defined]
            for tool_call in response.choices[0].message.tool_calls:  # type: ignore[attr-defined]
                tool_call = cast("ChatCompletionMessageToolCall", tool_call)
                tool_name = tool_call.function.name  # type: ignore[attr-defined]
                tool_args = json.loads(tool_call.function.arguments)  # type: ignore[attr-defined]
                result = await session.call_tool(tool_name, tool_args)
                # Add to messages
                messages.append(  # type: ignore[PGH003]
                    {"role": "assistant", "content": "", "tool_calls": [tool_call]}  # type: ignore[PGH003]
                )
                messages.append(  # type: ignore[PGH003]
                    {
                        "role": "tool",
                        "content": str(result),
                        "tool_call_id": tool_call.id,
                    }  # type: ignore[PGH003]
                )

            # Call LLM again with updated messages
            final_response = call_llm(
                config.llm.model,
                messages,  # type: ignore[arg-type]
                None,
                {"max_tokens": 200},  # type: ignore[arg-type]
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
