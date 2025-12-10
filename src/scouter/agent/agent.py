import json

from scouter.agent.tools import get_tools
from scouter.config.llm import (
    DEFAULT_MODEL,
    call_with_rate_limit,
    get_scouter_client,
)


def handle_tool_calls(response, tools, client, messages):
    if not response.choices[0].message.tool_calls:
        return response.choices[0].message.content, messages

    # Append assistant message with tool calls
    messages.append(response.choices[0].message)

    for tool_call in response.choices[0].message.tool_calls:
        tool_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        for tool in tools:
            if tool["function"]["name"] == tool_name:
                result = tool["callable"](**args)
                # Append tool result message
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

    # Follow-up call for multi-turn
    follow_up = call_with_rate_limit(
        client,
        model=DEFAULT_MODEL,
        messages=messages,  # type: ignore[arg-type]
        tools=[t["function"] for t in tools],
        tool_choice="auto",
    )
    return handle_tool_calls(follow_up, tools, client, messages)


def search_agent(query: str, hints: str = ""):
    client = get_scouter_client()
    tools = get_tools()
    openai_tools = [t["function"] for t in tools]
    system_content = (
        "You are a search agent. Use the available tools to answer the user's query."
    )
    if hints:
        system_content += f"\n\nHints:\n{hints}"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
    ]
    response = call_with_rate_limit(
        client,
        model=DEFAULT_MODEL,
        messages=messages,  # type: ignore[arg-type]
        tools=openai_tools,
        tool_choice="auto",
        max_tokens=200,
    )
    final_response, _ = handle_tool_calls(response, tools, client, messages)
    return final_response
