from collections.abc import Callable

tool_registry: dict[str, dict] = {}


def register_tool(name: str, fn: Callable, description: str, schema: dict):
    tool_registry[name] = {"fn": fn, "description": description, "schema": schema}


def run_tool(name: str, args: dict) -> str:
    if name not in tool_registry:
        return f"Unknown tool: {name}"
    return tool_registry[name]["fn"](args)
