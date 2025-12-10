from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable  # noqa: TC003
from typing import TYPE_CHECKING, Any, get_origin

from pydantic import BaseModel, Field

from .exceptions import ToolExecutionError

if TYPE_CHECKING:
    from .types import ChatCompletionToolParam


class Tool(BaseModel):
    name: str
    description: str
    handler: Callable[[BaseModel], BaseModel | str]

    # Auto-filled fields
    parameters_schema: dict = Field(default_factory=dict)
    output_schema: dict = Field(default_factory=dict)
    description_with_output: str = ""

    # Internal: Store the actual class types for runtime conversion
    input_type: type[BaseModel] | None = None

    def model_post_init(self, /, __context) -> None:
        # 1. Extract input model from handler signature
        sig = inspect.signature(self.handler)
        if not sig.parameters:
            msg = f"Handler for tool '{self.name}' must have at least one argument (the input Pydantic model)."
            raise TypeError(msg)

        param = next(iter(sig.parameters.values()))
        input_model = param.annotation

        origin = get_origin(input_model) or input_model
        if not (isinstance(origin, type) and issubclass(origin, BaseModel)):
            msg = f"Handler first param for '{self.name}' must be a Pydantic BaseModel, got {origin}"
            raise TypeError(msg)

        self.input_type = origin  # SAVE THIS for execute_tool

        # 2. Extract return type
        return_type = sig.return_annotation
        return_origin = get_origin(return_type) or return_type
        if return_origin is str:
            pass  # Allow str
        elif isinstance(return_origin, type) and issubclass(return_origin, BaseModel):
            pass  # Allow BaseModel
        else:
            msg = f"Handler for '{self.name}' must return a Pydantic BaseModel or str"
            raise TypeError(msg)

        # 3. Auto-fill everything
        self.parameters_schema = origin.model_json_schema()
        if return_origin is str:
            self.output_schema = {"type": "string"}
        else:
            self.output_schema = return_origin.model_json_schema()  # type: ignore[reportAttributeAccessIssue]

        # 4. Enrich description with pretty-printed output schema
        if return_origin is str:
            self.description_with_output = (
                f"{self.description}\n\nThe tool will **always return a string**."
            )
        else:
            pretty_output = json.dumps(self.output_schema, indent=2)
            self.description_with_output = (
                f"{self.description}\n\n"
                f"The tool will **always return JSON matching this exact schema**:\n"
                f"```json\n{pretty_output}\n```"
            )

    def openai_tool_spec(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description_with_output,
                "parameters": self.parameters_schema,
            },
        }


def create_tool(
    name: str, description: str, handler: Callable[[BaseModel], BaseModel | str]
) -> Tool:
    """
    Creates a Pydantic Tool instance.
    """
    return Tool(name=name, description=description, handler=handler)


def tool(name: str | None = None, description: str | None = None):
    """
    Decorator to create and register a Pydantic-based tool.
    The decorated function MUST take a Pydantic model and return a Pydantic model or a string.
    """

    def decorator(func: Callable[[BaseModel], BaseModel | str]):
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "No description.").strip()

        # Create the Tool instance
        t = create_tool(tool_name, tool_desc, func)

        # Register it
        register_tool(t)

        return func

    return decorator


def run_tool(name: str, raw_args: dict[str, Any]) -> str:
    """
    Looks up a tool by name and executes it.
    """
    tool_instance = lookup_tool(name)
    return execute_tool(tool_instance, raw_args)


def execute_tool(tool_instance: Tool, raw_args: dict[str, Any]) -> str:
    """
    Executes a Pydantic Tool.
    1. Converts raw_args (dict) -> InputModel (Pydantic).
    2. Calls handler(InputModel).
    3. Gets OutputModel or str.
    4. Returns OutputModel.model_dump_json() or the str.
    """
    try:
        # 1. Instantiate the specific input model
        input_model_cls = tool_instance.input_type
        assert input_model_cls is not None
        input_obj = input_model_cls(**raw_args)

        # 2. Call Handler
        handler = tool_instance.handler

        if inspect.iscoroutinefunction(handler):
            result_model = asyncio.run(handler(input_obj))
        else:
            result_model = handler(input_obj)

        # 3. Validate Return
        if not isinstance(result_model, (BaseModel, str)):
            msg = f"Tool '{tool_instance.name}' handler did not return a Pydantic model or str."
            raise ToolExecutionError(msg)  # noqa: TRY301

        # 4. Serialize Output
        if isinstance(result_model, str):
            return result_model
        return result_model.model_dump_json()

    except Exception as e:
        msg = f"Error executing tool '{tool_instance.name}': {e!s}"
        raise ToolExecutionError(msg) from e


# Global registry stores Tool instances
TOOL_REGISTRY: dict[str, Tool] = {}


def register_tool(tool_instance: Tool) -> None:
    """
    Registers a Tool object in the global registry.
    """
    if not tool_instance.name:
        msg = "Cannot register tool without a name."
        raise ToolExecutionError(msg)
    TOOL_REGISTRY[tool_instance.name] = tool_instance


def lookup_tool(name: str) -> Tool:
    """
    Retrieves a Tool object from the global registry.
    """
    if name not in TOOL_REGISTRY:
        msg = f"Tool '{name}' not found in registry."
        raise ToolExecutionError(msg)
    return TOOL_REGISTRY[name]
