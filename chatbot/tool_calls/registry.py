from typing import Any
import inspect

from openai.types.chat import ChatCompletionFunctionToolParam

from .handlers.database_retrieval import search_unified_knowledge

SEARCH_UNIFIED_KNOWLEDGE_TOOL: ChatCompletionFunctionToolParam = {
    "type": "function",
    "function": {
        "name": "search_unified_knowledge",
        "description": "Search the SQLite chunk database for retrieval-augmented answers with citations.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user question or search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

TOOLS: list[ChatCompletionFunctionToolParam] = [SEARCH_UNIFIED_KNOWLEDGE_TOOL]

TOOL_HANDLERS = {
    "search_unified_knowledge": search_unified_knowledge,
}

def _invoke_handler_with_supported_args(handler: Any, call_arguments: dict[str, Any]) -> Any:
    signature = inspect.signature(handler)
    params = signature.parameters
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    if accepts_var_kwargs:
        return handler(**call_arguments)

    supported_args = {key: value for key, value in call_arguments.items() if key in params}
    return handler(**supported_args)


def execute_tool_call(
    function_name: str,
    function_arguments: dict[str, Any],
    tool_context: dict[str, Any] | None = None,
) -> Any:
    handler = TOOL_HANDLERS.get(function_name)
    if not handler:
        return {"error": f"No handler for tool: {function_name}"}
    try:
        call_arguments = dict(function_arguments or {})
        if tool_context:
            for key, value in tool_context.items():
                call_arguments.setdefault(key, value)
        return _invoke_handler_with_supported_args(handler, call_arguments)
    except Exception as exc:
        return {"error": f"Error executing tool {function_name}: {exc}"}


__all__ = ["TOOLS", "TOOL_HANDLERS", "execute_tool_call"]