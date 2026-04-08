from typing import Any

from openai.types.chat import ChatCompletionFunctionToolParam

from .handlers.json_retrieval import search_unified_knowledge

SEARCH_UNIFIED_KNOWLEDGE_TOOL: ChatCompletionFunctionToolParam = {
    "type": "function",
    "function": {
        "name": "search_unified_knowledge",
        "description": "Search the unified Drive+Website chunk knowledge base for retrieval-augmented answers with citations.",
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

def execute_tool_call(function_name: str, function_arguments: dict[str, Any]) -> Any:
    handler = TOOL_HANDLERS.get(function_name)
    if not handler:
        return {"error": f"No handler for tool: {function_name}"}
    try:
        return handler(**function_arguments)
    except Exception as exc:
        return {"error": f"Error executing tool {function_name}: {exc}"}


__all__ = ["TOOLS", "TOOL_HANDLERS", "execute_tool_call"]