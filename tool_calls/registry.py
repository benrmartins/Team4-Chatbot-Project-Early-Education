# registry.py
# This module defines the registry of tools available for the chatbot, including their metadata and handlers. 
# It also provides a function to execute tool calls based on the function name and arguments provided by the assistant
from tool_calls.handlers.json_retrieval import search_unified_knowledge

TOOLS = [
    {
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
    },
]

TOOL_HANDLERS = {
    "search_unified_knowledge": search_unified_knowledge,
}

def execute_tool_call(function_name, function_arguments):
    handler = TOOL_HANDLERS.get(function_name)
    if not handler:
        return {"error": f"No handler for tool: {function_name}"}
    try:
        return handler(**function_arguments)
    except Exception as exc:
        return {"error": f"Error executing tool {function_name}: {exc}"}