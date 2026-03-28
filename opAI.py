from openai import OpenAI 
from dotenv import load_dotenv 
import json
import os
from json_retrieval import search_knowledge_base, search_unified_knowledge

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

SYSTEM_PROMPT = """
You are an Early Education Chatbot named Owlbear.

Answer questions about the Early Education Leaders Institute using retrieved source material whenever possible.
If tool results are available, prioritize them over guesswork.
Do not invent facts.
If the retrieved sources are insufficient, say you do not know and suggest contacting EarlyEdLeaders@umb.edu.
When you answer using retrieved sources, cite them clearly by title and URL.
"""

# increase if we want to do a larger context history collection for the chatbot
CONTEXT_LIMIT = 10

class ChatMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content
    def to_dict(self):
        return {"role": self.role, "content": self.content}
    def to_string(self):
        return f"{self.role.capitalize()}: {self.content}"
    def __str__(self):        
        return self.to_string()
    def __repr__(self):       
        return f"ChatMessage(role={self.role!r}, content={self.content!r})"
    def __eq__(self, other): 
        if isinstance(other, ChatMessage):
            return self.role == other.role and self.content == other.content
        return False

class ConversationHistory:
    def __init__(self, context_limit=10):
        self.messages = []
        self.context_limit = context_limit
    def add_message(self, role, content):
        if role not in ["system", "user", "assistant", "tool"]:
            raise ValueError(f"Invalid role: {role}. Must be 'system', 'user', 'assistant', or 'tool'.")
        if not isinstance(content, str):
            raise ValueError("Content must be a string.")
        if not content.strip():
            raise ValueError("Content cannot be empty or whitespace.")
        if len(content) > 10000:
            raise ValueError("Content is too long. Must be less than 10,000 characters.")
        if len(self.messages) >= self.context_limit:
            self.summarize() 
        self.messages.append(ChatMessage(role, content))
    def to_list(self):
        return [msg.to_dict() for msg in self.messages]
    def __str__(self):
        return "\n".join(msg.to_string() for msg in self.messages)
    def __repr__(self):
        return f"ConversationHistory(messages={self.messages!r})"
    def __eq__(self, other):
        if isinstance(other, ConversationHistory):
            return self.messages == other.messages
        return False
    def clear(self):
        self.messages = []
    def size(self):
        return len(self.messages)
    def recent(self, n=5):
        return self.messages[-n:]
    def recent_as_list(self, n=5):
        return [msg.to_dict() for msg in self.recent(n)]
    def summarize(self):
        # TODO: use openAI to summarize the conversation history into a shorter form, for now just keep the last half of the messages
        self.messages = self.messages[-self.context_limit//2:]

class Chatbot:
    def __init__(self, name, system_prompt, context_limit=10):
        self.name = name
        self.history = ConversationHistory(context_limit=context_limit)
        self.history.add_message("system", system_prompt)
        # self.history.add_message("system", self.data_provider.build_context_message(self.database))
    def handle_conversation(self):
        print(f"Welcome to the AI Chatbot {self.name}! Type 'bye', 'quit', 'exit' to close the chat.")

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["bye", "quit", "exit"]:
                break

            self.history.add_message("user", user_input)

            if self.history.size() > self.history.context_limit:
                self.history.summarize()

            request_messages = self.history.recent_as_list(self.history.context_limit)
            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=request_messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            assistant_message = response.choices[0].message
            tool_calls = assistant_message.tool_calls or []

            if tool_calls:
                # Add the assistant message with tool call metadata exactly as the API expects.
                request_messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content or "",
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": call.type,
                                "function": {
                                    "name": call.function.name,
                                    "arguments": call.function.arguments,
                                },
                            }
                            for call in tool_calls
                        ],
                    }
                )

                for call in tool_calls:
                    try:
                        call_arguments = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        call_arguments = {}

                    # Let the developer know which tool call is being executed and with what arguments, 
                    # this will help with debugging and understanding the flow of the conversation when tools are involved.
                    print(f"Executing tool call: {call.function.name} with arguments: {call_arguments}")

                    tool_result = execute_tool_call(
                        call.function.name,
                        call_arguments,
                    )
                    
                    request_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(tool_result),
                        }
                    )

                follow_up = client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=request_messages,
                    tools=TOOLS,
                    tool_choice="none",
                )
                result = (follow_up.choices[0].message.content or "").strip()
            else:
                result = (assistant_message.content or "").strip()

            print(f"{self.name}:", result)
            self.history.add_message("assistant", result)

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
    {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Legacy search for the clean website knowledge base JSON.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user question or search query."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return.",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
},
]

TOOL_HANDLERS = {
    "search_unified_knowledge": search_unified_knowledge,
    "search_knowledge_base": search_knowledge_base,
}

def execute_tool_call(function_name, function_arguments):
    handler = TOOL_HANDLERS.get(function_name)
    if not handler:
        return {"error": f"No handler for tool: {function_name}"}
    try:
        return handler(**function_arguments)
    except Exception as exc:
        return {"error": f"Error executing tool {function_name}: {exc}"}

if __name__ == "__main__":
    chatbot = Chatbot("Owlbear", SYSTEM_PROMPT, context_limit=CONTEXT_LIMIT)
    chatbot.handle_conversation()