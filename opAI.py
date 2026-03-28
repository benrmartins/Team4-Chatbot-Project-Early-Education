# make sure to open the terminal in the same directory as this file to run it
# to install dependencies run the following commands in the terminal:
#   source ./.venv/bin/activate (for linux/bash) or ./.venv/Scripts/activate.ps1 (for windows) 
#   pip3 install -r requirements.txt or py -m pip install -r requirements.txt
from openai import OpenAI 
from dotenv import load_dotenv 
import json
import os
import re
import zipfile
from pathlib import Path
from database import MockDatabase
from json_retrieval import search_knowledge_base
# We're using dotenv to load the env variables from the .env file
# you must first create a .env file in the same directory as this file and add your openrouter key like this:
# OPENROUTER_API_KEY="enter whatever your openrouter key is"
# This will prevent having to enter it everytime you open the terminal
load_dotenv()

# to test if openai works <type> py -c "import openai; print(openai.__file__)"
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",

    # if setting the env to your openrouter key doesn't work
    # then we can hard code the key in. The previous way is to avoid
    # any leaks of our api keys. <look down below for hardcode>

    # api_key="ENTER YOUR KEY HERE",
    # base_url="https://openrouter.ai/api/v1",
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


# Use a class to represent each message in the conversation history, this will make it easier to manage the messages and their metadata (like role, content, etc.) 
# and also make it easier to convert them to the format that the openAI API expects when we want to use them in a chat completion request.
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

# Conversation History class to keep track of the conversation history and summarize it when it gets too long.
# The chatbot implements a handle_conversation function that will use the conversation history when it is relevant to the client.
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

# Main chatbot implementation, handles queries and toolcalls in a loop to provide a response
class Chatbot:
    def __init__(self, name, system_prompt, context_limit=10, data_provider=None):
        self.name = name
        self.history = ConversationHistory(context_limit=context_limit)
        self.data_provider = data_provider or MockDatabase()
        self.database = self.data_provider.load()
        self.history.add_message("system", system_prompt)
        self.history.add_message("system", self.data_provider.build_context_message(self.database))
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

def extract_docx_text(file_path):
    """Read text content from a .docx without external dependencies."""
    try:
        with zipfile.ZipFile(file_path, "r") as archive:
            with archive.open("word/document.xml") as xml_file:
                xml_content = xml_file.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        return "", f"Failed to read {file_path}: {exc}"

    text = re.sub(r"<[^>]+>", " ", xml_content)
    text = re.sub(r"\s+", " ", text).strip()
    return text, None

########################################################
### INSTRUCTIONS FOR TOOL CALLS IMPLEMENTATION BELOW ###
########################################################
# we can add more tool calls here as needed, for example if we want to do a google search or 
# something like that we can add a tool call for that 

# First implement the function to handle that tool call like below
def keyword_search_docx(folder_path, keywords, recursive=True, max_results=50):
    """Search .docx files in a folder for one or more keywords."""
    return "This is a placeholder result for keyword_search_docx. Implement the actual search logic here."

# def another_tool(param1, param2):
#     """Example of another tool call that the chatbot can use."""
#     # Implement the logic for this tool here, for example if this tool is supposed to do a google search then we would implement the google search logic here and return the results in a format that the chatbot can use.
#     return {
#         "param1": param1,
#         "param2": param2,
#         "result": f"Processed {param1} and {param2} in another_tool.",
#     }

# Next we need to add the tool call to the list of tools that the chatbot can use like below
# Be sure to give a unique name to the tool call and a clear description of what it does and what parameters it takes so that the chatbot can use it effectively when needed.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "keyword_search_docx",
            "description": "Search a folder of DOCX files for one or more keywords.",
            "parameters": {
                "type": "object",
                "properties": {
                    "folder_path": {
                        "type": "string",
                        "description": "Path to folder containing DOCX files (e.g., Google Drive sync folder).",
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "One or more keywords to search for.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to search subfolders.",
                        "default": True,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max number of matching files to return.",
                        "default": 50,
                    },
                },
                "required": ["folder_path", "keywords"],
            },
        },
    },
    {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Search the scraped Early Education Leaders JSON knowledge base for relevant pages, blog posts, and institute information.",
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
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "another_tool",
    #         "description": "Description of another tool that the chatbot can use.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "param1": {
    #                     "type": "string",
    #                     "description": "Description of param1.",
    #                 },
    #                 "param2": {
    #                     "type": "integer",
    #                     "description": "Description of param2.",
    #                 },
    #             },
    #             "required": ["param1", "param2"],
    #         },
    #     },
    # } 
    # Add more tool definitions here as needed
]

# Then add the function to the handlers so the chatbot can execute it by string
TOOL_HANDLERS = {
    "keyword_search_docx": keyword_search_docx,
    "search_knowledge_base": search_knowledge_base,
    # "another_tool": another_tool,  # Add more handlers here as needed
}

# Finally we need to implement the function that will execute the tool call when the chatbot decides to use it, 
# this function will look up the tool call in the TOOL_HANDLERS dict and execute it with the provided arguments 
# and return the result back to the chatbot so it can use it in the conversation. The code for that is below.
# Be sure to handle any exceptions that may occur during the execution of the tool call and return an appropriate 
# error message back to the chatbot if something goes wrong.
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