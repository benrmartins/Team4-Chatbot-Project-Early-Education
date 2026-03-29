from openai import OpenAI 
from dotenv import load_dotenv 
import json
import os
from typing import Callable
from tool_calls.registry import execute_tool_call, TOOLS
from chatbot.prompts import ONBOARD_PROMPT, SYSTEM_PROMPT, format_tool_reprompt

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

def get_open_ai_response(messages):
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    return response

# Define a context limit for the conversation history, after which we will summarize to keep the most relevant information in context for the assistant.
CONTEXT_LIMIT = 10

class ChatMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content
    def to_dict(self):
        return {"role": self.role, "content": self.content}

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
    def clear(self):
        self.messages = []
    def size(self):
        return len(self.messages)
    def summarize(self):
        # TODO: use openAI to summarize the conversation history into a shorter form, for now just keep the last half of the messages
        self.messages = self.messages[-self.context_limit//2:]

class Chatbot:
    def __init__(self, system_prompt=SYSTEM_PROMPT, context_limit=CONTEXT_LIMIT, name="Owlbear"):
        self.name = name
        self.history = ConversationHistory(context_limit=context_limit)
        self.history.add_message("system", system_prompt)
        self.input_prompt = "You: "
        self.onboard_prompt = ONBOARD_PROMPT

    # main entry point for handling user input, generating assistant responses, and managing tool calls
    def create_response(self, user_input, status_callback: Callable[[str], None] = None):
        if not isinstance(user_input, str) or not user_input.strip():
            return "Please enter a non-empty question."

        # add user message to conversation history and summarize if we are at the context limit
        self.history.add_message("user", user_input)
        if self.history.size() > self.history.context_limit:
            self.history.summarize()

        # Prepare the messages for the API call as a list of dicts.
        request_messages = self.history.to_list()

        # Call the OpenAI API to get a response based on the recent conversation history.
        response = get_open_ai_response(request_messages)
        assistant_message = response.choices[0].message

        # Process the response before returning it
        return self._handle_response(assistant_message, status_callback=status_callback)

    # process the assistant message, execute any tool calls if present, 
    # then reprompt the assistant with the tool results included in the context 
    # to generate a final response for the user
    def _handle_response(self, message, status_callback: Callable[[str], None] = None):
        
        # process tool calls if present in the assistant message, execute them and collect results
        tool_calls = message.tool_calls or []
        if not tool_calls:
            return (message.content or "").strip()

        if status_callback:
            status_callback("running_tools")

        results = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments_raw = tool_call.function.arguments or "{}"

            try:
                arguments = json.loads(arguments_raw)
            except json.JSONDecodeError:
                arguments = {}

            try:
                tool_result = execute_tool_call(function_name, arguments)
                results.append(f"Tool: {function_name}, Result: {tool_result}")
            except Exception as e:
                error_msg = f"Error executing tool {function_name}: {str(e)}"
                results.append(error_msg)

        # combine tool results into a single string to include in the reprompt to the assistant, truncating if necessary to avoid exceeding token limits
        combined_results = "\n".join(results)

        if len(combined_results) > 2000:
            combined_results = combined_results[:2000] + "\n[Truncated additional results]"

        generated_reprompt = format_tool_reprompt(combined_results)

        # reprompt the assistant with the tool results included in the context, generate a final response for the user, and add it to the conversation history
        if status_callback:
            status_callback("generating_final")

        self.history.add_message("system", generated_reprompt)
        reprompt_messages = self.history.to_list()
        response = get_open_ai_response(reprompt_messages)
        final_message = response.choices[0].message
        result = (final_message.content or "").strip()
        self.history.add_message("assistant", result)
        return result