import json
import os
from dataclasses import dataclass
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI

from chatbot.tool_calls import TOOLS, execute_tool_call
from project_config import (
    CHATBOT_NAME,
    CHAT_CONTEXT_LIMIT,
    CHAT_INPUT_PROMPT,
    CHAT_MAX_MESSAGE_CHARS,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    ONBOARD_PROMPT,
    SYSTEM_PROMPT,
    TOOL_REPROMPT_TEMPLATE,
)

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=OPENROUTER_BASE_URL,
)


def get_open_ai_response(messages):
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    return response


ALLOWED_MESSAGE_ROLES = {"system", "user", "assistant", "tool"}


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class ConversationHistory:
    def __init__(self, context_limit: int = CHAT_CONTEXT_LIMIT):
        self.messages: list[ChatMessage] = []
        self.context_limit = context_limit

    def add_message(self, role: str, content: str) -> None:
        if role not in ALLOWED_MESSAGE_ROLES:
            raise ValueError(
                f"Invalid role: {role}. Must be 'system', 'user', 'assistant', or 'tool'."
            )
        if not isinstance(content, str):
            raise ValueError("Content must be a string.")
        if not content.strip():
            raise ValueError("Content cannot be empty or whitespace.")
        if len(content) > CHAT_MAX_MESSAGE_CHARS:
            raise ValueError(f"Content is too long. Must be less than {CHAT_MAX_MESSAGE_CHARS:,} characters.")
        if len(self.messages) >= self.context_limit:
            self.summarize()
        self.messages.append(ChatMessage(role, content))

    def to_list(self) -> list[dict[str, str]]:
        return [msg.to_dict() for msg in self.messages]

    def summarize(self) -> None:
        self.messages = self.messages[-self.context_limit // 2 :]


class Chatbot:
    def __init__(self, system_prompt=SYSTEM_PROMPT, context_limit=CHAT_CONTEXT_LIMIT, name=CHATBOT_NAME):
        self.name = name
        self.history = ConversationHistory(context_limit=context_limit)
        self.history.add_message("system", system_prompt)
        self.input_prompt = CHAT_INPUT_PROMPT
        self.onboard_prompt = ONBOARD_PROMPT

    def create_response(self, user_input: str, status_callback: Callable[[str], None] | None = None) -> str:
        if not isinstance(user_input, str) or not user_input.strip():
            return "Please enter a non-empty question."

        self.history.add_message("user", user_input)

        request_messages = self.history.to_list()
        response = get_open_ai_response(request_messages)
        assistant_message = response.choices[0].message

        return self._handle_response(assistant_message, status_callback=status_callback)

    def _handle_response(self, message: Any, status_callback: Callable[[str], None] | None = None) -> str:
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

        combined_results = "\n".join(results)

        if len(combined_results) > 2000:
            combined_results = combined_results[:2000] + "\n[Truncated additional results]"

        generated_reprompt = TOOL_REPROMPT_TEMPLATE.format(tool_results=combined_results)

        if status_callback:
            status_callback("generating_final")

        self.history.add_message("system", generated_reprompt)
        reprompt_messages = self.history.to_list()
        response = get_open_ai_response(reprompt_messages)
        final_message = response.choices[0].message
        result = (final_message.content or "").strip()
        self.history.add_message("assistant", result)
        return result
