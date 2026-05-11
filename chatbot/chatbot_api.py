import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable

from dotenv import load_dotenv
from openai import OpenAI
from logs.cost import log_api_usage

from chatbot.tool_calls import TOOLS, execute_tool_call
from ingestion_pipeline.schema import JSONDict
from project_config import (
    CHATBOT_NAME,
    CHAT_CONTEXT_LIMIT,
    CHAT_INPUT_PROMPT,
    CHAT_MAX_MESSAGE_CHARS,
    DEFAULT_VECTOR_DB_PATH,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    ONBOARD_PROMPT,
    SYSTEM_PROMPT,
    TOOL_REPROMPT_TEMPLATE,
)

ALLOWED_MESSAGE_ROLES = {"system", "user", "assistant", "tool"}

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url=OPENROUTER_BASE_URL)

def get_open_ai_response(messages, tool_choice: str = "auto"):
    request_kwargs = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "tool_choice": tool_choice,
    }
    if tool_choice != "none":
        request_kwargs["tools"] = TOOLS
    response = client.chat.completions.create(**request_kwargs)

    usage = getattr(response, "usage", None)
    if usage is not None:
        try:
            log_api_usage(
                model=getattr(response, "model", OPENROUTER_MODEL),
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
            )
        except Exception:
            pass

    return response

def get_name_from_query(query: str) -> str | None:
    prompt = """Create a Conversation Name for the following query: {query}. 
    The name should be concise (ideally under 5 words), descriptive of the query, 
    and engaging. Avoid generic names like 'Chat' or 'Conversation'. If the query is 
    about a specific topic, try to include that in the name. If the query is a question,
    consider phrasing the name as a question as well. 
    Return only the name without any additional text.""".format(query=query)
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b:free",
            messages=[{"role": "system", "content": prompt}],
        )
        name = response.choices[0].message.content
    except Exception as e:
        print(f"Error occurred while fetching conversation name: {e}")
        name = "Unnamed Conversation"

    return name

@dataclass
class ChatTurnPayload:
    role: str
    content: str
    citations: list[dict[str, str]]
    evidence: list[JSONDict]
    retrieval: JSONDict | None


class ConversationHistory:
    def __init__(self, context_limit: int = CHAT_CONTEXT_LIMIT):
        self.messages: list[ChatTurnPayload] = []
        self.context_limit = context_limit

    def add_message(self, message: ChatTurnPayload) -> None:

        if message.role not in ALLOWED_MESSAGE_ROLES:
            raise ValueError(
                f"Invalid role: {message.role}. Must be 'system', 'user', 'assistant', or 'tool'."
            )
        if not isinstance(message.content, str):
            raise ValueError("Content must be a string.")
        if not message.content.strip():
            raise ValueError("Content cannot be empty or whitespace.")
        if len(message.content) > CHAT_MAX_MESSAGE_CHARS:
            raise ValueError(f"Content is too long. Must be less than {CHAT_MAX_MESSAGE_CHARS:,} characters.")
        if len(self.messages) >= self.context_limit:
            self.summarize()
        self.messages.append(message)

    def to_list(self) -> list[dict[str, str]]:
        return [asdict(msg) for msg in self.messages]

    def summarize(self) -> None:
        # TO DO: Implement a summarization method to condense conversation history when context limit is reached.
        self.messages = self.messages[-self.context_limit // 2 :]


class Chatbot:
    def __init__(
            self, 
            system_prompt=SYSTEM_PROMPT, 
            context_limit=CHAT_CONTEXT_LIMIT, 
            name=CHATBOT_NAME, 
            input_prompt=CHAT_INPUT_PROMPT, 
            onboard_prompt=ONBOARD_PROMPT,
            database_path: str = str(DEFAULT_VECTOR_DB_PATH) + "_default.sqlite",
            chat_id: str | None = None,
        ):
        self.name = None
        self.chat_id = chat_id
        self.history = ConversationHistory(context_limit=context_limit)
        self.history.add_message(ChatTurnPayload(role="system", content=system_prompt, citations=[], evidence=[], retrieval=None))
        self.input_prompt = input_prompt
        self.onboard_prompt = onboard_prompt
        self.database_path = database_path

    def get_history(self) -> list[dict[str, str]]:
        return self.history.to_list()

    def create_response(
        self,
        user_input: str,
        status_callback: Callable[[str], None] | None = None,
    ) -> JSONDict:
        if not isinstance(user_input, str) or not user_input.strip():
            return asdict(ChatTurnPayload(
                role="assistant",
                content="Please enter a non-empty question.",
                citations=[],
                evidence=[],
                retrieval=None,
            ))

        self.history.add_message(ChatTurnPayload(role="user", content=user_input, citations=[], evidence=[], retrieval=None))
        self.get_name_from_query(user_input)
        request_messages = self.history.to_list()
        response = get_open_ai_response(request_messages, tool_choice="auto")
        assistant_message = response.choices[0].message

        payload = self._handle_response_payload(assistant_message, status_callback=status_callback)
        return asdict(payload)

    def _handle_response_payload(
        self,
        message: Any,
        status_callback: Callable[[str], None] | None = None,
    ) -> ChatTurnPayload:
        tool_calls = message.tool_calls or []
        if not tool_calls:
            reply = self._sanitize_reply(message.content)
            payload = ChatTurnPayload(role="assistant", content=reply, citations=[], evidence=[], retrieval=None)
            self.history.add_message(payload)
            return payload

        if status_callback:
            status_callback("running_tools")

        tool_results: list[JSONDict] = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            arguments_raw = tool_call.function.arguments or "{}"

            try:
                arguments = json.loads(arguments_raw)
            except json.JSONDecodeError:
                arguments = {}

            try:
                tool_result = execute_tool_call(
                    function_name,
                    arguments,
                    tool_context={"database_path": self.database_path},
                )
                tool_results.append(
                    {
                        "name": function_name,
                        "arguments": arguments,
                        "result": tool_result,
                    }
                )
            except Exception as e:
                tool_results.append(
                    {
                        "name": function_name,
                        "arguments": arguments,
                        "result": {"error": f"Error executing tool {function_name}: {str(e)}"},
                    }
                )

        formatted_results = self._format_tool_results_for_prompt(tool_results)
        generated_reprompt = TOOL_REPROMPT_TEMPLATE.format(tool_results=formatted_results)

        if status_callback:
            status_callback("generating_final")

        self.history.add_message(ChatTurnPayload(role="system", content=generated_reprompt, citations=[], evidence=[], retrieval=None))
        reprompt_messages = self.history.to_list()
        response = get_open_ai_response(reprompt_messages, tool_choice="none")
        final_message = response.choices[0].message
        reply = self._sanitize_reply(final_message.content)

        citations, evidence, retrieval = self._build_evidence_payload(tool_results)
        payload = ChatTurnPayload(role="assistant", content=reply, citations=citations, evidence=evidence, retrieval=retrieval)
        self.history.add_message(payload)
        return payload

    def _sanitize_reply(self, content: str | None) -> str:
        reply = (content or "").strip()
        if reply:
            return reply
        return (
            "I couldn't generate a grounded answer from the available sources. "
            "Please try rephrasing your question."
        )

    def _format_tool_results_for_prompt(self, tool_results: list[JSONDict]) -> str:
        prompt_blocks: list[str] = []
        for tool_result in tool_results:
            name = tool_result["name"]
            result = tool_result["result"]
            if isinstance(result, dict) and result.get("results") is not None:
                lines = [
                    f"Tool: {name}",
                    f"Query: {result.get('query', '')}",
                    f"Low confidence: {result.get('low_confidence', False)}",
                ]
                for item in result.get("results", [])[:5]:
                    lines.extend(
                        [
                            f"Rank {item.get('rank', '?')} | Title: {item.get('title', 'Untitled')}",
                            f"URL: {item.get('url', '')}",
                            f"Matched terms: {', '.join(item.get('matched_terms', [])) or 'n/a'}",
                            f"Why it matched: {', '.join(item.get('match_reasons', [])) or 'n/a'}",
                            f"Evidence: {item.get('evidence_snippet', '')}",
                        ]
                    )
                prompt_blocks.append("\n".join(lines))
                continue
            prompt_blocks.append(f"Tool: {name}\nResult: {result}")

        combined_results = "\n\n".join(prompt_blocks).strip()
        if len(combined_results) <= 3000:
            return combined_results
        return combined_results[:3000] + "\n[Truncated additional results]"

    def _build_evidence_payload(
        self,
        tool_results: list[JSONDict],
    ) -> tuple[list[JSONDict], list[JSONDict], JSONDict | None]:
        retrieval = None
        for tool_result in tool_results:
            result = tool_result["result"]
            if isinstance(result, dict) and "results" in result:
                retrieval = result
                break

        if retrieval is None:
            return [], [], None

        citations: list[JSONDict] = []
        seen_citations: set[tuple[str, str]] = set()
        evidence: list[JSONDict] = []

        for item in retrieval.get("results", []):
            citation = item.get("citation") or {
                "title": item.get("title", "Untitled"),
                "url": item.get("url", ""),
            }
            citation_key = (citation.get("title", ""), citation.get("url", ""))
            if citation_key not in seen_citations:
                citations.append(citation)
                seen_citations.add(citation_key)

            evidence.append(
                {
                    "rank": item.get("rank"),
                    "title": item.get("title", "Untitled"),
                    "url": item.get("url", ""),
                    "source_type": item.get("source_type", "unknown"),
                    "matched_terms": item.get("matched_terms", []),
                    "match_reasons": item.get("match_reasons", []),
                    "snippet": item.get("evidence_snippet") or item.get("excerpt", ""),
                    "document_id": item.get("document_id", ""),
                    "chunk_id": item.get("chunk_id", ""),
                    "score": item.get("score"),
                }
            )

        return citations, evidence, retrieval
    
    def get_name_from_query(self, query: str) -> str | None:
        if self.name is not None:
            return
        self.name = get_name_from_query(query)