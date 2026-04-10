import importlib
import sys
import types
import unittest
from unittest.mock import patch


def install_openai_stubs():
    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda: None
    sys.modules["dotenv"] = dotenv_module

    openai_module = types.ModuleType("openai")

    class DummyOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kwargs: None)
            )

    openai_module.OpenAI = DummyOpenAI
    sys.modules["openai"] = openai_module

    openai_types_module = types.ModuleType("openai.types")
    sys.modules["openai.types"] = openai_types_module

    chat_types_module = types.ModuleType("openai.types.chat")
    chat_types_module.ChatCompletionFunctionToolParam = dict
    sys.modules["openai.types.chat"] = chat_types_module


def import_chatbot_api():
    install_openai_stubs()
    for module_name in [
        "chatbot.chatbot_api",
        "chatbot.tool_calls.registry",
        "chatbot.tool_calls",
        "chatbot",
    ]:
        sys.modules.pop(module_name, None)
    return importlib.import_module("chatbot.chatbot_api")


def make_tool_call(name, arguments):
    return types.SimpleNamespace(
        function=types.SimpleNamespace(name=name, arguments=arguments)
    )


def make_response_message(content="", tool_calls=None):
    return types.SimpleNamespace(content=content, tool_calls=tool_calls or [])


def make_completion(message):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


class ChatbotPayloadTests(unittest.TestCase):
    def test_create_response_payload_returns_structured_evidence(self):
        chatbot_api = import_chatbot_api()
        bot = chatbot_api.Chatbot()

        retrieval_payload = {
            "query": "What is ExCELS?",
            "result_count": 1,
            "low_confidence": False,
            "results": [
                {
                    "rank": 1,
                    "title": "ExCELS Overview",
                    "url": "https://example.org/excels",
                    "document_id": "doc-1",
                    "chunk_id": "doc-1::chunk-0000",
                    "source_type": "website",
                    "score": 42.0,
                    "matched_terms": ["excels", "measure"],
                    "match_reasons": ["title match", "phrase match"],
                    "evidence_snippet": "ExCELS developed a new measure of leadership in center-based early care and education settings.",
                    "citation": {"title": "ExCELS Overview", "url": "https://example.org/excels"},
                }
            ],
        }

        first_completion = make_completion(
            make_response_message(
                tool_calls=[make_tool_call("search_unified_knowledge", '{"query":"What is ExCELS?"}')]
            )
        )
        second_completion = make_completion(
            make_response_message(
                content=(
                    "ExCELS is the Early Care and Education Leadership Study. "
                    "Title: ExCELS Overview, URL: https://example.org/excels"
                )
            )
        )

        with patch.object(
            chatbot_api,
            "get_open_ai_response",
            side_effect=[first_completion, second_completion],
        ), patch.object(
            chatbot_api,
            "execute_tool_call",
            return_value=retrieval_payload,
        ):
            payload = bot.create_response_payload("What is ExCELS?")

        self.assertIn("reply", payload)
        self.assertIn("citations", payload)
        self.assertIn("evidence", payload)
        self.assertIn("retrieval", payload)
        self.assertEqual("https://example.org/excels", payload["citations"][0]["url"])
        self.assertEqual(
            "ExCELS developed a new measure of leadership in center-based early care and education settings.",
            payload["evidence"][0]["snippet"],
        )

    def test_create_response_returns_safe_fallback_when_final_message_is_blank(self):
        chatbot_api = import_chatbot_api()
        bot = chatbot_api.Chatbot()

        first_completion = make_completion(
            make_response_message(
                tool_calls=[make_tool_call("search_unified_knowledge", '{"query":"What is ExCELS?"}')]
            )
        )
        second_completion = make_completion(make_response_message(content=""))

        with patch.object(
            chatbot_api,
            "get_open_ai_response",
            side_effect=[first_completion, second_completion],
        ), patch.object(
            chatbot_api,
            "execute_tool_call",
            return_value={"query": "What is ExCELS?", "result_count": 0, "low_confidence": True, "results": []},
        ):
            reply = bot.create_response("What is ExCELS?")

        self.assertTrue(reply)
        self.assertIn("couldn't generate", reply.lower())


if __name__ == "__main__":
    unittest.main()
