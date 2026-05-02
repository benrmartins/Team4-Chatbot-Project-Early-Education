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


def import_app_module():
    install_openai_stubs()
    for module_name in [
        "app",
        "chatbot.chatbot_api",
        "chatbot.tool_calls.registry",
        "chatbot.tool_calls",
        "chatbot",
    ]:
        sys.modules.pop(module_name, None)
    return importlib.import_module("app")


class AppFallbackTests(unittest.TestCase):
    def test_load_benchmark_chatbot_falls_back_to_default_when_variant_db_is_missing(self):
        app_module = import_app_module()

        with patch.object(
            app_module,
            "_load_best_variant",
            side_effect=FileNotFoundError("missing hpc variant db"),
        ), patch.object(app_module, "Chatbot") as chatbot_class:
            bot = app_module._load_benchmark_chatbot()

        self.assertEqual(chatbot_class.return_value, bot)
        chatbot_class.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
