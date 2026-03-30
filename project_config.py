from pathlib import Path

# Absolute project root used to derive all other paths.
PROJECT_ROOT = Path(__file__).resolve().parent
# Folder containing pipeline outputs and retrieval datasets.
DATA_DIR = PROJECT_ROOT / "data"

# OAuth client credentials JSON used by Google Drive ingestion.
CREDENTIALS_PATH = PROJECT_ROOT / "credentials.json"

# Raw source outputs written by each crawler.
DEFAULT_DRIVE_OUTPUT = DATA_DIR / "drive_data.json"
DEFAULT_WEB_OUTPUT = DATA_DIR / "web_data.json"
# Unified chunked payload consumed by retrieval tools.
DEFAULT_VECTOR_OUTPUT = DATA_DIR / "unified_vector_data.json"

# Primary retrieval file used by search_unified_knowledge.
UNIFIED_KNOWLEDGE_BASE_PATH = DEFAULT_VECTOR_OUTPUT

# OpenRouter API endpoint and model used by chatbot completions.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openai/gpt-4o-mini"

# Conversation history policy and CLI experience settings.
CHAT_CONTEXT_LIMIT = 10
CHAT_MAX_MESSAGE_CHARS = 10_000
CHAT_INPUT_PROMPT = "You: "
CHATBOT_NAME = "Knowledge Assistant"

# Intro text shown when a chat session starts.
ONBOARD_PROMPT = (
    "Welcome! I can answer questions grounded in your configured data sources. "
    "Ask me anything about the content you ingested."
)

SYSTEM_PROMPT = """
You are a grounded assistant for user-configured data sources.

Your goals:
- Provide accurate, helpful answers grounded in retrieved source material.
- Be transparent about uncertainty.
- Never fabricate facts, citations, links, dates, names, or policies.

Grounding and tools:
- If retrieval/tool results are available, use them as the primary source of truth.
- If tool output conflicts with prior chat context, trust the tool output and update your understanding accordingly.
- If no reliable source is available, say "I don't know based on the available sources."

Citation rules:
- For factual claims from retrieved sources, include citations at the end.
- Use this format: Title: <source title>, URL: <source url>.
- If multiple sources are used, list each on a new line.
- If no sources are used, do not include citations.

Response style:
- Be concise, clear, and supportive.
- Prefer bullet points for steps, requirements, or lists.
""".strip()

# Template inserted as a system message after tool calls execute.
TOOL_REPROMPT_TEMPLATE = """Tool call results:
{tool_results}

Incorporate these results into your next response to the user, using them as needed to answer the user's question.
If the tool results contain factual information relevant to the user's query, use that information in your response and cite it appropriately.
If the tool results indicate an error or issue with retrieving information, acknowledge that in your response and do not attempt to fabricate an answer.
Always prioritize accuracy and grounding in the provided tool results when formulating your response."""

# Default Google Drive folder target for Drive crawler runs.
DEFAULT_DRIVE_FOLDER_ID = "0AFlQ37_lQJh8Uk9PVA"
DEFAULT_DRIVE_FOLDER_LINK = "https://drive.google.com/drive/folders/0AFlQ37_lQJh8Uk9PVA"
# Seed URLs for website crawling.
DEFAULT_WEBSITE_SEED_URLS = [
    "https://www.umb.edu/early-education-leaders-institute/",
    "https://blogs.umb.edu/earlyed/",
]

# Pipeline execution controls (config-only, no CLI switches).
PIPELINE_RUN_DRIVE = True
PIPELINE_RUN_WEB = True
PIPELINE_CHUNK_SIZE = 700
PIPELINE_CHUNK_OVERLAP = 120


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "CREDENTIALS_PATH",
    "DEFAULT_DRIVE_OUTPUT",
    "DEFAULT_WEB_OUTPUT",
    "DEFAULT_VECTOR_OUTPUT",
    "UNIFIED_KNOWLEDGE_BASE_PATH",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_MODEL",
    "CHAT_CONTEXT_LIMIT",
    "CHAT_MAX_MESSAGE_CHARS",
    "CHAT_INPUT_PROMPT",
    "CHATBOT_NAME",
    "ONBOARD_PROMPT",
    "SYSTEM_PROMPT",
    "TOOL_REPROMPT_TEMPLATE",
    "DEFAULT_DRIVE_FOLDER_ID",
    "DEFAULT_DRIVE_FOLDER_LINK",
    "DEFAULT_WEBSITE_SEED_URLS",
    "PIPELINE_RUN_DRIVE",
    "PIPELINE_RUN_WEB",
    "PIPELINE_CHUNK_SIZE",
    "PIPELINE_CHUNK_OVERLAP",
]
