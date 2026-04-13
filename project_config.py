from pathlib import Path

# Absolute project root used to derive all other paths.
PROJECT_ROOT = Path(__file__).resolve().parent
# Folder containing pipeline outputs and retrieval datasets.
DATA_DIR = PROJECT_ROOT / "data"
COST_LOG_PATH = PROJECT_ROOT / "cost_events.jsonl"

# OAuth client credentials JSON used by Google Drive ingestion.
CREDENTIALS_PATH = PROJECT_ROOT / "credentials.json"

# Raw source outputs written by web crawlers. These are not directly consumed by retrieval tools but can be useful for debugging and pipeline iteration.
DEFAULT_WEB_OUTPUT = DATA_DIR / "web_data.json"

# Unified chunked payload consumed by retrieval tools.
DEFAULT_CHUNK_OUTPUT = DATA_DIR / "unified_chunk_data.json"

# Primary retrieval file used by search_unified_knowledge.
UNIFIED_KNOWLEDGE_BASE_PATH = DEFAULT_CHUNK_OUTPUT
UNIFIED_KNOWLEDGE_BASE_FALLBACK_PATH = PROJECT_ROOT / "unified_chunk_data.json"

# OpenRouter API endpoint and model used by chatbot completions.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openai/gpt-4o-mini"

# Conversation history policy and CLI experience settings.
CHAT_CONTEXT_LIMIT = 10
CHAT_MAX_MESSAGE_CHARS = 10_000
CHAT_INPUT_PROMPT = "You: "
CHATBOT_NAME = "Owlbot"

# Intro text shown when a chat session starts.
ONBOARD_PROMPT = """
    Welcome to the Early Education Leaders Institute Chatbot!
    I can answer your questions about our programs, resources, 
    and related early education topics. You can ask me anything related 
    to the Early Education Leaders Institute. How can I help you today?
"""

SYSTEM_PROMPT = """You are Owlbear, an Early Education Chatbot for the Early Education Leaders Institute.

Your goals:
- Provide accurate, helpful answers grounded in retrieved source material.
- Be transparent about uncertainty
- Never fabricate facts, citations, links, dates, names, or policies.

Grounding and tools:
- If retrieval/tool results are available, use them as the primary source of truth.
- If tool output conflicts with prior chat context, trust the tool output and update your understanding accordingly.
- If no reliable source is available, say "I don't know based on the available sources."
- Then suggest contacting EarlyEdLeaders@umb.edu for further assistance.

Citation rules:
- For any factual claim from retrieved sources, include citations at the end.
- Use this format: Title: <source title>, URL: <source url>.
- If multiple sources are used, list each on a new line.
- If no sources are used, do not include a citation section, do not cite sources you did not use, and do not fabricate sources.

Response style:
- Be concise, clear and supportive.
- If the user asks a broad question, ask one brief clarifying question to better understand their needs before answering.
- Prefer bullet points for steps, requirements, or lists.
- If the user asks for action not covered by available sources, explain the limitations and offer the closest supported help.

Scope:
- Focus on the Early Education Leaders Institute, related retrieved materials, its programs, resources, and related early education topics.
- If a question is out of scope, state that clearly and redirect to relevant institute topics or suggest contacting EarlyEdLeaders@umb.edu."""

# Template inserted as a system message after tool calls execute.
TOOL_REPROMPT_TEMPLATE = """Tool call results:
{tool_results}

Incorporate these results into your next response to the user, using them as needed to answer the user's question.
If the tool results contain factual information relevant to the user's query, use that information in your response and cite it appropriately.
If low confidence is False and at least one result is present, provide a direct answer from the top-ranked evidence and do not say you lack information.
Treat the top-ranked evidence snippet as the primary grounding, then use additional results only if they add non-duplicative detail.
If the tool results indicate an error or issue with retrieving information, acknowledge that in your response and do not attempt to fabricate an answer.
Always prioritize accuracy and grounding in the provided tool results when formulating your response."""

# Default Google Drive folder target for Drive crawler runs. This can be overridden by providing a different folder link when running the pipeline.
DEFAULT_DRIVE_FOLDER_URLS = [
    "https://drive.google.com/drive/folders/0AFlQ37_lQJh8Uk9PVA"
]
# Seed URLs for website crawling. This can be overridden by providing different URLs when running the pipeline.
DEFAULT_WEBSITE_SEED_URLS = [
    "https://www.umb.edu/early-education-leaders-institute/",
    "https://blogs.umb.edu/earlyed/",
]

# Pipeline execution controls
CRAWLER_DEPTH_LIMIT = 5
PIPELINE_RUN_DRIVE = True
PIPELINE_RUN_WEB = True
DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_MIN_CHUNK_SIZE = 220
EXCLUDED_PATH_SEGMENTS = {
        "tag",
        "tags",
        "category",
        "categories",
        "author",
        "feed",
        "search",
        "archive",
        "archives",
        "comments",
}
BLOCKED_FILE_EXTENSIONS = (
        ".pdf",
        ".jpg",
        ".png",
        ".zip",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".mp4",
        ".mp3",
)
MIN_CONTENT_CHARS = 100

# Embedding and vector store settings
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_BATCH_SIZE = 32
DEFAULT_VECTOR_DB_PATH = DATA_DIR / "db_store"

# Deterministic variant-testing matrix for retrieval experiments.
VARIANT_TEST_CHUNK_SIZES = [DEFAULT_CHUNK_SIZE, 500, 1000]
VARIANT_TEST_CHUNK_OVERLAPS = [DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_OVERLAP // 2]
VARIANT_TEST_EMBEDDING_DIMS = [DEFAULT_EMBEDDING_DIM, 1536]
VARIANT_TEST_BATCH_SIZES = [DEFAULT_BATCH_SIZE, DEFAULT_BATCH_SIZE * 2]
VARIANT_TEST_EMBEDDING_METHODS = ["openai_small"]
VARIANT_TEST_BENCHMARK_PATH = PROJECT_ROOT / "evaluation" / "retrieval_benchmark.json"
VARIANT_TEST_MAX_RESULTS = 20

# Retrieval tuning
RETRIEVAL_CANDIDATE_POOL = 40
RETRIEVAL_MAX_RESULTS_PER_DOCUMENT = 1
RETRIEVAL_LOW_CONFIDENCE_MIN_SCORE = 3.5
RETRIEVAL_LOW_CONFIDENCE_MIN_COVERAGE = 0.34
RETRIEVAL_SNIPPET_MAX_CHARS = 320
RETRIEVAL_SYNONYM_MAP = {
    "multilingual": ["bilingual", "language", "languages", "linguistically"],
    "bilingual": ["multilingual", "language", "languages", "linguistically"],
    "teachers": ["educators", "teacher", "instructors"],
    "teacher": ["educator", "teachers", "instructor"],
    "educators": ["teacher", "teachers", "workforce"],
    "support": ["help", "supports", "supporting", "assistance"],
    "fellowship": ["fellow", "fellows", "scholarship"],
    "fellows": ["fellowship", "fellow", "scholarship"],
    "bachelor": ["degree", "ba", "undergraduate"],
    "degree": ["bachelor", "undergraduate", "college"],
    "change": ["improvement", "improve", "innovation"],
    "policy": ["systems", "advocacy", "policymakers"],
    "infants": ["infant", "toddler", "toddlers"],
    "toddlers": ["toddler", "infant", "infants"],
    "contact": ["email", "reach", "learn", "information"],
}


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "COST_LOG_PATH",
    "CREDENTIALS_PATH",
    "DEFAULT_WEB_OUTPUT",
    "DEFAULT_CHUNK_OUTPUT",
    "DEFAULT_VECTOR_DB_PATH",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EMBEDDING_DIM",
    "UNIFIED_KNOWLEDGE_BASE_PATH",
    "UNIFIED_KNOWLEDGE_BASE_FALLBACK_PATH",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_MODEL",
    "CHAT_CONTEXT_LIMIT",
    "CHAT_MAX_MESSAGE_CHARS",
    "CHAT_INPUT_PROMPT",
    "CHATBOT_NAME",
    "ONBOARD_PROMPT",
    "SYSTEM_PROMPT",
    "TOOL_REPROMPT_TEMPLATE",
    "DEFAULT_DRIVE_FOLDER_URLS",
    "DEFAULT_WEBSITE_SEED_URLS",
    "CRAWLER_DEPTH_LIMIT",
    "PIPELINE_RUN_DRIVE",
    "PIPELINE_RUN_WEB",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_MIN_CHUNK_SIZE",
    "RETRIEVAL_CANDIDATE_POOL",
    "RETRIEVAL_MAX_RESULTS_PER_DOCUMENT",
    "RETRIEVAL_LOW_CONFIDENCE_MIN_SCORE",
    "RETRIEVAL_LOW_CONFIDENCE_MIN_COVERAGE",
    "RETRIEVAL_SNIPPET_MAX_CHARS",
    "RETRIEVAL_SYNONYM_MAP",
    "VARIANT_TEST_CHUNK_SIZES",
    "VARIANT_TEST_CHUNK_OVERLAPS",
    "VARIANT_TEST_EMBEDDING_DIMS",
    "VARIANT_TEST_BATCH_SIZES",
    "VARIANT_TEST_EMBEDDING_METHODS",
    "VARIANT_TEST_BENCHMARK_PATH",
    "VARIANT_TEST_MAX_RESULTS",
]
