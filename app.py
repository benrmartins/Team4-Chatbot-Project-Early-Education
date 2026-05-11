import io
import json
import os
import random
import shutil
import sqlite3
from typing import Any, List
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from uuid import uuid4

from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from markupsafe import Markup
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import pandas as pd
from PyPDF2 import PdfReader
from docx import Document

from chatbot import Chatbot
from ingestion_pipeline.scripts.build_chunk_payload import build_chunk_payload, normalize_text
from ingestion_pipeline.services.vector_store import (
    get_embedder_with_dimension,
    ingest_payload_to_sqlite,
    init_db,
)
from dashboard_support import (
    DASHBOARD_CATEGORIES,
    DASHBOARD_SOURCES_PATH,
    DASHBOARD_UPLOAD_DIR,
    DEFAULT_WEB_OUTPUT,
    PROCESSABLE_SOURCE_STATUSES,
    REVIEW_STALE_DAYS,
    allowed_file,
    build_content_library,
    build_coverage_summary,
    build_dashboard_summary,
    build_library_filters,
    build_needs_attention,
    build_processing_queue,
    build_review_items,
    load_dashboard_sources,
    save_dashboard_sources,
    process_dashboard_sources,
    _create_dashboard_db_backup,
    _delete_existing_document_rows,
    _friendly_source_type,
    _load_dashboard_source_by_id,
    _resolve_dashboard_db_path,
    _resolve_db_from_hpc,
    _reset_dashboard_source_states,
    _reset_database,
    _reset_web_payload,
    _restore_dashboard_backup,
    _unique_upload_filename,
)

from HPC.load_best_variant import load_best_variant
from project_config import UNIFIED_HPC_RESULTS_PATH
from project_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_VECTOR_DB_PATH,
)

app = Flask(__name__)
app.secret_key = "change-me-in-production"

# Session configuration for security
app.config["PERMANENT_SESSION_LIFETIME"] = 3600  # 1 hour session timeout
app.config["SESSION_COOKIE_SECURE"] = False  # Set to True in production with HTTPS
app.config["SESSION_COOKIE_HTTPONLY"] = True  # Prevent JavaScript access
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"  # CSRF protection

# Demo user credentials. For production, use a proper database with proper auth (e.g., Auth0, AWS Cognito).
# These are intentionally simple for demonstration purposes.
DEMO_USERS = {}
for line in os.environ.get("ADMIN_ACCOUNT", "user:[admin]|||password:[password]").splitlines():
    if "user:" in line:
        # Split by lines and clean each part
        parts = line.split('|||')
        user = parts[0].split('[')[1].split(']')[0]
        password = parts[1].split('[')[1].split(']')[0]

        DEMO_USERS[user] = generate_password_hash(password)

print(f"Demo users loaded: {list(DEMO_USERS.keys())}")

# In-memory chat state keyed by browser session id.
_CHATBOTS = {}
DASHBOARD_PROCESS_DB_PATH = Path(str(DEFAULT_VECTOR_DB_PATH) + "_default.sqlite")

def _load_benchmark_chatbot(chat_id: str | None = None) -> Chatbot:
    try:
        variant = load_best_variant(UNIFIED_HPC_RESULTS_PATH)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Could not load benchmark variant; using local default database. Reason: {exc}")
        return Chatbot(chat_id=chat_id)

    if not variant:
        print("No benchmark variant found; using local default database.")
        return Chatbot(chat_id=chat_id)
    output_path = variant.get("output_path")
    if not output_path:
        print("Benchmark variant does not specify a knowledge base path; using local default database.")
        return Chatbot(chat_id=chat_id)
    return Chatbot(database_path=output_path, chat_id=chat_id)

def _get_chatbot(chat_id: str | None = None) -> Chatbot:
    if not chat_id:
        chat_id = session.get("chat_id")
    if not chat_id:
        chat_id = str(uuid4())
    session["chat_id"] = chat_id

    bot = _CHATBOTS.get(chat_id)
    if bot is None:
        bot = _load_benchmark_chatbot(chat_id=chat_id)
        _CHATBOTS[chat_id] = bot
    return bot

def _get_all_chatbots():
    bots = list(_CHATBOTS.values())
    bots.reverse()  # show most recently created chats first
    return [
        {"name": bot.name or "New chat", "chat_id": bot.chat_id}
        for bot in bots
        if bot.chat_id
    ]


def _is_logged_in() -> bool:
    """Check if the current user is logged in."""
    return "user_email" in session and "user_id" in session


def _get_current_user() -> dict | None:
    """Get the current logged-in user info, or None if not logged in."""
    if _is_logged_in():
        return {
            "user_id": session.get("user_id"),
            "user_email": session.get("user_email"),
        }
    return None


def login_required(f):
    """Decorator to protect routes. Redirects to login if not authenticated."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not _is_logged_in():
            flash("Please log in to access this page.", "info")
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated_function


def load_dashboard_sources() -> list[dict]:
    if not DASHBOARD_SOURCES_PATH.exists():
        return []
    try:
        payload = json.loads(DASHBOARD_SOURCES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(payload, list):
        return [source for source in payload if isinstance(source, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("sources"), list):
        return [source for source in payload["sources"] if isinstance(source, dict)]
    return []


def save_dashboard_sources(sources: list[dict]) -> None:
    DASHBOARD_SOURCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_SOURCES_PATH.write_text(
        json.dumps(sources, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_dashboard_source_by_id(source_id: str) -> tuple[list[dict], dict | None]:
    sources = load_dashboard_sources()
    for source in sources:
        if str(source.get("id", "")) == str(source_id):
            return sources, source
    return sources, None


def _load_web_payload() -> dict:
    if not DEFAULT_WEB_OUTPUT.exists():
        return {}
    try:
        payload = json.loads(DEFAULT_WEB_OUTPUT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _friendly_source_type(source: dict) -> str:
    url = str(source.get("url", "") or "")
    if "blogs.umb.edu" in url:
        return "Blog Post"
    return "Website Page"


def _parse_dashboard_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _is_old_source(value: str | None) -> bool:
    parsed = _parse_dashboard_datetime(value)
    if not parsed:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - parsed).days > REVIEW_STALE_DAYS


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _library_filter_for_source(source: dict) -> str:
    source_type = str(source.get("source_type", "") or "").lower()
    category = str(source.get("category", "") or "").lower()
    filename = str(source.get("filename", "") or "").lower()

    if "uploaded" in source_type:
        return "uploaded-files"
    if "blog" in source_type or "blog" in category:
        return "blog-posts"
    if "report" in category or "pdf" in category or filename.endswith(".pdf"):
        return "reports-pdfs"
    if "faq" in category:
        return "faqs"
    if "website" in source_type:
        return "website-pages"
    return "all"


def _normalize_dashboard_source(source: dict, *, uploaded: bool = False) -> dict:
    filename = str(source.get("filename", "") or "")
    source_url = str(source.get("source_url", "") or "")
    url = str(source.get("url", "") or "")
    link = source_url or url
    source_type = source.get("source_type") or ("Uploaded File" if uploaded else "Website Page")
    category = source.get("category") or ("Uploaded File" if uploaded else source_type)
    status = source.get("status") or "Ready for Chatbot"
    uploaded_at = source.get("uploaded_at") or source.get("modified_time") or ""
    processed_at = source.get("processed_at") or ""
    saved_passages = source.get("saved_passages")

    return {
        "id": source.get("id") or source.get("document_id") or "",
        "title": source.get("title") or filename or "Untitled source",
        "source_type": source_type,
        "category": category,
        "status": status,
        "uploaded_at": uploaded_at,
        "processed_at": processed_at,
        "display_date": processed_at or uploaded_at or "Not available yet",
        "link": link if link.startswith("http") else "",
        "filename": filename,
        "description": source.get("description") or "",
        "ready_for_chatbot": bool(source.get("ready_for_chatbot", status == "Ready for Chatbot")),
        "saved_passages": saved_passages,
        "is_uploaded": uploaded,
        "can_delete_from_chatbot": uploaded and status == "Ready for Chatbot",
        "filter_group": _library_filter_for_source(
            {
                "source_type": source_type,
                "category": category,
                "filename": filename,
            }
        ),
    }


def _review_reasons_for_source(source: dict, *, uploaded: bool = False) -> list[str]:
    reasons = []
    status = str(source.get("status", "") or "")
    if status == "Needs Processing":
        reasons.append("Waiting to be prepared")
    if status == "Processing Failed":
        reasons.append("Could not be prepared yet")
    if uploaded and not (source.get("source_url") or source.get("link")):
        reasons.append("Missing public source link")
    if not source.get("title"):
        reasons.append("Missing source title")
    if uploaded and not source.get("filename"):
        reasons.append("Missing saved file")
    if _is_old_source(source.get("uploaded_at") or source.get("processed_at")):
        reasons.append("May need a freshness review")
    return reasons


def summarize_existing_sources() -> dict:
    payload = _load_web_payload()
    documents = payload.get("documents", [])
    documents = documents if isinstance(documents, list) else []
    summary = payload.get("summary", {})
    summary = summary if isinstance(summary, dict) else {}

    sources = []
    for document in documents:
        if not isinstance(document, dict):
            continue
        normalized = _normalize_dashboard_source(
            {
                "document_id": document.get("document_id", ""),
                "title": document.get("title") or "Untitled source",
                "source_type": _friendly_source_type(document),
                "category": "Institute Resource",
                "status": "Ready for Chatbot",
                "modified_time": document.get("modified_time") or "Not available yet",
                "url": document.get("url", ""),
                "description": "Current source already available in the chatbot collection.",
                "ready_for_chatbot": True,
            }
        )
        sources.append(normalized)

    return {
        "count": len(documents) if documents else None,
        "last_updated": payload.get("generated_at_utc") or "Not available yet",
        "sources": sources,
        "summary": summary,
    }


def build_dashboard_summary() -> dict:
    existing = summarize_existing_sources()
    uploads = load_dashboard_sources()
    existing_count = existing["count"]
    upload_count = len(uploads)
    ready_upload_count = sum(1 for source in uploads if source.get("ready_for_chatbot"))
    needs_processing_count = sum(
        1 for source in uploads if source.get("status") == "Needs Processing"
    )
    processing_failed_count = sum(
        1 for source in uploads if source.get("status") == "Processing Failed"
    )

    total_sources = (
        existing_count + upload_count if existing_count is not None else upload_count or "Not available yet"
    )
    ready_count = (
        existing_count + ready_upload_count if existing_count is not None else ready_upload_count
    )

    last_updated = "Not available yet"
    uploaded_dates = [source.get("uploaded_at", "") for source in uploads if source.get("uploaded_at")]
    if uploaded_dates:
        last_updated = sorted(uploaded_dates)[-1]
    elif existing.get("last_updated"):
        last_updated = existing["last_updated"]

    return {
        "total_sources": total_sources,
        "website_pages": existing_count if existing_count is not None else "Not available yet",
        "uploaded_files": upload_count,
        "ready_for_chatbot": ready_count,
        "needs_processing": needs_processing_count,
        "processing_failed": processing_failed_count,
        "last_updated": last_updated,
    }


def build_content_library() -> list[dict]:
    uploads = load_dashboard_sources()
    existing = summarize_existing_sources()["sources"]
    normalized_uploads = []
    for source in uploads:
        if source.get("status") != "Ready for Chatbot":
            continue
        normalized_uploads.append(_normalize_dashboard_source(source, uploaded=True))
    return normalized_uploads + existing


def build_processing_queue() -> list[dict]:
    queue = []
    for source in load_dashboard_sources():
        if source.get("status") not in PROCESSABLE_SOURCE_STATUSES:
            continue
        normalized = _normalize_dashboard_source(source, uploaded=True)
        normalized["button_label"] = (
            "Try Processing Again"
            if normalized.get("status") == "Processing Failed"
            else "Process for Chatbot"
        )
        queue.append(normalized)
    return queue


def build_review_items() -> list[dict]:
    review_items = []
    for source in load_dashboard_sources():
        normalized = _normalize_dashboard_source(source, uploaded=True)
        reasons = _review_reasons_for_source(source, uploaded=True)
        if reasons:
            normalized["review_reasons"] = reasons
            review_items.append(normalized)
    return review_items


def build_needs_attention(review_items: list[dict]) -> list[dict]:
    priority = {"Processing Failed": 0, "Needs Processing": 1, "Ready for Chatbot": 2}
    return sorted(
        review_items,
        key=lambda item: (
            priority.get(str(item.get("status", "")), 3),
            item.get("display_date") or "",
        ),
    )[:5]


def build_coverage_summary(sources: list[dict]) -> list[dict]:
    groups: dict[str, dict] = {}
    for source in sources:
        label = source.get("category") or source.get("source_type") or "Other"
        group = groups.setdefault(
            label,
            {
                "label": label,
                "source_count": 0,
                "saved_passages": 0,
                "source_types": set(),
                "filter_group": source.get("filter_group", "all"),
            },
        )
        group["source_count"] += 1
        group["saved_passages"] += _safe_int(source.get("saved_passages"))
        group["source_types"].add(source.get("source_type") or "Source")

    coverage = []
    for group in groups.values():
        coverage.append(
            {
                "label": group["label"],
                "source_count": group["source_count"],
                "saved_passages": group["saved_passages"],
                "source_types": ", ".join(sorted(group["source_types"])),
                "filter_group": group["filter_group"],
            }
        )
    return sorted(coverage, key=lambda item: (-item["source_count"], item["label"]))


def build_library_filters() -> list[dict]:
    return [
        {"label": "All", "value": "all"},
        {"label": "Uploaded Files", "value": "uploaded-files"},
        {"label": "Website Pages", "value": "website-pages"},
        {"label": "Blog Posts", "value": "blog-posts"},
        {"label": "Reports / PDFs", "value": "reports-pdfs"},
        {"label": "FAQs", "value": "faqs"},
    ]


def _dashboard_backup_path(db_path: Path | None = None) -> Path:
    target = Path(db_path or DASHBOARD_PROCESS_DB_PATH)
    return target.with_name(f"{target.stem}.backup_before_dashboard_processing{target.suffix}")


def _create_dashboard_db_backup(target_path: Path | None = None) -> dict[str, str]:
    target = Path(target_path or DASHBOARD_PROCESS_DB_PATH)
    if not target.exists():
        raise FileNotFoundError(f"Active database not found: {target}")

    backup_dir = Path("data") / "dashboard_backups" / datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)

    backup_db_path = backup_dir / target.name
    shutil.copy2(target, backup_db_path)

    session["db_path"] = str(target)
    session["current_db_path"] = str(target)
    session["backup_dir"] = str(backup_dir)
    session["backup_db_path"] = str(backup_db_path)
    session["current_backup"] = str(backup_db_path)
    session["current_backup_dir"] = str(backup_dir)

    return {
        "db_path": str(target),
        "backup_dir": str(backup_dir),
        "backup_db_path": str(backup_db_path),
    }


def _unique_upload_filename(filename: str) -> str:
    safe_name = secure_filename(filename)
    if not safe_name:
        return ""
    candidate = DASHBOARD_UPLOAD_DIR / safe_name
    if not candidate.exists():
        return safe_name
    stem = candidate.stem
    suffix = candidate.suffix
    return f"{stem}_{uuid4().hex[:8]}{suffix}"


def _extract_uploaded_text(file_path: Path, filename: str) -> str:
    suffix = file_path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        return json.dumps(payload, indent=2, ensure_ascii=False)

    if suffix == ".csv":
        dataframe = pd.read_csv(file_path)
        return dataframe.to_string(index=False)

    raw_bytes = file_path.read_bytes()
    buffer = io.BytesIO(raw_bytes)

    if suffix == ".pdf":
        reader = PdfReader(buffer)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        document = Document(buffer)
        return "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text and paragraph.text.strip()
        )

    raise ValueError(f"Unsupported file type for processing: {filename}")


def _build_uploaded_document(source: dict) -> dict:
    source_id = str(source.get("id", "")).strip()
    filename = str(source.get("filename", "")).strip()
    if not source_id:
        raise ValueError("Uploaded source is missing its dashboard id.")
    if not filename:
        raise ValueError("Uploaded source is missing its saved filename.")

    file_path = DASHBOARD_UPLOAD_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError("Uploaded file could not be found on disk.")

    raw_text = _extract_uploaded_text(file_path, filename)
    text = normalize_text(raw_text)
    if not text:
        raise ValueError("This file did not contain usable text for the chatbot.")

    return {
        "document_id": f"upload::{source_id}",
        "source_type": "uploaded_file",
        "source_name": "dashboard_upload",
        "source_locator": filename,
        "title": source.get("title") or filename,
        "mime_type": "",
        "url": (source.get("source_url") or "").strip(),
        "modified_time": source.get("uploaded_at"),
        "size_bytes": file_path.stat().st_size,
        "folder_path": "dashboard_uploads",
        "text": text,
        "char_count": len(text),
    }


def _delete_existing_document_rows(db_path: Path, document_id: str) -> None:
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DELETE FROM embeddings WHERE document_id = ?", (document_id,))
        conn.commit()
    finally:
        conn.close()


def process_dashboard_source(source: dict) -> None:
    document = _build_uploaded_document(source)
    payload = build_chunk_payload(
        [document],
        source={
            "type": "dashboard_upload",
            "source_id": source.get("id"),
            "filename": source.get("filename", ""),
        },
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    if not payload.get("chunks"):
        raise ValueError("This file could not be broken into chatbot-ready sections.")

    _create_dashboard_db_backup(DASHBOARD_PROCESS_DB_PATH)

    # use best variant instead of default embedding config for better performance and relevance in the dashboard context
    variant = load_best_variant(UNIFIED_HPC_RESULTS_PATH)
    method = variant.get("embedding_method", str("dummy"))
    dim = variant.get("embedding_dimension", DEFAULT_EMBEDDING_DIM) 

    embedder = get_embedder_with_dimension(dim=dim, embedding_method=method)

    _delete_existing_document_rows(DASHBOARD_PROCESS_DB_PATH, document["document_id"])
    ingest_payload_to_sqlite(
        payload,
        DASHBOARD_PROCESS_DB_PATH,
        embedder=embedder,
        batch_size=DEFAULT_BATCH_SIZE,
    )
    # return len(payload.get("chunks", []))

def getSamplePrompts(count: int = 3) -> list[str]:
    variant = load_best_variant(UNIFIED_HPC_RESULTS_PATH)

    prompts = variant.get("raw_variant", {}).get("details", [])

    # filter anything with low scores or errors to ensure the sample prompts are high quality and relevant to the context, 
    prompts = [prompt for prompt in prompts if prompt.get("error_code", None) is None]
    prompts = [prompt for prompt in prompts if prompt.get("retrieval_hit") == 1]
    prompts = [prompt for prompt in prompts if prompt.get("hint_match") == 1]
    prompts = [prompt for prompt in prompts if prompt.get("retrieval_status") == "ok"]
    prompts = [prompt for prompt in prompts if prompt.get("grounding_verified") == 1]
    prompts = [prompt for prompt in prompts if prompt.get("correct") == 1]
    prompts = [prompt for prompt in prompts if prompt.get("score", 0) >= 4]

    selected_prompts = ["What does Early Education Leaders do?"] # start with a fixed prompt that's relevant to the context, then add random ones after

    while len(selected_prompts) < count:
        prompt = random.choice(prompts)
        question = prompt.get("question", "").strip()
        if question and question not in selected_prompts:
            selected_prompts.append(question)

    return selected_prompts

@app.get("/")
def index():
    if not UNIFIED_HPC_RESULTS_PATH.exists():
        return "Unified HPC results not found.", 404
    return chat()

@app.get("/chat/<chat_id>")
def chat(chat_id: str | None = None):
    bot = _get_chatbot(chat_id)
    print(_resolve_db_from_hpc())
    session["db_path"] = str(_resolve_db_from_hpc())
    session["current_db_path"] = str(_resolve_db_from_hpc())
    return render_template(
        "chat.html",
        bot_name=bot.name,
        chat_id=bot.chat_id,
        chat_history=bot.get_history(),
        onboard_prompt=bot.onboard_prompt,
        is_logged_in=_is_logged_in(),
        current_user=_get_current_user(),
        prompt_suggestions=getSamplePrompts(3),
        chats=_get_all_chatbots()
    )

@app.get("/login")
def login_page():
    """Render the login page. Redirect to home if already logged in."""
    if _is_logged_in():
        return redirect(url_for("index"))
    return render_template("login.html")

def _format_db_name(name: str | None) -> Markup:
    if not name:
        return Markup("No database selected")
    parts = name.replace(".sqlite", "").split("_")

    def _extract(prefix: str, default: str = "Unknown") -> str:
        return next((part[len(prefix):] for part in parts if part.startswith(prefix)), default)

    model = parts[0] if parts and parts[0] else "Unknown model"
    chunk_size = _extract("cs")
    chunk_overlap = _extract("co")
    embedding_dim = _extract("ed")
    batch_size = _extract("bs")

    html = f"""
                <div style="background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 16px; max-width: 350px; font-family: sans-serif; line-height: 1.6;">
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #eee; padding: 4px 0;">
                        <small style="color: #495057;">Model:</small>
                        <span style="font-family: monospace; color: #007bff;">{model}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #eee; padding: 4px 0;">
                        <small style="color: #495057;">Chunk Size:</small>
                        <span style="font-family: monospace; color: #007bff;">{chunk_size}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #eee; padding: 4px 0;">
                        <small style="color: #495057;">Chunk Overlap:</small>
                        <span style="font-family: monospace; color: #007bff;">{chunk_overlap}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #eee; padding: 4px 0;">
                        <small style="color: #495057;">Embedding Dim:</small>
                        <span style="font-family: monospace; color: #007bff;">{embedding_dim}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                        <small style="color: #495057;">Batch Size:</small>
                        <span style="font-family: monospace; color: #007bff;">{batch_size}</span>
                    </div>
                </div>
        """
    return Markup(html)

@app.get("/dashboard")
@login_required
def dashboard():
    sources = build_content_library()
    review_items = build_review_items()
    user = _get_current_user()
    db_path_value = session.get("current_db_path") or session.get("db_path") or ""
    name = Path(str(db_path_value)).name if db_path_value else ""
    db = _format_db_name(name)
    backup_name = Path(str(session.get("current_backup"))) if session.get("current_backup") else None
    backup_dir = Path(str(session.get("current_backup_dir"))) if session.get("current_backup_dir") else None
    return render_template(
        "dashboard.html",
        summary=build_dashboard_summary(),
        processing_queue=build_processing_queue(),
        needs_attention=build_needs_attention(review_items),
        sources=sources,
        coverage_summary=build_coverage_summary(sources),
        review_items=review_items,
        library_filters=build_library_filters(),
        categories=DASHBOARD_CATEGORIES,
        current_user=user,
        is_logged_in = _is_logged_in(),
        current_db=db,
        current_backup_path=backup_name,
        current_backup_dir=backup_dir,
        chats=_get_all_chatbots(),
    )

@app.post("/dashboard/upload")
@login_required
def dashboard_upload():
    title = (request.form.get("title") or "").strip()
    category = (request.form.get("category") or "Other").strip() or "Other"
    description = (request.form.get("description") or "").strip()
    source_url = (request.form.get("source_url") or "").strip()
    upload = request.files.get("file")
    upload_filename = (upload.filename if upload else "") or ""

    if not title:
        flash("Please add a source title before submitting.", "error")
        return redirect(url_for("dashboard"))
    has_file = bool(upload and upload_filename)
    if has_file and not allowed_file(upload_filename):
        flash("This file type is not supported yet.", "error")
        return redirect(url_for("dashboard"))
    if not has_file and not source_url:
        flash("Please provide a source URL or upload a file.", "error")
        return redirect(url_for("dashboard"))

    try:
        saved_filename = ""
        if has_file:
            DASHBOARD_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            saved_filename = _unique_upload_filename(upload_filename)
            if not saved_filename:
                flash("This file type is not supported yet.", "error")
                return redirect(url_for("dashboard"))
            assert upload is not None
            upload.save(str(DASHBOARD_UPLOAD_DIR / saved_filename))
            upload.close()

        source_type = "Uploaded File" if has_file else _friendly_source_type({"url": source_url})

        sources = load_dashboard_sources()
        sources.insert(
            0,
            {
                "id": str(uuid4()),
                "title": title,
                "source_type": source_type,
                "category": category or source_type,
                "filename": saved_filename,
                "source_url": source_url,
                "description": description,
                "status": "Needs Processing",
                "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "ready_for_chatbot": False,
            },
        )
        save_dashboard_sources(sources)
    except OSError as exc:
        print(f"Dashboard upload save failed: {exc}")
        flash("Something went wrong while saving the file. Please try again.", "error")
        return redirect(url_for("dashboard"))

    flash(
        "Your content was uploaded successfully. This source is saved and waiting to be prepared for the chatbot.",
        "success",
    )
    return redirect(url_for("dashboard"))

@app.post("/dashboard/process")
@app.post("/dashboard/process/<source_id>")
@login_required
def dashboard_process(source_id: str | None = None):
    sources = load_dashboard_sources()
    processable_sources = [
        item
        for item in sources
        if str(item.get("status", "")).strip() in PROCESSABLE_SOURCE_STATUSES
    ]
    if not processable_sources:
        flash("There are no uploaded sources waiting to be processed.", "error")
        return redirect(url_for("dashboard"))

    try:
        saved_passages, db_created, db_path = process_dashboard_sources()
        processed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for item in processable_sources:
            item["status"] = "Ready for Chatbot"
            item["ready_for_chatbot"] = True
            item["processed_at"] = processed_at
            item["saved_passages"] = saved_passages
            item.pop("process_error", None)
        save_dashboard_sources(sources)
        if saved_passages > 0:
            flash(
                f"Processed {len(processable_sources)} source(s). Added {saved_passages} passages to chatbot database at {db_path}.",
                "success",
            )
        else:
            flash(
                f"All sources are already in the chatbot database at {db_path}.",
                "info",
            )
    except Exception as exc:
        error_message = str(exc).strip()[:300] or "Processing failed."
        for item in processable_sources:
            item["status"] = "Processing Failed"
            item["ready_for_chatbot"] = False
            item["process_error"] = error_message
        save_dashboard_sources(sources)
        flash(
            "These sources could not be prepared for the chatbot. Please check the files and try again.",
            "error",
        )

    return redirect(url_for("dashboard"))

@app.post("/dashboard/delete/<source_id>")
@login_required
def dashboard_delete(source_id: str):
    sources, source = _load_dashboard_source_by_id(source_id)
    if source is None:
        flash("We could not find that uploaded source.", "error")
        return redirect(url_for("dashboard"))

    status = str(source.get("status", "")).strip()
    is_ready_source = status == "Ready for Chatbot"
    if status not in PROCESSABLE_SOURCE_STATUSES and not is_ready_source:
        flash("This source cannot be deleted from the dashboard right now.", "error")
        return redirect(url_for("dashboard"))

    if is_ready_source:
        try:
            _delete_existing_document_rows(
                _resolve_dashboard_db_path(),
                f"upload::{source_id}",
            )
        except Exception as exc:
            print(f"Dashboard ready upload delete failed: {exc}")
            flash("This source could not be removed from the chatbot right now. Please try again.", "error")
            return redirect(url_for("dashboard"))

    filename = str(source.get("filename", "") or "").strip()
    if filename:
        upload_path = DASHBOARD_UPLOAD_DIR / filename
        try:
            upload_path.unlink(missing_ok=True)
        except OSError as exc:
            print(f"Dashboard upload delete failed: {exc}")
            flash("This file could not be removed right now. Please try again.", "error")
            return redirect(url_for("dashboard"))

    remaining_sources = [
        candidate
        for candidate in sources
        if str(candidate.get("id", "")) != str(source_id)
    ]
    save_dashboard_sources(remaining_sources)
    if is_ready_source:
        flash("This uploaded source was removed from the chatbot.", "success")
    else:
        flash("This uploaded file was removed from the dashboard.", "success")
    return redirect(url_for("dashboard"))

@app.post("/dashboard/reset")
@login_required
def dashboard_reset():
    # Check if a database exists before attempting to backup
    db_path = _resolve_dashboard_db_path()
    if not db_path.exists():
        flash("No database to reset. Please process sources first.", "info")
        return redirect(url_for("dashboard"))
    
    try:
        backup_state = _create_dashboard_db_backup()
        session["current_backup"] = backup_state.get("backup_db_path")
        session["current_backup_dir"] = backup_state.get("backup_dir")
        _reset_database()
        flash(
            "The chatbot database has been reset. A backup of the previous database was saved and a fresh database was created.",
            "success",
        )
    except Exception as exc:
        print(f"Dashboard reset failed: {exc}")
        flash("The chatbot's knowledge base could not be reset right now. Please try again.", "error")
    return redirect(url_for("dashboard"))


@app.post("/dashboard/reset_crawl")
@login_required
def dashboard_reset_crawl():
    try:
        _reset_web_payload()
        _reset_dashboard_source_states()
        flash(
            "The source snapshot has been cleared. Reprocess sources to rebuild web_data.json.",
            "success",
        )
    except Exception as exc:
        print(f"Dashboard crawl reset failed: {exc}")
        flash("The crawl snapshot could not be cleared right now. Please try again.", "error")
    return redirect(url_for("dashboard"))


@app.post("/dashboard/reset_sources")
@login_required
def dashboard_reset_sources():
    return dashboard_reset_crawl()

@app.post("/dashboard/restore_backup")
@login_required
def dashboard_restore_backup():
    try:
        backup_file = session.get("current_backup", None)
        
        if not backup_file:
            flash("No backup file could be found to restore.", "error")
            return redirect(url_for("dashboard"))
        
        backup_path = Path(str(backup_file))
        if not backup_path.exists():
            flash("No backup file could be found to restore.", "error")
            return redirect(url_for("dashboard"))

        restore_state = _restore_dashboard_backup()
        session["db_path"] = restore_state["db_path"]
        session["current_backup"] = restore_state["backup_db_path"]
        flash(
            "A backup has been restored to the chatbot's knowledge base.",
            "success",
        )
    except Exception as exc:
        print(f"Dashboard restore backup failed: {exc}")
        flash("The backup could not be restored right now. Please try again.", "error")
    return redirect(url_for("dashboard"))

@app.post("/login")
def login():
    """Handle login form submission. Validate credentials and set session."""
    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    # Validate input
    if not email or not password:
        flash("Email and password are required.", "error")
        return redirect(url_for("login_page"))

    # Check credentials against demo users
    is_valid_password = check_password_hash(DEMO_USERS.get(email, ""), password) # ALWAYS hash to ensure secure timing consistency, even for invalid emails
    if email not in DEMO_USERS or not is_valid_password:
        flash("Invalid email or password. Please try again.", "error")
        return redirect(url_for("login_page"))

    # Create session
    session["user_id"] = email
    session["user_email"] = email
    session.permanent = request.form.get("remember") == "on"

    flash(f"Welcome back, {email.split('@')[0]}!", "success")
    return redirect(url_for("index"))

@app.get("/logout")
def logout():
    """Log out the current user and clear the session."""
    session.clear()
    flash(f"You have been logged out. Goodbye!", "info")
    return redirect(url_for("login_page"))

@app.post("/chat")
def chatAPI():
    payload = request.get_json(silent=True) or {}
    user_input = (payload.get("message") or "").strip()

    if not user_input:
        return jsonify({"error": "Message cannot be empty."}), 400

    bot = _get_chatbot(session.get("chat_id"))
    status_events = []

    def _status_callback(status: str) -> None:
        status_events.append(status)

    response_payload = bot.create_response(user_input, status_callback=_status_callback)
    if isinstance(response_payload, dict):
        response_payload["chat_name"] = bot.name
    return jsonify(response_payload)


@app.post("/new_chat")
def new_chat():
    chat_id = str(uuid4())
    session["chat_id"] = chat_id
    _CHATBOTS[chat_id] = _load_benchmark_chatbot(chat_id=chat_id)
    return jsonify({"ok": True, "chat_id": chat_id})


def _clear_chat(chat_id: str) -> dict:
    if chat_id and chat_id in _CHATBOTS:
        del _CHATBOTS[chat_id]
    session.pop("chat_id", None)
    return {"ok": True}


@app.post("/reset/<chat_id>")
def reset(chat_id: str):
    return jsonify(_clear_chat(chat_id))


@app.post("/delete/<chat_id>")
def delete_chat(chat_id: str):
    return jsonify(_clear_chat(chat_id))


@app.get("/health")
def health():
    return jsonify({"status": "ok", "chatbot_count": len(_CHATBOTS)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
