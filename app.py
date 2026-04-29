import io
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
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
    read_db_embedding_config,
)

from scripts.run_retrieval_benchmark import _load_best_variant
from project_config import (
    DATA_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_VECTOR_DB_PATH,
    DEFAULT_WEB_OUTPUT,
    UNIFIED_HPC_RESULTS_PATH,
)

app = Flask(__name__)
app.secret_key = "change-me-in-production"

# In-memory chat state keyed by browser session id.
_CHATBOTS = {}

DASHBOARD_UPLOAD_DIR = DATA_DIR / "dashboard_uploads"
DASHBOARD_SOURCES_PATH = DATA_DIR / "dashboard_sources.json"
DASHBOARD_PROCESS_DB_PATH = Path(str(DEFAULT_VECTOR_DB_PATH) + "_default.sqlite")
ALLOWED_UPLOAD_EXTENSIONS = {"txt", "md", "pdf", "docx", "csv", "json"}
DASHBOARD_CATEGORIES = [
    "Institute Page",
    "Blog Post",
    "Program Information",
    "Report or PDF",
    "Contact Information",
    "FAQ",
    "Other",
]
PROCESSABLE_SOURCE_STATUSES = {"Needs Processing", "Processing Failed"}
REVIEW_STALE_DAYS = 365

def _load_benchmark_chatbot() -> Chatbot:
    try:
        variant = _load_best_variant(UNIFIED_HPC_RESULTS_PATH)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Could not load benchmark variant; using local default database. Reason: {exc}")
        return Chatbot()

    if not variant:
        print("No benchmark variant found; using local default database.")
        return Chatbot()
    output_path = variant.get("output_path")
    if not output_path:
        print("Benchmark variant does not specify a knowledge base path; using local default database.")
        return Chatbot()
    return Chatbot(database_path=output_path)

def _get_chatbot() -> Chatbot:
    chat_id = session.get("chat_id")
    if not chat_id:
        chat_id = str(uuid4())
        session["chat_id"] = chat_id

    bot = _CHATBOTS.get(chat_id)
    if bot is None:
        bot = _load_benchmark_chatbot()
        _CHATBOTS[chat_id] = bot
    return bot


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_UPLOAD_EXTENSIONS


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
            priority.get(item.get("status"), 3),
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


def _create_dashboard_db_backup(db_path: Path | None = None) -> Path:
    target = Path(db_path or DASHBOARD_PROCESS_DB_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        init_db(target)
    backup_path = _dashboard_backup_path(target)
    shutil.copy2(target, backup_path)
    return backup_path


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

    method, dim = read_db_embedding_config(
        DASHBOARD_PROCESS_DB_PATH,
        default_method="dummy",
        default_dim=DEFAULT_EMBEDDING_DIM,
    )
    embedder = get_embedder_with_dimension(dim=dim, embedding_method=method)

    _delete_existing_document_rows(DASHBOARD_PROCESS_DB_PATH, document["document_id"])
    ingest_payload_to_sqlite(
        payload,
        DASHBOARD_PROCESS_DB_PATH,
        embedder=embedder,
        batch_size=DEFAULT_BATCH_SIZE,
    )
    return len(payload.get("chunks", []))


@app.get("/")
def index():
    bot = _get_chatbot()
    return render_template("index.html", bot_name=bot.name, onboard_prompt=bot.onboard_prompt)


@app.get("/dashboard")
def dashboard():
    sources = build_content_library()
    review_items = build_review_items()
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
    )


@app.post("/dashboard/upload")
def dashboard_upload():
    title = (request.form.get("title") or "").strip()
    category = (request.form.get("category") or "Other").strip() or "Other"
    description = (request.form.get("description") or "").strip()
    source_url = (request.form.get("source_url") or "").strip()
    upload = request.files.get("file")

    if not title:
        flash("Please add a source title before submitting.", "error")
        return redirect(url_for("dashboard", _anchor="upload"))
    if upload is None or not upload.filename:
        flash("Please choose a file before submitting.", "error")
        return redirect(url_for("dashboard", _anchor="upload"))
    if not allowed_file(upload.filename):
        flash("This file type is not supported yet.", "error")
        return redirect(url_for("dashboard", _anchor="upload"))

    try:
        DASHBOARD_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        saved_filename = _unique_upload_filename(upload.filename)
        if not saved_filename:
            flash("This file type is not supported yet.", "error")
            return redirect(url_for("dashboard", _anchor="upload"))
        upload.save(str(DASHBOARD_UPLOAD_DIR / saved_filename))
        upload.close()

        sources = load_dashboard_sources()
        sources.insert(
            0,
            {
                "id": str(uuid4()),
                "title": title,
                "source_type": "Uploaded File",
                "category": category,
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
        return redirect(url_for("dashboard", _anchor="upload"))

    flash(
        "Your content was uploaded successfully. This source is saved and waiting to be prepared for the chatbot.",
        "success",
    )
    return redirect(url_for("dashboard", _anchor="content-library"))


@app.post("/dashboard/process/<source_id>")
def dashboard_process(source_id: str):
    sources, source = _load_dashboard_source_by_id(source_id)
    if source is None:
        flash("We could not find that uploaded source.", "error")
        return redirect(url_for("dashboard", _anchor="process-queue"))

    current_status = str(source.get("status", "")).strip()
    if current_status == "Ready for Chatbot":
        flash("This source is already ready for chatbot answers.", "error")
        return redirect(url_for("dashboard", _anchor="process-queue"))
    if current_status not in PROCESSABLE_SOURCE_STATUSES:
        flash("This source cannot be processed right now.", "error")
        return redirect(url_for("dashboard", _anchor="process-queue"))

    try:
        saved_passages = process_dashboard_source(source)
        source["status"] = "Ready for Chatbot"
        source["ready_for_chatbot"] = True
        source["processed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        source["saved_passages"] = saved_passages
        source.pop("process_error", None)
        save_dashboard_sources(sources)
        flash("This source is now ready for chatbot answers.", "success")
    except Exception as exc:
        source["status"] = "Processing Failed"
        source["ready_for_chatbot"] = False
        source["process_error"] = str(exc).strip()[:300] or "Processing failed."
        save_dashboard_sources(sources)
        flash(
            "This file could not be prepared for the chatbot. Please check the file and try again.",
            "error",
        )

    return redirect(url_for("dashboard", _anchor="process-queue"))


@app.post("/dashboard/delete/<source_id>")
def dashboard_delete(source_id: str):
    sources, source = _load_dashboard_source_by_id(source_id)
    if source is None:
        flash("We could not find that uploaded source.", "error")
        return redirect(url_for("dashboard", _anchor="process-queue"))

    if str(source.get("status", "")).strip() not in PROCESSABLE_SOURCE_STATUSES:
        flash("Ready sources cannot be deleted from this dashboard queue.", "error")
        return redirect(url_for("dashboard", _anchor="process-queue"))

    filename = str(source.get("filename", "") or "").strip()
    if filename:
        upload_path = DASHBOARD_UPLOAD_DIR / filename
        try:
            upload_path.unlink(missing_ok=True)
        except OSError as exc:
            print(f"Dashboard upload delete failed: {exc}")
            flash("This file could not be removed right now. Please try again.", "error")
            return redirect(url_for("dashboard", _anchor="process-queue"))

    remaining_sources = [
        candidate
        for candidate in sources
        if str(candidate.get("id", "")) != str(source_id)
    ]
    save_dashboard_sources(remaining_sources)
    flash("This uploaded file was removed from the dashboard.", "success")
    return redirect(url_for("dashboard", _anchor="process-queue"))


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    user_input = (payload.get("message") or "").strip()

    if not user_input:
        return jsonify({"error": "Message cannot be empty."}), 400

    bot = _get_chatbot()
    status_events = []

    def _status_callback(status: str) -> None:
        status_events.append(status)

    response_payload = bot.create_response(user_input, status_callback=_status_callback)
    return jsonify({
        "reply": response_payload["reply"],
        "citations": response_payload["citations"],
        "evidence": response_payload["evidence"],
        "retrieval": response_payload["retrieval"],
        "status_events": status_events,
    })


@app.post("/reset")
def reset():
    chat_id = session.get("chat_id")
    if chat_id and chat_id in _CHATBOTS:
        del _CHATBOTS[chat_id]
    session.pop("chat_id", None)
    return jsonify({"ok": True})


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
