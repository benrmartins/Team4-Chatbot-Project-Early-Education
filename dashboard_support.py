import hashlib
import io
import json
import random
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List
from uuid import uuid4

from flask import session
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from docx import Document
import pandas as pd

from ingestion_pipeline.scripts.build_chunk_payload import build_chunk_payload, normalize_text
from ingestion_pipeline.scripts.pipeline_runner import run_crawlers
from ingestion_pipeline.services.vector_store import (
    get_embedder_with_dimension,
    ingest_payload_to_sqlite,
    init_db,
)
from ingestion_pipeline.DataProcessor import DataProcessor
from HPC.load_best_variant import load_best_variant
from project_config import (
    PROJECT_ROOT,
    DATA_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_VECTOR_DB_PATH,
    DEFAULT_WEB_OUTPUT,
    DEFAULT_WEBSITE_SEED_URLS,
    DEFAULT_DRIVE_FOLDER_URLS,
    UNIFIED_HPC_RESULTS_PATH,
)

DASHBOARD_UPLOAD_DIR = DATA_DIR / "dashboard_uploads"
DASHBOARD_SOURCES_PATH = DATA_DIR / "dashboard_sources.json"
DASHBOARD_CACHE_DIR = DATA_DIR / "dashboard_cache"  # User cache directory with timestamped files
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


def _compute_sources_fingerprint(sources: list[dict]) -> str:
    """Compute a hash fingerprint of source URLs to detect changes."""
    urls = sorted([
        source.get("url") or source.get("link") or ""
        for source in sources
        if isinstance(source, dict)
    ])
    url_string = "|".join(urls)
    return hashlib.md5(url_string.encode()).hexdigest()


def _find_latest_user_cache() -> Path | None:
    """Find the most recently created user cache file."""
    if not DASHBOARD_CACHE_DIR.exists():
        return None
    cache_files = sorted(
        list(DASHBOARD_CACHE_DIR.glob("user_cache_*.json"))
        + list(DASHBOARD_CACHE_DIR.glob("web_crawl_*.json")),
        reverse=True,
    )
    return cache_files[0] if cache_files else None


def _load_latest_user_cache() -> dict:
    """Load the most recent user cache file if it exists."""
    latest_cache = _find_latest_user_cache()
    if not latest_cache:
        return {}
    try:
        payload = json.loads(latest_cache.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _generate_user_cache_filename() -> str:
    """Generate a timestamped user cache filename."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"user_cache_{timestamp}.json"


def allowed_file(filename: str | None) -> bool:
    if not filename:
        return True
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


def _resolve_db_from_hpc(hpc_results_path: Path = UNIFIED_HPC_RESULTS_PATH) -> Path | None:
    try:
        variant = load_best_variant(hpc_results_path)
        output_path = variant.get("output_path")
        if output_path:
            resolved = Path(output_path)
            return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved
    except (FileNotFoundError, ValueError) as exc:
        print(f"Could not load HPC benchmark variant. Reason: {exc}")
    return None


def _resolve_dashboard_db_path() -> Path:
    stored_path = session.get("current_db_path") or session.get("db_path")
    if stored_path:
        resolved = Path(str(stored_path))
        return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved

    hpc_path = _resolve_db_from_hpc()
    if hpc_path is not None:
        return hpc_path

    return DEFAULT_VECTOR_DB_PATH


def _load_dashboard_source_by_id(source_id: str) -> tuple[list[dict], dict | None]:
    sources = load_dashboard_sources()
    for source in sources:
        if str(source.get("id", "")) == str(source_id):
            return sources, source
    return sources, None


def _collect_dashboard_seed_links() -> tuple[list[str], list[str]]:
    web_seeds: list[str] = []
    drive_links: list[str] = []
    for source in load_dashboard_sources():
        source_type = str(source.get("source_type", "") or "").lower()
        if "uploaded" in source_type:
            continue
        candidates = [
            str(source.get("source_url", "") or "").strip(),
            str(source.get("url", "") or "").strip(),
            str(source.get("link", "") or "").strip(),
        ]
        for link in candidates:
            if not link or not link.startswith("http"):
                continue
            if "drive.google.com/drive/folders/" in link:
                if link not in drive_links:
                    drive_links.append(link)
            else:
                if link not in web_seeds:
                    web_seeds.append(link)
    return web_seeds, drive_links



def _reset_web_payload() -> None:
    if DEFAULT_WEB_OUTPUT.exists():
        DEFAULT_WEB_OUTPUT.unlink()


def _reset_database() -> None:
    path = _resolve_dashboard_db_path()
    if path.exists():
        path.unlink()
    session["current_db_path"] = None
    session["db_path"] = None


def _reset_dashboard_source_states() -> None:
    sources = load_dashboard_sources()
    changed = False
    for source in sources:
        if source.get("status") != "Needs Processing" or source.get("ready_for_chatbot") is not False:
            changed = True
        source["status"] = "Needs Processing"
        source["ready_for_chatbot"] = False
        source.pop("processed_at", None)
        source.pop("saved_passages", None)
        source.pop("process_error", None)
    if changed:
        save_dashboard_sources(sources)


def _load_web_payload(force: bool = False, use_defaults: bool = False) -> dict:
    # For defaults: use cache if it exists (unless force=True); respects DEFAULT_WEB_OUTPUT
    # For regular (user uploads): load latest user cache (no web crawl)
    if use_defaults:
        if force or not DEFAULT_WEB_OUTPUT.exists():
            try:
                if force and DEFAULT_WEB_OUTPUT.exists():
                    DEFAULT_WEB_OUTPUT.unlink()
                documents, summary = run_crawlers(
                    web_seeds=list(DEFAULT_WEBSITE_SEED_URLS),
                    drive_links=list(DEFAULT_DRIVE_FOLDER_URLS),
                    web_output_path=str(DEFAULT_WEB_OUTPUT),
                )
                return {"documents": documents, "summary": summary}
            except Exception as exc:
                print(f"Web crawl failed while building payload: {exc}")
                return {}
        # Use existing default cache
        try:
            payload = json.loads(DEFAULT_WEB_OUTPUT.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}
    else:
        # For user uploads: load latest user cache (no crawl, just load)
        return _load_latest_user_cache()


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
    # Dashboard reads the cached crawl output; explicit re-crawls happen only during processing.
    payload = _load_web_payload(force=False, use_defaults=False)
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
            priority.get(str(item.get("status") or ""), 3),
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


def _create_dashboard_db_backup(target_path: Path | None = None) -> dict[str, str]:
    db_path = Path(target_path or _resolve_dashboard_db_path())
    if not db_path.exists():
        raise FileNotFoundError(f"Active database not found: {db_path}")

    backup_dir = DATA_DIR / "dashboard_backups" / datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)

    backup_db_path = backup_dir / db_path.name
    shutil.copy2(db_path, backup_db_path)

    session["db_path"] = str(db_path)
    session["current_db_path"] = str(db_path)
    session["backup_dir"] = str(backup_dir)
    session["backup_db_path"] = str(backup_db_path)
    session["current_backup"] = str(backup_db_path)
    session["current_backup_dir"] = str(backup_dir)

    return {
        "db_path": str(db_path),
        "backup_dir": str(backup_dir),
        "backup_db_path": str(backup_db_path),
    }


def _restore_dashboard_backup() -> dict[str, str]:
    backup_db_path = Path(session.get("current_backup") or "")
    if not backup_db_path.exists():
        raise FileNotFoundError("No backup file could be found to restore.")

    db_path = Path(str(session.get("current_db_path") or session.get("db_path") or _resolve_dashboard_db_path()))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup_db_path, db_path)

    session["db_path"] = str(db_path)
    session["current_db_path"] = str(db_path)

    return {
        "db_path": str(db_path),
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


def _build_url_document(source: dict, url: str, crawled_text: str) -> dict:
    """Build a document from a crawled URL source."""
    source_id = str(source.get("id", "")).strip()
    if not source_id:
        raise ValueError("URL source is missing its dashboard id.")
    
    text = normalize_text(crawled_text)
    if not text:
        raise ValueError(f"URL {url} did not contain usable text for the chatbot.")
    
    return {
        "document_id": f"url::{source_id}",
        "source_type": "web_url",
        "source_name": "dashboard_url",
        "source_locator": url,
        "title": source.get("title") or url,
        "mime_type": "text/html",
        "url": url,
        "modified_time": source.get("uploaded_at"),
        "size_bytes": len(crawled_text.encode()),
        "folder_path": "dashboard_urls",
        "text": text,
        "char_count": len(text),
    }


def _dashboard_uploaded_documents(sources: list[dict]) -> list[dict]:
    documents: list[dict] = []
    
    # Process uploaded files
    for source in sources:
        filename = str(source.get("filename", "") or "").strip()
        if filename:
            file_path = DASHBOARD_UPLOAD_DIR / filename
            if not file_path.exists():
                continue
            documents.append(_build_uploaded_document(source))
    
    # Collect seeds from URL-only sources and crawl them
    web_seeds = []
    drive_links = []
    for source in sources:
        filename = str(source.get("filename", "") or "").strip()
        source_url = str(source.get("source_url", "") or "").strip()
        
        # Skip if it has a file (already processed above)
        if filename:
            continue
        
        # Collect URL seeds
        if source_url and source_url.startswith("http"):
            if "drive.google.com/drive/folders/" in source_url:
                if source_url not in drive_links:
                    drive_links.append(source_url)
            else:
                if source_url not in web_seeds:
                    web_seeds.append(source_url)
    
    # If there are URL sources, crawl them
    if web_seeds or drive_links:
        try:
            DASHBOARD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            output_path = str(DASHBOARD_CACHE_DIR / _generate_user_cache_filename())
            crawled_documents, crawled_summary = run_crawlers(
                web_seeds=web_seeds,
                drive_links=drive_links,
                web_output_path=output_path,
            )
            if isinstance(crawled_documents, list):
                documents.extend(crawled_documents)
        except Exception as exc:
            print(f"Failed to crawl user URL sources: {exc}")
    
    return documents


def _get_existing_document_ids(db_path: Path) -> set[str]:
    """Get all document IDs already ingested in the database."""
    if not db_path.exists():
        return set()

    try:
        init_db(db_path)
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.execute("SELECT DISTINCT document_id FROM embeddings")
            return {row[0] for row in cursor.fetchall()}
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return set()


def process_dashboard_sources(*, use_defaults: bool = False) -> tuple[int, bool, Path]:
    # Build from the best benchmark variant settings when available, otherwise use defaults.
    if use_defaults:
        variant = {}
        db_path = DEFAULT_VECTOR_DB_PATH
    else:
        try:
            variant = load_best_variant(UNIFIED_HPC_RESULTS_PATH)
        except (FileNotFoundError, ValueError):
            variant = {}

        output_path = str((variant or {}).get("output_path", "")).strip()
        db_path = Path(output_path) if output_path else _resolve_dashboard_db_path()
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    # Get existing document IDs to avoid re-processing
    existing_doc_ids = _get_existing_document_ids(db_path)

    # Extract variant parameters
    method = str((variant or {}).get("embedding_method", "dummy"))
    dim = int((variant or {}).get("embedding_dim", DEFAULT_EMBEDDING_DIM))
    batch_size = int((variant or {}).get("batch_size", DEFAULT_BATCH_SIZE))
    chunk_size = int((variant or {}).get("chunk_size", DEFAULT_CHUNK_SIZE))
    chunk_overlap = int((variant or {}).get("chunk_overlap", DEFAULT_CHUNK_OVERLAP))

    if use_defaults:
        # Use DataProcessor for defaults since no incremental logic needed
        processor = DataProcessor(
            name="dashboard_defaults",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_dim=dim,
            batch_size=batch_size,
            output_path=str(db_path),
            embedding_method=method,
        )
        
        # Crawl with default seeds
        processor.crawl(
            output_path=str(DEFAULT_WEB_OUTPUT),
            web_seeds=list(DEFAULT_WEBSITE_SEED_URLS),
            drive_links=list(DEFAULT_DRIVE_FOLDER_URLS),
        )
        
        # Get all documents
        all_documents = processor.web_data or []
        
        # Filter to only NEW documents for incremental ingestion
        new_documents = [
            doc for doc in all_documents
            if doc.get("document_id") not in existing_doc_ids
        ]
        
        if not new_documents:
            session["db_path"] = str(db_path)
            session["current_db_path"] = str(db_path)
            return 0, False, db_path
        
        # Use DataProcessor to chunk and embed only the new documents
        processor.web_data = new_documents
        processor.source_summary = processor.source_summary or {}
        processor.chunk()
        if not processor.chunk_payload or not processor.chunk_payload.get("chunks"):
            raise ValueError("New ingestion data could not be converted into chatbot-ready chunks.")
        processor.embed()

        # Cache the payload
        payload = {
            "documents": all_documents,
            "summary": processor.source_summary or {},
        }
        DEFAULT_WEB_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_WEB_OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        # User uploads with incremental logic
        dashboard_sources = load_dashboard_sources()
        uploaded_documents = _dashboard_uploaded_documents(dashboard_sources)
        all_documents = uploaded_documents

        if not all_documents:
            raise ValueError(
                "No ingestion sources are available in project defaults or dashboard uploads."
            )

        # Filter to only NEW documents (not already in DB)
        new_documents = [
            doc for doc in all_documents
            if doc.get("document_id") not in existing_doc_ids
        ]

        if not new_documents:
            # No new documents to process, but return success with existing DB
            session["db_path"] = str(db_path)
            session["current_db_path"] = str(db_path)
            return 0, False, db_path

        # Check if sources differ from latest cache before saving
        current_fingerprint = _compute_sources_fingerprint(dashboard_sources)
        latest_cache = _load_latest_user_cache()
        latest_fingerprint = latest_cache.get("_source_fingerprint", "")
        
        # Only write new user cache if sources have changed
        if current_fingerprint != latest_fingerprint:
            payload = {
                "documents": all_documents,
                "summary": latest_cache.get("summary") or {},
                "_source_fingerprint": current_fingerprint,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            DASHBOARD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_filename = _generate_user_cache_filename()
            cache_path = DASHBOARD_CACHE_DIR / cache_filename
            cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        # Use DataProcessor to chunk and embed only the new documents
        processor = DataProcessor(
            name="dashboard_user",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_dim=dim,
            batch_size=batch_size,
            output_path=str(db_path),
            embedding_method=method,
        )
        # Assign prepared new documents and an empty summary (uploads have no website summary)
        processor.web_data = new_documents
        processor.source_summary = {}
        processor.chunk()
        if not processor.chunk_payload or not processor.chunk_payload.get("chunks"):
            raise ValueError("New ingestion data could not be converted into chatbot-ready chunks.")

        processor.embed()

    session["db_path"] = str(db_path)
    session["current_db_path"] = str(db_path)

    return len(processor.chunk_payload.get("chunks", [])), False, db_path

