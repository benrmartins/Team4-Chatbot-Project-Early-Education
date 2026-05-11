import os
import random
from typing import Any, List
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from uuid import uuid4

from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from markupsafe import Markup
from werkzeug.security import check_password_hash, generate_password_hash

from chatbot import Chatbot
from dashboard_support import (
    DASHBOARD_CATEGORIES,
    DASHBOARD_UPLOAD_DIR,
    PROCESSABLE_SOURCE_STATUSES,
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
    _save_url_source_record,
    _save_uploaded_file,
    _unique_upload_filename,
)

from HPC.load_best_variant import load_best_variant
from project_config import UNIFIED_HPC_RESULTS_PATH

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

def _load_benchmark_chatbot(chat_id: str | None = None) -> Chatbot:
    path = session.get("db_path") or session.get("current_db_path")
    if path and Path(path).exists():
        print("Info: Session already has a database path set. Benchmark chatbot will use the existing path instead of loading from HPC variant.")
        return Chatbot(database_path=path, chat_id=chat_id)
    
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
        chat_id = str(uuid4())
        session["chat_id"] = chat_id

    bot = _CHATBOTS.get(chat_id)
    if bot is None:
        bot = _load_benchmark_chatbot(chat_id=chat_id)
        _CHATBOTS[chat_id] = bot
    return bot

def _get_all_chatbots():
    bots = _CHATBOTS.values()
    named_bots = [bot for bot in bots if bot.name]
    return [
        {"name": bot.name, "chat_id": bot.chat_id}
        for bot in named_bots
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
    source_id = str(uuid4())

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
            saved_filename = _unique_upload_filename(upload_filename)
            if not saved_filename:
                flash("This file type is not supported yet.", "error")
                return redirect(url_for("dashboard"))
            assert upload is not None
            _save_uploaded_file(upload, saved_filename)
            upload.close()
        else:
            saved_record_filename = _save_url_source_record(
                source_id=source_id,
                title=title,
                source_url=source_url,
                description=description,
                category=category or "Other",
            )

        source_type = "Uploaded File" if has_file else _friendly_source_type({"url": source_url})

        sources = load_dashboard_sources()
        new_source = {
            "id": source_id,
            "title": title,
            "source_type": source_type,
            "category": category or source_type,
            "filename": saved_filename,
            "saved_record_filename": saved_record_filename if not has_file else "",
            "source_url": source_url,
            "description": description,
            "status": "Needs Processing",
            "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ready_for_chatbot": False,
        }
        sources.insert(0, new_source)
        save_dashboard_sources(sources)
    except (OSError, ValueError) as exc:
        print(f"Dashboard upload save failed: {exc}")
        flash("Something went wrong while saving the file. Please try again.", "error")
        return redirect(url_for("dashboard"))

    if has_file:
        flash(
            "Your file was uploaded successfully. This source is saved and waiting to be prepared for the chatbot.",
            "success",
        )
    else:
        flash(
            "Your URL source was saved successfully. This source is waiting to be prepared for the chatbot.",
            "success",
        )
    return redirect(url_for("dashboard"))

@app.post("/dashboard/process")
@login_required
def dashboard_process():
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


@app.post("/dashboard/process_defaults")
@login_required
def dashboard_process_defaults():
    try:
        saved_passages, _, db_path = process_dashboard_sources(use_defaults=True)
        if saved_passages > 0:
            flash(
                f"Built a database from project defaults and added {saved_passages} passages at {db_path}.",
                "success",
            )
        else:
            flash(
                f"The project defaults database was refreshed at {db_path}.",
                "info",
            )
    except Exception as exc:
        error_message = str(exc).strip()[:300] or "Processing failed."
        flash(
            f"The project defaults database could not be built right now. {error_message}",
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

    saved_record_filename = str(source.get("saved_record_filename", "") or "").strip()
    if saved_record_filename:
        record_path = DASHBOARD_UPLOAD_DIR / saved_record_filename
        try:
            record_path.unlink(missing_ok=True)
        except OSError as exc:
            print(f"Dashboard URL record delete failed: {exc}")

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
        session["current_backup"] = backup_state["backup_db_path"]
        session["current_backup_dir"] = backup_state["backup_dir"]
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

    return bot.create_response(user_input, status_callback=_status_callback)

@app.post("/delete/<chat_id>")
def reset(chat_id):
    if chat_id and chat_id in _CHATBOTS:
        del _CHATBOTS[chat_id]
    session.pop("chat_id", None)
    return jsonify({"ok": True})


@app.get("/health")
def health():
    return jsonify({"status": "ok", "chatbot_count": len(_CHATBOTS)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
