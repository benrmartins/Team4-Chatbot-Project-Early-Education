from uuid import uuid4

from flask import Flask, jsonify, render_template, request, session

from chatbot import Chatbot

from scripts.run_retrieval_benchmark import _load_best_variant
from project_config import UNIFIED_HPC_RESULTS_PATH

app = Flask(__name__)
app.secret_key = "change-me-in-production"

# In-memory chat state keyed by browser session id.
_CHATBOTS = {}

def _load_benchmark_chatbot() -> Chatbot:
    variant = _load_best_variant(UNIFIED_HPC_RESULTS_PATH)
    if not variant:
        raise ValueError("No benchmark variant found to load chatbot from.")
    output_path = variant.get("output_path")
    if not output_path:
        raise ValueError("Benchmark variant does not specify a knowledge base path.")
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


@app.get("/")
def index():
    bot = _get_chatbot()
    return render_template("index.html", bot_name=bot.name, onboard_prompt=bot.onboard_prompt)


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
