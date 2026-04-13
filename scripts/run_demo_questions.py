from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chatbot import Chatbot
from metrics import score_final_answer_benchmark_item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a list of questions with Chatbot, score responses against a benchmark, and write results."
    )
    parser.add_argument(
        "--questions",
        default="demo_questions.txt",
        help="Path to a text file containing one question per line.",
    )
    parser.add_argument(
        "--benchmark",
        default="evaluation/retrieval_benchmark.json",
        help="Path to benchmark JSON file with ground truth answers. Scoring skipped if not found.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output text file path. Defaults to outputs/chatbot_responses_YYYYMMDD_HHMMSS.txt",
    )
    return parser.parse_args()


def load_questions(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    questions = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not questions:
        raise ValueError(f"No questions found in file: {path}")
    return questions


def load_benchmark(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        print(f"Benchmark file not found: {path} (scoring will be skipped)")
        return None
    
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            print(f"Benchmark file must be a JSON list (got {type(payload).__name__}); scoring will be skipped")
            return None
        return payload
    except json.JSONDecodeError as e:
        print(f"Failed to parse benchmark JSON: {e} (scoring will be skipped)")
        return None


def make_output_path(output_arg: str) -> Path:
    if output_arg:
        return Path(output_arg)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "outputs" / f"chatbot_responses_{stamp}.txt"


def status_callback_factory(bot: Chatbot):
    def _update_status(status: str) -> None:
        if status == "running_tools":
            print(f"{bot.name}: thinking... (running tools)")
        elif status == "generating_final":
            print(f"{bot.name}: thinking... (writing answer)")

    return _update_status


def _normalize_question_text(text: str) -> str:
    """Normalize question text for matching."""
    return str(text or "").strip().lower()


def find_benchmark_item(question: str, benchmark_items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Find a benchmark item that matches the question text."""
    normalized_question = _normalize_question_text(question)
    
    for item in benchmark_items:
        benchmark_question = str(item.get("question", "")).strip()
        if _normalize_question_text(benchmark_question) == normalized_question:
            return item
    
    return None


def _payload_evidence_to_retrieved_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert chatbot response evidence to retrieved_entries format for scoring."""
    retrieved_entries: list[dict[str, Any]] = []
    
    evidence = payload.get("evidence", [])
    citations = payload.get("citations", [])
    citation_map = {cit.get("url"): cit for cit in citations}
    
    for item in evidence:
        rank = item.get("rank", 0)
        title = item.get("title", "")
        url = item.get("url", "")
        snippet = item.get("snippet", "")
        
        citation = citation_map.get(url) or {"title": title, "url": url}
        
        entry = {
            "rank": rank,
            "title": title,
            "url": url,
            "document_id": url or title,
            "chunk_id": f"rank_{rank}",
            "evidence_snippet": snippet,
            "excerpt": snippet,
            "text": snippet,
            "citation": citation,
        }
        retrieved_entries.append(entry)
    
    return retrieved_entries


def score_response(
    question: str,
    answer_text: str,
    payload: dict[str, Any],
    benchmark_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Score a response using benchmark ground truth. Returns None if no benchmark match."""
    if not benchmark_items:
        return None
    
    benchmark_item = find_benchmark_item(question, benchmark_items)
    if not benchmark_item:
        return None
    
    retrieved_entries = _payload_evidence_to_retrieved_entries(payload)
    
    score_result = score_final_answer_benchmark_item(
        benchmark_item=benchmark_item,
        answer_text=answer_text,
        retrieved_entries=retrieved_entries,
    )
    
    return score_result


def format_sources_and_evidence(payload: dict) -> str:
    lines: list[str] = []

    citations = payload.get("citations", [])
    evidence = payload.get("evidence", [])
    retrieval = payload.get("retrieval") or {}

    if citations:
        lines.append("Citations:")
        for citation in citations:
            lines.append(f"- {citation.get('title', 'Untitled')} | {citation.get('url', '')}")

    if evidence:
        lines.append("Evidence:")
        for item in evidence:
            matched_terms = ", ".join(item.get("matched_terms", [])) or "n/a"
            lines.append(
                f"- Rank {item.get('rank', '?')} | {item.get('title', 'Untitled')} | matched: {matched_terms}"
            )
            lines.append(f"  Snippet: {item.get('snippet', '')}")

    if retrieval:
        lines.append(
            f"Retrieval confidence: {'low' if retrieval.get('low_confidence') else 'normal'}"
        )

    return "\n".join(lines)


def format_score(score_result: dict[str, Any] | None) -> str:
    """Format a score result for display."""
    if not score_result:
        return ""
    
    score = score_result.get("score", "N/A")
    correct = score_result.get("correct", 0)
    retrieval_hit = score_result.get("retrieval_hit", 0)
    hint_match = score_result.get("hint_match", 0)
    
    return f"Score: {score}/5 (correct={correct}, retrieval_hit={retrieval_hit}, hint_match={hint_match})"


def write_results(
    output_path: Path,
    results: list[tuple[int, str, dict[str, Any], dict[str, Any] | None]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("Automated Chatbot Responses with Scoring\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write("=" * 80 + "\n\n")
        for idx, question, payload, score_result in results:
            f.write(f"Question {idx}: {question}\n")
            f.write(f"Answer {idx}: {payload.get('reply', '')}\n")
            
            score_line = format_score(score_result)
            if score_line:
                f.write(f"{score_line}\n")
            
            evidence_block = format_sources_and_evidence(payload)
            if evidence_block:
                f.write(f"{evidence_block}\n")
            
            f.write("-" * 80 + "\n")


def main() -> None:
    args = parse_args()
    questions_path = Path(args.questions)
    if not questions_path.is_absolute():
        questions_path = PROJECT_ROOT / questions_path
    
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.is_absolute():
        benchmark_path = PROJECT_ROOT / benchmark_path
    
    output_path = make_output_path(args.output)

    questions = load_questions(questions_path)
    benchmark_items = load_benchmark(benchmark_path)
    bot = Chatbot()
    status_cb = status_callback_factory(bot)

    print(bot.onboard_prompt)
    print(f"Loaded {len(questions)} questions from: {questions_path}")
    if benchmark_items:
        print(f"Loaded {len(benchmark_items)} benchmark items from: {benchmark_path}")
    else:
        print(f"No benchmark available; responses will not be scored")

    results: list[tuple[int, str, dict[str, Any], dict[str, Any] | None]] = []
    for idx, question in enumerate(questions, start=1):
        print(f"\n[{idx}/{len(questions)}] {question}")
        try:
            payload = bot.create_response(question, status_callback=status_cb)
        except Exception as exc:
            payload = {"reply": f"[ERROR] {exc}", "citations": [], "evidence": [], "retrieval": None}
        
        answer_text = payload.get("reply", "")
        print(f"{bot.name}: {answer_text}")
        
        # Score the response if benchmark is available
        score_result = score_response(question, answer_text, payload, benchmark_items)
        if score_result:
            score_line = format_score(score_result)
            if score_line:
                print(score_line)
        
        evidence_block = format_sources_and_evidence(payload)
        if evidence_block:
            print(evidence_block)
        
        results.append((idx, question, payload, score_result))

    write_results(output_path, results)
    print(f"\nSaved responses to: {output_path}")


if __name__ == "__main__":
    main()