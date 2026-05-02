from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chatbot import Chatbot

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a list of questions with Chatbot and write responses to a text file."
    )
    parser.add_argument(
        "--questions",
        default="demo_questions.txt",
        help="Path to a text file containing one question per line.",
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


def write_results(output_path: Path, results: list[tuple[int, str, dict]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("Automated Chatbot Responses\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write("=" * 80 + "\n\n")
        for idx, question, payload in results:
            f.write(f"Question {idx}: {question}\n")
            f.write(f"Answer {idx}: {payload.get('reply', '')}\n")
            evidence_block = format_sources_and_evidence(payload)
            if evidence_block:
                f.write(f"{evidence_block}\n")
            f.write("-" * 80 + "\n")


def main() -> None:
    args = parse_args()
    questions_path = Path(args.questions)
    if not questions_path.is_absolute():
        questions_path = PROJECT_ROOT / questions_path
    output_path = make_output_path(args.output)

    questions = load_questions(questions_path)
    bot = Chatbot()
    status_cb = status_callback_factory(bot)

    print(bot.onboard_prompt)
    print(f"Loaded {len(questions)} questions from: {questions_path}")

    results: list[tuple[int, str, dict]] = []
    for idx, question in enumerate(questions, start=1):
        print(f"\n[{idx}/{len(questions)}] {question}")
        try:
            payload = bot.create_response(question, status_callback=status_cb)
        except Exception as exc:
            payload = {"reply": f"[ERROR] {exc}", "citations": [], "evidence": [], "retrieval": None}
        print(f"{bot.name}: {payload['reply']}")
        evidence_block = format_sources_and_evidence(payload)
        if evidence_block:
            print(evidence_block)
        results.append((idx, question, payload))

    write_results(output_path, results)
    print(f"\nSaved responses to: {output_path}")


if __name__ == "__main__":
    main()
