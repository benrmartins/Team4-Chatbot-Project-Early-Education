from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

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
    return Path("outputs") / f"chatbot_responses_{stamp}.txt"


def status_callback_factory(bot: Chatbot):
    def _update_status(status: str) -> None:
        if status == "running_tools":
            print(f"{bot.name}: thinking... (running tools)")
        elif status == "generating_final":
            print(f"{bot.name}: thinking... (writing answer)")

    return _update_status


def write_results(output_path: Path, results: list[tuple[int, str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("Automated Chatbot Responses\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write("=" * 80 + "\n\n")
        for idx, question, answer in results:
            f.write(f"Question {idx}: {question}\n")
            f.write(f"Answer {idx}: {answer}\n")
            f.write("-" * 80 + "\n")


def main() -> None:
    args = parse_args()
    questions_path = Path(args.questions)
    output_path = make_output_path(args.output)

    questions = load_questions(questions_path)
    bot = Chatbot()
    status_cb = status_callback_factory(bot)

    print(bot.onboard_prompt)
    print(f"Loaded {len(questions)} questions from: {questions_path}")

    results: list[tuple[int, str, str]] = []
    for idx, question in enumerate(questions, start=1):
        print(f"\n[{idx}/{len(questions)}] {question}")
        try:
            answer = bot.create_response(question, status_callback=status_cb)
        except Exception as exc:
            answer = f"[ERROR] {exc}"
        print(f"{bot.name}: {answer}")
        results.append((idx, question, answer))

    write_results(output_path, results)
    print(f"\nSaved responses to: {output_path}")


if __name__ == "__main__":
    main()