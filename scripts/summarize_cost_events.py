from __future__ import annotations

import argparse
import json
from pathlib import Path

from cost_logger import summarize_cost_events
from project_config import COST_LOG_PATH


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize append-only cost events written by cost_logger.",
    )
    parser.add_argument(
        "--cost-log",
        default=str(COST_LOG_PATH),
        help="Path to cost events JSONL file (default: project config path).",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write summary JSON.",
    )
    args = parser.parse_args()

    summary = summarize_cost_events(Path(args.cost_log))
    text = json.dumps(summary, indent=2)
    print(text)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
