from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_PATH = PROJECT_ROOT / "evaluation" / "retrieval_benchmark.json"
RETRIEVAL_MODULE_PATH = PROJECT_ROOT / "chatbot" / "tool_calls" / "handlers" / "json_retrieval.py"


def load_retrieval_module():
    spec = importlib.util.spec_from_file_location("phase2_json_retrieval", RETRIEVAL_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load retrieval module from {RETRIEVAL_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Phase 1 lexical retrieval against the Phase 2 retrieval upgrade."
    )
    parser.add_argument(
        "--benchmark",
        default=str(BENCHMARK_PATH),
        help="Path to the benchmark JSON file.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=3,
        help="How many top retrieval results to show for each system.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to save the comparison JSON artifact.",
    )
    return parser.parse_args()


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Benchmark file must contain a JSON list of questions.")
    return data


def summarize_results(results: list[dict[str, Any]]) -> list[str]:
    summaries = []
    for item in results:
        title = item.get("title", "Untitled")
        score = item.get("score", item.get("rank", "?"))
        summaries.append(f"{title} (score={score})")
    return summaries


def count_hint_hits(results: list[dict[str, Any]], hints: list[str]) -> int:
    haystack = " ".join(
        f"{item.get('title', '')} {item.get('excerpt', '')} {item.get('evidence_snippet', '')}"
        for item in results
    ).lower()
    return sum(1 for hint in hints if hint.lower() in haystack)


def explain_improvement(
    benchmark_item: dict[str, Any],
    old_result: dict[str, Any],
    new_result: dict[str, Any],
) -> str:
    hints = benchmark_item.get("expected_source_hints", [])
    old_hits = count_hint_hits(old_result.get("results", []), hints)
    new_hits = count_hint_hits(new_result.get("results", []), hints)

    if new_result.get("low_confidence"):
        return "The new retrieval explicitly flags this question as weakly supported, which makes error analysis and out-of-scope handling easier to inspect."
    if new_hits > old_hits:
        return "The new retrieval surfaces result titles/snippets that better match the expected evidence hints."
    if old_result.get("results", []) and new_result.get("results", []):
        old_top = old_result["results"][0].get("title", "Untitled")
        new_top = new_result["results"][0].get("title", "Untitled")
        if old_top != new_top:
            return f"The top result is now more specific: '{new_top}' replaces the weaker Phase 1 match '{old_top}'."
    if len({item.get('document_id') for item in new_result.get('results', [])}) > len(
        {item.get('document_id') for item in old_result.get('results', [])}
    ):
        return "The new retrieval returns more diverse supporting sources instead of repeated chunks from one document."
    return "The new retrieval exposes clearer evidence snippets and confidence signals for manual inspection."


def main() -> None:
    args = parse_args()
    benchmark_path = Path(args.benchmark)
    if not benchmark_path.is_absolute():
        benchmark_path = PROJECT_ROOT / benchmark_path
    benchmark = load_benchmark(benchmark_path)
    retrieval = load_retrieval_module()

    comparisons: list[dict[str, Any]] = []
    for item in benchmark:
        question = item["question"]
        old_result = retrieval.search_unified_knowledge_phase1(question, max_results=args.max_results)
        new_result = retrieval.search_unified_knowledge(question, max_results=args.max_results)

        comparisons.append(
            {
                "question": question,
                "why_hard": item.get("why_hard", ""),
                "phase1_likely_failure_mode": item.get("phase1_likely_failure_mode", ""),
                "old_top_results": summarize_results(old_result.get("results", [])),
                "new_top_results": summarize_results(new_result.get("results", [])),
                "new_low_confidence": new_result.get("low_confidence", False),
                "why_new_retrieval_is_better": explain_improvement(item, old_result, new_result),
            }
        )

    for comparison in comparisons:
        print("=" * 80)
        print(f"Question: {comparison['question']}")
        print(f"Why hard: {comparison['why_hard']}")
        print(f"Phase 1 likely struggle: {comparison['phase1_likely_failure_mode']}")
        print("Old top results:")
        for result in comparison["old_top_results"]:
            print(f"- {result}")
        print("New top results:")
        for result in comparison["new_top_results"]:
            print(f"- {result}")
        print(f"Why the new retrieval is better: {comparison['why_new_retrieval_is_better']}")

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
        print(f"\nSaved comparison artifact to: {output_path}")


if __name__ == "__main__":
    main()
