from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from project_config import PROJECT_ROOT, DATA_DIR

from metrics import aggregate_retrieval_benchmark_scores, score_retrieval_benchmark_item

BENCHMARK_PATH = PROJECT_ROOT / "evaluation" / "retrieval_benchmark.json"
UNIFIED_HPC_RESULTS_PATH = PROJECT_ROOT / "outputs" / "hpc" / "unified_variant_results.json"
PHASE1_RETRIEVAL_MODULE_PATH = PROJECT_ROOT / "chatbot" / "tool_calls" / "handlers" / "json_retrieval.py"
SQLITE_RETRIEVAL_MODULE_PATH = PROJECT_ROOT / "chatbot" / "tool_calls" / "handlers" / "database_retrieval.py"


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_phase1_search_fn():
    module = _load_module_from_path("phase1_json_retrieval", PHASE1_RETRIEVAL_MODULE_PATH)
    if not hasattr(module, "search_unified_knowledge"):
        raise RuntimeError("Phase 1 retrieval module does not expose search_unified_knowledge().")
    return module.search_unified_knowledge


def _load_sqlite_search_fn():
    try:
        module = _load_module_from_path("sqlite_variant_retrieval", SQLITE_RETRIEVAL_MODULE_PATH)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to load SQLite retrieval module due to a missing dependency. "
            "Install project requirements (for example: pip install -r requirements.txt)."
        ) from exc

    if not hasattr(module, "search_sqlite_knowledge"):
        raise RuntimeError("SQLite retrieval module does not expose search_sqlite_knowledge().")
    return module.search_sqlite_knowledge


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _variant_sort_key(variant: dict[str, Any]) -> tuple[float, float, float, float]:
    # Prefer newer normalized score keys; fallback to legacy retrieval_score.
    score_key = _to_float(variant.get("avg_score"), default=-1.0)
    if score_key < 0:
        score_key = _to_float(variant.get("retrieval_score"), default=0.0)

    hit_key = _to_float(variant.get("retrieval_hit_rate"), default=-1.0)
    if hit_key < 0:
        hit_key = _to_float(variant.get("hit_rate"), default=0.0)

    accuracy_key = _to_float(variant.get("accuracy"), default=0.0)
    failure_penalty = -_to_float(variant.get("retrieval_failures"), default=0.0)
    return (score_key, hit_key, accuracy_key, failure_penalty)


def _resolve_variant_db_path(raw_output_path: str) -> Path:
    candidate = Path(str(raw_output_path))
    if candidate.exists():
        return candidate

    by_name = DATA_DIR / candidate.name
    if by_name.exists():
        return by_name

    normalized = str(raw_output_path).replace("\\", "/")
    marker = "/data/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        by_suffix = PROJECT_ROOT / "data" / suffix
        if by_suffix.exists():
            return by_suffix

    raise FileNotFoundError(
        f"Could not resolve variant database path from '{raw_output_path}'. "
        f"Checked direct path and local data fallback in {PROJECT_ROOT / 'data'}."
    )


def _load_best_variant(unified_results_path: Path) -> dict[str, Any]:
    if not unified_results_path.exists():
        raise FileNotFoundError(f"Unified HPC results file not found: {unified_results_path}")

    payload = json.loads(unified_results_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Unified HPC results must be a JSON object.")

    variants = payload.get("variants", [])
    if not isinstance(variants, list) or not variants:
        raise ValueError("Unified HPC results must contain a non-empty 'variants' list.")

    ranked_variants = sorted(variants, key=_variant_sort_key, reverse=True)
    best_variant = ranked_variants[0]
    if not isinstance(best_variant, dict):
        raise ValueError("Best variant entry is not an object.")

    output_path = str(best_variant.get("output_path", "")).strip()
    if not output_path:
        raise ValueError("Best variant does not include output_path.")

    resolved_db = _resolve_variant_db_path(output_path)
    return {
        "name": str(best_variant.get("name", "best_variant")),
        "output_path": str(resolved_db),
        "raw_variant": best_variant,
    }


def _summarize_results(results: list[dict[str, Any]]) -> list[str]:
    summaries: list[str] = []
    for item in results:
        title = item.get("title", "Untitled")
        rank = item.get("rank", "?")
        score = item.get("score", "?")
        summaries.append(f"#{rank} {title} (score={score})")
    return summaries


def _item_metric_subset(item_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "correct": int(item_metrics.get("correct", 0)),
        "retrieval_hit": int(item_metrics.get("retrieval_hit", 0)),
        "score": int(item_metrics.get("score", 1)),
        "hint_match": int(item_metrics.get("hint_match", 0)),
        "notes": str(item_metrics.get("notes", "")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Phase 1 JSON retrieval against best SQLite variant retrieval."
    )
    parser.add_argument(
        "--benchmark",
        default=str(BENCHMARK_PATH),
        help="Path to the benchmark JSON file.",
    )
    parser.add_argument(
        "--unified-hpc-results",
        default=str(UNIFIED_HPC_RESULTS_PATH),
        help="Path to unified_hpc_results.json used to select best variant DB.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="Top-k retrieval results to request for each system.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to save comparison artifact JSON.",
    )
    return parser.parse_args()


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Benchmark file must contain a JSON list of questions.")
    return data


def main() -> None:
    args = parse_args()
    phase1_search = _load_phase1_search_fn()
    sqlite_search = _load_sqlite_search_fn()

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.is_absolute():
        benchmark_path = PROJECT_ROOT / benchmark_path

    unified_results_path = Path(args.unified_hpc_results)
    if not unified_results_path.is_absolute():
        unified_results_path = UNIFIED_HPC_RESULTS_PATH / "unified_variant_results.json"

    benchmark = load_benchmark(benchmark_path)
    best_variant = _load_best_variant(unified_results_path)
    best_db_path = best_variant["output_path"]

    phase1_details: list[dict[str, Any]] = []
    best_details: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    for item in benchmark:
        question = str(item.get("question", "")).strip()
        if not question:
            continue

        phase1_result = phase1_search(question, max_results=args.max_results)
        best_result = sqlite_search(
            query=question,
            max_results=args.max_results,
            database_path=best_db_path,
        )

        phase1_item_metrics = score_retrieval_benchmark_item(item, phase1_result)
        best_item_metrics = score_retrieval_benchmark_item(item, best_result)

        phase1_details.append(phase1_item_metrics)
        best_details.append(best_item_metrics)

        comparisons.append(
            {
                "question": question,
                "phase1": {
                    "metrics": _item_metric_subset(phase1_item_metrics),
                    "retrieval_status": phase1_item_metrics.get("retrieval_status", "ok"),
                    "top_results": _summarize_results(phase1_result.get("results", [])),
                },
                "best_variant": {
                    "metrics": _item_metric_subset(best_item_metrics),
                    "retrieval_status": best_item_metrics.get("retrieval_status", "ok"),
                    "top_results": _summarize_results(best_result.get("results", [])),
                },
            }
        )

    phase1_aggregate = aggregate_retrieval_benchmark_scores(phase1_details)
    best_aggregate = aggregate_retrieval_benchmark_scores(best_details)

    summary = {
        "benchmark_path": str(benchmark_path),
        "max_results": args.max_results,
        "phase1_source": str(PROJECT_ROOT / "data" / "unified_vector_data.json"),
        "best_variant": {
            "name": best_variant["name"],
            "database_path": best_db_path,
        },
        "phase1_metrics": {
            "accuracy": phase1_aggregate["accuracy"],
            "retrieval_hit_rate": phase1_aggregate["retrieval_hit_rate"],
            "avg_score": phase1_aggregate["avg_score"],
            "hint_match_rate": phase1_aggregate["hint_match_rate"],
        },
        "best_variant_metrics": {
            "accuracy": best_aggregate["accuracy"],
            "retrieval_hit_rate": best_aggregate["retrieval_hit_rate"],
            "avg_score": best_aggregate["avg_score"],
            "hint_match_rate": best_aggregate["hint_match_rate"],
        },
        "delta_best_minus_phase1": {
            "accuracy": round(best_aggregate["accuracy"] - phase1_aggregate["accuracy"], 4),
            "retrieval_hit_rate": round(best_aggregate["retrieval_hit_rate"] - phase1_aggregate["retrieval_hit_rate"], 4),
            "avg_score": round(best_aggregate["avg_score"] - phase1_aggregate["avg_score"], 4),
            "hint_match_rate": round(best_aggregate["hint_match_rate"] - phase1_aggregate["hint_match_rate"], 4),
        },
    }

    artifact = {
        "summary": summary,
        "per_question": comparisons,
    }

    print("=" * 80)
    print("PHASE 1 VS BEST VARIANT BENCHMARK")
    print("=" * 80)
    print(f"Best variant: {best_variant['name']}")
    print(f"Best variant DB: {best_db_path}")
    print("\nPhase 1 metrics:")
    print(f"  accuracy: {summary['phase1_metrics']['accuracy']}")
    print(f"  retrieval_hit_rate: {summary['phase1_metrics']['retrieval_hit_rate']}")
    print(f"  avg_score: {summary['phase1_metrics']['avg_score']}")
    print(f"  hint_match_rate: {summary['phase1_metrics']['hint_match_rate']}")
    print("\nBest variant metrics:")
    print(f"  accuracy: {summary['best_variant_metrics']['accuracy']}")
    print(f"  retrieval_hit_rate: {summary['best_variant_metrics']['retrieval_hit_rate']}")
    print(f"  avg_score: {summary['best_variant_metrics']['avg_score']}")
    print(f"  hint_match_rate: {summary['best_variant_metrics']['hint_match_rate']}")
    print("\nDelta (best - phase1):")
    print(f"  accuracy: {summary['delta_best_minus_phase1']['accuracy']}")
    print(f"  retrieval_hit_rate: {summary['delta_best_minus_phase1']['retrieval_hit_rate']}")
    print(f"  avg_score: {summary['delta_best_minus_phase1']['avg_score']}")
    print(f"  hint_match_rate: {summary['delta_best_minus_phase1']['hint_match_rate']}")

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print(f"\nSaved comparison artifact to: {output_path}")


if __name__ == "__main__":
    main()
