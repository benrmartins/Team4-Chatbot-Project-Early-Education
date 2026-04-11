from __future__ import annotations

import argparse
import datetime
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from chatbot.tool_calls.handlers.database_retrieval import search_sqlite_knowledge
from ingestion_pipeline import DataProcessor, DefaultDataProcessor, JSONDict
from project_config import (
    PROJECT_ROOT,
    VARIANT_TEST_BATCH_SIZES,
    VARIANT_TEST_BENCHMARK_PATH,
    VARIANT_TEST_CHUNK_OVERLAPS,
    VARIANT_TEST_CHUNK_SIZES,
    DATA_DIR,
    VARIANT_TEST_EMBEDDING_DIMS,
    VARIANT_TEST_EMBEDDING_METHODS,
    VARIANT_TEST_MAX_RESULTS,
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    embedding_method: str
    chunk_size: int
    chunk_overlap: int
    embedding_dim: int
    batch_size: int
    output_path: str


def _ordered_unique(values: list[int]) -> list[int]:
    return sorted(set(int(v) for v in values))


def _ordered_unique_str(values: list[str]) -> list[str]:
    return sorted(set(str(v).strip().lower() for v in values if str(v).strip()))


def _load_benchmark(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Benchmark file must contain a list of questions.")
    return payload


def _hit_rank(results: list[dict[str, Any]], hints: list[str]) -> int | None:
    normalized_hints = [hint.lower().strip() for hint in hints if hint.strip()]
    if not normalized_hints:
        return None

    for idx, result in enumerate(results, start=1):
        haystack = " ".join(
            [
                str(result.get("title", "")),
                str(result.get("evidence_snippet", "")),
                str(result.get("excerpt", "")),
                str(result.get("url", "")),
            ]
        ).lower()
        if any(hint in haystack for hint in normalized_hints):
            return idx
    return None


def _is_out_of_scope_case(item: dict[str, Any]) -> bool:
    return "out-of-scope" in str(item.get("why_hard", "")).lower()


def evaluate_variant(
    db_path: str,
    benchmark_items: list[dict[str, Any]],
    max_results: int,
) -> dict[str, Any]:
    hit_count = 0
    reciprocal_rank_total = 0.0
    confidence_points = 0
    citation_points = 0.0
    details: list[dict[str, Any]] = []

    for item in benchmark_items:
        query = str(item.get("question", "")).strip()
        if not query:
            continue

        expected_hints = item.get("expected_source_hints", []) or []
        result = search_sqlite_knowledge(
            query=query,
            max_results=max_results,
            database_path=db_path,
        )

        ranked_results = result.get("results", []) if isinstance(result, dict) else []
        first_hit_rank = _hit_rank(ranked_results, expected_hints)
        hit = 1 if first_hit_rank is not None else 0
        mrr = 1.0 / first_hit_rank if first_hit_rank else 0.0

        if _is_out_of_scope_case(item):
            confidence_score = 1 if result.get("low_confidence", False) else 0
        else:
            confidence_score = 0 if result.get("low_confidence", False) else 1

        citation_rows = [r for r in ranked_results if (r.get("title") and r.get("url"))]
        citation_score = len(citation_rows) / max(1, len(ranked_results)) if ranked_results else 0.0

        hit_count += hit
        reciprocal_rank_total += mrr
        confidence_points += confidence_score
        citation_points += citation_score

        details.append(
            {
                "question": query,
                "hit": hit,
                "first_hit_rank": first_hit_rank,
                "mrr": round(mrr, 4),
                "low_confidence": bool(result.get("low_confidence", False)),
                "confidence_score": confidence_score,
                "citation_score": round(citation_score, 4),
                "result_count": int(result.get("result_count", 0)),
            }
        )

    question_count = max(1, len(details))
    hit_rate = hit_count / question_count
    mrr = reciprocal_rank_total / question_count
    confidence_calibration = confidence_points / question_count
    citation_completeness = citation_points / question_count

    retrieval_score = (
        0.45 * hit_rate
        + 0.35 * mrr
        + 0.10 * confidence_calibration
        + 0.10 * citation_completeness
    )

    return {
        "retrieval_score": round(retrieval_score, 4),
        "hit_rate": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "confidence_calibration": round(confidence_calibration, 4),
        "citation_completeness": round(citation_completeness, 4),
        "question_count": question_count,
        "details": details,
    }


def build_variant_specs(limit: int | None = None) -> list[VariantSpec]:
    chunk_sizes = _ordered_unique(list(VARIANT_TEST_CHUNK_SIZES))
    chunk_overlaps = _ordered_unique(list(VARIANT_TEST_CHUNK_OVERLAPS))
    embedding_dims = _ordered_unique(list(VARIANT_TEST_EMBEDDING_DIMS))
    batch_sizes = _ordered_unique(list(VARIANT_TEST_BATCH_SIZES))
    embedding_methods = _ordered_unique_str(list(VARIANT_TEST_EMBEDDING_METHODS))

    db_dir = Path(DATA_DIR)
    db_dir.mkdir(parents=True, exist_ok=True)

    specs: list[VariantSpec] = []
    for method, chunk_size, chunk_overlap, embedding_dim, batch_size in itertools.product(
        embedding_methods,
        chunk_sizes,
        chunk_overlaps,
        embedding_dims,
        batch_sizes,
    ):
        name = DataProcessor.build_variant_name(
            embedding_method=method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
        )
        output_path = DataProcessor.build_variant_output_path(name=name, db_dir=db_dir)
        specs.append(
            VariantSpec(
                name=name,
                embedding_method=method,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_dim=embedding_dim,
                batch_size=batch_size,
                output_path=output_path,
            )
        )

    specs.sort(key=lambda s: s.name)
    if limit and limit > 0:
        return specs[:limit]
    return specs


class VariantTestRunner:
    def __init__(
        self,
        base_processor: DataProcessor | None = None,
        benchmark_path: Path | None = None,
        max_results: int = VARIANT_TEST_MAX_RESULTS,
        limit: int | None = None,
        rebuild_existing: bool = False,
    ):
        self.base_processor = base_processor or DefaultDataProcessor()
        self.benchmark_path = benchmark_path or Path(VARIANT_TEST_BENCHMARK_PATH)
        self.max_results = max_results
        self.limit = limit
        self.rebuild_existing = rebuild_existing
        self.specs = build_variant_specs(limit=limit)
        self.variant_results: list[dict[str, Any]] = []

    def _ensure_base_data(self) -> None:
        if self.base_processor.web_data is None or self.base_processor.source_summary is None:
            self.base_processor.crawl()

    def _create_or_rebuild_variant(self, spec: VariantSpec) -> DataProcessor:
        if spec.embedding_method != "default":
            raise ValueError(
                f"Unsupported embedding method '{spec.embedding_method}'. "
                "Currently supported methods: ['default']."
            )

        variant = DataProcessor(
            name=spec.name,
            chunk_size=spec.chunk_size,
            chunk_overlap=spec.chunk_overlap,
            embedding_dim=spec.embedding_dim,
            batch_size=spec.batch_size,
            output_path=spec.output_path,
        )
        variant.web_data = self.base_processor.web_data
        variant.source_summary = self.base_processor.source_summary

        db_path = Path(spec.output_path)
        if db_path.exists() and self.rebuild_existing:
            db_path.unlink()

        if not db_path.exists():
            variant.chunk()
            variant.embed()

        return variant

    def run_tests(self) -> None:
        self._ensure_base_data()
        benchmark_items = _load_benchmark(Path(self.benchmark_path))

        for spec in self.specs:
            print(f"Building variant: {spec.name}")
            variant = self._create_or_rebuild_variant(spec)
            metrics = evaluate_variant(
                db_path=variant.output_path,
                benchmark_items=benchmark_items,
                max_results=self.max_results,
            )
            payload = {
                **asdict(spec),
                **metrics,
            }
            self.variant_results.append(payload)

        self.variant_results.sort(key=lambda item: item["retrieval_score"], reverse=True)

    def print_results(self) -> None:
        print("Variant testing complete. Summary of results:")
        for payload in self.variant_results:
            print(
                f"{payload['name']}: score={payload['retrieval_score']} "
                f"(hit_rate={payload['hit_rate']}, mrr={payload['mrr']}, "
                f"confidence={payload['confidence_calibration']}, citations={payload['citation_completeness']})"
            )

    def write_results_to_json(self, output_path: str) -> None:
        output_file = Path(output_path)
        if not output_file.is_absolute():
            output_file = PROJECT_ROOT / output_file
        output_file.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "benchmark_path": str(Path(self.benchmark_path)),
            "max_results": self.max_results,
            "variant_count": len(self.variant_results),
            "variants": self.variant_results,
        }
        output_file.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print(f"Saved variant test artifact to: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic variant matrix tests for retrieval.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on how many matrix variants to run (0 means all).",
    )
    parser.add_argument(
        "--benchmark",
        default=str(VARIANT_TEST_BENCHMARK_PATH),
        help="Path to benchmark JSON file.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=VARIANT_TEST_MAX_RESULTS,
        help="Top-k retrieval results used for scoring.",
    )
    parser.add_argument(
        "--rebuild-existing",
        action="store_true",
        help="Delete and rebuild existing variant databases.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/variant_test_results.json",
        help="Where to write the test artifact.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = VariantTestRunner(
        benchmark_path=Path(args.benchmark),
        max_results=int(args.max_results),
        limit=int(args.limit) if int(args.limit) > 0 else None,
        rebuild_existing=bool(args.rebuild_existing),
    )
    runner.run_tests()
    runner.print_results()
    runner.write_results_to_json(args.output_json)
