from __future__ import annotations

import argparse
import datetime
import hashlib
import itertools
import json
import os
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


@dataclass(frozen=True)
class HPCExecutionConfig:
    num_shards: int
    shard_index: int
    node_count: int
    gpus_per_job: int
    cpus_per_job: int


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
    retrieval_failures = 0
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

        retrieval_status = result.get("retrieval_status", "ok") if isinstance(result, dict) else "failed"
        if retrieval_status != "ok":
            retrieval_failures += 1

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
                "retrieval_status": retrieval_status,
                "error_code": result.get("error_code") if isinstance(result, dict) else "unknown_error",
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
    failure_rate = retrieval_failures / question_count

    base_score = (
        0.45 * hit_rate
        + 0.35 * mrr
        + 0.10 * confidence_calibration
        + 0.10 * citation_completeness
    )
    retrieval_score = max(0.0, base_score - (0.50 * failure_rate))

    return {
        "retrieval_score": round(retrieval_score, 4),
        "base_score": round(base_score, 4),
        "hit_rate": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "confidence_calibration": round(confidence_calibration, 4),
        "citation_completeness": round(citation_completeness, 4),
        "retrieval_failures": retrieval_failures,
        "failure_rate": round(failure_rate, 4),
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
    name_set = {spec.name for spec in specs}
    if len(name_set) != len(specs):
        raise ValueError("Duplicate variant names detected; shard assignment would be ambiguous.")
    if limit and limit > 0:
        return specs[:limit]
    return specs


def _resolve_hpc_execution_config(args: argparse.Namespace) -> HPCExecutionConfig:
    env_shards = os.getenv("SLURM_ARRAY_TASK_COUNT", "").strip()
    env_index = os.getenv("SLURM_ARRAY_TASK_ID", "").strip()

    num_shards = int(args.num_shards)
    shard_index = int(args.shard_index)

    if int(args.hpc_job_count) > 0:
        num_shards = int(args.hpc_job_count)
    elif not args.no_slurm_autodetect and env_shards:
        num_shards = int(env_shards)

    if int(args.hpc_job_index) >= 0:
        shard_index = int(args.hpc_job_index)
    elif not args.no_slurm_autodetect and env_index:
        shard_index = int(env_index)

    if num_shards <= 0:
        raise ValueError("num_shards must be a positive integer.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index must be in range [0, {num_shards - 1}] for num_shards={num_shards}."
        )

    return HPCExecutionConfig(
        num_shards=num_shards,
        shard_index=shard_index,
        node_count=max(1, int(args.hpc_node_count)),
        gpus_per_job=max(0, int(args.hpc_gpus_per_job)),
        cpus_per_job=max(1, int(args.hpc_cpus_per_job)),
    )


def _stable_shard_for_variant(variant_name: str, num_shards: int) -> int:
    digest = hashlib.sha256(variant_name.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_shards


def _select_shard(specs: list[VariantSpec], cfg: HPCExecutionConfig) -> list[VariantSpec]:
    return [
        spec
        for spec in specs
        if _stable_shard_for_variant(spec.name, cfg.num_shards) == cfg.shard_index
    ]


class VariantTestRunner:
    def __init__(
        self,
        base_processor: DataProcessor | None = None,
        benchmark_path: Path | None = None,
        max_results: int = VARIANT_TEST_MAX_RESULTS,
        limit: int | None = None,
        rebuild_existing: bool = False,
        hpc_config: HPCExecutionConfig | None = None,
        include_baseline: bool = True,
        cleanup_variant_dbs: bool = False,
    ):
        self.base_processor = base_processor or DefaultDataProcessor()
        self.benchmark_path = benchmark_path or Path(VARIANT_TEST_BENCHMARK_PATH)
        self.max_results = max_results
        self.limit = limit
        self.rebuild_existing = rebuild_existing
        self.hpc_config = hpc_config or HPCExecutionConfig(
            num_shards=1,
            shard_index=0,
            node_count=1,
            gpus_per_job=0,
            cpus_per_job=1,
        )
        # In sharded runs, baseline should execute once on shard 0 only.
        self.include_baseline = include_baseline and not (
            self.hpc_config.num_shards > 1 and self.hpc_config.shard_index != 0
        )
        self.cleanup_variant_dbs = cleanup_variant_dbs
        full_specs = build_variant_specs(limit=None)
        sharded_specs = _select_shard(full_specs, self.hpc_config)
        if limit and limit > 0:
            sharded_specs = sharded_specs[:limit]
        self.specs = sharded_specs
        self.total_spec_count = len(full_specs)
        self.variant_results: list[dict[str, Any]] = []

    def _ensure_base_data(self) -> None:
        if self.base_processor.web_data is None or self.base_processor.source_summary is None:
            self.base_processor.crawl()

    def _ensure_default_database(self) -> None:
        """Ensure the canonical default datastore exists before variant evaluation."""
        self._ensure_base_data()
        default_db = Path(self.base_processor.output_path)
        if default_db.exists() and not self.rebuild_existing:
            return

        if default_db.exists() and self.rebuild_existing:
            default_db.unlink()

        print(f"Default database not found; creating: {default_db}")
        self.base_processor.chunk()
        self.base_processor.embed()

    def _create_or_rebuild_variant(self, spec: VariantSpec) -> DataProcessor:
        variant = DataProcessor(
            name=spec.name,
            chunk_size=spec.chunk_size,
            chunk_overlap=spec.chunk_overlap,
            embedding_dim=spec.embedding_dim,
            batch_size=spec.batch_size,
            output_path=spec.output_path,
            embedding_method=spec.embedding_method,
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
        if self.include_baseline:
            self._ensure_default_database()
        else:
            self._ensure_base_data()
        benchmark_items = _load_benchmark(Path(self.benchmark_path))

        print(
            f"Shard {self.hpc_config.shard_index + 1}/{self.hpc_config.num_shards}: "
            f"running {len(self.specs)} of {self.total_spec_count} variants"
        )

        if self.include_baseline:
            baseline_payload = {
                "name": self.base_processor.name,
                "embedding_method": "dummy",
                "chunk_size": self.base_processor.chunk_size,
                "chunk_overlap": self.base_processor.chunk_overlap,
                "embedding_dim": self.base_processor.embedding_dim,
                "batch_size": self.base_processor.batch_size,
                "output_path": self.base_processor.output_path,
                **evaluate_variant(
                    db_path=self.base_processor.output_path,
                    benchmark_items=benchmark_items,
                    max_results=self.max_results,
                ),
            }
            self.variant_results.append(baseline_payload)

        for spec in self.specs:
            if str(spec.output_path) == str(self.base_processor.output_path):
                continue
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

            if self.cleanup_variant_dbs:
                variant_db = Path(variant.output_path)
                if variant_db.exists():
                    variant_db.unlink()

        self.variant_results.sort(key=lambda item: item["retrieval_score"], reverse=True)

    def print_results(self) -> None:
        print("Variant testing complete. Summary of results:")
        for payload in self.variant_results:
            print(
                f"{payload['name']}: score={payload['retrieval_score']} "
                f"(hit_rate={payload['hit_rate']}, mrr={payload['mrr']}, "
                f"confidence={payload['confidence_calibration']}, citations={payload['citation_completeness']})"
                f" - {payload['question_count']} questions, {payload['retrieval_failures']} retrieval failures"
                f" - {payload['failure_rate']*100:.1f}% failure rate"
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
            "include_baseline": self.include_baseline,
            "cleanup_variant_dbs": self.cleanup_variant_dbs,
            "hpc": asdict(self.hpc_config),
            "total_spec_count": self.total_spec_count,
            "shard_spec_count": len(self.specs),
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
        default=False,
        help="Delete and rebuild existing variant databases.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        default=False,
        help="Skip evaluating the default baseline in this run.",
    )
    parser.add_argument(
        "--cleanup-variant-dbs",
        action="store_true",
        default=False,
        help="Delete each variant SQLite DB after it is scored to reduce disk usage.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/variant_test_results.json",
        help="Where to write the test artifact.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of parallel shard jobs. Use 1 for no sharding.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index for this run.",
    )
    parser.add_argument(
        "--hpc-job-count",
        type=int,
        default=0,
        help="Optional alias for --num-shards for scheduler integration.",
    )
    parser.add_argument(
        "--hpc-job-index",
        type=int,
        default=-1,
        help="Optional alias for --shard-index for scheduler integration.",
    )
    parser.add_argument(
        "--hpc-node-count",
        type=int,
        default=1,
        help="Metadata only: total HPC nodes allocated to this run.",
    )
    parser.add_argument(
        "--hpc-gpus-per-job",
        type=int,
        default=0,
        help="Metadata only: GPUs available per job.",
    )
    parser.add_argument(
        "--hpc-cpus-per-job",
        type=int,
        default=1,
        help="Metadata only: CPUs available per job.",
    )
    parser.add_argument(
        "--no-slurm-autodetect",
        action="store_true",
        help="Disable auto-detecting shard count/index from SLURM_ARRAY_TASK_COUNT/SLURM_ARRAY_TASK_ID.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hpc_config = _resolve_hpc_execution_config(args)
    runner = VariantTestRunner(
        benchmark_path=Path(args.benchmark),
        max_results=int(args.max_results),
        limit=int(args.limit) if int(args.limit) > 0 else None,
        rebuild_existing=bool(args.rebuild_existing),
        hpc_config=hpc_config,
        include_baseline=not bool(args.skip_baseline),
        cleanup_variant_dbs=bool(args.cleanup_variant_dbs),
    )
    runner.run_tests()
    runner.print_results()
    runner.write_results_to_json(args.output_json)
