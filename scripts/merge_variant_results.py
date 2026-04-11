from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any

from project_config import PROJECT_ROOT


def _load_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid artifact format in {path}")
    return payload


def _variant_sort_key(item: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(item.get("retrieval_score", 0.0)),
        float(item.get("hit_rate", 0.0)),
        float(item.get("mrr", 0.0)),
        -float(item.get("failure_rate", 1.0)),
    )


def merge_artifacts(files: list[Path]) -> dict[str, Any]:
    by_name: dict[str, dict[str, Any]] = {}
    source_files: list[str] = []
    benchmark_path = ""
    max_results = 0
    total_specs = 0

    for path in files:
        artifact = _load_artifact(path)
        source_files.append(str(path))
        benchmark_path = benchmark_path or str(artifact.get("benchmark_path", ""))
        max_results = max(max_results, int(artifact.get("max_results", 0) or 0))
        total_specs = max(total_specs, int(artifact.get("total_spec_count", 0) or 0))

        for variant in artifact.get("variants", []) or []:
            if not isinstance(variant, dict):
                continue
            name = str(variant.get("name", "")).strip()
            if not name:
                continue

            current = by_name.get(name)
            if current is None or _variant_sort_key(variant) > _variant_sort_key(current):
                by_name[name] = variant

    merged_variants = sorted(by_name.values(), key=_variant_sort_key, reverse=True)
    duplicate_count = max(0, sum(len((_load_artifact(p).get("variants", []) or [])) for p in files) - len(merged_variants))

    return {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "benchmark_path": benchmark_path,
        "max_results": max_results,
        "source_files": source_files,
        "input_file_count": len(files),
        "duplicate_variants_removed": duplicate_count,
        "total_spec_count": total_specs,
        "variant_count": len(merged_variants),
        "variants": merged_variants,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge shard variant-test JSON artifacts into one ranked report.")
    parser.add_argument(
        "--input-glob",
        default="outputs/variant_test_results_shard_*_of_*.json",
        help="Glob pattern (relative to project root unless absolute) for shard artifacts.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/variant_test_results_merged.json",
        help="Output path for merged artifact.",
    )
    parser.add_argument(
        "--delete-inputs",
        action="store_true",
        default=False,
        help="Delete shard artifact files after successful merge.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_pattern = Path(args.input_glob)
    if not input_pattern.is_absolute():
        input_pattern = PROJECT_ROOT / input_pattern

    files = sorted(input_pattern.parent.glob(input_pattern.name))
    if not files:
        raise FileNotFoundError(f"No shard artifacts matched: {input_pattern}")

    merged = merge_artifacts(files)

    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"Saved merged artifact: {output_path}")

    if args.delete_inputs:
        for file in files:
            file.unlink(missing_ok=True)
        print(f"Deleted {len(files)} shard artifact files.")


if __name__ == "__main__":
    main()
