import json
from pathlib import Path
from typing import Any

from ingestion_pipeline import DataProcessor, DefaultDataProcessor
from project_config import PROJECT_ROOT, DATA_DIR
from project_config import DEFAULT_BATCH_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_EMBEDDING_DIM

def load_best_variant(unified_results_path: Path) -> dict[str, Any]:
    if not unified_results_path.exists():
        raise FileNotFoundError(f"Unified HPC results file not found: {unified_results_path}")

    payload = json.loads(unified_results_path.read_text())
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
    if not output_path or not Path(output_path).exists():
        resolved_db = get_new_db_from_variant(best_variant)
    else:
        try:
            resolved_db = _resolve_variant_db_path(output_path)
        except FileNotFoundError as exc:
            # Generate a new database when the benchmark output path is stale or missing.
            print(f"Warning: {exc}. Attempting to create new database from variant configuration.")
            resolved_db = get_new_db_from_variant(best_variant)
    return {
        "name": str(best_variant.get("name", "best_variant")),
        "output_path": str(resolved_db),
        "raw_variant": best_variant,
    }

def get_new_db_from_variant(variant: dict[str, Any]) -> Path:
    embedding_method = str(variant.get("embedding_method", "dummy")).strip().lower() or "dummy"

    try:
        chunk_size = int(variant.get("chunk_size", DEFAULT_CHUNK_SIZE))
    except (TypeError, ValueError):
        chunk_size = DEFAULT_CHUNK_SIZE

    try:
        chunk_overlap = int(variant.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP))
    except (TypeError, ValueError):
        chunk_overlap = DEFAULT_CHUNK_OVERLAP

    try:
        embedding_dim = int(variant.get("embedding_dim", DEFAULT_EMBEDDING_DIM))
    except (TypeError, ValueError):
        embedding_dim = DEFAULT_EMBEDDING_DIM

    try:
        batch_size = int(variant.get("batch_size", DEFAULT_BATCH_SIZE))
    except (TypeError, ValueError):
        batch_size = DEFAULT_BATCH_SIZE

    name = str(variant.get("name", "")).strip() or DataProcessor.build_variant_name(
        embedding_method=embedding_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
    )

    raw_output_path = str(variant.get("output_path", "")).strip()
    if raw_output_path:
        target_path = Path(raw_output_path)
        if not target_path.is_absolute():
            target_path = PROJECT_ROOT / target_path
    else:
        target_path = Path(DataProcessor.build_variant_output_path(name=name, db_dir=DATA_DIR))

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return target_path

    base_processor = DefaultDataProcessor()
    created_variant = base_processor.create_variant(
        name=name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        output_path=str(target_path),
        embedding_method=embedding_method,
    )
    return Path(created_variant.output_path)

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

__all__ = [
    "load_best_variant",
    "_to_float",
    "_variant_sort_key",
    "_resolve_variant_db_path"
]
