import json
from pathlib import Path
from typing import Any

from project_config import PROJECT_ROOT, DATA_DIR

def load_best_variant(unified_results_path: Path) -> dict[str, Any]:
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
