from __future__ import annotations

import json
from pathlib import Path

from project_config import COST_LOG_PATH

# Public list of known per-model token pricing in USD per 1M tokens.
# These values are best-effort estimates and can be updated over time.
_MODEL_PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
    "text-embedding-3-large": {"input": 0.13, "output": 0.00},
}


def estimate_cost_usd(
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int | None = None,
) -> float:
    model_key = (model or "").strip().lower()
    pricing = _MODEL_PRICING_USD_PER_1M.get(model_key)
    if not pricing:
        return 0.0

    prompt = max(int(prompt_tokens or 0), 0)
    completion = max(int(completion_tokens or 0), 0)

    # Embedding APIs frequently report only total/prompt tokens.
    if completion == 0 and total_tokens is not None:
        prompt = max(int(total_tokens or 0), 0)

    input_cost = (prompt / 1_000_000.0) * float(pricing.get("input", 0.0))
    output_cost = (completion / 1_000_000.0) * float(pricing.get("output", 0.0))
    return input_cost + output_cost


def log_api_usage(
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int | None = None,
    cost_file_path: Path | str | None = None,
) -> float:
    estimated_cost = estimate_cost_usd(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )

    if estimated_cost <= 0:
        return 0.0

    cost_path = Path(cost_file_path) if cost_file_path else Path(COST_LOG_PATH)
    cost_path.parent.mkdir(parents=True, exist_ok=True)

    running_total = 0.0
    if cost_path.exists():
        try:
            existing = json.loads(cost_path.read_text(encoding="utf-8") or "{}")
            running_total = float(existing.get("total_estimated_cost", 0.0))
        except Exception:
            running_total = 0.0

    new_total = running_total + estimated_cost
    payload = {"total_estimated_cost": round(new_total, 10)}

    temp_path = cost_path.with_suffix(cost_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(cost_path)

    return estimated_cost


__all__ = [
    "estimate_cost_usd",
    "log_api_usage",
]
