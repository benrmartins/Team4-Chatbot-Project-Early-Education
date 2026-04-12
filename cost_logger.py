from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock
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

    # Append-only event records are safer than read-modify-write totals across HPC shards.
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": (model or "").strip().lower(),
        "prompt_tokens": max(int(prompt_tokens or 0), 0),
        "completion_tokens": max(int(completion_tokens or 0), 0),
        "total_tokens": max(int(total_tokens or 0), 0) if total_tokens is not None else None,
        "estimated_cost_usd": round(float(estimated_cost), 10),
        "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID", ""),
        "hostname": os.getenv("HOSTNAME", ""),
        "user": os.getenv("USER", ""),
    }
    line = json.dumps(event, separators=(",", ":")) + "\n"

    lock_path = cost_path.with_suffix(cost_path.suffix + ".lock")
    with FileLock(str(lock_path), timeout=30):
        with cost_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()

    return estimated_cost


def summarize_cost_events(cost_file_path: Path | str | None = None) -> dict[str, Any]:
    cost_path = Path(cost_file_path) if cost_file_path else Path(COST_LOG_PATH)
    if not cost_path.exists():
        return {
            "events": 0,
            "total_estimated_cost_usd": 0.0,
            "by_model": {},
        }

    by_model: dict[str, float] = {}
    events = 0
    total = 0.0
    with cost_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue

            model = str(event.get("model", "unknown") or "unknown")
            cost = float(event.get("estimated_cost_usd", 0.0) or 0.0)
            events += 1
            total += cost
            by_model[model] = by_model.get(model, 0.0) + cost

    return {
        "events": events,
        "total_estimated_cost_usd": round(total, 10),
        "by_model": {k: round(v, 10) for k, v in sorted(by_model.items())},
    }


__all__ = [
    "estimate_cost_usd",
    "log_api_usage",
    "summarize_cost_events",
]
