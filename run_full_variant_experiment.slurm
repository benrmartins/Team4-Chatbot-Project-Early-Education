#!/usr/bin/env bash
#SBATCH --job-name=variant-experiment
#SBATCH --output=outputs/hpc/%x-%A_%a.out
#SBATCH --error=outputs/hpc/%x-%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-7
#SBATCH --partition=pomplun         # Specify the partition
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --time=01:00:00             # Time limit hrs:min:sec
#SBATCH --account=cs438_team4       # Specify the account for the job


set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

mkdir -p outputs/hpc

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: python executable not found on PATH" >&2
  exit 1
fi

NUM_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
NODE_COUNT="${SLURM_JOB_NUM_NODES:-1}"
CPUS_PER_JOB="${SLURM_CPUS_PER_TASK:-1}"

# Best effort GPU count extraction (works for values like "2" or "gpu:2").
RAW_GPUS="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-0}}"
if [[ "$RAW_GPUS" =~ ([0-9]+) ]]; then
  GPUS_PER_JOB="${BASH_REMATCH[1]}"
else
  GPUS_PER_JOB="0"
fi

OUTPUT_JSON="outputs/variant_test_results_shard_${SHARD_INDEX}_of_${NUM_SHARDS}.json"

EXTRA_FLAGS=()
if [[ "$SHARD_INDEX" != "0" ]]; then
  EXTRA_FLAGS+=(--skip-baseline)
fi

# Keep disk usage low by deleting per-variant SQLite DBs after scoring.
if [[ "${VARIANT_CLEANUP_DBS:-1}" == "1" ]]; then
  EXTRA_FLAGS+=(--cleanup-variant-dbs)
fi

# Optional: force rebuild if you need a clean slate.
if [[ "${VARIANT_REBUILD_EXISTING:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(--rebuild-existing)
fi

echo "Running shard ${SHARD_INDEX}/${NUM_SHARDS} on node count ${NODE_COUNT}" 
echo "Writing artifact: ${OUTPUT_JSON}"

"$PYTHON_BIN" -m scripts.run_variant_tests \
  --output-json "$OUTPUT_JSON" \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  --hpc-node-count "$NODE_COUNT" \
  --hpc-cpus-per-job "$CPUS_PER_JOB" \
  --hpc-gpus-per-job "$GPUS_PER_JOB" \
  "${EXTRA_FLAGS[@]}"
