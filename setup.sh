#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f requirements.txt ]]; then
  echo "ERROR: requirements.txt not found. Run this script from the project root."
  exit 1
fi

RECREATE=0
SKIP_SMOKE_TEST=0
INIT_DATA_DIRS=1

is_supported_python() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
sys.exit(0 if sys.version_info >= (3, 9) else 1)
PY
}

find_python() {
  local candidate
  for candidate in python3.13 python3.12 python3.11 python3.10 python3.9 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && is_supported_python "$candidate"; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

for arg in "$@"; do
  case "$arg" in
    --recreate)
      RECREATE=1
      ;;
    --skip-smoke-test)
      SKIP_SMOKE_TEST=1
      ;;
    --skip-data-init)
      INIT_DATA_DIRS=0
      ;;
    *)
      echo "ERROR: Unknown argument '$arg'"
      echo "Usage: bash setup.sh [--recreate] [--skip-smoke-test] [--skip-data-init]"
      exit 1
      ;;
  esac
done

if [[ "$RECREATE" -eq 1 && -d .venv ]]; then
  echo "Removing existing .venv..."
  rm -rf .venv
fi

PYTHON_BIN="$(find_python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: No supported Python found on PATH. Python 3.9+ is required."
  exit 1
fi

if [[ -x .venv/bin/python ]] && ! is_supported_python .venv/bin/python; then
  echo "Existing .venv uses unsupported Python: $(.venv/bin/python -V 2>&1)"
  echo "Recreating .venv with: $PYTHON_BIN"
  rm -rf .venv
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "Creating virtual environment with: $PYTHON_BIN"
  "$PYTHON_BIN" -m venv .venv
fi

VENV_PY=".venv/bin/python"

echo "Upgrading pip..."
"$VENV_PY" -m pip install --upgrade pip

echo "Installing requirements..."
"$VENV_PY" -m pip install -r requirements.txt

if [[ ! -f .env ]]; then
  printf 'OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"\n' > .env
  echo "Created .env template. Update OPENROUTER_API_KEY before running the chatbot."
fi

if [[ "$SKIP_SMOKE_TEST" -eq 0 ]]; then
  echo "Running dependency smoke test..."
  "$VENV_PY" -c "import requests, bs4, urllib3, pandas, PyPDF2, docx, googleapiclient.http; print('Setup check: ok')"
else
  echo "Skipping dependency smoke test (--skip-smoke-test)."
fi

source .venv/bin/activate

if [[ "$INIT_DATA_DIRS" -eq 1 ]]; then
  echo "Initializing data directories..."
  "$VENV_PY" -m scripts.create_default_database
else
  echo "Skipping data directory initialization (INIT_DATA_DIRS=0)."
fi

echo
echo "Setup complete."
echo "Activate the virtual environment with: source .venv/bin/activate"
echo "Run CLI chatbot with: python cli.py"
echo "Run Flask app with: python app.py"
echo "Run HPC variant experiment with: sbatch run_full_variant_experiment.slurm"
