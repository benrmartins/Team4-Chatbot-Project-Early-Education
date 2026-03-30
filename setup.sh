#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f requirements.txt ]]; then
  echo "ERROR: requirements.txt not found. Run this script from the project root."
  exit 1
fi

RECREATE=0
if [[ "${1:-}" == "--recreate" ]]; then
  RECREATE=1
fi

if [[ "$RECREATE" -eq 1 && -d .venv ]]; then
  echo "Removing existing .venv..."
  rm -rf .venv
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "Creating virtual environment..."
  if command -v python3 >/dev/null 2>&1; then
    python3 -m venv .venv
  elif command -v python >/dev/null 2>&1; then
    python -m venv .venv
  else
    echo "ERROR: Python is not installed or not on PATH."
    exit 1
  fi
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

echo "Running dependency smoke test..."
"$VENV_PY" -c "import requests, bs4, urllib3, pandas, PyPDF2, docx, googleapiclient.http; print('Setup check: ok')"

echo
echo "Setup complete."
echo "Activate environment with: source .venv/bin/activate"
echo "Run CLI chatbot with: python cli.py"
echo "Run Flask app with: python app.py"
