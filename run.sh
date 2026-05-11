#!/usr/bin/env bash
set -euo pipefail

# Unified setup + run script: combines setup.sh, config, and app startup in one workflow.
# Usage: ./run.sh [--no-run]
#   --no-run : perform setup and initialization only, do not start app

NO_RUN=0
if [[ "${1:-}" == "--no-run" ]]; then
  NO_RUN=1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Detect OS and set venv paths accordingly
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
  # Windows (Git Bash, WSL, or Cygwin)
  VENV_PYTHON=".venv/Scripts/python.exe"
else
  # Unix-like (Linux, macOS)
  VENV_PYTHON=".venv/bin/python"
fi

# Ensure virtual environment exists; run setup.sh if needed
if ! "$VENV_PYTHON" --version >/dev/null 2>&1; then
  echo "Virtual environment not found or not working. Running setup.sh..."
  bash setup.sh
else
  echo "Virtual environment found and working at $VENV_PYTHON"
fi

VENV_PY="$VENV_PYTHON"

echo ""
echo "========================================="
echo "Chatbot Setup & Configuration"
echo "========================================="
echo ""
echo "This will prepare the environment and start the Flask app."
read -p "Admin username (default: admin): " ADMIN_USER
ADMIN_USER=${ADMIN_USER:-admin}
read -s -p "Admin password (default: password): " ADMIN_PASS
echo
ADMIN_PASS=${ADMIN_PASS:-password}
read -p "OpenRouter API key: " OPENROUTER_API_KEY
echo
OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}

export FLASK_ENV=development

# Set up directories
LOGS_DIR="$ROOT_DIR/logs"

echo "Ensuring project directories exist..."
mkdir -p "$LOGS_DIR"

# Create minimal default files so the app can start with an empty log dir.
[[ ! -f "$LOGS_DIR/cost_events.jsonl" ]] && touch "$LOGS_DIR/cost_events.jsonl"

echo "Created/verified:"
echo " - $LOGS_DIR/cost_events.jsonl"

# Create or update .env with OpenRouter API key
ENV_FILE="$ROOT_DIR/.env"
if [[ -n "${OPENROUTER_API_KEY}" ]]; then
  printf 'OPENROUTER_API_KEY="%s"\n' "$OPENROUTER_API_KEY" > "$ENV_FILE"
  printf 'ADMIN_ACCOUNT="user:[%s]|||password:[%s]"\n' "$ADMIN_USER" "$ADMIN_PASS" >> "$ENV_FILE"
  echo "Wrote API key to $ENV_FILE"
  export OPENROUTER_API_KEY
else
  # If .env doesn't exist, create a template
  if [[ ! -f "$ENV_FILE" ]]; then
    printf 'OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"\n' > "$ENV_FILE"
    printf 'ADMIN_ACCOUNT="user:[your-username]|||password:[your-password]"\n' >> "$ENV_FILE"
    echo "Created .env template at $ENV_FILE"
  fi
fi

echo ""
if [[ "$NO_RUN" -eq 1 ]]; then
  echo "✓ Setup complete. Environment ready for app start."
  echo "  To run the app, execute: bash run.sh"
  exit 0
fi

echo "Starting app using: $VENV_PY app.py"
echo "App will be available at: http://0.0.0.0:8000"
echo ""
exec "$VENV_PY" app.py
