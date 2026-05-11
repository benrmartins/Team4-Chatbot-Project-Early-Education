#!/usr/bin/env bash
set -euo pipefail

# Small helper to start the app with interactive prompts and safe defaults.
# Usage: ./run.sh [--no-run]
#   --no-run : perform initialization (create data/logs/outputs and default files) but do not start app

NO_RUN=0
if [[ "${1:-}" == "--no-run" ]]; then
  NO_RUN=1
fi

echo "This helper will prepare the environment and start the Flask app."
read -p "Admin username (default: admin): " ADMIN_USER
ADMIN_USER=${ADMIN_USER:-admin}
read -s -p "Admin password (default: password): " ADMIN_PASS
echo
ADMIN_PASS=${ADMIN_PASS:-password}
read -p "Python command to run the app (default: py): " PY_CMD
PY_CMD=${PY_CMD:-py}
read -s -p "OpenRouter API key (leave blank to skip): " OPENROUTER_API_KEY
echo
OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}

export ADMIN_ACCOUNT="user:[${ADMIN_USER}]|||password:[${ADMIN_PASS}]"
export FLASK_ENV=development
if [[ -n "${OPENROUTER_API_KEY}" ]]; then
  # create .env file with OPENROUTER_API_KEY for the app to pick up
  ENV_FILE="$ROOT_DIR/.env"
  printf "OPENROUTER_API_KEY=%s\n" "$OPENROUTER_API_KEY" > "$ENV_FILE"
  echo "Wrote API key to $ENV_FILE"
  export OPENROUTER_API_KEY
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$ROOT_DIR/data"
LOGS_DIR="$ROOT_DIR/logs"
OUTPUTS_DIR="$ROOT_DIR/outputs"

echo "Ensuring project directories exist..."
mkdir -p "$DATA_DIR" "$LOGS_DIR" "$OUTPUTS_DIR"

# Create minimal default files so the app can start with an empty data dir.
touch "$DATA_DIR/web_data.json"
touch "$DATA_DIR/unified_chunk_data.json"
touch "$LOGS_DIR/cost_events.jsonl"

echo "Created/verified:"
echo " - $DATA_DIR/web_data.json"
echo " - $DATA_DIR/unified_chunk_data.json"
echo " - $LOGS_DIR/cost_events.jsonl"

if [[ "$NO_RUN" -eq 1 ]]; then
  echo "Initialization complete (no run). Exiting."
  exit 0
fi

echo "Starting app using command: $PY_CMD app.py"
exec "$PY_CMD" app.py
