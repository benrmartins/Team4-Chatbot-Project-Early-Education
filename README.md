# Team4 Chatbot Project

This project provides a configurable chatbot and ingestion pipeline that can be reused across different data sources.

## Setup (Recommended)

Run commands from the project root.

### Prerequisites

- Python 3.10+ installed (3.13 tested)
- `pip` available

### One-command setup

Windows (PowerShell or CMD):

```bat
setup.bat
```

macOS/Linux/Git Bash:

```bash
bash setup.sh
```

To force a clean re-create of `.venv`:

```bash
setup.bat --recreate
bash setup.sh --recreate
```

These scripts will:

- create (or reuse) `.venv`
- install/upgrade dependencies from `requirements.txt`
- create `.env` template if missing
- run a dependency smoke test

### After setup

Activate virtual environment:

Windows CMD:

```bat
call .venv\Scripts\activate.bat
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux/Git Bash:

```bash
source .venv/bin/activate
```

Set your OpenRouter key in `.env`:

```env
OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"
```

Non-sensitive runtime defaults live in `project_config.py`, including:

- OpenRouter base URL/model
- Chatbot name, prompts, and behavior text
- Drive folder defaults
- Website seed URLs
- Pipeline toggles and chunk settings

Set VS Code interpreter to `.venv`:

- Command Palette -> Python: Select Interpreter
- Choose `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (macOS/Linux)

## Run Chatbot

CLI:

```powershell
python .\cli.py
```

Flask app:

```powershell
python .\app.py
```

Then open `http://localhost:8000`.

## Run Ingestion Pipeline

Default run (module):

```powershell
python -m ingestion_pipeline.scripts.pipeline_runner
```

Default run (convenience script):

```powershell
python .\collect_data.py
```

All pipeline behavior is config-driven from `project_config.py`:

- `PIPELINE_RUN_DRIVE`
- `PIPELINE_RUN_WEB`
- `PIPELINE_CHUNK_SIZE`
- `PIPELINE_CHUNK_OVERLAP`
- `DEFAULT_DRIVE_OUTPUT`, `DEFAULT_WEB_OUTPUT`, `DEFAULT_VECTOR_OUTPUT`

## Retrieval Tools

- `search_unified_knowledge`: primary retrieval over `data/unified_vector_data.json` chunks

## Project Structure

- `app.py` - Flask web app entrypoint
- `cli.py` - CLI chatbot entrypoint
- `collect_data.py` - convenience ingestion entrypoint
- `chatbot/`
  - `chatbot_api.py` - chatbot orchestration and tool-calling loop
  - `tool_calls/`
    - `registry.py` - tool schema + handler registry
    - `handlers/json_retrieval.py` - retrieval over local JSON knowledge files
- `ingestion_pipeline/`
  - `scripts/pipeline_runner.py` - unified ingestion runner
  - `vector_preprocess.py` - shared normalize/chunk/payload logic
  - `schema.py` - normalized schema contracts
  - `services/google_service.py` - Google API auth/service setup
  - `webcrawlers/google_drive_crawler.py` - Google Drive ingestion
  - `webcrawlers/website_crawler.py` - website ingestion
- `project_config.py` - centralized runtime config, prompts, and source defaults
- `data/`
  - `unified_vector_data.json`
- `templates/index.html` - Flask chat UI

