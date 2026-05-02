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

To skip the smoke test (useful on constrained HPC login nodes):

```bash
setup.bat --skip-smoke-test
bash setup.sh --skip-smoke-test
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
- Retrieval tuning and synonym expansion

Set VS Code interpreter to `.venv`:

- Command Palette -> Python: Select Interpreter
- Choose `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (macOS/Linux)

## HPC Quickstart (SLURM)

Run these from the project root on your HPC system.

### Setup

1. Copy or clone the repo to your home/scratch directory.
2. Create environment and install dependencies:

```bash
bash setup.sh --skip-smoke-test
```

3. Set your OpenRouter key:

```bash
nano .env
# OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"
```

4. Activate the virtual environment:

```bash
source .venv/bin/activate
```
### Before you run anything

- Run all Chimera/SLURM commands from the repo root.
- Make sure the repo root is writable by your job.
- Create the expected output directories before submitting jobs:

```bash
mkdir -p data outputs outputs/hpc
```

- If you already have a known-good `data/web_data.json`, keep it in place. This avoids a fresh crawl and makes HPC runs more reproducible.
- `credentials.json` is only needed for the Google Drive crawler path. If it is missing, the Drive portion is skipped, but website/prebuilt-data runs can still proceed.

### Recommended run order

Start with the cheapest/safest run first, then scale up:

1. Dummy smoke run

```bash
sbatch run_dummy_variant_experiment.slurm
```

2. Small sharded run

```bash
sbatch run_small_variant_experiment.slurm
```

3. Full run

```bash
sbatch run_full_variant_experiment.slurm
```

### Monitor jobs and inspect logs

Use these after any submission:

```bash
squeue -u $USER
ls outputs/hpc
tail -f outputs/hpc/variant-experiment-<jobid>_<arrayid>.out
```

Each shard writes a JSON artifact to:

```text
outputs/hpc/variant_test_results_shard_<index>_of_<count>.json
```

### Merge shard outputs

After the run finishes, merge the shard artifacts into one ranked report:

```bash
python -m scripts.merge_variant_results \
  --input-glob "outputs/hpc/variant_test_results_shard_*_of_*.json" \
  --output-json "outputs/hpc/variant_test_results_merged.json"
```

### Reruns after failed or interrupted jobs

Interrupted runs can leave behind incomplete SQLite DB files. If you are retrying after a failed run, force a rebuild:

```bash
export VARIANT_REBUILD_EXISTING=1
```

Then resubmit the job:

```bash
sbatch run_dummy_variant_experiment.slurm
```

If you want to delete per-variant DBs after scoring (optional):

```bash
export VARIANT_CLEANUP_DBS=1
```

Notes:

- `run_full_variant_experiment.slurm` now runs baseline on shard `0` only.
- Per-variant SQLite DB cleanup is opt-in.
- Set `VARIANT_REBUILD_EXISTING=1` before `sbatch` to force full DB rebuild.
- Set `VARIANT_CLEANUP_DBS=1` before `sbatch` if you want DBs deleted after scoring.
- If `data/web_data.json` is missing, some runs may trigger a fresh crawl.

## Cost Tracking (Team + HPC)

Cost usage is now logged as append-only JSONL events (one line per API call) to:

```text
outputs/cost/cost_events.jsonl
```

This avoids read-modify-write race conditions across concurrent SLURM jobs and avoids git merge conflicts on mutable totals.

To summarize cost events:

```bash
python -m scripts.summarize_cost_events
```

Optional: write summary to a JSON file:

```bash
python -m scripts.summarize_cost_events \
  --output-json outputs/cost/cost_summary.json
```

After any real HPC run, review the printed summary or saved JSON before launching larger runs. This is the easiest way to sanity-check API usage and cost growth across dummy, small, and full experiment passes.

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

The web UI now shows:

- the generated answer
- citations/source links
- retrieved evidence snippets used to support the answer
- a low-confidence retrieval note when matching is weak

## Run Ingestion Pipeline

Default run (module):

```powershell
python -m ingestion_pipeline.scripts.pipeline_runner
```

Default run (convenience script):

```powershell
python -m scripts.create_default_database
```

Default pipeline behavior is config-driven from `project_config.py`:

- `PIPELINE_RUN_DRIVE`
- `PIPELINE_RUN_WEB`
- `DEFAULT_CHUNK_SIZE`
- `DEFAULT_CHUNK_OVERLAP`
- `DEFAULT_MIN_CHUNK_SIZE`
- `DEFAULT_WEB_OUTPUT`, `DEFAULT_CHUNK_OUTPUT`, `DEFAULT_VECTOR_DB_PATH`

The chunking path is now section-aware for new ingestion runs:

- preserves heading/paragraph boundaries when possible
- uses sentence-grouped chunks instead of naive fixed slicing
- stores optional `metadata.section_hint` for retrieval/ranking

## Retrieval Tools

- `search_sqlite_knowledge`: primary retrieval over SQLite vector databases (default path derived from `project_config.py`)

Phase 2 retrieval improvements include:
- fallback loading from repo-root `unified_vector_data.json` if `data/` is missing
- query normalization and lightweight synonym expansion
- stronger lexical reranking with title/section weighting
- duplicate-result suppression
- focused evidence snippet extraction
- low-confidence retrieval signaling
- metadata-aware embedding-method selection from DB metadata
- lexical + similarity evidence shaping for ranked snippets
- low-confidence and structured failure signaling (`retrieval_status`, `error_code`, `error_message`)

For a lightweight before-vs-after comparison:

```powershell
python -m scripts.run_retrieval_benchmark
```

Benchmark questions live in `evaluation/retrieval_benchmark.json`.

## Project Structure

- `app.py` - Flask web app entrypoint
- `cli.py` - CLI chatbot entrypoint
- `run_full_variant_experiment.slurm` - SLURM entrypoint for full shard-based variant experiments
- `chatbot/`
  - `chatbot_api.py` - chatbot orchestration and tool-calling loop
  - `tool_calls/`
    - `registry.py` - tool schema + handler registry
    - `handlers/database_retrieval.py` - retrieval over SQLite vector databases
- `ingestion_pipeline/`
  - `scripts/pipeline_runner.py` - unified ingestion runner
  - `DataProcessor.py` - data processing orchestration for chunking and embedding
  - `schema.py` - normalized schema contracts
  - `services/`
    - `google_service.py` - Google API auth/service setup
    - `vector_store.py` - embedding + SQLite vector store helpers
  - `webcrawlers/google_drive_crawler.py` - Google Drive ingestion
  - `webcrawlers/website_crawler.py` - website ingestion
- `project_config.py` - centralized runtime config, prompts, and source defaults
- `scripts/`
  - `create_default_database.py` - build default SQLite vector database
  - `create_variant_database.py` - build one variant SQLite database
  - `run_variant_tests.py` - run deterministic matrix benchmark and scoring
  - `merge_variant_results.py` - merge shard outputs into one ranked report
  - `run_retrieval_benchmark.py` - compare retrieval behavior on benchmark prompts
  - `run_demo_questions.py` - batch demo runner for sample questions
- `tests/` - regression and integration tests
- `evaluation/retrieval_benchmark.json` - retrieval-focused benchmark questions for demos
- `data/`
  - `db_store_default.sqlite` (created on first embed)
- `templates/index.html` - Flask chat UI

