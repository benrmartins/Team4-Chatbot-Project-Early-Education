# Team4-Chatbot-Project-Early-Education

## Project Structure

- `app.py` - Flask web app entrypoint (simple deployable frontend)
- `frontend.py` - CLI chat frontend
- `chatbot/`
	- `API.py` - chatbot orchestration and tool-calling loop
	- `prompts.py` - system/onboarding/tool reprompt templates
- `tool_calls/`
	- `registry.py` - tool schema + handler registry
	- `handlers/json_retrieval.py` - retrieval over local JSON knowledge files
- `ingestion_pipeline/`
	- `vector_preprocess.py` - shared normalize/chunk/vector payload logic
	- `services/google_service.py` - Google API auth/service setup
	- `webcrawlers/googlescrape.py` - Google Drive crawler
	- `webcrawlers/webscrapegem.py` - website crawler
- `scripts/pipeline.py` - unified ingestion pipeline runner
- `data/`
	- `early_ed_clean_data.json`
	- `unified_vector_data.json`
- `templates/index.html` - Flask chat UI template

## Unified Ingestion Architecture

Chunking/vector preprocessing is centralized in `ingestion_pipeline/vector_preprocess.py`.

Current flow:

1. `scripts/pipeline.py` runs Google Drive ingestion (raw documents).
2. `scripts/pipeline.py` runs website ingestion (raw documents).
3. `scripts/pipeline.py` merges both sources.
4. `scripts/pipeline.py` calls `ingestion_pipeline/vector_preprocess.py` for unified chunking + vector-ready payload output.

This keeps Drive and web scraping consistent and makes future source integration easier.

## Google Drive Ingestion (Raw Documents)

Drive ingestion is folder-scoped and recursive (includes subfolders). It extracts readable text and outputs normalized `documents[]` records.

### Supported File Types

- Google Docs (`application/vnd.google-apps.document`)
- Plain text (`text/plain`)
- CSV (`text/csv`)
- PDF (`application/pdf`)
- DOCX (`application/vnd.openxmlformats-officedocument.wordprocessingml.document`)

Unsupported file types are tracked in `skipped_files` in the output JSON.

## Configure Drive Folder in `.env`

```env
DRIVE_FOLDER_LINK="https://drive.google.com/drive/folders/YOUR_FOLDER_ID?usp=sharing"
# Optional alternative:
DRIVE_FOLDER_ID="YOUR_FOLDER_ID"
```

## Run Unified Pipeline

```powershell
py .\scripts\pipeline.py
```

Optional:

```powershell
py .\scripts\pipeline.py --folder-id "YOUR_GOOGLE_DRIVE_FOLDER_ID" --drive-output "data\drive_data.json" --web-output "data\web_data.json" --vector-output "data\unified_vector_data.json" --chunk-size 700 --chunk-overlap 120
```

Source toggles:

```powershell
py .\scripts\pipeline.py --skip-web
py .\scripts\pipeline.py --skip-drive
```

## Output Files

- `data/unified_vector_data.json` (primary merged + chunked vector-ready payload used by retrieval)
- `data/early_ed_clean_data.json` (legacy clean dataset used by legacy retrieval)
- Optional pipeline outputs if you pass explicit args:
	- `data/drive_data.json`
	- `data/web_data.json`

## Unified Schema

Raw ingestors (Drive/Web) output matching `documents[]` fields:

- `document_id`
- `source_type`
- `source_name`
- `source_locator`
- `title`
- `mime_type`
- `url`
- `modified_time`
- `size_bytes`
- `folder_path`
- `text`
- `char_count`

Unified vector output adds `chunks[]`:

- `chunk_id`
- `document_id`
- `source_type`
- `title`
- `url`
- `chunk_index`
- `text`
- `char_count`
- `token_estimate`
- `metadata`

## Source Files

- `scripts/pipeline.py`: Orchestrates sources + calls unified chunking.
- `ingestion_pipeline/webcrawlers/googlescrape.py`: Google Drive crawling and raw document extraction only.
- `ingestion_pipeline/webcrawlers/webscrapegem.py`: Web crawling and raw document extraction only.
- `ingestion_pipeline/vector_preprocess.py`: Shared normalize/chunk/vector payload logic.
- `ingestion_pipeline/services/google_service.py`: Google API auth and service setup.

## Chatbot Setup

Run commands from the project root.

Install dependencies:

```powershell
py -m pip install -r requirements.txt
```

Add required API key in `.env`:

```env
OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"
```

## Chatbot Run (CLI)

```powershell
py .\frontend.py
```

## Simple Deployable Frontend (Flask)

This repo includes a basic web frontend in `app.py` and `templates/index.html`.

Run it locally:

```powershell
py .\app.py
```

Then open:

- `http://localhost:8000`

Health endpoint:

- `http://localhost:8000/health`

## Retrieval Tools Used By Chatbot

- `search_unified_knowledge`: Primary retrieval over `data/unified_vector_data.json` chunks for RAG answers and citations.
- `search_knowledge_base`: Legacy fallback retrieval over `data/early_ed_clean_data.json`.
