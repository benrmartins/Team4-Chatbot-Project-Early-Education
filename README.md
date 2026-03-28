# Team4-Chatbot-Project-Early-Education

## Unified Ingestion Architecture

Chunking/vector preprocessing has been moved out of source-specific scrapers and centralized in `vector_preprocess.py`.

Current flow:

1. `pipeline.py` runs Google Drive ingestion (raw documents).
2. `pipeline.py` runs website ingestion (raw documents).
3. `pipeline.py` merges both sources.
4. `pipeline.py` calls `vector_preprocess.py` for unified chunking + vector-ready payload output.

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
py .\pipeline.py
```

Optional:

```powershell
py .\pipeline.py --folder-id "YOUR_GOOGLE_DRIVE_FOLDER_ID" --drive-output "drive_data.json" --web-output "web_data.json" --vector-output "unified_vector_data.json" --chunk-size 700 --chunk-overlap 120
```

Source toggles:

```powershell
py .\pipeline.py --skip-web
py .\pipeline.py --skip-drive
```

## Output Files

- `drive_data.json` (Drive raw normalized documents)
- `web_data.json` (Web raw normalized documents)
- `unified_vector_data.json` (merged + chunked vector-ready payload)
- `early_ed_clean_data.json` (legacy raw list retained for compatibility)

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

- `pipeline.py`: Orchestrates sources + calls unified chunking.
- `googlescrape.py`: Google Drive crawling and raw document extraction only.
- `webscrapegem.py`: Web crawling and raw document extraction only.
- `vector_preprocess.py`: Shared normalize/chunk/vector payload logic.
- `google_service.py`: Google API auth and service setup.

## Chatbot Setup

Run commands from the project root.

Install dependencies:

```powershell
py -m pip install -r requirments.txt
```

Add required API key in `.env`:

```env
OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"
```

## Chatbot Run

```powershell
py .\opAI.py
```

## Retrieval Tools Used By Chatbot

- `search_unified_knowledge`: Primary retrieval over `unified_vector_data.json` chunks for RAG answers and citations.
- `search_knowledge_base`: Legacy fallback retrieval over `early_ed_clean_data.json`.
