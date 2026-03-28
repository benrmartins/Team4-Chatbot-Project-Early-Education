import io
import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from googleapiclient.http import MediaIoBaseDownload

from google_service import Create_Service
from vector_preprocess import normalize_text

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SCHEMA_VERSION = "1.0"
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
DEFAULT_DRIVE_OUTPUT = "drive_data.json"


def download_file_bytes(service, file_id: str) -> io.BytesIO:
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    file_buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(file_buffer, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    file_buffer.seek(0)
    return file_buffer


def export_google_doc_text(service, file_id: str) -> str:
    request = service.files().export_media(fileId=file_id, mimeType="text/plain")
    file_buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(file_buffer, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    file_buffer.seek(0)
    return file_buffer.read().decode("utf-8", errors="ignore")


def extract_text_from_file(file_buffer: io.BytesIO, mime_type: str) -> Optional[str]:
    if mime_type == "text/plain":
        return file_buffer.read().decode("utf-8", errors="ignore")

    if mime_type == "text/csv":
        dataframe = pd.read_csv(file_buffer)
        return dataframe.to_string(index=False)

    if mime_type == "application/pdf":
        reader = PdfReader(file_buffer)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    if mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        document = Document(file_buffer)
        paragraphs = [p.text for p in document.paragraphs if p.text and p.text.strip()]
        return "\n".join(paragraphs)

    return None


def list_folder_items(service, folder_id: str, page_size: int = 100) -> List[Dict]:
    query = f"'{folder_id}' in parents and trashed=false"
    fields = "nextPageToken, files(id, name, mimeType, webViewLink, modifiedTime, size, parents)"

    all_files: List[Dict] = []
    next_page_token = None

    while True:
        response = (
            service.files()
            .list(
                q=query,
                pageSize=page_size,
                fields=fields,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
                corpora="allDrives",
                pageToken=next_page_token,
            )
            .execute()
        )

        all_files.extend(response.get("files", []))
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return all_files


def list_folder_files_recursive(service, root_folder_id: str, page_size: int = 100) -> List[Dict]:
    queue: List[Dict] = [{"id": root_folder_id, "path": ""}]
    seen_folders = set()
    files: List[Dict] = []

    while queue:
        current = queue.pop(0)
        folder_id = current["id"]
        folder_path = current["path"]

        if folder_id in seen_folders:
            continue
        seen_folders.add(folder_id)

        children = list_folder_items(service, folder_id, page_size=page_size)
        for item in children:
            item_name = item.get("name", "")
            item_path = f"{folder_path}/{item_name}" if folder_path else item_name

            if item.get("mimeType") == FOLDER_MIME_TYPE:
                queue.append({"id": item["id"], "path": item_path})
                continue

            item["folder_path"] = folder_path
            files.append(item)

    return files


def extract_folder_id_from_link(folder_link: str) -> Optional[str]:
    if not folder_link:
        return None

    cleaned = folder_link.strip()

    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", cleaned)
    if match:
        return match.group(1)

    parsed = urlparse(cleaned)
    query_params = parse_qs(parsed.query)
    open_id = query_params.get("id", [])
    if open_id:
        return open_id[0]

    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", cleaned):
        return cleaned

    return None


def resolve_folder_id(cli_folder_id: Optional[str]) -> str:
    if cli_folder_id:
        return cli_folder_id

    env_folder_id = (os.getenv("DRIVE_FOLDER_ID") or "").strip()
    if env_folder_id:
        return env_folder_id

    env_folder_link = (os.getenv("DRIVE_FOLDER_LINK") or "").strip()
    parsed_id = extract_folder_id_from_link(env_folder_link)
    if parsed_id:
        return parsed_id

    raise ValueError(
        "Google Drive folder not provided. Set --folder-id, DRIVE_FOLDER_ID, or DRIVE_FOLDER_LINK in .env"
    )


def build_document_record(file_meta: Dict, source_folder_id: str, text: str) -> Dict:
    return {
        "document_id": file_meta["id"],
        "source_type": "google_drive",
        "source_name": "drive_folder",
        "source_locator": source_folder_id,
        "title": file_meta.get("name", "Untitled"),
        "mime_type": file_meta.get("mimeType", ""),
        "url": file_meta.get("webViewLink", ""),
        "modified_time": file_meta.get("modifiedTime"),
        "size_bytes": int(file_meta.get("size", 0)) if file_meta.get("size") else None,
        "folder_path": file_meta.get("folder_path", ""),
        "text": text,
        "char_count": len(text),
    }


def ingest_drive_folder(service, folder_id: str) -> Dict:
    files = list_folder_files_recursive(service, folder_id)
    documents: List[Dict] = []
    indexed_files: List[Dict] = []
    skipped_files: List[Dict] = []

    for file_meta in files:
        file_id = file_meta["id"]
        name = file_meta.get("name", "Untitled")
        mime_type = file_meta.get("mimeType", "")
        print(f"Processing: {name} ({mime_type})")

        text: Optional[str] = None
        reason: Optional[str] = None

        try:
            if mime_type == "application/vnd.google-apps.document":
                text = export_google_doc_text(service, file_id)
            else:
                file_buffer = download_file_bytes(service, file_id)
                text = extract_text_from_file(file_buffer, mime_type)
                if text is None:
                    reason = "Unsupported mime type"
        except Exception as exc:
            reason = str(exc)

        if not text:
            skipped_files.append(
                {
                    "file_id": file_id,
                    "name": name,
                    "mime_type": mime_type,
                    "reason": reason or "No extractable text",
                }
            )
            continue

        normalized_text = normalize_text(text)
        if not normalized_text:
            skipped_files.append(
                {
                    "file_id": file_id,
                    "name": name,
                    "mime_type": mime_type,
                    "reason": "Text was empty after normalization",
                }
            )
            continue

        doc_record = build_document_record(file_meta, folder_id, normalized_text)
        documents.append(doc_record)
        indexed_files.append(
            {
                "id": file_id,
                "name": name,
                "mimeType": mime_type,
                "url": file_meta.get("webViewLink", ""),
                "folder_path": file_meta.get("folder_path", ""),
                "char_count": doc_record["char_count"],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "type": "google_drive_folder",
            "folder_id": folder_id,
            "recursive": True,
        },
        "summary": {
            "files_seen": len(files),
            "files_indexed": len(indexed_files),
            "documents": len(documents),
            "files_skipped": len(skipped_files),
        },
        "documents": documents,
        "skipped_files": skipped_files,
        "indexed_files": indexed_files,
    }


def ingest_google_drive(folder_id: Optional[str] = None) -> Dict:
    if os.path.exists("drive_data.json"):
        print("Found existing drive_data.json, loading from file instead of re-ingesting.")
        with open("drive_data.json", "r", encoding="utf-8") as f:
            return json.load(f)

    resolved_folder_id = resolve_folder_id(folder_id)

    service = Create_Service("credentials.json", "drive", "v3", SCOPES)
    if service is None:
        raise RuntimeError("Unable to create Google Drive service.")

    return ingest_drive_folder(service=service, folder_id=resolved_folder_id)
