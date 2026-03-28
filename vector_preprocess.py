import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

SCHEMA_VERSION = "1.0"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 120) -> List[str]:
    clean = normalize_text(text)
    if not clean:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(clean):
        chunks.append(clean[start:start + chunk_size])
        start += step

    return chunks


def estimate_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def ensure_document_char_count(documents: List[Dict]) -> List[Dict]:
    normalized_docs: List[Dict] = []

    for doc in documents:
        text = normalize_text(doc.get("text", ""))
        if not text:
            continue

        merged = dict(doc)
        merged["text"] = text
        merged["char_count"] = len(text)
        normalized_docs.append(merged)

    return normalized_docs


def build_chunk_records(documents: List[Dict], chunk_size: int = 700, chunk_overlap: int = 120) -> List[Dict]:
    chunks: List[Dict] = []

    for doc in ensure_document_char_count(documents):
        document_id = doc["document_id"]
        for idx, chunk in enumerate(chunk_text(doc["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)):
            chunks.append(
                {
                    "chunk_id": f"{document_id}::chunk-{idx:04d}",
                    "document_id": document_id,
                    "source_type": doc.get("source_type", "unknown"),
                    "title": doc.get("title", "Untitled"),
                    "url": doc.get("url", ""),
                    "chunk_index": idx,
                    "text": chunk,
                    "char_count": len(chunk),
                    "token_estimate": estimate_token_count(chunk),
                    "metadata": {
                        "mime_type": doc.get("mime_type", ""),
                        "modified_time": doc.get("modified_time"),
                        "source_locator": doc.get("source_locator", ""),
                        "folder_path": doc.get("folder_path", ""),
                        "source_name": doc.get("source_name", ""),
                    },
                }
            )

    return chunks


def build_vector_payload(documents: List[Dict], source: Dict, chunk_size: int = 700, chunk_overlap: int = 120) -> Dict:
    normalized_documents = ensure_document_char_count(documents)
    chunks = build_chunk_records(normalized_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "summary": {
            "documents": len(normalized_documents),
            "chunks": len(chunks),
        },
        "documents": normalized_documents,
        "chunks": chunks,
    }


def save_json(payload: Dict, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
