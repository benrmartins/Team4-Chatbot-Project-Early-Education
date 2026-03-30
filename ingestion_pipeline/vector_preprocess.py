import json
import re
from pathlib import Path
from typing import List

from ingestion_pipeline.schema import JSONDict, build_base_payload

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


def ensure_document_char_count(documents: List[JSONDict]) -> List[JSONDict]:
    normalized_docs: List[JSONDict] = []

    for doc in documents:
        text = normalize_text(doc.get("text", ""))
        if not text:
            continue

        merged = dict(doc)
        merged["text"] = text
        merged["char_count"] = len(text)
        normalized_docs.append(merged)

    return normalized_docs


def build_chunk_records(documents: List[JSONDict], chunk_size: int = 700, chunk_overlap: int = 120) -> List[JSONDict]:
    chunks: List[JSONDict] = []

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


def build_vector_payload(
    documents: List[JSONDict],
    source: JSONDict,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> JSONDict:
    normalized_documents = ensure_document_char_count(documents)
    chunks = build_chunk_records(normalized_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    payload = build_base_payload(
        source=source,
        summary={
            "documents": len(normalized_documents),
            "chunks": len(chunks),
        },
    )
    payload["documents"] = normalized_documents
    payload["chunks"] = chunks
    return payload


def save_json(payload: JSONDict, output_path: str) -> None:
    output_file = Path(output_path)
    output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = [
    "JSONDict",
    "normalize_text",
    "chunk_text",
    "estimate_token_count",
    "ensure_document_char_count",
    "build_chunk_records",
    "build_vector_payload",
    "save_json",
]
