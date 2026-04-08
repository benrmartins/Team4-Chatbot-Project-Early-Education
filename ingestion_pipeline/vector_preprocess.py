import json
import re
from pathlib import Path
from typing import List

from ingestion_pipeline.schema import JSONDict, build_base_payload
from project_config import PIPELINE_MIN_CHUNK_SIZE


def normalize_text(text: str) -> str:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]

    blocks: List[str] = []
    current_lines: List[str] = []
    for line in lines:
        if line:
            current_lines.append(line)
            continue
        if current_lines:
            blocks.append(" ".join(current_lines))
            current_lines = []

    if current_lines:
        blocks.append(" ".join(current_lines))

    return "\n\n".join(blocks).strip()


def is_heading(block: str) -> bool:
    cleaned = block.strip()
    if not cleaned:
        return False
    words = cleaned.split()
    if len(words) > 8 or len(cleaned) > 90:
        return False
    if cleaned.endswith((".", "!", "?")):
        return False
    title_case_words = sum(1 for word in words if word[:1].isupper())
    return title_case_words >= max(1, len(words) - 1)


def split_sections(text: str) -> List[dict[str, str]]:
    blocks = [block.strip() for block in normalize_text(text).split("\n\n") if block.strip()]
    if not blocks:
        return []

    sections: List[dict[str, str]] = []
    current_heading = ""
    current_paragraphs: List[str] = []

    for block in blocks:
        if is_heading(block):
            if current_paragraphs:
                sections.append(
                    {
                        "section_hint": current_heading,
                        "text": "\n\n".join(current_paragraphs),
                    }
                )
                current_paragraphs = []
            current_heading = block
            continue
        current_paragraphs.append(block)

    if current_paragraphs:
        sections.append(
            {
                "section_hint": current_heading,
                "text": "\n\n".join(current_paragraphs),
            }
        )

    return sections or [{"section_hint": "", "text": normalize_text(text)}]


def split_sentences(text: str) -> List[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    sentences: List[str] = []
    for paragraph in paragraphs:
        parts = re.split(r"(?<=[.!?])\s+", paragraph)
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                sentences.append(cleaned)
    return sentences or [text.strip()]


def chunk_text_with_metadata(
    text: str,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    min_chunk_size: int = PIPELINE_MIN_CHUNK_SIZE,
) -> List[dict[str, str]]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    sections = split_sections(text)
    if not sections:
        return []

    chunks: List[dict[str, str]] = []
    for section in sections:
        section_hint = section["section_hint"]
        sentences = split_sentences(section["text"])
        start = 0

        while start < len(sentences):
            end = start
            current_sentences: List[str] = []
            current_length = len(section_hint) + 2 if section_hint else 0

            while end < len(sentences):
                sentence = sentences[end]
                sentence_length = len(sentence) + 1
                would_exceed = current_sentences and current_length + sentence_length > chunk_size
                if would_exceed and current_length >= min_chunk_size:
                    break
                current_sentences.append(sentence)
                current_length += sentence_length
                end += 1

            if not current_sentences:
                current_sentences.append(sentences[end])
                end += 1

            chunk_body = " ".join(current_sentences).strip()
            chunk_text = f"{section_hint}\n\n{chunk_body}".strip() if section_hint else chunk_body
            chunks.append(
                {
                    "text": chunk_text,
                    "section_hint": section_hint,
                }
            )

            if end >= len(sentences):
                break

            next_start = end
            carried_chars = 0
            while next_start - 1 > start and carried_chars < chunk_overlap:
                next_start -= 1
                carried_chars += len(sentences[next_start]) + 1

            start = next_start if next_start > start else end

    return chunks


def chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 120) -> List[str]:
    return [
        chunk["text"]
        for chunk in chunk_text_with_metadata(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    ]


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
        for idx, chunk in enumerate(
            chunk_text_with_metadata(doc["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            metadata = {
                "mime_type": doc.get("mime_type", ""),
                "modified_time": doc.get("modified_time"),
                "source_locator": doc.get("source_locator", ""),
                "folder_path": doc.get("folder_path", ""),
                "source_name": doc.get("source_name", ""),
            }
            if chunk.get("section_hint"):
                metadata["section_hint"] = chunk["section_hint"]

            chunks.append(
                {
                    "chunk_id": f"{document_id}::chunk-{idx:04d}",
                    "document_id": document_id,
                    "source_type": doc.get("source_type", "unknown"),
                    "title": doc.get("title", "Untitled"),
                    "url": doc.get("url", ""),
                    "chunk_index": idx,
                    "text": chunk["text"],
                    "char_count": len(chunk["text"]),
                    "token_estimate": estimate_token_count(chunk["text"]),
                    "metadata": metadata,
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
    "chunk_text_with_metadata",
    "chunk_text",
    "estimate_token_count",
    "ensure_document_char_count",
    "build_chunk_records",
    "build_vector_payload",
    "save_json",
]
