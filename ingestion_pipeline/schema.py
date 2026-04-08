from datetime import datetime, timezone
from typing import Any, Dict, Iterable

JSONDict = Dict[str, Any]

SCHEMA_VERSION = "1.0"
SOURCE_GOOGLE_DRIVE = "drive"
SOURCE_WEBSITE = "website"

DOCUMENT_REQUIRED_FIELDS = (
	"document_id",
	"source_type",
	"title",
	"mime_type",
	"url",
	"text",
	"char_count",
)

CHUNK_REQUIRED_FIELDS = (
	"chunk_id",
	"document_id",
	"source_type",
	"title",
	"url",
	"chunk_index",
	"text",
	"char_count",
	"token_estimate",
	"metadata",
)

CHUNK_METADATA_FIELDS = (
	"mime_type",
	"modified_time",
	"source_locator",
	"folder_path",
	"source_name",
	"section_hint",
)


def utc_now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def build_base_payload(source: JSONDict, summary: JSONDict) -> JSONDict:
	return {
		"schema_version": SCHEMA_VERSION,
		"generated_at_utc": utc_now_iso(),
		"source": source,
		"summary": summary,
	}


def _missing_required_fields(record: JSONDict, required_fields: Iterable[str]) -> list[str]:
	return [field for field in required_fields if field not in record]


def validate_document_record(document: JSONDict) -> None:
	missing = _missing_required_fields(document, DOCUMENT_REQUIRED_FIELDS)
	if missing:
		raise ValueError(f"Document missing required fields: {', '.join(missing)}")


def validate_chunk_record(chunk: JSONDict) -> None:
	missing = _missing_required_fields(chunk, CHUNK_REQUIRED_FIELDS)
	if missing:
		raise ValueError(f"Chunk missing required fields: {', '.join(missing)}")


__all__ = [
	"JSONDict",
	"SCHEMA_VERSION",
	"SOURCE_GOOGLE_DRIVE",
	"SOURCE_WEBSITE",
	"DOCUMENT_REQUIRED_FIELDS",
	"CHUNK_REQUIRED_FIELDS",
	"CHUNK_METADATA_FIELDS",
	"utc_now_iso",
	"build_base_payload",
	"validate_document_record",
	"validate_chunk_record",
]
