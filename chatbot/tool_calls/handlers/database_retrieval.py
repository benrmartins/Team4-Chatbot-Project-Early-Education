import re
from pathlib import Path
from typing import Any, Dict, List

from ingestion_pipeline.services.vector_store import (
	get_embedder_with_dimension,
	query_similar_by_text,
	read_db_embedding_config,
)
from project_config import RETRIEVAL_SNIPPET_MAX_CHARS

STOPWORDS = {
	"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
	"is", "are", "was", "were", "be", "by", "as", "at", "from", "that",
	"this", "it", "about", "what", "who", "when", "where", "why", "how",
	"does", "do", "did", "can", "could", "should", "would", "i", "we",
	"you", "they", "them", "their", "our", "your", "say", "anything",
	"into", "than", "then", "have", "has", "had", "will", "want", "need",
}
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9']+")


def _error_payload(
	query: str,
	database_path: str | None,
	error_code: str,
	error_message: str,
	max_results: int,
) -> Dict[str, Any]:
	return {
		"query": query,
		"normalized_query_terms": tokenize(query),
		"result_count": 0,
		"low_confidence": True,
		"results": [],
		"retrieval_status": "failed",
		"error_code": error_code,
		"error_message": error_message,
		"database_path": database_path or "",
		"max_results": max_results,
	}


def tokenize(text: str) -> List[str]:
	if not text:
		return []
	tokens: List[str] = []
	for raw_token in TOKEN_PATTERN.findall(text.lower()):
		token = raw_token.strip("'")
		if token and token not in STOPWORDS and len(token) > 1:
			tokens.append(token)
	return tokens


def build_excerpt(text: str, query_terms: List[str], max_chars: int = RETRIEVAL_SNIPPET_MAX_CHARS) -> str:
	normalized = re.sub(r"\s+", " ", (text or "")).strip()
	if not normalized:
		return ""

	lowered = normalized.lower()
	for term in query_terms:
		pos = lowered.find(term)
		if pos >= 0:
			start = max(0, pos - max_chars // 3)
			end = min(len(normalized), start + max_chars)
			excerpt = normalized[start:end].strip()
			if start > 0:
				excerpt = "..." + excerpt
			if end < len(normalized):
				excerpt = excerpt + "..."
			return excerpt

	if len(normalized) <= max_chars:
		return normalized
	return normalized[: max_chars - 3].rstrip() + "..."


def _coerce_metadata(raw_metadata: Any) -> Dict[str, Any]:
	if isinstance(raw_metadata, dict):
		return raw_metadata
	return {}


def _pick_title(row: Dict[str, Any], metadata: Dict[str, Any]) -> str:
	return str(
		row.get("title")
		or metadata.get("title")
		or metadata.get("source_name")
		or metadata.get("section_hint")
		or row.get("document_id")
		or "Untitled"
	).strip() or "Untitled"


def _pick_url(row: Dict[str, Any], metadata: Dict[str, Any]) -> str:
	return str(
		row.get("url")
		or metadata.get("url")
		or metadata.get("source_locator")
		or ""
	).strip()


def _pick_source_type(row: Dict[str, Any], metadata: Dict[str, Any]) -> str:
	return str(
		row.get("source_type")
		or metadata.get("source_type")
		or "unknown"
	).strip() or "unknown"


def _build_result_item(row: Dict[str, Any], query_terms: List[str], rank: int) -> Dict[str, Any]:
	metadata = _coerce_metadata(row.get("metadata"))
	text = (row.get("text") or "").strip()
	title = _pick_title(row, metadata)
	url = _pick_url(row, metadata)
	source_type = _pick_source_type(row, metadata)

	row_tokens = set(tokenize(text))
	matched_terms = [term for term in query_terms if term in row_tokens]
	coverage = len(matched_terms) / max(1, len(query_terms))
	score = float(row.get("score") or 0.0)
	evidence_snippet = build_excerpt(text, query_terms=query_terms)

	reasons: List[str] = []
	if matched_terms:
		reasons.append("body keyword match")
	if coverage >= 0.75:
		reasons.append("high term coverage")
	elif coverage >= 0.34:
		reasons.append("partial term coverage")
	if score >= 0.5:
		reasons.append("strong semantic similarity")
	elif score >= 0.2:
		reasons.append("moderate semantic similarity")
	if not reasons:
		reasons.append("weak lexical overlap")

	return {
		"rank": rank,
		"title": title,
		"url": url,
		"source_type": source_type,
		"document_id": row.get("document_id", ""),
		"chunk_id": row.get("id", ""),
		"score": round(score, 4),
		"recall_score": round(score, 4),
		"coverage": round(coverage, 3),
		"matched_terms": matched_terms,
		"match_reasons": reasons,
		"evidence_snippet": evidence_snippet,
		"excerpt": evidence_snippet,
		"citation": {
			"title": title,
			"url": url,
		},
	}


def search_sqlite_knowledge(
	query: str,
	max_results: int = 5,
	database_path: str | None = None,
) -> Dict[str, Any]:
	if not isinstance(query, str) or not query.strip():
		return _error_payload(
			query=query if isinstance(query, str) else "",
			database_path=database_path,
			error_code="invalid_query",
			error_message="Query must be a non-empty string.",
			max_results=max_results if isinstance(max_results, int) else 5,
		)
	if not database_path:
		return _error_payload(
			query=query,
			database_path=database_path,
			error_code="missing_database_path",
			error_message="database_path is required for SQLite retrieval.",
			max_results=max_results if isinstance(max_results, int) else 5,
		)
	if not isinstance(max_results, int) or max_results <= 0:
		max_results = 5

	db_file = Path(database_path)
	if not db_file.exists():
		return _error_payload(
			query=query,
			database_path=database_path,
			error_code="database_not_found",
			error_message=f"SQLite database not found: {db_file}",
			max_results=max_results,
		)

	query_terms = tokenize(query)
	semantic_pool = max(max_results * 4, 10)
	try:
		embedding_method, detected_dim = read_db_embedding_config(db_file)
		query_embedder = get_embedder_with_dimension(
			dim=detected_dim,
			embedding_method=embedding_method,
		)
		rows = query_similar_by_text(
			db_path=db_file,
			query_text=query,
			embedder=query_embedder,
			top_k=semantic_pool,
		)
	except ValueError as exc:
		return _error_payload(
			query=query,
			database_path=database_path,
			error_code="embedding_dim_mismatch",
			error_message=str(exc),
			max_results=max_results,
		)
	except Exception as exc:
		return _error_payload(
			query=query,
			database_path=database_path,
			error_code="embedder_unavailable",
			error_message=str(exc),
			max_results=max_results,
		)

	ranked_results = [
		_build_result_item(row, query_terms=query_terms, rank=index)
		for index, row in enumerate(rows[:max_results], start=1)
	]

	top_result = ranked_results[0] if ranked_results else None
	low_confidence = top_result is None or (
		top_result.get("score", 0.0) < 0.2 and top_result.get("coverage", 0.0) < 0.34
	)

	return {
		"query": query,
		"normalized_query_terms": query_terms,
		"result_count": len(ranked_results),
		"low_confidence": low_confidence,
		"retrieval_status": "ok",
		"results": ranked_results,
	}


def search_unified_knowledge(query: str, max_results: int = 5, database_path: str | None = None) -> Dict[str, Any]:
	"""Backward-compatible tool name that now reads from SQLite."""
	return search_sqlite_knowledge(query=query, max_results=max_results, database_path=database_path)


