import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

from project_config import (
    RETRIEVAL_CANDIDATE_POOL,
    RETRIEVAL_LOW_CONFIDENCE_MIN_COVERAGE,
    RETRIEVAL_LOW_CONFIDENCE_MIN_SCORE,
    RETRIEVAL_MAX_RESULTS_PER_DOCUMENT,
    RETRIEVAL_SNIPPET_MAX_CHARS,
    RETRIEVAL_SYNONYM_MAP,
    UNIFIED_KNOWLEDGE_BASE_FALLBACK_PATH,
    UNIFIED_KNOWLEDGE_BASE_PATH,
)

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "by", "as", "at", "from", "that",
    "this", "it", "about", "what", "who", "when", "where", "why", "how",
    "does", "do", "did", "can", "could", "should", "would", "i", "we",
    "you", "they", "them", "their", "our", "your", "say", "anything",
    "into", "than", "then", "have", "has", "had", "will", "want", "need",
}
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9']+")
QUOTE_PATTERN = re.compile(r'"([^"]+)"')
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def resolve_knowledge_base_path(
    preferred_path: Path = UNIFIED_KNOWLEDGE_BASE_PATH,
    fallback_path: Path = UNIFIED_KNOWLEDGE_BASE_FALLBACK_PATH,
) -> Path:
    if preferred_path.exists():
        return preferred_path
    if fallback_path.exists():
        return fallback_path
    raise FileNotFoundError(
        f"Unified knowledge base file not found. Tried: {preferred_path} and {fallback_path}"
    )


@lru_cache(maxsize=4)
def _load_cached_knowledge_base(path_str: str, mtime_ns: int) -> Dict[str, Any]:
    with open(path_str, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Unified knowledge base JSON must be an object.")

    chunks = data.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError("Unified knowledge base must contain a list field named 'chunks'.")

    return data


def load_unified_knowledge_base(path: Path | None = None) -> Dict[str, Any]:
    resolved_path = Path(path) if path else resolve_knowledge_base_path()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Unified knowledge base file not found: {resolved_path}")
    return _load_cached_knowledge_base(str(resolved_path), resolved_path.stat().st_mtime_ns)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_token(token: str) -> str:
    normalized = token.lower().strip("'")
    if len(normalized) <= 1:
        return ""
    if normalized.endswith("ies") and len(normalized) > 4:
        normalized = normalized[:-3] + "y"
    elif normalized.endswith("ing") and len(normalized) > 5:
        normalized = normalized[:-3]
    elif normalized.endswith("ed") and len(normalized) > 4:
        normalized = normalized[:-2]
    elif normalized.endswith("es") and len(normalized) > 4:
        normalized = normalized[:-2]
    elif normalized.endswith("s") and len(normalized) > 3 and not normalized.endswith("ss"):
        normalized = normalized[:-1]
    return normalized


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens: List[str] = []
    for raw_token in TOKEN_PATTERN.findall(text.lower()):
        normalized = normalize_token(raw_token)
        if normalized and normalized not in STOPWORDS:
            tokens.append(normalized)
    return tokens


def tokenize_phase1(text: str) -> List[str]:
    if not text:
        return []
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return [word for word in words if word not in STOPWORDS and len(word) > 1]


def extract_query_phrases(query: str) -> List[str]:
    phrases = [normalize_whitespace(match).lower() for match in QUOTE_PATTERN.findall(query)]
    raw_terms = [token for token in TOKEN_PATTERN.findall(query.lower()) if token not in STOPWORDS]

    if len(raw_terms) >= 2:
        phrases.append(" ".join(raw_terms[: min(4, len(raw_terms))]))
        for size in (2, 3):
            for idx in range(0, max(0, len(raw_terms) - size + 1)):
                phrases.append(" ".join(raw_terms[idx : idx + size]))

    unique_phrases: List[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        cleaned = normalize_whitespace(phrase)
        if len(cleaned) < 4 or cleaned in seen:
            continue
        seen.add(cleaned)
        unique_phrases.append(cleaned)
    return unique_phrases


def expand_query_term_groups(query_terms: Iterable[str]) -> Dict[str, set[str]]:
    term_groups: Dict[str, set[str]] = {}
    for term in query_terms:
        expansions = {term}
        for synonym in RETRIEVAL_SYNONYM_MAP.get(term, []):
            expansions.update(tokenize(synonym))
        term_groups[term] = expansions
    return term_groups


def build_phase1_excerpt(text: str, query_terms: List[str], max_chars: int = 350) -> str:
    clean_text = re.sub(r"\s+", " ", text).strip()
    if not clean_text:
        return ""

    lowered = clean_text.lower()

    for term in query_terms:
        idx = lowered.find(term.lower())
        if idx != -1:
            start = max(0, idx - 120)
            end = min(len(clean_text), idx + 220)
            snippet = clean_text[start:end].strip()

            if start > 0:
                snippet = "..." + snippet
            if end < len(clean_text):
                snippet = snippet + "..."

            return snippet[:max_chars]

    if len(clean_text) <= max_chars:
        return clean_text
    return clean_text[:max_chars] + "..."


def score_chunk_phase1(query_terms: List[str], title: str, text: str) -> int:
    title_tokens = tokenize_phase1(title)
    text_tokens = tokenize_phase1(text)

    title_counts = Counter(title_tokens)
    text_counts = Counter(text_tokens)

    score = 0
    for term in query_terms:
        score += title_counts.get(term, 0) * 4
        score += text_counts.get(term, 0)

    full_query = " ".join(query_terms).strip()
    if full_query:
        title_lower = title.lower()
        text_lower = text.lower()

        if full_query in title_lower:
            score += 15
        if full_query in text_lower:
            score += 12

    return score


def build_corpus_stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    doc_frequency: Counter[str] = Counter()

    for chunk in chunks:
        section_hint = ((chunk.get("metadata") or {}).get("section_hint") or "").strip()
        unique_tokens = set(tokenize(chunk.get("title", "")))
        unique_tokens.update(tokenize(section_hint))
        unique_tokens.update(tokenize(chunk.get("text", "")))
        doc_frequency.update(unique_tokens)

    return {
        "doc_count": max(1, len(chunks)),
        "doc_frequency": doc_frequency,
    }


def get_corpus_stats(data: Dict[str, Any]) -> Dict[str, Any]:
    cached = data.get("_cached_corpus_stats")
    if cached is None:
        cached = build_corpus_stats(data.get("chunks", []))
        data["_cached_corpus_stats"] = cached
    return cached


def compute_idf(token: str, corpus_stats: Dict[str, Any]) -> float:
    doc_count = corpus_stats["doc_count"]
    df = corpus_stats["doc_frequency"].get(token, 0)
    return math.log((1 + doc_count) / (1 + df)) + 1


def split_sentences(text: str) -> List[str]:
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    sentences = [segment.strip() for segment in SENTENCE_SPLIT_PATTERN.split(normalized) if segment.strip()]
    return sentences or [normalized]


def build_evidence_snippet(
    text: str,
    query_terms: List[str],
    term_groups: Dict[str, set[str]],
    phrases: List[str],
    max_chars: int = RETRIEVAL_SNIPPET_MAX_CHARS,
) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""

    best_score = -1.0
    best_index = 0
    for index, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        sentence_tokens = set(tokenize(sentence))
        score = 0.0

        for original_term, expansions in term_groups.items():
            if sentence_tokens.intersection(expansions):
                score += 3.0 if original_term in query_terms else 2.0
        for phrase in phrases:
            if phrase in sentence_lower:
                score += 4.0
        if score > best_score:
            best_score = score
            best_index = index

    snippet_parts = [sentences[best_index]]
    if len(snippet_parts[0]) < max_chars // 2 and best_index + 1 < len(sentences):
        snippet_parts.append(sentences[best_index + 1])
    snippet = " ".join(snippet_parts).strip()

    if len(snippet) <= max_chars:
        return snippet
    return snippet[: max_chars - 3].rstrip() + "..."


def build_match_reasons(
    title_hits: int,
    section_hits: int,
    body_hits: int,
    phrase_hits: int,
    coverage: float,
) -> List[str]:
    reasons: List[str] = []
    if title_hits:
        reasons.append("title match")
    if section_hits:
        reasons.append("section heading match")
    if body_hits:
        reasons.append("body keyword match")
    if phrase_hits:
        reasons.append("phrase match")
    if coverage >= 0.75:
        reasons.append("high term coverage")
    elif coverage >= 0.34:
        reasons.append("partial term coverage")
    return reasons or ["weak lexical overlap"]


def normalize_text_signature(text: str) -> str:
    return " ".join(tokenize(text))


def are_near_duplicates(candidate_a: Dict[str, Any], candidate_b: Dict[str, Any]) -> bool:
    if candidate_a["document_id"] != candidate_b["document_id"]:
        return False
    signature_a = set(normalize_text_signature(candidate_a["text"]).split())
    signature_b = set(normalize_text_signature(candidate_b["text"]).split())
    if not signature_a or not signature_b:
        return False
    overlap = len(signature_a & signature_b) / max(1, len(signature_a | signature_b))
    return overlap >= 0.8


def score_candidate(
    chunk: Dict[str, Any],
    query_terms: List[str],
    term_groups: Dict[str, set[str]],
    phrases: List[str],
    corpus_stats: Dict[str, Any],
) -> Dict[str, Any] | None:
    chunk_text = (chunk.get("text") or "").strip()
    title = (chunk.get("title") or "Untitled").strip()
    url = (chunk.get("url") or "No URL provided").strip()
    section_hint = ((chunk.get("metadata") or {}).get("section_hint") or "").strip()

    if not chunk_text:
        return None

    title_tokens = tokenize(title)
    section_tokens = tokenize(section_hint)
    body_tokens = tokenize(chunk_text)

    title_counts = Counter(title_tokens)
    section_counts = Counter(section_tokens)
    body_counts = Counter(body_tokens)
    matched_terms: List[str] = []
    matched_positions: List[int] = []
    recall_score = 0.0
    title_hits = 0
    section_hits = 0
    body_hits = 0
    rare_term_bonus = 0.0

    for original_term, expansions in term_groups.items():
        group_title_hits = sum(title_counts.get(term, 0) for term in expansions)
        group_section_hits = sum(section_counts.get(term, 0) for term in expansions)
        group_body_hits = sum(body_counts.get(term, 0) for term in expansions)

        if group_title_hits or group_section_hits or group_body_hits:
            matched_terms.append(original_term)
            title_hits += int(group_title_hits > 0)
            section_hits += int(group_section_hits > 0)
            body_hits += int(group_body_hits > 0)
            matched_positions.extend(
                idx for idx, token in enumerate(body_tokens) if token in expansions
            )
            best_idf = max((compute_idf(term, corpus_stats) for term in expansions), default=1.0)
            rare_term_bonus += best_idf
            recall_score += group_title_hits * 6.0 * best_idf
            recall_score += group_section_hits * 4.5 * best_idf
            recall_score += group_body_hits * 2.2 * best_idf

    title_lower = normalize_whitespace(title).lower()
    section_lower = normalize_whitespace(section_hint).lower()
    body_lower = normalize_whitespace(chunk_text).lower()
    phrase_hits = 0
    for phrase in phrases:
        if phrase in title_lower:
            recall_score += 9.0
            phrase_hits += 1
        if section_lower and phrase in section_lower:
            recall_score += 6.0
            phrase_hits += 1
        if phrase in body_lower:
            recall_score += 5.0
            phrase_hits += 1

    if recall_score <= 0:
        return None

    distinct_coverage = len(matched_terms) / max(1, len(term_groups))
    proximity_bonus = 0.0
    if len(matched_positions) >= 2:
        span = max(matched_positions) - min(matched_positions) + 1
        proximity_bonus = min(12.0, 20.0 * (len(set(matched_positions)) / max(1, span)))

    focus_multiplier = max(0.7, min(1.2, 260 / max(180, len(chunk_text))))
    rerank_score = (
        recall_score
        + distinct_coverage * 14.0
        + phrase_hits * 4.0
        + proximity_bonus
        + rare_term_bonus
    ) * focus_multiplier

    match_reasons = build_match_reasons(
        title_hits=title_hits,
        section_hits=section_hits,
        body_hits=body_hits,
        phrase_hits=phrase_hits,
        coverage=distinct_coverage,
    )

    evidence_snippet = build_evidence_snippet(
        chunk_text,
        query_terms=query_terms,
        term_groups=term_groups,
        phrases=phrases,
    )

    return {
        "title": title,
        "url": url,
        "source_type": chunk.get("source_type", "unknown"),
        "document_id": chunk.get("document_id", ""),
        "chunk_id": chunk.get("chunk_id", ""),
        "text": chunk_text,
        "score": round(rerank_score, 3),
        "recall_score": round(recall_score, 3),
        "coverage": round(distinct_coverage, 3),
        "matched_terms": matched_terms,
        "match_reasons": match_reasons,
        "evidence_snippet": evidence_snippet,
        "excerpt": evidence_snippet,
        "citation": {
            "title": title,
            "url": url,
        },
    }


def is_low_confidence(result: Dict[str, Any] | None) -> bool:
    if result is None:
        return True
    if result["score"] < RETRIEVAL_LOW_CONFIDENCE_MIN_SCORE:
        return True
    if result["coverage"] < RETRIEVAL_LOW_CONFIDENCE_MIN_COVERAGE:
        return True
    return False


def search_unified_knowledge_phase1(query: str, max_results: int = 5) -> Dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        return {"error": "Query must be a non-empty string."}

    if not isinstance(max_results, int) or max_results <= 0:
        max_results = 5

    data = load_unified_knowledge_base()
    query_terms = tokenize_phase1(query)
    ranked_results = []

    for chunk in data.get("chunks", []):
        chunk_text = (chunk.get("text") or "").strip()
        title = (chunk.get("title") or "Untitled").strip()
        url = (chunk.get("url") or "No URL provided").strip()

        if not chunk_text:
            continue

        score = score_chunk_phase1(query_terms, title, chunk_text)
        if score <= 0:
            continue

        ranked_results.append(
            {
                "title": title,
                "url": url,
                "source_type": chunk.get("source_type", "unknown"),
                "document_id": chunk.get("document_id", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "score": score,
                "excerpt": build_phase1_excerpt(chunk_text, query_terms),
                "citation": {
                    "title": title,
                    "url": url,
                },
            }
        )

    ranked_results.sort(key=lambda item: item["score"], reverse=True)
    return {
        "query": query,
        "result_count": min(len(ranked_results), max_results),
        "results": ranked_results[:max_results],
    }


def search_unified_knowledge(query: str, max_results: int = 5) -> Dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        return {"error": "Query must be a non-empty string."}

    if not isinstance(max_results, int) or max_results <= 0:
        max_results = 5

    data = load_unified_knowledge_base()
    query_terms = tokenize(query)
    phrases = extract_query_phrases(query)
    term_groups = expand_query_term_groups(query_terms)
    corpus_stats = get_corpus_stats(data)

    ranked_candidates: List[Dict[str, Any]] = []
    for chunk in data.get("chunks", []):
        scored = score_candidate(
            chunk=chunk,
            query_terms=query_terms,
            term_groups=term_groups,
            phrases=phrases,
            corpus_stats=corpus_stats,
        )
        if scored is not None:
            ranked_candidates.append(scored)

    ranked_candidates.sort(
        key=lambda item: (item["score"], item["coverage"], item["recall_score"]),
        reverse=True,
    )

    filtered_results: List[Dict[str, Any]] = []
    per_document_counts: Counter[str] = Counter()
    for candidate in ranked_candidates[: RETRIEVAL_CANDIDATE_POOL]:
        if per_document_counts[candidate["document_id"]] >= RETRIEVAL_MAX_RESULTS_PER_DOCUMENT:
            continue
        if any(are_near_duplicates(candidate, existing) for existing in filtered_results):
            continue
        filtered_results.append(candidate)
        per_document_counts[candidate["document_id"]] += 1
        if len(filtered_results) >= max_results:
            break

    for rank, result in enumerate(filtered_results, start=1):
        result["rank"] = rank

    top_result = filtered_results[0] if filtered_results else None
    return {
        "query": query,
        "normalized_query_terms": query_terms,
        "result_count": len(filtered_results),
        "low_confidence": is_low_confidence(top_result),
        "results": filtered_results,
    }
