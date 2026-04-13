import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

from project_config import UNIFIED_KNOWLEDGE_BASE_PATH

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "by", "as", "at", "from", "that",
    "this", "it", "about", "what", "who", "when", "where", "why", "how",
    "does", "do", "did", "can", "could", "should", "would", "i", "we",
    "you", "they", "them", "their", "our", "your"
}


def load_unified_knowledge_base(path: Path = UNIFIED_KNOWLEDGE_BASE_PATH) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Unified knowledge base file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Unified knowledge base JSON must be an object.")

    chunks = data.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError("Unified knowledge base must contain a list field named 'chunks'.")

    return data


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def build_excerpt(text: str, query_terms: List[str], max_chars: int = 350) -> str:
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


def score_chunk(query_terms: List[str], title: str, text: str) -> int:
    title_tokens = tokenize(title)
    text_tokens = tokenize(text)

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


def search_unified_knowledge(query: str, max_results: int = 5) -> Dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        return {"error": "Query must be a non-empty string."}

    if not isinstance(max_results, int) or max_results <= 0:
        max_results = 5

    data = load_unified_knowledge_base()
    query_terms = tokenize(query)
    ranked_results = []

    for chunk in data.get("chunks", []):
        chunk_text = (chunk.get("text") or "").strip()
        title = (chunk.get("title") or "Untitled").strip()
        url = (chunk.get("url") or "No URL provided").strip()

        if not chunk_text:
            continue

        score = score_chunk(query_terms, title, chunk_text)
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
                "excerpt": build_excerpt(chunk_text, query_terms),
                "citation": {
                    "title": title,
                    "url": url,
                },
            }
        )

    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    top_results = ranked_results[:max_results]

    return {
        "query": query,
        "result_count": len(top_results),
        "results": top_results,
    }