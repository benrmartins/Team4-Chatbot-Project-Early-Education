# json_retrieval.py
# This module provides functions to search the knowledge base JSON files for relevant information based on a user
# query. It includes functions to load the knowledge base, tokenize text, score relevance, and build excerpts for results.
import json
import re
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_BASE_PATH = DATA_DIR / "early_ed_clean_data.json"
UNIFIED_KNOWLEDGE_BASE_PATH = DATA_DIR / "unified_vector_data.json"

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "by", "as", "at", "from", "that",
    "this", "it", "about", "what", "who", "when", "where", "why", "how",
    "does", "do", "did", "can", "could", "should", "would", "i", "we",
    "you", "they", "them", "their", "our", "your"
}


def load_knowledge_base(path: Path = KNOWLEDGE_BASE_PATH) -> List[Dict[str, Any]]:
    # Loads the knowledge base JSON file and returns its content as a list of records.
    # Raises FileNotFoundError if the file does not exist, or ValueError if the JSON is not a list.
    # Loads the knowledge base JSON file and returns its content as a list of records.
    # Raises FileNotFoundError if the file does not exist, or ValueError if the JSON is not a list.
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Knowledge base JSON must be a list of records.")

    return data


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
    # Tokenizes the input text into lowercase words, removing stopwords and single-character words.
    # Returns a list of filtered tokens.
    # Tokenizes the input text into lowercase words, removing stopwords and single-character words.
    # Returns a list of filtered tokens.
    if not text:
        return []
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def score_document(query_terms: List[str], title: str, text: str) -> int:
    # Scores a document based on the presence and frequency of query terms in the title and text.
    # Gives higher weight to matches in the title and for full query matches.
    # Returns an integer score.
    # Scores a document based on the presence and frequency of query terms in the title and text.
    # Gives higher weight to matches in the title and for full query matches.
    # Returns an integer score.
    title_tokens = tokenize(title)
    text_tokens = tokenize(text)

    title_counts = Counter(title_tokens)
    text_counts = Counter(text_tokens)

    score = 0
    for term in query_terms:
        score += title_counts.get(term, 0) * 5
        score += text_counts.get(term, 0)

    full_query = " ".join(query_terms).strip()
    if full_query:
        title_lower = title.lower()
        text_lower = text.lower()

        if full_query in title_lower:
            score += 20
        if full_query in text_lower:
            score += 10

    return score


def build_excerpt(text: str, query_terms: List[str], max_chars: int = 350) -> str:
    # Builds a short excerpt from the text, centered around the first occurrence of any query term.
    # If no query term is found, returns the start of the text up to max_chars.
    # Builds a short excerpt from the text, centered around the first occurrence of any query term.
    # If no query term is found, returns the start of the text up to max_chars.
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


def search_knowledge_base(query: str, max_results: int = 5) -> Dict[str, Any]:
    # Searches the knowledge base for records relevant to the query.
    # Returns a dictionary with the query, result count, and a list of top results with excerpts.
    # Searches the knowledge base for records relevant to the query.
    # Returns a dictionary with the query, result count, and a list of top results with excerpts.
    if not isinstance(query, str) or not query.strip():
        return {"error": "Query must be a non-empty string."}

    if not isinstance(max_results, int) or max_results <= 0:
        max_results = 5

    data = load_knowledge_base()
    query_terms = tokenize(query)

    ranked_results = []
    for item in data:
        title = item.get("title", "").strip()
        url = item.get("url", "").strip()
        text = item.get("text", "").strip()

        if not text:
            continue

        score = score_document(query_terms, title, text)
        if score <= 0:
            continue

        ranked_results.append({
            "title": title or "Untitled",
            "url": url or "No URL provided",
            "score": score,
            "excerpt": build_excerpt(text, query_terms),
        })

    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    top_results = ranked_results[:max_results]

    return {
        "query": query,
        "result_count": len(top_results),
        "results": top_results,
    }


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


if __name__ == "__main__":
    test_query = "What is Leading for Change?"
    result = search_knowledge_base(test_query, max_results=3)
    print(json.dumps(result, indent=2))