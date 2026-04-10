"""Lightweight SQLite-backed vector store and embedding boilerplate.

This module provides a simple, dependency-free example of how to:
- initialize a SQLite database for storing text chunks + embeddings
- insert embeddings (batched)
- run a simple similarity search (in-memory cosine similarity)

It includes a deterministic `DummyEmbedder` for local testing and a
pluggable `BaseEmbedder` interface you can replace with a real
embedding client (OpenAI, OpenRouter, Hugging Face, etc.).

This is intended as boilerplate to adapt to your preferred vector
database or service (FAISS, Annoy, Pinecone, Weaviate, etc.).
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import hashlib
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ingestion_pipeline.schema import JSONDict


class BaseEmbedder:
    """Abstract embedder interface.

    Implementers must provide `embed(texts: List[str]) -> List[List[float]]`.
    """

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError()


class DummyEmbedder(BaseEmbedder):
    """Deterministic pseudo-embedder for local testing.

    Generates reproducible float vectors based on the SHA256 hash of the
    input text. This is intentionally lightweight and deterministic so
    tests and examples behave consistently without external APIs.
    """

    def __init__(self, dim: int = 128):
        self.dim = int(dim)

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256((text or "").encode("utf-8")).digest(), "big")
            rng = random.Random(seed)
            vec = [rng.random() for _ in range(self.dim)]
            # normalize to unit length
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out


def get_default_embedder() -> BaseEmbedder:
    """Return a sensible default embedder.

    If `OPENAI_API_KEY` is present and `openai` is importable, an OpenAI
    embedder could be used here. For now, default to `DummyEmbedder` so
    the pipeline works out of the box.
    """
    try:
        if os.environ.get("OPENAI_API_KEY"):
            # Lazy import to avoid hard dependency in environments without openai
            import openai  # type: ignore

            # If openai is installed and an API key is present, user may
            # plug a production embedder here. For now, fall back to dummy.
            return DummyEmbedder(dim=int(os.environ.get("EMBEDDING_DIM", "1536")))
    except Exception:
        pass
    return DummyEmbedder()


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def init_db(db_path: Path | str) -> None:
    db = Path(db_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            document_id TEXT,
            chunk_index INTEGER,
            text TEXT,
            metadata TEXT,
            embedding TEXT,
            created_at TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_document_id ON embeddings(document_id)")
    conn.commit()
    conn.close()


def ingest_payload_to_sqlite(
    payload: JSONDict,
    db_path: Path | str,
    embedder: Optional[BaseEmbedder] = None,
    batch_size: int = 32,
) -> int:
    """Insert chunk records from a payload into a SQLite vector DB.

    - `payload` should contain a `chunks` list matching the existing
      pipeline format.
    - `embedder` should implement `BaseEmbedder`. If omitted, a
      lightweight `DummyEmbedder` will be used.

    Returns the number of inserted rows.
    """
    if embedder is None:
        embedder = get_default_embedder()

    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    chunks = payload.get("chunks", []) or []
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.get("text", "") for c in batch]
        embeddings = embedder.embed(texts)
        now = _now_iso()
        rows: List[tuple] = []
        for chunk, emb in zip(batch, embeddings):
            chunk_id = chunk.get("chunk_id")
            document_id = chunk.get("document_id")
            chunk_index = chunk.get("chunk_index")
            text = chunk.get("text")
            metadata = chunk.get("metadata", {})
            rows.append((chunk_id, document_id, chunk_index, text, json.dumps(metadata, ensure_ascii=False), json.dumps(emb), now))

        cur.executemany(
            "INSERT OR REPLACE INTO embeddings (id, document_id, chunk_index, text, metadata, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        total += len(rows)

    conn.close()
    return total


def fetch_all_embeddings(db_path: Path | str) -> List[JSONDict]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    rows = cur.execute("SELECT id, document_id, chunk_index, text, metadata, embedding, created_at FROM embeddings").fetchall()
    conn.close()

    out: List[JSONDict] = []
    for id_, document_id, chunk_index, text, metadata_text, embedding_text, created_at in rows:
        try:
            metadata = json.loads(metadata_text) if metadata_text else {}
        except Exception:
            metadata = {}
        try:
            embedding = json.loads(embedding_text) if embedding_text else []
        except Exception:
            embedding = []
        out.append(
            {
                "id": id_,
                "document_id": document_id,
                "chunk_index": chunk_index,
                "text": text,
                "metadata": metadata,
                "embedding": embedding,
                "created_at": created_at,
            }
        )
    return out


def _cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    a_list = list(a)
    b_list = list(b)
    if not a_list or not b_list or len(a_list) != len(b_list):
        return 0.0
    dot = sum(x * y for x, y in zip(a_list, b_list))
    na = math.sqrt(sum(x * x for x in a_list))
    nb = math.sqrt(sum(x * x for x in b_list))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def query_similar_by_text(
    db_path: Path | str, query_text: str, embedder: Optional[BaseEmbedder] = None, top_k: int = 10
) -> List[JSONDict]:
    """Run a simple similarity search by embedding the query and returning top_k chunks.

    This implementation loads embeddings into memory and computes cosine
    similarity. It's intentionally simple — replace with a vector index
    (FAISS, Annoy, etc.) for production-scale workloads.
    """
    if embedder is None:
        embedder = get_default_embedder()

    q_emb = embedder.embed([query_text])[0]
    all_rows = fetch_all_embeddings(db_path)
    scored: List[tuple[float, JSONDict]] = []
    for row in all_rows:
        emb = row.get("embedding") or []
        score = _cosine_similarity(q_emb, emb)
        row_copy = dict(row)
        row_copy["score"] = score
        scored.append((score, row_copy))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [r for _, r in scored[:top_k]]


__all__ = [
    "BaseEmbedder",
    "DummyEmbedder",
    "get_default_embedder",
    "init_db",
    "ingest_payload_to_sqlite",
    "fetch_all_embeddings",
    "query_similar_by_text",
]
