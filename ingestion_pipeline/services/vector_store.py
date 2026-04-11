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
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from ingestion_pipeline.schema import JSONDict
from project_config import DEFAULT_BATCH_SIZE, OPENROUTER_BASE_URL, DEFAULT_EMBEDDING_DIM
from abc import ABC, abstractmethod

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY", None)
client = OpenAI(api_key=API_KEY, base_url=OPENROUTER_BASE_URL)

class BaseEmbedder(ABC):
    """Abstract embedder interface.

    Implementers must provide `embed(documents: List[Dict[str, str]]) -> List[List[float]]`.
    """
    @abstractmethod
    def embed(self, documents: JSONDict) -> List[List[float]]:
        pass

class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, embedding_method: str = "dummy", dim: int = 1024):
        self.embedding_method = embedding_method
        self.dim = dim

    def embed(self, documents: JSONDict) -> List[List[float]]:
        texts = [doc.get("text", "") for doc in documents.get("chunks", [])]
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small" if self.embedding_method == "openai_small" else
                  "text-embedding-3-large" if self.embedding_method == "openai_large" else
                  "text-embedding-3-small",
            dimensions=self.dim,
        )
        return [e.embedding for e in response.data]

class DummyEmbedder(BaseEmbedder):
    """Deterministic pseudo-embedder for local testing.

    Generates reproducible float vectors based on the SHA256 hash of the
    input text. This is intentionally lightweight and deterministic so
    tests and examples behave consistently without external APIs.
    """

    def __init__(self, dim: int = 128):
        self.dim = int(dim)

    def embed(self, documents: JSONDict) -> List[List[float]]:
        texts = [doc.get("text", "") for doc in documents.get("chunks", [])]
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
    """Helper to get a default embedder instance."""
    try:
        if API_KEY and client is not None:
            return OpenAIEmbedder(dim=DEFAULT_EMBEDDING_DIM)
    except Exception:
        pass
    raise Exception(f"Failed to initialize OpenAIEmbedder with dim {DEFAULT_EMBEDDING_DIM}. Check API key and client configuration.")

def get_embedder_with_dimension(dim: int, embedding_method: str = "dummy") -> BaseEmbedder:
    """Helper to get an embedder instance with a specific dimension."""
    if embedding_method == "dummy":
        return DummyEmbedder(dim=dim)
    else:
        try:
            if API_KEY and client is not None:
                return OpenAIEmbedder(embedding_method=embedding_method, dim=dim)
        except Exception:
            pass
    raise Exception(f"Failed to initialize OpenAIEmbedder with method '{embedding_method}' and dim {dim}. Check API key and client configuration.")

def _now_iso() -> str:
    return datetime.now().isoformat()

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
            title TEXT,
            url TEXT,
            source_type TEXT,
            text TEXT,
            metadata TEXT,
            embedding TEXT,
            created_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS db_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_document_id ON embeddings(document_id)")
    conn.commit()
    conn.close()


def _write_db_embedding_config(
    conn: sqlite3.Connection,
    embedding_config: Dict[str, Any],
) -> None:
    cur = conn.cursor()
    now = _now_iso()
    rows = [
        ("embedding_config", json.dumps(embedding_config)),
        ("updated_at", now),
    ]
    cur.executemany(
        "INSERT OR REPLACE INTO db_metadata (key, value) VALUES (?, ?)",
        rows,
    )


def _build_embedding_config(embedder: BaseEmbedder) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "method": str(getattr(embedder, "embedding_method", "dummy")).strip().lower() or "dummy",
        "dim": int(getattr(embedder, "dim", DEFAULT_EMBEDDING_DIM)),
        "embedder_class": embedder.__class__.__name__,
    }
    model_name = getattr(embedder, "model", None)
    if model_name:
        config["model"] = str(model_name)
    return config


def read_db_embedding_config(
    db_path: Path | str,
    default_method: str = "dummy",
    default_dim: int = DEFAULT_EMBEDDING_DIM,
) -> tuple[str, int]:
    """Resolve (embedding_method, embedding_dim) directly from DB metadata."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        table_exists = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='db_metadata'"
        ).fetchone()
        if not table_exists:
            return str(default_method or "dummy").strip().lower() or "dummy", int(default_dim)

        row = cur.execute(
            "SELECT value FROM db_metadata WHERE key='embedding_config' LIMIT 1"
        ).fetchone()
        if not row or not row[0]:
            return str(default_method or "dummy").strip().lower() or "dummy", int(default_dim)

        parsed = json.loads(row[0])
        if not isinstance(parsed, dict):
            return str(default_method or "dummy").strip().lower() or "dummy", int(default_dim)

        method = str(parsed.get("method", "")).strip().lower() or str(default_method or "dummy").strip().lower() or "dummy"
        try:
            dim = int(parsed.get("dim", default_dim))
        except Exception:
            dim = int(default_dim)
        if dim <= 0:
            dim = int(default_dim)
        return method, dim
    except Exception:
        return str(default_method or "dummy").strip().lower() or "dummy", int(default_dim)
    finally:
        conn.close()

def ingest_payload_to_sqlite(
    payload: JSONDict,
    db_path: Path | str,
    embedder: Optional[BaseEmbedder] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
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

    # Record embedder configuration for retrieval-time query embedding alignment.
    embedding_config = _build_embedding_config(embedder)
    _write_db_embedding_config(conn, embedding_config=embedding_config)

    chunks = payload.get("chunks", []) or []
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = embedder.embed({"chunks": batch})
        now = _now_iso()
        rows: List[tuple] = []
        for chunk, emb in zip(batch, embeddings):
            chunk_id = chunk.get("chunk_id")
            document_id = chunk.get("document_id")
            chunk_index = chunk.get("chunk_index")
            title = chunk.get("title")
            url = chunk.get("url")
            source_type = chunk.get("source_type")
            text = chunk.get("text")
            metadata = dict(chunk.get("metadata", {}) or {})
            rows.append(
                (
                    chunk_id,
                    document_id,
                    chunk_index,
                    title,
                    url,
                    source_type,
                    text,
                    json.dumps(metadata, ensure_ascii=False),
                    json.dumps(emb),
                    now,
                )
            )

        cur.executemany(
            "INSERT OR REPLACE INTO embeddings (id, document_id, chunk_index, title, url, source_type, text, metadata, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        total += len(rows)

    conn.close()
    return total


def fetch_all_embeddings(db_path: Path | str) -> List[JSONDict]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT id, document_id, chunk_index, title, url, source_type, text, metadata, embedding, created_at FROM embeddings"
    ).fetchall()
    
    conn.close()

    out: List[JSONDict] = []
    for row in rows:

        id_, document_id, chunk_index, title, url, source_type, text, metadata_text, embedding_text, created_at = row

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
                "title": title,
                "url": url,
                "source_type": source_type,
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

    q_emb = embedder.embed({"chunks": [{"text": query_text}]})[0]
    all_rows = fetch_all_embeddings(db_path)

    # compare size of query and db vector and error if they don't match
    if all_rows and "embedding" in all_rows[0]:
        db_emb = all_rows[0]["embedding"]
        if len(q_emb) != len(db_emb):
            raise ValueError(f"Query embedding dimension {len(q_emb)} does not match DB embedding dimension {len(db_emb)}.")

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
    "get_embedder_with_dimension",
    "read_db_embedding_config",
    "init_db",
    "ingest_payload_to_sqlite",
    "fetch_all_embeddings",
    "query_similar_by_text",
]
