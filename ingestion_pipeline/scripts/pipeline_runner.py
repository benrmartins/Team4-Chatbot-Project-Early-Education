import json
from typing import Dict, List

from ingestion_pipeline.schema import JSONDict
from ingestion_pipeline.vector_preprocess import build_vector_payload, save_json
from ingestion_pipeline.vector_store import ingest_payload_to_sqlite, get_default_embedder
from ingestion_pipeline.webcrawlers import GoogleDriveCrawler, WebsiteCrawler
from project_config import (
    DEFAULT_DRIVE_OUTPUT,
    DEFAULT_VECTOR_OUTPUT,
    DEFAULT_VECTOR_DB,
    DEFAULT_WEB_OUTPUT,
    PIPELINE_CHUNK_OVERLAP,
    PIPELINE_CHUNK_SIZE,
    PIPELINE_RUN_DRIVE,
    PIPELINE_RUN_WEB,
)


def run_pipeline() -> None:
    all_documents: List[JSONDict] = []
    source_summary: Dict[str, JSONDict] = {}

    if PIPELINE_RUN_DRIVE:
        drive_payload = GoogleDriveCrawler().scrape()
        save_json(drive_payload, str(DEFAULT_DRIVE_OUTPUT))
        all_documents.extend(drive_payload.get("documents", []))
        source_summary["google_drive"] = drive_payload.get("summary", {})
        print(f"Drive output written to: {DEFAULT_DRIVE_OUTPUT}")

    if PIPELINE_RUN_WEB:
        web_payload = WebsiteCrawler().scrape()
        save_json(web_payload, str(DEFAULT_WEB_OUTPUT))
        all_documents.extend(web_payload.get("documents", []))
        source_summary["website"] = web_payload.get("summary", {})
        print(f"Web output written to: {DEFAULT_WEB_OUTPUT}")

    unified_payload = build_vector_payload(
        documents=all_documents,
        source={
            "type": "combined_pipeline",
            "components": source_summary,
            "chunking": {
                "chunk_size": PIPELINE_CHUNK_SIZE,
                "chunk_overlap": PIPELINE_CHUNK_OVERLAP,
            },
        },
        chunk_size=PIPELINE_CHUNK_SIZE,
        chunk_overlap=PIPELINE_CHUNK_OVERLAP,
    )

    save_json(unified_payload, str(DEFAULT_VECTOR_OUTPUT))

    # Optional: ingest embeddings into a local SQLite vector DB. Uses a
    # deterministic dummy embedder by default so this works without API keys.
    try:
        db_path = DEFAULT_VECTOR_DB
        embedder = get_default_embedder()
        inserted = ingest_payload_to_sqlite(unified_payload, db_path, embedder=embedder)
        print(f"Inserted {inserted} embeddings into: {db_path}")
    except Exception as exc:  # pragma: no cover - optional step
        print(f"Warning: could not ingest embeddings to DB: {exc}")

    print("Unified pipeline complete.")
    print(json.dumps(unified_payload["summary"], indent=2))
    print(f"Vector output written to: {DEFAULT_VECTOR_OUTPUT}")


if __name__ == "__main__":
    run_pipeline()

