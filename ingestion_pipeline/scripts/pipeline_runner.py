import json
from pathlib import Path
from typing import Dict, List

from ingestion_pipeline.schema import JSONDict
from ingestion_pipeline.scripts.build_chunk_payload import build_chunk_payload, save_json
from ingestion_pipeline.services.vector_store import get_embedder_with_dimension, ingest_payload_to_sqlite, get_default_embedder
from ingestion_pipeline.webcrawlers import GoogleDriveCrawler, WebsiteCrawler
from project_config import (
    DEFAULT_WEB_OUTPUT,
    PIPELINE_RUN_DRIVE,
    PIPELINE_RUN_WEB,
)

def run_crawlers(
        run_drive: bool = PIPELINE_RUN_DRIVE, 
        run_web: bool = PIPELINE_RUN_WEB,
        web_output_path: str | None = str(DEFAULT_WEB_OUTPUT)
    ) -> tuple[List[JSONDict], Dict[str, JSONDict]]:
    all_documents: List[JSONDict] = []
    source_summary: Dict[str, JSONDict] = {}

    if run_drive:
        drive_payload = GoogleDriveCrawler().scrape()
        all_documents.extend(drive_payload.get("documents", []))
        source_summary["google_drive"] = drive_payload.get("summary", {})

    if run_web:
        web_payload = WebsiteCrawler().scrape()
        all_documents.extend(web_payload.get("documents", []))
        source_summary["website"] = web_payload.get("summary", {})

    if web_output_path:
        save_json(
            {"documents": all_documents, "summary": source_summary},
            web_output_path
        )
        print(f"Crawler output saved to: {web_output_path}")

    print("Crawling complete.")

    return all_documents, source_summary

def run_chunking(
        all_documents: List[JSONDict], 
        source_summary: Dict[str, JSONDict], 
        chunk_size: int, 
        chunk_overlap: int,
    ) -> JSONDict:
    unified_payload = build_chunk_payload(
        documents=all_documents,
        source={
            "type": "chunking_pipeline",
            "components": source_summary,
            "chunking": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
        },
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    print("Chunking complete.")

    return unified_payload

def run_embedder(
        unified_payload: JSONDict, 
        dimensions: int,
        batch_size: int,
        database_path: str,
        overwrite_db: bool = False,
    ) -> int:
    embedder = get_embedder_with_dimension(dim=dimensions)
    inserted = ingest_payload_to_sqlite(
        unified_payload, 
        database_path,
        embedder=embedder, 
        batch_size=batch_size, 
        overwrite_db=overwrite_db
    )
    print(f"Inserted {inserted} chunks into vector DB at: {database_path}")
    print("Embedding complete.")
    return inserted

__all__ = [
    "run_crawlers",
    "run_chunking",
    "run_embedder",
]
