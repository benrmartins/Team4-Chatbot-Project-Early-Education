import argparse
import json
from pathlib import Path
from ingestion_pipeline.scripts.pipeline_runner import *
from project_config import *

class DataProcessor():
    """Class to handle data processing steps for the chatbot ingestion pipeline."""
    def __init__(self, name: str, chunk_size: int, chunk_overlap: int, embedding_dim: int, batch_size: int, output_path: str, embedding_method: str = "default"):
        self.name = name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_dim = embedding_dim
        self.embedding_method = embedding_method
        self.batch_size = batch_size
        self.output_path = output_path
        self.web_data = None
        self.chunk_payload = None
        self.source_summary = None

    @staticmethod
    def build_variant_name(
        embedding_method: str,
        chunk_size: int,
        chunk_overlap: int,
        embedding_dim: int,
        batch_size: int,
    ) -> str:
        method = str(embedding_method or "default").strip().lower()
        return f"{method}_cs{int(chunk_size)}_co{int(chunk_overlap)}_ed{int(embedding_dim)}_bs{int(batch_size)}"

    @staticmethod
    def build_variant_output_path(name: str, db_dir: str | Path = DATA_DIR) -> str:
        directory = Path(db_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return str(directory / f"{name}.sqlite")

    def create_variant(
            self,
            name: str,
            chunk_size: int,
            chunk_overlap: int,
            embedding_dim: int,
            batch_size: int,
            output_path: str | None = None,
            embedding_method: str = "default",
    ) -> 'DataProcessor':
        path_to_db = output_path or self.build_variant_output_path(name)
        if not str(path_to_db).lower().endswith(".sqlite"):
            path_to_db = f"{path_to_db}.sqlite"
        if Path(path_to_db).exists():
            raise RuntimeError(f"Warning: Output database for variant '{name}' already exists at {path_to_db}.")

        copy = DataProcessor(
            name=name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            output_path=path_to_db,
            embedding_method=embedding_method,
        )

        copy.web_data = self.web_data
        copy.source_summary = self.source_summary
        copy.embed()
        return copy

    def process(self):
        self.crawl()
        self.chunk()
        self.embed()

    def crawl(self, output_path: str | None = None):
        self.web_data, self.source_summary = run_crawlers(web_output_path=output_path)
        if self.web_data is None or self.source_summary is None:
            raise ProcessLookupError("Crawling failed to retrieve data.")
        return self.web_data, self.source_summary
    
    def chunk(self):
        if self.web_data is None or self.source_summary is None:
            print("Web data or source summary not found. Running crawler first...")
            self.crawl()
    
        self.chunk_payload = run_chunking(
            all_documents=self.web_data or [],
            source_summary=self.source_summary or {},
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        return self.chunk_payload

    def embed(self):
        if self.chunk_payload is None:
            print("Chunk payload not found. Running chunking step first...")
            self.chunk()

        run_embedder(
            unified_payload=self.chunk_payload or {},
            dimensions=self.embedding_dim,
            batch_size=self.batch_size,
            database_path=self.output_path,
            embedding_method=self.embedding_method,
        )

class DefaultDataProcessor(DataProcessor):
    """Baseline pipeline variant with default parameters and webcrawler outputs defined in project_config.py."""
    def __init__(self):
        super().__init__(
            name="default",
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
            batch_size=DEFAULT_BATCH_SIZE,
            output_path=str(DEFAULT_VECTOR_DB_PATH) + "_default.sqlite",
            embedding_method="dummy",
        )
        drive_output_path = DATA_DIR / "drive_data.json"

        # If default crawler output is present, prefer it and merge with drive output when available.
        if DEFAULT_WEB_OUTPUT.exists():
            with open(DEFAULT_WEB_OUTPUT, "r", encoding="utf-8") as f:
                web_output = json.load(f)

            documents = list(web_output.get("documents", []) or [])
            summary = web_output.get("summary", {}) or {}

            if drive_output_path.exists():
                with open(drive_output_path, "r", encoding="utf-8") as f:
                    drive_output = json.load(f)

                drive_docs = list(drive_output.get("documents", []) or [])
                seen_ids = {doc.get("document_id") for doc in documents if doc.get("document_id")}
                for doc in drive_docs:
                    doc_id = doc.get("document_id")
                    if doc_id and doc_id in seen_ids:
                        continue
                    documents.append(doc)
                    if doc_id:
                        seen_ids.add(doc_id)

                summary = {
                    "website": web_output.get("summary", {}),
                    "google_drive": drive_output.get("summary", {}),
                }

            self.web_data = documents
            self.source_summary = summary if isinstance(summary, dict) else {"website": summary}
        else:
            print(f"Default web output not found at {DEFAULT_WEB_OUTPUT}. Running crawler to populate data...")
            self.crawl(output_path=str(DEFAULT_WEB_OUTPUT))
