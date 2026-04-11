import argparse
import json
from pathlib import Path
from ingestion_pipeline.scripts.pipeline_runner import *
from project_config import *

class DataProcessor():
    """Class to handle data processing steps for the chatbot ingestion pipeline."""
    def __init__(self, name: str, chunk_size: int, chunk_overlap: int, embedding_dim: int, batch_size: int, output_path: str):
        self.name = name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.output_path = output_path
        self.web_data = None
        self.chunk_payload = None
        self.source_summary = None

    def create_variant(
            self,
            name: str,
            chunk_size: int,
            chunk_overlap: int,
            embedding_dim: int,
            batch_size: int,
            output_path: str
    ) -> 'DataProcessor':
        path_to_db = output_path + f"_{name}.sqlite"
        if Path(path_to_db).exists():
            raise RuntimeError(f"Warning: Output database for variant '{name}' already exists at {path_to_db}.")

        copy = DataProcessor(
            name=name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            output_path=path_to_db
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
            output_path=str(DEFAULT_VECTOR_DB_PATH)
        )
        # if default web output is not found, run crawler to populate data for chunking and embedding steps
        if DEFAULT_WEB_OUTPUT.exists():
            with open(DEFAULT_WEB_OUTPUT, "r", encoding="utf-8") as f:
                web_output = json.load(f)
                self.web_data = web_output.get("documents", [])
                self.source_summary = {
                    "website": web_output.get("summary", {})
                }
        else:
            print(f"Default web output not found at {DEFAULT_WEB_OUTPUT}. Running crawler to populate data...")
            self.crawl(output_path=str(DEFAULT_WEB_OUTPUT))

