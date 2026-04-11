import argparse
from ingestion_pipeline import DefaultDataProcessor
from project_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_VECTOR_DB_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
)
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Name for the variant processor", type=str, default="variant")
    parser.add_argument("--chunk-size", help="Chunk size for the variant processor", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", help="Chunk overlap for the variant processor", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--embedding-dim", help="Embedding dimension for the variant processor", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--batch-size", help="Batch size for the variant processor", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output-path", help="Output path for the variant processor", type=str, default=str(DEFAULT_VECTOR_DB_PATH))
    args = parser.parse_args()

    # Create the variant
    name = args.name
    pipeline_chunk_size = args.chunk_size
    pipeline_chunk_overlap = args.chunk_overlap
    embedding_dim = args.embedding_dim
    batch_size = args.batch_size
    output_path = args.output_path

    variant_processor = DefaultDataProcessor().create_variant(
        name=name,
        chunk_size=pipeline_chunk_size,
        chunk_overlap=pipeline_chunk_overlap,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        output_path=output_path
    )


    

