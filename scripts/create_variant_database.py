import argparse
from ingestion_pipeline import DefaultDataProcessor
from project_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DATA_DIR
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Optional explicit name for the variant processor", type=str, default="")
    parser.add_argument("--embedding-method", help="Embedding method label used in variant naming", type=str, default="default")
    parser.add_argument("--chunk-size", help="Chunk size for the variant processor", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", help="Chunk overlap for the variant processor", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--embedding-dim", help="Embedding dimension for the variant processor", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--batch-size", help="Batch size for the variant processor", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output-path", help="Optional explicit output path for the variant processor", type=str, default="")
    args = parser.parse_args()

    # Create the variant
    pipeline_chunk_size = args.chunk_size
    pipeline_chunk_overlap = args.chunk_overlap
    embedding_dim = args.embedding_dim
    batch_size = args.batch_size
    embedding_method = args.embedding_method

    canonical_name = DefaultDataProcessor.build_variant_name(
        embedding_method=embedding_method,
        chunk_size=pipeline_chunk_size,
        chunk_overlap=pipeline_chunk_overlap,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
    )
    name = args.name.strip() or canonical_name

    output_path = args.output_path.strip() or DefaultDataProcessor.build_variant_output_path(
        name=name,
        db_dir=DATA_DIR ,
    )

    variant_processor = DefaultDataProcessor().create_variant(
        name=name,
        chunk_size=pipeline_chunk_size,
        chunk_overlap=pipeline_chunk_overlap,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        output_path=output_path
    )

    print(f"Created variant '{variant_processor.name}' at: {variant_processor.output_path}")


    

