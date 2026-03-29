# main ingestion pipeline to run both Google Drive and web crawling, then unify outputs with vector preprocessing
# Usage:
#   python pipeline.py --folder-id <GOOGLE_DRIVE_FOLDER_ID> --chunk-size 700 --chunk-overlap 120
#   python pipeline.py --skip-drive --chunk-size 700 --chunk-overlap 120
#   python pipeline.py --skip-web --chunk-size 700 --chunk-overlap 120
# Outputs:
#   - drive_data.json: raw Google Drive ingestion output
#   - web_data.json: raw web crawling ingestion output
#   - unified_vector_data.json: combined and preprocessed output ready for vectorization
# Note: Ensure .env file is set up with DRIVE_FOLDER_ID or DRIVE_FOLDER_LINK if not providing --folder-id
# This script is designed for modularity and can be extended to include additional data sources or preprocessing steps as needed.
import argparse
import json
from typing import Dict, List

from dotenv import load_dotenv

from ingestion_pipeline.webcrawlers.googlescrape import GoogleDriveCrawler
from ingestion_pipeline.webcrawlers.webscrapegem import EarlyEdCrawler
from ingestion_pipeline.vector_preprocess import build_vector_payload, save_json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Drive + web ingestion, then apply unified chunking/vector preprocessing."
    )
    parser.add_argument("--folder-id",required=False, help="Google Drive folder ID to ingest. If omitted, uses DRIVE_FOLDER_ID or DRIVE_FOLDER_LINK from .env")
    parser.add_argument("--drive-output", default="drive_data.json", help=f"Drive raw output path (default: drive_data.json)")
    parser.add_argument("--web-output", default="web_data.json", help="Website raw output path")
    parser.add_argument("--vector-output", default="unified_vector_data.json", help="Unified chunked output path")
    parser.add_argument("--chunk-size", type=int, default=700, help="Chunk size in characters.")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap in characters.")
    parser.add_argument("--skip-drive", action="store_true", help="Skip Google Drive ingestion")
    parser.add_argument("--skip-web", action="store_true", help="Skip web crawling ingestion")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    all_documents: List[Dict] = []
    source_summary: Dict[str, Dict] = {}

    if not args.skip_drive:
        drive_payload = GoogleDriveCrawler().scrape(folder_id=args.folder_id)
        save_json(drive_payload, args.drive_output)
        all_documents.extend(drive_payload.get("documents", []))
        source_summary["google_drive"] = drive_payload.get("summary", {})
        print(f"Drive output written to: {args.drive_output}")

    if not args.skip_web:
        web_payload = EarlyEdCrawler().scrape()
        save_json(web_payload, args.web_output)
        all_documents.extend(web_payload.get("documents", []))
        source_summary["website"] = web_payload.get("summary", {})
        print(f"Web output written to: {args.web_output}")

    unified_payload = build_vector_payload(
        documents=all_documents,
        source={
            "type": "combined_pipeline",
            "components": source_summary,
            "chunking": {
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
            },
        },
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    save_json(unified_payload, args.vector_output)

    print("Unified pipeline complete.")
    print(json.dumps(unified_payload["summary"], indent=2))
    print(f"Vector output written to: {args.vector_output}")


if __name__ == "__main__":
    main()