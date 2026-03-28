import argparse
import json
from typing import Dict, List

from dotenv import load_dotenv

from googlescrape import DEFAULT_DRIVE_OUTPUT, ingest_google_drive
from vector_preprocess import build_vector_payload, save_json
from webscrapegem import EarlyEdCrawler

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Drive + web ingestion, then apply unified chunking/vector preprocessing."
    )
    parser.add_argument(
        "--folder-id",
        required=False,
        help="Google Drive folder ID to ingest. If omitted, uses DRIVE_FOLDER_ID or DRIVE_FOLDER_LINK from .env",
    )
    parser.add_argument(
        "--drive-output",
        default=DEFAULT_DRIVE_OUTPUT,
        help=f"Drive raw output path (default: {DEFAULT_DRIVE_OUTPUT})",
    )
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
        drive_payload = ingest_google_drive(folder_id=args.folder_id)
        save_json(drive_payload, args.drive_output)
        all_documents.extend(drive_payload.get("documents", []))
        source_summary["google_drive"] = drive_payload.get("summary", {})
        print(f"Drive output written to: {args.drive_output}")

    if not args.skip_web:
        crawler = EarlyEdCrawler()
        web_payload = crawler.scrape()
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