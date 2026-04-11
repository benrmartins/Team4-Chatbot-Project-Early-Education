import tempfile
import unittest
from pathlib import Path

from chatbot.tool_calls.handlers.database_retrieval import search_sqlite_knowledge
from ingestion_pipeline.services.vector_store import DummyEmbedder, ingest_payload_to_sqlite


class DatabaseRetrievalTests(unittest.TestCase):
    def test_search_sqlite_knowledge_returns_ranked_results_with_citations(self):
        payload = {
            "chunks": [
                {
                    "chunk_id": "doc-1::chunk-0000",
                    "document_id": "doc-1",
                    "chunk_index": 0,
                    "title": "Language Access Program",
                    "url": "https://example.org/languages",
                    "source_type": "website",
                    "text": "Programs are offered in English, Spanish, and Mandarin to support multilingual educators.",
                    "metadata": {
                        "source_name": "Language Access Program",
                        "source_locator": "https://example.org/languages",
                        "section_hint": "Program Supports",
                    },
                },
                {
                    "chunk_id": "doc-2::chunk-0000",
                    "document_id": "doc-2",
                    "chunk_index": 0,
                    "title": "General Mission",
                    "url": "https://example.org/mission",
                    "source_type": "website",
                    "text": "We support children and families through leadership development.",
                    "metadata": {
                        "source_name": "General Mission",
                        "source_locator": "https://example.org/mission",
                    },
                },
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "knowledge.sqlite"
            ingest_payload_to_sqlite(
                payload,
                db_path,
                embedder=DummyEmbedder(dim=32),
                batch_size=8,
            )

            results = search_sqlite_knowledge(
                query="How do they support multilingual educators?",
                max_results=3,
                database_path=str(db_path),
            )

        self.assertEqual(1, results["results"][0]["rank"])
        self.assertIn("citation", results["results"][0])
        self.assertEqual("https://example.org/languages", results["results"][0]["citation"]["url"])
        self.assertIn("matched_terms", results["results"][0])

    def test_search_sqlite_knowledge_requires_database_path(self):
        result = search_sqlite_knowledge(query="test", max_results=3, database_path=None)
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
