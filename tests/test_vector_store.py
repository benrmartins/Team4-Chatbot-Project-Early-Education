import tempfile
import unittest
from pathlib import Path

from ingestion_pipeline.vector_preprocess import build_vector_payload
from ingestion_pipeline.vector_store import DummyEmbedder, ingest_payload_to_sqlite, query_similar_by_text


class VectorStoreTests(unittest.TestCase):
    def test_ingest_and_query(self):
        documents = [
            {
                "document_id": "doc-1",
                "source_type": "website",
                "title": "Leadership Guide",
                "url": "https://example.org/leadership",
                "text": (
                    "Program Overview\n\n"
                    "Leading for Change helps educators develop leadership mindsets and create change projects. "
                    "Participants work together in a professional learning community.\n\n"
                    "Eligibility\n\n"
                    "Applicants should be early educators interested in leadership and quality improvement."
                ),
            }
        ]

        payload = build_vector_payload(
            documents=documents,
            source={"type": "test_pipeline"},
            chunk_size=180,
            chunk_overlap=60,
        )

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "test_vectors.db"
            embedder = DummyEmbedder(dim=64)
            inserted = ingest_payload_to_sqlite(payload, db_path, embedder=embedder)
            self.assertGreater(inserted, 0)

            results = query_similar_by_text(db_path, "leadership project", embedder=embedder, top_k=3)
            self.assertTrue(len(results) >= 1)
            self.assertIn("score", results[0])
            self.assertGreaterEqual(results[0]["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
