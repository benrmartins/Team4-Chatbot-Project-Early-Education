import unittest

from ingestion_pipeline.vector_preprocess import build_chunk_records


class VectorPreprocessTests(unittest.TestCase):
    def test_build_chunk_records_adds_section_hint_for_heading_aware_chunks(self):
        documents = [
            {
                "document_id": "doc-1",
                "source_type": "website",
                "title": "Program Catalog",
                "url": "https://example.org/catalog",
                "text": (
                    "Program Overview\n\n"
                    "Leading for Change helps educators develop leadership mindsets and create change projects. "
                    "Participants work together in a professional learning community.\n\n"
                    "Eligibility\n\n"
                    "Applicants should be early educators interested in leadership and quality improvement."
                ),
            }
        ]

        chunks = build_chunk_records(documents, chunk_size=180, chunk_overlap=60)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual("Program Overview", chunks[0]["metadata"].get("section_hint"))
        self.assertIn("Leading for Change", chunks[0]["text"])
        self.assertTrue(chunks[1]["metadata"].get("section_hint"))


if __name__ == "__main__":
    unittest.main()
