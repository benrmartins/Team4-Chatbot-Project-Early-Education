import unittest

from ingestion_pipeline.vector_store import OpenAIEmbedder, get_default_embedder, ingest_payload_to_sqlite, query_similar_by_text

class OpenAIEmbeddingTests(unittest.TestCase):
    def test_openai_embedding(self):
        embedder = OpenAIEmbedder(model="text-embedding-3-small", dim=256)
        chunks = { "chunks": [
            {"text": "Hello World!"},
            {"text": "This is a test."},
        ]}
        embeddings = embedder.embed(chunks)
        self.assertEqual(len(embeddings), len(chunks["chunks"]))
        self.assertEqual(len(embeddings[0]), 256)

if __name__ == "__main__":
    unittest.main()