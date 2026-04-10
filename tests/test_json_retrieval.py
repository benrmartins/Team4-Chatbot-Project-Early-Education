import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "chatbot" / "tool_calls" / "handlers" / "json_retrieval.py"
SPEC = importlib.util.spec_from_file_location("json_retrieval_under_test", MODULE_PATH)
json_retrieval = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(json_retrieval)


def make_payload(chunks):
    return {
        "schema_version": "1.0",
        "summary": {"documents": len({chunk["document_id"] for chunk in chunks}), "chunks": len(chunks)},
        "chunks": chunks,
    }


class JsonRetrievalTests(unittest.TestCase):
    def test_search_returns_evidence_metadata_and_synonym_aware_ranking(self):
        payload = make_payload(
            [
                {
                    "chunk_id": "doc-1::chunk-0000",
                    "document_id": "doc-1",
                    "source_type": "website",
                    "title": "Language Access Program",
                    "url": "https://example.org/languages",
                    "chunk_index": 0,
                    "text": (
                        "Programs are offered in English, Spanish, Portuguese, Cantonese, and Mandarin "
                        "to create inclusive pathways for educators and families."
                    ),
                    "metadata": {"section_hint": "Program Supports"},
                },
                {
                    "chunk_id": "doc-2::chunk-0000",
                    "document_id": "doc-2",
                    "source_type": "website",
                    "title": "General Mission",
                    "url": "https://example.org/mission",
                    "chunk_index": 0,
                    "text": "We support children and families through leadership development.",
                    "metadata": {},
                },
            ]
        )

        with patch.object(json_retrieval, "load_unified_knowledge_base", return_value=payload):
            results = json_retrieval.search_unified_knowledge(
                "How do they support multilingual educators?", max_results=3
            )

        self.assertFalse(results["low_confidence"])
        self.assertEqual("Language Access Program", results["results"][0]["title"])
        self.assertEqual(1, results["results"][0]["rank"])
        self.assertIn("evidence_snippet", results["results"][0])
        self.assertIn("matched_terms", results["results"][0])
        self.assertIn("match_reasons", results["results"][0])
        self.assertTrue(results["results"][0]["matched_terms"])
        self.assertTrue(results["results"][0]["match_reasons"])

    def test_search_marks_empty_or_weak_results_as_low_confidence(self):
        payload = make_payload(
            [
                {
                    "chunk_id": "doc-1::chunk-0000",
                    "document_id": "doc-1",
                    "source_type": "website",
                    "title": "General Mission",
                    "url": "https://example.org/mission",
                    "chunk_index": 0,
                    "text": "We support children and families through leadership development.",
                    "metadata": {},
                }
            ]
        )

        with patch.object(json_retrieval, "load_unified_knowledge_base", return_value=payload):
            results = json_retrieval.search_unified_knowledge(
                "Does the institute say anything about campus parking permits?", max_results=3
            )

        self.assertTrue(results["low_confidence"])
        self.assertEqual(0, results["result_count"])

    def test_search_suppresses_duplicate_top_results_from_same_document(self):
        payload = make_payload(
            [
                {
                    "chunk_id": "doc-1::chunk-0000",
                    "document_id": "doc-1",
                    "source_type": "website",
                    "title": "Leading for Change Overview",
                    "url": "https://example.org/change-overview",
                    "chunk_index": 0,
                    "text": "Leading for Change helps educators become change agents and improve program quality.",
                    "metadata": {},
                },
                {
                    "chunk_id": "doc-1::chunk-0001",
                    "document_id": "doc-1",
                    "source_type": "website",
                    "title": "Leading for Change Overview",
                    "url": "https://example.org/change-overview",
                    "chunk_index": 1,
                    "text": "Leading for Change helps educators become change agents and improve program quality in practice.",
                    "metadata": {},
                },
                {
                    "chunk_id": "doc-2::chunk-0000",
                    "document_id": "doc-2",
                    "source_type": "google_drive",
                    "title": "Leadership Study",
                    "url": "https://example.org/study",
                    "chunk_index": 0,
                    "text": "Study participants reported stronger leadership identities and change agency after the program.",
                    "metadata": {},
                },
            ]
        )

        with patch.object(json_retrieval, "load_unified_knowledge_base", return_value=payload):
            results = json_retrieval.search_unified_knowledge(
                "How does Leading for Change help educators become change agents?",
                max_results=3,
            )

        top_doc_ids = [result["document_id"] for result in results["results"][:2]]
        self.assertEqual(2, len(set(top_doc_ids)))

    def test_resolve_knowledge_base_path_uses_repo_root_fallback(self):
        preferred = Mock(spec=Path)
        preferred.exists.return_value = False
        fallback = Mock(spec=Path)
        fallback.exists.return_value = True

        resolved = json_retrieval.resolve_knowledge_base_path(
            preferred_path=preferred,
            fallback_path=fallback,
        )
        self.assertEqual(fallback, resolved)


if __name__ == "__main__":
    unittest.main()
