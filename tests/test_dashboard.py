import importlib
import io
import json
import sqlite3
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4


def install_openai_stubs():
    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda: None
    sys.modules["dotenv"] = dotenv_module

    openai_module = types.ModuleType("openai")

    class DummyOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **kwargs: None)
            )

    openai_module.OpenAI = DummyOpenAI
    sys.modules["openai"] = openai_module

    openai_types_module = types.ModuleType("openai.types")
    sys.modules["openai.types"] = openai_types_module

    chat_types_module = types.ModuleType("openai.types.chat")
    chat_types_module.ChatCompletionFunctionToolParam = dict
    sys.modules["openai.types.chat"] = chat_types_module


def import_app_module():
    install_openai_stubs()
    for module_name in [
        "app",
        "chatbot.chatbot_api",
        "chatbot.tool_calls.registry",
        "chatbot.tool_calls",
        "chatbot",
    ]:
        sys.modules.pop(module_name, None)
    return importlib.import_module("app")


class DashboardTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(tempfile.gettempdir()) / "owlbot_dashboard_tests"
        temp_root.mkdir(exist_ok=True)
        self.root = temp_root / f"dashboard_test_{uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.upload_dir = self.root / "dashboard_uploads"
        self.sources_path = self.root / "dashboard_sources.json"
        self.web_output_path = self.root / "missing_web_data.json"
        self.process_db_path = Path(tempfile.gettempdir()) / f"dashboard_chatbot_{uuid4().hex}.sqlite"
        self.app_module = import_app_module()
        self.patches = [
            patch.object(self.app_module, "DASHBOARD_UPLOAD_DIR", self.upload_dir, create=True),
            patch.object(self.app_module, "DASHBOARD_SOURCES_PATH", self.sources_path, create=True),
            patch.object(self.app_module, "DEFAULT_WEB_OUTPUT", self.web_output_path, create=True),
            patch.object(self.app_module, "DASHBOARD_PROCESS_DB_PATH", self.process_db_path, create=True),
        ]
        for patcher in self.patches:
            patcher.start()
        self.client = self.app_module.app.test_client()

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()
        try:
            shutil.rmtree(self.root)
        except PermissionError:
            pass
        try:
            self.process_db_path.unlink(missing_ok=True)
        except OSError:
            pass

    def test_dashboard_loads_when_data_files_are_missing(self):
        response = self.client.get("/dashboard")

        self.assertEqual(200, response.status_code)
        body = response.get_data(as_text=True)
        self.assertIn("data-tab-target=\"overview\"", body)
        self.assertIn("data-tab-target=\"content-library\"", body)
        self.assertIn("data-tab-target=\"coverage\"", body)
        self.assertIn("data-tab-target=\"review-tracker\"", body)
        self.assertIn("Upload New Institute Content", body)
        self.assertIn("Process for Chatbot", body)
        self.assertIn("Content Library", body)
        self.assertIn("Content Coverage", body)
        self.assertIn("Content Review Tracker", body)
        self.assertIn("Not available yet", body)

    def test_library_filters_hide_non_matching_cards(self):
        self.sources_path.write_text(
            json.dumps(
                [
                    {
                        "id": "ready-source",
                        "title": "Ready Source",
                        "source_type": "Uploaded File",
                        "category": "FAQ",
                        "filename": "ready.txt",
                        "status": "Ready for Chatbot",
                        "ready_for_chatbot": True,
                    }
                ]
            ),
            encoding="utf-8",
        )

        response = self.client.get("/dashboard")

        self.assertEqual(200, response.status_code)
        body = response.get_data(as_text=True)
        self.assertIn("[data-library-item][hidden]", body)
        self.assertIn("item.style.display = visible ? \"\" : \"none\";", body)
        self.assertIn("data-filter-group=", body)

    def test_valid_upload_saves_file_and_needs_processing_metadata(self):
        response = self.client.post(
            "/dashboard/upload",
            data={
                "title": "Updated Program FAQ",
                "category": "FAQ",
                "description": "New public FAQ details",
                "source_url": "https://example.org/faq",
                "file": (io.BytesIO(b"Question and answer"), "program faq.txt"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )

        self.assertEqual(200, response.status_code)
        self.assertTrue((self.upload_dir / "program_faq.txt").exists())

        sources = json.loads(self.sources_path.read_text(encoding="utf-8"))
        self.assertEqual(1, len(sources))
        self.assertEqual("Updated Program FAQ", sources[0]["title"])
        self.assertEqual("Needs Processing", sources[0]["status"])
        self.assertFalse(sources[0]["ready_for_chatbot"])
        self.assertIn("waiting to be prepared for the chatbot", response.get_data(as_text=True))

    def test_pending_upload_shows_in_processing_queue_not_content_library(self):
        self.client.post(
            "/dashboard/upload",
            data={
                "title": "Draft FAQ",
                "category": "FAQ",
                "file": (io.BytesIO(b"Draft question and answer"), "draft.txt"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )

        response = self.client.get("/dashboard")

        self.assertEqual(200, response.status_code)
        body = response.get_data(as_text=True)
        self.assertIn("Process for Chatbot", body)
        self.assertIn("Draft FAQ", body)
        self.assertIn("action=\"/dashboard/delete/", body)
        review_markup = body.split('id="review-tracker"', 1)[1]
        self.assertIn("Draft FAQ", review_markup)
        library_markup = body.split('id="content-library"', 1)[1].split('id="coverage"', 1)[0]
        self.assertNotIn("Draft FAQ", library_markup)

    def test_failed_upload_shows_in_review_tracker_not_content_library(self):
        self.sources_path.write_text(
            json.dumps(
                [
                    {
                        "id": "failed-source",
                        "title": "Broken Upload",
                        "source_type": "Uploaded File",
                        "category": "FAQ",
                        "filename": "broken.json",
                        "status": "Processing Failed",
                        "ready_for_chatbot": False,
                        "process_error": "bad json",
                    }
                ]
            ),
            encoding="utf-8",
        )

        response = self.client.get("/dashboard")

        self.assertEqual(200, response.status_code)
        body = response.get_data(as_text=True)
        queue_markup = body.split('id="process-queue"', 1)[1].split('id="needs-attention"', 1)[0]
        review_markup = body.split('id="review-tracker"', 1)[1]
        library_markup = body.split('id="content-library"', 1)[1].split('id="coverage"', 1)[0]
        self.assertIn("Broken Upload", queue_markup)
        self.assertIn("Broken Upload", review_markup)
        self.assertNotIn("Broken Upload", library_markup)

    def test_ready_upload_shows_in_content_library_with_details(self):
        self.sources_path.write_text(
            json.dumps(
                [
                    {
                        "id": "ready-upload",
                        "title": "Ready Upload",
                        "source_type": "Uploaded File",
                        "category": "FAQ",
                        "filename": "ready.txt",
                        "source_url": "https://example.org/ready",
                        "description": "Prepared staff note",
                        "status": "Ready for Chatbot",
                        "ready_for_chatbot": True,
                        "uploaded_at": "2026-04-01T10:00:00+00:00",
                        "processed_at": "2026-04-02T10:00:00+00:00",
                        "saved_passages": 3,
                    }
                ]
            ),
            encoding="utf-8",
        )

        response = self.client.get("/dashboard")

        self.assertEqual(200, response.status_code)
        body = response.get_data(as_text=True)
        library_markup = body.split('id="content-library"', 1)[1].split('id="coverage"', 1)[0]
        self.assertIn("Ready Upload", library_markup)
        self.assertIn("Source details", library_markup)
        self.assertIn("View FAQ", library_markup)
        self.assertIn("source-action-link", library_markup)
        self.assertNotIn("Open source", library_markup)
        self.assertIn("Saved passages: 3", library_markup)
        self.assertIn("Prepared staff note", library_markup)
        self.assertNotIn(str(self.upload_dir), library_markup)
        self.assertIn(".source-card:hover .source-delete-form", body)
        self.assertIn(".source-card:focus-within .source-delete-form", body)

    def test_existing_website_sources_show_as_ready_library_content(self):
        self.web_output_path.write_text(
            json.dumps(
                {
                    "documents": [
                        {
                            "document_id": "web-1",
                            "title": "Institute Home",
                            "url": "https://www.umb.edu/early-education-leaders-institute/",
                            "source_type": "website",
                            "modified_time": "2026-04-01T00:00:00+00:00",
                            "char_count": 1200,
                        }
                    ],
                    "generated_at_utc": "2026-04-03T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )

        response = self.client.get("/dashboard")

        self.assertEqual(200, response.status_code)
        body = response.get_data(as_text=True)
        library_markup = body.split('id="content-library"', 1)[1].split('id="coverage"', 1)[0]
        self.assertIn("Institute Home", library_markup)
        self.assertIn("Ready for Chatbot", library_markup)
        self.assertIn("Website Page", library_markup)
        self.assertIn("View page", library_markup)

    def test_content_library_uses_client_friendly_action_labels(self):
        self.sources_path.write_text(
            json.dumps(
                [
                    {
                        "id": "report",
                        "title": "Annual Report",
                        "source_type": "Uploaded File",
                        "category": "Report or PDF",
                        "filename": "annual.pdf",
                        "source_url": "https://example.org/report.pdf",
                        "status": "Ready for Chatbot",
                        "ready_for_chatbot": True,
                    },
                    {
                        "id": "contact",
                        "title": "Contact Sheet",
                        "source_type": "Uploaded File",
                        "category": "Contact Information",
                        "filename": "contact.txt",
                        "source_url": "https://example.org/contact",
                        "status": "Ready for Chatbot",
                        "ready_for_chatbot": True,
                    },
                    {
                        "id": "upload",
                        "title": "Uploaded Notes",
                        "source_type": "Uploaded File",
                        "category": "Other",
                        "filename": "notes.txt",
                        "source_url": "https://example.org/notes",
                        "status": "Ready for Chatbot",
                        "ready_for_chatbot": True,
                    },
                ]
            ),
            encoding="utf-8",
        )
        self.web_output_path.write_text(
            json.dumps(
                {
                    "documents": [
                        {
                            "document_id": "blog-1",
                            "title": "Blog Update",
                            "url": "https://blogs.umb.edu/earlyed/2026/04/01/blog-update/",
                            "source_type": "website",
                            "char_count": 1200,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        response = self.client.get("/dashboard")

        self.assertEqual(200, response.status_code)
        library_markup = response.get_data(as_text=True).split('id="content-library"', 1)[1].split('id="coverage"', 1)[0]
        self.assertIn("View document", library_markup)
        self.assertIn("View contact page", library_markup)
        self.assertIn("View uploaded file", library_markup)
        self.assertIn("Read blog post", library_markup)

    def test_coverage_and_review_tracker_render_friendly_summary(self):
        self.sources_path.write_text(
            json.dumps(
                [
                    {
                        "id": "ready-report",
                        "title": "Ready Report",
                        "source_type": "Uploaded File",
                        "category": "Report or PDF",
                        "filename": "report.pdf",
                        "status": "Ready for Chatbot",
                        "ready_for_chatbot": True,
                        "saved_passages": 4,
                    },
                    {
                        "id": "needs-url",
                        "title": "Needs Public Link",
                        "source_type": "Uploaded File",
                        "category": "FAQ",
                        "filename": "needs.txt",
                        "status": "Ready for Chatbot",
                        "ready_for_chatbot": True,
                    },
                    {
                        "id": "waiting",
                        "title": "Waiting Upload",
                        "source_type": "Uploaded File",
                        "category": "FAQ",
                        "filename": "waiting.txt",
                        "status": "Needs Processing",
                        "ready_for_chatbot": False,
                    },
                ]
            ),
            encoding="utf-8",
        )

        response = self.client.get("/dashboard")

        self.assertEqual(200, response.status_code)
        body = response.get_data(as_text=True)
        coverage_markup = body.split('id="coverage"', 1)[1].split('id="review-tracker"', 1)[0]
        review_markup = body.split('id="review-tracker"', 1)[1]
        self.assertIn("Report or PDF", coverage_markup)
        self.assertIn("4 saved passages", coverage_markup)
        self.assertIn("FAQ", coverage_markup)
        self.assertIn("Needs Public Link", review_markup)
        self.assertIn("Missing public source link", review_markup)
        self.assertIn("Waiting Upload", review_markup)
        self.assertIn("Waiting to be prepared", review_markup)

    def test_delete_pending_upload_removes_file_and_metadata(self):
        self.sources_path.write_text(
            json.dumps(
                [
                    {
                        "id": "source-remove",
                        "title": "Remove Me",
                        "filename": "remove.txt",
                        "status": "Needs Processing",
                        "ready_for_chatbot": False,
                    }
                ]
            ),
            encoding="utf-8",
        )
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        uploaded_file = self.upload_dir / "remove.txt"
        uploaded_file.write_text("Temporary upload", encoding="utf-8")
        self.assertTrue(uploaded_file.exists())

        response = self.client.post(
            "/dashboard/delete/source-remove",
            follow_redirects=True,
        )

        self.assertEqual(200, response.status_code)
        self.assertIn("was removed from the dashboard.", response.get_data(as_text=True))
        self.assertFalse(uploaded_file.exists())
        self.assertEqual([], json.loads(self.sources_path.read_text(encoding="utf-8")))

    def test_delete_rejects_ready_upload(self):
        self.sources_path.write_text(
            json.dumps(
                [
                    {
                        "id": "source-ready",
                        "title": "Ready Source",
                        "filename": "ready.txt",
                        "status": "Ready for Chatbot",
                        "ready_for_chatbot": True,
                    }
                ]
            ),
            encoding="utf-8",
        )
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        ready_file = self.upload_dir / "ready.txt"
        ready_file.write_text("ready", encoding="utf-8")

        response = self.client.post(
            "/dashboard/delete/source-ready",
            follow_redirects=True,
        )

        self.assertEqual(200, response.status_code)
        self.assertIn("Ready sources cannot be deleted from this dashboard queue.", response.get_data(as_text=True))
        self.assertTrue(ready_file.exists())
        sources = json.loads(self.sources_path.read_text(encoding="utf-8"))
        self.assertEqual(1, len(sources))

    def test_upload_requires_title(self):
        response = self.client.post(
            "/dashboard/upload",
            data={
                "title": "",
                "category": "FAQ",
                "file": (io.BytesIO(b"content"), "source.txt"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )

        self.assertEqual(200, response.status_code)
        self.assertIn("Please add a source title before submitting.", response.get_data(as_text=True))
        self.assertFalse(self.sources_path.exists())

    def test_upload_requires_file(self):
        response = self.client.post(
            "/dashboard/upload",
            data={"title": "Program Update", "category": "Program Information"},
            content_type="multipart/form-data",
            follow_redirects=True,
        )

        self.assertEqual(200, response.status_code)
        self.assertIn("Please choose a file before submitting.", response.get_data(as_text=True))
        self.assertFalse(self.sources_path.exists())

    def test_upload_rejects_unsupported_file_type(self):
        response = self.client.post(
            "/dashboard/upload",
            data={
                "title": "Image Upload",
                "category": "Other",
                "file": (io.BytesIO(b"image"), "photo.exe"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )

        self.assertEqual(200, response.status_code)
        self.assertIn("This file type is not supported yet.", response.get_data(as_text=True))
        self.assertFalse(self.sources_path.exists())

    def test_processing_txt_upload_marks_source_ready_and_writes_chunks(self):
        self.client.post(
            "/dashboard/upload",
            data={
                "title": "Partner Brief",
                "category": "Report or PDF",
                "description": "Institute partner overview",
                "source_url": "",
                "file": (io.BytesIO(b"Partners include state agencies and local leaders."), "partners.txt"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )

        sources = json.loads(self.sources_path.read_text(encoding="utf-8"))
        source_id = sources[0]["id"]

        response = self.client.post(
            f"/dashboard/process/{source_id}",
            follow_redirects=True,
        )

        self.assertEqual(200, response.status_code)
        body = response.get_data(as_text=True)
        self.assertIn("is now ready for chatbot answers", body)

        updated_sources = json.loads(self.sources_path.read_text(encoding="utf-8"))
        self.assertEqual("Ready for Chatbot", updated_sources[0]["status"])
        self.assertTrue(updated_sources[0]["ready_for_chatbot"])
        self.assertIn("processed_at", updated_sources[0])
        self.assertNotIn("process_error", updated_sources[0])
        self.assertTrue((self.upload_dir / "partners.txt").exists())
        self.assertTrue(self.process_db_path.exists())
        backup_path = self.app_module._dashboard_backup_path(self.process_db_path)
        self.assertTrue(backup_path.exists())

        conn = sqlite3.connect(self.process_db_path)
        try:
            row_count = conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE document_id = ?",
                (f"upload::{source_id}",),
            ).fetchone()[0]
        finally:
            conn.close()

        self.assertGreater(row_count, 0)

    def test_backup_failure_marks_source_failed_without_writing_chunks(self):
        self.client.post(
            "/dashboard/upload",
            data={
                "title": "Backup Fail",
                "category": "FAQ",
                "file": (io.BytesIO(b"This file should not be written."), "backup.txt"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        sources = json.loads(self.sources_path.read_text(encoding="utf-8"))
        source_id = sources[0]["id"]

        with patch.object(self.app_module, "_create_dashboard_db_backup", side_effect=OSError("backup blocked")):
            response = self.client.post(
                f"/dashboard/process/{source_id}",
                follow_redirects=True,
            )

        self.assertEqual(200, response.status_code)
        self.assertIn("could not be prepared for the chatbot", response.get_data(as_text=True))
        updated_sources = json.loads(self.sources_path.read_text(encoding="utf-8"))
        self.assertEqual("Processing Failed", updated_sources[0]["status"])
        self.assertFalse(updated_sources[0]["ready_for_chatbot"])

        if self.process_db_path.exists():
            conn = sqlite3.connect(self.process_db_path)
            try:
                row_count = conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'embeddings'"
                ).fetchone()[0]
            finally:
                conn.close()
            self.assertEqual(0, row_count)

    def test_processing_failure_marks_source_failed_and_keeps_file(self):
        self.client.post(
            "/dashboard/upload",
            data={
                "title": "Broken JSON",
                "category": "Other",
                "file": (io.BytesIO(b"{not-json"), "broken.json"),
            },
            content_type="multipart/form-data",
            follow_redirects=True,
        )

        sources = json.loads(self.sources_path.read_text(encoding="utf-8"))
        source_id = sources[0]["id"]

        response = self.client.post(
            f"/dashboard/process/{source_id}",
            follow_redirects=True,
        )

        self.assertEqual(200, response.status_code)
        body = response.get_data(as_text=True)
        self.assertIn("could not be prepared for the chatbot", body)
        self.assertIn("Try Processing Again", body)

        updated_sources = json.loads(self.sources_path.read_text(encoding="utf-8"))
        self.assertEqual("Processing Failed", updated_sources[0]["status"])
        self.assertFalse(updated_sources[0]["ready_for_chatbot"])
        self.assertIn("process_error", updated_sources[0])
        self.assertTrue((self.upload_dir / "broken.json").exists())

    def test_processing_unknown_source_shows_friendly_error(self):
        response = self.client.post(
            "/dashboard/process/missing-source",
            follow_redirects=True,
        )

        self.assertEqual(200, response.status_code)
        self.assertIn("We could not find that uploaded source.", response.get_data(as_text=True))

    def test_processing_ready_source_is_rejected(self):
        self.sources_path.write_text(
            json.dumps(
                [
                    {
                        "id": "source-1",
                        "title": "Already ready",
                        "filename": "ready.txt",
                        "status": "Ready for Chatbot",
                        "ready_for_chatbot": True,
                    }
                ]
            ),
            encoding="utf-8",
        )
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        (self.upload_dir / "ready.txt").write_text("ready", encoding="utf-8")

        response = self.client.post(
            "/dashboard/process/source-1",
            follow_redirects=True,
        )

        self.assertEqual(200, response.status_code)
        self.assertIn("This source is already ready for chatbot answers.", response.get_data(as_text=True))

    def test_failed_source_can_retry_and_recover(self):
        self.sources_path.write_text(
            json.dumps(
                [
                    {
                        "id": "source-1",
                        "title": "Retry me",
                        "filename": "retry.txt",
                        "status": "Processing Failed",
                        "ready_for_chatbot": False,
                        "process_error": "old error",
                    }
                ]
            ),
            encoding="utf-8",
        )
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        (self.upload_dir / "retry.txt").write_text("Retry content for the chatbot.", encoding="utf-8")

        response = self.client.post(
            "/dashboard/process/source-1",
            follow_redirects=True,
        )

        self.assertEqual(200, response.status_code)
        updated_sources = json.loads(self.sources_path.read_text(encoding="utf-8"))
        self.assertEqual("Ready for Chatbot", updated_sources[0]["status"])
        self.assertTrue(updated_sources[0]["ready_for_chatbot"])
        self.assertNotIn("process_error", updated_sources[0])


if __name__ == "__main__":
    unittest.main()
