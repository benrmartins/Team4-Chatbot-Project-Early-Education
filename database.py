from abc import ABC, abstractmethod


class DatabaseProvider(ABC):
    """Contract for chatbot data providers.

    Any future provider (Supabase, Postgres, etc.) should implement the same interface.
    """

    @abstractmethod
    def load(self):
        """Return a dictionary containing initialization data for the chatbot."""

    def build_context_message(self, data):
        """Optional helper to convert provider data into a system context line."""
        drive_count = len(data.get("google_drive", {}).get("indexed_files", []))
        url_count = len(data.get("website_sources", {}).get("urls", []))
        return (
            "Data source context loaded for this session: "
            f"{drive_count} Google Drive files indexed, "
            f"{url_count} website sources configured. "
            "Use tools for retrieval when relevant."
        )


class MockDatabase(DatabaseProvider):
    """Simple in-memory provider used for local development."""

    def load(self):
        return {
            "google_drive": self._load_google_drive_source(),
            "website_sources": self._load_website_sources(),
            "notes": [
                "Mock database initialized at chatbot startup.",
                "Replace this provider with Supabase later.",
            ],
        }

    @staticmethod
    def _load_google_drive_source_from_json():
        return {
            "enabled": False,
            "sync_folder": "",
            "indexed_files": [
                "lesson_plan_grade1.docx",
                "math_vocab_week3.docx",
            ],
        }

    @staticmethod
    def _load_website_sources_from_json():
        return {
            "enabled": False,
            "urls": [
                "https://example-school-resources.org",
                "https://example-early-learning.net",
            ],
            "scraper_jobs": [
                {"name": "curriculum_scraper", "status": "not_started"},
                {"name": "activities_scraper", "status": "not_started"},
            ],
        }


class OtherDatabase(DatabaseProvider):
    """Placeholder for future implementation."""

    def load(self):
        raise NotImplementedError("OtherDatabase.load is not implemented yet.")