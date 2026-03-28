from abc import ABC, abstractmethod
import json


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
            "google_drive": self._load_google_drive_source_from_json(),
            "website_sources": self._load_website_sources_from_json(),
        }

    @staticmethod
    def _load_google_drive_source_from_json():
        jsonFile = open('drive_data.json')
        data = json.load(jsonFile)
        jsonFile.close()
        return data

    @staticmethod
    def _load_website_sources_from_json():
        jsonFile = open('website_sources.json')
        data = json.load(jsonFile)
        jsonFile.close()
        return data
    
class OtherDatabase(DatabaseProvider):
    """Placeholder for future implementation."""

    def load(self):
        raise NotImplementedError("OtherDatabase.load is not implemented yet.")