import unittest
from pathlib import Path


class ChatSourceDateTemplateTests(unittest.TestCase):
    def setUp(self):
        self.template = Path("templates/index.html").read_text(encoding="utf-8")

    def test_chat_sources_extract_publish_dates_from_blog_urls(self):
        self.assertIn("function formatSourceDate", self.template)
        self.assertIn("match(/\\/(\\d{4})\\/(\\d{2})\\/(\\d{2})", self.template)
        self.assertIn("source-date", self.template)
        self.assertIn("Published", self.template)
        self.assertIn("toLocaleDateString", self.template)

    def test_chat_sources_do_not_show_missing_date_label(self):
        self.assertNotIn("Date not listed", self.template)
        self.assertNotIn("Not available yet", self.template)

    def test_chat_sources_are_deduped_by_url_before_rendering(self):
        self.assertIn("function uniqueSourcesByUrl", self.template)
        self.assertIn("uniqueSourcesByUrl(payload.evidence || [])", self.template)
        self.assertIn("uniqueSourcesByUrl(payload.citations || [])", self.template)


if __name__ == "__main__":
    unittest.main()
