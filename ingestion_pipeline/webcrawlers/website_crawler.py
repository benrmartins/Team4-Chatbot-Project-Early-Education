import json
import time
from collections import deque
from urllib.parse import urljoin, urlparse
from typing import Iterable

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ingestion_pipeline.schema import SOURCE_WEBSITE, build_base_payload
from ingestion_pipeline.scripts.build_chunk_payload import normalize_text
from project_config import ( 
    CRAWLER_DEPTH_LIMIT,
    DEFAULT_WEBSITE_SEED_URLS, 
    MIN_CONTENT_CHARS, 
    BLOCKED_FILE_EXTENSIONS, 
    EXCLUDED_PATH_SEGMENTS
)

def build_web_document_record(index: int, title: str, url: str, text: str) -> dict:
    clean_text = normalize_text(text)
    return {
        "document_id": f"web::{index:05d}",
        "source_type": SOURCE_WEBSITE,
        "source_name": "web_crawler",
        "source_locator": url,
        "title": title,
        "mime_type": "text/html",
        "url": url,
        "modified_time": None,
        "size_bytes": None,
        "folder_path": "",
        "text": clean_text,
        "char_count": len(clean_text),
    }


def build_web_payload(documents: list[dict], skipped_pages: list[dict], pages_seen: int) -> dict:
    indexed_files = [
        {
            "id": doc["document_id"],
            "name": doc["title"],
            "mimeType": doc["mime_type"],
            "url": doc["url"],
            "folder_path": doc["folder_path"],
            "char_count": doc["char_count"],
        }
        for doc in documents
    ]

    payload = build_base_payload(
        source={
            "type": "website_crawler",
            "recursive": True,
        },
        summary={
            "pages_seen": pages_seen,
            "pages_indexed": len(documents),
            "documents": len(documents),
            "pages_skipped": len(skipped_pages),
        },
    )
    payload["documents"] = documents
    payload["skipped_files"] = skipped_pages
    payload["indexed_files"] = indexed_files
    return payload


class WebsiteCrawler:
    def __init__(self, seeds: Iterable[str] | None = None, max_depth: int = CRAWLER_DEPTH_LIMIT) -> None:
        configured_seeds = [url.strip() for url in (seeds or DEFAULT_WEBSITE_SEED_URLS) if url and url.strip()]
        if not configured_seeds:
            raise ValueError(
                "No website seed URLs configured. Set DEFAULT_WEBSITE_SEED_URLS in project_config.py"
            )

        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
        }

        # Normalize and deduplicate seed URLs to create a consistent set of starting points for crawling
        self.seed_prefixes = list(dict.fromkeys(filter(None, (self._canonicalize_url(seed) for seed in configured_seeds))))

        if not self.seed_prefixes:
            raise ValueError("No valid website seed URLs found after normalization.")

        self.allowed_domains = {urlparse(seed_prefix).netloc for seed_prefix in self.seed_prefixes}
        self.seed_prefix_set = set(self.seed_prefixes)
        self.seed_scope_prefixes = tuple(f"{seed_prefix}/" for seed_prefix in self.seed_prefixes)
        self.max_depth = max_depth
        self.queue = deque((url, 0) for url in self.seed_prefixes)
        self.queued_urls = set(self.seed_prefixes)
        self.visited = set()
        self.documents = []
        self.skipped_pages = []
        self.request_timeout = (5, 15)

        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    @classmethod
    # Normalizes URLs to a canonical form for consistent comparison and deduplication.
    def _canonicalize_url(cls, url: str) -> str | None:
        parsed = urlparse((url or "").strip())
        scheme = parsed.scheme.lower().strip()
        domain = parsed.netloc.lower().strip()
        if scheme not in {"http", "https"} or not domain:
            return None
        path = (parsed.path or "/").strip()
        path = f"/{path.lstrip('/')}"
        path = (path.rstrip("/") or "/").lower()
        return f"{scheme}://{domain}{path}"

    @classmethod
    # Extracts the main textual content from a BeautifulSoup-parsed HTML page, while removing common noise elements.
    def _extract_content_text(cls, soup: BeautifulSoup) -> tuple[str | None, str | None]:
        content_area = soup.find("article") or soup.find("main") or soup.find("div", class_="entry-content")
        if not content_area:
            return None, "No content area found"

        for noise in content_area(["nav", "footer", "script", "style"]):
            noise.decompose()

        text = " ".join(content_area.get_text(separator=" ", strip=True).split())
        if len(text) <= MIN_CONTENT_CHARS:
            return None, "Extracted text too short"
        return text, None
    
    # Checks if a normalized URL is relevant based on allowed domains, seed URL prefixes, and exclusion criteria.
    def _is_relevant_canonical(self, normalized_url: str) -> bool:
        parsed = urlparse(normalized_url)
        segments = [seg for seg in (parsed.path or "/").split("/") if seg]
        non_info = any(seg in EXCLUDED_PATH_SEGMENTS or seg.startswith(("page", "paged", "comment")) for seg in segments)
        return (
            parsed.netloc in self.allowed_domains
            and (normalized_url in self.seed_prefix_set or normalized_url.startswith(self.seed_scope_prefixes))
            and not non_info
            and not parsed.path.endswith(BLOCKED_FILE_EXTENSIONS)
        )

    def _skip(self, url: str, reason: str) -> None:
        self.skipped_pages.append({"url": url, "reason": reason})

    def is_relevant(self, url: str) -> bool:
        normalized_url = self._canonicalize_url(url)
        return bool(normalized_url and self._is_relevant_canonical(normalized_url))

    def scrape(self) -> dict:
        page_index = 0

        try:
            while self.queue:
                url, depth = self.queue.popleft()
                self.queued_urls.discard(url)
                if url in self.visited:
                    continue

                self.visited.add(url)

                try:
                    print(f"Requesting: {url}")
                    response = self.session.get(
                        url,
                        headers=self.headers,
                        timeout=self.request_timeout,
                    )

                    if response.status_code != 200:
                        print(f"Skipping {url} - Status: {response.status_code}")
                        self._skip(url, f"HTTP {response.status_code}")
                        continue

                    soup = BeautifulSoup(response.text, "html.parser")
                    text, skip_reason = self._extract_content_text(soup)

                    # If no meaningful text could be extracted, skip indexing but still attempt to find links for crawling.
                    if text is None:
                        self._skip(url, skip_reason or "Unknown reason")
                    else:
                    # Build the document record and add to the list of documents to be indexed.
                        title_text = soup.title.string if soup.title else ""
                        title = title_text.replace(" - UMass Boston", "").strip() if isinstance(title_text, str) else ""
                        title = title or "Untitled"
                        self.documents.append(build_web_document_record(page_index, title, url, text))
                        page_index += 1

                    if depth < self.max_depth:
                        link_depth = depth + 1

                        # Extract and enqueue new links from the page
                        for anchor in soup.find_all("a", href=True):
                            href_attr = anchor.get("href")
                            # Handle cases where href might be a list (e.g., from malformed HTML)
                            if isinstance(href_attr, list):
                                href_attr = href_attr[0] if href_attr else None
                            if not href_attr:
                                continue
                            normalized_link = self._canonicalize_url(urljoin(url, str(href_attr)))
                            if not normalized_link or not self._is_relevant_canonical(normalized_link):
                                continue
                            if normalized_link in self.visited or normalized_link in self.queued_urls:
                                continue
                            self.queue.append((normalized_link, link_depth))
                            self.queued_urls.add(normalized_link)

                    time.sleep(1.5)

                except requests.exceptions.Timeout:
                    print(f"Timeout while requesting {url}")
                    self._skip(url, "Timeout")
                except Exception as exc:
                    print(f"Could not process {url}: {exc}")
                    self._skip(url, str(exc))
        except KeyboardInterrupt:
            print("\nScrape interrupted by user. Saving partial results...")

        payload = build_web_payload(
            documents=self.documents,
            skipped_pages=self.skipped_pages,
            pages_seen=len(self.visited),
        )
        return payload


if __name__ == "__main__":
    crawler = WebsiteCrawler()
    payload = crawler.scrape()
    print(json.dumps(payload["summary"], indent=2))
