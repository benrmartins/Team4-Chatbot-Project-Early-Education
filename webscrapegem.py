import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import os
import time
import re
from datetime import datetime, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SCHEMA_VERSION = "1.0"
OUTPUT_JSON = "web_data.json"

def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def build_web_document_record(index, title, url, text):
    clean_text = normalize_text(text)
    return {
        "document_id": f"web::{index:05d}",
        "source_type": "website",
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


def build_web_payload(documents, skipped_pages, pages_seen):
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

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "type": "website_crawler",
            "recursive": True,
        },
        "summary": {
            "pages_seen": pages_seen,
            "pages_indexed": len(documents),
            "documents": len(documents),
            "pages_skipped": len(skipped_pages),
        },
        "documents": documents,
        "skipped_files": skipped_pages,
        "indexed_files": indexed_files,
    }

class EarlyEdCrawler:
    def __init__(self):
        # Faking a real browser to prevent "Connection Failed" / 403 Forbidden
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.seeds = [
            "https://www.umb.edu/early-education-leaders-institute/",
            "https://blogs.umb.edu/earlyed/"
        ]
        self.queue = list(self.seeds)
        self.visited = set()
        self.documents = []
        self.skipped_pages = []
        self.request_timeout = (5, 15)  # (connect timeout, read timeout)

        # Reuse connections and retry transient server/network failures.
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

    def is_relevant(self, url):
        parsed = urlparse(url)
        # STRICT RULE: Only stay inside these specific sub-folders
        is_blog = bool(re.match(r'.*/\d{4}/\d{2}/\d{2}/.*', url))
        is_institute = "umb.edu/early-education-leaders-institute" in url
        
        # Avoid huge files or external social media
        is_not_file = not url.lower().endswith(('.pdf', '.jpg', '.png', '.zip', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.mp4', '.mp3'))
        is_not_fragment = not parsed.fragment  # Avoid URLs with fragments (anchor links that don't lead to anything useful)
        
        return (is_blog or is_institute) and is_not_file and is_not_fragment
    
    def scrape(self):
        if os.path.exists(OUTPUT_JSON):
            print(f"Skipping scrape because {OUTPUT_JSON} already exists.")
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                return json.load(f)

        page_index = 0

        try:
            while self.queue:
                url = self.queue.pop(0)
                if url in self.visited:
                    continue

                # Mark early so bad links are not retried forever if rediscovered.
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
                        self.skipped_pages.append({
                            "url": url,
                            "reason": f"HTTP {response.status_code}",
                        })
                        continue

                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Extract Content
                    # Most UMB blogs use <article> or <div class="entry-content">
                    content_area = soup.find('article') or soup.find('main') or soup.find('div', class_='entry-content')

                    if content_area:
                        # Remove nav/footer inside the content area if they exist
                        for noise in content_area(['nav', 'footer', 'script', 'style']):
                            noise.decompose()

                        text = " ".join(content_area.get_text(separator=' ', strip=True).split())

                        if len(text) > 100: # Only save meaningful pages
                            title = soup.title.string.replace(" - UMass Boston", "").strip() if soup.title else "Untitled"
                            self.documents.append(build_web_document_record(page_index, title, url, text))
                            page_index += 1
                        else:
                            self.skipped_pages.append({
                                "url": url,
                                "reason": "Extracted text too short",
                            })
                    else:
                        self.skipped_pages.append({
                            "url": url,
                            "reason": "No content area found",
                        })

                    # Find more links
                    for a in soup.find_all('a', href=True):
                        full_link = urljoin(url, a['href']).split('?')[0].split('#')[0]
                        if self.is_relevant(full_link) and full_link not in self.visited and full_link not in self.queue:
                            self.queue.append(full_link)

                    time.sleep(1.5) # Slight delay so we don't get IP banned

                except requests.exceptions.Timeout:
                    print(f"Timeout while requesting {url}")
                    self.skipped_pages.append({
                        "url": url,
                        "reason": "Timeout",
                    })
                except requests.exceptions.RequestException as e:
                    print(f"Could not connect to {url}: {e}")
                    self.skipped_pages.append({
                        "url": url,
                        "reason": str(e),
                    })
                except Exception as e:
                    print(f"Unexpected error while processing {url}: {e}")
                    self.skipped_pages.append({
                        "url": url,
                        "reason": str(e),
                    })
        except KeyboardInterrupt:
            print("\nScrape interrupted by user. Saving partial results...")

        payload = build_web_payload(
            documents=self.documents,
            skipped_pages=self.skipped_pages,
            pages_seen=len(self.visited),
        )

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


        print(f"\nDone! Scraped {len(self.documents)} specific EarlyEd pages.")
        print(f"Output written to: {OUTPUT_JSON}")
        return payload

if __name__ == "__main__":
    crawler = EarlyEdCrawler()
    crawler.scrape()
    