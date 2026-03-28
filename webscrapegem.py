import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time
import re

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
        self.knowledge_base = []

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
        while self.queue:
            url = self.queue.pop(0)
            if url in self.visited:
                continue

            try:
                print(f"Requesting: {url}")
                # Added a 15-second timeout and headers to bypass blocks
                response = requests.get(url, headers=self.headers, timeout=15)
                
                if response.status_code != 200:
                    print(f"Skipping {url} - Status: {response.status_code}")
                    continue

                self.visited.add(url)
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
                        self.knowledge_base.append({
                            "title": soup.title.string.replace(" - UMass Boston", "").strip() if soup.title else "Untitled",
                            "url": url,
                            "text": text
                        })

                # Find more links
                for a in soup.find_all('a', href=True):
                    full_link = urljoin(url, a['href']).split('?')[0].split('#')[0]
                    if self.is_relevant(full_link) and full_link not in self.visited:
                        self.queue.append(full_link)

                time.sleep(1.5) # Slight delay so we don't get IP banned

            except Exception as e:
                print(f"Could not connect to {url}: {e}")

        # Final Export
        with open("early_ed_clean_data.json", "w", encoding="utf-8") as f:
            json.dump(self.knowledge_base, f, indent=4)
        print(f"\nDone! Scraped {len(self.knowledge_base)} specific EarlyEd pages.")

if __name__ == "__main__":
    crawler = EarlyEdCrawler()
    crawler.scrape()
    