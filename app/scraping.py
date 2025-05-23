import re
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import List, TypedDict
from urllib.parse import urljoin

import boto3
import requests
import warcio
from app.utils import MIME_TYPES, USER_AGENT, setup_logger
from bs4 import BeautifulSoup
from dotenv import load_dotenv

logger = setup_logger(__name__)
load_dotenv()


class WARCJob(TypedDict):
    url: str
    mime_type: str
    timestamp: str | int
    filename: str
    length: str | int
    offset: str | int


class RecordJob(TypedDict):
    url: str
    is_pdf: bool
    is_html: bool
    pdf_bytes: bytes | bytearray
    pdf_urls: List[str]
    timestamp: str | int


class Scraping:
    def __init__(self, session: boto3.Session):
        self.s3 = session.client("s3")

    def find_pdf_from_html(self, html_text, base_url):
        soup = BeautifulSoup(html_text, "html.parser")
        pdf_urls = []

        # Handle file viewer link
        link_pattern = r"=(https?://[^\s]*?\.pdf)"
        match = re.search(link_pattern, base_url)
        if match:
            pdf_urls.append(match.group(1))

        # Handle link in href tag
        for link in soup.find_all("a", href=True):
            href = link["href"].replace("\\", "/")

            if href.endswith(".pdf"):
                if href.startswith(("http://", "https://")):
                    pdf_urls.append(href)
                else:
                    # Convert relative url to absolute url
                    pdf_urls.append(urljoin(base_url, href))
        return pdf_urls

    def is_valid_pdf(self, content: bytes) -> bool:
        return content.startswith(b"%PDF-")

    def process_warc_record(self, warc_job: WARCJob) -> RecordJob:
        if warc_job["mime_type"] not in MIME_TYPES:
            return {}
        try:
            response = self.s3.get_object(
                Bucket="commoncrawl",
                Key=warc_job["filename"],
                Range=f"bytes={int(warc_job['offset'])}-{int(warc_job['offset']) + int(warc_job['length']) - 1}",
            )
            record_data = response["Body"].read()

            with BytesIO(record_data) as stream:
                for record in warcio.ArchiveIterator(stream):
                    if record.rec_type == "response":
                        is_pdf = "pdf" in warc_job["mime_type"]
                        is_html = "html" in warc_job["mime_type"]
                        response = record.content_stream().read()
                        pdf_bytes = (
                            response
                            if (is_pdf and self.is_valid_pdf(response))
                            else self.fetch_pdf(warc_job["url"]).get("pdf_bytes")
                        )
                        pdf_urls = (
                            self.find_pdf_from_html(response, warc_job["url"])
                            if is_html
                            else []
                        )
                        ts = warc_job["timestamp"]
                        if hasattr(ts, "timestamp"):
                            ts = ts.timestamp()
                        return RecordJob(
                            url=warc_job["url"],
                            is_pdf=is_pdf,
                            is_html=is_html,
                            pdf_bytes=pdf_bytes,
                            pdf_urls=pdf_urls,
                            timestamp=int(ts),
                        )
        except Exception as e:
            logger.error(f"Error processing {warc_job['url']}: {str(e)}")
        return {}

    def fetch_pdf(self, url) -> RecordJob:
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
            if response.status_code == 200:
                content_type = response.headers.get("Content-Type", "").lower()
                if "pdf" in content_type:
                    return RecordJob(
                        url=url,
                        is_pdf=True,
                        is_html=False,
                        pdf_bytes=response.content,
                        pdf_urls=[],
                        timestamp=int(time.time()),
                    )
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {str(e)}")
        return {}

    def process_pdf_urls(self, pdf_urls: List[str]) -> List[RecordJob]:
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.fetch_pdf, pdf_urls))
        return results
