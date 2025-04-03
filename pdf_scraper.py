import argparse
import asyncio
import json
import re
import time
from datetime import datetime
from multiprocessing import cpu_count
from queue import Queue
from random import random
from typing import Literal
from urllib.parse import urljoin

import aiohttp
import requests
from bs4 import BeautifulSoup
from milvus import Milvus
from tenacity import (
    AsyncRetrying,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from utils import setup_logger
from vectorisation import Vectorisation

logger = setup_logger()
INDEX_SERVER = "http://index.commoncrawl.org"
AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
URLS = [
    "*.gov",
    "*.com",
    "*.org",
    "*.edu",
    "*.net",
    "*.co",
]
MIME_TYPES = [
    "pdf",
    "html",
]

jobs_queue = Queue()
vect = Vectorisation()
mv = Milvus()
counter = {"success": 0, "failed": 0, "empty_or_duplicate": 0}
failed = {}


def is_retryable_error(e):
    return isinstance(
        e, (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError, TimeoutError)
    ) or ("Failed to open stream" in str(e))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(is_retryable_error),
)
def get_index_api(index_name):
    session = requests.Session()
    r = session.get(urljoin(INDEX_SERVER, "collinfo.json")).json()
    collinfo = list(filter(lambda c: c["id"] == index_name, r))
    return collinfo[0]["cdx-api"] if collinfo else None


def get_num_pages(index_api, url):
    params = {
        "url": url,
        "showNumPages": True,
    }

    session = requests.Session()
    r = session.get(index_api, params=params).json()
    pages = r.get("pages")
    return pages if pages else None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(is_retryable_error),
)
def get_jobs_from_url(
    index_api, url, page, format: Literal["pdf", "html"] = "pdf", limit=None
):
    """
    Fetch absolute urls from the Common Crawl index API and add them to the jobs queue.
    """

    jobs = []
    params = {
        "url": url,
        "filter": [f"~mime:.*/{format}$", "=status:200"],
        "output": "json",
        "fl": "url,timestamp,mime",
        "page": page,
    }

    if limit:
        params["limit"] = limit

    try:
        response = requests.get(
            index_api,
            params=params,
            headers={"user-agent": AGENT, "accept": "application/json"},
        )
        if response.status_code == 200:
            jobs = response.content.decode("utf-8").splitlines()
            for job in jobs:
                jobs_queue.put(json.loads(job))
        else:
            logger.warning(
                f"Failed to fetch jobs from {response.url}. Status code: {response.status_code}."
            )
        time.sleep(random() * 2)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch {index_api}: {e}")


async def find_pdf_url_from_html(html_content, base_url):
    soup = BeautifulSoup(html_content, "html.parser")
    pdf_urls = []

    # Handle file viewer link
    link_pattern = r"=(https?://[^\s]*?\.pdf)"
    match = re.search(link_pattern, base_url)
    if match:
        pdf_urls.append(match.group(1))

    # Handle link in href tag
    for link in soup.find_all("a", href=True):
        href = link["href"]

        if href.endswith(".pdf"):
            if href.startswith(("http://", "https://")):
                pdf_urls.append(href)
            else:
                # Convert relative url to absolute url
                pdf_urls.append(urljoin(base_url, href))

    return pdf_urls


async def fetch_job_pdf(session, job):
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(is_retryable_error),
        ):
            with attempt:
                async with session.get(
                    job["url"], timeout=10, headers={"user-agent": AGENT}
                ) as response:
                    pdf_bytes = await response.read()
                    is_duplicate = await mv.is_duplicate(job["url"], len(pdf_bytes))
                    if is_duplicate:
                        counter["empty_or_duplicate"] += 1
                    else:
                        vector_data = await vect.generate_embeddings_from_pdf_bytes(
                            pdf_bytes, job["url"], job["timestamp"]
                        )
                        inserted = await mv.insert_data(vector_data)

                        if inserted.get("insert_count", 0) > 0:
                            counter["success"] += 1
                            logger.info(
                                f"Successfully inserted {inserted['insert_count']} data of {job['url']}"
                            )
                        else:
                            counter["empty_or_duplicate"] += 1
    except Exception as e:
        counter["failed"] += 1
        failed[job["url"]] = str(e)
        logger.error(f"Failed to fetch pdf {job['url']}: {e}")


async def fetch_job_html(session, job):
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(is_retryable_error),
        ):
            with attempt:
                async with session.get(
                    job["url"], timeout=10, headers={"user-agent": AGENT}
                ) as response:
                    html = await response.text()
                    pdf_urls = await find_pdf_url_from_html(html, job["url"])
                    for pdf_url in pdf_urls:
                        await fetch_job_pdf(
                            session, {"url": pdf_url, "timestamp": job["timestamp"]}
                        )
    except Exception as e:
        logger.error(f"Failed to fetch html {job['url']}: {e}")


async def worker(session):
    while not jobs_queue.empty():
        job = jobs_queue.get()
        if "pdf" in job["mime"]:
            await fetch_job_pdf(session, job)
        elif "html" in job["mime"]:
            await fetch_job_html(session, job)
        jobs_queue.task_done()


async def run_workers(num_workers):
    async with aiohttp.ClientSession() as session:
        tasks = [worker(session) for _ in range(num_workers)]
        await asyncio.gather(*tasks)


def test():
    jobs = [
        {
            "timestamp": "20241204155509",
            "url": "https://liftoff.energy.gov/vpp/",
            "mime": "text/html",
        },
        {
            "timestamp": "20241213055142",
            "url": "https://repository.library.noaa.gov/pdfjs/web/viewer.html?file=https://repository.library.noaa.gov/view/noaa/48516/noaa_48516_DS1.pdf",
            "mime": "text/html",
        },
        {
            "timestamp": "20250324122124",
            "url": "https://www.36thdistrictcourtmi.gov/docs/default-source/general-information/ncsc-reports/admin_ncsc-d36-final-report_-20140612.pdf?sfvrsn=2",
            "mime": "application/pdf",
        },
    ]

    for job in jobs:
        jobs_queue.put(job)

    try:
        num_workers = cpu_count() * 2
    except NotImplementedError:
        num_workers = 4

    asyncio.run(run_workers(num_workers=num_workers))
    mv.reindex()
    logger.info(counter)
    logger.info(json.dumps(failed))


def main():
    parser = argparse.ArgumentParser(description="CommonCrawl PDF Scraper")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--index-name",
        type=str,
        required=False,
        help="CC index name to use. Format: CC-MAIN-YYYY-WW",
    )

    args = parser.parse_args()

    if args.index_name:
        raw_index_name = args.index_name
    else:
        current_week = datetime.now().isocalendar()[1]
        current_year = datetime.now().isocalendar()[0]
        raw_index_name = f"CC-MAIN-{current_year}-{current_week}"

    index_name = get_index_api(raw_index_name)

    if index_name:
        logger.info(f"Processing index {index_name}")
        for url in URLS:
            num_pages = get_num_pages(index_name, url)
            if num_pages:
                for page in range(num_pages):
                    logger.info(f"Processing page {page}/{num_pages - 1} for {url}")
                    for mime_type in MIME_TYPES:
                        get_jobs_from_url(
                            index_name, url, page, format=mime_type, limit=10
                        )
        try:
            num_workers = cpu_count() * 2
        except NotImplementedError:
            num_workers = 4

        asyncio.run(run_workers(num_workers=num_workers))
        mv.reindex()
        logger.info(counter)
        logger.info(json.dumps(failed))
    else:
        logger.warning(f"Index {raw_index_name} not found.")


if __name__ == "__main__":
    # test()

    main()
