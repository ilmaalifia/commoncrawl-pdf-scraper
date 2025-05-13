import argparse
import asyncio
import json
import os
import random
import re
from datetime import datetime
from multiprocessing import cpu_count
from urllib.parse import urljoin

import aiohttp
import requests
from bs4 import BeautifulSoup
from src.milvus import Milvus
from src.utils import USER_AGENT, setup_logger
from tenacity import AsyncRetrying, retry, stop_after_attempt, wait_exponential
from vectorisation import Vectorisation

logger = setup_logger(__name__)
INDEX_SERVER = "http://index.commoncrawl.org"
MIME_TYPES = [
    "pdf",
    "html",
]
CC_PAGE_SIZE = int(os.getenv("CC_PAGE_SIZE", "10"))
TEST = os.getenv("TEST", "false").lower() == "true"

pagination_url_queue = asyncio.Queue()
absolute_url_queue = asyncio.Queue()
vect = Vectorisation()
milvus = Milvus()
counter = {"success": 0, "failed": 0, "empty": 0, "duplicate": 0}
failed = {}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
def get_index_api(index_name):
    session = requests.Session()
    r = session.get(urljoin(INDEX_SERVER, "collinfo.json")).json()
    collinfo = list(filter(lambda c: c["id"] == index_name, r))
    return collinfo[0]["cdx-api"] if collinfo else None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
def get_num_pages(index_api, url):
    pages = 1

    if TEST:
        return pages

    params = {
        "url": url,
        "pageSize": CC_PAGE_SIZE,
        "showNumPages": True,
    }

    try:
        session = requests.Session()
        r = session.get(index_api, params=params).json()
        pages = r.get("pages", 1)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch num pages {index_api}: {e}")
    finally:
        return pages


async def fetch_pagination_url(session, job):
    """
    Fetch list of absolute urls from the Common Crawl pagination index API and add them to the jobs queue.
    """
    jobs = []
    index_api = job.pop("index_api", None)

    logger.info(f"Fetching pagination url for {job['url']} page {job['page']}")
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
        ):
            with attempt:
                async with session.get(
                    index_api,
                    params=job,
                    headers={"user-agent": USER_AGENT, "accept": "application/json"},
                    timeout=10,
                ) as response:
                    jobs = await response.read()
                    if jobs:
                        jobs = jobs.splitlines()
                        random.shuffle(jobs)
                        for job in jobs:
                            await absolute_url_queue.put(json.loads(job))
    except aiohttp.ClientResponseError as e:
        logger.warning(
            f"Failed to fetch pagination url {e.request_info.url}: {e.status} {e.message}"
        )
    finally:
        await asyncio.sleep(5 + random.random() * 5)


async def fetch_absolute_url_pdf(session, job):
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
        ):
            with attempt:
                async with session.get(
                    job["url"], timeout=10, headers={"user-agent": USER_AGENT}
                ) as response:
                    pdf_bytes = await response.read()
                    is_duplicate = await milvus.is_duplicate(job["url"], len(pdf_bytes))
                    if is_duplicate:
                        counter["duplicate"] += 1
                    else:
                        vector_data = await vect.generate_embeddings_from_pdf_bytes(
                            pdf_bytes, job["url"], job["timestamp"]
                        )
                        inserted = await milvus.insert_data(vector_data)

                        if inserted.get("insert_count", 0) > 0:
                            counter["success"] += 1
                            logger.info(
                                f"Successfully inserted {inserted['insert_count']} data of {job['url']}"
                            )
                        else:
                            counter["empty"] += 1
    except Exception as e:
        counter["failed"] += 1
        failed[job["url"]] = str(e)
        logger.error(f"Failed to fetch pdf {job['url']}: {e}")
    finally:
        logger.info(
            f"Success: {counter['success']}, Failed: {counter['failed']}, Empty: {counter['empty']}, Duplicate: {counter['duplicate']}"
        )


async def fetch_absolute_url_html(session, job):
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
        ):
            with attempt:
                async with session.get(
                    job["url"], timeout=10, headers={"user-agent": USER_AGENT}
                ) as response:
                    response.raise_for_status()
                    html = await response.text()
                    pdf_urls = await find_pdf_url_from_html(html, job["url"])
                    for pdf_url in pdf_urls:
                        await fetch_absolute_url_pdf(
                            session, {"url": pdf_url, "timestamp": job["timestamp"]}
                        )
    except Exception as e:
        counter["failed"] += 1
        failed[job["url"]] = str(e)
        logger.error(f"Failed to fetch html {job['url']}: {e}")
    finally:
        logger.info(
            f"Success: {counter['success']}, Failed: {counter['failed']}, Empty: {counter['empty']}, Duplicate: {counter['duplicate']}"
        )


async def find_pdf_url_from_html(html_text, base_url):
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


async def pagination_producer(index_api, urls):
    logger.info(f"Starting pagination producer for {index_api}")
    for url in urls:
        num_pages = get_num_pages(index_api, url)
        logger.info(f"Number of pages for {url}: {num_pages}")
        if num_pages:
            for page in range(num_pages):
                for mime_type in MIME_TYPES:
                    await pagination_url_queue.put(
                        {
                            "index_api": index_api,
                            "url": url,
                            "filter": [f"~mime:.*/{mime_type}$", "=status:200"],
                            "output": "json",
                            "fl": "url,timestamp,mime",
                            "page": page,
                            "pageSize": CC_PAGE_SIZE,
                        }
                    )
    logger.info(f"Finished pagination producer for {index_api}")


async def pagination_url_consumer(session):
    logger.info(f"Starting pagination url consumer")
    while True:
        try:
            job = await pagination_url_queue.get()
            if job:
                await fetch_pagination_url(session, job)
                pagination_url_queue.task_done()
        except asyncio.CancelledError:
            break
    logger.info(f"Finished pagination url consumer")


async def absolute_url_consumer(session):
    logger.info(f"Starting absolute url consumer")
    while True:
        try:
            job = await absolute_url_queue.get()
            if job:
                logger.info(f"Processing absolute url: {job.get('url')}")
                if "pdf" in job.get("mime", ""):
                    await fetch_absolute_url_pdf(session, job)
                elif "html" in job.get("mime", ""):
                    await fetch_absolute_url_html(session, job)
                absolute_url_queue.task_done()
        except asyncio.CancelledError:
            break

    logger.info(f"Finished absolute url consumer")


async def run_workers(num_workers, index_api, urls):
    async with aiohttp.ClientSession() as session:
        await pagination_producer(index_api, urls)

        pagination_url_consumers = [
            asyncio.create_task(pagination_url_consumer(session))
        ]

        absolute_url_consumers = [
            asyncio.create_task(absolute_url_consumer(session))
            for _ in range(num_workers * 2)
        ]

        await pagination_url_queue.join()
        await absolute_url_queue.join()

        for c in pagination_url_consumers + absolute_url_consumers:
            c.cancel()

        await asyncio.gather(*pagination_url_consumers, *absolute_url_consumers)


def generate_num_workers():
    try:
        num_workers = cpu_count()
    except NotImplementedError:
        num_workers = 4
    return num_workers


def save_dict_as_json(data: dict, filename: str, indent: int = 4):
    try:
        current_dir = os.getcwd()
        path = os.path.join(current_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except TypeError as e:
        raise TypeError(f"Data contains non-serializable objects: {e}")
    except Exception as e:
        raise IOError(f"Failed to write JSON file: {e}")


def main():
    parser = argparse.ArgumentParser(description="PDF Scraper")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--index-name",
        type=str,
        required=False,
        help="CC index name to use with format CC-MAIN-<YYYY>-<WW>. Example: CC-MAIN-2025-13",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        required=True,
        help='URL pattern to scrape. Example: --url="*.gov" --url="*.com" --url="*.org"',
    )

    args = parser.parse_args()

    if args.index_name:
        raw_index_name = args.index_name
    else:
        current_week = datetime.now().isocalendar()[1]
        current_year = datetime.now().isocalendar()[0]
        raw_index_name = f"CC-MAIN-{current_year}-{current_week}"

    urls = args.url

    try:
        index_name = get_index_api(raw_index_name)
        if index_name:
            num_workers = generate_num_workers()
            asyncio.run(run_workers(num_workers, index_name, urls))
        else:
            logger.warning(f"Index {raw_index_name} is not found")
    finally:
        milvus.reindex()
        url_filename = "_".join(urls).replace("*", "").replace(".", "")
        save_dict_as_json(failed, f"failed_urls_{url_filename}.json")
        logger.info(
            f"Success: {counter['success']}, Failed: {counter['failed']}, Empty: {counter['empty']}, Duplicate: {counter['duplicate']}"
        )


if __name__ == "__main__":
    main()
