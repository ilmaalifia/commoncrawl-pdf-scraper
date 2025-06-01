import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from queue import Queue
from threading import Lock, Semaphore, Thread, get_ident
from typing import List

import boto3
from app.athena import AthenaIndexQuery
from app.milvus import Milvus
from app.s3_reader import S3Reader
from app.scraping import Scraping
from app.utils import REGION_NAME, setup_logger
from app.vectorisation import Vectorisation
from dotenv import load_dotenv

load_dotenv()
logger = setup_logger(__name__)

session = boto3.Session(region_name=REGION_NAME)
athena_index_query = AthenaIndexQuery(session)
job_queue = Queue(maxsize=10_000)
s3_reader = S3Reader(job_queue)
scraping = Scraping(session)
vectorisation = Vectorisation()
milvus = Milvus()
counter = {"success": 0, "failed": 0, "empty": 0, "duplicate": 0}
counter_lock = Lock()
MAX_WORKERS = 64
VECTOR_WORKERS = 1
vector_semaphore = Semaphore(VECTOR_WORKERS)


def pipeline_worker(
    topics: List[str],
):
    while True:
        try:
            warc_job = job_queue.get()

            if warc_job is None:
                logger.info(f"[Thread ID {get_ident()}] Received sentinel, exiting.")
                break

            records = [scraping.process_warc_record(warc_job)]
            if records[0] and records[0].get("pdf_urls", []):
                records.extend(
                    scraping.process_pdf_urls(records[0].get("pdf_urls", []))
                )

            for record in records:
                pdf_bytes = record.get("pdf_bytes")
                failed = record.get("failed")
                if pdf_bytes:
                    if not milvus.is_duplicate(record.get("url"), len(pdf_bytes)):
                        with vector_semaphore:
                            vector_data = vectorisation.generate_vector_from_pdf_bytes(
                                topics=topics, job=record
                            )
                            if vector_data:
                                inserted = milvus.insert_data(vector_data)
                                if inserted.get("insert_count", 0) > 0:
                                    with counter_lock:
                                        counter["success"] += 1
                                    logger.info(
                                        f"[Thread ID {get_ident()}] Successfully inserted {inserted['insert_count']} data of {record['url']}"
                                    )
                                else:
                                    with counter_lock:
                                        counter["empty"] += 1
                    else:
                        with counter_lock:
                            counter["duplicate"] += 1
                elif failed:
                    logger.info(
                        f"[Thread ID {get_ident()}] Failed here {warc_job.get('url')}: {failed}"
                    )
                    raise Exception(failed)
                else:
                    with counter_lock:
                        counter["empty"] += 1
        except Exception as e:
            logger.error(f"[Thread ID {get_ident()}] Error processing: {e}")
            with counter_lock:
                counter["failed"] += 1
        finally:
            job_queue.task_done()
            logger.info(
                f"[Thread ID {get_ident()}] Task done for {warc_job.get('url')}"
            )
            logger.info(
                f"[Thread ID {get_ident()}] Success: {counter['success']}, Failed: {counter['failed']}, Empty: {counter['empty']}, Duplicate: {counter['duplicate']}"
            )


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="PDF Scraper")
    parser.add_argument(
        "--index",
        type=str,
        required=True,
        help='CC index name to use with format CC-MAIN-<YYYY>-<WW>. Example: --index="CC-MAIN-2025-13"',
    )
    parser.add_argument(
        "--topic",
        action="append",
        default=[],
        required=True,
        help='Topic to match with PDF document content. Example: --topic="virtual power plant" --topic="vertical farming"',
    )

    args = parser.parse_args()

    if args.index:
        index = args.index
    else:
        current_week = datetime.now().isocalendar()[1]
        current_year = datetime.now().isocalendar()[0]
        index = f"CC-MAIN-{current_year}-{current_week}"

    topics = args.topic

    logger.info(f"Processing index: {index}")
    try:
        status = athena_index_query.update_index()
        logger.info(f"Index status: {status}")
        s3_path = athena_index_query.run(index)
        logger.info(f"Index path: {s3_path}")

        loader = Thread(target=s3_reader.run, args=(s3_path,))
        loader.start()

        num_workers = min(MAX_WORKERS, os.cpu_count() * 4)
        workers = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(pipeline_worker, topics) for _ in range(num_workers)
            ]
            loader.join()
            for _ in range(num_workers):
                job_queue.put(None)  # Put sentinel values to stop workers
            job_queue.join()  # Wait for all tasks to be marked as done
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        logger.info(
            f"Success: {counter['success']}, Failed: {counter['failed']}, Empty: {counter['empty']}, Duplicate: {counter['duplicate']}"
        )
        logger.info(f"Total scanned: {sum(counter.values())}")
        logger.info(f"Total running time: {elapsed:.2f} seconds")
