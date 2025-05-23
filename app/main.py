import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from queue import Empty, Queue
from threading import Lock, Thread, get_ident
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
job_queue = Queue()
s3_reader = S3Reader(job_queue)
scraping = Scraping(session)
vectorisation = Vectorisation()
milvus = Milvus()
counter = {"success": 0, "failed": 0, "empty": 0, "duplicate": 0}
failed = {}
counter_lock = Lock()
failed_lock = Lock()


def pipeline_worker(
    topics: List[str],
):
    while True:
        try:
            warc_job = job_queue.get(timeout=20)
            if warc_job is None:
                break
            logger.info(
                f"[Thread ID {get_ident()}] Processing {warc_job['url']} with mime type {warc_job['mime_type']}"
            )
            records = [scraping.process_warc_record(warc_job)]
            if records[0] and records[0].get("pdf_urls", []):
                records.extend(
                    scraping.process_pdf_urls(records[0].get("pdf_urls", []))
                )

            for record in records:
                if record.get("pdf_bytes"):
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
                        counter["empty"] += 1
        except Empty:
            logger.info(f"[Thread ID {get_ident()}] Queue empty, waiting for jobs ...")
            continue
        except Exception as e:
            logger.error(f"[Thread ID {get_ident()}] Error processing: {e}")
            with counter_lock:
                counter["failed"] += 1
            with failed_lock:
                failed[warc_job.get("url")] = str(e)
        finally:
            job_queue.task_done()
            logger.info(
                f"[Thread ID {get_ident()}] Success: {counter['success']}, Failed: {counter['failed']}, Empty: {counter['empty']}, Duplicate: {counter['duplicate']}"
            )


def save_dict_as_json(data: dict, filename: str):
    try:
        current_dir = os.getcwd()
        path = os.path.join(current_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except TypeError as e:
        raise TypeError(f"Data contains non-serializable objects: {e}")
    except Exception as e:
        raise IOError(f"Failed to write JSON file: {e}")


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="PDF Scraper")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--index",
        type=str,
        required=False,
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

    try:
        s3_path = athena_index_query.run(index)

        loader = Thread(target=s3_reader.run, args=(s3_path,))
        loader.start()

        num_workers = min(10, os.cpu_count() * 2)
        workers = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for _ in range(num_workers):
                executor.submit(pipeline_worker, topics)

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
        save_dict_as_json(dict(failed), "failed_urls.json")
