import argparse
import os
from datetime import datetime
from multiprocessing import Manager, Process, Queue, cpu_count
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


def pipeline_worker(job_queue: Queue, topics: List[str], counter: dict):
    logger = setup_logger(f"worker-{os.getpid()}")
    scraping = Scraping(boto3.Session(region_name=REGION_NAME))
    vectorisation = Vectorisation()
    milvus = Milvus()
    while True:
        try:
            warc_job = job_queue.get(timeout=20)
            if warc_job is None:
                break
            logger.info(
                f"Processing {warc_job['url']} with mime type {warc_job['mime_type']}"
            )
            records = [scraping.process_warc_record(warc_job)]
            if records[0] and records[0].get("pdf_urls", []):
                records.extend(
                    scraping.process_pdf_urls(records[0].get("pdf_urls", []))
                )

            for record in records:
                if record["pdf_bytes"]:
                    vector_data = vectorisation.generate_vector_from_pdf_bytes(
                        topics=topics, job=record
                    )
                    if vector_data:
                        inserted = milvus.insert_data(vector_data, warc_job["metadata"])
                        if inserted.get("insert_count", 0) > 0:
                            counter["success"] += 1
                            logger.info(
                                f"Successfully inserted {inserted['insert_count']} data of {record['url']}"
                            )
                        else:
                            counter["empty"] += 1
                else:
                    counter["empty"] += 1
        except Exception as e:
            logger.error(f"Error processing: {e}")
            counter["failed"] += 1
            # failed[warc_job["url"]] = str(e)
        finally:
            logger.info(
                f"Success: {counter['success']}, Failed: {counter['failed']}, Empty: {counter['empty']}, Duplicate: {counter['duplicate']}"
            )


if __name__ == "__main__":
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
    manager = Manager()
    counter = manager.dict({"success": 0, "failed": 0, "empty": 0, "duplicate": 0})
    job_queue = Queue()
    athena_index_query = AthenaIndexQuery(boto3.Session(region_name=REGION_NAME))
    s3_reader = S3Reader(job_queue)
    logger = setup_logger(__name__)
    try:
        # s3_path = athena_index_query.run(index)
        s3_path = "s3://mahardika-bucket/athena-index-query/CC-MAIN-2025-18/2025-05-23_00-56-06"

        loader = Process(target=s3_reader.run, args=(s3_path,))
        loader.start()

        workers = []
        num_workers = cpu_count() // 2
        for _ in range(num_workers):
            p = Process(target=pipeline_worker, args=(job_queue, topics, counter))
            p.start()
            workers.append(p)
        loader.join()
        for _ in range(num_workers):
            job_queue.put(None)  # Put sentinel values to signal the workers to exit
        for p in workers:
            p.join()
    finally:
        logger.info(
            f"Success: {counter['success']}, Failed: {counter['failed']}, Empty: {counter['empty']}, Duplicate: {counter['duplicate']}"
        )
        logger.info(f"Total scanned: {sum(counter.values())}")
