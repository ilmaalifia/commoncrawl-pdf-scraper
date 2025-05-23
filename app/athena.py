import os
import time
from datetime import datetime

import boto3
from app.utils import MIME_TYPES, setup_logger
from dotenv import load_dotenv

load_dotenv()
logger = setup_logger(__name__)
BUCKET_NAME = os.getenv("BUCKET_NAME")


class AthenaIndexQuery:
    def __init__(self, session: boto3.Session):
        self.athena = session.client("athena")

    def run(self, index_name="CC-MAIN-2025-18"):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        bucket_path = (
            f"s3://{BUCKET_NAME}/athena-index-query/{index_name}/{current_time}"
        )
        query = f"""
        UNLOAD (
            SELECT 
                url,
                content_mime_type AS mime_type,
                fetch_time AS timestamp,
                warc_filename AS filename,
                warc_record_length AS length,
                warc_record_offset AS offset
            FROM "ccindex"
            WHERE crawl = '{index_name}'
                AND subset = 'warc'
                AND content_languages = 'eng'
                AND fetch_status = 200
                AND content_mime_type IN ({", ".join(f"'{mime}'" for mime in MIME_TYPES)})
            {'LIMIT 10000' if os.getenv("TEST").lower() == "true" else ''}
        ) TO '{bucket_path}' WITH (format = 'PARQUET', compression = 'SNAPPY')
        """

        execution = self.athena.start_query_execution(
            QueryString=query,
            ResultConfiguration={"OutputLocation": f"{bucket_path}"},
        )
        execution_id = execution["QueryExecutionId"]

        while True:
            status = self.athena.get_query_execution(QueryExecutionId=execution_id)
            state = status["QueryExecution"]["Status"]["State"]

            if state in ["SUCCEEDED"]:
                return bucket_path
            elif state in ["FAILED", "CANCELLED"]:
                raise Exception(f"Query failed with status {status}")
            time.sleep(5)
