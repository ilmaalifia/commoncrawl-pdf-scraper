from multiprocessing import Queue

import pyarrow.dataset as ds
from app.utils import setup_logger
from dotenv import load_dotenv
from pyarrow.fs import FileSelector, FileType, S3FileSystem

load_dotenv()
logger = setup_logger(__name__)


class S3Reader:
    def __init__(self, queue: Queue):
        self.queue = queue
        self.s3 = S3FileSystem()

    def run(self, s3_path: str):
        if s3_path.startswith("s3://"):
            s3_path = s3_path[5:]
        files_info = self.s3.get_file_info(FileSelector(s3_path))

        def is_data_file(file_info):
            if file_info.type == FileType.File and (
                file_info.path.endswith(".csv") or file_info.path.endswith(".metadata")
            ):
                return False
            return True

        data_files = [f.path for f in files_info if is_data_file(f)]

        dataset = ds.dataset(data_files, format="parquet", filesystem=self.s3)
        scanner = dataset.scanner(batch_size=1000)
        for batch in scanner.to_batches():
            for record in batch.to_pylist():
                try:
                    self.queue.put(record)
                except Exception as e:
                    logger.error(f"Failed to put record in queue: {e}")
        logger.info("Done pushing all indexes to queue")
