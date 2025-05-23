import logging
import sys

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
NUMBER_OF_PAGES_TO_CHECK = 5
SIMILARITY_THRESHOLD = 0.1
MIME_TYPES = ["application/pdf", "text/pdf", "text/html"]
REGION_NAME = "us-east-1"


def setup_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel("INFO")

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
