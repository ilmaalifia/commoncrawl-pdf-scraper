import os
import uuid

import fitz
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import setup_logger

load_dotenv()

logger = setup_logger(__name__)


class Vectorisation:
    def __init__(self):
        self.model_embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("DENSE_MODEL"),
        )

    def generate_embedding(self, data):
        return self.model_embeddings.embed_query(data)

    async def generate_embeddings_from_pdf_bytes(
        self, pdf_bytes: bytes | bytearray, url, timestamp
    ):
        if not pdf_bytes:
            return []

        chunks = []
        data = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:

                def document_generator():
                    for page in doc:
                        yield Document(
                            page_content=page.get_text(),
                            metadata={
                                "source": url,
                                "page": page.number + 1,
                                "total_size": len(pdf_bytes),
                            },
                        )

                chunks = splitter.split_documents(document_generator())

        except Exception as e:
            logger.error(f"Error opening PDF: {e}")
            return []

        for chunk in chunks:
            data.append(
                {
                    "id": str(uuid.uuid4()),
                    "dense_vector": self.generate_embedding(chunk.page_content),
                    "text": chunk.page_content,
                    "source": chunk.metadata["source"],
                    "page": chunk.metadata["page"],
                    "total_size": chunk.metadata["total_size"],
                    "timestamp": (
                        timestamp if isinstance(timestamp, int) else int(timestamp)
                    ),
                }
            )

        return data
