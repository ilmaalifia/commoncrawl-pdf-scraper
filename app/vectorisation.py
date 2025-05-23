import os
import uuid

import fitz
from app.scraping import RecordJob
from app.utils import NUMBER_OF_PAGES_TO_CHECK, SIMILARITY_THRESHOLD, setup_logger
from dotenv import load_dotenv
from langchain_community.utils.math import cosine_similarity
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from pymupdf import Page

load_dotenv()

logger = setup_logger(__name__)


class Vectorisation:
    def __init__(self):
        self.model_embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("DENSE_MODEL"),
        )
        self.splitter = SentenceTransformersTokenTextSplitter(
            model_name=os.getenv("DENSE_MODEL"),
        )

    def topic_checking(self, topics: list[str], doc: list[Page]) -> bool:
        # Pass checking if no topic is provided
        if not topics:
            return True

        pages = [" ".join([page.get_text() for page in doc])]
        pages_embedding = self.model_embeddings.embed_documents(pages)
        topic_embeddings = self.model_embeddings.embed_documents(topics)
        similarities = cosine_similarity(topic_embeddings, pages_embedding)

        for topic, similarity in zip(topics, similarities):
            logger.info(f"Similarity for topic '{topic}': {similarity[0]}")
            if similarity[0] >= SIMILARITY_THRESHOLD:
                return True
        return False

    def generate_vector_from_pdf_bytes(self, topics: list[str], job: RecordJob):
        if not job["pdf_bytes"]:
            return []

        chunks = []
        data = []

        try:
            with fitz.open(stream=job["pdf_bytes"], filetype="pdf") as doc:
                if not self.topic_checking(topics, doc[:NUMBER_OF_PAGES_TO_CHECK]):
                    return []

                def document_generator():
                    for page in doc:
                        yield Document(
                            page_content=page.get_text(),
                            metadata={
                                "source": job["url"],
                                "page": page.number + 1,
                                "total_size": len(job["pdf_bytes"]),
                            },
                        )

                chunks = self.splitter.split_documents(document_generator())
        except Exception as e:
            logger.error(f"Error opening PDF: {e}")
            return []

        dense_vectors = self.model_embeddings.embed_documents(
            [chunk.page_content for chunk in chunks]
        )
        for chunk, dense_vector in zip(chunks, dense_vectors):
            data.append(
                {
                    "id": str(uuid.uuid4()),
                    "dense_vector": dense_vector,
                    "text": chunk.page_content,
                    "source": chunk.metadata["source"],
                    "page": chunk.metadata["page"],
                    "total_size": chunk.metadata["total_size"],
                    "timestamp": int(job["timestamp"]),
                }
            )
        return data
