import uuid

import fitz
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class EmbeddingsTransformer:
    def __init__(self):
        self.model_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

    def generate_embedding(self, data):
        return self.model_embeddings.embed_query(data)

    async def generate_pdf_embeddings_from_bytes(self, pdf_bytes, url, timestamp):
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            documents = []
            for page in doc:
                documents.append(
                    Document(
                        page_content=page.get_text(),
                        metadata={
                            "source": url,
                            "page": page.number + 1,
                            "total_size": len(pdf_bytes),
                        },
                    )
                )

        chunks = splitter.split_documents(documents)

        del pdf_bytes
        del documents

        data = []

        for chunk in chunks:
            data.append(
                {
                    "id": str(uuid.uuid4()),
                    "vector": self.generate_embedding(chunk.page_content),
                    "text": chunk.page_content,
                    "source": chunk.metadata["source"],
                    "page": chunk.metadata["page"],
                    "total_size": chunk.metadata["total_size"],
                    "timestamp": int(timestamp),
                }
            )

        return data
