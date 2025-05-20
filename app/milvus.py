import os

from app.utils import setup_logger
from dotenv import load_dotenv
from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    MilvusClient,
)

load_dotenv()

logger = setup_logger(__name__)


class Milvus:
    def __init__(self):
        self.client = MilvusClient(
            uri=os.getenv("MILVUS_URI", "./milvus_demo.db"),
            token=os.getenv("MILVUS_TOKEN"),
        )
        self.collection_name = os.getenv("MILVUS_COLLECTION", "pdf")

        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=self.create_schema(),
            )
            self.create_index()

        state = self.client.get_load_state(collection_name=self.collection_name)

        logger.info(f"Collection state: {state.get('state')}")
        logger.info(f"Collection stats: {self.get_collection_stats()}")
        logger.info(f"Collection indexes: {self.list_indexes()}")

    async def insert_data(self, data):
        return self.client.insert(collection_name=self.collection_name, data=data)

    def create_schema(self):
        schema = CollectionSchema(
            fields=[
                FieldSchema(
                    name="id",
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=255,
                ),
                FieldSchema(
                    name="dense_vector",
                    description="Dense vector of the current snippet",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=768,
                ),
                FieldSchema(
                    name="sparse_vector",
                    description="Sparse vector embeddings of the current snippet",
                    dtype=DataType.SPARSE_FLOAT_VECTOR,
                ),
                FieldSchema(
                    name="text",
                    description="Text of the current snippet",
                    dtype=DataType.VARCHAR,
                    max_length=3000,
                    enable_analyzer=True,  # allows Milvus to tokenize text for sparse vectorisation
                    analyzer_params={"type": "english"},
                ),
                FieldSchema(
                    name="source",
                    description="Source link of the PDF document",
                    dtype=DataType.VARCHAR,
                    max_length=3000,
                ),
                FieldSchema(
                    name="page",
                    description="Page number of the current snippet",
                    dtype=DataType.INT64,
                ),
                FieldSchema(
                    name="total_size",
                    description="Total size of the entire PDF document in bytes",
                    dtype=DataType.INT64,
                ),
                FieldSchema(
                    name="timestamp",
                    description="Timestamp stored as Unix epoch time in milliseconds",
                    dtype=DataType.INT64,
                ),
            ],
            description="Vectors of all collected PDF documents from internet",
        )

        bm25_function = Function(
            name="bm25_function",
            input_field_names=["text"],
            output_field_names=["sparse_vector"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        return schema

    def create_index(self):
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            index_name="dense_index",
            field_name="dense_vector",
            index_type="IVF_FLAT",
            metric_type="COSINE",
        )
        index_params.add_index(
            index_name="sparse_index",
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
        index_params.add_index(
            field_name="source",
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params,
        )
        self.client.load_collection(collection_name=self.collection_name)
        logger.info(f"Creating index is completed")

    def dense_search(self, query_embedding, top_k: int = 3):
        return self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["text", "source", "page", "timestamp"],
            anns_field="dense_vector",
            search_params={"metric_type": "COSINE"},
        )

    def sparse_search(self, query):
        return self.client.search(
            collection_name=self.collection_name,
            data=[query],
            anns_field="sparse_vector",
            limit=10,
            search_params={"metric_type": "BM25"},
            output_fields=[
                "text",
                "page",
                "source",
                "timestamp",
            ],
        )

    def get_collection_stats(self):
        return self.client.get_collection_stats(self.collection_name)

    def list_indexes(self):
        return self.client.list_indexes(self.collection_name)

    def search_by_ids(self, ids):
        results = self.client.get(
            collection_name=self.collection_name,
            ids=ids,
            output_fields=[
                "text",
                "page",
                "source",
                "timestamp",
                "total_size",
            ],
        )
        return results

    def query_by_source(self, source):
        return self.client.query(
            collection_name=self.collection_name,
            filter=f'source=="{source}"',
            output_fields=[
                "page",
                "text",
                "source",
                "timestamp",
                "total_size",
            ],
        )

    async def is_duplicate(self, source, total_size):
        res = self.query_by_source(source)

        if (
            isinstance(res, list)
            and len(res) > 0
            and res[0].get("total_size") == total_size
        ):
            return True
        return False

    def clean(self):
        self.client.release_collection(collection_name=self.collection_name)
        state = self.client.get_load_state(collection_name=self.collection_name)
        logger.info(f"Collection {self.collection_name} state: {state.get('state')}")


if __name__ == "__main__":
    milvus = Milvus()
