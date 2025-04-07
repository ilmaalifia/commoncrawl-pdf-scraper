from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient


class Milvus:
    def __init__(self):
        self.client = MilvusClient("./milvus_demo.db")
        self.collection_name = "pdf"

        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=CollectionSchema(
                    fields=[
                        FieldSchema(
                            name="id",
                            dtype=DataType.VARCHAR,
                            is_primary=True,
                            max_length=100,
                        ),
                        FieldSchema(
                            name="vector",
                            dtype=DataType.FLOAT_VECTOR,
                            dim=384,
                            description="Vector embeddings of the current snippet",
                        ),  # all-MiniLM-L6-v2 dimension
                        FieldSchema(
                            name="text",
                            dtype=DataType.VARCHAR,
                            max_length=4000,
                            description="Text of the current snippet",
                        ),
                        FieldSchema(
                            name="source",
                            dtype=DataType.VARCHAR,
                            max_length=256,
                            description="Source link of the PDF document",
                        ),
                        FieldSchema(
                            name="page",
                            dtype=DataType.INT64,
                            description="Page number of the current snippet",
                        ),
                        FieldSchema(
                            name="total_size",
                            dtype=DataType.INT64,
                            description="Total size of the entire PDF document in bytes",
                        ),
                        FieldSchema(
                            name="timestamp",
                            dtype=DataType.INT64,
                            description="Timestamp stored as Unix epoch time in milliseconds",
                        ),
                    ],
                    description="Vector embeddings of all collected PDF documents from internet",
                ),
                consistency_level="Strong",
            )

    async def insert_data(self, data):
        return self.client.insert(collection_name=self.collection_name, data=data)

    def reindex(self):
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 1024},
        )
        index_params.add_index(
            field_name="source",
        )
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params,
        )
        self.client.refresh_load(collection_name=self.collection_name)

    def search(self, query_embedding, top_k: int = 3):
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["text", "source", "page", "timestamp"],
            search_params={"metric_type": "COSINE"},
        )

        return_results = []
        for hits in results:
            for hit in hits:
                return_results.append(
                    {
                        "score": hit["distance"],
                        "text": hit["entity"]["text"],
                        "page": hit["entity"]["page"],
                        "source": hit["entity"]["source"],
                        "timestamp": hit["entity"]["timestamp"],
                        "total_size": hit["entity"]["total_size"],
                    }
                )
        return return_results

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

    def delete_by_source(self, source):
        return self.client.delete(
            collection_name=self.collection_name,
            filter=f'source=="{source}"',
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


if __name__ == "__main__":
    milvus = Milvus()
    print(milvus.get_collection_stats())
    print(milvus.list_indexes())
    print(milvus.query_by_source(""))
