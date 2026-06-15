import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.http import models

class QdrantConfig:
    def __init__(self):
        load_dotenv()
        protocol = os.getenv("QDRANT_PROTOCOL", "http")
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))

        self.__qdrant_url = f"{protocol}://{host}:{port}"
        self.__collection_name = os.getenv("COLLECTION_NAME")
        self.__embedding = os.getenv("QDRANT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.__embedding_dimension = int(os.getenv("QDRANT_EMBEDDING_DIMENSION", 384)) # Dimension must match the embedding dimension of the model used for vectorization, which for default value (all-MiniLM-L6-v2) is 384.
        self.__client = QdrantClient(url=self.__qdrant_url)
        self.__vector_store = None
        self._ensure_subject_index()


    def create_collection(self):
        collection_names = [c.name for c in self.__client.get_collections().collections]

        if self.__collection_name not in collection_names:
            self.__client.create_collection(
                collection_name=self.__collection_name,
                vectors_config=VectorParams(
                    size=self.__embedding_dimension,
                    distance=Distance.COSINE
                )
            )

    def get_client(self):
        return self.__client
    
    def get_embedding_model(self):
        return self.__embedding
    
    def get_collection(self):
        return self.__collection_name
    
    def get_vector_store(self) -> QdrantVectorStore:
        if self.__vector_store is None:
            embedding = HuggingFaceEmbeddings(model_name=self.__embedding)
            self.__vector_store = QdrantVectorStore.from_existing_collection(
                embedding=embedding,
                url=self.__qdrant_url,
                collection_name=self.__collection_name,
            )
        return self.__vector_store


    def _ensure_subject_index(self):
        """
        Qdrant requires a keyword index on a field before it can be used in facet queries.
        Safe to call repeatedly — skips creation if the index already exists.
        """
        collection_info = self.__client.get_collection(self.__collection_name)
        indexed_fields = collection_info.payload_schema or {}

        if "metadata.subject" not in indexed_fields:
            self.__client.create_payload_index(
                collection_name=self.__collection_name,
                field_name="metadata.subject",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
