import os
import uuid
from config.qdrant_config import QdrantConfig
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http import models
from qdrant_client.models import Filter
from utils.file_utils import file_exists
from utils.language_detector import detect_language

class VectorDBRepository:
    def __init__(self):
        # Use high-resolution parsing to capture more semantic details, which can improve embedding quality and retrieval relevance.
        self.parsing_strategy = os.getenv("VECTOR_DB_PARSING_STRATEGY", "hi_res")
        self.config = QdrantConfig()
        self.config.create_collection()
        self.retriever = self.config.get_vector_store().as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": 0.7,
                "k": 5,
            },
        )


    def ingest_document(
        self, 
        session_id: str, 
        subject: str,
        docs_path: list[str]
    ):
        """
        Load PDF with Unstructured, chunk intelligently, attach tenant/document metadata, store in Qdrant.

        Args:
            session_id: str - Tenant identifier for multi-tenancy isolation.
            subject: str - Field for the document subject to include in the metadata 
            docs_path: list[str] - List of file paths to the documents to ingest.

        Returns:
            None
        """
        document_id = str(uuid.uuid4())
        for file_path in docs_path:
            if not file_exists(file_path):
                print(f"INFO - File {file_path} does not exist. Skipping.")
                continue

            print(f"INFO - Ingesting document: {file_path} for user: {session_id}")

            languages = detect_language(file_path) # Detect language to improve Unstructured parsing accuracy

            # Parse document structure

            loader = UnstructuredLoader(
                file_path=file_path,

                # split the document into elements preserve structure (paragraphs, headings, etc.) and 
                # return those as individual langchain Document objects for better chunking and retrieval
                chunking_strategy="by_title",
                
                strategy=self.parsing_strategy,
                
                # Use the detected language to improve parsing accuracy
                languages=languages
            )

            print(f"INFO - loader created for file: {file_path}, now loading document...")

            docs = loader.load()

            print(f"INFO - Document loaded: {file_path}, now chunking...")

            # Chunk while preserving semantics
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=500
            )

            chunks = splitter.split_documents(docs)

            print(f"INFO - Document split into {len(chunks)} chunks, now adding metadata and storing in Qdrant: {file_path}")

            # Inject multi-tenant metadata
            for doc in chunks:
                doc.metadata["session_id"] = session_id # Tenant isolation is metadata-based.
                doc.metadata["subject"] = subject
                doc.metadata["document_id"] = document_id
                doc.metadata["source_file"] = file_path
            # Store in Qdrant
            self.config.get_vector_store().add_documents(chunks)

            print(f"INFO - Document ingested and stored in Qdrant: {file_path} for user: {session_id}")


    def retrieve(
        self,
        query: str,
        k=5
    ) -> list[Document]:
        """
        Retrieve relevant document chunks from Qdrant based on query, with tenant and optional subject filtering.

        Args:
            query: str - User's search query.
            k: int - Number of top results to retrieve.

        Returns:
            List of relevant document chunks as strings.
        """
        
        results = self.retriever.invoke(query)
        if len(results) > k:
            results = results[:k]

        return results


    def get_collection_topics(
        self,
        session_id: str
    ) -> list[Document]:
        records, next_page_offset = self.config.get_client().scroll(
            collection_name=self.config.get_collection(),
            scroll_filter= Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.session_id",
                        match=models.MatchValue(value=session_id),
                    )
                ]
            ),
            limit=100,          # Adjust depending on collection size
            with_payload=True,  
            with_vectors=False   # No need to download heavy vectors for topic modeling
        )

        documents = [
            Document(page_content=record.payload["page_content"]) 
            for record in records if "page_content" in record.payload
        ]
        return documents


    def list_subjects(self, session_id: str) -> list[str]:
        """
        Return all distinct subject names ingested under this session.

        Uses the Qdrant Facet API to retrieve unique values of metadata.subject 
        filtered by metadata.session_id, without scanning all points.

        Args:
            session_id: str - Tenant identifier to scope the query.

        Returns:
            Sorted list of subject name strings.
        """

        _client = self.config.get_client()
        _collection = self.config.get_collection()

        response = _client.facet(
            collection_name=_collection,
            key="metadata.subject",
            facet_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.session_id",
                        match=models.MatchValue(value=session_id),
                    )
                ]
            ),
            # limit of subjects for the same session_id
            limit=200,
            exact=True
        )

        return sorted(hit.value for hit in response.hits)
