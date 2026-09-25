"""
Qdrant vector store utilities for the RAG Experiment Lab.

This module handles storage and retrieval of both dense and
sparse BM25 vectors in Qdrant.

The collection contains two named vectors:

    dense
        384-dimensional BGE embedding.

    bm25
        Sparse BM25 embedding.

Both vectors belong to the same Qdrant point and share the
same document metadata and chunk ID.

The store also ensures that:
    - The Qdrant collection exists.
    - Required payload indexes exist.
"""

import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    PayloadSchemaType,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams
)

from config import (
    EMBEDDING_DIMENSION,
    QDRANT_API_KEY,
    QDRANT_URL,
    QDRANT_COLLECTION_NAME
)


logger = logging.getLogger("rag_experiment")


class QdrantStore:
    """
    Wrapper around the Qdrant Cloud collection.
    """

    def __init__(self):
        """
        Initialize the Qdrant client and ensure that
        the collection and required indexes exist.
        """

        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )

        self._ensure_collection()
        self._ensure_payload_indexes()

    def _ensure_collection(self):
        """
        Create the Qdrant collection if it does not exist.

        The collection contains:
            - dense: 384-dimensional BGE embeddings
            - bm25: sparse BM25 embeddings
        """

        collections = self.client.get_collections()

        collection_names = [
            collection.name
            for collection in collections.collections
        ]

        if QDRANT_COLLECTION_NAME in collection_names:
            logger.info(
                "Qdrant collection already exists: %s",
                QDRANT_COLLECTION_NAME
            )
            return

        logger.info(
            "Creating Qdrant collection: %s",
            QDRANT_COLLECTION_NAME
        )

        self.client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                "bm25": SparseVectorParams(
                    modifier=Modifier.IDF
                )
            }
        )

        logger.info(
            "Created Qdrant collection: %s",
            QDRANT_COLLECTION_NAME
        )

    def _ensure_payload_indexes(self):
        """
        Ensure required payload indexes exist.

        document_id is indexed as a keyword because it is used
        for filtering and document-level retrieval.
        """

        collection_info = self.client.get_collection(
            collection_name=QDRANT_COLLECTION_NAME
        )

        payload_schema = collection_info.payload_schema

        if "document_id" in payload_schema:
            logger.info(
                "Payload index already exists for document_id."
            )
            return

        logger.info(
            "Creating payload index for document_id."
        )

        self.client.create_payload_index(
            collection_name=QDRANT_COLLECTION_NAME,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD
        )

        logger.info(
            "Created payload index for document_id."
        )

    def document_exists(self, document_id):
        """
        Check whether a document is already indexed.
        """

        result = self.client.count(
            collection_name=QDRANT_COLLECTION_NAME,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(
                            value=document_id
                        )
                    )
                ]
            ),
            exact=True
        )

        exists = result.count > 0

        logger.info(
            "Document %s exists: %s",
            document_id,
            exists
        )

        return exists

    def upsert_documents(
        self,
        chunks,
        dense_embeddings,
        sparse_embeddings,
        batch_size=64
    ):
        """
        Upload chunks with both dense and sparse vectors.

        Args:
            chunks:
                List of LangChain Document objects.

            dense_embeddings:
                Dense BGE embeddings.

            sparse_embeddings:
                Sparse BM25 embeddings.

            batch_size:
                Number of points uploaded per request.
        """

        points = []

        for index in range(len(chunks)):

            chunk = chunks[index]

            chunk_id = chunk.metadata["chunk_id"]

            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    chunk_id
                )
            )

            sparse_embedding = sparse_embeddings[index]

            payload = {
                "text": chunk.page_content,
                **chunk.metadata
            }

            point = PointStruct(
                id=point_id,
                vector={
                    "dense": dense_embeddings[index].tolist(),
                    "bm25": {
                        "indices": sparse_embedding.indices.tolist(),
                        "values": sparse_embedding.values.tolist()
                    }
                },
                payload=payload
            )

            points.append(point)

        logger.info(
            "Prepared %d points for Qdrant.",
            len(points)
        )

        for start in range(
            0,
            len(points),
            batch_size
        ):

            batch = points[
                start:start + batch_size
            ]

            self.client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=batch
            )

            logger.info(
                "Uploaded Qdrant points %d-%d.",
                start,
                start + len(batch) - 1
            )

        logger.info(
            "Successfully uploaded %d points.",
            len(points)
        )

    def search_dense(
        self,
        query_embedding,
        top_k
    ):
        """
        Search using the dense BGE vector.
        """

        results = self.client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_embedding.tolist(),
            using="dense",
            limit=top_k,
            with_payload=True
        )

        logger.info(
            "Dense retrieval returned %d results.",
            len(results.points)
        )

        return results

    def search_bm25(
        self,
        query_embedding,
        top_k=5
    ):
        """
        Search using the BM25 sparse vector.
        """

        query_vector = SparseVector(
            indices=query_embedding.indices.tolist(),
            values=query_embedding.values.tolist()
        )

        results = self.client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_vector,
            using="bm25",
            limit=top_k,
            with_payload=True
        )

        logger.info(
            "BM25 retrieval returned %d results.",
            len(results.points)
        )

        return results

    def get_document_chunks(
        self,
        document_id
    ):
        """
        Retrieve all chunks belonging to a document.

        Chunks are sorted by chunk ID.
        """

        results = self.client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(
                            value=document_id
                        )
                    )
                ]
            ),
            limit=1000,
            with_payload=True,
            with_vectors=False
        )

        points = results[0]

        points.sort(
            key=lambda point: point.payload.get(
                "chunk_id",
                ""
            )
        )

        logger.info(
            "Retrieved %d chunks for document %s.",
            len(points),
            document_id
        )

        return points