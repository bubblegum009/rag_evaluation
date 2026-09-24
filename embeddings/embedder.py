"""
Embedding utilities for the RAG Experiment Lab.

This module is responsible for converting text into dense
vector representations using the configured embedding model.

The current model is BAAI/bge-small-en-v1.5.

The module provides separate methods for:
    - document/passages
    - user queries

This distinction is important because BGE is used for
query-to-passage retrieval.
"""

from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSION,
)


class Embedder:
    """
    Wrapper around the configured Sentence Transformer model.
    """

    def __init__(self):
        """
        Load the embedding model and verify its dimension.
        """

        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        actual_dimension = (
            self.model.get_sentence_embedding_dimension()
        )

        if actual_dimension != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding dimension mismatch. "
                f"Expected {EMBEDDING_DIMENSION}, "
                f"but model produces {actual_dimension}."
            )

    def embed_documents(self, texts):
        """
        Generate embeddings for document passages.

        Args:
            texts: List of document/passages.

        Returns:
            List/array of dense embeddings.
        """

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def embed_query(self, query):
        """
        Generate an embedding for a user query.

        BGE recommends a query instruction for
        short-query-to-passage retrieval.

        Args:
            query: User search query.

        Returns:
            Dense query embedding.
        """

        query_instruction = (
            "Represent this sentence for searching "
            "relevant passages: "
        )

        query_with_instruction = (
            query_instruction + query
        )

        return self.model.encode(
            query_with_instruction,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )