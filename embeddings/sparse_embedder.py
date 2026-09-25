"""
Sparse embedding utilities for the RAG Experiment Lab.

This module generates BM25 sparse vector representations
using the Qdrant BM25 model through FastEmbed.

Dense embeddings are handled separately by embedder.py.
"""

import logging

from fastembed import SparseTextEmbedding

from config import BM25_MODEL_NAME


logger = logging.getLogger("rag_experiment")


class SparseEmbedder:
    """
    Wrapper around the BM25 sparse embedding model.
    """

    def __init__(self):
        """
        Load the configured BM25 sparse embedding model.
        """

        logger.info(
            "Loading sparse embedding model: %s",
            BM25_MODEL_NAME
        )

        self.model = SparseTextEmbedding(
            model_name=BM25_MODEL_NAME
        )

        logger.info(
            "Sparse embedding model loaded."
        )

    def embed_documents(self, texts):
        """
        Generate sparse BM25 embeddings for documents.

        Args:
            texts: List of document/chunk texts.

        Returns:
            List of sparse embedding objects.
        """

        embeddings = list(
            self.model.embed(texts)
        )

        logger.info(
            "Generated sparse embeddings for %d documents.",
            len(embeddings)
        )

        return embeddings

    def embed_query(self, query):
        """
        Generate a sparse BM25 embedding for a query.

        Args:
            query: User search query.

        Returns:
            Sparse embedding object.
        """

        embedding = next(
            self.model.embed([query])
        )

        logger.info(
            "Generated sparse query embedding."
        )

        return embedding