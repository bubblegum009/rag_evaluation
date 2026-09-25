"""
Sparse retrieval utilities for the RAG Experiment Lab.

This module performs BM25 retrieval using the sparse embedding
model and the BM25 sparse vector stored in Qdrant.

The flow is:

    User query
        ↓
    SparseEmbedder
        ↓
    BM25 sparse query vector
        ↓
    Qdrant BM25 search
        ↓
    Ranked chunks
"""

import logging
import time

from embeddings.sparse_embedder import SparseEmbedder
from retrieval.latency import RetrievalLatency


logger = logging.getLogger("rag_experiment")


def bm25_search(
    query,
    sparse_embedder,
    qdrant_store,
    top_k=5,
    return_latency=False
):
    """
    Retrieve relevant chunks using BM25.

    Args:
        query:
            User search query.

        sparse_embedder:
            SparseEmbedder instance used to generate
            the BM25 query vector.

        qdrant_store:
            QdrantStore instance used for retrieval.

        top_k:
            Number of chunks to retrieve.

        return_latency:
            Whether to return latency measurements along
            with the retrieval results.

    Returns:
        Qdrant search results when return_latency is False.

        When return_latency is True:
            tuple of (results, latency).
    """

    logger.info(
        "Running BM25 retrieval for query: %s",
        query
    )

    total_start = time.perf_counter()

    embedding_start = time.perf_counter()

    query_embedding = sparse_embedder.embed_query(
        query
    )

    embedding_ms = (
        time.perf_counter() - embedding_start
    ) * 1000

    logger.info(
        "Generated BM25 query embedding. "
        "Latency: %.2f ms",
        embedding_ms
    )

    retrieval_start = time.perf_counter()

    results = qdrant_store.search_bm25(
        query_embedding,
        top_k=top_k
    )

    retrieval_ms = (
        time.perf_counter() - retrieval_start
    ) * 1000

    total_ms = (
        time.perf_counter() - total_start
    ) * 1000

    logger.info(
        "BM25 retrieval returned %d results. "
        "Embedding: %.2f ms | "
        "Qdrant: %.2f ms | "
        "Total: %.2f ms",
        len(results.points),
        embedding_ms,
        retrieval_ms,
        total_ms
    )

    if not return_latency:
        return results

    latency = RetrievalLatency(
        embedding_ms=embedding_ms,
        retrieval_ms=retrieval_ms,
        total_ms=total_ms
    )

    return results, latency