"""
Dense retrieval utilities for the RAG Experiment Lab.

This module converts a user query into a dense embedding
and retrieves the most similar document chunks from Qdrant.
"""

import logging

logger = logging.getLogger("rag_experiment")


def dense_search(query, embedder, qdrant_store, top_k=5):
    """
    Retrieve the most relevant chunks for a query.

    Args:
        query: User's natural-language query.
        embedder: Initialized embedding model.
        qdrant_store: Initialized QdrantStore.
        top_k: Number of chunks to retrieve.

    Returns:
        Qdrant search results.
    """

    logger.info("Running dense retrieval for query: %s", query)
"""
Dense retrieval utilities for the RAG Experiment Lab.

This module converts a user query into a dense embedding
and retrieves the most similar document chunks from Qdrant.
"""

import logging
import time

from retrieval.latency import RetrievalLatency


logger = logging.getLogger("rag_experiment")


def dense_search(
    query,
    embedder,
    qdrant_store,
    top_k=5,
    return_latency=False
):
    """
    Retrieve the most relevant chunks for a query.

    Args:
        query:
            User's natural-language query.

        embedder:
            Initialized dense embedding model.

        qdrant_store:
            Initialized QdrantStore.

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
        "Running dense retrieval for query: %s",
        query
    )

    total_start = time.perf_counter()

    embedding_start = time.perf_counter()

    query_embedding = embedder.embed_query(
        query
    )

    embedding_ms = (
        time.perf_counter() - embedding_start
    ) * 1000

    logger.info(
        "Generated query embedding. Shape: %s. Latency: %.2f ms",
        query_embedding.shape,
        embedding_ms
    )

    retrieval_start = time.perf_counter()

    results = qdrant_store.search_dense(
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
        "Dense retrieval returned %d results. "
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
    query_embedding = embedder.embed_query(query)

    logger.info("Generated query embedding. Shape: %s", query_embedding.shape)

    results = qdrant_store.search_dense(query_embedding, top_k=top_k)

    logger.info("Retrieved %d results.", len(results.points))

    return results