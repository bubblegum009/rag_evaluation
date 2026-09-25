"""
Hybrid retrieval utilities for the RAG Experiment Lab.

This module combines dense and BM25 retrieval results using
Reciprocal Rank Fusion (RRF).

RRF combines rankings rather than raw retrieval scores, which
allows dense cosine similarity scores and BM25 scores to be
combined without requiring score normalization.
"""

import logging
import time

from config import RECIPROCAL_RANK_FUSION_K
from retrieval.latency import RetrievalLatency


logger = logging.getLogger("rag_experiment")


def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    top_k=5,
    return_latency=False
):
    """
    Combine dense and BM25 rankings using Reciprocal Rank Fusion.

    Args:
        dense_results:
            Qdrant results from dense retrieval.

        bm25_results:
            Qdrant results from BM25 retrieval.

        top_k:
            Number of final hybrid results to return.

        return_latency:
            Whether to return RRF latency along with results.

    Returns:
        Hybrid results when return_latency is False.

        When return_latency is True:
            tuple of (results, latency).
    """

    rrf_start = time.perf_counter()

    rrf_k = RECIPROCAL_RANK_FUSION_K
    scores = {}
    payloads = {}

    # --------------------------------------------------------
    # Dense results
    # --------------------------------------------------------

    for rank, point in enumerate(
        dense_results.points,
        start=1
    ):

        chunk_id = point.payload["chunk_id"]

        score = 1 / (
            rrf_k + rank
        )

        scores[chunk_id] = (
            scores.get(chunk_id, 0.0)
            + score
        )

        payloads[chunk_id] = point.payload


    # --------------------------------------------------------
    # BM25 results
    # --------------------------------------------------------

    for rank, point in enumerate(
        bm25_results.points,
        start=1
    ):

        chunk_id = point.payload["chunk_id"]

        score = 1 / (
            rrf_k + rank
        )

        scores[chunk_id] = (
            scores.get(chunk_id, 0.0)
            + score
        )

        payloads[chunk_id] = point.payload


    # --------------------------------------------------------
    # Sort by RRF score
    # --------------------------------------------------------

    ranked_chunks = []

    for chunk_id in scores:

        ranked_chunks.append(
            {
                "chunk_id": chunk_id,
                "score": scores[chunk_id],
                "payload": payloads[chunk_id]
            }
        )


    ranked_chunks.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    results = ranked_chunks[:top_k]

    rrf_ms = (
        time.perf_counter() - rrf_start
    ) * 1000

    logger.info(
        "RRF produced %d hybrid results. "
        "Latency: %.2f ms",
        len(results),
        rrf_ms
    )

    if not return_latency:
        return results

    latency = RetrievalLatency(
        retrieval_ms=rrf_ms,
        total_ms=rrf_ms
    )

    return results, latency