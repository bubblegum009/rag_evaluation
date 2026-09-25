"""
Retrieval evaluation metrics for the RAG Experiment Lab.

This module implements ranking-based metrics for evaluating
retrieval quality against a manually created relevance dataset.

Supported metrics:
    - Precision@K
    - Recall@K
    - Mean Reciprocal Rank (MRR)
    - nDCG@K

Relevance grades:
    3 -> Highly relevant
    2 -> Relevant supporting context
    1 -> Weakly relevant
    0 -> Not relevant

The retriever determines the ranking.

The evaluation dataset only tells us how relevant each
retrieved chunk is. Chunks that are not present in the
golden relevance dataset receive a relevance grade of 0.
"""

import math


def precision_at_k(retrieved_chunks, relevant_chunks, k):
    """
    Calculate Precision@K.

    The retriever provides the ranked chunks.

    For every retrieved chunk:
        - If it exists in the golden dataset with grade > 0,
          it is considered relevant.
        - Otherwise, it is considered irrelevant.

    Args:
        retrieved_chunks:
            Ranked list of chunk IDs returned by the retriever.

        relevant_chunks:
            Dictionary mapping gold chunk IDs to relevance grades.

        k:
            Number of retrieved chunks to evaluate.

    Returns:
        Precision@K.
    """

    retrieved_at_k = retrieved_chunks[:k]

    if not retrieved_at_k:
        return 0.0

    relevant_count = 0
    
    for chunk_id in retrieved_at_k:

        relevance = relevant_chunks.get(chunk_id, 0,)

        if relevance > 0:
            relevant_count += 1

    return relevant_count / len(retrieved_at_k)


def recall_at_k(retrieved_chunks, relevant_chunks, k):
    """
    Calculate Recall@K.

    Recall measures how many of the relevant gold chunks
    were successfully retrieved within the top K results.

    Chunks returned by the retriever that are not present
    in the golden dataset are treated as irrelevant.

    Args:
        retrieved_chunks:
            Ranked list of chunk IDs returned by the retriever.

        relevant_chunks:
            Dictionary mapping gold chunk IDs to relevance grades.

        k:
            Number of retrieved chunks to evaluate.

    Returns:
        Recall@K.
    """

    retrieved_at_k = retrieved_chunks[:k]

    total_relevant = 0

    for chunk_id in relevant_chunks:

        relevance = relevant_chunks[chunk_id]

        if relevance > 0:
            total_relevant += 1

    if total_relevant == 0:
        return 0.0

    relevant_retrieved = 0

    for chunk_id in retrieved_at_k:

        relevance = relevant_chunks.get(chunk_id,0)

        if relevance > 0:
            relevant_retrieved += 1

    return relevant_retrieved / total_relevant


def reciprocal_rank(retrieved_chunks, relevant_chunks):
    """
    Calculate Reciprocal Rank for one query.

    Reciprocal Rank only cares about the FIRST relevant
    chunk returned by the retriever.

    Examples:

        First relevant at rank 1 -> 1 / 1 = 1.0
        First relevant at rank 2 -> 1 / 2 = 0.5
        First relevant at rank 5 -> 1 / 5 = 0.2
        No relevant result       -> 0.0
    """

    for index in range(len(retrieved_chunks)):

        chunk_id = retrieved_chunks[index]

        relevance = relevant_chunks.get(chunk_id,0,)

        if relevance > 0:
            rank = index + 1
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    all_retrieved_chunks,
    relevant_chunks_list,
):
    """
    Calculate Mean Reciprocal Rank (MRR).

    MRR is the average Reciprocal Rank across all queries.
    """

    if not all_retrieved_chunks:
        return 0.0

    total_reciprocal_rank = 0.0

    for retrieved_chunks, relevant_chunks in zip(
        all_retrieved_chunks,
        relevant_chunks_list,
    ):

        rr = reciprocal_rank(
            retrieved_chunks,
            relevant_chunks,
        )

        total_reciprocal_rank += rr

    return total_reciprocal_rank / len(all_retrieved_chunks)


def dcg_at_k(retrieved_chunks, relevant_chunks, k):
    """
    Calculate Discounted Cumulative Gain@K.

    DCG considers:

        1. Whether a retrieved chunk is relevant.
        2. How relevant the chunk is.
        3. Where the chunk appears in the ranking.

    Higher relevance is better.

    Earlier positions are better.

    Formula:

        relevance / log2(rank + 1)

    where rank starts at 1.
    """

    retrieved_at_k = retrieved_chunks[:k]

    dcg = 0.0

    for index in range(len(retrieved_at_k)):

        chunk_id = retrieved_at_k[index]

        relevance = relevant_chunks.get(chunk_id, 0,)

        # The chunk is not present in the goldenrelevance dataset, so its relevance is 0.
        if relevance == 0:
            continue

        rank = index + 1

        dcg += relevance / math.log2(rank + 1)

    return dcg


def ideal_ranking(relevant_chunks):
    """
    Create the ideal ranking from the golden dataset.

    The most relevant chunks are placed first.

    Example:

        {
            "A": 3,
            "B": 1,
            "C": 3,
            "D": 2
        }

    becomes:

        A -> 3
        C -> 3
        D -> 2
        B -> 1

    The actual retriever ranking is NOT changed.
    This ranking is only used to calculate IDCG.
    """

    ideal_chunks = list(relevant_chunks.keys())

    ideal_chunks.sort(key=lambda chunk_id: relevant_chunks[chunk_id],reverse=True,)

    return ideal_chunks

def ndcg_at_k(retrieved_chunks, relevant_chunks, k):
    """
    Calculate normalized Discounted Cumulative Gain@K.

    nDCG compares:

        Actual retriever ranking
        ------------------------
        Ideal possible ranking

    The actual retriever ranking is never modified.

    Chunks that are not present in the golden dataset
    receive relevance 0 and therefore contribute nothing
    to DCG.

    Returns:
        nDCG@K between 0 and 1.
    """

    actual_dcg = dcg_at_k(retrieved_chunks,relevant_chunks, k,)

    ideal_chunks = ideal_ranking(relevant_chunks)

    ideal_dcg = dcg_at_k(ideal_chunks, relevant_chunks,k,)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg