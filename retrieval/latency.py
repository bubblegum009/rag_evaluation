"""
Latency utilities for the RAG Experiment Lab.

This module provides common structures for recording latency
during retrieval experiments.

Latency is measured in milliseconds.

The structure is intentionally independent of the retrieval
implementation so that future batch-processing experiments
can reuse the same latency representation.
"""

from dataclasses import dataclass


@dataclass
class RetrievalLatency:
    """
    Latency measurements for a retrieval operation.
    """

    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    total_ms: float = 0.0