"""
Retrieval experiment runner for the RAG Experiment Lab.

This module executes retrieval experiments across:

    - Dense retrieval
    - BM25 retrieval
    - Hybrid retrieval using RRF

For each retrieval strategy and top-K value, the runner
evaluates retrieval quality against a golden dataset.

Metrics:
    - Precision@K
    - Recall@K
    - MRR
    - nDCG@K

Latency:
    - Embedding latency
    - Retrieval latency
    - RRF latency
    - Total retrieval latency
    - Query execution latency

The runner executes queries sequentially in V1.

Batch processing can be added later as a separate experiment.
"""

import json
import logging
import statistics
import time
import uuid
from pathlib import Path

from evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k
)

from retrieval.dense import dense_search
from retrieval.sparse import bm25_search
from retrieval.hybrid import reciprocal_rank_fusion


logger = logging.getLogger("rag_experiment")


class RetrievalExperimentRunner:
    """
    Executes retrieval experiments against a golden dataset.
    """

    def __init__(
        self,
        embedder,
        sparse_embedder,
        qdrant_store,
        dataset_path
    ):
        """
        Initialize the experiment runner.
        """

        self.embedder = embedder
        self.sparse_embedder = sparse_embedder
        self.qdrant_store = qdrant_store

        self.dataset_path = Path(dataset_path)

        self.dataset = self._load_dataset()

        logger.info(
            "Loaded evaluation dataset: %s",
            self.dataset_path
        )

    def _load_dataset(self):
        """
        Load the golden evaluation dataset.

        Expected format:

        [
            {
                "query_id": "attention_01",
                "query": "...",
                "relevant_chunks": {
                    "attention_is_all_you_need_chunk_00004": 3,
                    "attention_is_all_you_need_chunk_00005": 3
                }
            }
        ]
        """

        with open(
            self.dataset_path,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    def _extract_chunk_ids(self, results):
        """
        Extract chunk IDs from Qdrant results.
        """

        return [
            point.payload["chunk_id"]
            for point in results.points
        ]

    def _evaluate_query(
        self,
        retrieved_chunks,
        relevant_chunks,
        k
    ):
        """
        Calculate retrieval metrics for a single query.
        """

        return {
            "precision": precision_at_k(
                retrieved_chunks,
                relevant_chunks,
                k
            ),
            "recall": recall_at_k(
                retrieved_chunks,
                relevant_chunks,
                k
            ),
            "mrr": reciprocal_rank(
                retrieved_chunks,
                relevant_chunks
            ),
            "ndcg": ndcg_at_k(
                retrieved_chunks,
                relevant_chunks,
                k
            )
        }

    def _aggregate_metrics(self, query_results):
        """
        Calculate aggregate retrieval metrics across queries.
        """

        if not query_results:
            return {}

        return {
            "precision": statistics.mean(
                result["metrics"]["precision"]
                for result in query_results
            ),
            "recall": statistics.mean(
                result["metrics"]["recall"]
                for result in query_results
            ),
            "mrr": statistics.mean(
                result["metrics"]["mrr"]
                for result in query_results
            ),
            "ndcg": statistics.mean(
                result["metrics"]["ndcg"]
                for result in query_results
            )
        }

    def _percentile(self, values, percentile):
        """
        Calculate a percentile using linear interpolation.

        Returns latency in milliseconds.
        """

        if not values:
            return 0.0

        sorted_values = sorted(values)

        if len(sorted_values) == 1:
            return sorted_values[0]

        position = (
            (len(sorted_values) - 1)
            * percentile
        )

        lower = int(position)
        upper = lower + 1

        if upper >= len(sorted_values):
            return sorted_values[lower]

        weight = position - lower

        return (
            sorted_values[lower]
            + weight
            * (
                sorted_values[upper]
                - sorted_values[lower]
            )
        )

    def _aggregate_latency(self, latency_values):
        """
        Calculate aggregate latency statistics.
        """

        if not latency_values:
            return {}

        return {
            "mean_ms": statistics.mean(
                latency_values
            ),
            "p50_ms": self._percentile(
                latency_values,
                0.50
            ),
            "p95_ms": self._percentile(
                latency_values,
                0.95
            )
        }

    def run(
        self,
        retrieval_strategies,
        top_k_values
    ):
        """
        Run the configured retrieval experiment.

        Args:
            retrieval_strategies:
                List containing:
                    dense
                    bm25
                    hybrid

            top_k_values:
                List of K values such as:
                    [5, 7, 10]

        Returns:
            Structured experiment results.
        """

        experiment_id = str(uuid.uuid4())

        logger.info(
            "Starting retrieval experiment: %s",
            experiment_id
        )

        experiment_start = time.perf_counter()

        strategy_results = {}

        for strategy in retrieval_strategies:

            if strategy not in {
                "dense",
                "bm25",
                "hybrid"
            }:
                raise ValueError(
                    f"Unsupported retrieval strategy: {strategy}"
                )

            strategy_results[strategy] = {}

            for k in top_k_values:

                logger.info(
                    "Running strategy=%s, top_k=%d",
                    strategy,
                    k
                )

                query_results = []

                total_latencies = []
                embedding_latencies = []
                retrieval_latencies = []
                rrf_latencies = []

                for item in self.dataset:

                    query_id = item["id"]
                    query = item["query"]
                    relevant_chunks = item[
                        "relevant_chunks"
                    ]

                    logger.info(
                        "Evaluating query %s",
                        query_id
                    )

                    query_execution_start = (
                        time.perf_counter()
                    )

                    if strategy == "dense":

                        results, latency = dense_search(
                            query,
                            self.embedder,
                            self.qdrant_store,
                            top_k=k,
                            return_latency=True
                        )

                        retrieved_chunks = (
                            self._extract_chunk_ids(
                                results
                            )
                        )

                        embedding_latencies.append(
                            latency.embedding_ms
                        )

                        retrieval_latencies.append(
                            latency.retrieval_ms
                        )

                        total_latencies.append(
                            latency.total_ms
                        )

                    elif strategy == "bm25":

                        results, latency = bm25_search(
                            query,
                            self.sparse_embedder,
                            self.qdrant_store,
                            top_k=k,
                            return_latency=True
                        )

                        retrieved_chunks = (
                            self._extract_chunk_ids(
                                results
                            )
                        )

                        embedding_latencies.append(
                            latency.embedding_ms
                        )

                        retrieval_latencies.append(
                            latency.retrieval_ms
                        )

                        total_latencies.append(
                            latency.total_ms
                        )

                    else:

                        dense_results, dense_latency = (
                            dense_search(
                                query,
                                self.embedder,
                                self.qdrant_store,
                                top_k=k,
                                return_latency=True
                            )
                        )

                        bm25_results, bm25_latency = (
                            bm25_search(
                                query,
                                self.sparse_embedder,
                                self.qdrant_store,
                                top_k=k,
                                return_latency=True
                            )
                        )

                        hybrid_results, rrf_latency = (
                            reciprocal_rank_fusion(
                                dense_results,
                                bm25_results,
                                top_k=k,
                                return_latency=True
                            )
                        )

                        retrieved_chunks = [
                            result["chunk_id"]
                            for result in hybrid_results
                        ]

                        embedding_latencies.append(
                            dense_latency.embedding_ms
                            + bm25_latency.embedding_ms
                        )

                        retrieval_latencies.append(
                            dense_latency.retrieval_ms
                            + bm25_latency.retrieval_ms
                        )

                        rrf_latencies.append(
                            rrf_latency.total_ms
                        )

                        total_latencies.append(
                            dense_latency.total_ms
                            + bm25_latency.total_ms
                            + rrf_latency.total_ms
                        )

                    metrics = self._evaluate_query(
                        retrieved_chunks,
                        relevant_chunks,
                        k
                    )

                    query_execution_ms = (
                        time.perf_counter()
                        - query_execution_start
                    ) * 1000

                    query_result = {
                        "query_id": query_id,
                        "metrics": metrics,
                        "retrieved_chunks": retrieved_chunks,
                        "query_execution_ms": query_execution_ms
                    }

                    query_results.append(
                        query_result
                    )

                aggregate_metrics = (
                    self._aggregate_metrics(
                        query_results
                    )
                )

                aggregate_latency = {
                    "embedding": self._aggregate_latency(
                        embedding_latencies
                    ),
                    "retrieval": self._aggregate_latency(
                        retrieval_latencies
                    ),
                    "total": self._aggregate_latency(
                        total_latencies
                    )
                }

                if rrf_latencies:
                    aggregate_latency["rrf"] = (
                        self._aggregate_latency(
                            rrf_latencies
                        )
                    )

                strategy_results[strategy][
                    str(k)
                ] = {
                    "metrics": aggregate_metrics,
                    "latency": aggregate_latency,
                    "queries": query_results
                }

                logger.info(
                    "Completed strategy=%s, top_k=%d",
                    strategy,
                    k
                )

        experiment_elapsed_ms = (
            time.perf_counter()
            - experiment_start
        ) * 1000

        logger.info(
            "Completed experiment %s in %.2f ms",
            experiment_id,
            experiment_elapsed_ms
        )

        experiment_results = {
        "experiment_id": experiment_id,
        "dataset": str(self.dataset_path),
        "retrieval_strategies": retrieval_strategies,
        "top_k_values": top_k_values,
        "duration_ms": experiment_elapsed_ms,
        "results": strategy_results
        }

        self._save_results(
            experiment_results
        )

        return experiment_results



    def _save_results(self, results):
        """
        Save experiment results as a JSON file.
        """

        results_directory = Path("results")
        results_directory.mkdir(
            exist_ok=True
        )

        result_file = (
            results_directory
            / f"experiment_{results['experiment_id']}.json"
        )

        with open(
            result_file,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                results,
                file,
                indent=4
            )

        logger.info(
            "Saved experiment results to: %s",
            result_file
        )