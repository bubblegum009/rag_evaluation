"""
API schemas for RAG experiments.

This module defines the request and response models used
by the experiment API.
"""

from pydantic import BaseModel, Field


class ExperimentRequest(BaseModel):
    """
    Configuration for a retrieval experiment.
    """

    retrieval_strategies: list[str] = Field(
        default=["dense", "bm25", "hybrid"]
    )

    top_k_values: list[int] = Field(
        default=[5, 7, 10]
    )


class ExperimentResponse(BaseModel):
    """
    Response returned after running an experiment.
    """

    experiment_id: str
    status: str
    results: list[dict]