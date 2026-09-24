"""
Fixed-size chunking utilities for the RAG Experiment Lab.

This module implements character-based fixed-size chunking.

Unlike recursive chunking, fixed-size chunking does not attempt
to preserve paragraph or line boundaries.
"""

from langchain_text_splitters import CharacterTextSplitter

from config import (
    FIXED_CHUNK_SIZE,
    FIXED_CHUNK_OVERLAP,
)


def create_fixed_splitter():
    """
    Create the configured fixed-size text splitter.

    Returns:
        CharacterTextSplitter.
    """

    return CharacterTextSplitter(
        chunk_size=FIXED_CHUNK_SIZE,
        chunk_overlap=FIXED_CHUNK_OVERLAP,
        separator="",
    )


def chunk_documents(documents):
    """
    Split documents using fixed-size character chunking.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of chunked Documents.
    """

    splitter = create_fixed_splitter()

    return splitter.split_documents(documents)