"""
Recursive chunking utilities for the RAG Experiment Lab.

This module implements recursive character-based chunking.

Recursive chunking attempts to preserve larger text boundaries
such as paragraphs and lines before falling back to smaller
boundaries such as words and characters.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    RECURSIVE_CHUNK_SIZE,
    RECURSIVE_CHUNK_OVERLAP,
)


def create_recursive_splitter():
    """
    Create the configured recursive text splitter.

    Returns:
        RecursiveCharacterTextSplitter.
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=RECURSIVE_CHUNK_SIZE,
        chunk_overlap=RECURSIVE_CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )


def chunk_documents(documents):
    """
    Split documents using recursive chunking.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        List of chunked Documents.
    """

    splitter = create_recursive_splitter()

    return splitter.split_documents(documents)