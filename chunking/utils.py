"""
Utilities for working with document chunks.

This module contains functionality shared by all chunking
strategies, such as assigning identifiers to generated chunks.
"""


def assign_chunk_ids(chunks):
    """
    Assign a unique identifier to each chunk.

    The identifier is based on the source document and the
    chunk's position in the generated chunk list.

    Args:
        chunks: List of LangChain Document objects.

    Returns:
        The same list of documents with chunk_id added to metadata.
    """

    for index, chunk in enumerate(chunks):
        document_id = chunk.metadata["document_id"]

        chunk.metadata["chunk_id"] = (
            f"{document_id}_chunk_{index:05d}"
        )

    return chunks