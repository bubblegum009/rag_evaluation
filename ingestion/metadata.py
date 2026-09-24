"""
Metadata utilities for the RAG Experiment Lab.

This module is responsible for enriching the metadata generated
by document loaders with application-level information.

Loader-provided metadata is preserved because it may contain
useful information such as page number, source path, and
document-level PDF metadata.
"""

from pathlib import Path


def enrich_metadata(documents):
    """
    Add consistent application-level metadata to documents.

    Existing metadata from the document loader is preserved.
    This function adds fields that are useful for identifying
    documents throughout the RAG pipeline.

    Added fields:
        - filename
        - file_type

    Args:
        documents: List of LangChain Document objects.

    Returns:
        The same list of Document objects with enriched metadata.
    """

    for document in documents:
        source = document.metadata.get("source")

        if source:
            path = Path(source)
            filename = path.name

            document.metadata["filename"] = path.name
            document.metadata["file_type"] = (
                path.suffix.lower().lstrip(".")
            )
            document.metadata["document_id"] = generate_document_id(
                filename
            )


    return documents


def generate_document_id(filename: str) -> str:
    """
    Generate a stable document identifier from a filename.

    Example:
        attention_is_all_you_need.pdf
        ->
        attention_is_all_you_need
    """

    return Path(filename).stem