"""
Document loaders for the RAG Experiment Lab.

This module loads PDF documents, converts their pages into
LangChain Document objects, enriches metadata, and applies
basic text cleaning.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from ingestion.metadata import enrich_metadata
from ingestion.cleaning import clean_text


def load_pdf(file_path: str):
    """
    Load and clean a single PDF.

    PyPDFLoader creates one LangChain Document per page.
    Existing metadata is preserved and enriched with
    application-level metadata.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of cleaned LangChain Document objects.
    """

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    documents = enrich_metadata(documents)

    for document in documents:
        document.page_content = clean_text(
            document.page_content
        )

    return documents


def load_all_pdfs(directory: str):
    """
    Load all PDFs from a directory.

    Args:
        directory: Directory containing PDF documents.

    Returns:
        A combined list of cleaned Document objects.
    """

    documents = []

    for file_path in Path(directory).glob("*.pdf"):
        documents.extend(load_pdf(str(file_path)))

    return documents