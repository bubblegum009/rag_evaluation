"""
Parent-child chunking utilities for the RAG Experiment Lab.

Parent-child chunking creates larger parent chunks and smaller
child chunks.

Children are intended for retrieval, while their corresponding
parents provide additional context to the generation stage.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
)


def create_parent_splitter():
    """
    Create the splitter used to generate parent chunks.
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )


def create_child_splitter():
    """
    Create the splitter used to generate child chunks.
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )


def create_parent_child_chunks(documents):
    """
    Create parent and child chunks.

    Each child stores the ID of its parent so that retrieval
    can later locate the larger parent context.

    Returns:
        Tuple containing:
            - parent_documents
            - child_documents
    """

    parent_splitter = create_parent_splitter()
    child_splitter = create_child_splitter()

    parent_documents = parent_splitter.split_documents(
        documents
    )

    child_documents = []

    for parent_index, parent in enumerate(parent_documents):

        parent_id = (
            f"{parent.metadata['document_id']}"
            f"_parent_{parent_index:04d}"
        )

        parent.metadata["parent_id"] = parent_id

        children = child_splitter.split_documents(
            [parent]
        )

        for child_index, child in enumerate(children):

            child.metadata["parent_id"] = parent_id

            child.metadata["child_id"] = (
                f"{parent_id}"
                f"_child_{child_index:04d}"
            )

            child_documents.append(child)

    return parent_documents, child_documents