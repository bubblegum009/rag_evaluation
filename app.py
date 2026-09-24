"""
Application entry point for the RAG Experiment Lab.

On application startup, the document pipeline is initialized:

    Local PDFs
        ↓
    Document extraction
        ↓
    Metadata enrichment
        ↓
    Text cleaning
        ↓
    Fixed-size chunking
        ↓
    Dense embeddings

The resulting chunks and embeddings will later be
stored in Qdrant for retrieval.
"""

from ingestion.loaders import load_all_pdfs
from chunking.fixed import chunk_documents
from embeddings.embedder import Embedder



# ============================================
# Configuration
# ============================================

DOCUMENT_DIRECTORY = "data/documents"


# ============================================
# Logging
# ============================================
import logging
from datetime import datetime
from pathlib import Path


LOG_DIRECTORY = Path("logs")


LOG_DIRECTORY.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

log_file = LOG_DIRECTORY / f"rag_experiment_{timestamp}.log"

logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        handlers=[
            logging.FileHandler(
                log_file,
                encoding="utf-8",
            ),
            logging.StreamHandler(),
        ],
    )
logger=logging.getLogger("rag_experiment")



# ============================================
# Application initialization
# ============================================

def initialize_pipeline():
    """
    Load documents, chunk them, and generate embeddings.
    """

    logger.info("Starting document pipeline.")

    logger.info(
        "Loading documents from: %s",
        DOCUMENT_DIRECTORY,
    )

    documents = load_all_pdfs(DOCUMENT_DIRECTORY)

    logger.info(
        "Loaded %d document pages.",
        len(documents),
    )

    logger.info("Creating fixed-size chunks.")

    chunks = chunk_documents(documents)

    logger.info(
        "Created %d chunks.",
        len(chunks),
    )

    logger.info("Loading embedding model.")

    embedder = Embedder()

    logger.info("Generating document embeddings.")

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = embedder.embed_documents(texts)

    logger.info(
        "Generated document embeddings. Shape: %s",
        embeddings.shape,
    )

    logger.info("Document pipeline completed successfully.")

    return chunks, embeddings, embedder


# ============================================
# Application startup
# ============================================

chunks, embeddings, embedder = initialize_pipeline()