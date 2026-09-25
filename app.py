"""
Application entry point for the RAG Experiment Lab.

The application performs document indexing during startup.

Startup flow:

    FastAPI starts
        ↓
    Configure logging
        ↓
    Initialize QdrantStore
        ↓
    Ensure Qdrant collection and payload indexes exist
        ↓
    Load dense and sparse embedding models
        ↓
    Discover local PDF documents
        ↓
    Check whether each document is already indexed
        ↓
    PDF extraction
        ↓
    Metadata enrichment and text cleaning
        ↓
    Recursive chunking
        ↓
    Chunk ID assignment
        ↓
    Dense embeddings
    Sparse BM25 embeddings
        ↓
    Upload to Qdrant

Already-indexed documents are skipped.

FastAPI manages the application lifecycle and stores shared
components in application state.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI

from chunking.recursive import chunk_documents
from chunking.utils import assign_chunk_ids
from config import (DOCUMENT_DIRECTORY,EVALUATION_DATASET_PATH)
from embeddings.embedder import Embedder
from embeddings.sparse_embedder import SparseEmbedder
from ingestion.loaders import load_pdf
from vectorstore.qdrant_store import QdrantStore
from experiments.schemas import ExperimentRequest

from experiments.retrieval_experiment import (RetrievalExperimentRunner)

# ============================================================
# Logging configuration
# ============================================================

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
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("rag_experiment")


# ============================================================
# Document indexing
# ============================================================

def initialize_pipeline():
    """
    Initialize the document indexing pipeline.

    The pipeline:

        1. Connects to Qdrant.
        2. Ensures the collection exists.
        3. Loads dense and sparse embedding models.
        4. Discovers local PDF documents.
        5. Skips documents already indexed.
        6. Loads and cleans each PDF.
        7. Creates recursive chunks.
        8. Assigns chunk IDs.
        9. Generates dense embeddings.
        10. Generates BM25 sparse embeddings.
        11. Uploads both vectors to Qdrant.

    Returns:
        A tuple containing the Qdrant store, dense embedder,
        and sparse embedder.
    """

    logger.info("Starting document indexing pipeline.")

    # --------------------------------------------------------
    # Qdrant
    # --------------------------------------------------------

    logger.info("Connecting to Qdrant.")

    qdrant_store = QdrantStore()

    logger.info("Qdrant initialization completed.")


    # --------------------------------------------------------
    # Embedding models
    # --------------------------------------------------------

    logger.info("Loading dense embedding model.")

    embedder = Embedder()

    logger.info("Dense embedding model loaded.")


    logger.info("Loading sparse BM25 embedding model.")

    sparse_embedder = SparseEmbedder()

    logger.info("Sparse embedding model loaded.")


    # --------------------------------------------------------
    # Discover documents
    # --------------------------------------------------------

    document_paths = Path(
        DOCUMENT_DIRECTORY
    ).glob("*.pdf")

    processed_documents = 0
    skipped_documents = 0


    # --------------------------------------------------------
    # Process each document
    # --------------------------------------------------------

    for file_path in document_paths:

        document_id = file_path.stem

        logger.info(
            "Checking document: %s",
            document_id
        )


        # ----------------------------------------------------
        # Check whether document is already indexed
        # ----------------------------------------------------

        if qdrant_store.document_exists(
            document_id
        ):

            logger.info(
                "Skipping already indexed document: %s",
                document_id
            )

            skipped_documents += 1

            continue


        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        logger.info(
            "Loading PDF: %s",
            file_path
        )

        documents = load_pdf(
            str(file_path)
        )

        logger.info(
            "Loaded %d pages from %s.",
            len(documents),
            document_id
        )


        # ----------------------------------------------------
        # Chunking
        # ----------------------------------------------------

        logger.info(
            "Creating recursive chunks for: %s",
            document_id
        )

        chunks = chunk_documents(
            documents
        )

        chunks = assign_chunk_ids(
            chunks
        )

        logger.info(
            "Created %d chunks for %s.",
            len(chunks),
            document_id
        )


        # ----------------------------------------------------
        # Prepare text
        # ----------------------------------------------------

        texts = [
            chunk.page_content
            for chunk in chunks
        ]


        # ----------------------------------------------------
        # Dense embeddings
        # ----------------------------------------------------

        logger.info(
            "Generating dense embeddings for %s.",
            document_id
        )

        dense_embeddings = embedder.embed_documents(
            texts,
            batch_size=32
        )

        logger.info(
            "Generated dense embeddings for %s. Shape: %s",
            document_id,
            dense_embeddings.shape
        )


        # ----------------------------------------------------
        # Sparse BM25 embeddings
        # ----------------------------------------------------

        logger.info(
            "Generating BM25 sparse embeddings for %s.",
            document_id
        )

        sparse_embeddings = sparse_embedder.embed_documents(
            texts
        )

        logger.info(
            "Generated BM25 sparse embeddings for %s.",
            document_id
        )


        # ----------------------------------------------------
        # Upload to Qdrant
        # ----------------------------------------------------

        logger.info(
            "Uploading %s to Qdrant.",
            document_id
        )

        qdrant_store.upsert_documents(
            chunks,
            dense_embeddings,
            sparse_embeddings,
            batch_size=64
        )

        logger.info(
            "Uploaded %d chunks for %s.",
            len(chunks),
            document_id
        )

        processed_documents += 1


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    logger.info(
        "Document indexing pipeline completed."
    )

    logger.info(
        "Documents processed: %d",
        processed_documents
    )

    logger.info(
        "Documents skipped: %d",
        skipped_documents
    )


    return (
        qdrant_store,
        embedder,
        sparse_embedder
    )


# ============================================================
# FastAPI lifecycle
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage FastAPI application startup and shutdown.

    Document indexing is performed during application startup.
    Shared components are stored in application state so that
    API endpoints can reuse them.
    """

    logger.info(
        "Starting RAG Experiment Lab API."
    )

    (
        app.state.qdrant_store,
        app.state.embedder,
        app.state.sparse_embedder
    ) = initialize_pipeline()

    logger.info(
        "RAG Experiment Lab API startup completed."
    )

    yield

    logger.info(
        "Shutting down RAG Experiment Lab API."
    )


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="RAG Experiment Lab",
    description=(
        "API-driven framework for evaluating "
        "RAG retrieval and generation strategies."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# Health endpoint
# ============================================================

@app.get("/health")
def health_check():
    """
    Check whether the API is running.
    """

    return {
        "status": "healthy"
    }

@app.post("/experiments")
def run_experiment(
    request: ExperimentRequest
):
    """
    Run a retrieval experiment using the configured
    golden evaluation dataset.
    """

    logger.info(
        "Received experiment request."
    )

    runner = RetrievalExperimentRunner(
        embedder=app.state.embedder,
        sparse_embedder=app.state.sparse_embedder,
        qdrant_store=app.state.qdrant_store,
        dataset_path=EVALUATION_DATASET_PATH
    )

    results = runner.run(
        retrieval_strategies=request.retrieval_strategies,
        top_k_values=request.top_k_values
    )

    return results