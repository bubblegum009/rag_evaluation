"""
Project configuration for the RAG Experiment Lab.

This module contains infrastructure settings and default pipeline
configuration. Experiment-specific parameters that we want to
vary through the API will be supplied separately.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Recursive Chunking 
RECURSIVE_CHUNK_SIZE = 1000
RECURSIVE_CHUNK_OVERLAP = 150

# Fixed-Size Chunking
FIXED_CHUNK_SIZE = 1000
FIXED_CHUNK_OVERLAP = 150

# Parent-Child Chunking
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 500
CHILD_CHUNK_OVERLAP = 50


#EMBEDINGS
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384
BM25_MODEL_NAME = "Qdrant/bm25"


#DATABASE
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "rag_documents_hybrid"
UPSERT_BATCH_SIZE=64

#FILE PATHS
DOCUMENT_DIRECTORY = "data/documents"
EVALUATION_DATASET_PATH = ("evaluation/dataset_rag.json")
RESULTS_DIRECTORY = "results"


#HYBRID RETRIEVL
RECIPROCAL_RANK_FUSION_K=60

