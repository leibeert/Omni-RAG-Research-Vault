import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")

# Docker / Server Settings
CHROMA_HOST = os.getenv("CHROMA_HOST", None)
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")


# Collection Settings
COLLECTION_NAME = "research_vault"

# Models
# Embedding model for vector search
EMBEDDING_MODEL = "nomic-embed-text" 

# Cross-encoder model for reranking
# Using a lightweight but effective model for reranking
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Search Parameters
TOP_K_RETRIEVAL = 10  # Number of results to fetch from Vector DB/Keyword Search
TOP_K_RERANK = 3      # Number of results to keep after reranking
