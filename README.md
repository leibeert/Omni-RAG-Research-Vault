# Omni-RAG Research Vault

A production-ready RAG (Retrieval-Augmented Generation) system designed for researching and querying documents with high precision and citation support.

## Features

- **Hybrid Search**: Combines ChromaDB vector search with BM25 keyword matching for optimal retrieval.
- **Reranking**: Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to refine search results.
- **Strict API Contracts**: Pydantic-validated Inputs/Outputs.
- **Global Error Handling**: Robust middleware for graceful failure management.
- **Dockerized**: specific `Dockerfile` and `docker-compose` setup for easy deployment.
- **Interactive API**: FastAPI generated Swagger UI at `/docs`.

## Quick Start (Docker)

The easiest way to run the system is with Docker Compose, which sets up the Application and the Vector Database automatically.

```bash
docker-compose up --build
```

Access the API documentation at: [http://localhost:8000/docs](http://localhost:8000/docs)

## Local Development

### Prerequisites

- Python 3.11+
- Ollama running locally (ensure `qwen2.5:14b` is pulled)

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   uvicorn main:app --reload
   ```

## Architecture

- **`src/`**: Core source code (cleaner, parser, search, database).
- **`data/`**: Stores raw documents and the persistent ChromaDB (when running locally).
- **`Dockerfile`**: Lightweight documentation-ready container.
