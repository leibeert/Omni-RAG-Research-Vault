from typing import List, Dict, Any, Optional
import chromadb
from chromadb.api import ClientAPI
from chromadb.utils import embedding_functions
import ollama
from src.parser import Document

class OllamaEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    Custom embedding function using Ollama's API.
    """
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for text in input:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            embeddings.append(response["embedding"])
        return embeddings

class ResearchIndex:
    """
    Manages the vector store for research documents.
    """
    def __init__(self, persist_directory: str = "data/chroma_db", collection_name: str = "research_vault"):
        """
        Initialize the ChromaDB client and collection.
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = OllamaEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )

    def add_documents(self, documents: List[Document]):
        """
        Adds Document objects to the collection.
        """
        if not documents:
            return

        ids = [doc.metadata["id"] for doc in documents]
        documents_content = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        self.collection.add(
            ids=ids,
            documents=documents_content,
            metadatas=metadatas
        )

    def search(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Search the collection for relevant documents.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
