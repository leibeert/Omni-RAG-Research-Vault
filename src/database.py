from typing import List, Dict, Any, Optional
import chromadb
import ollama
from rank_bm25 import BM25Okapi
import string

from src.parser import Document
from src.config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL

class OllamaEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    Custom embedding function using Ollama's API.
    """
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for text in input:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            embeddings.append(response["embedding"])
        return embeddings

class Database:
    """
    Manages the vector store and keyword search index (Hybrid Search).
    """
    def __init__(self, persist_directory: str = CHROMA_DB_PATH, collection_name: str = COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = OllamaEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )
        
        # In-memory store for BM25 (simple implementation for this scope)
        # In a production system, you might restart this or persist it separately.
        # For now, we will rebuild it on load if documents exist, or rely on runtime addition.
        # However, since BM25 needs the corpus to initialize, and we might not load everything at once,
        # we'll implement a simple keyword match fallback or strictly re-index on startup if needed.
        # Given the "Day 3" scope, we will initialize BM25 on the fly or just use simple substring/keyword matching
        # if BM25 is too heavy to reload every time.
        # BUT, the requirement asks for "Review Requirements: Hybrid Search: ... (BM25 or similar logic)".
        # To make it persistent without complex serialization, we will load *all* documents from Chroma 
        # into memory for BM25 construction upon initialization. 
        # This is okay for a "Research Vault" with < 10k documents.
        
        self.bm25 = None
        self.doc_store = {} # Map ID to content for BM25 retrieval
        self.doc_ids = []   # List of IDs parallel to BM25 corpus
        
        self._load_bm25()

    def _load_bm25(self):
        """
        Loads all documents from ChromaDB to initialize BM25.
        """
        # Fetch all documents
        existing_data = self.collection.get()
        if existing_data and existing_data['documents']:
            corpus = existing_data['documents']
            self.doc_ids = existing_data['ids']
            
            # Simple tokenization
            tokenized_corpus = [self._tokenize(doc) for doc in corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            
            # Store full objects/references if needed, but Chroma has the data.
            # We just need to map index ref -> chromadb ID.

    def _tokenize(self, text: str) -> List[str]:
        # Simple tokenizer: lowercase and remove punctuation
        text = text.lower()
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator).split()

    def add_documents(self, documents: List[Document]):
        """
        Adds documents to ChromaDB and updates BM25.
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
        
        # Re-build BM25 (not efficient for huge scale, but fine here)
        self._load_bm25()

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Hybrid search: combines Vector search + BM25 search.
        Returns a unified list of results format similar to ChromaDB query output.
        """
        # 1. Vector Search
        vector_results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format vector results to ease merging
        # vector_results structure: {'ids': [['id1', ...]], 'documents': [['text1', ...]], 'metadatas': [[{...}, ...]], 'distances': [[0.2, ...]]}
        combined_results = {}
        
        if vector_results['ids']:
             for i, doc_id in enumerate(vector_results['ids'][0]):
                 combined_results[doc_id] = {
                     'doc': vector_results['documents'][0][i],
                     'metadata': vector_results['metadatas'][0][i],
                     'score': 1.0 # Placeholder for vector "presence" score or normalized distance
                 }

        # 2. BM25 Search
        if self.bm25:
            tokenized_query = self._tokenize(query)
            # Get top N BM25 docs
            # BM25Okapi gives scores for *all* docs. We pick top N.
            doc_scores = self.bm25.get_scores(tokenized_query)
            top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:n_results]
            
            for idx in top_n_indices:
                score = doc_scores[idx]
                if score > 0: # Only relevant matches
                    doc_id = self.doc_ids[idx]
                    
                    if doc_id not in combined_results:
                        # Need to fetch details if not in vector results
                        # A bit inefficient to fetch one by one, but robust
                        d = self.collection.get(ids=[doc_id])
                        if d['documents']:
                            combined_results[doc_id] = {
                                'doc': d['documents'][0],
                                'metadata': d['metadatas'][0],
                                'score': 0.5 # Placeholder for keyword boost
                            }
                    else:
                        # Boost existing vector result
                        combined_results[doc_id]['score'] += 0.5
        
        # Convert back to list format expected by downstream
        # We just return a clean list of dicts directly, simplifying the messy Chroma structure
        final_docs = []
        final_metas = []
        final_ids = []
        
        # Sort by simple score
        sorted_ids = sorted(combined_results.keys(), key=lambda k: combined_results[k]['score'], reverse=True)
        
        for doc_id in sorted_ids[:n_results * 2]: # Keep a bit more for reranking
            final_ids.append(doc_id)
            final_docs.append(combined_results[doc_id]['doc'])
            final_metas.append(combined_results[doc_id]['metadata'])
            
        return {
            'ids': [final_ids],
            'documents': [final_docs],
            'metadatas': [final_metas]
        }
