from sentence_transformers import CrossEncoder
import ollama
from typing import List, Dict, Any
from src.database import Database
from src.config import RERANK_MODEL, TOP_K_RETRIEVAL, TOP_K_RERANK

class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[str], top_k: int = TOP_K_RERANK) -> List[int]:
        """
        Reranks documents based on query relevance using a CrossEncoder.
        Returns indices of the top_k documents from the original list.
        """
        if not docs:
            return []
            
        pairs = [[query, doc] for doc in docs]
        scores = self.model.predict(pairs)
        
        # Sort indices by score descending
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return sorted_indices[:top_k]

def search_and_answer(query: str, db: Database, reranker: Reranker) -> Dict[str, Any]:
    """
    Orchestrates the search pipeline:
    1. Hybrid Search (Database)
    2. Reranking (CrossEncoder)
    3. Answer Generation (Ollama)
    """
    
    # 1. Retrieval
    print(f"Retrieving top {TOP_K_RETRIEVAL} candidates...")
    results = db.search(query, n_results=TOP_K_RETRIEVAL)
    
    if not results['documents'] or not results['documents'][0]:
        return {"answer": "I found no relevant documents to answer your question.", "sources": []}

    candidates_text = results['documents'][0]
    candidates_meta = results['metadatas'][0]

    # 2. Reranking
    print(f"Reranking top {len(candidates_text)} candidates...")
    top_indices = reranker.rerank(query, candidates_text, top_k=TOP_K_RERANK)
    
    top_docs = [candidates_text[i] for i in top_indices]
    top_metas = [candidates_meta[i] for i in top_indices]

    # 3. Context Construction
    context = "\n\n".join([f"Source {i+1}: {doc}" for i, doc in enumerate(top_docs)])
    
    prompt = f"""You are a research assistant. Answer the question based ONLY on the following context.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:"""

    # 4. Generation
    print("Generating answer with Ollama...")
    response = ollama.generate(model="qwen2.5:14b", prompt=prompt)
    answer_text = response['response']

    # 5. Format Output
    sources = []
    for meta in top_metas:
        sources.append({
            "filename": meta.get("filename", "Unknown"),
            "page_number": meta.get("page_number", "Unknown")
        })

    return {
        "answer": answer_text,
        "sources": sources
    }
