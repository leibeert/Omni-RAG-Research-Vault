from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
import time
import logging

from src.schemas import QueryRequest, AnswerResponse, Source, ErrorResponse
import uvicorn
import json
import ollama

from src.database import Database
from src.search import Reranker, search_and_answer
from src.config import TOP_K_RETRIEVAL, TOP_K_RERANK
from src.memory import ChatHistory

app = FastAPI(
    title="Omni-RAG Research Vault",
    description="A production-ready RAG system for researching and querying documents with citation support.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global Error Handler Middleware
@app.middleware("http")
async def global_error_handler(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logging.error(f"Global Error: {str(e)}")
        # Check for common errors (DB, Timeout) - simplistic check
        error_msg = str(e)
        if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            return JSONResponse(
                status_code=503,
                content={"detail": "Service Unavailable: Database or LLM connection issue."}
            )
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal Server Error: {str(e)}"}
        )

# Initialize Global Resources
db = Database()
reranker = Reranker()
# Simple in-memory history for single-user dev (Global)
# In prod, this would be session-based or keyed by user_id
chat_history = ChatHistory()

# QueryRequest imported from src.schemas

@app.on_event("startup")
async def startup_event():
    print("Omni-RAG System Initialized.")
    # In a real app, we might check DB connection here

@app.get("/")
def read_root():
    return {"status": "active", "message": "Welcome to Omni-RAG Research Vault API"}

@app.post("/chat", response_model=AnswerResponse, tags=["Research"])
async def chat_endpoint(request: QueryRequest):
    """
    Search and Answer endpoint.
    
    Processing Steps:
    1. condensed_question = process_history(query)
    2. retrieval = vector_db.search(condensed_question)
    3. reranking = reranker.rank(retrieval)
    4. generation = llm.generate(prompt, context)
    
    Returns:
        AnswerResponse: The answer and list of sources.
    """
    user_query = request.query
    
    # 0. Conversation Memory
    query = chat_history.condense_question(user_query)
    chat_history.add_user_message(user_query)
    
    # 1. Retrieval
    results = db.search(query, n_results=TOP_K_RETRIEVAL)
    if not results['documents'] or not results['documents'][0]:
        return AnswerResponse(answer="No relevant documents found.", sources=[])
    
    candidates_text = results['documents'][0]
    candidates_meta = results['metadatas'][0]
    
    # 2. Reranking
    try:
        top_indices = reranker.rerank(query, candidates_text, top_k=TOP_K_RERANK)
        top_docs = [candidates_text[i] for i in top_indices]
        top_metas = [candidates_meta[i] for i in top_indices]
    except Exception as e:
        # Fallback if reranker fails (e.g. memory issue)
        print(f"Reranking failed: {e}. Falling back to top retrieval results.")
        top_docs = candidates_text[:TOP_K_RERANK]
        top_metas = candidates_meta[:TOP_K_RERANK]
    
    # 3. Context Construction
    context = "\n\n".join([f"Source {i+1}: {doc}" for i, doc in enumerate(top_docs)])
    
    prompt = f"""You are a research assistant. Answer the question based ONLY on the following context.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:"""
    
    # 4. Generation (Non-streaming for strict schema validation)
    response = ollama.chat(
        model='qwen2.5:14b',
        messages=[{'role': 'user', 'content': prompt}],
        stream=False,
    )
    
    answer_text = response['message']['content']
    
    # Update history
    chat_history.add_ai_message(answer_text)
    
    # 5. Response Formatting
    sources = []
    for meta in top_metas:
        sources.append(Source(
            filename=meta.get("filename", "Unknown"),
            page_number=meta.get("page_number") # Pydantic Optional[int] handles None/missing if passed correctly, but meta.get returns None by default
        ))

    return AnswerResponse(answer=answer_text, sources=sources)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
