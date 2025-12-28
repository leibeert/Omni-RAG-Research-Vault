from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import json
import ollama

from src.database import Database
from src.search import Reranker, search_and_answer
from src.config import TOP_K_RETRIEVAL, TOP_K_RERANK
from src.memory import ChatHistory

app = FastAPI(title="Omni-RAG Research Vault", version="1.0.0")

# Initialize Global Resources
db = Database()
reranker = Reranker()
# Simple in-memory history for single-user dev (Global)
# In prod, this would be session-based or keyed by user_id
chat_history = ChatHistory()

class QueryRequest(BaseModel):
    query: str
    pass

@app.on_event("startup")
async def startup_event():
    print("Omni-RAG System Initialized.")
    # In a real app, we might check DB connection here

@app.get("/")
def read_root():
    return {"status": "active", "message": "Welcome to Omni-RAG Research Vault API"}

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    """
    Streaming chat endpoint that:
    1. Searches (Hybrid)
    2. Reranks
    3. Streams response from Ollama
    """
    user_query = request.query
    
    # 0. Conversation Memory - Condense Question
    # Check if we have history
    query = chat_history.condense_question(user_query)
    print(f"Original: {user_query} -> Condensed: {query}")
    
    # Add user message to history
    chat_history.add_user_message(user_query) # Store original or condensed? Usually original for display, but context matters.
    
    # 1. Retrieval & Reranking (Non-streaming part)
    # We reuse the logic from search.py but we need to control the generation generation for streaming
    
    # Search
    results = db.search(query, n_results=TOP_K_RETRIEVAL)
    if not results['documents'] or not results['documents'][0]:
        return {"answer": "No relevant documents found.", "sources": []}
    
    candidates_text = results['documents'][0]
    candidates_meta = results['metadatas'][0]
    
    # Rerank
    top_indices = reranker.rerank(query, candidates_text, top_k=TOP_K_RERANK)
    top_docs = [candidates_text[i] for i in top_indices]
    top_metas = [candidates_meta[i] for i in top_indices]
    
    # Construct Context
    context = "\n\n".join([f"Source {i+1}: {doc}" for i, doc in enumerate(top_docs)])
    
    # Prepare Prompt
    prompt = f"""You are a research assistant. Answer the question based ONLY on the following context.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:"""
    
    # Sources for the client
    sources = []
    for meta in top_metas:
        sources.append({
            "filename": meta.get("filename", "Unknown"),
            "page_number": meta.get("page_number", "Unknown")
        })

    # Generator for Streaming
    def iter_response():
        # First yield the sources headers or metadata if needed? 
        # Usually checking streaming APIs, we send the text. 
        # We can send a JSON structure line by line, or just raw text.
        # For simplicity in this "Streaming" task, we'll stream the raw text of the answer.
        # But the user requirement says "Update the final output to be a dictionary containing... sources".
        # Streaming standard JSON is hard. 
        # PROPOSAL: We will stream the text chunks, and maybe append sources at the end?
        # OR: We yield a custom SSE event or JSON lines.
        # Let's try simple text streaming for the answer, and maybe print sources first?
        
        # Let's send a JSON object with sources first as a separate message if possible?
        # Or standard practice: Stream the markdown tokens.
        # We will separate sources and answer.
        # Since the user asked for a "dictionary containing... sources", that was for the static output.
        # For "Streaming", usually it's text.
        # Let's stream the text and return sources in a header? (Too rigid)
        # Let's stick to streaming the raw answer provided by Ollama.
        
        stream = ollama.chat(
            model='qwen2.5:14b',
            messages=[{'role': 'user', 'content': prompt}],
            stream=True,
        )
        
        full_response = ""
        for chunk in stream:
            content = chunk['message']['content']
            full_response += content
            yield content
        
        # Update History with full response
        chat_history.add_ai_message(full_response)
            
        # Optional: Append sources to the stream in markdown
        yield "\n\n**Sources:**\n"
        for s in sources:
            yield f"- {s['filename']} (Page {s['page_number']})\n"

    return StreamingResponse(iter_response(), media_type="text/plain")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
