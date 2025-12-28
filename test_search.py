import os
import shutil
from src.database import Database
from src.search import Reranker, search_and_answer
from src.parser import Document

def test_full_pipeline():
    print("=== Testing Hybrid Search, Reranking & Citations ===")
    
    # 1. Setup Test DB
    test_db_dir = "data/test_chroma_db_v2"
    if os.path.exists(test_db_dir):
        shutil.rmtree(test_db_dir)
        
    print(f"Initializing Database at {test_db_dir}...")
    db = Database(persist_directory=test_db_dir, collection_name="test_hybrid")
    
    # 2. Add Dummy Data
    print("Ingesting dummy documents...")
    docs = [
        Document(content="The capital of France is Paris. It is known for the Eiffel Tower.", metadata={"filename": "geography.pdf", "page_number": 10, "id": "1"}),
        Document(content="Python is a popular programming language. It is great for AI.", metadata={"filename": "coding.pdf", "page_number": 5, "id": "2"}),
        Document(content="The Eiffel Tower was constructed in 1889.", metadata={"filename": "history.pdf", "page_number": 3, "id": "3"}),
        Document(content="Machine learning is a subset of artificial intelligence.", metadata={"filename": "ai.pdf", "page_number": 1, "id": "4"}),
        Document(content="Paris is a beautiful city in Europe.", metadata={"filename": "travel.pdf", "page_number": 2, "id": "5"}),
        Document(content="Snakes are reptiles. Some are venomous.", metadata={"filename": "biology.pdf", "page_number": 99, "id": "6"}),
    ]
    db.add_documents(docs)
    
    # 3. Initialize Reranker
    print("Initializing Reranker...")
    reranker = Reranker()
    
    # 4. Perform Search
    query = "What is the capital of France?"
    print(f"\nQuery: '{query}'")
    
    result = search_and_answer(query, db, reranker)
    
    print("\n=== Final Answer ===")
    print(result['answer'])
    print("\n=== Sources ===")
    for source in result['sources']:
        print(f"- {source['filename']} (Page {source['page_number']})")
        
    # Validation
    assert result['sources'], "No sources returned!"
    assert any("geography.pdf" in s['filename'] for s in result['sources']), "Expected source not found!"
    print("\n[SUCCESS] Pipeline verified.")

    # Cleanup
    if os.path.exists(test_db_dir):
        shutil.rmtree(test_db_dir)

if __name__ == "__main__":
    test_full_pipeline()
