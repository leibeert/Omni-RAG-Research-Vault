import time
import os
import fitz
import ollama
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion import IngestionPipeline
from src.database import OllamaEmbeddingFunction

def create_dummy_pdf(path: Path, pages: int = 50):
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((50, 50), f"This is page {i+1} of the benchmark document.\nHere is some content to ingest.")
    doc.save(path)
    print(f"Created {pages}-page dummy PDF at {path}")

def run_benchmarks():
    print("=== Starting Performance Benchmarks ===")
    
    # Setup
    base_dir = Path(__file__).parent.parent
    bench_dir = base_dir / "benchmarks"
    bench_dir.mkdir(exist_ok=True)
    
    test_pdf = bench_dir / "benchmark_50pg.pdf"
    
    if not test_pdf.exists():
        create_dummy_pdf(test_pdf, pages=50)
        
    # 1. Ingestion Speed
    print("\n[Test 1] Ingestion Speed (50 Pages)")
    pipeline = IngestionPipeline()
    start_time = time.time()
    try:
        pipeline.ingest_file(test_pdf)
        ingest_time = time.time() - start_time
        print(f"Time taken: {ingest_time:.2f} seconds")
    except Exception as e:
        print(f"Ingestion failed: {e}")

    # 2. Embedding Generation Speed
    print("\n[Test 2] Embedding Generation Speed")
    embed_fn = OllamaEmbeddingFunction()
    query = "What is the content of the benchmark document?"
    
    start_time = time.time()
    _ = embed_fn([query])
    embed_time = time.time() - start_time
    print(f"Time taken (1 query): {embed_time:.4f} seconds")

    # 3. LLM Generation Speed
    print("\n[Test 3] LLM Response Speed")
    prompt = "Explain quantum computing in 50 words."
    
    start_time = time.time()
    _ = ollama.chat(model='qwen2.5:14b', messages=[{'role': 'user', 'content': prompt}])
    llm_time = time.time() - start_time
    print(f"Time taken (full response): {llm_time:.2f} seconds")

    print("\n=== Benchmarks Complete ===")

if __name__ == "__main__":
    run_benchmarks()
