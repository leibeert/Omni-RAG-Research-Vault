import json
import os
import sys
import ollama
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import Database
from src.search import Reranker, search_and_answer
from src.parser import Document

def load_dataset(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def evaluate_faithfulness(query, answer, context):
    """
    Checks if the answer is derived faithfully from the context.
    """
    prompt = f"""You are an impartial judge. Evaluate if the provided Answer is derived strictly from the Context.
    
    Context:
    {context}
    
    Answer:
    {answer}
    
    Return ONLY 'YES' if the answer is supported by the context, otherwise 'NO'.
    """
    
    response = ollama.chat(model='qwen2.5:14b', messages=[{'role': 'user', 'content': prompt}])
    content = response['message']['content'].strip().upper()
    return 1 if "YES" in content else 0

def evaluate_relevancy(query, answer):
    """
    Checks if the answer actually answers the question.
    """
    prompt = f"""You are an impartial judge. Evaluate if the provided Answer is a relevant and direct response to the Question.
    
    Question:
    {query}
    
    Answer:
    {answer}
    
    Return ONLY 'YES' if the answer is relevant, otherwise 'NO'.
    """
    
    response = ollama.chat(model='qwen2.5:14b', messages=[{'role': 'user', 'content': prompt}])
    content = response['message']['content'].strip().upper()
    return 1 if "YES" in content else 0

def run_eval():
    print("=== Starting RAG Evaluation ===")
    
    # Paths
    base_dir = Path(__file__).parent.parent
    dataset_path = base_dir / "tests" / "eval_dataset.json"
    
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return

    data = load_dataset(str(dataset_path))
    
    # Initialize Pipeline
    # Using the test DB we created in test_search.py or the main one?
    # Ideally we should use a controlled environment. 
    # For this script, let's assume valid data is in the default DB or we rely on what's there.
    # To make it robust for *this* specific test set (which matches the dummy data in test_search.py),
    # we might need to ingest that dummy data first if it's not there.
    # However, the user asked for a generic eval script. 
    # Let's assume the DB is populated with relevant data or we just run it.
    # Note: The provided eval_dataset.json matches the dummy data from test_search.py implicitly.
    
    print("Initializing Database...")
    db = Database() # Uses default config
    reranker = Reranker()
    
    faithfulness_scores = []
    relevancy_scores = []
    
    for item in tqdm(data, desc="Evaluating"):
        query = item['question']
        
        # Run Pipeline
        try:
            result = search_and_answer(query, db, reranker)
            generated_answer = result['answer']
            context = result['context']
            
            # Evaluate
            faith_score = evaluate_faithfulness(query, generated_answer, context)
            rel_score = evaluate_relevancy(query, generated_answer)
            
            faithfulness_scores.append(faith_score)
            relevancy_scores.append(rel_score)
            
            # Optional: Print detailed results for first few
            if len(faithfulness_scores) <= 3:
                print(f"\nQ: {query}\nA: {generated_answer}\nFaithful: {faith_score}, Relevant: {rel_score}\n")
                
        except Exception as e:
            print(f"Error evaluating {query}: {e}")
            
    # Aggregates
    avg_faith = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
    avg_rel = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0
    
    print("\n=== Evaluation Results ===")
    print(f"Average Faithfulness: {avg_faith:.2f}")
    print(f"Average Answer Relevancy: {avg_rel:.2f}")
    print("==========================")
    
if __name__ == "__main__":
    run_eval()
