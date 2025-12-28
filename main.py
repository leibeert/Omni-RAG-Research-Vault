from pathlib import Path
from src.parser import PDFParser
from src.database import Database

def main():
    """
    Main entry point for the RAG Research Vault ingestion pipeline.
    """
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    print(f"Scanning {data_dir} for PDFs...")
    
    parser = PDFParser()
    db = Database() # Initialize Database (Hybrid Search)
    
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found.")
        return

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        try:
            documents = []
            for document in parser.parse(pdf_file):
                print(f"  - Parsed Page {document.page_number}")
                documents.append(document)
            
            if documents:
                print(f"  > Indexing {len(documents)} pages into Vector DB & BM25...")
                db.add_documents(documents)
                print("  > Done.")
                
        except Exception as e:
            print(f"  X Failed to process {pdf_file.name}: {e}")

if __name__ == "__main__":
    main()
