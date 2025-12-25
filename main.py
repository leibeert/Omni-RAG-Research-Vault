from pathlib import Path
from src.parser import PDFParser

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
    
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found.")
        return

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        try:
            for document in parser.parse(pdf_file):
                print(f"  - Extracted Page {document.page_number} (Hash: {document.metadata['id'][:8]}...)")
                # Future: Ingest into vector DB here
        except Exception as e:
            print(f"  X Failed to process {pdf_file.name}: {e}")

if __name__ == "__main__":
    main()
