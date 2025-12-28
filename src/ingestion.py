from typing import List, Any
import logging
from src.parser import PDFParser, Document
from src.database import Database
from src.utils import setup_logging

logger = setup_logging(__name__)

class RecursiveCharacterTextSplitter:
    """
    Splits text into chunks of a given size with overlap, preserving words/paragraphs.
    Simulated simple logic to avoid langchain dependency if not needed, 
    but ensures context is kept better than naive splitting.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        chunks = []
        if not text:
            return []
            
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            
            # Adjust end to nearest whitespace to avoid cutting words
            if end < len(text):
                while end > start and text[end] not in [' ', '\n', '.']:
                    end -= 1
                if end == start: # Force split if no whitespace found
                     end = start + self.chunk_size
            
            chunks.append(text[start:end].strip())
            
            start = end - self.chunk_overlap
            if start < 0: start = 0 # Safety
            
            # Avoid infinite loop if overlap >= size (bad config) or progress stalled
            if end <= start:
                start = end 
                
        return chunks

class IngestionPipeline:
    def __init__(self):
        self.parser = PDFParser()
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.db = Database()

    def ingest_file(self, file_path: Any):
        logger.info(f"Starting ingestion for {file_path}")
        try:
            raw_docs = self.parser.parse(file_path)
            
            # Now Chunking
            chunked_docs = []
            for doc in raw_docs:
                chunks = self.splitter.split_text(doc.content)
                for i, chunk in enumerate(chunks):
                    new_doc = Document(
                        content=chunk, 
                        metadata={
                            **doc.metadata, 
                            "chunk_id": i,
                            "parent_id": doc.metadata.get("id", "unknown")
                        }
                    )
                    chunked_docs.append(new_doc)
            
            logger.info(f"Split into {len(chunked_docs)} chunks. Indexing...")
            self.db.add_documents(chunked_docs)
            logger.info("Ingestion complete.")
            
        except Exception as e:
            logger.error(f"Ingestion failed for {file_path}: {e}")
            raise

if __name__ == "__main__":
    # Example usage for manual execution
    import sys
    from pathlib import Path
    
    pipeline = IngestionPipeline()
    
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if target.exists():
             pipeline.ingest_file(target)
    else:
        # Default scan
        data_dir = Path("data")
        if data_dir.exists():
            for f in data_dir.glob("*.pdf"):
                pipeline.ingest_file(f)
