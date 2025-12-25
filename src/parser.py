import hashlib
import fitz  # type: ignore
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, Dict, Any, Optional

from src.cleaner import TextCleaner


@dataclass
class Document:
    """
    Represents a processed document page or chunk.
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def source_file(self) -> str:
        return self.metadata.get("filename", "unknown")
    
    @property
    def page_number(self) -> int:
        return self.metadata.get("page_number", -1)


class PDFParser:
    """
    Handles ingestion of PDF documents using PyMuPDF.
    """

    def __init__(self, cleaner: Optional[TextCleaner] = None):
        self.cleaner = cleaner or TextCleaner()

    def parse(self, file_path: Path) -> Iterator[Document]:
        """
        Parses a PDF file and yields Document objects (one per page).

        Args:
            file_path: Path to the PDF file.

        Yields:
            Document objects containing cleaned text and metadata.
        
        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file is not a PDF.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {file_path}")

        # Calculate file hash for deduplication
        file_hash = self._calculate_file_hash(file_path)
        filename = file_path.name

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            raise ValueError(f"Failed to open PDF {file_path}: {e}")

        for page_num, page in enumerate(doc):
            # Extract text
            raw_text = page.get_text()
            
            # Clean text
            cleaned_text = self.cleaner.clean_text(raw_text)
            
            # Skip empty pages
            if not cleaned_text:
                continue

            metadata = {
                "filename": filename,
                "file_path": str(file_path.absolute()),
                "page_number": page_num + 1,  # 1-based indexing for humans
                "file_hash": file_hash,
                # Create a unique ID for this page: hash of (file_hash + page)
                "id": hashlib.sha256(f"{file_hash}_{page_num}".encode()).hexdigest()
            }

            yield Document(content=cleaned_text, metadata=metadata)
        
        doc.close()

    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculates SHA-256 hash of the file content.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
