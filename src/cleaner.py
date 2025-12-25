import re


class TextCleaner:
    """
    Handles text normalization and cleaning for academic PDFs.
    """

    def clean_text(self, text: str) -> str:
        """
        Main entry point for cleaning text.
        
        Args:
            text: Raw text extracted from PDF.
            
        Returns:
            Cleaned and normalized text.
        """
        if not text:
            return ""

        text = self._fix_ligatures(text)
        text = self._remove_artifacts(text)
        text = self._normalize_whitespace(text)
        return text

    def _fix_ligatures(self, text: str) -> str:
        """
        Fixes common broken ligatures found in PDFs.
        """
        # Common ligatures mapping
        ligatures = {
            "ﬁ": "fi",
            "ﬂ": "fl",
            "ﬀ": "ff",
            "ﬃ": "ffi",
            "ﬄ": "ffl",
            "ﬅ": "ft",
            "ﬆ": "st",
        }
        for lig, rep in ligatures.items():
            text = text.replace(lig, rep)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalizes whitespace:
        - Replaces multiple spaces with single space.
        - Handles hyphenation at line breaks.
        """
        # Fix hyphenation (e.g., "communi-\ncation" -> "communication")
        text = re.sub(r"-\n\s*", "", text)
        
        # Replace multiple whitespace (including newlines) with single space
        # Note: Depending on downstream needs, we might want to keep paragraph breaks.
        # For now, we'll normalize to single spaces to form continuous text, 
        # but let's preserve double newlines as paragraph breaks if needed.
        # The requirement says "normalizing whitespace".
        # Let's collapse all whitespace to single space for RAG chunks usually, 
        # or keep \n for structure. Let's do standard normalization.
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _remove_artifacts(self, text: str) -> str:
        """
        Removes common artifacts like page numbers and headers/footers.
        Note: This is heuristic-based and might not catch everything without layout info.
        """
        # Remove standalone page numbers (e.g., " 1 ", "Page 1 of 10")
        # Matches "Page X" or "Page X of Y" or just digits on a line (if we had lines)
        # Since we might have already normalized newlines in a different order, 
        # it's safer to run this before whitespace normalization if we relied on \n.
        # But here I'm calling it before _normalize_whitespace in clean_text, 
        # so newlines are still present.
        
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip empty lines
            if not stripped:
                continue
            
            # Heuristic: Skip if line is just a number (page number)
            if re.match(r'^\d+$', stripped):
                continue
                
            # Heuristic: Skip common header/footer patterns
            if re.match(r'^Page \d+ of \d+$', stripped, re.IGNORECASE):
                continue
            
            cleaned_lines.append(line)
            
        return '\n'.join(cleaned_lines)
