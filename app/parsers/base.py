"""Base parser interface and common parsing utilities."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from app.schemas import ExtractedBlock, SourceLocation


class ParserError(Exception):
    """Raised when document parsing fails due to corruption, unreadable content, or invalid formats."""
    def __init__(self, message: str, is_ocr_limitation: bool = False):
        super().__init__(message)
        self.message = message
        self.is_ocr_limitation = is_ocr_limitation


class BaseParser(ABC):
    """Abstract base class for all file format parsers."""

    @abstractmethod
    def parse(self, file_path: Path, document_id: str, source_filename: str) -> List[ExtractedBlock]:
        """Parse the given file and return a list of structured ExtractedBlocks.
        
        Args:
            file_path: Local filesystem path to the uploaded file.
            document_id: Unique identifier for the document.
            source_filename: Original filename as uploaded.
            
        Returns:
            List of ExtractedBlock instances with format-specific SourceLocation metadata.
            
        Raises:
            ParserError: If the file is empty, corrupted, unreadable, or requires unsupported OCR.
        """
        pass

    def check_non_empty(self, file_path: Path) -> None:
        """Verify the file exists and is not zero bytes."""
        if not file_path.exists():
            raise ParserError(f"File not found: {file_path.name}")
        if file_path.stat().st_size == 0:
            raise ParserError(f"File is empty (0 bytes): {file_path.name}")
