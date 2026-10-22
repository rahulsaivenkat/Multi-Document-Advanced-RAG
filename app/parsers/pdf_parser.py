"""PDF document parser using PyMuPDF with page-level extraction and OCR limitation detection."""
from pathlib import Path
from typing import List
import pymupdf as fitz  # PyMuPDF
from app.parsers.base import BaseParser, ParserError
from app.schemas import ExtractedBlock, SourceLocation


class PDFParser(BaseParser):
    """Parses PDF documents into page-level blocks preserving 1-based page numbers."""

    # Minimum average extractable characters per page to consider a document non-scanned/readable
    MIN_CHAR_THRESHOLD = 25

    def parse(self, file_path: Path, document_id: str, source_filename: str) -> List[ExtractedBlock]:
        self.check_non_empty(file_path)

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            raise ParserError(f"Failed to open or decode PDF file '{source_filename}': {str(e)}")

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise ParserError(f"PDF file '{source_filename}' contains zero pages.")

        blocks: List[ExtractedBlock] = []
        total_extracted_chars = 0
        non_empty_pages = 0

        for page_idx in range(total_pages):
            page_num = page_idx + 1  # 1-based page index
            try:
                page = doc[page_idx]
                page_text = page.get_text("text").strip()
            except Exception as e:
                # Malformed page text extraction
                page_text = ""

            if page_text:
                total_extracted_chars += len(page_text)
                non_empty_pages += 1
                blocks.append(
                    ExtractedBlock(
                        document_id=document_id,
                        source_filename=source_filename,
                        file_type=".pdf",
                        content=page_text,
                        block_id=f"page_{page_num}",
                        source_location=SourceLocation(page_number=page_num),
                        metadata={"total_pages": total_pages, "page_char_count": len(page_text)}
                    )
                )

        doc.close()

        # Scanned PDF / Image-only PDF detection
        if total_extracted_chars < self.MIN_CHAR_THRESHOLD:
            raise ParserError(
                f"PDF '{source_filename}' contains insufficient extractable text "
                f"({total_extracted_chars} characters across {total_pages} pages). "
                "Scanned or image-only PDFs require an Optical Character Recognition (OCR) engine, "
                "which is not currently available.",
                is_ocr_limitation=True
            )

        return blocks
