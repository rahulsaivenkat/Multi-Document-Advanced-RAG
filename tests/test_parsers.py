"""Unit tests for all 8 document parsers and edge cases."""
from pathlib import Path
import pytest
from app.parsers.base import ParserError
from app.parsers.pdf_parser import PDFParser
from app.parsers.text_parser import TextParser
from app.parsers.markdown_parser import MarkdownParser
from app.parsers.docx_parser import DocxParser
from app.parsers.csv_parser import CSVParser
from app.parsers.html_parser import HTMLParser
from app.parsers.json_parser import JSONParser
from app.parsers.python_parser import PythonParser


def test_pdf_parser_success(sample_pdf: Path):
    parser = PDFParser()
    blocks = parser.parse(sample_pdf, "doc_pdf", sample_pdf.name)
    assert len(blocks) == 2
    # Verify 1-based page numbering
    assert blocks[0].source_location.page_number == 1
    assert "Overview" in blocks[0].content
    assert blocks[1].source_location.page_number == 2
    assert "ChromaDB" in blocks[1].content


def test_pdf_parser_scanned_ocr_rejection(sample_scanned_pdf: Path):
    parser = PDFParser()
    with pytest.raises(ParserError) as exc_info:
        parser.parse(sample_scanned_pdf, "doc_scanned", sample_scanned_pdf.name)
    assert exc_info.value.is_ocr_limitation is True
    assert "OCR" in str(exc_info.value)


def test_text_parser(sample_txt: Path):
    parser = TextParser()
    blocks = parser.parse(sample_txt, "doc_txt", sample_txt.name)
    assert len(blocks) == 2
    # Check 1-based line ranges
    assert blocks[0].source_location.start_line == 1
    assert blocks[0].source_location.end_line == 2
    assert "Architecture" in blocks[0].content
    assert blocks[1].source_location.start_line == 4
    assert "Evaluation metrics" in blocks[1].content


def test_markdown_parser(sample_md: Path):
    parser = MarkdownParser()
    blocks = parser.parse(sample_md, "doc_md", sample_md.name)
    assert len(blocks) >= 2
    
    # Check heading path tracking
    code_blocks = [b for b in blocks if b.metadata.get("is_code_block")]
    assert len(code_blocks) == 1
    assert "pip install" in code_blocks[0].content
    assert "Windows Setup" in code_blocks[0].source_location.heading_path
    assert code_blocks[0].source_location.start_line is not None


def test_docx_parser(sample_docx: Path):
    parser = DocxParser()
    blocks = parser.parse(sample_docx, "doc_docx", sample_docx.name)
    assert len(blocks) >= 2
    
    # Verify paragraph extraction and heading path
    para_blocks = [b for b in blocks if b.source_location.paragraph_index is not None]
    assert len(para_blocks) >= 2
    assert any("Architecture" in b.content for b in para_blocks)
    
    # Verify table extraction coordinates
    table_blocks = [b for b in blocks if b.source_location.table_index is not None]
    assert len(table_blocks) >= 1
    assert table_blocks[0].source_location.table_index == 0
    assert table_blocks[0].source_location.row_number is not None
    # Ensure no fabricated page numbers
    assert all(b.source_location.page_number is None for b in blocks)


def test_csv_parser(sample_csv: Path):
    parser = CSVParser()
    blocks = parser.parse(sample_csv, "doc_csv", sample_csv.name)
    assert len(blocks) >= 1
    
    first_block = blocks[0]
    assert "Alice Smith" in first_block.content
    assert "Principal Engineer" in first_block.content
    assert "id" in first_block.source_location.column_names
    assert "name" in first_block.source_location.column_names
    # Verify row numbering accounts for header row (Row 1)
    assert first_block.source_location.start_line == 2


def test_html_parser(sample_html: Path):
    parser = HTMLParser()
    blocks = parser.parse(sample_html, "doc_html", sample_html.name)
    assert len(blocks) >= 2
    
    # Verify scripts and styles were discarded
    all_content = " ".join(b.content for b in blocks)
    assert "malicious script" not in all_content
    assert "color: red" not in all_content
    assert "SentenceTransformers" in all_content
    assert any(b.source_location.tag_name == "table" for b in blocks)


def test_json_parser(sample_json: Path):
    parser = JSONParser()
    blocks = parser.parse(sample_json, "doc_json", sample_json.name)
    assert len(blocks) >= 3
    
    # Verify RFC 6901 JSON pointer paths
    paths = [b.source_location.json_path for b in blocks]
    assert "/servers/0" in paths
    assert "/servers/1" in paths
    assert any("prod-east" in b.content for b in blocks)


def test_python_parser(sample_python: Path):
    parser = PythonParser()
    blocks = parser.parse(sample_python, "doc_py", sample_python.name)
    assert len(blocks) >= 3
    
    # Verify symbols and line numbers
    symbols = [b.source_location.symbol_name for b in blocks if b.source_location.symbol_name]
    assert any("Calculator" in s for s in symbols)
    assert any("compute_square_root" in s for s in symbols)
    assert all(b.source_location.start_line is not None for b in blocks)
    assert all(b.source_location.end_line is not None for b in blocks)


def test_empty_files_rejection(tmp_path: Path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")
    
    parser = TextParser()
    with pytest.raises(ParserError):
        parser.parse(empty_file, "doc_empty", "empty.txt")


def test_malformed_json_rejection(tmp_path: Path):
    bad_json = tmp_path / "broken.json"
    bad_json.write_text("{ unquoted_key: 123, }", encoding="utf-8")
    
    parser = JSONParser()
    with pytest.raises(ParserError) as exc:
        parser.parse(bad_json, "doc_bad_json", "broken.json")
    assert "Invalid JSON syntax" in str(exc.value)
