"""Tests for format-aware chunking preserving source locations."""
from app.schemas import ExtractedBlock, SourceLocation
from app.services.chunking import FormatAwareChunker


def test_chunking_preserves_pdf_page_metadata():
    chunker = FormatAwareChunker(chunk_size=100, chunk_overlap=20)
    long_content = " ".join([f"Word{i}" for i in range(150)])
    
    block = ExtractedBlock(
        document_id="doc_1",
        source_filename="paper.pdf",
        file_type=".pdf",
        content=long_content,
        block_id="page_3",
        source_location=SourceLocation(page_number=3)
    )

    chunks = chunker.chunk_block(block)
    assert len(chunks) > 1
    # Verify every sub-chunk carries page 3
    for c in chunks:
        assert c.source_location.page_number == 3
        assert c.source_filename == "paper.pdf"
        assert c.file_type == ".pdf"


def test_chunking_preserves_python_symbol():
    chunker = FormatAwareChunker(chunk_size=200, chunk_overlap=30)
    code_content = "def test_func():\n    return 42"
    
    block = ExtractedBlock(
        document_id="doc_2",
        source_filename="app.py",
        file_type=".py",
        content=code_content,
        block_id="func_test",
        source_location=SourceLocation(
            symbol_name="def test_func()",
            start_line=10,
            end_line=12
        )
    )

    chunks = chunker.chunk_block(block)
    assert len(chunks) == 1
    assert chunks[0].source_location.symbol_name == "def test_func()"
    assert chunks[0].source_location.start_line == 10
    assert chunks[0].source_location.end_line == 12
