"""Tests for citation mapping, verification, and anti-hallucination guardrails."""
from app.schemas import Chunk, Citation, SourceLocation
from app.services.citations import CitationService


def test_citation_creation_from_chunk():
    chunk = Chunk(
        chunk_id="chk_pdf_1",
        document_id="doc_pdf",
        source_filename="research.pdf",
        file_type=".pdf",
        content="Evaluation results show MRR of 0.88 across 100 queries.",
        source_location=SourceLocation(page_number=5)
    )

    citation = CitationService.create_citation("C1", chunk)
    assert citation.citation_id == "C1"
    assert citation.document_id == "doc_pdf"
    assert citation.source_filename == "research.pdf"
    assert citation.readable_location == "Page 5"
    assert "Evaluation results" in citation.excerpt
    assert citation.chunk_id == "chk_pdf_1"


def test_prepare_evidence_context():
    c1 = Chunk(
        chunk_id="chk_1",
        document_id="doc_py",
        source_filename="core.py",
        file_type=".py",
        content="def calculate(): pass",
        source_location=SourceLocation(symbol_name="def calculate()", start_line=12, end_line=15)
    )
    evidence_text, citations = CitationService.prepare_evidence_context([c1])
    
    assert len(citations) == 1
    assert "[C1] Source: core.py (Symbol: def calculate(), Lines 12-15)" in evidence_text
    assert "def calculate(): pass" in evidence_text


def test_filter_and_validate_citations_with_inline_tags():
    c1 = Citation(
        citation_id="C1",
        document_id="d1",
        source_filename="f1.txt",
        file_type=".txt",
        source_location=SourceLocation(start_line=1),
        readable_location="Line 1",
        excerpt="Excerpt 1",
        chunk_id="chunk_1"
    )
    c2 = Citation(
        citation_id="C2",
        document_id="d2",
        source_filename="f2.txt",
        file_type=".txt",
        source_location=SourceLocation(start_line=10),
        readable_location="Line 10",
        excerpt="Excerpt 2",
        chunk_id="chunk_2"
    )

    # Answer references only C2, and hallucinated [C99]
    answer = "The system utilizes ChromaDB [C2] for persistence. Unverified claim [C99]."
    verified = CitationService.filter_and_validate_citations(
        answer=answer,
        available_citations=[c1, c2],
        insufficient_evidence=False
    )

    # C2 should be kept, C99 hallucination discarded, C1 was not cited so omitted
    assert len(verified) == 1
    assert verified[0].citation_id == "C2"


def test_filter_and_validate_citations_insufficient_evidence():
    c1 = Citation(
        citation_id="C1",
        document_id="d1",
        source_filename="f1.txt",
        file_type=".txt",
        source_location=SourceLocation(start_line=1),
        readable_location="Line 1",
        excerpt="Excerpt 1",
        chunk_id="chunk_1"
    )
    answer = "Based on the provided documents, there is insufficient evidence to answer."
    verified = CitationService.filter_and_validate_citations(
        answer=answer,
        available_citations=[c1],
        insufficient_evidence=True
    )

    # When insufficient_evidence is True, citations MUST be empty
    assert len(verified) == 0
