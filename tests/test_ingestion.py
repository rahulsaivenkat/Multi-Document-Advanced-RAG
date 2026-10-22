"""Tests for document ingestion, deduplication, and lifecycle deletion."""
from pathlib import Path
import pytest
from app.core.keyword_index import KeywordIndex
from app.core.vector_store import VectorStore
from app.services.ingestion import IngestionService


def test_ingestion_text_document(sample_txt: Path, tmp_path: Path):
    vs = VectorStore(persist_directory=str(tmp_path / "chroma"))
    ki = KeywordIndex(persist_directory=str(tmp_path / "bm25"))
    service = IngestionService(
        vector_store=vs,
        keyword_index=ki,
        storage_directory=str(tmp_path / "storage_txt"),
        upload_directory=str(tmp_path / "uploads_txt")
    )

    raw_bytes = sample_txt.read_bytes()
    summary = service.ingest_file(raw_bytes, sample_txt.name)

    assert summary.status == "success"
    assert summary.file_type == ".txt"
    assert summary.indexed_chunks >= 1
    assert vs.count() == summary.indexed_chunks
    assert ki.count() == summary.indexed_chunks

    # Verify inventory record
    doc_info = service.get_document(summary.document_id)
    assert doc_info is not None
    assert doc_info.source_filename == sample_txt.name


def test_ingestion_deduplication(sample_csv: Path, tmp_path: Path):
    vs = VectorStore(persist_directory=str(tmp_path / "chroma_dedup"))
    ki = KeywordIndex(persist_directory=str(tmp_path / "bm25_dedup"))
    service = IngestionService(
        vector_store=vs,
        keyword_index=ki,
        storage_directory=str(tmp_path / "storage_dedup"),
        upload_directory=str(tmp_path / "uploads_dedup")
    )

    raw_bytes = sample_csv.read_bytes()
    # First upload
    first = service.ingest_file(raw_bytes, sample_csv.name)
    assert first.status == "success"
    initial_chunks = vs.count()

    # Second upload with identical bytes
    second = service.ingest_file(raw_bytes, "duplicate_copy.csv")
    assert second.status == "warning"
    assert "already indexed" in second.warnings[0]
    # Verify index was not duplicated
    assert vs.count() == initial_chunks


def test_ingestion_unsupported_format(tmp_path: Path):
    service = IngestionService(
        storage_directory=str(tmp_path / "storage_unsupported"),
        upload_directory=str(tmp_path / "uploads_unsupported")
    )
    summary = service.ingest_file(b"some content", "archive.zip")
    assert summary.status == "rejected"
    assert "Unsupported file format" in summary.warnings[0]


def test_ingestion_empty_file(tmp_path: Path):
    service = IngestionService(
        storage_directory=str(tmp_path / "storage_empty"),
        upload_directory=str(tmp_path / "uploads_empty")
    )
    summary = service.ingest_file(b"", "empty.txt")
    assert summary.status == "rejected"
    assert "empty" in summary.warnings[0]


def test_ingestion_and_deletion(sample_python: Path, tmp_path: Path):
    vs = VectorStore(persist_directory=str(tmp_path / "chroma_del"))
    ki = KeywordIndex(persist_directory=str(tmp_path / "bm25_del"))
    service = IngestionService(
        vector_store=vs,
        keyword_index=ki,
        storage_directory=str(tmp_path / "storage_del"),
        upload_directory=str(tmp_path / "uploads_del")
    )

    raw_bytes = sample_python.read_bytes()
    summary = service.ingest_file(raw_bytes, sample_python.name)
    assert summary.status == "success"
    doc_id = summary.document_id

    # Delete document
    deleted = service.delete_document(doc_id)
    assert deleted is True
    assert vs.count() == 0
    assert ki.count() == 0
    assert service.get_document(doc_id) is None
