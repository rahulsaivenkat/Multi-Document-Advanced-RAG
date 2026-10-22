"""Tests for CrossEncoder reranker order and safe fallback."""
from unittest.mock import MagicMock
from app.schemas import Chunk, SourceLocation
from app.services.reranking import CrossEncoderReranker


def make_chunk(cid: str, content: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        document_id="doc1",
        source_filename="doc1.txt",
        file_type=".txt",
        content=content,
        source_location=SourceLocation(start_line=1, end_line=5)
    )


def test_cross_encoder_rerank_order():
    reranker = CrossEncoderReranker()
    
    # Mock the internal model to test scoring logic deterministically
    mock_model = MagicMock()
    # Assume 3 candidates, mock predicts highest score for index 1, medium for index 2, lowest for index 0
    mock_model.predict.return_value = [0.1, 0.9, 0.4]
    reranker._model = mock_model

    c0 = make_chunk("c0", "General introduction")
    c1 = make_chunk("c1", "Exact matching technical answer")
    c2 = make_chunk("c2", "Partially relevant content")
    candidates = [(c0, 0.5), (c1, 0.5), (c2, 0.5)]

    results = reranker.rerank(query="What is the exact answer?", candidates=candidates, top_k=2)

    assert len(results) == 2
    # c1 has score 0.9, so must be #1
    assert results[0][0].chunk_id == "c1"
    assert results[0][1] == 0.9
    # c2 has score 0.4, so must be #2
    assert results[1][0].chunk_id == "c2"
    assert results[1][1] == 0.4


def test_cross_encoder_fallback_on_exception():
    reranker = CrossEncoderReranker()
    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("Inference device error")
    reranker._model = mock_model

    c0 = make_chunk("c0", "Content A")
    c1 = make_chunk("c1", "Content B")
    candidates = [(c0, 0.8), (c1, 0.7)]

    # Must not raise an exception; falls back gracefully to candidate order
    results = reranker.rerank("Query", candidates, top_k=2)
    assert len(results) == 2
    assert results[0][0].chunk_id == "c0"
    assert results[1][0].chunk_id == "c1"
