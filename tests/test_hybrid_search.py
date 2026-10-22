"""Tests for BM25 keyword index, ChromaDB vector store, and Reciprocal Rank Fusion (RRF)."""
from pathlib import Path
import pytest
from app.core.hybrid_search import HybridSearchEngine
from app.core.keyword_index import KeywordIndex
from app.core.vector_store import VectorStore
from app.schemas import Chunk, SourceLocation


def make_chunk(cid: str, doc_id: str, content: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        document_id=doc_id,
        source_filename=f"{doc_id}.txt",
        file_type=".txt",
        content=content,
        source_location=SourceLocation(start_line=1, end_line=5)
    )


def test_bm25_keyword_index(tmp_path: Path):
    kw_index = KeywordIndex(persist_directory=str(tmp_path / "bm25"))
    c1 = make_chunk("c1", "doc1", "FastAPI is a modern Python web framework.")
    c2 = make_chunk("c2", "doc2", "ChromaDB provides persistent embedding vector storage.")
    c3 = make_chunk("c3", "doc3", "SentenceTransformers generates dense semantic embeddings.")

    kw_index.add_chunks([c1, c2, c3])
    assert kw_index.count() == 3

    # Query for exact keyword
    results = kw_index.query("FastAPI Python", top_k=2)
    assert len(results) > 0
    assert results[0][0].chunk_id == "c1"

    # Test document deletion
    deleted_count = kw_index.delete_document("doc1")
    assert deleted_count == 1
    assert kw_index.count() == 2
    post_results = kw_index.query("FastAPI", top_k=2)
    assert len(post_results) == 0


def test_reciprocal_rank_fusion_logic():
    engine = HybridSearchEngine(rrf_k=60)

    c1 = make_chunk("c1", "d1", "Chunk 1")
    c2 = make_chunk("c2", "d2", "Chunk 2")
    c3 = make_chunk("c3", "d3", "Chunk 3")

    # Semantic ranking: c1 (rank 1), c2 (rank 2)
    semantic_results = [(c1, 0.95), (c2, 0.80)]
    # BM25 ranking: c2 (rank 1), c3 (rank 2)
    bm25_results = [(c2, 5.2), (c3, 3.1)]

    # Expected RRF scores:
    # c2: rank 2 in semantic + rank 1 in bm25 -> 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    # c1: rank 1 in semantic -> 1/(60+1) = 0.016393
    # c3: rank 2 in bm25 -> 1/(60+2) = 0.016129
    fused = engine.fuse(semantic_results, bm25_results, candidate_limit=5)
    
    assert len(fused) == 3
    # c2 must rank #1 because it appeared in both rankings
    assert fused[0][0].chunk_id == "c2"
    assert fused[1][0].chunk_id == "c1"
    assert fused[2][0].chunk_id == "c3"


def test_vector_store_persistence(tmp_path: Path):
    vs = VectorStore(persist_directory=str(tmp_path / "chroma"))
    c1 = make_chunk("v1", "docA", "Artificial Intelligence research paper.")
    c2 = make_chunk("v2", "docB", "Database internals and storage engine design.")

    # Synthetic embeddings of dimension 4
    emb1 = [0.1, 0.2, 0.3, 0.4]
    emb2 = [0.9, 0.8, 0.7, 0.6]

    vs.add_chunks([c1, c2], [emb1, emb2])
    assert vs.count() == 2

    # Query with vector close to emb1
    results = vs.query(query_embedding=[0.11, 0.19, 0.31, 0.39], top_k=2)
    assert len(results) == 2
    assert results[0][0].chunk_id == "v1"

    # Test document deletion
    deleted = vs.delete_document("docA")
    assert deleted == 1
    assert vs.count() == 1
