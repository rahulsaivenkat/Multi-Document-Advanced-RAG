"""Hybrid retrieval engine combining semantic search and BM25 via Reciprocal Rank Fusion (RRF)."""
import logging
from typing import Dict, List, Optional, Tuple
from app.config import settings
from app.schemas import Chunk

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """Merges semantic and BM25 search rankings using standard Reciprocal Rank Fusion."""

    def __init__(self, rrf_k: int = settings.rrf_k):
        self.rrf_k = rrf_k

    def fuse(
        self,
        semantic_results: List[Tuple[Chunk, float]],
        bm25_results: List[Tuple[Chunk, float]],
        candidate_limit: int = 25
    ) -> List[Tuple[Chunk, float]]:
        """Combine dense and sparse search rankings using Reciprocal Rank Fusion.
        
        RRF(d) = sum_{r in R} 1 / (k + rank_r(d))
        Where rank is 1-indexed.
        """
        rrf_scores: Dict[str, float] = {}
        chunk_registry: Dict[str, Chunk] = {}

        # 1. Process Semantic Results
        for rank_zero, (chunk, _) in enumerate(semantic_results):
            rank = rank_zero + 1  # 1-indexed rank
            chunk_registry[chunk.chunk_id] = chunk
            score = 1.0 / (self.rrf_k + rank)
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + score

        # 2. Process BM25 Keyword Results
        for rank_zero, (chunk, _) in enumerate(bm25_results):
            rank = rank_zero + 1  # 1-indexed rank
            chunk_registry[chunk.chunk_id] = chunk
            score = 1.0 / (self.rrf_k + rank)
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + score

        # 3. Sort by aggregated RRF score descending
        sorted_chunk_ids = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True
        )

        # 4. Form final candidate list up to limit
        candidates: List[Tuple[Chunk, float]] = []
        for cid in sorted_chunk_ids[:candidate_limit]:
            candidates.append((chunk_registry[cid], rrf_scores[cid]))

        logger.debug(
            f"RRF fused {len(semantic_results)} semantic and {len(bm25_results)} BM25 results "
            f"into {len(candidates)} unique candidates."
        )
        return candidates
