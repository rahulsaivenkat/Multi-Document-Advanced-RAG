"""Unified hybrid retrieval pipeline combining query transform, dense search, BM25, RRF, and CrossEncoder."""
import logging
from typing import List, Optional, Tuple
from app.config import settings
from app.core.embeddings import EmbeddingManager
from app.core.vector_store import VectorStore
from app.core.keyword_index import KeywordIndex
from app.core.hybrid_search import HybridSearchEngine
from app.schemas import ChatMessage, Chunk, RetrievalDiagnostic
from app.services.query_transform import QueryTransformer
from app.services.reranking import CrossEncoderReranker

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Coordinates query transformation, semantic vector search, BM25 keyword search, RRF fusion, and reranking."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        keyword_index: Optional[KeywordIndex] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        hybrid_engine: Optional[HybridSearchEngine] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        query_transformer: Optional[QueryTransformer] = None,
    ):
        self.vector_store = vector_store or VectorStore()
        self.keyword_index = keyword_index or KeywordIndex()
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.hybrid_engine = hybrid_engine or HybridSearchEngine()
        self.reranker = reranker or CrossEncoderReranker()
        self.query_transformer = query_transformer or QueryTransformer()

    def retrieve(
        self,
        question: str,
        conversation_history: Optional[List[ChatMessage]] = None,
        document_ids: Optional[List[str]] = None,
        rerank_top_k: Optional[int] = None
    ) -> Tuple[List[Chunk], RetrievalDiagnostic]:
        """Execute full hybrid retrieval pipeline.
        
        Returns:
            Tuple of (top_evidence_chunks, diagnostic_metadata)
        """
        target_top_k = rerank_top_k or settings.rerank_top_k

        # 1. Query Transformation
        transformed_query = self.query_transformer.transform(question, conversation_history)
        query_transformed = (transformed_query.strip() != question.strip())
        search_query = transformed_query if query_transformed else question

        # 2. Semantic Search via ChromaDB
        query_embedding = self.embedding_manager.embed_text(search_query)
        semantic_results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=settings.semantic_top_k,
            document_ids=document_ids
        )

        # 3. Keyword Search via BM25
        bm25_results = self.keyword_index.query(
            query_text=search_query,
            top_k=settings.bm25_top_k,
            document_ids=document_ids
        )

        # 4. Reciprocal Rank Fusion (RRF)
        rrf_candidates = self.hybrid_engine.fuse(
            semantic_results=semantic_results,
            bm25_results=bm25_results,
            candidate_limit=25
        )

        # 5. CrossEncoder Reranking
        # We rerank against the user's intended question
        reranked_results = self.reranker.rerank(
            query=question,
            candidates=rrf_candidates,
            top_k=target_top_k
        )

        evidence_chunks = [chunk for chunk, _ in reranked_results]

        diagnostics = RetrievalDiagnostic(
            query_transformed=query_transformed,
            transformed_query=transformed_query if query_transformed else None,
            semantic_results_count=len(semantic_results),
            bm25_results_count=len(bm25_results),
            rrf_candidates_count=len(rrf_candidates),
            reranked_results_count=len(evidence_chunks)
        )

        return evidence_chunks, diagnostics
