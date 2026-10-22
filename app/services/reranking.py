"""CrossEncoder reranking service using SentenceTransformers."""
import logging
from typing import List, Optional, Tuple
from app.config import settings
from app.schemas import Chunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks candidate chunks against user questions using a pretrained CrossEncoder."""

    _instance: Optional["CrossEncoderReranker"] = None
    _model = None

    def __new__(cls, model_name: str = settings.cross_encoder_model):
        if cls._instance is None:
            cls._instance = super(CrossEncoderReranker, cls).__new__(cls)
            cls._instance.model_name = model_name
        return cls._instance

    def _get_model(self):
        """Lazy load CrossEncoder model once."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading CrossEncoder reranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Chunk, float]],
        top_k: int = settings.rerank_top_k
    ) -> List[Tuple[Chunk, float]]:
        """Score (query, chunk_content) pairs and return top_k reranked candidates.
        
        If model loading or inference fails, falls back safely to original candidate order.
        """
        if not candidates:
            return []

        # If candidates count is less than or equal to 1, reranking is trivial
        if len(candidates) <= 1:
            return candidates[:top_k]

        try:
            model = self._get_model()
            pairs = [[query, chunk.content] for chunk, _ in candidates]
            scores = model.predict(pairs)

            # Combine chunks with CrossEncoder scores
            scored_candidates: List[Tuple[Chunk, float]] = []
            for (chunk, _), score in zip(candidates, scores):
                scored_candidates.append((chunk, float(score)))

            # Sort descending by CrossEncoder score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            reranked = scored_candidates[:top_k]
            logger.debug(f"Reranked {len(candidates)} candidates to top {len(reranked)} evidence chunks.")
            return reranked

        except Exception as e:
            logger.warning(f"CrossEncoder inference failed ({e}). Falling back to RRF candidate order.")
            return candidates[:top_k]
