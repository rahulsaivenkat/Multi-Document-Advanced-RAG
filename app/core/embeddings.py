"""Embedding manager providing singleton SentenceTransformers model initialization."""
import logging
from typing import List, Optional
import numpy as np
from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Singleton wrapper for SentenceTransformer embedding generation."""

    _instance: Optional["EmbeddingManager"] = None
    _model = None

    def __new__(cls, model_name: str = settings.embedding_model):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            cls._instance.model_name = model_name
        return cls._instance

    def _get_model(self):
        """Lazy load the embedding model once."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Embed a single query or text string."""
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()
