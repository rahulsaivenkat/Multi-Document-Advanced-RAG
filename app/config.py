"""Application configuration and environment settings."""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application
    app_name: str = "PaperMind AI"
    app_version: str = "1.0.0"
    debug: bool = False

    # Groq API Configuration
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key for LLM queries")
    groq_model: str = Field(default="llama-3.3-70b-versatile", description="Default Groq LLM model")
    groq_temperature: float = Field(default=0.1, ge=0.0, le=1.0)

    # Persistence Directories (relative to project root or absolute)
    storage_directory: str = Field(default="data/storage")
    chroma_persist_directory: str = Field(default="data/storage/chroma")
    bm25_persist_directory: str = Field(default="data/storage/bm25")
    upload_directory: str = Field(default="data/uploads")

    # Embeddings & Reranking Models
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    cross_encoder_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Chunking Configuration
    chunk_size: int = Field(default=500, gt=50)
    chunk_overlap: int = Field(default=80, ge=0)

    # Retrieval & Reranking Parameters
    semantic_top_k: int = Field(default=15, gt=0)
    bm25_top_k: int = Field(default=15, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    rerank_top_k: int = Field(default=5, gt=0)

    # Constraints
    max_upload_size_mb: int = Field(default=25, gt=0)

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        for path_str in [self.chroma_persist_directory, self.bm25_persist_directory, self.upload_directory]:
            Path(path_str).mkdir(parents=True, exist_ok=True)


# Global settings singleton
settings = Settings()
settings.ensure_directories()
