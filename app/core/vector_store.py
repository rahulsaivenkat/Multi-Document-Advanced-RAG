"""Persistent ChromaDB vector store wrapper for semantic chunk search."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings
from app.schemas import Chunk, SourceLocation

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages persistent ChromaDB vector storage and semantic retrieval."""

    COLLECTION_NAME = "papermind_chunks"

    def __init__(self, persist_directory: str = settings.chroma_persist_directory):
        self.persist_directory = persist_directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "PaperMind AI Document Chunks"}
        )

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Add chunks and precomputed embeddings to ChromaDB."""
        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        
        # Chroma requires flat metadata values (str, int, float, bool)
        metadatas = []
        for chunk in chunks:
            meta = {
                "document_id": chunk.document_id,
                "source_filename": chunk.source_filename,
                "file_type": chunk.file_type,
                "source_location_json": chunk.source_location.model_dump_json(),
            }
            # Flatten primitive metadata if present
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[f"custom_{k}"] = v
            metadatas.append(meta)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Upserted {len(chunks)} chunks into ChromaDB collection '{self.COLLECTION_NAME}'.")

    def query(
        self,
        query_embedding: List[float],
        top_k: int = settings.semantic_top_k,
        document_ids: Optional[List[str]] = None
    ) -> List[Tuple[Chunk, float]]:
        """Semantic search using query vector."""
        total_count = self.collection.count()
        if total_count == 0:
            return []

        n_results = min(top_k, total_count)
        where_filter = None
        if document_ids:
            if len(document_ids) == 1:
                where_filter = {"document_id": document_ids[0]}
            else:
                where_filter = {"document_id": {"$in": document_ids}}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        chunks_with_scores: List[Tuple[Chunk, float]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(ids)

        for c_id, content, meta, dist in zip(ids, docs, metas, distances):
            loc_json = meta.get("source_location_json", "{}")
            source_loc = SourceLocation.model_validate_json(loc_json)
            chunk = Chunk(
                chunk_id=c_id,
                document_id=meta.get("document_id", ""),
                source_filename=meta.get("source_filename", ""),
                file_type=meta.get("file_type", ""),
                content=content,
                source_location=source_loc
            )
            # Cosine distance in chroma: similarity ~ 1.0 - distance
            similarity = max(0.0, 1.0 - float(dist))
            chunks_with_scores.append((chunk, similarity))

        return chunks_with_scores

    def delete_document(self, document_id: str) -> int:
        """Remove all chunks associated with a document ID from ChromaDB."""
        # Check current count
        try:
            current = self.collection.get(where={"document_id": document_id}, include=[])
            count = len(current["ids"]) if current and "ids" in current else 0
            if count > 0:
                self.collection.delete(where={"document_id": document_id})
                logger.info(f"Deleted {count} chunks for document '{document_id}' from ChromaDB.")
            return count
        except Exception as e:
            logger.error(f"Error deleting document '{document_id}' from ChromaDB: {e}")
            return 0

    def count(self) -> int:
        """Total number of chunks in ChromaDB."""
        return self.collection.count()
