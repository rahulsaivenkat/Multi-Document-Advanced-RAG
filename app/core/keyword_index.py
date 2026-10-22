"""Persistent BM25 keyword index using rank-bm25 with tokenization preserving code identifiers."""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from rank_bm25 import BM25Okapi
from app.config import settings
from app.schemas import Chunk, SourceLocation

logger = logging.getLogger(__name__)


class KeywordIndex:
    """Manages an inverted BM25 index with token preservation for identifiers, symbols, and text."""

    def __init__(self, persist_directory: str = settings.bm25_persist_directory):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.storage_file = self.persist_directory / "bm25_corpus.json"
        
        self.chunks_map: Dict[str, Chunk] = {}  # chunk_id -> Chunk
        self.tokenized_corpus: List[List[str]] = []
        self.chunk_ids: List[str] = []
        self.bm25: Optional[BM25Okapi] = None

        self._load_from_disk()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize preserving code identifiers, camelCase, snake_case, paths, and filenames."""
        # Find alphanumeric sequences with optional internal dots, underscores, dashes
        tokens = re.findall(r"[a-zA-Z0-9_\-\.\/]+", text.lower())
        # Also split snake_case and camelCase into subwords for hybrid matching
        expanded_tokens = list(tokens)
        for t in tokens:
            if "_" in t:
                expanded_tokens.extend([part for part in t.split("_") if part])
            if "." in t:
                expanded_tokens.extend([part for part in t.split(".") if part])
        return [t for t in expanded_tokens if len(t) > 1]

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 model from the in-memory chunks_map."""
        self.chunk_ids = list(self.chunks_map.keys())
        self.tokenized_corpus = [
            self.tokenize(self.chunks_map[cid].content) for cid in self.chunk_ids
        ]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def _save_to_disk(self) -> None:
        """Persist corpus and metadata to disk."""
        try:
            records = [
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "source_filename": chunk.source_filename,
                    "file_type": chunk.file_type,
                    "content": chunk.content,
                    "source_location": chunk.source_location.model_dump(),
                    "metadata": chunk.metadata
                }
                for chunk in self.chunks_map.values()
            ]
            self.storage_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to persist BM25 index: {e}")

    def _load_from_disk(self) -> None:
        """Load persisted BM25 corpus from disk on startup."""
        if self.storage_file.exists():
            try:
                data = json.loads(self.storage_file.read_text(encoding="utf-8"))
                for item in data:
                    loc = SourceLocation(**item["source_location"])
                    chunk = Chunk(
                        chunk_id=item["chunk_id"],
                        document_id=item["document_id"],
                        source_filename=item["source_filename"],
                        file_type=item["file_type"],
                        content=item["content"],
                        source_location=loc,
                        metadata=item.get("metadata", {})
                    )
                    self.chunks_map[chunk.chunk_id] = chunk
                self._rebuild_index()
                logger.info(f"Loaded {len(self.chunks_map)} chunks into BM25 index from disk.")
            except Exception as e:
                logger.error(f"Failed to load BM25 index from disk: {e}")

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Index new or updated chunks and persist."""
        if not chunks:
            return
        for chunk in chunks:
            self.chunks_map[chunk.chunk_id] = chunk
        self._rebuild_index()
        self._save_to_disk()
        logger.info(f"Indexed {len(chunks)} chunks in BM25. Total chunks: {len(self.chunks_map)}")

    def query(
        self,
        query_text: str,
        top_k: int = settings.bm25_top_k,
        document_ids: Optional[List[str]] = None
    ) -> List[Tuple[Chunk, float]]:
        """Run BM25 query over indexed corpus."""
        if self.bm25 is None or not self.chunk_ids:
            return []

        tokens = self.tokenize(query_text)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        
        # Pair with chunk IDs
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: List[Tuple[Chunk, float]] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0.0:
                break
            
            c_id = self.chunk_ids[idx]
            chunk = self.chunks_map[c_id]

            if document_ids and chunk.document_id not in document_ids:
                continue

            results.append((chunk, score))
            if len(results) >= top_k:
                break

        return results

    def delete_document(self, document_id: str) -> int:
        """Purge all chunks for a document from BM25 and rebuild."""
        to_delete = [
            cid for cid, chunk in self.chunks_map.items() if chunk.document_id == document_id
        ]
        if not to_delete:
            return 0

        for cid in to_delete:
            del self.chunks_map[cid]

        self._rebuild_index()
        self._save_to_disk()
        logger.info(f"Deleted {len(to_delete)} chunks for document '{document_id}' from BM25.")
        return len(to_delete)

    def count(self) -> int:
        """Total chunks in BM25 index."""
        return len(self.chunks_map)
