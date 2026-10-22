"""Document ingestion pipeline orchestrating validation, deduplication, parsing, chunking, and dual-indexing."""
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid
from app.config import settings
from app.core.embeddings import EmbeddingManager
from app.core.vector_store import VectorStore
from app.core.keyword_index import KeywordIndex
from app.parsers.base import BaseParser, ParserError
from app.parsers.pdf_parser import PDFParser
from app.parsers.text_parser import TextParser
from app.parsers.markdown_parser import MarkdownParser
from app.parsers.docx_parser import DocxParser
from app.parsers.csv_parser import CSVParser
from app.parsers.html_parser import HTMLParser
from app.parsers.json_parser import JSONParser
from app.parsers.python_parser import PythonParser
from app.schemas import (
    DocumentInfo,
    DocumentUploadSummary,
    ExtractedBlock,
    Chunk
)
from app.services.chunking import FormatAwareChunker

logger = logging.getLogger(__name__)


class IngestionService:
    """Manages document validation, parsing across 8 file types, format-aware chunking, and dual indexing."""

    SUPPORTED_EXTENSIONS = {
        ".pdf": PDFParser,
        ".txt": TextParser,
        ".md": MarkdownParser,
        ".docx": DocxParser,
        ".csv": CSVParser,
        ".html": HTMLParser,
        ".htm": HTMLParser,
        ".json": JSONParser,
        ".py": PythonParser,
    }

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        keyword_index: Optional[KeywordIndex] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        chunker: Optional[FormatAwareChunker] = None,
        storage_directory: Optional[str] = None,
        upload_directory: Optional[str] = None,
    ):
        self.vector_store = vector_store or VectorStore()
        self.keyword_index = keyword_index or KeywordIndex()
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.chunker = chunker or FormatAwareChunker()
        
        self.storage_dir = Path(storage_directory or settings.storage_directory)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir = Path(upload_directory or settings.upload_directory)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.inventory_file = self.storage_dir / "documents.json"
        self._inventory: Dict[str, DocumentInfo] = {}
        self._load_inventory()

    def _load_inventory(self) -> None:
        """Load document inventory from disk."""
        if self.inventory_file.exists():
            try:
                data = json.loads(self.inventory_file.read_text(encoding="utf-8"))
                for doc_id, item in data.items():
                    self._inventory[doc_id] = DocumentInfo(**item)
            except Exception as e:
                logger.error(f"Failed to load document inventory: {e}")

    def _save_inventory(self) -> None:
        """Save document inventory to disk."""
        try:
            serialized = {
                doc_id: info.model_dump() for doc_id, info in self._inventory.items()
            }
            self.inventory_file.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save document inventory: {e}")

    def get_inventory(self) -> List[DocumentInfo]:
        """Return list of all registered documents."""
        return list(self._inventory.values())

    def get_document(self, document_id: str) -> Optional[DocumentInfo]:
        """Retrieve document metadata by document ID."""
        return self._inventory.get(document_id)

    @staticmethod
    def calculate_sha256(content: bytes) -> str:
        """Compute SHA-256 hash of raw file content."""
        return hashlib.sha256(content).hexdigest()

    def ingest_file(
        self,
        file_bytes: bytes,
        original_filename: str
    ) -> DocumentUploadSummary:
        """Ingest, validate, parse, chunk, embed, and index a single uploaded file."""
        ext = Path(original_filename).suffix.lower()

        # 1. Extension Validation
        if ext not in self.SUPPORTED_EXTENSIONS:
            return DocumentUploadSummary(
                original_filename=original_filename,
                document_id="",
                file_type=ext or "unknown",
                extracted_blocks=0,
                indexed_chunks=0,
                status="rejected",
                warnings=[
                    f"Unsupported file format '{ext}'. Supported formats are: "
                    f"{', '.join(sorted(self.SUPPORTED_EXTENSIONS.keys()))}"
                ]
            )

        # 2. Size Validation
        max_bytes = settings.max_upload_size_mb * 1024 * 1024
        if len(file_bytes) > max_bytes:
            return DocumentUploadSummary(
                original_filename=original_filename,
                document_id="",
                file_type=ext,
                extracted_blocks=0,
                indexed_chunks=0,
                status="rejected",
                warnings=[
                    f"File size ({len(file_bytes) / (1024 * 1024):.2f} MB) exceeds "
                    f"maximum configured limit of {settings.max_upload_size_mb} MB."
                ]
            )

        # 3. Empty file validation
        if len(file_bytes) == 0:
            return DocumentUploadSummary(
                original_filename=original_filename,
                document_id="",
                file_type=ext,
                extracted_blocks=0,
                indexed_chunks=0,
                status="rejected",
                warnings=["Uploaded file is empty (0 bytes)."]
            )

        content_hash = self.calculate_sha256(file_bytes)

        # 4. Deduplication Check
        for existing in self._inventory.values():
            if existing.content_hash == content_hash:
                return DocumentUploadSummary(
                    original_filename=original_filename,
                    document_id=existing.document_id,
                    file_type=existing.file_type,
                    extracted_blocks=0,
                    indexed_chunks=existing.chunk_count,
                    status="warning",
                    warnings=[
                        f"Document with identical content is already indexed under ID '{existing.document_id}' "
                        f"(filename: '{existing.source_filename}'). Skipped duplicate indexing."
                    ]
                )

        # 5. Save staging file
        doc_id = f"doc_{uuid.uuid4().hex[:10]}"
        staged_file = self.upload_dir / f"{doc_id}_{original_filename}"
        staged_file.write_bytes(file_bytes)

        # 6. Parse file with format-specific parser
        parser_cls = self.SUPPORTED_EXTENSIONS[ext]
        parser = parser_cls()
        warnings = []

        try:
            extracted_blocks = parser.parse(staged_file, doc_id, original_filename)
        except ParserError as pe:
            # Clean up staged file on parse failure
            if staged_file.exists():
                staged_file.unlink()
            return DocumentUploadSummary(
                original_filename=original_filename,
                document_id=doc_id,
                file_type=ext,
                extracted_blocks=0,
                indexed_chunks=0,
                status="failed",
                warnings=[str(pe)]
            )
        except Exception as e:
            if staged_file.exists():
                staged_file.unlink()
            return DocumentUploadSummary(
                original_filename=original_filename,
                document_id=doc_id,
                file_type=ext,
                extracted_blocks=0,
                indexed_chunks=0,
                status="failed",
                warnings=[f"Unexpected parser failure: {str(e)}"]
            )

        # 7. Format-aware chunking preserving source metadata
        chunks = self.chunker.chunk_document(extracted_blocks)
        if not chunks:
            return DocumentUploadSummary(
                original_filename=original_filename,
                document_id=doc_id,
                file_type=ext,
                extracted_blocks=len(extracted_blocks),
                indexed_chunks=0,
                status="failed",
                warnings=["Parsing produced no indexable chunks."]
            )

        # 8. Dual Indexing: Vector Embeddings + Keyword Index
        try:
            chunk_texts = [c.content for c in chunks]
            embeddings = self.embedding_manager.embed_batch(chunk_texts)
            self.vector_store.add_chunks(chunks, embeddings)
            self.keyword_index.add_chunks(chunks)
        except Exception as e:
            logger.error(f"Indexing error for doc {doc_id}: {e}")
            # Rollback
            self.vector_store.delete_document(doc_id)
            self.keyword_index.delete_document(doc_id)
            if staged_file.exists():
                staged_file.unlink()
            return DocumentUploadSummary(
                original_filename=original_filename,
                document_id=doc_id,
                file_type=ext,
                extracted_blocks=len(extracted_blocks),
                indexed_chunks=0,
                status="failed",
                warnings=[f"Indexing failure: {str(e)}"]
            )

        # 9. Register document in inventory
        now_iso = datetime.now(timezone.utc).isoformat()
        info = DocumentInfo(
            document_id=doc_id,
            source_filename=original_filename,
            file_type=ext,
            chunk_count=len(chunks),
            uploaded_at=now_iso,
            file_size_bytes=len(file_bytes),
            content_hash=content_hash
        )
        self._inventory[doc_id] = info
        self._save_inventory()

        return DocumentUploadSummary(
            original_filename=original_filename,
            document_id=doc_id,
            file_type=ext,
            extracted_blocks=len(extracted_blocks),
            indexed_chunks=len(chunks),
            status="success",
            warnings=warnings
        )

    def delete_document(self, document_id: str) -> bool:
        """Purge document and all its chunks from persistent storage, vector store, and BM25 index."""
        if document_id not in self._inventory:
            return False

        doc_info = self._inventory[document_id]

        # 1. Remove chunks from ChromaDB
        self.vector_store.delete_document(document_id)

        # 2. Remove chunks from BM25
        self.keyword_index.delete_document(document_id)

        # 3. Remove staged file
        staged_file = self.upload_dir / f"{document_id}_{doc_info.source_filename}"
        if staged_file.exists():
            staged_file.unlink()

        # 4. Remove from inventory
        del self._inventory[document_id]
        self._save_inventory()
        logger.info(f"Purged document '{document_id}' completely from system.")
        return True
