"""Pydantic data validation schemas for PaperMind AI."""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class SourceLocation(BaseModel):
    """Format-specific source location metadata."""
    # PDF
    page_number: Optional[int] = Field(default=None, description="1-indexed PDF page number")

    # Plain text, Markdown, Python
    start_line: Optional[int] = Field(default=None, description="1-indexed starting line number")
    end_line: Optional[int] = Field(default=None, description="1-indexed ending line number")

    # Markdown, DOCX, HTML
    heading_path: Optional[str] = Field(default=None, description="Hierarchical heading path, e.g. Setup > Windows")

    # DOCX
    paragraph_index: Optional[int] = Field(default=None, description="0-indexed document paragraph order")
    table_index: Optional[int] = Field(default=None, description="0-indexed document table index")

    # CSV, DOCX tables
    row_number: Optional[int] = Field(default=None, description="CSV row number or table row index")
    column_names: Optional[List[str]] = Field(default=None, description="Column names present in record")

    # JSON
    json_path: Optional[str] = Field(default=None, description="JSON pointer path, e.g. /users/0/name")

    # Python
    symbol_name: Optional[str] = Field(default=None, description="Function, method, or class name")

    # HTML
    block_id: Optional[str] = Field(default=None, description="Stable DOM block identifier")
    tag_name: Optional[str] = Field(default=None, description="HTML tag, e.g. h1, p, table")

    def to_readable_string(self) -> str:
        """Render a concise human-readable citation location string."""
        parts = []
        if self.page_number is not None:
            parts.append(f"Page {self.page_number}")
        if self.symbol_name:
            parts.append(f"Symbol: {self.symbol_name}")
        if self.heading_path:
            parts.append(f"Section: {self.heading_path}")
        if self.start_line is not None:
            if self.end_line is not None and self.end_line != self.start_line:
                parts.append(f"Lines {self.start_line}-{self.end_line}")
            else:
                parts.append(f"Line {self.start_line}")
        if self.row_number is not None:
            parts.append(f"Row {self.row_number}")
        if self.column_names:
            cols = ", ".join(self.column_names[:4])
            if len(self.column_names) > 4:
                cols += f" (+{len(self.column_names) - 4} more)"
            parts.append(f"Cols: [{cols}]")
        if self.paragraph_index is not None:
            parts.append(f"Paragraph #{self.paragraph_index}")
        if self.table_index is not None:
            parts.append(f"Table #{self.table_index}")
        if self.json_path:
            parts.append(f"Path: {self.json_path}")
        if self.block_id:
            parts.append(f"Block: {self.block_id}")
        return ", ".join(parts) if parts else "Document Root"


class ExtractedBlock(BaseModel):
    """Raw parsed block extracted from a document before chunking."""
    document_id: str
    source_filename: str
    file_type: str
    content: str
    block_id: str
    source_location: SourceLocation
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Indexed chunk ready for embedding and keyword indexing."""
    chunk_id: str
    document_id: str
    source_filename: str
    file_type: str
    content: str
    source_location: SourceLocation
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Evidence citation mapped to a retrieved chunk."""
    citation_id: str = Field(description="Citation reference tag, e.g. C1, C2")
    document_id: str
    source_filename: str
    file_type: str
    source_location: SourceLocation
    readable_location: str = Field(description="Formatted human-readable location")
    excerpt: str = Field(description="Verifiable source excerpt from the chunk")
    chunk_id: str


class DocumentUploadSummary(BaseModel):
    """Summary of a single uploaded document's ingestion."""
    original_filename: str
    document_id: str
    file_type: str
    extracted_blocks: int
    indexed_chunks: int
    status: Literal["success", "warning", "rejected", "failed"]
    warnings: List[str] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    """Response returned after processing file uploads."""
    uploaded: List[DocumentUploadSummary]
    failed: List[Dict[str, str]] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    """Metadata record of an indexed document in the inventory."""
    document_id: str
    source_filename: str
    file_type: str
    chunk_count: int
    uploaded_at: str
    file_size_bytes: int
    content_hash: str


class DocumentListResponse(BaseModel):
    """Inventory of all active indexed documents."""
    documents: List[DocumentInfo]
    total_documents: int
    total_chunks: int


class ChatMessage(BaseModel):
    """Message in conversation history."""
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    """User query request payload."""
    question: str = Field(min_length=1, max_length=2000, description="Natural language question")
    conversation_history: Optional[List[ChatMessage]] = Field(default=None, description="Previous messages")
    document_ids: Optional[List[str]] = Field(default=None, description="Optional document filter")
    rerank_top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Override top evidence count")


class RetrievalDiagnostic(BaseModel):
    """Operational diagnostics for retrieval and reranking."""
    query_transformed: bool
    transformed_query: Optional[str] = None
    semantic_results_count: int
    bm25_results_count: int
    rrf_candidates_count: int
    reranked_results_count: int


class ChatResponse(BaseModel):
    """Evidence-grounded response with verified citations."""
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    insufficient_evidence: bool = False
    diagnostics: Optional[RetrievalDiagnostic] = None


class HealthResponse(BaseModel):
    """Safe system health report without credentials."""
    status: str
    version: str
    embedding_model: str
    cross_encoder_model: str
    groq_configured: bool
    indexed_documents_count: int
    indexed_chunks_count: int
