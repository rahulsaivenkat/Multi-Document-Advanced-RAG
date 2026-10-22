"""FastAPI router endpoints for health, document management, and RAG chat."""
import logging
from typing import List, Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from app.config import settings
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentUploadResponse,
    DocumentUploadSummary,
    HealthResponse
)
from app.services.generation import GenerationService
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalPipeline

logger = logging.getLogger(__name__)
router = APIRouter()

# Global service instances (initialized once)
ingestion_service = IngestionService()
retrieval_pipeline = RetrievalPipeline()
generation_service = GenerationService()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def get_health() -> HealthResponse:
    """Safe system health report without exposing credentials or internal secrets."""
    is_groq_set = bool(
        settings.groq_api_key
        and settings.groq_api_key.strip()
        and settings.groq_api_key != "your_groq_api_key_here"
    )
    docs = ingestion_service.get_inventory()
    total_chunks = ingestion_service.vector_store.count()

    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        embedding_model=settings.embedding_model,
        cross_encoder_model=settings.cross_encoder_model,
        groq_configured=is_groq_set,
        indexed_documents_count=len(docs),
        indexed_chunks_count=total_chunks
    )


@router.post("/documents/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_documents(
    files: List[UploadFile] = File(..., description="One or more documents (.pdf, .txt, .md, .docx, .csv, .html, .json, .py)")
) -> DocumentUploadResponse:
    """Upload, validate, parse, chunk, and index multiple supported documents."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided in upload request."
        )

    uploaded_summaries: List[DocumentUploadSummary] = []
    failed_items = []

    for file in files:
        filename = file.filename or "unknown"
        try:
            content = await file.read()
            summary = ingestion_service.ingest_file(
                file_bytes=content,
                original_filename=filename
            )
            uploaded_summaries.append(summary)
        except Exception as e:
            logger.error(f"Failed to process upload for '{filename}': {e}")
            failed_items.append({
                "filename": filename,
                "error": str(e)
            })

    return DocumentUploadResponse(
        uploaded=uploaded_summaries,
        failed=failed_items
    )


@router.get("/documents", response_model=DocumentListResponse, tags=["Documents"])
def list_documents() -> DocumentListResponse:
    """List all indexed documents with metadata and chunk counts."""
    docs = ingestion_service.get_inventory()
    total_chunks = sum(d.chunk_count for d in docs)
    return DocumentListResponse(
        documents=docs,
        total_documents=len(docs),
        total_chunks=total_chunks
    )


@router.delete("/documents/{document_id}", status_code=status.HTTP_200_OK, tags=["Documents"])
def delete_document(document_id: str):
    """Delete a document and all its chunks from persistent vector and keyword stores."""
    success = ingestion_service.delete_document(document_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' not found in index."
        )
    return {"message": f"Document '{document_id}' successfully purged from all indexes."}


@router.post("/chat", response_model=ChatResponse, tags=["RAG"])
def chat(request: ChatRequest) -> ChatResponse:
    """Query across uploaded documents with hybrid search, reranking, and verified citations."""
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or blank."
        )

    # 1. Retrieve evidence chunks via hybrid pipeline
    evidence_chunks, diagnostics = retrieval_pipeline.retrieve(
        question=question,
        conversation_history=request.conversation_history,
        document_ids=request.document_ids,
        rerank_top_k=request.rerank_top_k
    )

    # 2. Synthesize evidence-grounded answer
    answer, citations, insufficient_evidence = generation_service.generate(
        question=question,
        evidence_chunks=evidence_chunks
    )

    return ChatResponse(
        answer=answer,
        citations=citations,
        insufficient_evidence=insufficient_evidence,
        diagnostics=diagnostics
    )
