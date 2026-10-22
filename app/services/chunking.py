"""Format-aware chunking service preserving source metadata across all document types."""
import uuid
from typing import List
from app.config import settings
from app.schemas import Chunk, ExtractedBlock, SourceLocation


class FormatAwareChunker:
    """Chunks ExtractedBlocks into indexed Chunks while strictly preserving SourceLocation metadata."""

    def __init__(self, chunk_size: int = settings.chunk_size, chunk_overlap: int = settings.chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_block(self, block: ExtractedBlock) -> List[Chunk]:
        """Split a single ExtractedBlock into one or more Chunks based on format characteristics."""
        content = block.content.strip()
        if not content:
            return []

        # Formats that are already structured at block level (CSV batches, JSON records, Python functions/classes)
        # only split if they exceed the chunk size threshold by more than 1.5x
        if block.file_type in [".csv", ".json", ".py"]:
            if len(content) <= self.chunk_size * 2:
                return [
                    Chunk(
                        chunk_id=f"{block.document_id}_{block.block_id}_c1",
                        document_id=block.document_id,
                        source_filename=block.source_filename,
                        file_type=block.file_type,
                        content=content,
                        source_location=block.source_location.model_copy(),
                        metadata=block.metadata.copy()
                    )
                ]

        # For long text blocks (PDF pages, long Markdown sections, DOCX paragraphs, long text),
        # perform character/word sliding window chunking with overlap
        chunks: List[Chunk] = []
        words = content.split()
        
        # Approximate word limit assuming ~5 characters per word
        word_chunk_size = max(50, self.chunk_size // 5)
        word_overlap = max(10, self.chunk_overlap // 5)

        if len(words) <= word_chunk_size:
            chunks.append(
                Chunk(
                    chunk_id=f"{block.document_id}_{block.block_id}_c1",
                    document_id=block.document_id,
                    source_filename=block.source_filename,
                    file_type=block.file_type,
                    content=content,
                    source_location=block.source_location.model_copy(),
                    metadata=block.metadata.copy()
                )
            )
            return chunks

        sub_idx = 1
        start_w = 0
        while start_w < len(words):
            end_w = min(start_w + word_chunk_size, len(words))
            sub_content = " ".join(words[start_w:end_w]).strip()

            if sub_content:
                # Maintain the same source location (e.g. PDF page number, heading path, line range)
                loc_copy = block.source_location.model_copy()
                chunks.append(
                    Chunk(
                        chunk_id=f"{block.document_id}_{block.block_id}_c{sub_idx}",
                        document_id=block.document_id,
                        source_filename=block.source_filename,
                        file_type=block.file_type,
                        content=sub_content,
                        source_location=loc_copy,
                        metadata=block.metadata.copy()
                    )
                )
                sub_idx += 1

            if end_w >= len(words):
                break
            start_w += (word_chunk_size - word_overlap)

        return chunks

    def chunk_document(self, blocks: List[ExtractedBlock]) -> List[Chunk]:
        """Process all extracted blocks for a document into indexed chunks."""
        all_chunks: List[Chunk] = []
        for block in blocks:
            all_chunks.extend(self.chunk_block(block))
        return all_chunks
