"""Verifiable citation service building and mapping citations directly from retrieved chunk metadata."""
import re
from typing import Dict, List, Optional, Set, Tuple
from app.schemas import Citation, Chunk, SourceLocation


class CitationService:
    """Creates verifiable citations from retrieved chunks and maps citations referenced in generated answers."""

    @staticmethod
    def create_citation(citation_id: str, chunk: Chunk) -> Citation:
        """Construct a Citation object directly from chunk metadata and content."""
        clean_content = chunk.content.strip().replace("\n", " ")
        excerpt = clean_content[:220] + ("..." if len(clean_content) > 220 else "")

        return Citation(
            citation_id=citation_id,
            document_id=chunk.document_id,
            source_filename=chunk.source_filename,
            file_type=chunk.file_type,
            source_location=chunk.source_location,
            readable_location=chunk.source_location.to_readable_string(),
            excerpt=excerpt,
            chunk_id=chunk.chunk_id
        )

    @staticmethod
    def prepare_evidence_context(chunks: List[Chunk]) -> Tuple[str, List[Citation]]:
        """Format retrieved chunks into numbered context blocks [C1], [C2] for LLM prompt."""
        citations: List[Citation] = []
        context_blocks: List[str] = []

        for idx, chunk in enumerate(chunks, start=1):
            c_tag = f"C{idx}"
            citation = CitationService.create_citation(c_tag, chunk)
            citations.append(citation)

            block_text = (
                f"[{c_tag}] Source: {chunk.source_filename} ({citation.readable_location})\n"
                f"{chunk.content.strip()}"
            )
            context_blocks.append(block_text)

        evidence_str = "\n\n".join(context_blocks)
        return evidence_str, citations

    @staticmethod
    def filter_and_validate_citations(
        answer: str,
        available_citations: List[Citation],
        insufficient_evidence: bool = False
    ) -> List[Citation]:
        """Validate citation tags in the answer against available evidence chunks.
        
        Guarantees that:
        1. No unretrieved/hallucinated citations are returned.
        2. If evidence is insufficient, returns an empty citation list.
        3. If specific [C#] tags are in the answer, returns only those matching citations.
        4. If no explicit tags were used but answer is grounded, includes the provided evidence citations.
        """
        if insufficient_evidence or not available_citations:
            return []

        # Find tags like [C1], [C2], [c1] in the answer text
        cited_tags: Set[str] = set()
        matches = re.findall(r"\[C(\d+)\]", answer, flags=re.IGNORECASE)
        for num in matches:
            cited_tags.add(f"C{num}")

        citation_map: Dict[str, Citation] = {c.citation_id.upper(): c for c in available_citations}

        if cited_tags:
            # Return only citations explicitly referenced that exist in available evidence
            verified = [
                citation_map[tag.upper()] for tag in cited_tags if tag.upper() in citation_map
            ]
            # Maintain original presentation order
            verified.sort(key=lambda c: int(c.citation_id.replace("C", "")))
            return verified

        # If LLM answered without inline tags but evidence was grounded, provide the evidence chunks
        return available_citations
