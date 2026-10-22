"""Markdown parser with heading hierarchy tracking, line ranges, and code block preservation."""
from pathlib import Path
from typing import List, Tuple
from app.parsers.base import BaseParser, ParserError
from app.schemas import ExtractedBlock, SourceLocation


class MarkdownParser(BaseParser):
    """Parses Markdown files while tracking hierarchical heading paths and line ranges."""

    SUPPORTED_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1"]

    def _read_text(self, file_path: Path) -> Tuple[str, str]:
        raw_bytes = file_path.read_bytes()
        for enc in self.SUPPORTED_ENCODINGS:
            try:
                return raw_bytes.decode(enc), enc
            except UnicodeDecodeError:
                continue
        raise ParserError(f"Unable to decode markdown file '{file_path.name}'.")

    def parse(self, file_path: Path, document_id: str, source_filename: str) -> List[ExtractedBlock]:
        self.check_non_empty(file_path)

        text, encoding = self._read_text(file_path)
        lines = text.splitlines(keepends=False)

        if not lines or all(not line.strip() for line in lines):
            raise ParserError(f"Markdown file '{source_filename}' contains no readable content.")

        blocks: List[ExtractedBlock] = []
        heading_stack: List[Tuple[int, str]] = []  # list of (level, title)
        
        current_block_lines: List[str] = []
        block_start_line = 1
        block_counter = 1
        in_code_block = False
        code_fence = ""

        def current_heading_path() -> str:
            return " > ".join([title for _, title in heading_stack]) if heading_stack else "Document Root"

        def flush_current_block(end_line: int, is_code: bool = False):
            nonlocal current_block_lines, block_counter
            content = "\n".join(current_block_lines).strip()
            if content:
                blocks.append(
                    ExtractedBlock(
                        document_id=document_id,
                        source_filename=source_filename,
                        file_type=".md",
                        content=content,
                        block_id=f"md_block_{block_counter}",
                        source_location=SourceLocation(
                            start_line=block_start_line,
                            end_line=end_line,
                            heading_path=current_heading_path(),
                        ),
                        metadata={"is_code_block": is_code, "encoding": encoding}
                    )
                )
                block_counter += 1
            current_block_lines = []

        for line_idx, line in enumerate(lines, start=1):
            stripped = line.strip()

            # Handle fenced code blocks
            if stripped.startswith("```") or stripped.startswith("~~~"):
                fence = stripped[:3]
                if not in_code_block:
                    # Flush any pending prose before starting code block
                    if current_block_lines:
                        flush_current_block(line_idx - 1, is_code=False)
                    in_code_block = True
                    code_fence = fence
                    block_start_line = line_idx
                    current_block_lines.append(line)
                    continue
                elif in_code_block and stripped.startswith(code_fence):
                    current_block_lines.append(line)
                    flush_current_block(line_idx, is_code=True)
                    in_code_block = False
                    block_start_line = line_idx + 1
                    continue

            if in_code_block:
                current_block_lines.append(line)
                continue

            # Check for Markdown headings: # Heading
            if stripped.startswith("#"):
                header_parts = stripped.lstrip("#")
                level = len(stripped) - len(header_parts)
                title = header_parts.strip()
                if level > 0 and title:
                    # Flush prior content
                    if current_block_lines:
                        flush_current_block(line_idx - 1, is_code=False)

                    # Update heading stack
                    while heading_stack and heading_stack[-1][0] >= level:
                        heading_stack.pop()
                    heading_stack.append((level, title))

                    # Start next block with heading line
                    block_start_line = line_idx
                    current_block_lines.append(line)
                    continue

            # Blank line: paragraph boundary
            if not stripped:
                if current_block_lines:
                    flush_current_block(line_idx - 1, is_code=False)
                block_start_line = line_idx + 1
            else:
                if not current_block_lines:
                    block_start_line = line_idx
                current_block_lines.append(line)

        # Flush final block
        if current_block_lines:
            flush_current_block(len(lines), is_code=in_code_block)

        return blocks
