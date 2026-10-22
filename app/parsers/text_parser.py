"""Plain text parser with encoding detection, paragraph preservation, and line range tracking."""
from pathlib import Path
from typing import List, Tuple
from app.parsers.base import BaseParser, ParserError
from app.schemas import ExtractedBlock, SourceLocation


class TextParser(BaseParser):
    """Parses plain text files, preserving 1-based line ranges and meaningful paragraph structure."""

    SUPPORTED_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    def _read_with_encoding(self, file_path: Path) -> Tuple[str, str]:
        """Try reading the file with supported encodings."""
        raw_bytes = file_path.read_bytes()
        for enc in self.SUPPORTED_ENCODINGS:
            try:
                text = raw_bytes.decode(enc)
                return text, enc
            except UnicodeDecodeError:
                continue
        raise ParserError(
            f"Unable to decode text file '{file_path.name}' using supported encodings: "
            f"{', '.join(self.SUPPORTED_ENCODINGS)}."
        )

    def parse(self, file_path: Path, document_id: str, source_filename: str) -> List[ExtractedBlock]:
        self.check_non_empty(file_path)

        text, encoding = self._read_with_encoding(file_path)
        lines = text.splitlines(keepends=False)

        if not lines or all(not line.strip() for line in lines):
            raise ParserError(f"Text file '{source_filename}' contains no readable non-whitespace text.")

        blocks: List[ExtractedBlock] = []
        current_paragraph_lines: List[str] = []
        para_start_line = 1
        block_counter = 1

        for line_idx, line in enumerate(lines, start=1):
            if not line.strip():
                # Paragraph boundary
                if current_paragraph_lines:
                    content = "\n".join(current_paragraph_lines).strip()
                    if content:
                        end_line = line_idx - 1
                        blocks.append(
                            ExtractedBlock(
                                document_id=document_id,
                                source_filename=source_filename,
                                file_type=".txt",
                                content=content,
                                block_id=f"para_{block_counter}",
                                source_location=SourceLocation(
                                    start_line=para_start_line,
                                    end_line=end_line
                                ),
                                metadata={"encoding": encoding}
                            )
                        )
                        block_counter += 1
                    current_paragraph_lines = []
                para_start_line = line_idx + 1
            else:
                if not current_paragraph_lines:
                    para_start_line = line_idx
                current_paragraph_lines.append(line)

        # Append last paragraph if present
        if current_paragraph_lines:
            content = "\n".join(current_paragraph_lines).strip()
            if content:
                end_line = len(lines)
                blocks.append(
                    ExtractedBlock(
                        document_id=document_id,
                        source_filename=source_filename,
                        file_type=".txt",
                        content=content,
                        block_id=f"para_{block_counter}",
                        source_location=SourceLocation(
                            start_line=para_start_line,
                            end_line=end_line
                        ),
                        metadata={"encoding": encoding}
                    )
                )

        return blocks
