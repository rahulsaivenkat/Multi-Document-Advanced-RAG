"""CSV parser treating rows as structured records with column names and row numbers."""
import csv
import io
from pathlib import Path
from typing import List
from app.parsers.base import BaseParser, ParserError
from app.schemas import ExtractedBlock, SourceLocation


class CSVParser(BaseParser):
    """Parses CSV files into row-level structured records with exact row numbers and column names."""

    # Maximum rows to combine into a single block to balance retrieval granularity and context
    ROWS_PER_BLOCK = 5

    def parse(self, file_path: Path, document_id: str, source_filename: str) -> List[ExtractedBlock]:
        self.check_non_empty(file_path)

        raw_bytes = file_path.read_bytes()
        text = None
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                text = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise ParserError(f"Unable to decode CSV file '{source_filename}'.")

        stream = io.StringIO(text)
        try:
            reader = csv.reader(stream)
            rows = list(reader)
        except Exception as e:
            raise ParserError(f"Malformed CSV file '{source_filename}': {str(e)}")

        if not rows:
            raise ParserError(f"CSV file '{source_filename}' is empty.")

        # Header row is Row 1
        header_row = [col.strip() for col in rows[0]]
        if not any(header_row):
            raise ParserError(f"CSV file '{source_filename}' has an empty or invalid header row.")

        data_rows = rows[1:]
        if not data_rows:
            raise ParserError(f"CSV file '{source_filename}' contains headers but zero data rows.")

        blocks: List[ExtractedBlock] = []
        current_batch_lines: List[str] = []
        batch_start_row = 2
        block_counter = 1

        for idx, row in enumerate(data_rows):
            current_row_number = idx + 2  # 1-indexed (Row 1 is header)
            
            # Format row with column names
            formatted_cells = []
            for col_idx, header in enumerate(header_row):
                val = row[col_idx].strip() if col_idx < len(row) else ""
                if val:
                    formatted_cells.append(f"{header}: {val}")
                else:
                    formatted_cells.append(f"{header}: [N/A]")

            row_text = f"[Row {current_row_number}] " + " | ".join(formatted_cells)
            current_batch_lines.append(row_text)

            # Flush when batch reaches limit or at end of data
            if len(current_batch_lines) >= self.ROWS_PER_BLOCK or idx == len(data_rows) - 1:
                batch_end_row = current_row_number
                combined_content = "\n".join(current_batch_lines)
                
                blocks.append(
                    ExtractedBlock(
                        document_id=document_id,
                        source_filename=source_filename,
                        file_type=".csv",
                        content=combined_content,
                        block_id=f"row_{batch_start_row}_to_{batch_end_row}",
                        source_location=SourceLocation(
                            row_number=batch_start_row if batch_start_row == batch_end_row else None,
                            start_line=batch_start_row,
                            end_line=batch_end_row,
                            column_names=header_row
                        ),
                        metadata={
                            "start_row": batch_start_row,
                            "end_row": batch_end_row,
                            "column_count": len(header_row)
                        }
                    )
                )
                block_counter += 1
                current_batch_lines = []
                batch_start_row = current_row_number + 1

        return blocks
