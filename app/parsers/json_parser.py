"""JSON parser extracting structured records with exact JSON Pointer paths."""
import json
from pathlib import Path
from typing import Any, Dict, List
from app.parsers.base import BaseParser, ParserError
from app.schemas import ExtractedBlock, SourceLocation


class JSONParser(BaseParser):
    """Parses JSON files into structured records with RFC 6901 JSON Pointer paths."""

    def parse(self, file_path: Path, document_id: str, source_filename: str) -> List[ExtractedBlock]:
        self.check_non_empty(file_path)

        raw_bytes = file_path.read_bytes()
        text = None
        for enc in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                text = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise ParserError(f"Unable to decode JSON file '{source_filename}'.")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ParserError(
                f"Invalid JSON syntax in '{source_filename}': line {e.lineno}, column {e.colno} ({e.msg})"
            )

        blocks: List[ExtractedBlock] = []
        block_counter = 1

        def format_record(obj: Any) -> str:
            """Format a JSON object or primitive into human and LLM-readable text."""
            if isinstance(obj, dict):
                lines = []
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
                    else:
                        lines.append(f"{k}: {v}")
                return "\n".join(lines)
            elif isinstance(obj, list):
                return "\n".join(f"- {json.dumps(item, ensure_ascii=False)}" for item in obj)
            return str(obj)

        def traverse(node: Any, current_pointer: str):
            nonlocal block_counter

            if isinstance(node, list):
                if not node:
                    return
                # Check if elements are objects/records or primitives
                for idx, item in enumerate(node):
                    elem_ptr = f"{current_pointer}/{idx}"
                    if isinstance(item, (dict, list)):
                        content = f"[JSON Record at {elem_ptr}]\n{format_record(item)}"
                        blocks.append(
                            ExtractedBlock(
                                document_id=document_id,
                                source_filename=source_filename,
                                file_type=".json",
                                content=content,
                                block_id=f"json_block_{block_counter}",
                                source_location=SourceLocation(json_path=elem_ptr),
                                metadata={"json_type": type(item).__name__}
                            )
                        )
                        block_counter += 1
                    else:
                        # Primitives in array
                        content = f"[Value at {elem_ptr}]: {item}"
                        blocks.append(
                            ExtractedBlock(
                                document_id=document_id,
                                source_filename=source_filename,
                                file_type=".json",
                                content=content,
                                block_id=f"json_block_{block_counter}",
                                source_location=SourceLocation(json_path=elem_ptr),
                                metadata={"json_type": type(item).__name__}
                            )
                        )
                        block_counter += 1

            elif isinstance(node, dict):
                if not node:
                    return

                # Collect scalar properties into a top-level group
                scalars: Dict[str, Any] = {}
                complex_fields: Dict[str, Any] = {}

                for k, v in node.items():
                    # Escape JSON pointer tokens according to RFC 6901 (~ -> ~0, / -> ~1)
                    escaped_key = k.replace("~", "~0").replace("/", "~1")
                    if isinstance(v, (dict, list)):
                        complex_fields[escaped_key] = v
                    else:
                        scalars[k] = v

                if scalars:
                    ptr = current_pointer if current_pointer else "/"
                    content = f"[Properties at {ptr}]\n" + "\n".join(f"{k}: {v}" for k, v in scalars.items())
                    blocks.append(
                        ExtractedBlock(
                            document_id=document_id,
                            source_filename=source_filename,
                            file_type=".json",
                            content=content,
                            block_id=f"json_block_{block_counter}",
                            source_location=SourceLocation(json_path=ptr),
                            metadata={"is_property_group": True}
                        )
                    )
                    block_counter += 1

                for esc_k, v in complex_fields.items():
                    child_ptr = f"{current_pointer}/{esc_k}"
                    traverse(v, child_ptr)

            else:
                # Root primitive
                ptr = current_pointer if current_pointer else "/"
                content = f"[Root Value at {ptr}]: {node}"
                blocks.append(
                    ExtractedBlock(
                        document_id=document_id,
                        source_filename=source_filename,
                        file_type=".json",
                        content=content,
                        block_id=f"json_block_{block_counter}",
                        source_location=SourceLocation(json_path=ptr),
                        metadata={"json_type": type(node).__name__}
                    )
                )
                block_counter += 1

        traverse(data, "")

        if not blocks:
            raise ParserError(f"JSON document '{source_filename}' contains no extractable data.")

        return blocks
