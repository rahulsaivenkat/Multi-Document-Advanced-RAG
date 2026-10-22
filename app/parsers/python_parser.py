"""Python source code parser using AST and line slicing without code execution."""
import ast
from pathlib import Path
from typing import List, Optional, Set
from app.parsers.base import BaseParser, ParserError
from app.schemas import ExtractedBlock, SourceLocation


class PythonParser(BaseParser):
    """Parses Python source files using AST to extract syntax-aware functions, classes, and methods."""

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
            raise ParserError(f"Unable to decode Python source file '{source_filename}'.")

        lines = text.splitlines(keepends=True)
        if not lines or all(not line.strip() for line in lines):
            raise ParserError(f"Python file '{source_filename}' contains no readable code.")

        # Attempt AST parsing
        try:
            tree = ast.parse(text, filename=source_filename)
            return self._parse_ast(tree, lines, document_id, source_filename)
        except SyntaxError as e:
            # Graceful fallback for syntactically invalid Python code
            return self._parse_fallback_lines(lines, document_id, source_filename, str(e))

    def _get_slice(self, lines: List[str], start_line: int, end_line: int) -> str:
        """Extract lines slice 1-indexed."""
        return "".join(lines[start_line - 1:end_line]).rstrip()

    def _parse_ast(
        self,
        tree: ast.Module,
        lines: List[str],
        document_id: str,
        source_filename: str
    ) -> List[ExtractedBlock]:
        blocks: List[ExtractedBlock] = []
        covered_lines: Set[int] = set()
        block_counter = 1

        # Check module-level docstring
        docstring = ast.get_docstring(tree)
        if docstring:
            first_stmt = tree.body[0] if tree.body else None
            if first_stmt and hasattr(first_stmt, "lineno") and hasattr(first_stmt, "end_lineno"):
                s_line = first_stmt.lineno
                e_line = first_stmt.end_lineno or s_line
                blocks.append(
                    ExtractedBlock(
                        document_id=document_id,
                        source_filename=source_filename,
                        file_type=".py",
                        content=f'"""{docstring}"""',
                        block_id=f"py_module_doc_{block_counter}",
                        source_location=SourceLocation(
                            symbol_name="module_docstring",
                            start_line=s_line,
                            end_line=e_line
                        ),
                        metadata={"is_docstring": True}
                    )
                )
                block_counter += 1
                for l in range(s_line, e_line + 1):
                    covered_lines.add(l)

        # Traverse top-level AST statements
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                s_line = node.lineno
                e_line = getattr(node, "end_lineno", s_line)
                func_content = self._get_slice(lines, s_line, e_line)
                blocks.append(
                    ExtractedBlock(
                        document_id=document_id,
                        source_filename=source_filename,
                        file_type=".py",
                        content=func_content,
                        block_id=f"py_func_{node.name}_{block_counter}",
                        source_location=SourceLocation(
                            symbol_name=f"def {node.name}()",
                            start_line=s_line,
                            end_line=e_line
                        ),
                        metadata={"type": "function", "name": node.name}
                    )
                )
                block_counter += 1
                for l in range(s_line, e_line + 1):
                    covered_lines.add(l)

            elif isinstance(node, ast.ClassDef):
                cls_start = node.lineno
                cls_end = getattr(node, "end_lineno", cls_start)

                # Extract class methods individually if class is large, or whole class
                method_nodes = [
                    n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]

                if len(method_nodes) > 0 and (cls_end - cls_start > 30):
                    # Class header / docstring
                    header_end = method_nodes[0].lineno - 1
                    header_content = self._get_slice(lines, cls_start, header_end)
                    if header_content.strip():
                        blocks.append(
                            ExtractedBlock(
                                document_id=document_id,
                                source_filename=source_filename,
                                file_type=".py",
                                content=header_content,
                                block_id=f"py_class_header_{node.name}_{block_counter}",
                                source_location=SourceLocation(
                                    symbol_name=f"class {node.name}",
                                    start_line=cls_start,
                                    end_line=header_end
                                ),
                                metadata={"type": "class_definition", "class_name": node.name}
                            )
                        )
                        block_counter += 1
                        for l in range(cls_start, header_end + 1):
                            covered_lines.add(l)

                    # Extract each method
                    for m in method_nodes:
                        m_start = m.lineno
                        m_end = getattr(m, "end_lineno", m_start)
                        m_content = self._get_slice(lines, m_start, m_end)
                        blocks.append(
                            ExtractedBlock(
                                document_id=document_id,
                                source_filename=source_filename,
                                file_type=".py",
                                content=m_content,
                                block_id=f"py_method_{node.name}_{m.name}_{block_counter}",
                                source_location=SourceLocation(
                                    symbol_name=f"{node.name}.{m.name}()",
                                    start_line=m_start,
                                    end_line=m_end
                                ),
                                metadata={"type": "method", "class_name": node.name, "method_name": m.name}
                            )
                        )
                        block_counter += 1
                        for l in range(m_start, m_end + 1):
                            covered_lines.add(l)
                else:
                    # Entire class as single block
                    cls_content = self._get_slice(lines, cls_start, cls_end)
                    blocks.append(
                        ExtractedBlock(
                            document_id=document_id,
                            source_filename=source_filename,
                            file_type=".py",
                            content=cls_content,
                            block_id=f"py_class_{node.name}_{block_counter}",
                            source_location=SourceLocation(
                                symbol_name=f"class {node.name}",
                                start_line=cls_start,
                                end_line=cls_end
                            ),
                            metadata={"type": "class", "name": node.name}
                        )
                    )
                    block_counter += 1
                    for l in range(cls_start, cls_end + 1):
                        covered_lines.add(l)

        # Collect uncovered module-level lines (imports, constants, global code)
        uncovered_start = None
        for line_num in range(1, len(lines) + 1):
            if line_num not in covered_lines and lines[line_num - 1].strip():
                if uncovered_start is None:
                    uncovered_start = line_num
            else:
                if uncovered_start is not None:
                    uncovered_end = line_num - 1
                    snippet = self._get_slice(lines, uncovered_start, uncovered_end)
                    if snippet.strip():
                        blocks.append(
                            ExtractedBlock(
                                document_id=document_id,
                                source_filename=source_filename,
                                file_type=".py",
                                content=snippet,
                                block_id=f"py_global_{block_counter}",
                                source_location=SourceLocation(
                                    symbol_name="module_level_code",
                                    start_line=uncovered_start,
                                    end_line=uncovered_end
                                ),
                                metadata={"type": "module_code"}
                            )
                        )
                        block_counter += 1
                    uncovered_start = None

        if uncovered_start is not None:
            snippet = self._get_slice(lines, uncovered_start, len(lines))
            if snippet.strip():
                blocks.append(
                    ExtractedBlock(
                        document_id=document_id,
                        source_filename=source_filename,
                        file_type=".py",
                        content=snippet,
                        block_id=f"py_global_{block_counter}",
                        source_location=SourceLocation(
                            symbol_name="module_level_code",
                            start_line=uncovered_start,
                            end_line=len(lines)
                        ),
                        metadata={"type": "module_code"}
                    )
                )

        # Sort blocks by start_line
        blocks.sort(key=lambda b: b.source_location.start_line or 0)
        return blocks

    def _parse_fallback_lines(
        self,
        lines: List[str],
        document_id: str,
        source_filename: str,
        syntax_error_msg: str
    ) -> List[ExtractedBlock]:
        """Fallback chunking when AST syntax parsing fails."""
        blocks: List[ExtractedBlock] = []
        batch_size = 40
        total_lines = len(lines)

        for s in range(0, total_lines, batch_size):
            e = min(s + batch_size, total_lines)
            content = "".join(lines[s:e]).strip()
            if content:
                blocks.append(
                    ExtractedBlock(
                        document_id=document_id,
                        source_filename=source_filename,
                        file_type=".py",
                        content=content,
                        block_id=f"py_fallback_{s + 1}_{e}",
                        source_location=SourceLocation(
                            start_line=s + 1,
                            end_line=e,
                            symbol_name="syntax_error_fallback"
                        ),
                        metadata={"syntax_error": syntax_error_msg}
                    )
                )

        return blocks
