"""HTML parser with DOM sanitization, heading path tracking, and table structure preservation."""
from pathlib import Path
from typing import List, Optional
from bs4 import BeautifulSoup, Tag
from app.parsers.base import BaseParser, ParserError
from app.schemas import ExtractedBlock, SourceLocation


class HTMLParser(BaseParser):
    """Parses HTML documents, strips non-content tags, and extracts structured blocks with heading paths."""

    DISCARD_TAGS = {"script", "style", "nav", "header", "footer", "noscript", "aside", "svg"}

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
            raise ParserError(f"Unable to decode HTML file '{source_filename}'.")

        try:
            soup = BeautifulSoup(text, "html.parser")
        except Exception as e:
            raise ParserError(f"Malformed HTML in '{source_filename}': {str(e)}")

        # Remove irrelevant non-content elements
        for tag in soup.find_all(self.DISCARD_TAGS):
            tag.decompose()

        # Target content tags in document order
        content_tags = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "table", "pre", "blockquote"]
        elements = soup.find_all(content_tags)

        if not elements:
            # Fallback to body text if no semantic tags found
            body = soup.find("body") or soup
            body_text = body.get_text(separator="\n", strip=True)
            if not body_text:
                raise ParserError(f"HTML file '{source_filename}' contains no readable content.")
            return [
                ExtractedBlock(
                    document_id=document_id,
                    source_filename=source_filename,
                    file_type=".html",
                    content=body_text,
                    block_id="html_block_1",
                    source_location=SourceLocation(block_id="body_1", tag_name="body"),
                    metadata={"fallback": True}
                )
            ]

        blocks: List[ExtractedBlock] = []
        heading_stack: List[str] = []
        block_counter = 1

        def current_heading_path() -> Optional[str]:
            return " > ".join(heading_stack) if heading_stack else None

        for el in elements:
            tag_name = el.name.lower()

            # Handle Headings
            if tag_name.startswith("h") and len(tag_name) == 2 and tag_name[1].isdigit():
                level = int(tag_name[1])
                heading_text = el.get_text(strip=True)
                if heading_text:
                    while len(heading_stack) >= level:
                        heading_stack.pop()
                    heading_stack.append(heading_text)
                    blocks.append(
                        ExtractedBlock(
                            document_id=document_id,
                            source_filename=source_filename,
                            file_type=".html",
                            content=heading_text,
                            block_id=f"html_block_{block_counter}",
                            source_location=SourceLocation(
                                heading_path=current_heading_path(),
                                block_id=f"el_{block_counter}",
                                tag_name=tag_name
                            ),
                            metadata={"is_heading": True, "level": level}
                        )
                    )
                    block_counter += 1
                continue

            # Handle Tables
            if tag_name == "table":
                rows = el.find_all("tr")
                table_lines = []
                for r in rows:
                    cells = [c.get_text(strip=True) for c in r.find_all(["th", "td"])]
                    if any(cells):
                        table_lines.append(" | ".join(cells))
                if table_lines:
                    content = "[Table]\n" + "\n".join(table_lines)
                    blocks.append(
                        ExtractedBlock(
                            document_id=document_id,
                            source_filename=source_filename,
                            file_type=".html",
                            content=content,
                            block_id=f"html_block_{block_counter}",
                            source_location=SourceLocation(
                                heading_path=current_heading_path(),
                                block_id=f"el_{block_counter}",
                                tag_name="table"
                            ),
                            metadata={"is_table": True}
                        )
                    )
                    block_counter += 1
                continue

            # Handle Lists
            if tag_name in ("ul", "ol"):
                items = [li.get_text(strip=True) for li in el.find_all("li", recursive=False)]
                if items:
                    content = "\n".join(f"- {item}" for item in items if item)
                    if content:
                        blocks.append(
                            ExtractedBlock(
                                document_id=document_id,
                                source_filename=source_filename,
                                file_type=".html",
                                content=content,
                                block_id=f"html_block_{block_counter}",
                                source_location=SourceLocation(
                                    heading_path=current_heading_path(),
                                    block_id=f"el_{block_counter}",
                                    tag_name=tag_name
                                ),
                                metadata={"is_list": True}
                            )
                        )
                        block_counter += 1
                continue

            # Paragraphs, preformatted code, blockquotes
            text_content = el.get_text(strip=True)
            if text_content:
                blocks.append(
                    ExtractedBlock(
                        document_id=document_id,
                        source_filename=source_filename,
                        file_type=".html",
                        content=text_content,
                        block_id=f"html_block_{block_counter}",
                        source_location=SourceLocation(
                            heading_path=current_heading_path(),
                            block_id=f"el_{block_counter}",
                            tag_name=tag_name
                        ),
                        metadata={"tag": tag_name}
                    )
                )
                block_counter += 1

        if not blocks:
            raise ParserError(f"HTML file '{source_filename}' contains no extractable text.")

        return blocks
