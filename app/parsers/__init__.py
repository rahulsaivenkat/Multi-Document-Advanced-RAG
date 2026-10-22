"""Document parsers package for PaperMind AI."""
from app.parsers.base import BaseParser, ParserError
from app.parsers.pdf_parser import PDFParser
from app.parsers.text_parser import TextParser
from app.parsers.markdown_parser import MarkdownParser
from app.parsers.docx_parser import DocxParser
from app.parsers.csv_parser import CSVParser
from app.parsers.html_parser import HTMLParser
from app.parsers.json_parser import JSONParser
from app.parsers.python_parser import PythonParser

__all__ = [
    "BaseParser",
    "ParserError",
    "PDFParser",
    "TextParser",
    "MarkdownParser",
    "DocxParser",
    "CSVParser",
    "HTMLParser",
    "JSONParser",
    "PythonParser",
]
