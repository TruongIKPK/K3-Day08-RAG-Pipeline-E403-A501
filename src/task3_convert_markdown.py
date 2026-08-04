"""Task 3 - normalize every landing document into UTF-8 Markdown."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from ._rag_common import html_to_text

PROJECT_DIR = Path(__file__).resolve().parent.parent
LANDING_DIR = PROJECT_DIR / "data" / "landing"
OUTPUT_DIR = PROJECT_DIR / "data" / "standardized"

MARKITDOWN_EXTENSIONS = {".pdf", ".doc", ".docx", ".html", ".htm"}
JSON_EXTENSIONS = {".json"}
PLAIN_TEXT_EXTENSIONS = {".md", ".markdown", ".txt"}
SUPPORTED_EXTENSIONS = MARKITDOWN_EXTENSIONS | JSON_EXTENSIONS | PLAIN_TEXT_EXTENSIONS

try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None


def _iter_supported_files(input_dir: Path) -> Iterable[Path]:
    if not input_dir.exists():
        return ()
    return sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS and not path.name.startswith(".")
        ),
        key=lambda path: path.relative_to(input_dir).as_posix().casefold(),
    )


def _output_path(filepath: Path, input_dir: Path, output_dir: Path) -> Path:
    return (output_dir / filepath.relative_to(input_dir)).with_suffix(".md")


def _write_markdown(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    value = content if isinstance(content, str) else str(content)
    output_path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _get_markitdown() -> Any:
    return MarkItDown() if MarkItDown is not None else None


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def _first_text(data: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = _stringify(data.get(key))
        if value.strip():
            return value.strip()
    return ""


def _json_to_markdown(filepath: Path) -> str:
    data = json.loads(filepath.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return _stringify(data)
    title = _first_text(data, "title") or filepath.stem
    source = _first_text(data, "url", "source") or "N/A"
    crawled = _first_text(data, "date_crawled", "crawled_at", "date") or "N/A"
    content = _first_text(data, "content_markdown", "content", "text", "body", "description")
    if not content:
        content = json.dumps(data, ensure_ascii=False, indent=2)
    return f"# {title}\n\n**Source:** {source}\n**Crawled:** {crawled}\n\n---\n\n{content}"


def _fallback_binary_to_markdown(filepath: Path) -> str:
    """Best-effort conversion for optional PDF/DOC dependencies."""

    if filepath.suffix.casefold() in {".html", ".htm"}:
        return f"# {filepath.stem}\n\n{html_to_text(filepath.read_text(encoding='utf-8', errors='replace'))}"
    try:
        raw = filepath.read_bytes()
    except OSError:
        raw = b""
    text = raw.decode("utf-8", errors="ignore")
    text = re.sub(r"\s+", " ", text).strip()
    return f"# {filepath.stem}\n\n{text or 'Document content requires an optional converter.'}"


def _convert_files(input_dir: Path, output_dir: Path) -> int:
    files = list(_iter_supported_files(input_dir))
    converter = _get_markitdown() if any(path.suffix.casefold() in MARKITDOWN_EXTENSIONS for path in files) else None
    converted = 0
    for filepath in files:
        suffix = filepath.suffix.casefold()
        try:
            if suffix in JSON_EXTENSIONS:
                content = _json_to_markdown(filepath)
            elif suffix in PLAIN_TEXT_EXTENSIONS:
                content = filepath.read_text(encoding="utf-8", errors="replace")
            elif converter is not None:
                result = converter.convert(str(filepath))
                content = getattr(result, "text_content", str(result))
            else:
                content = _fallback_binary_to_markdown(filepath)
            _write_markdown(_output_path(filepath, input_dir, output_dir), content)
            converted += 1
        except (OSError, UnicodeError, ValueError, TypeError):
            continue
    return converted


def convert_legal_docs() -> int:
    return _convert_files(LANDING_DIR / "legal", OUTPUT_DIR / "legal")


def convert_news_articles() -> int:
    return _convert_files(LANDING_DIR / "news", OUTPUT_DIR / "news")


def convert_all() -> int:
    """Convert legal and news inputs and return the number of outputs written."""

    (OUTPUT_DIR / "legal").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "news").mkdir(parents=True, exist_ok=True)
    return convert_legal_docs() + convert_news_articles()


if __name__ == "__main__":
    print(f"Converted {convert_all()} document(s) into {OUTPUT_DIR}")

