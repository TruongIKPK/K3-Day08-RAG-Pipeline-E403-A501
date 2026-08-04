"""Convert documents in data/landing to standardized Markdown files.

Task 3 keeps the input directory layout intact. For example:

    data/landing/legal/tuition-fees.pdf
        -> data/standardized/legal/tuition-fees.md

PDF/DOC/DOCX/HTML files are converted with Microsoft's MarkItDown. News JSON
files produced by Task 2 already contain Markdown, so their article content
is extracted and written with a small metadata header instead of being passed
through MarkItDown.
"""

import json
from pathlib import Path
from typing import Any, Iterable

try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

PROJECT_DIR = Path(__file__).resolve().parent.parent
LANDING_DIR = PROJECT_DIR / "data" / "landing"
OUTPUT_DIR = PROJECT_DIR / "data" / "standardized"

MARKITDOWN_EXTENSIONS = {".pdf", ".doc", ".docx", ".html", ".htm"}
JSON_EXTENSIONS = {".json"}
PLAIN_TEXT_EXTENSIONS = {".md", ".markdown", ".txt"}
SUPPORTED_EXTENSIONS = (
    MARKITDOWN_EXTENSIONS | JSON_EXTENSIONS | PLAIN_TEXT_EXTENSIONS
)


def _iter_supported_files(input_dir: Path) -> Iterable[Path]:
    """Yield supported files below input_dir in deterministic order."""
    if not input_dir.exists():
        return ()

    return sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.relative_to(input_dir).as_posix().lower(),
    )


def _output_path(filepath: Path, input_dir: Path, output_dir: Path) -> Path:
    """Map an input file to the matching .md output path."""
    relative_path = filepath.relative_to(input_dir)
    return (output_dir / relative_path).with_suffix(".md")


def _write_markdown(output_path: Path, content: str) -> None:
    """Write UTF-8 Markdown, ensuring a final newline for clean diffs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = content if isinstance(content, str) else str(content)
    output_path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _get_markitdown() -> Any:
    """Create a converter or raise an actionable dependency error."""
    if MarkItDown is None:
        raise ImportError(
            "MarkItDown is required for PDF/DOC/DOCX/HTML conversion. "
            'Install it with: python -m pip install "markitdown[pdf]"'
        )
    return MarkItDown()


def _stringify(value: Any) -> str:
    """Convert JSON values to readable text without losing Unicode."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def _first_text(data: dict[str, Any], *keys: str) -> str:
    """Return the first non-empty value from data for the given keys."""
    for key in keys:
        value = _stringify(data.get(key))
        if value.strip():
            return value
    return ""


def _json_to_markdown(filepath: Path) -> str:
    """Extract a Task 2 article JSON file into Markdown with metadata."""
    data = json.loads(filepath.read_text(encoding="utf-8"))

    # A crawler normally writes an object. If a generic JSON list/value is
    # encountered, preserve it as formatted JSON rather than dropping data.
    if not isinstance(data, dict):
        return _stringify(data)

    title = _first_text(data, "title") or filepath.stem
    source = _first_text(data, "url", "source") or "N/A"
    crawled = _first_text(data, "date_crawled", "crawled_at", "date") or "N/A"
    content = _first_text(
        data,
        "content_markdown",
        "content",
        "text",
        "body",
        "description",
    )

    # Keep useful data even when a JSON file does not follow Task 2's schema.
    if not content:
        content = json.dumps(data, ensure_ascii=False, indent=2)

    return (
        f"# {title}\n\n"
        f"**Source:** {source}\n"
        f"**Crawled:** {crawled}\n\n"
        "---\n\n"
        f"{content}"
    )


def _convert_files(input_dir: Path, output_dir: Path) -> int:
    """Convert supported files under one landing subdirectory.

    Returns the number of files successfully written. Unsupported files and
    repository markers such as .gitkeep are intentionally ignored.
    """
    files = list(_iter_supported_files(input_dir))
    if not files:
        return 0

    converter = _get_markitdown() if any(
        path.suffix.lower() in MARKITDOWN_EXTENSIONS for path in files
    ) else None
    converted = 0

    for filepath in files:
        output_path = _output_path(filepath, input_dir, output_dir)
        suffix = filepath.suffix.lower()
        print(f"Converting: {filepath.relative_to(input_dir)}")

        if suffix in JSON_EXTENSIONS:
            content = _json_to_markdown(filepath)
        elif suffix in MARKITDOWN_EXTENSIONS:
            # MarkItDown returns a conversion result whose text is exposed as
            # text_content. Do not use str(result) because it can include
            # representation/debug data instead of document text.
            result = converter.convert(str(filepath))
            content = result.text_content
        else:  # .md/.markdown/.txt are already text-based formats.
            content = filepath.read_text(encoding="utf-8")

        _write_markdown(output_path, content)
        print(f"  Saved: {output_path}")
        converted += 1

    return converted


def convert_legal_docs() -> int:
    """Convert supported documents in data/landing/legal."""
    return _convert_files(LANDING_DIR / "legal", OUTPUT_DIR / "legal")


def convert_news_articles() -> int:
    """Convert supported crawled files in data/landing/news."""
    return _convert_files(LANDING_DIR / "news", OUTPUT_DIR / "news")


def convert_all() -> None:
    """Convert all legal and news landing files to standardized Markdown."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_count = convert_legal_docs()

    print("\n--- News Articles ---")
    news_count = convert_news_articles()

    # Create the expected output directories even when no input files exist.
    (OUTPUT_DIR / "legal").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "news").mkdir(parents=True, exist_ok=True)
    print(f"\nDone! Converted {legal_count + news_count} file(s) to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
