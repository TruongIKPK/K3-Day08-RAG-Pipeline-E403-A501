"""Small, dependency-free building blocks shared by the RAG tasks.

The lab can use Chroma, sentence-transformers and rank-bm25 when those
packages are installed.  These helpers deliberately keep the application
usable without them as well: the test runner and a local demo should not
need to download a model before they can answer a question.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from pathlib import Path
from typing import Iterable


TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)
MOJIBAKE_MARKERS = ("Ã", "Â", "Ä", "Å", "Æ", "áº", "â€", "ï¿")


def repair_mojibake(value: str) -> str:
    """Repair the common UTF-8-as-Latin-1 artefact found in sample data.

    The original text is kept in returned documents.  This function is only
    used for matching, so repairing a damaged sample cannot alter citations.
    Correct Unicode text is returned unchanged.
    """

    if not isinstance(value, str) or not any(marker in value for marker in MOJIBAKE_MARKERS):
        return value
    try:
        candidate = value.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value

    old_markers = sum(value.count(marker) for marker in MOJIBAKE_MARKERS)
    new_markers = sum(candidate.count(marker) for marker in MOJIBAKE_MARKERS)
    return candidate if new_markers < old_markers else value


def searchable_text(value: str) -> str:
    """Return normalized text for matching while retaining Unicode words."""

    repaired = repair_mojibake(value or "")
    return unicodedata.normalize("NFKC", repaired).casefold()


def tokenize(value: str) -> list[str]:
    """Tokenize Vietnamese/English text consistently for every retriever."""

    return TOKEN_RE.findall(searchable_text(value))


def safe_top_k(top_k: int, default: int = 10) -> int:
    """Normalize a public top-k argument and reject ambiguous values."""

    if isinstance(top_k, bool):
        raise TypeError("top_k must be an integer")
    if not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    return max(0, top_k if top_k is not None else default)


def recursive_split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: Iterable[str] = ("\n\n", "\n", ". ", " ", ""),
) -> list[str]:
    """Split text into bounded overlapping chunks without LangChain.

    The separator order mirrors ``RecursiveCharacterTextSplitter``.  The
    implementation is intentionally conservative: every returned chunk is at
    most ``chunk_size`` characters, including for very long unbroken strings.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    clean = text.strip()
    if not clean:
        return []
    if len(clean) <= chunk_size:
        return [clean]

    separator_list = tuple(separators)
    pieces: list[str] = []
    start = 0
    text_length = len(clean)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            cut = end
            for separator in separator_list:
                if not separator:
                    continue
                position = clean.rfind(separator, start, end)
                if position > start:
                    candidate = position + len(separator)
                    # Do not create tiny chunks merely because a separator is
                    # close to the beginning of the current window.
                    if candidate - start >= max(1, chunk_size // 4):
                        cut = candidate
                        break
            end = cut

        piece = clean[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= text_length:
            break

        next_start = max(0, end - chunk_overlap)
        if next_start <= start:
            next_start = end
        start = next_start

    return pieces


def hashed_embedding(text: str, dimension: int = 1024) -> list[float]:
    """Create a deterministic lightweight dense vector for offline fallback."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    tokens = tokenize(text)
    vector = [0.0] * dimension
    features = list(tokens)
    normalized = searchable_text(text)
    features.extend(normalized[index : index + 3] for index in range(max(0, len(normalized) - 2)))
    if not features:
        return vector

    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % dimension
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign

    norm = sum(value * value for value in vector) ** 0.5
    if norm:
        vector = [value / norm for value in vector]
    return vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Compute cosine similarity safely for vectors of different lengths."""

    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(float(left[index]) * float(right[index]) for index in range(size))
    left_norm = sum(float(value) * float(value) for value in left[:size]) ** 0.5
    right_norm = sum(float(value) * float(value) for value in right[:size]) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def html_to_text(raw_html: str) -> str:
    """Convert basic HTML to readable text without requiring BeautifulSoup."""

    without_scripts = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", raw_html, flags=re.I | re.S)
    with_breaks = re.sub(r"</?(?:p|div|br|h[1-6]|li|tr|section|article)[^>]*>", "\n", without_scripts, flags=re.I)
    plain = re.sub(r"<[^>]+>", " ", with_breaks)
    return html.unescape(re.sub(r"[ \t]+", " ", plain)).strip()


def structural_sections(content: str, source: str, doc_type: str) -> list[dict]:
    """Extract heading/paragraph sections for the vectorless fallback."""

    lines = content.splitlines()
    sections: list[dict] = []
    current_title = "Document"
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if not body:
            return
        sections.append(
            {
                "content": f"## {current_title}\n\n{body}",
                "metadata": {"source": source, "type": doc_type, "section": current_title},
            }
        )

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            flush()
            current_title = stripped.lstrip("# ").strip() or "Document"
            current_lines = []
        elif stripped:
            current_lines.append(stripped)
            if len("\n".join(current_lines)) >= 1200:
                flush()
                current_lines = []
    flush()
    return sections


def ensure_text(value: object, default: str = "") -> str:
    """Convert optional metadata values to a stable string."""

    return value if isinstance(value, str) else default if value is None else str(value)

