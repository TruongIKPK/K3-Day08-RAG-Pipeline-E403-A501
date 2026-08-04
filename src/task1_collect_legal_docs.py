"""Task 1 - collect and validate public legal/policy documents."""

from __future__ import annotations

import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "landing" / "legal"
SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".md", ".txt"}


def setup_directory() -> Path:
    """Create and return the legal landing directory."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def download_file(url: str, filename: str, timeout: int = 30) -> Path:
    """Download one public document with a path-traversal-safe filename."""

    if not isinstance(url, str) or not url.strip():
        raise ValueError("url must be a non-empty string")
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string")
    safe_name = Path(filename).name
    if safe_name != filename or Path(safe_name).suffix.casefold() not in SUPPORTED_EXTENSIONS:
        raise ValueError("filename must be a simple supported document name")
    destination = setup_directory() / safe_name
    try:
        from requests import get

        response = get(url, timeout=timeout, headers={"User-Agent": "RAG-lab/1.0"})
        response.raise_for_status()
        payload = response.content
    except ImportError:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read()
    destination.write_bytes(payload)
    return destination


def collect_legal_docs(sources: list[dict] | None = None) -> list[Path]:
    """Download a supplied list of ``{"url", "filename"}`` sources.

    Existing files are retained and returned, making the function safe to run
    repeatedly after a network interruption.
    """

    setup_directory()
    if not sources:
        return sorted(
            path for path in DATA_DIR.iterdir() if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS
        )
    output: list[Path] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        filename = source.get("filename")
        url = source.get("url")
        if not isinstance(filename, str) or not isinstance(url, str):
            continue
        destination = DATA_DIR / Path(filename).name
        if destination.exists() and destination.stat().st_size > 0:
            output.append(destination)
            continue
        try:
            output.append(download_file(url, filename))
        except Exception:
            # A single blocked public source should not discard collected data.
            continue
    return output


if __name__ == "__main__":
    print(f"Legal landing directory ready: {setup_directory()}")

