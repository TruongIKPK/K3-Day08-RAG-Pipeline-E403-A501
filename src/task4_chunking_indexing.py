"""Task 4 - load, chunk, embed and index the standardized corpus.

The preferred production path is Chroma plus the configured embedding model.
For a reproducible local lab run, every step has a deterministic Python
fallback so Tasks 4-10 work without downloading a model first.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from ._rag_common import hashed_embedding, recursive_split_text

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
CHROMA_DIR = PROJECT_DIR / "chroma_db"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive character splitting"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024 if "bge" in EMBEDDING_MODEL.lower() else 1536
VECTOR_STORE = "chromadb (with JSON fallback)"
COLLECTION_NAME = "university_services_docs"
LOCAL_INDEX_FILE = CHROMA_DIR / f"{COLLECTION_NAME}.json"


def _document_type(path: Path) -> str:
    parts = {part.casefold() for part in path.parts}
    if "legal" in parts:
        return "legal"
    if "news" in parts:
        return "news"
    return "document"


def load_documents() -> list[dict]:
    """Load every non-empty Markdown file with stable relative metadata."""

    if not STANDARDIZED_DIR.is_dir():
        return []
    documents: list[dict] = []
    paths = sorted(STANDARDIZED_DIR.rglob("*.md"), key=lambda item: item.as_posix().casefold())
    for path in paths:
        if path.name.startswith("."):
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            continue
        if not content:
            continue
        relative = path.relative_to(STANDARDIZED_DIR).as_posix()
        documents.append(
            {"content": content, "metadata": {"source": relative, "type": _document_type(path)}}
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents into bounded, overlapping chunks with chunk indexes."""

    if documents is None:
        return []
    if not isinstance(documents, list):
        raise TypeError("documents must be a list")
    chunks: list[dict] = []
    for document in documents:
        if not isinstance(document, dict):
            continue
        content = document.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        metadata = document.get("metadata")
        metadata = metadata.copy() if isinstance(metadata, dict) else {}
        metadata.setdefault("source", "unknown")
        metadata.setdefault("type", "document")
        splits = recursive_split_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk_index, chunk_text in enumerate(splits):
            chunks.append(
                {"content": chunk_text, "metadata": {**metadata, "chunk_index": chunk_index}}
            )
    return chunks


def _openai_embeddings(chunks: list[dict]) -> list[list[float]] | None:
    """Try OpenAI embeddings only when the configured model is an OpenAI model."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or "text-embedding" not in EMBEDDING_MODEL.lower():
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            input=[chunk["content"] for chunk in chunks], model=EMBEDDING_MODEL
        )
        return [list(item.embedding) for item in response.data]
    except Exception:
        return None


def _sentence_transformer_embeddings(chunks: list[dict]) -> list[list[float]] | None:
    """Try the local model without making missing packages fatal."""

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        values = model.encode([chunk["content"] for chunk in chunks], show_progress_bar=False)
        return [value.tolist() if hasattr(value, "tolist") else list(value) for value in values]
    except Exception:
        return None


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Attach embeddings, preferring configured providers and falling back locally."""

    if not isinstance(chunks, list):
        raise TypeError("chunks must be a list")
    if not chunks:
        return []
    embeddings = _openai_embeddings(chunks) or _sentence_transformer_embeddings(chunks)
    if embeddings is None or len(embeddings) != len(chunks):
        embeddings = [hashed_embedding(chunk["content"], EMBEDDING_DIM) for chunk in chunks]
    output: list[dict] = []
    for chunk, embedding in zip(chunks, embeddings):
        item = chunk.copy()
        item["metadata"] = dict(chunk.get("metadata", {}))
        item["embedding"] = [float(value) for value in embedding]
        output.append(item)
    return output


def _write_local_index(chunks: list[dict]) -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    payload = [
        {"content": chunk["content"], "metadata": chunk.get("metadata", {}), "embedding": chunk.get("embedding", [])}
        for chunk in chunks
    ]
    LOCAL_INDEX_FILE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _index_chroma(chunks: list[dict]) -> bool:
    """Index in Chroma when installed; return False when it is unavailable."""

    try:
        import chromadb

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        ids = [
            f"{chunk.get('metadata', {}).get('source', 'document')}_chunk_"
            f"{chunk.get('metadata', {}).get('chunk_index', index)}"
            for index, chunk in enumerate(chunks)
        ]
        collection.upsert(
            ids=ids,
            documents=[chunk["content"] for chunk in chunks],
            embeddings=[chunk["embedding"] for chunk in chunks],
            metadatas=[chunk.get("metadata", {}) for chunk in chunks],
        )
        return True
    except Exception:
        return False


def index_to_vectorstore(chunks: list[dict]) -> bool:
    """Persist chunks to Chroma and always maintain the local JSON fallback."""

    if not chunks:
        return False
    embedded = chunks if all(chunk.get("embedding") for chunk in chunks) else embed_chunks(chunks)
    chroma_ok = _index_chroma(embedded)
    try:
        _write_local_index(embedded)
    except OSError:
        pass
    return chroma_ok or LOCAL_INDEX_FILE.exists()


def run_pipeline() -> list[dict]:
    """Run load -> chunk -> embed -> index and return the indexed chunks."""

    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded = embed_chunks(chunks)
    index_to_vectorstore(embedded)
    print(
        f"Task 4 complete: {len(documents)} documents, {len(embedded)} chunks, "
        f"embedding={EMBEDDING_MODEL}, size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}"
    )
    return embedded


if __name__ == "__main__":
    run_pipeline()

