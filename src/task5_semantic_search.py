"""Task 5 - dense/semantic retrieval with an offline-safe fallback."""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv

from ._rag_common import cosine_similarity, hashed_embedding, safe_top_k
from .task4_chunking_indexing import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    LOCAL_INDEX_FILE,
    chunk_documents,
    load_documents,
)

load_dotenv()

_model = None
_collection = None
_local_chunks: list[dict] | None = None


def _get_embedding_model():
    """Load a local model only when explicitly enabled by the user."""

    global _model
    if _model is not None:
        return _model
    if os.getenv("RAG_USE_LOCAL_MODEL", "0").casefold() not in {"1", "true", "yes"}:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception:
        _model = None
    return _model


def _get_collection():
    """Return the Task 4 Chroma collection when it is available."""

    global _collection
    if _collection is not None:
        return _collection
    try:
        import chromadb

        if not CHROMA_DIR.exists():
            return None
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        _collection = None
    return _collection


def _get_query_embedding(query: str) -> list[float]:
    """Use the same configured model as indexing, else deterministic hashing."""

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and "text-embedding" in EMBEDDING_MODEL.lower():
        try:
            from openai import OpenAI

            response = OpenAI(api_key=api_key).embeddings.create(
                input=[query], model=EMBEDDING_MODEL
            )
            return list(response.data[0].embedding)
        except Exception:
            pass

    model = _get_embedding_model()
    if model is not None:
        try:
            value = model.encode(query)
            return value.tolist() if hasattr(value, "tolist") else list(value)
        except Exception:
            pass
    return hashed_embedding(query, EMBEDDING_DIM)


def _load_local_chunks() -> list[dict]:
    global _local_chunks
    if _local_chunks is not None:
        return _local_chunks

    if LOCAL_INDEX_FILE.exists():
        try:
            payload = json.loads(LOCAL_INDEX_FILE.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                _local_chunks = [item for item in payload if isinstance(item, dict) and item.get("content")]
        except (OSError, ValueError, TypeError):
            _local_chunks = None

    if _local_chunks is None:
        _local_chunks = chunk_documents(load_documents())
    return _local_chunks


def _semantic_search_local(query: str, top_k: int) -> list[dict]:
    query_vector = _get_query_embedding(query)
    results: list[dict] = []
    for index, chunk in enumerate(_load_local_chunks()):
        embedding = chunk.get("embedding")
        if not isinstance(embedding, list):
            embedding = hashed_embedding(chunk.get("content", ""), len(query_vector) or EMBEDDING_DIM)
        score = max(0.0, cosine_similarity(query_vector, embedding))
        item = {
            "content": str(chunk.get("content", "")),
            "score": round(score, 6),
            "metadata": dict(chunk.get("metadata", {})),
        }
        item["metadata"].setdefault("chunk_index", index)
        results.append(item)
    results.sort(key=lambda item: (-item["score"], item["metadata"].get("source", ""), item["metadata"].get("chunk_index", 0)))
    return results[:top_k]


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Return chunks ranked by cosine similarity, highest score first."""

    if not isinstance(query, str):
        raise TypeError("query must be a string")
    limit = safe_top_k(top_k)
    if not query.strip() or limit == 0:
        return []

    collection = _get_collection()
    if collection is not None:
        try:
            query_vector = _get_query_embedding(query)
            response = collection.query(
                query_embeddings=[query_vector],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )
            documents = (response.get("documents") or [[]])[0]
            metadatas = (response.get("metadatas") or [[]])[0]
            distances = (response.get("distances") or [[]])[0]
            results = []
            for document, metadata, distance in zip(documents, metadatas, distances):
                score = max(0.0, 1.0 - float(distance))
                results.append(
                    {"content": document, "score": round(score, 6), "metadata": metadata or {}}
                )
            results.sort(key=lambda item: item["score"], reverse=True)
            if results:
                return results[:limit]
        except Exception:
            # A stale/incompatible Chroma index should not take down the assistant.
            pass

    return _semantic_search_local(query, limit)


if __name__ == "__main__":
    for result in semantic_search("shipping policy", top_k=5):
        print(f"[{result['score']:.3f}] {result['metadata'].get('source')}")

