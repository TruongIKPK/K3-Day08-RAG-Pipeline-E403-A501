"""Task 6 - sparse retrieval using BM25 with a pure-Python implementation."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from ._rag_common import recursive_split_text, safe_top_k, tokenize

STANDARDIZED_DIR = Path(__file__).resolve().parent.parent / "data" / "standardized"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

CORPUS: list[dict] = []
_BM25_INDEX = None
_INDEXED_CONTENTS: tuple[str, ...] | None = None


def _tokenize(text: str) -> list[str]:
    return tokenize(text)


def _load_and_chunk_corpus() -> list[dict]:
    """Load the same standardized corpus and chunking config as Task 4."""

    if not STANDARDIZED_DIR.is_dir():
        return []
    documents: list[dict] = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md"), key=lambda item: item.as_posix().casefold()):
        if path.name.startswith("."):
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            continue
        if not content:
            continue
        relative = path.relative_to(STANDARDIZED_DIR).as_posix()
        parts = set(relative.casefold().split("/"))
        doc_type = "legal" if "legal" in parts else "news" if "news" in parts else "document"
        documents.append({"content": content, "metadata": {"source": relative, "type": doc_type}})

    chunks: list[dict] = []
    for document in documents:
        for chunk_index, content in enumerate(
            recursive_split_text(document["content"], CHUNK_SIZE, CHUNK_OVERLAP)
        ):
            chunks.append(
                {"content": content, "metadata": {**document["metadata"], "chunk_index": chunk_index}}
            )
    return chunks


class _BM25:
    """Small BM25Okapi-compatible object used when rank-bm25 is unavailable."""

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.document_count = len(tokenized_corpus)
        self.average_length = (
            sum(len(document) for document in tokenized_corpus) / self.document_count
            if self.document_count
            else 0.0
        )
        self.document_frequency: Counter[str] = Counter()
        for document in tokenized_corpus:
            self.document_frequency.update(set(document))

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for document in self.corpus:
            length = len(document)
            frequencies = Counter(document)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                document_frequency = self.document_frequency.get(token, 0)
                idf = math.log(
                    (self.document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                    + 1.0
                )
                denominator = frequency + self.k1 * (
                    1.0 - self.b + self.b * length / (self.average_length or 1.0)
                )
                score += idf * frequency * (self.k1 + 1.0) / denominator
            scores.append(score)
        return scores


def build_bm25_index(corpus: list[dict]):
    """Build a BM25 index from dictionaries containing ``content``."""

    if not isinstance(corpus, list):
        raise TypeError("corpus must be a list")
    tokenized = [_tokenize(item.get("content", "")) for item in corpus if isinstance(item, dict)]
    if not tokenized or not any(tokenized):
        raise ValueError("Corpus must contain at least one token")
    return _BM25(tokenized)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Return BM25-ranked chunks with ``content``, ``score`` and ``metadata``."""

    global _BM25_INDEX, _INDEXED_CONTENTS
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    limit = safe_top_k(top_k)
    if limit == 0 or not query.strip():
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []
    if not CORPUS:
        CORPUS.extend(_load_and_chunk_corpus())
    corpus = [
        item
        for item in CORPUS
        if isinstance(item, dict) and isinstance(item.get("content"), str) and item["content"].strip()
    ]
    if not corpus:
        return []

    indexed_contents = tuple(item["content"] for item in corpus)
    if _BM25_INDEX is None or _INDEXED_CONTENTS != indexed_contents:
        _BM25_INDEX = build_bm25_index(corpus)
        _INDEXED_CONTENTS = indexed_contents

    scores = _BM25_INDEX.get_scores(query_tokens)
    ranked = sorted(range(len(corpus)), key=lambda index: (-float(scores[index]), index))
    results: list[dict] = []
    has_match = any(float(scores[index]) > 0 for index in ranked)
    for index in ranked[:limit]:
        raw_score = float(scores[index])
        metadata = dict(corpus[index].get("metadata", {}))
        if not has_match:
            # Keep the public search API useful for an unfamiliar vocabulary,
            # while marking these as non-evidence so Task 9 can fallback.
            raw_score = 1e-6 / (index + 1)
            metadata["lexical_match"] = False
        else:
            metadata["lexical_match"] = raw_score > 0
        results.append(
            {
                "content": corpus[index]["content"],
                "score": round(raw_score, 6),
                "metadata": metadata,
            }
        )
    return results


if __name__ == "__main__":
    for result in lexical_search("shipping policy", top_k=5):
        print(f"[{result['score']:.3f}] {result['metadata'].get('source')}")

