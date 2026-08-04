"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25 (BM25Plus để tránh lỗi IDF âm cho các từ xuất hiện phổ biến).

Cài đặt:
    pip install rank-bm25
"""

import math
import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}

_BM25_INDEX = None
_INDEXED_CONTENTS: tuple[str, ...] | None = None


def _tokenize(text: str) -> list[str]:
    """Tokenize nhất quán cho cả corpus và query, hỗ trợ Unicode tiếng Việt."""
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def _load_and_chunk_corpus() -> list[dict]:
    """Nạp các tài liệu Markdown và cắt nhỏ thành chunks (CHUNK_SIZE=800, OVERLAP=100)."""
    if not STANDARDIZED_DIR.is_dir():
        return []

    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        path_parts = {part.casefold() for part in relative_path.parts}
        doc_type = "legal" if "legal" in path_parts else ("news" if "news" in path_parts else "document")

        documents.append({
            "content": content,
            "metadata": {"source": relative_path.as_posix(), "type": doc_type}
        })

    # Chunking corpus thành các phần nhỏ 800 chars
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, split in enumerate(splits):
            if split.strip():
                chunks.append({
                    "content": split,
                    "metadata": {**doc["metadata"], "chunk_index": i}
                })

    return chunks


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index (BM25Plus) từ corpus.
    """
    if not corpus:
        raise ValueError("Corpus must contain at least one non-empty document")

    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus if isinstance(doc.get("content"), str)]
    if not any(tokenized_corpus):
        raise ValueError("Corpus must contain at least one token")

    from rank_bm25 import BM25Plus
    return BM25Plus(tokenized_corpus, k1=1.5, b=0.75)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25Plus.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    global _BM25_INDEX, _INDEXED_CONTENTS

    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k <= 0:
        return []

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    if not CORPUS:
        CORPUS.extend(_load_and_chunk_corpus())
    if not CORPUS:
        return []

    searchable_corpus = [
        doc for doc in CORPUS
        if isinstance(doc, dict) and isinstance(doc.get("content"), str) and _tokenize(doc["content"])
    ]
    if not searchable_corpus:
        return []

    indexed_contents = tuple(doc["content"] for doc in searchable_corpus)
    if _BM25_INDEX is None or _INDEXED_CONTENTS != indexed_contents:
        _BM25_INDEX = build_bm25_index(searchable_corpus)
        _INDEXED_CONTENTS = indexed_contents

    scores = _BM25_INDEX.get_scores(tokenized_query)
    ranked_indices = sorted(
        range(len(searchable_corpus)),
        key=lambda index: (-float(scores[index]), index),
    )

    results = []
    for index in ranked_indices:
        score = float(scores[index])
        if not math.isfinite(score) or score <= 0:
            continue

        document = searchable_corpus[index]
        metadata = document.get("metadata", {})
        results.append({
            "content": document["content"],
            "score": round(score, 4),
            "metadata": metadata.copy() if isinstance(metadata, dict) else {},
        })
        if len(results) == top_k:
            break

    return results


if __name__ == "__main__":
    results = lexical_search("Chính sách vận chuyển Shopee Mall", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
