"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import math
import re
from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}

_BM25_INDEX = None
_INDEXED_CONTENTS: tuple[str, ...] | None = None


def _tokenize(text: str) -> list[str]:
    """Tokenize nhất quán cho cả corpus và query, có hỗ trợ Unicode."""
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def _load_standardized_corpus() -> list[dict]:
    """Nạp các tài liệu Markdown đã chuẩn hóa theo thứ tự ổn định."""
    if not STANDARDIZED_DIR.is_dir():
        return []

    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        path_parts = {part.casefold() for part in relative_path.parts}
        if "legal" in path_parts:
            doc_type = "legal"
        elif "news" in path_parts:
            doc_type = "news"
        else:
            doc_type = "document"

        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": relative_path.as_posix(),
                    "type": doc_type,
                },
            }
        )

    return documents


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    if not corpus:
        raise ValueError("Corpus must contain at least one non-empty document")

    tokenized_corpus = []
    for index, document in enumerate(corpus):
        if not isinstance(document, dict):
            raise TypeError(f"Corpus item at index {index} must be a dict")

        content = document.get("content")
        if not isinstance(content, str):
            raise TypeError(
                f"Corpus item at index {index} must contain a string 'content'"
            )
        tokenized_corpus.append(_tokenize(content))

    if not any(tokenized_corpus):
        raise ValueError("Corpus must contain at least one token")

    from rank_bm25 import BM25Okapi

    return BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

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
        CORPUS.extend(_load_standardized_corpus())
    if not CORPUS:
        return []

    # Chỉ index các document hợp lệ; giữ corpus và index cùng thứ tự.
    searchable_corpus = [
        document
        for document in CORPUS
        if isinstance(document, dict)
        and isinstance(document.get("content"), str)
        and _tokenize(document["content"])
    ]
    if not searchable_corpus:
        return []

    indexed_contents = tuple(document["content"] for document in searchable_corpus)
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
        results.append(
            {
                "content": document["content"],
                "score": score,
                "metadata": metadata.copy() if isinstance(metadata, dict) else {},
            }
        )
        if len(results) == top_k:
            break

    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("tuition fee payment methods", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
