"""Task 9 - hybrid retrieval, reranking and PageIndex fallback."""

from __future__ import annotations

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search
from ._rag_common import safe_top_k

SCORE_THRESHOLD = 0.48
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"
RRF_CANDIDATE_MULTIPLIER = 2


def _with_source(items: list[dict], source: str) -> list[dict]:
    output = []
    for item in items:
        if not isinstance(item, dict) or not item.get("content"):
            continue
        result = item.copy()
        result["metadata"] = dict(item.get("metadata", {}))
        result["source"] = source
        try:
            result["score"] = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            result["score"] = 0.0
        output.append(result)
    return output


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Return grounded chunks from dense+sparse retrieval or PageIndex fallback."""

    if not isinstance(query, str):
        raise TypeError("query must be a string")
    limit = safe_top_k(top_k)
    if limit == 0 or not query.strip():
        return []
    try:
        threshold = float(score_threshold)
    except (TypeError, ValueError):
        raise TypeError("score_threshold must be numeric") from None

    candidate_k = max(limit, limit * RRF_CANDIDATE_MULTIPLIER)
    try:
        dense = semantic_search(query, top_k=candidate_k)
    except Exception:
        dense = []
    try:
        sparse = lexical_search(query, top_k=candidate_k)
    except Exception:
        sparse = []

    dense = _with_source(dense, "dense")
    sparse = _with_source(sparse, "sparse")
    best_dense_score = dense[0].get("score", 0.0) if dense else 0.0

    # The threshold is intentionally evaluated on the original dense score,
    # never on an RRF score. This is the key safety boundary in the lab.
    if not dense or best_dense_score < threshold:
        try:
            fallback = _with_source(pageindex_search(query, top_k=limit), "pageindex")
        except Exception:
            fallback = []
        if fallback:
            return fallback[:limit]

    # Zero-score lexical placeholders are useful to direct Task 6 callers but
    # must not become evidence in the assistant.
    sparse_evidence = [
        item for item in sparse if item.get("metadata", {}).get("lexical_match", True)
    ]
    if not dense and not sparse_evidence:
        return []
    merged = rerank_rrf([dense, sparse_evidence], top_k=candidate_k)
    merged = _with_source(merged, "hybrid")
    if not merged:
        return []

    if use_reranking:
        try:
            merged = rerank(query, merged, top_k=limit, method=RERANK_METHOD)
            merged = _with_source(merged, "hybrid")
        except Exception:
            merged = merged[:limit]
    return merged[:limit]


if __name__ == "__main__":
    for result in retrieve("shipping policy", top_k=3):
        print(f"[{result['source']}] {result['score']:.4f} {result['metadata'].get('source')}")

