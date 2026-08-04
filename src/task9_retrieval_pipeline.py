"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.48   # Ngưỡng điểm cosine gốc tối thiểu (Semantic Search) khi < 0.48 -> PageIndex Fallback
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (Semantic Search)
        use_reranking: Có áp dụng RRF reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    dense_results = []
    try:
        dense_results = semantic_search(query, top_k=top_k * 2)
    except Exception as e:
        print(f"⚠ Semantic search warning: {e}")

    sparse_results = []
    try:
        sparse_results = lexical_search(query, top_k=top_k * 2)
    except Exception as e:
        print(f"⚠ Lexical search warning: {e}")

    # Kiểm tra Fallback dùng Cosine Score GỐC của Semantic Search
    best_dense_score = dense_results[0]["score"] if dense_results else 0.0

    if best_dense_score < score_threshold and not sparse_results:
        print(f"  ⚠ Best dense score ({best_dense_score:.3f}) < threshold ({score_threshold}) → Kích hoạt PageIndex Fallback")
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback

    # Merge kết quả bằng RRF (Reciprocal Rank Fusion k=60)
    merged_results = rerank_rrf([dense_results, sparse_results], top_k=top_k)
    for item in merged_results:
        item["source"] = "hybrid"

    if not merged_results:
        return pageindex_search(query, top_k=top_k)

    return merged_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Chính sách vận chuyển Shopee",
        "Quy chế hoạt động sàn Shopee.vn",
        "xyzabc123nonsense_out_of_domain",  # Query ngoài domain → kích hoạt Fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.4f}] [{r['source']}] {r['content'][:80]}...")
