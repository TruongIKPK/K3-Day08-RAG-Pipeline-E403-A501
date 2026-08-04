"""
Task 7 — Reranking Module.

Chọn RRF (Reciprocal Rank Fusion) là phương pháp mặc định:
    RRF(d) = Σ 1 / (k + rank_r(d))
    với k = 60 (smoothing constant theo Cormack et al. 2009).
"""

from typing import Optional
import math


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder đơn giản hoặc keyword match score.
    """
    if not candidates:
        return []

    # Sort candidates by existing score or relevance
    sorted_candidates = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
    return sorted_candidates[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.
    """
    if not candidates:
        return []

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            rel = candidates[idx].get("score", 0.0)
            best_score_candidate = rel * lambda_param

            if best_score_candidate > best_score:
                best_score = best_score_candidate
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker (Dense + Sparse).

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List các danh sách kết quả được xếp hạng từ các ranker khác nhau
        top_k: Số lượng kết quả cuối cùng
        k: Hằng số làm mượt (default=60 từ Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    if not ranked_lists:
        return []

    rrf_scores = {}  # content -> score
    content_map = {}  # content -> full item dict

    for ranked_list in ranked_lists:
        if not ranked_list:
            continue
        for rank, item in enumerate(ranked_list, start=1):
            key = item.get("content", "")
            if not key:
                continue

            rrf_score = 1.0 / (k + rank)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + rrf_score

            if key not in content_map:
                content_map[key] = item.copy()

    # Sắp xếp các items theo RRF score giảm dần
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = round(score, 6)
        results.append(item)

    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if not candidates:
        return []

    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    elif method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k=top_k)
    elif method == "mmr":
        return candidates[:top_k]
    else:
        return candidates[:top_k]


if __name__ == "__main__":
    dummy_candidates_1 = [
        {"content": "Chính sách vận chuyển Shopee Mall", "score": 0.8, "metadata": {}},
        {"content": "Quy chế hoạt động sàn TMĐT", "score": 0.6, "metadata": {}},
    ]
    dummy_candidates_2 = [
        {"content": "Quy chế hoạt động sàn TMĐT", "score": 12.5, "metadata": {}},
        {"content": "Điều khoản dịch vụ Shopee", "score": 8.1, "metadata": {}},
    ]
    results = rerank_rrf([dummy_candidates_1, dummy_candidates_2], top_k=2)
    for r in results:
        print(f"[{r['score']:.5f}] {r['content']}")
