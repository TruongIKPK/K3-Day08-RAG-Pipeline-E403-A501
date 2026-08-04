"""Task 7 - reranking and Reciprocal Rank Fusion utilities."""

from __future__ import annotations

import math

from ._rag_common import cosine_similarity, hashed_embedding, safe_top_k, tokenize


def _base_score(item: dict) -> float:
    try:
        value = float(item.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def rerank_cross_encoder(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Use lexical relevance as a local cross-encoder substitute.

    A real cross encoder can be added later without changing the public
    contract. This deterministic scorer is particularly useful for an offline
    classroom demo and is multilingual because it uses Unicode tokens.
    """

    if not isinstance(query, str):
        raise TypeError("query must be a string")
    limit = safe_top_k(top_k)
    if limit == 0 or not isinstance(candidates, list):
        return []
    query_tokens = set(tokenize(query))
    scored: list[tuple[float, int, dict]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        content = str(candidate.get("content", ""))
        content_tokens = set(tokenize(content))
        overlap = len(query_tokens & content_tokens) / len(query_tokens) if query_tokens else 0.0
        phrase_bonus = 0.15 if query.strip().casefold() in content.casefold() else 0.0
        base = _base_score(candidate)
        normalized_base = base / (1.0 + abs(base))
        final_score = 0.65 * min(1.0, overlap + phrase_bonus) + 0.35 * normalized_base
        item = candidate.copy()
        item["score"] = round(final_score, 6)
        scored.append((final_score, index, item))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [item for _, _, item in scored[:limit]]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Select relevant but non-duplicate candidates using MMR."""

    limit = safe_top_k(top_k)
    if not isinstance(candidates, list) or limit == 0:
        return []
    if not 0.0 <= float(lambda_param) <= 1.0:
        raise ValueError("lambda_param must be between 0 and 1")
    query_vector = list(query_embedding or [])
    remaining = [index for index, item in enumerate(candidates) if isinstance(item, dict)]
    selected: list[int] = []
    vectors: dict[int, list[float]] = {}
    for index in remaining:
        value = candidates[index].get("embedding")
        vectors[index] = list(value) if isinstance(value, list) else hashed_embedding(candidates[index].get("content", ""), len(query_vector) or 256)

    while remaining and len(selected) < limit:
        best_index = remaining[0]
        best_value = float("-inf")
        for index in remaining:
            relevance = cosine_similarity(query_vector, vectors[index]) if query_vector else _base_score(candidates[index])
            redundancy = max(
                (cosine_similarity(vectors[index], vectors[chosen]) for chosen in selected),
                default=0.0,
            )
            value = float(lambda_param) * relevance - (1.0 - float(lambda_param)) * redundancy
            if value > best_value:
                best_value, best_index = value, index
        selected.append(best_index)
        remaining.remove(best_index)

    output = []
    for index in selected:
        item = candidates[index].copy()
        item["score"] = round(_base_score(item), 6)
        output.append(item)
    return output


def rerank_rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """Fuse ranked lists with ``RRF(d) = sum(1 / (k + rank))``."""

    limit = safe_top_k(top_k)
    if not isinstance(ranked_lists, list) or limit == 0:
        return []
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")

    scores: dict[str, float] = {}
    representatives: dict[str, dict] = {}
    first_seen: dict[str, int] = {}
    serial = 0
    for ranked_list in ranked_lists:
        if not isinstance(ranked_list, list):
            continue
        seen: set[str] = set()
        for rank, item in enumerate(ranked_list, start=1):
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content or content in seen:
                continue
            seen.add(content)
            serial += 1
            scores[content] = scores.get(content, 0.0) + 1.0 / (k + rank)
            representatives.setdefault(content, item.copy())
            first_seen.setdefault(content, serial)

    ordered = sorted(scores, key=lambda content: (-scores[content], first_seen[content]))
    output = []
    for content in ordered[:limit]:
        item = representatives[content].copy()
        item["score"] = round(scores[content], 6)
        output.append(item)
    return output


def rerank(
    query: str,
    candidates: list[dict] | list[list[dict]],
    top_k: int = 5,
    method: str = "rrf",
) -> list[dict]:
    """Unified reranking interface for a flat or multi-ranker candidate set."""

    limit = safe_top_k(top_k)
    if limit == 0 or not candidates:
        return []
    is_nested = isinstance(candidates[0], list) if isinstance(candidates, list) else False
    ranked_lists = candidates if is_nested else [candidates]
    method_name = (method or "rrf").casefold()
    if method_name == "rrf":
        return rerank_rrf(ranked_lists, top_k=limit)
    flat = [item for ranked_list in ranked_lists for item in ranked_list]
    if method_name in {"cross_encoder", "cross-encoder", "cross"}:
        return rerank_cross_encoder(query, flat, top_k=limit)
    if method_name == "mmr":
        query_vector = hashed_embedding(query, 256)
        return rerank_mmr(query_vector, flat, top_k=limit)
    return [item.copy() for item in flat[:limit] if isinstance(item, dict)]


if __name__ == "__main__":
    print(rerank("shipping policy", [{"content": "shipping policy", "score": 0.8}], top_k=1))

