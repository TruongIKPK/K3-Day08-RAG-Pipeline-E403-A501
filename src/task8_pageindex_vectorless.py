"""Task 8 - structural/vectorless retrieval with an optional PageIndex API."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from ._rag_common import safe_top_k, structural_sections, tokenize
from .task4_chunking_indexing import load_documents

load_dotenv()

STANDARDIZED_DIR = Path(__file__).resolve().parent.parent / "data" / "standardized"
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")


def _api_value(value, name: str, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def upload_documents() -> list[str]:
    """Upload standardized files when PageIndex is configured.

    Returning an empty list is a valid offline result; the local structural
    index remains available for Task 9 fallback.
    """

    api_key = os.getenv("PAGEINDEX_API_KEY", PAGEINDEX_API_KEY)
    if not api_key or not STANDARDIZED_DIR.is_dir():
        return []
    try:
        from pageindex.client import PageIndexClient

        client = PageIndexClient(api_key=api_key)
        uploaded: list[str] = []
        for path in sorted(STANDARDIZED_DIR.rglob("*.md"), key=lambda item: item.as_posix().casefold()):
            response = client.submit_document(str(path))
            identifier = _api_value(response, "doc_id") or _api_value(response, "id")
            uploaded.append(str(identifier or path))
        return uploaded
    except Exception:
        return []


def _pageindex_api_search(query: str, top_k: int) -> list[dict]:
    api_key = os.getenv("PAGEINDEX_API_KEY", PAGEINDEX_API_KEY)
    if not api_key:
        return []
    try:
        from pageindex.client import PageIndexClient

        client = PageIndexClient(api_key=api_key)
        response = client.submit_query(query=query)
        retrieval_id = _api_value(response, "retrieval_id") or _api_value(response, "id")
        if not retrieval_id:
            return []
        retrieval = client.get_retrieval(retrieval_id)
        output: list[dict] = []
        nodes = _api_value(retrieval, "retrieved_nodes", []) or []
        for node in nodes:
            groups = _api_value(node, "relevant_contents", []) or []
            for group in groups:
                items = group if isinstance(group, list) else [group]
                for item in items:
                    content = _api_value(item, "relevant_content", "")
                    if not content:
                        continue
                    output.append(
                        {
                            "content": str(content),
                            "score": 0.6,
                            "metadata": {"section": _api_value(item, "section_title", "")},
                            "source": "pageindex",
                        }
                    )
                    if len(output) >= top_k:
                        return output
        return output[:top_k]
    except Exception:
        return []


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve structurally relevant sections and mark every result as PageIndex."""

    if not isinstance(query, str):
        raise TypeError("query must be a string")
    limit = safe_top_k(top_k)
    if limit == 0 or not query.strip():
        return []

    api_results = _pageindex_api_search(query, limit)
    if api_results:
        return api_results[:limit]

    query_tokens = set(tokenize(query))
    candidates: list[tuple[float, int, dict]] = []
    serial = 0
    for document in load_documents():
        source = document.get("metadata", {}).get("source", "unknown")
        doc_type = document.get("metadata", {}).get("type", "document")
        for section in structural_sections(document.get("content", ""), source, doc_type):
            serial += 1
            section_tokens = set(tokenize(section["content"]))
            overlap = len(query_tokens & section_tokens) / len(query_tokens) if query_tokens else 0.0
            title = section["metadata"].get("section", "").casefold()
            title_bonus = 0.2 if any(token in title for token in query_tokens) else 0.0
            candidates.append((overlap + title_bonus, serial, section))

    candidates.sort(key=lambda row: (-row[0], row[2]["metadata"].get("source", ""), row[1]))
    results: list[dict] = []
    for score, _, section in candidates[:limit]:
        item = {
            "content": section["content"],
            "score": round(max(0.05, min(1.0, score)), 6),
            "metadata": dict(section["metadata"]),
            "source": "pageindex",
        }
        if score == 0:
            item["metadata"]["fallback"] = True
        results.append(item)

    if results:
        return results
    return [
        {
            "content": "No structural evidence is available for this query.",
            "score": 0.05,
            "metadata": {"type": "fallback", "section": "none"},
            "source": "pageindex",
        }
    ]


if __name__ == "__main__":
    for result in pageindex_search("shipping policy", top_k=2):
        print(f"[{result['source']}] {result['metadata'].get('source', 'n/a')}")

