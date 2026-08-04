"""Task 10 - grounded generation with citations and offline extractive mode."""

from __future__ import annotations

import os

from dotenv import load_dotenv

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:  # pragma: no cover - supports direct script execution
    from task9_retrieval_pipeline import retrieve

load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
REFUSAL = "T\u00f4i kh\u00f4ng th\u1ec3 x\u00e1c minh th\u00f4ng tin n\u00e0y t\u1eeb ngu\u1ed3n hi\u1ec7n c\u00f3."

SYSTEM_PROMPT = """You are a grounded RAG assistant.
Use only the supplied context. Every factual claim must be followed by a
citation such as [Source: file.md]. If the context does not support the
question, say that the information cannot be verified from the available
sources. Do not invent facts or citations. Answer in the user's language."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Place high-ranked chunks at the context edges to reduce lost-in-middle."""

    if not isinstance(chunks, list):
        raise TypeError("chunks must be a list")
    if len(chunks) <= 2:
        return list(chunks)
    return list(chunks[::2]) + list(chunks[1::2][::-1])


def format_context(chunks: list[dict]) -> str:
    """Format evidence with stable source labels for the LLM and UI."""

    if not chunks:
        return "No context is available."
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        source = metadata.get("source", f"Source {index}")
        doc_type = metadata.get("type", metadata.get("doc_type", "unknown"))
        page = metadata.get("page", metadata.get("page_number", "unknown"))
        content = chunk.get("content", "") if isinstance(chunk, dict) else ""
        parts.append(
            f"[Document {index} | Source: {source} | Type: {doc_type} | Page: {page}]\n{content}"
        )
    return "\n\n---\n\n".join(parts)


def _should_call_llm() -> bool:
    """Keep tests and local demos deterministic; opt into network generation."""

    return os.getenv("RAG_ENABLE_LLM", "0").casefold() in {"1", "true", "yes"}


def _generate_with_llm(query: str, context: str) -> str | None:
    if not _should_call_llm():
        return None
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI

        options = {"api_key": api_key}
        if os.getenv("OPENROUTER_API_KEY"):
            options["base_url"] = "https://openrouter.ai/api/v1"
        response = OpenAI(**options).chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        answer = response.choices[0].message.content
        return answer.strip() if isinstance(answer, str) and answer.strip() else None
    except Exception:
        return None


def _extractive_answer(chunks: list[dict]) -> str:
    """Produce a useful grounded answer when no LLM is enabled."""

    if not chunks:
        return REFUSAL
    snippets: list[str] = []
    for chunk in chunks[:2]:
        content = " ".join(str(chunk.get("content", "")).split())
        if not content:
            continue
        metadata = chunk.get("metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        source = metadata.get("source", "unknown")
        snippet = content[:420].rstrip()
        if len(content) > 420:
            snippet += "..."
        snippets.append(f"{snippet} [Source: {source}]")
    return "\n\n".join(snippets) if snippets else REFUSAL


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Run retrieval and return an answer, source chunks and retrieval source."""

    if not isinstance(query, str):
        raise TypeError("query must be a string")
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    answer = _generate_with_llm(query, context) or _extractive_answer(reordered)
    retrieval_source = chunks[0].get("source", "none") if chunks else "none"
    return {"answer": answer, "sources": chunks, "retrieval_source": retrieval_source}


if __name__ == "__main__":
    result = generate_with_citation("What is the shipping policy?", top_k=3)
    print(result["answer"])

