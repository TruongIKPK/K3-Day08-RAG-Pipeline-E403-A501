"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "Tôi không thể xác minh thông tin này từ nguồn hiện có"
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION
# =============================================================================

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
LLM_MODEL = "openai/gpt-4o-mini"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về dịch vụ và chính sách (pháp lý, mua sắm, học phí, quy chế).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Mỗi khẳng định phải có trích dẫn ngay sau, ví dụ: [Source: file_name.md] hoặc [Document X]
3. Nếu context không đủ thông tin → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
4. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo đoạn văn
5. Không suy luận hay mở rộng ngoài những gì được nêu trong context"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    """
    if len(chunks) <= 2:
        return chunks

    front = chunks[::2]   # index 0, 2, 4 -> đặt ở đầu
    back = chunks[1::2]   # index 1, 3    -> đặt ở cuối (reversed)
    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.
    """
<<<<<<< HEAD
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"Source {i}")
        doc_type = metadata.get("type", "unknown")
=======
    if not chunks:
        return "Không có context."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
>>>>>>> 36cad2057c36ad41222d74ddd2afd90287d824be
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk.get('content', '')}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.
    """
<<<<<<< HEAD
    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có. (Thiếu API Key trong file .env)",
            "sources": chunks,
            "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
        }

    try:
        from openai import OpenAI
        base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        answer = f"Tôi không thể xác minh thông tin này từ nguồn hiện có. ({e})"
=======
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    # Step 2: Reorder để tránh lost in the middle
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Call LLM (nếu có API Key) hoặc tổng hợp câu trả lời chuẩn hóa có citation
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

    answer = ""
    if api_key and not api_key.startswith("sk-proj-invalid"):
        try:
            from openai import OpenAI
            base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
            client = OpenAI(api_key=api_key, base_url=base_url)

            user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            print(f"⚠ LLM Generation API Call Warning: {e}")

    if not answer:
        if chunks and chunks[0].get("source") != "pageindex":
            top_doc = chunks[0]
            src_name = top_doc.get("metadata", {}).get("source", "Tài liệu hệ thống")
            snippet = top_doc.get("content", "")[:300].strip()
            answer = f"Dựa trên tài liệu hệ thống [{src_name}]: {snippet}..."
        else:
            answer = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    retrieval_source = chunks[0].get("source", "hybrid") if chunks else "none"
>>>>>>> 36cad2057c36ad41222d74ddd2afd90287d824be

    return {
        "answer": answer,
        "sources": chunks,
<<<<<<< HEAD
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none"
    }

=======
        "retrieval_source": retrieval_source
    }
>>>>>>> 36cad2057c36ad41222d74ddd2afd90287d824be


if __name__ == "__main__":
    test_queries = [
        "Chính sách vận chuyển Shopee Mall",
        "Điều khoản dịch vụ Shopee Mall",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
