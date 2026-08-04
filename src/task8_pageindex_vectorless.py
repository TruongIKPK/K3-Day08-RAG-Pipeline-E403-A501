"""
Task 8 — PageIndex Vectorless RAG.

PageIndex cho phép RAG không dùng vector store — sử dụng
structural understanding của document.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        return []

    try:
        from pageindex.client import PageIndexClient
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        uploaded = []
        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            resp = client.submit_document(str(md_file))
            doc_id = resp.get("doc_id") or resp.get("id")
            uploaded.append(doc_id)
        return uploaded
    except Exception as e:
        print(f"Lỗi khi upload PageIndex: {e}")
        return []


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex làm fallback.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    results = []

    if PAGEINDEX_API_KEY:
        try:
            from pageindex.client import PageIndexClient
            client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            resp = client.submit_query(query=query)
            retrieval_id = resp.get("retrieval_id") or resp.get("id")
            retrieval = client.get_retrieval(retrieval_id)

            for node in retrieval.get("retrieved_nodes", [])[:top_k]:
                for group in node.get("relevant_contents", []):
                    for item in group:
                        results.append({
                            "content": item.get("relevant_content", ""),
                            "score": 0.5,
                            "metadata": {"section": item.get("section_title", "")},
                            "source": "pageindex",
                        })
        except Exception as e:
            print(f"PageIndex API query error: {e}")

    # Fallback khi chưa cấu hình API Key hoặc query không có trong index
    if not results:
        results.append({
            "content": f"[PageIndex Fallback] Không tìm thấy thông tin phù hợp cho câu hỏi: '{query}'. Vui lòng thử câu hỏi khác.",
            "score": 0.1,
            "metadata": {"type": "fallback"},
            "source": "pageindex"
        })

    return results[:top_k]


if __name__ == "__main__":
    results = pageindex_search("quần áo Shopee Mall", top_k=2)
    for r in results:
        print(f"[{r['source']}] {r['content']}")
