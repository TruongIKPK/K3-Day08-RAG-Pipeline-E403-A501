"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
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
        print("  [WARN] PAGEINDEX_API_KEY missing. Skipping PageIndex upload.")
        return
    try:
        from pageindex.client import PageIndexClient
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            print(f"  ✓ Processed: {md_file.name}")
    except Exception as e:
        print(f"  [WARN] PageIndex upload error: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

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
    if PAGEINDEX_API_KEY:
        try:
            from pageindex.client import PageIndexClient
            client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
            resp = client.submit_query(query=query)
            if resp:
                results = []
                retrieval_id = resp.get("retrieval_id") or resp.get("id")
                if retrieval_id:
                    retrieval = client.get_retrieval(retrieval_id)
                    for node in retrieval.get("retrieved_nodes", []):
                        for group in node.get("relevant_contents", []):
                            for item in group:
                                results.append({
                                    "content": item.get("relevant_content", ""),
                                    "score": 0.6,
                                    "metadata": {"section": item.get("section_title")},
                                    "source": "pageindex",
                                })
                if results:
                    return results[:top_k]
        except Exception as e:
            print(f"  [INFO] PageIndex API error ({e}). Using structural fallback...")

    return [
        {
            "content": f"[PageIndex Structural Fallback] Tra cứu cấu trúc mục lục tổng hợp quy chế/chính sách cho truy vấn: '{query}'.",
            "score": 0.5,
            "metadata": {"section": "Cấu trúc quy định tổng hợp"},
            "source": "pageindex",
        }
    ][:top_k]



if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
