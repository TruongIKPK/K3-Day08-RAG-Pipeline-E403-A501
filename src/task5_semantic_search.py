"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb

from .task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

load_dotenv()

_collection = None


def _get_collection():
    """Kết nối (và cache) tới ChromaDB collection đã index ở Task 4."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


def _get_query_embedding(query: str) -> list[float]:
    """Tạo embedding cho query câu hỏi từ người dùng."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            res = client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
            return res.data[0].embedding
        except Exception as e:
            print(f"⚠ Semantic search API warning: {e}")
            return []
    else:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(EMBEDDING_MODEL)
            return model.encode(query).tolist()
        except Exception as e:
            print(f"⚠ Local sentence-transformers warning: {e}")
            return []


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    collection = _get_collection()

    # Tạo vector cho query câu hỏi
    query_vector = _get_query_embedding(query)
    if not query_vector:
        return []

    # Truy vấn vector store (cosine similarity)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results and results.get("documents") and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            score = max(0.0, 1.0 - dist)  # cosine distance → similarity
            output.append({"content": doc, "score": round(score, 4), "metadata": meta})

    output.sort(key=lambda x: x["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    results = semantic_search("quần áo Shopee Mall", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
