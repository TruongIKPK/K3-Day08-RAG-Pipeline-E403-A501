"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from .task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

_model = None
_collection = None


def _get_embedding_model():
    """Load (và cache) embedding model — cùng model dùng ở Task 4."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as e:
            print(f"  [WARN] Cannot load embedding model ({e})")
            return None
    return _model


def _get_collection():
    """Kết nối (và cache) tới ChromaDB collection đã index ở Task 4."""
    global _collection
    if _collection is None:
        try:
            import chromadb
            if CHROMA_DIR.exists():
                client = chromadb.PersistentClient(path=str(CHROMA_DIR))
                _collection = client.get_collection(name=COLLECTION_NAME)
        except Exception as e:
            print(f"  [WARN] Cannot load ChromaDB collection ({e})")
            return None
    return _collection



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
    model = _get_embedding_model()
    collection = _get_collection()

    if model is None or collection is None:
        return []

    try:
        query_vector = model.encode(query).tolist()
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
    except Exception as e:
        print(f"  [WARN] Semantic search error: {e}")

    return []



if __name__ == "__main__":
    # Test
    results = semantic_search("what is the tuition fee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
