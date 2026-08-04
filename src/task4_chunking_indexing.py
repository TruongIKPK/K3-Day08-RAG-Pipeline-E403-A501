"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import chromadb

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — OpenAI Cloud API Embedding
# =============================================================================

# Chunking Configuration
CHUNK_SIZE = 800        # 800 ký tự (~150-200 từ) vừa đủ giữ trọn 1-2 điều khoản/đoạn văn bản
CHUNK_OVERLAP = 100      # Overlap 100 ký tự (~12.5%) bảo toàn ngữ cảnh ranh giới
CHUNKING_METHOD = "recursive"

# Embedding Model (OpenAI API - Không tải weights về local)
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Vector store configuration
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "university_services_docs"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        print(f"Thư mục {STANDARDIZED_DIR} không tồn tại!")
        return documents

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        if md_file.name.startswith("."):
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
            if not content.strip():
                continue
            doc_type = "legal" if "legal" in str(md_file.parent) or "legal" in str(md_file) else "news"
            documents.append({
                "content": content,
                "metadata": {"source": md_file.name, "type": doc_type}
            })
        except Exception as e:
            print(f"Lỗi khi đọc file {md_file}: {e}")

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn (RecursiveCharacterTextSplitter).

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            if chunk_text.strip():
                chunks.append({
                    "content": chunk_text,
                    "metadata": {**doc["metadata"], "chunk_index": i}
                })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng OpenAI API (text-embedding-3-small).

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    if not chunks:
        return []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Thiếu OPENAI_API_KEY trong file .env!\n"
            "Vui lòng thêm OPENAI_API_KEY=sk-... vào file .env để chạy embedding API."
        )

    print(f"Đang gọi OpenAI API ({EMBEDDING_MODEL}) để tạo embeddings...")
    client = OpenAI(api_key=api_key)
    texts = [c["content"] for c in chunks]

    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        response = client.embeddings.create(
            input=batch_texts,
            model=EMBEDDING_MODEL
        )
        all_embeddings.extend([item.embedding for item in response.data])

    for chunk, emb in zip(chunks, all_embeddings):
        chunk["embedding"] = emb

    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào Vector Store ChromaDB.
    """
    if not chunks:
        print("Không có chunks để index!")
        return

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Reset collection khi re-index
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]
    documents = [c["content"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    batch_size = 100
    total = len(chunks)
    for i in range(0, total, batch_size):
        end_idx = min(i + batch_size, total)
        collection.upsert(
            ids=ids[i:end_idx],
            documents=documents[i:end_idx],
            embeddings=embeddings[i:end_idx],
            metadatas=metadatas[i:end_idx],
        )

    print(f"[OK] Da luu thanh cong {total} vectors vao ChromaDB collection '{COLLECTION_NAME}' (duong dan: {CHROMA_DIR})")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing (OpenAI API Embedding)")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding Model: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("[OK] Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
