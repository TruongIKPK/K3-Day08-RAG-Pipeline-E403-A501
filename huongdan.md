⚙️ 1. Cấu Hình Môi Trường & File 

.env
Bước 1: Khởi tạo Virtual Environment & Cài đặt Thư viện
Mở terminal (PowerShell) tại thư mục dự án và chạy các lệnh sau:

powershell
# 1. Tạo môi trường ảo .venv (nếu chưa có)
python -m venv .venv
# 2. Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1
# 3. Cài đặt các gói phụ thuộc từ requirements.txt
pip install --upgrade pip
pip install -r requirements.txt
# 4. Cài đặt các gói bổ sung cho PDF conversion và Crawl4AI
pip install "markitdown[pdf]"
playwright install chromium
Bước 2: Điền API Keys vào file 

.env
Mở file 

.env
 và khai báo các API Key cần thiết (tham khảo mẫu tại 

.env.example
):

env
# OPENROUTER API Key (Khuyên dùng cho bài lab - có thể dùng các model miễn phí)
OPENROUTER_API_KEY=sk-or-v1-your-openrouter-key...
# HOẶC OPENAI API Key (Tùy chọn)
OPENAI_API_KEY=sk-proj-your-openai-key...
# PageIndex API Key (Tùy chọn - dùng cho Task 8 Fallback)
PAGEINDEX_API_KEY=pix_your-pageindex-key...
# Jina API Key (Tùy chọn - dùng cho Task 7 Jina Reranker)
JINA_API_KEY=jina_your-jina-key...
NOTE

Chỉ cần có tối thiểu OPENROUTER_API_KEY hoặc OPENAI_API_KEY là hệ thống đã có thể chạy đầy đủ luồng Generation.

🚀 2. Vận Hành Lần Lượt Các Task Trong 

src/
Vận hành hệ thống theo đúng thứ tự luồng dữ liệu (chạy từ thư mục gốc của dự án):

powershell
# Task 1: Thu thập file PDF/DOCX chính sách đại học -> data/landing/legal/
python -m src.task1_collect_legal_docs
# Task 2: Crawl các bài báo/tin tức dịch vụ -> data/landing/news/
python -m src.task2_crawl_news
# Task 3: Convert toàn bộ dữ liệu PDF/JSON sang Markdown -> data/standardized/
python -m src.task3_convert_markdown
# Task 4: Chunking văn bản (800 chars, 100 overlap), embed BAAI/bge-m3 và index vào ChromaDB -> chroma_db/
python -m src.task4_chunking_indexing
# Task 5: Kiểm tra module Semantic Search (Dense Retrieval + HyDE)
python -m src.task5_semantic_search
# Task 6: Kiểm tra module Lexical Search (BM25 Sparse Retrieval)
python -m src.task6_lexical_search
# Task 7: Kiểm tra module Reranking (Reciprocal Rank Fusion - RRF)
python -m src.task7_reranking
# Task 8: Kiểm tra module Vectorless RAG với PageIndex
python -m src.task8_pageindex_vectorless
# Task 9: Chạy kiểm thử Pipeline Retrieval hoàn chỉnh (Hybrid + Threshold Fallback 0.48)
python -m src.task9_retrieval_pipeline
# Task 10: Chạy kiểm thử Generation trả lời có Citation & Reorder chunks
python -m src.task10_generation
🧪 3. Kiểm Tra Tự Động Với Pytest (Chấm Điểm Bài Cá Nhân)
Để kiểm tra xem tất cả các module trong thư mục 

src/
 đã đạt yêu cầu chưa:

powershell
# Chạy toàn bộ test suite (mục tiêu đạt 35/35 passed = 50 điểm)
pytest tests/test_individual.py -v
# Chạy test cho riêng từng task (ví dụ Task 5)
pytest tests/test_individual.py::TestTask5 -v
🖥️ 4. Khởi Chạy Chatbot Web UI & Evaluation (Bài Tập Nhóm)
powershell
# 1. Khởi chạy ứng dụng Chatbot Streamlit UI
streamlit run app.py
# 2. Khởi chạy pipeline đánh giá RAGAS (trên golden_dataset.json)
python -m group_project.evaluation.eval_pipeline