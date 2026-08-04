# RAGAS Evaluation — Trợ Lý Pháp Lý Khởi Nghiệp & Shopee

## Trạng thái đánh giá

- **Phạm vi:** pháp lý khởi nghiệp, doanh nghiệp và hoạt động bán hàng trên Shopee tại Việt Nam
- **Framework:** RAGAS
- **Golden dataset:** 20 câu hỏi, đáp ứng yêu cầu 15–20 mẫu
- **Config A:** Dense + BM25 + Reciprocal Rank Fusion (RRF)
- **Config B:** Dense-only
- **PageIndex:** không sử dụng
- **Ngày chuẩn bị artifact:** 04/08/2026
- **Trạng thái chạy:** **chưa thực thi theo yêu cầu của người dùng**

> Báo cáo không tự tạo hoặc ước lượng điểm. Bốn metric và phần bottom-3 giữ N/A cho đến khi pipeline thật được chạy. Sau khi chạy thành công, eval_pipeline.py sẽ ghi lại báo cáo bằng kết quả RAGAS thực tế.

## Phạm vi golden dataset

Bộ dữ liệu được chia cân bằng giữa chính sách nền tảng và pháp luật doanh nghiệp:

| Nhóm | Số câu | Nội dung |
|---|---:|---|
| Chính sách Shopee | 6 | Quan hệ Người Mua–Người Bán–Shopee, thông tin sản phẩm, giá, hóa đơn/thuế, chứng từ, xử lý vi phạm |
| Pháp luật thương mại điện tử | 4 | Trách nhiệm của sàn và người bán, quy chế hoạt động theo Nghị định 52/2013/NĐ-CP |
| Pháp luật khởi nghiệp/doanh nghiệp | 10 | Đại diện pháp luật, tên, dấu, công ty TNHH, công ty cổ phần, doanh nghiệp tư nhân |
| **Tổng** | **20** | |

Các câu hỏi về Nghị định 52 và Luật Doanh nghiệp được diễn đạt là “theo văn bản trong corpus”. Cách viết này cố ý không biến bản PDF cũ thành kết luận về pháp luật đang có hiệu lực.

## Nguồn chuẩn để xây dựng ground truth

- [Điều khoản dịch vụ Shopee](https://help.shopee.vn/portal/4/article/77243-%C4%90I%E1%BB%80U-KHO%E1%BA%A2N-D%E1%BB%8ACH-V%E1%BB%A4): mục 1.2, 15.1, 15.2, 15.5 và 15.6.
- [Quy định về đăng bán sản phẩm trên Shopee](https://help.shopee.vn/portal/4/article/77246): yêu cầu về chứng từ và phần xử lý vi phạm.
- [Nghị định 52/2013/NĐ-CP trên CSDL quốc gia về VBPL](https://vbpl.moj.gov.vn/hanoi/Pages/vbpq-toanvan.aspx?ItemID=30470&Keyword=): Điều 36, 37 và 38.
- [Luật Doanh nghiệp 59/2020/QH14 trên CSDL quốc gia về VBPL](https://vbpl.vn/bokehoachvadautu/Pages/vbpq-toanvan.aspx?ItemID=142881): các Điều 12, 37, 43, 46, 47, 74, 111, 113 và 188.

## Kiểm tra corpus và blocker trước khi chạy

1. **Nguồn Shopee chưa nằm trong corpus cục bộ.** Thư mục data/landing/legal hiện chỉ có ba PDF pháp luật. Sáu câu hỏi chính sách Shopee sẽ không đo đúng Context Recall/Precision cho đến khi hai trang Shopee ở trên được thu thập, chuẩn hóa và index kèm URL, tiêu đề và ngày thu thập.

2. **Nghị định 52 trong corpus không phải trạng thái pháp lý mới nhất.** File data/landing/legal/52-nd.pdf là Nghị định 52/2013/NĐ-CP; văn bản này đã được [Nghị định 85/2021/NĐ-CP](https://vanban.chinhphu.vn/default.aspx?docid=204191&pageid=27160) sửa đổi, bổ sung và CSDL quốc gia đánh dấu hết hiệu lực một phần. Trước khi dùng chatbot ngoài bài lab, cần bổ sung bản sửa đổi/bản hợp nhất và metadata hiệu lực.

3. **Luật Doanh nghiệp trong corpus đã được sửa đổi.** Hai file luatdoanhnghiep1.pdf và luatdoanhnghiep2.pdf là Luật 59/2020/QH14, hiện được CSDL quốc gia đánh dấu hết hiệu lực một phần. Ngoài Luật 03/2022/QH15, [Luật 76/2025/QH15](https://vanban.chinhphu.vn/?classid=1&docid=214562&orggroupi=&pageid=27160) có hiệu lực từ 01/07/2025 cũng sửa đổi, bổ sung Luật Doanh nghiệp. Corpus phải lưu version và trạng thái hiệu lực để tránh trả lời theo bản cũ.

4. **Ba PDF pháp luật là tài liệu scan.** Cần OCR và kiểm tra thủ công số Điều/Khoản trước khi tạo Markdown; nếu chỉ index ảnh hoặc text OCR lỗi, cả Context Recall lẫn Faithfulness đều mất ý nghĩa.

5. **Upstream chưa sẵn sàng để chạy end-to-end.** Dense retrieval trong src/task5_semantic_search.py vẫn chưa được triển khai; data/standardized cũng phải có Markdown đã chuẩn hóa. Đây là blocker thực tế, không được thay thế bằng điểm giả.

6. **Không phụ thuộc PageIndex.** eval_pipeline.py kết hợp trực tiếp semantic_search và lexical_search rồi tự fuse bằng RRF, vì vậy không import Task 8/Task 9 và không cần cài pageindex/litellm cho bài đánh giá này.

## Overall Scores

| Metric | Config A (Dense + BM25 + RRF) | Config B (Dense-only) | Δ (A - B) |
|---|---:|---:|---:|
| Faithfulness | N/A | N/A | N/A |
| Answer Relevance | N/A | N/A | N/A |
| Context Recall | N/A | N/A | N/A |
| Context Precision | N/A | N/A | N/A |
| **Average** | **N/A** | **N/A** | **N/A** |

### Quy tắc diễn giải

- Mỗi metric nằm trong khoảng 0–1; càng cao càng tốt.
- Ngưỡng mục tiêu ban đầu đề xuất: 0,70 cho từng metric.
- Chỉ kết luận Config A tốt hơn khi điểm trung bình cao hơn và Faithfulness không giảm đáng kể.
- Nếu Recall tăng nhưng Precision giảm, cần điều chỉnh candidate pool, top_k hoặc cách chunk theo Điều/Khoản.
- Với domain pháp lý, một câu trả lời có vẻ liên quan nhưng dùng sai phiên bản văn bản vẫn phải được xem là lỗi Faithfulness/corpus.

## Thiết kế A/B

### Config A — Dense + BM25 + RRF

- Dense retrieval tìm các cách diễn đạt gần nghĩa.
- BM25 ưu tiên từ khóa pháp lý chính xác như số văn bản, số Điều/Khoản, tên loại hình doanh nghiệp và tên chính sách Shopee.
- RRF hợp nhất thứ hạng của hai danh sách mà không so sánh trực tiếp hai thang điểm khác nhau.
- Lấy cùng TOP_K = 5 context cho generation.

### Config B — Dense-only

- Chỉ dùng semantic_search với cùng candidate size và TOP_K.
- Dùng cùng prompt pháp lý, model sinh, nhiệt độ và RAGAS judge như Config A.
- Vì chỉ thay retrieval, chênh lệch metric phản ánh tác động của BM25 + RRF.

### Kết luận

**Chưa thể kết luận trước khi chạy.** eval_pipeline.py sẽ tính Δ theo Config A trừ Config B và sinh kết luận từ điểm thật.

## Worst Performers của Config A (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Precision | Failure Stage | Root Cause |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | N/A — chưa chạy | N/A | N/A | N/A | N/A | N/A | Không bịa dữ liệu khi chưa có kết quả RAGAS |
| 2 | N/A — chưa chạy | N/A | N/A | N/A | N/A | N/A | Không bịa dữ liệu khi chưa có kết quả RAGAS |
| 3 | N/A — chưa chạy | N/A | N/A | N/A | N/A | N/A | Không bịa dữ liệu khi chưa có kết quả RAGAS |

Sau khi chạy, script xếp hạng từng câu theo trung bình bốn metric. Metric thấp nhất cung cấp chẩn đoán sơ bộ:

- **Faithfulness thấp:** generation suy diễn ngoài chứng cứ, trộn luật với chính sách Shopee hoặc bỏ qua trạng thái hiệu lực.
- **Answer Relevance thấp:** câu trả lời không trực tiếp giải quyết vấn đề pháp lý.
- **Context Recall thấp:** thiếu Điều/Khoản cần thiết, thiếu trang Shopee hoặc thiếu văn bản sửa đổi trong corpus.
- **Context Precision thấp:** chunk sai ranh giới, trùng lặp hoặc truy xuất nhầm văn bản/phiên bản.

## Khuyến nghị QA

1. Thu thập hai trang Shopee chính thức trong golden dataset, lưu snapshot hoặc nội dung chuẩn hóa cùng URL và ngày thu thập.
2. Bổ sung Nghị định 85/2021/NĐ-CP và các luật sửa đổi Luật Doanh nghiệp; gắn metadata số văn bản, Điều, Khoản, ngày hiệu lực và trạng thái hiệu lực.
3. OCR ba PDF, sau đó đối soát thủ công các Điều 12, 37, 43, 46, 47, 74, 111, 113, 188 và các Điều 36–38 của Nghị định 52.
4. Chunk theo cấu trúc Chương → Mục → Điều → Khoản; không cắt một khoản giữa hai chunk.
5. Yêu cầu generation trích dẫn từng kết luận, phân loại nguồn là pháp luật hay chính sách nền tảng, và từ chối khi không đủ căn cứ.
6. Không dùng điểm RAGAS để che lỗi corpus: nếu sáu câu Shopee không có tài liệu nguồn, phải sửa ingestion trước khi tối ưu retrieval.

## Cách chạy sau khi gỡ blocker

Từ thư mục gốc repository:

    python -m group_project.evaluation.eval_pipeline

Script sẽ:

1. Xác thực golden dataset có 15–20 mẫu, đủ ba trường bắt buộc và không trùng câu hỏi.
2. Chạy cùng 20 câu trên Config A và Config B.
3. Thu thập Faithfulness, Answer Relevance, Context Recall và Context Precision.
4. Tính trung bình, Δ, bottom-3, chẩn đoán failure stage và ghi lại results.md.

Các biến model tùy chọn: EVAL_GENERATION_MODEL, RAGAS_JUDGE_MODEL và RAGAS_EMBEDDING_MODEL.
