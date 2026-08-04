"""RAGAS evaluation for the Startup & Shopee Legal Assistant.

The same golden dataset is evaluated with:
- Config A: dense retrieval + BM25, fused by Reciprocal Rank Fusion (RRF).
- Config B: dense-only retrieval.

PageIndex is intentionally excluded. Run from the repository root only after the
upstream ingestion, standardization and dense-search tasks are complete:

    python -m group_project.evaluation.eval_pipeline
"""

import json
import math
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

TOP_K = 5
CANDIDATE_MULTIPLIER = 2
RRF_K = 60
TEMPERATURE = 0.1
TOP_P = 0.9

METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
)

CONFIGS = {
    "hybrid_rrf": {
        "label": "Config A (Dense + BM25 + RRF)",
        "retrieval_mode": "hybrid",
    },
    "dense_only": {
        "label": "Config B (Dense-only)",
        "retrieval_mode": "dense",
    },
}

LEGAL_SYSTEM_PROMPT = """Bạn là Trợ Lý Pháp Lý Khởi Nghiệp & Thương Mại Điện Tử,
tập trung vào hoạt động bán hàng trên Shopee tại Việt Nam.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin có trong Context; không tự bổ sung điều luật, thời hạn,
   mức phạt hoặc chính sách nền tảng.
2. Phân biệt rõ nguồn pháp luật với Điều khoản/Chính sách Shopee. Chính sách
   Shopee không phải là văn bản quy phạm pháp luật.
3. Với mỗi kết luận quan trọng, trích dẫn [Document N] tương ứng.
4. Nếu Context cho thấy văn bản đã bị sửa đổi, hết hiệu lực một phần, có xung
   đột hoặc thiếu ngày hiệu lực, phải nêu giới hạn đó và không khẳng định đây là
   quy định hiện hành.
5. Nếu không đủ căn cứ, trả lời đúng câu:
   "Tôi không thể xác minh thông tin này từ nguồn hiện có."
6. Trả lời ngắn gọn, thực tiễn, không suy đoán. Đây là thông tin tham khảo,
   không thay thế tư vấn của luật sư cho tình huống cụ thể.
"""


def load_golden_dataset() -> list[dict]:
    """Load and validate a 15-20 item golden dataset."""
    with GOLDEN_DATASET_PATH.open("r", encoding="utf-8") as stream:
        dataset = json.load(stream)

    if not isinstance(dataset, list) or not 15 <= len(dataset) <= 20:
        raise ValueError("golden_dataset.json must contain between 15 and 20 items")

    required_fields = ("question", "expected_answer", "expected_context")
    seen_questions: set[str] = set()

    for index, item in enumerate(dataset):
        if not isinstance(item, dict):
            raise TypeError(f"Golden item {index} must be an object")

        for field in required_fields:
            value = item.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Golden item {index} has invalid field: {field}")

        normalized_question = item["question"].strip().casefold()
        if normalized_question in seen_questions:
            raise ValueError(f"Duplicate golden question at item {index}")
        seen_questions.add(normalized_question)

    return dataset


def _api_settings() -> tuple[str, str | None, str, str, str]:
    """Resolve generation, judge and embedding models for an OpenAI-compatible API."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if openrouter_key:
        return (
            openrouter_key,
            "https://openrouter.ai/api/v1",
            os.getenv("EVAL_GENERATION_MODEL", "openai/gpt-4o-mini"),
            os.getenv("RAGAS_JUDGE_MODEL", "openai/gpt-4o-mini"),
            os.getenv("RAGAS_EMBEDDING_MODEL", "openai/text-embedding-3-small"),
        )

    if openai_key:
        return (
            openai_key,
            None,
            os.getenv("EVAL_GENERATION_MODEL", "gpt-4o-mini"),
            os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini"),
            os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small"),
        )

    raise EnvironmentError(
        "Set OPENROUTER_API_KEY or OPENAI_API_KEY before running evaluation"
    )


def _normalize_chunk(item: Any, retrieval_source: str) -> dict | None:
    """Normalize one retrieval result and reject malformed/empty chunks."""
    if not isinstance(item, dict):
        return None

    content = item.get("content")
    if not isinstance(content, str) or not content.strip():
        return None

    metadata = item.get("metadata")
    metadata = metadata.copy() if isinstance(metadata, dict) else {}

    normalized = item.copy()
    normalized["content"] = content.strip()
    normalized["metadata"] = metadata
    normalized["retrieval_source"] = retrieval_source
    return normalized


def _normalize_chunks(items: Any, retrieval_source: str) -> list[dict]:
    if not isinstance(items, list):
        raise TypeError(f"{retrieval_source} retrieval must return a list")

    output = []
    for item in items:
        normalized = _normalize_chunk(item, retrieval_source)
        if normalized is not None:
            output.append(normalized)
    return output


def _rrf_fuse(
    ranked_lists: list[list[dict]],
    top_k: int,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """Fuse ranked lists by content identity using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    representatives: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        seen_in_list: set[str] = set()
        for rank, item in enumerate(ranked_list, start=1):
            key = item["content"].strip()
            if not key or key in seen_in_list:
                continue
            seen_in_list.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            representatives.setdefault(key, item)

    ranked_keys = sorted(scores, key=lambda key: (-scores[key], key))

    results = []
    for key in ranked_keys[:top_k]:
        item = representatives[key].copy()
        item["score"] = scores[key]
        item["retrieval_source"] = "hybrid_rrf"
        results.append(item)
    return results


def _reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Place stronger chunks at the beginning and end of the LLM context."""
    if len(chunks) <= 2:
        return list(chunks)
    return chunks[::2] + chunks[1::2][::-1]


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks with stable citation labels and legal metadata."""
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "unknown")
        doc_type = metadata.get("type", metadata.get("doc_type", "unknown"))
        page = metadata.get("page", metadata.get("page_number", "unknown"))
        parts.append(
            f"[Document {index} | Source: {source} | Type: {doc_type} | Page: {page}]\n"
            f"{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


class EvaluationRAGPipeline:
    """Keep generation fixed while varying only the retrieval configuration."""

    def __init__(self, retrieval_mode: str) -> None:
        if retrieval_mode not in {"hybrid", "dense"}:
            raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")
        self.retrieval_mode = retrieval_mode

    def _retrieve(self, question: str) -> list[dict]:
        from src.task5_semantic_search import semantic_search

        candidate_k = TOP_K * CANDIDATE_MULTIPLIER
        dense = _normalize_chunks(
            semantic_search(question, top_k=candidate_k),
            retrieval_source="dense",
        )

        if self.retrieval_mode == "dense":
            return dense[:TOP_K]

        from src.task6_lexical_search import lexical_search

        lexical = _normalize_chunks(
            lexical_search(question, top_k=candidate_k),
            retrieval_source="bm25",
        )
        return _rrf_fuse([dense, lexical], top_k=TOP_K)

    def generate_with_citation(self, question: str) -> dict:
        """Retrieve evidence and generate one grounded legal-information answer."""
        from openai import OpenAI

        chunks = _reorder_for_llm(self._retrieve(question))
        if not chunks:
            return {
                "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
                "sources": [],
                "retrieval_source": self.retrieval_mode,
            }

        api_key, base_url, generation_model, _, _ = _api_settings()
        client_options: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_options["base_url"] = base_url

        client = OpenAI(**client_options)
        context = _format_context(chunks)
        response = client.chat.completions.create(
            model=generation_model,
            messages=[
                {"role": "system", "content": LEGAL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\n---\n\nCâu hỏi: {question}",
                },
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

        answer = response.choices[0].message.content or ""
        return {
            "answer": answer.strip(),
            "sources": chunks,
            "retrieval_source": self.retrieval_mode,
        }


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: list[Any]) -> float | None:
    valid = [
        number
        for value in values
        if (number := _finite_float(value)) is not None
    ]
    return sum(valid) / len(valid) if valid else None


def _build_ragas_judge():
    """Create explicit judge and embedding clients for RAGAS."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    api_key, base_url, _, judge_model, embedding_model = _api_settings()
    options: dict[str, Any] = {"api_key": api_key}
    if base_url:
        options["base_url"] = base_url

    judge_llm = ChatOpenAI(model=judge_model, temperature=0, **options)
    judge_embeddings = OpenAIEmbeddings(model=embedding_model, **options)
    return judge_llm, judge_embeddings


def evaluate_with_ragas(
    rag_pipeline: EvaluationRAGPipeline,
    golden_dataset: list[dict],
) -> dict:
    """Generate answers and calculate the four required RAGAS metrics."""
    from datasets import Dataset
    # pyrefly: ignore [missing-import]
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for item in golden_dataset:
        generated = rag_pipeline.generate_with_citation(item["question"])
        contexts = [
            source["content"]
            for source in generated["sources"]
            if source.get("content")
        ]
        eval_data["question"].append(item["question"])
        eval_data["answer"].append(generated["answer"])
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append(item["expected_answer"])

    judge_llm, judge_embeddings = _build_ragas_judge()
    result = evaluate(
        Dataset.from_dict(eval_data),
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
    )
    frame = result.to_pandas()

    rows = []
    for record in frame.to_dict(orient="records"):
        rows.append(
            {
                "question": record.get("question", ""),
                "answer": record.get("answer", ""),
                **{
                    metric: _finite_float(record.get(metric))
                    for metric in METRIC_NAMES
                },
            }
        )

    summary = {
        metric: _mean([row[metric] for row in rows])
        for metric in METRIC_NAMES
    }
    return {"summary": summary, "rows": rows}


def compare_configs(golden_dataset: list[dict]) -> dict:
    """Evaluate both retrieval configurations on exactly the same questions."""
    comparison = {}
    for config_name, config in CONFIGS.items():
        print(
            f"Evaluating {config['label']} on "
            f"{len(golden_dataset)} legal questions..."
        )
        pipeline = EvaluationRAGPipeline(
            retrieval_mode=config["retrieval_mode"],
        )
        comparison[config_name] = {
            "label": config["label"],
            **evaluate_with_ragas(pipeline, golden_dataset),
        }
    return comparison


def _display_score(value: Any) -> str:
    number = _finite_float(value)
    return f"{number:.4f}" if number is not None else "N/A"


def _escape_markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _row_average(row: dict) -> float:
    value = _mean([row.get(metric) for metric in METRIC_NAMES])
    return value if value is not None else float("inf")


def _failure_diagnosis(row: dict) -> tuple[str, str]:
    valid_scores = {
        metric: score
        for metric in METRIC_NAMES
        if (score := _finite_float(row.get(metric))) is not None
    }
    if not valid_scores:
        return "Evaluation", "RAGAS không trả về điểm hợp lệ cho mẫu này."

    weakest = min(valid_scores, key=valid_scores.get)
    diagnoses = {
        "faithfulness": (
            "Generation",
            "Câu trả lời có thể suy diễn vượt quá chứng cứ hoặc trộn luật với chính sách Shopee.",
        ),
        "answer_relevancy": (
            "Generation",
            "Câu trả lời chưa tập trung trực tiếp vào vấn đề pháp lý được hỏi.",
        ),
        "context_recall": (
            "Retrieval/Corpus",
            "Retriever bỏ sót điều khoản cần thiết hoặc corpus còn thiếu nguồn Shopee/văn bản sửa đổi.",
        ),
        "context_precision": (
            "Retrieval/RRF",
            "Context chứa điều luật, phiên bản văn bản hoặc chính sách không liên quan.",
        ),
    }
    return diagnoses[weakest]


def export_results(comparison: dict) -> None:
    """Export real aggregate scores, bottom-3 cases and recommendations."""
    config_a = comparison["hybrid_rrf"]
    config_b = comparison["dense_only"]
    summary_a = config_a["summary"]
    summary_b = config_b["summary"]

    labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevance",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
    }

    lines = [
        "# RAGAS Evaluation — Trợ Lý Pháp Lý Khởi Nghiệp & Shopee",
        "",
        "## Trạng thái",
        "",
        "- Framework: **RAGAS**",
        f"- Golden dataset: **{len(config_a['rows'])} câu hỏi**",
        f"- Config A: **{config_a['label']}**",
        f"- Config B: **{config_b['label']}**",
        "- PageIndex: **không sử dụng**",
        "",
        "## Overall Scores",
        "",
        "| Metric | Config A | Config B | Δ (A - B) |",
        "|---|---:|---:|---:|",
    ]

    for metric, label in labels.items():
        score_a = _finite_float(summary_a.get(metric))
        score_b = _finite_float(summary_b.get(metric))
        delta = (
            score_a - score_b
            if score_a is not None and score_b is not None
            else None
        )
        lines.append(
            f"| {label} | {_display_score(score_a)} | "
            f"{_display_score(score_b)} | {_display_score(delta)} |"
        )

    average_a = _mean(list(summary_a.values()))
    average_b = _mean(list(summary_b.values()))
    average_delta = (
        average_a - average_b
        if average_a is not None and average_b is not None
        else None
    )
    lines.extend(
        [
            f"| **Average** | **{_display_score(average_a)}** | "
            f"**{_display_score(average_b)}** | "
            f"**{_display_score(average_delta)}** |",
            "",
            "## A/B Comparison Analysis",
            "",
        ]
    )

    if average_a is None or average_b is None:
        conclusion = "Chưa đủ điểm hợp lệ để kết luận cấu hình nào tốt hơn."
    elif average_a > average_b:
        conclusion = (
            "Config A có điểm trung bình cao hơn; BM25 và RRF cải thiện "
            "chất lượng tổng thể trên bộ câu hỏi pháp lý này."
        )
    elif average_a < average_b:
        conclusion = (
            "Config B có điểm trung bình cao hơn; cần kiểm tra nhiễu từ BM25, "
            "chunk văn bản pháp luật hoặc cách hợp nhất RRF của Config A."
        )
    else:
        conclusion = (
            "Hai cấu hình có cùng điểm trung bình; cần so sánh từng metric "
            "và từng câu hỏi trước khi chọn."
        )

    lines.extend(
        [
            f"**Kết luận:** {conclusion}",
            "",
            "## Worst Performers của Config A (Bottom 3)",
            "",
            "| # | Question | Faithfulness | Relevance | Recall | Precision | "
            "Failure Stage | Root Cause |",
            "|---:|---|---:|---:|---:|---:|---|---|",
        ]
    )

    for index, row in enumerate(
        sorted(config_a["rows"], key=_row_average)[:3],
        start=1,
    ):
        stage, cause = _failure_diagnosis(row)
        lines.append(
            "| {index} | {question} | {faithfulness} | {relevance} | "
            "{recall} | {precision} | {stage} | {cause} |".format(
                index=index,
                question=_escape_markdown(row.get("question", "")),
                faithfulness=_display_score(row.get("faithfulness")),
                relevance=_display_score(row.get("answer_relevancy")),
                recall=_display_score(row.get("context_recall")),
                precision=_display_score(row.get("context_precision")),
                stage=_escape_markdown(stage),
                cause=_escape_markdown(cause),
            )
        )

    lines.extend(
        [
            "",
            "## Khuyến nghị",
            "",
            "1. Bổ sung Điều khoản dịch vụ và Quy định đăng bán Shopee vào corpus, "
            "kèm URL và ngày thu thập.",
            "2. Lập version metadata cho Nghị định 52/2013/NĐ-CP và Luật Doanh "
            "nghiệp; ưu tiên bản hợp nhất hoặc văn bản sửa đổi còn hiệu lực.",
            "3. Giữ nguyên ranh giới Điều/Khoản khi chunk; gắn số văn bản, điều, "
            "khoản, trang và trạng thái hiệu lực vào metadata.",
            "4. Không trộn chính sách Shopee với quy phạm pháp luật trong cùng một "
            "kết luận nếu thiếu trích dẫn riêng cho từng nguồn.",
            "",
            "## Reproduction",
            "",
            "    python -m group_project.evaluation.eval_pipeline",
            "",
            "Biến tùy chọn: EVAL_GENERATION_MODEL, RAGAS_JUDGE_MODEL, "
            "RAGAS_EMBEDDING_MODEL.",
        ]
    )

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} golden test cases")
    comparison = compare_configs(golden_dataset)
    export_results(comparison)
    print(f"Evaluation report written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
