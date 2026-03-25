from __future__ import annotations

import argparse
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from rank_bm25 import BM25Okapi

from src.api_requests import APIProcessor
from src.build_public_review_report import parse_manual_audit
from src.lexical_retrieval import tokenize_for_bm25
from src.public_dataset_eval import (
    PublicCorpus,
    answer_match,
    answer_question,
    is_rate_limit_error,
    normalize_answer,
    safe_json_dump,
    safe_json_load,
)
from src.public_retrieval_benchmark import run_strategy


STRUCTURED_TYPES = {"boolean", "number", "name", "names", "date"}


def normalize_manual_gold(answer_type: str, raw_value: str) -> Any:
    raw_text = (raw_value or "").strip()
    code_spans = [part.strip() for part in re.findall(r"`([^`]+)`", raw_text) if part.strip()]
    text = raw_text.replace("`", "").strip()
    if code_spans:
        if answer_type == "names":
            text = " | ".join(code_spans)
        else:
            text = code_spans[0]
    if answer_type == "boolean":
        lowered = text.lower()
        if "true" in lowered:
            return True
        if "false" in lowered:
            return False
    if answer_type in {"name", "names", "date"}:
        text = text.split(";", 1)[0].strip()
    return normalize_answer(answer_type, text)


def evaluate_structured_questions(
    dataset_path: Path,
    audit_path: Path,
    pdf_dir: Path,
    chunked_dir: Path,
    work_dir: Path,
    provider: str,
    model: str,
    strategy: str,
    top_k_chunks: int,
    limit: int | None,
) -> Dict[str, Any]:
    corpus = PublicCorpus(work_dir, pdf_dir, chunked_dir)
    bm25_index = BM25Okapi([tokenize_for_bm25(chunk.text) for chunk in corpus.chunks])
    audit_rows = parse_manual_audit(audit_path)
    questions = safe_json_load(dataset_path)
    api = APIProcessor(provider=provider)

    results: List[Dict[str, Any]] = []
    type_totals: Counter[str] = Counter()
    type_correct: Counter[str] = Counter()
    error_count = 0

    for index, item in enumerate(questions, start=1):
        answer_type = item["answer_type"]
        if answer_type not in STRUCTURED_TYPES:
            continue

        manual_row = audit_rows.get(index)
        if not manual_row or not manual_row.get("expected_answer"):
            continue

        question = item["question"]
        gold_raw = manual_row["expected_answer"]
        gold_norm = normalize_manual_gold(answer_type, gold_raw)
        route = corpus.route_question(question, expansive=False)
        reranked = run_strategy(corpus, bm25_index, question, route["candidate_shas"], strategy)
        selected_chunks = reranked[:top_k_chunks]
        last_error: Exception | None = None
        predicted = None
        for attempt in range(3):
            try:
                predicted = answer_question(
                    api=api,
                    model=model,
                    question=question,
                    answer_type=answer_type,
                    chunks=selected_chunks,
                )
                break
            except Exception as err:
                last_error = err
                if is_rate_limit_error(err) and attempt < 2:
                    time.sleep(10 * (attempt + 1))
                    continue
                break
        if predicted is None:
            error_count += 1
            results.append(
                {
                    "index": index,
                    "id": item["id"],
                    "question": question,
                    "answer_type": answer_type,
                    "expected_answer": gold_raw,
                    "expected_normalized": gold_norm,
                    "predicted_answer": "",
                    "predicted_normalized": None,
                    "correct": False,
                    "error": str(last_error),
                    "citations": [],
                    "reasoning": "",
                    "top_chunk_refs": [chunk["ref"] for chunk in selected_chunks],
                    "top_chunk_titles": [chunk["title"] for chunk in selected_chunks],
                }
            )
            if limit is not None and len(results) >= limit:
                break
            continue

        pred_norm = predicted["normalized_answer"]
        correct = answer_match(answer_type, gold_norm, pred_norm)

        type_totals[answer_type] += 1
        if correct:
            type_correct[answer_type] += 1

        results.append(
            {
                "index": index,
                "id": item["id"],
                "question": question,
                "answer_type": answer_type,
                "expected_answer": gold_raw,
                "expected_normalized": gold_norm,
                "predicted_answer": predicted["raw_answer"],
                "predicted_normalized": pred_norm,
                "correct": correct,
                "citations": predicted.get("citations", []),
                "reasoning": predicted.get("reasoning", ""),
                "top_chunk_refs": [chunk["ref"] for chunk in selected_chunks],
                "top_chunk_titles": [chunk["title"] for chunk in selected_chunks],
            }
        )
        if limit is not None and len(results) >= limit:
            break

    summary = summarize_results(
        results=results,
        provider=provider,
        model=model,
        strategy=strategy,
        top_k_chunks=top_k_chunks,
    )
    summary["error_count"] = error_count
    return {"summary": summary, "results": results}


def build_markdown_report(payload: Dict[str, Any]) -> str:
    summary = payload["summary"]
    results = payload["results"]
    incorrect_rows = [row for row in results if not row["correct"]]

    lines = [
        "# Structured API Eval",
        "",
        f"- Provider: `{summary['provider']}`",
        f"- Model: `{summary['model']}`",
        f"- Retrieval strategy: `{summary['strategy']}`",
        f"- Context size: top `{summary['top_k_chunks']}` reranked chunks",
        f"- Structured questions evaluated: `{summary['question_count']}`",
        f"- Exact-match accuracy: `{summary['accuracy']:.4f}` ({summary['correct_count']}/{summary['question_count']})",
        "",
        "## By Answer Type",
        "",
    ]

    for answer_type, stats in summary["by_answer_type"].items():
        lines.append(
            f"- `{answer_type}`: `{stats['accuracy']:.4f}` ({stats['correct']}/{stats['count']})"
        )

    lines.extend(["", "## Incorrect Cases", ""])

    if not incorrect_rows:
        lines.append("- None.")
    else:
        for row in incorrect_rows:
            lines.extend(
                [
                    f"### Q{row['index']:03d}",
                    "",
                    f"- Question: {row['question']}",
                    f"- Type: `{row['answer_type']}`",
                    f"- Expected: `{row['expected_answer']}`",
                    f"- Predicted: `{row['predicted_answer']}`",
                    f"- Error: `{row.get('error', '')}`" if row.get("error") else "- Error: none",
                    f"- Top chunk titles: {' | '.join(row['top_chunk_titles'])}",
                    f"- Citations: {', '.join(row['citations']) or 'none'}",
                    "",
                ]
            )

    return "\n".join(lines)


def summarize_results(results: Sequence[Dict[str, Any]], provider: str, model: str, strategy: str, top_k_chunks: int) -> Dict[str, Any]:
    type_totals: Counter[str] = Counter()
    type_correct: Counter[str] = Counter()
    error_count = 0
    for row in results:
        answer_type = row["answer_type"]
        type_totals[answer_type] += 1
        if row.get("correct"):
            type_correct[answer_type] += 1
        if row.get("error"):
            error_count += 1
    total = len(results)
    correct_total = sum(1 for item in results if item.get("correct"))
    return {
        "provider": provider,
        "model": model,
        "strategy": strategy,
        "top_k_chunks": top_k_chunks,
        "question_count": total,
        "correct_count": correct_total,
        "error_count": error_count,
        "accuracy": round(correct_total / max(1, total), 4),
        "by_answer_type": {
            answer_type: {
                "count": type_totals[answer_type],
                "correct": type_correct[answer_type],
                "accuracy": round(type_correct[answer_type] / max(1, type_totals[answer_type]), 4),
            }
            for answer_type in sorted(type_totals)
        },
    }


def rescore_saved_payload(path: Path) -> Dict[str, Any]:
    payload = safe_json_load(path)
    results = payload["results"]
    for row in results:
        answer_type = row["answer_type"]
        row["expected_normalized"] = normalize_manual_gold(answer_type, row.get("expected_answer", ""))
        row["predicted_normalized"] = normalize_answer(answer_type, row.get("predicted_answer", ""))
        row["correct"] = answer_match(answer_type, row["expected_normalized"], row["predicted_normalized"])
    payload["summary"] = summarize_results(
        results=results,
        provider=payload["summary"]["provider"],
        model=payload["summary"]["model"],
        strategy=payload["summary"]["strategy"],
        top_k_chunks=payload["summary"]["top_k_chunks"],
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an API LLM on structured public_dataset questions.")
    parser.add_argument("--dataset", default="public_dataset.json")
    parser.add_argument("--audit", default="artifacts/public_review/manual_answers_and_pre_rerank_audit.md")
    parser.add_argument("--pdf-dir", default="pdfs")
    parser.add_argument("--chunked-dir", default="artifacts/public_eval/docling/chunked")
    parser.add_argument("--work-dir", default="artifacts/public_eval")
    parser.add_argument("--provider", default="cohere")
    parser.add_argument("--model", default="command-a-03-2025")
    parser.add_argument("--strategy", default="dense_doc_diverse")
    parser.add_argument("--top-k-chunks", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rescore-json", default="")
    parser.add_argument("--json-out", default="artifacts/public_review/cohere_structured_eval.json")
    parser.add_argument("--md-out", default="artifacts/public_review/cohere_structured_eval.md")
    args = parser.parse_args()

    if args.rescore_json:
        payload = rescore_saved_payload(Path(args.rescore_json))
    else:
        payload = evaluate_structured_questions(
            dataset_path=Path(args.dataset),
            audit_path=Path(args.audit),
            pdf_dir=Path(args.pdf_dir),
            chunked_dir=Path(args.chunked_dir),
            work_dir=Path(args.work_dir),
            provider=args.provider,
            model=args.model,
            strategy=args.strategy,
            top_k_chunks=args.top_k_chunks,
            limit=args.limit or None,
        )
    safe_json_dump(Path(args.json_out), payload)
    Path(args.md_out).write_text(build_markdown_report(payload), encoding="utf-8")
    print(payload["summary"])


if __name__ == "__main__":
    main()
