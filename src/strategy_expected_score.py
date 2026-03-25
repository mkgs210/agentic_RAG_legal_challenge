from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence

import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.advanced_retrieval import AdvancedRetriever
from src.platform_submission import merge_routes
from src.production_doc_level_audit import (
    DOC_AUDIT_SUMMARY_PATH,
    QUESTIONS_PATH,
    evaluate_row,
    infer_expected_groups,
    unique_doc_rows,
)
from src.public_dataset_eval import PublicCorpus, safe_json_dump
from src.public_retrieval_benchmark import run_strategy
from src.query_analysis import QuestionAnalyzer


BASELINE_ACTUAL = {
    "total": 0.754,
    "det": 0.986,
    "asst": 0.673,
    "g": 0.905,
    "t": 0.996,
    "f": 0.938,
}
OUTPUT_DIR = Path("/home/mkgs/hackaton/starter_kit/challenge_workdir/strategy_expected_scores")
STRATEGIES = [
    "baseline",
    "citation_focus",
    "late_interaction",
    "multi_query_expansion",
    "corrective_retrieval",
    "evidence_first",
    "hybrid_mq_evidence",
    "hybrid_mq_late",
    "hybrid_prod_v1",
    "hybrid_prod_v2",
]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def load_baseline_doc_summary() -> Dict[str, Any]:
    rows = json.loads(DOC_AUDIT_SUMMARY_PATH.read_text(encoding="utf-8"))
    for row in rows:
        if row["variant"] == "baseline":
            return row
    raise ValueError("Baseline doc-level audit summary not found.")


def estimate_metrics(
    *,
    summary: Dict[str, Any],
    baseline_summary: Dict[str, Any],
    baseline_retrieval_ms: float,
) -> Dict[str, float]:
    weak_rate = summary["weak"] / summary["question_count"]
    baseline_weak_rate = baseline_summary["weak"] / baseline_summary["question_count"]
    precision_delta = summary["mean_precision_top5"] - baseline_summary["mean_precision_top5"]
    coverage3_delta = summary["mean_coverage_top3"] - baseline_summary["mean_coverage_top3"]
    coverage5_delta = summary["mean_coverage_top5"] - baseline_summary["mean_coverage_top5"]
    retrieval_ms = summary["mean_retrieval_ms"]

    det = BASELINE_ACTUAL["det"]
    det += 0.010 * precision_delta
    det += 0.010 * coverage3_delta
    det -= 0.12 * max(0.0, weak_rate - baseline_weak_rate)
    det = clamp(det, 0.94, 0.995)

    asst = BASELINE_ACTUAL["asst"]
    asst += 0.14 * precision_delta
    asst += 0.06 * coverage3_delta
    asst += 0.03 * coverage5_delta
    asst -= 0.08 * max(0.0, weak_rate - baseline_weak_rate)
    asst = clamp(asst, 0.60, 0.80)

    g = BASELINE_ACTUAL["g"]
    g += 0.18 * precision_delta
    g += 0.10 * coverage3_delta
    g += 0.60 * coverage5_delta
    g -= 0.18 * max(0.0, weak_rate - baseline_weak_rate)
    g = clamp(g, 0.82, 0.97)

    t = BASELINE_ACTUAL["t"]

    retrieval_delta_ms = retrieval_ms - baseline_retrieval_ms
    f = BASELINE_ACTUAL["f"]
    if retrieval_delta_ms >= 0:
        f -= 0.00008 * retrieval_delta_ms
    else:
        f += 0.00003 * abs(retrieval_delta_ms)
    f = clamp(f, 0.88, 0.97)

    total = (0.7 * det + 0.3 * asst) * g * t * f
    return {
        "det": round(det, 3),
        "asst": round(asst, 3),
        "g": round(g, 3),
        "t": round(t, 3),
        "f": round(f, 3),
        "total": round(total, 3),
    }


def evaluate_strategy(
    *,
    strategy: str,
    corpus: PublicCorpus,
    questions: Sequence[Dict[str, Any]],
    analyzer: QuestionAnalyzer,
    advanced_retriever: AdvancedRetriever,
    expectations: Dict[int, Any],
) -> Dict[str, Any]:
    rows = []
    retrieval_ms_values: List[float] = []
    status_counts = {"sufficient": 0, "weak": 0, "miss": 0}
    precision_values: List[float] = []
    coverage3_values: List[float] = []
    coverage5_values: List[float] = []
    top1_values: List[float] = []

    for index, item in enumerate(questions, start=1):
        analysis = analyzer.analyze(item["question"], item["answer_type"])
        effective_question, route = merge_routes(corpus, item["question"], analysis)
        started = time.perf_counter()
        if strategy == "baseline":
            reranked = run_strategy(corpus, corpus.bm25_index, effective_question, route["candidate_shas"], "dense_doc_diverse")
        else:
            reranked = advanced_retriever.retrieve(
                strategy=strategy,
                question=effective_question,
                analysis=analysis,
                candidate_shas=route["candidate_shas"],
            )
        retrieval_ms_values.append((time.perf_counter() - started) * 1000.0)

        top_docs = unique_doc_rows(reranked[:5], limit=5)
        evaluated = evaluate_row(expectation=expectations[index], top_rows=top_docs)
        status_counts[evaluated["status"]] += 1
        precision_values.append(evaluated["precision_top5"])
        coverage5_values.append(evaluated["coverage_ratio"])
        coverage3_values.append(evaluate_row(expectation=expectations[index], top_rows=top_docs[:3])["coverage_ratio"])
        top1_values.append(1.0 if evaluated["top1_on_target"] else 0.0)
        rows.append(
            {
                "index": index,
                "question": item["question"],
                "answer_type": item["answer_type"],
                "method": strategy,
                "top_chunks": top_docs,
                "top_titles": [chunk["title"] for chunk in top_docs],
                "doc_status": evaluated["status"],
                "doc_note": evaluated["note"],
            }
        )

    return {
        "variant": strategy,
        "question_count": len(rows),
        "sufficient": status_counts["sufficient"],
        "weak": status_counts["weak"],
        "miss": status_counts["miss"],
        "top1_on_target_rate": round(mean(top1_values), 3),
        "mean_precision_top5": round(mean(precision_values), 3),
        "mean_coverage_top3": round(mean(coverage3_values), 3),
        "mean_coverage_top5": round(mean(coverage5_values), 3),
        "mean_retrieval_ms": round(mean(retrieval_ms_values), 1),
        "rows": rows,
    }


def write_report(rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    lines = [
        "# Expected Final Score By Strategy",
        "",
        "Anchor used for estimation: warm-up platform submit `Total=0.754 / Det=0.986 / Asst=0.673 / G=0.905 / T=0.996 / F=0.938`.",
        "The estimates below are not platform scores; they are modeled from local doc-level coverage/precision deltas and retrieval latency deltas against that anchor.",
        "",
        "| Strategy | Suff | Weak | Miss | Precision@5 | Coverage@5 | Retrieval ms | Est Det | Est Asst | Est G | Est F | Est Total |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ranked = sorted(rows, key=lambda row: row["estimated"]["total"], reverse=True)
    for row in ranked:
        est = row["estimated"]
        lines.append(
            f"| {row['variant']} | {row['sufficient']} | {row['weak']} | {row['miss']} | "
            f"{row['mean_precision_top5']:.3f} | {row['mean_coverage_top5']:.3f} | {row['mean_retrieval_ms']:.1f} | "
            f"{est['det']:.3f} | {est['asst']:.3f} | {est['g']:.3f} | {est['f']:.3f} | {est['total']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `hybrid_prod_v1` is the strongest prod-safe composite if it preserves full doc coverage without blowing up retrieval latency.",
            "- `multi_query_expansion` is the safest single change: good coverage lift with limited complexity.",
            "- `late_interaction` is useful, but it can become latency-heavy if left unbounded.",
            "- `evidence_first` improves noise in answer context, but by itself is not a retrieval recall fix.",
            "- `corrective_retrieval` is valuable as a fallback when the initial top-5 misses one side of a comparison.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate final warm-up score for prod-safe retrieval strategies.")
    parser.add_argument("--work-dir", type=Path, default=Path("/home/mkgs/hackaton/starter_kit/challenge_workdir"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("/home/mkgs/hackaton/starter_kit/docs_corpus"))
    parser.add_argument("--chunked-dir", type=Path, default=Path("/home/mkgs/hackaton/starter_kit/challenge_workdir/docling/chunked"))
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--strategies", nargs="*", default=STRATEGIES)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus = PublicCorpus(work_dir=args.work_dir, pdf_dir=args.pdf_dir, chunked_dir=args.chunked_dir)
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    expectations = infer_expected_groups(corpus=corpus, questions=questions)
    analyzer = QuestionAnalyzer(
        provider=args.provider,
        model=args.model,
        cache_path=args.work_dir / "query_analysis" / f"{args.provider}__{args.model.replace('/', '_')}.json",
        corpus=corpus,
    )
    advanced_retriever = AdvancedRetriever(corpus=corpus, work_dir=args.work_dir / "advanced_retrieval")

    summaries: List[Dict[str, Any]] = []
    for strategy in args.strategies:
        summary = evaluate_strategy(
            strategy=strategy,
            corpus=corpus,
            questions=questions,
            analyzer=analyzer,
            advanced_retriever=advanced_retriever,
            expectations=expectations,
        )
        pack_path = OUTPUT_DIR / f"review_pack_{strategy}.json"
        safe_json_dump(pack_path, {"questions": summary["rows"], "summary": {k: v for k, v in summary.items() if k != "rows"}})
        csv_path = OUTPUT_DIR / f"doc_status_{strategy}.json"
        safe_json_dump(csv_path, summary)
        summaries.append(summary)
        safe_json_dump(OUTPUT_DIR / "expected_score_summary.partial.json", summaries)
        print(f"finished:{strategy}", flush=True)

    baseline_summary = load_baseline_doc_summary()
    baseline_runtime_ms = next(item["mean_retrieval_ms"] for item in summaries if item["variant"] == "baseline")
    for summary in summaries:
        summary["estimated"] = estimate_metrics(
            summary=summary,
            baseline_summary=baseline_summary,
            baseline_retrieval_ms=baseline_runtime_ms,
        )

    safe_json_dump(OUTPUT_DIR / "expected_score_summary.json", summaries)
    write_report(summaries, OUTPUT_DIR / "expected_score_report.md")
    print(OUTPUT_DIR / "expected_score_report.md")
    print(OUTPUT_DIR / "expected_score_summary.json")


if __name__ == "__main__":
    main()
