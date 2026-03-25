from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path("/home/mkgs/hackaton")
DEFAULT_QUESTIONS = ROOT / "starter_kit" / "submission_history" / "20260312_011749" / "questions.json"
DEFAULT_DEBUG = ROOT / "starter_kit" / "challenge_workdir" / "submission_debug.json"
DEFAULT_CHUNKED_DIR = ROOT / "starter_kit" / "challenge_workdir" / "docling" / "chunked"
DEFAULT_OUTPUT_DIR = ROOT / "starter_kit" / "challenge_workdir" / "manual_audit_current"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manual audit pack for a submission_debug snapshot.")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--snapshot-debug", default=str(DEFAULT_DEBUG))
    parser.add_argument("--chunked-dir", default=str(DEFAULT_CHUNKED_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=0, help="Optional batch size for large runs such as final-stage review.")
    parser.add_argument("--batch-index", type=int, default=0, help="0-based batch index when --batch-size is set.")
    return parser.parse_args()


def load_chunked_page_text(chunked_dir: Path, doc_id: str, page_number: int) -> str:
    path = chunked_dir / f"{doc_id}.json"
    if not path.exists():
        return ""
    payload = json.loads(path.read_text())
    pages = ((payload.get("content") or {}).get("pages") or [])
    if page_number < 1 or page_number > len(pages):
        return ""
    page = pages[page_number - 1]
    return (page.get("page_text") or page.get("text") or "").strip()


def summarize_page(text: str, limit: int = 1200) -> str:
    text = " ".join(text.split())
    return text[:limit]


def likely_issue_tags(row: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    raw_answer = (row.get("raw_answer") or "").strip()
    answer_type = row.get("answer_type")
    grounding_method = row.get("grounding_method") or ""
    refs = row.get("retrieval_refs") or []
    candidate_page_refs = row.get("candidate_page_refs") or []
    missing_terms = row.get("support_gate_missing_terms") or []
    timing = row.get("timing") or {}
    page_count = sum(len(ref.get("page_numbers") or []) for ref in refs)
    if answer_type == "free_text" and len(raw_answer) < 90:
        tags.append("free_text_short")
    if answer_type == "free_text" and len(raw_answer) > 260:
        tags.append("free_text_long")
    if answer_type == "free_text" and raw_answer.lower().startswith("there is no information"):
        tags.append("free_text_absence")
    if answer_type == "free_text" and "does not mention" in raw_answer.lower():
        tags.append("weak_absence_phrasing")
    if grounding_method == "citation_pages":
        tags.append("citation_grounding")
    if grounding_method.startswith("selector"):
        tags.append("selector_grounding")
    if page_count == 0:
        tags.append("no_refs")
    if answer_type == "free_text" and page_count == 1:
        tags.append("single_page_free_text")
    if answer_type == "free_text" and len(candidate_page_refs) > page_count:
        tags.append("narrowed_grounding")
    if missing_terms:
        tags.append("support_gate_missing_terms")
    if row.get("support_gate_abstained"):
        tags.append("support_gate_abstained")
    if timing.get("ttft_ms") is not None and float(timing["ttft_ms"]) > 5000:
        tags.append("slow_ttft")
    return tags


def main() -> None:
    args = parse_args()
    questions = json.loads(Path(args.questions).read_text())
    qmap = {str(item["id"]): item for item in questions}
    snapshot = json.loads(Path(args.snapshot_debug).read_text())
    chunked_dir = Path(args.chunked_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for item in snapshot:
        qid = str(item["question_id"])
        question = qmap.get(qid, {})
        refs = item.get("retrieval_refs") or []
        ref_details = []
        for ref in refs:
            doc_id = str(ref.get("doc_id"))
            for page_number in ref.get("page_numbers") or []:
                text = load_chunked_page_text(chunked_dir, doc_id, int(page_number))
                ref_details.append(
                    {
                        "page_ref": f"{doc_id}:{int(page_number)}",
                        "excerpt": summarize_page(text),
                    }
                )

        row = {
            "index": item.get("index"),
            "question_id": qid,
            "answer_type": item.get("answer_type") or question.get("answer_type"),
            "question": item.get("question") or question.get("question"),
            "raw_answer": item.get("raw_answer"),
            "normalized_answer": item.get("normalized_answer"),
            "submission_answer": item.get("submission_answer"),
            "grounding_method": item.get("grounding_method"),
            "strategy": item.get("strategy"),
            "solver_handled": item.get("solver_handled"),
            "retrieval_refs": refs,
            "candidate_page_refs": item.get("candidate_page_refs") or [],
            "ref_details": ref_details,
            "answer_chunk_titles": item.get("answer_chunk_titles") or [],
            "answer_chunk_refs": item.get("answer_chunk_refs") or [],
            "top_chunk_titles": item.get("top_chunk_titles") or [],
            "top_chunk_refs": item.get("top_chunk_refs") or [],
            "citations": item.get("citations") or [],
            "support_gate_missing_terms": item.get("support_gate_missing_terms") or [],
            "support_gate_abstained": bool(item.get("support_gate_abstained")),
            "timing": item.get("timing") or {},
            "error": item.get("error") or "",
        }
        row["likely_issue_tags"] = likely_issue_tags(row)
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda item: (item["index"] if item["index"] is not None else 9999))
    batch_suffix = ""
    if args.batch_size and args.batch_size > 0:
        batch_start = args.batch_index * args.batch_size
        batch_end = batch_start + args.batch_size
        rows_sorted = rows_sorted[batch_start:batch_end]
        batch_suffix = f"_batch_{args.batch_index:02d}"
    (output_dir / f"manual_audit_pack{batch_suffix}.json").write_text(json.dumps(rows_sorted, indent=2, ensure_ascii=False))

    with (output_dir / f"manual_audit_pack{batch_suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "question_id",
                "answer_type",
                "question",
                "raw_answer",
                "grounding_method",
                "strategy",
                "solver_handled",
                "candidate_page_refs",
                "support_gate_missing_terms",
                "support_gate_abstained",
                "ttft_ms",
                "error",
                "likely_issue_tags",
            ],
        )
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(
                {
                    "index": row["index"],
                    "question_id": row["question_id"],
                    "answer_type": row["answer_type"],
                    "question": row["question"],
                    "raw_answer": row["raw_answer"],
                    "grounding_method": row["grounding_method"],
                    "strategy": row["strategy"],
                    "solver_handled": row["solver_handled"],
                    "candidate_page_refs": ",".join(row["candidate_page_refs"]),
                    "support_gate_missing_terms": ",".join(row["support_gate_missing_terms"]),
                    "support_gate_abstained": row["support_gate_abstained"],
                    "ttft_ms": (row.get("timing") or {}).get("ttft_ms"),
                    "error": row.get("error") or "",
                    "likely_issue_tags": ",".join(row["likely_issue_tags"]),
                }
            )

    lines = [
        "# Manual Audit Pack",
        "",
        f"- total_questions: `{len(rows_sorted)}`",
        f"- free_text: `{sum(1 for row in rows_sorted if row['answer_type'] == 'free_text')}`",
        f"- with_issue_tags: `{sum(1 for row in rows_sorted if row['likely_issue_tags'])}`",
        "",
        "## Likely Issue Shortlist",
        "",
    ]
    shortlist = [
        row
        for row in rows_sorted
        if row["likely_issue_tags"]
    ]
    shortlist.sort(key=lambda item: (0 if item["answer_type"] == "free_text" else 1, item["index"]))
    for row in shortlist[:30]:
        lines.append(
            f"- Q{int(row['index']):03d} `{row['answer_type']}` `{row['question_id']}`: "
            f"{','.join(row['likely_issue_tags'])} | answer `{(row['raw_answer'] or '')[:140]}`"
        )
    lines.extend(
        [
            "",
            "## Usage",
            "",
            f"- Use `manual_audit_pack{batch_suffix}.json` to inspect page excerpts for each retrieval ref.",
            "- For large runs, generate multiple batches with `--batch-size` and `--batch-index`.",
        ]
    )
    (output_dir / f"manual_audit_pack{batch_suffix}.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
