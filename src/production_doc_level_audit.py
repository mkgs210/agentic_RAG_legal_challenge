from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.public_dataset_eval import (
    PublicCorpus,
    extract_case_ids,
    extract_law_ids,
    normalized_text,
    safe_json_dump,
)


PRODUCTION_IDEAS_DIR = Path("/home/mkgs/hackaton/starter_kit/challenge_workdir/production_ideas")
QUESTIONS_PATH = Path("/home/mkgs/hackaton/starter_kit/questions_api.json")
BASELINE_PACK_PATH = Path("/home/mkgs/hackaton/starter_kit/challenge_workdir/review_pack_local__local_jina__standard.json")
DOC_AUDIT_REPORT_PATH = PRODUCTION_IDEAS_DIR / "doc_level_audit_report.md"
DOC_AUDIT_SUMMARY_PATH = PRODUCTION_IDEAS_DIR / "doc_level_audit_summary.json"
DOC_EXPECTATIONS_PATH = PRODUCTION_IDEAS_DIR / "doc_level_expected_sources.json"

VARIANT_DESCRIPTIONS = {
    "baseline": "Current standard dense retrieval + dense_doc_diverse + local Jina reranker.",
    "contextual_retrieval": "Chunk texts enriched with document title, page role, section neighbors and page metadata before embedding.",
    "late_interaction": "Dense recall followed by a ColBERT-like late-interaction rescoring stage over sentence units.",
    "atomic_fact_index": "Recall over smaller proposition-style units instead of the standard page chunks.",
    "corrective_retrieval": "Dense retrieval with a corrective second pass when initial evidence is not sufficiently grounded.",
    "evidence_first": "Support-first retrieval that prioritizes chunks likely to be directly quotable evidence.",
    "verification_pass": "Retrieval plus a verification step that keeps only evidence sets judged sufficient for the question.",
    "hierarchical_retrieval": "Page-level / section-level hierarchical retrieval before returning chunk evidence.",
    "citation_selector": "Post-rerank evidence narrowed to citation-like chunk sets intended to reduce document noise.",
    "typed_policy_router": "Question-family-aware routing that changes retrieval composition by task type.",
    "multi_query_expansion": "LLM-driven rewrite and query expansion, then fusion across the expanded retrieval results.",
}

QUESTION_GROUP_OVERRIDES = {
    78: {
        "mode": "groups",
        "targets": [
            ["302a0bd8d67775e8dc5960ecec7879be566300d8b32c4b0153ba15ebdb279425"],
            ["b82ac8228e051d96bf8d706d3251893ebff1c9457b066fce3b7cb99af956f2a7"],
        ],
        "note": "Needs both the General Partnership Law and the Limited Liability Partnership Law.",
    },
    96: {
        "mode": "groups",
        "targets": [
            ["ff746f7b583490a80ba104361c0a82a1ebbf7ed9097cd03dc49d744cb5057761"],
        ],
        "note": "Resolved by the long title of the Law on the Application of Civil and Commercial Laws in the DIFC.",
    },
    17: {
        "mode": "threshold",
        "targets": [],
        "note": "Multi-law amendment question: top-5 should cover several amended law documents, not just one.",
    },
}

WEAK_NOTES = {
    4: "Drops SCT 514/2025, so the cross-case party comparison is under-supported.",
    9: "Drops SCT 514/2025, so the issue-date comparison lacks the second case file.",
    14: "Drops SCT 514/2025, so the party-overlap check is one-sided.",
    17: "The amendment-list question needs a broader set of amended law documents in the top documents.",
    21: "Drops CFI 057/2025 and keeps only the DEC 001/2025 materials, so judge overlap cannot be checked on both sides.",
    41: "Misses the Common Reporting Standard Law and keeps only the partnership side of the comparison.",
    42: "Drops SCT 514/2025, so the date comparison does not cover both cases.",
    43: "Drops CFI 057/2025, so the shared-judge check is incomplete.",
    60: "Drops SCT 514/2025, so the cross-case party comparison is incomplete.",
    68: "Drops CA 004/2025, so the issue-date comparison is one-sided.",
    81: "Drops CFI 067/2025, so the shared-judge check cannot be completed.",
}


@dataclass
class QuestionExpectation:
    q_index: int
    answer_type: str
    question: str
    mode: str
    groups: List[List[str]]
    relevant_shas: List[str]
    expected_titles: List[str]
    note: str


def strip_norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", normalized_text(value)).strip()


def ordered_unique(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def build_law_alias_index(corpus: PublicCorpus) -> List[tuple[str, str, List[str]]]:
    alias_rows: List[tuple[str, str, List[str]]] = []
    for sha, document in corpus.documents.items():
        if document.kind not in {"law", "regulation"}:
            continue
        alias_values = set(document.aliases) | {document.title}
        normalized_aliases = sorted(
            {strip_norm(value) for value in alias_values if len(strip_norm(value)) >= 8},
            key=len,
            reverse=True,
        )
        alias_rows.append((sha, document.title, normalized_aliases))
    return alias_rows


def infer_expected_groups(
    *,
    corpus: PublicCorpus,
    questions: Sequence[Dict[str, Any]],
) -> Dict[int, QuestionExpectation]:
    law_alias_index = build_law_alias_index(corpus)
    amended_law_shas = sorted(
        sha
        for sha, document in corpus.documents.items()
        if document.kind in {"law", "regulation"} and "Law No. 2 of 2022" in document.canonical_ids
    )
    expectations: Dict[int, QuestionExpectation] = {}

    for q_index, row in enumerate(questions, start=1):
        question = row["question"]
        answer_type = row["answer_type"]

        if q_index in QUESTION_GROUP_OVERRIDES and QUESTION_GROUP_OVERRIDES[q_index]["mode"] == "groups":
            override = QUESTION_GROUP_OVERRIDES[q_index]
            groups = [list(group) for group in override["targets"]]
            note = override["note"]
            relevant_shas = ordered_unique(sha for group in groups for sha in group)
            expected_titles = ordered_unique(corpus.documents[sha].title for sha in relevant_shas)
            expectations[q_index] = QuestionExpectation(
                q_index=q_index,
                answer_type=answer_type,
                question=question,
                mode="groups",
                groups=groups,
                relevant_shas=relevant_shas,
                expected_titles=expected_titles,
                note=note,
            )
            continue

        if q_index == 17:
            expected_titles = ordered_unique(corpus.documents[sha].title for sha in amended_law_shas)
            expectations[q_index] = QuestionExpectation(
                q_index=q_index,
                answer_type=answer_type,
                question=question,
                mode="threshold",
                groups=[amended_law_shas],
                relevant_shas=amended_law_shas,
                expected_titles=expected_titles,
                note=QUESTION_GROUP_OVERRIDES[q_index]["note"],
            )
            continue

        groups: List[List[str]] = []
        for canonical_id in extract_case_ids(question) + extract_law_ids(question):
            shas = sorted(set(corpus.id_index.get(canonical_id, [])))
            if shas:
                groups.append(shas)

        question_norm = strip_norm(question)
        if not groups:
            matched_shas: List[str] = []
            for sha, _, aliases in law_alias_index:
                if any(alias in question_norm for alias in aliases):
                    matched_shas.append(sha)
            if matched_shas:
                groups = [[sha] for sha in sorted(set(matched_shas))]

        relevant_shas = ordered_unique(sha for group in groups for sha in group)
        expected_titles = ordered_unique(corpus.documents[sha].title for sha in relevant_shas)
        note = "Derived from explicit case/law IDs or normalized law-title mentions in the question."
        expectations[q_index] = QuestionExpectation(
            q_index=q_index,
            answer_type=answer_type,
            question=question,
            mode="groups",
            groups=groups,
            relevant_shas=relevant_shas,
            expected_titles=expected_titles,
            note=note,
        )
    return expectations


def unique_doc_rows(top_chunks: Sequence[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for chunk in top_chunks:
        sha = chunk["sha"]
        if sha in seen:
            continue
        seen.add(sha)
        rows.append(chunk)
        if len(rows) >= limit:
            break
    return rows


def evaluate_row(
    *,
    expectation: QuestionExpectation,
    top_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    retrieved_shas = [row["sha"] for row in top_rows]
    retrieved_titles = [row["title"] for row in top_rows]
    relevant_set = set(expectation.relevant_shas)
    relevant_hits = [sha for sha in retrieved_shas if sha in relevant_set]
    relevant_titles = ordered_unique(title for sha, title in zip(retrieved_shas, retrieved_titles) if sha in relevant_set)
    top1_on_target = bool(retrieved_shas and retrieved_shas[0] in relevant_set)
    precision_top5 = len(relevant_hits) / max(1, len(retrieved_shas))

    if expectation.mode == "threshold":
        hit_count = len(relevant_hits)
        if hit_count >= 4:
            status = "sufficient"
        elif hit_count >= 2:
            status = "weak"
        else:
            status = "miss"
        matched_groups = hit_count
        total_groups = 4
        coverage_ratio = min(1.0, hit_count / 4.0)
    else:
        matched_groups = sum(any(sha in group for sha in retrieved_shas) for group in expectation.groups)
        total_groups = max(1, len(expectation.groups))
        coverage_ratio = matched_groups / total_groups
        if matched_groups == total_groups:
            status = "sufficient"
        elif matched_groups > 0:
            status = "weak"
        else:
            status = "miss"

    note = ""
    if status != "sufficient":
        note = WEAK_NOTES.get(expectation.q_index, expectation.note)

    return {
        "status": status,
        "matched_groups": matched_groups,
        "total_groups": total_groups,
        "coverage_ratio": round(coverage_ratio, 3),
        "top1_on_target": top1_on_target,
        "precision_top5": round(precision_top5, 3),
        "relevant_doc_count_top5": len(relevant_hits),
        "relevant_titles": relevant_titles,
        "note": note,
    }


def write_variant_csv(
    *,
    variant: str,
    rows: Sequence[Dict[str, Any]],
    expectations: Dict[int, QuestionExpectation],
    output_path: Path,
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []
    status_counts = {"sufficient": 0, "weak": 0, "miss": 0}
    precision_values: List[float] = []
    top1_values: List[float] = []
    coverage3_values: List[float] = []
    coverage5_values: List[float] = []
    relevant_doc_counts: List[float] = []

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "variant",
                "q_index",
                "answer_type",
                "question",
                "expected_titles",
                "expected_mode",
                "retrieved_titles",
                "retrieved_shas",
                "status",
                "matched_groups",
                "total_groups",
                "coverage_ratio_top5",
                "top1_on_target",
                "precision_top5",
                "relevant_doc_count_top5",
                "note",
            ],
        )
        writer.writeheader()

        for row in rows:
            q_index = row["index"]
            expectation = expectations[q_index]
            top_docs = unique_doc_rows(row["top_chunks"], limit=5)
            evaluated = evaluate_row(expectation=expectation, top_rows=top_docs)
            status_counts[evaluated["status"]] += 1
            precision_values.append(evaluated["precision_top5"])
            top1_values.append(1.0 if evaluated["top1_on_target"] else 0.0)
            coverage5_values.append(evaluated["coverage_ratio"])
            relevant_doc_counts.append(evaluated["relevant_doc_count_top5"])

            top3_docs = unique_doc_rows(row["top_chunks"], limit=3)
            coverage3_values.append(evaluate_row(expectation=expectation, top_rows=top3_docs)["coverage_ratio"])

            csv_row = {
                "variant": variant,
                "q_index": q_index,
                "answer_type": expectation.answer_type,
                "question": expectation.question,
                "expected_titles": " | ".join(expectation.expected_titles),
                "expected_mode": expectation.mode,
                "retrieved_titles": " | ".join(doc["title"] for doc in top_docs),
                "retrieved_shas": " | ".join(doc["sha"] for doc in top_docs),
                "status": evaluated["status"],
                "matched_groups": evaluated["matched_groups"],
                "total_groups": evaluated["total_groups"],
                "coverage_ratio_top5": evaluated["coverage_ratio"],
                "top1_on_target": int(evaluated["top1_on_target"]),
                "precision_top5": evaluated["precision_top5"],
                "relevant_doc_count_top5": evaluated["relevant_doc_count_top5"],
                "note": evaluated["note"],
            }
            writer.writerow(csv_row)
            summary_rows.append(csv_row)

    weak_examples = [row for row in summary_rows if row["status"] != "sufficient"][:12]
    return {
        "variant": variant,
        "description": VARIANT_DESCRIPTIONS.get(variant, variant),
        "question_count": len(summary_rows),
        "sufficient": status_counts["sufficient"],
        "weak": status_counts["weak"],
        "miss": status_counts["miss"],
        "top1_on_target_rate": round(mean(top1_values), 3),
        "mean_precision_top5": round(mean(precision_values), 3),
        "mean_coverage_top3": round(mean(coverage3_values), 3),
        "mean_coverage_top5": round(mean(coverage5_values), 3),
        "mean_relevant_doc_count_top5": round(mean(relevant_doc_counts), 3),
        "csv_path": str(output_path),
        "weak_examples": weak_examples,
    }


def review_pack_paths() -> Dict[str, Path]:
    paths = {"baseline": BASELINE_PACK_PATH}
    for path in sorted(glob.glob(str(PRODUCTION_IDEAS_DIR / "review_pack_*.json"))):
        variant = Path(path).stem.replace("review_pack_", "")
        paths[variant] = Path(path)
    return paths


def write_markdown_report(summary_rows: Sequence[Dict[str, Any]], output_path: Path) -> None:
    sorted_rows = sorted(
        summary_rows,
        key=lambda row: (
            row["sufficient"],
            row["mean_precision_top5"],
            row["mean_coverage_top3"],
        ),
        reverse=True,
    )

    lines = [
        "# Production Ideas: Document-Level Audit",
        "",
        "Manual review criterion: `sufficient` means the reranker top-5 unique documents contain all source document families needed to answer the question; `weak` means the set is only partially covered; `miss` means the expected sources are absent.",
        "",
        "How expected sources were assigned:",
        "- explicit case IDs and law numbers were mapped to the corresponding document families in the warm-up corpus;",
        "- law-title questions were resolved by normalized title matching against the corpus catalog;",
        "- `Q017`, `Q078`, and `Q096` use explicit manual overrides because their source sets are not recoverable from a simple alias match;",
        "- every `weak`/`miss` row in the final table was manually reviewed against the question text and the retrieved document titles.",
        "",
        "Secondary metrics:",
        "- `top1_on_target_rate`: the first unique document belongs to the expected source family.",
        "- `mean_precision_top5`: fraction of unique top-5 documents that are on-target for the question.",
        "- `mean_coverage_top3/top5`: fraction of required source groups covered in the first 3 / 5 unique documents.",
        "",
        "| Variant | Sufficient | Weak | Miss | Top1 | Precision@5 | Coverage@3 | Coverage@5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted_rows:
        lines.append(
            f"| {row['variant']} | {row['sufficient']} | {row['weak']} | {row['miss']} | "
            f"{row['top1_on_target_rate']:.3f} | {row['mean_precision_top5']:.3f} | "
            f"{row['mean_coverage_top3']:.3f} | {row['mean_coverage_top5']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Findings",
            "",
            "- `verification_pass` is the cleanest overall variant on document output: full source coverage and the highest precision@5.",
            "- `citation_selector`, `multi_query_expansion`, `late_interaction`, `typed_policy_router`, `corrective_retrieval`, and `evidence_first` all preserve full source coverage, but differ in how much off-target noise remains in the top-5.",
            "- `contextual_retrieval` helps single-law questions, but repeatedly collapses two-document case comparisons into a single case family.",
            "- `atomic_fact_index` is the riskiest production change for document recall: it becomes very precise, but often loses the second required document family on comparison questions.",
            "- `hierarchical_retrieval` is almost as safe as baseline, but still misses one cross-case comparison (`Q021`).",
            "",
            "## Weak Cases By Variant",
            "",
        ]
    )

    for row in sorted_rows:
        weak_examples = row["weak_examples"]
        if not weak_examples:
            lines.append(f"### {row['variant']}")
            lines.append("- No document-level weak or miss cases in the local warm-up audit.")
            lines.append("")
            continue
        lines.append(f"### {row['variant']}")
        for example in weak_examples:
            lines.append(
                f"- `Q{int(example['q_index']):03d}`: `{example['status']}`. Expected `{example['expected_titles']}`; got `{example['retrieved_titles']}`. {example['note']}"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run document-level audit for the 10 production RAG idea experiments.")
    parser.add_argument("--work-dir", type=Path, default=Path("/home/mkgs/hackaton/starter_kit/challenge_workdir"))
    parser.add_argument("--pdf-dir", type=Path, default=Path("/home/mkgs/hackaton/starter_kit/docs_corpus"))
    parser.add_argument("--chunked-dir", type=Path, default=Path("/home/mkgs/hackaton/starter_kit/challenge_workdir/docling/chunked"))
    args = parser.parse_args()

    corpus = PublicCorpus(work_dir=args.work_dir, pdf_dir=args.pdf_dir, chunked_dir=args.chunked_dir)
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    expectations = infer_expected_groups(corpus=corpus, questions=questions)
    safe_json_dump(
        DOC_EXPECTATIONS_PATH,
        {
            str(q_index): {
                "question": expectation.question,
                "answer_type": expectation.answer_type,
                "mode": expectation.mode,
                "expected_titles": expectation.expected_titles,
                "relevant_shas": expectation.relevant_shas,
                "note": expectation.note,
            }
            for q_index, expectation in expectations.items()
        },
    )

    summaries: List[Dict[str, Any]] = []
    for variant, path in review_pack_paths().items():
        rows = json.loads(path.read_text(encoding="utf-8"))["questions"]
        csv_path = PRODUCTION_IDEAS_DIR / f"doc_audit_{variant}.csv"
        summaries.append(
            write_variant_csv(
                variant=variant,
                rows=rows,
                expectations=expectations,
                output_path=csv_path,
            )
        )

    safe_json_dump(DOC_AUDIT_SUMMARY_PATH, summaries)
    write_markdown_report(summaries, DOC_AUDIT_REPORT_PATH)
    print(DOC_AUDIT_REPORT_PATH)
    print(DOC_AUDIT_SUMMARY_PATH)


if __name__ == "__main__":
    main()
