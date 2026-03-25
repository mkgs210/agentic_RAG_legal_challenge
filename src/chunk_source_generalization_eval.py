from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Sequence
import sys

from rank_bm25 import BM25Okapi

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.production_idea_corpora import build_contextual_chunk_corpus, build_summary_augmented_chunk_corpus
from src.public_dataset_eval import (
    PublicCorpus,
    prepare_docling_artifacts,
    safe_json_dump,
    safe_json_load,
)
from src.public_retrieval_benchmark import (
    build_oracle_documents,
    dedupe_by_ref,
    doc_diversified_candidates,
    lexical_candidates,
    parse_manual_audit,
    unique_titles,
    vector_candidates,
)


def warmup_explicit_target_metrics(corpus: PublicCorpus, questions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    explicit_count = 0
    hit_at_1 = 0
    hit_at_5 = 0
    unique_doc_hit_at_5 = 0
    rows = []

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        route = corpus.route_question(question, expansive=False)
        explicit_targets = set(route["explicit_case_ids"] or []) | set(route["explicit_law_ids"] or []) | set(route["alias_hits"] or [])
        if not explicit_targets:
            continue
        explicit_count += 1
        vector = vector_candidates(corpus, question, route["candidate_shas"], vector_k=16)
        lexical = lexical_candidates(corpus, corpus.bm25_index, question, route["candidate_shas"], bm25_k=16)
        fused = dedupe_by_ref(vector[:6] + doc_diversified_candidates(vector, head_keep=4, max_unique_docs=6) + lexical[:4])
        top1 = fused[:1]
        top5 = fused[:5]
        top5_titles = unique_titles(top5, limit=5)

        def hit(results: Sequence[Dict[str, Any]]) -> bool:
            for row in results:
                canonical_ids = set(row.get("canonical_ids") or [])
                if canonical_ids & explicit_targets:
                    return True
                title = str(row.get("title") or "").lower()
                if any(target.lower() in title for target in explicit_targets):
                    return True
            return False

        if hit(top1):
            hit_at_1 += 1
        if hit(top5):
            hit_at_5 += 1
        if any(any(target.lower() in title.lower() for target in explicit_targets) for title in top5_titles):
            unique_doc_hit_at_5 += 1

        rows.append(
            {
                "index": index,
                "id": item["id"],
                "question": question,
                "explicit_targets": sorted(explicit_targets),
                "top1_title": top1[0]["title"] if top1 else "",
                "top5_titles": top5_titles,
                "hit_at_1": hit(top1),
                "hit_at_5": hit(top5),
            }
        )

    return {
        "explicit_question_count": explicit_count,
        "hit_at_1": round(hit_at_1 / max(1, explicit_count), 4),
        "hit_at_5": round(hit_at_5 / max(1, explicit_count), 4),
        "unique_doc_hit_at_5": round(unique_doc_hit_at_5 / max(1, explicit_count), 4),
        "rows": rows,
    }


def public_oracle_metrics(
    corpus: PublicCorpus,
    dataset: Sequence[Dict[str, Any]],
    audit_rows: Dict[int, Dict[str, str]],
) -> Dict[str, Any]:
    oracle_docs = build_oracle_documents(corpus, audit_rows, dataset)
    top1_hits = 0
    top5_hits = 0
    coverage_total = 0.0
    anchored = 0
    rows = []

    for index, item in enumerate(dataset, start=1):
        oracle = oracle_docs.get(index, {})
        oracle_shas = list(oracle.get("oracle_shas") or [])
        if not oracle_shas:
            continue
        anchored += 1
        question = item["question"]
        route = corpus.route_question(question, expansive=False)
        vector = vector_candidates(corpus, question, route["candidate_shas"], vector_k=24)
        lexical = lexical_candidates(corpus, corpus.bm25_index, question, route["candidate_shas"], bm25_k=12)
        fused = dedupe_by_ref(vector[:8] + doc_diversified_candidates(vector, head_keep=5, max_unique_docs=8) + lexical[:4])
        top1 = fused[:1]
        top5 = fused[:5]
        top5_shas = []
        seen = set()
        for row in top5:
            sha = row["sha"]
            if sha in seen:
                continue
            seen.add(sha)
            top5_shas.append(sha)
        if any(row["sha"] in oracle_shas for row in top1):
            top1_hits += 1
        if any(row["sha"] in oracle_shas for row in top5):
            top5_hits += 1
        coverage = len(set(top5_shas) & set(oracle_shas)) / max(1, len(set(oracle_shas)))
        coverage_total += coverage
        rows.append(
            {
                "index": index,
                "id": item["id"],
                "question": question,
                "oracle_titles": oracle.get("oracle_titles", []),
                "top5_titles": unique_titles(top5, limit=5),
                "top1_hit": any(row["sha"] in oracle_shas for row in top1),
                "top5_hit": any(row["sha"] in oracle_shas for row in top5),
                "coverage": round(coverage, 4),
            }
        )

    return {
        "anchored_question_count": anchored,
        "top1_hit": round(top1_hits / max(1, anchored), 4),
        "top5_hit": round(top5_hits / max(1, anchored), 4),
        "mean_coverage": round(coverage_total / max(1, anchored), 4),
        "rows": rows,
    }


def build_corpus(work_dir: Path, pdf_dir: Path, chunk_source: str) -> PublicCorpus:
    artifacts = prepare_docling_artifacts(work_dir, pdf_dir)
    chunked_dir = artifacts["chunked_dir"]
    if chunk_source == "contextual":
        chunked_dir = build_contextual_chunk_corpus(
            chunked_dir=artifacts["chunked_dir"],
            output_dir=work_dir / "docling" / "chunked_contextual",
        )
    elif chunk_source == "summary_augmented":
        chunked_dir = build_summary_augmented_chunk_corpus(
            chunked_dir=artifacts["chunked_dir"],
            output_dir=work_dir / "docling" / "chunked_summary_augmented",
        )
    return PublicCorpus(work_dir=work_dir, pdf_dir=pdf_dir, chunked_dir=chunked_dir)


def run_eval(
    *,
    chunk_source: str,
    warmup_work_dir: Path,
    warmup_pdf_dir: Path,
    warmup_questions_path: Path,
    public_work_dir: Path,
    public_pdf_dir: Path,
    public_dataset_path: Path,
    public_audit_path: Path,
) -> Dict[str, Any]:
    warmup_corpus = build_corpus(warmup_work_dir, warmup_pdf_dir, chunk_source)
    warmup_questions = safe_json_load(warmup_questions_path)
    warmup_metrics = warmup_explicit_target_metrics(warmup_corpus, warmup_questions)

    public_corpus = build_corpus(public_work_dir, public_pdf_dir, chunk_source)
    public_dataset = safe_json_load(public_dataset_path)
    public_audit_rows = parse_manual_audit(public_audit_path)
    public_metrics = public_oracle_metrics(public_corpus, public_dataset, public_audit_rows)

    return {
        "chunk_source": chunk_source,
        "warmup_explicit": warmup_metrics,
        "public_oracle": public_metrics,
    }


def build_markdown(results: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# Chunk Source Generalization Eval",
        "",
        "Compares retrieval behavior across warm-up questions and `public_dataset` using the same chunk-source variant.",
        "",
    ]
    for result in results:
        warm = result["warmup_explicit"]
        pub = result["public_oracle"]
        lines.extend(
            [
                f"## `{result['chunk_source']}`",
                "",
                f"- Warm-up explicit questions: `{warm['explicit_question_count']}`",
                f"- Warm-up hit@1: `{warm['hit_at_1']}`",
                f"- Warm-up hit@5: `{warm['hit_at_5']}`",
                f"- Warm-up unique-doc hit@5: `{warm['unique_doc_hit_at_5']}`",
                f"- Public anchored questions: `{pub['anchored_question_count']}`",
                f"- Public top1 hit: `{pub['top1_hit']}`",
                f"- Public top5 hit: `{pub['top5_hit']}`",
                f"- Public mean coverage: `{pub['mean_coverage']}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare chunk corpus variants on warm-up and public retrieval benchmarks.")
    parser.add_argument("--warmup-work-dir", default="starter_kit/challenge_workdir")
    parser.add_argument("--warmup-pdf-dir", default="starter_kit/docs_corpus")
    parser.add_argument("--warmup-questions", default="starter_kit/questions_api.json")
    parser.add_argument("--public-work-dir", default="artifacts/public_eval")
    parser.add_argument("--public-pdf-dir", default="data/pdf_reports")
    parser.add_argument("--public-dataset", default="public_dataset.json")
    parser.add_argument("--public-audit", default="artifacts/public_review/manual_answers_and_pre_rerank_audit.md")
    parser.add_argument("--out-dir", default="starter_kit/challenge_workdir/chunk_source_eval")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for chunk_source in ("standard", "contextual", "summary_augmented"):
        result = run_eval(
            chunk_source=chunk_source,
            warmup_work_dir=Path(args.warmup_work_dir),
            warmup_pdf_dir=Path(args.warmup_pdf_dir),
            warmup_questions_path=Path(args.warmup_questions),
            public_work_dir=Path(args.public_work_dir),
            public_pdf_dir=Path(args.public_pdf_dir),
            public_dataset_path=Path(args.public_dataset),
            public_audit_path=Path(args.public_audit),
        )
        results.append(result)

    safe_json_dump(out_dir / "chunk_source_generalization_eval.json", results)
    (out_dir / "chunk_source_generalization_eval.md").write_text(build_markdown(results), encoding="utf-8")


if __name__ == "__main__":
    main()
