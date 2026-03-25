from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

from rank_bm25 import BM25Okapi

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.advanced_retrieval import AdvancedRetriever
from src.lexical_retrieval import tokenize_for_bm25
from src.public_dataset_eval import (
    PublicCorpus,
    prepare_docling_artifacts,
    safe_json_dump,
    safe_json_load,
)
from src.public_retrieval_benchmark import (
    build_oracle_documents,
    parse_manual_audit,
    run_strategy,
    unique_titles,
)
from src.query_analysis import QuestionAnalyzer


ADVANCED_STRATEGIES = {"intent_hybrid", "doc_profile_hybrid", "multi_index_hybrid"}


def build_corpus(work_dir: Path, pdf_dir: Path) -> PublicCorpus:
    artifacts = prepare_docling_artifacts(work_dir, pdf_dir)
    return PublicCorpus(work_dir=work_dir, pdf_dir=pdf_dir, chunked_dir=artifacts["chunked_dir"])


def retrieve(
    *,
    corpus: PublicCorpus,
    advanced: AdvancedRetriever,
    analyzer: QuestionAnalyzer,
    question: str,
    answer_type: str,
    strategy: str,
) -> tuple[list[dict[str, Any]], Any | None]:
    route = corpus.route_question(question, expansive=False)
    if strategy in ADVANCED_STRATEGIES:
        analysis = analyzer.analyze(question, answer_type)
        rows = advanced.retrieve(strategy=strategy, question=question, analysis=analysis, candidate_shas=route["candidate_shas"])
        return rows, analysis
    rows = run_strategy(corpus, corpus.bm25_index, question, route["candidate_shas"], strategy)
    return rows, None


def warmup_explicit_target_metrics(
    *,
    corpus: PublicCorpus,
    advanced: AdvancedRetriever,
    analyzer: QuestionAnalyzer,
    questions: Sequence[Dict[str, Any]],
    strategy: str,
) -> Dict[str, Any]:
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
        reranked, _ = retrieve(
            corpus=corpus,
            advanced=advanced,
            analyzer=analyzer,
            question=question,
            answer_type=item["answer_type"],
            strategy=strategy,
        )
        top1 = reranked[:1]
        top5 = reranked[:5]
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
    *,
    corpus: PublicCorpus,
    advanced: AdvancedRetriever,
    analyzer: QuestionAnalyzer,
    dataset: Sequence[Dict[str, Any]],
    audit_rows: Dict[int, Dict[str, str]],
    strategy: str,
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
        reranked, _ = retrieve(
            corpus=corpus,
            advanced=advanced,
            analyzer=analyzer,
            question=item["question"],
            answer_type=item["answer_type"],
            strategy=strategy,
        )
        top1 = reranked[:1]
        top5 = reranked[:5]
        top5_shas = []
        seen = set()
        for row in top5:
            sha = row["sha"]
            if sha in seen:
                continue
            seen.add(sha)
            top5_shas.append(sha)
        top1_hit = any(row["sha"] in oracle_shas for row in top1)
        top5_hit = any(row["sha"] in oracle_shas for row in top5)
        if top1_hit:
            top1_hits += 1
        if top5_hit:
            top5_hits += 1
        coverage = len(set(top5_shas) & set(oracle_shas)) / max(1, len(set(oracle_shas)))
        coverage_total += coverage
        rows.append(
            {
                "index": index,
                "id": item["id"],
                "question": item["question"],
                "oracle_titles": oracle.get("oracle_titles", []),
                "top5_titles": unique_titles(top5, limit=5),
                "top1_hit": top1_hit,
                "top5_hit": top5_hit,
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


def build_markdown(results: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# Advanced Strategy Generalization Eval",
        "",
        "Compares retrieval strategies on warm-up explicit-target questions and the anchored `public_dataset` benchmark.",
        "",
    ]
    for result in results:
        warm = result["warmup_explicit"]
        pub = result["public_oracle"]
        lines.extend(
            [
                f"## `{result['strategy']}`",
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
    parser = argparse.ArgumentParser(description="Compare advanced retrieval strategies across warm-up and public benchmarks.")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--analysis-model", default="gpt-4.1-mini")
    parser.add_argument("--warmup-work-dir", default="starter_kit/challenge_workdir")
    parser.add_argument("--warmup-pdf-dir", default="starter_kit/docs_corpus")
    parser.add_argument("--warmup-questions", default="starter_kit/questions_api.json")
    parser.add_argument("--public-work-dir", default="artifacts/public_eval")
    parser.add_argument("--public-pdf-dir", default="data/pdf_reports")
    parser.add_argument("--public-dataset", default="public_dataset.json")
    parser.add_argument("--public-audit", default="artifacts/public_review/manual_answers_and_pre_rerank_audit.md")
    parser.add_argument("--out-dir", default="starter_kit/challenge_workdir/advanced_strategy_eval")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warmup_corpus = build_corpus(Path(args.warmup_work_dir), Path(args.warmup_pdf_dir))
    public_corpus = build_corpus(Path(args.public_work_dir), Path(args.public_pdf_dir))

    warmup_questions = safe_json_load(Path(args.warmup_questions))
    public_dataset = safe_json_load(Path(args.public_dataset))
    public_audit_rows = parse_manual_audit(Path(args.public_audit))

    warmup_analyzer = QuestionAnalyzer(
        provider=args.provider,
        model=args.analysis_model,
        cache_path=Path(args.warmup_work_dir) / "query_analysis" / f"{args.provider}__{args.analysis_model.replace('/', '_')}.json",
        corpus=warmup_corpus,
    )
    public_analyzer = QuestionAnalyzer(
        provider=args.provider,
        model=args.analysis_model,
        cache_path=Path(args.public_work_dir) / "query_analysis" / f"{args.provider}__{args.analysis_model.replace('/', '_')}.json",
        corpus=public_corpus,
    )
    warmup_advanced = AdvancedRetriever(warmup_corpus, Path(args.warmup_work_dir) / "advanced_retrieval_eval")
    public_advanced = AdvancedRetriever(public_corpus, Path(args.public_work_dir) / "advanced_retrieval_eval")

    results = []
    for strategy in ("dense_doc_diverse", "intent_hybrid", "multi_index_hybrid", "doc_profile_hybrid"):
        results.append(
            {
                "strategy": strategy,
                "warmup_explicit": warmup_explicit_target_metrics(
                    corpus=warmup_corpus,
                    advanced=warmup_advanced,
                    analyzer=warmup_analyzer,
                    questions=warmup_questions,
                    strategy=strategy,
                ),
                "public_oracle": public_oracle_metrics(
                    corpus=public_corpus,
                    advanced=public_advanced,
                    analyzer=public_analyzer,
                    dataset=public_dataset,
                    audit_rows=public_audit_rows,
                    strategy=strategy,
                ),
            }
        )

    safe_json_dump(out_dir / "advanced_strategy_generalization_eval.json", results)
    (out_dir / "advanced_strategy_generalization_eval.md").write_text(build_markdown(results), encoding="utf-8")


if __name__ == "__main__":
    main()
