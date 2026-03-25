from __future__ import annotations

import argparse
from pathlib import Path

from src.public_dataset_eval import PublicCorpus, safe_json_dump, safe_json_load
from src.public_retrieval_benchmark import (
    enrich_question_with_memory,
    question_has_explicit_targets,
    run_strategy,
    update_memory_from_reranked,
)


def build_strategy_pack(
    dataset_path: Path,
    pdf_dir: Path,
    chunked_dir: Path,
    work_dir: Path,
    strategy: str,
) -> dict:
    corpus = PublicCorpus(work_dir, pdf_dir, chunked_dir)
    questions = safe_json_load(dataset_path)
    use_memory = strategy.endswith("_with_context")
    base_strategy = strategy.replace("_with_context", "")
    memory_shas = []

    records = []
    for index, item in enumerate(questions, start=1):
        original_question = item["question"]
        base_route = corpus.route_question(original_question, expansive=False)
        effective_question = (
            enrich_question_with_memory(corpus, original_question, memory_shas)
            if use_memory
            else original_question
        )
        route = corpus.route_question(effective_question, expansive=True)
        reranked = run_strategy(corpus, corpus.bm25_index, effective_question, route["candidate_shas"], base_strategy)

        records.append(
            {
                "index": index,
                "id": item["id"],
                "question": original_question,
                "effective_question": effective_question,
                "answer_type": item["answer_type"],
                "route": {
                    "candidate_shas": route["candidate_shas"],
                    "candidate_docs": [
                        {
                            "sha": sha,
                            "title": corpus.documents[sha].title,
                            "kind": corpus.documents[sha].kind,
                            "canonical_ids": corpus.documents[sha].canonical_ids,
                        }
                        for sha in route["candidate_shas"]
                    ],
                    "explicit_case_ids": route["explicit_case_ids"],
                    "explicit_law_ids": route["explicit_law_ids"],
                    "alias_hits": route["alias_hits"],
                },
                "pre_rerank": reranked[:5],
                "post_rerank": reranked[:5],
            }
        )

        if use_memory and question_has_explicit_targets(base_route):
            new_memory = update_memory_from_reranked(base_route, reranked)
            if new_memory:
                memory_shas = new_memory

    payload = {"strategy": strategy, "questions": records}
    safe_json_dump(work_dir / "review_pack.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a review pack for a benchmarked public retrieval strategy.")
    parser.add_argument("--dataset", default="public_dataset.json")
    parser.add_argument("--pdf-dir", default="pdfs")
    parser.add_argument("--chunked-dir", default="artifacts/public_eval/docling/chunked")
    parser.add_argument("--work-dir", default="artifacts/public_review_doc_diverse")
    parser.add_argument("--strategy", default="dense_doc_diverse")
    args = parser.parse_args()

    payload = build_strategy_pack(
        dataset_path=Path(args.dataset),
        pdf_dir=Path(args.pdf_dir),
        chunked_dir=Path(args.chunked_dir),
        work_dir=Path(args.work_dir),
        strategy=args.strategy,
    )
    print({"questions": len(payload["questions"]), "strategy": payload["strategy"]})


if __name__ == "__main__":
    main()
