from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.public_dataset_eval import PublicCorpus, prepare_docling_artifacts, safe_json_dump, safe_json_load


def build_review_pack(
    dataset_path: Path,
    pdf_dir: Path,
    work_dir: Path,
    bm25_k: int = 0,
    bm25_weight: float = 0.0,
    bm25_auto: bool = False,
) -> dict:
    artifacts = prepare_docling_artifacts(work_dir, pdf_dir)
    corpus = PublicCorpus(work_dir, pdf_dir, artifacts["chunked_dir"])
    questions = safe_json_load(dataset_path)

    records = []
    for index, item in enumerate(questions, start=1):
        route = corpus.route_question(item["question"], expansive=True)
        retrieval = corpus.retrieve(
            question=item["question"],
            candidate_shas=route["candidate_shas"],
            vector_k=8,
            rerank_k=5,
            lexical_boost=False,
            bm25_k=bm25_k,
            bm25_weight=bm25_weight,
            bm25_auto=bm25_auto,
        )

        records.append(
            {
                "index": index,
                "id": item["id"],
                "question": item["question"],
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
                "pre_rerank": retrieval["vector_results"][:5],
                "post_rerank": retrieval["reranked_results"][:5],
            }
        )

    payload = {"questions": records}
    safe_json_dump(work_dir / "review_pack.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a retrieval review pack for public_dataset.json")
    parser.add_argument("--dataset", default="public_dataset.json")
    parser.add_argument("--pdf-dir", default="pdfs")
    parser.add_argument("--work-dir", default="artifacts/public_review")
    parser.add_argument("--bm25-k", type=int, default=0)
    parser.add_argument("--bm25-weight", type=float, default=0.0)
    parser.add_argument("--bm25-auto", action="store_true")
    args = parser.parse_args()

    payload = build_review_pack(
        dataset_path=Path(args.dataset),
        pdf_dir=Path(args.pdf_dir),
        work_dir=Path(args.work_dir),
        bm25_k=args.bm25_k,
        bm25_weight=args.bm25_weight,
        bm25_auto=args.bm25_auto,
    )
    print(json.dumps({"questions": len(payload["questions"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
