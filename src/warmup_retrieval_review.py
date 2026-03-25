from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.isaacus_enrichment import build_enriched_chunk_corpus
from src.isaacus_models import IsaacusEmbeddingModel, IsaacusReranker
from src.local_models import DEFAULT_LOCAL_EMBEDDING_MODEL, DEFAULT_LOCAL_RERANKER_MODEL
from src.public_dataset_eval import PublicCorpus, prepare_docling_artifacts, safe_json_dump, safe_json_load
from src.public_retrieval_benchmark import run_strategy
from src.section_chunking import build_section_aware_chunk_corpus


def cache_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def preview(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def build_stack(
    work_dir: Path,
    pdf_dir: Path,
    embedding_backend: str,
    reranker_backend: str,
    chunk_source: str,
) -> PublicCorpus:
    artifacts = prepare_docling_artifacts(work_dir, pdf_dir)
    if chunk_source == "standard":
        chunked_dir = artifacts["chunked_dir"]
    elif chunk_source == "section_aware":
        chunked_dir = build_section_aware_chunk_corpus(
            merged_dir=artifacts["merged_dir"],
            output_dir=work_dir / "docling" / "chunked_section_aware",
        )
    elif chunk_source == "isaacus_enriched":
        chunked_dir = build_enriched_chunk_corpus(
            merged_dir=artifacts["merged_dir"],
            output_dir=work_dir / "docling" / "chunked_isaacus_enriched",
            cache_dir=work_dir / "isaacus_cache",
        )
    else:
        raise ValueError(f"Unsupported chunk source: {chunk_source}")

    if embedding_backend == "isaacus":
        embedder = IsaacusEmbeddingModel(cache_dir=work_dir / "isaacus_cache")
        embedding_model = embedder.model_name
    elif embedding_backend == "local":
        embedder = None
        embedding_model = DEFAULT_LOCAL_EMBEDDING_MODEL
    else:
        raise ValueError(f"Unsupported embedding backend: {embedding_backend}")

    if reranker_backend == "isaacus":
        reranker = IsaacusReranker(cache_dir=work_dir / "isaacus_cache")
        reranker_model = reranker.model_name
    elif reranker_backend == "local_jina":
        reranker = None
        reranker_model = DEFAULT_LOCAL_RERANKER_MODEL
    else:
        raise ValueError(f"Unsupported reranker backend: {reranker_backend}")

    return PublicCorpus(
        work_dir=work_dir,
        pdf_dir=pdf_dir,
        chunked_dir=chunked_dir,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        embedder=embedder,
        reranker=reranker,
    )


def explicit_targets(route: Dict[str, Any]) -> List[str]:
    return sorted(set(route.get("explicit_case_ids", []) + route.get("explicit_law_ids", []) + route.get("alias_hits", [])))


def retrieved_target_hit(route: Dict[str, Any], reranked: Sequence[Dict[str, Any]], limit: int = 5) -> bool:
    targets = set(route.get("explicit_case_ids", []) + route.get("explicit_law_ids", []))
    if not targets:
        return False
    for item in reranked[:limit]:
        ids = set(item.get("canonical_ids") or [])
        if ids & targets:
            return True
    return False


def pack_name(embedding_backend: str, reranker_backend: str) -> str:
    return f"{embedding_backend}__{reranker_backend}"


def build_review_pack(
    questions_path: Path,
    pdf_dir: Path,
    work_dir: Path,
    embedding_backend: str,
    reranker_backend: str,
    strategy: str,
    chunk_source: str,
) -> Dict[str, Any]:
    corpus = build_stack(
        work_dir=work_dir,
        pdf_dir=pdf_dir,
        embedding_backend=embedding_backend,
        reranker_backend=reranker_backend,
        chunk_source=chunk_source,
    )
    questions = safe_json_load(questions_path)
    rows: List[Dict[str, Any]] = []

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        route = corpus.route_question(question, expansive=False)
        reranked = run_strategy(corpus, corpus.bm25_index, question, route["candidate_shas"], strategy)
        top_chunks = []
        for rank, chunk in enumerate(reranked[:5], start=1):
            top_chunks.append(
                {
                    "rank": rank,
                    "title": chunk["title"],
                    "page": chunk["page"],
                    "ref": chunk["ref"],
                    "sha": chunk["sha"],
                    "distance": float(chunk.get("distance", 0.0)),
                    "relevance_score": float(chunk.get("relevance_score", 0.0)),
                    "normalized_relevance_score": float(chunk.get("normalized_relevance_score", chunk.get("relevance_score", 0.0))),
                    "canonical_ids": chunk.get("canonical_ids", []),
                    "preview": preview(chunk.get("text", "")),
                }
            )

        rows.append(
            {
                "index": index,
                "id": item["id"],
                "question": question,
                "answer_type": item["answer_type"],
                "route": route,
                "explicit_targets": explicit_targets(route),
                "explicit_target_hit_top5": retrieved_target_hit(route, reranked, limit=5),
                "top_chunks": top_chunks,
                "top_titles": [chunk["title"] for chunk in top_chunks],
            }
        )

    summary = {
        "question_count": len(rows),
        "explicit_target_questions": sum(1 for row in rows if row["explicit_targets"]),
        "explicit_target_hit_top5": sum(1 for row in rows if row["explicit_targets"] and row["explicit_target_hit_top5"]),
    }
    payload = {
        "config": {
            "embedding_backend": embedding_backend,
            "reranker_backend": reranker_backend,
            "strategy": strategy,
            "chunk_source": chunk_source,
        },
        "summary": summary,
        "questions": rows,
    }
    safe_json_dump(work_dir / f"review_pack_{pack_name(embedding_backend, reranker_backend)}__{chunk_source}.json", payload)
    return payload


def load_pack(path: Path) -> Dict[str, Any]:
    return safe_json_load(path)


def build_comparison(
    baseline_pack: Dict[str, Any],
    candidate_pack: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    baseline_rows = {row["id"]: row for row in baseline_pack["questions"]}
    candidate_rows = {row["id"]: row for row in candidate_pack["questions"]}
    rows: List[Dict[str, Any]] = []

    improved_explicit = 0
    regressed_explicit = 0
    changed_top1 = 0
    changed_top5 = 0

    for candidate in candidate_pack["questions"]:
        baseline = baseline_rows[candidate["id"]]
        base_top_titles = baseline.get("top_titles", [])
        candidate_top_titles = candidate.get("top_titles", [])
        base_top1 = base_top_titles[0] if base_top_titles else ""
        candidate_top1 = candidate_top_titles[0] if candidate_top_titles else ""
        explicit_targets_row = ", ".join(candidate.get("explicit_targets", []))
        base_hit = bool(baseline.get("explicit_target_hit_top5"))
        candidate_hit = bool(candidate.get("explicit_target_hit_top5"))
        if explicit_targets_row and (not base_hit) and candidate_hit:
            improved_explicit += 1
        if explicit_targets_row and base_hit and (not candidate_hit):
            regressed_explicit += 1
        if base_top1 != candidate_top1:
            changed_top1 += 1
        if base_top_titles[:5] != candidate_top_titles[:5]:
            changed_top5 += 1

        rows.append(
            {
                "index": candidate["index"],
                "id": candidate["id"],
                "answer_type": candidate["answer_type"],
                "question": candidate["question"],
                "explicit_targets": explicit_targets_row,
                "baseline_explicit_hit_top5": "yes" if base_hit else "no",
                "candidate_explicit_hit_top5": "yes" if candidate_hit else "no",
                "baseline_top1_title": base_top1,
                "candidate_top1_title": candidate_top1,
                "baseline_top5_titles": " | ".join(base_top_titles[:5]),
                "candidate_top5_titles": " | ".join(candidate_top_titles[:5]),
                "baseline_top1_preview": baseline.get("top_chunks", [{}])[0].get("preview", "") if baseline.get("top_chunks") else "",
                "candidate_top1_preview": candidate.get("top_chunks", [{}])[0].get("preview", "") if candidate.get("top_chunks") else "",
                "baseline_refs": " | ".join(chunk["ref"] for chunk in baseline.get("top_chunks", [])[:5]),
                "candidate_refs": " | ".join(chunk["ref"] for chunk in candidate.get("top_chunks", [])[:5]),
            }
        )

    summary = {
        "question_count": len(rows),
        "explicit_target_questions": sum(1 for row in rows if row["explicit_targets"]),
        "baseline_explicit_hit_top5": sum(1 for row in rows if row["explicit_targets"] and row["baseline_explicit_hit_top5"] == "yes"),
        "candidate_explicit_hit_top5": sum(1 for row in rows if row["explicit_targets"] and row["candidate_explicit_hit_top5"] == "yes"),
        "improved_explicit_target_hits": improved_explicit,
        "regressed_explicit_target_hits": regressed_explicit,
        "changed_top1": changed_top1,
        "changed_top5": changed_top5,
    }
    return rows, summary


def save_comparison(outputs_dir: Path, rows: Sequence[Dict[str, Any]], summary: Dict[str, Any], baseline_label: str, candidate_label: str) -> None:
    csv_path = outputs_dir / f"comparison_{baseline_label}__vs__{candidate_label}.csv"
    md_path = outputs_dir / f"comparison_{baseline_label}__vs__{candidate_label}.md"

    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        f"# Warm-up retrieval comparison: `{baseline_label}` vs `{candidate_label}`",
        "",
        "## Summary",
        f"- Questions: `{summary['question_count']}`",
        f"- Explicit-target questions: `{summary['explicit_target_questions']}`",
        f"- Baseline explicit hit@5: `{summary['baseline_explicit_hit_top5']}`",
        f"- Candidate explicit hit@5: `{summary['candidate_explicit_hit_top5']}`",
        f"- Improved explicit hit@5 count: `{summary['improved_explicit_target_hits']}`",
        f"- Regressed explicit hit@5 count: `{summary['regressed_explicit_target_hits']}`",
        f"- Top-1 title changed: `{summary['changed_top1']}` questions",
        f"- Top-5 title set/order changed: `{summary['changed_top5']}` questions",
        "",
        "## Per-question comparison",
        "",
    ]

    for row in rows:
        lines.extend(
            [
                f"### Q{row['index']:03d} [{row['answer_type']}]",
                row["question"],
                "",
                f"- Explicit targets: `{row['explicit_targets'] or 'none'}`",
                f"- Baseline explicit hit@5: `{row['baseline_explicit_hit_top5']}`",
                f"- Candidate explicit hit@5: `{row['candidate_explicit_hit_top5']}`",
                f"- Baseline top-1: `{row['baseline_top1_title']}`",
                f"- Candidate top-1: `{row['candidate_top1_title']}`",
                f"- Baseline top-5 titles: `{row['baseline_top5_titles']}`",
                f"- Candidate top-5 titles: `{row['candidate_top5_titles']}`",
                f"- Baseline top-1 preview: {row['baseline_top1_preview']}",
                f"- Candidate top-1 preview: {row['candidate_top1_preview']}",
                "",
            ]
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build warm-up retrieval packs and side-by-side comparison reports.")
    parser.add_argument("--questions", default="starter_kit/questions_api.json")
    parser.add_argument("--pdf-dir", default="starter_kit/docs_corpus")
    parser.add_argument("--work-dir", default="starter_kit/challenge_workdir")
    parser.add_argument("--strategy", default="dense_doc_diverse")
    parser.add_argument("--baseline-embedding", default="local")
    parser.add_argument("--baseline-reranker", default="local_jina")
    parser.add_argument("--baseline-chunk-source", default="standard")
    parser.add_argument("--candidate-embedding", default="isaacus")
    parser.add_argument("--candidate-reranker", default="local_jina")
    parser.add_argument("--candidate-chunk-source", default="standard")
    args = parser.parse_args()

    questions_path = Path(args.questions)
    pdf_dir = Path(args.pdf_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    baseline = build_review_pack(
        questions_path=questions_path,
        pdf_dir=pdf_dir,
        work_dir=work_dir,
        embedding_backend=args.baseline_embedding,
        reranker_backend=args.baseline_reranker,
        strategy=args.strategy,
        chunk_source=args.baseline_chunk_source,
    )
    candidate = build_review_pack(
        questions_path=questions_path,
        pdf_dir=pdf_dir,
        work_dir=work_dir,
        embedding_backend=args.candidate_embedding,
        reranker_backend=args.candidate_reranker,
        strategy=args.strategy,
        chunk_source=args.candidate_chunk_source,
    )

    rows, summary = build_comparison(baseline, candidate)
    baseline_label = f"{pack_name(args.baseline_embedding, args.baseline_reranker)}__{args.baseline_chunk_source}"
    candidate_label = f"{pack_name(args.candidate_embedding, args.candidate_reranker)}__{args.candidate_chunk_source}"
    save_comparison(work_dir, rows, summary, baseline_label, candidate_label)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
