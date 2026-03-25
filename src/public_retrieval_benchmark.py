from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from src.build_public_review_report import parse_manual_audit
from src.lexical_retrieval import tokenize_for_bm25
from src.public_dataset_eval import PublicCorpus, normalize_space, safe_json_dump, safe_json_load


def normalized_text(value: str) -> str:
    return normalize_space(value).lower()


def build_alias_index(corpus: PublicCorpus) -> List[Tuple[str, str]]:
    alias_pairs: List[Tuple[str, str]] = []
    for sha, document in corpus.documents.items():
        values = set(document.aliases) | set(document.canonical_ids) | {document.title}
        for alias in values:
            alias_norm = normalized_text(alias)
            if len(alias_norm) < 5:
                continue
            alias_pairs.append((alias_norm, sha))
    alias_pairs.sort(key=lambda item: len(item[0]), reverse=True)
    return alias_pairs


def explicit_question_oracle_shas(corpus: PublicCorpus, question: str) -> Set[str]:
    route = corpus.route_question(question, expansive=False)
    has_explicit_targets = bool(
        route["explicit_case_ids"] or route["explicit_law_ids"] or route["alias_hits"]
    )
    if not has_explicit_targets:
        return set()
    return set(route["candidate_shas"])


def resolve_reference_links(text: str) -> List[int]:
    refs = []
    for match in re.finditer(r"\bQ(\d{1,3})\b", text or "", re.I):
        refs.append(int(match.group(1)))
    return refs


def build_oracle_documents(
    corpus: PublicCorpus,
    audit_rows: Dict[int, Dict[str, str]],
    dataset: Sequence[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    alias_pairs = build_alias_index(corpus)
    question_by_index = {index: item["question"] for index, item in enumerate(dataset, start=1)}
    resolved: Dict[int, Dict[str, Any]] = {}

    def resolve_question(index: int, stack: Set[int]) -> Dict[str, Any]:
        if index in resolved:
            return resolved[index]
        if index in stack:
            return {"oracle_shas": [], "oracle_titles": []}
        stack = set(stack)
        stack.add(index)

        row = audit_rows.get(index, {})
        text = " ".join(
            [
                row.get("expected_answer", ""),
                row.get("expected_source", ""),
            ]
        )
        text_norm = normalized_text(text)
        oracle_shas: Set[str] = set()

        for alias_norm, sha in alias_pairs:
            if alias_norm in text_norm:
                oracle_shas.add(sha)

        oracle_shas.update(explicit_question_oracle_shas(corpus, question_by_index.get(index, "")))

        for ref_index in resolve_reference_links(text):
            oracle_shas.update(resolve_question(ref_index, stack)["oracle_shas"])

        result = {
            "oracle_shas": sorted(oracle_shas),
            "oracle_titles": sorted({corpus.documents[sha].title for sha in oracle_shas}),
        }
        resolved[index] = result
        return result

    for index in sorted(audit_rows):
        resolve_question(index, set())

    return resolved


DEICTIC_RE = re.compile(
    r"\b(this law|this case|this regulation|the law|the case|the regulation|the enacted law|the enactment notice)\b",
    re.I,
)


def memory_context_suffix(corpus: PublicCorpus, memory_shas: Sequence[str]) -> str:
    parts = []
    for sha in memory_shas[:2]:
        document = corpus.documents.get(sha)
        if not document:
            continue
        ids = ", ".join(document.canonical_ids[:2])
        if ids:
            parts.append(f"{document.title} ({ids})")
        else:
            parts.append(document.title)
    return "; ".join(parts)


def enrich_question_with_memory(
    corpus: PublicCorpus,
    question: str,
    memory_shas: Sequence[str],
) -> str:
    if not memory_shas:
        return question
    route = corpus.route_question(question, expansive=False)
    has_explicit_targets = bool(
        route["explicit_case_ids"] or route["explicit_law_ids"] or route["alias_hits"]
    )
    if has_explicit_targets:
        return question
    if not DEICTIC_RE.search(question):
        return question
    suffix = memory_context_suffix(corpus, memory_shas)
    if not suffix:
        return question
    return f"{question}\n\nContext entity: {suffix}"


def question_has_explicit_targets(route: Dict[str, Any]) -> bool:
    return bool(route["explicit_case_ids"] or route["explicit_law_ids"] or route["alias_hits"])


def update_memory_from_reranked(
    explicit_route: Dict[str, Any],
    reranked: Sequence[Dict[str, Any]],
) -> List[str]:
    if not question_has_explicit_targets(explicit_route):
        return []
    memory: List[str] = []
    seen = set()
    for item in reranked:
        sha = item.get("sha")
        if not sha or sha in seen:
            continue
        memory.append(sha)
        seen.add(sha)
        if len(memory) >= 2:
            break
    return memory


def unique_titles(results: Sequence[Dict[str, Any]], limit: int = 5) -> List[str]:
    titles: List[str] = []
    seen = set()
    for item in results:
        title = item.get("title", "")
        if not title or title in seen:
            continue
        titles.append(title)
        seen.add(title)
        if len(titles) >= limit:
            break
    return titles


def dedupe_by_ref(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for item in results:
        ref = item.get("ref")
        if ref in seen:
            continue
        seen.add(ref)
        merged.append(item)
    return merged


def vector_candidates(corpus: PublicCorpus, question: str, candidate_shas: Sequence[str], vector_k: int) -> List[Dict[str, Any]]:
    candidate_indices = corpus._candidate_chunk_indices(candidate_shas)
    query_embedding = corpus.embedder.embed_query(question)
    candidate_embeddings = corpus.chunk_embeddings[candidate_indices]
    scores = candidate_embeddings @ query_embedding
    top_order = np.argsort(scores)[::-1][: min(vector_k, len(candidate_indices))]
    rows: List[Dict[str, Any]] = []
    for rank in top_order:
        chunk_index = candidate_indices[int(rank)]
        chunk = corpus.chunks[chunk_index]
        rows.append(
            {
                "ref": chunk.ref,
                "distance": round(float(scores[int(rank)]), 4),
                "page": chunk.page,
                "text": chunk.text,
                "sha": chunk.sha,
                "title": chunk.title,
                "kind": chunk.kind,
                "canonical_ids": chunk.canonical_ids,
                "retrieval_sources": ["vector"],
            }
        )
    return rows


def lexical_candidates(
    corpus: PublicCorpus,
    bm25_index: BM25Okapi,
    question: str,
    candidate_shas: Sequence[str],
    bm25_k: int,
) -> List[Dict[str, Any]]:
    candidate_indices = corpus._candidate_chunk_indices(candidate_shas)
    scores = bm25_index.get_scores(tokenize_for_bm25(question))
    ranked_indices = sorted(candidate_indices, key=lambda idx: scores[idx], reverse=True)[: min(bm25_k, len(candidate_indices))]
    rows: List[Dict[str, Any]] = []
    for chunk_index in ranked_indices:
        chunk = corpus.chunks[chunk_index]
        rows.append(
            {
                "ref": chunk.ref,
                "distance": round(float(scores[chunk_index]), 4),
                "page": chunk.page,
                "text": chunk.text,
                "sha": chunk.sha,
                "title": chunk.title,
                "kind": chunk.kind,
                "canonical_ids": chunk.canonical_ids,
                "retrieval_sources": ["bm25"],
            }
        )
    return rows


def doc_diversified_candidates(results: Sequence[Dict[str, Any]], head_keep: int, max_unique_docs: int) -> List[Dict[str, Any]]:
    diversified: List[Dict[str, Any]] = list(results[:head_keep])
    seen_refs = {item["ref"] for item in diversified}
    seen_docs = {item["sha"] for item in diversified}

    for item in results:
        if item["ref"] in seen_refs:
            continue
        if item["sha"] in seen_docs:
            continue
        diversified.append(item)
        seen_refs.add(item["ref"])
        seen_docs.add(item["sha"])
        if len(seen_docs) >= max_unique_docs:
            break
    return diversified


def run_strategy(
    corpus: PublicCorpus,
    bm25_index: BM25Okapi,
    question: str,
    candidate_shas: Sequence[str],
    strategy: str,
) -> List[Dict[str, Any]]:
    vector = vector_candidates(corpus, question, candidate_shas, vector_k=16 if strategy != "baseline" else 8)

    if strategy == "baseline":
        candidates = vector[:8]
    elif strategy == "adaptive_lexical":
        lexical = lexical_candidates(corpus, bm25_index, question, candidate_shas, bm25_k=8)
        novel = []
        seen_refs = {item["ref"] for item in vector[:8]}
        for item in lexical:
            if item["ref"] in seen_refs:
                continue
            novel.append(item)
            if len(novel) >= 4:
                break
        candidates = dedupe_by_ref(vector[:8] + novel)
    elif strategy == "dense_doc_diverse":
        candidates = doc_diversified_candidates(vector, head_keep=4, max_unique_docs=10)
    elif strategy == "dense_doc_diverse_lexical":
        lexical = lexical_candidates(corpus, bm25_index, question, candidate_shas, bm25_k=8)
        diverse = doc_diversified_candidates(vector, head_keep=4, max_unique_docs=10)
        lexical_diverse = doc_diversified_candidates(lexical, head_keep=0, max_unique_docs=6)
        candidates = dedupe_by_ref(diverse + lexical_diverse[:4])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    reranked = corpus.reranker.rerank_documents(question, candidates, llm_weight=0.7)[:5]
    return reranked


def evaluate_strategy(
    corpus: PublicCorpus,
    dataset: Sequence[Dict[str, Any]],
    oracle_docs: Dict[int, Dict[str, Any]],
    strategy: str,
) -> Dict[str, Any]:
    bm25_index = BM25Okapi([tokenize_for_bm25(chunk.text) for chunk in corpus.chunks])
    rows = []
    top1_hit = 0
    top3_hit = 0
    top5_hit = 0
    anchored_count = 0
    anchored_top1_hit = 0
    anchored_top3_hit = 0
    anchored_top5_hit = 0
    coverage_top3_scores: List[float] = []
    coverage_scores: List[float] = []
    multi_doc_rows = 0
    memory_shas: List[str] = []

    use_memory = strategy.endswith("_with_context")
    base_strategy = strategy.replace("_with_context", "")

    for index, item in enumerate(dataset, start=1):
        original_question = item["question"]
        base_route = corpus.route_question(original_question, expansive=False)
        effective_question = (
            enrich_question_with_memory(corpus, original_question, memory_shas)
            if use_memory
            else original_question
        )
        route = corpus.route_question(effective_question, expansive=True)
        reranked = run_strategy(corpus, bm25_index, effective_question, route["candidate_shas"], base_strategy)
        retrieved_titles = unique_titles(reranked, limit=5)
        retrieved_titles_top3 = retrieved_titles[:3]
        oracle_titles = oracle_docs.get(index, {}).get("oracle_titles", [])
        oracle_set = set(oracle_titles)
        hit1 = bool(retrieved_titles[:1] and retrieved_titles[0] in oracle_set)
        hit3 = any(title in oracle_set for title in retrieved_titles_top3)
        hit5 = any(title in oracle_set for title in retrieved_titles)
        if hit1:
            top1_hit += 1
        if hit3:
            top3_hit += 1
        if hit5:
            top5_hit += 1

        if oracle_titles:
            anchored_count += 1
            if hit1:
                anchored_top1_hit += 1
            if hit3:
                anchored_top3_hit += 1
            if hit5:
                anchored_top5_hit += 1
            coverage_top3 = len(set(retrieved_titles_top3).intersection(oracle_set)) / len(oracle_set)
            coverage = len(set(retrieved_titles).intersection(oracle_set)) / len(oracle_set)
            coverage_top3_scores.append(coverage_top3)
            coverage_scores.append(coverage)
            if len(oracle_titles) > 1:
                multi_doc_rows += 1

        if use_memory:
            new_memory = update_memory_from_reranked(base_route, reranked)
            if new_memory:
                memory_shas = new_memory

        rows.append(
            {
                "index": index,
                "question": original_question,
                "effective_question": effective_question,
                "oracle_titles": oracle_titles,
                "retrieved_titles": retrieved_titles,
                "top1_hit": hit1,
                "top3_hit": hit3,
                "top5_hit": hit5,
            }
        )

    result = {
        "strategy": strategy,
        "question_count": len(dataset),
        "anchored_question_count": anchored_count,
        "top1_hit_rate": round(top1_hit / max(1, len(dataset)), 4),
        "top3_hit_rate": round(top3_hit / max(1, len(dataset)), 4),
        "top5_hit_rate": round(top5_hit / max(1, len(dataset)), 4),
        "anchored_top1_hit_rate": round(anchored_top1_hit / max(1, anchored_count), 4),
        "anchored_top3_hit_rate": round(anchored_top3_hit / max(1, anchored_count), 4),
        "anchored_top5_hit_rate": round(anchored_top5_hit / max(1, anchored_count), 4),
        "mean_oracle_coverage_top3": round(sum(coverage_top3_scores) / max(1, len(coverage_top3_scores)), 4),
        "mean_oracle_coverage": round(sum(coverage_scores) / max(1, len(coverage_scores)), 4),
        "multi_doc_question_count": multi_doc_rows,
        "rows": rows,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval strategies on public_dataset using manual oracle docs.")
    parser.add_argument("--dataset", default="public_dataset.json")
    parser.add_argument("--audit", default="artifacts/public_review/manual_answers_and_pre_rerank_audit.md")
    parser.add_argument("--work-dir", default="artifacts/public_review")
    parser.add_argument("--pdf-dir", default="pdfs")
    parser.add_argument("--chunked-dir", default="artifacts/public_eval/docling/chunked")
    parser.add_argument("--out", default="artifacts/public_review/retrieval_benchmark.json")
    parser.add_argument("--embedding-model", default="")
    parser.add_argument(
        "--strategies",
        default="baseline,baseline_with_context,adaptive_lexical,dense_doc_diverse,dense_doc_diverse_with_context,dense_doc_diverse_lexical",
    )
    args = parser.parse_args()

    corpus = PublicCorpus(
        Path(args.work_dir),
        Path(args.pdf_dir),
        Path(args.chunked_dir),
        embedding_model=args.embedding_model or "intfloat/multilingual-e5-base",
    )
    dataset = safe_json_load(Path(args.dataset))
    audit_rows = parse_manual_audit(Path(args.audit))
    oracle_docs = build_oracle_documents(corpus, audit_rows, dataset)

    strategy_names = [item.strip() for item in args.strategies.split(",") if item.strip()]
    results = []
    for strategy in strategy_names:
        results.append(evaluate_strategy(corpus, dataset, oracle_docs, strategy))

    payload = {"embedding_model": corpus.embedding_model, "results": results}
    safe_json_dump(Path(args.out), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
