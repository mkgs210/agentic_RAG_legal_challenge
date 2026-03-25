from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.api_requests import APIProcessor
from src.local_models import DEFAULT_LOCAL_EMBEDDING_MODEL, DEFAULT_LOCAL_RERANKER_MODEL
from src.production_idea_corpora import (
    build_atomic_fact_chunk_corpus,
    build_contextual_chunk_corpus,
)
from src.public_dataset_eval import (
    PublicCorpus,
    extract_article_refs,
    normalize_space,
    prepare_docling_artifacts,
    safe_json_dump,
    safe_json_load,
)
from src.public_retrieval_benchmark import run_strategy
from src.query_analysis import QuestionAnalysis, QuestionAnalyzer


BASELINE_PACK_PATH = Path("/home/mkgs/hackaton/starter_kit/challenge_workdir/review_pack_local__local_jina__standard.json")
BASELINE_AUDIT_PATH = Path("/home/mkgs/hackaton/starter_kit/submission_history/20260312_011749/WARMUP_RERANKER_MANUAL_AUDIT.csv")
VARIANTS = [
    "contextual_retrieval",
    "late_interaction",
    "atomic_fact_index",
    "corrective_retrieval",
    "evidence_first",
    "verification_pass",
    "hierarchical_retrieval",
    "citation_selector",
    "typed_policy_router",
    "multi_query_expansion",
]

STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "to",
    "in",
    "on",
    "for",
    "under",
    "with",
    "what",
    "which",
    "who",
    "when",
    "where",
    "is",
    "was",
    "were",
    "be",
    "that",
    "this",
    "those",
    "these",
    "there",
    "their",
    "according",
}

ORDER_MARKERS = (
    "it is hereby ordered that",
    "order with reasons",
    "permission to appeal",
    "application is dismissed",
    "application is refused",
)

ARTICLE_RE = re.compile(r"article\s+(\d+)", re.I)
TOKEN_RE = re.compile(r"[A-Za-z0-9/.-]+")
HEADING_RE = re.compile(r"(?:^|\n)##\s+([^\n]+)")


class CitationSelection(BaseModel):
    selected_refs: List[str] = Field(default_factory=list)
    coverage: str = "partial"
    rationale: str = ""


class VerificationDecision(BaseModel):
    supported: bool = False
    missing_aspects: List[str] = Field(default_factory=list)
    preferred_refs: List[str] = Field(default_factory=list)
    rationale: str = ""


def ordered_unique(values: Iterable[Any]) -> List[Any]:
    result: List[Any] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def normalized_text(value: Any) -> str:
    return normalize_space(str(value or "")).lower()


def preview(text: str, limit: int = 220) -> str:
    text = normalize_space(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def reciprocal_rank_fuse(
    ranked_lists: Sequence[tuple[str, Sequence[Dict[str, Any]], float]],
    fusion_k: int = 60,
) -> List[Dict[str, Any]]:
    fused: Dict[str, Dict[str, Any]] = {}
    max_rrf_score = sum(weight / (fusion_k + 1) for _, _, weight in ranked_lists if weight > 0)
    max_rrf_score = max_rrf_score or 1.0
    for source_name, results, weight in ranked_lists:
        for rank, item in enumerate(results, start=1):
            ref = item["ref"]
            record = fused.setdefault(ref, dict(item))
            record.setdefault("retrieval_sources", [])
            record["retrieval_sources"].append(source_name)
            record.setdefault("rrf_score", 0.0)
            record["rrf_score"] += weight / (fusion_k + rank)
    merged = list(fused.values())
    for item in merged:
        item["retrieval_sources"] = sorted(set(item.get("retrieval_sources", [])))
        item["distance"] = round(float(item["rrf_score"]) / max_rrf_score, 4)
    merged.sort(key=lambda row: row["distance"], reverse=True)
    return merged


def tokenize_keywords(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        text = " ".join(str(item) for item in value)
    else:
        text = str(value)
    tokens = [token.lower() for token in TOKEN_RE.findall(text) if len(token) > 2 and token.lower() not in STOPWORDS]
    return ordered_unique(tokens)


def significant_title_tokens(value: str) -> set[str]:
    return {token for token in tokenize_keywords(value) if not (token.isdigit() and len(token) == 4)}


def article_reference_score(text: str, article_refs: Sequence[str]) -> float:
    text_norm = normalized_text(text)
    score = 0.0
    for article_ref in article_refs:
        ref_norm = normalized_text(article_ref)
        if ref_norm and ref_norm in text_norm:
            score += 24.0
        match = ARTICLE_RE.search(article_ref or "")
        if match and re.search(rf"(?:^|\b|##\s*){match.group(1)}\.", text_norm):
            score += 16.0
        for part in re.findall(r"\(([^)]+)\)", article_ref or ""):
            if re.search(rf"\(\s*{re.escape(part.lower())}\s*\)", text_norm):
                score += 6.0
    return score


def parse_baseline_audit(path: Path) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[str(row["q_index"]).strip()] = row
    return rows


def exact_ref_signature(chunks: Sequence[Dict[str, Any]]) -> str:
    return " | ".join(str(chunk["ref"]) for chunk in chunks[:5])


def title_signature(titles: Sequence[str]) -> str:
    return " | ".join(str(title) for title in titles[:5])


def title_set_signature(titles: Sequence[str]) -> str:
    return " | ".join(sorted(set(str(title) for title in titles[:5])))


def split_interaction_units(text: str) -> List[str]:
    units: List[str] = []
    for block in re.split(r"\n{2,}", text or ""):
        block = normalize_space(block)
        if not block:
            continue
        if block.startswith(("-", "•")) or re.match(r"^\(?[a-z0-9]+\)", block, re.I):
            units.append(block)
            continue
        pieces = re.split(r"(?<=[\.\?\!;:])\s+(?=[A-Z0-9(\[])", block)
        units.extend(normalize_space(piece) for piece in pieces if len(normalize_space(piece)) >= 24)
    return ordered_unique(unit for unit in units if len(unit) >= 24)[:8]


class LateInteractionReranker:
    def __init__(self, corpus: PublicCorpus):
        self.corpus = corpus
        self._unit_cache: Dict[str, np.ndarray] = {}

    def _unit_embeddings(self, ref: str, text: str) -> np.ndarray:
        cached = self._unit_cache.get(ref)
        if cached is not None:
            return cached
        units = split_interaction_units(text)
        if not units:
            units = [normalize_space(text)]
        embeddings = self.corpus.embedder.embed_documents(units)
        self._unit_cache[ref] = embeddings
        return embeddings

    def rerank(self, query: str, candidates: Sequence[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.corpus.embedder.embed_query(query)
        rescored: List[Dict[str, Any]] = []
        for row in candidates:
            unit_embeddings = self._unit_embeddings(row["ref"], row["text"])
            unit_score = float(np.max(unit_embeddings @ query_embedding)) if len(unit_embeddings) else 0.0
            base_score = float(row.get("combined_score", row.get("relevance_score", row.get("distance", 0.0))))
            rescored_row = dict(row)
            rescored_row["late_interaction_score"] = round(unit_score, 4)
            rescored_row["combined_score"] = round(0.55 * unit_score + 0.45 * base_score, 4)
            rescored.append(rescored_row)
        rescored.sort(key=lambda item: item["combined_score"], reverse=True)
        return rescored[:limit]


@dataclass
class PageRow:
    page_ref: str
    sha: str
    page: int
    title: str
    text: str
    canonical_ids: List[str]


class PageIndex:
    def __init__(self, corpus: PublicCorpus, cache_dir: Path):
        self.corpus = corpus
        self.cache_dir = cache_dir
        self.page_rows: List[PageRow] = []
        self.page_ref_to_row: Dict[str, PageRow] = {}
        self.page_embeddings = self._load_or_build_page_embeddings()
        self.doc_embeddings = self._load_or_build_doc_embeddings()

    def _build_page_rows(self) -> List[PageRow]:
        rows: List[PageRow] = []
        for sha, payload in self.corpus.documents_payload.items():
            document = self.corpus.documents[sha]
            for page in payload["content"]["pages"]:
                page_number = int(page["page"])
                text = str(page.get("text", "") or "")
                page_text = f"[DOC_TITLE] {document.title}\n[PAGE] {page_number}\n{text}"
                row = PageRow(
                    page_ref=f"{sha}:{page_number}",
                    sha=sha,
                    page=page_number,
                    title=document.title,
                    text=page_text,
                    canonical_ids=document.canonical_ids,
                )
                rows.append(row)
        return rows

    def _load_or_build_page_embeddings(self) -> np.ndarray:
        self.page_rows = self._build_page_rows()
        self.page_ref_to_row = {row.page_ref: row for row in self.page_rows}
        cache_slug = re.sub(r"[^a-zA-Z0-9]+", "_", getattr(self.corpus.embedder, "model_name", DEFAULT_LOCAL_EMBEDDING_MODEL)).strip("_").lower()
        signature = hashlib.sha1("\n".join(row.page_ref for row in self.page_rows).encode("utf-8")).hexdigest()[:12]
        meta_path = self.cache_dir / f"{cache_slug}__{signature}__page_meta.json"
        emb_path = self.cache_dir / f"{cache_slug}__{signature}__page_embeddings.npy"
        if meta_path.exists() and emb_path.exists():
            cached = safe_json_load(meta_path)
            if cached.get("page_refs") == [row.page_ref for row in self.page_rows]:
                return np.load(emb_path)
        embeddings = self.corpus.embedder.embed_documents([row.text for row in self.page_rows])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, embeddings)
        safe_json_dump(meta_path, {"page_refs": [row.page_ref for row in self.page_rows]})
        return embeddings

    def _load_or_build_doc_embeddings(self) -> np.ndarray:
        cache_slug = re.sub(r"[^a-zA-Z0-9]+", "_", getattr(self.corpus.embedder, "model_name", DEFAULT_LOCAL_EMBEDDING_MODEL)).strip("_").lower()
        ordered_docs = [self.corpus.documents[sha] for sha in sorted(self.corpus.documents)]
        signature = hashlib.sha1("\n".join(doc.sha for doc in ordered_docs).encode("utf-8")).hexdigest()[:12]
        meta_path = self.cache_dir / f"{cache_slug}__{signature}__doc_meta.json"
        emb_path = self.cache_dir / f"{cache_slug}__{signature}__doc_embeddings.npy"
        if meta_path.exists() and emb_path.exists():
            cached = safe_json_load(meta_path)
            if cached.get("shas") == [doc.sha for doc in ordered_docs]:
                return np.load(emb_path)
        texts = [f"{doc.title}\n{' '.join(doc.aliases[:6])}\n{doc.first_page_text[:2000]}" for doc in ordered_docs]
        embeddings = self.corpus.embedder.embed_documents(texts)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, embeddings)
        safe_json_dump(meta_path, {"shas": [doc.sha for doc in ordered_docs]})
        return embeddings

    def top_docs(self, question: str, candidate_shas: Sequence[str], limit: int = 4) -> List[str]:
        ordered_docs = [self.corpus.documents[sha] for sha in sorted(self.corpus.documents)]
        query_embedding = self.corpus.embedder.embed_query(question)
        scores = self.doc_embeddings @ query_embedding
        rows = []
        candidate_set = set(candidate_shas)
        for index, document in enumerate(ordered_docs):
            if document.sha not in candidate_set:
                continue
            rows.append((float(scores[index]), document.sha))
        rows.sort(key=lambda item: item[0], reverse=True)
        return [sha for _, sha in rows[:limit]]

    def top_pages(self, question: str, shas: Sequence[str], limit: int = 8) -> List[PageRow]:
        query_embedding = self.corpus.embedder.embed_query(question)
        scores = self.page_embeddings @ query_embedding
        rows = []
        target = set(shas)
        for index, page_row in enumerate(self.page_rows):
            if page_row.sha not in target:
                continue
            rows.append((float(scores[index]), page_row))
        rows.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in rows[:limit]]


class StrategyRunner:
    def __init__(
        self,
        *,
        work_dir: Path,
        pdf_dir: Path,
        questions_path: Path,
        provider: str,
        model: str,
    ):
        self.work_dir = work_dir
        self.pdf_dir = pdf_dir
        self.questions_path = questions_path
        self.provider = provider
        self.model = model
        self.api = APIProcessor(provider=provider)
        self.analysis_cache = work_dir / "production_ideas" / "query_analysis" / f"{provider}__{model}.json"
        self.analyzer = QuestionAnalyzer(provider=provider, model=model, cache_path=self.analysis_cache)
        self.citation_cache_path = work_dir / "production_ideas" / "citation_selector" / f"{provider}__{model}.json"
        self.verification_cache_path = work_dir / "production_ideas" / "verification" / f"{provider}__{model}.json"
        self.citation_cache = self._load_cache(self.citation_cache_path)
        self.verification_cache = self._load_cache(self.verification_cache_path)
        artifacts = prepare_docling_artifacts(work_dir, pdf_dir)
        self.standard_chunked_dir = artifacts["chunked_dir"]
        self.contextual_chunked_dir = build_contextual_chunk_corpus(
            chunked_dir=self.standard_chunked_dir,
            output_dir=work_dir / "docling" / "chunked_contextual",
        )
        self.atomic_chunked_dir = build_atomic_fact_chunk_corpus(
            chunked_dir=self.standard_chunked_dir,
            output_dir=work_dir / "docling" / "chunked_atomic_facts",
        )
        self._corpora: Dict[str, PublicCorpus] = {}
        self._page_indices: Dict[str, PageIndex] = {}
        self._late_rerankers: Dict[str, LateInteractionReranker] = {}

    def _load_cache(self, path: Path) -> Dict[str, Any]:
        if path.exists():
            try:
                return safe_json_load(path)
            except Exception:
                return {}
        return {}

    def _save_cache(self, path: Path, payload: Dict[str, Any]) -> None:
        safe_json_dump(path, payload)

    def corpus(self, kind: str) -> PublicCorpus:
        if kind in self._corpora:
            return self._corpora[kind]
        if kind == "standard":
            chunked_dir = self.standard_chunked_dir
        elif kind == "contextual":
            chunked_dir = self.contextual_chunked_dir
        elif kind == "atomic":
            chunked_dir = self.atomic_chunked_dir
        else:
            raise ValueError(f"Unknown corpus kind: {kind}")
        corpus = PublicCorpus(
            work_dir=self.work_dir / "production_ideas" / kind,
            pdf_dir=self.pdf_dir,
            chunked_dir=chunked_dir,
            embedding_model=DEFAULT_LOCAL_EMBEDDING_MODEL,
            reranker_model=DEFAULT_LOCAL_RERANKER_MODEL,
        )
        self._corpora[kind] = corpus
        return corpus

    def page_index(self, corpus_kind: str) -> PageIndex:
        if corpus_kind not in self._page_indices:
            corpus = self.corpus(corpus_kind)
            self._page_indices[corpus_kind] = PageIndex(
                corpus=corpus,
                cache_dir=self.work_dir / "production_ideas" / corpus_kind / "page_index",
            )
        return self._page_indices[corpus_kind]

    def late_reranker(self, corpus_kind: str) -> LateInteractionReranker:
        if corpus_kind not in self._late_rerankers:
            self._late_rerankers[corpus_kind] = LateInteractionReranker(self.corpus(corpus_kind))
        return self._late_rerankers[corpus_kind]

    def vector_candidates(
        self,
        corpus: PublicCorpus,
        question: str,
        candidate_shas: Sequence[str],
        vector_k: int = 16,
    ) -> List[Dict[str, Any]]:
        candidate_indices = corpus._candidate_chunk_indices(candidate_shas)
        query_embedding = corpus.embedder.embed_query(question)
        candidate_embeddings = corpus.chunk_embeddings[candidate_indices]
        scores = candidate_embeddings @ query_embedding
        top_order = np.argsort(scores)[::-1][: min(vector_k, len(candidate_indices))]
        rows = []
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
                }
            )
        return rows

    def multi_query_variants(self, analysis: QuestionAnalysis, question: str) -> List[str]:
        variants = [question, analysis.standalone_question, analysis.retrieval_query]
        explicit_parts = ordered_unique(
            list(analysis.target_case_ids)
            + list(analysis.target_law_ids)
            + list(analysis.target_titles)
            + list(analysis.target_article_refs)
            + list(analysis.must_support_terms)
        )
        if explicit_parts:
            variants.append(" ".join(explicit_parts))
        if analysis.target_field == "order_result":
            variants.append(f"{analysis.standalone_question} IT IS HEREBY ORDERED THAT conclusion order result")
        if analysis.target_field == "article_content" and analysis.target_article_refs:
            variants.append(f"{analysis.standalone_question} {' '.join(analysis.target_article_refs)}")
        return [normalize_space(variant) for variant in ordered_unique(variants) if normalize_space(variant)]

    def _route_with_titles(self, corpus: PublicCorpus, analysis: QuestionAnalysis, question: str) -> Dict[str, Any]:
        route = corpus.route_question(analysis.retrieval_query or question, expansive=False)
        candidate_shas = list(route["candidate_shas"])
        question_norm = normalized_text(question)
        for sha, document in corpus.documents.items():
            doc_tokens = significant_title_tokens(document.title)
            for target in analysis.target_titles:
                target_tokens = significant_title_tokens(target)
                if target_tokens and len(doc_tokens & target_tokens) >= min(2, len(target_tokens)):
                    candidate_shas.append(sha)
            if any(case_id.lower() in question_norm for case_id in [item.lower() for item in document.canonical_ids]):
                candidate_shas.append(sha)
        route["candidate_shas"] = ordered_unique(candidate_shas)
        return route

    def hierarchical_retrieve(
        self,
        corpus: PublicCorpus,
        analysis: QuestionAnalysis,
        question: str,
    ) -> List[Dict[str, Any]]:
        route = self._route_with_titles(corpus, analysis, question)
        page_index = self.page_index("standard")
        top_docs = ordered_unique(route["candidate_shas"][:2] + page_index.top_docs(analysis.retrieval_query or question, route["candidate_shas"], limit=4))
        if analysis.needs_multi_document_support and len(route["candidate_shas"]) >= 2:
            top_docs = ordered_unique(route["candidate_shas"][:2] + top_docs)
        page_rows = page_index.top_pages(analysis.retrieval_query or question, top_docs, limit=10)
        allowed_pages = {(row.sha, row.page) for row in page_rows}
        if analysis.target_field in {"law_number", "made_by", "administered_by", "publication_text", "enacted_text"}:
            for sha in top_docs:
                allowed_pages.add((sha, 1))
        if "last_page" in analysis.support_focus or analysis.target_field == "order_result":
            for sha in top_docs:
                allowed_pages.add((sha, corpus.documents[sha].page_count))
        candidate_chunks = [
            {
                "ref": chunk.ref,
                "distance": 0.5,
                "page": chunk.page,
                "text": chunk.text,
                "sha": chunk.sha,
                "title": chunk.title,
                "kind": chunk.kind,
                "canonical_ids": chunk.canonical_ids,
            }
            for chunk in corpus.chunks
            if (chunk.sha, chunk.page) in allowed_pages
        ]
        reranked = corpus.reranker.rerank_documents(analysis.retrieval_query or question, candidate_chunks, llm_weight=0.7)
        return reranked[:5]

    def multi_query_retrieve(
        self,
        corpus: PublicCorpus,
        analysis: QuestionAnalysis,
        question: str,
    ) -> List[Dict[str, Any]]:
        route = self._route_with_titles(corpus, analysis, question)
        ranked_lists = []
        for rank, query in enumerate(self.multi_query_variants(analysis, question), start=1):
            results = self.vector_candidates(corpus, query, route["candidate_shas"], vector_k=12)
            ranked_lists.append((f"q{rank}", results, max(0.4, 1.0 - (rank - 1) * 0.12)))
        fused = reciprocal_rank_fuse(ranked_lists)
        return corpus.reranker.rerank_documents(question, fused, llm_weight=0.7)[:5]

    def corrective_retrieve(
        self,
        corpus: PublicCorpus,
        analysis: QuestionAnalysis,
        question: str,
    ) -> List[Dict[str, Any]]:
        route = self._route_with_titles(corpus, analysis, question)
        baseline = run_strategy(corpus, corpus.bm25_index, analysis.retrieval_query or question, route["candidate_shas"], "dense_doc_diverse")
        enough_docs = len(ordered_unique(chunk["sha"] for chunk in baseline[:5])) >= (2 if analysis.needs_multi_document_support else 1)
        explicit_hits = True
        if analysis.target_case_ids or analysis.target_law_ids:
            targets = set(analysis.target_case_ids + analysis.target_law_ids)
            explicit_hits = any(set(item.get("canonical_ids") or []) & targets for item in baseline[:5])
        article_supported = True
        if analysis.target_field == "article_content" and analysis.target_article_refs:
            article_supported = any(article_reference_score(chunk["text"], analysis.target_article_refs) >= 16 for chunk in baseline[:5])
        order_supported = True
        if analysis.target_field == "order_result":
            order_supported = any(any(marker in normalized_text(chunk["text"]) for marker in ORDER_MARKERS) for chunk in baseline[:5])
        if enough_docs and explicit_hits and article_supported and order_supported:
            return baseline[:5]

        expansion_queries = self.multi_query_variants(analysis, question)
        if analysis.must_support_terms:
            expansion_queries.append(f"{question} {' '.join(analysis.must_support_terms)}")
        if analysis.target_field == "order_result":
            expansion_queries.append(f"{question} IT IS HEREBY ORDERED THAT")
        ranked_lists = [("baseline", baseline[:8], 1.0)]
        for rank, query in enumerate(ordered_unique(expansion_queries), start=1):
            results = self.vector_candidates(corpus, query, route["candidate_shas"], vector_k=14)
            ranked_lists.append((f"exp{rank}", results, 0.85))
        bm25_results = corpus.retrieve(
            question=question,
            candidate_shas=route["candidate_shas"],
            vector_k=16,
            rerank_k=8,
            bm25_k=8,
            bm25_weight=0.35,
            bm25_auto=True,
        )["vector_results"][:8]
        ranked_lists.append(("bm25", bm25_results, 0.55))
        fused = reciprocal_rank_fuse(ranked_lists)
        return corpus.reranker.rerank_documents(question, fused, llm_weight=0.7)[:5]

    def evidence_first_select(
        self,
        corpus: PublicCorpus,
        analysis: QuestionAnalysis,
        question: str,
        candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        title_tokens = set()
        for title in analysis.target_titles:
            title_tokens.update(significant_title_tokens(title))
        question_tokens = set(tokenize_keywords(question))
        scored = []
        for rank, chunk in enumerate(candidates[:8], start=1):
            text_norm = normalized_text(chunk["text"])
            score = float(chunk.get("combined_score", chunk.get("relevance_score", chunk.get("distance", 0.0))))
            score += article_reference_score(chunk["text"], analysis.target_article_refs)
            score += 8.0 * sum(1 for term in analysis.must_support_terms if normalized_text(term) in text_norm)
            score += 3.0 * len(question_tokens & set(tokenize_keywords(chunk["text"])) & set(tokenize_keywords(" ".join(analysis.must_support_terms))))
            if title_tokens:
                score += 6.0 * len(title_tokens & significant_title_tokens(chunk["title"]))
            if "first_page" in analysis.support_focus or "title_page" in analysis.support_focus:
                if int(chunk["page"]) == 1:
                    score += 18.0
            if "last_page" in analysis.support_focus and int(chunk["page"]) == corpus.documents[chunk["sha"]].page_count:
                score += 18.0
            if analysis.target_field == "order_result" and any(marker in text_norm for marker in ORDER_MARKERS):
                score += 16.0
            if analysis.target_field in {"law_number", "made_by", "administered_by", "publication_text", "enacted_text"} and int(chunk["page"]) == 1:
                score += 14.0
            scored.append((score - rank * 0.01, dict(chunk)))
        scored.sort(key=lambda item: item[0], reverse=True)

        selected: List[Dict[str, Any]] = []
        seen_refs = set()
        covered_docs = set()
        expected_docs = max(1, min(2, len(analysis.target_case_ids) + len(analysis.target_law_ids) + len(analysis.target_titles)))
        for score, chunk in scored:
            if chunk["ref"] in seen_refs:
                continue
            if analysis.needs_multi_document_support and len(covered_docs) < expected_docs:
                if chunk["sha"] in covered_docs and any(other["sha"] != chunk["sha"] for _, other in scored):
                    continue
            selected.append(chunk)
            seen_refs.add(chunk["ref"])
            covered_docs.add(chunk["sha"])
            if len(selected) >= 5:
                break
        if not selected:
            return list(candidates[:5])
        return selected[:5]

    def _citation_cache_key(self, question: str, refs: Sequence[str]) -> str:
        raw = "\n".join([self.provider, self.model, question, *refs])
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def citation_select(
        self,
        question: str,
        chunks: Sequence[Dict[str, Any]],
    ) -> List[str]:
        refs = [chunk["ref"] for chunk in chunks[:6]]
        key = self._citation_cache_key(question, refs)
        cached = self.citation_cache.get(key)
        if cached:
            return [ref for ref in cached.get("selected_refs", []) if ref in refs][:5]
        context_blocks = []
        for chunk in chunks[:6]:
            context_blocks.append(
                f"[REF {chunk['ref']} | title={chunk['title']} | page={chunk['page']}]\n{chunk['text']}"
            )
        try:
            response = self.api.send_message(
                model=self.model,
                temperature=0.0,
                system_content=(
                    "You are selecting supporting citations for a DIFC legal RAG system. "
                    "Return only exact REF ids that directly support answering the question. "
                    "Prefer the smallest sufficient set. Do not invent refs."
                ),
                human_content=(
                    f"Question: {question}\n\n"
                    f"Context:\n\n{'\n\n'.join(context_blocks)}\n\n"
                    "Return JSON with selected_refs, coverage, rationale."
                ),
                is_structured=True,
                response_format=CitationSelection,
                max_tokens=180,
                request_timeout=60,
            )
            payload = CitationSelection.model_validate(response).model_dump()
        except Exception:
            payload = {"selected_refs": [], "coverage": "partial", "rationale": "fallback"}
        self.citation_cache[key] = payload
        self._save_cache(self.citation_cache_path, self.citation_cache)
        return [ref for ref in payload.get("selected_refs", []) if ref in refs][:5]

    def _verification_cache_key(self, question: str, refs: Sequence[str]) -> str:
        raw = "\n".join([self.provider, self.model, "verification", question, *refs])
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def verification_select(
        self,
        corpus: PublicCorpus,
        analysis: QuestionAnalysis,
        question: str,
        chunks: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        refs = [chunk["ref"] for chunk in chunks[:6]]
        key = self._verification_cache_key(question, refs)
        cached = self.verification_cache.get(key)
        if cached:
            payload = cached
        else:
            context_blocks = []
            for chunk in chunks[:6]:
                context_blocks.append(
                    f"[REF {chunk['ref']} | title={chunk['title']} | page={chunk['page']}]\n{chunk['text']}"
                )
            try:
                response = self.api.send_message(
                    model=self.model,
                    temperature=0.0,
                    system_content=(
                        "You are verifying whether the retrieved DIFC legal passages are sufficient to answer the question. "
                        "Be conservative. If a key aspect is missing, mark supported=false and list the missing aspects. "
                        "Also list any preferred_refs already present in the context."
                    ),
                    human_content=(
                        f"Question: {question}\n\n"
                        f"Context:\n\n{'\n\n'.join(context_blocks)}\n\n"
                        "Return JSON with supported, missing_aspects, preferred_refs, rationale."
                    ),
                    is_structured=True,
                    response_format=VerificationDecision,
                    max_tokens=180,
                    request_timeout=60,
                )
                payload = VerificationDecision.model_validate(response).model_dump()
            except Exception:
                payload = {
                    "supported": True,
                    "missing_aspects": [],
                    "preferred_refs": [],
                    "rationale": "fallback",
                }
            self.verification_cache[key] = payload
            self._save_cache(self.verification_cache_path, self.verification_cache)

        preferred_refs = [ref for ref in payload.get("preferred_refs", []) if ref in refs]
        if payload.get("supported") and preferred_refs:
            preferred = [chunk for chunk in chunks if chunk["ref"] in preferred_refs]
            if preferred:
                return preferred[:5]

        if payload.get("supported"):
            return list(chunks[:5])

        expansion_terms = [normalize_space(item) for item in payload.get("missing_aspects", []) if normalize_space(item)]
        if not expansion_terms:
            return list(chunks[:5])
        route = self._route_with_titles(corpus, analysis, question)
        expanded_query = f"{question} {' '.join(expansion_terms[:3])}"
        extra = self.vector_candidates(corpus, expanded_query, route["candidate_shas"], vector_k=14)
        fused = reciprocal_rank_fuse([("initial", chunks[:8], 1.0), ("expanded", extra, 0.85)])
        reranked = corpus.reranker.rerank_documents(question, fused, llm_weight=0.7)
        return self.evidence_first_select(corpus, analysis, question, reranked)

    def late_interaction_retrieve(
        self,
        corpus: PublicCorpus,
        question: str,
        candidate_shas: Sequence[str],
    ) -> List[Dict[str, Any]]:
        initial = corpus.retrieve(question=question, candidate_shas=candidate_shas, vector_k=18, rerank_k=12)
        return self.late_reranker("standard").rerank(question, initial["reranked_results"], limit=5)

    def variant_output(self, variant: str, item: Dict[str, Any]) -> Dict[str, Any]:
        question = item["question"]
        answer_type = item["answer_type"]
        analysis = self.analyzer.analyze(question, answer_type)

        if variant == "contextual_retrieval":
            corpus = self.corpus("contextual")
            route = self._route_with_titles(corpus, analysis, question)
            top_chunks = run_strategy(corpus, corpus.bm25_index, analysis.retrieval_query or question, route["candidate_shas"], "dense_doc_diverse")
            method = "contextual_dense_doc_diverse"
        elif variant == "late_interaction":
            corpus = self.corpus("standard")
            route = self._route_with_titles(corpus, analysis, question)
            top_chunks = self.late_interaction_retrieve(corpus, question, route["candidate_shas"])
            method = "dense_doc_diverse_plus_late_interaction"
        elif variant == "atomic_fact_index":
            corpus = self.corpus("atomic")
            route = self._route_with_titles(corpus, analysis, question)
            top_chunks = run_strategy(corpus, corpus.bm25_index, analysis.retrieval_query or question, route["candidate_shas"], "dense_doc_diverse")
            method = "atomic_fact_dense_doc_diverse"
        elif variant == "corrective_retrieval":
            corpus = self.corpus("standard")
            top_chunks = self.corrective_retrieve(corpus, analysis, question)
            method = "corrective_retrieval"
        elif variant == "evidence_first":
            corpus = self.corpus("standard")
            route = self._route_with_titles(corpus, analysis, question)
            initial = run_strategy(corpus, corpus.bm25_index, analysis.retrieval_query or question, route["candidate_shas"], "dense_doc_diverse")
            top_chunks = self.evidence_first_select(corpus, analysis, question, initial)
            method = "evidence_first_support_selection"
        elif variant == "verification_pass":
            corpus = self.corpus("standard")
            route = self._route_with_titles(corpus, analysis, question)
            initial = run_strategy(corpus, corpus.bm25_index, analysis.retrieval_query or question, route["candidate_shas"], "dense_doc_diverse")
            focused = self.evidence_first_select(corpus, analysis, question, initial)
            top_chunks = self.verification_select(corpus, analysis, question, focused)
            method = "evidence_verification_and_retrieval"
        elif variant == "hierarchical_retrieval":
            corpus = self.corpus("standard")
            top_chunks = self.hierarchical_retrieve(corpus, analysis, question)
            method = "hierarchical_doc_page_chunk"
        elif variant == "citation_selector":
            corpus = self.corpus("standard")
            route = self._route_with_titles(corpus, analysis, question)
            initial = run_strategy(corpus, corpus.bm25_index, analysis.retrieval_query or question, route["candidate_shas"], "dense_doc_diverse")
            selected_refs = self.citation_select(question, initial)
            top_chunks = [chunk for chunk in initial if chunk["ref"] in selected_refs] or list(initial[:5])
            method = "structured_citation_selector"
        elif variant == "typed_policy_router":
            corpus = self.corpus("standard")
            if analysis.target_field in {"law_number", "made_by", "administered_by", "publication_text", "enacted_text"}:
                top_chunks = self.hierarchical_retrieve(corpus, analysis, question)
                top_chunks = self.evidence_first_select(corpus, analysis, question, top_chunks)
                method = "typed_router:hierarchical+evidence"
            elif analysis.target_field in {"article_content", "order_result"}:
                top_chunks = self.corrective_retrieve(corpus, analysis, question)
                top_chunks = self.evidence_first_select(corpus, analysis, question, top_chunks)
                method = "typed_router:corrective+evidence"
            elif analysis.needs_multi_document_support:
                top_chunks = self.multi_query_retrieve(corpus, analysis, question)
                top_chunks = self.evidence_first_select(corpus, analysis, question, top_chunks)
                method = "typed_router:multi_query+evidence"
            else:
                contextual = self.corpus("contextual")
                route = self._route_with_titles(contextual, analysis, question)
                top_chunks = run_strategy(contextual, contextual.bm25_index, analysis.retrieval_query or question, route["candidate_shas"], "dense_doc_diverse")
                method = "typed_router:contextual"
        elif variant == "multi_query_expansion":
            corpus = self.corpus("standard")
            top_chunks = self.multi_query_retrieve(corpus, analysis, question)
            method = "multi_query_expansion_rrf"
        else:
            raise ValueError(f"Unknown variant: {variant}")

        packed_chunks = []
        for rank, chunk in enumerate(top_chunks[:5], start=1):
            packed_chunks.append(
                {
                    "rank": rank,
                    "title": chunk["title"],
                    "page": int(chunk["page"]),
                    "ref": chunk["ref"],
                    "sha": chunk["sha"],
                    "distance": float(chunk.get("distance", 0.0)),
                    "relevance_score": float(chunk.get("relevance_score", 0.0)),
                    "combined_score": float(chunk.get("combined_score", chunk.get("distance", 0.0))),
                    "canonical_ids": chunk.get("canonical_ids", []),
                    "preview": preview(chunk.get("text", "")),
                }
            )

        return {
            "id": item["id"],
            "question": question,
            "answer_type": answer_type,
            "analysis": analysis.model_dump(),
            "method": method,
            "top_chunks": packed_chunks,
            "top_titles": [chunk["title"] for chunk in packed_chunks],
        }


def build_pack(
    *,
    runner: StrategyRunner,
    variant: str,
    output_dir: Path,
) -> Dict[str, Any]:
    questions = safe_json_load(runner.questions_path)
    rows = []
    for index, item in enumerate(questions, start=1):
        row = runner.variant_output(variant, item)
        row["index"] = index
        rows.append(row)
    payload = {
        "config": {
            "variant": variant,
            "provider": runner.provider,
            "model": runner.model,
        },
        "summary": {
            "question_count": len(rows),
        },
        "questions": rows,
    }
    safe_json_dump(output_dir / f"review_pack_{variant}.json", payload)
    return payload


def load_pack(path: Path) -> Dict[str, Any]:
    return safe_json_load(path)


def build_variant_audit(
    *,
    baseline_pack: Dict[str, Any],
    baseline_audit_rows: Dict[str, Dict[str, str]],
    candidate_pack: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    baseline_by_id = {row["id"]: row for row in baseline_pack["questions"]}
    audit_rows = []
    changed_rows = []
    inherited = 0

    for row in candidate_pack["questions"]:
        index = str(row["index"])
        baseline_row = baseline_by_id[row["id"]]
        baseline_audit = baseline_audit_rows.get(index, {})
        same_title_signature = title_signature(row.get("top_titles", [])) == title_signature(baseline_row.get("top_titles", []))
        same_title_set_signature = title_set_signature(row.get("top_titles", [])) == title_set_signature(baseline_row.get("top_titles", []))
        same_signature = exact_ref_signature(row["top_chunks"]) == exact_ref_signature(baseline_row["top_chunks"])
        inheritable = same_title_signature or same_title_set_signature or same_signature
        manual_relevance = baseline_audit.get("reranker_top5_relevance", "") if inheritable else ""
        manual_comment = baseline_audit.get("comment", "") if inheritable else ""
        if inheritable:
            inherited += 1
        else:
            changed_rows.append(
                {
                    "index": row["index"],
                    "question": row["question"],
                    "baseline_label": baseline_audit.get("reranker_top5_relevance", ""),
                    "baseline_comment": baseline_audit.get("comment", ""),
                    "baseline_titles": " | ".join(baseline_row.get("top_titles", [])[:5]),
                    "candidate_titles": " | ".join(row.get("top_titles", [])[:5]),
                    "baseline_refs": exact_ref_signature(baseline_row["top_chunks"]),
                    "candidate_refs": exact_ref_signature(row["top_chunks"]),
                    "baseline_top1_preview": baseline_row.get("top_chunks", [{}])[0].get("preview", "") if baseline_row.get("top_chunks") else "",
                    "candidate_top1_preview": row.get("top_chunks", [{}])[0].get("preview", "") if row.get("top_chunks") else "",
                }
            )
        audit_rows.append(
            {
                "q_index": row["index"],
                "answer_type": row["answer_type"],
                "question": row["question"],
                "manual_relevance": manual_relevance,
                "manual_comment": manual_comment,
                "inherited_from_baseline": "yes" if inheritable else "no",
                "baseline_label": baseline_audit.get("reranker_top5_relevance", ""),
                "baseline_titles": " | ".join(baseline_row.get("top_titles", [])[:5]),
                "candidate_titles": " | ".join(row.get("top_titles", [])[:5]),
                "baseline_refs": exact_ref_signature(baseline_row["top_chunks"]),
                "candidate_refs": exact_ref_signature(row["top_chunks"]),
                "baseline_top1_preview": baseline_row.get("top_chunks", [{}])[0].get("preview", "") if baseline_row.get("top_chunks") else "",
                "candidate_top1_preview": row.get("top_chunks", [{}])[0].get("preview", "") if row.get("top_chunks") else "",
                "method": row.get("method", ""),
            }
        )

    variant = candidate_pack["config"]["variant"]
    csv_path = output_dir / f"audit_{variant}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(audit_rows)

    md_path = output_dir / f"changed_{variant}.md"
    lines = [
        f"# Changed rows for `{variant}`",
        "",
        f"- Questions: `{len(candidate_pack['questions'])}`",
        f"- Inherited labels from baseline: `{inherited}`",
        f"- Changed rows requiring manual review: `{len(changed_rows)}`",
        "",
    ]
    for row in changed_rows:
        lines.extend(
            [
                f"## Q{int(row['index']):03d}",
                f"- Question: {row['question']}",
                f"- Baseline label: `{row['baseline_label']}`",
                f"- Baseline comment: {row['baseline_comment']}",
                f"- Baseline titles: `{row['baseline_titles']}`",
                f"- Candidate titles: `{row['candidate_titles']}`",
                f"- Baseline refs: `{row['baseline_refs']}`",
                f"- Candidate refs: `{row['candidate_refs']}`",
                f"- Baseline top1: {row['baseline_top1_preview']}",
                f"- Candidate top1: {row['candidate_top1_preview']}",
                "",
            ]
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "variant": variant,
        "question_count": len(candidate_pack["questions"]),
        "inherited_from_baseline": inherited,
        "changed_rows": len(changed_rows),
        "audit_csv": str(csv_path),
        "changed_md": str(md_path),
    }
    safe_json_dump(output_dir / f"audit_{variant}_summary.json", summary)
    return summary


def summarize_filled_audit(path: Path) -> Dict[str, Any]:
    counts = {"sufficient": 0, "weak": 0, "miss": 0, "blank": 0}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = normalized_text(row.get("manual_relevance", ""))
            if value in counts:
                counts[value] += 1
            else:
                counts["blank"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run production-style warm-up retrieval idea experiments and generate manual audit packs.")
    parser.add_argument("--work-dir", default="/home/mkgs/hackaton/starter_kit/challenge_workdir")
    parser.add_argument("--pdf-dir", default="/home/mkgs/hackaton/starter_kit/docs_corpus")
    parser.add_argument("--questions-path", default="/home/mkgs/hackaton/starter_kit/questions_api.json")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--variant", default="all", help="One of 'all' or a specific variant name.")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    output_dir = work_dir / "production_ideas"
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = StrategyRunner(
        work_dir=work_dir,
        pdf_dir=Path(args.pdf_dir),
        questions_path=Path(args.questions_path),
        provider=args.provider,
        model=args.model,
    )

    baseline_pack = load_pack(BASELINE_PACK_PATH)
    baseline_audit_rows = parse_baseline_audit(BASELINE_AUDIT_PATH)

    variants = VARIANTS if args.variant == "all" else [args.variant]
    summaries = []
    for variant in variants:
        pack = build_pack(runner=runner, variant=variant, output_dir=output_dir)
        summary = build_variant_audit(
            baseline_pack=baseline_pack,
            baseline_audit_rows=baseline_audit_rows,
            candidate_pack=pack,
            output_dir=output_dir,
        )
        summaries.append(summary)

    safe_json_dump(output_dir / "experiment_summaries.json", summaries)


if __name__ == "__main__":
    main()
