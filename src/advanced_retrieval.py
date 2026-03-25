from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from src.legal_runtime_index import LegalRuntimeIndex, build_index, open_index
from src.production_idea_corpora import build_summary_augmented_chunk_corpus
from src.public_dataset_eval import PublicCorpus, normalize_space
from src.public_retrieval_benchmark import dedupe_by_ref, doc_diversified_candidates, lexical_candidates, run_strategy, vector_candidates
from src.query_analysis import QuestionAnalysis
from src.lexical_retrieval import tokenize_for_bm25
from src.section_chunking import build_section_aware_chunk_corpus


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
    "law",
    "laws",
    "case",
}
TITLE_MATCH_STOPWORDS = {
    "law",
    "laws",
    "difc",
    "case",
    "court",
    "order",
    "judgment",
    "judgement",
    "rules",
    "regulations",
    "regulation",
}
ORDER_MARKERS = (
    "it is hereby ordered that",
    "order with reasons",
    "permission to appeal",
    "application is dismissed",
    "application is refused",
)

GENERIC_NOISE_TOKENS = {
    "difc",
    "law",
    "laws",
    "court",
    "courts",
    "judge",
    "judgment",
    "judgement",
    "order",
    "claimant",
    "defendant",
    "appellant",
}


def normalized_text(value: Any) -> str:
    return normalize_space(str(value or "")).lower()


def ordered_unique(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def tokenize_keywords(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        text = " ".join(str(item) for item in value)
    else:
        text = str(value)
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z0-9/.-]+", text)
        if len(token) > 2 and token.lower() not in STOPWORDS
    ]
    return ordered_unique(tokens)


def significant_title_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value)
        if len(token) > 3 and not (token.isdigit() and len(token) == 4) and token not in TITLE_MATCH_STOPWORDS
    }


def article_reference_score(text: str, article_refs: Sequence[str]) -> float:
    text_norm = normalized_text(text)
    score = 0.0
    for article_ref in article_refs:
        ref_norm = normalized_text(article_ref)
        if ref_norm and ref_norm in text_norm:
            score += 24.0
        match = re.search(r"article\s+(\d+)", article_ref or "", re.I)
        if match and re.search(rf"(?:^|\b|##\s*){match.group(1)}\.", text_norm):
            score += 16.0
        for part in re.findall(r"\(([^)]+)\)", article_ref or ""):
            if re.search(rf"\(\s*{re.escape(part.lower())}\s*\)", text_norm):
                score += 6.0
    return score


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
    def __init__(self, corpus: PublicCorpus, cache_dir: Path):
        self.corpus = corpus
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
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


class AdvancedRetriever:
    SUPPORTED_STRATEGIES = {
        "late_interaction",
        "multi_query_expansion",
        "corrective_retrieval",
        "evidence_first",
        "citation_focus",
        "route_guard_hybrid",
        "intent_hybrid",
        "multi_index_hybrid",
        "hybrid_mq_evidence",
        "hybrid_mq_late",
        "hybrid_prod_v1",
        "hybrid_prod_v2",
        "doc_profile_hybrid",
    }

    def __init__(self, corpus: PublicCorpus, work_dir: Path, runtime_index: LegalRuntimeIndex | None = None):
        self.corpus = corpus
        self.work_dir = work_dir
        self._late_reranker = LateInteractionReranker(corpus, work_dir / "late_interaction")
        self.doc_profile_rows = self._build_doc_profile_rows()
        self.doc_profile_embeddings = self._load_or_build_doc_profile_embeddings()
        self.doc_profile_bm25 = BM25Okapi([tokenize_for_bm25(row["text"]) for row in self.doc_profile_rows]) if self.doc_profile_rows else None
        self.runtime_index = runtime_index or self._ensure_runtime_index()
        self._aux_corpora: Dict[str, PublicCorpus | None] = {}

    def _ensure_runtime_index(self) -> LegalRuntimeIndex | None:
        db_path = self.corpus.work_dir / "runtime_index" / "legal_runtime_index.sqlite"
        try:
            if db_path.exists():
                return open_index(db_path)
            return build_index(self.corpus.work_dir, db_path, source_variant="chunked_section_aware")
        except Exception:
            return None

    def _auxiliary_corpus(self, variant: str) -> PublicCorpus | None:
        if variant in self._aux_corpora:
            return self._aux_corpora[variant]
        try:
            if variant == "section_aware":
                chunked_dir = build_section_aware_chunk_corpus(
                    self.corpus.work_dir / "docling" / "merged",
                    self.corpus.work_dir / "docling" / "chunked_section_aware",
                )
            elif variant == "summary_augmented":
                chunked_dir = build_summary_augmented_chunk_corpus(
                    chunked_dir=self.corpus.chunked_dir,
                    output_dir=self.corpus.work_dir / "docling" / "chunked_summary_augmented",
                )
            else:
                self._aux_corpora[variant] = None
                return None
            aux = PublicCorpus(
                work_dir=self.corpus.work_dir,
                pdf_dir=self.corpus.pdf_dir,
                chunked_dir=chunked_dir,
                embedder=self.corpus.embedder,
                reranker=self.corpus.reranker,
            )
            self._aux_corpora[variant] = aux
            return aux
        except Exception:
            self._aux_corpora[variant] = None
            return None

    def _build_doc_profile_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for sha, document in self.corpus.documents.items():
            title = normalize_space(document.title)
            aliases = ordered_unique([normalize_space(alias) for alias in document.aliases if normalize_space(alias)])[:6]
            canonical_ids = ordered_unique([normalize_space(item) for item in document.canonical_ids if normalize_space(item)])[:4]
            page_one = normalize_space(document.first_page_text[:1200])
            parts = [
                f"[DOC_TITLE] {title}",
                f"[DOC_KIND] {document.kind}",
            ]
            if canonical_ids:
                parts.append(f"[DOC_IDS] {' | '.join(canonical_ids)}")
            if aliases:
                parts.append(f"[ALIASES] {' | '.join(aliases)}")
            if document.is_enactment_notice:
                parts.append("[DOC_ROLE] enactment_notice")
            if document.is_consolidated:
                parts.append("[DOC_ROLE] consolidated_version")
            if page_one:
                parts.append(f"[FIRST_PAGE] {page_one}")
            rows.append({"sha": sha, "title": title, "text": "\n".join(parts)})
        return rows

    def _load_or_build_doc_profile_embeddings(self) -> np.ndarray:
        cache_dir = self.work_dir / "doc_profile_index"
        cache_dir.mkdir(parents=True, exist_ok=True)
        texts = [row["text"] for row in self.doc_profile_rows]
        signature = hashlib.sha1("\n".join(f"{row['sha']}\n{row['text']}" for row in self.doc_profile_rows).encode("utf-8")).hexdigest()[:12]
        embedder_cache_key = getattr(self.corpus.embedder, "cache_key", self.corpus.embedding_model)
        cache_slug = re.sub(r"[^a-zA-Z0-9]+", "_", embedder_cache_key).strip("_").lower()
        emb_path = cache_dir / f"{cache_slug}__{signature}__doc_profiles.npy"
        if emb_path.exists():
            return np.load(emb_path)
        embeddings = self.corpus.embedder.embed_documents(texts)
        np.save(emb_path, embeddings)
        return embeddings

    def document_profile_shortlist(
        self,
        *,
        question: str,
        analysis: QuestionAnalysis | None,
        limit: int = 8,
    ) -> List[str]:
        if not self.doc_profile_rows:
            return []
        query = question
        if analysis is not None:
            query = normalize_space(
                " ".join(
                    part
                    for part in [
                        analysis.retrieval_query,
                        " ".join(analysis.target_case_ids),
                        " ".join(analysis.target_law_ids),
                        " ".join(analysis.target_titles),
                        " ".join(analysis.target_article_refs),
                        " ".join(analysis.must_support_terms[:3]),
                    ]
                    if normalize_space(part)
                )
            ) or question
        query_embedding = self.corpus.embedder.embed_query(query)
        vector_scores = self.doc_profile_embeddings @ query_embedding
        vector_order = np.argsort(vector_scores)[::-1][: min(12, len(self.doc_profile_rows))]
        ranked_lists: List[tuple[str, Sequence[Dict[str, Any]], float]] = []
        vector_rows = [
            {"ref": self.doc_profile_rows[int(rank)]["sha"], "sha": self.doc_profile_rows[int(rank)]["sha"], "distance": float(vector_scores[int(rank)])}
            for rank in vector_order
        ]
        ranked_lists.append(("doc_vector", vector_rows, 1.0))
        if self.doc_profile_bm25 is not None:
            bm25_scores = self.doc_profile_bm25.get_scores(tokenize_for_bm25(query))
            bm25_order = np.argsort(bm25_scores)[::-1][: min(12, len(self.doc_profile_rows))]
            bm25_rows = [
                {"ref": self.doc_profile_rows[int(rank)]["sha"], "sha": self.doc_profile_rows[int(rank)]["sha"], "distance": float(bm25_scores[int(rank)])}
                for rank in bm25_order
            ]
            ranked_lists.append(("doc_bm25", bm25_rows, 0.7))
        fused = reciprocal_rank_fuse(ranked_lists)
        return ordered_unique(row["sha"] for row in fused[:limit])

    def route_with_titles(
        self,
        *,
        route_candidate_shas: Sequence[str],
        analysis: QuestionAnalysis | None,
        question: str,
    ) -> List[str]:
        if analysis is None:
            return list(route_candidate_shas)
        candidate_shas = list(route_candidate_shas)
        question_norm = normalized_text(question)
        for sha, document in self.corpus.documents.items():
            doc_tokens = significant_title_tokens(normalized_text(document.title))
            for target in analysis.target_titles:
                target_tokens = significant_title_tokens(normalized_text(target))
                if target_tokens and len(doc_tokens & target_tokens) >= min(2, len(target_tokens)):
                    candidate_shas.append(sha)
            if any(item.lower() in question_norm for item in [canonical.lower() for canonical in document.canonical_ids]):
                candidate_shas.append(sha)
        candidate_shas.extend(self.document_profile_shortlist(question=question, analysis=analysis, limit=8))
        return ordered_unique(candidate_shas)

    def multi_query_variants(self, analysis: QuestionAnalysis | None, question: str) -> List[str]:
        if analysis is None:
            return [question]
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
        if analysis.target_titles:
            variants.append(" ".join(analysis.target_titles))
        if analysis.task_family in {"comparison_rag", "comparison"} and explicit_parts:
            shared_terms = " ".join(analysis.must_support_terms[:3])
            for entity in ordered_unique(list(analysis.target_case_ids) + list(analysis.target_law_ids) + list(analysis.target_titles)):
                variants.append(normalize_space(f"{entity} {shared_terms}"))
        if analysis.task_family in {"clause_lookup", "article_lookup"} or analysis.target_field in {"article_content", "clause_summary"}:
            clause_terms = " ".join(ordered_unique(list(analysis.target_article_refs) + analysis.must_support_terms)[:5])
            if clause_terms:
                variants.append(normalize_space(f"{analysis.standalone_question} {clause_terms} operative clause"))
        if analysis.task_family == "enumeration_lookup":
            variants.append(normalize_space(f"{analysis.standalone_question} list all relevant entities"))
        if analysis.task_family == "absence_probe":
            term_string = " ".join(analysis.must_support_terms[:4])
            if term_string:
                variants.append(normalize_space(f"{term_string} explicit mention definition clause"))
        if analysis.target_field == "order_result":
            variants.append(f"{analysis.standalone_question} IT IS HEREBY ORDERED THAT conclusion order result")
        if analysis.target_field == "article_content" and analysis.target_article_refs:
            variants.append(f"{analysis.standalone_question} {' '.join(analysis.target_article_refs)}")
        return [normalize_space(variant) for variant in ordered_unique(variants) if normalize_space(variant)]

    def vector_candidates(
        self,
        *,
        question: str,
        candidate_shas: Sequence[str],
        vector_k: int = 16,
    ) -> List[Dict[str, Any]]:
        candidate_indices = self.corpus._candidate_chunk_indices(candidate_shas)
        query_embedding = self.corpus.embedder.embed_query(question)
        candidate_embeddings = self.corpus.chunk_embeddings[candidate_indices]
        scores = candidate_embeddings @ query_embedding
        top_order = np.argsort(scores)[::-1][: min(vector_k, len(candidate_indices))]
        rows = []
        for rank in top_order:
            chunk_index = candidate_indices[int(rank)]
            chunk = self.corpus.chunks[chunk_index]
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

    def auxiliary_candidates(
        self,
        *,
        variant: str,
        question: str,
        candidate_shas: Sequence[str],
        strategy: str = "dense_doc_diverse",
    ) -> List[Dict[str, Any]]:
        aux_corpus = self._auxiliary_corpus(variant)
        if aux_corpus is None:
            return []
        return run_strategy(aux_corpus, aux_corpus.bm25_index, question, candidate_shas, strategy)

    def _page_chunk_row(
        self,
        *,
        doc_id: str,
        page_number: int,
        ref_suffix: str,
        distance: float,
    ) -> Dict[str, Any] | None:
        payload = self.corpus.documents_payload.get(doc_id)
        document = self.corpus.documents.get(doc_id)
        if payload is None or document is None:
            return None
        page_text = ""
        for page in payload["content"]["pages"]:
            if int(page["page"]) == int(page_number):
                page_text = str(page.get("text", "") or "")
                break
        if not page_text:
            return None
        return {
            "ref": f"{doc_id}:{int(page_number)}:{ref_suffix}",
            "distance": round(float(distance), 4),
            "page": int(page_number),
            "text": page_text,
            "sha": doc_id,
            "title": document.title,
            "kind": document.kind,
            "canonical_ids": list(document.canonical_ids),
        }

    def runtime_index_retrieve(
        self,
        *,
        question: str,
        analysis: QuestionAnalysis | None,
        candidate_shas: Sequence[str],
    ) -> List[Dict[str, Any]]:
        if self.runtime_index is None:
            return []
        query_variants = ordered_unique(
            [
                question,
                analysis.standalone_question if analysis is not None else "",
                analysis.retrieval_query if analysis is not None else "",
                " ".join((analysis.target_article_refs if analysis is not None else []) + (analysis.must_support_terms[:4] if analysis is not None else [])),
                " ".join(analysis.target_titles[:2]) if analysis is not None else "",
            ]
        )
        query_variants = [variant for variant in query_variants if normalize_space(variant)]
        if not query_variants:
            return []

        page_roles: List[str] = []
        tags: List[str] = []
        if analysis is not None:
            if "title_page" in analysis.support_focus or "first_page" in analysis.support_focus:
                page_roles.append("title_page")
            if "last_page" in analysis.support_focus:
                page_roles.append("last_page")
            if analysis.target_field == "order_result":
                tags.extend(["order_section", "conclusion_section"])
            if analysis.target_field in {"administered_by", "made_by", "publication_text", "enacted_text", "effective_dates"}:
                tags.extend(["administration_clause", "made_by_clause", "publication_line", "enactment_clause"])
            if analysis.target_field in {"article_content", "clause_summary", "absence_check"} or analysis.target_article_refs:
                tags.append("article_section")
            if analysis.target_field in {"claim_number", "claim_amount"}:
                tags.append("case_id_line")
            if analysis.target_field in {"judge_name", "judges", "common_judges"}:
                tags.append("judge_block")
            if analysis.target_field in {"claimant", "defendant", "parties", "common_parties"}:
                tags.append("party_block")

        rows: List[Dict[str, Any]] = []
        seen_pages: set[tuple[str, int]] = set()

        for query_index, query_variant in enumerate(query_variants[:3], start=1):
            page_hits = self.runtime_index.search_pages(
                query_variant,
                limit=8,
                doc_ids=candidate_shas,
                page_roles=page_roles or None,
                tags=tags or None,
            )
            for rank, hit in enumerate(page_hits, start=1):
                key = (hit.doc_id, hit.page_number)
                if key in seen_pages:
                    continue
                seen_pages.add(key)
                row = self._page_chunk_row(
                    doc_id=hit.doc_id,
                    page_number=hit.page_number,
                    ref_suffix="rtidx",
                    distance=max(0.2, 1.0 - (query_index - 1) * 0.08 - (rank - 1) * 0.03),
                )
                if row is not None:
                    row["retrieval_sources"] = ["runtime_page"]
                    rows.append(row)

            if analysis is not None and (analysis.target_article_refs or analysis.target_field in {"article_content", "clause_summary", "absence_check"}):
                article_hits = self.runtime_index.search_articles(query_variant, limit=8, doc_ids=candidate_shas)
                for rank, hit in enumerate(article_hits, start=1):
                    key = (hit.doc_id, hit.page_number)
                    if key in seen_pages:
                        continue
                    seen_pages.add(key)
                    row = self._page_chunk_row(
                        doc_id=hit.doc_id,
                        page_number=hit.page_number,
                        ref_suffix="rtidx",
                        distance=max(0.25, 1.02 - (query_index - 1) * 0.08 - (rank - 1) * 0.03),
                    )
                    if row is not None:
                        row["text"] = f"[ARTICLE_MATCH] {hit.article_ref}\n{row['text']}"
                        row["retrieval_sources"] = ["runtime_article"]
                        rows.append(row)
        return rows[:8]

    def bm25_hybrid_candidates(
        self,
        *,
        question: str,
        candidate_shas: Sequence[str],
        vector_k: int = 16,
        rerank_k: int = 10,
        bm25_k: int = 10,
        bm25_weight: float = 0.35,
    ) -> List[Dict[str, Any]]:
        result = self.corpus.retrieve(
            question=question,
            candidate_shas=candidate_shas,
            vector_k=vector_k,
            rerank_k=rerank_k,
            bm25_k=bm25_k,
            bm25_weight=bm25_weight,
            bm25_auto=True,
        )
        return result["reranked_results"][: max(5, min(rerank_k, 8))]

    def multi_query_retrieve(
        self,
        *,
        question: str,
        analysis: QuestionAnalysis | None,
        candidate_shas: Sequence[str],
    ) -> List[Dict[str, Any]]:
        ranked_lists = []
        for rank, query in enumerate(self.multi_query_variants(analysis, question), start=1):
            results = self.vector_candidates(question=query, candidate_shas=candidate_shas, vector_k=12)
            ranked_lists.append((f"q{rank}", results, max(0.45, 1.0 - (rank - 1) * 0.12)))
        fused = reciprocal_rank_fuse(ranked_lists)
        return self.corpus.reranker.rerank_documents(question, fused, llm_weight=0.7)[:5]

    def late_interaction_retrieve(
        self,
        *,
        question: str,
        candidate_shas: Sequence[str],
    ) -> List[Dict[str, Any]]:
        initial = self.corpus.retrieve(question=question, candidate_shas=candidate_shas, vector_k=18, rerank_k=12)
        return self._late_reranker.rerank(question, initial["reranked_results"], limit=5)

    def corrective_retrieve(
        self,
        *,
        question: str,
        analysis: QuestionAnalysis | None,
        candidate_shas: Sequence[str],
    ) -> List[Dict[str, Any]]:
        baseline = run_strategy(self.corpus, self.corpus.bm25_index, question, candidate_shas, "dense_doc_diverse")
        needs_multi = bool(analysis and analysis.needs_multi_document_support)
        enough_docs = len(ordered_unique(chunk["sha"] for chunk in baseline[:5])) >= (2 if needs_multi else 1)
        explicit_hits = True
        if analysis is not None and (analysis.target_case_ids or analysis.target_law_ids):
            targets = set(analysis.target_case_ids + analysis.target_law_ids)
            explicit_hits = any(set(item.get("canonical_ids") or []) & targets for item in baseline[:5])
        article_supported = True
        if analysis is not None and analysis.target_field == "article_content" and analysis.target_article_refs:
            article_supported = any(article_reference_score(chunk["text"], analysis.target_article_refs) >= 16 for chunk in baseline[:5])
        order_supported = True
        if analysis is not None and analysis.target_field == "order_result":
            order_supported = any(any(marker in normalized_text(chunk["text"]) for marker in ORDER_MARKERS) for chunk in baseline[:5])
        if enough_docs and explicit_hits and article_supported and order_supported:
            return baseline[:5]

        expansion_queries = self.multi_query_variants(analysis, question)
        if analysis is not None and analysis.must_support_terms:
            expansion_queries.append(f"{question} {' '.join(analysis.must_support_terms)}")
        ranked_lists = [("baseline", baseline[:8], 1.0)]
        for rank, query in enumerate(ordered_unique(expansion_queries), start=1):
            results = self.vector_candidates(question=query, candidate_shas=candidate_shas, vector_k=14)
            ranked_lists.append((f"exp{rank}", results, 0.85))
        bm25_results = self.corpus.retrieve(
            question=question,
            candidate_shas=candidate_shas,
            vector_k=16,
            rerank_k=8,
            bm25_k=8,
            bm25_weight=0.35,
            bm25_auto=True,
        )["vector_results"][:8]
        ranked_lists.append(("bm25", bm25_results, 0.55))
        fused = reciprocal_rank_fuse(ranked_lists)
        return self.corpus.reranker.rerank_documents(question, fused, llm_weight=0.7)[:5]

    def evidence_first_select(
        self,
        *,
        question: str,
        analysis: QuestionAnalysis | None,
        candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if analysis is None:
            return list(candidates[:5])
        title_tokens = set()
        for title in analysis.target_titles:
            title_tokens.update(significant_title_tokens(normalized_text(title)))
        question_tokens = set(tokenize_keywords(question))
        scored = []
        for rank, chunk in enumerate(candidates[:8], start=1):
            text_norm = normalized_text(chunk["text"])
            score = float(chunk.get("combined_score", chunk.get("relevance_score", chunk.get("distance", 0.0))))
            score += article_reference_score(chunk["text"], analysis.target_article_refs)
            score += 8.0 * sum(1 for term in analysis.must_support_terms if normalized_text(term) in text_norm)
            score += 3.0 * len(question_tokens & set(tokenize_keywords(chunk["text"])) & set(tokenize_keywords(" ".join(analysis.must_support_terms))))
            if title_tokens:
                score += 6.0 * len(title_tokens & significant_title_tokens(normalized_text(chunk["title"])))
            if "first_page" in analysis.support_focus or "title_page" in analysis.support_focus:
                if int(chunk["page"]) == 1:
                    score += 18.0
            if "last_page" in analysis.support_focus and int(chunk["page"]) == self.corpus.documents[chunk["sha"]].page_count:
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
        for _, chunk in scored:
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
        return selected[:5] or list(candidates[:5])

    def citation_like_select(
        self,
        *,
        question: str,
        analysis: QuestionAnalysis | None,
        candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if analysis is None:
            return list(candidates[:5])

        question_tokens = set(tokenize_keywords(question))
        title_tokens = set()
        for title in analysis.target_titles:
            title_tokens.update(significant_title_tokens(normalized_text(title)))
        target_ids = set(analysis.target_case_ids + analysis.target_law_ids)
        expected_docs = max(1, min(2, len(target_ids) + len(analysis.target_titles)))

        rescored = []
        for rank, chunk in enumerate(candidates[:10], start=1):
            text_norm = normalized_text(chunk["text"])
            title_norm = normalized_text(chunk["title"])
            title_keyword_overlap = len(title_tokens & significant_title_tokens(title_norm))
            text_tokens = set(tokenize_keywords(chunk["text"]))
            base_score = float(chunk.get("combined_score", chunk.get("relevance_score", chunk.get("distance", 0.0))))

            score = base_score
            score += 12.0 * sum(1 for term in analysis.must_support_terms if normalized_text(term) in text_norm)
            score += 10.0 * article_reference_score(chunk["text"], analysis.target_article_refs)
            score += 6.0 * title_keyword_overlap
            score += 2.0 * len(question_tokens & text_tokens)
            score += 10.0 * bool(target_ids & set(chunk.get("canonical_ids") or []))

            if analysis.target_field in {"law_number", "made_by", "administered_by", "publication_text", "enacted_text"}:
                if int(chunk["page"]) == 1:
                    score += 18.0
                else:
                    score -= 4.0
            if analysis.target_field == "order_result":
                if any(marker in text_norm for marker in ORDER_MARKERS):
                    score += 18.0
                elif int(chunk["page"]) == self.corpus.documents[chunk["sha"]].page_count:
                    score += 8.0
            if "last_page" in analysis.support_focus and int(chunk["page"]) == self.corpus.documents[chunk["sha"]].page_count:
                score += 10.0
            if "first_page" in analysis.support_focus or "title_page" in analysis.support_focus:
                if int(chunk["page"]) == 1:
                    score += 12.0

            # Penalize very generic headings that often look relevant but ground poorly.
            noise_overlap = len(GENERIC_NOISE_TOKENS & significant_title_tokens(title_norm))
            if noise_overlap and not title_keyword_overlap and not (target_ids & set(chunk.get("canonical_ids") or [])):
                score -= 3.0 * noise_overlap
            if "contents" in title_norm or "index" in title_norm:
                score -= 12.0

            rescored.append((score - rank * 0.02, dict(chunk)))

        rescored.sort(key=lambda item: item[0], reverse=True)
        selected: List[Dict[str, Any]] = []
        seen_refs = set()
        covered_docs = set()
        for _, chunk in rescored:
            if chunk["ref"] in seen_refs:
                continue
            if analysis.needs_multi_document_support and len(covered_docs) < expected_docs:
                if chunk["sha"] in covered_docs and any(other["sha"] != chunk["sha"] for _, other in rescored):
                    continue
            selected.append(chunk)
            seen_refs.add(chunk["ref"])
            covered_docs.add(chunk["sha"])
            enough_support = (
                len(selected) >= 3
                and (
                    not analysis.needs_multi_document_support
                    or len(covered_docs) >= min(expected_docs, 2)
                )
            )
            if enough_support and len(selected) >= 3:
                break
            if len(selected) >= 5:
                break
        return selected[:5] or list(candidates[:5])

    def intent_hybrid_retrieve(
        self,
        *,
        question: str,
        analysis: QuestionAnalysis | None,
        candidate_shas: Sequence[str],
    ) -> List[Dict[str, Any]]:
        expanded_candidates = self.route_with_titles(
            route_candidate_shas=candidate_shas,
            analysis=analysis,
            question=question,
        )
        base = run_strategy(self.corpus, self.corpus.bm25_index, question, expanded_candidates, "dense_doc_diverse")
        ranked_lists: List[tuple[str, Sequence[Dict[str, Any]], float]] = [("base", base, 0.7)]
        multi = self.multi_query_retrieve(question=question, analysis=analysis, candidate_shas=expanded_candidates)
        ranked_lists.append(("multi", multi, 1.0))
        needs_clause_precision = bool(
            analysis
            and (
                analysis.task_family in {"clause_lookup", "comparison_rag", "absence_probe", "enumeration_lookup"}
                or analysis.target_field in {"article_content", "order_result", "clause_summary", "comparison_answer", "absence_check", "enumeration_answer"}
                or "clause_localized" in analysis.intent_tags
            )
        )
        if needs_clause_precision:
            late = self.late_interaction_retrieve(question=question, candidate_shas=expanded_candidates)
            ranked_lists.append(("late", late, 0.9))
            bm25_hybrid = self.bm25_hybrid_candidates(question=question, candidate_shas=expanded_candidates)
            ranked_lists.append(("bm25_hybrid", bm25_hybrid, 0.8))
        if analysis and (analysis.needs_multi_document_support or analysis.task_family in {"comparison_rag", "enumeration_lookup"}):
            corrective = self.corrective_retrieve(question=question, analysis=analysis, candidate_shas=expanded_candidates)
            ranked_lists.append(("corrective", corrective, 0.85))
        fused = reciprocal_rank_fuse(ranked_lists)
        diversified = doc_diversified_candidates(dedupe_by_ref(fused), head_keep=4, max_unique_docs=12)
        reranked = self.corpus.reranker.rerank_documents(question, diversified, llm_weight=0.7)
        if analysis and (analysis.needs_multi_document_support or analysis.task_family in {"comparison_rag", "enumeration_lookup"}):
            return self.evidence_first_select(question=question, analysis=analysis, candidates=reranked)
        return self.citation_like_select(question=question, analysis=analysis, candidates=reranked)

    def route_guard_hybrid_retrieve(
        self,
        *,
        question: str,
        analysis: QuestionAnalysis | None,
        candidate_shas: Sequence[str],
    ) -> List[Dict[str, Any]]:
        baseline = run_strategy(self.corpus, self.corpus.bm25_index, question, candidate_shas, "dense_doc_diverse")
        profile_candidates = self.document_profile_shortlist(question=question, analysis=analysis, limit=12)
        if not profile_candidates:
            return baseline[:5]

        shortlist = ordered_unique(list(profile_candidates) + list(candidate_shas[:4]))
        shortlist_base = run_strategy(self.corpus, self.corpus.bm25_index, question, shortlist, "dense_doc_diverse")
        shortlist_bm25 = self.bm25_hybrid_candidates(
            question=question,
            candidate_shas=shortlist,
            vector_k=18,
            rerank_k=12,
            bm25_k=12,
            bm25_weight=0.45,
        )

        ranked_lists: List[tuple[str, Sequence[Dict[str, Any]], float]] = [
            ("baseline", baseline, 0.8),
            ("shortlist_base", shortlist_base, 0.95),
            ("shortlist_bm25", shortlist_bm25, 1.0),
        ]

        needs_clause_precision = bool(
            analysis
            and (
                analysis.task_family in {"clause_lookup", "comparison_rag", "absence_probe", "enumeration_lookup"}
                or analysis.target_field in {"article_content", "order_result", "clause_summary", "comparison_answer", "absence_check", "enumeration_answer"}
                or "clause_localized" in analysis.intent_tags
            )
        )
        if needs_clause_precision:
            late = self.late_interaction_retrieve(question=question, candidate_shas=shortlist)
            ranked_lists.append(("shortlist_late", late, 0.8))

        fused = reciprocal_rank_fuse(ranked_lists)
        diversified = doc_diversified_candidates(dedupe_by_ref(fused), head_keep=4, max_unique_docs=10)
        reranked = self.corpus.reranker.rerank_documents(question, diversified, llm_weight=0.7)

        if analysis and (analysis.needs_multi_document_support or analysis.task_family in {"comparison_rag", "enumeration_lookup"}):
            return self.evidence_first_select(question=question, analysis=analysis, candidates=reranked)
        return self.citation_like_select(question=question, analysis=analysis, candidates=reranked)

    def multi_index_hybrid_retrieve(
        self,
        *,
        question: str,
        analysis: QuestionAnalysis | None,
        candidate_shas: Sequence[str],
    ) -> List[Dict[str, Any]]:
        expanded_candidates = self.route_with_titles(
            route_candidate_shas=candidate_shas,
            analysis=analysis,
            question=question,
        )
        base = run_strategy(self.corpus, self.corpus.bm25_index, question, expanded_candidates, "dense_doc_diverse")
        ranked_lists: List[tuple[str, Sequence[Dict[str, Any]], float]] = [("base", base, 0.8)]

        multi = self.multi_query_retrieve(question=question, analysis=analysis, candidate_shas=expanded_candidates)
        ranked_lists.append(("multi", multi, 1.0))

        summary_rows = self.auxiliary_candidates(
            variant="summary_augmented",
            question=question,
            candidate_shas=expanded_candidates,
        )
        if summary_rows:
            ranked_lists.append(("summary", summary_rows, 0.7))

        needs_clause_precision = bool(
            analysis
            and (
                analysis.task_family in {"clause_lookup", "comparison_rag", "absence_probe", "enumeration_lookup"}
                or analysis.target_field in {"article_content", "order_result", "clause_summary", "comparison_answer", "absence_check", "enumeration_answer"}
                or "clause_localized" in analysis.intent_tags
            )
        )
        if needs_clause_precision:
            section_rows = self.auxiliary_candidates(
                variant="section_aware",
                question=question,
                candidate_shas=expanded_candidates,
            )
            if section_rows:
                ranked_lists.append(("section", section_rows, 0.8))
            runtime_rows = self.runtime_index_retrieve(
                question=question,
                analysis=analysis,
                candidate_shas=expanded_candidates,
            )
            if runtime_rows:
                ranked_lists.append(("runtime", runtime_rows, 0.75))

        fused = reciprocal_rank_fuse(ranked_lists)
        diversified = doc_diversified_candidates(dedupe_by_ref(fused), head_keep=4, max_unique_docs=12)
        reranked = self.corpus.reranker.rerank_documents(question, diversified, llm_weight=0.7)
        if analysis and (analysis.needs_multi_document_support or analysis.task_family in {"comparison_rag", "enumeration_lookup"}):
            return self.evidence_first_select(question=question, analysis=analysis, candidates=reranked)
        return self.citation_like_select(question=question, analysis=analysis, candidates=reranked)

    def retrieve(
        self,
        *,
        strategy: str,
        question: str,
        analysis: QuestionAnalysis | None,
        candidate_shas: Sequence[str],
    ) -> List[Dict[str, Any]]:
        expanded_candidates = self.route_with_titles(
            route_candidate_shas=candidate_shas,
            analysis=analysis,
            question=question,
        )

        if strategy == "late_interaction":
            return self.late_interaction_retrieve(question=question, candidate_shas=expanded_candidates)
        if strategy == "multi_query_expansion":
            return self.multi_query_retrieve(question=question, analysis=analysis, candidate_shas=expanded_candidates)
        if strategy == "corrective_retrieval":
            return self.corrective_retrieve(question=question, analysis=analysis, candidate_shas=expanded_candidates)
        if strategy == "evidence_first":
            initial = run_strategy(self.corpus, self.corpus.bm25_index, question, expanded_candidates, "dense_doc_diverse")
            return self.evidence_first_select(question=question, analysis=analysis, candidates=initial)
        if strategy == "citation_focus":
            initial = run_strategy(self.corpus, self.corpus.bm25_index, question, expanded_candidates, "dense_doc_diverse")
            return self.citation_like_select(question=question, analysis=analysis, candidates=initial)
        if strategy == "route_guard_hybrid":
            return self.route_guard_hybrid_retrieve(question=question, analysis=analysis, candidate_shas=expanded_candidates)
        if strategy == "intent_hybrid":
            return self.intent_hybrid_retrieve(question=question, analysis=analysis, candidate_shas=expanded_candidates)
        if strategy == "multi_index_hybrid":
            return self.multi_index_hybrid_retrieve(question=question, analysis=analysis, candidate_shas=expanded_candidates)
        if strategy == "hybrid_mq_evidence":
            multi = self.multi_query_retrieve(question=question, analysis=analysis, candidate_shas=expanded_candidates)
            base = run_strategy(self.corpus, self.corpus.bm25_index, question, expanded_candidates, "dense_doc_diverse")
            fused = reciprocal_rank_fuse([("multi", multi, 1.0), ("base", base, 0.7)])
            reranked = self.corpus.reranker.rerank_documents(question, fused, llm_weight=0.7)
            return self.evidence_first_select(question=question, analysis=analysis, candidates=reranked)
        if strategy == "hybrid_mq_late":
            multi = self.multi_query_retrieve(question=question, analysis=analysis, candidate_shas=expanded_candidates)
            late = self.late_interaction_retrieve(question=question, candidate_shas=expanded_candidates)
            base = run_strategy(self.corpus, self.corpus.bm25_index, question, expanded_candidates, "dense_doc_diverse")
            fused = reciprocal_rank_fuse([("multi", multi, 1.0), ("late", late, 0.9), ("base", base, 0.6)])
            return self.corpus.reranker.rerank_documents(question, fused, llm_weight=0.7)[:5]
        if strategy == "hybrid_prod_v1":
            multi = self.multi_query_retrieve(question=question, analysis=analysis, candidate_shas=expanded_candidates)
            late = self.late_interaction_retrieve(question=question, candidate_shas=expanded_candidates)
            corrective = self.corrective_retrieve(question=question, analysis=analysis, candidate_shas=expanded_candidates)
            base = run_strategy(self.corpus, self.corpus.bm25_index, question, expanded_candidates, "dense_doc_diverse")
            diversified = doc_diversified_candidates(dedupe_by_ref(multi + late + corrective + base), head_keep=4, max_unique_docs=10)
            reranked = self.corpus.reranker.rerank_documents(question, diversified, llm_weight=0.7)
            return self.evidence_first_select(question=question, analysis=analysis, candidates=reranked)
        if strategy == "hybrid_prod_v2":
            multi = self.multi_query_retrieve(question=question, analysis=analysis, candidate_shas=expanded_candidates)
            corrective = self.corrective_retrieve(question=question, analysis=analysis, candidate_shas=expanded_candidates)
            base = run_strategy(self.corpus, self.corpus.bm25_index, question, expanded_candidates, "dense_doc_diverse")
            fused = reciprocal_rank_fuse([("multi", multi, 1.0), ("corrective", corrective, 0.9), ("base", base, 0.6)])
            reranked = self.corpus.reranker.rerank_documents(question, fused, llm_weight=0.7)
            return self.citation_like_select(question=question, analysis=analysis, candidates=reranked)
        if strategy == "doc_profile_hybrid":
            profile_candidates = self.document_profile_shortlist(question=question, analysis=analysis, limit=10)
            expanded = ordered_unique(list(expanded_candidates) + list(profile_candidates))
            base = run_strategy(self.corpus, self.corpus.bm25_index, question, expanded, "dense_doc_diverse")
            multi = self.multi_query_retrieve(question=question, analysis=analysis, candidate_shas=expanded)
            fused = reciprocal_rank_fuse([("base", base, 0.8), ("multi", multi, 1.0)])
            reranked = self.corpus.reranker.rerank_documents(question, fused, llm_weight=0.7)
            return self.citation_like_select(question=question, analysis=analysis, candidates=reranked)
        raise ValueError(f"Unsupported advanced retrieval strategy: {strategy}")
