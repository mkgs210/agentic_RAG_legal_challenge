from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from rank_bm25 import BM25Okapi

ROOT_DIR = Path(__file__).resolve().parent.parent
STARTER_KIT_DIR = ROOT_DIR / "starter_kit"

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
if str(STARTER_KIT_DIR) not in sys.path:
    sys.path.append(str(STARTER_KIT_DIR))

from arlc import (  # noqa: E402
    EvaluationClient,
    RetrievalRef,
    SubmissionAnswer,
    SubmissionBuilder,
    Telemetry,
    TimingMetrics,
    UsageMetrics,
    get_config,
    normalize_retrieved_pages,
)
from src.advanced_retrieval import AdvancedRetriever  # noqa: E402
from src.api_requests import APIProcessor  # noqa: E402
from src.evidence_selection import (  # noqa: E402
    EvidenceSelector,
    GroundingIndex,
    is_absence_answer,
    ordered_unique as ordered_unique_refs,
    parse_page_ref,
)
from src.lexical_retrieval import tokenize_for_bm25  # noqa: E402
from src.legal_runtime_index import LegalRuntimeIndex, build_index, open_index  # noqa: E402
from src.production_idea_corpora import build_contextual_chunk_corpus, build_summary_augmented_chunk_corpus  # noqa: E402
from src.public_dataset_eval import (  # noqa: E402
    PublicCorpus,
    answer_contextual_absence,
    answer_question,
    compress_free_text_answer,
    extract_article_refs,
    extract_case_ids,
    extract_law_ids,
    normalize_answer,
    normalize_space,
    prefer_draft_citations,
    prepare_docling_artifacts,
    refine_free_text_answer,
    resolve_answer_citations,
    safe_json_dump,
    select_best_free_text_payload,
)
from src.public_retrieval_benchmark import run_strategy  # noqa: E402
from src.query_analysis import QuestionAnalysis, QuestionAnalyzer  # noqa: E402
from src.structured_solver import StructuredWarmupSolver  # noqa: E402


ABSENT_MARKERS = (
    "not present in the context",
    "not available in the context",
    "there is no information",
    "contain no information",
    "contains no information",
    "do not provide information about",
    "does not provide information about",
    "do not provide any information about",
    "does not provide any information about",
    "do not state anything about",
    "does not state anything about",
    "do not state",
    "does not state",
    "do not mention",
    "does not mention",
    "do not contain information",
    "does not contain information",
    "do not contain enough information",
    "does not contain enough information",
    "cannot be answered from the corpus",
    "cannot be found",
    "absent from the context",
)

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

ORDER_SECTION_MARKERS = (
    "it is hereby ordered that",
    "conclusion",
    "order with reasons",
    "judgment",
    "application",
)

GENERIC_SUPPORT_TERMS = {
    "appeal",
    "permission to appeal",
    "application",
    "result",
    "outcome",
    "final ruling",
    "cost",
    "costs",
    "trial",
    "order",
    "conclusion",
    "ruling",
    "article",
    "law",
    "case",
    "purpose",
}

JUDGE_MARKERS = (
    "before ",
    "justice ",
    "judge ",
    "chief justice",
    "deputy chief justice",
    "hearing :",
    "coram",
)


def article_reference_score(text_norm: str, article_ref: str) -> float:
    article_norm = normalized_text(article_ref)
    if not article_norm:
        return 0.0
    score = 0.0
    if article_norm in text_norm:
        score += 28.0

    number_match = re.search(r"article\s+(\d+)", article_norm)
    if number_match:
        number = number_match.group(1)
        if re.search(rf"(?:^|\b|##\s*){number}\.", text_norm):
            score += 20.0

    parts = re.findall(r"\(([^)]+)\)", article_ref)
    for part in parts:
        if re.search(rf"\(\s*{re.escape(part.lower())}\s*\)", text_norm):
            score += 8.0
    return score


def normalized_text(value: Any) -> str:
    return normalize_space(str(value or "")).lower()


def answer_is_absent(raw_answer: str) -> bool:
    lowered = normalized_text(raw_answer)
    return any(marker in lowered for marker in ABSENT_MARKERS)


def token_keywords(value: Any) -> list[str]:
    if value is None or isinstance(value, bool):
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
    return list(dict.fromkeys(tokens))[:12]


def context_text(chunks: Sequence[Dict[str, Any]]) -> str:
    return "\n".join(str(chunk.get("text", "") or "") for chunk in chunks)


def term_supported_in_context(term: str, context_norm: str) -> bool:
    term_norm = normalized_text(term)
    if not term_norm:
        return True
    if term_norm in context_norm:
        return True
    tokens = [
        token.lower()
        for token in re.findall(r"[a-z0-9/.-]+", term_norm)
        if len(token) > 3 and token.lower() not in STOPWORDS
    ]
    if not tokens:
        return term_norm in context_norm
    hits = sum(1 for token in tokens if token in context_norm)
    if len(tokens) == 1:
        return hits == 1
    if len(tokens) == 2:
        return hits == 2
    return hits >= len(tokens) - 1


def distinctive_support_terms(analysis: QuestionAnalysis | None) -> list[str]:
    if analysis is None or not analysis.must_support_terms:
        return []
    result: list[str] = []
    for term in analysis.must_support_terms:
        term_norm = normalized_text(term)
        if not term_norm:
            continue
        if re.fullmatch(r"(article\s+\d+(?:\([^)]+\))*)", term_norm):
            result.append(term)
            continue
        if re.fullmatch(r"(?:cfi|ca|sct|enf|arb)\s+\d+/\d{4}", term_norm, re.I):
            continue
        if term_norm in GENERIC_SUPPORT_TERMS:
            continue
        tokens = [
            token.lower()
            for token in re.findall(r"[a-z0-9/.-]+", term_norm)
            if len(token) > 3 and token.lower() not in STOPWORDS and token.lower() not in GENERIC_SUPPORT_TERMS
        ]
        if not tokens:
            continue
        result.append(term)
    return list(dict.fromkeys(result))[:4]


def missing_support_terms(analysis: QuestionAnalysis | None, chunks: Sequence[Dict[str, Any]]) -> list[str]:
    terms = distinctive_support_terms(analysis)
    if not terms:
        return []
    context_norm = normalized_text(context_text(chunks))
    if not context_norm:
        return list(terms)
    return [term for term in terms if not term_supported_in_context(term, context_norm)]


def should_abstain_for_missing_support_terms(
    answer_type: str,
    answer_payload: Dict[str, Any],
    analysis: QuestionAnalysis | None,
    chunks: Sequence[Dict[str, Any]],
) -> tuple[bool, list[str]]:
    if (
        answer_type != "free_text"
        or analysis is None
        or not analysis.must_support_terms
    ):
        return False, []
    raw_answer = str(answer_payload.get("raw_answer", "") or "").strip()
    if not raw_answer or answer_is_absent(raw_answer):
        return False, []
    missing = missing_support_terms(analysis, chunks)
    if not missing:
        return False, []
    distinctive_terms = distinctive_support_terms(analysis)
    if not distinctive_terms:
        return False, []
    if len(distinctive_terms) == 1:
        return True, missing
    if len(missing) == len(distinctive_terms):
        return True, missing
    if analysis.target_field in {"order_result", "article_content", "clause_summary", "comparison_answer", "enumeration_answer", "absence_check"}:
        if len(missing) >= max(1, len(distinctive_terms) - 1):
            return True, missing
    if len(distinctive_terms) >= 3 and len(missing) >= len(distinctive_terms) - 1:
        return True, missing
    return False, missing


def derive_absence_concept(question: str, analysis: QuestionAnalysis | None = None) -> str:
    question_text = normalize_space(question)
    lowered = question_text.lower().rstrip("?.!")
    if analysis is not None and analysis.target_field == "absence_check":
        if "parole" in lowered:
            return "parole hearings"
        if "jury" in lowered:
            return "a jury decision"
        if "miranda" in lowered:
            return "Miranda rights"
        if "plea bargain" in lowered:
            return "a plea bargain"
    if analysis is not None:
        for term in analysis.must_support_terms:
            term_text = normalize_space(term)
            if term_text and term_text.lower() not in {"appeal", "costs", "records", "translation", "liability"}:
                return term_text
    patterns = (
        r"(?:any information about|information about)\s+(.+)$",
        r"(?:what was the plea bargain in)\s+(.+)$",
        r"(?:what did the jury decide in)\s+(.+)$",
        r"(?:were the)\s+(.+?)\s+properly administered(?:\s+in\s+.+)?$",
        r"(?:does(?:\s+the\s+document)?\s+contain)\s+(.+)$",
        r"(?:does(?:\s+the\s+document)?\s+mention)\s+(.+)$",
        r"(?:is there)\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered, re.I)
        if not match:
            continue
        concept = normalize_space(match.group(1))
        concept = re.sub(r"\s+in\s+case\s+[A-Z]{2,4}\s+\d+/\d+$", "", concept, flags=re.I)
        concept = re.sub(r"\s+under\s+article\s+.+$", "", concept, flags=re.I)
        concept = concept.strip(" .,:;")
        if concept:
            return concept
    return ""


def absence_answer_text(question: str, analysis: QuestionAnalysis | None = None) -> str:
    concept = derive_absence_concept(question, analysis)
    subject = ""
    case_ids = extract_case_ids(question)
    law_ids = extract_law_ids(question)
    if case_ids:
        subject = case_ids[0]
    elif law_ids:
        subject = law_ids[0]
    question_norm = normalized_text(question)
    if concept:
        if subject and subject.lower() not in concept.lower():
            scoped = f"{concept} in {subject}"
        else:
            scoped = concept
        if question_norm.startswith("is there any information about") or question_norm.startswith("is there information about"):
            return f"There is no information about {scoped} in the provided documents."
        if re.match(r"^(were|was|did|does|can|could|is|are|what)\b", question_norm):
            return f"There is no information about {scoped} in the provided documents."
        return f"There is no information about {scoped} in the provided documents."
    if subject:
        if question_norm.startswith("is there any information about") or question_norm.startswith("is there information about"):
            return f"There is no information about {subject} in the provided documents."
        return f"There is no information about {subject} in the provided documents."
    if question_norm.startswith("is there any information about") or question_norm.startswith("is there information about"):
        return "There is no information about this in the provided documents."
    return "There is no information on this question in the provided documents."


def absence_answer_payload(
    question: str = "",
    analysis: QuestionAnalysis | None = None,
) -> Dict[str, Any]:
    answer = absence_answer_text(question, analysis)
    return {
        "raw_answer": answer,
        "normalized_answer": normalize_answer("free_text", answer),
        "citations": [],
        "reasoning": "Unsupported by provided context",
        "confidence": "high",
        "response_data": {},
    }


def contextual_absence_payload(
    *,
    api: APIProcessor,
    model: str,
    question: str,
    analysis: QuestionAnalysis | None,
    chunks: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if analysis is not None and analysis.target_field == "absence_check":
        return absence_answer_payload(question, analysis)
    if not chunks:
        return absence_answer_payload(question, analysis)
    try:
        payload = answer_contextual_absence(
            api=api,
            model=model,
            question=question,
            chunks=chunks,
            analysis=analysis,
        )
        raw_answer = normalize_space(str(payload.get("raw_answer", "") or ""))
        better_answer = absence_answer_text(question, analysis)
        better_norm = normalize_space(better_answer).lower()
        if not raw_answer or raw_answer.lower() == "there is no information on this question in the provided documents.":
            payload["raw_answer"] = better_answer
            payload["normalized_answer"] = normalize_answer("free_text", better_answer)
            return payload
        if better_norm != raw_answer.lower():
            subject_tokens = [token.lower() for token in extract_case_ids(question) + extract_law_ids(question)]
            if subject_tokens and not any(token in raw_answer.lower() for token in subject_tokens):
                payload["raw_answer"] = better_answer
                payload["normalized_answer"] = normalize_answer("free_text", better_answer)
                return payload
        lowered = raw_answer.lower()
        if (
            lowered.startswith("there is no information about ")
            or lowered.startswith("the provided materials do not state anything about ")
        ) and ("provided documents" in lowered or "provided materials" in lowered):
            better_answer = absence_answer_text(question, analysis)
            payload["raw_answer"] = better_answer
            payload["normalized_answer"] = normalize_answer("free_text", better_answer)
        return payload
    except Exception:
        return absence_answer_payload(question, analysis)


def ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def significant_title_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value)
        if len(token) > 3 and not (token.isdigit() and len(token) == 4) and token not in TITLE_MATCH_STOPWORDS
    }


def clean_citation_ref(citation: Any) -> str:
    text = normalize_space(str(citation or ""))
    text = re.sub(r"^ref\s+", "", text, flags=re.I)
    text = re.sub(r"^ref", "", text, flags=re.I)
    return text.strip()


def page_refs_from_citations(answer_payload: Dict[str, Any], chunks: Sequence[Dict[str, Any]]) -> list[str]:
    chunk_by_ref = {chunk["ref"]: chunk for chunk in chunks}
    refs: list[str] = []
    for citation in answer_payload.get("citations", []):
        cleaned = clean_citation_ref(citation)
        parts = cleaned.split(":")
        if len(parts) >= 2:
            try:
                refs.append(f"{parts[0]}:{int(parts[1])}")
                continue
            except ValueError:
                pass
        chunk = chunk_by_ref.get(cleaned)
        if not chunk:
            continue
        refs.append(f"{chunk['sha']}:{int(chunk['page'])}")
    return ordered_unique(refs)


def page_refs_from_retrieval_refs(retrieval_refs: Sequence[RetrievalRef]) -> list[str]:
    refs: list[str] = []
    for retrieval_ref in retrieval_refs:
        for page_number in retrieval_ref.page_numbers:
            refs.append(f"{retrieval_ref.doc_id}:{int(page_number)}")
    return ordered_unique(refs)


def primary_chunk_refs(answer_payload: Dict[str, Any], chunks: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    chunk_by_ref = {chunk["ref"]: chunk for chunk in chunks}
    cited_chunks = []
    for citation in answer_payload.get("citations", []):
        cleaned = clean_citation_ref(citation)
        if cleaned in chunk_by_ref:
            cited_chunks.append(chunk_by_ref[cleaned])
    if cited_chunks:
        return cited_chunks
    return list(chunks[:5])


def chunks_from_citations(corpus: PublicCorpus, citations: Sequence[Any]) -> list[Dict[str, Any]]:
    refs = [clean_citation_ref(citation) for citation in citations if clean_citation_ref(citation)]
    if not refs:
        return []
    selected: list[Dict[str, Any]] = []
    seen = set()
    for ref in refs:
        parts = ref.split(":")
        if len(parts) != 3:
            continue
        sha, page_text, chunk_text = parts
        try:
            page_number = int(page_text)
            chunk_id = int(chunk_text)
        except ValueError:
            continue
        payload = corpus.documents_payload.get(sha)
        document = corpus.documents.get(sha)
        if payload is None or document is None:
            continue
        matched = False
        for chunk in payload["content"]["chunks"]:
            if int(chunk["page"]) != page_number or int(chunk["id"]) != chunk_id:
                continue
            chunk_ref = f"{sha}:{page_number}:{chunk_id}"
            if chunk_ref in seen:
                break
            seen.add(chunk_ref)
            selected.append(
                {
                    "ref": chunk_ref,
                    "distance": 1.0,
                    "page": page_number,
                    "text": str(chunk["text"]),
                    "sha": sha,
                    "title": document.title,
                    "kind": document.kind,
                    "canonical_ids": document.canonical_ids,
                }
            )
            matched = True
            break
        if matched:
            continue
        pages = (payload.get("content") or {}).get("pages") or []
        if not 1 <= page_number <= len(pages):
            continue
        page_text = str(pages[page_number - 1].get("text") or "")
        if not page_text:
            continue
        chunk_ref = f"{sha}:{page_number}:{chunk_id}"
        if chunk_ref in seen:
            continue
        seen.add(chunk_ref)
        selected.append(
            {
                "ref": chunk_ref,
                "distance": 1.0,
                "page": page_number,
                "text": page_text,
                "sha": sha,
                "title": document.title,
                "kind": document.kind,
                "canonical_ids": document.canonical_ids,
            }
        )
    return selected


def target_shas_for_question(corpus: PublicCorpus, question: str, chunks: Sequence[Dict[str, Any]]) -> list[str]:
    route = corpus.route_question(question, expansive=False)
    explicit_ids = route["explicit_case_ids"] + route["explicit_law_ids"]
    selected = ordered_unique(chunk["sha"] for chunk in chunks)
    question_norm = normalized_text(question)
    comparison_mode = len(explicit_ids) >= 2 or len(route["alias_hits"]) >= 2 or any(
        marker in question_norm for marker in (" both ", " between ", " common ", " same ")
    )
    if comparison_mode:
        ranked = [sha for sha in route["candidate_shas"] if sha in selected]
        ranked.extend(sha for sha in selected if sha not in ranked)
        ranked.extend(sha for sha in route["candidate_shas"] if sha not in ranked)
        return ranked[:4]
    if selected:
        return selected[:1]
    return route["candidate_shas"][:1]


def candidate_shas_from_titles(corpus: PublicCorpus, titles: Sequence[str]) -> list[str]:
    candidates: list[tuple[float, str]] = []
    for title in titles:
        title_norm = normalized_text(title)
        if len(title_norm) < 4:
            continue
        explicit_ids = extract_law_ids(title) + extract_case_ids(title)
        explicit_shas = ordered_unique(
            sha
            for item_id in explicit_ids
            for sha in corpus.id_index.get(item_id, [])
        )
        if explicit_shas:
            for rank, sha in enumerate(explicit_shas):
                candidates.append((300.0 - rank, sha))
            continue
        title_tokens = significant_title_tokens(title_norm)
        for sha, document in corpus.documents.items():
            values = [document.title, *document.aliases, *document.canonical_ids]
            score = 0.0
            for value in values:
                alias_norm = normalized_text(value)
                if not alias_norm:
                    continue
                alias_tokens = significant_title_tokens(alias_norm)
                if title_norm == alias_norm:
                    score = max(score, 200.0)
                elif title_norm in alias_norm and len(title_tokens) >= 2:
                    score = max(score, 140.0 + len(title_norm))
                elif alias_norm in title_norm and len(alias_tokens) >= 2:
                    score = max(score, 120.0 + len(alias_norm))
                else:
                    overlap_tokens = title_tokens & alias_tokens
                    required_overlap = min(2, len(title_tokens), len(alias_tokens))
                    if required_overlap and len(overlap_tokens) >= required_overlap:
                        score = max(score, len(overlap_tokens) * 20.0 + len(" ".join(overlap_tokens)))
            if score > 0:
                candidates.append((score, sha))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return ordered_unique(sha for _, sha in candidates)


def article_refs_for_title(article_refs: Sequence[str], title: str, index: int, total_titles: int) -> list[str]:
    title_norm = normalized_text(title)
    title_tokens = {token for token in re.findall(r"[a-z0-9]+", title_norm) if len(token) > 3 and token not in STOPWORDS}
    explicit: list[str] = []
    plain: list[str] = []
    for article_ref in article_refs:
        ref_norm = normalized_text(article_ref)
        ref_tokens = {token for token in re.findall(r"[a-z0-9]+", ref_norm) if len(token) > 3 and token not in STOPWORDS}
        if title_tokens and len(title_tokens & ref_tokens) >= max(1, min(2, len(title_tokens))):
            explicit.append(article_ref)
        else:
            plain.append(article_ref)
    if explicit:
        return list(dict.fromkeys(explicit))
    if total_titles > 1 and len(plain) >= total_titles and index < len(plain):
        return [plain[index]]
    return list(dict.fromkeys(plain))


def analysis_target_shas(corpus: PublicCorpus, analysis: QuestionAnalysis | None) -> list[str]:
    if analysis is None:
        return []
    candidate_shas: list[str] = []
    for case_id in analysis.target_case_ids:
        candidate_shas.extend(corpus.id_index.get(case_id, []))
    for law_id in analysis.target_law_ids:
        candidate_shas.extend(corpus.id_index.get(law_id, []))
    candidate_shas.extend(candidate_shas_from_titles(corpus, analysis.target_titles))
    return ordered_unique(candidate_shas)


def analysis_explicit_target_shas(corpus: PublicCorpus, analysis: QuestionAnalysis | None) -> list[str]:
    if analysis is None:
        return []
    candidate_shas: list[str] = []
    for case_id in analysis.target_case_ids:
        candidate_shas.extend(corpus.id_index.get(case_id, []))
    for law_id in analysis.target_law_ids:
        candidate_shas.extend(corpus.id_index.get(law_id, []))
    return ordered_unique(candidate_shas)


def should_prefer_exact_grounding_for_deterministic(
    answer_type: str,
    analysis: QuestionAnalysis | None,
    exact_page_refs: Sequence[str],
) -> bool:
    if answer_type == "free_text" or analysis is None or not exact_page_refs:
        return False
    anchored = bool(
        analysis.target_titles
        or analysis.target_law_ids
        or analysis.target_case_ids
        or analysis.target_article_refs
    )
    if not anchored:
        return False
    return analysis.target_field in {
        "clause_summary",
        "enumeration_answer",
        "comparison_answer",
        "effective_dates",
        "enacted_text",
    }


def analysis_target_docs_with_article_refs(
    corpus: PublicCorpus,
    analysis: QuestionAnalysis | None,
) -> list[tuple[str, list[str]]]:
    if analysis is None:
        return []
    rows: list[tuple[str, list[str]]] = []
    seen = set()
    explicit_target_shas = analysis_explicit_target_shas(corpus, analysis)
    if explicit_target_shas:
        for sha in explicit_target_shas:
            if sha in seen:
                continue
            seen.add(sha)
            rows.append((sha, list(analysis.target_article_refs)))
        if analysis.target_article_refs and not analysis.needs_multi_document_support:
            return rows
    if analysis.target_titles:
        for index, title in enumerate(analysis.target_titles):
            shas = candidate_shas_from_titles(corpus, [title])
            if not shas:
                continue
            scoped_refs = article_refs_for_title(
                analysis.target_article_refs,
                title=title,
                index=index,
                total_titles=len(analysis.target_titles),
            )
            for sha in shas[:1]:
                if sha in seen:
                    continue
                seen.add(sha)
                rows.append((sha, scoped_refs))
    for sha in analysis_target_shas(corpus, analysis):
        if sha in seen:
            continue
        rows.append((sha, list(analysis.target_article_refs)))
        seen.add(sha)
    return rows


def merge_routes(
    corpus: PublicCorpus,
    question: str,
    analysis: QuestionAnalysis | None,
) -> tuple[str, Dict[str, Any]]:
    base_route = corpus.route_question(question, expansive=False)
    if analysis is None:
        return question, base_route

    effective_question = analysis.retrieval_query or analysis.standalone_question or question
    effective_question = normalize_space(effective_question) or question
    alt_route = base_route if effective_question == question else corpus.route_question(effective_question, expansive=False)
    explicit_target_shas = analysis_explicit_target_shas(corpus, analysis)
    title_target_shas = candidate_shas_from_titles(corpus, analysis.target_titles)
    single_source_fields = {
        "article_content",
        "law_number",
        "made_by",
        "administered_by",
        "publication_text",
        "enacted_text",
        "effective_dates",
        "amended_laws",
        "absence_check",
        "clause_summary",
    }
    narrow_targets = explicit_target_shas or (title_target_shas if len(title_target_shas) == 1 else [])
    if (
        narrow_targets
        and not analysis.needs_multi_document_support
        and analysis.target_field in single_source_fields
    ):
        candidate_shas = ordered_unique(narrow_targets)
    else:
        candidate_shas = ordered_unique(
            list(base_route["candidate_shas"])
            + list(alt_route["candidate_shas"])
            + list(title_target_shas)
        )
    alias_hits = ordered_unique(list(base_route["alias_hits"]) + list(alt_route["alias_hits"]) + list(analysis.target_titles))
    explicit_case_ids = ordered_unique(
        list(base_route["explicit_case_ids"]) + list(alt_route["explicit_case_ids"]) + list(analysis.target_case_ids)
    )
    explicit_law_ids = ordered_unique(
        list(base_route["explicit_law_ids"]) + list(alt_route["explicit_law_ids"]) + list(analysis.target_law_ids)
    )

    merged_route = {
        "question": question,
        "standalone_question": analysis.standalone_question or question,
        "retrieval_query": effective_question,
        "explicit_case_ids": explicit_case_ids,
        "explicit_law_ids": explicit_law_ids,
        "alias_hits": alias_hits,
        "candidate_shas": candidate_shas or list(base_route["candidate_shas"]),
    }
    return effective_question, merged_route


def should_expand_with_runtime_index(
    corpus: PublicCorpus,
    route: Dict[str, Any],
    analysis: QuestionAnalysis | None,
) -> bool:
    if analysis is None:
        return False
    if analysis.use_structured_executor and analysis.target_field not in {
        "generic_answer",
        "article_content",
        "order_result",
        "comparison_answer",
        "enumeration_answer",
        "absence_check",
    }:
        return False

    explicit_anchor = bool(
        route.get("explicit_case_ids")
        or route.get("explicit_law_ids")
        or route.get("alias_hits")
        or analysis.target_case_ids
        or analysis.target_law_ids
        or analysis.target_titles
    )
    candidate_count = len(route.get("candidate_shas") or [])
    total_docs = max(1, len(corpus.documents))
    broad_route = candidate_count >= max(18, (total_docs * 3) // 5)
    weak_route = candidate_count <= 4 and not explicit_anchor
    article_or_order = analysis.target_field in {"article_content", "order_result", "clause_summary"}
    comparative = analysis.target_field in {"comparison_answer", "enumeration_answer"} or analysis.needs_multi_document_support
    generic_fallback = analysis.task_family in {"generic_rag", "comparison_rag", "clause_lookup", "enumeration_lookup", "absence_probe"}
    return bool(broad_route or weak_route or article_or_order or comparative or generic_fallback)


def runtime_index_candidate_shas(
    runtime_index: LegalRuntimeIndex,
    question: str,
    effective_question: str,
    analysis: QuestionAnalysis,
    current_candidates: Sequence[str],
) -> list[str]:
    candidate_scores: dict[str, float] = {}
    queries = ordered_unique(
        [
            normalize_space(effective_question),
            normalize_space(question),
            *[normalize_space(title) for title in (analysis.target_titles or [])],
            *[normalize_space(ref) for ref in (analysis.target_article_refs or [])],
        ]
    )
    for query in queries:
        if not query:
            continue
        route = runtime_index.route_query(query)
        doc_hits = runtime_index.search_documents(query, limit=6, kinds=route.get("kinds"))
        page_hits = runtime_index.search_pages(
            query,
            limit=6,
            kinds=route.get("kinds"),
            page_roles=route.get("page_roles"),
        )
        entity_hits = runtime_index.search_entities(
            query,
            limit=4,
            kinds=route.get("entity_kinds"),
            doc_ids=None,
        )
        article_hits = runtime_index.search_articles(
            query,
            limit=6,
            doc_ids=None,
        )

        for rank, hit in enumerate(doc_hits, start=1):
            candidate_scores[hit.doc_id] = candidate_scores.get(hit.doc_id, 0.0) + (8.0 - rank)
        for rank, hit in enumerate(page_hits, start=1):
            candidate_scores[hit.doc_id] = candidate_scores.get(hit.doc_id, 0.0) + (5.0 - 0.5 * rank)
        for rank, hit in enumerate(entity_hits, start=1):
            candidate_scores[hit.doc_id] = candidate_scores.get(hit.doc_id, 0.0) + (4.5 - 0.5 * rank)
        for rank, hit in enumerate(article_hits, start=1):
            candidate_scores[hit.doc_id] = candidate_scores.get(hit.doc_id, 0.0) + (5.5 - 0.5 * rank)

    ranked = sorted(candidate_scores.items(), key=lambda item: (-item[1], item[0]))
    runtime_candidates = [doc_id for doc_id, _score in ranked[:8]]
    if not runtime_candidates:
        return []
    existing = set(current_candidates)
    novel = [doc_id for doc_id in runtime_candidates if doc_id not in existing]
    if novel:
        return novel[:4]
    return runtime_candidates[:2]


def build_chunk_record(corpus: PublicCorpus, sha: str, chunk: Dict[str, Any], score: float) -> Dict[str, Any]:
    document = corpus.documents[sha]
    return {
        "ref": f"{sha}:{int(chunk['page'])}:{int(chunk['id'])}",
        "distance": round(float(score), 4),
        "page": int(chunk["page"]),
        "text": str(chunk.get("text", "") or ""),
        "sha": sha,
        "title": document.title,
        "kind": document.kind,
        "canonical_ids": document.canonical_ids,
    }


def heuristic_support_chunks(
    corpus: PublicCorpus,
    sha: str,
    question_norm: str,
    article_refs: Sequence[str],
    analysis: QuestionAnalysis | None = None,
) -> list[Dict[str, Any]]:
    payload = corpus.documents_payload.get(sha)
    document = corpus.documents.get(sha)
    if payload is None or document is None:
        return []

    focus = set(analysis.support_focus) if analysis is not None else set()
    focused_request = bool(focus)
    wants_cover = "title_page" in focus or (
        not focused_request
        and any(
            marker in question_norm
            for marker in (
                "title page",
                "cover page",
                "official law number",
                "who made this law",
                "published",
                "consolidated version",
            )
        )
    )
    wants_first = "first_page" in focus or (not focused_request and "first page" in question_norm)
    wants_second = "second_page" in focus or (
        not focused_request and ("page 2" in question_norm or "second page" in question_norm)
    )
    wants_last = "last_page" in focus or (
        not focused_request and any(marker in question_norm for marker in ("last page", "conclusion section"))
    )
    wants_date = "issue_date_line" in focus or (
        not focused_request and any(marker in question_norm for marker in ("date of issue", "issue date"))
    )
    wants_administer = "administration_clause" in focus or (not focused_request and "administer" in question_norm)
    wants_order = "order_section" in focus or "conclusion_section" in focus or (
        not focused_request and any(marker in question_norm for marker in ORDER_SECTION_MARKERS)
    )
    wants_judge = "judge_block" in focus or (
        not focused_request and any(marker in question_norm for marker in ("judge", "judges", "preside", "presiding"))
    )

    rows: list[Dict[str, Any]] = []
    for chunk in payload["content"]["chunks"]:
        text_norm = normalized_text(chunk.get("text", ""))
        score = 0.0
        page_number = int(chunk["page"])
        if (wants_cover or "title_page" in focus) and page_number == 1:
            score += 30.0
        if (wants_first or "first_page" in focus) and page_number == 1:
            score += 24.0
        if (wants_second or "second_page" in focus) and page_number == 2:
            score += 22.0
        if wants_last and page_number == document.page_count:
            score += 28.0
        if wants_date and "date of issue" in text_norm:
            score += 24.0
        if wants_administer and "administered by" in text_norm:
            score += 26.0
        if "enactment_clause" in focus and "this law is enacted on" in text_norm:
            score += 26.0
        if "publication_line" in focus and "consolidated version" in text_norm:
            score += 24.0
        if wants_order:
            if "it is hereby ordered that" in text_norm:
                score += 30.0
            if "conclusion" in text_norm:
                score += 18.0
            if "order with reasons" in text_norm:
                score += 18.0
        if "order_section" in focus and "it is hereby ordered that" in text_norm:
            score += 28.0
        if "conclusion_section" in focus and "conclusion" in text_norm:
            score += 24.0
        if wants_judge and any(marker in text_norm for marker in JUDGE_MARKERS):
            score += 18.0
        if "judge_block" in focus and any(marker in text_norm for marker in JUDGE_MARKERS):
            score += 22.0
        if "party_block" in focus and "between" in text_norm:
            score += 18.0
        for article_ref in article_refs:
            score += article_reference_score(text_norm, article_ref)
        if score > 0:
            rows.append(build_chunk_record(corpus, sha, chunk, score))
    rows.sort(key=lambda item: (-item["distance"], item["page"], item["ref"]))
    return rows


def augment_chunks_for_answering(
    corpus: PublicCorpus,
    question: str,
    chunks: Sequence[Dict[str, Any]],
    analysis: QuestionAnalysis | None = None,
    max_chunks: int = 7,
) -> list[Dict[str, Any]]:
    if not chunks:
        return []
    question_norm = normalized_text(question)
    article_refs = analysis.target_article_refs if analysis is not None and analysis.target_article_refs else extract_article_refs(question)
    support: list[Dict[str, Any]] = []
    forced_target_docs = analysis_target_docs_with_article_refs(corpus, analysis)
    preferred_shas = ordered_unique([sha for sha, _ in forced_target_docs] + analysis_target_shas(corpus, analysis))
    for sha, scoped_article_refs in forced_target_docs[:4]:
        support.extend(heuristic_support_chunks(corpus, sha, question_norm, scoped_article_refs or article_refs, analysis=analysis)[:2])
    for sha in target_shas_for_question(corpus, question, chunks):
        if preferred_shas and sha not in preferred_shas and not (analysis is not None and analysis.needs_multi_document_support):
            continue
        support.extend(heuristic_support_chunks(corpus, sha, question_norm, article_refs, analysis=analysis)[:2])

    base_chunks = list(chunks)
    if preferred_shas and not (analysis is not None and analysis.needs_multi_document_support):
        filtered = [chunk for chunk in base_chunks if chunk["sha"] in set(preferred_shas)]
        if filtered:
            base_chunks = filtered

    merged: list[Dict[str, Any]] = []
    seen = set()
    for chunk in support + base_chunks:
        ref = chunk["ref"]
        if ref in seen:
            continue
        seen.add(ref)
        merged.append(chunk)
        if len(merged) >= max_chunks:
            break
    return merged


def page_chunks(corpus: PublicCorpus, sha: str, page_number: int) -> list[Dict[str, Any]]:
    payload = corpus.documents_payload.get(sha)
    if payload is None:
        return []
    chunks = [
        chunk
        for chunk in payload["content"]["chunks"]
        if int(chunk["page"]) == int(page_number)
    ]
    chunks.sort(key=lambda chunk: int(chunk["id"]))
    rows = [
        build_chunk_record(corpus, sha, chunk, 0.0)
        for chunk in chunks
    ]
    return rows


def question_keywords(question: str, analysis: QuestionAnalysis | None) -> list[str]:
    tokens: list[str] = []
    sources: list[str] = [question]
    if analysis is not None:
        sources.extend(analysis.must_support_terms)
        sources.extend(analysis.target_titles)
        sources.extend(analysis.target_article_refs)
    for value in sources:
        for token in re.findall(r"[a-z0-9/.-]+", normalized_text(value)):
            if len(token) <= 3 or token in STOPWORDS or token in TITLE_MATCH_STOPWORDS:
                continue
            tokens.append(token)
    return ordered_unique(tokens)[:18]


def score_answer_chunk(
    chunk: Dict[str, Any],
    *,
    analysis: QuestionAnalysis | None,
    question_norm: str,
    keywords: Sequence[str],
    article_refs: Sequence[str],
    exact_page_pairs: set[tuple[str, int]],
    document_page_count: int,
) -> float:
    text_norm = normalized_text(chunk.get("text", ""))
    page_number = int(chunk["page"])
    question_norm = normalize_space(question_norm).lower()
    score = 0.0
    if (chunk["sha"], page_number) in exact_page_pairs:
        score += 30.0
    if analysis is not None:
        focus = set(analysis.support_focus)
        if "title_page" in focus and page_number == 1:
            score += 14.0
        if "first_page" in focus and page_number == 1:
            score += 12.0
        if "second_page" in focus and page_number == 2:
            score += 12.0
        if "last_page" in focus and page_number == document_page_count:
            score += 14.0
        if "conclusion_section" in focus and page_number >= max(1, document_page_count - 1):
            score += 10.0
        if analysis.target_field == "order_result":
            if "it is hereby ordered that" in text_norm:
                score += 22.0
            if "conclusion" in text_norm:
                score += 12.0
            if "order with reasons" in text_norm:
                score += 10.0
            if "cost" in question_norm and any(
                marker in text_norm
                for marker in ("## costs", "costs", "statement of costs", "award of costs", "entitled to its costs")
            ):
                score += 12.0
        if analysis.target_field in {"article_content", "clause_summary"}:
            score += sum(article_reference_score(text_norm, article_ref) for article_ref in article_refs)
        if analysis.target_field in {"made_by", "administered_by", "publication_text", "enacted_text", "effective_dates"}:
            if "this law is made by" in text_norm:
                score += 16.0
            if "administered by" in text_norm:
                score += 16.0
            if "consolidated version" in text_norm:
                score += 12.0
            if "this law is enacted on" in text_norm:
                score += 16.0
            if "effective date" in text_norm:
                score += 14.0
    overlap = sum(1 for token in keywords if token in text_norm)
    score += overlap * 2.0
    if re.match(r"^#+\s*", str(chunk.get("text", "")).strip()):
        score -= 3.0
    if len(text_norm) < 24:
        score -= 2.0
    return score


def focused_free_text_chunks(
    corpus: PublicCorpus,
    grounding_index: GroundingIndex,
    question: str,
    chunks: Sequence[Dict[str, Any]],
    analysis: QuestionAnalysis | None = None,
    max_chunks: int = 6,
) -> list[Dict[str, Any]]:
    if not chunks:
        return []

    question_norm = normalized_text(question)
    article_refs = analysis.target_article_refs if analysis is not None and analysis.target_article_refs else extract_article_refs(question)
    strict_page_pairs: set[tuple[str, int]] = set()
    if analysis is not None and analysis.target_field in {"article_content", "clause_summary"} and article_refs:
        for sha, scoped_article_refs in analysis_target_docs_with_article_refs(corpus, analysis):
            refs = scoped_article_refs or article_refs
            for page_number in grounding_index.article_pages_for_doc(sha, refs):
                strict_page_pairs.add((sha, int(page_number)))
    if analysis is not None and analysis.target_field == "order_result":
        focus = set(analysis.support_focus)
        for case_id in analysis.target_case_ids:
            case = grounding_index.solver.case_by_id.get(case_id)
            if case is None:
                continue
            page_count = corpus.documents[case.sha].page_count
            if "conclusion_section" in focus:
                start_page = max(1, page_count - 1)
                for page_number in range(start_page, page_count + 1):
                    strict_page_pairs.add((case.sha, page_number))
            elif "last_page" in focus:
                strict_page_pairs.add((case.sha, page_count))
                if "order_section" in focus:
                    for page_number in case.order_pages[:1]:
                        strict_page_pairs.add((case.sha, int(page_number)))
            else:
                for page_number in case.order_pages[:2]:
                    strict_page_pairs.add((case.sha, int(page_number)))
    exact_page_pairs = {
        parsed
        for ref in grounding_index.exact_page_refs(question, "free_text", analysis, chunks)
        if (parsed := parse_page_ref(ref)) is not None
    }
    if strict_page_pairs:
        exact_page_pairs = strict_page_pairs
    exact_only = bool(strict_page_pairs)
    keywords = question_keywords(question, analysis)
    forced_target_docs = analysis_target_docs_with_article_refs(corpus, analysis)
    preferred_shas = ordered_unique([sha for sha, _ in forced_target_docs] + analysis_target_shas(corpus, analysis))
    if not preferred_shas:
        preferred_shas = target_shas_for_question(corpus, question, chunks)

    candidates: list[tuple[float, Dict[str, Any]]] = []
    seen_refs = set()
    for sha, scoped_article_refs in forced_target_docs[:4]:
        for chunk in heuristic_support_chunks(
            corpus,
            sha,
            question_norm,
            scoped_article_refs or article_refs,
            analysis=analysis,
        )[:3]:
            if exact_only and (chunk["sha"], int(chunk["page"])) not in exact_page_pairs:
                continue
            if chunk["ref"] in seen_refs:
                continue
            seen_refs.add(chunk["ref"])
            page_count = corpus.documents[sha].page_count
            score = score_answer_chunk(
                chunk,
                analysis=analysis,
                question_norm=question_norm,
                keywords=keywords,
                article_refs=scoped_article_refs or article_refs,
                exact_page_pairs=exact_page_pairs,
                document_page_count=page_count,
            )
            candidates.append((score, chunk))

    for chunk in chunks:
        if exact_only and (chunk["sha"], int(chunk["page"])) not in exact_page_pairs:
            continue
        if preferred_shas and chunk["sha"] not in set(preferred_shas) and not (analysis and analysis.needs_multi_document_support):
            continue
        if chunk["ref"] in seen_refs:
            continue
        seen_refs.add(chunk["ref"])
        page_count = corpus.documents[chunk["sha"]].page_count
        score = score_answer_chunk(
            chunk,
            analysis=analysis,
            question_norm=question_norm,
            keywords=keywords,
            article_refs=article_refs,
            exact_page_pairs=exact_page_pairs,
            document_page_count=page_count,
        )
        candidates.append((score, chunk))

    for sha, page_number in exact_page_pairs:
        if preferred_shas and sha not in set(preferred_shas) and not (analysis and analysis.needs_multi_document_support):
            continue
        for chunk in page_chunks(corpus, sha, page_number):
            if chunk["ref"] in seen_refs:
                continue
            seen_refs.add(chunk["ref"])
            page_count = corpus.documents[sha].page_count
            score = score_answer_chunk(
                chunk,
                analysis=analysis,
                question_norm=question_norm,
                keywords=keywords,
                article_refs=article_refs,
                exact_page_pairs=exact_page_pairs,
                document_page_count=page_count,
            )
            candidates.append((score, chunk))

    if not candidates:
        return list(chunks[:max_chunks])

    candidates.sort(key=lambda item: (-item[0], item[1]["sha"], int(item[1]["page"]), item[1]["ref"]))
    result: list[Dict[str, Any]] = []
    counts: dict[str, int] = {}
    target_count = 2 if analysis is not None and analysis.needs_multi_document_support else 1
    for score, chunk in candidates:
        if score <= 0 and result:
            break
        sha = chunk["sha"]
        per_doc_limit = 3 if analysis is not None and analysis.target_field == "order_result" else 2
        if counts.get(sha, 0) >= per_doc_limit:
            continue
        counts[sha] = counts.get(sha, 0) + 1
        result.append(chunk)
        if len(result) >= max_chunks:
            break

    if preferred_shas and len(counts) < min(target_count, len(preferred_shas)):
        return list(chunks[:max_chunks])
    return result or list(chunks[:max_chunks])


def should_focus_free_text_context(analysis: QuestionAnalysis | None) -> bool:
    if analysis is None:
        return False
    focus = set(analysis.support_focus)
    page_or_section_focus = {
        "title_page",
        "first_page",
        "second_page",
        "last_page",
        "order_section",
        "conclusion_section",
        "publication_line",
        "administration_clause",
        "issue_date_line",
    }
    if analysis.needs_multi_document_support:
        return True
    if analysis.target_field == "order_result":
        return True
    if analysis.target_field in {"article_content", "clause_summary"} and ("clause_localized" in set(analysis.intent_tags) or bool(analysis.target_article_refs)):
        return True
    if analysis.target_field in {"made_by", "administered_by", "publication_text", "enacted_text", "effective_dates", "enumeration_answer", "absence_check"}:
        return True
    if focus & page_or_section_focus:
        return True
    return False


def choose_answer_model(
    *,
    default_model: str,
    free_text_model: str | None,
    answer_type: str,
    question: str,
    analysis: QuestionAnalysis | None,
) -> str:
    if answer_type != "free_text" or not free_text_model:
        return default_model
    focus = set(analysis.support_focus) if analysis is not None else set()
    if analysis is not None and (
        analysis.needs_multi_document_support
        or analysis.target_field in {"order_result", "article_content", "generic_answer", "effective_dates"}
    ):
        return free_text_model
    if focus.intersection({"order_section", "conclusion_section", "last_page", "first_page", "title_page"}):
        return free_text_model
    return free_text_model


def should_polish_free_text_answer(
    *,
    answer_payload: Dict[str, Any],
    analysis: QuestionAnalysis | None,
    solver_handled: bool,
) -> bool:
    raw_answer = normalize_space(str(answer_payload.get("raw_answer", "") or ""))
    if not raw_answer or is_absence_answer(raw_answer):
        return False
    if answer_payload.get("skip_refine"):
        return False
    if solver_handled:
        return True
    if len(raw_answer) < 90 or len(raw_answer) > 220:
        return True
    if analysis is None:
        return False
    if analysis.target_field in {"order_result", "clause_summary", "comparison_answer", "enumeration_answer", "absence_check"}:
        return True
    return False


def render_structured_free_text_if_needed(
    *,
    solver: StructuredWarmupSolver,
    payload: Dict[str, Any],
    question: str,
    route: Dict[str, Any],
    analysis: QuestionAnalysis | None,
    solver_handled: bool,
) -> Dict[str, Any]:
    if not solver_handled or not payload.get("raw_answer"):
        return payload
    try:
        return solver.render_free_text_payload(
            payload,
            question=question,
            route=route,
            analysis=analysis,
        )
    except Exception:
        return payload


def should_select_free_text_candidates(
    *,
    analysis: QuestionAnalysis | None,
    draft_payload: Dict[str, Any],
    refined_payload: Dict[str, Any],
) -> bool:
    if analysis is None:
        return False
    draft_answer = normalize_space(str(draft_payload.get("raw_answer", "") or ""))
    refined_answer = normalize_space(str(refined_payload.get("raw_answer", "") or ""))
    if not draft_answer or not refined_answer or draft_answer == refined_answer:
        return False
    if analysis.needs_multi_document_support:
        return True
    return analysis.target_field == "order_result"


ORDER_RESULT_OUTCOME_MARKERS = (
    "refused",
    "dismissed",
    "granted",
    "allowed",
    "discharged",
    "rejected",
    "denied",
    "proceed to trial",
    "set aside",
)


def order_result_answer_quality(answer: str, question: str) -> float:
    text = normalize_space(answer)
    lowered = text.lower()
    question_norm = normalize_space(question).lower()
    score = 0.0
    if not text:
        return -100.0
    if answer_is_absent(text):
        return -50.0
    if any(marker in lowered for marker in ORDER_RESULT_OUTCOME_MARKERS):
        score += 8.0
    if any(marker in lowered for marker in ("cost", "costs", "statement of costs", "bear its own costs", "no order as to costs")):
        score += 3.0
    if any(marker in question_norm for marker in ("result of the application", "what was the result", "what was the outcome", "how did the court", "what did the court decide")):
        if "cost" in lowered:
            score += 4.0
    if any(marker in lowered for marker in ("this order concerns", "this is an application", "for the reasons set out", "background", "schedule of reasons")):
        score -= 12.0
    if "..." in text:
        score -= 16.0
    if "visa application is refused" in lowered:
        score -= 20.0
    if len(text) < 70:
        score -= 3.0
    if len(text) > 280:
        score -= 8.0
    return score


def prefer_draft_free_text_answer(
    *,
    draft_payload: Dict[str, Any],
    refined_payload: Dict[str, Any],
    question: str,
    analysis: QuestionAnalysis | None,
    solver_handled: bool,
) -> bool:
    draft_answer = normalize_space(str(draft_payload.get("raw_answer", "") or ""))
    refined_answer = normalize_space(str(refined_payload.get("raw_answer", "") or ""))
    if not draft_answer or not refined_answer or draft_answer == refined_answer:
        return False
    if analysis is None:
        return False
    if analysis.target_field == "order_result" and solver_handled:
        question_norm = normalize_space(question).lower()
        broad_outcome_question = any(
            marker in question_norm
            for marker in (
                "what was the result",
                "what was the outcome",
                "result of the application",
                "outcome of the application",
                "what did the court decide",
                "how did the court",
                "final ruling",
            )
        )
        cost_markers = (
            "costs",
            "cost ",
            "bear its own costs",
            "no order as to costs",
            "statement of costs",
        )
        awkward_scope_markers = (
            "the last page shows",
            "the first page shows",
            "the conclusion section shows",
            "the order section shows",
        )
        if any(marker in refined_answer.lower() for marker in awkward_scope_markers) and not any(
            marker in draft_answer.lower() for marker in awkward_scope_markers
        ):
            return True
        if broad_outcome_question and any(marker in draft_answer.lower() for marker in cost_markers) and not any(
            marker in refined_answer.lower() for marker in cost_markers
        ):
            return True
        for marker in ("adjournment", "set aside", "trial", "no order as to costs"):
            if marker in draft_answer.lower() and marker not in refined_answer.lower():
                return True
        return order_result_answer_quality(draft_answer, question) > order_result_answer_quality(refined_answer, question) + 1.0
    return False


def citations_cover_expected_support(
    answer_type: str,
    analysis: QuestionAnalysis | None,
    citation_page_refs: Sequence[str],
) -> bool:
    if answer_type != "free_text" or not citation_page_refs:
        return False
    cited_docs = {ref.split(":")[0] for ref in citation_page_refs if ":" in ref}
    if analysis is None or not analysis.needs_multi_document_support:
        return True
    expected = len(analysis.target_titles) or len(analysis.target_case_ids) + len(analysis.target_law_ids)
    if expected <= 1:
        expected = 2
    return len(cited_docs) >= min(expected, 2)


def score_page(
    *,
    question_norm: str,
    answer_terms: Sequence[str],
    page_text_norm: str,
    page_number: int,
    page_count: int,
    article_refs: Sequence[str],
    selected_pages: set[int],
    cited_pages: set[int],
) -> float:
    score = 0.0
    if page_number in selected_pages:
        score += 4.0
    if page_number in cited_pages:
        score += 8.0

    if any(marker in question_norm for marker in ("title page", "cover page", "official law number", "who made this law", "published", "consolidated version")):
        if page_number == 1:
            score += 18.0
    if "first page" in question_norm and page_number == 1:
        score += 18.0
    if any(marker in question_norm for marker in ("last page", "conclusion section")) and page_number == page_count:
        score += 18.0
    if any(marker in question_norm for marker in ("date of issue", "issue date")) and "date of issue" in page_text_norm:
        score += 14.0
    if "administer" in question_norm and "administered by" in page_text_norm:
        score += 14.0
    if any(marker in question_norm for marker in ORDER_SECTION_MARKERS):
        if "it is hereby ordered that" in page_text_norm:
            score += 16.0
        if "conclusion" in page_text_norm:
            score += 8.0
        if "order with reasons" in page_text_norm:
            score += 8.0
    if any(marker in question_norm for marker in ("judge", "judges", "preside", "presiding")):
        for marker in JUDGE_MARKERS:
            if marker in page_text_norm:
                score += 3.0
    for article_ref in article_refs:
        score += min(article_reference_score(page_text_norm, article_ref), 16.0)
    overlap = sum(1 for term in answer_terms if term in page_text_norm)
    score += min(overlap, 8) * 1.5
    return score


def select_evidence_refs(
    corpus: PublicCorpus,
    question: str,
    answer_type: str,
    answer_payload: Dict[str, Any],
    chunks: Sequence[Dict[str, Any]],
    answer_value: Any,
    analysis: QuestionAnalysis | None = None,
) -> list[RetrievalRef]:
    raw_answer = str(answer_payload.get("raw_answer", "") or "").strip()
    absent = answer_is_absent(raw_answer)
    if answer_type == "free_text" and absent:
        return []
    if answer_type in {"boolean", "number", "name", "names", "date"} and answer_value is None and absent:
        return []

    seed_chunks = primary_chunk_refs(answer_payload, chunks)
    if not seed_chunks:
        return []

    question_norm = normalized_text(question)
    article_refs = analysis.target_article_refs if analysis is not None and analysis.target_article_refs else extract_article_refs(question)
    answer_terms = token_keywords(answer_value if answer_value is not None else raw_answer)
    target_shas = target_shas_for_question(corpus, question, seed_chunks)
    comparison_mode = len(target_shas) >= 2 or any(
        marker in question_norm for marker in (" both ", " between ", " same ", " common ")
    )

    selected_pages_by_sha: Dict[str, set[int]] = {}
    cited_pages_by_sha: Dict[str, set[int]] = {}
    for chunk in chunks:
        selected_pages_by_sha.setdefault(chunk["sha"], set()).add(int(chunk["page"]))
    for chunk in seed_chunks:
        cited_pages_by_sha.setdefault(chunk["sha"], set()).add(int(chunk["page"]))

    raw_refs: list[RetrievalRef] = []
    seen_pairs = set()

    for sha in target_shas:
        document = corpus.documents.get(sha)
        payload = corpus.documents_payload.get(sha)
        if document is None or payload is None:
            continue
        pages = payload["content"]["pages"]
        scored_pages = []
        for page in pages:
            page_number = int(page["page"])
            page_text_norm = normalized_text(page.get("text", ""))
            score = score_page(
                question_norm=question_norm,
                answer_terms=answer_terms,
                page_text_norm=page_text_norm,
                page_number=page_number,
                page_count=document.page_count,
                article_refs=article_refs,
                selected_pages=selected_pages_by_sha.get(sha, set()),
                cited_pages=cited_pages_by_sha.get(sha, set()),
            )
            if score > 0:
                scored_pages.append((score, page_number))
        scored_pages.sort(key=lambda item: (-item[0], item[1]))
        if scored_pages:
            keep = 2 if not comparison_mode and sha == target_shas[0] else 1
            for _, page_number in scored_pages[:keep]:
                pair = (sha, page_number)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                raw_refs.append(RetrievalRef(doc_id=sha, page_numbers=[page_number]))

    if raw_refs:
        return normalize_retrieved_pages(raw_refs)

    fallback_refs = [
        RetrievalRef(doc_id=chunk["sha"], page_numbers=[int(chunk["page"])])
        for chunk in seed_chunks[: (4 if comparison_mode else 3)]
    ]
    return normalize_retrieved_pages(fallback_refs)


@dataclass
class PlatformRunConfig:
    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    analysis_model: str | None = None
    free_text_model: str | None = None
    strategy: str = "dense_doc_diverse"
    top_k_chunks: int = 5
    chunk_source: str = "standard"
    work_dir: str = "challenge_workdir"
    limit: int = 0
    reuse_downloads: bool = False
    download_only: bool = False
    submit: bool = False


def resolve_runtime_strategy(
    configured_strategy: str,
    *,
    answer_type: str,
    analysis: QuestionAnalysis | None,
) -> str:
    if configured_strategy != "prod_auto_v1":
        return configured_strategy
    if analysis is None:
        return "dense_doc_diverse"
    if analysis.needs_multi_document_support:
        return "intent_hybrid"
    if analysis.task_family in {"comparison_rag", "clause_lookup", "enumeration_lookup", "absence_probe"}:
        return "intent_hybrid"
    if analysis.target_field in {"article_content", "order_result", "effective_dates", "clause_summary", "comparison_answer", "enumeration_answer", "absence_check"}:
        return "intent_hybrid"
    if answer_type == "free_text":
        return "citation_focus"
    if analysis.target_field in {
        "law_number",
        "made_by",
        "administered_by",
        "publication_text",
        "enacted_text",
    }:
        return "citation_focus"
    return "dense_doc_diverse"


def should_use_route_guard(
    corpus: PublicCorpus,
    route: Dict[str, Any],
    analysis: QuestionAnalysis | None,
    active_strategy: str,
) -> bool:
    if active_strategy not in {"dense_doc_diverse", "citation_focus"}:
        return False
    if analysis is None:
        return False

    total_docs = max(1, len(corpus.documents))
    candidate_count = len(route.get("candidate_shas") or [])
    broad_route = candidate_count >= max(18, (total_docs * 3) // 5)
    if not broad_route:
        return False

    explicit_anchor = bool(
        route.get("explicit_case_ids")
        or route.get("explicit_law_ids")
        or route.get("alias_hits")
        or analysis.target_case_ids
        or analysis.target_law_ids
    )
    title_anchor = bool(analysis.target_titles and candidate_shas_from_titles(corpus, analysis.target_titles))
    if explicit_anchor or title_anchor:
        return False

    return bool(
        analysis.needs_multi_document_support
        or analysis.task_family in {"generic_rag", "comparison_rag", "clause_lookup", "enumeration_lookup", "absence_probe"}
        or analysis.target_field in {"generic_answer", "article_content", "order_result", "clause_summary", "comparison_answer", "enumeration_answer", "absence_check"}
    )


def should_isolate_structured_solver(
    *,
    answer_type: str,
    analysis: QuestionAnalysis | None,
) -> bool:
    if analysis is None or answer_type != "free_text":
        return False
    if analysis.target_field != "order_result":
        return False
    focus = set(analysis.support_focus or [])
    return bool(focus.intersection({"last_page", "first_page"}))


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == STARTER_KIT_DIR.name:
        return ROOT_DIR / path
    return STARTER_KIT_DIR / path


def resolve_docs_dir(raw_docs_dir: str) -> Path:
    return resolve_path(Path(raw_docs_dir))


def ensure_pdf_stage_dir(docs_root: Path) -> Path:
    pdfs = sorted(docs_root.rglob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found under {docs_root}")
    direct_pdfs = sorted(docs_root.glob("*.pdf"))
    if len(direct_pdfs) == len(pdfs):
        return docs_root

    stage_dir = docs_root / "_flat_pdfs"
    stage_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in pdfs:
        target = stage_dir / pdf_path.name
        if target.exists():
            continue
        target.write_bytes(pdf_path.read_bytes())
    return stage_dir


def download_resources(
    client: EvaluationClient,
    questions_path: Path,
    docs_dir: Path,
    reuse_downloads: bool = False,
) -> list[dict[str, Any]]:
    if reuse_downloads and questions_path.exists():
        questions = json.loads(questions_path.read_text(encoding="utf-8"))
    else:
        questions = client.download_questions(target_path=questions_path)

    existing_pdfs = list(docs_dir.rglob("*.pdf")) if docs_dir.exists() else []
    if not (reuse_downloads and existing_pdfs):
        client.download_documents(docs_dir)
    return questions


def build_submission_value(answer_type: str, answer_payload: Dict[str, Any]) -> Any:
    raw_answer = str(answer_payload.get("raw_answer", "") or "").strip()
    normalized = answer_payload.get("normalized_answer")
    absent = answer_is_absent(raw_answer)

    if answer_type in {"boolean", "number", "name", "names", "date"} and (absent or normalized is None):
        return None

    if answer_type == "boolean":
        return bool(normalized) if normalized is not None else None
    if answer_type == "number":
        if normalized is None:
            return None
        if float(normalized).is_integer():
            return int(normalized)
        return float(normalized)
    if answer_type == "name":
        return raw_answer or None
    if answer_type == "names":
        parts = [part.strip() for part in raw_answer.replace("|", ";").split(";") if part.strip()]
        return parts or None
    if answer_type == "date":
        return normalized or None

    if absent or not raw_answer:
        return raw_answer or "There is no information on this question in the provided documents."
    return compress_free_text_answer(raw_answer)


def extract_usage(answer_payload: Dict[str, Any]) -> UsageMetrics:
    usage = answer_payload.get("response_data") or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    return UsageMetrics(input_tokens=input_tokens, output_tokens=output_tokens)


def build_code_archive(client: EvaluationClient, archive_path: Path) -> Path:
    include_paths = [
        ROOT_DIR / "src",
        ROOT_DIR / "main.py",
        ROOT_DIR / "requirements.txt",
        ROOT_DIR / "README.md",
        STARTER_KIT_DIR / "arlc",
        STARTER_KIT_DIR / "API.md",
        STARTER_KIT_DIR / "BENCHMARK_RUN_NOTES.md",
        STARTER_KIT_DIR / "COMPLIANCE_CHECKLIST.md",
        STARTER_KIT_DIR / "EVALUATION.md",
        STARTER_KIT_DIR / "README.md",
        STARTER_KIT_DIR / "openapi.yaml",
        STARTER_KIT_DIR / ".env.example",
        STARTER_KIT_DIR / "PARTICIPANT_GUIDE_NOTES.md",
        STARTER_KIT_DIR / "validate_submission_local.py",
        STARTER_KIT_DIR / "examples" / "hackaton_difc_runner.py",
    ]
    return client.create_code_archive(include_paths=include_paths, archive_path=archive_path, root_dir=ROOT_DIR)


def save_progress(
    submission_path: Path | None,
    debug_path: Path | None,
    architecture_summary: str | None,
    submission_answers: Sequence[SubmissionAnswer],
    debug_rows: Sequence[Dict[str, Any]],
) -> None:
    if submission_path is not None and architecture_summary is not None:
        safe_json_dump(
            submission_path,
            {
                "architecture_summary": architecture_summary,
                "answers": [answer.to_dict() for answer in submission_answers],
            },
        )
        if debug_path is not None:
            safe_json_dump(debug_path, list(debug_rows))


def ensure_runtime_index(corpus: PublicCorpus) -> LegalRuntimeIndex | None:
    db_path = corpus.work_dir / "runtime_index" / "legal_runtime_index.sqlite"
    try:
        if not db_path.exists():
            return build_index(corpus.work_dir, db_path, source_variant="chunked_section_aware")
        return open_index(db_path)
    except Exception:
        return None


def answer_all_questions(
    questions: Sequence[Dict[str, Any]],
    corpus: PublicCorpus,
    provider: str,
    model: str,
    analysis_model: str | None,
    free_text_model: str | None,
    strategy: str,
    top_k_chunks: int,
    limit: int | None = None,
    submission_path: Path | None = None,
    debug_path: Path | None = None,
    architecture_summary: str | None = None,
) -> tuple[list[SubmissionAnswer], list[Dict[str, Any]]]:
    api = APIProcessor(provider=provider)
    bm25_index = BM25Okapi([tokenize_for_bm25(chunk.text) for chunk in corpus.chunks])
    advanced_retriever = AdvancedRetriever(corpus=corpus, work_dir=corpus.work_dir / "advanced_retrieval")
    runtime_index = ensure_runtime_index(corpus)
    analyzer = QuestionAnalyzer(
        provider=provider,
        model=analysis_model or model,
        cache_path=corpus.work_dir / "query_analysis" / f"{provider}__{(analysis_model or model).replace('/', '_')}.json",
        corpus=corpus,
    )
    solver = StructuredWarmupSolver(corpus)
    grounding_index = GroundingIndex(corpus, solver)
    evidence_selector = EvidenceSelector(
        provider=provider,
        model=analysis_model or model,
        cache_path=corpus.work_dir / "grounding_selection" / f"{provider}__{(analysis_model or model).replace('/', '_')}.json",
    )
    submission_answers: list[SubmissionAnswer] = []
    debug_rows: list[Dict[str, Any]] = []
    total_questions = len(questions)

    for index, item in enumerate(questions, start=1):
        question_started = time.perf_counter()
        question = item["question"]
        answer_type = item["answer_type"]
        analysis = analyzer.analyze(question, answer_type)
        question_solver = solver
        if should_isolate_structured_solver(answer_type=answer_type, analysis=analysis):
            question_solver = StructuredWarmupSolver(corpus)
        effective_question, route = merge_routes(corpus, question, analysis)
        pre_decision = question_solver.prepare(question, answer_type, route, [], analysis=analysis)
        prehandled_chunks = list(pre_decision.chunks_override or [])
        if not prehandled_chunks and pre_decision.handled and pre_decision.answer_payload is not None:
            prehandled_chunks = chunks_from_citations(corpus, pre_decision.answer_payload.get("citations", []))
        if pre_decision.handled and prehandled_chunks:
            active_strategy = "prehandled"
            reranked = list(prehandled_chunks)
            selected_chunks = list(prehandled_chunks)
            decision = pre_decision
            answer_chunks = list(prehandled_chunks)
        else:
            if runtime_index is not None and should_expand_with_runtime_index(corpus, route, analysis):
                runtime_candidates = runtime_index_candidate_shas(
                    runtime_index=runtime_index,
                    question=question,
                    effective_question=effective_question,
                    analysis=analysis,
                    current_candidates=route.get("candidate_shas") or [],
                )
                if runtime_candidates:
                    route = dict(route)
                    route["runtime_candidates"] = runtime_candidates
                    route["candidate_shas"] = ordered_unique(list(route["candidate_shas"]) + list(runtime_candidates))
            active_strategy = resolve_runtime_strategy(
                strategy,
                answer_type=answer_type,
                analysis=analysis,
            )
            if should_use_route_guard(corpus, route, analysis, active_strategy):
                active_strategy = "route_guard_hybrid"
            if active_strategy in AdvancedRetriever.SUPPORTED_STRATEGIES:
                reranked = advanced_retriever.retrieve(
                    strategy=active_strategy,
                    question=effective_question,
                    analysis=analysis,
                    candidate_shas=route["candidate_shas"],
                )
            else:
                reranked = run_strategy(corpus, bm25_index, effective_question, route["candidate_shas"], active_strategy)
            selected_chunks = augment_chunks_for_answering(corpus, effective_question, reranked[:top_k_chunks], analysis=analysis)
            decision = question_solver.prepare(question, answer_type, route, reranked, analysis=analysis)
            if decision.chunks_override:
                selected_chunks = decision.chunks_override
            answer_chunks = list(selected_chunks)
            if not decision.handled and answer_type == "free_text" and should_focus_free_text_context(analysis):
                answer_chunks = focused_free_text_chunks(
                    corpus=corpus,
                    grounding_index=grounding_index,
                    question=analysis.standalone_question or question,
                    chunks=selected_chunks,
                    analysis=analysis,
                )

        started = time.perf_counter()
        error_text = None
        try:
            if decision.handled and decision.answer_payload is not None:
                answer_payload = decision.answer_payload
            else:
                answer_model = choose_answer_model(
                    default_model=model,
                    free_text_model=free_text_model,
                    answer_type=answer_type,
                    question=analysis.standalone_question or question,
                    analysis=analysis,
                )
                answer_payload = answer_question(
                    api=api,
                    model=answer_model,
                    question=analysis.standalone_question or question,
                    answer_type=answer_type,
                    chunks=answer_chunks,
                    analysis=analysis,
                    stream=(provider == "openai"),
                )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            answer_payload = {
                "raw_answer": "",
                "citations": [],
                "response_data": {},
            }
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        support_gate_abstained, missing_terms = should_abstain_for_missing_support_terms(
            answer_type=answer_type,
            answer_payload=answer_payload,
            analysis=analysis,
            chunks=answer_chunks,
        )
        absence_model = free_text_model or model
        if support_gate_abstained:
            answer_payload = contextual_absence_payload(
                api=api,
                model=absence_model,
                question=analysis.standalone_question or question,
                analysis=analysis,
                chunks=answer_chunks,
            )
        elif answer_type == "free_text" and answer_is_absent(str(answer_payload.get("raw_answer", "") or "")):
            answer_payload = contextual_absence_payload(
                api=api,
                model=absence_model,
                question=analysis.standalone_question or question,
                analysis=analysis,
                chunks=answer_chunks,
            )
        elif answer_type == "free_text" and should_polish_free_text_answer(
            answer_payload=answer_payload,
            analysis=analysis,
            solver_handled=bool(decision.handled),
        ) and not answer_payload.get("skip_refine"):
            try:
                polish_model = free_text_model or model
                draft_payload = dict(answer_payload)
                draft_rendered = render_structured_free_text_if_needed(
                    solver=question_solver,
                    payload=dict(draft_payload),
                    question=question,
                    route=route,
                    analysis=analysis,
                    solver_handled=bool(decision.handled),
                )
                draft_citations = list(draft_rendered.get("citations", []))
                refined_payload = refine_free_text_answer(
                    api=api,
                    model=polish_model,
                    question=analysis.standalone_question or question,
                    draft_payload=draft_rendered,
                    chunks=answer_chunks,
                    analysis=analysis,
                    multi_source_context=bool(analysis and analysis.needs_multi_document_support),
                )
                refined_rendered = render_structured_free_text_if_needed(
                    solver=question_solver,
                    payload=dict(refined_payload),
                    question=question,
                    route=route,
                    analysis=analysis,
                    solver_handled=bool(decision.handled),
                )
                refined_citations = resolve_answer_citations(refined_rendered.get("citations", []), answer_chunks)
                if prefer_draft_free_text_answer(
                    draft_payload=draft_rendered,
                    refined_payload=refined_rendered,
                    question=analysis.standalone_question or question,
                    analysis=analysis,
                    solver_handled=bool(decision.handled),
                ):
                    answer_payload = dict(draft_rendered)
                    answer_payload["citations"] = draft_citations
                elif should_select_free_text_candidates(
                    analysis=analysis,
                    draft_payload=draft_rendered,
                    refined_payload=refined_rendered,
                ):
                    answer_payload = select_best_free_text_payload(
                        api=api,
                        model=polish_model,
                        question=analysis.standalone_question or question,
                        candidate_a=draft_rendered,
                        candidate_b=refined_rendered,
                        chunks=answer_chunks,
                        analysis=analysis,
                        multi_source_context=bool(analysis and analysis.needs_multi_document_support),
                    )
                elif prefer_draft_citations(draft_citations, refined_citations):
                    answer_payload = dict(refined_rendered)
                    answer_payload["citations"] = draft_citations
                else:
                    answer_payload = dict(refined_rendered)
                    answer_payload["citations"] = refined_citations
            except Exception:
                pass

        if answer_type == "free_text" and decision.handled and answer_payload.get("raw_answer"):
            answer_payload = render_structured_free_text_if_needed(
                solver=question_solver,
                payload=answer_payload,
                question=question,
                route=route,
                analysis=analysis,
                solver_handled=True,
            )

        answer_value = build_submission_value(answer_type, answer_payload)
        grounding_method = "evidence_selector"
        candidate_page_refs: list[str] = []
        grounding_chunks = primary_chunk_refs(answer_payload, answer_chunks) if decision.handled else list(answer_chunks)
        exact_page_refs = grounding_index.exact_page_refs(
            question=analysis.standalone_question or question,
            answer_type=answer_type,
            analysis=analysis,
            selected_chunks=grounding_chunks,
        )
        citation_page_refs = page_refs_from_citations(answer_payload, grounding_chunks)
        if answer_type == "free_text" and answer_is_absent(str(answer_value or "")):
            retrieval_refs = []
            grounding_method = "absence_answer"
            candidate_page_refs = []
        else:
            page_specific_focus = bool(
                analysis is not None
                and set(analysis.support_focus).intersection({"title_page", "first_page", "second_page", "last_page"})
            )
            safe_single_doc_citation_grounding = bool(
                analysis is not None
                and citation_page_refs
                and not analysis.needs_multi_document_support
                and analysis.target_field
                    in {
                        "article_content",
                        "clause_summary",
                        "generic_answer",
                        "enumeration_answer",
                        "comparison_answer",
                        "order_result",
                        "made_by",
                        "administered_by",
                        "publication_text",
                        "enacted_text",
                        "effective_dates",
                        "law_number",
                        "issue_date",
                        "claim_amount",
                        "claim_number",
                        "absence_check",
                        "consultation_deadline",
                        "consultation_email",
                        "consultation_topic",
                    }
            )
            if decision.handled:
                if (
                    answer_type != "free_text"
                    and analysis is not None
                    and analysis.target_field
                    in {
                        "common_parties",
                        "common_judges",
                        "claimant_names",
                        "defendant_name",
                        "claim_amount",
                        "claim_number",
                        "higher_claim_amount_case",
                        "earlier_issue_date_case",
                        "issue_date",
                    }
                    and citation_page_refs
                ):
                    handled_candidate_page_refs = ordered_unique_refs(citation_page_refs)
                elif (
                    answer_type != "free_text"
                    and analysis is not None
                    and analysis.target_field == "absence_check"
                    and "title_page" in set(analysis.support_focus or [])
                    and exact_page_refs
                ):
                    handled_candidate_page_refs = ordered_unique_refs(exact_page_refs)
                elif safe_single_doc_citation_grounding:
                    handled_candidate_page_refs = ordered_unique_refs(citation_page_refs)
                elif (
                    answer_type == "free_text"
                    and analysis is not None
                    and analysis.target_field
                    in {
                        "order_result",
                        "article_content",
                        "clause_summary",
                        "comparison_answer",
                        "enumeration_answer",
                        "amended_laws",
                        "made_by",
                        "administered_by",
                        "publication_text",
                        "enacted_text",
                        "effective_dates",
                    }
                    and citation_page_refs
                ):
                    handled_candidate_page_refs = ordered_unique_refs(citation_page_refs)
                else:
                    handled_candidate_page_refs = ordered_unique_refs(
                        exact_page_refs + citation_page_refs if page_specific_focus and exact_page_refs else (citation_page_refs or exact_page_refs)
                    )
            else:
                handled_candidate_page_refs = ordered_unique_refs(exact_page_refs + citation_page_refs)
            if decision.handled and handled_candidate_page_refs:
                candidate_page_refs, _ = grounding_index.format_candidates(handled_candidate_page_refs, max_pages=10)
                retrieval_refs = normalize_retrieved_pages(
                    [
                        RetrievalRef(doc_id=parsed[0], page_numbers=[parsed[1]])
                        for ref in candidate_page_refs
                        if (parsed := parse_page_ref(ref)) is not None
                    ]
                )
                grounding_method = "exact_pages"
                candidate_context = ""
            elif (
                analysis.needs_multi_document_support
                and analysis.target_titles
                and analysis.target_article_refs
                and handled_candidate_page_refs
            ):
                article_page_refs = exact_page_refs or handled_candidate_page_refs
                retrieval_refs = normalize_retrieved_pages(
                    [
                        RetrievalRef(doc_id=parsed[0], page_numbers=[parsed[1]])
                        for ref in article_page_refs
                        if (parsed := parse_page_ref(ref)) is not None
                    ]
                )
                grounding_method = "exact_multi_doc_articles"
                candidate_page_refs = article_page_refs
                candidate_context = ""
            elif (
                answer_type == "free_text"
                and analysis is not None
                and analysis.target_field == "comparison_answer"
                and analysis.needs_multi_document_support
                and citation_page_refs
                and citations_cover_expected_support(answer_type, analysis, citation_page_refs)
            ):
                retrieval_refs = normalize_retrieved_pages(
                    [
                        RetrievalRef(doc_id=parsed[0], page_numbers=[parsed[1]])
                        for ref in citation_page_refs
                        if (parsed := parse_page_ref(ref)) is not None
                    ]
                )
                grounding_method = "exact_pages"
                candidate_page_refs = citation_page_refs
                candidate_context = ""
            elif (
                answer_type != "free_text"
                and safe_single_doc_citation_grounding
            ):
                preferred_page_refs = citation_page_refs
                if exact_page_refs and (
                    analysis.target_field in {"article_content", "absence_check"}
                    or len(exact_page_refs) <= len(citation_page_refs)
                ):
                    preferred_page_refs = exact_page_refs
                retrieval_refs = normalize_retrieved_pages(
                    [
                        RetrievalRef(doc_id=parsed[0], page_numbers=[parsed[1]])
                        for ref in preferred_page_refs
                        if (parsed := parse_page_ref(ref)) is not None
                    ]
                )
                grounding_method = "exact_pages"
                candidate_page_refs = preferred_page_refs
                candidate_context = ""
            elif (
                answer_type != "free_text"
                and analysis is not None
                and analysis.target_field in {
                    "article_content",
                    "absence_check",
                    "common_parties",
                    "common_judges",
                    "claimant_names",
                    "defendant_name",
                    "claim_amount",
                    "claim_number",
                    "higher_claim_amount_case",
                    "earlier_issue_date_case",
                    "issue_date",
                    "order_result",
                }
                and handled_candidate_page_refs
            ):
                retrieval_refs = normalize_retrieved_pages(
                    [
                        RetrievalRef(doc_id=parsed[0], page_numbers=[parsed[1]])
                        for ref in handled_candidate_page_refs
                        if (parsed := parse_page_ref(ref)) is not None
                    ]
                )
                grounding_method = "exact_pages"
                candidate_page_refs = handled_candidate_page_refs
                candidate_context = ""
            elif should_prefer_exact_grounding_for_deterministic(
                answer_type=answer_type,
                analysis=analysis,
                exact_page_refs=exact_page_refs,
            ):
                retrieval_refs = normalize_retrieved_pages(
                    [
                        RetrievalRef(doc_id=parsed[0], page_numbers=[parsed[1]])
                        for ref in exact_page_refs
                        if (parsed := parse_page_ref(ref)) is not None
                    ]
                )
                grounding_method = "exact_pages"
                candidate_page_refs = list(exact_page_refs)
                candidate_context = ""
            elif citations_cover_expected_support(answer_type, analysis, citation_page_refs):
                retrieval_refs = normalize_retrieved_pages(
                    [
                        RetrievalRef(doc_id=parsed[0], page_numbers=[parsed[1]])
                        for ref in citation_page_refs
                        if (parsed := parse_page_ref(ref)) is not None
                    ]
                )
                grounding_method = "citation_pages"
                candidate_page_refs = citation_page_refs
                candidate_context = ""
            else:
                heuristic_page_refs = page_refs_from_retrieval_refs(
                    select_evidence_refs(
                        corpus=corpus,
                        question=analysis.standalone_question or question,
                        answer_type=answer_type,
                        answer_payload=answer_payload,
                        chunks=answer_chunks,
                        answer_value=answer_value,
                        analysis=analysis,
                    )
                )
                candidate_page_refs = ordered_unique_refs(handled_candidate_page_refs + heuristic_page_refs)
                candidate_page_refs, candidate_context = grounding_index.format_candidates(candidate_page_refs, max_pages=10)
                if candidate_page_refs and candidate_context:
                    selection = evidence_selector.select(
                        question=analysis.standalone_question or question,
                        answer_type=answer_type,
                        answer=str(answer_payload.get("raw_answer", "") or ""),
                        analysis=analysis,
                        candidate_page_refs=candidate_page_refs,
                        candidate_context=candidate_context,
                    )
                    retrieval_refs = normalize_retrieved_pages(
                        [
                            RetrievalRef(doc_id=parsed[0], page_numbers=[parsed[1]])
                            for ref in selection.selected_page_refs
                            if (parsed := parse_page_ref(ref)) is not None
                        ]
                    )
                else:
                    retrieval_refs = []
                if not retrieval_refs:
                    grounding_method = "legacy_fallback"
                    retrieval_refs = select_evidence_refs(
                        corpus=corpus,
                        question=analysis.standalone_question or question,
                        answer_type=answer_type,
                        answer_payload=answer_payload,
                        chunks=answer_chunks,
                        answer_value=answer_value,
                        analysis=analysis,
                    )
                else:
                    grounding_method = f"selector:{selection.coverage}"
        usage = extract_usage(answer_payload)
        response_data = answer_payload.get("response_data") or {}
        pre_answer_ms = int((started - question_started) * 1000)
        ttft_ms = int((response_data.get("ttft_ms", elapsed_ms) or 0) + pre_answer_ms)
        tpot_ms = int(response_data.get("tpot_ms", 0) or 0)
        total_time_ms = int((time.perf_counter() - question_started) * 1000)
        telemetry = Telemetry(
            timing=TimingMetrics(ttft_ms=ttft_ms, tpot_ms=tpot_ms, total_time_ms=total_time_ms),
            retrieval=retrieval_refs,
            usage=usage,
            model_name=str(response_data.get("model") or (free_text_model if answer_type == "free_text" and free_text_model else model)),
        )
        submission_answers.append(
            SubmissionAnswer(
                question_id=item["id"],
                answer=answer_value,
                telemetry=telemetry,
            )
        )
        debug_rows.append(
            {
                "index": index,
                "question_id": item["id"],
                "question": question,
                "answer_type": answer_type,
                "strategy": active_strategy,
                "raw_answer": answer_payload.get("raw_answer"),
                "normalized_answer": normalize_answer(answer_type, answer_payload.get("raw_answer")),
                "submission_answer": answer_value,
                "citations": answer_payload.get("citations", []),
                "retrieval_refs": [ref.__dict__ for ref in retrieval_refs],
                "top_chunk_titles": [chunk["title"] for chunk in selected_chunks],
                "top_chunk_refs": [chunk["ref"] for chunk in selected_chunks],
                "answer_chunk_titles": [chunk["title"] for chunk in answer_chunks],
                "answer_chunk_refs": [chunk["ref"] for chunk in answer_chunks],
                "solver_handled": bool(decision.handled),
                "analysis": analysis.model_dump(),
                "effective_question": effective_question,
                "route_candidate_shas": list(route.get("candidate_shas") or []),
                "route_runtime_candidates": list(route.get("runtime_candidates") or []),
                "grounding_method": grounding_method,
                "candidate_page_refs": candidate_page_refs,
                "support_gate_missing_terms": missing_terms,
                "support_gate_abstained": support_gate_abstained,
                "timing": {
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot_ms,
                    "total_time_ms": total_time_ms,
                },
                "error": error_text,
            }
        )
        status = "error" if error_text else "ok"
        print(f"[{index}/{total_questions}] {item['id']} {status} {elapsed_ms}ms")
        if limit is not None and len(submission_answers) >= limit:
            break

    return submission_answers, debug_rows


def run_platform_pipeline(run_config: PlatformRunConfig) -> dict[str, Any]:
    config = get_config()
    client = EvaluationClient.from_env()

    submission_path = resolve_path(Path(config.submission_path))
    code_archive_path = resolve_path(Path(config.code_archive_path))
    questions_path = resolve_path(Path(config.questions_path))
    docs_dir = resolve_docs_dir(config.docs_dir)
    work_dir = resolve_path(Path(run_config.work_dir))
    docs_dir.mkdir(parents=True, exist_ok=True)
    questions_path.parent.mkdir(parents=True, exist_ok=True)

    questions = download_resources(client, questions_path, docs_dir, reuse_downloads=run_config.reuse_downloads)
    pdf_dir = ensure_pdf_stage_dir(docs_dir)
    artifacts = prepare_docling_artifacts(work_dir, pdf_dir)
    chunked_dir = artifacts["chunked_dir"]
    if run_config.chunk_source == "contextual":
        chunked_dir = build_contextual_chunk_corpus(
            chunked_dir=artifacts["chunked_dir"],
            output_dir=work_dir / "docling" / "chunked_contextual",
        )
    elif run_config.chunk_source == "summary_augmented":
        chunked_dir = build_summary_augmented_chunk_corpus(
            chunked_dir=artifacts["chunked_dir"],
            output_dir=work_dir / "docling" / "chunked_summary_augmented",
        )
    corpus = PublicCorpus(work_dir, pdf_dir, chunked_dir)
    corpus.save_catalog()

    if run_config.download_only:
        summary = {
            "question_count": len(questions),
            "docs_dir": str(docs_dir),
            "pdf_dir": str(pdf_dir),
            "work_dir": str(work_dir),
            "document_count": len(corpus.documents),
            "chunk_count": len(corpus.chunks),
            "chunk_source": run_config.chunk_source,
        }
        safe_json_dump(work_dir / "download_summary.json", summary)
        return summary

    architecture_summary = (
        "Docling parsing + local multilingual-e5 embeddings + local Jina reranker + "
        f"{run_config.strategy} retrieval + chunk_source={run_config.chunk_source} + "
        f"LLM query analysis ({run_config.analysis_model or run_config.model}) + "
        f"page-level evidence selection ({run_config.analysis_model or run_config.model}) + "
        f"{run_config.model} answer model"
    )
    if run_config.free_text_model:
        architecture_summary += f" + {run_config.free_text_model} free_text override"
    if run_config.provider == "openai":
        architecture_summary += " + streaming final-answer telemetry"
    builder = SubmissionBuilder(
        architecture_summary=architecture_summary,
        target_path=submission_path,
    )
    submission_answers, debug_rows = answer_all_questions(
        questions=questions,
        corpus=corpus,
        provider=run_config.provider,
        model=run_config.model,
        analysis_model=run_config.analysis_model,
        free_text_model=run_config.free_text_model,
        strategy=run_config.strategy,
        top_k_chunks=run_config.top_k_chunks,
        limit=run_config.limit or None,
        submission_path=submission_path,
        debug_path=work_dir / "submission_debug.json",
        architecture_summary=architecture_summary,
    )
    for submission_answer in submission_answers:
        builder.add_answer(submission_answer)
    builder.save(submission_path)
    safe_json_dump(work_dir / "submission_debug.json", debug_rows)

    archive_path = build_code_archive(client, code_archive_path)

    result = {
        "submission_path": str(submission_path),
        "code_archive_path": str(archive_path),
        "question_count": len(submission_answers),
        "work_dir": str(work_dir),
    }
    if run_config.submit:
        result["submit_response"] = client.submit_submission(submission_path, archive_path)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the src DIFC pipeline against the Agentic Challenge API.")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--analysis-model", default=None)
    parser.add_argument("--free-text-model", default=None)
    parser.add_argument("--strategy", default="dense_doc_diverse")
    parser.add_argument("--top-k-chunks", type=int, default=5)
    parser.add_argument("--chunk-source", choices=["standard", "contextual", "summary_augmented"], default="standard")
    parser.add_argument("--work-dir", default="challenge_workdir")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reuse-downloads", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--submit", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_config = PlatformRunConfig(
        provider=args.provider,
        model=args.model,
        analysis_model=args.analysis_model,
        free_text_model=args.free_text_model,
        strategy=args.strategy,
        top_k_chunks=args.top_k_chunks,
        chunk_source=args.chunk_source,
        work_dir=args.work_dir,
        limit=args.limit,
        reuse_downloads=args.reuse_downloads,
        download_only=args.download_only,
        submit=args.submit,
    )
    result = run_platform_pipeline(run_config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
