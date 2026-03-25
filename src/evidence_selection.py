from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from pydantic import BaseModel, Field

from src.api_requests import APIProcessor
from src.pdf_text_fallback import merged_pdf_page_texts
from src.public_dataset_eval import PublicCorpus, extract_article_refs, normalize_space
from src.query_analysis import QuestionAnalysis
from src.structured_solver import StructuredWarmupSolver


GROUNDING_SELECTOR_VERSION = "v3"
ABSENT_ANSWER = "there is no information on this question in the provided documents."
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


class EvidenceSelection(BaseModel):
    selected_page_refs: List[str] = Field(default_factory=list)
    rationale: str = ""
    coverage: str = "partial"


def normalize_coverage(value: Any) -> str:
    text = normalized_text(value)
    if "full" in text:
        return "full"
    if "low" in text:
        return "low"
    return "partial"


def normalized_text(value: Any) -> str:
    return normalize_space(str(value or "")).lower()


def significant_title_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value)
        if len(token) > 3 and not (token.isdigit() and len(token) == 4) and token not in TITLE_MATCH_STOPWORDS
    }


def ordered_unique(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def page_ref(sha: str, page: int) -> str:
    return f"{sha}:{int(page)}"


def minimal_term_cover_pages(
    page_term_hits: Dict[int, set[str]],
    required_terms: Sequence[str],
    *,
    max_pages: int = 3,
) -> List[int]:
    uncovered = {term for term in required_terms if term}
    selected: List[int] = []
    available = {page: set(terms) for page, terms in page_term_hits.items() if terms}
    while uncovered and available and len(selected) < max_pages:
        best_page = None
        best_hits: set[str] = set()
        for page, hits in available.items():
            covered = hits & uncovered
            if not covered:
                continue
            if best_page is None or len(covered) > len(best_hits) or (
                len(covered) == len(best_hits) and page < best_page
            ):
                best_page = page
                best_hits = covered
        if best_page is None:
            break
        selected.append(best_page)
        uncovered -= best_hits
        available.pop(best_page, None)
    return sorted(selected)


def parse_page_ref(value: str) -> tuple[str, int] | None:
    match = re.match(r"^([0-9a-f]{64}):(\d+)$", str(value).strip())
    if not match:
        return None
    return match.group(1), int(match.group(2))


def is_absence_answer(answer: str) -> bool:
    text = normalized_text(answer)
    return (
        text == ABSENT_ANSWER
        or "there is no information" in text
        or "contain no information" in text
        or "contains no information" in text
        or "does not mention" in text
        or "do not mention" in text
        or "does not contain information" in text
        or "do not contain information" in text
    )


def match_docs_for_titles(corpus: PublicCorpus, titles: Sequence[str]) -> List[str]:
    rows: List[tuple[float, str]] = []
    for title in titles:
        title_norm = normalized_text(title)
        if len(title_norm) < 4:
            continue
        explicit_ids = extract_article_refs(title)
        # Titles may include a law or case id; prefer direct id matches when present.
        direct_ids = re.findall(r"(?:CFI|CA|ARB|ENF|SCT|DEC|TCD)\s+\d+/\d+|Law No\.\s*\d+\s+of\s+\d{4}|DIFC Law No\.\s*\d+\s+of\s+\d{4}", title, re.I)
        explicit_shas = ordered_unique(
            sha
            for item_id in direct_ids
            for sha in corpus.id_index.get(normalize_space(item_id), [])
        )
        if explicit_shas:
            for rank, sha in enumerate(explicit_shas):
                rows.append((300.0 - rank, sha))
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
                rows.append((score, sha))
    rows.sort(key=lambda item: (-item[0], item[1]))
    return ordered_unique(sha for _, sha in rows)


def article_refs_for_title(article_refs: Sequence[str], title: str, index: int, total_titles: int) -> List[str]:
    title_norm = normalized_text(title)
    title_tokens = {token for token in re.findall(r"[a-z0-9]+", title_norm) if len(token) > 3}
    explicit: List[str] = []
    plain: List[str] = []
    for article_ref in article_refs:
        ref_norm = normalized_text(article_ref)
        ref_tokens = {token for token in re.findall(r"[a-z0-9]+", ref_norm) if len(token) > 3}
        if title_tokens and len(title_tokens & ref_tokens) >= max(1, min(2, len(title_tokens))):
            explicit.append(article_ref)
        else:
            plain.append(article_ref)
    if explicit:
        return ordered_unique(explicit)
    if total_titles > 1 and len(plain) >= total_titles and index < len(plain):
        return [plain[index]]
    return ordered_unique(plain)


def analysis_target_docs_with_articles(
    corpus: PublicCorpus,
    analysis: QuestionAnalysis | None,
) -> List[tuple[str, List[str]]]:
    if analysis is None:
        return []
    rows: List[tuple[str, List[str]]] = []
    seen = set()
    if analysis.target_titles:
        for index, title in enumerate(analysis.target_titles):
            shas = match_docs_for_titles(corpus, [title])
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
    for case_id in analysis.target_case_ids:
        for sha in corpus.id_index.get(case_id, []):
            if sha in seen:
                continue
            rows.append((sha, list(analysis.target_article_refs)))
            seen.add(sha)
    for law_id in analysis.target_law_ids:
        for sha in corpus.id_index.get(law_id, []):
            if sha in seen:
                continue
            rows.append((sha, list(analysis.target_article_refs)))
            seen.add(sha)
    return rows


class GroundingIndex:
    def __init__(self, corpus: PublicCorpus, solver: StructuredWarmupSolver):
        self.corpus = corpus
        self.solver = solver
        self.page_texts: Dict[str, Dict[int, str]] = {}
        self.article_pages: Dict[str, Dict[str, List[int]]] = {}
        self._build()

    def _build(self) -> None:
        for sha, payload in self.corpus.documents_payload.items():
            self.page_texts[sha] = merged_pdf_page_texts(
                self.corpus.pdf_dir,
                sha,
                payload["content"]["pages"],
            )
            self.article_pages[sha] = self._build_article_map(sha, payload)

    def _build_article_map(self, sha: str, payload: Dict[str, Any]) -> Dict[str, List[int]]:
        document = self.corpus.documents[sha]
        if document.kind not in {"law", "regulation"}:
            return {}
        article_pages: Dict[str, List[int]] = {}
        current_article: str | None = None
        for page in payload["content"]["pages"]:
            page_number = int(page["page"])
            text = str(page["text"] or "")
            headings = re.findall(r"(?:^|\n)##\s*(\d+)\.\s+[^\n]+", text)
            if headings:
                current_article = headings[-1]
                article_pages.setdefault(current_article, [])
            if current_article is not None:
                article_pages.setdefault(current_article, []).append(page_number)
        return {key: sorted(dict.fromkeys(value)) for key, value in article_pages.items()}

    def page_text(self, sha: str, page: int) -> str:
        return self.page_texts.get(sha, {}).get(int(page), "")

    def article_pages_for_doc(self, sha: str, article_refs: Sequence[str]) -> List[int]:
        result: List[int] = []
        for article_ref in article_refs:
            result.extend(self._article_pages_from_chunks(sha, article_ref))
        return ordered_unique(str(page) for page in result) and [int(page) for page in ordered_unique(str(page) for page in result)]

    def _article_pages_from_chunks(self, sha: str, article_ref: str) -> List[int]:
        payload = self.corpus.documents_payload[sha]
        article_norm = normalized_text(article_ref)
        number_match = re.search(r"article\s+(\d+)", article_ref, re.I)
        parts = [part.lower() for part in re.findall(r"\(([^)]+)\)", article_ref)]
        pages: List[int] = []
        for chunk in payload["content"]["chunks"]:
            text_norm = normalized_text(chunk["text"])
            raw_text = str(chunk["text"] or "")
            page_number = int(chunk["page"])
            page_text = self.page_text(sha, page_number)
            page_text_norm = normalized_text(page_text)
            if "table of contents" in text_norm or "contents" in text_norm and raw_text.count("|") >= 4:
                continue
            if "..." in raw_text and raw_text.count("|") >= 2:
                continue
            if raw_text.count("|") >= 3 and "##" not in raw_text:
                continue
            if "contents" in page_text_norm and page_text.count("|") >= 4:
                continue
            if "..." in page_text and page_text.count("|") >= 4:
                continue
            score = 0.0
            if article_norm and article_norm in text_norm:
                score += 40.0
                if any(
                    f"{prefix}{article_norm}" in text_norm
                    for prefix in (
                        "under ",
                        "pursuant to ",
                        "specified in ",
                        "in accordance with ",
                        "reference to ",
                    )
                ):
                    score -= 30.0
            if number_match and re.search(rf"(?:^|\b|##\s*){number_match.group(1)}\.", text_norm):
                score += 24.0
            if parts:
                parent_marker = parts[0] if len(parts) >= 2 else None
                if parent_marker and re.search(rf"\(\s*{re.escape(parent_marker)}\s*\)", text_norm):
                    score += 18.0
            for part in parts:
                if re.search(rf"\(\s*{re.escape(part)}\s*\)", text_norm):
                    score += 12.0
            if score >= 24.0:
                pages.append(page_number)
        return pages

    def pages_with_phrase(self, sha: str, patterns: Sequence[str]) -> List[int]:
        result: List[int] = []
        for page, text in self.page_texts.get(sha, {}).items():
            text_norm = normalized_text(text)
            if any(pattern in text_norm for pattern in patterns):
                result.append(page)
        return result

    def exact_page_refs(
        self,
        question: str,
        answer_type: str,
        analysis: QuestionAnalysis | None,
        selected_chunks: Sequence[Dict[str, Any]],
    ) -> List[str]:
        refs: List[str] = []
        focus = set(analysis.support_focus) if analysis is not None else set()
        field = analysis.target_field if analysis is not None else ""
        structured_law_fields = {"law_number", "made_by", "administered_by", "publication_text", "enacted_text", "effective_dates"}
        question_norm = normalized_text(question)

        for case_id in (analysis.target_case_ids if analysis is not None else []):
            case = self.solver.case_by_id.get(case_id)
            if not case:
                continue
            if analysis and analysis.target_field in {"issue_date", "earlier_issue_date_case"}:
                refs.append(page_ref(case.sha, case.issue_page or case.first_page))
            if analysis and analysis.target_field in {"common_parties", "common_judges", "claimant_names", "defendant_name"}:
                if analysis.target_field in {"common_parties", "common_judges"} and any(
                    marker in question_norm for marker in ("all documents", "full case files", "full case file")
                ):
                    for sha in ordered_unique(self.corpus.id_index.get(case_id, [])) or [case.sha]:
                        refs.append(page_ref(sha, 1))
                else:
                    refs.append(page_ref(case.sha, case.first_page))
            if analysis and analysis.target_field == "claim_amount":
                refs.append(page_ref(case.sha, case.claim_amount_page or case.first_page))
            if analysis and analysis.target_field == "claim_number":
                refs.append(page_ref(case.sha, case.origin_claim_page or case.first_page))
            if analysis and analysis.target_field == "order_result":
                if "page 2" in question_norm or "second page" in question_norm:
                    refs.append(page_ref(case.sha, 2))
                elif "first page" in question_norm or "page 1" in question_norm:
                    refs.append(page_ref(case.sha, 1))
                elif "last_page" in focus or "last page" in question_norm:
                    refs.append(page_ref(case.sha, self.corpus.documents[case.sha].page_count))
                else:
                    refs.extend(page_ref(case.sha, page) for page in case.order_pages[:2])
            if field not in {"issue_date", "earlier_issue_date_case", "claim_amount", "claim_number", "claimant_names", "defendant_name", "common_parties", "common_judges"} and focus.intersection({"first_page", "issue_date_line", "party_block", "judge_block"}):
                refs.append(page_ref(case.sha, case.first_page))
            if field != "order_result" and focus.intersection({"order_section", "conclusion_section"}):
                refs.extend(page_ref(case.sha, page) for page in case.order_pages[:2])

        for law_id in (analysis.target_law_ids if analysis is not None else []):
            law = self.solver.law_by_id.get(law_id)
            if not law:
                continue
            if field in structured_law_fields:
                refs.extend(self._law_field_refs(law, analysis))

        if analysis is not None and analysis.target_titles:
            for sha, scoped_article_refs in analysis_target_docs_with_articles(self.corpus, analysis):
                law = self.solver.law_by_id.get(self.corpus.documents[sha].title)
                if law and field in structured_law_fields:
                    refs.extend(self._law_field_refs(law, analysis))
                if scoped_article_refs and field not in structured_law_fields:
                    exact_support_refs: List[str] = []
                    if len(scoped_article_refs) == 1:
                        exact_clause = self.solver._article_clause_answer(sha, scoped_article_refs[0], question)
                        if exact_clause is not None:
                            exact_support_refs = [
                                page_ref(chunk["sha"], int(chunk["page"]))
                                for chunk in exact_clause[1]
                            ]
                    article_page_refs = exact_support_refs or [page_ref(sha, page) for page in self.article_pages_for_doc(sha, scoped_article_refs)]
                    if article_page_refs:
                        refs.extend(article_page_refs)
                    else:
                        refs.extend(
                            page_ref(chunk["sha"], int(chunk["page"]))
                            for chunk in selected_chunks
                            if chunk["sha"] == sha
                        )
                elif field not in structured_law_fields and focus.intersection({"title_page", "administration_clause", "enactment_clause", "publication_line", "commencement_clause"}):
                    selected_title_refs = [
                        page_ref(chunk["sha"], int(chunk["page"]))
                        for chunk in selected_chunks
                        if chunk["sha"] == sha
                    ]
                    if selected_title_refs and focus.intersection({"administration_clause", "enactment_clause", "publication_line", "commencement_clause"}):
                        refs.extend(selected_title_refs[:2])
                    else:
                        refs.append(page_ref(sha, 1))

        if (analysis is None or not refs) and field not in structured_law_fields:
            article_refs = extract_article_refs(question)
            if article_refs:
                for chunk in selected_chunks:
                    sha = chunk["sha"]
                    for page in self.article_pages_for_doc(sha, article_refs):
                        refs.append(page_ref(sha, page))

        if analysis is not None and field == "absence_check" and answer_type != "free_text" and analysis.must_support_terms:
            target_shas = ordered_unique(
                [
                    *[sha for case_id in analysis.target_case_ids for sha in self.corpus.id_index.get(case_id, [])],
                    *[sha for law_id in analysis.target_law_ids for sha in self.corpus.id_index.get(law_id, [])],
                    *[sha for sha, _ in analysis_target_docs_with_articles(self.corpus, analysis)],
                ]
            )
            term_norms = [normalized_text(term) for term in analysis.must_support_terms if len(normalized_text(term)) > 3]
            anchor_terms = {
                normalized_text(term)
                for term in [
                    *(analysis.target_case_ids or []),
                    *(analysis.target_law_ids or []),
                    *(analysis.target_titles or []),
                ]
                if len(normalized_text(term)) > 3
            }
            content_term_norms = [
                term
                for term in term_norms
                if term not in anchor_terms and not any(term in anchor for anchor in anchor_terms)
            ]
            page_match_terms = content_term_norms or term_norms
            matched_refs: List[str] = []
            for sha in target_shas:
                page_term_hits = {
                    page: {term for term in page_match_terms if term in normalized_text(text)}
                    for page, text in self.page_texts.get(sha, {}).items()
                }
                matched_pages = minimal_term_cover_pages(page_term_hits, page_match_terms, max_pages=3)
                if not matched_pages:
                    matched_pages = [
                        page
                        for page, hits in page_term_hits.items()
                        if hits
                    ][:4]
                for page in matched_pages:
                    matched_refs.append(page_ref(sha, page))
            if matched_refs:
                if (
                    "title_page" in focus
                    and len(target_shas) == 1
                    and self.corpus.documents.get(target_shas[0]) is not None
                    and self.corpus.documents[target_shas[0]].kind in {"law", "regulation"}
                ):
                    title_ref = page_ref(target_shas[0], 1)
                    if title_ref not in matched_refs:
                        matched_refs = [title_ref, *matched_refs]
                refs = matched_refs

        if field not in {"issue_date", "claim_amount", "law_number", "made_by", "administered_by", "publication_text", "enacted_text"} and ("page 2" in question_norm or "second page" in question_norm or "second_page" in focus):
            for chunk in selected_chunks[:2]:
                refs.append(page_ref(chunk["sha"], 2))

        if not refs:
            refs.extend(page_ref(chunk["sha"], int(chunk["page"])) for chunk in selected_chunks[:4])
        elif (
            analysis is not None
            and analysis.target_field == "generic_answer"
            and not (analysis.needs_multi_document_support and analysis.target_titles and analysis.target_article_refs)
        ):
            refs.extend(page_ref(chunk["sha"], int(chunk["page"])) for chunk in selected_chunks[:2])
        return ordered_unique(refs)

    def _law_field_refs(self, law: Any, analysis: QuestionAnalysis | None) -> List[str]:
        if analysis is None:
            return [page_ref(law.sha, law.page_one)]
        field = analysis.target_field
        refs: List[str] = []
        if field in {"law_number", "publication_text"}:
            refs.append(page_ref(law.sha, law.publication_page or law.page_one))
        if field == "made_by":
            refs.append(page_ref(law.sha, law.made_by_page or law.page_one))
        if field == "administered_by":
            refs.append(page_ref(law.sha, law.administered_by_page or law.page_one))
        if field == "enacted_text":
            refs.append(page_ref(law.sha, law.enacted_page or law.page_one))
        if field == "effective_dates":
            refs.append(page_ref(law.sha, law.effective_date_page or law.commencement_page or law.page_one))
        if not refs:
            refs.append(page_ref(law.sha, law.page_one))
        return refs

    def format_candidates(self, page_refs: Sequence[str], max_pages: int = 10, max_chars: int = 1400) -> tuple[List[str], str]:
        selected: List[str] = []
        parts: List[str] = []
        for ref in page_refs:
            parsed = parse_page_ref(ref)
            if parsed is None:
                continue
            sha, page = parsed
            text = normalize_space(self.page_text(sha, page))
            if not text:
                continue
            selected.append(ref)
            title = self.corpus.documents[sha].title
            excerpt = text[:max_chars]
            parts.append(f"[PAGE {ref} | title={title}]\n{excerpt}")
            if len(selected) >= max_pages:
                break
        return selected, "\n\n---\n\n".join(parts)


class EvidenceSelector:
    def __init__(self, provider: str, model: str, cache_path: Path):
        self.provider = provider
        self.model = model
        self.api = APIProcessor(provider=provider)
        self.cache_path = cache_path
        self.cache = self._load_cache()

    def _load_cache(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
        return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as handle:
            json.dump(self.cache, handle, indent=2, ensure_ascii=False)

    def _cache_key(
        self,
        question: str,
        answer_type: str,
        answer: str,
        page_refs: Sequence[str],
    ) -> str:
        raw = "\n".join([GROUNDING_SELECTOR_VERSION, self.provider, self.model, question, answer_type, answer, *page_refs])
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def select(
        self,
        question: str,
        answer_type: str,
        answer: str,
        analysis: QuestionAnalysis | None,
        candidate_page_refs: Sequence[str],
        candidate_context: str,
    ) -> EvidenceSelection:
        if is_absence_answer(answer):
            return EvidenceSelection(selected_page_refs=[], rationale="Unanswerable", coverage="full")
        if not candidate_page_refs:
            return EvidenceSelection(selected_page_refs=[], rationale="No candidates", coverage="low")

        key = self._cache_key(question, answer_type, answer, candidate_page_refs)
        cached = self.cache.get(key)
        if cached:
            try:
                return EvidenceSelection.model_validate(cached)
            except Exception:
                pass

        try:
            response = self.api.send_message(
                model=self.model,
                temperature=0.0,
                system_content=self._system_prompt(),
                human_content=self._user_prompt(question, answer_type, answer, analysis, candidate_context),
                is_structured=True,
                response_format=EvidenceSelection,
                max_tokens=300,
                request_timeout=60,
            )
            selection = EvidenceSelection.model_validate(response)
        except Exception:
            selection = EvidenceSelection(
                selected_page_refs=list(candidate_page_refs[:2]),
                rationale="Fallback to top candidate pages",
                coverage="partial",
            )

        valid = [ref for ref in selection.selected_page_refs if ref in set(candidate_page_refs)]
        if not valid:
            valid = list(candidate_page_refs[:2])
        selection = EvidenceSelection(
            selected_page_refs=ordered_unique(valid),
            rationale=selection.rationale,
            coverage=normalize_coverage(selection.coverage),
        )
        self.cache[key] = selection.model_dump()
        self._save_cache()
        return selection

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are the grounding selection layer for a production legal RAG system. "
            "Do not answer the question. Choose the minimal set of page refs that support the final answer. "
            "Recall matters more than precision, but do not add unrelated pages. "
            "If the question compares multiple sources, include support from every source needed. "
            "If the question analysis specifies a page or section focus, prefer pages that match that focus. "
            "If the analysis includes must_support_terms, pages missing those concepts are weak support. "
            "Select only page refs that appear in the provided candidate pages."
        )

    @staticmethod
    def _user_prompt(
        question: str,
        answer_type: str,
        answer: str,
        analysis: QuestionAnalysis | None,
        candidate_context: str,
    ) -> str:
        analysis_text = json.dumps(analysis.model_dump(), ensure_ascii=False) if analysis is not None else "{}"
        return (
            f"Question: {question}\n"
            f"Answer type: {answer_type}\n"
            f"Final answer: {answer}\n"
            f"Question analysis: {analysis_text}\n\n"
            "Candidate pages are below. Return the smallest set of PAGE refs that is sufficient to support the answer. "
            "If a fully correct answer would require two sources, include both. "
            "Coverage values: full, partial, low.\n\n"
            f"{candidate_context}"
        )
