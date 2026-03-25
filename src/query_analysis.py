from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, List, Sequence

from pydantic import BaseModel, Field

from src.api_requests import APIProcessor
from src.evidence_contracts import EntityAnchor, SlotRequirement, TaskContract
from src.public_dataset_eval import extract_article_refs, extract_case_ids, extract_law_ids, normalize_space


STRUCTURED_FIELDS = {
    "issue_date",
    "earlier_issue_date_case",
    "common_parties",
    "common_judges",
    "claimant_names",
    "claimant_count",
    "defendant_name",
    "claim_amount",
    "claim_number",
    "higher_claim_amount_case",
    "law_number",
    "same_law_number",
    "made_by",
    "administered_by",
    "publication_text",
    "enacted_text",
    "effective_dates",
    "amended_laws",
    "consultation_deadline",
    "consultation_email",
    "consultation_topic",
    "consultation_issuer",
    "case_relation",
}
ANALYSIS_PROMPT_VERSION = "v18"

TITLE_ANCHOR_STOPWORDS = {
    "law",
    "laws",
    "difc",
    "regulation",
    "regulations",
    "the",
    "and",
    "of",
    "in",
    "on",
    "for",
    "application",
    "commercial",
    "civil",
}

NOISY_TITLE_MARKERS = (
    "consultation paper",
    "court administrative orders",
    " annex ",
    "# annex",
    "part 1 citation",
    "small claims tribunal",
    "dfsa translation following publication",
    "regarding the financial free zones",
    "hereby enact the following law",
)
QUESTION_INSTRUMENT_RE = re.compile(
    r"(?:about|under|according\s+to|contrary\s+to)\s+Article\s+\d+(?:\([^)]+\))*\s+of\s+(?:the\s+)?(?P<title>[A-Za-z][A-Za-z/&'()\- ]+?(?:Law|Regulations|Rules)(?:\s+\d{4})?(?:\s+(?:DIFC\s+)?Law\s+No\.?\s*\(?\d+\)?\s+of\s+\d{4})?)",
    re.I,
)
BARE_QUESTION_INSTRUMENT_RE = re.compile(
    r"(?:\bof\b|\bunder\b|\bfor\b|\bin\b|\bon\b)\s+(?:the\s+)?(?P<title>[A-Za-z][A-Za-z0-9/&'()\- ]+?(?:Law|Regulations|Rules)(?:\s+\d{4})?)\b",
    re.I,
)
DEICTIC_QUESTION_INSTRUMENT_RE = re.compile(
    r"\b(?:this|these)\s+(?P<title>[A-Za-z][A-Za-z0-9/&'()\- ]+?(?:Law|Regulations|Rules)(?:\s+\d{4})?)\b",
    re.I,
)
QUOTED_TITLE_RE = re.compile(r"[\"“”']([^\"“”']{6,200})[\"“”']")


def question_implies_multi_support(question_norm: str) -> bool:
    scoped_markers = (
        " both ",
        " between ",
        " compare ",
        " compared to ",
        " compared with ",
        " same judge",
        " same judges",
        " same party",
        " same parties",
        " in common ",
        " common party",
        " common parties",
        " common judge",
        " common judges",
    )
    return any(marker in f" {question_norm} " for marker in scoped_markers)


def question_is_absence_sensitive(question_norm: str) -> bool:
    markers = (
        " contain ",
        " contains ",
        " mention ",
        " mentions ",
        " specify ",
        " specifies ",
        " provide ",
        " provides ",
        " any information",
        " can be inferred",
        " miranda ",
        " jury ",
        " plea bargain",
        " parole ",
    )
    wrapped = f" {question_norm} "
    return any(marker in wrapped for marker in markers)


def question_is_party_overlap(question_norm: str) -> bool:
    wrapped = f" {question_norm} "
    judge_markers = (
        " judge ",
        " judges ",
        " justice ",
        " justices ",
        " presided ",
        " preside ",
        " presiding ",
    )
    if any(marker in wrapped for marker in judge_markers):
        return False
    overlap_markers = (
        " same legal entities",
        " same legal entity",
        " same main party",
        " same main parties",
        " common to both cases",
        " common to both",
        " appeared in both cases",
        " appear in both cases",
        " appears as a main party in both",
        " named as a main party in both",
        " main party to both",
        " main party in both",
        " party in both cases",
        " legal entity or individual",
        " person or company",
        " individual or company",
        " involve any of the same",
        " main party common",
    )
    overlap_trigger = any(marker in wrapped for marker in overlap_markers)
    return overlap_trigger and any(marker in wrapped for marker in (" both ", " common ", " same "))


def question_is_judge_overlap(question_norm: str) -> bool:
    wrapped = f" {question_norm} "
    overlap_markers = (
        " judge involved in both",
        " judge who presided over both",
        " same judge",
        " same judges",
        " common judge",
        " common judges",
        " judge who participated in both",
        " judge participated in both",
        " participated in both cases",
        " preside over proceedings in both",
        " did any judge appear in both",
        " any judge appear in both",
    )
    return ("judge" in wrapped or "judges" in wrapped) and any(marker in wrapped for marker in overlap_markers)


def question_is_enumeration_like(question_norm: str, answer_type: str) -> bool:
    if answer_type == "names":
        return True
    markers = (
        " which ",
        " what bodies",
        " what entities",
        " list ",
        " identify all ",
        " what are the ",
    )
    wrapped = f" {question_norm} "
    return any(marker in wrapped for marker in markers)


def question_is_clause_reasoning(question_norm: str) -> bool:
    markers = (
        " under what conditions",
        " in what circumstances",
        " how does ",
        " explain ",
        " summarize ",
        " what does ",
        " what is the purpose",
        " what are the obligations",
        " what obligations",
        " what are the consequences",
        " what happens if",
        " how is ",
    )
    return any(marker in question_norm for marker in markers)


def question_is_consultation_paper(question_norm: str, target_titles: Sequence[str]) -> bool:
    if "consultation paper" in question_norm:
        return True
    return any("consultation paper" in normalize_space(title).lower() for title in target_titles)


def question_targets_claimant_side(question_norm: str) -> bool:
    wrapped = f" {question_norm} "
    markers = (
        " claimant ",
        " claimants ",
        " applicant ",
        " applicants ",
        " appellant ",
        " appellants ",
        " judgment creditor ",
        " award creditor ",
        " sought recognition ",
        " sought enforcement ",
        " brought the claim ",
        " held the judgment being enforced ",
    )
    return any(marker in wrapped for marker in markers)


def question_targets_defendant_side(question_norm: str) -> bool:
    wrapped = f" {question_norm} "
    markers = (
        " defendant ",
        " defendants ",
        " respondent ",
        " respondents ",
        " opposed the appeal ",
        " oppose the appeal ",
        " opposed recognition ",
        " resisted recognition ",
        " enforcement sought in ",
        " enforcement sought against ",
        " judgment debtor ",
        " opposed the recognition ",
        " who opposed ",
    )
    return any(marker in wrapped for marker in markers)


def title_anchor_score(question: str, title: str) -> float:
    question_norm = normalize_space(question).lower()
    title_norm = normalize_space(title).lower().replace("#", " ")
    if not title_norm:
        return 0.0
    explicit = normalize_space(re.sub(r"\b20\d{2}\b", "", title_norm))
    if explicit and explicit in question_norm:
        return 120.0 + len(explicit)
    title_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", title_norm)
        if len(token) > 3 and token not in TITLE_ANCHOR_STOPWORDS
    }
    if not title_tokens:
        return 0.0
    overlap = sum(1 for token in title_tokens if token in question_norm)
    if overlap == 0:
        return 0.0
    missing = len(title_tokens) - overlap
    score = overlap * 15.0 - missing * 8.0
    if any(token in question_norm for token in ("limited", "liability", "general", "employment", "foundations", "operating", "trust", "personal", "common", "reporting", "standard")):
        score += overlap * 2.0
    return score


def _normalized_title_surface(title: str) -> str:
    title_norm = normalize_space(title).lower().replace("#", " ")
    title_norm = normalize_space(re.sub(r"\b20\d{2}\b", "", title_norm))
    title_norm = re.sub(r"\s+", " ", title_norm).strip()
    return title_norm


def _normalized_title_surface_exact(title: str) -> str:
    title_norm = normalize_space(title).lower().replace("#", " ")
    title_norm = re.sub(r"\s+", " ", title_norm).strip()
    return title_norm


def _question_contains_title(question: str, title: str) -> bool:
    question_norm = normalize_space(question).lower()
    title_norm = _normalized_title_surface(title)
    if not title_norm or len(title_norm) < 6:
        return False
    return re.search(rf"\b{re.escape(title_norm)}\b", question_norm) is not None


def _question_instrument_mentions(question: str) -> List[str]:
    mentions: List[str] = []
    for match in QUESTION_INSTRUMENT_RE.finditer(question or ""):
        mention = normalize_space(match.group("title") or "")
        if mention:
            mentions.append(mention)
    return list(dict.fromkeys(mentions))


def _bare_question_instrument_titles(question: str) -> List[str]:
    titles: List[str] = []
    question_norm = normalize_space(question)
    for match in BARE_QUESTION_INSTRUMENT_RE.finditer(question_norm or ""):
        title = normalize_space(match.group("title") or "")
        if not title:
            continue
        exact = _normalized_title_surface_exact(title)
        if not exact or len(exact) < 6:
            continue
        if any(token in exact for token in ("this ", "these ", "that ", "those ")):
            continue
        if exact in {"these regulations", "the regulations", "this law", "the law", "these rules", "the rules"}:
            continue
        titles.append(title)
    return list(dict.fromkeys(titles))


def _deictic_question_instrument_titles(question: str) -> List[str]:
    titles: List[str] = []
    for match in DEICTIC_QUESTION_INSTRUMENT_RE.finditer(normalize_space(question or "")):
        title = normalize_space(match.group("title") or "")
        if not title:
            continue
        exact = _normalized_title_surface_exact(title)
        if exact in {"law", "the law", "regulations", "the regulations", "rules", "the rules"}:
            continue
        titles.append(title)
    return list(dict.fromkeys(titles))


def _quoted_question_titles(question: str) -> List[str]:
    titles: List[str] = []
    for match in QUOTED_TITLE_RE.finditer(question or ""):
        title = normalize_space(match.group(1) or "")
        if title:
            titles.append(title)
    return list(dict.fromkeys(titles))


def _title_matches_instrument_mention(title: str, mention: str) -> bool:
    title_exact = _normalized_title_surface_exact(title)
    mention_exact = _normalized_title_surface_exact(mention)
    if not title_exact or not mention_exact:
        return False
    mention_instrument = _instrument_type(mention)
    title_instrument = _instrument_type(title)
    if mention_instrument and title_instrument and mention_instrument != title_instrument:
        return False
    mention_year = re.search(r"\b(19|20)\d{2}\b", mention_exact)
    if mention_year and mention_year.group(0) not in title_exact:
        return False
    if mention_exact in title_exact:
        return True
    mention_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", mention_exact)
        if len(token) > 3 and token not in TITLE_ANCHOR_STOPWORDS and not token.isdigit()
    }
    title_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", title_exact)
        if len(token) > 3 and token not in TITLE_ANCHOR_STOPWORDS and not token.isdigit()
    }
    if not mention_tokens:
        return False
    return mention_tokens.issubset(title_tokens)


def _is_noisy_title(title: str) -> bool:
    title_norm = normalize_space(title).lower().replace("#", " ")
    if len(title_norm) > 180:
        return True
    return any(marker in title_norm for marker in NOISY_TITLE_MARKERS)


def _prefer_question_instrument_titles(question_instrument: str, titles: List[str]) -> List[str]:
    if question_instrument not in {"law", "regulation", "rules"}:
        return titles
    preferred = [title for title in titles if _instrument_type(title) == question_instrument]
    return preferred or titles


def _instrument_type(text: str) -> str:
    text_norm = normalize_space(text).lower()
    head = re.split(r"[\(\[:\-]", text_norm, maxsplit=1)[0]
    if re.search(r"\bregulations?\b", text_norm):
        if re.search(r"\bregulations?\b", head) or not re.search(r"\blaw\b", head):
            return "regulation"
    if re.search(r"\brules\b", text_norm):
        if re.search(r"\brules\b", head) or not re.search(r"\blaw\b", head):
            return "rules"
    if re.search(r"\blaw\b", text_norm):
        return "law"
    return ""


def filter_target_titles_for_question(question: str, titles: List[str]) -> List[str]:
    if not titles:
        return []
    question_norm = normalize_space(question).lower()
    question_law_ids = extract_law_ids(question)
    law_or_article_question = bool(question_law_ids or extract_article_refs(question) or " law " in f" {question_norm} ")
    question_instrument = _instrument_type(question)
    question_mentions = _question_instrument_mentions(question)
    non_noisy_titles = [title for title in titles if not _is_noisy_title(title)]
    if non_noisy_titles:
        titles = non_noisy_titles
    if question_law_ids:
        exact_law_matches = [
            title
            for title in titles
            if set(extract_law_ids(title)) & set(question_law_ids)
        ]
        if exact_law_matches:
            return list(dict.fromkeys(exact_law_matches))
        return []
    if question_mentions:
        mention_matches = [
            title
            for title in titles
            if any(_title_matches_instrument_mention(title, mention) for mention in question_mentions)
        ]
        if mention_matches:
            return _prefer_question_instrument_titles(question_instrument, list(dict.fromkeys(mention_matches)))
    explicit_hits = [title for title in titles if _question_contains_title(question, title)]
    if explicit_hits:
        return _prefer_question_instrument_titles(question_instrument, list(dict.fromkeys(explicit_hits)))
    scored = [(title_anchor_score(question, title), title) for title in titles]
    scored = [(score, title) for score, title in scored if score > 0]
    if not scored:
        titles = _prefer_question_instrument_titles(question_instrument, titles)
        if law_or_article_question and all(_is_noisy_title(title) for title in titles):
            return []
        return titles
    anchored = [title for score, title in scored if score >= 18.0]
    if len(anchored) >= 2:
        anchored = _prefer_question_instrument_titles(question_instrument, anchored)
        if law_or_article_question:
            anchored = [title for title in anchored if not _is_noisy_title(title)]
        return anchored
    best_score = max(score for score, _ in scored)
    keep = [
        title
        for score, title in scored
        if score >= max(18.0, best_score - 6.0)
    ]
    keep = _prefer_question_instrument_titles(question_instrument, keep)
    if law_or_article_question:
        keep = [title for title in keep if not _is_noisy_title(title)]
        if not keep:
            return []
    return keep or titles


class QuestionAnalysis(BaseModel):
    standalone_question: str = ""
    retrieval_query: str = ""
    task_family: str = "generic_rag"
    target_field: str = "generic_answer"
    support_focus: List[str] = Field(default_factory=list)
    target_case_ids: List[str] = Field(default_factory=list)
    target_law_ids: List[str] = Field(default_factory=list)
    target_article_refs: List[str] = Field(default_factory=list)
    target_titles: List[str] = Field(default_factory=list)
    must_support_terms: List[str] = Field(default_factory=list)
    intent_tags: List[str] = Field(default_factory=list)
    needs_multi_document_support: bool = False
    use_structured_executor: bool = False
    confidence: float = 0.0


def _anchor_from_surface(kind: str, surface: str) -> EntityAnchor:
    return EntityAnchor(kind=kind, canonical_id=surface, surface_form=surface, confidence=0.9)


def _page_focus_from_support_focus(support_focus: List[str]) -> List[str]:
    allowed = {
        "title_page",
        "first_page",
        "second_page",
        "last_page",
        "article_section",
        "order_section",
        "conclusion_section",
    }
    return [item for item in support_focus if item in allowed]


def _slot(name: str, description: str, cardinality: str = "one", required: bool = True) -> SlotRequirement:
    return SlotRequirement(name=name, description=description, cardinality=cardinality, required=required)


def build_task_contract(question: str, answer_type: str, analysis: QuestionAnalysis) -> TaskContract:
    anchors: List[EntityAnchor] = []
    anchors.extend(_anchor_from_surface("case", item) for item in analysis.target_case_ids)
    anchors.extend(_anchor_from_surface("law", item) for item in analysis.target_law_ids)
    anchors.extend(_anchor_from_surface("article", item) for item in analysis.target_article_refs)
    anchors.extend(_anchor_from_surface("law", item) for item in analysis.target_titles)
    answer_kind = answer_type if answer_type in {"boolean", "number", "name", "names", "date", "free_text", "null"} else "free_text"
    target_field = analysis.target_field
    question_norm = normalize_space(question).lower()

    field_contracts: dict[str, tuple[str, list[SlotRequirement]]] = {
        "issue_date": ("extract_scalar", [_slot("issue_date", "Issue date of the target case.")]),
        "claim_amount": ("extract_scalar", [_slot("claim_amount", "Claim amount or claim value for the target case.")]),
        "claim_number": ("extract_scalar", [_slot("claim_number", "Origin claim number for the target appeal case.")]),
        "law_number": ("extract_scalar", [_slot("law_number", "Official law number on the governing instrument.")]),
        "claimant_count": ("extract_scalar", [_slot("claimant_count", "Number of distinct claimants in the case.")]),
        "same_law_number": ("compare_entities", [_slot("same_law_number", "Whether the referenced instruments point to the same law number.")]),
        "made_by": ("extract_scalar", [_slot("made_by", "Authority that made the governing law.")]),
        "administered_by": ("extract_scalar", [_slot("administered_by", "Authority that administers the governing law.")]),
        "publication_text": ("extract_scalar", [_slot("publication_text", "Publication or consolidated-version statement of the law.")]),
        "enacted_text": ("extract_scalar", [_slot("enacted_text", "Enactment statement of the law.")]),
        "claimant_names": ("extract_set", [_slot("claimant_names", "Claimant names in the case.", cardinality="many")]),
        "amended_laws": ("extract_set", [_slot("amended_laws", "List of amended laws.", cardinality="many")]),
        "common_parties": ("compare_entities", [_slot("common_parties", "Whether both cases share a common main party.")]),
        "common_judges": ("compare_entities", [_slot("common_judges", "Whether both cases share a common judge.")]),
        "earlier_issue_date_case": ("compare_entities", [_slot("earlier_issue_date_case", "Which case has the earlier issue date.")]),
        "higher_claim_amount_case": ("compare_entities", [_slot("higher_claim_amount_case", "Which case has the higher claim amount.")]),
        "effective_dates": (
            "locate_clause",
            [
                _slot("pre_existing_account_date", "Effective date for pre-existing accounts."),
                _slot("new_account_date", "Effective date for new accounts."),
            ],
        ),
        "order_result": ("locate_clause", [_slot("operative_result", "Operative order or final result of the case.")]),
        "article_content": ("locate_clause", [_slot("article_clause", "Clause or article content responsive to the question.")]),
        "clause_summary": ("locate_clause", [_slot("clause_summary", "Clause-localized answer for the requested provision.")]),
        "consultation_issuer": ("extract_scalar", [_slot("consultation_issuer", "Authority that issued the consultation paper.")]),
        "comparison_answer": ("summarize_evidence", [_slot("comparison_summary", "Comparison answer across multiple sources.")]),
        "enumeration_answer": ("extract_set", [_slot("enumerated_items", "Enumerated items or entities.", cardinality="many")]),
        "absence_check": ("detect_absence", [_slot("absence_check", "Whether the requested information is absent from the available evidence.")]),
        "generic_answer": ("summarize_evidence", [_slot("generic_answer", "Grounded answer synthesized from supporting evidence.")]),
        "defendant_name": ("extract_scalar", [_slot("defendant_name", "Name of the defendant or respondent.")]),
    }
    operation, slots = field_contracts.get(target_field, ("summarize_evidence", [_slot("generic_answer", "Grounded answer synthesized from supporting evidence.")]))

    if (
        operation == "summarize_evidence"
        and answer_kind == "free_text"
        and analysis.needs_multi_document_support
        and (" compare " in f" {question_norm} " or " comparison" in analysis.intent_tags)
    ):
        slots = [_slot("comparison_summary", "Comparison answer across multiple sources.")]

    return TaskContract(
        operation=operation,
        answer_kind=answer_kind,
        slots=slots,
        anchors=anchors,
        needs_multi_document_support=analysis.needs_multi_document_support,
        page_focus=_page_focus_from_support_focus(analysis.support_focus),
        should_abstain_if_missing=("absence_sensitive" in analysis.intent_tags or target_field == "absence_check"),
        confidence=analysis.confidence,
    )


class QuestionAnalyzer:
    def __init__(
        self,
        provider: str,
        model: str,
        cache_path: Path,
        corpus: Any | None = None,
    ):
        self.provider = provider
        self.model = model
        self.api = APIProcessor(provider=provider)
        self.cache_path = cache_path
        self.corpus = corpus
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

    def _cache_key(self, question: str, answer_type: str) -> str:
        raw = f"{ANALYSIS_PROMPT_VERSION}\n{self.provider}\n{self.model}\n{answer_type}\n{question}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def analyze(self, question: str, answer_type: str) -> QuestionAnalysis:
        key = self._cache_key(question, answer_type)
        fallback = self._fallback(question)
        fast = self._fast_path(question, answer_type, fallback)
        if fast is not None:
            analysis = self._postprocess(question, answer_type, fast, fallback)
            self.cache[key] = analysis.model_dump()
            self._save_cache()
            return analysis
        cached = self.cache.get(key)
        if cached:
            try:
                return self._postprocess(question, answer_type, QuestionAnalysis.model_validate(cached), fallback)
            except Exception:
                pass

        try:
            response = self.api.send_message(
                model=self.model,
                temperature=0.0,
                system_content=self._system_prompt(),
                human_content=self._user_prompt(question=question, answer_type=answer_type),
                is_structured=True,
                response_format=QuestionAnalysis,
                max_tokens=400,
                request_timeout=60,
            )
            analysis = QuestionAnalysis.model_validate(response)
            analysis = self._postprocess(question, answer_type, analysis, fallback)
        except Exception:
            analysis = fallback

        self.cache[key] = analysis.model_dump()
        self._save_cache()
        return analysis

    def _corpus_title_hits(self, question: str) -> List[str]:
        if self.corpus is None:
            return []
        question_norm = normalize_space(question).lower()
        question_law_ids = extract_law_ids(question)
        question_mentions = _question_instrument_mentions(question)
        exact_id_hits: list[str] = []
        if question_law_ids:
            for document in self.corpus.documents.values():
                if document.kind not in {"law", "regulation"}:
                    continue
                if set(document.canonical_ids) & set(question_law_ids):
                    exact_id_hits.append(normalize_space(document.title))
            if exact_id_hits:
                return list(dict.fromkeys(exact_id_hits))[:4]

        mention_hits: list[tuple[float, int, str]] = []
        if question_mentions:
            for document in self.corpus.documents.values():
                if document.kind not in {"law", "regulation"}:
                    continue
                candidates = [document.title, *document.aliases]
                best_score = 0.0
                for candidate in candidates:
                    candidate_norm = _normalized_title_surface_exact(candidate)
                    if not candidate_norm:
                        continue
                    for mention in question_mentions:
                        mention_norm = _normalized_title_surface_exact(mention)
                        if not _title_matches_instrument_mention(candidate, mention):
                            continue
                        local_score = 0.0
                        if mention_norm == candidate_norm:
                            local_score = 220.0
                        elif mention_norm == _normalized_title_surface_exact(document.title):
                            local_score = 210.0
                        elif mention_norm in candidate_norm:
                            local_score = 180.0 - max(0, len(candidate_norm) - len(mention_norm)) * 0.05
                        else:
                            mention_tokens = {
                                token
                                for token in re.findall(r"[a-z0-9]+", mention_norm)
                                if len(token) > 3 and token not in TITLE_ANCHOR_STOPWORDS and not token.isdigit()
                            }
                            candidate_tokens = {
                                token
                                for token in re.findall(r"[a-z0-9]+", candidate_norm)
                                if len(token) > 3 and token not in TITLE_ANCHOR_STOPWORDS and not token.isdigit()
                            }
                            overlap = len(mention_tokens & candidate_tokens)
                            if overlap:
                                local_score = overlap * 25.0 - max(0, len(candidate_tokens) - len(mention_tokens)) * 0.5
                        best_score = max(best_score, local_score)
                if best_score > 0:
                    mention_hits.append((best_score, len(normalize_space(document.title)), normalize_space(document.title)))
            if mention_hits:
                mention_hits.sort(key=lambda item: (-item[0], item[1], item[2]))
                seen = set()
                result = []
                for _, _, title in mention_hits:
                    if title.lower() in seen:
                        continue
                    seen.add(title.lower())
                    result.append(title)
                return result[:4]

        rows: list[tuple[float, int, str]] = []
        for document in self.corpus.documents.values():
            if document.kind not in {"law", "regulation"}:
                continue
            candidates = [document.title, *document.aliases, *document.canonical_ids]
            best = 0.0
            best_title = normalize_space(document.title)
            for candidate in candidates:
                candidate_norm = normalize_space(str(candidate or "")).lower().replace("#", " ")
                candidate_norm = normalize_space(re.sub(r"\b20\d{2}\b", "", candidate_norm))
                if len(candidate_norm) < 6:
                    continue
                if re.search(rf"\b{re.escape(candidate_norm)}\b", question_norm):
                    local_score = 120.0 + len(candidate_norm)
                    if candidate_norm == _normalized_title_surface_exact(document.title):
                        local_score += 20.0
                    best = max(best, local_score)
                    continue
                candidate_tokens = {
                    token
                    for token in re.findall(r"[a-z0-9]+", candidate_norm)
                    if len(token) > 3 and token not in {"law", "laws", "difc", "regulation", "regulations"}
                }
                if len(candidate_tokens) >= 2:
                    overlap = sum(1 for token in candidate_tokens if token in question_norm)
                    if overlap >= min(2, len(candidate_tokens)):
                        best = max(best, overlap * 20.0)
            if best > 0 and best_title:
                rows.append((best, len(best_title), best_title))
        rows.sort(key=lambda item: (-item[0], item[1], item[2]))
        seen = set()
        result = []
        for _, _, title in rows:
            title_norm = title.lower()
            if title_norm in seen:
                continue
            seen.add(title_norm)
            result.append(title)
        return result[:4]

    def _fast_path(self, question: str, answer_type: str, fallback: QuestionAnalysis) -> QuestionAnalysis | None:
        question = normalize_space(question)
        question_norm = question.lower()
        case_ids = fallback.target_case_ids
        law_ids = fallback.target_law_ids
        article_refs = fallback.target_article_refs
        quoted_titles = _quoted_question_titles(question)
        deictic_titles = _deictic_question_instrument_titles(question)
        bare_titles = _bare_question_instrument_titles(question)
        corpus_title_hits = filter_target_titles_for_question(question, self._corpus_title_hits(question))
        consultation_like = "consultation paper" in question_norm or any(
            marker in question_norm for marker in ("public responses", "response deadline", "submitting comments", "provide comments")
        )
        quoted_document_focus = bool(quoted_titles) and any(
            marker in question_norm
            for marker in (
                "issuing body",
                "which authority issued",
                "who issued the difc document",
                "email address",
                "what area of law",
                "area of law is covered",
                "response deadline",
                "consultation deadline",
                "deadline for providing comments",
                "deadline for submitting comments",
            )
        )
        metadata_like_question = any(
            marker in question_norm
            for marker in (
                "effective date",
                "come into force",
                "came into force",
                "in force on",
                "enactment date",
                "enacted on",
                "date of enactment",
                "who administers",
                "administered by",
                "who made this law",
                "who made the law",
            )
        )
        specific_question_titles = deictic_titles or bare_titles
        if metadata_like_question and specific_question_titles:
            target_titles = list(specific_question_titles)
        else:
            target_titles = list(quoted_titles) if ((consultation_like or quoted_document_focus) and quoted_titles) else corpus_title_hits
        if not target_titles:
            target_titles = quoted_titles
        if not target_titles and specific_question_titles:
            target_titles = specific_question_titles
        if not target_titles and fallback.target_titles:
            target_titles = list(fallback.target_titles)
        needs_multi = fallback.needs_multi_document_support or len(target_titles) >= 2

        # Keep the fast path intentionally narrow and layout-stable.
        # If a question can be handled by the general analyzer + contract layer,
        # prefer that over adding another benchmark-shaped shortcut here.

        def analysis(
            *,
            task_family: str,
            target_field: str,
            support_focus: list[str] | None = None,
            intent_tags: list[str] | None = None,
            use_structured_executor: bool = False,
            confidence: float = 0.92,
            retrieval_query: str | None = None,
            must_support_terms: list[str] | None = None,
            force_multi: bool | None = None,
        ) -> QuestionAnalysis:
            return QuestionAnalysis(
                standalone_question=question,
                retrieval_query=retrieval_query or question,
                task_family=task_family,
                target_field=target_field,
                support_focus=support_focus or [],
                target_case_ids=list(case_ids),
                target_law_ids=list(law_ids),
                target_article_refs=list(article_refs),
                target_titles=list(target_titles),
                must_support_terms=must_support_terms or [],
                intent_tags=intent_tags or [],
                needs_multi_document_support=needs_multi if force_multi is None else force_multi,
                use_structured_executor=use_structured_executor,
                confidence=confidence,
            )

        clause_like_request = any(
            marker in question_norm
            for marker in (
                "what must",
                "what may",
                "what is required",
                "what are required",
                "what is the purpose",
                "what are the retention",
                "what is the retention",
                "minimum period",
                "maximum fine",
                "provide upon request",
                "language other than english",
                "translation",
                "preserve",
                "retained",
                "records",
                "liable",
                "liability",
                "appoint",
                "dismiss",
                "permitted to",
            )
        )

        if answer_type in {"date", "name"} and case_ids and (
            "date of issue" in question_norm
            or "issued first" in question_norm
            or "earlier issue date" in question_norm
            or "earlier date of issue" in question_norm
            or "which was issued first" in question_norm
        ):
            if len(case_ids) == 2 or "earlier" in question_norm:
                return analysis(task_family="comparison", target_field="earlier_issue_date_case", support_focus=["issue_date_line"], intent_tags=["single_fact", "comparison"], use_structured_executor=True)
            if answer_type == "date":
                return analysis(task_family="case_metadata", target_field="issue_date", support_focus=["issue_date_line"], intent_tags=["single_fact"], use_structured_executor=True)

        if answer_type == "boolean" and len(case_ids) == 2 and question_is_party_overlap(question_norm):
            return analysis(
                task_family="comparison",
                target_field="common_parties",
                support_focus=["party_block", "first_page"],
                intent_tags=["comparison", "single_fact"],
                use_structured_executor=True,
            )

        if answer_type == "boolean" and len(case_ids) == 2 and question_is_judge_overlap(question_norm):
            return analysis(
                task_family="comparison",
                target_field="common_judges",
                support_focus=["judge_block", "first_page"],
                intent_tags=["comparison", "single_fact"],
                use_structured_executor=True,
            )

        if answer_type == "names" and case_ids and any(marker in question_norm for marker in ("claimant", "claimants", "listed as claimant")):
            return analysis(task_family="case_metadata", target_field="claimant_names", support_focus=["party_block", "first_page"], intent_tags=["enumeration", "single_fact"], use_structured_executor=True)

        if answer_type == "names" and case_ids and question_targets_claimant_side(question_norm):
            return analysis(task_family="case_metadata", target_field="claimant_names", support_focus=["party_block", "first_page"], intent_tags=["enumeration", "single_fact"], use_structured_executor=True)

        if answer_type == "number" and case_ids and any(
            marker in question_norm for marker in ("number of distinct claimants", "how many claimants", "total number of distinct claimants")
        ):
            return analysis(task_family="case_metadata", target_field="claimant_count", support_focus=["party_block", "first_page"], intent_tags=["single_fact"], use_structured_executor=True)

        if answer_type in {"name", "names"} and case_ids and question_targets_defendant_side(question_norm):
            return analysis(task_family="case_metadata", target_field="defendant_name", support_focus=["party_block", "first_page"], intent_tags=["single_fact"], use_structured_executor=True)

        if answer_type == "number" and case_ids and any(marker in question_norm for marker in ("claim value", "claim amount", "value of the claim")):
            return analysis(task_family="case_metadata", target_field="claim_amount", support_focus=["first_page"], intent_tags=["single_fact"], use_structured_executor=True)

        if answer_type == "boolean" and len(case_ids) == 2 and any(marker in question_norm for marker in ("enforce the arbitration", "related to the arbitration award", "arbitral award", "arbitration decision made in")):
            return analysis(task_family="comparison", target_field="case_relation", support_focus=["first_page"], intent_tags=["comparison", "single_fact"], use_structured_executor=True)

        consultation_deadline_markers = (
            "by when should public responses be submitted",
            "response deadline",
            "consultation deadline",
            "deadline for providing comments",
            "deadline for submitting comments",
            "deadline for submitting public responses",
            "what was the deadline for providing comments",
            "what is the response deadline",
            "what is the consultation deadline",
        )

        if answer_type == "date" and (
            question_is_consultation_paper(question_norm, target_titles)
            or bool(law_ids)
            or bool(target_titles)
        ) and any(
            marker in question_norm
            for marker in consultation_deadline_markers
        ):
            return analysis(task_family="consultation_metadata", target_field="consultation_deadline", support_focus=["title_page"], intent_tags=["single_fact"], use_structured_executor=True)

        if answer_type in {"name", "free_text"} and question_is_consultation_paper(question_norm, target_titles) and "email address" in question_norm:
            return analysis(task_family="consultation_metadata", target_field="consultation_email", support_focus=["title_page"], intent_tags=["single_fact"], use_structured_executor=True)

        if answer_type == "name" and (question_is_consultation_paper(question_norm, target_titles) or bool(target_titles)) and any(
            marker in question_norm
            for marker in ("issuing body", "which authority issued", "who issued the difc document")
        ):
            return analysis(task_family="consultation_metadata", target_field="consultation_issuer", support_focus=["title_page"], intent_tags=["single_fact"], use_structured_executor=True)

        if answer_type == "name" and question_is_consultation_paper(question_norm, target_titles) and any(
            marker in question_norm
            for marker in ("what area of law", "area of law is covered", "what area of law is covered")
        ):
            return analysis(task_family="consultation_metadata", target_field="consultation_topic", support_focus=["title_page"], intent_tags=["single_fact"], use_structured_executor=True)

        if answer_type == "boolean" and "same law number" in question_norm and len(target_titles) >= 2:
            return analysis(task_family="comparison", target_field="same_law_number", support_focus=["title_page", "first_page"], intent_tags=["comparison", "single_fact"], use_structured_executor=True, force_multi=True)

        if answer_type == "name" and any(marker in question_norm for marker in ("who made this law", "who made the law")) and (law_ids or target_titles):
            return analysis(task_family="law_metadata", target_field="made_by", support_focus=["title_page"], intent_tags=["single_fact"], use_structured_executor=True)

        if answer_type == "free_text" and any(marker in question_norm for marker in ("who made this law", "who made the law")) and (law_ids or target_titles):
            return analysis(task_family="law_metadata", target_field="made_by", support_focus=["title_page"], intent_tags=["single_fact"], use_structured_executor=True)

        if (
            not article_refs
            and any(marker in question_norm for marker in ("who administers", "responsible for administering", "administered by"))
            and (law_ids or target_titles)
        ):
            return analysis(task_family="law_metadata", target_field="administered_by", support_focus=["title_page", "administration_clause"], intent_tags=["single_fact"], use_structured_executor=(answer_type == "free_text"))

        if answer_type == "free_text" and "consolidated version" in question_norm and "publish" in question_norm and (law_ids or target_titles):
            return analysis(task_family="law_metadata", target_field="publication_text", support_focus=["title_page", "publication_line"], intent_tags=["single_fact"], use_structured_executor=True)

        if answer_type == "date" and any(
            marker in question_norm
            for marker in (
                "effective date",
                "come into force",
                "came into force",
                "in force on",
            )
        ) and (law_ids or target_titles):
            return analysis(task_family="law_metadata", target_field="effective_dates", support_focus=["title_page", "commencement_clause"], intent_tags=["single_fact"], use_structured_executor=True)

        if answer_type == "free_text" and "effective date" in question_norm and ("pre-existing" in question_norm or "new account" in question_norm):
            return analysis(task_family="law_metadata", target_field="effective_dates", support_focus=["commencement_clause"], intent_tags=["comparison", "clause_localized"], use_structured_executor=True)

        if answer_type == "date" and any(
            marker in question_norm
            for marker in (
                "enactment date",
                "enacted on",
                "date of enactment",
            )
        ) and (law_ids or target_titles):
            return analysis(task_family="law_metadata", target_field="enacted_text", support_focus=["title_page", "enactment_clause"], intent_tags=["single_fact"], use_structured_executor=True)

        if answer_type == "free_text" and any(marker in question_norm for marker in ("enacted", "enactment notice")) and (law_ids or target_titles):
            return analysis(task_family="law_metadata", target_field="enacted_text", support_focus=["title_page", "enactment_clause"], intent_tags=["single_fact"], use_structured_executor=True)

        if answer_type == "free_text" and "amended by" in question_norm and law_ids:
            return analysis(task_family="amendment_lookup", target_field="amended_laws", support_focus=["title_page"], intent_tags=["enumeration"], use_structured_executor=True)

        if answer_type == "name" and len(case_ids) == 2 and any(
            marker in question_norm
            for marker in (
                "larger sum claimed",
                "larger claim amount",
                "higher claim amount",
                "larger sum sought",
            )
        ):
            return analysis(task_family="comparison", target_field="higher_claim_amount_case", support_focus=["first_page"], intent_tags=["comparison", "single_fact"], use_structured_executor=True)

        if answer_type == "name" and case_ids and any(
            marker in question_norm
            for marker in (
                "original case number",
                "award reference",
                "being enforced",
                "award being enforced",
            )
        ):
            return analysis(task_family="case_metadata", target_field="claim_number", support_focus=["first_page"], intent_tags=["single_fact"], use_structured_executor=True)

        if answer_type == "free_text" and case_ids and any(
            marker in question_norm
            for marker in ("miranda", "jury", "plea bargain", "parole")
        ):
            return analysis(
                task_family="absence_probe",
                target_field="absence_check",
                support_focus=[],
                intent_tags=["absence_sensitive"],
                use_structured_executor=True,
            )

        if answer_type == "free_text" and case_ids and any(
            marker in question_norm
            for marker in (
                "it is hereby ordered that",
                "final ruling",
                "what was the result",
                "what was the outcome",
                "outcome of the",
                "result of the application",
                "according to the conclusion section",
                "how did the court",
                "what did the court decide",
                "last page",
                "first page",
            )
        ):
            focus: list[str] = ["order_section"]
            if "conclusion section" in question_norm:
                focus = ["conclusion_section", "order_section"]
            elif "last page" in question_norm:
                focus = ["last_page", "order_section"]
            elif "first page" in question_norm:
                focus = ["first_page", "order_section"]
            return analysis(
                task_family="case_order",
                target_field="order_result",
                support_focus=focus,
                intent_tags=["section_specific", "operative_result"],
                use_structured_executor=False,
                must_support_terms=[term for term in ["permission to appeal", "appeal", "costs", "trial", "set aside"] if term in question_norm][:4],
            )

        if article_refs:
            focus = ["article_section"]
            if "second page" in question_norm or "page 2" in question_norm:
                focus.append("second_page")
            return analysis(
                task_family="article_lookup",
                target_field="article_content",
                support_focus=focus,
                intent_tags=["clause_localized"],
                use_structured_executor=False,
                confidence=0.95,
                retrieval_query=question,
                must_support_terms=list(article_refs[:2]) + [term for term in ["records", "retention", "translation", "purpose", "liability", "remuneration", "registrar"] if term in question_norm][:2],
                force_multi=(needs_multi or len(article_refs) >= 2),
            )

        if (
            answer_type == "free_text"
            and clause_like_request
            and (target_titles or law_ids)
            and not case_ids
            and not needs_multi
        ):
            must_support_terms = [
                term
                for term in (
                    "translation",
                    "relevant authority",
                    "records",
                    "retention",
                    "minimum period",
                    "maximum fine",
                    "fine",
                    "fee",
                    "penalty",
                    "assignment of rights",
                    "registered agent",
                    "registrar",
                    "appoint",
                    "dismiss",
                    "liability",
                    "purpose",
                )
                if term in question_norm
            ][:4]
            return analysis(
                task_family="clause_lookup",
                target_field="clause_summary",
                support_focus=["article_section"],
                intent_tags=["clause_localized", "single_source"],
                use_structured_executor=False,
                confidence=0.88,
                must_support_terms=must_support_terms,
                force_multi=False,
            )

        if (
            answer_type == "number"
            and any(marker in question_norm for marker in ("fine", "fee", "penalty", "usd", "aed"))
            and (target_titles or law_ids)
            and not case_ids
            and not needs_multi
        ):
            must_support_terms = [
                term
                for term in (
                    "fine",
                    "fee",
                    "penalty",
                    "certificate of compliance",
                    "exemption",
                    "family business register",
                    "founding member",
                    "registered agent",
                    "annual accounts",
                    "entertainment events",
                )
                if term in question_norm
            ][:4]
            return analysis(
                task_family="clause_lookup",
                target_field="clause_summary",
                support_focus=["article_section"],
                intent_tags=["clause_localized", "single_source"],
                use_structured_executor=True,
                confidence=0.9,
                must_support_terms=must_support_terms,
                force_multi=False,
            )

        if (
            answer_type == "name"
            and any(marker in question_norm for marker in ("defined term", "what is the term for", "what term", "referred to"))
            and (target_titles or law_ids)
            and not case_ids
            and not needs_multi
        ):
            must_support_terms = [
                term
                for term in (
                    "defined term",
                    "designation",
                    "notices",
                    "regulations refer to",
                    "law that these regulations refer to",
                )
                if term in question_norm
            ][:4]
            return analysis(
                task_family="clause_lookup",
                target_field="clause_summary",
                support_focus=["article_section"],
                intent_tags=["clause_localized", "single_source"],
                use_structured_executor=True,
                confidence=0.9,
                must_support_terms=must_support_terms,
                force_multi=False,
            )

        if (
            answer_type == "name"
            and ("defined term for the law" in question_norm or "these regulations refer to" in question_norm)
            and not case_ids
            and not needs_multi
        ):
            must_support_terms = [
                term
                for term in (
                    "defined term",
                    "law",
                    "regulations",
                    "refer to",
                )
                if term in question_norm
            ][:4]
            return analysis(
                task_family="clause_lookup",
                target_field="clause_summary",
                support_focus=["article_section"],
                intent_tags=["clause_localized", "single_source"],
                use_structured_executor=True,
                confidence=0.84,
                must_support_terms=must_support_terms,
                force_multi=False,
            )

        if answer_type == "free_text" and needs_multi and (case_ids or law_ids or len(target_titles) >= 2):
            focus: list[str] = []
            if "article" in question_norm or "section" in question_norm:
                focus.append("article_section")
            if "last page" in question_norm or "conclusion section" in question_norm:
                focus.append("order_section")
            return analysis(
                task_family="comparison_rag",
                target_field="comparison_answer",
                support_focus=focus,
                intent_tags=["comparison", "multi_source"],
                use_structured_executor=False,
                confidence=0.9,
                must_support_terms=[term for term in ["retention", "costs", "appeal", "judge", "party", "records", "translation", "liability"] if term in question_norm][:4],
                force_multi=True,
            )

        if answer_type == "free_text" and question_is_clause_reasoning(question_norm) and (target_titles or law_ids or article_refs):
            return analysis(
                task_family="clause_lookup",
                target_field="clause_summary",
                support_focus=["article_section"] if (article_refs or target_titles or law_ids) else [],
                intent_tags=["clause_localized"],
                use_structured_executor=False,
                confidence=0.84,
                must_support_terms=[term for term in ["conditions", "consequences", "obligations", "purpose", "liability", "records"] if term in question_norm][:4],
                force_multi=needs_multi,
            )

        if answer_type == "free_text" and question_is_enumeration_like(question_norm, answer_type) and (target_titles or law_ids or case_ids):
            return analysis(
                task_family="enumeration_lookup",
                target_field="enumeration_answer",
                support_focus=["article_section"] if article_refs else [],
                intent_tags=["enumeration"] + (["multi_source"] if needs_multi else []),
                use_structured_executor=False,
                confidence=0.82,
                must_support_terms=[term for term in ["bodies", "jurisdictions", "exceptions", "rules", "records"] if term in question_norm][:4],
                force_multi=needs_multi,
            )

        if answer_type == "free_text" and question_is_absence_sensitive(question_norm):
            return analysis(
                task_family="absence_probe",
                target_field="absence_check",
                support_focus=["article_section"] if (article_refs or target_titles or law_ids) else [],
                intent_tags=["absence_sensitive"],
                use_structured_executor=False,
                confidence=0.8,
                must_support_terms=[term for term in ["force majeure", "penalty", "liability", "reasonable time", "definition"] if term in question_norm][:4],
                force_multi=needs_multi,
            )

        return None

    def _fallback(self, question: str) -> QuestionAnalysis:
        question = normalize_space(question)
        question_norm = question.lower()
        return QuestionAnalysis(
            standalone_question=question,
            retrieval_query=question,
            task_family="generic_rag",
            target_field="generic_answer",
            support_focus=[],
            target_case_ids=extract_case_ids(question),
            target_law_ids=extract_law_ids(question),
            target_article_refs=extract_article_refs(question),
            target_titles=[],
            must_support_terms=[],
            needs_multi_document_support=question_implies_multi_support(question_norm)
            or len(extract_case_ids(question)) + len(extract_law_ids(question)) >= 2,
            use_structured_executor=False,
            confidence=0.0,
        )

    def _postprocess(
        self,
        question: str,
        answer_type: str,
        analysis: QuestionAnalysis,
        fallback: QuestionAnalysis,
    ) -> QuestionAnalysis:
        data = analysis.model_dump()
        data["standalone_question"] = normalize_space(data.get("standalone_question") or question)
        data["retrieval_query"] = normalize_space(data.get("retrieval_query") or data["standalone_question"])
        data["task_family"] = normalize_space(data.get("task_family") or "generic_rag") or "generic_rag"
        data["target_field"] = normalize_space(data.get("target_field") or "generic_answer") or "generic_answer"
        data["support_focus"] = [
            normalize_space(str(item)).lower().replace(" ", "_")
            for item in data.get("support_focus", [])
            if normalize_space(str(item))
        ]
        data["support_focus"] = list(dict.fromkeys(data["support_focus"]))
        data["target_case_ids"] = sorted(
            dict.fromkeys(list(data.get("target_case_ids", [])) + fallback.target_case_ids)
        )
        data["target_law_ids"] = sorted(
            dict.fromkeys(list(data.get("target_law_ids", [])) + fallback.target_law_ids)
        )
        data["target_article_refs"] = sorted(
            dict.fromkeys(list(data.get("target_article_refs", [])) + fallback.target_article_refs)
        )
        data["target_titles"] = [
            normalize_space(str(item))
            for item in data.get("target_titles", [])
            if normalize_space(str(item))
        ]
        data["target_titles"] = list(dict.fromkeys(data["target_titles"]))
        data["target_titles"] = filter_target_titles_for_question(question, data["target_titles"])
        data["must_support_terms"] = [
            normalize_space(str(item))
            for item in data.get("must_support_terms", [])
            if normalize_space(str(item))
        ]
        data["must_support_terms"] = list(dict.fromkeys(data["must_support_terms"]))[:4]
        data["intent_tags"] = [
            normalize_space(str(item)).lower().replace(" ", "_")
            for item in data.get("intent_tags", [])
            if normalize_space(str(item))
        ]
        data["intent_tags"] = list(dict.fromkeys(data["intent_tags"]))
        data["confidence"] = float(max(0.0, min(1.0, data.get("confidence", 0.0) or 0.0)))
        if data["needs_multi_document_support"] is False and fallback.needs_multi_document_support:
            data["needs_multi_document_support"] = True

        question_norm = normalize_space(question).lower()
        explicit_multi = question_implies_multi_support(question_norm)
        entity_count = len(data["target_case_ids"]) + len(data["target_law_ids"]) + len(data["target_titles"])
        if (
            data["needs_multi_document_support"]
            and not explicit_multi
            and entity_count <= 1
            and len(data["target_article_refs"]) <= 1
        ):
            data["needs_multi_document_support"] = False
        if not explicit_multi and len(data["target_titles"]) <= 1 and len(data["target_case_ids"]) + len(data["target_law_ids"]) <= 1:
            data["needs_multi_document_support"] = False

        if explicit_multi and "comparison" not in data["intent_tags"]:
            data["intent_tags"].append("comparison")
        if question_is_absence_sensitive(question_norm) and "absence_sensitive" not in data["intent_tags"]:
            data["intent_tags"].append("absence_sensitive")
        if question_is_enumeration_like(question_norm, answer_type) and "enumeration" not in data["intent_tags"]:
            data["intent_tags"].append("enumeration")
        if data["target_field"] in {"article_content", "clause_summary"} and "clause_localized" not in data["intent_tags"]:
            data["intent_tags"].append("clause_localized")
        if any(item in data["support_focus"] for item in {"title_page", "first_page", "last_page", "conclusion_section", "order_section"}):
            if "section_specific" not in data["intent_tags"]:
                data["intent_tags"].append("section_specific")

        if answer_type == "free_text" and "effective date" in question_norm and (
            "pre-existing" in question_norm or "new account" in question_norm
        ):
            data["task_family"] = "law_metadata"
            data["target_field"] = "effective_dates"
            data["support_focus"] = list(dict.fromkeys(data["support_focus"] + ["commencement_clause"]))
            if data["target_titles"]:
                data["use_structured_executor"] = True

        if answer_type == "free_text" and "consolidated version" in question_norm and "publish" in question_norm:
            data["task_family"] = "law_metadata"
            data["target_field"] = "publication_text"
            data["support_focus"] = list(dict.fromkeys(data["support_focus"] + ["title_page", "publication_line"]))
            if data["target_titles"] and data["confidence"] >= 0.75:
                data["use_structured_executor"] = True

        if (
            answer_type == "free_text"
            and data["target_case_ids"]
            and any(marker in question_norm for marker in ("final ruling", "what was the result", "what was the outcome", "court's final ruling", "how did the court"))
        ):
            data["task_family"] = "case_order"
            data["target_field"] = "order_result"
            additional_focus = ["order_section"]
            if not any(marker in question_norm for marker in ("last page", "conclusion section", "title page")):
                additional_focus.append("first_page")
            data["support_focus"] = list(dict.fromkeys(data["support_focus"] + additional_focus))
            data["use_structured_executor"] = False

        if answer_type == "free_text" and explicit_multi and data["task_family"] == "generic_rag":
            data["task_family"] = "comparison_rag"
            data["target_field"] = "comparison_answer"

        if answer_type == "free_text" and data["task_family"] == "generic_rag" and "clause_localized" in data["intent_tags"]:
            data["task_family"] = "clause_lookup"
            if data["target_field"] == "generic_answer":
                data["target_field"] = "clause_summary"

        if answer_type == "free_text" and data["task_family"] == "generic_rag" and "enumeration" in data["intent_tags"]:
            data["task_family"] = "enumeration_lookup"

        if (
            data["target_field"] in {"made_by", "administered_by", "publication_text", "enacted_text"}
            and any(marker in question_norm for marker in ("appoint", "dismiss", "responsible for", "purpose of", "fine for", "penalty for", "permitted to"))
            and "administer" not in question_norm
        ):
            data["task_family"] = "article_lookup"
            data["target_field"] = "article_content"
            data["support_focus"] = list(dict.fromkeys(data["support_focus"] + ["article_section"]))
            data["use_structured_executor"] = False

        if data["target_field"] in {"made_by", "administered_by"} and data["target_article_refs"]:
            data["task_family"] = "article_lookup"
            data["target_field"] = "article_content"
            data["support_focus"] = list(dict.fromkeys(data["support_focus"] + ["article_section"]))
            data["use_structured_executor"] = False

        if answer_type not in {"boolean", "number", "name", "names", "date", "free_text"}:
            data["use_structured_executor"] = False
        if data["target_field"] not in STRUCTURED_FIELDS:
            data["use_structured_executor"] = False
        if data["confidence"] < 0.6:
            data["use_structured_executor"] = False
        if (
            data["target_field"] in STRUCTURED_FIELDS
            and data["confidence"] >= 0.75
            and data["task_family"] in {"case_metadata", "law_metadata", "comparison", "amendment_lookup", "article_lookup"}
        ):
            data["use_structured_executor"] = True
        if (
            data["target_field"] in {"law_number", "made_by", "administered_by", "publication_text", "enacted_text"}
            and data["confidence"] >= 0.8
            and data["target_titles"]
            and not data["needs_multi_document_support"]
        ):
            data["use_structured_executor"] = True
        if data["target_field"] == "effective_dates" and data["target_titles"] and data["confidence"] >= 0.75:
            data["use_structured_executor"] = True
        if (
            answer_type in {"boolean", "number", "name"}
            and data["target_field"] in {"article_content", "clause_summary"}
            and data["confidence"] >= 0.75
            and (data["target_titles"] or data["target_law_ids"])
            and not data["needs_multi_document_support"]
        ):
            data["use_structured_executor"] = True

        return QuestionAnalysis.model_validate(data)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are the query-analysis layer for a production DIFC legal RAG system. "
            "Do not answer the question. Rewrite it into a clean standalone version, "
            "produce a retrieval-friendly query, classify the task, and extract entities. "
            "Only set use_structured_executor=true for narrow factual lookups that can be "
            "answered directly from a cover page, issue-date line, party block, judge line, "
            "or simple two-entity comparison. For open-ended legal interpretation, article "
            "substance, or broad reasoning questions, keep use_structured_executor=false.\n\n"
            "Prefer clause_lookup for narrow clause-focused questions that ask what a provision says, requires, permits, prohibits, or explains.\n"
            "Prefer comparison_rag when a free-text answer must synthesize information from multiple laws or cases.\n"
            "Prefer enumeration_lookup when the answer is a list of names, entities, or jurisdictions rather than a single fact.\n"
            "Prefer absence_probe when the user is explicitly asking whether the corpus contains, mentions, specifies, or provides something.\n"
            "When the question explicitly asks what a provision says, requires a retention period, "
            "asks for the operative outcome of an application, or names a quoted section such as "
            "'IT IS HEREBY ORDERED THAT', prefer article_lookup/article_content or case_order/order_result "
            "rather than generic_answer.\n"
            "When the question asks for the first page, title page, last page, or conclusion section, "
            "reflect that in support_focus.\n"
            "Do not classify a question as generic_answer if it is really asking for a narrow operative clause.\n\n"
            "Allowed task_family values: generic_rag, case_metadata, law_metadata, comparison, "
            "article_lookup, case_order, amendment_lookup, clause_lookup, comparison_rag, enumeration_lookup, absence_probe.\n"
            "Allowed target_field values: generic_answer, issue_date, earlier_issue_date_case, "
            "common_parties, common_judges, claimant_names, claimant_count, defendant_name, claim_amount, claim_number, higher_claim_amount_case, law_number, same_law_number, made_by, "
            "administered_by, publication_text, enacted_text, effective_dates, amended_laws, article_content, order_result, clause_summary, comparison_answer, enumeration_answer, absence_check.\n"
            "Allowed support_focus values: title_page, first_page, issue_date_line, party_block, "
            "judge_block, article_section, order_section, conclusion_section, administration_clause, "
            "second_page, "
            "enactment_clause, commencement_clause, publication_line, last_page.\n"
            "Also extract up to 4 distinctive must_support_terms: question concepts that must appear in the supporting text "
            "for a non-abstain answer to be credible. Include only distinctive concepts, not generic legal words.\n"
            "Also extract intent_tags from this set when applicable: clause_localized, comparison, multi_source, enumeration, absence_sensitive, section_specific, single_fact, single_source, operative_result.\n"
            "Be conservative. Prefer generic_rag over a brittle specialized classification."
        )

    @staticmethod
    def _user_prompt(question: str, answer_type: str) -> str:
        return (
            f"Question: {question}\n"
            f"Answer type: {answer_type}\n\n"
            "Return a JSON object only. Preserve the user's meaning exactly. "
            "If the question mentions a law title without a canonical law number, place that law "
            "title into target_titles. If the question refers to multiple entities or requires both "
            "sides of a comparison, set needs_multi_document_support=true. "
            "If the question starts with 'According to Article ...' or cites one or more articles, "
            "prefer target_field=article_content. "
            "If the question asks for the final ruling, outcome, result, or names the section "
            "'IT IS HEREBY ORDERED THAT', prefer target_field=order_result. "
            "If the answer is expected to be a list, prefer enumeration_lookup. "
            "If the question is asking whether the corpus contains or specifies something, prefer absence_probe."
        )
