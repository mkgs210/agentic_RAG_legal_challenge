from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from dateutil import parser as date_parser

from src.evidence_contracts import TaskContract
from src.pdf_text_fallback import merged_pdf_page_texts
from src.public_dataset_eval import (
    FREE_TEXT_ABSENCE_ANSWER,
    PublicCorpus,
    compress_free_text_answer,
    extract_article_refs,
    extract_case_ids,
    extract_law_ids,
    normalize_answer,
    normalize_number,
    normalize_space,
)
from src.query_analysis import QuestionAnalysis, build_task_contract


ROLE_TITLES = (
    "claimant",
    "defendant",
    "respondent",
    "appellant",
    "applicants",
    "claimant/respondent",
    "defendant/appellant",
)

JUDGE_PATTERNS = [
    re.compile(
        r"(?:H\.E\.\s+)?(?:Deputy Chief Justice|Chief Justice|Justice|SCT Judge)\s+"
        r"((?:Sir\s+)?(?:[A-Z][A-Za-z'\-]+|Al|al|Le|le|De|de|Van|van)"
        r"(?:\s+(?:[A-Z][A-Za-z'\-]+|Al|al|Le|le|De|de|Van|van|KC)){0,4})",
        re.I,
    ),
]

CLAIM_AMOUNT_PATTERNS = [
    re.compile(r"\bentered\s+judgment(?:[^.\n]{0,160})?\bamount of A[DE]D?\s*([0-9,]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"\bjudgment is entered(?:[^.\n]{0,120})?\bamount of A[DE]D?\s*([0-9,]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"\bshall pay(?:[^.\n]{0,120})?A[DE]D?\s*([0-9,]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"\bamount of AED\s+([0-9,]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"\bamount of A[DE]D?\s*([0-9,]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"\bseeking payment .*? AED\s+([0-9,]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"\bclaimed that an amount of A[DE]D?\s*([0-9,]+(?:\.[0-9]+)?)\s+was owed\b", re.I),
    re.compile(r"\bclaim(?:ed)? .*? AED\s+([0-9,]+(?:\.[0-9]+)?)", re.I),
    re.compile(r"\bclaim(?:ed|s)?\s+([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+\.[0-9]{2})\b", re.I),
    re.compile(r"\bJudgment Sum .*? AED\s+([0-9,]+(?:\.[0-9]+)?)", re.I),
]
CLAIM_AMOUNT_SCORED_PATTERNS = [
    (re.compile(r"\bentered\s+judgment(?:[^.\n]{0,160})?\bamount of A[DE]D?\s*([0-9,]+(?:\.[0-9]+)?)", re.I), 130),
    (re.compile(r"\bjudgment is entered(?:[^.\n]{0,120})?\bamount of A[DE]D?\s*([0-9,]+(?:\.[0-9]+)?)", re.I), 125),
    (re.compile(r"\bshall pay(?:[^.\n]{0,120})?A[DE]D?\s*([0-9,]+(?:\.[0-9]+)?)", re.I), 115),
    (re.compile(r"\bclaim form\b.*?\bclaimed\s+([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+\.[0-9]{2})\b", re.I | re.S), 120),
    (re.compile(r"\bclaimed that an amount of A[DE]D?\s*([0-9,]+(?:\.[0-9]+)?)\s+was owed\b", re.I), 110),
    (re.compile(r"\bclaim(?:ed|s)? .*? AED\s+([0-9,]+(?:\.[0-9]+)?)", re.I), 100),
    (re.compile(r"\bseeking payment .*? AED\s+([0-9,]+(?:\.[0-9]+)?)", re.I), 90),
    (re.compile(r"\bamount of AED\s+([0-9,]+(?:\.[0-9]+)?)", re.I), 80),
    (re.compile(r"\bamount of A[DE]D?\s*([0-9,]+(?:\.[0-9]+)?)", re.I), 80),
    (re.compile(r"\bclaim(?:ed|s)?\s+([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+\.[0-9]{2})\b", re.I), 70),
    (re.compile(r"\bJudgment Sum .*? AED\s+([0-9,]+(?:\.[0-9]+)?)", re.I), 20),
]

ORIGIN_CLAIM_NUMBER_PATTERNS = [
    re.compile(
        r"(?:in|from)\s+claim\s+no\.?\s*((?:CFI|ARB|TCD|CA|ENF|DEC|SCT)\s*[- ]?\d{3}(?:[-/]\d{4})(?:/\d+)?)",
        re.I,
    ),
    re.compile(
        r"(?:proceedings|application)\s*((?:CFI|ARB|TCD|CA|ENF|DEC|SCT)\s*[- ]?\d{3}(?:[-/]\d{4})(?:/\d+)?)",
        re.I,
    ),
    re.compile(
        r"(?:in|under)\s+claim\s+no\.?\s*([A-Z]{2,}(?:\s*\([A-Z]+\))?\s*\d+\s+of\s+\d{4})",
        re.I,
    ),
    re.compile(
        r"(?:award\s+reference|award|arbitration)\s*(?:no\.?|number)?\s*[:\-]?\s*([A-Z0-9][A-Za-z0-9() /.-]{4,80}\d{4})",
        re.I,
    ),
]

ORDER_MARKERS = (
    "it is hereby ordered that",
    "conclusion",
    "order with reasons",
    "result of the application",
)

ORDER_OUTCOME_TERMS = (
    "refused",
    "dismissed",
    "granted",
    "allowed",
    "discharged",
    "rejected",
    "denied",
    "struck out",
    "proceed to trial",
)

MONTH_NAME_RE = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"


def ordered_unique(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result

JUDGE_NOISE_TOKENS = {
    "and",
    "on",
    "sir",
    "the",
    "court",
    "courts",
    "order",
    "orders",
}


@dataclass
class CaseMetadata:
    sha: str
    case_id: str
    issue_date: Optional[str]
    first_page: int
    issue_page: Optional[int]
    claimant_names: List[str]
    defendant_names: List[str]
    all_main_parties: List[str]
    judges: List[str]
    claim_amount_aed: Optional[float]
    claim_amount_page: Optional[int]
    origin_claim_number: Optional[str]
    origin_claim_page: Optional[int]
    order_pages: List[int]


@dataclass
class LawMetadata:
    sha: str
    law_id: str
    title: str
    law_number: Optional[int]
    publication_text: Optional[str]
    publication_page: Optional[int]
    enacted_text: Optional[str]
    enacted_page: Optional[int]
    made_by: Optional[str]
    made_by_page: Optional[int]
    administered_by: Optional[str]
    administered_by_page: Optional[int]
    page_one: int
    commencement_page: Optional[int]
    consolidated_version_number: Optional[int]
    effective_date: Optional[str] = None
    effective_date_page: Optional[int] = None


@dataclass
class ConsultationMetadata:
    sha: str
    title: str
    deadline_date: Optional[str]
    deadline_page: Optional[int]
    submission_email: Optional[str]
    email_page: Optional[int]
    topic: Optional[str]
    topic_page: int
    issuing_body: Optional[str] = None
    issuing_body_page: Optional[int] = None
    related_law_ids: List[str] = field(default_factory=list)


@dataclass
class StructuredDecision:
    handled: bool = False
    answer_payload: Optional[Dict[str, Any]] = None
    chunks_override: Optional[List[Dict[str, Any]]] = None


class LazyMergedPageTexts(dict):
    def __init__(self, corpus: PublicCorpus):
        super().__init__()
        self._corpus = corpus

    def _load(self, sha: str) -> Dict[int, str]:
        if sha not in self:
            self[sha] = merged_pdf_page_texts(
                self._corpus.pdf_dir,
                sha,
                self._corpus.documents_payload[sha]["content"]["pages"],
            )
        return dict.__getitem__(self, sha)

    def get(self, key: str, default=None):  # type: ignore[override]
        if key in self._corpus.documents:
            return self._load(key)
        return default

    def __missing__(self, key: str):
        if key in self._corpus.documents:
            return self._load(key)
        raise KeyError(key)


def normalized_text(value: Any) -> str:
    return normalize_space(str(value or "")).lower()


ARTICLE_STOPWORDS = {
    "article",
    "law",
    "laws",
    "difc",
    "trust",
    "employment",
    "operating",
    "general",
    "partnership",
    "case",
    "question",
    "under",
    "according",
    "what",
    "which",
    "does",
    "is",
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
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


def significant_title_match_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value)
        if len(token) > 3 and not (token.isdigit() and len(token) == 4) and token not in TITLE_MATCH_STOPWORDS
    }


def title_instrument_kind(value: str) -> str:
    text = normalized_text(value)
    if "regulations" in text or "regulation" in text or "rules" in text:
        return "regulation"
    if "law" in text:
        return "law"
    return ""


def clean_clause_text(text: str) -> str:
    value = normalize_space(text)
    value = re.sub(r"^\-\s*", "", value)
    value = re.sub(r"^\d+\.\s*", "", value)
    value = re.sub(r"^\(([0-9]+|[a-z])\)\s*", "", value, flags=re.I)
    return normalize_space(value)


def sentence_level_order_candidates(text: str) -> List[str]:
    cleaned = clean_clause_text(text)
    if not cleaned:
        return []
    candidates = [cleaned]
    list_items = [
        normalize_space(match.group(1))
        for match in re.finditer(r"(?:^|\s)-\s*\d+\.\s*(.*?)(?=(?:\s+-\s*\d+\.\s)|$)", cleaned, re.S)
        if normalize_space(match.group(1))
    ]
    if not list_items:
        list_items = [
            normalize_space(match.group(1))
            for match in re.finditer(r"(?:^|\s)\d+\.\s*(.*?)(?=(?:\s+\d+\.\s)|$)", cleaned, re.S)
            if normalize_space(match.group(1))
        ]
    candidates.extend(item.rstrip(".") + "." for item in list_items)
    if len(cleaned) < 80:
        return candidates
    sentences = [
        normalize_space(part)
        for part in re.split(r"(?<=[\.;])\s+(?=[A-Z])", cleaned)
        if normalize_space(part)
    ]
    for sentence in sentences:
        sentence_norm = normalized_text(sentence)
        if any(term in sentence_norm for term in ORDER_OUTCOME_TERMS) or (
            "permission to appeal" in sentence_norm and "refused" in sentence_norm
        ):
            candidates.append(sentence.rstrip(".") + ".")
    deduped: List[str] = []
    seen = set()
    for candidate in candidates:
        key = normalized_text(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def extract_order_section_list_items(text: str) -> List[str]:
    if not text:
        return []
    match = re.search(
        r"IT IS HEREBY ORDERED THAT:(.*?)(?:##\s*SCHEDULE OF REASONS|##\s*Conclusion|Issued by:|$)",
        text,
        re.I | re.S,
    )
    block = ""
    if match:
        block = match.group(1)
    else:
        text_norm = normalized_text(text)
        has_continued_order_list = re.search(r"(?:^|\n)\s*-?\s*\d+\.\s+", text) is not None
        has_order_tail_signal = any(
            marker in text_norm
            for marker in (
                "costs award",
                "within 14 days",
                "interest shall accrue",
                "no order as to costs",
                "bear its own costs",
                "statement of costs",
                "proceed to trial",
                "set aside application",
                "permission to appeal",
                "application was dismissed",
                "application was refused",
            )
        )
        if not (has_continued_order_list and has_order_tail_signal):
            return []
        block = re.split(r"##\s*SCHEDULE OF REASONS|##\s*Conclusion|Issued by:|Date of issue:|At:\s*", text, maxsplit=1, flags=re.I)[0]
    items = [
        normalize_space(match.group(1))
        for match in re.finditer(r"(?:^|\n)\s*-\s*\d+\.\s*(.*?)(?=(?:\n\s*-\s*\d+\.\s)|$)", block, re.S)
        if normalize_space(match.group(1))
    ]
    if not items:
        items = [
            normalize_space(match.group(1))
            for match in re.finditer(r"(?:^|\n)\s*\d+\.\s*(.*?)(?=(?:\n\s*\d+\.\s)|$)", block, re.S)
            if normalize_space(match.group(1))
        ]
    deduped: List[str] = []
    seen = set()
    for item in items:
        key = normalized_text(item)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(item.rstrip(".") + ".")
    return deduped


def compact_clause_text(text: str) -> str:
    value = normalize_space(text)
    replacements = [
        (
            r"^All records required to be kept by Reporting Financial Institutions pursuant to the provisions of this Law and the Regulations shall be retained in an electronically readable format for a retention period of ",
            "records must be retained electronically for ",
        ),
        (
            r"^A General Partnership'?s Accounting Records shall be:\s*preserved by the General Partnership for ",
            "accounting records must be preserved for ",
        ),
        (
            r"^A Limited Liability Partnership'?s Accounting Records shall be:\s*preserved by the Limited Liability Partnership for ",
            "accounting records must be preserved for ",
        ),
        (
            r"^preserved by the General Partnership for ",
            "accounting records must be preserved for ",
        ),
        (
            r"^preserved by the Limited Liability Partnership for ",
            "accounting records must be preserved for ",
        ),
    ]
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value, flags=re.I)
    value = value.replace(" shall be ", " must be ")
    value = value.replace(" shall ", " must ")
    value = re.sub(r"after the date of reporting the information\b", "after reporting", value, flags=re.I)
    value = re.sub(r"from the date upon which they were created\b", "from creation", value, flags=re.I)
    value = re.sub(r",?\s*as follows:?\s*$", "", value, flags=re.I)
    value = re.sub(r"\s*[-–]\s*\d+\.?\s*$", "", value)
    value = re.sub(r"\s*\[\d+\]\s*$", "", value)
    value = normalize_space(value)
    value = re.sub(r",\s*(?=[.!?]$)", "", value)
    value = value.rstrip(" ;,")
    return value


def clause_marker_kind(marker: str) -> str:
    value = normalize_space(marker).lower()
    if re.fullmatch(r"\d+", value):
        return "digit"
    if re.fullmatch(r"(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)+", value):
        return "roman"
    if re.fullmatch(r"[a-z]", value):
        return "alpha"
    return "other"


def normalize_party_role_sentence(text: str) -> str:
    value = normalize_space(text)
    replacements = (
        (r"^the\s+applicant\b", "The Applicant"),
        (r"^the\s+claimant\b", "The Claimant"),
        (r"^the\s+defendant\b", "The Defendant"),
        (r"^the\s+respondent\b", "The Respondent"),
        (r"^the\s+appellant\b", "The Appellant"),
        (r"^applicant\b", "The Applicant"),
        (r"^claimant\b", "The Claimant"),
        (r"^defendant\b", "The Defendant"),
        (r"^respondent\b", "The Respondent"),
        (r"^appellant\b", "The Appellant"),
    )
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, value, flags=re.I)
        if updated != value:
            return normalize_space(updated)
    return value


def concise_disposition_clause(text: str) -> str:
    value = compact_clause_text(text)
    value_norm = normalized_text(value)
    if re.search(r"\bby which the application was dismissed\b", value_norm):
        return "The application was dismissed."
    if re.search(r"\bby which the application was refused\b", value_norm):
        return "The application was refused."
    if re.search(r"\bby which the application was granted\b", value_norm):
        return "The application was granted."
    if "no realistic prospect" in value_norm and "totally without merit" in value_norm:
        return "There was no realistic prospect of the applicant succeeding on appeal, and the application was totally without merit."
    if "no realistic prospect" in value_norm and "permission to appeal" in value_norm:
        return "There was no realistic prospect of the applicant succeeding on appeal."
    if "claim is to proceed to trial to be determined alongside the main issues" in value_norm:
        return "The application was dismissed. The claim will proceed to trial alongside the main issues."
    if "restoring the second part 50 order" in value_norm and "dismissing the appeal" in value_norm:
        value = re.sub(r"^.*?allowed\s+the\s+appeal\s+to\s+the\s+extent\s+shown\s+in\s+our\s+order\s+above,\s*", "", value, flags=re.I)
        value = normalize_space(value.replace("Ms Onitaand", "Ms Onita and"))
        value = re.sub(r"^restoring\b", "The appeal was allowed in part, restoring", value, flags=re.I)
        value = re.sub(
            r"\band\s+dismissing\s+the\s+Appeal\s+against\s+the\s+Order,\s+insofar\s+as\s+it\s+relates\s+to\b",
            "but dismissing the appeal against the Order as it relates to",
            value,
            flags=re.I,
        )
        return value.rstrip(".") + "."
    if (
        "appeal" in value_norm
        and "allowed" in value_norm
        and any(marker in value_norm for marker in ("to the extent shown", "insofar as", "restoring", "dismissing"))
    ):
        return "The appeal was allowed in part."
    value = re.sub(r"^For all of the foregoing reasons,\s*", "", value, flags=re.I)
    value = re.sub(r"^For the reasons set out above,\s*", "", value, flags=re.I)
    value = re.sub(r"^In those circumstances,\s*", "", value, flags=re.I)
    value = re.sub(r"^we have\s+", "", value, flags=re.I)
    value = re.sub(r"^The Appeal is allowed, to the following extent\.?$", "The appeal was allowed.", value, flags=re.I)
    value = normalize_space(value)
    value = normalize_party_role_sentence(value)
    value = re.sub(
        r"^The\s+(.+?)\s+is\s+(dismissed|refused|granted|allowed|rejected|discharged|set aside)\b",
        lambda m: f"The {normalize_space(m.group(1))} was {m.group(2)}",
        value,
        flags=re.I,
    )
    return value.rstrip(".") + "."


def infer_application_descriptor(text: str) -> str:
    value = normalize_space(text)
    if not value:
        return ""
    value_norm = normalized_text(value)
    actor = ""
    actor_patterns = (
        r"\bUPON\s+the\s+([A-Za-z][A-Za-z ]+?)'?s\s+(?:Application|Appeal Notice)\b",
        r"\bThis\s+is\s+an\s+application\s+by\s+the\s+([A-Za-z][A-Za-z ]+?)\b",
        r"\bThis\s+Application\s+is\s+brought\s+.*?\bby\s+the\s+([A-Za-z][A-Za-z ]+?)\b",
    )
    for pattern in actor_patterns:
        match = re.search(pattern, value, re.I)
        if not match:
            continue
        actor = normalize_space(match.group(1))
        actor = re.sub(r"\bthe\b", "", actor, flags=re.I).strip()
        if actor:
            break

    kind = ""
    if "permission to appeal" in value_norm:
        kind = "for permission to appeal"
    elif "default costs certificate" in value_norm:
        kind = "for a Default Costs Certificate"
    elif "no costs application" in value_norm:
        kind = "for a No Costs Application"
    elif "immediate judgment" in value_norm and "strike out" in value_norm:
        kind = "for immediate judgment and/or strike out"
    elif "immediate judgment" in value_norm:
        kind = "for immediate judgment"
    elif "strike out" in value_norm:
        kind = "to strike out"
    elif "set aside application" in value_norm:
        kind = "to set aside"
    elif "adjournment" in value_norm:
        kind = "for an adjournment"

    if not kind:
        return ""
    if actor:
        suffix = "'" if actor.lower().endswith("s") else "'s"
        descriptor = f"the {actor}{suffix} application"
    else:
        descriptor = "the application"
    if descriptor and kind and kind not in descriptor:
        descriptor = f"{descriptor} {kind}"
    return normalize_space(descriptor)


def cost_clause_kind(text: str) -> str:
    clause = normalized_text(text)
    if not clause:
        return "none"
    if "within 14 days" in clause or "interest shall accrue" in clause:
        return "payment_terms"
    if "statement of costs" in clause and "within" in clause:
        return "statement_deadline"
    if "no order as to costs" in clause or "no costs ordered" in clause or "no costs order" in clause:
        return "no_order"
    if "bear its own costs" in clause:
        return "own_costs"
    if any(token in clause for token in ("usd ", "aed ", "standard basis", "to be assessed", "costs are awarded", "costs awarded", "costs for the claim")):
        return "award"
    if "shall pay" in clause or "must pay" in clause or "costs of the application" in clause or "costs of the appeal" in clause:
        return "award"
    return "other_cost"


def enhance_order_answer_with_descriptor(answer: str, descriptor: str) -> str:
    value = normalize_space(answer)
    if not value or not descriptor or descriptor == "the application":
        return value
    descriptor_norm = normalized_text(descriptor)
    if re.fullmatch(r"the application(?:\s+(?:for|to)\s+[a-z ]+)?", descriptor_norm):
        return value
    descriptor_cap = descriptor[0].upper() + descriptor[1:]
    replacements = (
        (r"^The court dismissed the application\.", f"{descriptor_cap} was dismissed."),
        (r"^The court dismissed the application\b", f"{descriptor_cap} was dismissed"),
        (r"^The application was dismissed\.", f"{descriptor_cap} was dismissed."),
        (r"^The application was dismissed\b", f"{descriptor_cap} was dismissed"),
        (r"^The court granted the application\.", f"{descriptor_cap} was granted."),
        (r"^The court granted the application\b", f"{descriptor_cap} was granted"),
        (r"^The application was granted\.", f"{descriptor_cap} was granted."),
        (r"^The application was granted\b", f"{descriptor_cap} was granted"),
        (r"^The court rejected the application\.", f"{descriptor_cap} was rejected."),
        (r"^The court rejected the application\b", f"{descriptor_cap} was rejected"),
        (r"^The application was rejected\.", f"{descriptor_cap} was rejected."),
        (r"^The application was rejected\b", f"{descriptor_cap} was rejected"),
        (r"^The court refused the application for permission to appeal\.", f"{descriptor_cap} was refused."),
        (r"^The court refused the application for permission to appeal\b", f"{descriptor_cap} was refused"),
        (r"^The court refused the defendant's application for permission to appeal\.", f"{descriptor_cap} was refused."),
        (r"^The court refused the defendant's application for permission to appeal\b", f"{descriptor_cap} was refused"),
        (r"^The court refused the application\.", f"{descriptor_cap} was refused."),
        (r"^The court refused the application\b", f"{descriptor_cap} was refused"),
        (r"^The application was refused\.", f"{descriptor_cap} was refused."),
        (r"^The application was refused\b", f"{descriptor_cap} was refused"),
    )
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, value, flags=re.I)
        if updated != value:
            return normalize_space(updated)
    return value


def concise_costs_clause(text: str, *, include_followup: bool = False) -> str:
    value = compact_clause_text(text)
    value_norm = normalized_text(value)
    if "no order as to costs" in value_norm:
        return "No order as to costs was made."
    match = re.search(
        r"there\s+shall\s+be\s+no\s+costs\s+ordered\s+in\s+relation\s+to\s+the\s+(.+?)(?:\.|$)",
        value,
        re.I,
    )
    if match:
        subject = normalize_space(match.group(1))
        return f"No costs were ordered in relation to the {subject}."
    if "no costs ordered" in value_norm or "no costs order" in value_norm:
        return "No order as to costs was made."
    if "bear its own costs" in value_norm:
        value = normalize_party_role_sentence(value)
        value = normalize_space(value)
        return value.rstrip(".") + "."
    match = re.search(
        r"the\s+([a-z]+)\s+should\s+pay\s+the\s+([a-z]+)'s\s+costs\s+of\s+the\s+application.*?to\s+be\s+assessed",
        value,
        re.I,
    )
    if match:
        payer = match.group(1).capitalize()
        payee = match.group(2).capitalize()
        return f"The {payer} must pay the {payee}'s costs of the application, to be assessed."
    match = re.search(
        r"the\s+([a-z]+)\s+(?:shall|must)\s+pay\s+the\s+([a-z]+)\s+its\s+costs\s+of\s+the\s+(.+?)\s+on\s+the\s+standard\s+basis",
        value,
        re.I,
    )
    if match:
        payer = match.group(1).capitalize()
        payee = match.group(2).capitalize()
        subject = normalize_space(match.group(3))
        day_match = re.search(r"within\s+(\d+)\s+days?", value, re.I)
        days = normalize_space(day_match.group(1) if day_match else "")
        sentence = f"The {payer} must pay the {payee}'s costs of the {subject} on the standard basis"
        if days and include_followup:
            sentence += f", and the {payee} must file a Statement of Costs within {days} days"
        return sentence.rstrip(".") + "."
    if re.search(
        r"costs\s+are\s+awarded\s+on\s+the\s+standard\s+basis\s+to\s+be\s+assessed\s+by\s+way\s+of\s+parties'\s+submissions",
        value,
        re.I,
    ):
        return "Costs were awarded on the standard basis, to be assessed by way of parties' submissions."
    if "statement of costs" in value_norm:
        if not include_followup:
            return ""
        value = normalize_party_role_sentence(value)
        value = normalize_space(value)
        return value.rstrip(".") + "."
    match = re.search(
        r"^the\s+([a-z]+)\s+must\s+pay\s+the\s+([a-z]+)\s+([0-9]+%?)\s+of\s+the\s+\2'?s\s+costs(?:\s+of\s+the\s+application)?(?:,\s*assessed\s+on\s+the\s+standard\s+basis)?(?:,\s*in\s+the\s+total\s+sum\s+of\s+(USD\s*[0-9,]+(?:\.[0-9]+)?))?",
        value,
        re.I,
    )
    if match:
        payer = match.group(1).capitalize()
        payee = match.group(2).capitalize()
        percentage = normalize_space(match.group(3))
        amount = normalize_space(match.group(4) or "")
        sentence = f"The {payer} must pay {percentage} of the {payee}'s costs"
        if "standard basis" in value_norm:
            sentence += " on the standard basis"
        if amount:
            sentence += f", totaling {amount}"
        return sentence.rstrip(".") + "."
    if "within 14 days" in value_norm or "interest shall accrue" in value_norm:
        if "costs award" in value_norm and "within 14 days" in value_norm:
            sentence = "The Costs Award must be paid within 14 days."
            if "interest shall accrue" in value_norm:
                sentence = sentence.rstrip(".") + ", after which interest accrues."
            return sentence
        value = normalize_party_role_sentence(value)
        value = re.sub(
            r"the\s+applicant\s+must\s+pay\s+the\s+costs\s+award\s+within\s+14\s+days\s+of\s+the\s+date\s+of\s+this\s+order(?:,\s*pursuant\s+to\s+rdc\s+[0-9.]+)?",
            "The Applicant must pay the Costs Award within 14 days",
            value,
            flags=re.I,
        )
        value = normalize_space(value)
        return value.rstrip(".") + "."
    match = re.search(
        r"award of costs for the\s+(.+?)\s*(?:in this appeal)?\s*,?\s*is\s+([0-9]+%?)\s+of the amount it has claimed,\s+namely the sum of\s+(USD\s*[0-9,]+(?:\.[0-9]+)?)",
        value,
        re.I,
    )
    if match:
        party = normalize_space(match.group(1))
        percentage = normalize_space(match.group(2))
        amount = normalize_space(match.group(3))
        return f"{party} was awarded {percentage} of its claimed costs, namely {amount}."
    value = re.sub(r"^Taking those [^.]+ into account,\s*", "", value, flags=re.I)
    value = re.sub(r"^Consequently,\s*", "", value, flags=re.I)
    value = normalize_space(value)
    value = normalize_party_role_sentence(value)
    return value.rstrip(".") + "."


def strip_heading_prefix(text: str) -> str:
    value = normalize_space(text)
    value = re.sub(r"^#+\s*", "", value)
    return normalize_space(value)


def is_heading_line(text: str) -> bool:
    value = normalize_space(text)
    if not value:
        return True
    if value.startswith("#"):
        return True
    return bool(re.fullmatch(r"[A-Z][A-Z\s/&\-]{3,}", value))


def short_title_label(title: str) -> str:
    tokens = [token for token in re.findall(r"[A-Za-z0-9]+", normalize_space(title.replace("#", ""))) if len(token) > 3 and token.lower() not in ARTICLE_STOPWORDS]
    if not tokens:
        return normalize_space(title.replace("#", ""))
    if len(tokens) >= 3:
        return "".join(token[0].upper() for token in tokens[:3])
    return " ".join(token.title() for token in tokens[:2])


def readable_source_label(title: str) -> str:
    value = normalize_space(title.replace("#", " "))
    value = re.sub(r"\bDIFC\b", "", value, flags=re.I)
    value = re.sub(r"\bLaw No\.?\s*\d+\s+of\s+\d{4}\b", "", value, flags=re.I)
    value = re.sub(r"\b20\d{2}\b", "", value)
    value = normalize_space(value).strip(" -")
    if value.isupper():
        value = value.title()
    if value and len(value) <= 44:
        return value
    return short_title_label(title)


def answer_source_label(title: str) -> str:
    value = normalize_space(title.replace("#", " "))
    value = re.sub(r"\bDIFC Law No\.?\s*\d+\s+of\s+\d{4}\b", "", value, flags=re.I)
    value = re.sub(r"\bLaw No\.?\s*\d+\s+of\s+\d{4}\b", "", value, flags=re.I)
    value = re.sub(r"\b20\d{2}\b", "", value)
    value = normalize_space(value).strip(" -")
    if value.isupper():
        value = value.title()
    value = re.sub(
        r"\b(On|The|Of|And|In|To|For|Under|At|By|Or)\b",
        lambda match: match.group(1).lower(),
        value,
    )
    if value:
        value = value[0].upper() + value[1:]
    value = re.sub(r"\bDifc\b", "DIFC", value)
    return value or readable_source_label(title)


def question_subject_label(question: str, article_ref: str | None = None) -> str:
    text = normalize_space(question)
    if not text:
        return ""
    if article_ref:
        pattern = re.compile(
            rf"{re.escape(normalize_space(article_ref))}\s+of\s+(.+?)(?:,|\?|$)",
            re.I,
        )
        match = pattern.search(text)
        if match:
            subject = normalize_space(match.group(1))
            subject = re.sub(r"^the\s+", "", subject, flags=re.I)
            return subject
    for pattern in (
        re.compile(r"who administers\s+the\s+(.+?)\??$", re.I),
        re.compile(r"who is responsible for administering\s+the\s+(.+?)\??$", re.I),
    ):
        match = pattern.search(text)
        if match:
            subject = normalize_space(match.group(1))
            subject = re.sub(r"^the\s+", "", subject, flags=re.I)
            return subject
    return ""


def is_operative_order_clause(text: str) -> bool:
    clause = normalized_text(text)
    if not clause:
        return False
    if clause.startswith(
        (
            "this is ",
            "the test for ",
            "under rdc ",
            "by rdc ",
            "in the notice ",
            "in those circumstances",
            "for the reasons set out below",
            "the issue for determination",
            "on the above",
            "i granted ",
            "i do not consider ",
        )
    ):
        return False
    patterns = (
        r"^the\s+.+?\s+is\s+(?:refused|dismissed|granted|allowed|discharged|rejected|denied|struck out)\b",
        r"^the\s+.+?\s+shall\s+pay\b",
        r"^the\s+claim\s+is\s+to\s+proceed\s+to\s+trial\b",
        r"^the\s+request\s+.+?\s+is\s+(?:refused|rejected|denied)\b",
        r"^(?:the\s+)?permission\s+to\s+appeal\b.*\b(?:refused|rejected|denied|granted|allowed)\b",
        r"^we\s+have\s+(?:allowed|dismissed|granted|upheld|restored|rejected)\b",
        r"^for\s+all\s+.+?\s+we\s+have\s+(?:allowed|dismissed|granted|upheld|restored|rejected)\b",
        r"^there\s+shall\s+be\s+no\s+order\s+as\s+to\s+costs\b",
        r"^costs\s+(?:are|shall\s+be)\s+awarded\b",
        r"^the\s+asi\s+order\s+is\s+discharged\b",
    )
    return any(re.search(pattern, clause, re.I) for pattern in patterns)


def is_primary_outcome_order_clause(text: str) -> bool:
    clause = normalized_text(text)
    clause = re.sub(r"^(for all of the foregoing reasons|for the reasons set out above|in those circumstances),\s*", "", clause)
    patterns = (
        r"^the\s+.+?\s+is\s+(?:refused|dismissed|granted|allowed|discharged|rejected|denied|struck out)\b",
        r"^the\s+request\s+.+?\s+is\s+(?:refused|rejected|denied)\b",
        r"^(?:the\s+)?permission\s+to\s+appeal\b.*\b(?:refused|rejected|denied|granted|allowed)\b",
        r"^the\s+claim\s+is\s+to\s+proceed\s+to\s+trial\b",
        r"^we\s+have\s+(?:allowed|dismissed|granted|upheld|restored|rejected)\b",
        r"^for\s+all\s+.+?\s+we\s+have\s+(?:allowed|dismissed|granted|upheld|restored|rejected)\b",
        r"^the\s+asi\s+order\s+is\s+discharged\b",
    )
    return any(re.search(pattern, clause, re.I) for pattern in patterns)


def is_cost_related_order_clause(text: str) -> bool:
    clause = normalized_text(text)
    explicit_markers = (
        "shall pay",
        "statement of costs",
        "within 14 days",
        "interest shall accrue",
        "bear its own costs",
        "no order as to costs",
        "award of costs",
        "entitled to its costs",
        "costs are awarded",
        "costs awarded",
        "costs of the application",
        "costs of the appeal",
    )
    if any(marker in clause for marker in explicit_markers):
        return True
    return bool(
        re.search(r"\bcosts?\b.{0,60}\b(?:assessed|fixed|sum|amount|awarded|payable|payment|discount)\b", clause, re.I)
    )


def order_clause_topics(text: str) -> set[str]:
    clause = normalized_text(text)
    topics: set[str] = set()
    topic_markers = {
        "permission_to_appeal": ("permission to appeal",),
        "appeal": ("appeal",),
        "adjournment": ("adjournment",),
        "set_aside": ("set aside",),
        "asi_order": ("asi order",),
        "trial": ("trial", "proceed to trial"),
        "reconsideration": ("reconsidered at a hearing", "reconsideration", "request that the decision be reconsidered"),
        "costs": ("cost", "shall pay", "statement of costs", "bear its own costs", "no order as to costs", "award of costs"),
        "payment_terms": ("within 14 days", "interest shall accrue"),
    }
    for topic, markers in topic_markers.items():
        if any(marker in clause for marker in markers):
            topics.add(topic)
    if is_primary_outcome_order_clause(text):
        topics.add("primary_outcome")
    return topics


def law_id_matches_text(text: str, law_id: str) -> bool:
    text_ids = set(extract_law_ids(text))
    if law_id in text_ids:
        return True
    match = re.search(r"Law No\.?\s*(\d+)\s+of\s+(\d{4})", law_id, re.I)
    if not match:
        return False
    number, year = match.groups()
    pattern = re.compile(rf"(?:DIFC\s+)?Law\s+No\.?\s*0*{int(number)}\s+of\s+{re.escape(year)}", re.I)
    return pattern.search(text) is not None


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
        return list(dict.fromkeys(explicit))
    if total_titles > 1 and len(plain) >= total_titles and index < len(plain):
        return [plain[index]]
    return list(dict.fromkeys(plain))


def article_query_tokens(analysis: QuestionAnalysis, question: str) -> set[str]:
    tokens = set()
    for source in [question, *analysis.must_support_terms]:
        for token in re.findall(r"[a-z0-9]+", normalized_text(source)):
            if len(token) > 3 and token not in ARTICLE_STOPWORDS:
                tokens.add(token)
    return tokens


def canonical_person_name(value: str) -> str:
    value = re.sub(r"\b(H\.E\.|Justice|Judge|Chief Justice|Deputy Chief Justice|KC)\b", "", value, flags=re.I)
    return normalize_space(value).lower()


def iso_date_from_text(value: str) -> Optional[str]:
    try:
        return date_parser.parse(value, fuzzy=True, dayfirst=True).date().isoformat()
    except Exception:
        return None


def display_date(value: str) -> str:
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
            parsed = date_parser.isoparse(value.strip()).date()
        else:
            parsed = date_parser.parse(value, fuzzy=True, dayfirst=True).date()
        return f"{parsed.day} {parsed.strftime('%B %Y')}"
    except Exception:
        return normalize_space(value)


def parse_between_block(text: str) -> tuple[List[str], List[str]]:
    match = re.search(
        r"\bBETWEEN\b(.*?)(?:ORDER WITH REASONS|REASONS FOR THE ORDER|JUDGMENT OF THE COURT|SCHEDULE OF REASONS|IT IS HEREBY ORDERED THAT)",
        text,
        re.I | re.S,
    )
    if not match:
        return [], []
    block = match.group(1)
    lines = [normalize_space(line) for line in block.splitlines() if normalize_space(line)]
    left: List[str] = []
    right: List[str] = []
    current = left
    for line in lines:
        lower = line.lower()
        if lower == "between":
            continue
        if lower == "and":
            current = right
            continue
        if any(title in lower for title in ROLE_TITLES):
            continue
        current.append(line)
    return clean_party_names(left), clean_party_names(right)


def clean_party_names(lines: Sequence[str]) -> List[str]:
    result: List[str] = []
    for line in lines:
        numbered_parts = [
            normalize_space(match.group(1))
            for match in re.finditer(r"\(\d+\)\s*([^()]+?)(?=(?:\(\d+\)|$))", line)
            if normalize_space(match.group(1))
        ]
        if len(numbered_parts) >= 2:
            result.extend(clean_party_names(numbered_parts))
            continue
        line = re.sub(r"^\(\d+\)\s*", "", line).strip()
        line = re.sub(r"^[A-Z]{2,4}\s+\d{3}/\d{4}\s+", "", line, flags=re.I)
        line = normalize_space(line)
        if not line:
            continue
        if set(line) <= {"#"}:
            continue
        if line.lower() in {"between", "and"}:
            continue
        if "claim no" in line.lower():
            continue
        if re.search(r"\bthe dubai international financial centre courts\b", line, re.I):
            continue
        if re.search(r"\bjudgments? and orders?\b", line, re.I):
            continue
        if re.search(r"\border with reasons\b", line, re.I):
            continue
        if re.search(r"\[\d{4}\]\s+difc\b", line, re.I):
            continue
        if re.search(r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", line, re.I):
            continue
        if re.search(r"\b[a-z0-9 .'\-]+\s+v\s+[a-z0-9 .'\-]+\b", line, re.I):
            continue
        result.append(line)
    return result


def party_names_look_valid(names: Sequence[str]) -> bool:
    if not names:
        return False
    for name in names:
        value = normalize_space(name)
        if not value:
            return False
        if re.search(r"\bthe dubai international financial centre courts\b", value, re.I):
            return False
        if re.search(r"\[\d{4}\]\s+difc\b", value, re.I):
            return False
        if re.search(r"\bjudgments? and orders?\b", value, re.I):
            return False
        if re.search(r"\border with reasons\b", value, re.I):
            return False
        if re.search(r"\bclaim no\b", value, re.I):
            return False
        if re.search(r"\b[a-z0-9 .'\-]+\s+v\s+[a-z0-9 .'\-]+\b", value, re.I):
            return False
    return True


def parse_parties_from_title(title_line: str) -> tuple[List[str], List[str]]:
    title_line = normalize_space(title_line.lstrip("# "))
    title_line = re.sub(r"^[A-Z]{2,4}\s+\d{3}/\d{4}\s+", "", title_line, flags=re.I)
    match = re.match(r"(.+?)\s+v\s+(.+?)(?:\s+\[\d{4}\]|\s+(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)|\s+Claim No|\Z)", title_line, re.I)
    if not match:
        return [], []
    left = clean_party_names([match.group(1)])
    right = clean_party_names([match.group(2)])
    return left, right


def extract_judges(text: str) -> List[str]:
    names: List[str] = []
    for pattern in JUDGE_PATTERNS:
        for match in pattern.finditer(text):
            name = normalize_space(match.group(1))
            if not name:
                continue
            tokens = [
                token
                for token in re.split(r"\s+", name)
                if token and normalized_text(token) not in JUDGE_NOISE_TOKENS and len(token) >= 3
            ]
            if not tokens:
                continue
            name = normalize_space(" ".join(tokens[:5]))
            names.append(name)
    deduped: List[str] = []
    seen = set()
    for name in names:
        key = canonical_person_name(name)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(name)
    return deduped


def extract_presiding_judges(page_texts: Dict[int, str]) -> List[str]:
    heading_lines: List[str] = []
    fallback_lines: List[str] = []
    procedural_markers = (
        "upon ",
        "and upon ",
        "it is hereby ordered",
        "issued by",
        "date of issue",
        "schedule of reasons",
    )
    heading_markers = (
        "order with reasons of",
        "judgment of",
        "before ",
        "reasons of the court of appeal",
    )

    for page_number in sorted(page_texts)[:2]:
        raw_lines = [normalize_space(line) for line in str(page_texts.get(page_number) or "").splitlines()]
        lines = [line for line in raw_lines if line]
        for line in lines[:40]:
            line_norm = normalized_text(line)
            if not line_norm:
                continue
            if any(line_norm.startswith(marker) for marker in procedural_markers):
                break
            if any(marker in line_norm for marker in heading_markers):
                heading_lines.append(line)
            else:
                fallback_lines.append(line)

    names = extract_judges("\n".join(heading_lines))
    if names:
        return names

    # Fallback to the short preamble only; avoid procedural-history mentions of prior judges.
    names = extract_judges("\n".join(fallback_lines))
    if names:
        return names

    return extract_judges("\n".join(page_texts.get(page, "") for page in sorted(page_texts)[:2]))


def extract_claim_amount(text: str) -> Optional[float]:
    for pattern in CLAIM_AMOUNT_PATTERNS:
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


def extract_claim_amount_candidate(text: str) -> Optional[tuple[float, int]]:
    best: Optional[tuple[float, int]] = None
    for pattern, score in CLAIM_AMOUNT_SCORED_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        try:
            amount = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if best is None or score > best[1]:
            best = (amount, score)
    return best


def normalize_claim_number(value: str) -> str:
    normalized = normalize_space(value).upper()
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    return normalized


def claim_number_key(value: str) -> str:
    return re.sub(r"[^a-z0-9/]", "", normalized_text(value))


def extract_origin_claim_number(text: str, current_case_id: str) -> Optional[str]:
    text_norm = normalized_text(text)
    if not any(marker in text_norm for marker in ("appeal from", "originated from", "originating from", "appeal notice", "lower court", "judgment")):
        return None
    current_key = claim_number_key(current_case_id)
    for pattern in ORIGIN_CLAIM_NUMBER_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        candidate = normalize_claim_number(match.group(1))
        if claim_number_key(candidate) != current_key:
            return candidate
    return None


def extract_order_pages(page_texts: Dict[int, str]) -> List[int]:
    result: List[int] = []
    for page_number, text in page_texts.items():
        text = normalized_text(text)
        if any(marker in text for marker in ORDER_MARKERS):
            result.append(int(page_number))
    return result


def build_case_metadata(corpus: PublicCorpus, sha: str) -> Optional[CaseMetadata]:
    document = corpus.documents[sha]
    if document.kind != "case":
        return None
    payload = corpus.documents_payload[sha]
    pages = payload["content"]["pages"]
    page_texts = merged_pdf_page_texts(corpus.pdf_dir, sha, pages)
    joined_first_pages = "\n".join(page_texts.get(int(page["page"]), str(page.get("text") or "")) for page in pages[:3])
    issue_page = None
    issue_date = None
    for page in pages[:3]:
        text = page_texts.get(int(page["page"]), str(page.get("text") or ""))
        m = re.search(r"Date of Issue:\s*([^\n]+)", text, re.I)
        if m:
            issue_page = int(page["page"])
            issue_date = iso_date_from_text(m.group(1))
            break
    if issue_date is None and pages:
        first_page_head = " ".join(page_texts.get(1, str(pages[0].get("text") or "")).splitlines()[:5])
        issue_date = iso_date_from_text(first_page_head)
        issue_page = 1 if issue_date else None
    claimants, defendants = parse_between_block(joined_first_pages)
    if pages:
        first_page_lines = page_texts.get(1, str(pages[0].get("text") or "")).splitlines()
        title_line = next((normalize_space(line) for line in first_page_lines if normalize_space(line)), "")
        fallback_claimants, fallback_defendants = parse_parties_from_title(title_line)
        if fallback_claimants and not party_names_look_valid(claimants):
            claimants = fallback_claimants
        if fallback_defendants and not party_names_look_valid(defendants):
            defendants = fallback_defendants
    claim_amount = None
    claim_amount_page = None
    claim_amount_score = -1
    origin_claim_number = None
    origin_claim_page = None
    for page in pages[:25]:
        text = page_texts.get(int(page["page"]), str(page.get("text") or ""))
        candidate = extract_claim_amount_candidate(text)
        if candidate is None:
            continue
        amount, score = candidate
        if score > claim_amount_score:
            claim_amount = amount
            claim_amount_page = int(page["page"])
            claim_amount_score = score
    case_id = next((item for item in document.canonical_ids if re.match(r"^(CFI|ARB|TCD|CA|ENF|DEC|SCT)", item)), document.title)
    for page in pages[:6]:
        text = page_texts.get(int(page["page"]), str(page.get("text") or ""))
        origin_claim_number = extract_origin_claim_number(text, case_id)
        if origin_claim_number is not None:
            origin_claim_page = int(page["page"])
            break
    return CaseMetadata(
        sha=sha,
        case_id=case_id,
        issue_date=issue_date,
        first_page=1,
        issue_page=issue_page,
        claimant_names=claimants,
        defendant_names=defendants,
        all_main_parties=claimants + defendants,
        judges=extract_presiding_judges({int(page["page"]): page_texts.get(int(page["page"]), str(page.get("text") or "")) for page in pages[:3]}),
        claim_amount_aed=claim_amount,
        claim_amount_page=claim_amount_page,
        origin_claim_number=origin_claim_number,
        origin_claim_page=origin_claim_page,
        order_pages=extract_order_pages(page_texts),
    )


def case_metadata_score(meta: CaseMetadata) -> int:
    score = 0
    if meta.issue_date:
        score += 2
    if meta.claimant_names:
        score += 2
    if meta.defendant_names:
        score += 2
    if meta.judges:
        score += 1
    if meta.claim_amount_aed is not None:
        score += 4
    if meta.origin_claim_number:
        score += 3
    if meta.order_pages:
        score += 1
    return score


def build_law_metadata(corpus: PublicCorpus, sha: str) -> Optional[LawMetadata]:
    document = corpus.documents[sha]
    if document.kind not in {"law", "regulation"}:
        return None
    payload = corpus.documents_payload[sha]
    pages = payload["content"]["pages"]
    page_texts = merged_pdf_page_texts(corpus.pdf_dir, sha, pages)
    first_page_text = page_texts.get(1, pages[0]["text"] if pages else "")
    law_id = document.title
    law_number = None
    primary_match = re.search(r"DIFC\s+LAW\s+NO\.?\s*(\d+)\s+OF\s+(\d{4})", first_page_text, re.I)
    if primary_match:
        law_number = int(primary_match.group(1))
        law_id = f"Law No. {law_number} of {primary_match.group(2)}"
    else:
        fallback_match = next((item for item in document.canonical_ids if item.startswith("Law No.")), None)
        if fallback_match:
            law_id = fallback_match
            number_match = re.search(r"Law No\.?\s*(\d+)\s+of\s+(\d{4})", fallback_match, re.I)
            if number_match:
                law_number = int(number_match.group(1))
    publication_match = re.search(r"Consolidated Version(?: No\.\s*\d+)?\s*\(?([A-Za-z]+\s+\d{4})\)?", first_page_text, re.I)
    publication_text = normalize_space(publication_match.group(1)) if publication_match else None
    publication_page = 1 if publication_text else None
    consolidated_version_number = None
    version_match = re.search(r"Consolidated Version\s+No\.\s*(\d+)", first_page_text, re.I)
    if version_match:
        consolidated_version_number = int(version_match.group(1))
    made_by = None
    made_by_page = None
    administered_by = None
    administered_by_page = None
    enacted_text = None
    enacted_page = None
    commencement_page = None
    effective_date = None
    effective_date_page = None
    for page in pages[:8]:
        text = page_texts.get(int(page["page"]), str(page.get("text") or ""))
        if made_by is None:
            m = re.search(r"(?:This|The)\s+Law\s+is\s+made\s+by\s+the\s+([^.]+)\.", text, re.I)
            if not m:
                m = re.search(
                    rf"These\s+Regulations\s+are\s+made\s+by\s+the\s+(.+?)(?:\s+pursuant\s+to|\s+on\s+[0-9]{{1,2}}\s+{MONTH_NAME_RE}\s+[0-9]{{4}})",
                    text,
                    re.I,
                )
            if m:
                made_by = normalize_space(m.group(1))
                made_by_page = int(page["page"])
        if administered_by is None:
            m = re.search(
                r"(?:This|The)\s+Law(?:\s+and\s+any\s+Regulations\s+made\s+under\s+it)?\s+"
                r"(?:shall\s+be|is)\s+administered\s+by\s+the\s+([^.]+)\.",
                text,
                re.I,
            )
            if m:
                administered_by = normalize_space(m.group(1))
                administered_by_page = int(page["page"])
        if enacted_text is None:
            m = re.search(r"(?:This|The)\s+Law\s+is\s+enacted\s+on\s+([^.]+)\.", text, re.I)
            if not m:
                m = re.search(
                    rf"These\s+Regulations\s+are\s+made\s+by\s+the\s+.+?\s+on\s+([0-9]{{1,2}}\s+{MONTH_NAME_RE}\s+[0-9]{{4}})\b",
                    text,
                    re.I,
                )
            if m:
                enacted_text = normalize_space(m.group(1))
                enacted_page = int(page["page"])
        if commencement_page is None and re.search(r"##\s*6\.\s*Commencement", text, re.I):
            commencement_page = int(page["page"])
        if effective_date is None:
            m = re.search(rf"\bIn force on\s+([0-9]{{1,2}}\s+{MONTH_NAME_RE}\s+[0-9]{{4}})\b", text, re.I)
            if not m:
                m = re.search(
                    rf"\b(?:shall\s+come\s+into\s+force|comes\s+into\s+force|came\s+into\s+force|effective\s+from)\s+on\s+([0-9]{{1,2}}\s+{MONTH_NAME_RE}\s+[0-9]{{4}})\b",
                    text,
                    re.I,
                )
            if not m:
                m = re.search(
                    rf"\b(?:shall\s+come\s+into\s+force|comes\s+into\s+force|came\s+into\s+force|effective\s+from)\s+([0-9]{{1,2}}\s+{MONTH_NAME_RE}\s+[0-9]{{4}})\b",
                    text,
                    re.I,
                )
            if m:
                effective_date = iso_date_from_text(m.group(1))
                effective_date_page = int(page["page"])
    return LawMetadata(
        sha=sha,
        law_id=law_id,
        title=document.title,
        law_number=law_number,
        publication_text=publication_text,
        publication_page=publication_page,
        enacted_text=enacted_text,
        enacted_page=enacted_page,
        made_by=made_by,
        made_by_page=made_by_page,
        administered_by=administered_by,
        administered_by_page=administered_by_page,
        page_one=1,
        commencement_page=commencement_page,
        consolidated_version_number=consolidated_version_number,
        effective_date=effective_date,
        effective_date_page=effective_date_page,
    )


def consultation_topic_from_title(title: str) -> Optional[str]:
    value = normalize_space(title.replace("#", " "))
    value = re.sub(r"^\s*Annex\s+\d+\s*", "", value, flags=re.I)
    value = re.sub(
        r"\bCONSULTATION PAPER NO\.?\s*\d+(?:\s+[A-Za-z]+)?(?:\s+\d{4})?\b",
        " ",
        value,
        flags=re.I,
    )
    value = re.sub(r"\bDIFC LAW NO\.?\s*\d+\s+of\s+\d{4}\b", "", value, flags=re.I)
    value = re.sub(r"^\s*PROPOSED AMENDMENTS TO\s+", "", value, flags=re.I)
    value = re.sub(r"^\s*PROPOSALS RELATING TO AN AMENDMENT TO THE\s+", "", value, flags=re.I)
    value = re.sub(r"^\s*PROPOSALS RELATING TO A NEW\s+", "", value, flags=re.I)
    value = re.sub(r"^\s*PROPOSALS RELATING TO\s+", "", value, flags=re.I)
    value = re.sub(r"^\s*PROPOSED NEW\s+", "", value, flags=re.I)
    value = re.sub(r"^\s*ON\s+", "", value, flags=re.I)
    value = re.sub(r"^\s*AMENDED\s+", "", value, flags=re.I)
    value = normalize_space(value).strip(" -")
    if not value:
        return None
    if value.isupper():
        value = value.title()
    value = re.sub(r"\b(On|The|Of|And|In|To|For|Under|At|By|Or)\b", lambda match: match.group(1).lower(), value)
    return value[0].upper() + value[1:] if value else None


def consultation_heading_subject(text: str) -> Optional[str]:
    lines = [
        normalize_space(re.sub(r"^#+\s*", "", line))
        for line in str(text or "").splitlines()
        if normalize_space(re.sub(r"^#+\s*", "", line))
    ]
    if not lines:
        return None
    seen_cp = False
    for line in lines:
        upper = line.upper()
        if "CONSULTATION PAPER" in upper:
            inline_candidate = consultation_topic_from_title(line)
            if inline_candidate and len(consultation_match_tokens(inline_candidate)) >= 2:
                return inline_candidate
            seen_cp = True
            continue
        if not seen_cp:
            continue
        if re.fullmatch(r"[A-Za-z]+\s+\d{4}", line):
            continue
        if re.fullmatch(r"DIFC\s+LAW\s+NO\.?\s*\d+\s+OF\s+\d{4}", line, re.I):
            continue
        if line.lower().startswith(("why are we issuing this paper", "who should read this paper", "how to provide comments", "what happens next")):
            continue
        if any(keyword in upper for keyword in ("LAW", "REGULATION", "REGULATIONS", "RULES", "LEGISLATION", "DIGITAL ASSETS")):
            return consultation_topic_from_title(line)
        candidate = consultation_topic_from_title(line)
        if candidate and len(consultation_match_tokens(candidate)) >= 2:
            return candidate
    for line in lines:
        upper = line.upper()
        if "CONSULTATION PAPER" in upper:
            continue
        if re.fullmatch(r"[A-Za-z]+\s+\d{4}", line):
            continue
        if re.fullmatch(r"DIFC\s+LAW\s+NO\.?\s*\d+\s+OF\s+\d{4}", line, re.I):
            continue
        if line.lower().startswith(("why are we issuing this paper", "who should read this paper", "how to provide comments", "what happens next")):
            continue
        candidate = consultation_topic_from_title(line)
        if candidate and len(consultation_match_tokens(candidate)) >= 2:
            return candidate
    return None


CONSULTATION_TOPIC_STOPWORDS = {
    "consultation",
    "paper",
    "proposal",
    "proposals",
    "proposed",
    "relating",
    "amendment",
    "amendments",
    "difc",
    "new",
}


def consultation_focus_tokens(value: str) -> set[str]:
    normalized = normalized_text(value)
    normalized = re.sub(r"consultation\s+paper\s+no\.?\s*\d+", " ", normalized)
    normalized = re.sub(r"\blaw\s+no\.?\s*\d+\s+of\s+\d{4}\b", " ", normalized)
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) > 3 and token not in CONSULTATION_TOPIC_STOPWORDS and not token.isdigit()
    }


def consultation_topic_is_generic(topic: Optional[str]) -> bool:
    if not topic:
        return True
    topic_norm = normalized_text(topic)
    if "consultation paper no" in topic_norm or topic_norm.startswith("annex"):
        return True
    return len(consultation_match_tokens(topic)) == 0


def is_primary_consultation_document(title: str, first_page_text: str = "", body_text: str = "") -> bool:
    combined = normalize_space(" ".join([title, first_page_text, body_text])).lower()
    if "consultation paper" not in combined and "issued for consultation purposes only" not in combined:
        return False
    strong_body_markers = (
        "why are we issuing this paper",
        "who should read this paper",
        "how to provide comments",
        "what happens next",
        "this consultation paper seeks public comments",
        "this consultation paper seeks public comment",
    )
    if any(marker in combined for marker in strong_body_markers):
        return True
    primary_markers = (
        "annex ",
        "appendix ",
        "roadmap",
        "format for providing public comments",
        "table of comments",
        "comments received",
    )
    if any(marker in combined for marker in primary_markers):
        return False
    return normalize_space(title).lower().startswith("consultation paper")


def extract_consultation_paper_number(text: str) -> Optional[str]:
    match = re.search(r"consultation\s+paper\s+no\.?\s*(\d+)", str(text or ""), re.I)
    if not match:
        return None
    return match.group(1)


CONSULTATION_TITLE_STOPWORDS = {
    "consultation",
    "paper",
    "proposed",
    "proposal",
    "proposals",
    "relating",
    "amendment",
    "amendments",
    "difc",
    "only",
    "issued",
    "purposes",
    "purpose",
    "new",
    "law",
    "laws",
    "regulation",
    "regulations",
    "rule",
    "rules",
    "what",
    "when",
    "deadline",
    "comment",
    "comments",
    "providing",
    "provide",
    "submitted",
    "submitting",
    "responses",
    "response",
    "public",
    "read",
    "issuing",
    "issued",
    "dubai",
    "international",
    "financial",
    "centre",
    "authority",
    "according",
    "operating",
    "operations",
    "business",
    "conduct",
    "conducting",
    "proposing",
    "persons",
    "companies",
    "should",
}


def consultation_match_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized_text(text))
        if len(token) > 3 and token not in CONSULTATION_TITLE_STOPWORDS and not token.isdigit()
    }


def consultation_instrument_kind(text: str) -> str:
    norm = normalized_text(text)
    if "regulations" in norm or " regulation " in f" {norm} " or "rules" in norm:
        return "regulation"
    if " law " in f" {norm} " or "laws" in norm:
        return "law"
    return ""


def build_consultation_metadata(corpus: PublicCorpus, sha: str) -> Optional[ConsultationMetadata]:
    document = corpus.documents[sha]
    payload = corpus.documents_payload[sha]
    pages = payload["content"]["pages"]
    document_title = normalize_space(getattr(document, "title", "") or "")
    first_page_raw = ""
    if pages:
        first_page_raw = str(pages[0].get("text") or "")
    first_page_hint = normalize_space(getattr(document, "first_page_text", "") or first_page_raw).lower()
    consultation_body_hint = "\n".join(
        str(page.get("text") or "")
        for page in pages[:3]
    )
    if not is_primary_consultation_document(document_title, first_page_hint, consultation_body_hint):
        return None
    page_texts = merged_pdf_page_texts(corpus.pdf_dir, sha, pages)
    deadline_date = None
    deadline_page = None
    best_deadline_score = -1.0
    submission_email = None
    email_page = None
    issuing_body = None
    issuing_body_page = None
    related_law_ids = ordered_unique(
        extract_law_ids(
            "\n".join(
                [
                    document_title,
                    *(page_texts.get(int(page["page"]), str(page.get("text") or "")) for page in pages),
                ]
            )
        )
    )

    for page in pages[:6]:
        page_number = int(page["page"])
        text = page_texts.get(page_number, str(page.get("text") or ""))
        text_norm = normalized_text(text)
        email_match = re.search(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", text, re.I)
        if submission_email is None and email_match:
            submission_email = normalize_space(email_match.group(0))
            email_page = page_number
        if issuing_body is None:
            authority_patterns = (
                "Dubai International Financial Centre Authority",
                "Dubai Financial Services Authority",
                "DIFC Authority",
                "DFSA",
            )
            for authority in authority_patterns:
                if authority.lower() in text_norm:
                    issuing_body = authority
                    issuing_body_page = page_number
                    break
        for segment in [normalize_space(line) for line in text.splitlines() if normalize_space(line)]:
            if not re.search(r"\d{4}", segment):
                continue
            date_match = re.search(
                rf"\b((?:\d{{1,2}}\s*(?:st|nd|rd|th)?\s+{MONTH_NAME_RE}\s+\d{{4}})|(?:{MONTH_NAME_RE}\s+\d{{1,2}}(?:,\s*|\s+)\d{{4}}))\b",
                segment,
                re.I,
            )
            if not date_match:
                continue
            segment_norm = normalized_text(segment)
            score = 0.0
            if any(marker in segment_norm for marker in ("comment", "comments", "response", "responses")):
                score += 8.0
            if any(marker in segment_norm for marker in ("deadline", "submit", "submitted", "provide", "sent")):
                score += 8.0
            if "how to provide comments" in text_norm or "what happens next" in text_norm:
                score += 4.0
            if "deadline for providing comments" in segment_norm or "response deadline" in segment_norm:
                score += 12.0
            if score <= 0:
                continue
            if score > best_deadline_score:
                best_deadline_score = score
                deadline_date = iso_date_from_text(date_match.group(1))
                deadline_page = page_number

    heading_hint = "\n".join(
        normalize_space(line)
        for page in pages[:2]
        for line in str(page_texts.get(int(page["page"]), str(page.get("text") or ""))).splitlines()[:8]
        if normalize_space(line)
    )
    topic = consultation_topic_from_title(document.title)
    if consultation_topic_is_generic(topic):
        topic = consultation_heading_subject(re.sub(r"^Annex\s+\d+\s*", "", heading_hint, flags=re.I))
    if deadline_date is None and submission_email is None and not topic:
        return None
    return ConsultationMetadata(
        sha=sha,
        title=document_title,
        deadline_date=deadline_date,
        deadline_page=deadline_page,
        submission_email=submission_email,
        email_page=email_page,
        topic=topic,
        topic_page=1,
        issuing_body=issuing_body,
        issuing_body_page=issuing_body_page,
        related_law_ids=related_law_ids,
    )


class StructuredWarmupSolver:
    def __init__(self, corpus: PublicCorpus):
        self.corpus = corpus
        self.page_texts: Dict[str, Dict[int, str]] = LazyMergedPageTexts(corpus)
        self.case_by_id: Dict[str, CaseMetadata] = {}
        self.case_meta_by_sha: Dict[str, CaseMetadata] = {}
        self.case_metas_by_id: Dict[str, List[CaseMetadata]] = defaultdict(list)
        self.law_by_id: Dict[str, LawMetadata] = {}
        self.consultations_by_sha: Dict[str, ConsultationMetadata] = {}
        for sha in corpus.documents:
            case_meta = build_case_metadata(corpus, sha)
            if case_meta is not None:
                self.case_meta_by_sha[sha] = case_meta
                self.case_metas_by_id[case_meta.case_id].append(case_meta)
                existing = self.case_by_id.get(case_meta.case_id)
                if existing is None or case_metadata_score(case_meta) >= case_metadata_score(existing):
                    self.case_by_id[case_meta.case_id] = case_meta
            law_meta = build_law_metadata(corpus, sha)
            if law_meta is not None:
                self.law_by_id[law_meta.law_id] = law_meta
                for alias in corpus.documents[sha].aliases:
                    if alias.startswith("Law No.") and alias != law_meta.law_id:
                        continue
                    self.law_by_id.setdefault(alias, law_meta)
            consultation_meta = build_consultation_metadata(corpus, sha)
            if consultation_meta is not None:
                self.consultations_by_sha[sha] = consultation_meta

    def prepare(
        self,
        question: str,
        answer_type: str,
        route: Dict[str, Any],
        reranked: Sequence[Dict[str, Any]],
        analysis: QuestionAnalysis | None = None,
    ) -> StructuredDecision:
        route = dict(route)
        route["question"] = question
        if analysis is not None:
            contract_direct = self._solve_from_contract(question, analysis, answer_type, route, reranked)
            if contract_direct is not None:
                return self._postprocess_decision(contract_direct, question, answer_type, route, analysis)
            direct = self._solve_from_analysis(analysis, answer_type, route)
            if direct is not None:
                return self._postprocess_decision(direct, question, answer_type, route, analysis)
            order_boolean = self._solve_case_order_boolean(analysis, answer_type, route)
            if order_boolean is not None:
                return self._postprocess_decision(order_boolean, question, answer_type, route, analysis)
            law_boolean = self._solve_single_law_topic_boolean(analysis, answer_type, route)
            if law_boolean is not None:
                return self._postprocess_decision(law_boolean, question, answer_type, route, analysis)
            clause_fact = self._solve_single_source_clause_deterministic(analysis, answer_type, route, reranked)
            if clause_fact is not None:
                return self._postprocess_decision(clause_fact, question, answer_type, route, analysis)
            article_fact = self._solve_single_source_article_deterministic(analysis, answer_type, route)
            if article_fact is not None:
                return self._postprocess_decision(article_fact, question, answer_type, route, analysis)
            order_answer = self._solve_case_order_free_text(analysis, answer_type, route)
            if order_answer is not None:
                return self._postprocess_decision(order_answer, question, answer_type, route, analysis)
            single_article_answer = self._solve_single_source_article_content_free_text(analysis, answer_type, route)
            if single_article_answer is not None:
                return self._postprocess_decision(single_article_answer, question, answer_type, route, analysis)
            generic_clause_answer = self._solve_single_source_clause_free_text(analysis, answer_type, route, reranked)
            if generic_clause_answer is not None:
                return self._postprocess_decision(generic_clause_answer, question, answer_type, route, analysis)
            article_answer = self._solve_article_content_free_text(analysis, answer_type, route)
            if article_answer is not None:
                return self._postprocess_decision(article_answer, question, answer_type, route, analysis)
            override = self._override_from_analysis(analysis, answer_type, route)
            if override is not None:
                return StructuredDecision(handled=False, chunks_override=override)
        return StructuredDecision()

    def _postprocess_decision(
        self,
        decision: StructuredDecision,
        question: str,
        answer_type: str,
        route: Dict[str, Any],
        analysis: QuestionAnalysis,
    ) -> StructuredDecision:
        if not decision.handled or answer_type != "free_text" or not decision.answer_payload:
            return decision
        payload = self._render_structured_free_text_payload(
            decision.answer_payload,
            question=question,
            route=route,
            analysis=analysis,
        )
        if analysis.target_field in {
            "amended_laws",
        }:
            payload["skip_refine"] = True
        elif analysis.needs_multi_document_support and analysis.target_field in {
            "comparison_answer",
            "enumeration_answer",
            "article_content",
        }:
            payload["skip_refine"] = True
        return StructuredDecision(
            handled=decision.handled,
            answer_payload=payload,
            chunks_override=decision.chunks_override,
        )

    def _render_structured_free_text_payload(
        self,
        payload: Dict[str, Any],
        *,
        question: str,
        route: Dict[str, Any],
        analysis: QuestionAnalysis,
    ) -> Dict[str, Any]:
        raw_answer = normalize_space(str(payload.get("raw_answer", "") or ""))
        if not raw_answer or raw_answer == FREE_TEXT_ABSENCE_ANSWER:
            return payload
        rendered = self._render_structured_free_text_answer(
            raw_answer,
            question=question,
            route=route,
            analysis=analysis,
        )
        if rendered == raw_answer:
            return payload
        updated = dict(payload)
        updated["raw_answer"] = rendered
        updated["normalized_answer"] = normalize_answer("free_text", rendered)
        return updated

    def render_free_text_payload(
        self,
        payload: Dict[str, Any],
        *,
        question: str,
        route: Dict[str, Any],
        analysis: QuestionAnalysis,
    ) -> Dict[str, Any]:
        return self._render_structured_free_text_payload(
            payload,
            question=question,
            route=route,
            analysis=analysis,
        )

    def _render_structured_free_text_answer(
        self,
        raw_answer: str,
        *,
        question: str,
        route: Dict[str, Any],
        analysis: QuestionAnalysis,
    ) -> str:
        answer = normalize_space(raw_answer)
        field = analysis.target_field
        if field == "order_result":
            return self._render_order_result_free_text(answer, analysis)
        if field in {"made_by", "administered_by", "publication_text", "enacted_text", "effective_dates"}:
            return self._render_law_metadata_free_text(answer, route, analysis)
        if field in {"article_content", "clause_summary"}:
            return self._render_clause_free_text(answer, route, analysis)
        if field == "amended_laws":
            target = (analysis.target_law_ids or [None])[0]
            if target and "this difc law" in normalized_text(answer):
                answer = re.sub(r"\bthis DIFC Law\b", target, answer, flags=re.I)
        return answer

    def _render_order_result_free_text(self, raw_answer: str, analysis: QuestionAnalysis) -> str:
        answer = normalize_space(raw_answer)
        case_id = (analysis.target_case_ids or [None])[0]
        focus = set(analysis.support_focus or [])
        question_norm = normalized_text(analysis.standalone_question or "")
        full_operatives_question = any(
            marker in question_norm
            for marker in (
                "what was the result of the application heard",
                "what was the result of the application",
                "what did the court decide",
                "how did the court",
                "final ruling",
            )
        ) and all(
            marker not in question_norm
            for marker in (
                "specific order or application described",
                "outcome of the application for permission to appeal",
                "conclusion section",
                "it is hereby ordered",
                "ordered that",
                "last page",
                "first page",
            )
        )
        sentences = [normalize_space(part) for part in re.split(r"(?<=[.!?])\s+", answer) if normalize_space(part)]
        if not sentences:
            return answer

        first = sentences[0].rstrip(".")
        if case_id:
            first = re.sub(rf"^In\s+{re.escape(case_id)}\s*,\s*", "", first, flags=re.I)
            first = re.sub(rf"\s+in\s+{re.escape(case_id)}\b", "", first, flags=re.I)
            first = re.sub(rf"\s+in\s+case\s+{re.escape(case_id)}\b", "", first, flags=re.I)
        first_norm = normalized_text(first)
        if first_norm.startswith("the permission to appeal application is refused"):
            first = "the court refused the application for permission to appeal"
        elif first_norm.startswith("the permission to appeal application was refused"):
            first = "the court refused the application for permission to appeal"
        elif first_norm.startswith("the application for permission to appeal was refused"):
            first = "the court refused the application for permission to appeal"
        elif first_norm.startswith("the application is refused") and "permission to appeal" in normalized_text(analysis.standalone_question):
            first = "the court refused the application for permission to appeal"
        elif first_norm.startswith("the application was refused") and "permission to appeal" in normalized_text(analysis.standalone_question):
            first = "the court refused the application for permission to appeal"
        elif first_norm.startswith("the application was refused"):
            first = re.sub(r"^the application was refused", "the court refused the application", first, flags=re.I)
        elif first_norm.startswith("the application is granted"):
            first = re.sub(r"^the application is granted", "the court granted the application", first, flags=re.I)
        elif first_norm.startswith("the application was granted"):
            first = re.sub(r"^the application was granted", "the court granted the application", first, flags=re.I)
        elif first_norm.startswith("the application is dismissed"):
            first = re.sub(r"^the application is dismissed", "the court dismissed the application", first, flags=re.I)
        elif first_norm.startswith("application was dismissed"):
            first = re.sub(r"^application was dismissed", "the court dismissed the application", first, flags=re.I)
        elif first_norm.startswith("application is dismissed"):
            first = re.sub(r"^application is dismissed", "the court dismissed the application", first, flags=re.I)
        elif "by which the application was dismissed" in first_norm:
            first = "the court dismissed the application"
        elif first_norm.startswith("permission to appeal is therefore refused"):
            first = re.sub(
                r"^permission to appeal is therefore refused",
                "the court refused the application for permission to appeal",
                first,
                flags=re.I,
            )
        elif first_norm.startswith("permission to appeal is refused"):
            first = re.sub(r"^permission to appeal is refused", "the court refused permission to appeal", first, flags=re.I)
        elif first_norm.startswith("the defendant's application for permission to appeal is refused"):
            first = re.sub(
                r"^the defendant's application for permission to appeal is refused",
                "the court refused the defendant's application for permission to appeal",
                first,
                flags=re.I,
            )
        elif first_norm.startswith("the defendant's application for permission to appeal was refused"):
            first = re.sub(
                r"^the defendant's application for permission to appeal was refused",
                "the court refused the defendant's application for permission to appeal",
                first,
                flags=re.I,
            )
        elif first_norm.startswith("the no costs application was rejected"):
            first = re.sub(
                r"^the no costs application was rejected",
                "the court rejected the No Costs Application",
                first,
                flags=re.I,
            )

        rendered_parts = [first.rstrip(".") + "."]
        for tail in sentences[1:]:
            tail_clean = tail.rstrip(".")
            tail_norm = normalized_text(tail_clean)
            if "no order as to costs" in tail_norm:
                rendered_parts.append("No order as to costs was made.")
                continue
            if "no costs ordered" in tail_norm or "no costs order" in tail_norm:
                rendered_parts.append("No order as to costs was made.")
                continue
            own_costs_match = re.search(
                r"the\s+([a-z]+)\s+(?:shall|must)\s+bear\s+its\s+own\s+costs(?:\s+of\s+the\s+application)?",
                tail_clean,
                re.I,
            )
            if own_costs_match:
                role = own_costs_match.group(1).capitalize()
                rendered_parts.append(f"The {role} was ordered to bear its own costs of the application.")
                continue
            match = re.search(
                r"the\s+([a-z]+)\s+should\s+pay\s+the\s+([a-z]+)'s\s+costs\s+of\s+the\s+application.*?to\s+be\s+assessed",
                tail_clean,
                re.I,
            )
            if match:
                payer = match.group(1).capitalize()
                payee = match.group(2).capitalize()
                rendered_parts.append(f"The {payer} must pay the {payee}'s costs of the application, to be assessed.")
                continue
            match = re.search(
                r"the\s+([a-z]+)\s+(?:shall|must)\s+pay\s+the\s+([a-z]+)'s\s+costs\s+of\s+the\s+(.+?)\s+on\s+the\s+standard\s+basis",
                tail_clean,
                re.I,
            )
            if match:
                payer = match.group(1).capitalize()
                payee = match.group(2).capitalize()
                subject = normalize_space(match.group(3))
                sentence = f"The {payer} must pay the {payee}'s costs of the {subject} on the standard basis"
                day_match = re.search(r"within\s+(\d+)\s+(?:working\s+)?days?", tail_clean, re.I)
                if "statement of costs" in tail_norm and day_match:
                    sentence += f", and the {payee} must file a Statement of Costs within {day_match.group(1)} days"
                rendered_parts.append(sentence.rstrip(".") + ".")
                continue
            match = re.search(
                r"the\s+([a-z]+)\s+shall\s+pay\s+the\s+([a-z]+)\s+its\s+costs\s+of\s+the\s+(.+?)\s+on\s+the\s+standard\s+basis",
                tail_clean,
                re.I,
            )
            if match:
                payer = match.group(1).capitalize()
                payee = match.group(2).capitalize()
                subject = normalize_space(match.group(3))
                sentence = f"The {payer} must pay the {payee}'s costs of the {subject} on the standard basis"
                day_match = re.search(r"within\s+(\d+)\s+(?:working\s+)?days?", tail_clean, re.I)
                if "statement of costs" in tail_norm and day_match:
                    sentence += f", and the {payee} must file a Statement of Costs within {day_match.group(1)} days"
                rendered_parts.append(sentence.rstrip(".") + ".")
                continue
            if "costs are awarded on the standard basis to be assessed by way of parties' submissions" in tail_norm:
                rendered_parts.append("Costs were awarded on the standard basis, to be assessed by way of parties' submissions.")
                continue
            if "statement of costs" in tail_norm and "within" in tail_norm:
                day_match = re.search(r"within\s+(\d+)\s+(?:working\s+)?days?", tail_clean, re.I)
                actor_match = re.search(r"the\s+([a-z]+)\s+(?:shall|must)\s+(?:submit|file)", tail_clean, re.I)
                actor = actor_match.group(1).capitalize() if actor_match else "Party"
                if day_match:
                    rendered_parts.append(f"The {actor} must file a Statement of Costs within {day_match.group(1)} days.")
                else:
                    rendered_parts.append(f"The {actor} must file a Statement of Costs.")
                continue
            if "request for aed" in tail_norm and "cost" in tail_norm and "denied" in tail_norm:
                tail_clean = re.sub(r"^the defendant's request", "The defendant's request", tail_clean, flags=re.I)
                if tail_clean and tail_clean[-1] not in ".!?":
                    tail_clean += "."
                rendered_parts.append(tail_clean)
                continue
            tail_clean = normalize_party_role_sentence(tail_clean)
            if tail_clean and tail_clean[-1] not in ".!?":
                tail_clean += "."
            rendered_parts.append(tail_clean)

        answer = " ".join(normalize_space(part) for part in rendered_parts if normalize_space(part))
        answer = normalize_space(answer).rstrip(".")
        answer = re.sub(r"\bin its amended form\b", "in amended form", answer, flags=re.I)
        answer = re.sub(r"\bdismissed the appeal as it relates to\b", "dismissed the appeal against", answer, flags=re.I)
        if case_id:
            case_meta = self.case_by_id.get(case_id)
            if case_meta is not None:
                descriptor = ""
                for page in case_meta.order_pages or [case_meta.first_page]:
                    page_text = self.page_texts.get(case_meta.sha, {}).get(page, "")
                    descriptor = infer_application_descriptor(page_text)
                    if descriptor:
                        break
                if descriptor:
                    answer = enhance_order_answer_with_descriptor(answer, descriptor)
                if full_operatives_question:
                    existing_norm = normalized_text(answer)
                    supplemental_parts: List[str] = []
                    seen_parts = {existing_norm}
                    for page in case_meta.order_pages or [case_meta.first_page]:
                        page_text = self.page_texts.get(case_meta.sha, {}).get(page, "")
                        for clause in extract_order_section_list_items(page_text):
                            clause_norm = normalized_text(clause)
                            if not clause_norm or clause_norm in seen_parts:
                                continue
                            rendered_part = ""
                            if "adjournment" in clause_norm and "adjournment" not in existing_norm:
                                rendered_part = concise_disposition_clause(clause)
                            elif "set aside" in clause_norm and "set aside" not in existing_norm:
                                rendered_part = concise_disposition_clause(clause)
                            elif "asi order" in clause_norm and "asi order" not in existing_norm:
                                rendered_part = concise_disposition_clause(clause)
                            elif "trial" in clause_norm and "trial" not in existing_norm:
                                rendered_part = concise_disposition_clause(clause)
                            elif "no order as to costs" in clause_norm and "no order as to costs" not in existing_norm:
                                rendered_part = concise_costs_clause(clause)
                            if not rendered_part:
                                continue
                            seen_parts.add(clause_norm)
                            existing_norm += " " + normalized_text(rendered_part)
                            supplemental_parts.append(rendered_part)
                            if len(supplemental_parts) >= 2:
                                break
                        if len(supplemental_parts) >= 2:
                            break
                    if supplemental_parts:
                        base_answer = answer
                        if base_answer and base_answer[-1] not in ".!?":
                            base_answer += "."
                        answer = normalize_space(" ".join([base_answer, *supplemental_parts]))
        if case_id:
            answer = re.sub(
                rf"^(?:According to|On)\s+the\s+last\s+page\s+of\s+{re.escape(case_id)}\s*,\s*",
                "",
                answer,
                flags=re.I,
            )
            answer = re.sub(
                rf"^(?:According to|On)\s+the\s+first\s+page\s+of\s+{re.escape(case_id)}\s*,\s*",
                "",
                answer,
                flags=re.I,
            )
            answer = re.sub(
                rf"^In\s+the\s+Conclusion\s+section\s+of\s+{re.escape(case_id)}\s*,\s*",
                "",
                answer,
                flags=re.I,
            )
            answer = re.sub(
                rf"^In\s+the\s+(?:'IT IS HEREBY ORDERED THAT'|order)\s+section\s+of\s+{re.escape(case_id)}\s*,\s*",
                "",
                answer,
                flags=re.I,
            )
            answer = re.sub(rf"\s+in\s+case\s+{re.escape(case_id)}\b", "", answer, flags=re.I)
            lowered = answer[0].lower() + answer[1:] if answer.startswith("The ") else answer
            if "last page" in question_norm and len(answer) <= 150 and not answer.lower().startswith("according to the last page"):
                answer = f"According to the last page of {case_id}, {lowered}"
            elif "first page" in question_norm and len(answer) <= 180 and not answer.lower().startswith("according to the first page"):
                answer = f"According to the first page of {case_id}, {lowered}"
            elif "conclusion" in question_norm and len(answer) <= 260 and "conclusion section" not in answer.lower():
                answer = f"In the Conclusion section of {case_id}, {lowered}"
            elif ("it is hereby ordered" in question_norm or "ordered that" in question_norm) and len(answer) <= 280 and "order section" not in answer.lower():
                answer = f"In the order section of {case_id}, {lowered}"
            elif case_id.lower() not in answer.lower():
                answer = f"In {case_id}, {lowered}"
        if case_id and "last page" in question_norm:
            answer = re.sub(
                rf"^According to the last page of {re.escape(case_id)},\s*",
                "On the last page, ",
                answer,
                flags=re.I,
            )
            answer = re.sub(
                rf"^In\s+{re.escape(case_id)},\s*On the last page,\s*",
                "On the last page, ",
                answer,
                flags=re.I,
            )
            answer = re.sub(r"^(?:On the last page,\s*){2,}", "On the last page, ", answer, flags=re.I)
        if case_id and "first page" in question_norm:
            answer = re.sub(
                rf"^According to the first page of {re.escape(case_id)},\s*",
                "On the first page, ",
                answer,
                flags=re.I,
            )
            answer = re.sub(
                rf"^In\s+{re.escape(case_id)},\s*On the first page,\s*",
                "On the first page, ",
                answer,
                flags=re.I,
            )
            answer = re.sub(r"^(?:On the first page,\s*){2,}", "On the first page, ", answer, flags=re.I)
        if case_id and ("it is hereby ordered" in question_norm or "ordered that" in question_norm):
            answer = re.sub(
                rf"^In the order section of {re.escape(case_id)},\s*",
                "",
                answer,
                flags=re.I,
            )
            if case_id.lower() not in answer.lower() and len(answer) <= 120:
                lowered = answer[0].lower() + answer[1:] if answer.startswith("The ") else answer
                answer = f"In {case_id}, {lowered}"
        if case_id and "conclusion" in question_norm:
            answer = re.sub(
                rf"^In the Conclusion section of {re.escape(case_id)},\s*",
                "",
                answer,
                flags=re.I,
            )
        if case_id and ("last page" in question_norm or "first page" in question_norm):
            answer_norm = normalized_text(answer)
            sentence_count = len([part for part in re.split(r"(?<=[.!?])\s+", answer) if normalize_space(part)])
            has_secondary_limb = any(
                marker in answer_norm
                for marker in (
                    "cost",
                    "statement of costs",
                    "adjournment",
                    "set aside",
                    "asi order",
                    "trial",
                    "interest",
                    "within 14 days",
                )
            )
            scoped_page = None
            if "last page" in question_norm and case_meta is not None:
                scoped_page = self.corpus.documents[case_meta.sha].page_count
            elif "first page" in question_norm:
                scoped_page = 1
            if (
                sentence_count == 1
                and not has_secondary_limb
                and scoped_page is not None
                and case_meta is not None
                and "no further order is stated on that page" not in answer_norm
            ):
                page_text = self.page_texts.get(case_meta.sha, {}).get(scoped_page, "")
                page_norm = normalized_text(page_text)
                other_order_pages_have_operatives = False
                for page in case_meta.order_pages:
                    if int(page) == scoped_page:
                        continue
                    other_text = self.page_texts.get(case_meta.sha, {}).get(int(page), "")
                    for clause in extract_order_section_list_items(other_text):
                        if is_cost_related_order_clause(clause) or is_primary_outcome_order_clause(clause):
                            other_order_pages_have_operatives = True
                            break
                    if other_order_pages_have_operatives:
                        break
                if not any(
                    marker in page_norm
                    for marker in (
                        "no order as to costs",
                        "bear its own costs",
                        "shall pay",
                        "must pay",
                        "costs are awarded",
                        "statement of costs",
                        "set aside",
                        "adjournment",
                        "proceed to trial",
                        "interest shall accrue",
                    )
                ) and not other_order_pages_have_operatives:
                    answer = answer.rstrip(".") + "; no further order is stated on that page."
        answer = normalize_space(answer)
        answer = re.sub(
            r"^(On the (?:first|last) page, .+?)\.\s+No order as to costs was made\.$",
            r"\1 and made no order as to costs.",
            answer,
            flags=re.I,
        )
        if answer:
            answer = answer[0].upper() + answer[1:]
        if answer and answer[-1] not in ".!?":
            answer += "."
        return answer

    def _render_law_metadata_free_text(
        self,
        raw_answer: str,
        route: Dict[str, Any],
        analysis: QuestionAnalysis,
    ) -> str:
        answer = normalize_space(raw_answer)
        question_norm = normalized_text(analysis.standalone_question or route.get("question", ""))
        explicit_article_scope = bool(analysis.target_article_refs) or "article" in question_norm or "section" in question_norm
        law = self._candidate_law(analysis.target_law_ids, route, analysis.target_titles)
        subject = law.title if law is not None else normalize_space((analysis.target_titles or ["This Law"])[0])
        subject = answer_source_label(subject)
        question_subject = question_subject_label(route.get("question", ""))
        if question_subject and len(question_subject) > len(subject):
            subject = question_subject
        subject = normalize_space(subject)
        if not subject:
            return answer
        basis = self._law_basis_label(law, analysis.target_field)
        short_groundable_answer = len(answer) < 90 and bool(basis)
        basis_preferred = bool(basis and explicit_article_scope)
        if analysis.target_field == "administered_by":
            if re.search(r"administered by the\s+.+?[\.;]?$", answer, re.I):
                match = re.search(r"administered by the\s+(.+?)[\.;]?$", answer, re.I)
                if match:
                    entity = normalize_space(match.group(1))
                    if basis_preferred:
                        if "regulations made under it" in normalized_text(answer):
                            return f"Under {basis} of the {subject}, the {entity} administers the Law and any Regulations made under it."
                        return f"Under {basis} of the {subject}, the {entity} administers the Law."
                    if "regulations made under it" in normalized_text(answer):
                        return f"The {subject} and any Regulations made under it are administered by the {entity}."
                    return f"The {subject} is administered by the {entity}."
        if analysis.target_field == "made_by" and re.search(r"made by the\s+.+?[\.;]?$", answer, re.I):
            match = re.search(r"made by the\s+(.+?)[\.;]?$", answer, re.I)
            if match:
                entity = normalize_space(match.group(1))
                if basis and (explicit_article_scope or short_groundable_answer):
                    return f"Under {basis} of the {subject}, the {entity} made the Law."
                return f"The {subject} was made by the {entity}."
        if analysis.target_field == "enacted_text" and "date specified in the enactment notice" in normalized_text(answer):
            if basis and (explicit_article_scope or short_groundable_answer):
                return f"Under {basis} of the {subject}, the Law is enacted on the date specified in the Enactment Notice; no calendar date is provided."
            return f"The {subject} is enacted on the date specified in the Enactment Notice; no calendar date is provided."
        if analysis.target_field == "administered_by":
            if re.search(r"is responsible for administering\b", answer, re.I):
                entity_match = re.match(r"^The\s+(.+?)\s+is responsible for administering\b", answer, re.I)
                if entity_match:
                    entity = normalize_space(entity_match.group(1))
                    if basis_preferred:
                        return f"Under {basis} of the {subject}, the {entity} administers the Law and any Regulations made under it."
                    return f"The {subject} and any Regulations made under it are administered by the {entity}."
            if "and any regulations made under it are administered by" in normalized_text(answer):
                match = re.search(r"administered by the\s+(.+?)[\.;]?$", answer, re.I)
                if match:
                    entity = normalize_space(match.group(1))
                    if basis_preferred:
                        return f"Under {basis} of the {subject}, the {entity} administers the Law and any Regulations made under it."
                    return f"The {subject} and any Regulations made under it are administered by the {entity}."
            if re.match(r"^The\s+.+?\s+is administered by\b", answer, re.I):
                match = re.search(r"administered by the\s+(.+?)[\.;]?$", answer, re.I)
                if match:
                    entity = normalize_space(match.group(1))
                    if basis_preferred:
                        return f"Under {basis} of the {subject}, the {entity} administers the Law."
                    return f"The {subject} is administered by the {entity}."
        if re.match(rf"^(?:The\s+)?{re.escape(subject)}\s+is administered by\b", answer, re.I):
            if basis and explicit_article_scope:
                match = re.search(r"administered by the\s+(.+?)[\.;]?$", answer, re.I)
                if match:
                    entity = normalize_space(match.group(1))
                    return f"{basis} of the {subject} provides that the Law is administered by the {entity}."
            return answer.rstrip(".") + "."
        if re.match(rf"^The\s+{re.escape(subject)}\s+and any Regulations made under it are administered by\b", answer, re.I):
            return answer.rstrip(".") + "."
        if analysis.target_field == "made_by" and re.search(r"made by the\s+.+?[\.;]?$", answer, re.I):
            match = re.search(r"made by the\s+(.+?)[\.;]?$", answer, re.I)
            if match:
                entity = normalize_space(match.group(1))
                if basis and explicit_article_scope:
                    return f"{basis} of the {subject} provides that the Law was made by the {entity}."
        if re.match(r"^This Law was made by\b", answer, re.I):
            return re.sub(r"^This Law\b", f"The {subject}", answer, flags=re.I).rstrip(".") + "."
        if re.match(r"^This Law is administered by\b", answer, re.I):
            answer = re.sub(r"^This Law\b", f"The {subject}", answer, flags=re.I)
            return answer.rstrip(".") + "."
        if re.match(r"^The consolidated version was published in\b", answer, re.I):
            return re.sub(
                r"^The consolidated version was published in\b",
                f"The consolidated version of {subject} was published in",
                answer,
                flags=re.I,
            ).rstrip(".") + "."
        if re.match(r"^This Law was enacted on\b", answer, re.I):
            rendered = re.sub(r"^This Law\b", f"The {subject}", answer, flags=re.I)
            if "date specified in the enactment notice" in normalized_text(rendered):
                return f"The {subject} was enacted on the date specified in the Enactment Notice; the specific date is not provided."
            return rendered.rstrip(".") + "."
        if analysis.target_field == "effective_dates" and subject.lower() not in answer.lower():
            return f"For {subject}, {answer[0].lower() + answer[1:]}" if answer else answer
        return answer

    def _law_basis_label(self, law: LawMetadata | None, field: str) -> str:
        if law is None:
            return ""
        page = None
        if field == "administered_by":
            page = law.administered_by_page
        elif field == "made_by":
            page = law.made_by_page
        elif field == "publication_text":
            page = law.publication_page
        elif field in {"enacted_text", "effective_dates"}:
            page = law.enacted_page or law.effective_date_page or law.commencement_page
        if not page:
            return ""
        page_text_candidates: List[str] = []
        merged_page_text = str(self.page_texts.get(law.sha, {}).get(page, "") or "")
        if merged_page_text:
            page_text_candidates.append(merged_page_text)
        payload = self.corpus.documents_payload.get(law.sha) or {}
        for page_payload in payload.get("content", {}).get("pages", []):
            if int(page_payload.get("page") or 0) == int(page):
                payload_text = str(page_payload.get("text") or "")
                if payload_text and normalized_text(payload_text) not in {normalized_text(item) for item in page_text_candidates}:
                    page_text_candidates.append(payload_text)
                break
        if not page_text_candidates:
            return ""
        heading_markers: tuple[str, ...] = ()
        if field == "administered_by":
            heading_markers = ("administration", "administer")
        elif field == "made_by":
            heading_markers = ("legislative authority",)
        elif field == "enacted_text":
            heading_markers = ("date of enactment", "commencement")
        elif field == "effective_dates":
            heading_markers = ("commencement", "date of enactment")
        all_headings: List[tuple[str, str]] = []
        for page_text in page_text_candidates:
            headings = re.findall(r"(?:^|\n)\s*#+\s*(\d+)\.\s+([^\n]+)", page_text)
            if not headings:
                headings = re.findall(r"(?:^|\n)\s*(\d+)\.\s+([^\n]+)", page_text)
            all_headings.extend(headings)
        for number, heading in all_headings:
            heading_norm = normalized_text(heading)
            if heading_markers and any(marker in heading_norm for marker in heading_markers):
                return f"Article {number}"
        if len(all_headings) == 1:
            return f"Article {all_headings[0][0]}"
        return ""

    def _render_clause_free_text(
        self,
        raw_answer: str,
        route: Dict[str, Any],
        analysis: QuestionAnalysis,
    ) -> str:
        answer = normalize_space(raw_answer)
        if analysis.needs_multi_document_support:
            if answer and answer[-1] not in ".!?":
                answer += "."
            return answer
        article_ref = (analysis.target_article_refs or [None])[0]
        law = self._candidate_law(analysis.target_law_ids, route, analysis.target_titles)
        subject = answer_source_label(law.title) if law is not None else answer_source_label(normalize_space((analysis.target_titles or [""])[0]))
        question_subject = question_subject_label(route.get("question", ""), article_ref=article_ref)
        if question_subject and len(question_subject) > len(subject):
            subject = question_subject
        if subject:
            answer = re.sub(r"^This Law\b", f"The {subject}", answer, flags=re.I)
            answer = re.sub(
                r"^The\s+law\s+",
                f"The {subject} ",
                answer,
                flags=re.I,
            )
        answer = re.sub(r"\bRemuneration\b", "remuneration", answer)
        answer = re.sub(r"\bPay Period\b", "pay period", answer)
        question_norm = normalized_text(route.get("question", ""))
        if article_ref and subject and "who made" in question_norm and any(
            marker in normalized_text(answer) for marker in ("was made by", "is made by")
        ):
            entity_match = re.search(r"made by the\s+(.+?)[\.;]?$", answer, re.I)
            if entity_match:
                entity = normalize_space(entity_match.group(1))
                return f"Under {article_ref} of {subject}, the Law was made by the {entity}."
        if article_ref and subject and any(
            marker in normalized_text(answer) for marker in ("was made by", "is made by")
        ) and article_ref.lower() not in answer.lower():
            lowered = answer[0].lower() + answer[1:] if answer and answer[0].isupper() else answer
            lowered = re.sub(rf"^the\s+{re.escape(subject.lower())}\s+", "the Law ", lowered, flags=re.I)
            lowered = re.sub(r"\bis made by\b", "was made by", lowered, flags=re.I)
            return f"Under {article_ref} of {subject}, {lowered}"
        if article_ref and subject and article_ref.lower() not in answer.lower() and subject.lower() not in answer.lower():
            lowered = answer[0].lower() + answer[1:] if answer and answer[0].isupper() else answer
            return f"Under {article_ref} of {subject}, {lowered}"
        if subject and analysis.target_field == "clause_summary" and subject.lower() not in answer.lower():
            lowered = answer[0].lower() + answer[1:] if answer and answer[0].isupper() else answer
            return f"Under {subject}, {lowered}"
        return answer

    def _extract_numeric_clause_answer(
        self,
        question: str,
        clause_text: str,
    ) -> Optional[float]:
        question_norm = normalized_text(question)
        text = normalize_space(clause_text)
        text_norm = normalized_text(text)
        preferred_units = [
            unit
            for unit in ("business days", "days", "months", "years", "hours")
            if unit in question_norm
        ]
        patterns = [
            re.compile(
                r"\b(?:[A-Za-z\-]+\s*)?\(\s*(\d+)\s*\)\s+(business days?|days?|months?|years?|hours?)\b",
                re.I,
            ),
            re.compile(r"\b(\d+)\s+(business days?|days?|months?|years?|hours?)\b", re.I),
        ]
        matches: List[tuple[int, str]] = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                number = int(match.group(1))
                unit = normalized_text(match.group(2))
                matches.append((number, unit))
        if preferred_units:
            for preferred in preferred_units:
                for number, unit in matches:
                    if preferred.rstrip("s") in unit:
                        return float(number)
            return None
        if matches:
            return float(matches[0][0])
        paren_match = re.search(r"\(\s*(\d+)\s*\)", text)
        if paren_match:
            return float(paren_match.group(1))
        bare_numbers = [
            int(value)
            for value in re.findall(r"\b(\d+)\b", text_norm)
            if int(value) < 4000
        ]
        if bare_numbers:
            return float(bare_numbers[0])
        return None

    def _extract_numeric_from_evidence_context(
        self,
        question: str,
        evidence_text: str,
        article_ref: str | None = None,
    ) -> Optional[float]:
        question_norm = normalized_text(question)
        question_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", question_norm)
            if len(token) > 3 and token not in ARTICLE_STOPWORDS
        }
        preferred_units = [
            unit
            for unit in ("business days", "days", "months", "years", "hours")
            if unit in question_norm
        ]
        article_norm = normalized_text(article_ref or "")
        best_score = -1.0
        best_value: Optional[float] = None

        raw_segments = [normalize_space(line) for line in str(evidence_text or "").splitlines() if normalize_space(line)]
        if not raw_segments:
            raw_segments = [
                normalize_space(part)
                for part in re.split(r"(?<=[.;])\s+", str(evidence_text or ""))
                if normalize_space(part)
            ]
        if "confirm" in question_norm and article_ref:
            escaped_article_ref = re.escape(normalize_space(article_ref))
            confirmed_match = re.search(
                rf"confirmed\s+pursuant\s+to\s+{escaped_article_ref}.*?within\s+[A-Za-z-]+\s*\((\d+)\)\s+(business days?|days?)",
                normalize_space(str(evidence_text or "")),
                re.I,
            )
            if confirmed_match:
                return float(confirmed_match.group(1))
        if "confirm" in question_norm and article_norm:
            for segment in raw_segments:
                segment_norm = normalized_text(segment)
                if article_norm not in segment_norm or "confirm" not in segment_norm:
                    continue
                value = self._extract_numeric_clause_answer(question, segment)
                if value is not None:
                    return value
        for segment in raw_segments:
            if not re.search(r"\d", segment):
                continue
            segment_norm = normalized_text(segment)
            overlap = sum(1 for token in question_tokens if token in segment_norm)
            score = overlap * 4.0
            if article_norm and article_norm in segment_norm:
                score += 10.0
                if f"pursuant to {article_norm}" in segment_norm or f"under {article_norm}" in segment_norm:
                    score += 10.0
            if preferred_units and any(unit.rstrip("s") in segment_norm for unit in preferred_units):
                score += 8.0
            if "within" in segment_norm or "period of" in segment_norm:
                score += 2.0
            if any(marker in question_norm for marker in ("maternity", "nursing", "childbirth")) and any(
                marker in segment_norm for marker in ("maternity", "nursing", "childbirth")
            ):
                score += 8.0
            if any(marker in question_norm for marker in ("misleading", "deceptive", "conflicting", "change its name")) and any(
                marker in segment_norm for marker in ("misleading", "deceptive", "conflicting", "change it", "change of name")
            ):
                score += 8.0
            if any(marker in question_norm for marker in ("appeal", "confirms", "pay the fine", "perform the action")) and any(
                marker in segment_norm for marker in ("confirmed pursuant to", "pay the fine", "perform the action", "written notice")
            ):
                score += 8.0
            if "confirmed" in question_norm and "confirmed" in segment_norm:
                score += 8.0
            if "appeal" in question_norm and "has not made an appeal" in segment_norm:
                score -= 10.0
            value = self._extract_numeric_clause_answer(question, segment)
            if value is None:
                continue
            if score > best_score:
                best_score = score
                best_value = value
        return best_value

    def _extract_boolean_clause_answer(
        self,
        question: str,
        clause_text: str,
        evidence_text: str,
    ) -> Optional[bool]:
        question_norm = normalized_text(question)
        clause_norm = normalized_text(clause_text)
        evidence_norm = normalized_text(evidence_text or clause_text)

        if "constitute obstruction" in question_norm or "obstruction of inspector" in question_norm:
            if any(
                marker in evidence_norm
                for marker in (
                    "failure to give or produce information",
                    "failure to provide information",
                    "failure to produce information",
                    "failure to give or produce",
                )
            ):
                return True

        if "invalid" in question_norm and "unless" in question_norm and "explicitly allows" in question_norm:
            if "void" in evidence_norm and "expressly permitted" in evidence_norm:
                return True

        if any(re.search(rf"\b{re.escape(marker)}\b", question_norm) for marker in ("valid", "effective", "conclusive")):
            if all(marker in evidence_norm for marker in ("valid", "effective", "conclusive")):
                return True
            if any(marker in evidence_norm for marker in ("void", "invalid", "not valid")):
                return False

        if "void" in question_norm and "except where expressly permitted" in question_norm:
            if "void in all circumstances" in evidence_norm or ("void" in evidence_norm and "expressly permitted" in evidence_norm):
                return True

        if "liable" in question_norm:
            if "bad faith" in question_norm and "does not apply if" in evidence_norm and "bad faith" in evidence_norm:
                return True
            if "not liable" in evidence_norm or "no liability" in evidence_norm:
                return False
            if re.search(r"\bliable\b", evidence_norm) and "not liable" not in evidence_norm:
                return True

        if any(phrase in question_norm for phrase in ("without the approval", "without obtaining consent")) and any(term in question_norm for term in ("officers", "employees", "staff")):
            if "to such officers or employees" in evidence_norm:
                return True

        if "agreement specifying" in question_norm and "body corporate" in question_norm:
            if "unless" in evidence_norm and "body corporate" in evidence_norm:
                return False

        if any(marker in question_norm for marker in ("independent legal advice", "mediation", "terminate employment")):
            if (
                any(marker in evidence_norm for marker in ("independent legal advice", "mediation"))
                and any(marker in evidence_norm for marker in ("prevents an employee from waiving", "nothing in this law prevents", "provided"))
            ):
                return True

        if any(marker in question_norm for marker in (" bound to ", "obliged to", "required to provide a reason", "required to give a reason")):
            if any(marker in evidence_norm for marker in ("shall not be bound", "not be bound", "shall not be required", "not required to")):
                return False
            if any(marker in evidence_norm for marker in ("shall be bound", "is bound", "must provide a reason", "required to provide a reason")):
                return True

        negative_permission_markers = (
            "must not",
            "may not",
            "no person must",
            "no person shall",
            "no person may",
            "shall not",
            "no order may",
            "is void",
            "are void",
        )
        positive_permission_markers = (
            " may ",
            " may be ",
            "to such officers or employees",
            "nothing in this law prevents",
            "prevents an employee from waiving",
            "is valid",
            "does not apply if",
        )

        asks_permission = any(
            marker in question_norm
            for marker in (
                " can ",
                "is a person permitted",
                "is an employee permitted",
                "is an employer permitted",
                "is the registrar permitted",
                "can an employee",
                "can an employer",
                "can the registrar",
                "can an order",
                "is an unincorporated body",
                "does this constitute",
            )
        )
        if asks_permission:
            if any(marker in evidence_norm for marker in negative_permission_markers):
                return False
            if any(marker in evidence_norm for marker in positive_permission_markers):
                return True

        if "without notice" in question_norm and "no order may" in evidence_norm:
            return False

        if "under sixteen" in question_norm and "must not employ a child" in evidence_norm:
            return False

        if "unless" in evidence_norm and "without" in question_norm and any(marker in clause_norm for marker in ("must not", "no person must")):
            return False

        return None

    def _extract_name_clause_answer(
        self,
        question: str,
        clause_text: str,
        evidence_text: str,
    ) -> Optional[str]:
        question_norm = normalized_text(question)
        text = normalize_space(evidence_text or clause_text)
        if any(
            marker in question_norm
            for marker in (
                "what is the term for the person",
                "what is the defined term for an entity",
                "what is the defined term for the person",
                "what is the term for an entity",
            )
        ):
            match = re.search(
                r"\b(?:person|entity)\b[^()]{0,160}\(\s*the\s+([A-Z][A-Za-z0-9' \-/&]+?)\s*\)",
                text,
            )
            if match:
                return normalize_space(match.group(1))
            match = re.search(r"\b([A-Z][A-Za-z0-9' \-/&]+?)\s+means\s+the\s+(?:person|entity)\b", text)
            if match:
                return normalize_space(match.group(1))
            table_match = re.search(
                r"\|\s*([A-Z][A-Za-z0-9' \-/&]+?)\s*\|\s*(?:an?\s+)?\1\s+is\s+an?\s+(?:person|entity)\b",
                text,
                re.I,
            )
            if table_match:
                return normalize_space(table_match.group(1))
        if "defined term for the law" in question_norm or "these regulations refer to" in question_norm:
            if re.search(r"\bthis\s+Law\s+is\s+the\s+[A-Z][A-Za-z0-9' \-(),/&]+?\bLaw\b", text):
                return "Law"
            if re.search(r"\bdefinition\s+of\s+['\"]Law['\"]\s+is\s+modified\s+to\b", text, re.I):
                return "Law"
            table_match = re.search(
                r"\|\s*([A-Z][A-Za-z0-9' \-/&]+?)\s*\|\s*(?:the\s+)?Law\b[^|]*\|",
                text,
                re.I,
            )
            if table_match:
                return normalize_space(table_match.group(1))
            match = re.search(r"\b([A-Z][A-Za-z0-9' \-/&]+?)\s+means\s+(?:the\s+)?[A-Z][A-Za-z0-9' \-(),/&]+?(?:[.;]|$)", text)
            if match:
                candidate = normalize_space(match.group(1))
                if normalized_text(candidate) not in {"all", "these", "such", "any", "other"}:
                    return candidate
        if "document" not in question_norm or not any(marker in question_norm for marker in ("file", "submit", "submitted", "application")):
            return None
        patterns = [
            re.compile(
                r"\bfile with the Registrar,\s+(?:a|an|the)\s+([A-Z][A-Za-z0-9' \-]+?)(?:\s+containing|\s+under|\s+pursuant|\s+for\b|[.;])"
            ),
            re.compile(
                r"\bmust file\b.*?\b(?:a|an|the)\s+([A-Z][A-Za-z0-9' \-]+?)(?:\s+containing|\s+under|\s+pursuant|\s+for\b|[.;])"
            ),
        ]
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return normalize_space(match.group(1))
        return None

    def _contract_slot_names(self, contract: TaskContract) -> set[str]:
        return {slot.name for slot in contract.slots}

    def _solve_from_contract(
        self,
        question: str,
        analysis: QuestionAnalysis,
        answer_type: str,
        route: Dict[str, Any],
        reranked: Sequence[Dict[str, Any]],
    ) -> Optional[StructuredDecision]:
        contract = build_task_contract(question, answer_type, analysis)
        if contract.confidence < 0.72:
            return None
        if contract.operation == "extract_scalar":
            return self._solve_contract_extract_scalar(contract, analysis, answer_type, route)
        if contract.operation == "extract_set":
            return self._solve_contract_extract_set(contract, analysis, answer_type, route)
        if contract.operation == "compare_entities":
            return self._solve_contract_compare_entities(contract, answer_type, route)
        if contract.operation == "locate_clause":
            return self._solve_contract_locate_clause(contract, analysis, answer_type, route, reranked)
        return None

    def _solve_contract_extract_scalar(
        self,
        contract: TaskContract,
        analysis: QuestionAnalysis,
        answer_type: str,
        route: Dict[str, Any],
    ) -> Optional[StructuredDecision]:
        slot_names = self._contract_slot_names(contract)
        case_ids = [anchor.canonical_id for anchor in contract.anchors if anchor.kind == "case" and anchor.canonical_id]
        law_ids = [anchor.canonical_id for anchor in contract.anchors if anchor.kind == "law" and anchor.canonical_id and anchor.canonical_id.startswith("Law No.")]

        if "issue_date" in slot_names and answer_type == "date":
            return self._solve_issue_date_single(answer_type, case_ids)
        if "claim_amount" in slot_names and answer_type == "number":
            return self._solve_claim_amount(answer_type, case_ids)
        if "claim_number" in slot_names and answer_type == "name":
            return self._solve_origin_claim_number(answer_type, case_ids)
        if "defendant_name" in slot_names and answer_type == "name":
            return self._solve_case_party_lists(answer_type, case_ids, role="defendant")

        law_scalar_fields = {"law_number", "made_by", "administered_by", "publication_text", "enacted_text", "effective_dates"}
        selected = next((field for field in law_scalar_fields if field in slot_names), None)
        if selected is not None:
            return self._solve_law_cover_metadata(
                answer_type,
                law_ids,
                route,
                analysis,
                field_override=selected,
            )
        return None

    def _solve_contract_extract_set(
        self,
        contract: TaskContract,
        analysis: QuestionAnalysis,
        answer_type: str,
        route: Dict[str, Any],
    ) -> Optional[StructuredDecision]:
        slot_names = self._contract_slot_names(contract)
        case_ids = [anchor.canonical_id for anchor in contract.anchors if anchor.kind == "case" and anchor.canonical_id]
        law_ids = [anchor.canonical_id for anchor in contract.anchors if anchor.kind == "law" and anchor.canonical_id and anchor.canonical_id.startswith("Law No.")]

        if "claimant_names" in slot_names and answer_type == "names":
            return self._solve_case_party_lists(answer_type, case_ids, role="claimant")
        if "amended_laws" in slot_names and answer_type in {"names", "free_text"}:
            return self._solve_amended_laws(answer_type, law_ids)
        return None

    def _solve_contract_compare_entities(
        self,
        contract: TaskContract,
        answer_type: str,
        route: Dict[str, Any],
    ) -> Optional[StructuredDecision]:
        slot_names = self._contract_slot_names(contract)
        case_ids = [anchor.canonical_id for anchor in contract.anchors if anchor.kind == "case" and anchor.canonical_id]

        if "earlier_issue_date_case" in slot_names and answer_type == "name":
            return self._solve_issue_date_compare(answer_type, case_ids)
        if "common_parties" in slot_names and answer_type == "boolean":
            return self._solve_common_parties(answer_type, case_ids, route.get("question", ""))
        if "common_judges" in slot_names and answer_type == "boolean":
            return self._solve_common_judges(answer_type, case_ids, route.get("question", ""))
        if "higher_claim_amount_case" in slot_names and answer_type == "name":
            return self._solve_higher_claim_amount_case(answer_type, case_ids)
        return None

    def _solve_contract_locate_clause(
        self,
        contract: TaskContract,
        analysis: QuestionAnalysis,
        answer_type: str,
        route: Dict[str, Any],
        reranked: Sequence[Dict[str, Any]],
    ) -> Optional[StructuredDecision]:
        slot_names = self._contract_slot_names(contract)
        if "operative_result" in slot_names:
            if answer_type == "boolean":
                return self._solve_case_order_boolean(analysis, answer_type, route)
            if answer_type == "free_text":
                return self._solve_case_order_free_text(analysis, answer_type, route)
        if "article_clause" in slot_names:
            if answer_type == "free_text":
                direct = self._solve_single_source_article_content_free_text(analysis, answer_type, route)
                if direct is not None:
                    return direct
                return self._solve_article_content_free_text(analysis, answer_type, route)
        if "clause_summary" in slot_names:
            if answer_type == "free_text":
                direct = self._solve_single_source_clause_free_text(analysis, answer_type, route, reranked)
                if direct is not None:
                    return direct
        if {"pre_existing_account_date", "new_account_date"} & slot_names:
            law_ids = [anchor.canonical_id for anchor in contract.anchors if anchor.kind == "law" and anchor.canonical_id and anchor.canonical_id.startswith("Law No.")]
            return self._solve_law_cover_metadata(
                answer_type,
                law_ids,
                route,
                analysis,
                field_override="effective_dates",
            )
        return None

    def _candidate_law(
        self,
        law_ids: Sequence[str],
        route: Dict[str, Any],
        target_titles: Sequence[str] = (),
    ) -> Optional[LawMetadata]:
        for law_id in law_ids:
            if law_id in self.law_by_id:
                return self.law_by_id[law_id]

        title_candidates = []
        for title in target_titles:
            cleaned = normalize_space(re.sub(r"#+", " ", str(title or "")))
            cleaned_norm = normalized_text(cleaned)
            if cleaned_norm:
                title_candidates.append(cleaned_norm)
        if title_candidates:
            best_meta = None
            best_score = 0.0
            for sha, document in self.corpus.documents.items():
                if document.kind not in {"law", "regulation"}:
                    continue
                aliases = [document.title, *document.aliases, *document.canonical_ids]
                alias_norms = [normalized_text(alias) for alias in aliases if normalized_text(alias)]
                score = 0.0
                for target in title_candidates:
                    target_kind = title_instrument_kind(target)
                    for alias_norm in alias_norms:
                        local_score = 0.0
                        if target == alias_norm:
                            local_score = max(local_score, 200.0)
                        elif target in alias_norm:
                            local_score = max(local_score, 140.0 + len(target))
                        elif alias_norm in target:
                            local_score = max(local_score, 120.0 + len(alias_norm))
                        else:
                            target_tokens = significant_title_match_tokens(target)
                            alias_tokens = significant_title_match_tokens(alias_norm)
                            overlap_tokens = target_tokens & alias_tokens
                            required_overlap = min(2, len(target_tokens), len(alias_tokens))
                            if required_overlap and len(overlap_tokens) >= required_overlap:
                                local_score = max(local_score, len(overlap_tokens) * 20.0 + len(" ".join(overlap_tokens)))
                        alias_kind = title_instrument_kind(alias_norm)
                        if local_score > 0 and target_kind and alias_kind:
                            if target_kind == alias_kind:
                                local_score += 12.0
                            else:
                                local_score -= 12.0
                        score = max(score, local_score)
                if score > best_score:
                    best_score = score
                    best_meta = build_law_metadata(self.corpus, sha)
            if best_meta is not None:
                return best_meta

        question_norm = normalized_text(route.get("question", ""))
        best_meta = None
        best_score = 0
        for sha, document in self.corpus.documents.items():
            document = self.corpus.documents[sha]
            if document.kind not in {"law", "regulation"}:
                continue
            score = 0
            for alias in [document.title, *document.aliases, *document.canonical_ids]:
                alias_norm = normalized_text(alias)
                if len(alias_norm) < 6:
                    continue
                if alias_norm in question_norm:
                    score = max(score, len(alias_norm))
            if document.title and normalized_text(document.title) in question_norm:
                score = max(score, len(normalized_text(document.title)) + 10)
            if score > best_score:
                best_score = score
                best_meta = build_law_metadata(self.corpus, sha)
        if best_meta is not None:
            return best_meta
        for sha in route["candidate_shas"]:
            document = self.corpus.documents[sha]
            if document.kind in {"law", "regulation"}:
                meta = build_law_metadata(self.corpus, sha)
                if meta is not None:
                    return meta
        return None

    def _candidate_consultation(
        self,
        route: Dict[str, Any],
        target_titles: Sequence[str] = (),
        law_ids: Sequence[str] = (),
    ) -> Optional[ConsultationMetadata]:
        title_candidates = [normalize_space(title) for title in target_titles if normalize_space(title)]
        all_law_ids = ordered_unique([*law_ids, *extract_law_ids(" ".join([route.get("question", ""), *title_candidates]))])
        best: Optional[ConsultationMetadata] = None
        best_score = 0.0
        question_norm = normalized_text(route.get("question", ""))
        question_cp_number = extract_consultation_paper_number(route.get("question", ""))
        title_token_sets = [consultation_match_tokens(title) for title in title_candidates if consultation_match_tokens(title)]
        question_topic_tokens = consultation_match_tokens(route.get("question", ""))
        strong_topic_tokens: set[str] = set(question_topic_tokens)
        for token_set in title_token_sets:
            strong_topic_tokens |= set(token_set)
        for sha in list(route.get("candidate_shas", [])) + list(self.consultations_by_sha):
            meta = self.consultations_by_sha.get(sha)
            if meta is None:
                continue
            score = 0.0
            title_norm = normalized_text(meta.title)
            meta_cp_number = extract_consultation_paper_number(meta.title)
            if question_cp_number and meta_cp_number and question_cp_number != meta_cp_number and (title_candidates or strong_topic_tokens):
                continue
            meta_tokens = consultation_match_tokens(meta.title)
            if meta.topic:
                meta_tokens |= consultation_match_tokens(meta.topic)
            topic_overlap = len(strong_topic_tokens & meta_tokens) if strong_topic_tokens else 0
            law_id_match = False
            target_kind = consultation_instrument_kind(" ".join(title_candidates) or route.get("question", ""))
            meta_kind = consultation_instrument_kind(" ".join([meta.title, meta.topic or ""]))
            if "consultation paper" in question_norm and title_norm in question_norm:
                score += 200.0
            if question_cp_number and meta_cp_number and question_cp_number == meta_cp_number:
                required_overlap = min(2, len(strong_topic_tokens), len(meta_tokens)) if strong_topic_tokens else 0
                if not strong_topic_tokens or (required_overlap and topic_overlap >= required_overlap):
                    score = max(score, 160.0 + topic_overlap * 8.0)
            for law_id in all_law_ids:
                if normalize_space(law_id) in meta.related_law_ids:
                    law_id_match = True
                    score = max(score, 190.0)
            for target in title_candidates:
                target_norm = normalized_text(target)
                if not target_norm:
                    continue
                if target_norm == title_norm:
                    score = max(score, 200.0)
                elif target_norm in title_norm or title_norm in target_norm:
                    score = max(score, 140.0)
            for target_tokens in title_token_sets:
                overlap = len(target_tokens & meta_tokens)
                required_overlap = min(2, len(target_tokens), len(meta_tokens))
                if len(target_tokens) <= 2:
                    required_overlap = min(1, len(target_tokens), len(meta_tokens))
                if required_overlap and overlap >= required_overlap:
                    score = max(score, 180.0 + overlap * 10.0)
            topic_norm = normalized_text(meta.topic or "")
            if topic_norm and topic_norm in question_norm:
                score = max(score, 150.0)
            if target_kind and meta_kind:
                if target_kind == meta_kind:
                    score += 12.0
                else:
                    score -= 20.0
            if strong_topic_tokens and topic_overlap == 0 and not law_id_match and score < 200.0:
                continue
            if score > best_score:
                best_score = score
                best = meta
        if best is not None and best_score > 0:
            return best
        route_consultations = [
            self.consultations_by_sha[sha]
            for sha in route.get("candidate_shas", [])
            if sha in self.consultations_by_sha
        ]
        if len(route_consultations) == 1:
            only = route_consultations[0]
            only_tokens = consultation_match_tokens(only.title)
            if only.topic:
                only_tokens |= consultation_match_tokens(only.topic)
            if any(normalize_space(law_id) in only.related_law_ids for law_id in all_law_ids):
                return only
            if not strong_topic_tokens or (strong_topic_tokens & only_tokens):
                return only
        return None

    def _solve_consultation_metadata(
        self,
        answer_type: str,
        analysis: QuestionAnalysis,
        route: Dict[str, Any],
    ) -> Optional[StructuredDecision]:
        meta = self._candidate_consultation(route, analysis.target_titles, analysis.target_law_ids)
        if meta is None:
            if answer_type in {"boolean", "number", "name", "names", "date"}:
                return StructuredDecision(True, self._deterministic_absence_payload(), [])
            return None
        field = analysis.target_field
        if field == "consultation_deadline" and answer_type == "date" and meta.deadline_date:
            pages = [1]
            if meta.deadline_page and meta.deadline_page not in pages:
                pages.append(meta.deadline_page)
            support = [chunk for page in pages for chunk in self._page_chunks(meta.sha, page)[:2]]
            return StructuredDecision(True, self._answer_payload(answer_type, meta.deadline_date, support), support)
        if field == "consultation_email" and answer_type in {"name", "free_text"} and meta.submission_email:
            pages = [1]
            if meta.email_page and meta.email_page not in pages:
                pages.append(meta.email_page)
            support = [chunk for page in pages for chunk in self._page_chunks(meta.sha, page)[:2]]
            return StructuredDecision(True, self._answer_payload(answer_type, meta.submission_email, support), support)
        if field == "consultation_topic" and answer_type == "name" and meta.topic:
            support = self._page_chunks(meta.sha, meta.topic_page)[:2]
            return StructuredDecision(True, self._answer_payload(answer_type, meta.topic, support), support)
        if field == "consultation_issuer" and answer_type == "name" and meta.issuing_body:
            pages = [1]
            if meta.issuing_body_page and meta.issuing_body_page not in pages:
                pages.append(meta.issuing_body_page)
            support = [chunk for page in pages for chunk in self._page_chunks(meta.sha, page)[:2]]
            return StructuredDecision(True, self._answer_payload(answer_type, meta.issuing_body, support), support)
        return None

    def _solve_from_analysis(
        self,
        analysis: QuestionAnalysis,
        answer_type: str,
        route: Dict[str, Any],
    ) -> Optional[StructuredDecision]:
        if not analysis.use_structured_executor:
            return None
        case_ids = analysis.target_case_ids or extract_case_ids(route.get("question", ""))
        law_ids = analysis.target_law_ids or extract_law_ids(route.get("question", ""))
        field = analysis.target_field
        if field == "earlier_issue_date_case":
            return self._solve_issue_date_compare(answer_type, case_ids)
        if field == "issue_date":
            return self._solve_issue_date_single(answer_type, case_ids)
        if field == "common_parties":
            return self._solve_common_parties(answer_type, case_ids, route.get("question", ""))
        if field == "common_judges":
            return self._solve_common_judges(answer_type, case_ids, route.get("question", ""))
        if field == "same_law_number":
            return self._solve_same_law_number(answer_type, analysis, route)
        if field in {"consultation_deadline", "consultation_email", "consultation_topic", "consultation_issuer"}:
            return self._solve_consultation_metadata(answer_type, analysis, route)
        if field == "claimant_names":
            return self._solve_case_party_lists(answer_type, case_ids, role="claimant")
        if field == "claimant_count":
            return self._solve_case_party_count(answer_type, case_ids, role="claimant")
        if field == "defendant_name":
            return self._solve_case_party_lists(answer_type, case_ids, role="defendant")
        if field == "claim_amount":
            return self._solve_claim_amount(answer_type, case_ids)
        if field == "claim_number":
            return self._solve_origin_claim_number(answer_type, case_ids)
        if field == "higher_claim_amount_case":
            return self._solve_higher_claim_amount_case(answer_type, case_ids)
        if field == "absence_check":
            appeal_decision = self._solve_case_appeal_to_cfi_boolean(answer_type, case_ids, route.get("question", ""))
            if appeal_decision is not None:
                return appeal_decision
        if field == "generic_answer":
            appeal_decision = self._solve_case_appeal_to_cfi_boolean(answer_type, case_ids, route.get("question", ""))
            if appeal_decision is not None:
                return appeal_decision
            generic_numeric = self._solve_case_generic_number(answer_type, case_ids, route.get("question", ""))
            if generic_numeric is not None:
                return generic_numeric
        if field == "case_relation":
            relation_decision = self._solve_case_relation_boolean(answer_type, case_ids)
            if relation_decision is not None:
                return relation_decision
        if field in {"law_number", "made_by", "administered_by", "publication_text", "enacted_text", "effective_dates"}:
            return self._solve_law_cover_metadata(answer_type, law_ids, route, analysis)
        if field == "amended_laws":
            return self._solve_amended_laws(answer_type, law_ids)
        return None

    def _override_from_analysis(
        self,
        analysis: QuestionAnalysis,
        answer_type: str,
        route: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        case_ids = analysis.target_case_ids or extract_case_ids(route.get("question", ""))
        law_ids = analysis.target_law_ids or extract_law_ids(route.get("question", ""))
        focus = set(analysis.support_focus)
        if analysis.target_field == "article_content" or "article_section" in focus:
            if analysis.needs_multi_document_support and len(analysis.target_titles) > 1:
                return self._override_multi_document_article_context(route, analysis)
            return self._override_article_context(answer_type, law_ids, route, analysis)
        if analysis.target_field == "order_result" or focus.intersection({"order_section", "conclusion_section", "last_page"}):
            override = self._override_case_order_context(answer_type, case_ids, focus)
            if override is not None:
                return override
        if focus.intersection({"title_page", "first_page", "issue_date_line", "party_block", "judge_block"}):
            override = self._override_case_first_page_context(case_ids, focus)
            if override is not None:
                return override
        if focus.intersection({"administration_clause", "enactment_clause", "publication_line", "title_page"}):
            override = self._override_law_focus_context(law_ids, route, analysis, focus)
            if override is not None:
                return override
        return None

    def _chunk_record(self, sha: str, page: int, chunk_id: int, text: str, score: float = 1.0) -> Dict[str, Any]:
        document = self.corpus.documents[sha]
        return {
            "ref": f"{sha}:{page}:{chunk_id}",
            "distance": round(float(score), 4),
            "page": page,
            "text": text,
            "sha": sha,
            "title": document.title,
            "kind": document.kind,
            "canonical_ids": document.canonical_ids,
        }

    def _page_chunks(self, sha: str, page_number: int) -> List[Dict[str, Any]]:
        payload = self.corpus.documents_payload[sha]
        chunks = []
        max_chunk_id = -1
        for chunk in payload["content"]["chunks"]:
            if int(chunk["page"]) == int(page_number):
                chunk_id = int(chunk["id"])
                max_chunk_id = max(max_chunk_id, chunk_id)
                chunks.append(self._chunk_record(sha, int(chunk["page"]), chunk_id, str(chunk["text"])))
        merged_page_text = self.page_texts.get(sha, {}).get(int(page_number), "")
        if chunks and normalized_text(merged_page_text) and normalized_text("\n".join(chunk["text"] for chunk in chunks)) != normalized_text(merged_page_text):
            synthetic_chunk_id = max_chunk_id + 1000 if max_chunk_id >= 0 else 1000
            chunks.append(self._chunk_record(sha, int(page_number), synthetic_chunk_id, merged_page_text, score=1.0))
        return chunks

    def _all_case_metas(self, case_id: str) -> List[CaseMetadata]:
        metas = getattr(self, "case_metas_by_id", {}).get(case_id, [])
        if metas:
            return metas
        case = getattr(self, "case_by_id", {}).get(case_id)
        return [case] if case is not None else []

    def _best_case_meta(self, case_id: str, field: str | None = None) -> Optional[CaseMetadata]:
        metas = self._all_case_metas(case_id)
        if not metas:
            return None
        if field is None:
            return max(metas, key=case_metadata_score)
        valued = [meta for meta in metas if getattr(meta, field)]
        if valued:
            return max(valued, key=case_metadata_score)
        if field == "claim_amount_aed":
            valued = [meta for meta in metas if getattr(meta, field) is not None]
            if valued:
                return max(valued, key=case_metadata_score)
        return max(metas, key=case_metadata_score)

    def _explicit_page_focus_pages(self, question: str, case: CaseMetadata) -> List[int]:
        question_norm = normalized_text(question)
        page_count = self.corpus.documents[case.sha].page_count
        pages: List[int] = []
        if "first page" in question_norm or "page 1" in question_norm:
            pages.append(1)
        if "second page" in question_norm or "page 2" in question_norm:
            pages.append(2)
        if "last page" in question_norm:
            pages.append(page_count)
        return [page for page in dict.fromkeys(pages) if 1 <= page <= page_count]

    def _candidate_case_order_pages(self, case: CaseMetadata, focus: set[str]) -> List[int]:
        payload = self.corpus.documents_payload[case.sha]
        pages_payload = payload["content"]["pages"]
        page_count = self.corpus.documents[case.sha].page_count
        candidate_pages: List[int] = list(case.order_pages)
        if "last_page" in focus:
            candidate_pages.append(page_count)
        for page in case.order_pages:
            for offset in (1, 2):
                if page + offset <= page_count:
                    candidate_pages.append(page + offset)
        if case.order_pages and min(case.order_pages) <= 2:
            candidate_pages.extend(range(1, min(page_count, 4) + 1))
        order_only = "order_section" in focus and "conclusion_section" not in focus
        conclusion_only = "conclusion_section" in focus and "order_section" not in focus
        order_page_cap = min(case.order_pages) + 1 if case.order_pages else 2

        for index, page in enumerate(pages_payload):
            page_number = int(page["page"])
            if order_only and page_number > order_page_cap and "last_page" not in focus:
                continue
            raw_text = self.page_texts.get(case.sha, {}).get(page_number, str(page["text"] or ""))
            text_norm = normalized_text(raw_text)
            has_order_marker = "it is hereby ordered that" in text_norm or "result of the application" in text_norm
            has_conclusion_marker = "## conclusion" in text_norm or re.search(r"(?:^|\n)#+\s*conclusion\b", raw_text, re.I) is not None
            if order_only:
                has_strong_marker = has_order_marker
            elif conclusion_only:
                has_strong_marker = has_conclusion_marker
            else:
                has_strong_marker = has_order_marker or has_conclusion_marker
            has_operative_clause = any(
                is_operative_order_clause(clean_clause_text(normalize_space(line)))
                for line in raw_text.splitlines()
                if normalize_space(line)
            )
            if has_strong_marker or has_operative_clause:
                candidate_pages.append(page_number)
                continue
            header_only = "order with reasons" in text_norm and not has_strong_marker and not has_operative_clause
            if header_only and index + 1 < len(pages_payload):
                next_page = pages_payload[index + 1]
                next_raw = str(next_page["text"] or "")
                next_text_norm = normalized_text(next_raw)
                next_has_operative_clause = any(
                    is_operative_order_clause(clean_clause_text(normalize_space(line)))
                    for line in next_raw.splitlines()
                    if normalize_space(line)
                )
                next_has_order_marker = "it is hereby ordered that" in next_text_norm or "result of the application" in next_text_norm
                next_has_conclusion_marker = "## conclusion" in next_text_norm or re.search(r"(?:^|\n)#+\s*conclusion\b", next_raw, re.I) is not None
                if order_only:
                    next_has_strong_marker = next_has_order_marker
                elif conclusion_only:
                    next_has_strong_marker = next_has_conclusion_marker
                else:
                    next_has_strong_marker = next_has_order_marker or next_has_conclusion_marker
                if next_has_strong_marker or next_has_operative_clause:
                    candidate_pages.append(int(next_page["page"]))
        filtered = [page for page in candidate_pages if 1 <= page <= page_count]
        if order_only and "last_page" not in focus:
            filtered = [page for page in filtered if page <= order_page_cap]
        return list(dict.fromkeys(filtered))

    def _answer_payload(self, answer_type: str, answer: Any, support_chunks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if answer_type == "boolean":
            raw_answer = "true" if answer else "false"
        elif answer_type == "names":
            raw_answer = " | ".join(answer)
        else:
            raw_answer = str(answer)
        return {
            "raw_answer": raw_answer,
            "normalized_answer": normalize_answer(answer_type, raw_answer),
            "citations": [chunk["ref"] for chunk in support_chunks],
            "reasoning": "Structured solver",
            "confidence": "high",
            "response_data": {},
        }

    def _deterministic_absence_payload(self) -> Dict[str, Any]:
        return {
            "raw_answer": FREE_TEXT_ABSENCE_ANSWER,
            "normalized_answer": None,
            "citations": [],
            "reasoning": "Requested document or clause is not present in the corpus.",
            "confidence": "high",
            "response_data": {},
        }

    def _reranked_single_document_support(
        self,
        reranked: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        lawish = [
            chunk
            for chunk in reranked
            if chunk.get("kind") in {"law", "regulation"} and chunk.get("sha")
        ]
        if not lawish:
            return []
        counts: Dict[str, int] = defaultdict(int)
        for chunk in lawish:
            counts[str(chunk["sha"])] += 1
        best_sha, _ = max(counts.items(), key=lambda item: item[1])
        support = [chunk for chunk in lawish if str(chunk["sha"]) == best_sha]
        return self._representative_support_chunks(support, per_doc=3)

    def _solve_issue_date_single(self, answer_type: str, case_ids: Sequence[str]) -> Optional[StructuredDecision]:
        if answer_type != "date" or len(case_ids) != 1:
            return None
        case = self._best_case_meta(case_ids[0], "issue_date")
        if case is None or case.issue_date is None:
            return None
        support = self._page_chunks(case.sha, case.issue_page or case.first_page)
        return StructuredDecision(True, self._answer_payload(answer_type, case.issue_date, support), support)

    def _solve_issue_date_compare(self, answer_type: str, case_ids: Sequence[str]) -> Optional[StructuredDecision]:
        if answer_type != "name" or len(case_ids) != 2:
            return None
        left = self._best_case_meta(case_ids[0], "issue_date")
        right = self._best_case_meta(case_ids[1], "issue_date")
        if not left or not right or not left.issue_date or not right.issue_date:
            return None
        answer = left.case_id if left.issue_date <= right.issue_date else right.case_id
        support = self._page_chunks(left.sha, left.issue_page or 1)[:1] + self._page_chunks(right.sha, right.issue_page or 1)[:1]
        return StructuredDecision(True, self._answer_payload(answer_type, answer, support), support)

    def _minimal_overlap_support(
        self,
        left_cases: Sequence[CaseMetadata],
        right_cases: Sequence[CaseMetadata],
        shared_keys: Sequence[str] | set[str],
        values_getter,
    ) -> List[Dict[str, Any]]:
        for shared_key in sorted(shared_keys):
            left_case = next(
                (
                    case
                    for case in left_cases
                    if shared_key in {canonical_person_name(name) for name in values_getter(case)}
                ),
                None,
            )
            right_case = next(
                (
                    case
                    for case in right_cases
                    if shared_key in {canonical_person_name(name) for name in values_getter(case)}
                ),
                None,
            )
            if left_case and right_case:
                return self._page_chunks(left_case.sha, left_case.first_page)[:1] + self._page_chunks(right_case.sha, right_case.first_page)[:1]
        return [
            chunk
            for case in [*left_cases, *right_cases]
            for chunk in self._page_chunks(case.sha, case.first_page)[:1]
        ]

    def _solve_common_parties(self, answer_type: str, case_ids: Sequence[str], question: str = "") -> Optional[StructuredDecision]:
        if answer_type != "boolean" or len(case_ids) != 2:
            return None
        question_norm = normalized_text(question)
        full_case_scope = any(marker in question_norm for marker in ("all documents", "full case files", "full case file"))
        left_cases = self._all_case_metas(case_ids[0]) if full_case_scope else ([self.case_by_id.get(case_ids[0])] if self.case_by_id.get(case_ids[0]) else [])
        right_cases = self._all_case_metas(case_ids[1]) if full_case_scope else ([self.case_by_id.get(case_ids[1])] if self.case_by_id.get(case_ids[1]) else [])
        if not left_cases or not right_cases:
            return None
        left_parties = {
            canonical_person_name(name)
            for case in left_cases
            for name in case.all_main_parties
        }
        right_parties = {
            canonical_person_name(name)
            for case in right_cases
            for name in case.all_main_parties
        }
        shared_parties = left_parties & right_parties
        answer = bool(shared_parties)
        if answer:
            support = self._minimal_overlap_support(left_cases, right_cases, shared_parties, lambda case: case.all_main_parties)
        else:
            support = [
                chunk
                for case in [*left_cases, *right_cases]
                for chunk in self._page_chunks(case.sha, case.first_page)[:1]
            ]
        return StructuredDecision(True, self._answer_payload(answer_type, answer, support), support)

    def _solve_common_judges(self, answer_type: str, case_ids: Sequence[str], question: str = "") -> Optional[StructuredDecision]:
        if answer_type != "boolean" or len(case_ids) != 2:
            return None
        question_norm = normalized_text(question)
        full_case_scope = any(marker in question_norm for marker in ("all documents", "full case files", "full case file"))
        left_cases = self._all_case_metas(case_ids[0]) if full_case_scope else ([self.case_by_id.get(case_ids[0])] if self.case_by_id.get(case_ids[0]) else [])
        right_cases = self._all_case_metas(case_ids[1]) if full_case_scope else ([self.case_by_id.get(case_ids[1])] if self.case_by_id.get(case_ids[1]) else [])
        if not left_cases or not right_cases:
            return None
        left_judges = {
            canonical_person_name(name)
            for case in left_cases
            for name in case.judges
        }
        right_judges = {
            canonical_person_name(name)
            for case in right_cases
            for name in case.judges
        }
        if not left_judges or not right_judges:
            return None
        shared_judges = left_judges & right_judges
        answer = bool(shared_judges)
        if answer:
            support = self._minimal_overlap_support(left_cases, right_cases, shared_judges, lambda case: case.judges)
        else:
            support = [
                chunk
                for case in [*left_cases, *right_cases]
                for chunk in self._page_chunks(case.sha, case.first_page)[:1]
            ]
        return StructuredDecision(True, self._answer_payload(answer_type, answer, support), support)

    def _solve_case_party_lists(self, answer_type: str, case_ids: Sequence[str], role: str) -> Optional[StructuredDecision]:
        if len(case_ids) != 1:
            return None
        cases = self._all_case_metas(case_ids[0])
        if not cases:
            return None
        names: List[str] = []
        support: List[Dict[str, Any]] = []
        seen_names = set()
        for case in cases:
            role_names = case.claimant_names if role == "claimant" else case.defendant_names
            for name in role_names:
                key = canonical_person_name(name)
                if not key or key in seen_names:
                    continue
                seen_names.add(key)
                names.append(name)
            if role_names and len(support) < 2:
                support.extend(self._page_chunks(case.sha, 1)[:1])
        if answer_type == "names" and names:
            return StructuredDecision(True, self._answer_payload(answer_type, names, support or self._page_chunks(cases[0].sha, 1)[:1]), support or self._page_chunks(cases[0].sha, 1)[:1])
        if answer_type == "name" and role == "defendant" and names:
            support_rows = support or self._page_chunks(cases[0].sha, 1)[:1]
            return StructuredDecision(True, self._answer_payload(answer_type, names[0], support_rows), support_rows)
        return None

    def _solve_case_party_count(self, answer_type: str, case_ids: Sequence[str], role: str) -> Optional[StructuredDecision]:
        if answer_type != "number" or len(case_ids) != 1:
            return None
        cases = self._all_case_metas(case_ids[0])
        if not cases:
            return None
        distinct = {
            canonical_person_name(name)
            for case in cases
            for name in (case.claimant_names if role == "claimant" else case.defendant_names)
            if canonical_person_name(name)
        }
        if not distinct:
            return None
        support = []
        for case in cases:
            rows = self._page_chunks(case.sha, case.first_page)[:1]
            if rows:
                support.extend(rows)
            if len(support) >= 2:
                break
        return StructuredDecision(True, self._answer_payload(answer_type, float(len(distinct)), support), support)

    def _solve_claim_amount(self, answer_type: str, case_ids: Sequence[str]) -> Optional[StructuredDecision]:
        if answer_type != "number" or len(case_ids) != 1:
            return None
        case = self._best_case_meta(case_ids[0], "claim_amount_aed")
        if not case or case.claim_amount_aed is None:
            return None
        support_page = case.claim_amount_page or 1
        support = self._page_chunks(case.sha, support_page)[:2]
        return StructuredDecision(True, self._answer_payload(answer_type, case.claim_amount_aed, support), support)

    def _solve_origin_claim_number(self, answer_type: str, case_ids: Sequence[str]) -> Optional[StructuredDecision]:
        if answer_type != "name" or len(case_ids) != 1:
            return None
        case = self._best_case_meta(case_ids[0], "origin_claim_number")
        if not case or not case.origin_claim_number:
            return None
        support_page = case.origin_claim_page or case.first_page
        support = self._page_chunks(case.sha, support_page)[:2]
        return StructuredDecision(True, self._answer_payload(answer_type, case.origin_claim_number, support), support)

    def _solve_higher_claim_amount_case(self, answer_type: str, case_ids: Sequence[str]) -> Optional[StructuredDecision]:
        if answer_type != "name" or len(case_ids) != 2:
            return None
        left = self._best_case_meta(case_ids[0], "claim_amount_aed")
        right = self._best_case_meta(case_ids[1], "claim_amount_aed")
        if not left or not right or left.claim_amount_aed is None or right.claim_amount_aed is None:
            return None
        answer = left.case_id if float(left.claim_amount_aed) >= float(right.claim_amount_aed) else right.case_id
        support = self._page_chunks(left.sha, left.claim_amount_page or left.first_page)[:1] + self._page_chunks(right.sha, right.claim_amount_page or right.first_page)[:1]
        return StructuredDecision(True, self._answer_payload(answer_type, answer, support), support)

    def _solve_case_appeal_to_cfi_boolean(
        self,
        answer_type: str,
        case_ids: Sequence[str],
        question: str,
    ) -> Optional[StructuredDecision]:
        if answer_type != "boolean" or len(case_ids) != 1:
            return None
        question_norm = normalized_text(question)
        if not any(marker in question_norm for marker in ("appealed to the cfi", "appealed to cfi", "appealed to the court of first instance", "appeal to the cfi", "appeal to the court of first instance")):
            return None
        target_case_id = case_ids[0]
        matching_support: List[Dict[str, Any]] = []
        base_support: List[Dict[str, Any]] = []
        for sha, document in self.corpus.documents.items():
            if document.kind != "case":
                continue
            first_page_text = self.page_texts.get(sha, {}).get(1, document.first_page_text)
            page_case_ids = set(extract_case_ids(first_page_text)) | set(document.canonical_ids)
            if target_case_id not in page_case_ids:
                continue
            page_support = self._page_chunks(sha, 1)[:2]
            if page_support and not base_support:
                base_support = page_support
            page_norm = normalized_text(first_page_text)
            if "court of first instance" in page_norm and any(marker in page_norm for marker in ("appeal notice", "permission to appeal", "appeal")):
                matching_support = page_support
                break
        if matching_support:
            return StructuredDecision(True, self._answer_payload(answer_type, True, matching_support), matching_support)
        if base_support:
            return StructuredDecision(True, self._answer_payload(answer_type, False, base_support), base_support)
        return None

    def _best_enforcement_subject_amount(self, case_id: str) -> Optional[tuple[float, List[Dict[str, Any]]]]:
        best: Optional[tuple[float, int, str, int]] = None
        patterns = [
            (re.compile(r"\bentered\s+judgment(?:[^.\n]{0,160})?\bamount of A[DE]D?\s*([0-9,]+(?:\.[0-9]+)?)", re.I), 140),
            (re.compile(r"\bjudgment debt(?:[^.\n]{0,80})?\bamount of A[DE]D?\s*([0-9,]+(?:\.[0-9]+)?)", re.I), 130),
            (re.compile(r"\bjudgment ordering .*? to pay .*? amount of A[DE]D?\s*([0-9,]+(?:\.[0-9]+)?)", re.I), 125),
            (re.compile(r"\bclaimed that an amount of A[DE]D?\s*([0-9,]+(?:\.[0-9]+)?)\s+was owed\b", re.I), 85),
            (re.compile(r"\bSpecial Committee concluded that an amount of A[DE]D?\s*([0-9,]+(?:\.[0-9]+)?)\s+should be paid\b", re.I), 80),
        ]
        for meta in self._all_case_metas(case_id):
            payload = self.corpus.documents_payload[meta.sha]
            for page in payload["content"]["pages"][:25]:
                page_number = int(page["page"])
                text = self.page_texts.get(meta.sha, {}).get(page_number, str(page.get("text") or ""))
                for pattern, score in patterns:
                    match = pattern.search(text)
                    if not match:
                        continue
                    try:
                        amount = float(match.group(1).replace(",", ""))
                    except ValueError:
                        continue
                    if best is None or score > best[1]:
                        best = (amount, score, meta.sha, page_number)
        if best is None:
            return None
        amount, _score, sha, page_number = best
        support = self._page_chunks(sha, page_number)[:2]
        return amount, support

    def _solve_case_generic_number(
        self,
        answer_type: str,
        case_ids: Sequence[str],
        question: str,
    ) -> Optional[StructuredDecision]:
        if answer_type != "number" or len(case_ids) != 1:
            return None
        question_norm = normalized_text(question)
        target_case_id = case_ids[0]
        if any(marker in question_norm for marker in ("subject of the enforcement proceedings", "enforcement proceedings")):
            enforcement_amount = self._best_enforcement_subject_amount(target_case_id)
            if enforcement_amount is not None:
                amount, support = enforcement_amount
                return StructuredDecision(True, self._answer_payload(answer_type, amount, support), support)
        if any(
            marker in question_norm
            for marker in (
                "claimant request",
                "claimant requested",
                "claimant claim",
                "in the claim form",
                "amount was awarded",
                "amount awarded",
                "how much money",
                "what amount",
                "what sum",
            )
        ):
            amount_decision = self._solve_claim_amount(answer_type, [target_case_id])
            if amount_decision is not None:
                return amount_decision
        return None

    def _solve_case_relation_boolean(
        self,
        answer_type: str,
        case_ids: Sequence[str],
    ) -> Optional[StructuredDecision]:
        if answer_type != "boolean" or len(case_ids) != 2:
            return None
        left_id, right_id = case_ids
        right_key = claim_number_key(right_id)
        left_cases = self._all_case_metas(left_id)
        right_cases = self._all_case_metas(right_id)
        if not left_cases or not right_cases:
            return None
        positive_support: List[Dict[str, Any]] = []
        for case in left_cases:
            for page in range(1, min(self.corpus.documents[case.sha].page_count, 3) + 1):
                text = self.page_texts.get(case.sha, {}).get(page, "")
                text_keys = {claim_number_key(value) for value in extract_case_ids(text)}
                if right_key in text_keys:
                    positive_support = self._page_chunks(case.sha, page)[:2]
                    break
            if positive_support:
                break
        if positive_support:
            return StructuredDecision(True, self._answer_payload(answer_type, True, positive_support), positive_support)
        support = self._page_chunks(left_cases[0].sha, 1)[:1] + self._page_chunks(right_cases[0].sha, 1)[:1]
        return StructuredDecision(True, self._answer_payload(answer_type, False, support), support)

    def _solve_same_law_number(
        self,
        answer_type: str,
        analysis: QuestionAnalysis,
        route: Dict[str, Any],
    ) -> Optional[StructuredDecision]:
        if answer_type != "boolean" or len(analysis.target_titles) < 2:
            return None
        extracted: List[tuple[int, List[Dict[str, Any]]]] = []
        for title in analysis.target_titles:
            law = self._candidate_law([], route, [title])
            if law is None:
                continue
            support = self._page_chunks(law.sha, law.page_one)[:2]
            value: Optional[int] = law.law_number
            if value is None:
                first_page_text = self.page_texts.get(law.sha, {}).get(law.page_one, "")
                match = re.search(r"(?:'Law'|Law)\s+(?:means|the)\s+[^.\n|]*?DIFC\s+Law\s+No\.?\s*(\d+)\s+of\s+\d{4}", first_page_text, re.I)
                if match:
                    value = int(match.group(1))
            if value is None:
                continue
            extracted.append((value, support))
        if len(extracted) < 2:
            return None
        answer = len({value for value, _ in extracted}) == 1
        support = []
        for _value, rows in extracted[:2]:
            support.extend(rows[:1])
        return StructuredDecision(True, self._answer_payload(answer_type, answer, support), support)

    def _solve_law_cover_metadata(
        self,
        answer_type: str,
        law_ids: Sequence[str],
        route: Dict[str, Any],
        analysis: QuestionAnalysis,
        field_override: Optional[str] = None,
    ) -> Optional[StructuredDecision]:
        law = self._candidate_law(law_ids, route, analysis.target_titles)
        if law is None:
            return None
        support = self._page_chunks(law.sha, law.page_one)[:2]
        field = field_override or analysis.target_field
        question_norm = normalized_text(route.get("question", ""))
        def article_label(page_number: int | None, heading_markers: Sequence[str]) -> str:
            if not page_number:
                return ""
            payload = self.corpus.documents_payload.get(law.sha) or {}
            pages = (payload.get("content") or {}).get("pages") or []
            if not 1 <= int(page_number) <= len(pages):
                return ""
            text = str(pages[int(page_number) - 1].get("text") or "")
            for number, heading in re.findall(r"##\s*(\d+)\.\s+([^\n]+)", text):
                heading_norm = normalized_text(heading)
                if any(marker in heading_norm for marker in heading_markers):
                    return f"Article {number} of the {answer_source_label(law.title)}"
            return ""
        if answer_type == "number" and field == "law_number":
            if "consolidated version" in question_norm and law.consolidated_version_number is not None:
                return StructuredDecision(True, self._answer_payload(answer_type, law.consolidated_version_number, support), support)
            if law.law_number is not None:
                return StructuredDecision(True, self._answer_payload(answer_type, law.law_number, support), support)
        if answer_type == "date" and field == "effective_dates" and law.effective_date:
            page_support = self._page_chunks(law.sha, law.effective_date_page or law.commencement_page or law.page_one) or support
            return StructuredDecision(True, self._answer_payload(answer_type, law.effective_date, page_support), page_support)
        if answer_type == "date" and field == "enacted_text" and law.enacted_text:
            enacted_date = iso_date_from_text(law.enacted_text)
            if enacted_date:
                page_support = self._page_chunks(law.sha, law.enacted_page or law.page_one) or support
                return StructuredDecision(True, self._answer_payload(answer_type, enacted_date, page_support), page_support)
        if (
            answer_type == "date"
            and field == "enacted_text"
            and law.enacted_text is None
            and any(marker in normalized_text(law.title) for marker in ("regulation", "regulations", "rules"))
        ):
            return StructuredDecision(True, self._deterministic_absence_payload(), [])
        if answer_type == "free_text" and field == "made_by" and law.made_by:
            page_support = self._page_chunks(law.sha, law.made_by_page or law.page_one) or support
            subject = answer_source_label(law.title)
            basis = self._law_basis_label(law, "made_by")
            if basis:
                answer = f"Under {basis} of the {subject}, the {law.made_by} made the Law."
            else:
                answer = f"The {law.made_by} made the {subject}."
            return StructuredDecision(True, self._answer_payload(answer_type, answer, page_support), page_support)
        if answer_type == "free_text" and field == "administered_by" and law.administered_by:
            page_support = self._page_chunks(law.sha, law.administered_by_page or law.page_one) or support
            title = answer_source_label(law.title)
            basis = self._law_basis_label(law, "administered_by")
            if "regulations made under it" in normalize_space(route.get("question") or "").lower():
                if basis:
                    answer = f"Under {basis} of the {title}, the {law.administered_by} administers the Law and any Regulations made under it."
                else:
                    answer = f"The {law.administered_by} administers the {title} and any Regulations made under it."
            else:
                if basis:
                    answer = f"Under {basis} of the {title}, the {law.administered_by} administers the Law."
                else:
                    answer = f"The {law.administered_by} administers the {title}."
            return StructuredDecision(True, self._answer_payload(answer_type, answer, page_support), page_support)
        if answer_type == "free_text" and field == "publication_text" and law.publication_text:
            page_support = self._page_chunks(law.sha, law.publication_page or law.page_one) or support
            subject = answer_source_label(law.title)
            answer = f"The consolidated version of the {subject} was published in {law.publication_text}."
            return StructuredDecision(True, self._answer_payload(answer_type, answer, page_support), page_support)
        if answer_type == "free_text" and field == "enacted_text" and law.enacted_text:
            page_support = self._page_chunks(law.sha, law.enacted_page or law.page_one) or support
            subject = answer_source_label(law.title)
            if "date specified in the enactment notice" in normalized_text(law.enacted_text):
                answer = f"The {subject} was enacted on the date specified in the Enactment Notice; the specific date is not provided in the evidence."
            else:
                answer = f"The {subject} was enacted on {law.enacted_text}."
            return StructuredDecision(True, self._answer_payload(answer_type, answer, page_support), page_support)
        if answer_type == "free_text" and field == "effective_dates":
            if law.effective_date:
                page_support = self._page_chunks(law.sha, law.effective_date_page or law.commencement_page or law.page_one) or support
                subject = answer_source_label(law.title)
                answer = f"The {subject} came into force on {display_date(law.effective_date)}."
                return StructuredDecision(True, self._answer_payload(answer_type, answer, page_support), page_support)
            payload = self.corpus.documents_payload[law.sha]
            for page in payload["content"]["pages"][:10]:
                text = str(page["text"] or "")
                pre_match = re.search(
                    r"Pre-existing Accounts.*?effective date is\s+([0-9]{1,2}\s+[A-Za-z]+,?\s+[0-9]{4})",
                    text,
                    re.I | re.S,
                )
                new_match = re.search(
                    r"New Accounts.*?effective date is\s+([0-9]{1,2}\s+[A-Za-z]+,?\s+[0-9]{4})",
                    text,
                    re.I | re.S,
                )
                if pre_match and new_match:
                    page_support = self._page_chunks(law.sha, int(page["page"])) or support
                    answer = (
                        f"The due diligence requirements take effect on {normalize_space(pre_match.group(1))} for Pre-existing Accounts "
                        f"and {normalize_space(new_match.group(1))} for New Accounts."
                    )
                    return StructuredDecision(True, self._answer_payload(answer_type, answer, page_support), page_support)
        return None

    def _law_topic_groups(self, analysis: QuestionAnalysis, law: LawMetadata) -> List[List[str]]:
        law_aliases = {
            normalized_text(alias)
            for alias in [law.title, law.law_id, *self.corpus.documents[law.sha].canonical_ids, *analysis.target_titles, *analysis.target_law_ids]
            if normalized_text(alias)
        }
        generic_tokens = {
            "does",
            "did",
            "do",
            "is",
            "are",
            "the",
            "law",
            "laws",
            "numbered",
            "apply",
            "applies",
            "within",
            "deal",
            "deals",
            "address",
            "addresses",
            "topic",
            "topics",
            "with",
            "of",
            "in",
            "to",
            "any",
            "there",
            "information",
            "about",
            "this",
            "that",
            "jurisdiction",
        }
        groups: List[List[str]] = []
        for term in analysis.must_support_terms:
            term_norm = normalized_text(term)
            if not term_norm:
                continue
            if any(term_norm == alias or term_norm in alias or alias in term_norm for alias in law_aliases if len(alias) >= 6):
                continue
            tokens = [
                token
                for token in re.findall(r"[a-z0-9]+", term_norm)
                if len(token) > 2 and token not in generic_tokens
            ]
            if tokens:
                groups.append(tokens)
        if groups:
            return groups

        question_norm = normalized_text(analysis.standalone_question or "")
        for alias in law_aliases:
            question_norm = question_norm.replace(alias, " ")
        clauses = re.split(r"\band\b|,|;|/|\?", question_norm)
        for clause in clauses:
            tokens = [
                token
                for token in re.findall(r"[a-z0-9]+", clause)
                if len(token) > 2 and token not in generic_tokens
            ]
            if tokens:
                groups.append(tokens)
        return groups

    def _solve_single_law_topic_boolean(
        self,
        analysis: QuestionAnalysis,
        answer_type: str,
        route: Dict[str, Any],
    ) -> Optional[StructuredDecision]:
        if answer_type != "boolean" or analysis.target_field != "absence_check":
            return None
        law_ids = analysis.target_law_ids or extract_law_ids(route.get("question", ""))
        law = self._candidate_law(law_ids, route, analysis.target_titles)
        if law is None:
            return None
        groups = self._law_topic_groups(analysis, law)
        if not groups:
            return None

        payload = self.corpus.documents_payload.get(law.sha) or {}
        pages = (payload.get("content") or {}).get("pages") or []
        page_scores: List[tuple[int, set[int], int]] = []
        for page in pages[: max(12, min(len(pages), 25))]:
            page_number = int(page["page"])
            text = normalized_text(page.get("text") or "")
            matched_groups = {
                index
                for index, group in enumerate(groups)
                if all(token in text for token in group)
            }
            if matched_groups:
                page_scores.append((page_number, matched_groups, len(text)))

        if not page_scores:
            return None

        target_group_ids = set(range(len(groups)))
        best_true: Optional[tuple[List[int], set[int]]] = None
        for page_number, matched_groups, _ in page_scores:
            if matched_groups == target_group_ids:
                best_true = ([page_number], matched_groups)
                break
        if best_true is None:
            for left_page, left_groups, _ in page_scores:
                for right_page, right_groups, _ in page_scores:
                    merged = left_groups | right_groups
                    if merged == target_group_ids:
                        pages_pair = sorted(dict.fromkeys([left_page, right_page]))
                        best_true = (pages_pair, merged)
                        break
                if best_true is not None:
                    break

        if best_true is not None:
            support_pages = best_true[0]
            support = [chunk for page_number in support_pages for chunk in self._page_chunks(law.sha, page_number)[:2]]
            return StructuredDecision(True, self._answer_payload(answer_type, True, support), support)

        best_false_page = max(page_scores, key=lambda item: (len(item[1]), -item[0]))
        support = self._page_chunks(law.sha, best_false_page[0])[:2]
        return StructuredDecision(True, self._answer_payload(answer_type, False, support), support)

    def _solve_amended_laws(self, answer_type: str, law_ids: Sequence[str]) -> Optional[StructuredDecision]:
        if answer_type != "free_text" or not law_ids:
            return None
        target_law_id = law_ids[0]
        titles = []
        support: List[Dict[str, Any]] = []
        for sha, document in self.corpus.documents.items():
            if document.kind not in {"law", "regulation"}:
                continue
            text = self.corpus.documents_payload[sha]["content"]["pages"][0]["text"]
            if law_id_matches_text(text, target_law_id):
                title = answer_source_label(document.title)
                titles.append(title)
                support.extend(self._page_chunks(sha, 1)[:1])
        if not titles:
            return None
        answer = "The laws amended by this DIFC Law are: " + ", ".join(sorted(dict.fromkeys(titles))) + "."
        return StructuredDecision(True, self._answer_payload(answer_type, answer, support), support)

    def _extract_order_clause_candidates(
        self,
        chunks: Sequence[Dict[str, Any]],
    ) -> List[tuple[float, str, Dict[str, Any]]]:
        candidates: List[tuple[float, str, Dict[str, Any]]] = []
        for chunk in chunks:
            raw_lines = [line for line in str(chunk["text"] or "").splitlines() if normalize_space(line)]
            stop_section = False
            quoted_prior_order_block = False
            for raw_line in raw_lines:
                line = normalize_space(raw_line)
                line_norm = normalized_text(line)
                if not line_norm or is_heading_line(line):
                    continue
                if line_norm.startswith(
                    (
                        "document_type:",
                        "jurisdiction:",
                        "segment_category:",
                        "segment_kind:",
                        "mentioned_persons:",
                        "nearest_heading:",
                        "parent_context:",
                    )
                ):
                    continue
                if line_norm.startswith(("issued by", "schedule of reasons", "schedule of reason", "date of issue", "at:")):
                    stop_section = True
                    break
                if stop_section:
                    break
                if line_norm.startswith(("issued by", "date of issue", "at:", "schedule of reasons", "background", "introduction")):
                    continue
                if any(
                    marker in line_norm
                    for marker in (
                        "the learned judge ordered as follows",
                        "the learned justice ordered as follows",
                        "the lower court ordered as follows",
                    )
                ):
                    quoted_prior_order_block = True
                    continue
                if quoted_prior_order_block:
                    if (
                        line_norm.startswith(("consideration of permission", "the court thus has the power", "in those circumstances"))
                        or "permission to appeal is therefore" in line_norm
                        or "application is therefore" in line_norm
                    ):
                        quoted_prior_order_block = False
                    elif re.match(r"^(?:[-*]\s*)?(?:'?\d+\.)\s*", line, re.I) or line_norm in {"'", '"'}:
                        continue
                    elif line_norm.startswith(("53.", "54.", "55.")):
                        continue

                score = 0.0
                body = line
                main_match = re.match(r"^(?:-\s*)?(\d+)\.\s*(.+)$", line)
                if main_match:
                    body = main_match.group(2)
                    score += 12.0
                elif re.match(r"^(?:-\s*)?\(([0-9]+|[a-z]+)\)\s*", line, re.I):
                    # Sub-clauses usually contain cost-assessment mechanics, not the main disposition.
                    continue
                else:
                    if not any(
                        marker in line_norm
                        for marker in (
                            "refused",
                            "dismissed",
                            "granted",
                            "allowed",
                            "discharged",
                            "rejected",
                            "denied",
                            "struck out",
                            "proceed to trial",
                            "no order as to costs",
                            "costs are awarded",
                            "costs awarded",
                            "award of costs",
                            "entitled to its costs",
                            "bear its own costs",
                            "shall pay",
                            "statement of costs",
                            "within 14 days",
                            "interest shall accrue",
                            "reconsidered at a hearing",
                            "reconsideration",
                        )
                    ):
                        continue
                    score += 3.0

                cleaned = clean_clause_text(body)
                cleaned = re.sub(r"\bto be assessed as follows:\s*$", "to be assessed.", cleaned, flags=re.I)
                cleaned = re.sub(r"\bas follows:\s*$", "", cleaned, flags=re.I)
                for candidate_text in sentence_level_order_candidates(cleaned):
                    cleaned_norm = normalized_text(candidate_text)
                    candidate_score = score
                    if not cleaned_norm or len(cleaned_norm) < 12:
                        continue
                    if cleaned_norm.startswith(("and upon", "upon ", "pursuant to")):
                        continue
                    if any(
                        marker in cleaned_norm
                        for marker in ("real prospect of success", "background", "reasons set out below")
                    ):
                        candidate_score -= 4.0
                    if any(marker in cleaned_norm for marker in ORDER_OUTCOME_TERMS):
                        candidate_score += 10.0
                    if is_cost_related_order_clause(candidate_text):
                        candidate_score += 3.0
                    if "reconsidered at a hearing" in cleaned_norm or "reconsideration" in cleaned_norm:
                        candidate_score += 5.0
                    candidates.append((candidate_score, candidate_text.rstrip(".") + ".", chunk))
        candidates.sort(key=lambda item: (-item[0], item[2]["page"], item[2]["ref"]))
        return candidates

    def _score_order_clause(
        self,
        clause: str,
        question: str,
        analysis: QuestionAnalysis,
    ) -> float:
        clause_norm = normalized_text(clause)
        question_norm = normalized_text(question)
        score = 0.0
        if is_operative_order_clause(clause):
            score += 12.0
        if any(
            marker in clause_norm
            for marker in (*ORDER_OUTCOME_TERMS,)
        ):
            score += 8.0
        if re.match(r"^(the\s+)?(permission to appeal application|application|appeal|asi order|set aside application|claim)\b", clause_norm):
            score += 8.0
        if clause_norm.startswith("there shall"):
            score += 4.0
        if is_cost_related_order_clause(clause):
            score += 4.0
        for case_id in analysis.target_case_ids:
            if normalized_text(case_id) in clause_norm:
                score += 2.0
        question_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", question_norm)
            if len(token) > 3 and token not in ARTICLE_STOPWORDS
        }
        overlap = sum(1 for token in question_tokens if token in clause_norm)
        score += min(overlap, 8) * 2.5
        if "permission to appeal" in question_norm and "permission to appeal" in clause_norm:
            score += 12.0
        if "permission to appeal" in question_norm and "oral hearing" in clause_norm:
            score -= 12.0
        if "set aside" in question_norm and "set aside" in clause_norm:
            score += 12.0
        if "adjournment" in question_norm and "adjournment" in clause_norm:
            score += 12.0
        if "defendants" in question_norm and "defendant" in clause_norm:
            score += 4.0
        if "trial" in question_norm and "trial" in clause_norm:
            score += 4.0
        if "cost" in question_norm and is_cost_related_order_clause(clause):
            score += 8.0
        if "permission to appeal" in question_norm and "reconsidered at a hearing" in clause_norm:
            score += 6.0
        if any(marker in question_norm for marker in ("final ruling", "what did the court decide", "how did the court")) and (
            is_cost_related_order_clause(clause)
        ):
            score += 5.0
        if "last page" in question_norm and "cost" in clause_norm:
            score += 2.0
        if clause_norm.endswith(("of the", "of the.", "to the", "to the.")):
            score -= 14.0
        if clause_norm.startswith(("this is an application", "the test for", "by rdc ", "i granted")):
            score -= 20.0
        if clause_norm.startswith(("this order concerns", "by that order", "in determining the appropriate costs recovery")):
            score -= 24.0
        if any(marker in clause_norm for marker in ("realistic prospect of succeeding", "for the reasons set out below", "issue for determination")):
            score -= 10.0
        if "order_section" in set(analysis.support_focus) and "conclusion_section" not in set(analysis.support_focus):
            if len(clause.split()) > 35:
                score -= min(10.0, (len(clause.split()) - 35) * 0.35)
        if "it is hereby ordered that" in question_norm and len(clause.split()) > 40:
            score -= 12.0
        if "permission to appeal" in question_norm and not any(
            marker in clause_norm
            for marker in (
                "permission to appeal",
                "appeal",
                "application",
                "reconsidered at a hearing",
                "bear its own costs",
                "no order as to costs",
                "shall pay",
                "costs",
            )
        ):
            score -= 16.0
        if "permission to appeal" in question_norm and any(
            marker in clause_norm for marker in ("salary", "court fee", "training costs")
        ):
            score -= 40.0
        if "permission to appeal" in question_norm and "the learned judge ordered as follows" in clause_norm:
            score -= 24.0
        return score

    def _compose_case_order_answer(
        self,
        question: str,
        analysis: QuestionAnalysis,
        clauses: Sequence[tuple[float, str, Dict[str, Any]]],
    ) -> Optional[tuple[str, List[Dict[str, Any]]]]:
        if not clauses:
            return None
        question_norm = normalized_text(question)
        focus = set(analysis.support_focus)
        max_page = max((int(chunk["page"]) for _, _, chunk in clauses), default=1)
        scored = []
        for base_score, clause, chunk in clauses:
            score = self._score_order_clause(clause, question, analysis) + base_score
            page = int(chunk["page"])
            if "conclusion_section" in focus:
                if page >= max(1, max_page - 1):
                    score += 10.0
                elif page <= max(1, max_page - 3):
                    score -= 8.0
            if "last_page" in focus:
                if page >= max(1, max_page - 1):
                    score += 8.0
            if "order_section" in focus and "conclusion_section" not in focus:
                if page <= 2:
                    score += 6.0
                elif page >= 4 and len(clause.split()) > 35:
                    score -= 8.0
            if "first_page" in focus:
                if page <= 2:
                    score += 8.0
                elif page >= max(3, max_page - 1):
                    score -= 6.0
            scored.append((score, clause, chunk))
        scored.sort(key=lambda item: (-item[0], item[2]["page"], item[2]["ref"]))
        if "permission to appeal" in question_norm:
            scored = [
                item
                for item in scored
                if not any(marker in normalized_text(item[1]) for marker in ("salary", "court fee", "training costs"))
            ]
        operative = [
            (score, clause, chunk)
            for score, clause, chunk in scored
            if is_operative_order_clause(clause)
        ]
        primary = [(score, clause, chunk) for score, clause, chunk in operative if is_primary_outcome_order_clause(clause)]
        if "conclusion_section" in focus:
            late_primary = [
                (score, clause, chunk)
                for score, clause, chunk in primary
                if int(chunk["page"]) >= max(1, max_page - 1)
            ]
            if late_primary:
                primary = late_primary
        if "last_page" in focus:
            late_primary = [
                (score, clause, chunk)
                for score, clause, chunk in primary
                if int(chunk["page"]) >= max(1, max_page - 1)
            ]
            if late_primary:
                primary = late_primary
        if not primary or primary[0][0] < 12.0:
            fallback_primary = [
                (score, clause, chunk)
                for score, clause, chunk in scored
                if any(term in normalized_text(clause) for term in ORDER_OUTCOME_TERMS)
                and (
                    "application" in normalized_text(clause)
                    or "appeal" in normalized_text(clause)
                    or "claim" in normalized_text(clause)
                    or "permission to appeal" in normalized_text(clause)
                )
            ]
            fallback_primary = [
                item
                for item in fallback_primary
                if not normalized_text(item[1]).startswith(
                    (
                        "this order concerns",
                        "the judgment was",
                        "by that order",
                        "the issue for determination",
                    )
                )
            ]
            if "permission to appeal" in question_norm:
                fallback_primary = [
                    item
                    for item in fallback_primary
                    if "permission to appeal" in normalized_text(item[1]) or "appeal" in normalized_text(item[1])
                ]
            if fallback_primary:
                primary = fallback_primary
            else:
                return None

        explicit_cost_request = any(
            marker in question_norm
            for marker in (
                "cost",
                "costs awarded",
                "costs were awarded",
                "what costs",
            )
        )
        broad_decision_request = any(
            marker in question_norm
            for marker in (
                "final ruling",
                "what did the court decide",
                "how did the court",
            )
        )
        asks_payment_terms = any(
            marker in question_norm
            for marker in (
                "within 14 days",
                "when must",
                "when should",
                "when is",
                "deadline",
                "interest",
                "accrue",
                "payment term",
                "payment terms",
                "paid within",
            )
        )
        asks_reconsideration = any(
            marker in question_norm
            for marker in (
                "reconsider",
                "reconsideration",
                "hearing",
                "oral hearing",
            )
        )
        asks_merits = any(
            marker in question_norm
            for marker in (
                "totally without merit",
                "without merit",
                "realistic prospect",
                "prospect of success",
                "merits",
            )
        )
        generic_order_outcome_prompt = (
            not explicit_cost_request
            and not broad_decision_request
            and any(
                marker in question_norm
                for marker in (
                    "permission to appeal",
                    "it is hereby ordered that",
                    "last page",
                    "specific order or application described",
                    "outcome of the application",
                    "result of the application",
                    "what was the result",
                    "what was the outcome",
                )
            )
        )

        wants_full_outcome = any(
            marker in question_norm
            for marker in (
                "what was the result",
                "what was the outcome",
                "result of the application",
                "what did the court decide",
                "how did the court",
                "final ruling",
            )
        )
        full_operatives_question = any(
            marker in question_norm
            for marker in (
                "what was the result of the application heard",
                "what was the result of the application",
                "what did the court decide",
                "how did the court",
                "final ruling",
            )
        ) and all(
            marker not in question_norm
            for marker in (
                "specific order or application described",
                "outcome of the application for permission to appeal",
                "conclusion section",
                "it is hereby ordered",
                "ordered that",
                "last page",
                "first page",
            )
        )
        required_topics: set[str] = set()
        if "permission to appeal" in question_norm:
            required_topics.add("permission_to_appeal")
        if "adjournment" in question_norm:
            required_topics.add("adjournment")
        if "set aside" in question_norm:
            required_topics.add("set_aside")
        if "asi order" in question_norm:
            required_topics.add("asi_order")
        if "trial" in question_norm:
            required_topics.add("trial")
        narrow_permission_outcome = (
            "permission to appeal" in question_norm
            and (
                "outcome of the application" in question_norm
                or "what was the outcome" in question_norm
                or "what was the result" in question_norm
            )
            and not explicit_cost_request
            and not asks_reconsideration
            and not asks_payment_terms
            and not asks_merits
            and not full_operatives_question
            and "specific order or application described" not in question_norm
            and "how did the court" not in question_norm
            and "what did the court decide" not in question_norm
            and "final ruling" not in question_norm
        )

        selected: List[tuple[str, Dict[str, Any]]] = []
        seen_clauses = set()
        seen_topics: set[str] = set()
        max_primary = 1 if required_topics else (3 if wants_full_outcome else 2)
        if "last page" in question_norm and primary:
            latest_primary = max(primary, key=lambda item: (int(item[2]["page"]), item[0]))
            selected.append((latest_primary[1], latest_primary[2]))
            seen_clauses.add(normalized_text(latest_primary[1]))
            seen_topics.update(order_clause_topics(latest_primary[1]))
        prefer_direct_permission_clause = narrow_permission_outcome or (
            "permission to appeal" in question_norm
            and ("last_page" in focus or "last page" in question_norm)
        )
        if prefer_direct_permission_clause:
            direct_permission_candidates = [
                (clause, chunk)
                for score, clause, chunk in scored
                if any(
                    marker in normalized_text(clause)
                    for marker in (
                        "permission to appeal is therefore refused",
                        "permission to appeal is refused",
                        "permission to appeal was refused",
                        "application for permission to appeal is therefore refused",
                        "application for permission to appeal is refused",
                        "application for permission to appeal was refused",
                        "permission to appeal is granted",
                        "permission to appeal was granted",
                        "permission to appeal is allowed",
                        "permission to appeal was allowed",
                    )
                )
            ]
            if "last_page" in focus:
                direct_permission_candidates.sort(
                    key=lambda item: (-int(item[1]["page"]), item[1]["ref"])
                )
            direct_permission_clause = direct_permission_candidates[0] if direct_permission_candidates else None
            if direct_permission_clause is not None:
                selected.append(direct_permission_clause)
                seen_clauses.add(normalized_text(direct_permission_clause[0]))
                seen_topics.update(order_clause_topics(direct_permission_clause[0]))
        for index, (score, clause, chunk) in enumerate(primary):
            key = normalized_text(clause)
            if key in seen_clauses:
                continue
            clause_topics = order_clause_topics(clause)
            if not selected:
                seen_clauses.add(key)
                seen_topics.update(clause_topics)
                selected.append((clause, chunk))
                continue
            if required_topics and not (clause_topics & required_topics):
                continue
            distinct_topics = clause_topics - seen_topics
            if len(selected) >= max_primary:
                continue
            if distinct_topics and score >= max(18.0, primary[0][0] - 18.0):
                seen_clauses.add(key)
                seen_topics.update(clause_topics)
                selected.append((clause, chunk))
                continue
            if wants_full_outcome and distinct_topics and index < 4 and score >= max(18.0, primary[0][0] - 12.0):
                seen_clauses.add(key)
                seen_topics.update(clause_topics)
                selected.append((clause, chunk))

        primary_has_clear_outcome = any(
            any(term in normalized_text(clause) for term in ORDER_OUTCOME_TERMS)
            and not is_cost_related_order_clause(clause)
            for clause, _ in selected
        )
        include_costs = explicit_cost_request or (
            analysis.target_field == "order_result" and not narrow_permission_outcome
        )
        include_payment_followup = asks_payment_terms
        selected_pages = {int(chunk["page"]) for _, chunk in selected}
        early_order_pages = {
            int(chunk["page"])
            for _, _, chunk in scored
            if int(chunk["page"]) <= 2
        }

        if (
            "permission to appeal" in question_norm
            and selected_pages
            and early_order_pages
            and min(selected_pages) > max(early_order_pages)
            and not narrow_permission_outcome
        ):
            reconsideration_clause = next(
                (
                    (clause, chunk)
                    for score, clause, chunk in scored
                    if int(chunk["page"]) in early_order_pages
                    and normalized_text(clause) not in seen_clauses
                    and "reconsideration" in order_clause_topics(clause)
                ),
                None,
            )
            if reconsideration_clause is not None and len(selected) < 4:
                selected.append(reconsideration_clause)
                seen_clauses.add(normalized_text(reconsideration_clause[0]))
                seen_topics.update(order_clause_topics(reconsideration_clause[0]))
            if not any(is_cost_related_order_clause(clause) for clause, _ in selected):
                early_cost_clause = next(
                    (
                        (clause, chunk)
                        for score, clause, chunk in scored
                        if int(chunk["page"]) in early_order_pages
                        and normalized_text(clause) not in seen_clauses
                        and is_cost_related_order_clause(clause)
                    ),
                    None,
                )
                if early_cost_clause is not None and len(selected) < 5:
                    selected.append(early_cost_clause)
                    seen_clauses.add(normalized_text(early_cost_clause[0]))
                    seen_topics.update(order_clause_topics(early_cost_clause[0]))

        if include_costs:
            preferred_pages = {chunk["page"] for _, chunk in selected}
            broad_outcome_costs = explicit_cost_request or any(
                marker in question_norm
                for marker in (
                    "what was the result",
                    "what was the outcome",
                    "result of the application",
                    "final ruling",
                    "what did the court decide",
                    "how did the court",
                )
            )
            def preferred_cost_clause(item: tuple[float, str, Dict[str, Any]]) -> bool:
                _, clause, _chunk = item
                clause_norm = normalized_text(clause)
                return is_cost_related_order_clause(clause) or clause_norm.startswith(("there shall", "costs "))
            cost_clause = next(
                (
                    (clause, chunk)
                    for score, clause, chunk in scored
                    if chunk["page"] in preferred_pages
                    and preferred_cost_clause((score, clause, chunk))
                    and normalized_text(clause) not in seen_clauses
                ),
                None,
            )
            if cost_clause is None and broad_outcome_costs:
                cost_clause = next(
                    (
                        (clause, chunk)
                        for score, clause, chunk in scored
                        if preferred_cost_clause((score, clause, chunk))
                        and normalized_text(clause) not in seen_clauses
                    ),
                    None,
                )
            if cost_clause is None and broad_outcome_costs:
                cost_clause = next(
                    (
                        (clause, chunk)
                        for score, clause, chunk in scored
                        if normalized_text(clause) not in seen_clauses
                        and is_cost_related_order_clause(clause)
                    ),
                    None,
                )
            if cost_clause is not None:
                selected.append(cost_clause)
                seen_clauses.add(normalized_text(cost_clause[0]))
                seen_topics.update(order_clause_topics(cost_clause[0]))
                cost_clause_norm = normalized_text(cost_clause[0])
                primary_cost_kind = cost_clause_kind(cost_clause[0])
                if (
                    not include_payment_followup
                    and broad_outcome_costs
                    and "last_page" not in focus
                    and (
                        "costs award" in cost_clause_norm
                        or "usd" in cost_clause_norm
                        or "statement of costs" in cost_clause_norm
                        or re.search(r"\b\d+% of\b", cost_clause_norm)
                    )
                ):
                    include_payment_followup = True

                if broad_outcome_costs and primary_cost_kind in {"no_order", "own_costs"}:
                    additional_award_clause = next(
                        (
                            (clause, chunk)
                            for score, clause, chunk in scored
                            if normalized_text(clause) not in seen_clauses
                            and is_cost_related_order_clause(clause)
                            and cost_clause_kind(clause) == "award"
                        ),
                        None,
                    )
                    if additional_award_clause is not None and len(selected) < 4:
                        selected.append(additional_award_clause)
                        seen_clauses.add(normalized_text(additional_award_clause[0]))
                        seen_topics.update(order_clause_topics(additional_award_clause[0]))

                extra_cost_clause = next(
                    (
                        (clause, chunk)
                        for score, clause, chunk in scored
                        if normalized_text(clause) not in seen_clauses
                        and "payment_terms" in order_clause_topics(clause)
                        and chunk["page"] in preferred_pages
                    ),
                    None,
                )
                if include_payment_followup and extra_cost_clause is not None and len(selected) < 4:
                    selected.append(extra_cost_clause)
                    seen_clauses.add(normalized_text(extra_cost_clause[0]))
                elif include_payment_followup:
                    later_payment_clause = next(
                        (
                            (clause, chunk)
                            for score, clause, chunk in scored
                            if normalized_text(clause) not in seen_clauses
                            and "payment_terms" in order_clause_topics(clause)
                        ),
                        None,
                    )
                    if later_payment_clause is not None and len(selected) < 4:
                        selected.append(later_payment_clause)
                        seen_clauses.add(normalized_text(later_payment_clause[0]))

        if "permission to appeal" in question_norm and asks_reconsideration:
            reconsideration_candidates = [
                (score, clause, chunk)
                for score, clause, chunk in scored
                if normalized_text(clause) not in seen_clauses
                and "reconsideration" in order_clause_topics(clause)
            ]
            reconsideration_candidates.sort(key=lambda item: (len(item[1].split()) > 18, len(item[1]), item[2]["page"]))
            if reconsideration_candidates and len(selected) < 4:
                clause, chunk = reconsideration_candidates[0][1], reconsideration_candidates[0][2]
                selected.append((clause, chunk))
                seen_clauses.add(normalized_text(clause))
        if "permission to appeal" in question_norm and asks_merits:
            merits_candidates = [
                (score, clause, chunk)
                for score, clause, chunk in scored
                if normalized_text(clause) not in seen_clauses
                and any(marker in normalized_text(clause) for marker in ("no realistic prospect", "totally without merit"))
            ]
            merits_candidates.sort(key=lambda item: (-item[0], item[2]["page"], item[2]["ref"]))
            if merits_candidates and len(selected) < 5:
                clause, chunk = merits_candidates[0][1], merits_candidates[0][2]
                selected.append((clause, chunk))
                seen_clauses.add(normalized_text(clause))

        if full_operatives_question:
            extra_operatives = [
                (score, clause, chunk)
                for score, clause, chunk in scored
                if normalized_text(clause) not in seen_clauses
                and not is_cost_related_order_clause(clause)
                and any(term in normalized_text(clause) for term in ORDER_OUTCOME_TERMS)
            ]
            extra_operatives.sort(key=lambda item: (-item[0], item[2]["page"], item[2]["ref"]))
            for score, clause, chunk in extra_operatives:
                clause_topics = order_clause_topics(clause)
                if clause_topics and clause_topics - seen_topics:
                    selected.append((clause, chunk))
                    seen_clauses.add(normalized_text(clause))
                    seen_topics.update(clause_topics)
                if len(selected) >= 4:
                    break
            if not any(is_cost_related_order_clause(clause) for clause, _ in selected):
                no_cost_clause = next(
                    (
                        (clause, chunk)
                        for score, clause, chunk in scored
                        if normalized_text(clause) not in seen_clauses
                        and is_cost_related_order_clause(clause)
                    ),
                    None,
                )
                if no_cost_clause is not None and len(selected) < 5:
                    selected.append(no_cost_clause)
                    seen_clauses.add(normalized_text(no_cost_clause[0]))

        if full_operatives_question and selected:
            seed_chunk = selected[0][1]
            page_text = self.page_texts.get(seed_chunk["sha"], {}).get(int(seed_chunk["page"]), seed_chunk.get("text", ""))
            for clause in sentence_level_order_candidates(page_text):
                key = normalized_text(clause)
                if not key or key in seen_clauses:
                    continue
                topics = order_clause_topics(clause)
                if not topics:
                    continue
                if "payment_terms" in topics and not include_payment_followup:
                    continue
                if "reconsideration" in topics and not asks_reconsideration:
                    continue
                if not (topics - seen_topics) and not (
                    "costs" in topics and not any(is_cost_related_order_clause(existing) for existing, _ in selected)
                ):
                    continue
                selected.append((clause, seed_chunk))
                seen_clauses.add(key)
                seen_topics.update(topics)
                if len(selected) >= 5:
                    break

        answer_parts: List[str] = []
        for clause, _ in selected:
            clause_norm = normalized_text(clause)
            if is_cost_related_order_clause(clause):
                rendered_cost = concise_costs_clause(clause, include_followup=include_payment_followup)
                if rendered_cost:
                    answer_parts.append(rendered_cost)
            else:
                answer_parts.append(concise_disposition_clause(clause))
        answer = " ".join(answer_parts)
        descriptor = ""
        for _, chunk in selected:
            page_text = self.page_texts.get(chunk["sha"], {}).get(int(chunk["page"]), chunk.get("text", ""))
            descriptor = infer_application_descriptor(page_text)
            if descriptor:
                break
        answer = enhance_order_answer_with_descriptor(answer, descriptor)
        support = self._representative_support_chunks([chunk for _, chunk in selected], per_doc=2)
        return answer, support

    def _order_answer_covers_question(
        self,
        answer: str,
        question: str,
        analysis: QuestionAnalysis,
    ) -> bool:
        answer_norm = normalized_text(answer)
        question_norm = normalized_text(question)
        required_pairs = (
            ("permission to appeal", ("permission to appeal", "appeal")),
            ("adjournment", ("adjournment",)),
            ("set aside", ("set aside",)),
            ("asi order", ("asi order",)),
            ("trial", ("trial",)),
        )
        for trigger, acceptable in required_pairs:
            if trigger == "permission to appeal" and trigger in question_norm:
                if any(item in answer_norm for item in acceptable):
                    continue
                if "application is refused" in answer_norm or "application was refused" in answer_norm:
                    continue
                return False
            if trigger in question_norm and not any(item in answer_norm for item in acceptable):
                return False
        if analysis.must_support_terms:
            for term in analysis.must_support_terms:
                term_norm = normalized_text(term)
                if term_norm in {"result", "outcome", "it is hereby ordered that"}:
                    continue
                if "permission to appeal" in term_norm and (
                    "application is refused" in answer_norm
                    or "application was refused" in answer_norm
                    or "application is granted" in answer_norm
                    or "application was granted" in answer_norm
                ):
                    continue
                if len(term_norm) < 8:
                    continue
                if term_norm in question_norm and term_norm not in answer_norm:
                    if any(keyword in term_norm for keyword in ("appeal", "adjournment", "trial", "set aside", "asi order")):
                        return False
        return True

    def _solve_case_order_free_text(
        self,
        analysis: QuestionAnalysis,
        answer_type: str,
        route: Dict[str, Any],
    ) -> Optional[StructuredDecision]:
        if answer_type != "free_text" or analysis.target_field != "order_result":
            return None
        case_ids = analysis.target_case_ids or extract_case_ids(route.get("question", ""))
        if len(case_ids) != 1:
            return None
        case = self.case_by_id.get(case_ids[0])
        if case is None or not case.order_pages:
            return None

        focus = set(analysis.support_focus)
        explicit_focus_pages = self._explicit_page_focus_pages(route.get("question", ""), case)
        question_norm = normalized_text(route.get("question", ""))
        candidate_pages = explicit_focus_pages or self._candidate_case_order_pages(case, focus)
        if (
            explicit_focus_pages
            and "last page" in question_norm
            and any(
                marker in question_norm
                for marker in (
                    "what was the outcome",
                    "what was the result",
                    "specific order or application described",
                )
            )
        ):
            candidate_pages = list(dict.fromkeys([*explicit_focus_pages, *self._candidate_case_order_pages(case, focus)]))
        anchor_order_pages = list(dict.fromkeys(explicit_focus_pages or candidate_pages))

        chunks: List[Dict[str, Any]] = []
        for page in candidate_pages:
            chunks.extend(self._page_chunks(case.sha, page))
        clauses = self._extract_order_clause_candidates(chunks)
        missing_permission_outcome_clause = (
            "permission to appeal" in question_norm
            and not any(
                any(
                    marker in normalized_text(clause)
                    for marker in (
                        "permission to appeal is therefore refused",
                        "permission to appeal is refused",
                        "application for permission to appeal is refused",
                        "application for permission to appeal was refused",
                        "application for permission to appeal is therefore refused",
                    )
                )
                for _, clause, _ in clauses
            )
        )
        composed = self._compose_case_order_answer(route.get("question", ""), analysis, clauses)
        composed_covers = bool(
            composed is not None
            and self._order_answer_covers_question(composed[0], route.get("question", ""), analysis)
        )
        if (
            (composed is None or not composed_covers or missing_permission_outcome_clause)
            and "permission to appeal" in question_norm
        ):
            expanded_pages = list(candidate_pages)
            for page_number, page_text in self.page_texts.get(case.sha, {}).items():
                page_norm = normalized_text(page_text)
                if any(
                    marker in page_norm
                    for marker in (
                        "no realistic prospect",
                        "totally without merit",
                        "permission to appeal is therefore",
                        "permission to appeal is refused",
                        "application for permission to appeal is refused",
                    )
                ):
                    expanded_pages.append(int(page_number))
            expanded_pages = list(dict.fromkeys(expanded_pages))
            chunks = []
            for page in expanded_pages:
                chunks.extend(self._page_chunks(case.sha, page))
            clauses = self._extract_order_clause_candidates(chunks)
            composed = self._compose_case_order_answer(route.get("question", ""), analysis, clauses)
        if composed is None:
            return None
        answer, support = composed
        support_pages = {int(chunk["page"]) for chunk in support}
        if (
            anchor_order_pages
            and focus.intersection({"order_section", "last_page"})
            and "permission to appeal" not in question_norm
            and not (
            support_pages & set(anchor_order_pages)
            )
        ):
            preferred_anchor_page = max(anchor_order_pages)
            anchor_chunks = self._page_chunks(case.sha, preferred_anchor_page)
            if anchor_chunks:
                support = self._representative_support_chunks([*support, anchor_chunks[0]], per_doc=2)
        if not self._order_answer_covers_question(answer, route.get("question", ""), analysis):
            return None
        return StructuredDecision(True, self._answer_payload(answer_type, answer, support), support)

    def _solve_case_order_boolean(
        self,
        analysis: QuestionAnalysis,
        answer_type: str,
        route: Dict[str, Any],
    ) -> Optional[StructuredDecision]:
        if answer_type != "boolean" or analysis.target_field != "order_result":
            return None
        case_ids = analysis.target_case_ids or extract_case_ids(route.get("question", ""))
        if len(case_ids) != 1:
            return None
        case = self.case_by_id.get(case_ids[0])
        if case is None:
            return None
        focus = set(analysis.support_focus)
        explicit_focus_pages = self._explicit_page_focus_pages(route.get("question", ""), case)
        candidate_pages = explicit_focus_pages or self._candidate_case_order_pages(case, focus)
        chunks: List[Dict[str, Any]] = []
        for page in candidate_pages:
            chunks.extend(self._page_chunks(case.sha, page))
        clauses = self._extract_order_clause_candidates(chunks)
        if not clauses:
            return None
        question_norm = normalized_text(route.get("question", ""))
        scored = []
        for base_score, clause, chunk in clauses:
            clause_norm = normalized_text(clause)
            score = self._score_order_clause(clause, route.get("question", ""), analysis) + base_score
            if "granted" in question_norm or "allowed" in question_norm or "continuation" in question_norm:
                if any(term in clause_norm for term in ("granted", "allowed", "continued", "restored", "upheld")):
                    score += 10.0
                if any(term in clause_norm for term in ("refused", "dismissed", "rejected", "denied")):
                    score += 6.0
            if "asi order" in question_norm and "asi order" in clause_norm:
                score += 12.0
            scored.append((score, clause, chunk))
        scored.sort(key=lambda item: (-item[0], item[2]["page"], item[2]["ref"]))
        best_score, best_clause, best_chunk = scored[0]
        clause_norm = normalized_text(best_clause)
        positive = any(term in clause_norm for term in ("granted", "allowed", "continued", "restored", "upheld", "proceed to trial"))
        negative = any(term in clause_norm for term in ("refused", "dismissed", "rejected", "denied", "discharged", "struck out"))
        if not positive and not negative:
            return None
        if any(term in question_norm for term in ("granted", "allowed", "continuation", "continued")):
            answer = positive and not negative
        elif any(term in question_norm for term in ("refused", "dismissed", "rejected", "denied", "discharged")):
            answer = negative
        else:
            answer = positive and not negative
        support = self._representative_support_chunks([best_chunk], per_doc=1)
        return StructuredDecision(True, self._answer_payload(answer_type, answer, support), support)

    def _solve_single_source_article_content_free_text(
        self,
        analysis: QuestionAnalysis,
        answer_type: str,
        route: Dict[str, Any],
    ) -> Optional[StructuredDecision]:
        if (
            answer_type != "free_text"
            or analysis.target_field != "article_content"
            or analysis.confidence < 0.72
            or analysis.needs_multi_document_support
        ):
            return None

        support = self._override_article_context("free_text", [], route, analysis)
        if not support:
            return None
        article_refs = analysis.target_article_refs or extract_article_refs(route.get("question", ""))
        law = self._candidate_law(analysis.target_law_ids, route, analysis.target_titles)
        snippet = None
        payload_support = self._representative_support_chunks(support, per_doc=2)
        if law is not None and article_refs:
            exact = self._article_clause_answer(law.sha, article_refs[0], route.get("question", ""))
            if exact is not None:
                snippet, payload_support = exact
        if not snippet:
            snippet = self._best_snippet_from_chunks(support, analysis, route.get("question", ""), article_refs)
        if not snippet:
            return None
        return StructuredDecision(True, self._answer_payload(answer_type, snippet, payload_support), payload_support)

    def _solve_single_source_article_deterministic(
        self,
        analysis: QuestionAnalysis,
        answer_type: str,
        route: Dict[str, Any],
    ) -> Optional[StructuredDecision]:
        if (
            analysis.target_field != "article_content"
            or analysis.confidence < 0.72
            or len(analysis.target_article_refs or []) != 1
            or answer_type not in {"number", "boolean", "name"}
        ):
            return None
        law = self._candidate_law(analysis.target_law_ids, route, analysis.target_titles)
        if law is None:
            return None
        question_text = route.get("question", "")
        question_norm = normalized_text(question_text)
        if answer_type == "number" and any(marker in question_norm for marker in ("fine", "fee", "penalty", "schedule")):
            schedule_support = self._schedule_amount_support(law.sha, question_text)
            table_answer = self._best_schedule_amount_answer(schedule_support, question_text)
            if table_answer is not None:
                answer_text, payload_support = table_answer
                numeric_value = normalize_number(answer_text)
                if numeric_value is not None:
                    return StructuredDecision(True, self._answer_payload(answer_type, numeric_value, payload_support), payload_support)
        exact = self._article_clause_answer(law.sha, analysis.target_article_refs[0], route.get("question", ""))
        if exact is None:
            return None
        clause_text, support = exact
        evidence_pages = sorted({int(chunk["page"]) for chunk in support})
        evidence_text = normalize_space(
            " ".join(
                self.page_texts.get(law.sha, {}).get(page, "") or " ".join(
                    str(chunk.get("text") or "") for chunk in support if int(chunk["page"]) == page
                )
                for page in evidence_pages
            )
        )
        if answer_type == "number":
            numeric = self._extract_numeric_clause_answer(route.get("question", ""), clause_text)
            context_numeric = self._extract_numeric_from_evidence_context(
                route.get("question", ""),
                evidence_text,
                analysis.target_article_refs[0],
            )
            if numeric is None:
                numeric = context_numeric
            elif (
                context_numeric is not None
                and context_numeric != numeric
                and any(
                    marker in question_norm
                    for marker in (
                        "confirm",
                        "appeal",
                        "misleading",
                        "deceptive",
                        "conflicting",
                        "maternity",
                        "nursing",
                    )
                )
            ):
                numeric = context_numeric
            if numeric is None:
                return None
            return StructuredDecision(True, self._answer_payload(answer_type, numeric, support), support)
        if answer_type == "boolean":
            answer = self._extract_boolean_clause_answer(route.get("question", ""), clause_text, evidence_text)
            if answer is None:
                return None
            return StructuredDecision(True, self._answer_payload(answer_type, answer, support), support)
        if answer_type == "name":
            answer = self._extract_name_clause_answer(route.get("question", ""), clause_text, evidence_text)
            if not answer:
                return None
            return StructuredDecision(True, self._answer_payload(answer_type, answer, support), support)
        return None

    def _solve_single_source_clause_deterministic(
        self,
        analysis: QuestionAnalysis,
        answer_type: str,
        route: Dict[str, Any],
        reranked: Sequence[Dict[str, Any]] = (),
    ) -> Optional[StructuredDecision]:
        question_norm = normalized_text(route.get("question", ""))
        deictic_defined_term = (
            answer_type == "name"
            and (
                "defined term for the law" in question_norm
                or "these regulations refer to" in question_norm
            )
        )
        if (
            analysis.target_field not in {"clause_summary", "generic_answer"}
            or analysis.confidence < 0.8
            or analysis.needs_multi_document_support
            or analysis.target_case_ids
            or (not (analysis.target_law_ids or analysis.target_titles) and not deictic_defined_term)
            or answer_type not in {"number", "boolean", "name"}
        ):
            return None
        support = self._override_clause_focus_context(route, analysis) or []
        law = self._candidate_law(analysis.target_law_ids, route, analysis.target_titles)
        if not support and deictic_defined_term:
            support = self._reranked_single_document_support(reranked)
        if law is not None:
            schedule_support = self._schedule_amount_support(law.sha, route.get("question", ""))
            if schedule_support:
                merged: List[Dict[str, Any]] = []
                seen = set()
                for chunk in list(support) + schedule_support:
                    ref = str(chunk.get("ref") or "")
                    if not ref or ref in seen:
                        continue
                    seen.add(ref)
                    merged.append(chunk)
                support = merged
        if not support:
            return None
        evidence_text = normalize_space("\n".join(str(chunk.get("text") or "") for chunk in support))
        snippet = self._best_snippet_from_chunks(support, analysis, route.get("question", ""), [])
        if answer_type == "number":
            table_answer = self._best_schedule_amount_answer(support, route.get("question", ""))
            if table_answer is not None:
                answer_text, payload_support = table_answer
                numeric_value = normalize_number(answer_text)
                if numeric_value is not None:
                    return StructuredDecision(True, self._answer_payload(answer_type, numeric_value, payload_support), payload_support)
            if any(marker in question_norm for marker in ("fee", "fine", "penalty", "usd", "aed", "schedule")):
                return None
            numeric = self._extract_numeric_from_evidence_context(route.get("question", ""), evidence_text, None)
            if numeric is None and snippet:
                numeric = self._extract_numeric_clause_answer(route.get("question", ""), snippet)
            if numeric is None:
                return None
            payload_support = self._representative_support_chunks(support, per_doc=2)
            return StructuredDecision(True, self._answer_payload(answer_type, numeric, payload_support), payload_support)
        if answer_type == "boolean":
            bool_answer = self._extract_boolean_clause_answer(route.get("question", ""), snippet or evidence_text, evidence_text)
            if bool_answer is None:
                return None
            payload_support = self._representative_support_chunks(support, per_doc=2)
            return StructuredDecision(True, self._answer_payload(answer_type, bool_answer, payload_support), payload_support)
        if answer_type == "name":
            name_answer = self._extract_name_clause_answer(route.get("question", ""), snippet or evidence_text, evidence_text)
            if not name_answer:
                return None
            if deictic_defined_term:
                focused_support = [
                    chunk
                    for chunk in support
                    if (
                        "defined terms" in normalized_text(chunk.get("text") or "")
                        or re.search(r"\|\s*Law\s*\|", str(chunk.get("text") or ""))
                        or "this law is the" in normalized_text(chunk.get("text") or "")
                        or re.search(r"\bdefinition\s+of\s+['\"]Law['\"]\s+is\s+modified\s+to\b", str(chunk.get("text") or ""), re.I)
                    )
                ]
                if focused_support:
                    support = focused_support
            payload_support = self._representative_support_chunks(support, per_doc=1 if deictic_defined_term else 2)
            return StructuredDecision(True, self._answer_payload(answer_type, name_answer, payload_support), payload_support)
        return None

    def _solve_single_source_clause_free_text(
        self,
        analysis: QuestionAnalysis,
        answer_type: str,
        route: Dict[str, Any],
        reranked: Sequence[Dict[str, Any]],
    ) -> Optional[StructuredDecision]:
        question_norm = normalized_text(route.get("question", ""))
        clause_like_fallback = (
            analysis.target_field in {"generic_answer", "comparison_answer", "clause_summary"}
            and not analysis.needs_multi_document_support
            and not analysis.target_case_ids
            and (analysis.target_titles or analysis.target_law_ids)
            and any(
                marker in question_norm
                for marker in (
                    "fine",
                    "fee",
                    "penalty",
                    "liability",
                    "records",
                    "retain",
                    "preserve",
                    "translation",
                    "purpose",
                    "appoint",
                    "dismiss",
                    "responsible for",
                    "permitted to",
                )
            )
        )
        if (
            answer_type != "free_text"
            or not clause_like_fallback
            or analysis.confidence < 0.8
        ):
            return None
        support = self._override_clause_focus_context(route, analysis) or []
        law = self._candidate_law(analysis.target_law_ids, route, analysis.target_titles)
        if law is not None:
            schedule_support = self._schedule_amount_support(law.sha, route.get("question", ""))
            if schedule_support:
                merged: List[Dict[str, Any]] = []
                seen = set()
                for chunk in list(support) + schedule_support:
                    ref = str(chunk.get("ref") or "")
                    if not ref or ref in seen:
                        continue
                    seen.add(ref)
                    merged.append(chunk)
                support = merged
        if not support:
            return None
        table_answer = self._best_schedule_amount_answer(support, route.get("question", ""))
        if table_answer is not None:
            answer, payload_support = table_answer
            return StructuredDecision(True, self._answer_payload(answer_type, answer, payload_support), payload_support)
        snippet = self._best_snippet_from_chunks(support, analysis, route.get("question", ""), [])
        if not snippet:
            return None
        payload_support = self._representative_support_chunks(support, per_doc=2)
        return StructuredDecision(True, self._answer_payload(answer_type, snippet, payload_support), payload_support)

    def _solve_article_content_free_text(
        self,
        analysis: QuestionAnalysis,
        answer_type: str,
        route: Dict[str, Any],
    ) -> Optional[StructuredDecision]:
        if (
            answer_type != "free_text"
            or analysis.target_field != "article_content"
            or analysis.confidence < 0.8
            or not analysis.needs_multi_document_support
            or len(analysis.target_titles) <= 1
        ):
            return None

        support = self._override_multi_document_article_context(route, analysis)
        if not support:
            return None
        answer = self._compose_multi_source_article_answer(support, analysis, route.get("question", ""))
        if not answer:
            return None
        payload_support = self._representative_support_chunks(support, per_doc=2)
        return StructuredDecision(True, self._answer_payload(answer_type, answer, payload_support), support)

    def _representative_support_chunks(
        self,
        chunks: Sequence[Dict[str, Any]],
        per_doc: int = 2,
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}
        seen_refs: set[str] = set()
        seen_doc_pages: set[tuple[str, int]] = set()

        def add_chunk(chunk: Dict[str, Any]) -> bool:
            ref = str(chunk.get("ref") or "")
            if ref and ref in seen_refs:
                return False
            sha = str(chunk["sha"])
            used = counts.get(sha, 0)
            if used >= per_doc:
                return False
            counts[sha] = used + 1
            if ref:
                seen_refs.add(ref)
            seen_doc_pages.add((sha, int(chunk["page"])))
            result.append(chunk)
            return True

        for chunk in chunks:
            sha = str(chunk["sha"])
            page = int(chunk["page"])
            if (sha, page) in seen_doc_pages:
                continue
            add_chunk(chunk)

        for chunk in chunks:
            add_chunk(chunk)

        return result

    def _question_content_tokens(self, analysis: QuestionAnalysis, question: str) -> set[str]:
        tokens = article_query_tokens(analysis, question)
        for term in analysis.must_support_terms:
            for token in re.findall(r"[a-z0-9]+", normalized_text(term)):
                if len(token) > 3 and token not in ARTICLE_STOPWORDS:
                    tokens.add(token)
        return tokens

    def _iter_document_lines(self, sha: str) -> List[Dict[str, Any]]:
        payload = self.corpus.documents_payload[sha]
        rows = list(payload["content"]["chunks"])
        rows.sort(key=lambda chunk: (int(chunk["page"]), int(chunk["id"])))
        result: List[Dict[str, Any]] = []
        for chunk in rows:
            page = int(chunk["page"])
            chunk_id = int(chunk["id"])
            text = str(chunk["text"] or "")
            record = self._chunk_record(sha, page, chunk_id, text)
            for raw_line in text.splitlines():
                line = normalize_space(raw_line)
                if not line:
                    continue
                result.append(
                    {
                        "line": line,
                        "page": page,
                        "chunk_id": chunk_id,
                        "record": record,
                    }
                )
        return result

    def _parse_article_ref(self, article_ref: str) -> tuple[Optional[str], List[str]]:
        ref_norm = normalize_space(article_ref)
        match = re.search(r"Article\s+(\d+)", ref_norm, re.I)
        if not match:
            return None, []
        parts = [normalize_space(part).lower() for part in re.findall(r"\(([^)]+)\)", ref_norm)]
        return match.group(1), parts

    def _split_clause_segments(self, text: str) -> List[str]:
        value = str(text or "").replace("\r", "\n")
        value = re.sub(r"\s+(?=##\s+)", "\n", value)
        value = re.sub(r"\s+(?=-\s*\((?:[0-9]+|[a-z]+)\))", "\n", value, flags=re.I)
        return [normalize_space(line) for line in value.splitlines() if normalize_space(line)]

    def _question_keyword_tokens(self, question: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", normalized_text(question))
            if len(token) > 3 and token not in ARTICLE_STOPWORDS
        }

    def _score_clause_segment(
        self,
        segment: str,
        question_tokens: set[str],
        marker: str | None,
        article_number: str,
        article_ref: str,
    ) -> float:
        segment_norm = normalized_text(segment)
        if not segment_norm or segment.startswith("#"):
            return -100.0
        if "|" in segment:
            return -100.0
        score = 0.0
        article_norm = normalized_text(article_ref)
        if article_norm and article_norm in segment_norm:
            score += 30.0
            if any(
                f"{prefix}{article_norm}" in segment_norm
                for prefix in (
                    "under ",
                    "pursuant to ",
                    "specified in ",
                    "in accordance with ",
                    "reference to ",
                )
            ):
                score -= 24.0
        if re.search(rf"\barticle\s+{re.escape(article_number)}(?:\b|\()", segment_norm):
            score += 16.0
        if marker and re.match(rf"^\-\s*\(\s*{re.escape(marker)}\s*\)(?:\s|$)", segment, re.I):
            score += 26.0
        overlap = sum(1 for token in question_tokens if token in segment_norm)
        score += overlap * 4.0
        if marker and re.search(rf"\(\s*{re.escape(marker)}\s*\)", segment_norm):
            score += 10.0
        if any(unit in segment_norm for unit in (" day", "days", "month", "months", "year", "years", "hour", "hours")):
            score += 3.0
        if any(term in segment_norm for term in ("true", "false", "liable", "valid", "effective", "conclusive", "must", "may", "shall", "fine", "costs")):
            score += 2.0
        return score

    def _article_clause_answer_from_chunks(
        self,
        law_sha: str,
        article_ref: str,
        question: str,
    ) -> Optional[tuple[str, List[Dict[str, Any]]]]:
        article_number, clause_path = self._parse_article_ref(article_ref)
        if not article_number:
            return None

        payload = self.corpus.documents_payload[law_sha]
        chunk_rows = payload["content"]["chunks"]
        question_tokens = self._question_keyword_tokens(question)
        article_norm = normalized_text(article_ref)
        target_marker = clause_path[-1] if clause_path else None
        parent_marker = clause_path[0] if len(clause_path) >= 2 else None
        best: tuple[float, Dict[str, Any], List[str], int] | None = None

        for chunk_index, chunk in enumerate(chunk_rows):
            text = str(chunk["text"] or "")
            text_norm = normalized_text(text)
            if not text_norm:
                continue
            local_context_text = " ".join(
                str(chunk_rows[index]["text"] or "")
                for index in range(max(0, chunk_index - 4), min(len(chunk_rows), chunk_index + 2))
            )
            local_context_norm = normalized_text(local_context_text)
            page_text = self.page_texts.get(law_sha, {}).get(int(chunk["page"]), "")
            page_intro_norm = normalized_text("\n".join(page_text.splitlines()[:8]))
            local_article_supported = False
            chunk_score = 0.0
            overlap = sum(1 for token in question_tokens if token in text_norm)
            if article_norm and article_norm in text_norm:
                chunk_score += 45.0
                local_article_supported = True
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
                    chunk_score -= 60.0 if not parent_marker or not re.search(rf"\(\s*{re.escape(parent_marker)}\s*\)", text_norm) else 30.0
            elif article_norm and article_norm in local_context_norm:
                chunk_score += 28.0
                local_article_supported = True
            if re.search(rf"\barticle\s+{re.escape(article_number)}\b", text_norm):
                chunk_score += 20.0
                local_article_supported = True
            elif re.search(rf"\barticle\s+{re.escape(article_number)}(?:\b|\()", local_context_norm):
                chunk_score += 16.0
                local_article_supported = True
            if re.search(rf"(?:^|\b|##\s*){re.escape(article_number)}\.", text_norm):
                chunk_score += 18.0
                local_article_supported = True
            elif re.search(rf"(?:^|\b|##\s*){re.escape(article_number)}\.", local_context_norm):
                chunk_score += 12.0
                local_article_supported = True
            if clause_path:
                if parent_marker and re.search(rf"\(\s*{re.escape(parent_marker)}\s*\)", text_norm):
                    chunk_score += 18.0
                for part in clause_path:
                    if re.search(rf"\(\s*{re.escape(part)}\s*\)", text_norm):
                        chunk_score += 12.0
                if target_marker and re.search(rf"\(\s*{re.escape(target_marker)}\s*\)", text_norm):
                    chunk_score += 18.0
                if parent_marker and re.search(rf"\(\s*{re.escape(parent_marker)}\s*\)", local_context_norm):
                    chunk_score += 8.0
            chunk_score += overlap * 2.0
            if clause_path and not local_article_supported:
                if any(marker in page_intro_norm for marker in ("schedule", "interpretation")):
                    continue
                chunk_score -= 18.0
            if chunk_score < 14.0:
                continue

            segments = self._split_clause_segments(text)
            if not segments:
                continue
            candidate_segments: List[tuple[int, str]] = list(enumerate(segments))
            if clause_path:
                marker_segments = [
                    (index, segment)
                    for index, segment in candidate_segments
                    if re.match(rf"^\-\s*\(\s*{re.escape(target_marker or '')}\s*\)(?:\s|$)", segment, re.I)
                ]
                if marker_segments:
                    candidate_segments = marker_segments
                else:
                    continue
            best_index = -1
            best_segment_score = -100.0
            for index, segment in candidate_segments:
                marker_match = re.match(r"^\-\s*\(\s*([^)]+)\s*\)", segment, re.I)
                marker = marker_match.group(1).lower() if marker_match else None
                segment_score = self._score_clause_segment(segment, question_tokens, target_marker, article_number, article_ref)
                if clause_path:
                    if marker and marker == target_marker:
                        segment_score += 12.0
                    if marker and len(clause_path) >= 2 and marker == clause_path[0]:
                        segment_score += 3.0
                if segment_score > best_segment_score:
                    best_segment_score = segment_score
                    best_index = index
            if best_index < 0 or best_segment_score < 10.0:
                continue

            selected = [segments[best_index]]
            marker_match = re.match(r"^\-\s*\(\s*([^)]+)\s*\)", segments[best_index], re.I)
            selected_marker = marker_match.group(1).lower() if marker_match else ""
            selected_kind = clause_marker_kind(selected_marker)
            current = segments[best_index]
            if current.rstrip().endswith(":"):
                for segment in segments[best_index + 1 :]:
                    next_match = re.match(r"^\-\s*\(\s*([^)]+)\s*\)", segment, re.I)
                    if not next_match:
                        if len(selected) >= 2:
                            break
                        selected.append(segment)
                        continue
                    next_marker = next_match.group(1).lower()
                    next_kind = clause_marker_kind(next_marker)
                    if selected_kind == "alpha" and next_kind == "roman":
                        selected.append(segment)
                        continue
                    if selected_kind == "digit" and next_kind in {"alpha", "roman"}:
                        selected.append(segment)
                        continue
                    break

            joined = compact_clause_text(" ".join(selected))
            if not joined:
                continue
            joined_norm = normalized_text(joined)
            if article_norm and article_norm in joined_norm and any(
                f"{prefix}{article_norm}" in joined_norm
                for prefix in (
                    "under ",
                    "pursuant to ",
                    "specified in ",
                    "in accordance with ",
                    "reference to ",
                )
            ):
                continue

            candidate_score = chunk_score + best_segment_score
            record = self._chunk_record(law_sha, int(chunk["page"]), int(chunk["id"]), text)
            if best is None or candidate_score > best[0]:
                best = (candidate_score, record, selected, best_index)

        if best is None:
            return None

        _, record, selected_segments, _ = best
        body = compact_clause_text(" ".join(selected_segments))
        question_norm = normalized_text(question)
        body_norm = normalized_text(body)
        if not re.search(r"\b(is|are|must|may|shall|can|requires?|permits?|prohibits?|liable)\b", body_norm):
            if "record" in question_norm:
                body = f"The provision requires records of {body.rstrip(';,.')}."
            else:
                body = f"The provision states that {body.rstrip(';,.')}."
        else:
            body = body.rstrip(".") + "."
        return body, [record]

    def _article_line_records(self, sha: str, article_number: str) -> List[Dict[str, Any]]:
        lines = self._iter_document_lines(sha)
        collected: List[Dict[str, Any]] = []
        collecting = False
        for item in lines:
            heading_match = re.match(r"^(?:##\s*)?(\d+)\.\s*(.+)$", item["line"])
            if heading_match:
                current_article = heading_match.group(1)
                if collecting and current_article != article_number:
                    break
                collecting = current_article == article_number
            if collecting:
                collected.append(item)
        return collected

    def _article_clause_answer(
        self,
        law_sha: str,
        article_ref: str,
        question: str,
    ) -> Optional[tuple[str, List[Dict[str, Any]]]]:
        article_number, clause_path = self._parse_article_ref(article_ref)
        if not article_number:
            return None
        lines = self._article_line_records(law_sha, article_number)
        if not lines:
            return self._article_clause_answer_from_chunks(law_sha, article_ref, question)

        section_start = 0
        section_end = len(lines)
        if clause_path:
            first = clause_path[0]
            first_pattern = re.compile(rf"^(?:-\s*)?\({re.escape(first)}\)\s*(.*)$", re.I)
            sibling_pattern = re.compile(r"^(?:-\s*)?\(([0-9]+)\)\s*(.*)$", re.I)
            for index, item in enumerate(lines):
                if first_pattern.match(item["line"]):
                    section_start = index
                    break
            for index in range(section_start + 1, len(lines)):
                sibling = sibling_pattern.match(lines[index]["line"])
                if sibling and sibling.group(1).lower() != first:
                    section_end = index
                    break

        target_line: Optional[Dict[str, Any]] = None
        if clause_path:
            current_range = lines[section_start:section_end]
            if len(clause_path) >= 2:
                sub = clause_path[1]
                sub_pattern = re.compile(rf"^(?:-\s*)?\({re.escape(sub)}\)\s*(.*)$", re.I)
                for item in current_range:
                    if sub_pattern.match(item["line"]):
                        target_line = item
                        break
            if target_line is None:
                target_line = lines[section_start]
        else:
            for item in lines:
                line = item["line"]
                if re.match(r"^(?:##\s*)?\d+\.\s*", line):
                    continue
                if is_heading_line(line):
                    continue
                target_line = item
                break

        if target_line is None:
            return self._article_clause_answer_from_chunks(law_sha, article_ref, question)
        if is_heading_line(target_line["line"]) or re.match(r"^(?:##\s*)?\d+\.\s*", target_line["line"]):
            return self._article_clause_answer_from_chunks(law_sha, article_ref, question)

        target_index = lines.index(target_line)
        body = clean_clause_text(target_line["line"])
        continuation_record: Optional[Dict[str, Any]] = None
        if body.endswith(":") and target_index + 1 < len(lines):
            next_line = clean_clause_text(lines[target_index + 1]["line"])
            if next_line and not is_heading_line(next_line):
                body = normalize_space(f"{body} {next_line}")
                if lines[target_index + 1]["record"]["ref"] != target_line["record"]["ref"]:
                    continuation_record = lines[target_index + 1]["record"]
        body = compact_clause_text(body)

        question_norm = normalized_text(question)
        body_norm = normalized_text(body)
        if not re.search(r"\b(is|are|must|may|shall|can|requires?|permits?|prohibits?|liable)\b", body_norm):
            if "record" in question_norm:
                body = f"The provision requires records of {body.rstrip(';,.')}."
            else:
                body = f"The provision states that {body.rstrip(';,.')}."
        else:
            body = body.rstrip(".") + "."

        support_records = [target_line["record"]]
        if continuation_record is not None:
            support_records.append(continuation_record)
        support = self._representative_support_chunks(support_records, per_doc=2)
        return body, support

    def _best_snippet_from_chunks(
        self,
        chunks: Sequence[Dict[str, Any]],
        analysis: QuestionAnalysis,
        question: str,
        article_refs: Sequence[str],
    ) -> Optional[str]:
        question_tokens = self._question_content_tokens(analysis, question)
        best_score = 0.0
        best_line: Optional[str] = None

        for chunk in chunks:
            lines = [normalize_space(line) for line in str(chunk["text"] or "").splitlines() if normalize_space(line)]
            for index, line in enumerate(lines):
                line_norm = normalized_text(line)
                if not line_norm or line.strip().startswith("#"):
                    continue
                score = 0.0
                if len(line_norm) >= 20:
                    score += 1.0
                overlap = sum(1 for token in question_tokens if token in line_norm)
                score += overlap * 3.0
                if line_norm.endswith(":"):
                    score -= 4.0
                for article_ref in article_refs:
                    article_norm = normalized_text(article_ref)
                    if article_norm and article_norm in line_norm:
                        score += 12.0
                    for part in re.findall(r"\([^)]+\)", article_ref):
                        if part.lower() in line_norm:
                            score += 4.0
                if re.match(r"^(?:-\s*)?\(([0-9]+|[a-z])\)", line.strip(), re.I):
                    score += 1.5
                if any(marker in line_norm for marker in ("liable", "retain", "retained", "preserved", "regulations", "administer", "records", "purpose", "responsible", "required", "translation", "fine", "appoint", "dismiss", "remuneration")):
                    score += 4.0
                if any(marker in line_norm for marker in ("six (6) years", "years from", "after the date of reporting", "at least six", "jointly and severally")):
                    score += 4.0
                if score > best_score:
                    candidate = clean_clause_text(line)
                    if (
                        index + 1 < len(lines)
                        and len(candidate) < 120
                        and candidate.rstrip().endswith(":")
                    ):
                        candidate = normalize_space(candidate + " " + clean_clause_text(lines[index + 1]))
                    best_score = score
                    best_line = compact_clause_text(candidate)

        if best_line:
            return best_line.rstrip(".") + "."
        return None

    def _best_schedule_amount_answer(
        self,
        chunks: Sequence[Dict[str, Any]],
        question: str,
    ) -> Optional[tuple[str, List[Dict[str, Any]]]]:
        question_norm = normalized_text(question)
        question_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", question_norm)
            if len(token) > 3 and token not in ARTICLE_STOPWORDS
        }
        ref_tokens = {
            normalize_space(match.group(1)).lower()
            for match in re.finditer(r"\b(?:article|regulation|rule)\s+([0-9]+(?:\.[0-9]+)*(?:\([0-9a-z]+\))?)", question, re.I)
        }
        best_score = 0.0
        best_amount: Optional[str] = None
        best_chunk: Optional[Dict[str, Any]] = None
        best_line: Optional[str] = None
        for chunk in chunks:
            chunk_text = str(chunk["text"] or "")
            chunk_norm = normalized_text(chunk_text)
            for raw_line in str(chunk["text"] or "").splitlines():
                line = normalize_space(raw_line)
                if not line:
                    continue
                cells = [normalize_space(cell) for cell in raw_line.split("|") if normalize_space(cell)]
                amount_match = (
                    re.search(r"\b(US\$\s*[0-9,]+(?:\.[0-9]+)?)\b", line, re.I)
                    or re.search(r"\b(AED\s*[0-9,]+(?:\.[0-9]+)?)\b", line, re.I)
                    or re.search(r"(\$[0-9,]+(?:\.[0-9]+)?)", line)
                )
                if amount_match is None and cells:
                    last_cell = cells[-1]
                    if (
                        re.fullmatch(r"[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?", last_cell)
                        and any(marker in chunk_norm for marker in ("fee (usd)", "maximum fine", "table of fees", "contraventions with fines stipulated"))
                    ):
                        amount_match = re.match(r"(.+)", last_cell)
                nil_match = re.search(r"\bNil\b", line, re.I)
                if amount_match is None and nil_match is None:
                    continue
                line_norm = normalized_text(line)
                amount_text = normalize_space(amount_match.group(1)) if amount_match else "Nil"
                score = sum(1 for token in question_tokens if token in line_norm) * 4.0
                if "accounts" in question_norm and "accounts" in line_norm:
                    score += 10.0
                if "accounts" in question_norm and "accounts" not in line_norm:
                    score -= 4.0
                if "file" in question_norm and "file" in line_norm:
                    score += 8.0
                if "file" in question_norm and "approved" in line_norm and "file" not in line_norm:
                    score -= 10.0
                if "annual" in question_norm and "annual" in line_norm:
                    score += 4.0
                if "provide" in question_norm and "provide" in line_norm:
                    score += 6.0
                if "provide" in question_norm and "approved" in line_norm and all(
                    token not in line_norm for token in ("provide", "file", "copy")
                ):
                    score -= 6.0
                if "fine" in question_norm and "fine" in line_norm:
                    score += 6.0
                if "fee" in question_norm and "fee" in line_norm:
                    score += 6.0
                if "fee" in question_norm and "fine" in line_norm and "fee" not in line_norm:
                    score -= 6.0
                if "fine" in question_norm and "fee" in line_norm and "fine" not in line_norm:
                    score -= 6.0
                for ref_token in ref_tokens:
                    if ref_token in line_norm:
                        score += 14.0
                if "assignment of rights" in question_norm and "assignment of rights" in line_norm:
                    score += 10.0
                if "registered agent" in question_norm and "registered agent" in line_norm:
                    score += 4.0
                if "registered agent" in line_norm and "accounts" in question_norm and "accounts" not in line_norm:
                    score -= 8.0
                if "registrar" in question_norm and "registrar" in line_norm:
                    score += 4.0
                if "notice of change of founding member" in question_norm and "notice of change of founding member" in line_norm:
                    score += 12.0
                if "certificate of compliance" in question_norm and "certificate of compliance" in line_norm:
                    score += 12.0
                if "exemption" in question_norm and "exemption" in line_norm:
                    score += 12.0
                if "entertainment events" in question_norm and "entertainment events" in line_norm:
                    score += 12.0
                if score > best_score:
                    best_score = score
                    best_amount = amount_text
                    best_chunk = chunk
                    best_line = line
        if best_amount is None or best_chunk is None or best_score < 8.0:
            return None
        support_chunks = [best_chunk]
        asks_article_basis = any(
            marker in question_norm
            for marker in (
                "article ",
                "which article",
                "under article",
                "according to article",
                "clause",
                "schedule 3",
            )
        )
        needs_clause_basis = asks_article_basis or any(
            marker in question_norm
            for marker in (
                "within ",
                "failing to",
                "fails to",
                "failure to",
                "must ",
            )
        )
        direct_row_sufficient = (
            best_score >= 18.0
            and not needs_clause_basis
        )
        if best_line and "|" not in best_line:
            direct_row_sufficient = True
        if not direct_row_sufficient:
            article_match = re.search(r"\|\s*([0-9]+\([0-9a-z]+\))\s*\|", best_line or "", re.I)
            for chunk in chunks:
                if chunk["ref"] == best_chunk["ref"]:
                    continue
                text_norm = normalized_text(chunk["text"])
                score = 0.0
                if article_match and article_match.group(1).lower() in text_norm:
                    score += 12.0
                if "liable to a fine" in text_norm:
                    score += 10.0
                if "schedule 3" in text_norm:
                    score += 6.0
                overlap = sum(1 for token in question_tokens if token in text_norm)
                score += overlap * 2.0
                if score >= 10.0:
                    support_chunks.append(chunk)
                    break
        if "fee" in question_norm and "fine" not in question_norm:
            answer = f"The fee is {best_amount}."
        else:
            answer = f"The maximum fine is {best_amount}."
        return answer, self._representative_support_chunks(support_chunks, per_doc=2)

    def _schedule_amount_support(
        self,
        law_sha: str,
        question: str,
    ) -> List[Dict[str, Any]]:
        question_norm = normalized_text(question)
        if not any(marker in question_norm for marker in ("fine", "fee", "penalty", "schedule")):
            return []
        question_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", question_norm)
            if len(token) > 3 and token not in ARTICLE_STOPWORDS
        }
        payload = self.corpus.documents_payload[law_sha]
        scored: List[tuple[float, Dict[str, Any]]] = []
        for chunk in payload["content"]["chunks"]:
            raw_text = str(chunk["text"] or "")
            text_norm = normalized_text(raw_text)
            if not raw_text:
                continue
            if (
                "$" not in raw_text
                and "|" not in raw_text
                and "schedule" not in text_norm
                and "fine" not in text_norm
                and "fee" not in text_norm
            ):
                continue
            score = 0.0
            if "|" in raw_text and "$" in raw_text:
                score += 20.0
            if "schedule" in text_norm:
                score += 8.0
            if "fine" in question_norm and "fine" in text_norm:
                score += 8.0
            if "fee" in question_norm and "fee" in text_norm:
                score += 8.0
            overlap = sum(1 for token in question_tokens if token in text_norm)
            score += overlap * 4.0
            if "assignment of rights" in question_norm and "assignment of rights" in text_norm:
                score += 12.0
            if "registered agent" in question_norm and "registered agent" in text_norm:
                score += 6.0
            if "registrar" in question_norm and "registrar" in text_norm:
                score += 4.0
            if score < 8.0:
                continue
            record = self._chunk_record(
                law_sha,
                int(chunk["page"]),
                int(chunk["id"]),
                raw_text,
                score,
            )
            scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], item[1]["page"], item[1]["ref"]))
        result: List[Dict[str, Any]] = []
        seen = set()
        for _, record in scored[:4]:
            ref = record["ref"]
            if ref in seen:
                continue
            seen.add(ref)
            result.append(record)
        return result

    def _compose_multi_source_article_answer(
        self,
        support: Sequence[Dict[str, Any]],
        analysis: QuestionAnalysis,
        question: str,
    ) -> Optional[str]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in support:
            grouped.setdefault(chunk["sha"], []).append(chunk)

        parts: List[str] = []
        seen = set()
        for index, title in enumerate(analysis.target_titles):
            title_norm = normalized_text(title)
            rows: List[tuple[float, str]] = []
            for sha in grouped:
                document = self.corpus.documents[sha]
                values = [document.title, *document.aliases, *document.canonical_ids]
                score = 0.0
                for value in values:
                    value_norm = normalized_text(value)
                    if not value_norm:
                        continue
                    if title_norm == value_norm:
                        score = max(score, 200.0)
                    elif title_norm in value_norm:
                        score = max(score, 140.0 + len(title_norm))
                    elif value_norm in title_norm:
                        score = max(score, 120.0 + len(value_norm))
                    else:
                        title_tokens = {token for token in title_norm.split() if len(token) > 3 and token not in TITLE_MATCH_STOPWORDS}
                        value_tokens = {token for token in value_norm.split() if len(token) > 3 and token not in TITLE_MATCH_STOPWORDS}
                        overlap_tokens = title_tokens & value_tokens
                        required_overlap = min(2, len(title_tokens), len(value_tokens))
                        if required_overlap and len(overlap_tokens) >= required_overlap:
                            score = max(score, len(overlap_tokens) * 20.0)
                if score > 0:
                    rows.append((score, sha))
            rows.sort(key=lambda item: (-item[0], item[1]))
            scoped_refs = article_refs_for_title(
                analysis.target_article_refs,
                title=title,
                index=index,
                total_titles=len(analysis.target_titles),
            )
            snippet = None
            chosen_sha = None
            for _, sha in rows:
                source_chunks = grouped.get(sha, [])
                if not source_chunks:
                    continue
                snippet = None
                if scoped_refs:
                    exact = self._article_clause_answer(sha, scoped_refs[0], question)
                    if exact is not None:
                        snippet = exact[0]
                if not snippet:
                    snippet = self._best_snippet_from_chunks(source_chunks, analysis, question, scoped_refs)
                if snippet:
                    chosen_sha = sha
                    break
            if not snippet or not chosen_sha:
                continue
            label = readable_source_label(title) or readable_source_label(self.corpus.documents[chosen_sha].title)
            label_norm = normalized_text(label)
            snippet = re.sub(
                r"^Under\s+Article\s+([0-9]+(?:\([^)]+\))*)\s+of\s+the\s+.+?,\s*",
                "",
                snippet,
                flags=re.I,
            )
            snippet = re.sub(
                r"^Article\s+([0-9]+(?:\([^)]+\))*)\s+of\s+the\s+.+?\s+(requires?|provides?|empowers?|states?|permits?|prohibits?|makes?|sets?|establishes?)\s+",
                lambda match: f"Article {match.group(1)} {match.group(2).lower()} ",
                snippet,
                flags=re.I,
            )
            snippet = normalize_space(snippet)
            if snippet:
                if not snippet.startswith("Article "):
                    snippet = snippet[0].lower() + snippet[1:]
                snippet = snippet.rstrip(".") + "."
            key = (label, snippet)
            if key in seen:
                continue
            seen.add(key)
            parts.append(f"{label}: {snippet}")
        if len(parts) >= 2:
            return " ".join(parts)
        return None

    def _override_article_context(
        self,
        answer_type: str,
        law_ids: Sequence[str],
        route: Dict[str, Any],
        analysis: QuestionAnalysis,
    ) -> Optional[List[Dict[str, Any]]]:
        article_refs = analysis.target_article_refs or extract_article_refs(route.get("question", ""))
        if not article_refs:
            return None
        law = self._candidate_law(law_ids, route, analysis.target_titles)
        if law is None:
            return None
        if len(article_refs) == 1:
            exact = self._article_clause_answer(law.sha, article_refs[0], route.get("question", ""))
            if exact is not None:
                return exact[1]
        payload = self.corpus.documents_payload[law.sha]
        chunk_rows = list(payload["content"]["chunks"])
        chunk_rows.sort(key=lambda chunk: (int(chunk["id"]), int(chunk["page"])))
        chunk_id_to_pos = {int(chunk["id"]): index for index, chunk in enumerate(chunk_rows)}
        chunk_by_page_and_id = {
            (int(chunk["page"]), int(chunk["id"])): chunk
            for chunk in chunk_rows
        }
        keyword_tokens = article_query_tokens(analysis, route.get("question", ""))
        seed_rows: List[tuple[float, Dict[str, Any], str]] = []
        for chunk in chunk_rows:
            text_norm = normalized_text(chunk["text"])
            raw_text = str(chunk["text"] or "")
            if "table of contents" in text_norm or ("contents" in text_norm and raw_text.count("|") >= 2):
                continue
            if "..." in raw_text and raw_text.count("|") >= 2:
                continue
            if raw_text.count("|") >= 3 and "##" not in raw_text:
                continue
            score = 0.0
            for article_ref in article_refs:
                article_norm = normalized_text(article_ref)
                number_match = re.search(r"article\s+(\d+)", article_norm)
                if number_match and re.search(rf"(?:^|\b|##\s*)({number_match.group(1)})\.", text_norm):
                    score += 25.0
                if article_norm and article_norm in text_norm:
                    score += 35.0
                for part in re.findall(r"\([^)]+\)", article_ref):
                    if part.lower() in text_norm:
                        score += 6.0
            if keyword_tokens:
                overlap = sum(1 for token in keyword_tokens if token in text_norm)
                score += min(overlap, 8) * 3.0
            if score >= 20.0:
                seed_rows.append(
                    (
                        score,
                        self._chunk_record(law.sha, int(chunk["page"]), int(chunk["id"]), str(chunk["text"]), score),
                        raw_text,
                    )
                )
        if not seed_rows:
            return None
        seed_rows.sort(key=lambda item: (-item[0], item[1]["page"], item[1]["ref"]))

        merged: List[Dict[str, Any]] = []
        seen = set()
        for score, record, raw_text in seed_rows[:3]:
            if record["ref"] not in seen:
                seen.add(record["ref"])
                merged.append(record)
            chunk_id = int(record["ref"].split(":")[-1])
            pos = chunk_id_to_pos.get(chunk_id)
            if pos is not None:
                for offset in (1, 2):
                    if pos + offset >= len(chunk_rows):
                        break
                    neighbor = chunk_rows[pos + offset]
                    neighbor_record = self._chunk_record(
                        law.sha,
                        int(neighbor["page"]),
                        int(neighbor["id"]),
                        str(neighbor["text"]),
                        max(score - offset, 1.0),
                    )
                    if neighbor_record["ref"] not in seen:
                        seen.add(neighbor_record["ref"])
                        merged.append(neighbor_record)
            if "##" in raw_text and raw_text.count("- (") == 0:
                neighbor = chunk_by_page_and_id.get((record["page"], chunk_id + 1))
                if neighbor is not None:
                    neighbor_record = self._chunk_record(
                        law.sha,
                        int(neighbor["page"]),
                        int(neighbor["id"]),
                        str(neighbor["text"]),
                        max(score - 0.5, 1.0),
                    )
                    if neighbor_record["ref"] not in seen:
                        seen.add(neighbor_record["ref"])
                        merged.append(neighbor_record)
        merged.sort(key=lambda item: (-item["distance"], item["page"]))
        return merged[:5]

    def _override_clause_focus_context(
        self,
        route: Dict[str, Any],
        analysis: QuestionAnalysis,
    ) -> Optional[List[Dict[str, Any]]]:
        law = self._candidate_law(analysis.target_law_ids, route, analysis.target_titles)
        if law is None:
            return None
        payload = self.corpus.documents_payload[law.sha]
        chunk_rows = list(payload["content"]["chunks"])
        chunk_rows.sort(key=lambda chunk: (int(chunk["id"]), int(chunk["page"])))
        chunk_id_to_pos = {int(chunk["id"]): index for index, chunk in enumerate(chunk_rows)}
        keyword_tokens = self._question_content_tokens(analysis, route.get("question", ""))
        question_norm = normalized_text(route.get("question", ""))
        deictic_defined_term = (
            "defined term for the law" in question_norm
            or "these regulations refer to" in question_norm
        )
        seed_rows: List[tuple[float, Dict[str, Any]]] = []
        for chunk in chunk_rows:
            text_norm = normalized_text(chunk["text"])
            raw_text = str(chunk["text"] or "")
            if "table of contents" in text_norm or ("contents" in text_norm and raw_text.count("|") >= 2):
                continue
            if "..." in raw_text and raw_text.count("|") >= 2:
                continue
            if (
                raw_text.count("|") >= 3
                and "##" not in raw_text
                and not any(token in text_norm for token in keyword_tokens)
            ):
                continue
            score = 0.0
            if keyword_tokens:
                overlap = sum(1 for token in keyword_tokens if token in text_norm)
                score += min(overlap, 8) * 3.0
            score += 6.0 * sum(1 for term in analysis.must_support_terms if normalized_text(term) in text_norm)
            if deictic_defined_term:
                if "defined terms" in text_norm or re.search(r"\|\s*Law\s*\|", raw_text):
                    score += 8.0
                if "this law is the" in text_norm:
                    score += 6.0
                if "modified to" in text_norm or "deleted" in text_norm:
                    score -= 6.0
            if any(marker in text_norm for marker in ("translation", "retain", "preserve", "records", "fine", "liability", "administer")):
                score += 2.0
            if score >= 8.0:
                seed_rows.append(
                    (
                        score,
                        self._chunk_record(law.sha, int(chunk["page"]), int(chunk["id"]), str(chunk["text"]), score),
                    )
                )
        if not seed_rows:
            return None
        seed_rows.sort(key=lambda item: (-item[0], item[1]["page"], item[1]["ref"]))
        merged: List[Dict[str, Any]] = []
        seen = set()
        for score, record in seed_rows[:3]:
            if record["ref"] not in seen:
                seen.add(record["ref"])
                merged.append(record)
            chunk_id = int(record["ref"].split(":")[-1])
            pos = chunk_id_to_pos.get(chunk_id)
            if pos is None:
                continue
            for offset in (1, 2):
                if pos + offset >= len(chunk_rows):
                    break
                neighbor = chunk_rows[pos + offset]
                neighbor_record = self._chunk_record(
                    law.sha,
                    int(neighbor["page"]),
                    int(neighbor["id"]),
                    str(neighbor["text"]),
                    max(score - offset, 1.0),
                )
                if neighbor_record["ref"] in seen:
                    continue
                seen.add(neighbor_record["ref"])
                merged.append(neighbor_record)
        merged.sort(key=lambda item: (-item["distance"], item["page"]))
        return merged[:5]

    def _override_multi_document_article_context(
        self,
        route: Dict[str, Any],
        analysis: QuestionAnalysis,
    ) -> Optional[List[Dict[str, Any]]]:
        merged: List[Dict[str, Any]] = []
        seen = set()
        for index, title in enumerate(analysis.target_titles):
            scoped_analysis = analysis.model_copy(
                update={
                    "target_titles": [title],
                    "target_article_refs": article_refs_for_title(
                        analysis.target_article_refs,
                        title=title,
                        index=index,
                        total_titles=len(analysis.target_titles),
                    ),
                }
            )
            chunks = self._override_article_context("free_text", [], route, scoped_analysis)
            if not chunks:
                continue
            for chunk in chunks[:3]:
                if chunk["ref"] in seen:
                    continue
                seen.add(chunk["ref"])
                merged.append(chunk)
        return merged[:6] or None

    def _override_case_order_context(
        self,
        answer_type: str,
        case_ids: Sequence[str],
        focus: set[str],
    ) -> Optional[List[Dict[str, Any]]]:
        if len(case_ids) != 1:
            return None
        case = self.case_by_id.get(case_ids[0])
        if case is None:
            return None
        pages = []
        if "first_page" in focus:
            pages.append(1)
        if "last_page" in focus:
            pages.append(self.corpus.documents[case.sha].page_count)
        if "last_page" not in focus or "order_section" in focus:
            pages.extend(self._candidate_case_order_pages(case, focus))
        pages = [page for page in dict.fromkeys(pages) if page]
        chunks: List[Dict[str, Any]] = []
        for page in pages:
            chunks.extend(self._page_chunks(case.sha, page)[:3])
        return chunks[:6] or None

    def _override_case_first_page_context(
        self,
        case_ids: Sequence[str],
        focus: set[str],
    ) -> Optional[List[Dict[str, Any]]]:
        if len(case_ids) != 1:
            return None
        case = self.case_by_id.get(case_ids[0])
        if case is None:
            return None
        pages = [1]
        if "issue_date_line" in focus and case.issue_page:
            pages.insert(0, case.issue_page)
        chunks: List[Dict[str, Any]] = []
        for page in dict.fromkeys(pages):
            chunks.extend(self._page_chunks(case.sha, page)[:4])
        return chunks[:6] or None

    def _override_law_focus_context(
        self,
        law_ids: Sequence[str],
        route: Dict[str, Any],
        analysis: QuestionAnalysis,
        focus: set[str],
    ) -> Optional[List[Dict[str, Any]]]:
        law = self._candidate_law(law_ids, route, analysis.target_titles)
        if law is None:
            return None
        pages: List[int] = []
        if analysis.target_field in {"law_number", "publication_text"} or (
            "title_page" in focus and not focus.intersection({"administration_clause", "enactment_clause", "commencement_clause"})
        ):
            pages.append(law.page_one)
        if "administration_clause" in focus and law.administered_by_page:
            pages.append(law.administered_by_page)
        if "enactment_clause" in focus and law.enacted_page:
            pages.append(law.enacted_page)
        if "commencement_clause" in focus and law.commencement_page:
            pages.append(law.commencement_page)
        if not pages:
            pages.append(law.page_one)
        chunks: List[Dict[str, Any]] = []
        for page in dict.fromkeys(pages):
            chunks.extend(self._page_chunks(law.sha, page)[:4])
        return chunks[:6] or None
