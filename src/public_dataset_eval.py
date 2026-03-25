from __future__ import annotations

import argparse
import json
import math
import hashlib
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from dateutil import parser as date_parser
from rank_bm25 import BM25Okapi
from pydantic import BaseModel, Field
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.api_requests import APIProcessor
from src.lexical_retrieval import select_novel_results, tokenize_for_bm25
from src.local_models import (
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    DEFAULT_LOCAL_RERANKER_MODEL,
    LocalEmbeddingModel,
    LocalJinaReranker,
)
from src.parsed_reports_merging import PageTextPreparation
from src.pdf_parsing import PDFParser
from src.text_splitter import TextSplitter

if TYPE_CHECKING:
    from src.query_analysis import QuestionAnalysis


CASE_ID_RE = re.compile(
    r"\b((?:C\s*F\s*I)|(?:A\s*R\s*B)|(?:T\s*C\s*D)|(?:C\s*A)|(?:E\s*N\s*F)|(?:D\s*E\s*C)|(?:S\s*C\s*T))\s*[- ]?(\d{3})/(\d{4})\b",
    re.I,
)
LAW_ID_RE = re.compile(r"\b(?:DIFC\s+)?Law\s+No\.?\s*\(?(\d+)\)?\s+of\s+(\d{4})\b", re.I)
LAW_ID_VARIANT_RE = re.compile(
    r"\bLaw\s+of\s+the\s+Dubai\s+International\s+Financial\s+Centre\s+No\.?\s*\(?(\d+)\)?\s+of\s+(\d{4})\b",
    re.I,
)
ARTICLE_RE = re.compile(r"Article\s+\d+(?:\([^)]+\))*", re.I)
MONTH_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b",
    re.I,
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
    "is",
    "was",
    "were",
    "does",
    "did",
    "how",
    "all",
    "any",
    "same",
    "their",
    "that",
    "this",
    "these",
    "those",
    "case",
    "law",
    "laws",
}

FREE_TEXT_ABSENCE_ANSWER = "There is no information on this question in the provided documents."
DOCUMENT_KIND_NOISE_MARKERS = (
    "CONSULTATION PAPER",
    "COURT ADMINISTRATIVE ORDERS",
    "ANNEX ",
    "# ANNEX",
    "ROADMAP TO THE PROPOSED CHANGES",
    "TABLE OF COMMENTS ON CONSULTATION PAPER",
)


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
            record.setdefault("source_scores", {})
            record["source_scores"][source_name] = round(float(item.get("distance", 0.0)), 4)
            record.setdefault("rrf_score", 0.0)
            record["rrf_score"] += weight / (fusion_k + rank)

    merged = list(fused.values())
    for item in merged:
        item["retrieval_sources"] = sorted(set(item.get("retrieval_sources", [])))
        item["distance"] = round(float(item["rrf_score"]) / max_rrf_score, 4)
    merged.sort(key=lambda item: item["distance"], reverse=True)
    return merged


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def canonical_case_id(prefix: str, number: str, year: str) -> str:
    compact_prefix = re.sub(r"[^A-Za-z]", "", prefix or "").upper()
    return f"{compact_prefix} {int(number):03d}/{year}"


def canonical_law_id(number: str, year: str) -> str:
    return f"Law No. {int(number)} of {year}"


def extract_case_ids(text: str) -> List[str]:
    return sorted({canonical_case_id(*match.groups()) for match in CASE_ID_RE.finditer(text or "")})


def extract_law_ids(text: str) -> List[str]:
    text = text or ""
    return sorted(
        {
            canonical_law_id(*match.groups())
            for pattern in (LAW_ID_RE, LAW_ID_VARIANT_RE)
            for match in pattern.finditer(text)
        }
    )


def infer_document_kind(first_page_text: str, case_ids: Sequence[str] | None = None) -> str:
    head = normalize_space(first_page_text[:2500])
    head_upper = head.upper()
    case_ids = list(case_ids or extract_case_ids(head[:800]))
    if case_ids:
        return "case"
    if any(marker in head_upper for marker in DOCUMENT_KIND_NOISE_MARKERS):
        return "other"
    if "ORDER NO." in head_upper and "COURTS" in head_upper:
        return "other"
    if "THESE RULES MAY BE CITED AS" in head_upper:
        return "regulation"
    if "REGULATIONS" in head_upper or "RULES OF THE DUBAI INTERNATIONAL FINANCIAL CENTRE COURTS" in head_upper:
        return "regulation"
    if "LAW" in head_upper:
        return "law"
    return "other"


def extract_primary_law_ids(first_page_text: str) -> List[str]:
    head = normalize_space(first_page_text[:2500])
    if not head:
        return []
    title_zone = re.split(
        r"\b(?:As\s+Amended\s+by|TABLE\s+OF\s+CONTENTS|CONTENTS|Having\s+reviewed|Why\s+are\s+we\s+issuing\s+this\s+paper\?|##\s+CONTENTS)\b",
        head,
        maxsplit=1,
        flags=re.I,
    )[0]
    ids = extract_law_ids(title_zone)
    if ids:
        return ids
    return extract_law_ids(head[:800])


def extract_article_refs(text: str) -> List[str]:
    return sorted({normalize_space(match.group(0)) for match in ARTICLE_RE.finditer(text or "")})


def extract_quoted_phrases(text: str) -> List[str]:
    phrases = re.findall(r"'([^']+)'", text or "")
    phrases.extend(re.findall(r'"([^"]+)"', text or ""))
    return sorted({normalize_space(phrase) for phrase in phrases if normalize_space(phrase)})


def safe_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def safe_json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalized_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return normalize_space(str(value)).lower()


def normalize_name(value: str) -> str:
    value = normalized_text(value)
    value = re.sub(r"[“”\"'`]", "", value)
    value = re.sub(r"[\.,;:()\[\]]", " ", value)
    return normalize_space(value)


def normalize_names(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        parts = value
    else:
        text = normalize_space(str(value))
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    parts = parsed
                else:
                    parts = [text]
            except Exception:
                parts = re.split(r"\||;|,|\band\b", text, flags=re.I)
        else:
            parts = re.split(r"\||;|,|\band\b", text, flags=re.I)
    cleaned = [normalize_name(part) for part in parts if normalize_name(part)]
    return sorted(dict.fromkeys(cleaned))


def normalize_boolean(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = normalized_text(value)
    if text in {"true", "yes", "y", "approved", "granted"}:
        return True
    if text in {"false", "no", "n", "not approved", "not granted"}:
        return False
    if text.startswith("yes"):
        return True
    if text.startswith("no"):
        return False
    return None


def normalize_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if re.search(r"\bnil\b", normalized_text(value)):
        return 0.0
    text = str(value).replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def normalize_date_string(value: Any) -> str:
    text = normalized_text(value)
    if not text:
        return text
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    text = re.sub(r"\b0(\d)\b", r"\1", text)
    try:
        parsed = date_parser.parse(text, fuzzy=True, dayfirst=True)
        return parsed.date().isoformat()
    except Exception:
        return text


def normalize_answer(answer_type: str, value: Any) -> Any:
    if answer_type == "boolean":
        return normalize_boolean(value)
    if answer_type == "number":
        return normalize_number(value)
    if answer_type == "name":
        return normalize_name(str(value))
    if answer_type == "names":
        return normalize_names(value)
    if answer_type == "date":
        return normalize_date_string(value)
    return normalized_text(value)


def answer_match(answer_type: str, gold: Any, predicted: Any) -> bool:
    if answer_type == "number":
        if gold is None or predicted is None:
            return False
        return math.isclose(float(gold), float(predicted), rel_tol=1e-6, abs_tol=1e-6)
    if answer_type == "names":
        return list(gold or []) == list(predicted or [])
    return gold == predicted


@dataclass
class DocumentMeta:
    sha: str
    kind: str
    title: str
    first_page_text: str
    aliases: List[str]
    canonical_ids: List[str]
    page_count: int
    is_enactment_notice: bool
    is_consolidated: bool


@dataclass
class ChunkMeta:
    ref: str
    sha: str
    page: int
    chunk_id: int
    text: str
    title: str
    kind: str
    canonical_ids: List[str]


class QAResponse(BaseModel):
    answer: Any
    citations: List[str] = Field(default_factory=list)
    reasoning: str = ""
    confidence: str = "medium"


class JudgeVerdict(BaseModel):
    verdict: str = "incorrect"
    score: float = 0.0
    explanation: str = ""


class ExtraQuestion(BaseModel):
    question: str
    answer_type: str
    answer: Any
    supporting_refs: List[str] = Field(default_factory=list)


class ExtraQuestionBundle(BaseModel):
    questions: List[ExtraQuestion] = Field(default_factory=list)
    qa_examples: List[ExtraQuestion] = Field(default_factory=list)


class SourceFreeTextAnswer(BaseModel):
    answer_fragment: str = ""
    citations: List[str] = Field(default_factory=list)
    absent: bool = False
    reasoning: str = ""


class OrderResultExtraction(BaseModel):
    disposition: str = ""
    disposition_citations: List[str] = Field(default_factory=list)
    costs: str = ""
    costs_citations: List[str] = Field(default_factory=list)
    absent: bool = False
    reasoning: str = ""


class AbsenceExplanation(BaseModel):
    answer: str = ""
    citations: List[str] = Field(default_factory=list)
    reasoning: str = ""


class FreeTextCandidateSelection(BaseModel):
    choice: str = "A"
    reasoning: str = ""


def ordered_unique(values: Sequence[Any]) -> List[Any]:
    result: List[Any] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def infer_document_title(first_page_text: str, canonical_ids: Sequence[str], kind: str | None = None) -> str:
    head = normalize_space(first_page_text[:2500])
    if not head:
        return "Unknown Document"
    kind = kind or infer_document_kind(first_page_text)

    def consultation_heading_title(raw_text: str) -> Optional[str]:
        lines = [
            normalize_space(re.sub(r"^#+\s*", "", line))
            for line in str(raw_text or "").splitlines()
            if normalize_space(re.sub(r"^#+\s*", "", line))
        ]
        cp_index = next((index for index, line in enumerate(lines) if "CONSULTATION PAPER" in line.upper()), -1)
        if cp_index < 0:
            return None
        parts: List[str] = [lines[cp_index]]
        subtitle: Optional[str] = None
        for line in lines[cp_index + 1 : cp_index + 6]:
            if re.fullmatch(r"[A-Za-z]+\s+\d{4}", line):
                parts.append(line)
                continue
            if re.fullmatch(r"DIFC\s+LAW\s+NO\.?\s*\d+\s+OF\s+\d{4}", line, re.I):
                continue
            if line.upper().startswith(("ANNEX ", "APPENDIX ")):
                continue
            if any(keyword in line.upper() for keyword in ("LAW", "REGULATION", "REGULATIONS", "RULES", "AMENDMENT", "AMENDMENTS", "DIGITAL ASSETS", "PRESCRIBED COMPANY")):
                subtitle = line
                break
        if subtitle:
            parts.append(subtitle)
        title = normalize_space(" ".join(parts))
        return title or None

    citation_match = re.search(
        r"These\s+(Rules|Regulations)\s+may\s+be\s+cited\s+as\s+(.*?)(?:\.|;|\s+and\s+may\s+be\s+abbreviated)",
        head,
        re.I,
    )
    if citation_match:
        return normalize_space(citation_match.group(2).strip(" '\""))

    if canonical_ids and canonical_ids[0].startswith(("CFI", "ARB", "TCD", "CA", "ENF", "DEC", "SCT")):
        case_id = re.escape(canonical_ids[0])
        month_match = MONTH_RE.search(head)
        if month_match:
            left = head[: month_match.start()].strip()
            match = re.search(case_id + r"\s+(.*)$", left, re.I)
            if match:
                return normalize_space(match.group(1))
        match = re.search(case_id + r"\s+(.*?)\s+Claim No", head, re.I)
        if match:
            return normalize_space(match.group(1))
        return canonical_ids[0]

    enactment_match = re.search(r"attached the\s+(.*?)\s+This Law shall come into force", head, re.I)
    if enactment_match:
        return normalize_space(enactment_match.group(1))

    heading_line = normalize_space(re.sub(r"^#+\s*", "", head.split("##", 1)[0].splitlines()[0]))
    if heading_line and not heading_line.upper().startswith("PART 1"):
        if kind == "other" and "CONSULTATION PAPER" in heading_line.upper():
            consultation_title = consultation_heading_title(first_page_text)
            if consultation_title:
                return consultation_title
        if kind == "law" and "LAW" in heading_line.upper():
            return heading_line
        if kind == "regulation" and any(marker in heading_line.upper() for marker in ("RULES", "REGULATIONS")):
            return heading_line
        if kind == "other" and any(marker in heading_line.upper() for marker in ("CONSULTATION PAPER", "ANNEX", "ORDER NO.")):
            return heading_line

    title_match = re.search(r"^(.*?)(?:\s+DIFC\s+LAW\s+NO\.?\s*\d+|\s+Consolidated Version|\s+In force\b|\s+TABLE OF CONTENTS|\s+CONTENTS\b)", head, re.I)
    if title_match:
        return normalize_space(title_match.group(1).strip(" ."))

    reg_match = re.search(r"^(.*?REGULATIONS)", head, re.I)
    if reg_match:
        return normalize_space(reg_match.group(1))

    return normalize_space(head[:140])


def infer_aliases(title: str, canonical_ids: Sequence[str], first_page_text: str) -> List[str]:
    aliases = {normalize_space(title)}
    head = normalize_space(first_page_text[:1500])

    if title.upper().startswith("THE "):
        aliases.add(normalize_space(title[4:]))

    for canonical_id in canonical_ids:
        aliases.add(canonical_id)
        if canonical_id.startswith("Law No."):
            year_match = re.search(r"(\d{4})$", canonical_id)
            if year_match and title:
                aliases.add(f"{title} {year_match.group(1)}")

    if "DIFC" in title:
        aliases.add(normalize_space(title.replace("DIFC", "")))

    if "LAW" in title.upper():
        aliases.add(normalize_space(title.replace("Law", "Law").replace("LAW", "Law")))

    if "REGULATIONS" in title.upper():
        aliases.add(normalize_space(title.replace("REGULATIONS", "Regulations")))

    short_title_match = re.search(r"^(.*?)(?:\s+Amendment Law)$", title, re.I)
    if short_title_match:
        aliases.add(normalize_space(short_title_match.group(1)))

    if "CONSULTATION PAPER" in title.upper():
        subtitle_match = re.search(r"##\s+([^\n]+)", str(first_page_text or ""))
        if subtitle_match:
            subtitle = normalize_space(subtitle_match.group(1))
            if subtitle:
                aliases.add(subtitle)

    acronym_match = re.search(r"\(([A-Z]{2,})\)", head)
    if acronym_match:
        aliases.add(acronym_match.group(1))

    aliases = {alias for alias in aliases if alias and len(alias) > 2}
    return sorted(aliases)


def prepare_docling_artifacts(work_dir: Path, pdf_dir: Path) -> Dict[str, Path]:
    docling_dir = work_dir / "docling"
    parsed_dir = docling_dir / "parsed"
    merged_dir = docling_dir / "merged"
    chunked_dir = docling_dir / "chunked"

    total_pdfs = len(list(pdf_dir.glob("*.pdf")))
    existing = len(list(parsed_dir.glob("*.json"))) if parsed_dir.exists() else 0
    if existing < total_pdfs:
        parsed_dir.mkdir(parents=True, exist_ok=True)
        parser = PDFParser(output_dir=parsed_dir)
        parser.parse_and_export_parallel(doc_dir=pdf_dir, optimal_workers=4, chunk_size=4)

    if len(list(merged_dir.glob("*.json"))) < total_pdfs:
        PageTextPreparation().process_reports(reports_dir=parsed_dir, output_dir=merged_dir)

    if len(list(chunked_dir.glob("*.json"))) < total_pdfs:
        TextSplitter().split_all_reports(merged_dir, chunked_dir)

    return {"parsed_dir": parsed_dir, "merged_dir": merged_dir, "chunked_dir": chunked_dir}


def load_corpus(chunked_dir: Path) -> tuple[Dict[str, DocumentMeta], List[ChunkMeta]]:
    documents: Dict[str, DocumentMeta] = {}
    chunks: List[ChunkMeta] = []

    for path in sorted(chunked_dir.glob("*.json")):
        payload = safe_json_load(path)
        sha = payload["metainfo"]["sha1_name"]
        pages = payload["content"]["pages"]
        first_page_text = pages[0]["text"] if pages else ""
        case_ids = extract_case_ids(first_page_text)
        kind = infer_document_kind(first_page_text, case_ids)
        if kind == "case":
            canonical_ids = sorted(dict.fromkeys(case_ids))
        elif kind in {"law", "regulation"}:
            canonical_ids = extract_primary_law_ids(first_page_text)
        else:
            canonical_ids = []
        title = infer_document_title(first_page_text, canonical_ids, kind=kind)
        aliases = infer_aliases(title, canonical_ids, first_page_text)
        head_upper = first_page_text.upper()
        documents[sha] = DocumentMeta(
            sha=sha,
            kind=kind,
            title=title,
            first_page_text=first_page_text,
            aliases=aliases,
            canonical_ids=canonical_ids,
            page_count=len(pages),
            is_enactment_notice="ENACTMENT NOTICE" in head_upper,
            is_consolidated="CONSOLIDATED VERSION" in head_upper,
        )
        for chunk in payload["content"]["chunks"]:
            ref = f"{sha}:{chunk['page']}:{chunk['id']}"
            chunks.append(
                ChunkMeta(
                    ref=ref,
                    sha=sha,
                    page=int(chunk["page"]),
                    chunk_id=int(chunk["id"]),
                    text=chunk["text"],
                    title=title,
                    kind=kind,
                    canonical_ids=canonical_ids,
                )
            )

    return documents, chunks


class PublicCorpus:
    def __init__(
        self,
        work_dir: Path,
        pdf_dir: Path,
        chunked_dir: Path,
        embedding_model: str = DEFAULT_LOCAL_EMBEDDING_MODEL,
        reranker_model: str = DEFAULT_LOCAL_RERANKER_MODEL,
        embedder: Any | None = None,
        reranker: Any | None = None,
    ):
        self.work_dir = work_dir
        self.pdf_dir = pdf_dir
        self.chunked_dir = chunked_dir
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.documents, self.chunks = load_corpus(chunked_dir)
        self.documents_payload = {sha: safe_json_load(chunked_dir / f"{sha}.json") for sha in self.documents}
        self.id_index = defaultdict(list)
        self.alias_index = defaultdict(list)
        for sha, document in self.documents.items():
            for canonical_id in document.canonical_ids:
                self.id_index[canonical_id].append(sha)
            for alias in document.aliases:
                self.alias_index[normalized_text(alias)].append(sha)
        self.embedder = embedder or LocalEmbeddingModel(self.embedding_model)
        self.reranker = reranker or LocalJinaReranker(self.reranker_model)
        self.chunk_refs = [chunk.ref for chunk in self.chunks]
        self.chunk_ref_to_meta = {chunk.ref: chunk for chunk in self.chunks}
        self.chunk_embeddings = self._load_or_build_embeddings()
        self.bm25_index = BM25Okapi([tokenize_for_bm25(chunk.text) for chunk in self.chunks])

    def _load_or_build_embeddings(self) -> np.ndarray:
        cache_dir = self.work_dir / "index"
        cache_dir.mkdir(parents=True, exist_ok=True)
        embedder_cache_key = getattr(self.embedder, "cache_key", self.embedding_model)
        cache_slug = re.sub(r"[^a-zA-Z0-9]+", "_", embedder_cache_key).strip("_").lower()
        refs_signature = hashlib.sha1("\n".join(self.chunk_refs).encode("utf-8")).hexdigest()[:12]
        text_signature = hashlib.sha1(
            "\n".join(f"{chunk.ref}\n{chunk.text}" for chunk in self.chunks).encode("utf-8")
        ).hexdigest()[:12]
        meta_path = cache_dir / f"{cache_slug}__{refs_signature}__{text_signature}__chunk_metadata.json"
        emb_path = cache_dir / f"{cache_slug}__{refs_signature}__{text_signature}__chunk_embeddings.npy"
        current_refs = self.chunk_refs
        current_text_signature = text_signature
        if meta_path.exists() and emb_path.exists():
            cached_meta = safe_json_load(meta_path)
            if (
                cached_meta.get("refs") == current_refs
                and cached_meta.get("embedding_model") == embedder_cache_key
                and cached_meta.get("text_signature") == current_text_signature
            ):
                return np.load(emb_path)

        texts = [chunk.text for chunk in self.chunks]
        embeddings = self.embedder.embed_documents(texts)
        np.save(emb_path, embeddings)
        safe_json_dump(
            meta_path,
            {
                "refs": current_refs,
                "embedding_model": embedder_cache_key,
                "text_signature": current_text_signature,
            },
        )
        return embeddings

    def save_catalog(self) -> None:
        safe_json_dump(
            self.work_dir / "doc_catalog.json",
            {
                "documents": [asdict(document) for document in self.documents.values()],
                "chunk_count": len(self.chunks),
            },
        )

    def route_question(self, question: str, expansive: bool = False) -> Dict[str, Any]:
        explicit_case_ids = extract_case_ids(question)
        explicit_law_ids = extract_law_ids(question)
        question_norm = normalized_text(question)
        aliases_hit: List[str] = []
        candidate_shas: List[str] = []

        for canonical_id in explicit_case_ids + explicit_law_ids:
            candidate_shas.extend(self.id_index.get(canonical_id, []))

        for alias_norm, shas in self.alias_index.items():
            if len(alias_norm) < 6:
                continue
            if re.search(rf"\b{re.escape(alias_norm)}\b", question_norm):
                candidate_shas.extend(shas)
                aliases_hit.append(alias_norm)

        if expansive:
            if not candidate_shas and ("law" in question_norm or "regulation" in question_norm):
                candidate_shas.extend([sha for sha, document in self.documents.items() if document.kind in {"law", "regulation"}])
            elif not candidate_shas:
                candidate_shas.extend(self.documents.keys())
        else:
            if not candidate_shas and explicit_case_ids:
                candidate_shas.extend([sha for sha, document in self.documents.items() if document.kind == "case"])
            elif not candidate_shas and ("law" in question_norm or "regulation" in question_norm or "article" in question_norm):
                candidate_shas.extend([sha for sha, document in self.documents.items() if document.kind in {"law", "regulation"}])
            elif not candidate_shas:
                candidate_shas.extend(self.documents.keys())

        deduped = sorted(dict.fromkeys(candidate_shas))
        return {
            "candidate_shas": deduped,
            "explicit_case_ids": explicit_case_ids,
            "explicit_law_ids": explicit_law_ids,
            "alias_hits": sorted(set(aliases_hit)),
        }

    def _candidate_chunk_indices(self, candidate_shas: Sequence[str]) -> List[int]:
        candidate_sha_set = set(candidate_shas)
        return [index for index, chunk in enumerate(self.chunks) if chunk.sha in candidate_sha_set]

    def _lexical_refs(self, question: str, candidate_shas: Sequence[str]) -> List[str]:
        refs: List[str] = []
        phrases = extract_quoted_phrases(question)
        phrases.extend(extract_article_refs(question))
        if not phrases:
            law_ids = extract_law_ids(question)
            if law_ids:
                phrases.extend(law_ids)

        question_tokens = [token for token in re.findall(r"[A-Za-z0-9/]+", question) if token.lower() not in STOPWORDS]
        long_tokens = [token for token in question_tokens if len(token) > 7]
        phrases.extend(long_tokens[:3])
        phrases = [normalized_text(phrase) for phrase in phrases if normalized_text(phrase)]

        candidate_sha_set = set(candidate_shas)
        for chunk in self.chunks:
            if chunk.sha not in candidate_sha_set:
                continue
            chunk_norm = normalized_text(chunk.text)
            if any(phrase in chunk_norm for phrase in phrases):
                refs.append(chunk.ref)
        return refs[:24]

    def retrieve(
        self,
        question: str,
        candidate_shas: Sequence[str],
        vector_k: int,
        rerank_k: int,
        lexical_boost: bool = False,
        bm25_k: int = 0,
        bm25_weight: float = 0.0,
        bm25_auto: bool = False,
        adaptive_bm25_weight: float = 0.25,
        adaptive_bm25_max_novel: int = 4,
    ) -> Dict[str, Any]:
        candidate_indices = self._candidate_chunk_indices(candidate_shas)
        if not candidate_indices:
            raise ValueError("No candidate chunks found for question routing.")

        query_embedding = self.embedder.embed_query(question)
        candidate_embeddings = self.chunk_embeddings[candidate_indices]
        scores = candidate_embeddings @ query_embedding
        top_count = min(vector_k, len(candidate_indices))
        top_order = np.argsort(scores)[::-1][:top_count]
        vector_results: List[Dict[str, Any]] = []
        for rank in top_order:
            chunk_index = candidate_indices[int(rank)]
            chunk = self.chunks[chunk_index]
            vector_results.append(
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

        bm25_results: List[Dict[str, Any]] = []
        should_use_bm25 = bm25_k > 0 and bm25_weight > 0
        if should_use_bm25:
            bm25_scores = self.bm25_index.get_scores(tokenize_for_bm25(question))
            ranked_indices = sorted(candidate_indices, key=lambda idx: bm25_scores[idx], reverse=True)[: min(bm25_k, len(candidate_indices))]
            for chunk_index in ranked_indices:
                chunk = self.chunks[chunk_index]
                bm25_results.append(
                    {
                        "ref": chunk.ref,
                        "distance": round(float(bm25_scores[chunk_index]), 4),
                        "page": chunk.page,
                        "text": chunk.text,
                        "sha": chunk.sha,
                        "title": chunk.title,
                        "kind": chunk.kind,
                        "canonical_ids": chunk.canonical_ids,
                    }
                )
            if bm25_auto:
                bm25_results = select_novel_results(
                    primary_results=vector_results,
                    secondary_results=bm25_results,
                    max_new_results=min(adaptive_bm25_max_novel, bm25_k),
                )

        retrieval_results = vector_results

        if lexical_boost:
            lexical_refs = self._lexical_refs(question, candidate_shas)
            seen_refs = {result["ref"] for result in retrieval_results}
            for ref in lexical_refs:
                if ref in seen_refs:
                    continue
                chunk = self.chunk_ref_to_meta[ref]
                retrieval_results.append(
                    {
                        "ref": ref,
                        "distance": 0.95,
                        "page": chunk.page,
                        "text": chunk.text,
                        "sha": chunk.sha,
                        "title": chunk.title,
                        "kind": chunk.kind,
                        "canonical_ids": chunk.canonical_ids,
                    }
                )

        if bm25_results:
            retrieval_results = reciprocal_rank_fuse(
                [
                    ("vector", retrieval_results, 1.0),
                    ("bm25", bm25_results, adaptive_bm25_weight if bm25_auto else bm25_weight),
                ]
            )

        reranked = self.reranker.rerank_documents(question, retrieval_results, llm_weight=0.7)[:rerank_k]
        return {"vector_results": retrieval_results, "reranked_results": reranked, "bm25_results": bm25_results}


def build_context(
    chunks: Sequence[Dict[str, Any]],
    analysis: "QuestionAnalysis | None" = None,
    question: str = "",
) -> str:
    question_norm = normalize_space(question).lower()
    comparison_markers = (" both ", " between ", " compare ", " common ", " same ", " and ")
    unique_shas = ordered_unique(chunk.get("sha") for chunk in chunks if chunk.get("sha"))
    multi_source = (
        (analysis is not None and analysis.needs_multi_document_support)
        or (
            len(unique_shas) >= 2
            and any(marker in f" {question_norm} " for marker in comparison_markers)
        )
    )

    if not multi_source:
        parts = []
        for chunk in chunks:
            ids = ", ".join(chunk.get("canonical_ids") or [])
            meta = f"REF {chunk['ref']} | title={chunk['title']} | page={chunk['page']}"
            if ids:
                meta += f" | ids={ids}"
            parts.append(f"[{meta}]\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    grouped: Dict[str, list[Dict[str, Any]]] = {}
    title_by_sha: Dict[str, str] = {}
    ids_by_sha: Dict[str, str] = {}
    for chunk in chunks:
        sha = str(chunk.get("sha") or "")
        if not sha:
            continue
        grouped.setdefault(sha, []).append(chunk)
        title_by_sha.setdefault(sha, str(chunk.get("title") or "Unknown"))
        ids_by_sha.setdefault(sha, ", ".join(chunk.get("canonical_ids") or []))

    parts = []
    for source_index, sha in enumerate(unique_shas, start=1):
        source_chunks = grouped.get(sha, [])
        if not source_chunks:
            continue
        header = f"SOURCE {source_index} | title={title_by_sha.get(sha, 'Unknown')}"
        if ids_by_sha.get(sha):
            header += f" | ids={ids_by_sha[sha]}"
        chunk_parts = []
        for chunk in source_chunks:
            chunk_parts.append(
                f"[REF {chunk['ref']} | page={chunk['page']}]\n{chunk['text']}"
            )
        parts.append(f"[{header}]\n" + "\n\n".join(chunk_parts))
    return "\n\n===\n\n".join(parts)


def parse_tagged_field(text: str, field_name: str, next_fields: Sequence[str]) -> str:
    next_pattern = "|".join(re.escape(field) for field in next_fields)
    if next_pattern:
        pattern = rf"{field_name}:\s*(.*?)(?=\n(?:{next_pattern}):|\Z)"
    else:
        pattern = rf"{field_name}:\s*(.*)\Z"
    match = re.search(pattern, text, re.I | re.S)
    if match:
        return normalize_space(match.group(1))
    return ""


def parse_answer_text(answer_type: str, response_text: str) -> Dict[str, Any]:
    text = response_text.strip().replace("```", "")
    raw_answer = parse_tagged_field(text, "ANSWER", ["CITATIONS", "REASONING"])
    citations_text = parse_tagged_field(text, "CITATIONS", ["REASONING"])
    reasoning = parse_tagged_field(text, "REASONING", [])
    citations = [normalize_space(part) for part in re.split(r",|;", citations_text) if normalize_space(part)]
    if not raw_answer:
        lines = [normalize_space(line) for line in text.splitlines() if normalize_space(line)]
        raw_answer = lines[0] if lines else ""
    citation_limit = 5 if answer_type == "free_text" else 3
    return {
        "raw_answer": raw_answer,
        "normalized_answer": normalize_answer(answer_type, raw_answer),
        "citations": citations[:citation_limit],
        "reasoning": reasoning,
        "confidence": "medium",
    }


def parse_judge_text(response_text: str) -> Dict[str, Any]:
    text = response_text.strip().replace("```", "")
    verdict = parse_tagged_field(text, "VERDICT", ["SCORE", "EXPLANATION"]).lower() or "incorrect"
    score_text = parse_tagged_field(text, "SCORE", ["EXPLANATION"])
    explanation = parse_tagged_field(text, "EXPLANATION", [])
    score = normalize_number(score_text)
    if verdict == "correct":
        score = 1.0 if score is None else float(score)
    elif verdict == "partial":
        score = 0.5 if score is None else float(score)
    else:
        score = 0.0
    return {"verdict": verdict, "score": score, "explanation": explanation}


def compress_free_text_answer(text: str, limit: int = 320) -> str:
    text = normalize_space(text)
    if not text:
        return FREE_TEXT_ABSENCE_ANSWER
    replacements = (
        (r"\bsix\s*\(6\)", "six"),
        (r"\btwelve\s*\(12\)", "twelve"),
        (r"\bat least six(?:\s*\(6\))? years from the date they were created\b", "at least six years from creation"),
        (
            r"\bat least six(?:\s*\(6\))? years from (?:the date they were created|creation),\s*or\s*for\s*(?:another|some other)\s+period\s+as\s+may\s+be\s+prescribed\s+in\s+the\s+Regulations\b",
            "at least six years from creation, or longer if Regulations prescribe",
        ),
        (
            r"\bor\s+for\s+(?:another|some other)\s+period\s+as\s+may\s+be\s+prescribed\s+in\s+the\s+Regulations\b",
            "or longer if Regulations prescribe",
        ),
        (r"\bor\s+or\s+longer if Regulations prescribe\b", "or longer if Regulations prescribe"),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.I)
    text = normalize_space(text)
    if len(text) <= limit:
        return text

    for separator in (". ", "; ", ": ", ", "):
        cut = text.rfind(separator, 0, limit + 1)
        if cut >= 140:
            trimmed = text[:cut].rstrip(" ,;:.")
            if trimmed and trimmed[-1] not in ".!?":
                trimmed += "."
            return trimmed

    cut = text.rfind(" ", 0, limit - 3)
    if cut == -1:
        cut = limit
    trimmed = text[:cut].rstrip(" ,;:.")
    if trimmed and trimmed[-1] not in ".!?":
        trimmed += "."
    return trimmed


def support_query_tokens(question: str, analysis: "QuestionAnalysis | None" = None) -> List[str]:
    values = [question]
    if analysis is not None:
        values.extend(analysis.must_support_terms)
        values.extend(analysis.target_titles)
        values.extend(analysis.target_article_refs)
    tokens: List[str] = []
    for value in values:
        for token in re.findall(r"[a-z0-9/.-]+", normalize_space(value).lower()):
            if len(token) <= 3 or token in STOPWORDS:
                continue
            tokens.append(token)
    return ordered_unique(tokens)[:20]


def extract_support_units(text: str) -> List[str]:
    units: List[str] = []
    for block in re.split(r"\n{2,}", text or ""):
        block = normalize_space(block)
        if not block:
            continue
        if len(block) <= 260:
            units.append(block)
            continue
        pieces = re.split(r"(?<=[\.\?\!;:])\s+(?=[A-Z0-9(\[])", block)
        for piece in pieces:
            piece = normalize_space(piece)
            if len(piece) >= 28:
                units.append(piece)
    return ordered_unique(unit for unit in units if len(unit) >= 24)[:12]


def score_support_unit(
    unit: str,
    *,
    question: str,
    analysis: "QuestionAnalysis | None" = None,
) -> float:
    question_norm = normalize_space(question).lower()
    unit_norm = normalize_space(unit).lower()
    score = 0.0
    tokens = support_query_tokens(question, analysis)
    score += 1.5 * sum(1 for token in tokens if token in unit_norm)
    if "cost" in question_norm and any(
        marker in unit_norm
        for marker in (
            "## costs",
            "costs",
            "statement of costs",
            "award of costs",
            "entitled to its costs",
            "pay ",
            "paid by",
        )
    ):
        score += 10.0
    if analysis is not None:
        score += 6.0 * sum(1 for term in analysis.must_support_terms if normalize_space(term).lower() in unit_norm)
        if analysis.target_field == "order_result":
            if "it is hereby ordered that" in unit_norm:
                score += 16.0
            if "conclusion" in unit_norm:
                score += 8.0
            if any(marker in unit_norm for marker in ("refused", "dismissed", "granted", "allowed", "discharged")):
                score += 6.0
        if analysis.target_field in {"article_content", "clause_summary"}:
            for article_ref in analysis.target_article_refs:
                ref_norm = normalize_space(article_ref).lower()
                if ref_norm and ref_norm in unit_norm:
                    score += 14.0
                number_match = re.search(r"article\s+(\d+)", ref_norm)
                if number_match and re.search(rf"(?:^|\b){number_match.group(1)}\.", unit_norm):
                    score += 10.0
        if analysis.target_field in {"made_by", "administered_by", "publication_text", "enacted_text", "effective_dates"}:
            for marker in ("this law is made by", "administered by", "consolidated version", "this law is enacted on", "effective date"):
                if marker in unit_norm:
                    score += 8.0
    if len(unit_norm) < 35:
        score -= 1.0
    return score


def build_focused_free_text_context(
    chunks: Sequence[Dict[str, Any]],
    *,
    question: str,
    analysis: "QuestionAnalysis | None" = None,
    multi_source_context: bool = False,
) -> str:
    scored_units: List[tuple[float, str, Dict[str, Any], str]] = []
    for chunk in chunks:
        for unit in extract_support_units(str(chunk.get("text", "") or "")):
            score = score_support_unit(unit, question=question, analysis=analysis)
            if score <= 0:
                continue
            scored_units.append((score, str(chunk.get("sha") or ""), chunk, unit))

    if not scored_units:
        return build_context(chunks, analysis=analysis, question=question)

    scored_units.sort(key=lambda item: (-item[0], int(item[2].get("page", 0)), str(item[2].get("ref", ""))))
    if not multi_source_context:
        parts = []
        seen_refs = set()
        for _, _, chunk, unit in scored_units:
            ref = str(chunk.get("ref") or "")
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            ids = ", ".join(chunk.get("canonical_ids") or [])
            meta = f"REF {ref} | title={chunk['title']} | page={chunk['page']}"
            if ids:
                meta += f" | ids={ids}"
            parts.append(f"[{meta}]\n{unit}")
            if len(parts) >= 5:
                break
        return "\n\n---\n\n".join(parts)

    grouped_units: Dict[str, List[tuple[float, Dict[str, Any], str]]] = defaultdict(list)
    for score, sha, chunk, unit in scored_units:
        grouped_units[sha].append((score, chunk, unit))

    parts = []
    source_index = 0
    for sha, items in grouped_units.items():
        if not items:
            continue
        source_index += 1
        title = str(items[0][1].get("title") or "Unknown")
        ids = ", ".join(items[0][1].get("canonical_ids") or [])
        header = f"SOURCE {source_index} | title={title}"
        if ids:
            header += f" | ids={ids}"
        chunk_parts = []
        seen_refs = set()
        for _, chunk, unit in items:
            ref = str(chunk.get("ref") or "")
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            chunk_parts.append(f"[REF {ref} | page={chunk['page']}]\n{unit}")
            if len(chunk_parts) >= 2:
                break
        if chunk_parts:
            parts.append(f"[{header}]\n" + "\n\n".join(chunk_parts))
    return "\n\n===\n\n".join(parts) or build_context(chunks, analysis=analysis, question=question)


def cited_chunks_from_payload(
    chunks: Sequence[Dict[str, Any]],
    answer_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    chunk_by_ref = {str(chunk.get("ref") or ""): chunk for chunk in chunks}
    selected: List[Dict[str, Any]] = []
    for citation in answer_payload.get("citations", []):
        citation_ref = normalize_space(str(citation or ""))
        if citation_ref in chunk_by_ref:
            selected.append(chunk_by_ref[citation_ref])
    return selected


def resolve_answer_citations(
    citations: Sequence[str],
    chunks: Sequence[Dict[str, Any]],
) -> List[str]:
    resolved: List[str] = []
    chunk_by_ref = {str(chunk.get("ref") or ""): chunk for chunk in chunks}
    first_chunk_by_sha: Dict[str, str] = {}
    first_chunk_by_page: Dict[tuple[str, int], str] = {}
    for chunk in chunks:
        ref = str(chunk.get("ref") or "")
        sha = str(chunk.get("sha") or "")
        page = int(chunk.get("page") or 0)
        if sha and sha not in first_chunk_by_sha:
            first_chunk_by_sha[sha] = ref
        if sha and page and (sha, page) not in first_chunk_by_page:
            first_chunk_by_page[(sha, page)] = ref
    for citation in citations:
        citation_text = normalize_space(str(citation or "")).strip("[]")
        prefixed_ref = re.fullmatch(r"ref\s+([a-f0-9]{32,64}(?::\d+(?::\d+)?)?)", citation_text, re.I)
        if prefixed_ref:
            citation_text = prefixed_ref.group(1)
        if citation_text in chunk_by_ref:
            resolved.append(citation_text)
            continue
        placeholder = re.fullmatch(r"ref\s*(\d+)", citation_text, re.I)
        if placeholder:
            index = int(placeholder.group(1)) - 1
            if 0 <= index < len(chunks):
                resolved.append(str(chunks[index].get("ref") or citation_text))
                continue
        page_like = re.fullmatch(r"([a-f0-9]{32,64}):(\d+)", citation_text, re.I)
        if page_like:
            mapped = first_chunk_by_page.get((page_like.group(1), int(page_like.group(2))))
            if mapped:
                resolved.append(mapped)
                continue
        if citation_text in first_chunk_by_sha:
            resolved.append(first_chunk_by_sha[citation_text])
            continue
        resolved.append(citation_text)
    return [citation for citation in ordered_unique(resolved) if citation]


def citation_specificity_score(citations: Sequence[str]) -> tuple[int, int]:
    chunk_specific = 0
    page_specific = 0
    for citation in citations:
        ref = normalize_space(str(citation or ""))
        colon_count = ref.count(":")
        if colon_count >= 2:
            chunk_specific += 1
        elif colon_count == 1:
            page_specific += 1
    return chunk_specific, page_specific


def prefer_draft_citations(
    draft_citations: Sequence[str],
    refined_citations: Sequence[str],
) -> bool:
    if not draft_citations:
        return False
    if not refined_citations:
        return True
    draft_chunk, draft_page = citation_specificity_score(draft_citations)
    refined_chunk, refined_page = citation_specificity_score(refined_citations)
    if refined_chunk < draft_chunk:
        return True
    if refined_chunk == draft_chunk and refined_page < draft_page:
        return True
    if refined_chunk == draft_chunk and refined_page == draft_page and len(refined_citations) < len(draft_citations):
        return True
    return False


def merge_response_data(*payloads: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    input_tokens = 0
    output_tokens = 0
    model_name = None
    for payload in payloads:
        if not payload:
            continue
        input_tokens += int(payload.get("input_tokens", 0) or 0)
        output_tokens += int(payload.get("output_tokens", 0) or 0)
        if payload.get("model"):
            model_name = payload.get("model")
    if model_name:
        merged["model"] = model_name
    merged["input_tokens"] = input_tokens
    merged["output_tokens"] = output_tokens
    return merged


def should_refine_free_text_answer(
    answer_payload: Dict[str, Any],
    chunks: Sequence[Dict[str, Any]],
    analysis: "QuestionAnalysis | None" = None,
    multi_source_context: bool = False,
) -> bool:
    raw_answer = normalize_space(str(answer_payload.get("raw_answer", "") or ""))
    if not raw_answer or raw_answer == FREE_TEXT_ABSENCE_ANSWER:
        return False
    if len(raw_answer) > 210:
        return True
    if raw_answer.count(".") >= 2 or raw_answer.count(";") >= 1:
        return True
    cited_chunks = cited_chunks_from_payload(chunks, answer_payload)
    if not cited_chunks:
        return True
    if multi_source_context:
        cited_shas = ordered_unique(chunk.get("sha") for chunk in cited_chunks if chunk.get("sha"))
        if len(cited_shas) < 2:
            return True
    if analysis is None:
        return False
    if analysis.target_field in {
        "order_result",
        "article_content",
        "clause_summary",
        "comparison_answer",
        "enumeration_answer",
        "absence_check",
        "effective_dates",
        "made_by",
        "administered_by",
        "publication_text",
        "enacted_text",
    }:
        return True
    return False


def free_text_refine_prompt(
    analysis: "QuestionAnalysis | None" = None,
    multi_source_context: bool = False,
) -> str:
    hints = ""
    if analysis is not None:
        if analysis.target_field == "order_result":
            hints += "Keep only the operative result and any explicit costs direction.\n"
        if analysis.target_field in {"article_content", "clause_summary"}:
            hints += "Stay inside the named article or clause and preserve the exact operative content.\n"
            hints += "If the question asks who made or administers a law under a named article, answer that direct legal fact rather than abstaining.\n"
        if analysis.target_field in {"made_by", "administered_by", "publication_text", "enacted_text", "effective_dates"}:
            hints += "For law-metadata questions, prefer a complete standalone sentence naming the law and the direct factual statement from the evidence.\n"
        if analysis.needs_multi_document_support or multi_source_context:
            hints += "If the question requires multiple sources, the answer must explicitly cover every supported source.\n"
    return (
        "You are revising a DIFC legal assistant answer using only the evidence below.\n"
        "Optimize for these criteria simultaneously: factual correctness, completeness, grounding, calibrated confidence, and clarity.\n"
        "Remove any unsupported detail from the draft answer.\n"
        "Keep the answer legally precise, direct, coherent, and concise; ideally stay within about 320 characters unless the evidence requires slightly more.\n"
        "Prefer 1-2 complete sentences rather than a bare extracted span.\n"
        "If the question explicitly scopes the answer to a page or section (for example, first page, last page, Conclusion, or 'IT IS HEREBY ORDERED THAT'), reflect that scope briefly when it improves clarity.\n"
        "Use natural phrasing such as 'On the last page, ...' rather than meta wording like 'the last page shows' or 'the section shows'.\n"
        "If the question identifies a case, law, regulation, article, or section, name it in the answer when that improves standalone clarity.\n"
        "For court-result questions, state what application, appeal, or order was decided and include any explicit costs direction when relevant.\n"
        "For law-metadata questions, answer as a standalone statement about the named law rather than saying only 'This Law'.\n"
        "Do not replace a clearly supported answer with an abstention.\n"
        "If the evidence supports only part of the question, answer the supported part and explicitly say what is not stated.\n"
        "If the evidence does not answer the question, use a short calibrated absence answer that says the materials do not mention the requested information.\n"
        "If the draft already states the operative answer and the evidence supports it, preserve that answer and only improve wording and completeness.\n"
        f"If the evidence is insufficient, answer exactly: {FREE_TEXT_ABSENCE_ANSWER}\n"
        "Use only exact REF ids copied from the evidence. Do not invent citations.\n"
        f"{hints}"
        "Return exactly these lines and nothing else:\n"
        "ANSWER: ...\n"
        "CITATIONS: ref1, ref2, ref3, ref4, ref5\n"
        "REASONING: one short sentence\n"
    )


def single_source_free_text_prompt(
    analysis: "QuestionAnalysis | None" = None,
) -> str:
    hints = ""
    if analysis is not None and analysis.target_field in {"article_content", "clause_summary"}:
        hints += "Answer only from the named article or clause in this single source.\n"
        hints += "If the clause states who made, administers, empowers, or requires something, answer that direct legal fact explicitly.\n"
    return (
        "You answer a DIFC legal question using only one source block.\n"
        "Do not infer from any other source.\n"
        "If one or more sentences in this source directly answer the question, you must answer from them and cite them.\n"
        "Do not abstain merely because the wording is formal, legal, or split across two nearby sentences.\n"
        "Answer in assistant style: concise, complete for the asked point, and clear.\n"
        "Prefer 1-2 short sentences that directly answer the question, not a bare copied fragment.\n"
        "If the question identifies a law, case, article, or order section, name that subject in the answer when it improves clarity.\n"
        "For result questions, say what was refused, dismissed, granted, or allowed, and include any explicit costs order if it matters.\n"
        "If the context supports only part of the requested point, say exactly which part is supported.\n"
        f"If this source does not answer the question, answer exactly: {FREE_TEXT_ABSENCE_ANSWER}\n"
        "Keep the answer short and factual.\n"
        "Use only exact REF ids copied from the context.\n"
        f"{hints}"
        "Return exactly these lines and nothing else:\n"
        "ANSWER: ...\n"
        "CITATIONS: ref1, ref2, ref3\n"
        "REASONING: one short sentence\n"
    )


def source_free_text_structured_prompt(
    analysis: "QuestionAnalysis | None" = None,
) -> str:
    hints = ""
    if analysis is not None:
        if analysis.target_field == "order_result":
            hints += "Extract only the operative result and any explicit costs direction when present.\n"
        if analysis.target_field in {"article_content", "clause_summary"}:
            hints += "Stay inside the named article or clause and do not paraphrase away the operative obligation, period, or power.\n"
        if analysis.target_field in {"effective_dates", "made_by", "administered_by", "publication_text", "enacted_text"}:
            hints += "Prefer the exact statement from the relevant line rather than a broad summary.\n"
            hints += "A line such as 'This Law is administered by ...' or 'This Law is made by ...' is a direct answer and should not be treated as absent.\n"
    return (
        "You extract a source-grounded final answer fragment from a single DIFC legal source block.\n"
        "Use only the provided source block.\n"
        "Set absent=true only when the source block contains no sentence, clause, or short span that directly answers the question.\n"
        "If the answer is present in legal wording, still answer it rather than abstaining.\n"
        "Preserve exact operative content, but make the fragment readable as a standalone answer.\n"
        "Prefer a complete 1-sentence answer fragment over a bare noun phrase when the source supports it.\n"
        "If the question names a law, case, article, or order section, include that subject when it improves clarity.\n"
        "For article questions, preserve the exact duration, duty, exception, power, or prohibition stated in the article.\n"
        "For comparison questions answered source-by-source, extract the local answer for this source only.\n"
        f"If the source does not answer the question, set absent=true and leave answer_fragment empty.\n"
        "Every citation must be an exact REF id copied from the context.\n"
        f"{hints}"
        "Return JSON only with these fields:\n"
        "{\n"
        '  "answer_fragment": "short standalone answer fragment, usually 60-220 chars",\n'
        '  "citations": ["sha:page:chunk"],\n'
        '  "absent": false,\n'
        '  "reasoning": "one short sentence"\n'
        "}\n"
    )


def contextual_absence_prompt(
    analysis: "QuestionAnalysis | None" = None,
) -> str:
    hints = ""
    if analysis is not None:
        if analysis.target_field == "order_result":
            hints += "If the requested concept is absent, answer directly that the materials do not provide that outcome.\n"
        if analysis.target_field in {"article_content", "clause_summary"}:
            hints += "If the clause points to another schedule, table, notice, or regulation instead of stating the requested detail, say that explicitly.\n"
        if analysis.target_field == "absence_check":
            hints += "Prefer a direct answer like 'There is no information about X in the provided documents.'\n"
    return (
        "You are answering a DIFC legal question from evidence that may not contain the requested fact.\n"
        "Write a short, grounded assistant answer that explains either:\n"
        "1. the requested fact is not mentioned in the provided evidence, or\n"
        "2. the evidence supports only part of the answer and the missing part is not stated.\n"
        "Do not hallucinate missing facts.\n"
        "Prefer a direct answer that names the missing concept, such as 'There is no information about parole hearings in the provided documents.'\n"
        "Do not describe the retrieval process or say 'the retrieved pages discuss ...' unless that contrast is necessary to explain partial support.\n"
        "Keep the answer under 220 characters, in 1-2 sentences, and use only exact REF ids from the evidence.\n"
        f"{hints}"
        "Return JSON only with these fields:\n"
        "{\n"
        '  "answer": "grounded absence or partial-support answer",\n'
        '  "citations": ["sha:page:chunk"],\n'
        '  "reasoning": "one short sentence"\n'
        "}\n"
    )


def free_text_candidate_selector_prompt(
    analysis: "QuestionAnalysis | None" = None,
    multi_source_context: bool = False,
) -> str:
    hints = ""
    if analysis is not None:
        if analysis.target_field == "order_result":
            hints += "For court-result questions, prefer the answer that states the operative disposition clearly and keeps any material costs direction when that direction is part of the asked outcome.\n"
        if analysis.target_field in {"made_by", "administered_by", "publication_text", "enacted_text", "effective_dates"}:
            hints += "For law-metadata questions, prefer a direct factual answer over a roundabout formulation if both are equally grounded.\n"
        if analysis.target_field in {"article_content", "clause_summary"}:
            hints += "For article or clause questions, prefer the answer that preserves the operative legal rule, duration, duty, prohibition, or power most precisely.\n"
        if analysis.needs_multi_document_support or multi_source_context:
            hints += "For multi-source questions, the better answer must explicitly cover every supported source.\n"
    return (
        "You are choosing the better DIFC legal assistant answer between Candidate A and Candidate B using only the evidence.\n"
        "Choose the answer that is better overall on these criteria: correctness, completeness, grounding, calibrated confidence, and clarity/relevance.\n"
        "Do not reward verbosity for its own sake.\n"
        "Penalize unsupported detail, overconfident wording, missing a material limb of the answer, awkward phrasing, and failing to answer the exact question asked.\n"
        "If both candidates are equally correct, choose the clearer and more concise one.\n"
        "Use only the provided evidence. Do not invent facts.\n"
        f"{hints}"
        "Return JSON only with these fields:\n"
        "{\n"
        '  "choice": "A or B",\n'
        '  "reasoning": "one short sentence"\n'
        "}\n"
    )


def select_best_free_text_payload(
    api: APIProcessor,
    model: str,
    question: str,
    candidate_a: Dict[str, Any],
    candidate_b: Dict[str, Any],
    chunks: Sequence[Dict[str, Any]],
    analysis: "QuestionAnalysis | None" = None,
    multi_source_context: bool = False,
) -> Dict[str, Any]:
    answer_a = normalize_space(str(candidate_a.get("raw_answer", "") or ""))
    answer_b = normalize_space(str(candidate_b.get("raw_answer", "") or ""))
    if not answer_a:
        return candidate_b
    if not answer_b:
        return candidate_a
    if normalize_answer("free_text", answer_a) == normalize_answer("free_text", answer_b):
        return candidate_a if len(answer_a) <= len(answer_b) else candidate_b

    selector_chunks = ordered_unique(
        cited_chunks_from_payload(chunks, candidate_a) + cited_chunks_from_payload(chunks, candidate_b)
    )
    chunk_limit = 10 if multi_source_context else 6
    selector_chunks = selector_chunks[:chunk_limit] or list(chunks[:chunk_limit])
    context = build_focused_free_text_context(
        selector_chunks,
        question=question,
        analysis=analysis,
        multi_source_context=multi_source_context,
    )
    response = api.send_message(
        model=model,
        temperature=0.0,
        system_content=free_text_candidate_selector_prompt(
            analysis=analysis,
            multi_source_context=multi_source_context,
        ),
        human_content=(
            f"Question: {question}\n\n"
            f"Candidate A: {answer_a}\n\n"
            f"Candidate B: {answer_b}\n\n"
            f"Evidence:\n{context}"
        ),
        is_structured=True,
        response_format=FreeTextCandidateSelection,
        max_tokens=120,
        request_timeout=60,
    )
    choice = normalize_space(str(response.get("choice", "A") or "A")).upper()
    selected = candidate_b if choice == "B" else candidate_a
    selected = dict(selected)
    selected["response_data"] = merge_response_data(
        candidate_a.get("response_data") or {},
        candidate_b.get("response_data") or {},
        getattr(api.processor, "response_data", {}) or {},
    )
    return selected


def answer_contextual_absence(
    api: APIProcessor,
    model: str,
    question: str,
    chunks: Sequence[Dict[str, Any]],
    analysis: "QuestionAnalysis | None" = None,
) -> Dict[str, Any]:
    context_chunks = list(chunks[:6])
    context = build_focused_free_text_context(
        context_chunks,
        question=question,
        analysis=analysis,
        multi_source_context=bool(analysis and analysis.needs_multi_document_support),
    )
    response = api.send_message(
        model=model,
        temperature=0.0,
        system_content=contextual_absence_prompt(analysis=analysis),
        human_content=f"Question: {question}\n\nEvidence:\n{context}",
        is_structured=True,
        response_format=AbsenceExplanation,
        max_tokens=180,
        request_timeout=60,
    )
    citations = resolve_answer_citations(response.get("citations", []), context_chunks)
    answer = compress_free_text_answer(str(response.get("answer", "") or ""))
    lowered = normalize_space(answer).lower()
    mention_match = re.search(r"\bdoes not mention\s+(.+?)(?:[.?!]|$)", lowered, re.I)
    if mention_match:
        concept = normalize_space(mention_match.group(1)).rstrip(".")
        if concept:
            answer = f"There is no information about {concept} in the provided documents."
    if not answer:
        answer = FREE_TEXT_ABSENCE_ANSWER
    return {
        "raw_answer": answer,
        "normalized_answer": normalize_answer("free_text", answer),
        "citations": citations[:5],
        "reasoning": normalize_space(str(response.get("reasoning", "") or "")),
        "confidence": "high",
        "skip_refine": True,
        "response_data": getattr(api.processor, "response_data", {}) or {},
    }


def order_result_structured_prompt() -> str:
    return (
        "You extract the operative judicial outcome from a single DIFC court source block.\n"
        "Preserve the court's operative verb exactly when it is stated, such as refused, dismissed, granted, allowed, discharged, upheld, restored, or set aside.\n"
        "Do not replace one operative verb with another.\n"
        "If the context contains an operative sentence stating the result, you must extract it and must not abstain.\n"
        "Prefer explicit order language over background reasoning or procedural history.\n"
        "If the question asks about the 'IT IS HEREBY ORDERED THAT' section, answer from that section only.\n"
        "If the question asks for the first-page ruling, prefer the operative statement on the first page over later reasoning.\n"
        "If the same source also states an explicit costs direction relevant to the result, extract it separately.\n"
        "If the order contains more than one operative limb, preserve each limb that directly answers the question.\n"
        "Use only the provided source block.\n"
        "Every citation must be an exact REF id copied from the context.\n"
        "Return JSON only with these fields:\n"
        "{\n"
        '  "disposition": "operative result",\n'
        '  "disposition_citations": ["sha:page:chunk"],\n'
        '  "costs": "explicit costs direction if present",\n'
        '  "costs_citations": ["sha:page:chunk"],\n'
        '  "absent": false,\n'
        '  "reasoning": "one short sentence"\n'
        "}\n"
    )


def cleaned_title_for_answer(title: str) -> str:
    cleaned = normalize_space(re.sub(r"#+", " ", title or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
    return cleaned or "Source"


def source_specific_article_refs(
    analysis: "QuestionAnalysis | None",
    source_title: str,
    source_index: int,
    total_sources: int,
) -> List[str]:
    if analysis is None or not analysis.target_article_refs:
        return []
    title_norm = normalize_space(source_title).lower()
    title_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", title_norm)
        if len(token) > 3 and token not in STOPWORDS
    }
    explicit: List[str] = []
    fallback: List[str] = []
    for article_ref in analysis.target_article_refs:
        ref_norm = normalize_space(article_ref).lower()
        ref_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", ref_norm)
            if len(token) > 3 and token not in STOPWORDS
        }
        if title_tokens and len(title_tokens & ref_tokens) >= min(2, len(title_tokens)):
            explicit.append(article_ref)
        else:
            fallback.append(article_ref)
    if explicit:
        return ordered_unique(explicit)
    if total_sources > 1 and len(fallback) >= total_sources and source_index < len(fallback):
        return [fallback[source_index]]
    return ordered_unique(fallback)


def strip_leading_marker(text: str) -> str:
    return normalize_space(re.sub(r"^(?:[-*]\s*)?(?:\([a-z0-9]+\)|[a-z0-9]+\.)\s*", "", text, flags=re.I))


def score_source_context(
    source_chunks: Sequence[Dict[str, Any]],
    question: str,
    analysis: "QuestionAnalysis | None" = None,
) -> float:
    score = 0.0
    for chunk in source_chunks:
        text = str(chunk.get("text", "") or "")
        for unit in extract_support_units(text):
            score += max(score_support_unit(unit, question=question, analysis=analysis), 0.0)
    return score


def select_primary_source_chunks(
    chunks: Sequence[Dict[str, Any]],
    question: str,
    analysis: "QuestionAnalysis | None" = None,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        sha = str(chunk.get("sha") or "")
        if sha:
            grouped[sha].append(chunk)
    if len(grouped) <= 1:
        return list(chunks)

    preferred_titles = {normalize_space(title).lower() for title in (analysis.target_titles if analysis else [])}
    preferred_case_ids = {normalize_space(value).lower() for value in (analysis.target_case_ids if analysis else [])}
    preferred_law_ids = {normalize_space(value).lower() for value in (analysis.target_law_ids if analysis else [])}
    best_sha = None
    best_score = float("-inf")
    for sha, source_chunks in grouped.items():
        title = normalize_space(str(source_chunks[0].get("title") or "")).lower()
        ids = {normalize_space(value).lower() for value in (source_chunks[0].get("canonical_ids") or [])}
        score = score_source_context(source_chunks, question=question, analysis=analysis)
        if preferred_titles and any(title_part in title for title_part in preferred_titles):
            score += 20.0
        if preferred_case_ids & ids:
            score += 18.0
        if preferred_law_ids & ids:
            score += 18.0
        score += min(len(source_chunks), 3) * 2.0
        if score > best_score:
            best_sha = sha
            best_score = score
    if best_sha is None:
        return list(chunks)
    return grouped[best_sha]


def source_label_for_answer(
    source_chunks: Sequence[Dict[str, Any]],
    analysis: "QuestionAnalysis | None" = None,
) -> str:
    if not source_chunks:
        return "Source"
    canonical_ids = [normalize_space(value) for value in (source_chunks[0].get("canonical_ids") or []) if normalize_space(value)]
    source_title = cleaned_title_for_answer(str(source_chunks[0].get("title", "")))
    if analysis is not None and analysis.target_titles:
        title_norm = source_title.lower()
        for target_title in analysis.target_titles:
            target_norm = normalize_space(target_title).lower()
            if target_norm and (target_norm in title_norm or title_norm in target_norm):
                return normalize_space(target_title)
    if canonical_ids:
        return canonical_ids[0]
    return source_title


def extract_exact_article_clause(
    source_chunks: Sequence[Dict[str, Any]],
    article_ref: str,
    question: str,
) -> tuple[str | None, List[str]]:
    ref_norm = normalize_space(article_ref).lower()
    clause_parts = re.findall(r"\(([^)]+)\)", ref_norm)
    preferred_part = clause_parts[-1].lower() if clause_parts else ""
    scored_lines: List[tuple[float, str, str]] = []
    for chunk in source_chunks:
        lines = [normalize_space(line) for line in str(chunk.get("text", "") or "").splitlines() if normalize_space(line)]
        for line in lines:
            line_norm = line.lower()
            score = 0.0
            if preferred_part and re.match(rf"^(?:[-*]\s*)?\(\s*{re.escape(preferred_part)}\s*\)", line_norm):
                score += 40.0
            if preferred_part and re.match(rf"^(?:[-*]\s*)?{re.escape(preferred_part)}\.", line_norm):
                score += 36.0
            if ref_norm and ref_norm in line_norm:
                score += 24.0
            for token in support_query_tokens(question):
                if token in line_norm:
                    score += 1.2
            if score > 0:
                scored_lines.append((score, strip_leading_marker(line), str(chunk.get("ref") or "")))
    scored_lines.sort(key=lambda item: (-item[0], len(item[1])))
    if not scored_lines:
        return None, []
    best_text = scored_lines[0][1]
    best_ref = scored_lines[0][2]
    return best_text, [best_ref] if best_ref else []


def extract_order_result_support_units(
    source_chunks: Sequence[Dict[str, Any]],
    question: str,
    analysis: "QuestionAnalysis | None" = None,
) -> tuple[tuple[str | None, List[str]], tuple[str | None, List[str]]]:
    def concise_cost_candidate(unit: str) -> str:
        unit_clean = strip_leading_marker(unit)
        unit_norm = normalize_space(unit_clean).lower()
        if "no order as to costs" in unit_norm:
            return "No order as to costs was made."
        own_costs_match = re.search(
            r"the\s+([a-z]+)\s+(?:shall|must)\s+bear\s+its\s+own\s+costs(?:\s+of\s+the\s+application)?",
            unit_clean,
            re.I,
        )
        if own_costs_match:
            role = own_costs_match.group(1).capitalize()
            return f"The {role} was ordered to bear its own costs of the application."
        assessed_match = re.search(
            r"the\s+([a-z]+)\s+(?:shall|must)\s+pay\s+the\s+([a-z]+)'s\s+costs\s+of\s+the\s+application.*?to\s+be\s+assessed",
            unit_clean,
            re.I,
        )
        if assessed_match:
            payer = assessed_match.group(1).capitalize()
            payee = assessed_match.group(2).capitalize()
            return f"The {payer} must pay the {payee}'s costs of the application, to be assessed."
        return unit_clean

    question_norm = normalize_space(question).lower()
    focus = set(analysis.support_focus) if analysis is not None else set()
    page_count = max((int(chunk.get("page") or 0) for chunk in source_chunks), default=1)
    best_disposition: tuple[float, str, str] | None = None
    best_costs: tuple[float, str, str] | None = None
    operative_markers = (
        "permission to appeal is therefore refused",
        "application is dismissed",
        "appeal must be allowed",
        "application is refused",
        "is discharged",
        "is granted",
        "is allowed",
        "is rejected",
        "set aside application is granted",
    )
    cost_markers = (
        "## costs",
        "statement of costs",
        "award of costs",
        "entitled to its costs",
        "costs of the appeal",
        "shall pay",
        "costs are awarded",
        "bear its own costs",
        "no order as to costs",
    )

    for chunk in source_chunks:
        page_number = int(chunk.get("page") or 0)
        text_norm = normalize_space(str(chunk.get("text", "") or "")).lower()
        for unit in extract_support_units(str(chunk.get("text", "") or "")):
            unit_norm = normalize_space(unit).lower()
            disposition_score = 0.0
            costs_score = 0.0

            if any(marker in unit_norm for marker in operative_markers):
                disposition_score += 18.0
            if any(marker in unit_norm for marker in ("refused", "dismissed", "granted", "allowed", "discharged", "restored", "uphold")):
                disposition_score += 10.0
            if "permission to appeal" in question_norm and "permission to appeal" in unit_norm:
                disposition_score += 12.0
            if "appeal" in question_norm and "appeal" in unit_norm:
                disposition_score += 6.0
            if "order_result" == (analysis.target_field if analysis is not None else ""):
                if "it is hereby ordered that" in text_norm:
                    disposition_score += 12.0
                if "conclusion_section" in focus and page_number >= max(1, page_count - 1):
                    disposition_score += 8.0

            if any(marker in unit_norm for marker in cost_markers):
                costs_score += 14.0
            if "cost" in question_norm and any(marker in unit_norm for marker in cost_markers):
                costs_score += 12.0
            if "cost" in question_norm and page_number >= max(1, page_count - 1):
                costs_score += 6.0

            if disposition_score > 0:
                candidate = (disposition_score, strip_leading_marker(unit), str(chunk.get("ref") or ""))
                if best_disposition is None or candidate[0] > best_disposition[0]:
                    best_disposition = candidate
            if costs_score > 0:
                candidate = (costs_score, concise_cost_candidate(unit), str(chunk.get("ref") or ""))
                if best_costs is None or candidate[0] > best_costs[0]:
                    best_costs = candidate

    disposition = (best_disposition[1], [best_disposition[2]]) if best_disposition and best_disposition[1] else (None, [])
    costs = (best_costs[1], [best_costs[2]]) if best_costs and best_costs[1] else (None, [])
    return disposition, costs


def expand_short_disposition(question: str, disposition: str) -> str:
    disposition = normalize_space(disposition)
    if not disposition:
        return disposition
    if len(disposition.split()) > 2:
        return disposition
    question_norm = normalize_space(question).lower()
    lowered = disposition.lower()
    if "permission to appeal" in question_norm:
        return f"The application for permission to appeal was {lowered}."
    if "appeal" in question_norm:
        return f"The appeal was {lowered}."
    if "application" in question_norm:
        return f"The application was {lowered}."
    if "order" in question_norm:
        return f"The order was {lowered}."
    return disposition


def extract_source_free_text_answer(
    api: APIProcessor,
    model: str,
    question: str,
    source_chunks: Sequence[Dict[str, Any]],
    analysis: "QuestionAnalysis | None" = None,
    source_title: str | None = None,
) -> Dict[str, Any] | None:
    if not source_chunks:
        return None

    effective_title = source_title or source_label_for_answer(source_chunks, analysis)
    source_context = build_focused_free_text_context(
        source_chunks,
        question=question,
        analysis=analysis,
        multi_source_context=False,
    )
    scoped_question = f"For the source titled '{effective_title}', answer this question using only this source: {question}"
    response = api.send_message(
        model=model,
        temperature=0.0,
        system_content=source_free_text_structured_prompt(analysis=analysis),
        human_content=f"Question: {scoped_question}\n\nContext:\n{source_context}",
        is_structured=True,
        response_format=SourceFreeTextAnswer,
        max_tokens=220,
        request_timeout=60,
    )
    citations = resolve_answer_citations(response.get("citations", []), source_chunks)
    raw_fragment = normalize_space(str(response.get("answer_fragment", "") or ""))
    absent = bool(response.get("absent")) or not raw_fragment
    if absent:
        return {
            "source_title": effective_title,
            "raw_answer": FREE_TEXT_ABSENCE_ANSWER,
            "citations": [],
            "reasoning": normalize_space(str(response.get("reasoning", "") or "")),
            "absent": True,
            "response_data": getattr(api.processor, "response_data", {}) or {},
        }
    if not citations:
        return None
    raw_answer = compress_free_text_answer(raw_fragment, limit=260)
    return {
        "source_title": effective_title,
        "raw_answer": raw_answer,
        "citations": citations[:3],
        "reasoning": normalize_space(str(response.get("reasoning", "") or "")),
        "absent": False,
        "response_data": getattr(api.processor, "response_data", {}) or {},
    }


def extract_order_result_answer(
    api: APIProcessor,
    model: str,
    question: str,
    source_chunks: Sequence[Dict[str, Any]],
    analysis: "QuestionAnalysis | None" = None,
) -> Dict[str, Any] | None:
    if not source_chunks:
        return None

    source_context = build_focused_free_text_context(
        source_chunks,
        question=question,
        analysis=analysis,
        multi_source_context=False,
    )
    response = api.send_message(
        model=model,
        temperature=0.0,
        system_content=order_result_structured_prompt(),
        human_content=f"Question: {question}\n\nContext:\n{source_context}",
        is_structured=True,
        response_format=OrderResultExtraction,
        max_tokens=260,
        request_timeout=60,
    )
    disposition = normalize_space(str(response.get("disposition", "") or ""))
    costs = normalize_space(str(response.get("costs", "") or ""))
    absent = bool(response.get("absent")) or (not disposition and not costs)
    if absent:
        return None

    disposition_citations = resolve_answer_citations(response.get("disposition_citations", []), source_chunks)
    costs_citations = resolve_answer_citations(response.get("costs_citations", []), source_chunks)
    fallback_disposition, fallback_costs = extract_order_result_support_units(
        source_chunks,
        question=question,
        analysis=analysis,
    )
    if not disposition:
        disposition = fallback_disposition[0] or disposition
        if not disposition_citations:
            disposition_citations = fallback_disposition[1]
    if not costs or len(costs.split()) <= 2:
        costs = fallback_costs[0] or costs
        if not costs_citations:
            costs_citations = fallback_costs[1]
    disposition = expand_short_disposition(question, disposition)
    if not costs_citations and costs:
        for chunk in source_chunks:
            text_norm = normalize_space(str(chunk.get("text", "") or "")).lower()
            if any(
                marker in text_norm
                for marker in (
                    "## costs",
                    "statement of costs",
                    "award of costs",
                    "entitled to its costs",
                    "costs of the appeal",
                    "shall pay",
                    "costs are awarded",
                    "bear its own costs",
                )
            ):
                chunk_ref = str(chunk.get("ref") or "")
                if chunk_ref:
                    costs_citations.append(chunk_ref)
                    break
    citations = ordered_unique(disposition_citations + costs_citations)[:5]
    if not citations or not disposition:
        return None

    question_norm = normalize_space(question).lower()
    include_costs = bool(costs) and (
        "cost" in question_norm
        or "final ruling" in question_norm
        or "result" in question_norm
        or len(disposition) + len(costs) <= 240
    )
    answer_text = disposition
    if include_costs:
        answer_text = compress_free_text_answer(f"{disposition} {costs}")

    return {
        "raw_answer": answer_text,
        "normalized_answer": normalize_answer("free_text", answer_text),
        "citations": citations,
        "reasoning": normalize_space(str(response.get("reasoning", "") or "")),
        "confidence": "medium",
        "skip_refine": True,
        "response_data": getattr(api.processor, "response_data", {}) or {},
    }


def answer_free_text_via_sources(
    api: APIProcessor,
    model: str,
    question: str,
    chunks: Sequence[Dict[str, Any]],
    analysis: "QuestionAnalysis | None" = None,
    multi_source_context: bool = False,
) -> Dict[str, Any] | None:
    if not chunks:
        return None

    if not multi_source_context:
        primary_chunks = select_primary_source_chunks(chunks, question=question, analysis=analysis)
        if analysis is not None and analysis.target_field == "order_result":
            order_payload = extract_order_result_answer(
                api=api,
                model=model,
                question=question,
                source_chunks=primary_chunks,
                analysis=analysis,
            )
            if order_payload is not None:
                return order_payload
        extracted = extract_source_free_text_answer(
            api=api,
            model=model,
            question=question,
            source_chunks=primary_chunks,
            analysis=analysis,
        )
        if extracted is None:
            return None
        if extracted.get("absent"):
            return None
        answer_text = extracted["raw_answer"]
        return {
            "raw_answer": answer_text,
            "normalized_answer": normalize_answer("free_text", answer_text),
            "citations": extracted.get("citations", [])[:5],
            "reasoning": extracted.get("reasoning", ""),
            "confidence": "medium",
            "response_data": extracted.get("response_data", {}),
        }

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    order: List[str] = []
    for chunk in chunks:
        sha = str(chunk.get("sha") or "")
        if not sha:
            continue
        if sha not in grouped:
            order.append(sha)
        grouped[sha].append(chunk)

    source_answers: List[Dict[str, Any]] = []
    usage_payloads: List[Dict[str, Any]] = []
    for sha in order:
        source_chunks = grouped[sha]
        extracted = extract_source_free_text_answer(
            api=api,
            model=model,
            question=question,
            source_chunks=source_chunks,
            analysis=analysis,
            source_title=source_label_for_answer(source_chunks, analysis),
        )
        if extracted is None:
            continue
        usage_payloads.append(extracted.get("response_data", {}) or {})
        if extracted.get("absent"):
            continue
        source_answers.append(extracted)

    expected_sources = 2 if len(order) >= 2 else 1
    if analysis is not None and analysis.needs_multi_document_support:
        expected_sources = max(
            expected_sources,
            min(
                2,
                max(
                    len(analysis.target_titles),
                    len(analysis.target_case_ids) + len(analysis.target_law_ids),
                    1,
                ),
            ),
        )

    if len(source_answers) < expected_sources:
        return None

    answer_parts = [
        f"{item['source_title']}: {item['raw_answer']}"
        for item in source_answers
    ]
    final_answer = compress_free_text_answer("; ".join(answer_parts))
    citations = ordered_unique(
        citation
        for item in source_answers
        for citation in item.get("citations", [])
    )[:5]
    return {
        "raw_answer": final_answer,
        "normalized_answer": normalize_answer("free_text", final_answer),
        "citations": citations,
        "reasoning": "Source-grounded extraction",
        "confidence": "medium",
        "response_data": merge_response_data(*usage_payloads),
    }


def answer_multi_source_article_free_text(
    api: APIProcessor,
    model: str,
    question: str,
    chunks: Sequence[Dict[str, Any]],
    analysis: "QuestionAnalysis | None" = None,
) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    order: List[str] = []
    for chunk in chunks:
        sha = str(chunk.get("sha") or "")
        if not sha:
            continue
        if sha not in grouped:
            order.append(sha)
        grouped[sha].append(chunk)

    source_answers: List[tuple[str, Dict[str, Any]]] = []
    usage_payloads: List[Dict[str, Any]] = []
    for sha in order:
        source_chunks = grouped[sha]
        source_title = cleaned_title_for_answer(source_chunks[0].get("title", ""))
        scoped_article_refs = source_specific_article_refs(
            analysis,
            source_title=source_title,
            source_index=len(source_answers),
            total_sources=len(order),
        )
        exact_answer = None
        exact_citations: List[str] = []
        for scoped_article_ref in scoped_article_refs:
            exact_answer, exact_citations = extract_exact_article_clause(source_chunks, scoped_article_ref, question)
            if exact_answer:
                break
        if exact_answer:
            parsed = {
                "raw_answer": compress_free_text_answer(exact_answer, limit=260),
                "normalized_answer": normalize_answer("free_text", exact_answer),
                "citations": exact_citations,
                "reasoning": "Exact clause extraction",
            }
        else:
            source_context = build_focused_free_text_context(
                source_chunks,
                question=question,
                analysis=analysis,
                multi_source_context=False,
            )
            if scoped_article_refs:
                source_question = (
                    f"For the source titled '{source_title}', answer only from "
                    f"{', '.join(scoped_article_refs)}: {question}"
                )
            else:
                source_question = f"For the source titled '{source_title}', answer only from this source: {question}"
            response = api.send_message(
                model=model,
                temperature=0.0,
                system_content=single_source_free_text_prompt(analysis=analysis),
                human_content=f"Question: {source_question}\n\nContext:\n{source_context}",
                is_structured=False,
                max_tokens=140,
                request_timeout=60,
                stream=False,
            )
            parsed = parse_answer_text("free_text", response)
            parsed["citations"] = resolve_answer_citations(parsed.get("citations", []), source_chunks)
            parsed["raw_answer"] = compress_free_text_answer(parsed["raw_answer"], limit=260)
            parsed["normalized_answer"] = normalize_answer("free_text", parsed["raw_answer"])
            usage_payloads.append(getattr(api.processor, "response_data", {}) or {})
        source_answers.append((source_title, parsed))

    non_absent = [
        (title, parsed)
        for title, parsed in source_answers
        if normalize_space(str(parsed.get("raw_answer", "") or "")) != FREE_TEXT_ABSENCE_ANSWER
    ]
    if not non_absent:
        return {
            "raw_answer": FREE_TEXT_ABSENCE_ANSWER,
            "normalized_answer": normalize_answer("free_text", FREE_TEXT_ABSENCE_ANSWER),
            "citations": [],
            "reasoning": "No source-specific answer found",
            "confidence": "medium",
            "response_data": merge_response_data(*usage_payloads),
        }

    answer_parts = [f"{title}: {parsed['raw_answer']}" for title, parsed in non_absent]
    citations = ordered_unique(
        citation
        for _, parsed in non_absent
        for citation in parsed.get("citations", [])
    )[:5]
    final_answer = compress_free_text_answer("; ".join(answer_parts))
    return {
        "raw_answer": final_answer,
        "normalized_answer": normalize_answer("free_text", final_answer),
        "citations": citations,
        "reasoning": "Aggregated source-specific answers",
        "confidence": "medium",
        "response_data": merge_response_data(*usage_payloads),
    }


def refine_free_text_answer(
    api: APIProcessor,
    model: str,
    question: str,
    draft_payload: Dict[str, Any],
    chunks: Sequence[Dict[str, Any]],
    analysis: "QuestionAnalysis | None" = None,
    multi_source_context: bool = False,
) -> Dict[str, Any]:
    cited_chunks = cited_chunks_from_payload(chunks, draft_payload)
    refine_limit = 5
    if analysis is not None and (
        analysis.needs_multi_document_support
        or analysis.target_field in {"comparison_answer", "enumeration_answer", "amended_laws"}
    ):
        refine_limit = 10
    elif analysis is not None and analysis.target_field in {"order_result", "article_content"}:
        refine_limit = 6
    refine_chunks = cited_chunks[:refine_limit] or list(chunks[: min(refine_limit, 6)])
    context = build_focused_free_text_context(
        refine_chunks,
        question=question,
        analysis=analysis,
        multi_source_context=multi_source_context,
    )
    response = api.send_message(
        model=model,
        temperature=0.0,
        system_content=free_text_refine_prompt(analysis=analysis, multi_source_context=multi_source_context),
        human_content=(
            f"Question: {question}\n\n"
            f"Draft answer: {draft_payload.get('raw_answer', '')}\n\n"
            f"Evidence:\n{context}"
        ),
        is_structured=False,
        max_tokens=180,
        request_timeout=60,
        stream=False,
    )
    parsed = parse_answer_text("free_text", response)
    parsed["raw_answer"] = compress_free_text_answer(parsed["raw_answer"])
    parsed["normalized_answer"] = normalize_answer("free_text", parsed["raw_answer"])
    parsed["response_data"] = merge_response_data(
        draft_payload.get("response_data") or {},
        getattr(api.processor, "response_data", {}) or {},
    )
    return parsed


def answer_prompt(
    answer_type: str,
    analysis: "QuestionAnalysis | None" = None,
    multi_source_context: bool = False,
    question: str = "",
) -> str:
    if answer_type == "free_text":
        analysis_hints = ""
        question_norm = normalize_space(question).lower()
        if analysis is not None:
            focus = set(analysis.support_focus)
            if analysis.target_field == "order_result":
                analysis_hints += (
                    "Answer only with the operative result or disposition. "
                    "Do not add procedural history, reasoning, or unrelated background.\n"
                )
                analysis_hints += (
                    "If the order also makes an explicit costs ruling and the question asks for the outcome or final ruling, "
                    "include that costs ruling.\n"
                )
            if analysis.target_field in {"article_content", "clause_summary"}:
                analysis_hints += (
                    "Answer only from the named article or clause. "
                    "Do not infer from other provisions unless the context explicitly cross-refers.\n"
                )
            if analysis.needs_multi_document_support:
                analysis_hints += (
                    "If the question compares multiple sources, explicitly cover every source required by the question. "
                    "If one side is missing from the context, abstain.\n"
                )
            if analysis.needs_multi_document_support and analysis.target_field in {"article_content", "clause_summary", "comparison_answer"}:
                analysis_hints += (
                    "For multi-source article questions, the ANSWER line must explicitly cover each source block one by one. "
                    "Use the pattern 'Source 1: ... Source 2: ...'. "
                    "State what each named article says before any comparison or summary.\n"
                )
            if "last_page" in focus:
                analysis_hints += "If the question points to the last page, prioritize that page over earlier material.\n"
            if "first_page" in focus or "title_page" in focus:
                analysis_hints += "If the question points to the first or title page, prioritize that page over later material.\n"
            if "conclusion_section" in focus:
                analysis_hints += "If the question points to the conclusion, use the conclusion wording rather than earlier discussion.\n"
            if analysis.must_support_terms:
                terms = "; ".join(analysis.must_support_terms)
                analysis_hints += (
                    "A non-abstain answer is credible only if the context supports these distinctive concepts: "
                    f"{terms}.\n"
                )
        if any(marker in question_norm for marker in ("cost", "costs awarded", "costs were awarded", "what costs")):
            analysis_hints += "If the question asks about costs, include the explicit costs order when the context provides one.\n"
        if any(marker in question_norm for marker in ("result of the application", "outcome of the application", "final ruling", "what did the court decide", "how did the court", "how did the court of appeal rule")):
            analysis_hints += (
                "For result or ruling questions, cover both the operative disposition and any explicit costs direction if the context provides it.\n"
            )
        if "liability" in question_norm:
            analysis_hints += "For liability questions, preserve the exact liability formula and any stated qualifier or exception.\n"
        if "purpose of article" in question_norm or question_norm.startswith("what is the purpose"):
            analysis_hints += "For purpose questions, state the operative purpose or power of the provision in one sentence.\n"
        if any(marker in question_norm for marker in ("retention period", "retention periods", "minimum period", "how long", "preserve their accounting records")):
            analysis_hints += "For period questions, include the exact duration and any express qualifier such as a regulatory exception, if stated.\n"
        if multi_source_context:
            analysis_hints += (
                "The context is grouped by source. The answer is incomplete unless it explicitly covers each required "
                "source block supported by the context.\n"
            )
        return (
            "You answer questions about DIFC legal documents using only the provided context.\n"
            "Return a direct legally precise answer in 1-3 sentences and keep it concise; ideally stay within about 320 characters unless the evidence requires slightly more.\n"
            "Do not add facts, implications, dates, or background unless the question asks for them.\n"
            "If the context contains a direct operative answer, state it directly rather than summarizing surrounding reasoning.\n"
            "Do not abstain when the answer is explicitly stated in the cited context.\n"
            "If the answer is not present in the context, answer exactly: "
            f"{FREE_TEXT_ABSENCE_ANSWER}\n"
            "For comparison questions, mention both sides when supported by the context.\n"
            "Every citation must be an exact REF id copied verbatim from the context, such as "
            "'sha:page:chunk'. Never output placeholder labels like ref1 or ref2.\n"
            f"{analysis_hints}"
            "Return exactly these lines and nothing else:\n"
            "ANSWER: ...\n"
            "CITATIONS: ref1, ref2, ref3, ref4, ref5\n"
            "REASONING: one short sentence\n"
            "Use only REF ids from the context. Use up to 5 citations when needed.\n"
            "Expected answer type: free_text."
        )
    return (
        "You answer questions about DIFC legal documents using only the provided context.\n"
        "If the answer is not present in the context, say that directly.\n"
        "For boolean answers, answer true only when the context clearly supports yes; otherwise answer false.\n"
        "Evaluate the exact proposition asked, not a nearby alternative.\n"
        "Do not answer false merely because the context also contains a different restriction, exception, approval requirement, or branch that applies to another person, circumstance, or option.\n"
        "If the context expressly permits the specific actor or situation named in the question, answer true even when a different actor or alternative case requires extra approval or conditions.\n"
        "For number answers, return only the relevant number without units in the answer field when possible.\n"
        "For names answers, return names separated by ' | ' preserving the names as written in the context.\n"
        "For name/date answers, return a concise string.\n"
        "Return exactly these lines and nothing else:\n"
        "ANSWER: ...\n"
        "CITATIONS: ref1, ref2\n"
        "REASONING: one short sentence\n"
        "Use at most 3 citations. Citations must be REF ids taken from the provided context.\n"
        f"Expected answer type: {answer_type}."
    )


def answer_question(
    api: APIProcessor,
    model: str,
    question: str,
    answer_type: str,
    chunks: Sequence[Dict[str, Any]],
    analysis: "QuestionAnalysis | None" = None,
    stream: bool = False,
) -> Dict[str, Any]:
    question_norm = normalize_space(question).lower()
    comparison_markers = (" both ", " between ", " compare ", " common ", " same ", " and ")
    unique_shas = ordered_unique(chunk.get("sha") for chunk in chunks if chunk.get("sha"))
    multi_source_context = (
        answer_type == "free_text"
        and (
            (analysis is not None and analysis.needs_multi_document_support)
            or (len(unique_shas) >= 2 and any(marker in f" {question_norm} " for marker in comparison_markers))
        )
    )
    context = (
        build_focused_free_text_context(
            chunks,
            question=question,
            analysis=analysis,
            multi_source_context=multi_source_context,
        )
        if answer_type == "free_text"
        else build_context(chunks, analysis=analysis, question=question)
    )
    if answer_type == "free_text":
        max_tokens = 220
    elif answer_type in {"names", "date"}:
        max_tokens = 220
    else:
        max_tokens = 120
    parsed = None
    if answer_type == "free_text":
        parsed = answer_free_text_via_sources(
            api=api,
            model=model,
            question=question,
            chunks=chunks,
            analysis=analysis,
            multi_source_context=multi_source_context,
        )
    if parsed is None and (
        answer_type == "free_text"
        and analysis is not None
        and analysis.needs_multi_document_support
        and analysis.target_field in {"article_content", "clause_summary", "comparison_answer"}
    ):
        parsed = answer_multi_source_article_free_text(
            api=api,
            model=model,
            question=question,
            chunks=chunks,
            analysis=analysis,
        )
    if parsed is None:
        response = api.send_message(
            model=model,
            temperature=0.0,
            system_content=answer_prompt(answer_type, analysis=analysis, multi_source_context=multi_source_context, question=question),
            human_content=f"Question: {question}\n\nContext:\n{context}",
            is_structured=False,
            max_tokens=max_tokens,
            request_timeout=60,
            stream=stream,
        )
        parsed = parse_answer_text(answer_type, response)
        parsed["citations"] = resolve_answer_citations(parsed.get("citations", []), chunks)
    if answer_type == "free_text":
        parsed["response_data"] = parsed.get("response_data") or getattr(api.processor, "response_data", {}) or {}
        parsed["raw_answer"] = compress_free_text_answer(parsed["raw_answer"])
        parsed["normalized_answer"] = normalize_answer(answer_type, parsed["raw_answer"])
        if should_refine_free_text_answer(
            parsed,
            chunks,
            analysis=analysis,
            multi_source_context=multi_source_context,
        ) and not parsed.get("skip_refine"):
            draft_citations = list(parsed.get("citations", []))
            parsed = refine_free_text_answer(
                api=api,
                model=model,
                question=question,
                draft_payload=parsed,
                chunks=chunks,
                analysis=analysis,
                multi_source_context=multi_source_context,
            )
            refined_citations = resolve_answer_citations(parsed.get("citations", []), chunks)
            if prefer_draft_citations(draft_citations, refined_citations):
                parsed["citations"] = draft_citations
            else:
                parsed["citations"] = refined_citations
    else:
        parsed["response_data"] = getattr(api.processor, "response_data", {}) or {}
    return parsed


def judge_free_text(
    api: APIProcessor,
    model: str,
    question: str,
    gold_answer: str,
    predicted_answer: str,
    gold_context_chunks: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    context = build_context(gold_context_chunks[:8])
    prompt = (
        "You are grading a predicted answer against a gold answer using DIFC legal source excerpts.\n"
        "Return verdict as one of: correct, partial, incorrect.\n"
        "A paraphrase is correct if it preserves the legal substance.\n"
        "If the gold answer says the source does not mention something, the predicted answer is correct only if it clearly says the same.\n"
    )
    response = api.send_message(
        model=model,
        temperature=0.0,
        system_content=(
            prompt
            + " Return exactly these lines and nothing else:\n"
            + "VERDICT: correct|partial|incorrect\n"
            + "SCORE: 1 or 0.5 or 0\n"
            + "EXPLANATION: one short sentence\n"
        ),
        human_content=(
            f"Question: {question}\n"
            f"Gold answer: {gold_answer}\n"
            f"Predicted answer: {predicted_answer}\n\n"
            f"Source excerpts:\n{context}"
        ),
        is_structured=False,
        max_tokens=180,
    )
    return parse_judge_text(response)


def gold_support_refs(gold_item: Dict[str, Any], gold_chunks: Sequence[Dict[str, Any]]) -> List[str]:
    cited = [ref for ref in gold_item.get("citations", []) if ref in {chunk["ref"] for chunk in gold_chunks}]
    if cited:
        return cited
    return [chunk["ref"] for chunk in gold_chunks[:3]]


def retrieval_relevance(gold_refs: Sequence[str], gold_shas: Sequence[str], reranked: Sequence[Dict[str, Any]]) -> str:
    top_refs = [item["ref"] for item in reranked[:3]]
    top_shas = [item["sha"] for item in reranked[:8]]
    if any(ref in gold_refs for ref in top_refs):
        return "high"
    if any(sha in gold_shas for sha in top_shas):
        return "medium"
    return "low"


def evaluate_prediction(
    api: APIProcessor,
    judge_model: str,
    question: str,
    answer_type: str,
    gold: Dict[str, Any],
    predicted: Dict[str, Any],
    gold_chunks: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    gold_norm = gold["normalized_answer"]
    pred_norm = predicted["normalized_answer"]

    if answer_type in {"boolean", "number", "name", "names", "date"}:
        correct = answer_match(answer_type, gold_norm, pred_norm)
        return {
            "verdict": "correct" if correct else "incorrect",
            "score": 1.0 if correct else 0.0,
            "explanation": "" if correct else f"Gold={gold_norm!r}; Predicted={pred_norm!r}",
        }

    verdict = judge_free_text(
        api=api,
        model=judge_model,
        question=question,
        gold_answer=str(gold.get("raw_answer")),
        predicted_answer=str(predicted.get("raw_answer")),
        gold_context_chunks=gold_chunks,
    )
    verdict_name = normalized_text(verdict.get("verdict"))
    if verdict_name not in {"correct", "partial", "incorrect"}:
        verdict_name = "incorrect"
    score = float(verdict.get("score", 0.0))
    if verdict_name == "correct" and score < 1.0:
        score = 1.0
    if verdict_name == "partial" and not (0.0 < score < 1.0):
        score = 0.5
    if verdict_name == "incorrect":
        score = 0.0
    return {
        "verdict": verdict_name,
        "score": score,
        "explanation": verdict.get("explanation", ""),
    }


def detect_stage_error(
    routed_shas: Sequence[str],
    gold_shas: Sequence[str],
    retrieval_label: str,
    answer_verdict: str,
) -> Optional[str]:
    if not set(routed_shas).intersection(gold_shas):
        return "routing"
    if retrieval_label == "low":
        return "retrieval"
    if answer_verdict != "correct":
        return "answering"
    return None


def is_rate_limit_error(error: Exception) -> bool:
    text = normalized_text(error)
    return "429" in text or "rate limit" in text


def run_full_eval(
    dataset_path: Path,
    pdf_dir: Path,
    work_dir: Path,
    provider: str,
    answer_model: str,
    gold_provider: str,
    gold_model: Optional[str],
    judge_provider: str,
    judge_model: Optional[str],
) -> Dict[str, Any]:
    artifacts = prepare_docling_artifacts(work_dir, pdf_dir)
    corpus = PublicCorpus(work_dir, pdf_dir, artifacts["chunked_dir"])
    corpus.save_catalog()
    questions = safe_json_load(dataset_path)

    answer_api = APIProcessor(provider=provider)
    gold_api = APIProcessor(provider=gold_provider)
    judge_api = APIProcessor(provider=judge_provider)
    results_path = work_dir / "evaluation_results.json"
    if results_path.exists():
        results = safe_json_load(results_path)
    else:
        results = []
    results = [item for item in results if item.get("stage_error") != "runtime"]
    safe_json_dump(results_path, results)
    completed_ids = {item.get("id") for item in results if item.get("stage_error") != "runtime"}

    for index, item in enumerate(tqdm(questions, desc="Evaluating public_dataset"), start=1):
        if item["id"] in completed_ids:
            continue
        question = item["question"]
        answer_type = item["answer_type"]

        last_error: Optional[Exception] = None
        result: Dict[str, Any]
        for attempt in range(4):
            try:
                gold_route = corpus.route_question(question, expansive=True)
                gold_retrieval = corpus.retrieve(
                    question=question,
                    candidate_shas=gold_route["candidate_shas"],
                    vector_k=24,
                    rerank_k=8,
                    lexical_boost=True,
                )
                gold_answer = answer_question(
                    api=gold_api,
                    model=gold_model,
                    question=question,
                    answer_type=answer_type,
                    chunks=gold_retrieval["reranked_results"],
                )
                gold_refs = gold_support_refs(gold_answer, gold_retrieval["reranked_results"])
                gold_shas = sorted({corpus.chunk_ref_to_meta[ref].sha for ref in gold_refs if ref in corpus.chunk_ref_to_meta})

                system_route = corpus.route_question(question, expansive=False)
                system_retrieval = corpus.retrieve(
                    question=question,
                    candidate_shas=system_route["candidate_shas"],
                    vector_k=16,
                    rerank_k=6,
                    lexical_boost=False,
                )
                system_answer = answer_question(
                    api=answer_api,
                    model=answer_model,
                    question=question,
                    answer_type=answer_type,
                    chunks=system_retrieval["reranked_results"],
                )

                retrieval_label = retrieval_relevance(gold_refs, gold_shas, system_retrieval["reranked_results"])
                answer_eval = evaluate_prediction(
                    api=judge_api,
                    judge_model=judge_model,
                    question=question,
                    answer_type=answer_type,
                    gold=gold_answer,
                    predicted=system_answer,
                    gold_chunks=gold_retrieval["reranked_results"],
                )
                stage_error = detect_stage_error(
                    routed_shas=system_route["candidate_shas"],
                    gold_shas=gold_shas,
                    retrieval_label=retrieval_label,
                    answer_verdict=answer_eval["verdict"],
                )

                result = {
                    "index": index,
                    "id": item["id"],
                    "question": question,
                    "answer_type": answer_type,
                    "gold_route": gold_route,
                    "gold_answer": gold_answer,
                    "gold_support_refs": gold_refs,
                    "gold_reranked": gold_retrieval["reranked_results"],
                    "system_route": system_route,
                    "system_vector_top": system_retrieval["vector_results"][:8],
                    "system_reranked": system_retrieval["reranked_results"],
                    "system_answer": system_answer,
                    "retrieval_relevance": retrieval_label,
                    "answer_eval": answer_eval,
                    "stage_error": stage_error,
                }
                break
            except Exception as err:
                last_error = err
                if is_rate_limit_error(err) and attempt < 3:
                    wait_seconds = 20 * (attempt + 1)
                    print(f"Question-level rate limit retry for item {index} in {wait_seconds} seconds...")
                    time.sleep(wait_seconds)
                    continue
                result = {
                    "index": index,
                    "id": item["id"],
                    "question": question,
                    "answer_type": answer_type,
                    "error": str(err),
                    "answer_eval": {"verdict": "incorrect", "score": 0.0, "explanation": str(err)},
                    "retrieval_relevance": "low",
                    "stage_error": "runtime",
                }
                break
        results = [existing for existing in results if existing.get("id") != item["id"]]
        results.append(result)
        completed_ids.add(item["id"])
        safe_json_dump(results_path, results)

    metrics = summarize_metrics(results)
    safe_json_dump(work_dir / "evaluation_summary.json", metrics)
    extra_questions = {"questions": []}
    safe_json_dump(work_dir / "extra_questions.json", extra_questions)
    return {"results": results, "metrics": metrics, "extra_questions": extra_questions}


def summarize_metrics(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    answer_type_scores = defaultdict(list)
    retrieval_counts = Counter()
    stage_errors = Counter()
    exact_count = 0
    partial_count = 0
    structured_total = 0
    structured_correct = 0

    for result in results:
        answer_type = result["answer_type"]
        score = float(result["answer_eval"]["score"])
        verdict = result["answer_eval"]["verdict"]
        answer_type_scores[answer_type].append(score)
        retrieval_counts[result["retrieval_relevance"]] += 1
        if result["stage_error"]:
            stage_errors[result["stage_error"]] += 1
        if verdict == "correct":
            exact_count += 1
        elif verdict == "partial":
            partial_count += 1
        if answer_type in {"boolean", "number", "name", "names", "date"}:
            structured_total += 1
            if verdict == "correct":
                structured_correct += 1

    total = len(results)
    macro_scores = {key: round(sum(values) / max(1, len(values)), 4) for key, values in answer_type_scores.items()}
    overall_score = round(sum(sum(values) for values in answer_type_scores.values()) / max(1, total), 4)

    return {
        "total_questions": total,
        "overall_score": overall_score,
        "exact_match_count": exact_count,
        "partial_match_count": partial_count,
        "incorrect_count": total - exact_count - partial_count,
        "structured_exact_accuracy": round(structured_correct / max(1, structured_total), 4),
        "retrieval_relevance_distribution": dict(retrieval_counts),
        "stage_error_distribution": dict(stage_errors),
        "answer_type_scores": macro_scores,
    }


def generate_extra_questions(corpus: PublicCorpus, api: APIProcessor, model: str) -> Dict[str, Any]:
    selected_refs = []
    for chunk in corpus.chunks:
        if chunk.kind == "case" and len(selected_refs) < 4:
            selected_refs.append(chunk.ref)
        elif chunk.kind in {"law", "regulation"} and len(selected_refs) < 10:
            selected_refs.append(chunk.ref)
        if len(selected_refs) >= 10:
            break

    selected_chunks = [asdict(corpus.chunk_ref_to_meta[ref]) for ref in selected_refs]
    context = build_context(selected_chunks)
    response = api.send_message(
        model=model,
        temperature=0.0,
        system_content=(
            "Create 8 additional DIFC document QA examples grounded only in the provided excerpts.\n"
            "Use answer types from this set: boolean, number, name, names, free_text, date.\n"
            "Return concise gold answers and cite REF ids."
        ),
        human_content=f"Source excerpts:\n{context}",
        is_structured=True,
        response_format=ExtraQuestionBundle,
        max_tokens=900,
    )
    if response.get("questions"):
        return {"questions": response["questions"]}
    if response.get("qa_examples"):
        return {"questions": response["qa_examples"]}
    return {"questions": []}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the project on public_dataset.json")
    parser.add_argument("--dataset", default="public_dataset.json")
    parser.add_argument("--pdf-dir", default="pdfs")
    parser.add_argument("--work-dir", default="artifacts/public_eval")
    parser.add_argument("--provider", default="sambanova")
    parser.add_argument("--answer-model", default="Meta-Llama-3.3-70B-Instruct")
    parser.add_argument("--gold-provider", default="local")
    parser.add_argument("--gold-model", default="")
    parser.add_argument("--judge-provider", default="local")
    parser.add_argument("--judge-model", default="")
    args = parser.parse_args()

    result = run_full_eval(
        dataset_path=Path(args.dataset),
        pdf_dir=Path(args.pdf_dir),
        work_dir=Path(args.work_dir),
        provider=args.provider,
        answer_model=args.answer_model,
        gold_provider=args.gold_provider,
        gold_model=args.gold_model or None,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model or None,
    )
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
