from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


AnswerKind = Literal["boolean", "number", "name", "names", "date", "free_text", "null"]
OperationKind = Literal[
    "extract_scalar",
    "extract_set",
    "compare_entities",
    "locate_clause",
    "summarize_evidence",
    "detect_absence",
]
EntityKind = Literal["case", "law", "regulation", "article", "party", "judge", "unknown"]


class EvidenceRef(BaseModel):
    doc_id: str
    page_number: int
    chunk_ref: Optional[str] = None
    role: Literal["primary", "supporting", "comparison", "negative"] = "primary"
    score: float = 0.0


class EntityAnchor(BaseModel):
    kind: EntityKind
    canonical_id: Optional[str] = None
    surface_form: str
    confidence: float = 0.0


class SlotRequirement(BaseModel):
    name: str
    description: str
    required: bool = True
    cardinality: Literal["one", "many"] = "one"


class TaskContract(BaseModel):
    operation: OperationKind
    answer_kind: AnswerKind
    slots: List[SlotRequirement] = Field(default_factory=list)
    anchors: List[EntityAnchor] = Field(default_factory=list)
    needs_multi_document_support: bool = False
    page_focus: List[Literal["title_page", "first_page", "second_page", "last_page", "article_section", "order_section", "conclusion_section"]] = Field(
        default_factory=list
    )
    should_abstain_if_missing: bool = False
    confidence: float = 0.0


class EvidenceBundle(BaseModel):
    refs: List[EvidenceRef] = Field(default_factory=list)
    extracted_claims: List[str] = Field(default_factory=list)
    coverage_ok: bool = False
    rationale: str = ""


class ExecutionResult(BaseModel):
    handled: bool = False
    answer: Optional[str] = None
    normalized_answer: Optional[str] = None
    evidence: EvidenceBundle = Field(default_factory=EvidenceBundle)
    confidence: float = 0.0
