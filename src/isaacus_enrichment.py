from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.isaacus_models import IsaacusEnricher
from src.public_dataset_eval import safe_json_dump


JOINER = "\n\n"


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def slice_span(text: str, span: Dict[str, int] | None) -> str:
    if not span:
        return ""
    return text[int(span["start"]) : int(span["end"])]


def build_full_text(pages: Sequence[Dict[str, Any]]) -> tuple[str, List[tuple[int, int, int]]]:
    parts: List[str] = []
    offsets: List[tuple[int, int, int]] = []
    cursor = 0
    for index, page in enumerate(pages):
        page_text = page.get("text", "")
        start = cursor
        end = start + len(page_text)
        offsets.append((int(page["page"]), start, end))
        parts.append(page_text)
        cursor = end
        if index < len(pages) - 1:
            cursor += len(JOINER)
    return JOINER.join(parts), offsets


def offset_to_page(offset: int, page_offsets: Sequence[tuple[int, int, int]]) -> int:
    for page, start, end in page_offsets:
        if start <= offset < end:
            return page
    return page_offsets[-1][0] if page_offsets else 1


def span_inside_junk(span: Dict[str, int], junk_spans: Sequence[Dict[str, int]]) -> bool:
    start = int(span["start"])
    end = int(span["end"])
    for junk in junk_spans:
        junk_start = int(junk["start"])
        junk_end = int(junk["end"])
        overlap = max(0, min(end, junk_end) - max(start, junk_start))
        if overlap and overlap >= 0.8 * max(1, end - start):
            return True
    return False


def nearest_heading_text(document: Dict[str, Any], segment_start: int) -> str:
    best = ""
    best_distance = 10**9
    for heading in document.get("headings", []):
        start = int(heading["start"])
        end = int(heading["end"])
        if end > segment_start:
            continue
        distance = segment_start - end
        if distance < best_distance and distance <= 400:
            best_distance = distance
            best = normalize_space(document["text"][start:end])
    return best


def extract_person_name(document: Dict[str, Any], person: Dict[str, Any]) -> str:
    return normalize_space(slice_span(document["text"], person.get("name")))


def mentions_intersect(span: Dict[str, int], mention: Dict[str, int]) -> bool:
    start = int(span["start"])
    end = int(span["end"])
    mention_start = int(mention["start"])
    mention_end = int(mention["end"])
    return max(start, mention_start) < min(end, mention_end)


def persons_for_span(document: Dict[str, Any], span: Dict[str, int], limit: int = 6) -> List[str]:
    rows: List[str] = []
    for person in document.get("persons", []):
        mentions = person.get("mentions", [])
        if not any(mentions_intersect(span, mention) for mention in mentions):
            continue
        name = extract_person_name(document, person)
        if not name:
            continue
        role = person.get("role")
        rows.append(f"{name} ({role})" if role else name)
    deduped = list(dict.fromkeys(rows))
    return deduped[:limit]


def dates_for_span(document: Dict[str, Any], span: Dict[str, int], limit: int = 4) -> List[str]:
    rows: List[str] = []
    for date in document.get("dates", []):
        mentions = date.get("mentions", [])
        if not any(mentions_intersect(span, mention) for mention in mentions):
            continue
        value = normalize_space(str(date.get("value") or ""))
        kind = normalize_space(str(date.get("type") or ""))
        if value:
            rows.append(f"{value} ({kind})" if kind else value)
    deduped = list(dict.fromkeys(rows))
    return deduped[:limit]


def terms_summary(document: Dict[str, Any], limit: int = 6) -> List[str]:
    rows: List[str] = []
    for term in document.get("terms", [])[:limit]:
        name = normalize_space(slice_span(document["text"], term.get("name")))
        meaning = normalize_space(slice_span(document["text"], term.get("meaning")))
        if not name or not meaning:
            continue
        rows.append(f"{name} = {meaning[:120]}")
    return rows


def persons_summary(document: Dict[str, Any], limit: int = 10) -> List[str]:
    rows: List[str] = []
    for person in document.get("persons", []):
        name = extract_person_name(document, person)
        if not name:
            continue
        role = person.get("role")
        rows.append(f"{name} ({role})" if role else name)
    deduped = list(dict.fromkeys(rows))
    return deduped[:limit]


def dates_summary(document: Dict[str, Any], limit: int = 6) -> List[str]:
    rows: List[str] = []
    for date in document.get("dates", [])[:limit]:
        value = normalize_space(str(date.get("value") or ""))
        kind = normalize_space(str(date.get("type") or ""))
        if value:
            rows.append(f"{value} ({kind})" if kind else value)
    return rows


def build_metadata_chunk(document: Dict[str, Any]) -> str:
    lines = []
    doc_type = normalize_space(str(document.get("type") or ""))
    jurisdiction = normalize_space(str(document.get("jurisdiction") or ""))
    if doc_type:
        lines.append(f"DOCUMENT_TYPE: {doc_type}")
    if jurisdiction:
        lines.append(f"JURISDICTION: {jurisdiction}")

    persons = persons_summary(document)
    if persons:
        lines.append("PERSONS: " + " | ".join(persons))

    dates = dates_summary(document)
    if dates:
        lines.append("DATES: " + " | ".join(dates))

    terms = terms_summary(document)
    if terms:
        lines.append("TERMS: " + " | ".join(terms))

    return "\n".join(lines).strip()


def build_segment_text(document: Dict[str, Any], segment: Dict[str, Any], parent_text: str = "") -> str:
    body = normalize_space(slice_span(document["text"], segment.get("span")))
    if not body:
        return ""

    lines = []
    doc_type = normalize_space(str(document.get("type") or ""))
    jurisdiction = normalize_space(str(document.get("jurisdiction") or ""))
    if doc_type:
        lines.append(f"DOCUMENT_TYPE: {doc_type}")
    if jurisdiction:
        lines.append(f"JURISDICTION: {jurisdiction}")
    if segment.get("category"):
        lines.append(f"SEGMENT_CATEGORY: {segment['category']}")
    if segment.get("kind"):
        lines.append(f"SEGMENT_KIND: {segment['kind']}")
    if segment.get("type_name"):
        lines.append(f"SEGMENT_TYPE_NAME: {segment['type_name']}")
    if segment.get("code"):
        lines.append(f"SEGMENT_CODE: {segment['code']}")
    if segment.get("title"):
        lines.append(f"SEGMENT_TITLE: {segment['title']}")

    heading = nearest_heading_text(document, int(segment["span"]["start"]))
    if heading:
        lines.append(f"NEAREST_HEADING: {heading}")

    mentioned_persons = persons_for_span(document, segment["span"])
    if mentioned_persons:
        lines.append("MENTIONED_PERSONS: " + " | ".join(mentioned_persons))

    mentioned_dates = dates_for_span(document, segment["span"])
    if mentioned_dates:
        lines.append("MENTIONED_DATES: " + " | ".join(mentioned_dates))

    if len(body) < 80 and parent_text and normalize_space(parent_text) != body:
        lines.append(f"PARENT_CONTEXT: {normalize_space(parent_text)[:240]}")

    lines.append(body)
    return "\n".join(line for line in lines if line).strip()


def build_enriched_chunk_payload(merged_payload: Dict[str, Any], document: Dict[str, Any]) -> Dict[str, Any]:
    pages = merged_payload["content"]["pages"]
    full_text, page_offsets = build_full_text(pages)
    if normalize_space(full_text) != normalize_space(document["text"]):
        # The Enricher should receive exactly the same text, but normalize-space protects against trivial spacing drift.
        document_text = document["text"]
    else:
        document_text = full_text
    junk_spans = document.get("junk", [])

    segments_by_id = {segment["id"]: segment for segment in document.get("segments", [])}
    chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    metadata_text = build_metadata_chunk(document)
    if metadata_text:
        chunks.append({"id": chunk_id, "page": 1, "text": metadata_text})
        chunk_id += 1

    for segment in document.get("segments", []):
        if segment.get("kind") == "container" and segment.get("children"):
            continue
        if span_inside_junk(segment["span"], junk_spans):
            continue
        parent_text = ""
        parent_id = segment.get("parent")
        if parent_id and parent_id in segments_by_id:
            parent_text = slice_span(document_text, segments_by_id[parent_id].get("span"))
        text = build_segment_text(document, segment, parent_text=parent_text)
        if len(normalize_space(text)) < 8:
            continue
        page = offset_to_page(int(segment["span"]["start"]), page_offsets)
        chunks.append({"id": chunk_id, "page": page, "text": text})
        chunk_id += 1

    return {
        "metainfo": dict(merged_payload["metainfo"]),
        "content": {
            "pages": list(pages),
            "chunks": chunks,
        },
    }


def build_enriched_chunk_corpus(
    merged_dir: Path,
    output_dir: Path,
    cache_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    enricher = IsaacusEnricher(cache_dir=cache_dir)

    for merged_path in sorted(merged_dir.glob("*.json")):
        output_path = output_dir / merged_path.name
        if output_path.exists():
            continue
        merged_payload = json.loads(merged_path.read_text(encoding="utf-8"))
        full_text, _ = build_full_text(merged_payload["content"]["pages"])
        enriched_document = enricher.enrich_text(full_text)
        payload = build_enriched_chunk_payload(merged_payload, enriched_document)
        safe_json_dump(output_path, payload)

    return output_dir
