from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.public_dataset_eval import normalize_space, safe_json_dump, safe_json_load


HEADING_RE = re.compile(r"(?:^|\n)##\s+([^\n]+)")
SENTENCE_RE = re.compile(r"(?<=[\.\?\!;:])\s+(?=[A-Z0-9(\[])")
ARTICLE_RE = re.compile(r"Article\s+\d+(?:\([^)]+\))*", re.I)


def _ordered_unique(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _page_role(page_number: int, page_count: int, text: str) -> str:
    text_norm = normalize_space(text).lower()
    if page_number == 1:
        return "title_page"
    if page_number == page_count:
        return "last_page"
    if "it is hereby ordered that" in text_norm or "order with reasons" in text_norm:
        return "order_page"
    if "conclusion" in text_norm:
        return "conclusion_page"
    if "table of contents" in text_norm or text_norm.startswith("contents"):
        return "toc_page"
    return "body_page"


def _extract_heading(text: str) -> str:
    match = HEADING_RE.search(text or "")
    return normalize_space(match.group(1)) if match else ""


def _neighbor_heading(chunks: List[Dict[str, Any]], index: int, direction: int) -> str:
    cursor = index + direction
    while 0 <= cursor < len(chunks):
        heading = _extract_heading(str(chunks[cursor].get("text", "")))
        if heading:
            return heading
        cursor += direction
    return ""


def _build_contextual_text(
    *,
    title: str,
    canonical_ids: List[str],
    chunk_text: str,
    page_text: str,
    page_number: int,
    page_count: int,
    local_heading: str,
    prev_heading: str,
    next_heading: str,
) -> str:
    parts = [
        f"[DOC_TITLE] {title}",
        f"[PAGE_ROLE] {_page_role(page_number, page_count, page_text)}",
        f"[PAGE_NUMBER] {page_number} of {page_count}",
    ]
    if canonical_ids:
        parts.append(f"[DOC_IDS] {' | '.join(canonical_ids[:3])}")
    if local_heading:
        parts.append(f"[LOCAL_SECTION] {local_heading}")
    if prev_heading:
        parts.append(f"[PREV_SECTION] {prev_heading}")
    if next_heading:
        parts.append(f"[NEXT_SECTION] {next_heading}")
    parts.append(chunk_text)
    return "\n".join(part for part in parts if part)


def build_contextual_chunk_corpus(
    *,
    chunked_dir: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(output_dir.glob("*.json")))
    source_files = sorted(chunked_dir.glob("*.json"))
    if existing >= len(source_files) and existing > 0:
        return output_dir

    for path in source_files:
        payload = safe_json_load(path)
        first_page_text = str(payload["content"]["pages"][0]["text"] or "")
        title = normalize_space(first_page_text.split("\n", 1)[0].lstrip("# "))
        canonical_ids = _ordered_unique(
            re.findall(r"(?:CFI|CA|ARB|ENF|DEC|TCD|SCT)\s+\d+/\d+|Law No\.\s*\d+\s+of\s+\d{4}|DIFC Law No\.\s*\d+\s+of\s+\d{4}", first_page_text, re.I)
        )

        page_map = {
            int(page["page"]): str(page.get("text", "") or "")
            for page in payload["content"]["pages"]
        }
        page_count = len(page_map)
        chunks = list(payload["content"]["chunks"])
        contextual_chunks: List[Dict[str, Any]] = []

        for index, chunk in enumerate(chunks):
            chunk_text = str(chunk.get("text", "") or "")
            page_number = int(chunk["page"])
            page_text = page_map.get(page_number, "")
            local_heading = _extract_heading(chunk_text) or _extract_heading(page_text)
            prev_heading = _neighbor_heading(chunks, index, -1)
            next_heading = _neighbor_heading(chunks, index, 1)
            contextual_chunks.append(
                {
                    "id": int(chunk["id"]),
                    "page": page_number,
                    "text": _build_contextual_text(
                        title=title,
                        canonical_ids=canonical_ids,
                        chunk_text=chunk_text,
                        page_text=page_text,
                        page_number=page_number,
                        page_count=page_count,
                        local_heading=local_heading,
                        prev_heading=prev_heading,
                        next_heading=next_heading,
                    ),
                }
            )

        payload["content"]["chunks"] = contextual_chunks
        safe_json_dump(output_dir / path.name, payload)

    return output_dir


def _split_atomic_units(text: str) -> List[str]:
    text = text or ""
    units: List[str] = []
    for block in re.split(r"\n{2,}", text):
        block = normalize_space(block)
        if not block:
            continue
        if block.startswith(("-", "•")) or re.match(r"^\(?[a-z0-9]+\)", block, re.I):
            units.append(block)
            continue
        for part in SENTENCE_RE.split(block):
            part = normalize_space(part)
            if len(part) < 24:
                continue
            units.append(part)
    cleaned = []
    for unit in units:
        if len(unit) > 500:
            cleaned.extend(normalize_space(piece) for piece in re.split(r";\s+", unit) if len(normalize_space(piece)) >= 24)
        else:
            cleaned.append(unit)
    return _ordered_unique(unit for unit in cleaned if len(unit) >= 24)


def _document_kind(first_page_text: str) -> str:
    head_upper = (first_page_text or "").upper()
    if "REGULATIONS" in head_upper:
        return "regulation"
    if "LAW" in head_upper:
        return "law"
    if re.search(r"\b(CFI|ARB|TCD|CA|ENF|DEC|SCT)\b", head_upper):
        return "case"
    return "other"


def _first_sentences(text: str, limit: int = 3) -> List[str]:
    cleaned = normalize_space(text.replace("\n", " "))
    if not cleaned:
        return []
    sentences = [
        normalize_space(sentence)
        for sentence in SENTENCE_RE.split(cleaned)
        if len(normalize_space(sentence)) >= 32
    ]
    return _ordered_unique(sentences)[:limit]


def _collect_headings(pages: List[Dict[str, Any]], limit: int = 8) -> List[str]:
    headings: List[str] = []
    for page in pages:
        text = str(page.get("text", "") or "")
        for match in HEADING_RE.finditer(text):
            heading = normalize_space(match.group(1))
            if heading and heading.lower() not in {"contents", "table of contents"}:
                headings.append(heading)
        for article_ref in ARTICLE_RE.findall(text):
            headings.append(normalize_space(article_ref))
        if len(headings) >= limit * 2:
            break
    return _ordered_unique(headings)[:limit]


def _document_synopsis(payload: Dict[str, Any]) -> str:
    pages = list(payload["content"]["pages"])
    if not pages:
        return ""
    first_page_text = str(pages[0].get("text", "") or "")
    last_page_text = str(pages[-1].get("text", "") or "")
    title = normalize_space(first_page_text.split("\n", 1)[0].lstrip("# "))
    canonical_ids = _ordered_unique(
        re.findall(
            r"(?:CFI|CA|ARB|ENF|DEC|TCD|SCT)\s+\d+/\d+|Law No\.\s*\d+\s+of\s+\d{4}|DIFC Law No\.\s*\d+\s+of\s+\d{4}",
            first_page_text,
            re.I,
        )
    )
    headings = _collect_headings(pages)
    cover_facts = _first_sentences(first_page_text[:1800], limit=3)
    ending_facts: List[str] = []
    if any(marker in last_page_text.lower() for marker in ("ordered", "conclusion", "application", "costs", "trial")):
        ending_facts = _first_sentences(last_page_text[:1400], limit=2)

    parts = [
        f"[DOC_TITLE] {title}",
        f"[DOC_KIND] {_document_kind(first_page_text)}",
    ]
    if canonical_ids:
        parts.append(f"[DOC_IDS] {' | '.join(canonical_ids[:4])}")
    if cover_facts:
        parts.append(f"[DOC_SUMMARY] {' '.join(cover_facts)}")
    if headings:
        parts.append(f"[KEY_SECTIONS] {' | '.join(headings[:8])}")
    if ending_facts:
        parts.append(f"[ENDING_SUMMARY] {' '.join(ending_facts)}")
    return "\n".join(parts)


def build_atomic_fact_chunk_corpus(
    *,
    chunked_dir: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(output_dir.glob("*.json")))
    source_files = sorted(chunked_dir.glob("*.json"))
    if existing >= len(source_files) and existing > 0:
        return output_dir

    for path in source_files:
        payload = safe_json_load(path)
        first_page_text = str(payload["content"]["pages"][0]["text"] or "")
        title = normalize_space(first_page_text.split("\n", 1)[0].lstrip("# "))
        canonical_ids = _ordered_unique(
            re.findall(r"(?:CFI|CA|ARB|ENF|DEC|TCD|SCT)\s+\d+/\d+|Law No\.\s*\d+\s+of\s+\d{4}|DIFC Law No\.\s*\d+\s+of\s+\d{4}", first_page_text, re.I)
        )
        atomic_chunks: List[Dict[str, Any]] = []
        next_id = 0
        for chunk in payload["content"]["chunks"]:
            page_number = int(chunk["page"])
            chunk_text = str(chunk.get("text", "") or "")
            units = _split_atomic_units(chunk_text)
            if not units:
                units = [normalize_space(chunk_text)]
            for unit in units:
                prefix_parts = [f"[DOC_TITLE] {title}", f"[PAGE_NUMBER] {page_number}"]
                if canonical_ids:
                    prefix_parts.append(f"[DOC_IDS] {' | '.join(canonical_ids[:3])}")
                prefix_parts.append(unit)
                atomic_chunks.append(
                    {
                        "id": next_id,
                        "page": page_number,
                        "text": "\n".join(prefix_parts),
                    }
                )
                next_id += 1

        payload["content"]["chunks"] = atomic_chunks
        safe_json_dump(output_dir / path.name, payload)

    return output_dir


def build_summary_augmented_chunk_corpus(
    *,
    chunked_dir: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(output_dir.glob("*.json")))
    source_files = sorted(chunked_dir.glob("*.json"))
    if existing >= len(source_files) and existing > 0:
        return output_dir

    for path in source_files:
        payload = safe_json_load(path)
        synopsis = _document_synopsis(payload)
        augmented_chunks: List[Dict[str, Any]] = []
        for chunk in payload["content"]["chunks"]:
            chunk_text = normalize_space(str(chunk.get("text", "") or ""))
            augmented_text = "\n".join(part for part in [synopsis, chunk_text] if part)
            augmented_chunks.append(
                {
                    "id": int(chunk["id"]),
                    "page": int(chunk["page"]),
                    "text": augmented_text,
                }
            )
        payload["content"]["chunks"] = augmented_chunks
        safe_json_dump(output_dir / path.name, payload)

    return output_dir
