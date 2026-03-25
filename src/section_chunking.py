from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter


HEADING_RE = re.compile(r"^\s*#{1,6}\s+")
CASE_BOUNDARY_RE = re.compile(
    r"^\s*(IT IS HEREBY ORDERED THAT|ORDERED THAT|THE COURT ORDERS|JUDGMENT|ORDER|DATE OF ISSUE\b|HEARING\s*:|CORAM\b|BEFORE\b)",
    re.I,
)


def count_tokens(text: str, encoding_name: str = "o200k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text or ""))


def normalize_block(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def split_front_matter(page_text: str) -> List[str]:
    head = normalize_block(page_text[:1800])
    if not head:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", head) if part.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for paragraph in paragraphs:
        para_tokens = count_tokens(paragraph)
        if current and current_tokens + para_tokens > 140:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_tokens = para_tokens
        else:
            current.append(paragraph)
            current_tokens += para_tokens
    if current:
        chunks.append("\n\n".join(current))
    return chunks[:4]


def split_legal_sections(page_text: str) -> List[str]:
    text = normalize_block(page_text)
    if not text:
        return []

    lines = text.splitlines()
    sections: List[str] = []
    current: List[str] = []

    def flush() -> None:
        nonlocal current
        block = "\n".join(current).strip()
        if block:
            sections.append(block)
        current = []

    for line in lines:
        stripped = line.strip()
        boundary = bool(HEADING_RE.match(stripped) or CASE_BOUNDARY_RE.match(stripped))
        if boundary and current:
            flush()
        current.append(line)
    flush()
    return sections


def split_to_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]


def section_aware_page_chunks(page: Dict[str, object]) -> List[Dict[str, object]]:
    page_number = int(page["page"])
    page_text = str(page.get("text", "") or "")
    raw_sections: List[str] = []

    if page_number == 1:
        raw_sections.extend(split_front_matter(page_text))
    raw_sections.extend(split_legal_sections(page_text))
    if not raw_sections:
        raw_sections = [normalize_block(page_text)]

    chunks: List[Dict[str, object]] = []
    seen = set()
    for section in raw_sections:
        if not section:
            continue
        section_tokens = count_tokens(section)
        if section_tokens > 260:
            parts = split_to_chunks(section, chunk_size=260, chunk_overlap=40)
        elif section_tokens > 180:
            parts = split_to_chunks(section, chunk_size=180, chunk_overlap=20)
        else:
            parts = [section]
        for part in parts:
            key = re.sub(r"\s+", " ", part).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            chunks.append(
                {
                    "page": page_number,
                    "length_tokens": count_tokens(part),
                    "text": part,
                }
            )
    return chunks


def build_section_aware_chunk_corpus(merged_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for report_path in sorted(merged_dir.glob("*.json")):
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        chunk_id = 0
        chunks: List[Dict[str, object]] = []
        for page in payload["content"]["pages"]:
            for chunk in section_aware_page_chunks(page):
                chunk["id"] = chunk_id
                chunk["type"] = "content"
                chunk_id += 1
                chunks.append(chunk)
        payload["content"]["chunks"] = chunks
        (output_dir / report_path.name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return output_dir
