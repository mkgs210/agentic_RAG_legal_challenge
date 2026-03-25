from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List

from src.public_dataset_eval import normalize_space

try:
    from PyPDF2 import PdfReader
except Exception:  # pragma: no cover - optional runtime dependency guard
    PdfReader = None  # type: ignore[assignment]


CACHE_DIRNAME = ".merged_page_text_cache_v1"


def normalized_line(value: str) -> str:
    return normalize_space(value).lower()


def merged_text_lines(*texts: str) -> List[str]:
    lines: List[str] = []
    seen = set()
    for text in texts:
        for raw_line in str(text or "").splitlines():
            line = normalize_space(raw_line)
            if not line:
                continue
            key = normalized_line(line)
            if key in seen:
                continue
            seen.add(key)
            lines.append(line)
    return lines


@lru_cache(maxsize=256)
def load_pdf_page_texts(pdf_path: str) -> Dict[int, str]:
    path = Path(pdf_path)
    if not path.exists() or PdfReader is None:
        return {}
    try:
        reader = PdfReader(str(path))
    except Exception:
        return {}

    page_texts: Dict[int, str] = {}
    for index, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""
        if normalize_space(raw_text):
            page_texts[index] = "\n".join(merged_text_lines(raw_text))
    return page_texts


def merged_page_text(docling_text: str, pdf_text: str) -> str:
    return "\n".join(merged_text_lines(docling_text, pdf_text))


def cache_dir(pdf_dir: Path) -> Path:
    return pdf_dir / CACHE_DIRNAME


def cache_path(pdf_dir: Path, sha: str) -> Path:
    return cache_dir(pdf_dir) / f"{sha}.json"


def load_cached_merged_page_texts(pdf_dir: Path, sha: str) -> Dict[int, str]:
    path = cache_path(pdf_dir, sha)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    result: Dict[int, str] = {}
    for key, value in payload.items():
        try:
            page = int(key)
        except Exception:
            continue
        text = normalize_space(str(value or ""))
        if text:
            result[page] = text
    return result


def save_cached_merged_page_texts(pdf_dir: Path, sha: str, page_texts: Dict[int, str]) -> None:
    directory = cache_dir(pdf_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {str(page): normalize_space(text) for page, text in sorted(page_texts.items()) if normalize_space(text)}
    cache_path(pdf_dir, sha).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def merged_pdf_page_texts(
    pdf_dir: Path,
    sha: str,
    docling_pages: Iterable[dict],
) -> Dict[int, str]:
    cached = load_cached_merged_page_texts(pdf_dir, sha)
    if cached:
        return cached
    pdf_path = pdf_dir / f"{sha}.pdf"
    raw_texts = load_pdf_page_texts(str(pdf_path))
    merged: Dict[int, str] = {}
    for page in docling_pages:
        page_number = int(page["page"])
        merged[page_number] = merged_page_text(str(page.get("text") or ""), raw_texts.get(page_number, ""))
    for page_number, pdf_text in raw_texts.items():
        merged.setdefault(page_number, pdf_text)
    if merged:
        save_cached_merged_page_texts(pdf_dir, sha, merged)
    return merged
