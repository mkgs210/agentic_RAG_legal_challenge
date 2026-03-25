import re
from typing import Dict, List


WORD_RE = re.compile(r"[A-Za-z0-9/]+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_lexical_text(text: str) -> str:
    return (text or "").lower()


def tokenize_for_bm25(text: str, min_char_ngram: int = 3, max_char_ngram: int = 3) -> List[str]:
    normalized = normalize_lexical_text(text)
    word_tokens = WORD_RE.findall(normalized)

    compact = NON_ALNUM_RE.sub("", normalized)
    char_tokens: List[str] = []
    if compact:
        for size in range(min_char_ngram, max_char_ngram + 1):
            if len(compact) < size:
                continue
            char_tokens.extend(compact[index:index + size] for index in range(len(compact) - size + 1))

    return word_tokens + char_tokens


def select_novel_results(
    primary_results: List[Dict],
    secondary_results: List[Dict],
    max_new_results: int,
) -> List[Dict]:
    existing_refs = {item.get("ref") for item in primary_results}
    novel_results: List[Dict] = []

    for item in secondary_results:
        ref = item.get("ref")
        if ref in existing_refs:
            continue
        novel_results.append(item)
        if len(novel_results) >= max_new_results:
            break

    return novel_results
