from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.public_dataset_eval import PublicCorpus, safe_json_load


def iter_page_hits(
    corpus: PublicCorpus,
    candidate_shas: list[str],
    patterns: list[str],
) -> Iterable[dict]:
    lowered = [pattern.lower() for pattern in patterns if pattern]
    for sha in candidate_shas:
        payload = corpus.documents_payload[sha]
        document = corpus.documents[sha]
        for page in payload["content"]["pages"]:
            text = page["text"]
            text_lower = text.lower()
            if lowered and not all(pattern in text_lower for pattern in lowered):
                continue
            yield {
                "sha": sha,
                "title": document.title,
                "kind": document.kind,
                "canonical_ids": document.canonical_ids,
                "page": page["page"],
                "text": text,
            }


def iter_chunk_hits(
    corpus: PublicCorpus,
    candidate_shas: list[str],
    patterns: list[str],
) -> Iterable[dict]:
    lowered = [pattern.lower() for pattern in patterns if pattern]
    for chunk in corpus.chunks:
        if chunk.sha not in candidate_shas:
            continue
        text_lower = chunk.text.lower()
        if lowered and not all(pattern in text_lower for pattern in lowered):
            continue
        yield {
            "ref": chunk.ref,
            "sha": chunk.sha,
            "title": chunk.title,
            "kind": chunk.kind,
            "canonical_ids": chunk.canonical_ids,
            "page": chunk.page,
            "text": chunk.text,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Lookup public corpus documents and snippets.")
    parser.add_argument("--work-dir", default="artifacts/public_eval")
    parser.add_argument("--pdf-dir", default="pdfs")
    parser.add_argument("--law-id")
    parser.add_argument("--case-id")
    parser.add_argument("--title")
    parser.add_argument("--search", action="append", default=[])
    parser.add_argument("--page-limit", type=int, default=8)
    parser.add_argument("--chunk-limit", type=int, default=8)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    corpus = PublicCorpus(work_dir, Path(args.pdf_dir), work_dir / "docling" / "chunked")

    candidate_shas: list[str] = []
    if args.law_id:
        candidate_shas.extend(corpus.id_index.get(args.law_id, []))
    if args.case_id:
        candidate_shas.extend(corpus.id_index.get(args.case_id, []))
    if args.title:
        title_norm = re.sub(r"\s+", " ", args.title).strip().lower()
        for sha, doc in corpus.documents.items():
            if title_norm in doc.title.lower():
                candidate_shas.append(sha)
    if not candidate_shas:
        raise SystemExit("No candidate documents found. Supply --law-id, --case-id, or --title.")

    candidate_shas = sorted(dict.fromkeys(candidate_shas))
    print(json.dumps(
        {
            "documents": [
                {
                    "sha": sha,
                    "title": corpus.documents[sha].title,
                    "kind": corpus.documents[sha].kind,
                    "canonical_ids": corpus.documents[sha].canonical_ids,
                }
                for sha in candidate_shas
            ]
        },
        ensure_ascii=False,
        indent=2,
    ))

    page_hits = list(iter_page_hits(corpus, candidate_shas, args.search))[: args.page_limit]
    if page_hits:
        print("\nPAGE HITS")
        for hit in page_hits:
            print(
                json.dumps(
                    {
                        "title": hit["title"],
                        "page": hit["page"],
                        "sha": hit["sha"],
                        "canonical_ids": hit["canonical_ids"],
                        "text": hit["text"][:4000],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

    chunk_hits = list(iter_chunk_hits(corpus, candidate_shas, args.search))[: args.chunk_limit]
    if chunk_hits:
        print("\nCHUNK HITS")
        for hit in chunk_hits:
            print(json.dumps(hit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
