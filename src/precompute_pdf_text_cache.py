from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pdf_text_fallback import merged_pdf_page_texts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute merged PDF page-text cache for a corpus.")
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--merged-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged_files = sorted(args.merged_dir.glob("*.json"))
    if args.limit is not None:
        merged_files = merged_files[: args.limit]

    total = len(merged_files)
    for index, path in enumerate(merged_files, start=1):
        payload = json.loads(path.read_text(encoding="utf-8"))
        pages = payload["content"]["pages"]
        sha = path.stem
        merged = merged_pdf_page_texts(args.pdf_dir, sha, pages)
        print(f"[{index}/{total}] {sha} pages={len(merged)}")


if __name__ == "__main__":
    main()
