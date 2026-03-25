from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from pypdf import PdfReader

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.public_dataset_eval import normalize_space


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill missing parsed report JSONs from PDFs using a fast pypdf text fallback.")
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--parsed-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def build_fallback_report(pdf_path: Path) -> Dict[str, Any]:
    reader = PdfReader(str(pdf_path))
    content: List[Dict[str, Any]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""
        text = normalize_space(raw_text)
        blocks: List[Dict[str, Any]] = []
        if text:
            blocks.append({"type": "text", "text": text})
        content.append({"page": page_number, "content": blocks})

    return {
        "metainfo": {
            "sha1_name": pdf_path.stem,
            "pages_amount": len(reader.pages),
        },
        "content": content,
        "tables": [],
        "pictures": [],
    }


def main() -> None:
    args = parse_args()
    pdf_dir = Path(args.pdf_dir)
    parsed_dir = Path(args.parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    missing = [path for path in pdf_paths if not (parsed_dir / f"{path.stem}.json").exists()]
    if args.limit and args.limit > 0:
        missing = missing[: args.limit]

    print(f"missing_reports={len(missing)}")
    for index, pdf_path in enumerate(missing, start=1):
        report = build_fallback_report(pdf_path)
        out_path = parsed_dir / f"{pdf_path.stem}.json"
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if index <= 10 or index == len(missing) or index % 25 == 0:
            print(f"[{index}/{len(missing)}] wrote {out_path.name}")


if __name__ == "__main__":
    main()
