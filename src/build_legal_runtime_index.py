from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.legal_runtime_index import build_index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the legal runtime SQLite/FTS index.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("starter_kit/challenge_workdir"),
        help="Path to the challenge workdir.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("starter_kit/challenge_workdir/runtime_index/legal_runtime_index.sqlite"),
        help="Output SQLite database path.",
    )
    parser.add_argument(
        "--source-variant",
        type=str,
        default="chunked_section_aware",
        help="Docling variant directory to read from.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    index = build_index(args.work_dir, args.db_path, source_variant=args.source_variant)
    print(index.healthcheck())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
