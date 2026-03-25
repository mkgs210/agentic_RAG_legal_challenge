from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile


def _validate_timing(timing: dict[str, Any], issues: list[str], prefix: str) -> None:
    required = ("ttft_ms", "tpot_ms", "total_time_ms")
    for key in required:
        if key not in timing:
            issues.append(f"{prefix}.timing missing {key}")
            continue
        value = timing[key]
        if not isinstance(value, int) or value < 0:
            issues.append(f"{prefix}.timing.{key} must be a non-negative int")
    if all(isinstance(timing.get(key), int) for key in required):
        if timing["ttft_ms"] > timing["total_time_ms"]:
            issues.append(f"{prefix}.timing ttft_ms must be <= total_time_ms")


def _validate_usage(usage: dict[str, Any], issues: list[str], prefix: str) -> None:
    required = ("input_tokens", "output_tokens")
    for key in required:
        if key not in usage:
            issues.append(f"{prefix}.usage missing {key}")
            continue
        value = usage[key]
        if not isinstance(value, int) or value < 0:
            issues.append(f"{prefix}.usage.{key} must be a non-negative int")


def _validate_retrieval(retrieval: dict[str, Any], issues: list[str], prefix: str) -> None:
    refs = retrieval.get("retrieved_chunk_pages")
    if not isinstance(refs, list):
        issues.append(f"{prefix}.retrieval.retrieved_chunk_pages must be a list")
        return
    for idx, ref in enumerate(refs):
        item_prefix = f"{prefix}.retrieval.retrieved_chunk_pages[{idx}]"
        if not isinstance(ref, dict):
            issues.append(f"{item_prefix} must be an object")
            continue
        doc_id = ref.get("doc_id")
        page_numbers = ref.get("page_numbers")
        if not isinstance(doc_id, str) or not doc_id.strip():
            issues.append(f"{item_prefix}.doc_id must be a non-empty string")
        if not isinstance(page_numbers, list) or not page_numbers:
            issues.append(f"{item_prefix}.page_numbers must be a non-empty list")
            continue
        bad_pages = [page for page in page_numbers if not isinstance(page, int) or page <= 0]
        if bad_pages:
            issues.append(f"{item_prefix}.page_numbers must contain only positive ints")


def validate_submission(submission_path: Path, code_archive_path: Path | None = None) -> dict[str, Any]:
    payload = json.loads(submission_path.read_text(encoding="utf-8"))
    issues: list[str] = []

    if not isinstance(payload.get("architecture_summary"), str) or not payload["architecture_summary"].strip():
        issues.append("architecture_summary must be a non-empty string")

    answers = payload.get("answers")
    if not isinstance(answers, list) or not answers:
        issues.append("answers must be a non-empty list")
        answers = []

    for idx, answer in enumerate(answers):
        prefix = f"answers[{idx}]"
        if not isinstance(answer, dict):
            issues.append(f"{prefix} must be an object")
            continue
        question_id = answer.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            issues.append(f"{prefix}.question_id must be a non-empty string")
        if "answer" not in answer:
            issues.append(f"{prefix}.answer missing")
        telemetry = answer.get("telemetry")
        if not isinstance(telemetry, dict):
            issues.append(f"{prefix}.telemetry must be an object")
            continue
        timing = telemetry.get("timing")
        if not isinstance(timing, dict):
            issues.append(f"{prefix}.timing must be an object")
        else:
            _validate_timing(timing, issues, prefix)
        usage = telemetry.get("usage")
        if not isinstance(usage, dict):
            issues.append(f"{prefix}.usage must be an object")
        else:
            _validate_usage(usage, issues, prefix)
        retrieval = telemetry.get("retrieval")
        if not isinstance(retrieval, dict):
            issues.append(f"{prefix}.retrieval must be an object")
        else:
            _validate_retrieval(retrieval, issues, prefix)
        model_name = telemetry.get("model_name")
        if model_name is not None and not isinstance(model_name, str):
            issues.append(f"{prefix}.model_name must be null or string")

    archive_size_bytes = None
    archive_members = None
    if code_archive_path is not None and code_archive_path.exists():
        archive_size_bytes = code_archive_path.stat().st_size
        if archive_size_bytes > 25 * 1024 * 1024:
            issues.append("code archive exceeds 25 MB limit")
        with ZipFile(code_archive_path) as archive:
            archive_members = len(archive.infolist())

    return {
        "submission_path": str(submission_path),
        "code_archive_path": str(code_archive_path) if code_archive_path else None,
        "answer_count": len(answers),
        "archive_size_bytes": archive_size_bytes,
        "archive_member_count": archive_members,
        "issues": issues,
        "ok": not issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a local submission.json against core challenge rules.")
    parser.add_argument("--submission", default="starter_kit/submission.json")
    parser.add_argument("--archive", default="starter_kit/code_archive.zip")
    args = parser.parse_args()

    report = validate_submission(Path(args.submission), Path(args.archive))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
