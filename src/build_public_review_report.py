from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List

from src.public_dataset_eval import safe_json_load


AUDIT_LINE_RE = re.compile(r"^- Q(?P<index>\d+)\.\s+(?P<body>.*)$")


def normalize_preview(text: str, max_len: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def parse_manual_audit(path: Path) -> Dict[int, Dict[str, str]]:
    rows: Dict[int, Dict[str, str]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        match = AUDIT_LINE_RE.match(line)
        if not match:
            continue
        data = match.groupdict()
        body = data["body"]
        if " Pre-rerank: `" not in body:
            continue
        prefix, remainder = body.split(" Pre-rerank: `", 1)
        verdict, _, note = remainder.partition("`;")

        source = ""
        if " Sources: " in prefix:
            answer_block, source = prefix.split(" Sources: ", 1)
        elif " Source: " in prefix:
            answer_block, source = prefix.split(" Source: ", 1)
        else:
            answer_block = prefix

        expected_answer = answer_block
        if "? " in answer_block:
            expected_answer = answer_block.split("? ", 1)[1]
        elif ": " in answer_block:
            expected_answer = answer_block.split(": ", 1)[1]

        rows[int(data["index"])] = {
            "expected_answer": expected_answer.strip(),
            "expected_source": source.strip(),
            "manual_pre_rerank_verdict": verdict.strip(),
            "manual_note": note.strip(),
        }
    return rows


def top_titles(entry: Dict, limit: int = 5) -> List[str]:
    titles: List[str] = []
    seen = set()
    for item in post_rerank_rows(entry):
        title = item.get("title", "")
        if not title or title in seen:
            continue
        titles.append(title)
        seen.add(title)
        if len(titles) >= limit:
            break
    return titles


def post_rerank_rows(entry: Dict) -> List[Dict]:
    return entry.get("post_rerank") or entry.get("top_reranked_chunks") or []


def load_pack(path: Path) -> tuple[str, Dict[int, Dict]]:
    payload = safe_json_load(path)
    strategy = payload.get("strategy") or ("baseline" if path.stem == "review_pack" else path.stem)
    questions = {item["index"]: item for item in payload["questions"]}
    return strategy, questions


def build_rows(
    dataset_path: Path,
    audit_path: Path,
    baseline_pack_path: Path,
    candidate_pack_path: Path,
) -> List[Dict[str, str]]:
    questions = safe_json_load(dataset_path)
    manual = parse_manual_audit(audit_path)
    baseline_strategy, baseline = load_pack(baseline_pack_path)
    candidate_strategy, candidate = load_pack(candidate_pack_path)

    rows: List[Dict[str, str]] = []
    for index, item in enumerate(questions, start=1):
        base_entry = baseline[index]
        candidate_entry = candidate[index]
        base_post_rerank = post_rerank_rows(base_entry)
        candidate_post_rerank = post_rerank_rows(candidate_entry)
        base_top1 = base_post_rerank[0] if base_post_rerank else {}
        candidate_top1 = candidate_post_rerank[0] if candidate_post_rerank else {}
        candidate_sources = {
            source
            for result in candidate_post_rerank[:5]
            for source in result.get("retrieval_sources", [])
        }
        manual_row = manual.get(index, {})

        rows.append(
            {
                "index": str(index),
                "id": item["id"],
                "answer_type": item["answer_type"],
                "question": item["question"],
                "expected_answer": manual_row.get("expected_answer", ""),
                "expected_source": manual_row.get("expected_source", ""),
                "manual_pre_rerank_verdict": manual_row.get("manual_pre_rerank_verdict", ""),
                "baseline_top1_title": base_top1.get("title", ""),
                "baseline_top1_ref": base_top1.get("ref", ""),
                "baseline_top1_preview": normalize_preview(base_top1.get("text", "")),
                "baseline_top3_titles": " | ".join(top_titles(base_entry, limit=3)),
                "baseline_top5_titles": " | ".join(top_titles(base_entry)),
                "candidate_top1_title": candidate_top1.get("title", ""),
                "candidate_top1_ref": candidate_top1.get("ref", ""),
                "candidate_top1_preview": normalize_preview(candidate_top1.get("text", "")),
                "candidate_top3_titles": " | ".join(top_titles(candidate_entry, limit=3)),
                "candidate_top5_titles": " | ".join(top_titles(candidate_entry)),
                "candidate_lexical_contributed": "yes" if "bm25" in candidate_sources else "no",
                "top1_changed": "yes" if base_top1.get("title", "") != candidate_top1.get("title", "") else "no",
                "top3_changed": "yes"
                if top_titles(base_entry, limit=3) != top_titles(candidate_entry, limit=3)
                else "no",
                "top5_changed": "yes"
                if top_titles(base_entry, limit=5) != top_titles(candidate_entry, limit=5)
                else "no",
            }
        )
    rows.sort(key=lambda row: int(row["index"]))
    return baseline_strategy, candidate_strategy, rows


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_strategy_name(value: str) -> str:
    return value.replace("_", " ")


def write_markdown(
    path: Path,
    rows: List[Dict[str, str]],
    baseline_strategy: str,
    candidate_strategy: str,
) -> None:
    changed_rows = [row for row in rows if row["top1_changed"] == "yes"]
    top3_changed_rows = [row for row in rows if row["top3_changed"] == "yes"]
    top5_changed_rows = [row for row in rows if row["top5_changed"] == "yes"]
    lexical_rows = [row for row in rows if row["candidate_lexical_contributed"] == "yes"]
    baseline_label = format_strategy_name(baseline_strategy)
    candidate_label = format_strategy_name(candidate_strategy)

    lines: List[str] = [
        "# Public Review Comparison",
        "",
        "This report compares the manual answer key with the retrieval output.",
        f"Baseline strategy: `{baseline_label}`.",
        f"Candidate strategy: `{candidate_label}`.",
        "",
        f"- Questions: `{len(rows)}`",
        f"- Top-1 changed in candidate strategy: `{len(changed_rows)}`",
        f"- Top-3 changed in candidate strategy: `{len(top3_changed_rows)}`",
        f"- Top-5 changed in candidate strategy: `{len(top5_changed_rows)}`",
        f"- Candidate lexical contribution present in top-5: `{len(lexical_rows)}`",
        "",
        "## Changed Top-1 Cases",
        "",
    ]

    if not changed_rows:
        lines.append("- None.")
    else:
        for row in changed_rows:
            lines.append(
                f"- Q{int(row['index']):03d}: baseline `{row['baseline_top1_title']}` -> candidate `{row['candidate_top1_title']}`"
            )

    lines.extend(["", "## Per-Question Review", ""])

    for row in rows:
        lines.extend(
            [
                f"### Q{int(row['index']):03d}",
                "",
                f"- Question: {row['question']}",
                f"- Expected answer: {row['expected_answer']}",
                f"- Expected source: {row['expected_source']}",
                f"- Manual pre-rerank verdict: `{row['manual_pre_rerank_verdict']}`",
                f"- Baseline top-1: `{row['baseline_top1_title']}`",
                f"- Baseline ref: `{row['baseline_top1_ref']}`",
                f"- Baseline preview: {row['baseline_top1_preview']}",
                f"- Candidate top-1: `{row['candidate_top1_title']}`",
                f"- Candidate ref: `{row['candidate_top1_ref']}`",
                f"- Candidate preview: {row['candidate_top1_preview']}",
                f"- Baseline top-3 titles: {row['baseline_top3_titles']}",
                f"- Candidate top-3 titles: {row['candidate_top3_titles']}",
                f"- Baseline top-5 titles: {row['baseline_top5_titles']}",
                f"- Candidate top-5 titles: {row['candidate_top5_titles']}",
                f"- Candidate lexical contributed: `{row['candidate_lexical_contributed']}`",
                f"- Top-1 changed: `{row['top1_changed']}`",
                f"- Top-3 changed: `{row['top3_changed']}`",
                f"- Top-5 changed: `{row['top5_changed']}`",
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a human-readable retrieval comparison report.")
    parser.add_argument("--dataset", default="public_dataset.json")
    parser.add_argument("--audit", default="artifacts/public_review/manual_answers_and_pre_rerank_audit.md")
    parser.add_argument("--baseline-pack", default="artifacts/public_review/review_pack.json")
    parser.add_argument("--candidate-pack", default="artifacts/public_review_bm25_adaptive/review_pack.json")
    parser.add_argument("--csv-out", default="artifacts/public_review/human_review_comparison.csv")
    parser.add_argument("--md-out", default="artifacts/public_review/human_review_comparison.md")
    args = parser.parse_args()

    baseline_strategy, candidate_strategy, rows = build_rows(
        dataset_path=Path(args.dataset),
        audit_path=Path(args.audit),
        baseline_pack_path=Path(args.baseline_pack),
        candidate_pack_path=Path(args.candidate_pack),
    )
    write_csv(Path(args.csv_out), rows)
    write_markdown(Path(args.md_out), rows, baseline_strategy, candidate_strategy)


if __name__ == "__main__":
    main()
