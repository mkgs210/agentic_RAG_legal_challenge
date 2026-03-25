from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from pydantic import BaseModel, Field

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.api_requests import APIProcessor
from src.public_dataset_eval import load_corpus, normalize_space


class FreeTextRubricVerdict(BaseModel):
    correctness: int = Field(0, ge=0, le=1)
    completeness: int = Field(0, ge=0, le=1)
    grounding: int = Field(0, ge=0, le=1)
    confidence: int = Field(0, ge=0, le=1)
    clarity: int = Field(0, ge=0, le=1)
    explanation: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--chunked-dir", default="/home/mkgs/hackaton/starter_kit/challenge_workdir/docling/chunked")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", default="/home/mkgs/hackaton/starter_kit/challenge_workdir/free_text_rubric_eval")
    return parser.parse_args()


def load_snapshot(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("items") or payload.get("responses") or []
    raise TypeError(f"Unsupported snapshot payload in {path}")


def build_context(item: Dict[str, Any], ref_to_chunk: Dict[str, Dict[str, Any]]) -> str:
    refs = list(item.get("answer_chunk_refs") or [])
    if not refs:
        refs = list(item.get("citations") or [])
    if not refs:
        refs = list(item.get("top_chunk_refs") or [])[:5]
    parts: List[str] = []
    seen = set()
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)
        chunk = ref_to_chunk.get(ref)
        if not chunk:
            continue
        parts.append(
            f"[REF {ref} | title={chunk['title']} | page={chunk['page']}]\n{chunk['text']}"
        )
        if len(parts) >= 5:
            break
    return "\n\n---\n\n".join(parts)


def judge_prompt() -> str:
    return (
        "You are grading a DIFC legal assistant answer using only the provided evidence.\n"
        "Score each criterion as 1 only if it is clearly satisfied; otherwise score 0.\n"
        "Criteria:\n"
        "1. correctness: every factual statement is accurate from the evidence.\n"
        "2. completeness: the answer covers the key parts of the question.\n"
        "3. grounding: every statement is explicitly supported by the evidence and there is no hallucination.\n"
        "4. confidence: uncertainty is expressed appropriately; the answer does not overclaim.\n"
        "5. clarity: the answer is clear, concise, coherent, and directly answers the question.\n"
        "Judge only from the evidence below. If the evidence does not support the answer, score correctness and grounding as 0.\n"
        "Return JSON only."
    )


def main() -> None:
    args = parse_args()
    snapshot_path = Path(args.snapshot)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, chunks = load_corpus(Path(args.chunked_dir))
    ref_to_chunk = {
        chunk.ref: {
            "ref": chunk.ref,
            "title": chunk.title,
            "page": chunk.page,
            "text": chunk.text,
        }
        for chunk in chunks
    }

    items = [item for item in load_snapshot(snapshot_path) if item.get("answer_type") == "free_text"]
    if args.limit:
        items = items[: args.limit]

    api = APIProcessor(provider=args.provider)
    rows: List[Dict[str, Any]] = []
    for item in items:
        context = build_context(item, ref_to_chunk)
        if not context:
            continue
        response = api.send_message(
            model=args.model,
            temperature=0.0,
            system_content=judge_prompt(),
            human_content=(
                f"Question: {item.get('question', '')}\n\n"
                f"Answer: {item.get('raw_answer', '')}\n\n"
                f"Evidence:\n{context}"
            ),
            is_structured=True,
            response_format=FreeTextRubricVerdict,
            max_tokens=220,
            request_timeout=60,
        )
        total = (
            int(response.get("correctness", 0))
            + int(response.get("completeness", 0))
            + int(response.get("grounding", 0))
            + int(response.get("confidence", 0))
            + int(response.get("clarity", 0))
        ) / 5.0
        rows.append(
            {
                "question_id": item.get("question_id"),
                "question": item.get("question"),
                "answer": item.get("raw_answer"),
                "correctness": int(response.get("correctness", 0)),
                "completeness": int(response.get("completeness", 0)),
                "grounding": int(response.get("grounding", 0)),
                "confidence": int(response.get("confidence", 0)),
                "clarity": int(response.get("clarity", 0)),
                "score": round(total, 4),
                "explanation": normalize_space(str(response.get("explanation", "") or "")),
            }
        )

    summary = {
        "snapshot": str(snapshot_path),
        "provider": args.provider,
        "model": args.model,
        "count": len(rows),
        "self_consistency_proxy": round(mean(row["score"] for row in rows), 4) if rows else 0.0,
        "correctness": round(mean(row["correctness"] for row in rows), 4) if rows else 0.0,
        "completeness": round(mean(row["completeness"] for row in rows), 4) if rows else 0.0,
        "grounding": round(mean(row["grounding"] for row in rows), 4) if rows else 0.0,
        "confidence": round(mean(row["confidence"] for row in rows), 4) if rows else 0.0,
        "clarity": round(mean(row["clarity"] for row in rows), 4) if rows else 0.0,
        "limitations": [
            "This is not a leaderboard proxy.",
            "The judge only sees evidence selected by the current pipeline.",
            "It measures self-consistency against selected refs, not hidden gold completeness.",
        ],
    }

    slug = snapshot_path.stem
    (output_dir / f"{slug}_rubric_rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    (output_dir / f"{slug}_rubric_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    md = [
        f"# Free Text Rubric Eval: {slug}",
        "",
        f"- provider: `{args.provider}`",
        f"- model: `{args.model}`",
        f"- count: `{summary['count']}`",
        f"- self_consistency_proxy: `{summary['self_consistency_proxy']}`",
        f"- correctness: `{summary['correctness']}`",
        f"- completeness: `{summary['completeness']}`",
        f"- grounding: `{summary['grounding']}`",
        f"- confidence: `{summary['confidence']}`",
        f"- clarity: `{summary['clarity']}`",
        "",
        "> Warning: this judge sees only the evidence selected by the pipeline itself. "
        "Use it as a self-consistency check, not as a proxy for platform `Assistant` or `Grounding`.",
        "",
    ]
    worst = sorted(rows, key=lambda row: (row["score"], row["question_id"]))[:10]
    if worst:
        md.append("## Lowest-Scoring Answers")
        md.append("")
        for row in worst:
            md.append(f"- `{row['question_id']}` score `{row['score']}`: {row['answer']}")
            md.append(f"  - {row['explanation']}")
    (output_dir / f"{slug}_rubric_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
