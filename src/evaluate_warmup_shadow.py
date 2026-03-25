from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.api_requests import APIProcessor
from src.public_dataset_eval import answer_match, normalize_answer, normalize_space, load_corpus


ROOT = Path("/home/mkgs/hackaton")
DEFAULT_SHADOW_GOLD = ROOT / "starter_kit" / "challenge_workdir" / "warmup_shadow_eval" / "warmup_shadow_gold.json"
DEFAULT_DEBUG = ROOT / "starter_kit" / "challenge_workdir" / "submission_debug.json"
DEFAULT_SUBMISSION = ROOT / "starter_kit" / "submission.json"
DEFAULT_CHUNKED_DIR = ROOT / "starter_kit" / "challenge_workdir" / "docling" / "chunked"
DEFAULT_OUTPUT_DIR = ROOT / "starter_kit" / "challenge_workdir" / "warmup_shadow_eval"
STRUCTURED_TYPES = {"boolean", "number", "name", "names", "date"}
FREE_TEXT_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "under", "case", "law", "court",
    "application", "question", "provided", "documents", "there", "is", "are", "was", "were",
    "of", "to", "in", "on", "a", "an", "by", "as", "at", "it", "be", "or", "any", "no",
}


class FreeTextJudgeVerdict(BaseModel):
    correctness: int = Field(0, ge=0, le=1)
    completeness: int = Field(0, ge=0, le=1)
    grounding: int = Field(0, ge=0, le=1)
    confidence: int = Field(0, ge=0, le=1)
    clarity: int = Field(0, ge=0, le=1)
    explanation: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a warm-up snapshot against local shadow gold.")
    parser.add_argument("--shadow-gold", default=str(DEFAULT_SHADOW_GOLD))
    parser.add_argument("--snapshot-debug", default=str(DEFAULT_DEBUG))
    parser.add_argument("--submission", default=str(DEFAULT_SUBMISSION))
    parser.add_argument("--chunked-dir", default=str(DEFAULT_CHUNKED_DIR))
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--calibration", default="")
    return parser.parse_args()


def load_snapshot(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("items") or payload.get("responses") or []
    raise TypeError(f"Unsupported snapshot payload: {path}")


def load_submission_answers(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(path.read_text())
    answers = payload.get("answers") or []
    return {str(item["question_id"]): item for item in answers if "question_id" in item}


def extract_predicted_pages(answer_payload: Dict[str, Any]) -> List[str]:
    refs = (((answer_payload or {}).get("telemetry") or {}).get("retrieval") or {}).get("retrieved_chunk_pages") or []
    page_refs: List[str] = []
    for ref in refs:
        doc_id = ref.get("doc_id")
        for page in ref.get("page_numbers") or []:
            page_refs.append(f"{doc_id}:{int(page)}")
    return sorted(set(page_refs), key=page_sort_key)


def page_sort_key(ref: str) -> tuple[str, int]:
    sha, page = ref.split(":")
    return sha, int(page)


def f_beta(predicted: Sequence[str], gold: Sequence[str], beta: float = 2.5) -> float:
    pred = set(predicted)
    target = set(gold)
    if not pred and not target:
        return 1.0
    if not pred or not target:
        return 0.0
    tp = len(pred & target)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(target) if target else 0.0
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def load_calibration(path: str) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise TypeError("Calibration must be a JSON object.")
    return payload


def apply_calibration(metrics: Dict[str, float], calibration: Dict[str, Any]) -> Dict[str, float]:
    scale = calibration.get("scale") or {}
    bias = calibration.get("bias") or {}
    adjusted: Dict[str, float] = {}
    for key in ("s_det", "s_asst_shadow", "g_shadow", "t_avg", "f_avg"):
        value = metrics[key]
        adj = value * float(scale.get(key, 1.0)) + float(bias.get(key, 0.0))
        if key in {"s_det", "s_asst_shadow", "g_shadow"}:
            adj = max(0.0, min(1.0, adj))
        if key in {"t_avg", "f_avg"}:
            adj = max(0.0, min(1.2, adj))
        adjusted[key] = round(adj, 4)
    total_adj = (0.7 * adjusted["s_det"] + 0.3 * adjusted["s_asst_shadow"]) * adjusted["g_shadow"] * adjusted["t_avg"] * adjusted["f_avg"]
    adjusted["total_shadow_adjusted"] = round(total_adj, 4)
    return adjusted


def ttft_modifier(ttft_ms: Optional[float]) -> float:
    if ttft_ms is None:
        return 0.85
    if ttft_ms < 1000:
        return 1.05
    if ttft_ms < 2000:
        return 1.02
    if ttft_ms < 3000:
        return 1.00
    if ttft_ms < 5000:
        return 0.95
    return 0.85


def telemetry_modifier(answer_payload: Dict[str, Any]) -> float:
    telemetry = (answer_payload or {}).get("telemetry") or {}
    timing = telemetry.get("timing") or {}
    usage = telemetry.get("usage") or {}
    retrieval = telemetry.get("retrieval") or {}
    required = [
        timing.get("ttft_ms") is not None,
        timing.get("total_time_ms") is not None,
        usage.get("input_tokens") is not None,
        usage.get("output_tokens") is not None,
        retrieval.get("retrieved_chunk_pages") is not None,
    ]
    return 1.0 if all(required) else 0.9


def build_ref_context(gold_row: Dict[str, Any], ref_to_chunk: Dict[str, Dict[str, Any]]) -> str:
    parts: List[str] = []
    for page_ref in gold_row.get("support_pages") or []:
        sha, page = page_ref.split(":")
        matches = [
            chunk for ref, chunk in ref_to_chunk.items()
            if chunk["sha"] == sha and int(chunk["page"]) == int(page)
        ]
        if matches:
            text = "\n\n".join(chunk["text"] for chunk in matches[:2])
            title = matches[0]["title"]
        else:
            text = ""
            title = sha
        parts.append(f"[PAGE {page_ref} | title={title}]\n{text}")
    return "\n\n---\n\n".join(parts)


def judge_prompt() -> str:
    return (
        "You are grading a DIFC legal assistant answer against a reference answer and its supporting pages.\n"
        "Return 1 only if a criterion is clearly satisfied, else 0.\n"
        "Grade strictly.\n"
        "Important strictness rules:\n"
        "- If the predicted answer omits a material fact, legal qualifier, disposition, party, amount, date, or costs consequence that appears in the reference answer, set completeness=0.\n"
        "- If that omission changes the practical meaning of the answer, also set correctness=0.\n"
        "- If the predicted answer adds any fact that is not directly supported by the supporting pages or not present in the reference answer, set grounding=0 and correctness=0.\n"
        "- Bare-span answers to explanatory or outcome questions are incomplete unless they still preserve the full operative meaning.\n"
        "- For absence answers, score correctness=1 only if the reference answer is also an absence answer.\n"
        "Criteria:\n"
        "1. correctness: factual content materially matches the reference answer.\n"
        "2. completeness: covers the key aspects asked in the question.\n"
        "3. grounding: the answer stays within what the supporting pages justify; no unsupported additions.\n"
        "4. confidence: uncertainty is expressed appropriately; no overclaim.\n"
        "5. clarity: concise, coherent, and directly answers the question.\n"
        "When in doubt, score 0.\n"
        "Return JSON only."
    )


def judge_free_text(
    api: APIProcessor,
    model: str,
    question: str,
    reference_answer: str,
    predicted_answer: str,
    supporting_context: str,
) -> Dict[str, Any]:
    system_content = judge_prompt()
    human_content = (
        f"Question: {question}\n\n"
        f"Reference answer:\n{reference_answer}\n\n"
        f"Predicted answer:\n{predicted_answer}\n\n"
        f"Supporting pages for the reference answer:\n{supporting_context}"
    )
    try:
        response = api.send_message(
            model=model,
            temperature=0.0,
            system_content=system_content,
            human_content=human_content,
            is_structured=True,
            response_format=FreeTextJudgeVerdict,
            max_tokens=400,
            request_timeout=90,
        )
    except Exception:
        # Fallback: ask for raw JSON and parse loosely.
        raw = api.send_message(
            model=model,
            temperature=0.0,
            system_content=system_content,
            human_content=human_content,
            is_structured=False,
            max_tokens=400,
            request_timeout=90,
        )
        response = {}
        try:
            response = json.loads(raw)
        except Exception:
            match = re.search(r"\{.*\}", raw or "", re.S)
            if match:
                try:
                    response = json.loads(match.group(0))
                except Exception:
                    response = {}
    score = (
        int(response.get("correctness", 0))
        + int(response.get("completeness", 0))
        + int(response.get("grounding", 0))
        + int(response.get("confidence", 0))
        + int(response.get("clarity", 0))
    ) / 5.0
    return {
        "correctness": int(response.get("correctness", 0)),
        "completeness": int(response.get("completeness", 0)),
        "grounding": int(response.get("grounding", 0)),
        "confidence": int(response.get("confidence", 0)),
        "clarity": int(response.get("clarity", 0)),
        "score": round(score, 4),
        "explanation": normalize_space(str(response.get("explanation", "") or "")),
    }


def free_text_salient_recall(reference_answer: str, predicted_answer: str) -> float:
    ref_tokens = {
        token
        for token in re.findall(r"[a-z0-9/.-]+", normalize_space(reference_answer).lower())
        if len(token) > 2 and token not in FREE_TEXT_STOPWORDS
    }
    if not ref_tokens:
        return 1.0
    pred_tokens = {
        token
        for token in re.findall(r"[a-z0-9/.-]+", normalize_space(predicted_answer).lower())
        if len(token) > 2 and token not in FREE_TEXT_STOPWORDS
    }
    return len(ref_tokens & pred_tokens) / len(ref_tokens)


def main() -> None:
    args = parse_args()
    shadow_gold = json.loads(Path(args.shadow_gold).read_text())
    snapshot_rows = load_snapshot(Path(args.snapshot_debug))
    submission_answers = load_submission_answers(Path(args.submission))
    _, chunks = load_corpus(Path(args.chunked_dir))
    ref_to_chunk = {
        chunk.ref: {
            "ref": chunk.ref,
            "sha": chunk.sha,
            "page": chunk.page,
            "title": chunk.title,
            "text": chunk.text,
        }
        for chunk in chunks
    }
    by_index = {int(row["index"]): row for row in snapshot_rows}
    gold_by_index = {int(row["index"]): row for row in shadow_gold}
    indices = sorted(gold_by_index)
    if args.limit:
        indices = indices[: args.limit]

    api = APIProcessor(provider=args.provider)
    rows: List[Dict[str, Any]] = []
    for index in indices:
        gold = gold_by_index[index]
        pred = by_index[index]
        submission_answer = submission_answers.get(str(gold["question_id"]), {})
        answer_type = str(gold["answer_type"])
        predicted_value = pred.get("submission_answer")
        gold_value = None if gold.get("unanswerable") else gold.get("reference_answer_text")
        if answer_type in STRUCTURED_TYPES:
            predicted_norm = normalize_answer(answer_type, predicted_value)
            gold_norm = normalize_answer(answer_type, gold_value)
            det_score = 1.0 if answer_match(answer_type, gold_norm, predicted_norm) else 0.0
            answer_eval = {"score": det_score, "verdict": "correct" if det_score else "incorrect"}
        else:
            answer_eval = judge_free_text(
                api=api,
                model=args.model,
                question=gold["question"],
                reference_answer=str(gold.get("reference_answer_text") or gold.get("reference_answer") or ""),
                predicted_answer=str(pred.get("raw_answer") or ""),
                supporting_context=build_ref_context(gold, ref_to_chunk),
            )
            coverage_recall = free_text_salient_recall(
                str(gold.get("reference_answer_text") or ""),
                str(pred.get("raw_answer") or ""),
            )
            answer_eval["coverage_recall"] = round(coverage_recall, 4)
            answer_eval["score_llm"] = answer_eval["score"]
            answer_eval["score"] = round(math.sqrt(answer_eval["score"] * coverage_recall), 4)

        predicted_pages = extract_predicted_pages(submission_answer)
        g_score = round(f_beta(predicted_pages, gold.get("support_pages") or []), 4)
        t_score = telemetry_modifier(submission_answer)
        timing = (((submission_answer or {}).get("telemetry") or {}).get("timing") or {})
        f_score = ttft_modifier(timing.get("ttft_ms"))

        rows.append(
            {
                "index": index,
                "question_id": gold["question_id"],
                "answer_type": answer_type,
                "question": gold["question"],
                "reference_answer_text": gold.get("reference_answer_text"),
                "predicted_answer": predicted_value,
                "answer_score": float(answer_eval["score"]),
                "answer_eval": answer_eval,
                "gold_support_pages": gold.get("support_pages") or [],
                "predicted_pages": predicted_pages,
                "g_score": g_score,
                "t_score": t_score,
                "f_score": f_score,
                "ttft_ms": timing.get("ttft_ms"),
                "total_time_ms": timing.get("total_time_ms"),
            }
        )

    det_scores = [row["answer_score"] for row in rows if row["answer_type"] in STRUCTURED_TYPES]
    asst_scores = [row["answer_score"] for row in rows if row["answer_type"] == "free_text"]
    g_scores = [row["g_score"] for row in rows]
    t_scores = [row["t_score"] for row in rows]
    f_scores = [row["f_score"] for row in rows]
    s_det = mean(det_scores) if det_scores else 0.0
    s_asst = mean(asst_scores) if asst_scores else 0.0
    g_avg = mean(g_scores) if g_scores else 0.0
    t_avg = mean(t_scores) if t_scores else 0.0
    f_avg = mean(f_scores) if f_scores else 0.0
    total = (0.7 * s_det + 0.3 * s_asst) * g_avg * t_avg * f_avg
    calibration = load_calibration(args.calibration)

    summary = {
        "count": len(rows),
        "structured_count": len(det_scores),
        "free_text_count": len(asst_scores),
        "s_det": round(s_det, 4),
        "s_asst_shadow": round(s_asst, 4),
        "g_shadow": round(g_avg, 4),
        "t_avg": round(t_avg, 4),
        "f_avg": round(f_avg, 4),
        "total_shadow": round(total, 4),
        "limitations": [
            "This is a local shadow evaluator, not the official platform evaluator.",
            "Reference answers and support pages come from local shadow gold.",
            "Use it for relative iteration and regression detection, not leaderboard claims.",
        ],
    }
    if calibration:
        summary["calibration_path"] = args.calibration
        summary["adjusted"] = apply_calibration(
            {
                "s_det": s_det,
                "s_asst_shadow": s_asst,
                "g_shadow": g_avg,
                "t_avg": t_avg,
                "f_avg": f_avg,
            },
            calibration,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = Path(args.snapshot_debug).stem
    (output_dir / f"{slug}_shadow_rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    (output_dir / f"{slug}_shadow_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    md_lines = [
        f"# Warm-up Shadow Eval: {slug}",
        "",
        f"- count: `{summary['count']}`",
        f"- s_det: `{summary['s_det']}`",
        f"- s_asst_shadow: `{summary['s_asst_shadow']}`",
        f"- g_shadow: `{summary['g_shadow']}`",
        f"- t_avg: `{summary['t_avg']}`",
        f"- f_avg: `{summary['f_avg']}`",
        f"- total_shadow: `{summary['total_shadow']}`",
        f"- calibration: `{summary.get('calibration_path', '')}`",
        "",
        "## Limitations",
        "",
    ]
    if "adjusted" in summary:
        adjusted = summary["adjusted"]
        md_lines.extend(
            [
                "",
                "## Adjusted (Calibrated)",
                "",
                f"- s_det_adjusted: `{adjusted['s_det']}`",
                f"- s_asst_shadow_adjusted: `{adjusted['s_asst_shadow']}`",
                f"- g_shadow_adjusted: `{adjusted['g_shadow']}`",
                f"- t_avg_adjusted: `{adjusted['t_avg']}`",
                f"- f_avg_adjusted: `{adjusted['f_avg']}`",
                f"- total_shadow_adjusted: `{adjusted['total_shadow_adjusted']}`",
            ]
        )
    for item in summary["limitations"]:
        md_lines.append(f"- {item}")
    md_lines.extend(["", "## Lowest-Scoring Questions", ""])
    for row in sorted(rows, key=lambda item: (item["answer_score"] * item["g_score"], item["index"]))[:15]:
        md_lines.append(
            f"- Q{row['index']:03d} `{row['answer_type']}`: answer `{row['answer_score']}` | G `{row['g_score']}` | predicted `{row['predicted_answer']}`"
        )
    (output_dir / f"{slug}_shadow_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
