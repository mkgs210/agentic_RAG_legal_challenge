from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from rank_bm25 import BM25Okapi

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.api_requests import APIProcessor
from src.lexical_retrieval import tokenize_for_bm25
from src.public_dataset_eval import (
    PublicCorpus,
    build_context,
    normalize_space,
    parse_tagged_field,
    prepare_docling_artifacts,
    safe_json_dump,
    safe_json_load,
)
from src.public_retrieval_benchmark import run_strategy


FREE_TEXT_SYSTEM_PROMPT = """
You answer DIFC legal questions using only the provided context excerpts.
Your job is to produce a final answer that will be judged for correctness, completeness, grounding, confidence calibration, and clarity.

Rules:
- Answer the question directly in 1-3 sentences.
- Keep the final answer at or under 280 characters.
- Cover every part of the question that is supported by the context.
- Do not add facts that are not supported by the context.
- If the answer is absent from the context, return exactly: There is no information on this question in the provided documents.
- Prefer legally precise wording over generic paraphrase.
- For comparison questions, mention both sides when supported by context.

Return exactly these lines and nothing else:
ANSWER: ...
CITATIONS: ref1, ref2, ref3, ref4, ref5
REASONING: one short sentence

Use the minimum number of citations needed, but include multiple citations when the answer relies on multiple clauses or documents.
Citations must be REF ids copied from the context.
""".strip()


COMPRESS_PROMPT = """
Rewrite the answer below to stay within 280 characters.
Do not add new facts.
Keep the legal meaning unchanged.
Return only the rewritten answer text.
""".strip()


def parse_answer_text_loose(response_text: str) -> Dict[str, Any]:
    text = response_text.strip().replace("```", "")
    raw_answer = parse_tagged_field(text, "ANSWER", ["CITATIONS", "REASONING"])
    citations_text = parse_tagged_field(text, "CITATIONS", ["REASONING"])
    reasoning = parse_tagged_field(text, "REASONING", [])
    citations = [normalize_space(part) for part in re.split(r",|;", citations_text) if normalize_space(part)]
    if not raw_answer:
        lines = [normalize_space(line) for line in text.splitlines() if normalize_space(line)]
        raw_answer = lines[0] if lines else ""
    return {
        "raw_answer": raw_answer,
        "citations": citations[:5],
        "reasoning": reasoning,
    }


def compress_answer(api: APIProcessor, model: str, answer: str) -> str:
    if len(answer) <= 280:
        return answer
    response = api.send_message(
        model=model,
        temperature=0.0,
        system_content=COMPRESS_PROMPT,
        human_content=answer,
        is_structured=False,
        max_tokens=140,
        request_timeout=60,
    )
    compressed = normalize_space(response)
    return compressed[:280] if len(compressed) > 280 else compressed


def preview(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def answer_free_text(api: APIProcessor, model: str, question: str, chunks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    context = build_context(chunks)
    response = api.send_message(
        model=model,
        temperature=0.0,
        system_content=FREE_TEXT_SYSTEM_PROMPT,
        human_content=f"Question: {question}\n\nContext:\n{context}",
        is_structured=False,
        max_tokens=260,
        request_timeout=60,
    )
    parsed = parse_answer_text_loose(response)
    parsed["raw_answer"] = compress_answer(api, model, parsed["raw_answer"])
    parsed["response_data"] = getattr(api.processor, "response_data", {})
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a free-text prompt A/B experiment on warm-up questions.")
    parser.add_argument("--questions", default="starter_kit/questions_api.json")
    parser.add_argument("--pdf-dir", default="starter_kit/docs_corpus")
    parser.add_argument("--work-dir", default="starter_kit/challenge_workdir")
    parser.add_argument("--debug-path", default="starter_kit/challenge_workdir/submission_debug.json")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--strategy", default="dense_doc_diverse")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    pdf_dir = Path(args.pdf_dir)
    questions = safe_json_load(Path(args.questions))
    baseline_rows = {row["question_id"]: row for row in safe_json_load(Path(args.debug_path))}
    artifacts = prepare_docling_artifacts(work_dir, pdf_dir)
    corpus = PublicCorpus(work_dir=work_dir, pdf_dir=pdf_dir, chunked_dir=artifacts["chunked_dir"])
    bm25_index = BM25Okapi([tokenize_for_bm25(chunk.text) for chunk in corpus.chunks])
    api = APIProcessor(provider=args.provider)

    results: List[Dict[str, Any]] = []
    for index, item in enumerate(questions, start=1):
        if item["answer_type"] != "free_text":
            continue
        route = corpus.route_question(item["question"], expansive=False)
        reranked = run_strategy(corpus, bm25_index, item["question"], route["candidate_shas"], args.strategy)
        selected_chunks = reranked[: args.top_k]
        answer_payload = answer_free_text(api, args.model, item["question"], selected_chunks)
        baseline = baseline_rows[item["id"]]
        results.append(
            {
                "index": index,
                "question_id": item["id"],
                "question": item["question"],
                "baseline_answer": baseline["submission_answer"],
                "candidate_answer": answer_payload["raw_answer"],
                "candidate_citations": answer_payload["citations"],
                "top_titles": [chunk["title"] for chunk in selected_chunks],
                "top_refs": [chunk["ref"] for chunk in selected_chunks],
                "top_previews": [preview(chunk["text"]) for chunk in selected_chunks],
                "changed": normalize_space(str(baseline["submission_answer"])) != normalize_space(answer_payload["raw_answer"]),
                "baseline_length": len(str(baseline["submission_answer"] or "")),
                "candidate_length": len(answer_payload["raw_answer"]),
            }
        )

    summary = {
        "question_count": len(results),
        "changed_answers": sum(1 for row in results if row["changed"]),
        "baseline_avg_length": round(sum(row["baseline_length"] for row in results) / max(len(results), 1), 1),
        "candidate_avg_length": round(sum(row["candidate_length"] for row in results) / max(len(results), 1), 1),
        "candidate_absent_answers": sum(
            1
            for row in results
            if row["candidate_answer"] == "There is no information on this question in the provided documents."
        ),
    }

    out_json = work_dir / "free_text_prompt_experiment.json"
    out_md = work_dir / "free_text_prompt_experiment.md"
    safe_json_dump(out_json, {"summary": summary, "rows": results})

    lines = [
        "# Warm-up free-text prompt experiment",
        "",
        "## Summary",
        f"- Questions: `{summary['question_count']}`",
        f"- Changed answers: `{summary['changed_answers']}`",
        f"- Baseline average answer length: `{summary['baseline_avg_length']}`",
        f"- Candidate average answer length: `{summary['candidate_avg_length']}`",
        f"- Candidate abstention answers: `{summary['candidate_absent_answers']}`",
        "",
        "## Per-question output",
        "",
    ]
    for row in results:
        lines.extend(
            [
                f"### Q{row['index']:03d}",
                row["question"],
                "",
                f"- Baseline: {row['baseline_answer']}",
                f"- Candidate: {row['candidate_answer']}",
                f"- Candidate citations: {' | '.join(row['candidate_citations']) or 'none'}",
                f"- Top titles: {' | '.join(row['top_titles'])}",
                f"- Top refs: {' | '.join(row['top_refs'])}",
            ]
        )
        for idx, snippet in enumerate(row["top_previews"], start=1):
            lines.append(f"- Top preview {idx}: {snippet}")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
