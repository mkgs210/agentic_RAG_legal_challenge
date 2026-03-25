from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.api_requests import APIProcessor
from src.platform_submission import (
    STARTER_KIT_DIR,
    answer_all_questions,
    ensure_pdf_stage_dir,
    get_config,
    prepare_docling_artifacts,
    resolve_docs_dir,
)
from src.public_dataset_eval import PublicCorpus, safe_json_dump


class ParaphraseResponse(BaseModel):
    paraphrases: List[str]


def canonical_retrieval_pages(row: Dict[str, Any]) -> List[tuple[str, tuple[int, ...]]]:
    result: List[tuple[str, tuple[int, ...]]] = []
    for ref in row.get("retrieval_refs") or []:
        doc_id = str(ref.get("doc_id") or "")
        pages = tuple(sorted(int(page) for page in (ref.get("page_numbers") or [])))
        if doc_id and pages:
            result.append((doc_id, pages))
    return sorted(set(result))


def load_current_snapshot() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    questions = json.loads((STARTER_KIT_DIR / "questions.json").read_text(encoding="utf-8"))
    debug_rows = json.loads((STARTER_KIT_DIR / "challenge_workdir" / "submission_debug.json").read_text(encoding="utf-8"))
    return questions, debug_rows


def select_article_lookup_questions(
    questions: list[dict[str, Any]],
    debug_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for index, (question, row) in enumerate(zip(questions, debug_rows), start=1):
        analysis = row.get("analysis") or {}
        if (row.get("strategy") or "") == "prehandled":
            continue
        if analysis.get("task_family") != "article_lookup":
            continue
        if question.get("answer_type") not in {"boolean", "number", "name"}:
            continue
        selected.append(
            {
                "index": index,
                "id": question["id"],
                "question": question["question"],
                "answer_type": question["answer_type"],
                "baseline_answer": row.get("submission_answer"),
                "baseline_refs": canonical_retrieval_pages(row),
            }
        )
    return selected


def paraphrase_question_batch(
    api: APIProcessor,
    model: str,
    question: str,
    answer_type: str,
    variants: int,
) -> list[str]:
    response = api.send_message(
        model=model,
        temperature=0.0,
        system_content=(
            "You rewrite DIFC legal questions while preserving exact meaning.\n"
            "Do not change scope, answer type, article number, law identity, or legal effect.\n"
            "Keep the rewritten questions answerable from the same evidence.\n"
            "Vary wording and sentence structure substantially.\n"
            "Return JSON only."
        ),
        human_content=(
            f"Original question: {question}\n"
            f"Answer type: {answer_type}\n"
            f"Return exactly {variants} paraphrases.\n"
            "At least one paraphrase should reduce overlap with the original wording while preserving the same legal target.\n"
        ),
        is_structured=True,
        response_format=ParaphraseResponse,
        max_tokens=400,
        request_timeout=60,
    )
    paraphrases = []
    for candidate in response.get("paraphrases", []):
        text = str(candidate or "").strip()
        if text and text.lower() != question.lower() and text not in paraphrases:
            paraphrases.append(text)
    return paraphrases[:variants]


def build_corpus(work_dir: Path) -> PublicCorpus:
    config = get_config()
    docs_dir = resolve_docs_dir(config.docs_dir)
    pdf_dir = ensure_pdf_stage_dir(docs_dir)
    artifacts = prepare_docling_artifacts(work_dir, pdf_dir)
    corpus = PublicCorpus(work_dir, pdf_dir, artifacts["chunked_dir"])
    corpus.save_catalog()
    return corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test deterministic article questions on paraphrased wording.")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--analysis-model", default="gpt-4.1-mini")
    parser.add_argument("--free-text-model", default="gpt-4.1")
    parser.add_argument("--strategy", default="prod_auto_v1")
    parser.add_argument("--top-k-chunks", type=int, default=5)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--work-dir", default="challenge_workdir")
    parser.add_argument("--output-dir", default="starter_kit/challenge_workdir/article_paraphrase_stress")
    args = parser.parse_args()

    questions, debug_rows = load_current_snapshot()
    selected = select_article_lookup_questions(questions, debug_rows)
    if args.limit:
        selected = selected[: args.limit]

    api = APIProcessor(provider=args.provider)
    paraphrased_questions: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for item in selected:
        variants = paraphrase_question_batch(
            api=api,
            model=args.analysis_model,
            question=item["question"],
            answer_type=item["answer_type"],
            variants=args.variants,
        )
        for variant_index, paraphrase in enumerate(variants, start=1):
            manifest.append(
                {
                    **item,
                    "variant_index": variant_index,
                    "paraphrase": paraphrase,
                }
            )
            paraphrased_questions.append(
                {
                    "id": f"{item['id']}__para_{variant_index}",
                    "question": paraphrase,
                    "answer_type": item["answer_type"],
                }
            )

    work_dir = Path(args.work_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = build_corpus(work_dir)

    _, paraphrase_debug = answer_all_questions(
        questions=paraphrased_questions,
        corpus=corpus,
        provider=args.provider,
        model=args.model,
        analysis_model=args.analysis_model,
        free_text_model=args.free_text_model,
        strategy=args.strategy,
        top_k_chunks=args.top_k_chunks,
        limit=None,
        submission_path=None,
        debug_path=None,
        architecture_summary=None,
    )

    results: list[dict[str, Any]] = []
    answer_match_count = 0
    ref_match_count = 0
    for item, row in zip(manifest, paraphrase_debug):
        answer_match = row.get("submission_answer") == item["baseline_answer"]
        ref_match = canonical_retrieval_pages(row) == item["baseline_refs"]
        if answer_match:
            answer_match_count += 1
        if ref_match:
            ref_match_count += 1
        results.append(
            {
                **item,
                "paraphrase_answer": row.get("submission_answer"),
                "paraphrase_refs": canonical_retrieval_pages(row),
                "paraphrase_grounding_method": row.get("grounding_method"),
                "paraphrase_strategy": row.get("strategy"),
                "answer_match": answer_match,
                "ref_match": ref_match,
            }
        )

    summary = {
        "question_count": len(selected),
        "variant_count": len(manifest),
        "answer_match_rate": round(answer_match_count / len(manifest), 4) if manifest else 0.0,
        "ref_match_rate": round(ref_match_count / len(manifest), 4) if manifest else 0.0,
    }

    safe_json_dump(output_dir / "paraphrase_article_stress_results.json", results)
    safe_json_dump(output_dir / "paraphrase_article_stress_summary.json", summary)

    lines = [
        "# Article Paraphrase Stress Eval",
        "",
        f"- questions: `{summary['question_count']}`",
        f"- variants: `{summary['variant_count']}`",
        f"- answer_match_rate: `{summary['answer_match_rate']}`",
        f"- ref_match_rate: `{summary['ref_match_rate']}`",
        "",
        "## Mismatches",
        "",
    ]
    mismatch_count = 0
    for result in results:
        if result["answer_match"] and result["ref_match"]:
            continue
        mismatch_count += 1
        lines.extend(
            [
                f"- Q{result['index']:03d} v{result['variant_index']}",
                f"  original: {result['question']}",
                f"  paraphrase: {result['paraphrase']}",
                f"  baseline_answer: {result['baseline_answer']}",
                f"  paraphrase_answer: {result['paraphrase_answer']}",
                f"  baseline_refs: {result['baseline_refs']}",
                f"  paraphrase_refs: {result['paraphrase_refs']}",
                f"  strategy: {result['paraphrase_strategy']} | grounding: {result['paraphrase_grounding_method']}",
                "",
            ]
        )
    if mismatch_count == 0:
        lines.append("- None")
    (output_dir / "paraphrase_article_stress_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
