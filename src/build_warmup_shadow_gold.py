from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.api_requests import APIProcessor
from src.public_dataset_eval import PublicCorpus, safe_json_dump, safe_json_load


ROOT = Path("/home/mkgs/hackaton")
DEFAULT_WORK_DIR = ROOT / "starter_kit" / "challenge_workdir"
DEFAULT_CHUNKED_DIR = DEFAULT_WORK_DIR / "docling" / "chunked"
DEFAULT_MERGED_DIR = DEFAULT_WORK_DIR / "docling" / "merged"
DEFAULT_QUESTIONS = ROOT / "starter_kit" / "submission_history" / "20260312_011749" / "questions.json"
DEFAULT_AUDIT_CSV = ROOT / "starter_kit" / "submission_history" / "20260312_011749" / "WARMUP_RERANKER_MANUAL_AUDIT.csv"
DEFAULT_OUTPUT = ROOT / "starter_kit" / "challenge_workdir" / "warmup_shadow_eval" / "warmup_shadow_gold.json"
DEFAULT_SNAPSHOTS = [
    ROOT / "starter_kit" / "submission_history" / "20260312_011749" / "challenge_workdir__submission_debug_iter5_mini.json",
    ROOT / "starter_kit" / "submission_history" / "20260312_011749" / "challenge_workdir__submission_debug_iter10_rubric_fix.json",
    ROOT / "starter_kit" / "submission_history" / "20260312_011749" / "challenge_workdir__submission_debug_iter11_grounding_complete.json",
    ROOT / "starter_kit" / "submission_history" / "20260312_011749" / "challenge_workdir__submission_debug_iter14_debt_reduction.json",
    ROOT / "starter_kit" / "submission_history" / "20260312_011749" / "challenge_workdir__submission_debug_iter15_route_guard.json",
    ROOT / "starter_kit" / "challenge_workdir" / "submission_debug.json",
]


class GoldRecord(BaseModel):
    answer_text: str = ""
    support_pages: List[str] = Field(default_factory=list)
    unanswerable: bool = False
    confidence: str = "medium"
    rationale: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local warm-up shadow gold from questions and candidate evidence pages.")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--chunked-dir", default=str(DEFAULT_CHUNKED_DIR))
    parser.add_argument("--merged-dir", default=str(DEFAULT_MERGED_DIR))
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--audit-csv", default=str(DEFAULT_AUDIT_CSV))
    parser.add_argument("--snapshot", action="append", dest="snapshots", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--with-fresh-retrieval", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_snapshot_rows(path: Path) -> Dict[int, Dict[str, Any]]:
    payload = safe_json_load(path)
    if isinstance(payload, dict):
        rows = payload.get("items") or payload.get("responses") or []
    else:
        rows = payload
    if not isinstance(rows, list):
        return {}
    return {int(row["index"]): row for row in rows if isinstance(row, dict) and "index" in row}


def parse_audit_answers(path: Path) -> Dict[int, str]:
    rows: Dict[int, str] = {}
    if not path.exists():
        return rows
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return rows
    headers = [part.strip() for part in lines[0].split(",")]
    index_pos = headers.index("q_index")
    answer_pos = headers.index("submission_answer")
    for raw_line in lines[1:]:
        parts = [part.strip() for part in next(csv_row_iter(raw_line))]
        if len(parts) <= max(index_pos, answer_pos):
            continue
        try:
            q_index = int(parts[index_pos])
        except ValueError:
            continue
        rows[q_index] = parts[answer_pos]
    return rows


def csv_row_iter(line: str) -> Iterable[List[str]]:
    import csv

    yield from csv.reader([line])


def parse_chunk_page_ref(ref: str) -> Optional[str]:
    match = re.match(r"^([0-9a-f]{64}):(\d+)(?::\d+)?$", str(ref).strip())
    if not match:
        return None
    return f"{match.group(1)}:{int(match.group(2))}"


def page_sort_key(ref: str) -> Tuple[str, int]:
    sha, page = ref.split(":")
    return sha, int(page)


def load_page_texts(merged_dir: Path) -> Dict[str, Dict[int, str]]:
    page_texts: Dict[str, Dict[int, str]] = {}
    for path in sorted(merged_dir.glob("*.json")):
        payload = safe_json_load(path)
        sha = payload["metainfo"]["sha1_name"]
        pages = payload["content"]["pages"]
        page_texts[sha] = {int(page["page"]): str(page["text"]) for page in pages}
    return page_texts


def collect_snapshot_page_refs(row: Dict[str, Any]) -> List[str]:
    refs: List[str] = []
    for retrieval in row.get("retrieval_refs") or []:
        doc_id = retrieval.get("doc_id")
        for page in retrieval.get("page_numbers") or []:
            refs.append(f"{doc_id}:{int(page)}")
    for field in ("answer_chunk_refs", "top_chunk_refs", "candidate_page_refs", "citations"):
        for raw_ref in row.get(field) or []:
            page_ref = parse_chunk_page_ref(str(raw_ref))
            if page_ref:
                refs.append(page_ref)
    return refs


def collect_candidate_answers(snapshot_rows: Sequence[Dict[str, Any]], audit_answer: str) -> List[str]:
    answers: List[str] = []
    if audit_answer:
        answers.append(audit_answer.strip())
    for row in snapshot_rows:
        raw = str(row.get("raw_answer") or "").strip()
        if raw:
            answers.append(raw)
    return answers


def pick_consensus_answer(answer_type: str, answers: Sequence[str]) -> str:
    normalized: Counter[str] = Counter()
    raw_by_norm: Dict[str, str] = {}
    for raw in answers:
        compact = re.sub(r"\s+", " ", raw).strip()
        if not compact:
            continue
        norm = compact.lower()
        normalized[norm] += 1
        raw_by_norm.setdefault(norm, compact)
    if not normalized:
        return ""
    best_norm, _ = normalized.most_common(1)[0]
    if answer_type == "free_text":
        candidates = [raw_by_norm[norm] for norm, count in normalized.items() if count == normalized[best_norm]]
        return max(candidates, key=len)
    return raw_by_norm[best_norm]


def build_candidate_pages(
    *,
    corpus: PublicCorpus,
    question: str,
    snapshot_rows: Sequence[Dict[str, Any]],
    page_texts: Dict[str, Dict[int, str]],
    with_fresh_retrieval: bool,
) -> List[Dict[str, Any]]:
    counts: Counter[str] = Counter()
    sources: Dict[str, set[str]] = defaultdict(set)
    for row in snapshot_rows:
        for page_ref in collect_snapshot_page_refs(row):
            counts[page_ref] += 1
            sources[page_ref].add("snapshot")

    top_docs: List[str] = []
    if with_fresh_retrieval:
        route = corpus.route_question(question, expansive=True)
        retrieval = corpus.retrieve(
            question=question,
            candidate_shas=route["candidate_shas"],
            vector_k=16,
            rerank_k=8,
            lexical_boost=True,
            bm25_k=6,
            bm25_weight=0.2,
            bm25_auto=True,
        )
        for item in retrieval["reranked_results"]:
            page_ref = f"{item['sha']}:{int(item['page'])}"
            counts[page_ref] += 1
            sources[page_ref].add("retrieval")
        top_docs = [item["sha"] for item in retrieval["reranked_results"][:4]]
    else:
        doc_counts: Counter[str] = Counter()
        for page_ref in counts:
            doc_counts[page_ref.split(":")[0]] += counts[page_ref]
        top_docs = [sha for sha, _ in doc_counts.most_common(4)]

    for sha in top_docs:
        page_ref = f"{sha}:1"
        counts[page_ref] += 1
        sources[page_ref].add("doc_page_1")

    ranked_refs = sorted(
        counts,
        key=lambda ref: (
            -counts[ref],
            0 if "doc_page_1" in sources[ref] else 1,
            page_sort_key(ref),
        ),
    )
    selected = ranked_refs[:12]
    pages: List[Dict[str, Any]] = []
    for page_ref in selected:
        sha, raw_page = page_ref.split(":")
        page = int(raw_page)
        text = (page_texts.get(sha, {}) or {}).get(page, "")
        if not text:
            continue
        document = corpus.documents[sha]
        pages.append(
            {
                "page_ref": page_ref,
                "doc_id": sha,
                "page": page,
                "title": document.title,
                "kind": document.kind,
                "canonical_ids": document.canonical_ids,
                "text": text[:4000],
                "score_hint": counts[page_ref],
                "source_tags": sorted(sources[page_ref]),
            }
        )
    return pages


def build_gold_prompt(question: str, answer_type: str, candidate_answer: str, pages: Sequence[Dict[str, Any]]) -> str:
    page_blocks: List[str] = []
    for page in pages:
        ids = ", ".join(page["canonical_ids"])
        header = f"PAGE_REF {page['page_ref']} | title={page['title']} | page={page['page']}"
        if ids:
            header += f" | ids={ids}"
        page_blocks.append(f"[{header}]\n{page['text']}")
    guidance = (
        "Task: build a local reference answer and the minimal exact page set needed to answer.\n"
        "Rules:\n"
        "- Use only the candidate pages below.\n"
        "- support_pages must be the minimal exact page_ref list needed for the answer.\n"
        "- Do not include extra pages.\n"
        "- If the question is unanswerable from the pages, set unanswerable=true and support_pages=[].\n"
        "- For deterministic types (boolean, number, name, names, date), use JSON null for an unanswerable answer.\n"
        "- For free_text, if unanswerable, answer with a short natural-language absence answer.\n"
        "- Prefer the latest/consolidated edition if the same information appears in multiple editions.\n"
        "- Page numbers are 1-based.\n"
    )
    seed = ""
    if candidate_answer and answer_type in {"boolean", "number", "name", "names", "date"}:
        seed = f"Candidate answer hint: {candidate_answer}\n\n"
    return (
        f"{guidance}\nQuestion: {question}\nAnswer type: {answer_type}\n\n{seed}"
        f"Candidate pages:\n\n" + "\n\n---\n\n".join(page_blocks)
    )


def build_gold_system_prompt() -> str:
    return (
        "You are building a local evaluation key for DIFC legal QA.\n"
        "Be conservative and exact.\n"
        "The answer must be fully supported by the selected pages.\n"
        "Choose the minimal page set that is sufficient.\n"
        "Return structured JSON only."
    )


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.force:
        print(json.dumps({"status": "exists", "output": str(output_path)}, ensure_ascii=False))
        return

    questions = safe_json_load(Path(args.questions))
    corpus = PublicCorpus(Path(args.work_dir), ROOT / "starter_kit" / "challenge_workdir" / "pdfs", Path(args.chunked_dir))
    page_texts = load_page_texts(Path(args.merged_dir))
    audit_answers = parse_audit_answers(Path(args.audit_csv))
    snapshot_paths = [Path(item) for item in (args.snapshots or [str(path) for path in DEFAULT_SNAPSHOTS]) if Path(item).exists()]
    snapshots = {path.name: load_snapshot_rows(path) for path in snapshot_paths}

    api = APIProcessor(provider=args.provider)
    rows: List[Dict[str, Any]] = []
    limit = args.limit or len(questions)

    for index, item in enumerate(questions[:limit], start=1):
        snapshot_rows = [rows_by_index[index] for rows_by_index in snapshots.values() if index in rows_by_index]
        candidate_answer = pick_consensus_answer(
            item["answer_type"],
            collect_candidate_answers(snapshot_rows, audit_answers.get(index, "")),
        )
        candidate_pages = build_candidate_pages(
            corpus=corpus,
            question=item["question"],
            snapshot_rows=snapshot_rows,
            page_texts=page_texts,
            with_fresh_retrieval=args.with_fresh_retrieval,
        )
        response = api.send_message(
            model=args.model,
            temperature=0.0,
            system_content=build_gold_system_prompt(),
            human_content=build_gold_prompt(
                question=item["question"],
                answer_type=item["answer_type"],
                candidate_answer=candidate_answer,
                pages=candidate_pages,
            ),
            is_structured=True,
            response_format=GoldRecord,
            max_tokens=700,
            request_timeout=120,
        )
        rows.append(
            {
                "index": index,
                "question_id": item["id"],
                "question": item["question"],
                "answer_type": item["answer_type"],
                "reference_answer_text": str(response.get("answer_text") or ""),
                "support_pages": sorted(set(response.get("support_pages") or []), key=page_sort_key),
                "unanswerable": bool(response.get("unanswerable")),
                "confidence": str(response.get("confidence") or "medium"),
                "rationale": str(response.get("rationale") or ""),
                "candidate_answer_seed": candidate_answer,
                "candidate_pages": candidate_pages,
            }
        )
        safe_json_dump(output_path, rows)

    summary = {
        "count": len(rows),
        "questions_path": str(Path(args.questions)),
        "snapshots": [str(path) for path in snapshot_paths],
        "provider": args.provider,
        "model": args.model,
        "output": str(output_path),
    }
    safe_json_dump(output_path.with_name(output_path.stem + "_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
