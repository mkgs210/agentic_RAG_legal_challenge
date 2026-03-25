from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.legal_runtime_index import (
    ArticleHit,
    EntityHit,
    LegalRuntimeIndex,
    PageHit,
    build_index,
    extract_article_refs,
    extract_case_ids,
    extract_law_ids,
    normalized_text,
    normalize_space,
    open_index,
    tokenize_words,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORK_DIR = ROOT / "starter_kit" / "challenge_workdir"
DEFAULT_DOC_CATALOG_PATH = DEFAULT_WORK_DIR / "doc_catalog.json"
DEFAULT_QUESTIONS_PATH = ROOT / "starter_kit" / "submission_history" / "20260312_011749" / "questions.json"
DEFAULT_DB_PATH = DEFAULT_WORK_DIR / "runtime_index" / "legal_runtime_index.sqlite"
DEFAULT_REPORT_DIR = (
    ROOT / "starter_kit" / "submission_history" / "20260312_011749" / "runtime_index_route_harness_20260318"
)

STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "to",
    "in",
    "on",
    "for",
    "under",
    "with",
    "what",
    "which",
    "who",
    "when",
    "where",
    "is",
    "was",
    "were",
    "be",
    "that",
    "this",
    "those",
    "these",
    "there",
    "their",
    "same",
}


@dataclass(slots=True)
class RouteDiffRow:
    index: int
    question_id: str
    answer_type: str
    question: str
    base_docs: list[str]
    fts_docs: list[str]
    added_docs: list[str]
    fts_pages: list[str]
    net_new_pages: list[str]
    page_refs_in_base_docs: list[str]
    route_mode: str
    query_variants: list[str]
    doc_sources: dict[str, list[str]]
    page_sources: dict[str, list[str]]
    entity_sources: dict[str, list[str]]
    article_sources: dict[str, list[str]]


@dataclass(slots=True)
class BaseDocument:
    sha: str
    kind: str
    aliases: list[str]
    canonical_ids: list[str]


class BaseRouteCorpus:
    def __init__(self, doc_catalog_path: Path) -> None:
        payload = load_json(doc_catalog_path)
        documents = payload.get("documents", []) if isinstance(payload, dict) else []
        self.documents: dict[str, BaseDocument] = {}
        self.id_index: dict[str, list[str]] = defaultdict(list)
        self.alias_index: dict[str, list[str]] = defaultdict(list)
        for raw in documents:
            sha = str(raw.get("sha") or raw.get("sha1_name") or "").strip()
            if not sha:
                continue
            kind = str(raw.get("kind") or "unknown")
            aliases = [normalize_space(str(item)) for item in (raw.get("aliases") or []) if normalize_space(str(item))]
            canonical_ids = [normalize_space(str(item)) for item in (raw.get("canonical_ids") or []) if normalize_space(str(item))]
            document = BaseDocument(sha=sha, kind=kind, aliases=aliases, canonical_ids=canonical_ids)
            self.documents[sha] = document
            for canonical_id in canonical_ids:
                self.id_index[canonical_id].append(sha)
            for alias in aliases:
                self.alias_index[normalized_text(alias)].append(sha)

    def route_question(self, question: str, expansive: bool = False) -> dict[str, Any]:
        explicit_case_ids = extract_case_ids(question)
        explicit_law_ids = extract_law_ids(question)
        question_norm = normalized_text(question)
        aliases_hit: list[str] = []
        candidate_shas: list[str] = []

        for canonical_id in explicit_case_ids + explicit_law_ids:
            candidate_shas.extend(self.id_index.get(canonical_id, []))

        for alias_norm, shas in self.alias_index.items():
            if len(alias_norm) < 6:
                continue
            if re.search(rf"\b{re.escape(alias_norm)}\b", question_norm):
                candidate_shas.extend(shas)
                aliases_hit.append(alias_norm)

        if expansive:
            if not candidate_shas and ("law" in question_norm or "regulation" in question_norm):
                candidate_shas.extend([sha for sha, document in self.documents.items() if document.kind in {"law", "regulation"}])
            elif not candidate_shas:
                candidate_shas.extend(self.documents.keys())
        else:
            if not candidate_shas and explicit_case_ids:
                candidate_shas.extend([sha for sha, document in self.documents.items() if document.kind == "case"])
            elif not candidate_shas and ("law" in question_norm or "regulation" in question_norm or "article" in question_norm):
                candidate_shas.extend([sha for sha, document in self.documents.items() if document.kind in {"law", "regulation"}])
            elif not candidate_shas:
                candidate_shas.extend(self.documents.keys())

        deduped = sorted(dict.fromkeys(candidate_shas))
        return {
            "candidate_shas": deduped,
            "explicit_case_ids": explicit_case_ids,
            "explicit_law_ids": explicit_law_ids,
            "alias_hits": sorted(set(aliases_hit)),
        }


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_questions(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise TypeError(f"Expected a list of questions in {path}")
    return payload


def load_runtime_index(work_dir: Path, db_path: Path, rebuild: bool) -> LegalRuntimeIndex:
    if rebuild or not db_path.exists():
        build_index(work_dir, db_path, source_variant="chunked_section_aware")
    return open_index(db_path)


def query_variants(question: str) -> list[str]:
    variants: list[str] = []
    question = normalize_space(question)
    if question:
        variants.append(question)

    anchors = extract_case_ids(question) + extract_law_ids(question) + extract_article_refs(question)
    if anchors:
        variants.append(" ".join(anchors))

    words = [
        token
        for token in re.findall(r"[A-Za-z0-9/.-]+", question)
        if len(token) > 2 and token.lower() not in STOPWORDS
    ]
    if words:
        variants.append(" ".join(words[:12]))

    quoted = re.findall(r'"([^"]+)"', question)
    for phrase in quoted[:2]:
        phrase = normalize_space(phrase)
        if phrase:
            variants.append(phrase)

    return list(dict.fromkeys(variant for variant in variants if variant))


def _sqlite_list(values: Sequence[str]) -> str:
    return ",".join("?" for _ in values)


def _run_document_hits(db_path: Path, fts_query: str, *, kinds: Sequence[str] | None, limit: int) -> list[dict[str, Any]]:
    clauses: list[str] = []
    args: list[Any] = []
    if kinds:
        clauses.append(f"d.kind IN ({_sqlite_list(kinds)})")
        args.extend(kinds)
    where_clause = f" AND {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT d.doc_id, d.kind, d.title, d.aliases, d.first_page_text, d.page_count,
               d.case_id, d.law_id, bm25(document_fts) AS score
        FROM document_fts
        JOIN documents AS d USING(doc_id)
        WHERE document_fts MATCH ? {where_clause}
        ORDER BY score ASC
        LIMIT ?
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, [fts_query, *args, int(limit)]).fetchall()
    return [dict(row) for row in rows]


def _run_page_hits(
    db_path: Path,
    fts_query: str,
    *,
    kinds: Sequence[str] | None,
    page_roles: Sequence[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    args: list[Any] = []
    if kinds:
        clauses.append(f"doc_kind IN ({_sqlite_list(kinds)})")
        args.extend(kinds)
    if page_roles:
        clauses.append(f"page_role IN ({_sqlite_list(page_roles)})")
        args.extend(page_roles)
    where_clause = f" AND {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT doc_id, page_number, doc_kind, title, page_role, tags, content, bm25(page_fts) AS score
        FROM page_fts
        WHERE page_fts MATCH ? {where_clause}
        ORDER BY score ASC
        LIMIT ?
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, [fts_query, *args, int(limit)]).fetchall()
    return [dict(row) for row in rows]


def _run_entity_hits(
    db_path: Path,
    fts_query: str,
    *,
    kinds: Sequence[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    args: list[Any] = []
    if kinds:
        clauses.append(f"kind IN ({_sqlite_list(kinds)})")
        args.extend(kinds)
    where_clause = f" AND {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT doc_id, page_number, kind, value, surface, tags, bm25(anchor_fts) AS score
        FROM anchor_fts
        WHERE anchor_fts MATCH ? {where_clause}
        ORDER BY score ASC
        LIMIT ?
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, [fts_query, *args, int(limit)]).fetchall()
    return [dict(row) for row in rows]


def _run_article_hits(db_path: Path, fts_query: str, *, doc_ids: Sequence[str] | None, limit: int) -> list[dict[str, Any]]:
    clauses: list[str] = []
    args: list[Any] = []
    if doc_ids:
        clauses.append(f"doc_id IN ({_sqlite_list(doc_ids)})")
        args.extend(doc_ids)
    where_clause = f" AND {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT doc_id, page_number, article_ref, heading, section_kind, content, bm25(article_fts) AS score
        FROM article_fts
        WHERE article_fts MATCH ? {where_clause}
        ORDER BY score ASC
        LIMIT ?
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, [fts_query, *args, int(limit)]).fetchall()
    return [dict(row) for row in rows]


def _add_doc_hit(store: dict[str, set[str]], hit: Any, source: str) -> None:
    store.setdefault(hit.doc_id, set()).add(source)


def _add_page_hit(store: dict[str, set[str]], hit: Any, source: str) -> None:
    store.setdefault(f"{hit.doc_id}:{int(hit.page_number)}", set()).add(source)


def expand_with_runtime_index(index: LegalRuntimeIndex, question: str, limit: int = 8) -> dict[str, Any]:
    route = index.route_query(question)
    variants = query_variants(question)

    doc_sources: dict[str, set[str]] = defaultdict(set)
    page_sources: dict[str, set[str]] = defaultdict(set)
    entity_sources: dict[str, set[str]] = defaultdict(set)
    article_sources: dict[str, set[str]] = defaultdict(set)

    kinds = route.get("kinds") or None
    page_roles = route.get("page_roles") or None
    entity_kinds = route.get("entity_kinds") or None

    for variant in variants:
        fts_query = index._fts_query(variant)
        for row in _run_document_hits(index.db_path, fts_query, kinds=kinds, limit=limit):
            _add_doc_hit(doc_sources, type("Hit", (), row)(), f"documents:{variant}")
        for row in _run_page_hits(index.db_path, fts_query, kinds=kinds, page_roles=page_roles, limit=limit):
            hit = type("Hit", (), row)()
            _add_doc_hit(doc_sources, hit, f"pages:{variant}")
            _add_page_hit(page_sources, hit, f"pages:{variant}")
        for row in _run_entity_hits(index.db_path, fts_query, kinds=entity_kinds, limit=limit):
            hit = type("Hit", (), row)()
            _add_doc_hit(doc_sources, hit, f"entities:{variant}")
            _add_page_hit(entity_sources, hit, f"entities:{variant}")
        for row in _run_article_hits(index.db_path, fts_query, doc_ids=None, limit=limit):
            hit = type("Hit", (), row)()
            _add_doc_hit(doc_sources, hit, f"articles:{variant}")
            _add_page_hit(article_sources, hit, f"articles:{variant}")

    docs = sorted(doc_sources)
    page_refs = sorted(set(page_sources) | set(entity_sources) | set(article_sources))
    return {
        "route_mode": route.get("mode", ""),
        "route": route,
        "query_variants": variants,
        "docs": docs,
        "page_refs": page_refs,
        "doc_sources": {key: sorted(value) for key, value in doc_sources.items()},
        "page_sources": {key: sorted(value) for key, value in page_sources.items()},
        "entity_sources": {key: sorted(value) for key, value in entity_sources.items()},
        "article_sources": {key: sorted(value) for key, value in article_sources.items()},
    }


def compare_routes(
    corpus: BaseRouteCorpus,
    index: LegalRuntimeIndex,
    question_row: dict[str, Any],
    *,
    row_index: int,
    limit: int = 8,
) -> RouteDiffRow:
    question = str(question_row.get("question") or "")
    base_route = corpus.route_question(question, expansive=False)
    runtime = expand_with_runtime_index(index, question, limit=limit)

    base_docs = sorted(dict.fromkeys(base_route.get("candidate_shas") or []))
    fts_docs = runtime["docs"]
    added_docs = sorted(set(fts_docs) - set(base_docs))

    fts_pages = runtime["page_refs"]
    net_new_pages = []
    page_refs_in_base_docs = []
    for ref in fts_pages:
        doc_id, _, page_number = ref.partition(":")
        if doc_id in base_docs:
            page_refs_in_base_docs.append(ref)
        else:
            net_new_pages.append(ref)

    return RouteDiffRow(
        index=int(row_index),
        question_id=str(question_row.get("id") or ""),
        answer_type=str(question_row.get("answer_type") or ""),
        question=question,
        base_docs=base_docs,
        fts_docs=fts_docs,
        added_docs=added_docs,
        fts_pages=fts_pages,
        net_new_pages=sorted(net_new_pages),
        page_refs_in_base_docs=sorted(page_refs_in_base_docs),
        route_mode=str(runtime["route_mode"]),
        query_variants=list(runtime["query_variants"]),
        doc_sources={key: sorted(value) for key, value in runtime["doc_sources"].items()},
        page_sources={key: sorted(value) for key, value in runtime["page_sources"].items()},
        entity_sources={key: sorted(value) for key, value in runtime["entity_sources"].items()},
        article_sources={key: sorted(value) for key, value in runtime["article_sources"].items()},
    )


def summarize(rows: Sequence[RouteDiffRow]) -> dict[str, Any]:
    questions_with_added_docs = sum(1 for row in rows if row.added_docs)
    questions_with_new_pages = sum(1 for row in rows if row.net_new_pages)
    total_added_docs = sum(len(row.added_docs) for row in rows)
    total_added_pages = sum(len(row.net_new_pages) for row in rows)
    total_base_docs = sum(len(row.base_docs) for row in rows)
    total_fts_docs = sum(len(row.fts_docs) for row in rows)
    total_fts_pages = sum(len(row.fts_pages) for row in rows)
    route_modes = Counter(row.route_mode for row in rows)
    top_doc_expansion = sorted(rows, key=lambda row: (len(row.added_docs), len(row.net_new_pages), -len(row.base_docs)), reverse=True)[:10]
    top_page_expansion = sorted(rows, key=lambda row: (len(row.net_new_pages), len(row.added_docs), -len(row.base_docs)), reverse=True)[:10]

    return {
        "questions": len(rows),
        "questions_with_added_docs": questions_with_added_docs,
        "questions_with_new_pages": questions_with_new_pages,
        "total_base_docs": total_base_docs,
        "total_fts_docs": total_fts_docs,
        "total_added_docs": total_added_docs,
        "total_fts_pages": total_fts_pages,
        "total_added_pages": total_added_pages,
        "avg_base_docs": round(total_base_docs / max(1, len(rows)), 3),
        "avg_fts_docs": round(total_fts_docs / max(1, len(rows)), 3),
        "avg_added_docs": round(total_added_docs / max(1, len(rows)), 3),
        "avg_fts_pages": round(total_fts_pages / max(1, len(rows)), 3),
        "avg_added_pages": round(total_added_pages / max(1, len(rows)), 3),
        "route_modes": dict(route_modes),
        "top_doc_expansion": [asdict(row) for row in top_doc_expansion],
        "top_page_expansion": [asdict(row) for row in top_page_expansion],
    }


def render_markdown(summary: dict[str, Any], rows: Sequence[RouteDiffRow]) -> str:
    lines: list[str] = []
    lines.append("# Runtime Index Route Harness")
    lines.append("")
    lines.append("This report compares the current coarse route against a SQLite/FTS expansion built from the legal runtime index.")
    lines.append("")
    lines.append("## Summary")
    for key in [
        "questions",
        "questions_with_added_docs",
        "questions_with_new_pages",
        "total_base_docs",
        "total_fts_docs",
        "total_added_docs",
        "total_fts_pages",
        "total_added_pages",
        "avg_base_docs",
        "avg_fts_docs",
        "avg_added_docs",
        "avg_fts_pages",
        "avg_added_pages",
    ]:
        lines.append(f"- `{key}`: `{summary[key]}`")
    lines.append("")
    lines.append("## Route Modes")
    for mode, count in sorted(summary["route_modes"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{mode}`: `{count}`")
    lines.append("")
    lines.append("## Top Doc Expansions")
    lines.append("| idx | answer_type | added_docs | net_new_pages | base_docs | question |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in summary["top_doc_expansion"]:
        lines.append(
            f"| {row['index']} | {row['answer_type']} | {len(row['added_docs'])} | {len(row['net_new_pages'])} | {len(row['base_docs'])} | {row['question'][:120].replace('|', '\\|')} |"
        )
    lines.append("")
    lines.append("## Top Page Expansions")
    lines.append("| idx | answer_type | net_new_pages | added_docs | question |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in summary["top_page_expansion"]:
        lines.append(
            f"| {row['index']} | {row['answer_type']} | {len(row['net_new_pages'])} | {len(row['added_docs'])} | {row['question'][:120].replace('|', '\\|')} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- `base_docs` comes from the existing coarse `PublicCorpus.route_question(...)` route.")
    lines.append("- `fts_docs` and `fts_pages` are the union of multiple SQLite/FTS query variants.")
    lines.append("- `net_new_pages` counts page refs whose document was not already in the base route.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare base route candidates with SQLite/FTS route expansion.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--doc-catalog-path", type=Path, default=DEFAULT_DOC_CATALOG_PATH)
    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--max-questions", type=int, default=0, help="Optional cap for faster smoke runs.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    questions = load_questions(args.questions_path)
    if args.max_questions and args.max_questions > 0:
        questions = questions[: args.max_questions]

    corpus = BaseRouteCorpus(args.doc_catalog_path)
    index = load_runtime_index(args.work_dir, args.db_path, rebuild=args.rebuild_index)

    rows = [
        compare_routes(corpus, index, question_row, row_index=row_index, limit=args.limit)
        for row_index, question_row in enumerate(questions, start=1)
    ]
    summary = summarize(rows)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report_json = args.report_dir / "runtime_index_route_harness_report.json"
    report_md = args.report_dir / "runtime_index_route_harness_report.md"

    payload = {
        "questions_path": str(args.questions_path),
        "work_dir": str(args.work_dir),
        "db_path": str(args.db_path),
        "limit": int(args.limit),
        "summary": summary,
        "rows": [asdict(row) for row in rows],
    }
    dump_json(report_json, payload)
    report_md.write_text(render_markdown(summary, rows), encoding="utf-8")

    print(json.dumps({"report_json": str(report_json), "report_md": str(report_md), "summary": summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
