from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


CASE_ID_RE = re.compile(r"\b(CFI|ARB|TCD|CA|ENF|DEC|SCT)\s*[- ]?(\d{3})/(\d{4})\b", re.I)
LAW_ID_RE = re.compile(r"\b(?:DIFC\s+)?Law\s+No\.?\s*\(?(\d+)\)?\s+of\s+(\d{4})\b", re.I)
LAW_ID_VARIANT_RE = re.compile(
    r"\bLaw\s+of\s+the\s+Dubai\s+International\s+Financial\s+Centre\s+No\.?\s*\(?(\d+)\)?\s+of\s+(\d{4})\b",
    re.I,
)
ARTICLE_RE = re.compile(r"Article\s+\d+(?:\([^)]+\))*", re.I)
TITLE_STOPWORDS = {
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
    "is",
    "was",
    "were",
    "does",
    "did",
    "how",
    "all",
    "any",
    "same",
    "their",
    "that",
    "this",
    "these",
    "those",
    "case",
    "law",
    "laws",
}

PREFIX_EXPANSIONS = {
    "administers": "administer*",
    "administered": "administer*",
    "administering": "administer*",
    "administer": "administer*",
    "appointed": "appoint*",
    "appoints": "appoint*",
    "appointing": "appoint*",
    "publishes": "publish*",
    "published": "publish*",
    "publishing": "publish*",
    "publish": "publish*",
    "enacted": "enact*",
    "enacting": "enact*",
    "enacts": "enact*",
    "enact": "enact*",
    "amended": "amend*",
    "amending": "amend*",
    "amends": "amend*",
    "amend": "amend*",
    "retention": "retain*",
    "retained": "retain*",
    "retaining": "retain*",
    "preserved": "preserv*",
    "preserving": "preserv*",
    "preserve": "preserv*",
    "responsible": "responsib*",
    "responsibility": "responsib*",
    "outcome": "outcom*",
    "result": "result*",
}


@dataclass(slots=True)
class DocumentHit:
    doc_id: str
    kind: str
    title: str
    score: float
    page_count: int
    case_id: str = ""
    law_id: str = ""
    snippet: str = ""


@dataclass(slots=True)
class PageHit:
    doc_id: str
    page_number: int
    title: str
    kind: str
    page_role: str
    score: float
    snippet: str
    tags: list[str]


@dataclass(slots=True)
class EntityHit:
    entity_kind: str
    surface: str
    canonical: str
    doc_id: str
    page_number: int
    score: float
    tags: list[str]


@dataclass(slots=True)
class ArticleHit:
    article_ref: str
    heading: str
    section_kind: str
    doc_id: str
    page_number: int
    score: float
    snippet: str


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalized_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return normalize_space(str(value)).lower()


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", normalized_text(text))


def canonical_case_id(prefix: str, number: str, year: str) -> str:
    return f"{prefix.upper()} {int(number):03d}/{year}"


def canonical_law_id(number: str, year: str) -> str:
    return f"Law No. {int(number)} of {year}"


def extract_case_ids(text: str) -> list[str]:
    return sorted({canonical_case_id(*match.groups()) for match in CASE_ID_RE.finditer(text or "")})


def extract_law_ids(text: str) -> list[str]:
    text = text or ""
    return sorted(
        {
            canonical_law_id(*match.groups())
            for pattern in (LAW_ID_RE, LAW_ID_VARIANT_RE)
            for match in pattern.finditer(text)
        }
    )


def extract_article_refs(text: str) -> list[str]:
    return sorted({normalize_space(match.group(0)) for match in ARTICLE_RE.finditer(text or "") if normalize_space(match.group(0))})


def safe_json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _sqlite_list(values: Sequence[str]) -> str:
    return ",".join("?" for _ in values)


def _page_tags(page_text: str, page_number: int, page_count: int) -> list[str]:
    text = normalized_text(page_text)
    tags: list[str] = []
    if page_number == 1:
        tags.append("first_page")
        tags.append("title_page")
    if page_number == 2:
        tags.append("second_page")
    if page_number == page_count:
        tags.append("last_page")
    if "contents" in text:
        tags.append("contents")
    if "it is hereby ordered that" in text or "order of the court" in text or "order" in text and "judgment" in text:
        tags.append("order_section")
    if "conclusion" in text:
        tags.append("conclusion_section")
    if "before h.e." in text or "before" in text and "justice" in text:
        tags.append("judge_block")
    if "between" in text and "and" in text:
        tags.append("party_block")
    if "administered by" in text:
        tags.append("administration_clause")
    if "made by" in text:
        tags.append("made_by_clause")
    if "published in" in text:
        tags.append("publication_line")
    if "enacted on" in text or "enactment notice" in text:
        tags.append("enactment_clause")
    if "schedule" in text:
        tags.append("schedule")
    article_refs = extract_article_refs(page_text)
    if article_refs:
        tags.append("article_section")
    if any(marker in text for marker in ("claimant", "defendant", "appellant", "respondent", "applicant")):
        tags.append("party_names")
    if any(marker in text for marker in ("claim no", "claim number", "case no", "case number")):
        tags.append("case_id_line")
    return sorted(dict.fromkeys(tags))


def _extract_judges(page_text: str) -> list[str]:
    text = page_text or ""
    judges: list[str] = []
    patterns = [
        re.compile(
            r"(?:H\.E\.\s+)?(?:Deputy Chief Justice|Chief Justice|Justice|SCT Judge)\s+"
            r"((?:Sir\s+)?(?:[A-Z][A-Za-z'\-]+|Al|al|Le|le|De|de|Van|van)"
            r"(?:\s+(?:[A-Z][A-Za-z'\-]+|Al|al|Le|le|De|de|Van|van|KC)){0,4})",
            re.I,
        ),
        re.compile(r"before\s+(?:H\.E\.\s+)?((?:Chief Justice|Justice)\s+[^,\n\)]+)", re.I),
    ]
    for pattern in patterns:
        for match in pattern.finditer(text):
            candidate = normalize_space(match.group(1))
            candidate = re.sub(r"\b(the court|the hearing|the appeal)\b.*$", "", candidate, flags=re.I)
            candidate = candidate.replace("  ", " ")
            if len(candidate) >= 4:
                judges.append(candidate)
    return sorted(dict.fromkeys(judges))


def _extract_parties(page_text: str) -> list[str]:
    lines = [normalize_space(line) for line in (page_text or "").splitlines() if normalize_space(line)]
    candidates: list[str] = []
    in_between = False
    for line in lines:
        norm = normalized_text(line)
        if norm == "between":
            in_between = True
            continue
        if in_between and norm in {"and", "versus"}:
            continue
        if in_between and any(marker in norm for marker in ("order of the court", "judgment of", "claim no", "before ", "it is hereby ordered")):
            in_between = False
        if in_between and line:
            if len(line.split()) <= 10 and any(ch.isalpha() for ch in line):
                candidates.append(line)
        if any(marker in norm for marker in ("claimant", "defendant", "appellant", "respondent", "applicant")):
            if len(line.split()) <= 10:
                candidates.append(line)
    cleaned: list[str] = []
    for item in candidates:
        item = re.sub(r"\s+\(\s*the\s+'.*?'\s*\)$", "", item, flags=re.I)
        item = re.sub(r"\s+\(.*?\)$", "", item)
        item = normalize_space(item)
        if item and len(item) >= 2:
            cleaned.append(item)
    return sorted(dict.fromkeys(cleaned))


def _extract_anchor_rows(doc_kind: str, doc_title: str, first_page_text: str, page_texts: dict[int, str]) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    page_one = first_page_text or page_texts.get(1, "")
    case_ids = extract_case_ids(doc_title + "\n" + page_one)
    law_ids = extract_law_ids(doc_title + "\n" + page_one)
    article_refs = []
    for page_number, text in page_texts.items():
        refs = extract_article_refs(text)
        if refs:
            article_refs.extend(refs)
    article_refs = sorted(dict.fromkeys(article_refs))

    for case_id in case_ids:
        anchors.append({"kind": "case_id", "value": case_id, "surface": case_id, "page_number": 1, "confidence": 0.99})
    for law_id in law_ids:
        anchors.append({"kind": "law_id", "value": law_id, "surface": law_id, "page_number": 1, "confidence": 0.99})
    for article_ref in article_refs:
        for page_number, text in page_texts.items():
            if article_ref.lower() in normalized_text(text):
                anchors.append({"kind": "article_ref", "value": article_ref, "surface": article_ref, "page_number": page_number, "confidence": 0.95})
                break

    parties = _extract_parties(page_one)
    for party in parties:
        anchors.append({"kind": "party", "value": party, "surface": party, "page_number": 1, "confidence": 0.75})
    judges = _extract_judges(page_one)
    for judge in judges:
        anchors.append({"kind": "judge", "value": judge, "surface": judge, "page_number": 1, "confidence": 0.75})
    if doc_kind in {"law", "regulation"}:
        for label, marker in (
            ("administration", "administered by"),
            ("publication", "published in"),
            ("enactment", "enacted on"),
        ):
            for page_number, text in page_texts.items():
                if marker in normalized_text(text):
                    anchors.append({"kind": label, "value": marker, "surface": marker, "page_number": page_number, "confidence": 0.9})
                    break
    return anchors


def _guess_primary_document_id(title: str, first_page_text: str, page_texts: dict[int, str]) -> tuple[str, str]:
    case_ids = extract_case_ids(title + "\n" + first_page_text)
    law_ids = extract_law_ids(title + "\n" + first_page_text)
    if case_ids:
        return "case", case_ids[0]
    if law_ids:
        return "law", law_ids[0]
    for text in page_texts.values():
        case_ids = extract_case_ids(text)
        if case_ids:
            return "case", case_ids[0]
        law_ids = extract_law_ids(text)
        if law_ids:
            return "law", law_ids[0]
    return "unknown", ""


def _document_alias_blob(document: dict[str, Any]) -> str:
    aliases = [document.get("title", ""), *(document.get("aliases") or []), *(document.get("canonical_ids") or [])]
    return " | ".join(normalize_space(str(alias)) for alias in aliases if normalize_space(str(alias)))


def _infer_edition(first_page_text: str) -> str:
    text = normalize_space(first_page_text)
    match = re.search(r"Consolidated Version(?: No\.?\s*([^\n]+))?", text, re.I)
    if match:
        return normalize_space(match.group(1) or "consolidated")
    return ""


def _build_fts_query(text: str) -> str:
    words = [token for token in tokenize_words(text) if len(token) > 2 and token not in TITLE_STOPWORDS]
    if not words:
        return ""
    groups: list[str] = []
    case_ids = extract_case_ids(text)
    law_ids = extract_law_ids(text)
    article_refs = extract_article_refs(text)
    if case_ids:
        for case_id in case_ids:
            groups.append(" ".join(tokenize_words(case_id)))
    if law_ids:
        for law_id in law_ids:
            groups.append(" ".join(tokenize_words(law_id)))
    if article_refs:
        for article_ref in article_refs:
            groups.append(" ".join(tokenize_words(article_ref)))
    if words:
        normalized_terms: list[str] = []
        for token in words[:10]:
            normalized_terms.append(PREFIX_EXPANSIONS.get(token, token))
        groups.append(" ".join(normalized_terms))
    groups = [group for group in groups if group]
    if not groups:
        return ""
    if len(groups) == 1:
        return groups[0]
    return " OR ".join(f"({group})" for group in groups)


class LegalRuntimeIndexBuilder:
    def __init__(
        self,
        work_dir: Path,
        db_path: Path,
        source_variant: str = "chunked_section_aware",
    ) -> None:
        self.work_dir = Path(work_dir)
        self.db_path = Path(db_path)
        self.source_variant = source_variant

    def build(self) -> "LegalRuntimeIndex":
        docs_catalog = safe_json_load(self.work_dir / "doc_catalog.json")
        documents = docs_catalog.get("documents", [])
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.db_path.exists():
            self.db_path.unlink()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA foreign_keys=ON")
            self._create_schema(conn)
            self._populate(conn, documents)
            conn.commit()
        return LegalRuntimeIndex(self.db_path)

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE index_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                case_id TEXT NOT NULL DEFAULT '',
                law_id TEXT NOT NULL DEFAULT '',
                edition TEXT NOT NULL DEFAULT '',
                aliases TEXT NOT NULL DEFAULT '',
                source_variant TEXT NOT NULL,
                source_path TEXT NOT NULL,
                first_page_text TEXT NOT NULL DEFAULT '',
                is_consolidated INTEGER NOT NULL DEFAULT 0,
                is_enactment_notice INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE pages (
                doc_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                page_role TEXT NOT NULL DEFAULT '',
                page_text TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '',
                article_refs TEXT NOT NULL DEFAULT '',
                case_ids TEXT NOT NULL DEFAULT '',
                law_ids TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (doc_id, page_number),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            );

            CREATE TABLE anchors (
                anchor_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                kind TEXT NOT NULL,
                value TEXT NOT NULL,
                surface TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            );

            CREATE TABLE article_refs (
                doc_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                article_ref TEXT NOT NULL,
                heading TEXT NOT NULL DEFAULT '',
                section_kind TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (doc_id, page_number, article_ref),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            );

            CREATE TABLE page_tags (
                doc_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (doc_id, page_number, tag),
                FOREIGN KEY (doc_id, page_number) REFERENCES pages(doc_id, page_number) ON DELETE CASCADE
            );

            CREATE VIRTUAL TABLE document_fts USING fts5(
                doc_id UNINDEXED,
                kind UNINDEXED,
                title,
                aliases,
                first_page_text,
                tokenize = 'unicode61'
            );

            CREATE VIRTUAL TABLE page_fts USING fts5(
                doc_id UNINDEXED,
                page_number UNINDEXED,
                doc_kind UNINDEXED,
                title,
                page_role,
                tags,
                content,
                tokenize = 'unicode61'
            );

            CREATE VIRTUAL TABLE anchor_fts USING fts5(
                doc_id UNINDEXED,
                page_number UNINDEXED,
                kind UNINDEXED,
                value,
                surface,
                tags,
                tokenize = 'unicode61'
            );

            CREATE VIRTUAL TABLE article_fts USING fts5(
                doc_id UNINDEXED,
                page_number UNINDEXED,
                article_ref,
                heading,
                section_kind,
                content,
                tokenize = 'unicode61'
            );
            """
        )

    def _load_document_payload(self, sha: str) -> dict[str, Any]:
        variant_dir = self.work_dir / "docling" / self.source_variant
        candidate_paths = [
            variant_dir / f"{sha}.json",
            self.work_dir / "docling" / "chunked_section_aware" / f"{sha}.json",
            self.work_dir / "docling" / "chunked" / f"{sha}.json",
            self.work_dir / "docling" / "chunked_contextual" / f"{sha}.json",
            self.work_dir / "docling" / "chunked_atomic_facts" / f"{sha}.json",
            self.work_dir / "docling" / "merged" / f"{sha}.json",
            self.work_dir / "docling" / "parsed" / f"{sha}.json",
        ]
        for path in candidate_paths:
            if path.exists():
                return safe_json_load(path)
        raise FileNotFoundError(f"Could not locate document payload for {sha}")

    def _populate(self, conn: sqlite3.Connection, documents: Sequence[dict[str, Any]]) -> None:
        index_meta = {
            "work_dir": str(self.work_dir),
            "source_variant": self.source_variant,
            "document_count": str(len(documents)),
            "db_sha256": hashlib.sha256(str(self.db_path).encode("utf-8")).hexdigest(),
        }
        conn.executemany("INSERT INTO index_meta(key, value) VALUES (?, ?)", index_meta.items())

        for document in documents:
            sha = str(document.get("sha") or document.get("sha1_name") or "").strip()
            if not sha:
                continue
            payload = self._load_document_payload(sha)
            content = payload.get("content", {})
            pages = content.get("pages", []) or []
            page_texts: dict[int, str] = {
                int(page.get("page")): str(page.get("text") or "")
                for page in pages
                if page.get("page") is not None
            }
            first_page_text = page_texts.get(1, "")
            doc_kind = str(document.get("kind") or "unknown")
            title = normalize_space(str(document.get("title") or ""))
            page_count = int(document.get("page_count") or len(page_texts) or len(pages) or 0)
            case_kind, primary_id = _guess_primary_document_id(title, first_page_text, page_texts)
            case_id = primary_id if case_kind == "case" else ""
            law_id = primary_id if case_kind == "law" else ""
            aliases_blob = _document_alias_blob(document)
            edition = _infer_edition(first_page_text)
            is_consolidated = 1 if "consolidated version" in normalized_text(first_page_text) else 0
            is_enactment_notice = 1 if "enactment notice" in normalized_text(first_page_text) else 0

            conn.execute(
                """
                INSERT INTO documents(
                    doc_id, kind, title, page_count, case_id, law_id, edition, aliases,
                    source_variant, source_path, first_page_text, is_consolidated, is_enactment_notice
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sha,
                    doc_kind,
                    title,
                    page_count,
                    case_id,
                    law_id,
                    edition,
                    aliases_blob,
                    self.source_variant,
                    f"docling/{self.source_variant}/{sha}.json",
                    first_page_text,
                    is_consolidated,
                    is_enactment_notice,
                ),
            )

            conn.execute(
                "INSERT INTO document_fts(doc_id, kind, title, aliases, first_page_text) VALUES (?, ?, ?, ?, ?)",
                (sha, doc_kind, title, aliases_blob, first_page_text),
            )

            for page in pages:
                page_number = int(page.get("page"))
                page_text = normalize_space(str(page.get("text") or ""))
                tags = _page_tags(page_text, page_number, page_count)
                article_refs = extract_article_refs(page_text)
                case_ids = extract_case_ids(page_text or title)
                law_ids = extract_law_ids(page_text or title)
                role = tags[0] if tags else ("title_page" if page_number == 1 else "")

                conn.execute(
                    """
                    INSERT INTO pages(
                        doc_id, page_number, page_role, page_text, tags, article_refs, case_ids, law_ids
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sha,
                        page_number,
                        role,
                        page_text,
                        " ".join(tags),
                        " | ".join(article_refs),
                        " | ".join(case_ids),
                        " | ".join(law_ids),
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO page_fts(doc_id, page_number, doc_kind, title, page_role, tags, content)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (sha, page_number, doc_kind, title, role, " ".join(tags), page_text),
                )

                for tag in tags:
                    conn.execute(
                        "INSERT INTO page_tags(doc_id, page_number, tag) VALUES (?, ?, ?)",
                        (sha, page_number, tag),
                    )

                for article_ref in article_refs:
                    heading = article_ref
                    section_kind = "article"
                    if page_number == 1 and "contents" in normalized_text(page_text):
                        section_kind = "contents"
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO article_refs(
                            doc_id, page_number, article_ref, heading, section_kind, tags
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            sha,
                            page_number,
                            article_ref,
                            heading,
                            section_kind,
                            " ".join(tags),
                        ),
                    )
                    conn.execute(
                        """
                        INSERT INTO article_fts(doc_id, page_number, article_ref, heading, section_kind, content)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (sha, page_number, article_ref, heading, section_kind, page_text),
                    )

                anchors = _extract_anchor_rows(doc_kind, title, first_page_text, page_texts)
                for anchor in anchors:
                    if int(anchor["page_number"]) != page_number:
                        continue
                    conn.execute(
                        """
                        INSERT INTO anchors(doc_id, page_number, kind, value, surface, confidence)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            sha,
                            page_number,
                            anchor["kind"],
                            anchor["value"],
                            anchor["surface"],
                            float(anchor["confidence"]),
                        ),
                    )
                    conn.execute(
                        """
                        INSERT INTO anchor_fts(doc_id, page_number, kind, value, surface, tags)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (sha, page_number, anchor["kind"], anchor["value"], anchor["surface"], " ".join(tags)),
                    )


class LegalRuntimeIndex:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def healthcheck(self) -> dict[str, Any]:
        with self._connect() as conn:
            doc_count = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            page_count = conn.execute("SELECT COUNT(*) AS n FROM pages").fetchone()["n"]
            anchor_count = conn.execute("SELECT COUNT(*) AS n FROM anchors").fetchone()["n"]
            article_count = conn.execute("SELECT COUNT(*) AS n FROM article_refs").fetchone()["n"]
        return {
            "db_path": str(self.db_path),
            "documents": int(doc_count),
            "pages": int(page_count),
            "anchors": int(anchor_count),
            "articles": int(article_count),
        }

    def get_document(self, doc_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
            return dict(row) if row else None

    def get_page(self, doc_id: str, page_number: int) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM pages WHERE doc_id = ? AND page_number = ?",
                (doc_id, int(page_number)),
            ).fetchone()
            return dict(row) if row else None

    def _filter_clause(self, filters: dict[str, Any]) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        args: list[Any] = []

        doc_ids = filters.get("doc_ids") or []
        if doc_ids:
            clauses.append(f"doc_id IN ({_sqlite_list(doc_ids)})")
            args.extend(doc_ids)

        kinds = filters.get("kinds") or []
        if kinds:
            kind_column = str(filters.get("kind_column") or "kind")
            clauses.append(f"{kind_column} IN ({_sqlite_list(kinds)})")
            args.extend(kinds)

        page_roles = filters.get("page_roles") or []
        if page_roles:
            clauses.append(f"page_role IN ({_sqlite_list(page_roles)})")
            args.extend(page_roles)

        tags = filters.get("tags") or []
        if tags:
            for tag in tags:
                clauses.append("tags LIKE ?")
                args.append(f"%{tag}%")

        page_numbers = filters.get("page_numbers") or []
        if page_numbers:
            clauses.append(f"page_number IN ({_sqlite_list([str(p) for p in page_numbers])})")
            args.extend(int(p) for p in page_numbers)

        if clauses:
            return " AND " + " AND ".join(clauses), args
        return "", args

    def _fts_query(self, text: str) -> str:
        query = _build_fts_query(text)
        if query:
            return query
        return "\"\""

    def search_pages(
        self,
        query: str,
        limit: int = 10,
        *,
        doc_ids: Optional[Sequence[str]] = None,
        kinds: Optional[Sequence[str]] = None,
        page_roles: Optional[Sequence[str]] = None,
        tags: Optional[Sequence[str]] = None,
        page_numbers: Optional[Sequence[int]] = None,
    ) -> list[PageHit]:
        query = normalize_space(query)
        if not query:
            return []
        fts_query = self._fts_query(query)
        filters = {
            "doc_ids": list(doc_ids or []),
            "kinds": list(kinds or []),
            "kind_column": "doc_kind",
            "page_roles": list(page_roles or []),
            "tags": list(tags or []),
            "page_numbers": list(page_numbers or []),
        }
        where_clause, args = self._filter_clause(filters)
        sql = f"""
            SELECT
                doc_id, page_number, doc_kind, title, page_role, tags, content,
                bm25(page_fts) AS score
            FROM page_fts
            WHERE page_fts MATCH ? {where_clause}
            ORDER BY score ASC
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, [fts_query, *args, int(limit)]).fetchall()
        hits: list[PageHit] = []
        for row in rows:
            hits.append(
                PageHit(
                    doc_id=row["doc_id"],
                    page_number=int(row["page_number"]),
                    title=row["title"],
                    kind=row["doc_kind"],
                    page_role=row["page_role"] or "",
                    score=float(row["score"]),
                    snippet=normalize_space(str(row["content"])[:400]),
                    tags=[tag for tag in normalize_space(row["tags"]).split(" ") if tag],
                )
            )
        return hits

    def search_documents(
        self,
        query: str,
        limit: int = 10,
        *,
        kinds: Optional[Sequence[str]] = None,
    ) -> list[DocumentHit]:
        query = normalize_space(query)
        if not query:
            return []
        fts_query = self._fts_query(query)
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
        with self._connect() as conn:
            rows = conn.execute(sql, [fts_query, *args, int(limit)]).fetchall()
            if rows:
                hits: list[DocumentHit] = []
                for row in rows:
                    hits.append(
                        DocumentHit(
                            doc_id=row["doc_id"],
                            kind=row["kind"],
                            title=row["title"],
                            score=float(row["score"]),
                            page_count=int(row["page_count"]),
                            case_id=row["case_id"] or "",
                            law_id=row["law_id"] or "",
                            snippet=normalize_space(str(row["first_page_text"])[:400]),
                        )
                    )
                return hits

            page_hits = self.search_pages(query, limit=max(limit * 3, 10), kinds=kinds)
            aggregated: dict[str, dict[str, Any]] = {}
            for hit in page_hits:
                bucket = aggregated.setdefault(
                    hit.doc_id,
                    {
                        "doc_id": hit.doc_id,
                        "best_score": hit.score,
                        "count": 0,
                        "snippet": hit.snippet,
                    },
                )
                bucket["count"] += 1
                bucket["best_score"] = min(bucket["best_score"], hit.score)
                if not bucket["snippet"]:
                    bucket["snippet"] = hit.snippet
            if not aggregated:
                return []

            doc_rows = conn.execute(
                f"SELECT doc_id, kind, title, page_count, case_id, law_id, first_page_text FROM documents WHERE doc_id IN ({_sqlite_list(list(aggregated))})",
                list(aggregated),
            ).fetchall()
            docs_by_id = {row["doc_id"]: row for row in doc_rows}
            results: list[DocumentHit] = []
            for doc_id, bucket in aggregated.items():
                row = docs_by_id.get(doc_id)
                if row is None:
                    continue
                score = float(bucket["best_score"]) - 0.05 * float(bucket["count"] - 1)
                if kinds and row["kind"] not in kinds:
                    continue
                results.append(
                    DocumentHit(
                        doc_id=row["doc_id"],
                        kind=row["kind"],
                        title=row["title"],
                        score=score,
                        page_count=int(row["page_count"]),
                        case_id=row["case_id"] or "",
                        law_id=row["law_id"] or "",
                        snippet=normalize_space(str(bucket["snippet"])[:400] or str(row["first_page_text"])[:400]),
                    )
                )
            results.sort(key=lambda item: item.score)
            return results[:limit]
        return []

    def search_entities(
        self,
        query: str,
        limit: int = 10,
        *,
        kinds: Optional[Sequence[str]] = None,
        doc_ids: Optional[Sequence[str]] = None,
    ) -> list[EntityHit]:
        query = normalize_space(query)
        if not query:
            return []
        fts_query = self._fts_query(query)
        clauses: list[str] = []
        args: list[Any] = []
        if kinds:
            clauses.append(f"kind IN ({_sqlite_list(kinds)})")
            args.extend(kinds)
        if doc_ids:
            clauses.append(f"doc_id IN ({_sqlite_list(doc_ids)})")
            args.extend(doc_ids)
        where_clause = f" AND {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT doc_id, page_number, kind, value, surface, tags, bm25(anchor_fts) AS score
            FROM anchor_fts
            WHERE anchor_fts MATCH ? {where_clause}
            ORDER BY score ASC
            LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, [fts_query, *args, int(limit)]).fetchall()
        hits: list[EntityHit] = []
        for row in rows:
            hits.append(
                EntityHit(
                    entity_kind=row["kind"],
                    surface=row["surface"],
                    canonical=row["value"],
                    doc_id=row["doc_id"],
                    page_number=int(row["page_number"]),
                    score=float(row["score"]),
                    tags=[tag for tag in normalize_space(row["tags"]).split(" ") if tag],
                )
            )
        return hits

    def search_articles(
        self,
        query: str,
        limit: int = 10,
        *,
        doc_ids: Optional[Sequence[str]] = None,
    ) -> list[ArticleHit]:
        query = normalize_space(query)
        if not query:
            return []
        fts_query = self._fts_query(query)
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
        with self._connect() as conn:
            rows = conn.execute(sql, [fts_query, *args, int(limit)]).fetchall()
        hits: list[ArticleHit] = []
        for row in rows:
            hits.append(
                ArticleHit(
                    article_ref=row["article_ref"],
                    heading=row["heading"],
                    section_kind=row["section_kind"],
                    doc_id=row["doc_id"],
                    page_number=int(row["page_number"]),
                    score=float(row["score"]),
                    snippet=normalize_space(str(row["content"])[:400]),
                )
            )
        return hits

    def route_query(self, query: str) -> dict[str, Any]:
        text = normalize_space(query)
        lower = text.lower()
        case_ids = extract_case_ids(text)
        law_ids = extract_law_ids(text)
        article_refs = extract_article_refs(text)
        if article_refs or "article" in lower or "section" in lower:
            return {
                "mode": "article",
                "page_roles": ["article_section", "title_page", "second_page"],
                "kinds": ["law", "regulation"],
                "anchors": {"case_ids": case_ids, "law_ids": law_ids, "article_refs": article_refs},
            }
        if any(marker in lower for marker in ("judge", "judges")):
            return {
                "mode": "entity",
                "page_roles": ["judge_block", "case_id_line", "title_page", "first_page"],
                "kinds": ["case"],
                "entity_kinds": ["judge"],
                "anchors": {"case_ids": case_ids, "law_ids": law_ids, "article_refs": article_refs},
            }
        if any(marker in lower for marker in ("party", "parties", "claimant", "defendant", "appellant", "respondent")):
            return {
                "mode": "entity",
                "page_roles": ["party_block", "case_id_line", "title_page", "first_page"],
                "kinds": ["case"],
                "entity_kinds": ["party"],
                "anchors": {"case_ids": case_ids, "law_ids": law_ids, "article_refs": article_refs},
            }
        if any(marker in lower for marker in ("administered by", "made by", "published", "enacted", "effective date", "law", "regulation")):
            return {
                "mode": "document",
                "page_roles": ["title_page", "publication_line", "administration_clause", "enactment_clause"],
                "kinds": ["law", "regulation"],
                "anchors": {"case_ids": case_ids, "law_ids": law_ids, "article_refs": article_refs},
            }
        if any(marker in lower for marker in ("result", "outcome", "ordered", "dismissed", "granted", "refused", "appeal", "costs")):
            return {
                "mode": "page",
                "page_roles": ["order_section", "conclusion_section", "last_page", "case_id_line", "judge_block", "first_page"],
                "kinds": ["case"],
                "anchors": {"case_ids": case_ids, "law_ids": law_ids, "article_refs": article_refs},
            }
        return {
            "mode": "page",
            "page_roles": ["case_id_line", "first_page", "title_page"],
            "kinds": ["case", "law", "regulation"],
            "anchors": {"case_ids": case_ids, "law_ids": law_ids, "article_refs": article_refs},
        }

    def query(self, query: str, limit: int = 5) -> dict[str, list[Any]]:
        route = self.route_query(query)
        result: dict[str, list[Any]] = {"documents": [], "pages": [], "entities": [], "articles": []}
        if route["mode"] in {"document", "page", "article", "entity"}:
            result["pages"] = self.search_pages(
                query,
                limit=limit,
                kinds=route.get("kinds"),
                page_roles=route.get("page_roles"),
            )
        if route["mode"] in {"document", "article"}:
            result["articles"] = self.search_articles(query, limit=limit, doc_ids=None)
        if route["mode"] in {"document", "entity"}:
            entity_kinds = route.get("entity_kinds")
            result["entities"] = self.search_entities(query, limit=limit, kinds=entity_kinds)
        result["documents"] = self.search_documents(query, limit=limit, kinds=route.get("kinds"))
        return result


def build_index(
    work_dir: str | Path,
    db_path: str | Path,
    source_variant: str = "chunked_section_aware",
) -> LegalRuntimeIndex:
    builder = LegalRuntimeIndexBuilder(Path(work_dir), Path(db_path), source_variant=source_variant)
    return builder.build()


def open_index(db_path: str | Path) -> LegalRuntimeIndex:
    return LegalRuntimeIndex(Path(db_path))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build or query the legal runtime index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the SQLite/FTS index from a workdir.")
    build_parser.add_argument("--work-dir", type=Path, required=True)
    build_parser.add_argument("--db-path", type=Path, required=True)
    build_parser.add_argument("--source-variant", type=str, default="chunked_section_aware")

    query_parser = subparsers.add_parser("query", help="Query an existing index.")
    query_parser.add_argument("--db-path", type=Path, required=True)
    query_parser.add_argument("--text", type=str, required=True)
    query_parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "build":
        index = build_index(args.work_dir, args.db_path, source_variant=args.source_variant)
        print(json.dumps(index.healthcheck(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "query":
        index = open_index(args.db_path)
        payload = index.query(args.text, limit=args.limit)
        serializable = {
            key: [dataclasses.asdict(item) for item in value]
            for key, value in payload.items()
        }
        print(json.dumps(serializable, indent=2, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
