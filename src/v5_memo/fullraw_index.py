"""Persistent FTS index for the full raw corpus.

The cold scanner proves the raw archive is reachable, but it returns early hits
in file order. This module builds a durable SQLite FTS5 index so memo retrieval
can rank by relevance instead of archive order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from concurrent.futures import (
    TimeoutError as FuturesTimeoutError,
)
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from v5_memo.fullraw_service import RawFile, iter_raw_file_hits, load_or_build_manifest

_WORD = re.compile(r"[A-Za-z0-9]+")
_STOP = {
    "about", "above", "abstract", "after", "also", "among", "and", "are", "because", "been", "both",
    "but", "can", "could", "different", "does", "during", "for", "from", "had", "has", "have", "high",
    "important", "international", "into", "its",
    "may", "most", "no", "not", "off", "or", "our", "out", "over", "per", "such", "than", "that", "the",
    "their", "these", "this", "those", "through", "was", "were", "with", "within",
    "without", "would", "all", "any", "between", "each", "other",
    "should", "there", "using", "which", "while", "who", "whose", "will", "you", "your",
    "of", "in", "by", "to", "is", "be", "at", "on", "as", "an",
    "analysis", "associated", "based", "better", "compared", "data", "effect",
    "effects", "first", "form", "found", "here", "it", "method", "methods",
    "journal", "model", "models", "more", "name", "new", "none", "one", "only", "pages", "paper",
    "provide", "provided", "research", "related", "result", "results", "show", "showed", "shown", "significant",
    "significantly", "studies", "study", "two", "use", "used", "we", "work",
    "volume", "dan", "de", "del", "di", "es", "findings", "general", "however", "ini", "many", "merupakan",
    "para", "process", "que", "role", "specific", "therefore", "un", "understanding", "well", "yang",
    "adalah", "analisis", "dengan", "dilakukan", "hasil", "including", "made",
    "el", "en", "la", "los", "metode", "pada", "penelitian",
}
_BACKEND = "v5-fullraw-indexed-fts5"
_DEFAULT_TERM_MAP = (
    ("management", ("management", "manager", "managers", "managerial")),
    ("forecast", ("forecast", "forecasts", "forecasting", "guidance", "estimate", "estimates")),
    ("disclosure", ("disclosure", "disclosures", "disclose", "discloses", "disclosed")),
    ("earnings", ("earnings", "income", "profit", "profits")),
    ("analyst", ("analyst", "analysts", "analysis")),
)


@dataclass(frozen=True, slots=True)
class IndexStats:
    papers_indexed: int
    files_indexed: int
    files_total: int
    bytes_used: int

    @property
    def complete(self) -> bool:
        return self.files_total > 0 and self.files_indexed >= self.files_total


@dataclass(frozen=True, slots=True)
class IndexBuildResult:
    files_attempted: int
    files_completed: int
    files_failed: int
    papers_inserted: int
    stopped_for_budget: bool
    elapsed_seconds: float
    file_errors: str = ""


@dataclass(frozen=True, slots=True)
class ShardBuildResult:
    shard_id: int
    index_path: str
    files_total: int
    files_attempted: int
    files_completed: int
    files_failed: int
    papers_inserted: int
    stopped_for_budget: bool
    elapsed_seconds: float
    bytes_used: int
    year_min: int | None = None
    year_max: int | None = None
    cited_by_min: int = 0
    cited_by_max: int = 0
    cited_by_avg: float = 0.0
    topic_terms: tuple[str, ...] = ()
    file_errors: str = ""
    error: str = ""


@dataclass(frozen=True, slots=True)
class ShardBuildJob:
    shard_id: int
    index_path: str
    files: list[RawFile]
    rclone_bin: str
    time_budget_seconds: float | None
    commit_interval: int
    min_free_bytes: int


@dataclass(frozen=True, slots=True)
class ShardBatchResult:
    batch_id: int
    batch_dir: str
    remote_dir: str
    files_total: int
    files_completed: int
    files_failed: int
    papers_inserted: int
    bytes_used: int
    uploaded: bool
    deleted_local: bool
    skipped: bool
    elapsed_seconds: float
    file_errors: str = ""
    error: str = ""


@dataclass(frozen=True, slots=True)
class ShardCatalogEntry:
    path: Path
    batch_id: int
    shard_id: int
    sources: tuple[str, ...]
    files_completed: int
    papers_inserted: int
    bytes_used: int
    year_min: int | None = None
    year_max: int | None = None
    cited_by_min: int = 0
    cited_by_max: int = 0
    cited_by_avg: float = 0.0
    topic_terms: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SweepCacheEntry:
    created_at: float
    hits: list[dict[str, object]]
    receipt: dict[str, object]


class FullRawFtsIndex:
    """SQLite FTS5 index over normalized raw-corpus records."""

    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = path
        self._read_only = read_only
        if read_only:
            uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
            self._conn = sqlite3.connect(uri, uri=True, timeout=60.0, check_same_thread=False)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), timeout=60.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()

    def close(self) -> None:
        self._conn.close()

    def initialize(self) -> None:
        if self._read_only:
            return
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA temp_store=MEMORY")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS papers (
                  id INTEGER PRIMARY KEY,
                  source_key TEXT NOT NULL UNIQUE,
                  title TEXT NOT NULL,
                  abstract TEXT NOT NULL,
                  doi TEXT,
                  pmid TEXT,
                  pmcid TEXT,
                  openalex_id TEXT,
                  semantic_scholar_id TEXT,
                  year INTEGER,
                  journal TEXT,
                  source TEXT NOT NULL,
                  source_remote TEXT NOT NULL DEFAULT '',
                  url TEXT,
                  cited_by_count INTEGER,
                  raw_score REAL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS paper_fts USING fts5(
                  title,
                  abstract,
                  journal,
                  content='papers',
                  content_rowid='id',
                  tokenize='unicode61 remove_diacritics 2'
                );

                CREATE TABLE IF NOT EXISTS indexed_files (
                  remote TEXT PRIMARY KEY,
                  source TEXT NOT NULL,
                  format TEXT NOT NULL,
                  status TEXT NOT NULL,
                  docs_seen INTEGER NOT NULL DEFAULT 0,
                  docs_indexed INTEGER NOT NULL DEFAULT 0,
                  error TEXT NOT NULL DEFAULT '',
                  started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  finished_at TEXT
                );

                CREATE TABLE IF NOT EXISTS index_meta (
                  key TEXT PRIMARY KEY,
                  value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS term_map (
                  term TEXT PRIMARY KEY,
                  expansions_json TEXT NOT NULL,
                  source TEXT NOT NULL,
                  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
                CREATE INDEX IF NOT EXISTS idx_indexed_files_status ON indexed_files(status);
                """
            )
            self._ensure_source_remote_column()
            self._seed_default_term_map()
            self._conn.commit()

    def completed_remotes(self) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT remote FROM indexed_files WHERE status = 'complete'"
            ).fetchall()
        return {str(row["remote"]) for row in rows}

    def stats(self, *, files_total: int = 0) -> IndexStats:
        with self._lock:
            papers_indexed = int(self._get_meta("papers_indexed") or "0")
            files_indexed = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM indexed_files WHERE status = 'complete'"
                ).fetchone()[0]
            )
            bytes_used = self.path.stat().st_size if self.path.exists() else 0
        return IndexStats(
            papers_indexed=papers_indexed,
            files_indexed=files_indexed,
            files_total=files_total,
            bytes_used=bytes_used,
        )

    def profile(self, *, topic_limit: int = 12, sample_limit: int = 200) -> dict[str, object]:
        with self._lock:
            if not self._read_only:
                self.initialize()
            year_min_row = self._conn.execute(
                "SELECT year FROM papers WHERE year IS NOT NULL ORDER BY year ASC LIMIT 1"
            ).fetchone()
            year_max_row = self._conn.execute(
                "SELECT year FROM papers WHERE year IS NOT NULL ORDER BY year DESC LIMIT 1"
            ).fetchone()
            sample_cap = max(1, min(sample_limit, 1000))
            papers_indexed = _int_or_none(self._get_meta("papers_indexed")) or 0
            if papers_indexed <= 0:
                max_id_row = self._conn.execute("SELECT MAX(id) AS max_id FROM papers").fetchone()
                papers_indexed = _int_or_none(max_id_row["max_id"]) or 0 if max_id_row is not None else 0
            topic_rows: list[sqlite3.Row] = []
            seen_ids: set[int] = set()
            starts = _sample_row_starts(papers_indexed)
            rows_per_start = max(1, -(-sample_cap // max(1, len(starts))))
            for start in starts:
                rows = self._conn.execute(
                    """
                    SELECT id, title, abstract, journal, cited_by_count
                    FROM papers
                    WHERE id >= ?
                    ORDER BY id
                    LIMIT ?
                    """,
                    (start, rows_per_start),
                ).fetchall()
                for sample_row in rows:
                    row_id = int(sample_row["id"])
                    if row_id in seen_ids:
                        continue
                    seen_ids.add(row_id)
                    topic_rows.append(sample_row)
                    if len(topic_rows) >= sample_cap:
                        break
                if len(topic_rows) >= sample_cap:
                    break
            if not topic_rows:
                topic_rows = self._conn.execute(
                    """
                    SELECT id, title, abstract, journal, cited_by_count
                    FROM papers
                    ORDER BY id
                    LIMIT ?
                    """,
                    (sample_cap,),
                ).fetchall()
        citation_counts = [_int_or_none(row["cited_by_count"]) or 0 for row in topic_rows]
        return {
            "year_min": _int_or_none(year_min_row["year"]) if year_min_row is not None else None,
            "year_max": _int_or_none(year_max_row["year"]) if year_max_row is not None else None,
            "cited_by_min": min(citation_counts) if citation_counts else 0,
            "cited_by_max": max(citation_counts) if citation_counts else 0,
            "cited_by_avg": round(sum(citation_counts) / len(citation_counts), 3) if citation_counts else 0.0,
            "profile_sample_size": len(topic_rows),
            "topic_terms": _profile_topic_terms(topic_rows, limit=topic_limit),
        }

    def upsert_term_map(
        self,
        term: str,
        expansions: tuple[str, ...],
        *,
        source: str = "manual",
    ) -> None:
        """Persist a search-time expansion map for one canonical term."""
        clean_term = _first_fts_term(term)
        clean_expansions = _unique_terms((*expansions, clean_term))
        if not clean_term or not clean_expansions:
            raise ValueError("term map requires at least one searchable token")
        with self._lock:
            self.initialize()
            self._conn.execute(
                """
                INSERT INTO term_map(term, expansions_json, source, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(term) DO UPDATE SET
                  expansions_json=excluded.expansions_json,
                  source=excluded.source,
                  updated_at=CURRENT_TIMESTAMP
                """,
                (clean_term, json.dumps(clean_expansions), source),
            )
            self._conn.commit()

    def explain_query(self, query: str) -> dict[str, object]:
        """Return the persisted term-map expansion used for an FTS query."""
        with self._lock:
            self.initialize()
            terms = _fts_terms(query)
            groups = self._expanded_term_groups(terms)
        return {
            "query": query,
            "terms": terms,
            "groups": groups,
            "fts_match": _fts_match_query(groups),
        }

    def index_files(
        self,
        files: list[RawFile],
        *,
        rclone_bin: str = "rclone",
        max_files: int | None = None,
        time_budget_seconds: float | None = None,
        commit_interval: int = 1000,
        min_free_bytes: int = 0,
    ) -> IndexBuildResult:
        with self._lock:
            self.initialize()
            started = time.monotonic()
            deadline = started + time_budget_seconds if time_budget_seconds else None
            completed = self.completed_remotes()
            files_attempted = 0
            files_completed = 0
            files_failed = 0
            file_errors: list[str] = []
            papers_inserted = 0
            stopped_for_budget = False

            for raw_file in files:
                if min_free_bytes > 0 and _free_bytes(self.path.parent) < min_free_bytes:
                    stopped_for_budget = True
                    break
                if raw_file.remote in completed:
                    continue
                if max_files is not None and files_attempted >= max_files:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    stopped_for_budget = True
                    break
                files_attempted += 1
                try:
                    result = self._index_one_file(
                        raw_file,
                        rclone_bin=rclone_bin,
                        deadline=deadline,
                        commit_interval=max(1, commit_interval),
                    )
                except Exception as exc:
                    files_failed += 1
                    file_errors.append(f"{raw_file.remote}: {str(exc)[:240]}")
                    continue
                papers_inserted += result["inserted"]
                if result["complete"]:
                    files_completed += 1
                else:
                    stopped_for_budget = True
                    break

            return IndexBuildResult(
                files_attempted=files_attempted,
                files_completed=files_completed,
                files_failed=files_failed,
                papers_inserted=papers_inserted,
                stopped_for_budget=stopped_for_budget,
                elapsed_seconds=round(time.monotonic() - started, 3),
                file_errors="; ".join(file_errors)[:1000],
            )

    def search(
        self,
        query: str,
        *,
        limit: int = 25,
        year_min: int = 1900,
        year_max: int = 2100,
        rank_mode: str = "relevance",
        timeout_seconds: float | None = None,
    ) -> list[dict[str, object]]:
        with self._lock:
            if not self._read_only:
                self.initialize()
            terms = _fts_terms(query)
            if not terms:
                return []
            match_query = _fts_match_query(self._expanded_term_groups(terms))
            order_by = _search_order_by(rank_mode)
            if timeout_seconds is not None and timeout_seconds > 0:
                deadline = time.monotonic() + timeout_seconds
                self._conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            try:
                rows = self._conn.execute(
                    f"""
                    SELECT
                      p.title,
                      p.abstract,
                      p.doi,
                      p.pmid,
                      p.pmcid,
                      p.openalex_id,
                      p.semantic_scholar_id,
                      p.year,
                      p.journal,
                      p.source,
                      p.url,
                      p.cited_by_count,
                      bm25(paper_fts, 8.0, 3.0, 1.0) AS rank
                    FROM paper_fts
                    JOIN papers p ON p.id = paper_fts.rowid
                    WHERE paper_fts MATCH ?
                      AND (p.year IS NULL OR (p.year >= ? AND p.year <= ?))
                    ORDER BY {order_by}
                    LIMIT ?
                    """,
                    (match_query, year_min, year_max, max(1, min(limit, 200))),
                ).fetchall()
            finally:
                if timeout_seconds is not None and timeout_seconds > 0:
                    self._conn.set_progress_handler(None, 0)
        return [_row_to_hit(row) for row in rows]

    def _expanded_term_groups(self, terms: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        groups: list[tuple[str, ...]] = []
        for term in terms:
            groups.append(self._term_expansions(term) or (term,))
        return tuple(groups)

    def _index_one_file(
        self,
        raw_file: RawFile,
        *,
        rclone_bin: str,
        deadline: float | None,
        commit_interval: int,
    ) -> dict[str, int | bool]:
        docs_seen = 0
        inserted_total = 0
        inserted_since_commit = 0
        complete = True
        self._conn.execute("BEGIN")
        try:
            self._mark_file(raw_file, status="running", docs_seen=0, docs_indexed=0)
            for hit in iter_raw_file_hits(raw_file, rclone_bin=rclone_bin):
                docs_seen += 1
                if self._insert_hit(hit, source_remote=raw_file.remote):
                    inserted_total += 1
                    inserted_since_commit += 1
                if docs_seen % commit_interval == 0:
                    self._bump_papers(inserted_since_commit)
                    inserted_since_commit = 0
                    self._conn.commit()
                    if deadline is not None and time.monotonic() >= deadline:
                        complete = False
                        self._conn.execute("BEGIN")
                        break
                    self._conn.execute("BEGIN")
            status = "complete" if complete else "partial"
            self._bump_papers(inserted_since_commit)
            self._mark_file(
                raw_file,
                status=status,
                docs_seen=docs_seen,
                docs_indexed=inserted_total,
            )
            self._conn.commit()
        except Exception as exc:
            self._conn.rollback()
            self._conn.execute("BEGIN")
            self._remove_file_papers(raw_file.remote)
            self._mark_file(
                raw_file,
                status="error",
                docs_seen=docs_seen,
                docs_indexed=inserted_total,
                error=str(exc)[:500],
            )
            self._conn.commit()
            raise
        return {"inserted": inserted_total, "complete": complete}

    def _insert_hit(self, hit: dict[str, object], *, source_remote: str) -> bool:
        title = _clean(hit.get("title"))
        if not title:
            return self._update_hit_abstract(hit)
        abstract = _clean(hit.get("abstract"))
        journal = _clean(hit.get("journal") or hit.get("venue"))
        source_key = _dedupe_key(hit)
        cursor = self._conn.execute(
            """
            INSERT OR IGNORE INTO papers (
              source_key,
              title,
              abstract,
              doi,
              pmid,
              pmcid,
              openalex_id,
              semantic_scholar_id,
              year,
              journal,
              source,
              source_remote,
              url,
              cited_by_count,
              raw_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_key,
                title,
                abstract,
                _clean(hit.get("doi")),
                _clean(hit.get("pmid")),
                _clean(hit.get("pmcid")),
                _clean(hit.get("openalex_id")),
                _clean(hit.get("semantic_scholar_id")),
                _int_or_none(hit.get("year")),
                journal,
                _clean(hit.get("source")) or "unknown",
                source_remote,
                _clean(hit.get("url")),
                _int_or_none(hit.get("cited_by_count")),
                _float_or_none(hit.get("score")),
            ),
        )
        if cursor.rowcount == 0:
            return False
        paper_id = cursor.lastrowid
        if paper_id is None:
            return False
        self._conn.execute(
            "INSERT INTO paper_fts(rowid, title, abstract, journal) VALUES (?, ?, ?, ?)",
            (paper_id, title, abstract, journal),
        )
        return True

    def _update_hit_abstract(self, hit: dict[str, object]) -> bool:
        abstract = _clean(hit.get("abstract"))
        if not abstract:
            return False
        clauses: list[str] = []
        params: list[str] = []
        for column, key in (
            ("doi", "doi"),
            ("pmid", "pmid"),
            ("pmcid", "pmcid"),
            ("semantic_scholar_id", "semantic_scholar_id"),
            ("openalex_id", "openalex_id"),
        ):
            value = _clean(hit.get(key))
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        if not clauses:
            return False
        row = self._conn.execute(
            f"SELECT id, title, abstract, journal FROM papers WHERE {' OR '.join(clauses)} LIMIT 1",
            params,
        ).fetchone()
        if row is None or _clean(row["abstract"]):
            return False
        paper_id = int(row["id"])
        title = _clean(row["title"])
        journal = _clean(row["journal"])
        self._conn.execute("UPDATE papers SET abstract = ? WHERE id = ?", (abstract, paper_id))
        self._conn.execute(
            "INSERT INTO paper_fts(paper_fts, rowid, title, abstract, journal) VALUES ('delete', ?, ?, ?, ?)",
            (paper_id, title, "", journal),
        )
        self._conn.execute(
            "INSERT INTO paper_fts(rowid, title, abstract, journal) VALUES (?, ?, ?, ?)",
            (paper_id, title, abstract, journal),
        )
        return True

    def _mark_file(
        self,
        raw_file: RawFile,
        *,
        status: str,
        docs_seen: int,
        docs_indexed: int,
        error: str = "",
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO indexed_files (
              remote,
              source,
              format,
              status,
              docs_seen,
              docs_indexed,
              error,
              started_at,
              finished_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CASE WHEN ? = 'complete' THEN CURRENT_TIMESTAMP ELSE NULL END)
            ON CONFLICT(remote) DO UPDATE SET
              status=excluded.status,
              docs_seen=excluded.docs_seen,
              docs_indexed=excluded.docs_indexed,
              error=excluded.error,
              finished_at=CASE WHEN excluded.status = 'complete' THEN CURRENT_TIMESTAMP ELSE NULL END
            """,
            (
                raw_file.remote,
                raw_file.source,
                raw_file.format,
                status,
                docs_seen,
                docs_indexed,
                error,
                status,
            ),
        )

    def _bump_papers(self, delta: int) -> None:
        if delta == 0:
            return
        current = int(self._get_meta("papers_indexed") or "0")
        self._set_meta("papers_indexed", str(max(0, current + delta)))

    def _remove_file_papers(self, remote: str) -> None:
        deleted = int(
            self._conn.execute(
                "SELECT COUNT(*) FROM papers WHERE source_remote = ?",
                (remote,),
            ).fetchone()[0]
        )
        if deleted <= 0:
            return
        self._conn.execute(
            "DELETE FROM paper_fts WHERE rowid IN (SELECT id FROM papers WHERE source_remote = ?)",
            (remote,),
        )
        self._conn.execute("DELETE FROM papers WHERE source_remote = ?", (remote,))
        self._bump_papers(-deleted)

    def _ensure_source_remote_column(self) -> None:
        columns = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(papers)").fetchall()
        }
        if "source_remote" not in columns:
            self._conn.execute("ALTER TABLE papers ADD COLUMN source_remote TEXT NOT NULL DEFAULT ''")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_source_remote ON papers(source_remote)")

    def _get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM index_meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return str(row["value"])

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            """
            INSERT INTO index_meta(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (key, value),
        )

    def _seed_default_term_map(self) -> None:
        for term, expansions in _DEFAULT_TERM_MAP:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO term_map(term, expansions_json, source)
                VALUES (?, ?, 'default')
                """,
                (term, json.dumps(_unique_terms(expansions))),
            )

    def _term_expansions(self, term: str) -> tuple[str, ...]:
        row = self._conn.execute(
            "SELECT expansions_json FROM term_map WHERE term = ?",
            (term,),
        ).fetchone()
        if row is None:
            return ()
        try:
            data = json.loads(str(row["expansions_json"]))
        except json.JSONDecodeError:
            return ()
        if not isinstance(data, list):
            return ()
        return _unique_terms(str(item) for item in data)


def build_shards(
    files: list[RawFile],
    *,
    shard_dir: Path,
    shard_count: int,
    workers: int,
    rclone_bin: str = "rclone",
    max_files: int | None = None,
    time_budget_seconds: float | None = None,
    commit_interval: int = 1000,
    min_free_bytes: int = 0,
) -> list[ShardBuildResult]:
    """Build independent SQLite shard indexes in parallel."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    selected_files = files[:max_files] if max_files is not None else files
    chunks = _split_shard_files(selected_files, max(1, shard_count))
    jobs = [
        ShardBuildJob(
            shard_id=shard_id,
            index_path=str(shard_dir / f"fullraw_shard_{shard_id:04d}.sqlite"),
            files=chunk,
            rclone_bin=rclone_bin,
            time_budget_seconds=time_budget_seconds,
            commit_interval=max(1, commit_interval),
            min_free_bytes=min_free_bytes,
        )
        for shard_id, chunk in enumerate(chunks)
        if chunk
    ]
    if not jobs:
        return []
    max_workers = max(1, min(workers, len(jobs)))
    results: list[ShardBuildResult] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_build_shard_worker, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    ShardBuildResult(
                        shard_id=job.shard_id,
                        index_path=job.index_path,
                        files_total=len(job.files),
                        files_attempted=0,
                        files_completed=0,
                        files_failed=0,
                        papers_inserted=0,
                        stopped_for_budget=True,
                        elapsed_seconds=0.0,
                        bytes_used=0,
                        error=str(exc)[:500],
                    )
                )
    return sorted(results, key=lambda item: item.shard_id)


def build_upload_shard_batches(
    files: list[RawFile],
    *,
    shard_dir: Path,
    upload_remote: str,
    batch_files: int,
    shard_count: int,
    workers: int,
    rclone_bin: str = "rclone",
    max_files: int | None = None,
    commit_interval: int = 1000,
    min_free_bytes: int = 0,
    delete_local: bool = True,
) -> list[ShardBatchResult]:
    """Build local shard batches, upload completed batches, then free local disk."""
    if not upload_remote.strip():
        raise ValueError("upload remote is required for build-upload-shards")
    selected_files = files[:max_files] if max_files is not None else files
    results: list[ShardBatchResult] = []
    for batch_id, start in enumerate(range(0, len(selected_files), max(1, batch_files))):
        batch_started = time.monotonic()
        batch = selected_files[start:start + max(1, batch_files)]
        batch_name = f"batch_{batch_id:05d}"
        local_batch_dir = shard_dir / batch_name
        remote_batch_dir = f"{upload_remote.rstrip('/')}/{batch_name}"
        if _remote_complete_exists(remote_batch_dir, rclone_bin=rclone_bin):
            results.append(
                ShardBatchResult(
                    batch_id=batch_id,
                    batch_dir=str(local_batch_dir),
                    remote_dir=remote_batch_dir,
                    files_total=len(batch),
                    files_completed=len(batch),
                    files_failed=0,
                    papers_inserted=0,
                    bytes_used=0,
                    uploaded=True,
                    deleted_local=not local_batch_dir.exists(),
                    skipped=True,
                    elapsed_seconds=round(time.monotonic() - batch_started, 3),
                )
            )
            continue
        if local_batch_dir.exists():
            shutil.rmtree(local_batch_dir)
        shard_results = build_shards(
            batch,
            shard_dir=local_batch_dir,
            shard_count=shard_count,
            workers=workers,
            rclone_bin=rclone_bin,
            commit_interval=commit_interval,
            min_free_bytes=min_free_bytes,
        )
        files_completed = sum(result.files_completed for result in shard_results)
        files_failed = sum(result.files_failed for result in shard_results)
        papers_inserted = sum(result.papers_inserted for result in shard_results)
        bytes_used = sum(result.bytes_used for result in shard_results)
        error = "; ".join(result.error for result in shard_results if result.error)
        file_errors = "; ".join(result.file_errors for result in shard_results if result.file_errors)
        stopped = any(result.stopped_for_budget for result in shard_results)
        if not error and files_failed and files_completed == 0:
            error = "all files failed; treating as fatal source or transport failure"
        if error or stopped or files_completed + files_failed < len(batch):
            results.append(
                ShardBatchResult(
                    batch_id=batch_id,
                    batch_dir=str(local_batch_dir),
                    remote_dir=remote_batch_dir,
                    files_total=len(batch),
                    files_completed=files_completed,
                    files_failed=files_failed,
                    papers_inserted=papers_inserted,
                    bytes_used=bytes_used,
                    uploaded=False,
                    deleted_local=False,
                    skipped=False,
                    elapsed_seconds=round(time.monotonic() - batch_started, 3),
                    file_errors=file_errors,
                    error=error or "batch did not complete",
                )
            )
            break
        _write_batch_manifest(local_batch_dir, batch_id=batch_id, files=batch, shard_results=shard_results)
        _copy_batch_to_remote(local_batch_dir, remote_batch_dir, rclone_bin=rclone_bin)
        deleted = False
        if delete_local:
            shutil.rmtree(local_batch_dir)
            deleted = True
        result = ShardBatchResult(
            batch_id=batch_id,
            batch_dir=str(local_batch_dir),
            remote_dir=remote_batch_dir,
            files_total=len(batch),
            files_completed=files_completed,
            files_failed=files_failed,
            papers_inserted=papers_inserted,
            bytes_used=bytes_used,
            uploaded=True,
            deleted_local=deleted,
            skipped=False,
            elapsed_seconds=round(time.monotonic() - batch_started, 3),
            file_errors=file_errors,
        )
        print(json.dumps(asdict(result), sort_keys=True), flush=True)
        results.append(result)
    return results


def search_shards(
    index_paths: list[Path],
    query: str,
    *,
    limit: int = 25,
    year_min: int = 1900,
    year_max: int = 2100,
    rank_mode: str = "relevance",
    catalog: list[ShardCatalogEntry] | None = None,
    timeout_seconds: float | None = None,
) -> list[dict[str, object]]:
    """Search multiple shard DBs and merge ranked, deduped results."""
    if catalog is None:
        paths = select_search_shard_paths([path for path in index_paths if path.exists()])
    else:
        paths = [entry.path for entry in select_search_shard_entries(catalog, query=query)]
    return _search_shard_paths(
        paths,
        query,
        limit=limit,
        year_min=year_min,
        year_max=year_max,
        rank_mode=rank_mode,
        timeout_seconds=timeout_seconds,
        shard_timeout_seconds=timeout_seconds,
    )


def search_shard_entries(
    entries: list[ShardCatalogEntry],
    query: str,
    *,
    limit: int = 25,
    year_min: int = 1900,
    year_max: int = 2100,
    rank_mode: str = "relevance",
    workers: int | None = None,
    timeout_seconds: float | None = None,
    shard_timeout_seconds: float | None = None,
) -> list[dict[str, object]]:
    return _search_shard_paths(
        [entry.path for entry in entries],
        query,
        limit=limit,
        year_min=year_min,
        year_max=year_max,
        rank_mode=rank_mode,
        workers=workers,
        timeout_seconds=timeout_seconds,
        shard_timeout_seconds=shard_timeout_seconds,
    )


def _search_shard_paths(
    paths: list[Path],
    query: str,
    *,
    limit: int,
    year_min: int,
    year_max: int,
    rank_mode: str,
    workers: int | None = None,
    timeout_seconds: float | None = None,
    shard_timeout_seconds: float | None = None,
) -> list[dict[str, object]]:
    hits, _completed_paths, _timed_out = _search_shard_paths_with_paths(
        paths,
        query,
        limit=limit,
        year_min=year_min,
        year_max=year_max,
        rank_mode=rank_mode,
        workers=workers,
        timeout_seconds=timeout_seconds,
        shard_timeout_seconds=shard_timeout_seconds,
    )
    return hits


def _search_shard_paths_with_paths(
    paths: list[Path],
    query: str,
    *,
    limit: int,
    year_min: int,
    year_max: int,
    rank_mode: str,
    workers: int | None = None,
    timeout_seconds: float | None = None,
    shard_timeout_seconds: float | None = None,
) -> tuple[list[dict[str, object]], list[Path], bool]:
    merged: dict[str, dict[str, object]] = {}
    completed_paths: list[Path] = []
    timed_out = False
    worker_count = workers if workers is not None else int(os.environ.get("V5_MEMO_FULL_RAW_SEARCH_WORKERS", "8"))
    worker_count = max(1, min(worker_count, len(paths) or 1))
    pool = ThreadPoolExecutor(max_workers=worker_count)
    try:
        futures = {
            pool.submit(
                _search_one_shard,
                path,
                query,
                limit,
                year_min,
                year_max,
                rank_mode,
                shard_timeout_seconds,
            ): path
            for path in paths
        }
        completed = as_completed(futures, timeout=timeout_seconds) if timeout_seconds else as_completed(futures)
        for future in completed:
            path = futures[future]
            try:
                hits = future.result()
            except sqlite3.Error:
                continue
            completed_paths.append(path)
            for hit in hits:
                key = _dedupe_key(hit)
                existing = merged.get(key)
                if existing is None or _hit_score(hit) > _hit_score(existing):
                    merged[key] = hit
    except FuturesTimeoutError:
        timed_out = True
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    hits = sorted(
        merged.values(),
        key=lambda hit: (_hit_score(hit), _int_or_none(hit.get("cited_by_count")) or 0),
        reverse=True,
    )[: max(1, min(limit, 200))]
    return hits, completed_paths, timed_out


def select_search_shard_paths(paths: list[Path]) -> list[Path]:
    limit = _positive_int_env("V5_MEMO_FULL_RAW_SEARCH_SHARD_LIMIT")
    if limit is None or limit >= len(paths):
        return paths
    order = os.environ.get("V5_MEMO_FULL_RAW_SEARCH_SHARD_ORDER", "newest").casefold()
    if order in {"oldest", "first"}:
        return paths[:limit]
    if order == "spread" and limit > 1:
        step = (len(paths) - 1) / (limit - 1)
        return [paths[round(index * step)] for index in range(limit)]
    return paths[-limit:]


def select_search_shard_entries(
    entries: list[ShardCatalogEntry],
    *,
    query: str = "",
) -> list[ShardCatalogEntry]:
    entries = sorted(entries, key=lambda entry: (entry.batch_id, entry.shard_id, str(entry.path)))
    limit = _positive_int_env("V5_MEMO_FULL_RAW_SEARCH_SHARD_LIMIT")
    if limit is None or limit >= len(entries):
        return entries
    order = os.environ.get("V5_MEMO_FULL_RAW_SEARCH_SHARD_ORDER", "balanced").casefold()
    if order in {"oldest", "first", "newest", "spread"}:
        paths = select_search_shard_paths([entry.path for entry in entries])
        by_path = {entry.path: entry for entry in entries}
        return [by_path[path] for path in paths if path in by_path]
    return _select_balanced_shard_entries(entries, limit, query=query)


def select_sweep_shard_entries(
    entries: list[ShardCatalogEntry],
    *,
    query: str = "",
    limit: int | None = None,
) -> list[ShardCatalogEntry]:
    entries = sorted(entries, key=lambda entry: (entry.batch_id, entry.shard_id, str(entry.path)))
    if not entries:
        return []
    sweep_limit = limit if limit is not None else (_positive_int_env("V5_MEMO_FULL_RAW_SWEEP_SHARD_LIMIT") or 128)
    if sweep_limit >= len(entries):
        return entries
    sweep_limit = max(1, sweep_limit)
    selected: list[ShardCatalogEntry] = []
    query_terms = set(_fts_terms(query))
    if query_terms:
        relevant = [
            entry
            for entry in _rotate_entries(entries, _query_offset(f"sweep:{query}", len(entries)))
            if query_terms & set(entry.topic_terms)
        ]
        relevant.sort(
            key=lambda entry: (
                len(query_terms & set(entry.topic_terms)),
                entry.cited_by_max,
                entry.papers_inserted,
            ),
            reverse=True,
        )
        _extend_unique_entries(selected, relevant, min(sweep_limit, max(1, sweep_limit * 2 // 3)))
    _extend_unique_entries(selected, _select_balanced_shard_entries(entries, sweep_limit, query=query), sweep_limit)
    return selected


def _select_balanced_shard_entries(
    entries: list[ShardCatalogEntry],
    limit: int,
    *,
    query: str,
) -> list[ShardCatalogEntry]:
    by_source: dict[str, list[ShardCatalogEntry]] = {}
    for entry in entries:
        source = entry.sources[0] if entry.sources else "unknown"
        by_source.setdefault(source, []).append(entry)
    selected: list[ShardCatalogEntry] = []
    sources = sorted(by_source)
    per_source = max(1, limit // max(1, len(sources)))
    for source in sources:
        source_entries = _rotate_entries(by_source[source], _query_offset(f"{query}:{source}", len(by_source[source])))
        _extend_unique_entries(selected, _profiled_spread_entries(source_entries, per_source, query=query), limit)
    if len(selected) < limit:
        rotated = _rotate_entries(entries, _query_offset(query, len(entries)))
        _extend_unique_entries(selected, _profiled_spread_entries(rotated, limit, query=query), limit)
    if len(selected) < limit:
        _extend_unique_entries(selected, entries[-limit:], limit)
    return selected


def _profiled_spread_entries(
    entries: list[ShardCatalogEntry],
    limit: int,
    *,
    query: str,
) -> list[ShardCatalogEntry]:
    if limit <= 0:
        return []
    rotated = _rotate_entries(entries, _query_offset(query, len(entries)))
    if not any(entry.topic_terms or entry.year_min is not None or entry.cited_by_max for entry in entries):
        return _spread_entries(rotated, limit)
    selected: list[ShardCatalogEntry] = []
    query_terms = set(_fts_terms(query))
    if query_terms:
        relevant = sorted(
            rotated,
            key=lambda entry: (
                len(query_terms & set(entry.topic_terms)),
                entry.cited_by_max,
                entry.papers_inserted,
            ),
            reverse=True,
        )
        _extend_unique_entries(selected, relevant, min(limit, len(selected) + max(1, limit // 3)))
    _extend_unique_entries(
        selected,
        sorted(rotated, key=lambda entry: entry.cited_by_max, reverse=True),
        min(limit, len(selected) + 1),
    )
    _extend_unique_entries(
        selected,
        sorted(rotated, key=lambda entry: entry.cited_by_min),
        min(limit, len(selected) + 1),
    )
    _extend_unique_entries(
        selected,
        sorted(rotated, key=lambda entry: entry.year_max or -1, reverse=True),
        min(limit, len(selected) + 1),
    )
    _extend_unique_entries(
        selected,
        sorted(rotated, key=lambda entry: entry.year_min if entry.year_min is not None else 9999),
        min(limit, len(selected) + 1),
    )
    _extend_unique_entries(selected, _spread_entries(rotated, limit), limit)
    return selected


def _rotate_entries(
    entries: list[ShardCatalogEntry],
    offset: int,
) -> list[ShardCatalogEntry]:
    if not entries:
        return []
    offset %= len(entries)
    return [*entries[offset:], *entries[:offset]]


def _query_offset(query: str, size: int) -> int:
    if size <= 0 or not query:
        return 0
    return sum((index + 1) * ord(char) for index, char in enumerate(query)) % size


def _spread_entries(
    entries: list[ShardCatalogEntry],
    count: int,
) -> list[ShardCatalogEntry]:
    if count <= 0 or not entries:
        return []
    if count >= len(entries):
        return list(entries)
    if count == 1:
        return [entries[len(entries) // 2]]
    step = (len(entries) - 1) / (count - 1)
    out: list[ShardCatalogEntry] = []
    seen: set[Path] = set()
    for index in range(count):
        entry = entries[round(index * step)]
        if entry.path not in seen:
            out.append(entry)
            seen.add(entry.path)
    return out


def _extend_unique_entries(
    out: list[ShardCatalogEntry],
    candidates: Iterable[ShardCatalogEntry],
    limit: int,
) -> None:
    seen = {entry.path for entry in out}
    for entry in candidates:
        if len(out) >= limit:
            return
        if entry.path in seen:
            continue
        out.append(entry)
        seen.add(entry.path)


def _search_one_shard(
    path: Path,
    query: str,
    limit: int,
    year_min: int,
    year_max: int,
    rank_mode: str,
    timeout_seconds: float | None = None,
) -> list[dict[str, object]]:
    index = FullRawFtsIndex(path, read_only=True)
    try:
        return index.search(
            query,
            limit=limit,
            year_min=year_min,
            year_max=year_max,
            rank_mode=rank_mode,
            timeout_seconds=timeout_seconds,
        )
    finally:
        index.close()


def discover_shard_paths(shard_dir: Path, *, trust_filenames: bool = False) -> list[Path]:
    paths = sorted(path for path in shard_dir.rglob("*.sqlite") if path.is_file())
    if trust_filenames:
        return paths
    return [path for path in paths if _has_index_meta(path)]


def _has_index_meta(path: Path) -> bool:
    try:
        uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
        with sqlite3.connect(uri, uri=True, timeout=10.0) as conn:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'index_meta'"
            ).fetchone()
    except sqlite3.Error:
        return False
    return row is not None


def aggregate_shard_stats(index_paths: list[Path], *, files_total: int = 0) -> IndexStats:
    papers_indexed = 0
    files_indexed = 0
    bytes_used = 0
    for path in index_paths:
        if not path.exists():
            continue
        index = FullRawFtsIndex(path, read_only=True)
        try:
            stats = index.stats(files_total=0)
        except sqlite3.Error:
            continue
        finally:
            index.close()
        papers_indexed += stats.papers_indexed
        files_indexed += stats.files_indexed
        bytes_used += stats.bytes_used
    return IndexStats(
        papers_indexed=papers_indexed,
        files_indexed=files_indexed,
        files_total=files_total,
        bytes_used=bytes_used,
    )


def aggregate_shard_manifest_stats(shard_dir: Path, *, files_total: int = 0) -> IndexStats:
    catalog = build_shard_catalog(shard_dir, trust_filenames=True)
    return IndexStats(
        papers_indexed=sum(entry.papers_inserted for entry in catalog),
        files_indexed=sum(entry.files_completed for entry in catalog),
        files_total=files_total,
        bytes_used=sum(entry.bytes_used for entry in catalog),
    )


def backfill_shard_profiles(
    shard_dir: Path,
    *,
    trust_filenames: bool = True,
    max_shards: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    upload_remote: str = "",
    rclone_bin: str = "rclone",
    progress_interval: int = 0,
) -> dict[str, object]:
    manifests = _read_batch_manifests(shard_dir)
    changed_batches: set[Path] = set()
    updated_batches: set[Path] = set()
    current_batch: Path | None = None
    profiled = 0
    skipped = 0
    missing_manifest = 0
    failed: list[str] = []

    def flush_batch(batch_dir: Path | None) -> None:
        if batch_dir is None or batch_dir not in changed_batches:
            return
        manifest = manifests[batch_dir]
        manifest["profiled_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        updated_batches.add(batch_dir)
        if not dry_run:
            if upload_remote:
                _upload_manifest_payload(
                    manifest,
                    f"{upload_remote.rstrip('/')}/{batch_dir.name}/complete.json",
                    rclone_bin=rclone_bin,
                )
            else:
                _write_json_file(batch_dir / "complete.json", manifest)
        changed_batches.remove(batch_dir)
        if progress_interval > 0:
            print(
                json.dumps({
                    "batch": batch_dir.name,
                    "batches_updated": len(updated_batches),
                    "event": "profile_backfill_batch_flushed",
                    "shards_profiled": profiled,
                }, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )

    for path in discover_shard_paths(shard_dir, trust_filenames=trust_filenames):
        if max_shards is not None and profiled >= max_shards:
            break
        if current_batch is not None and path.parent != current_batch:
            flush_batch(current_batch)
        current_batch = path.parent
        manifest = manifests.get(path.parent)
        if manifest is None:
            missing_manifest += 1
            continue
        shard_meta = _manifest_shard(manifest, _shard_id_from_path(path))
        if shard_meta is None:
            missing_manifest += 1
            continue
        if not force and _shard_has_profile(shard_meta):
            skipped += 1
            continue
        try:
            profile = _profile_shard_path(path)
        except sqlite3.Error as exc:
            failed.append(f"{path}: {str(exc)[:200]}")
            continue
        shard_meta.update(profile)
        changed_batches.add(path.parent)
        profiled += 1
        if progress_interval > 0 and profiled % progress_interval == 0:
            print(
                json.dumps({
                    "event": "profile_backfill_progress",
                    "shards_profiled": profiled,
                    "shards_skipped": skipped,
                    "missing_manifest": missing_manifest,
                    "batch": path.parent.name,
                }, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )
    flush_batch(current_batch)
    for batch_dir in sorted(changed_batches):
        flush_batch(batch_dir)
    return {
        "shards_profiled": profiled,
        "shards_skipped": skipped,
        "missing_manifest": missing_manifest,
        "batches_updated": len(updated_batches),
        "dry_run": dry_run,
        "upload_remote": upload_remote,
        "errors": failed,
    }


def build_shard_catalog(
    shard_dir: Path,
    *,
    trust_filenames: bool = False,
) -> list[ShardCatalogEntry]:
    manifests = _read_batch_manifests(shard_dir)
    out: list[ShardCatalogEntry] = []
    for path in discover_shard_paths(shard_dir, trust_filenames=trust_filenames):
        manifest = manifests.get(path.parent)
        batch_id = _batch_id_from_path(path.parent)
        shard_id = _shard_id_from_path(path)
        sources: tuple[str, ...] = ()
        files_completed = 0
        papers_inserted = 0
        bytes_used = path.stat().st_size if path.exists() else 0
        year_min: int | None = None
        year_max: int | None = None
        cited_by_min = 0
        cited_by_max = 0
        cited_by_avg = 0.0
        topic_terms: tuple[str, ...] = ()
        if manifest is not None:
            batch_id = _int_or_none(manifest.get("batch_id")) or batch_id
            sources = _manifest_sources(manifest)
            shard_meta = _manifest_shard(manifest, shard_id)
            if shard_meta is not None:
                files_completed = _int_or_none(shard_meta.get("files_completed")) or 0
                papers_inserted = _int_or_none(shard_meta.get("papers_inserted")) or 0
                bytes_used = _int_or_none(shard_meta.get("bytes_used")) or bytes_used
                year_min = _int_or_none(shard_meta.get("year_min"))
                year_max = _int_or_none(shard_meta.get("year_max"))
                cited_by_min = _int_or_none(shard_meta.get("cited_by_min")) or 0
                cited_by_max = _int_or_none(shard_meta.get("cited_by_max")) or 0
                cited_by_avg = round(_float_or_none(shard_meta.get("cited_by_avg")) or 0.0, 3)
                topic_terms = _string_tuple(shard_meta.get("topic_terms"))
        out.append(ShardCatalogEntry(
            path=path,
            batch_id=batch_id,
            shard_id=shard_id,
            sources=sources,
            files_completed=files_completed,
            papers_inserted=papers_inserted,
            bytes_used=bytes_used,
            year_min=year_min,
            year_max=year_max,
            cited_by_min=cited_by_min,
            cited_by_max=cited_by_max,
            cited_by_avg=cited_by_avg,
            topic_terms=topic_terms,
        ))
    return out


def shard_coverage_receipt(
    entries: list[ShardCatalogEntry],
    selected: list[ShardCatalogEntry],
) -> dict[str, object]:
    return {
        "shards_total": len(entries),
        "shards_searched": len(selected),
        "partial_shard_search": len(entries) > len(selected),
        "sources_total": _source_counts(entries),
        "sources_searched": _source_counts(selected),
        "batch_range_total": _batch_range(entries),
        "batch_range_searched": _batch_range(selected),
        "year_range_total": _year_range(entries),
        "year_range_searched": _year_range(selected),
        "cited_by_range_total": _cited_by_range(entries),
        "cited_by_range_searched": _cited_by_range(selected),
        "topic_terms_searched": _topic_terms(selected),
        "papers_total": sum(entry.papers_inserted for entry in entries),
        "papers_searched": sum(entry.papers_inserted for entry in selected),
    }


def shard_coverage_gate_response(
    receipt: dict[str, object],
    *,
    min_shards_searched: int = 0,
    min_sources_searched: int = 0,
) -> tuple[int, dict[str, object]] | None:
    failures: list[str] = []
    shards_searched = _int_or_none(receipt.get("shards_searched")) or 0
    sources_searched = receipt.get("sources_searched")
    source_count = len(sources_searched) if isinstance(sources_searched, dict) else 0
    if min_shards_searched > 0 and shards_searched < min_shards_searched:
        failures.append(f"shards_searched {shards_searched} < required {min_shards_searched}")
    if min_sources_searched > 0 and source_count < min_sources_searched:
        failures.append(f"sources_searched {source_count} < required {min_sources_searched}")
    if not failures:
        return None
    return 422, {
        "error": "coverage_too_narrow",
        "failures": failures,
        "requirements": {
            "min_shards_searched": min_shards_searched,
            "min_sources_searched": min_sources_searched,
        },
        "shard_receipt": receipt,
    }


def _read_batch_manifests(shard_dir: Path) -> dict[Path, dict[str, object]]:
    manifests: dict[Path, dict[str, object]] = {}
    for manifest_path in sorted(shard_dir.rglob("complete.json")):
        try:
            payload = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            manifests[manifest_path.parent] = payload
    return manifests


def _shard_has_profile(shard_meta: dict[str, object]) -> bool:
    return bool(
        _int_or_none(shard_meta.get("year_min")) is not None
        or _int_or_none(shard_meta.get("year_max")) is not None
        or _int_or_none(shard_meta.get("cited_by_max"))
        or _string_tuple(shard_meta.get("topic_terms"))
    )


def _profile_shard_path(path: Path) -> dict[str, object]:
    index = FullRawFtsIndex(path, read_only=True)
    try:
        return index.profile()
    finally:
        index.close()


def _write_json_file(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _upload_manifest_payload(
    payload: dict[str, object],
    remote_path: str,
    *,
    rclone_bin: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="v5-fullraw-manifest-") as tmp:
        local_path = Path(tmp) / "complete.json"
        _write_json_file(local_path, payload)
        if remote_path.startswith("file://"):
            destination = Path(remote_path.removeprefix("file://"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, destination)
            return
        copied = subprocess.run(
            [rclone_bin, "copyto", str(local_path), remote_path],
            text=True,
            capture_output=True,
            timeout=None,
            check=False,
        )
        if copied.returncode != 0:
            raise RuntimeError((copied.stderr or copied.stdout or "rclone copyto failed").strip()[:500])


def _manifest_sources(manifest: dict[str, object]) -> tuple[str, ...]:
    sources: set[str] = set()
    files = manifest.get("files", [])
    if isinstance(files, list):
        for raw_file in files:
            if isinstance(raw_file, dict):
                source = _clean(raw_file.get("source"))
                if source:
                    sources.add(source)
    return tuple(sorted(sources))


def _manifest_shard(
    manifest: dict[str, object],
    shard_id: int,
) -> dict[str, object] | None:
    shards = manifest.get("shards", [])
    if not isinstance(shards, list):
        return None
    for shard in shards:
        if not isinstance(shard, dict):
            continue
        parsed_shard_id = _int_or_none(shard.get("shard_id"))
        if parsed_shard_id is not None and parsed_shard_id == shard_id:
            return shard
    return None


def _source_counts(entries: list[ShardCatalogEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        sources = entry.sources or ("unknown",)
        for source in sources:
            counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items()))


def _batch_range(entries: list[ShardCatalogEntry]) -> dict[str, int | None]:
    if not entries:
        return {"min": None, "max": None}
    batch_ids = [entry.batch_id for entry in entries]
    return {"min": min(batch_ids), "max": max(batch_ids)}


def _year_range(entries: list[ShardCatalogEntry]) -> dict[str, int | None]:
    years = [
        year
        for entry in entries
        for year in (entry.year_min, entry.year_max)
        if year is not None
    ]
    return {"min": min(years), "max": max(years)} if years else {"min": None, "max": None}


def _cited_by_range(entries: list[ShardCatalogEntry]) -> dict[str, int]:
    if not entries:
        return {"min": 0, "max": 0}
    return {
        "min": min(entry.cited_by_min for entry in entries),
        "max": max(entry.cited_by_max for entry in entries),
    }


def _topic_terms(entries: list[ShardCatalogEntry], *, limit: int = 12) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    for entry in entries:
        counts.update(entry.topic_terms)
    return tuple(term for term, _count in counts.most_common(limit))


def _batch_id_from_path(path: Path) -> int:
    match = re.search(r"batch_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _shard_id_from_path(path: Path) -> int:
    match = re.search(r"shard_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _split_shard_files(files: list[RawFile], shard_count: int) -> list[list[RawFile]]:
    chunks: list[list[RawFile]] = [[] for _ in range(max(1, shard_count))]
    for index, raw_file in enumerate(files):
        chunks[index % len(chunks)].append(raw_file)
    return chunks


def _remote_complete_exists(remote_dir: str, *, rclone_bin: str) -> bool:
    if remote_dir.startswith("file://"):
        return (Path(remote_dir.removeprefix("file://")) / "complete.json").exists()
    checked = subprocess.run(
        [rclone_bin, "lsf", f"{remote_dir.rstrip('/')}/complete.json"],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return checked.returncode == 0 and "complete.json" in checked.stdout


def _write_batch_manifest(
    batch_dir: Path,
    *,
    batch_id: int,
    files: list[RawFile],
    shard_results: list[ShardBuildResult],
) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "batch_id": batch_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": [asdict(raw_file) for raw_file in files],
        "shards": [asdict(result) for result in shard_results],
        "totals": {
            "files_total": len(files),
            "files_completed": sum(result.files_completed for result in shard_results),
            "files_failed": sum(result.files_failed for result in shard_results),
            "papers_inserted": sum(result.papers_inserted for result in shard_results),
            "bytes_used": sum(result.bytes_used for result in shard_results),
            "file_errors": [
                result.file_errors for result in shard_results if result.file_errors
            ],
        },
    }
    (batch_dir / "complete.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _copy_batch_to_remote(batch_dir: Path, remote_dir: str, *, rclone_bin: str) -> None:
    if remote_dir.startswith("file://"):
        destination = Path(remote_dir.removeprefix("file://"))
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(batch_dir, destination)
        return
    copied = subprocess.run(
        [rclone_bin, "copy", str(batch_dir), remote_dir],
        text=True,
        capture_output=True,
        timeout=None,
        check=False,
    )
    if copied.returncode != 0:
        raise RuntimeError((copied.stderr or copied.stdout or "rclone copy failed").strip()[:500])


def _build_shard_worker(job: ShardBuildJob) -> ShardBuildResult:
    index = FullRawFtsIndex(Path(job.index_path))
    try:
        result = index.index_files(
            job.files,
            rclone_bin=job.rclone_bin,
            time_budget_seconds=job.time_budget_seconds,
            commit_interval=job.commit_interval,
            min_free_bytes=job.min_free_bytes,
        )
        stats = index.stats(files_total=len(job.files))
        profile = index.profile()
    finally:
        index.close()
    return ShardBuildResult(
        shard_id=job.shard_id,
        index_path=job.index_path,
        files_total=len(job.files),
        files_attempted=result.files_attempted,
        files_completed=result.files_completed,
        files_failed=result.files_failed,
        papers_inserted=result.papers_inserted,
        stopped_for_budget=result.stopped_for_budget,
        elapsed_seconds=result.elapsed_seconds,
        bytes_used=stats.bytes_used,
        year_min=_int_or_none(profile.get("year_min")),
        year_max=_int_or_none(profile.get("year_max")),
        cited_by_min=_int_or_none(profile.get("cited_by_min")) or 0,
        cited_by_max=_int_or_none(profile.get("cited_by_max")) or 0,
        cited_by_avg=round(_float_or_none(profile.get("cited_by_avg")) or 0.0, 3),
        topic_terms=_string_tuple(profile.get("topic_terms")),
        file_errors=result.file_errors,
    )


def run_server() -> None:
    host = os.environ.get("V5_MEMO_FULL_RAW_INDEX_HOST", "127.0.0.1")
    port = int(os.environ.get("V5_MEMO_FULL_RAW_INDEX_PORT", "9902"))
    index_path = Path(
        os.environ.get("V5_MEMO_FULL_RAW_INDEX_PATH", "/var/lib/v5-memo/fullraw_index.sqlite")
    )
    shard_dir_config = os.environ.get("V5_MEMO_FULL_RAW_SHARD_DIR", "").strip()
    shard_dir = Path(shard_dir_config) if shard_dir_config else None
    trust_shard_filenames = os.environ.get(
        "V5_MEMO_FULL_RAW_SHARD_TRUST_FILENAMES", ""
    ).casefold() in {"1", "true", "yes"}
    shard_manifest_stats = os.environ.get(
        "V5_MEMO_FULL_RAW_SHARD_MANIFEST_STATS", ""
    ).casefold() in {"1", "true", "yes"}
    shard_catalog_ttl = _float_or_none(
        os.environ.get("V5_MEMO_FULL_RAW_SHARD_CATALOG_TTL_SECONDS", "")
    ) or 60.0
    min_shards_searched = _positive_int_env("V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED") or 0
    min_sources_searched = _positive_int_env("V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED") or 0
    sweep_enabled = os.environ.get("V5_MEMO_FULL_RAW_ASYNC_SWEEP", "").casefold() in {"1", "true", "yes"}
    sweep_ttl = _float_or_none(os.environ.get("V5_MEMO_FULL_RAW_SWEEP_TTL_SECONDS", "")) or 86400.0
    sweep_workers = _positive_int_env("V5_MEMO_FULL_RAW_SWEEP_WORKERS") or 1
    sweep_max_inflight = _positive_int_env("V5_MEMO_FULL_RAW_SWEEP_MAX_INFLIGHT") or 1
    sweep_shard_limit = _positive_int_env("V5_MEMO_FULL_RAW_SWEEP_SHARD_LIMIT") or 128
    sweep_timeout_seconds = _float_or_none(os.environ.get("V5_MEMO_FULL_RAW_SWEEP_TIMEOUT_SECONDS", "")) or 300.0
    sweep_timeout_seconds = max(1.0, min(sweep_timeout_seconds, 3600.0))
    sweep_shard_timeout_seconds = _float_or_none(
        os.environ.get("V5_MEMO_FULL_RAW_SWEEP_SHARD_TIMEOUT_SECONDS", "")
    ) or 10.0
    sweep_shard_timeout_seconds = max(0.1, min(sweep_shard_timeout_seconds, sweep_timeout_seconds))
    sweep_cache_dir_config = os.environ.get("V5_MEMO_FULL_RAW_SWEEP_CACHE_DIR", "").strip()
    sweep_cache_dir = Path(sweep_cache_dir_config) if sweep_cache_dir_config else None
    manifest_path = Path(
        os.environ.get("V5_MEMO_FULL_RAW_MANIFEST", "/var/lib/v5-memo/fullraw_manifest.json")
    )
    rclone_bin = os.environ.get("V5_MEMO_FULL_RAW_RCLONE", "rclone")
    refresh = os.environ.get("V5_MEMO_FULL_RAW_REFRESH_MANIFEST", "").casefold() in {"1", "true", "yes"}
    files = load_or_build_manifest(manifest_path, refresh=refresh, rclone_bin=rclone_bin)
    token = (
        os.environ.get("V5_MEMO_FULL_RAW_INDEX_TOKEN", "")
        or os.environ.get("V5_MEMO_FULL_RAW_TOKEN", "")
    ).strip()
    index = None if shard_dir else FullRawFtsIndex(index_path)
    if index is not None:
        index.initialize()
    catalog_cache: tuple[float, list[ShardCatalogEntry]] = (0.0, [])
    sweep_cache: dict[str, SweepCacheEntry] = {}
    sweep_inflight: set[str] = set()
    sweep_lock = threading.RLock()

    def current_catalog() -> list[ShardCatalogEntry]:
        nonlocal catalog_cache
        if shard_dir is None:
            return []
        now = time.monotonic()
        if catalog_cache[1] and now - catalog_cache[0] < shard_catalog_ttl:
            return catalog_cache[1]
        catalog = build_shard_catalog(shard_dir, trust_filenames=trust_shard_filenames)
        catalog_cache = (now, catalog)
        return catalog

    def stats_from_catalog(catalog: list[ShardCatalogEntry]) -> IndexStats:
        return IndexStats(
            papers_indexed=sum(entry.papers_inserted for entry in catalog),
            files_indexed=sum(entry.files_completed for entry in catalog),
            files_total=len(files),
            bytes_used=sum(entry.bytes_used for entry in catalog),
        )

    def current_stats() -> IndexStats:
        if shard_dir is not None:
            if shard_manifest_stats:
                return stats_from_catalog(current_catalog())
            return aggregate_shard_stats(
                discover_shard_paths(shard_dir, trust_filenames=trust_shard_filenames),
                files_total=len(files),
            )
        assert index is not None
        return index.stats(files_total=len(files))

    def current_search(
        query: str,
        *,
        limit: int,
        year_min: int,
        year_max: int,
        rank_mode: str,
        timeout_seconds: float | None,
    ) -> list[dict[str, object]]:
        if shard_dir is not None:
            catalog = current_catalog()
            return search_shards(
                [entry.path for entry in catalog],
                query,
                limit=limit,
                year_min=year_min,
                year_max=year_max,
                rank_mode=rank_mode,
                catalog=catalog,
                timeout_seconds=timeout_seconds,
            )
        assert index is not None
        return index.search(query, limit=limit, year_min=year_min, year_max=year_max, rank_mode=rank_mode)

    def sweep_cache_get(key: str) -> SweepCacheEntry | None:
        with sweep_lock:
            entry = sweep_cache.get(key)
            if entry is not None and (sweep_ttl <= 0 or time.time() - entry.created_at <= sweep_ttl):
                return entry
        cache_path = _sweep_cache_path(sweep_cache_dir, key)
        if cache_path is None or not cache_path.exists():
            return None
        entry = _load_sweep_cache(cache_path, ttl_seconds=sweep_ttl)
        if entry is not None:
            with sweep_lock:
                sweep_cache[key] = entry
        return entry

    def sweep_cache_put(key: str, entry: SweepCacheEntry) -> None:
        with sweep_lock:
            sweep_cache[key] = entry
            sweep_inflight.discard(key)
        cache_path = _sweep_cache_path(sweep_cache_dir, key)
        if cache_path is not None:
            _write_sweep_cache(cache_path, entry)

    def enqueue_sweep(
        *,
        key: str,
        query: str,
        limit: int,
        year_min: int,
        year_max: int,
        rank_mode: str,
        catalog: list[ShardCatalogEntry],
    ) -> str:
        if not sweep_enabled or shard_dir is None:
            return "disabled"
        if sweep_cache_get(key) is not None:
            return "hit"
        with sweep_lock:
            if key in sweep_inflight:
                return "running"
            if len(sweep_inflight) >= sweep_max_inflight:
                return "busy"
            sweep_inflight.add(key)

        def worker() -> None:
            try:
                sweep_entries = select_sweep_shard_entries(catalog, query=query, limit=sweep_shard_limit)
                hits, completed_paths, timed_out = _search_shard_paths_with_paths(
                    [entry.path for entry in sweep_entries],
                    query,
                    limit=limit,
                    year_min=year_min,
                    year_max=year_max,
                    rank_mode=rank_mode,
                    workers=sweep_workers,
                    timeout_seconds=sweep_timeout_seconds,
                    shard_timeout_seconds=sweep_shard_timeout_seconds,
                )
                completed = set(completed_paths)
                searched_entries = [entry for entry in sweep_entries if entry.path in completed]
                receipt = shard_coverage_receipt(catalog, searched_entries)
                receipt["sweep_scope"] = "relevant"
                receipt["sweep_shard_limit"] = sweep_shard_limit
                receipt["sweep_selected_shards"] = len(sweep_entries)
                receipt["sweep_timed_out"] = timed_out
                receipt["sweep_timeout_seconds"] = sweep_timeout_seconds
                receipt["sweep_shard_timeout_seconds"] = sweep_shard_timeout_seconds
                sweep_cache_put(key, SweepCacheEntry(time.time(), hits, receipt))
            except Exception:
                with sweep_lock:
                    sweep_inflight.discard(key)

        threading.Thread(target=worker, daemon=True).start()
        return "queued"

    def current_receipt(query: str) -> dict[str, object]:
        if shard_dir is None:
            return {}
        catalog = current_catalog()
        return shard_coverage_receipt(catalog, select_search_shard_entries(catalog, query=query))

    def coverage_requirements() -> dict[str, int]:
        return {
            "min_shards_searched": min_shards_searched,
            "min_sources_searched": min_sources_searched,
        }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404)
                return
            stats = current_stats()
            _write_json(self, 200, {
                "ok": True,
                "backend": _BACKEND,
                "index_path": str(index_path),
                "shard_dir": str(shard_dir) if shard_dir is not None else "",
                "papers_indexed": stats.papers_indexed,
                "files_indexed": stats.files_indexed,
                "files_total": stats.files_total,
                "complete": stats.complete,
                "bytes_used": stats.bytes_used,
                "shard_receipt": current_receipt("") if shard_dir is not None else {},
                "coverage_requirements": coverage_requirements(),
            })

        def do_POST(self) -> None:
            if self.path != "/search":
                self.send_error(404)
                return
            if token and self.headers.get("Authorization", "") != f"Bearer {token}":
                _write_json(self, 401, {"error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                query = str(payload.get("query", "")).strip()
                limit = max(1, min(int(payload.get("limit") or payload.get("top_k") or 25), 200))
                year_min = int(payload.get("year_min") or 1900)
                year_max = int(payload.get("year_max") or 2100)
                rank_mode = _rank_mode(payload.get("rank_mode"))
                timeout_seconds = _float_or_none(payload.get("timeout_seconds"))
                if timeout_seconds is not None:
                    timeout_seconds = max(0.1, min(timeout_seconds, 600.0))
                raw_cache_only = payload.get("cache_only")
                cache_only = raw_cache_only is True or (
                    isinstance(raw_cache_only, str)
                    and raw_cache_only.strip().casefold() in {"1", "true", "yes", "on"}
                )
                raw_queue_if_missing = payload.get("queue_if_missing")
                queue_if_missing = raw_queue_if_missing is True or (
                    isinstance(raw_queue_if_missing, str)
                    and raw_queue_if_missing.strip().casefold() in {"1", "true", "yes", "on"}
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                _write_json(self, 400, {"error": "bad request"})
                return
            if not query:
                _write_json(self, 400, {"error": "query is required"})
                return
            started = time.monotonic()
            catalog = current_catalog() if shard_dir is not None else []
            cache_key = _sweep_cache_key(
                query,
                limit=limit,
                year_min=year_min,
                year_max=year_max,
                rank_mode=rank_mode,
                sweep_shard_limit=sweep_shard_limit,
            )
            cached = sweep_cache_get(cache_key) if catalog else None
            sweep_status = "disabled"
            if cached is not None:
                hits = cached.hits
                receipt = cached.receipt
                sweep_status = "hit"
            else:
                if cache_only:
                    hits = []
                    receipt = {}
                    if not sweep_enabled or not catalog:
                        sweep_status = "disabled"
                    elif queue_if_missing:
                        sweep_status = enqueue_sweep(
                            key=cache_key,
                            query=query,
                            limit=limit,
                            year_min=year_min,
                            year_max=year_max,
                            rank_mode=rank_mode,
                            catalog=catalog,
                        )
                    else:
                        with sweep_lock:
                            sweep_status = "running" if cache_key in sweep_inflight else "miss"
                    _write_json(self, 200, {
                        "meta": {
                            "count": len(hits),
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                            "backend": _BACKEND,
                            "rank_mode": rank_mode,
                            "shard_receipt": receipt,
                            "cache_only": True,
                            "async_sweep": {
                                "enabled": sweep_enabled and bool(catalog),
                                "status": sweep_status,
                                "cache_key": cache_key if sweep_enabled and catalog else "",
                                "scope": "relevant",
                                "shard_limit": sweep_shard_limit,
                            },
                        },
                        "results": hits,
                    })
                    return
                else:
                    receipt = (
                        shard_coverage_receipt(catalog, select_search_shard_entries(catalog, query=query))
                        if catalog
                        else {}
                    )
                    coverage_gate = shard_coverage_gate_response(
                        receipt,
                        min_shards_searched=min_shards_searched,
                        min_sources_searched=min_sources_searched,
                    )
                    if coverage_gate is not None:
                        status, body = coverage_gate
                        _write_json(self, status, body)
                        return
                    hits = current_search(
                        query,
                        limit=limit,
                        year_min=year_min,
                        year_max=year_max,
                        rank_mode=rank_mode,
                        timeout_seconds=timeout_seconds,
                    )
                    sweep_status = enqueue_sweep(
                        key=cache_key,
                        query=query,
                        limit=limit,
                        year_min=year_min,
                        year_max=year_max,
                        rank_mode=rank_mode,
                        catalog=catalog,
                    )
                    if sweep_status == "hit" and (cached := sweep_cache_get(cache_key)) is not None:
                        hits = cached.hits
                        receipt = cached.receipt
            stats = current_stats()
            explain = (
                {"fts_match": query, "groups": []}
                if shard_dir is not None
                else index.explain_query(query)  # type: ignore[union-attr]
            )
            _write_json(self, 200, {
                "meta": {
                    "count": len(hits),
                    "papers_indexed": stats.papers_indexed,
                    "files_indexed": stats.files_indexed,
                    "files_total": stats.files_total,
                    "complete": stats.complete,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "backend": _BACKEND,
                    "expanded_query": explain["fts_match"],
                    "term_groups": explain["groups"],
                    "rank_mode": rank_mode,
                    "shard_receipt": receipt,
                    "cache_only": cache_only,
                    "async_sweep": {
                        "enabled": sweep_enabled and bool(catalog),
                        "status": sweep_status,
                        "cache_key": cache_key if sweep_enabled and catalog else "",
                        "scope": "relevant",
                        "shard_limit": sweep_shard_limit,
                    },
                },
                "results": hits,
            })

        def log_message(self, fmt: str, *args: object) -> None:
            return

    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and serve the V5 fullraw FTS index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--index-path", default=os.environ.get("V5_MEMO_FULL_RAW_INDEX_PATH", "/var/lib/v5-memo/fullraw_index.sqlite"))
    build.add_argument("--manifest", default=os.environ.get("V5_MEMO_FULL_RAW_MANIFEST", "/var/lib/v5-memo/fullraw_manifest.json"))
    build.add_argument("--refresh-manifest", action="store_true")
    build.add_argument("--rclone-bin", default=os.environ.get("V5_MEMO_FULL_RAW_RCLONE", "rclone"))
    build.add_argument("--max-files", type=int)
    build.add_argument("--time-budget-seconds", type=float)
    build.add_argument("--commit-interval", type=int, default=1000)
    build.add_argument(
        "--min-free-gb",
        type=float,
        default=float(os.environ.get("V5_MEMO_FULL_RAW_INDEX_MIN_FREE_GB", "40")),
    )

    shard_dir_default = os.environ.get(
        "V5_MEMO_FULL_RAW_SHARD_DIR",
        os.environ.get("V5_MEMO_FULL_RAW_SHARD_BUILD_DIR", "/var/lib/v5-memo/fullraw-shards"),
    )
    build_shards_parser = subparsers.add_parser("build-shards")
    build_shards_parser.add_argument("--shard-dir", default=shard_dir_default)
    build_shards_parser.add_argument("--manifest", default=os.environ.get("V5_MEMO_FULL_RAW_MANIFEST", "/var/lib/v5-memo/fullraw_manifest.json"))
    build_shards_parser.add_argument("--refresh-manifest", action="store_true")
    build_shards_parser.add_argument("--rclone-bin", default=os.environ.get("V5_MEMO_FULL_RAW_RCLONE", "rclone"))
    build_shards_parser.add_argument("--shards", type=int, default=int(os.environ.get("V5_MEMO_FULL_RAW_SHARDS", "4")))
    build_shards_parser.add_argument("--workers", type=int, default=int(os.environ.get("V5_MEMO_FULL_RAW_SHARD_WORKERS", "4")))
    build_shards_parser.add_argument("--max-files", type=int)
    build_shards_parser.add_argument("--time-budget-seconds", type=float)
    build_shards_parser.add_argument("--commit-interval", type=int, default=1000)
    build_shards_parser.add_argument(
        "--min-free-gb",
        type=float,
        default=float(os.environ.get("V5_MEMO_FULL_RAW_INDEX_MIN_FREE_GB", "40")),
    )

    build_upload_parser = subparsers.add_parser("build-upload-shards")
    build_upload_parser.add_argument("--shard-dir", default=shard_dir_default)
    build_upload_parser.add_argument("--upload-remote", default=os.environ.get("V5_MEMO_FULL_RAW_SHARD_REMOTE", ""))
    build_upload_parser.add_argument("--manifest", default=os.environ.get("V5_MEMO_FULL_RAW_MANIFEST", "/var/lib/v5-memo/fullraw_manifest.json"))
    build_upload_parser.add_argument("--refresh-manifest", action="store_true")
    build_upload_parser.add_argument("--rclone-bin", default=os.environ.get("V5_MEMO_FULL_RAW_RCLONE", "rclone"))
    build_upload_parser.add_argument("--batch-files", type=int, default=int(os.environ.get("V5_MEMO_FULL_RAW_SHARD_BATCH_FILES", "16")))
    build_upload_parser.add_argument("--shards", type=int, default=int(os.environ.get("V5_MEMO_FULL_RAW_SHARDS", "4")))
    build_upload_parser.add_argument("--workers", type=int, default=int(os.environ.get("V5_MEMO_FULL_RAW_SHARD_WORKERS", "4")))
    build_upload_parser.add_argument("--max-files", type=int)
    build_upload_parser.add_argument("--commit-interval", type=int, default=1000)
    build_upload_parser.add_argument(
        "--min-free-gb",
        type=float,
        default=float(os.environ.get("V5_MEMO_FULL_RAW_INDEX_MIN_FREE_GB", "40")),
    )
    build_upload_parser.add_argument("--keep-local", action="store_true")

    search = subparsers.add_parser("search")
    search.add_argument("query")
    search.add_argument("--index-path", default=os.environ.get("V5_MEMO_FULL_RAW_INDEX_PATH", "/var/lib/v5-memo/fullraw_index.sqlite"))
    search.add_argument("--limit", type=int, default=10)

    search_shards_parser = subparsers.add_parser("search-shards")
    search_shards_parser.add_argument("query")
    search_shards_parser.add_argument("--shard-dir", default=shard_dir_default)
    search_shards_parser.add_argument("--limit", type=int, default=10)
    search_shards_parser.add_argument("--year-min", type=int, default=1900)
    search_shards_parser.add_argument("--year-max", type=int, default=2100)

    explain = subparsers.add_parser("explain")
    explain.add_argument("query")
    explain.add_argument("--index-path", default=os.environ.get("V5_MEMO_FULL_RAW_INDEX_PATH", "/var/lib/v5-memo/fullraw_index.sqlite"))

    stats = subparsers.add_parser("stats")
    stats.add_argument("--index-path", default=os.environ.get("V5_MEMO_FULL_RAW_INDEX_PATH", "/var/lib/v5-memo/fullraw_index.sqlite"))
    stats.add_argument("--manifest", default=os.environ.get("V5_MEMO_FULL_RAW_MANIFEST", "/var/lib/v5-memo/fullraw_manifest.json"))

    stats_shards = subparsers.add_parser("stats-shards")
    stats_shards.add_argument("--shard-dir", default=shard_dir_default)
    stats_shards.add_argument("--manifest", default=os.environ.get("V5_MEMO_FULL_RAW_MANIFEST", "/var/lib/v5-memo/fullraw_manifest.json"))

    backfill_profiles = subparsers.add_parser("backfill-shard-profiles")
    backfill_profiles.add_argument("--shard-dir", default=shard_dir_default)
    backfill_profiles.add_argument("--max-shards", type=int)
    backfill_profiles.add_argument("--force", action="store_true")
    backfill_profiles.add_argument("--dry-run", action="store_true")
    backfill_profiles.add_argument("--upload-remote", default="")
    backfill_profiles.add_argument("--rclone-bin", default=os.environ.get("V5_MEMO_FULL_RAW_RCLONE", "rclone"))
    backfill_profiles.add_argument("--verify-sqlite", action="store_true")
    backfill_profiles.add_argument("--progress-interval", type=int, default=25)

    subparsers.add_parser("serve")
    args = parser.parse_args()

    if args.command == "serve":
        run_server()
        return

    if args.command == "build":
        manifest_path = Path(args.manifest)
        files = load_or_build_manifest(
            manifest_path,
            refresh=bool(args.refresh_manifest),
            rclone_bin=str(args.rclone_bin),
        )
        index = FullRawFtsIndex(Path(args.index_path))
        try:
            result = index.index_files(
                files,
                rclone_bin=str(args.rclone_bin),
                max_files=args.max_files,
                time_budget_seconds=args.time_budget_seconds,
                commit_interval=args.commit_interval,
                min_free_bytes=int(max(0.0, args.min_free_gb) * 1024**3),
            )
            print(json.dumps(asdict(result), sort_keys=True))
        finally:
            index.close()
        return

    if args.command == "build-shards":
        manifest_path = Path(args.manifest)
        files = load_or_build_manifest(
            manifest_path,
            refresh=bool(args.refresh_manifest),
            rclone_bin=str(args.rclone_bin),
        )
        results = build_shards(
            files,
            shard_dir=Path(args.shard_dir),
            shard_count=max(1, int(args.shards)),
            workers=max(1, int(args.workers)),
            rclone_bin=str(args.rclone_bin),
            max_files=args.max_files,
            time_budget_seconds=args.time_budget_seconds,
            commit_interval=args.commit_interval,
            min_free_bytes=int(max(0.0, args.min_free_gb) * 1024**3),
        )
        print(json.dumps({
            "shards": [asdict(result) for result in results],
            "totals": {
                "shards": len(results),
                "files_attempted": sum(result.files_attempted for result in results),
                "files_completed": sum(result.files_completed for result in results),
                "files_failed": sum(result.files_failed for result in results),
                "papers_inserted": sum(result.papers_inserted for result in results),
                "bytes_used": sum(result.bytes_used for result in results),
                "file_error_shards": sum(1 for result in results if result.file_errors),
                "errors": sum(1 for result in results if result.error),
                "stopped_for_budget": any(result.stopped_for_budget for result in results),
            },
        }, sort_keys=True))
        return

    if args.command == "build-upload-shards":
        manifest_path = Path(args.manifest)
        files = load_or_build_manifest(
            manifest_path,
            refresh=bool(args.refresh_manifest),
            rclone_bin=str(args.rclone_bin),
        )
        batch_results = build_upload_shard_batches(
            files,
            shard_dir=Path(args.shard_dir),
            upload_remote=str(args.upload_remote),
            batch_files=max(1, int(args.batch_files)),
            shard_count=max(1, int(args.shards)),
            workers=max(1, int(args.workers)),
            rclone_bin=str(args.rclone_bin),
            max_files=args.max_files,
            commit_interval=args.commit_interval,
            min_free_bytes=int(max(0.0, args.min_free_gb) * 1024**3),
            delete_local=not bool(args.keep_local),
        )
        print(json.dumps({
            "batches": [asdict(result) for result in batch_results],
            "totals": {
                "batches": len(batch_results),
                "files_completed": sum(result.files_completed for result in batch_results),
                "files_failed": sum(result.files_failed for result in batch_results),
                "papers_inserted": sum(result.papers_inserted for result in batch_results),
                "bytes_used": sum(result.bytes_used for result in batch_results),
                "uploaded": sum(1 for result in batch_results if result.uploaded),
                "skipped": sum(1 for result in batch_results if result.skipped),
                "file_error_batches": sum(1 for result in batch_results if result.file_errors),
                "errors": sum(1 for result in batch_results if result.error),
            },
        }, sort_keys=True))
        if any(result.error for result in batch_results):
            raise SystemExit(2)
        return

    if args.command == "search":
        index = FullRawFtsIndex(Path(args.index_path))
        try:
            print(json.dumps(index.search(args.query, limit=args.limit), indent=2, sort_keys=True))
        finally:
            index.close()
        return

    if args.command == "search-shards":
        hits = search_shards(
            discover_shard_paths(Path(args.shard_dir)),
            str(args.query),
            limit=int(args.limit),
            year_min=int(args.year_min),
            year_max=int(args.year_max),
        )
        print(json.dumps(hits, indent=2, sort_keys=True))
        return

    if args.command == "explain":
        index = FullRawFtsIndex(Path(args.index_path))
        try:
            print(json.dumps(index.explain_query(args.query), indent=2, sort_keys=True))
        finally:
            index.close()
        return

    if args.command == "stats":
        files_total = 0
        manifest_path = Path(args.manifest)
        if manifest_path.exists():
            files_total = len(load_or_build_manifest(manifest_path))
        index = FullRawFtsIndex(Path(args.index_path))
        try:
            index.initialize()
            print(json.dumps(asdict(index.stats(files_total=files_total)), sort_keys=True))
        finally:
            index.close()

    if args.command == "stats-shards":
        files_total = 0
        manifest_path = Path(args.manifest)
        if manifest_path.exists():
            files_total = len(load_or_build_manifest(manifest_path))
        shard_paths = discover_shard_paths(Path(args.shard_dir))
        stats_payload = asdict(aggregate_shard_stats(shard_paths, files_total=files_total))
        stats_payload["shards"] = len(shard_paths)
        stats_payload["shard_dir"] = str(args.shard_dir)
        print(json.dumps(stats_payload, sort_keys=True))
        return

    if args.command == "backfill-shard-profiles":
        profile_result = backfill_shard_profiles(
            Path(args.shard_dir),
            trust_filenames=not bool(args.verify_sqlite),
            max_shards=args.max_shards,
            force=bool(args.force),
            dry_run=bool(args.dry_run),
            upload_remote=str(args.upload_remote),
            rclone_bin=str(args.rclone_bin),
            progress_interval=max(0, int(args.progress_interval)),
        )
        print(json.dumps(profile_result, sort_keys=True))
        if profile_result["errors"]:
            raise SystemExit(2)
        return


def _row_to_hit(row: sqlite3.Row) -> dict[str, object]:
    rank = _float_or_none(row["rank"]) or 0.0
    return {
        "title": _clean(row["title"]),
        "abstract": _clean(row["abstract"]),
        "doi": _clean(row["doi"]),
        "pmid": _clean(row["pmid"]),
        "pmcid": _clean(row["pmcid"]),
        "openalex_id": _clean(row["openalex_id"]),
        "semantic_scholar_id": _clean(row["semantic_scholar_id"]),
        "year": _int_or_none(row["year"]),
        "journal": _clean(row["journal"]),
        "source": _clean(row["source"]),
        "url": _clean(row["url"]),
        "cited_by_count": _int_or_none(row["cited_by_count"]),
        "score": round(-rank, 6),
    }


def _hit_score(hit: dict[str, object]) -> float:
    return _float_or_none(hit.get("score")) or 0.0


def _sweep_cache_key(
    query: str,
    *,
    limit: int,
    year_min: int,
    year_max: int,
    rank_mode: str,
    sweep_shard_limit: int,
) -> str:
    payload = json.dumps(
        {
            "query": query,
            "limit": limit,
            "year_min": year_min,
            "year_max": year_max,
            "rank_mode": _rank_mode(rank_mode),
            "sweep_shard_limit": sweep_shard_limit,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sweep_cache_path(cache_dir: Path | None, key: str) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / f"{key}.json"


def _load_sweep_cache(path: Path, *, ttl_seconds: float) -> SweepCacheEntry | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    created_at = _float_or_none(payload.get("created_at")) or 0.0
    if ttl_seconds > 0 and time.time() - created_at > ttl_seconds:
        return None
    hits = payload.get("hits")
    receipt = payload.get("receipt")
    if not isinstance(hits, list) or not isinstance(receipt, dict):
        return None
    return SweepCacheEntry(
        created_at=created_at,
        hits=[hit for hit in hits if isinstance(hit, dict)],
        receipt=receipt,
    )


def _write_sweep_cache(path: Path, entry: SweepCacheEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_file(path, {
        "created_at": entry.created_at,
        "hits": entry.hits,
        "receipt": entry.receipt,
    })


def _free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return int(usage.free)


def _fts_terms(query: str) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in _WORD.findall(query.casefold()):
        if len(token) <= 1 or token.isdigit() or token in _STOP or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return tuple(terms)


def _first_fts_term(value: str) -> str:
    terms = _fts_terms(value)
    return terms[0] if terms else ""


def _profile_topic_terms(rows: Iterable[sqlite3.Row], *, limit: int) -> tuple[str, ...]:
    counts: Counter[str] = Counter()
    for row in rows:
        text = f"{row['title'] or ''} {row['abstract'] or ''}"
        counts.update(_fts_terms(text))
    return tuple(term for term, _count in counts.most_common(max(1, limit)))


def _sample_row_starts(total_rows: int, *, buckets: int = 5) -> tuple[int, ...]:
    if total_rows <= 1:
        return (1,)
    bucket_count = max(2, buckets)
    starts = {1, total_rows}
    for index in range(1, bucket_count - 1):
        starts.add(max(1, round(total_rows * index / (bucket_count - 1))))
    return tuple(sorted(starts))


def _search_order_by(rank_mode: str) -> str:
    mode = _rank_mode(rank_mode)
    if mode in {"citation", "citation_heavy", "cited"}:
        return "COALESCE(p.cited_by_count, 0) DESC, rank ASC"
    if mode in {"recency", "recent"}:
        return "COALESCE(p.year, 0) DESC, rank ASC, COALESCE(p.cited_by_count, 0) DESC"
    return "rank ASC, COALESCE(p.cited_by_count, 0) DESC"


def _rank_mode(value: object) -> str:
    mode = str(value or "relevance").casefold()
    if mode in {"citation", "citation_heavy", "cited"}:
        return "citation"
    if mode in {"recency", "recent"}:
        return "recency"
    return "relevance"


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value if str(item))


def _unique_terms(values: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for term in _fts_terms(value):
            if term in seen:
                continue
            seen.add(term)
            out.append(term)
    return tuple(out)


def _fts_match_query(groups: tuple[tuple[str, ...], ...]) -> str:
    clauses: list[str] = []
    for group in groups:
        clean_group = _unique_terms(group)
        if not clean_group:
            continue
        if len(clean_group) == 1:
            clauses.append(f'"{clean_group[0]}"')
            continue
        clauses.append("(" + " OR ".join(f'"{term}"' for term in clean_group) + ")")
    return " AND ".join(clauses)


def _dedupe_key(hit: dict[str, object]) -> str:
    for key in ("doi", "pmid", "pmcid", "semantic_scholar_id", "openalex_id"):
        value = _clean(hit.get(key))
        if value:
            return f"{key}:{value.casefold()}"
    return f"title:{_clean(hit.get('title')).casefold()}:{_int_or_none(hit.get('year')) or ''}"


def _clean(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _int_or_none(value: object) -> int | None:
    if not isinstance(value, int | float | str | bytes | bytearray):
        return None
    try:
        return int(value) if value not in {"", None} else None
    except (TypeError, ValueError):
        return None


def _positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _float_or_none(value: object) -> float | None:
    if not isinstance(value, int | float | str | bytes | bytearray):
        return None
    try:
        return float(value) if value not in {"", None} else None
    except (TypeError, ValueError):
        return None


def _write_json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, object]) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


if __name__ == "__main__":
    main()
