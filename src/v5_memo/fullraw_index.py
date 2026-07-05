"""Persistent FTS index for the full raw corpus.

The cold scanner proves the raw archive is reachable, but it returns early hits
in file order. This module builds a durable SQLite FTS5 index so memo retrieval
can rank by relevance instead of archive order.
"""
from __future__ import annotations

import argparse
import fcntl
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
from contextlib import suppress
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from v5_memo.fullraw.http import write_json
from v5_memo.fullraw_service import RawFile, iter_raw_file_hits, load_or_build_manifest

_write_json = write_json
_WORD = re.compile(r"[A-Za-z0-9]+")
_STOP = frozenset(
    (  # noqa: SIM905
        "about above abstract after also among and are because been both but can could different "
        "does during for from had has have high important international into its may most no not "
        "off or our out over per such than that the their these this those through was were with "
        "within without would all any between each other should there using which while who whose "
        "will you your of in by to is be at on as an analysis associated based better compared "
        "data effect effects first form found here it method methods journal model models more "
        "name new none one only pages paper provide provided research related result results show "
        "showed shown significant significantly studies study two use used we work volume dan de "
        "del di es findings general however ini many merupakan para process que role specific "
        "therefore un understanding well yang adalah analisis dengan dilakukan hasil including "
        "made el en la los metode pada penelitian"
    ).split()
)
_BACKEND = "researka-fullraw-indexed-fts5"
_ALPHA_SWEEP_TERMS = frozenset({
    "attenuate", "attenuated", "attenuates", "blunt", "blunted", "blunts",
    "endpoint", "failed", "impair", "impaired", "impairs", "null", "primary",
    "protocol", "reduced", "reduces", "replication", "subgroup", "timing",
})
_SWEEP_QUERY_POPULATION_FILLER_TERMS = frozenset({
    "adult", "adults", "aged", "elderly", "human", "humans", "older", "senior", "seniors",
})
_SWEEP_QUERY_FILLER_TERMS = frozenset({
    *_SWEEP_QUERY_POPULATION_FILLER_TERMS,
    "function", "muscle", "performance", "strength",
})
_FULLRAW_LEGACY_PREFIX = "V5_MEMO_FULL_RAW_"
_FULLRAW_GENERIC_PREFIX = "RESEARKA_FULLRAW_"
_FULLRAW_SPECIAL_ALIASES = {
    "V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL": ("RESEARKA_FULLRAW_SEARCH_URL",),
    "V5_MEMO_FULL_RAW_CORPUS_TOKEN": ("RESEARKA_FULLRAW_TOKEN",),
    "V5_MEMO_FULL_RAW_INDEX_TOKEN": ("RESEARKA_FULLRAW_INDEX_TOKEN", "RESEARKA_FULLRAW_TOKEN"),
    "V5_MEMO_FULL_RAW_TOKEN": ("RESEARKA_FULLRAW_TOKEN",),
}


def _fullraw_env_names(name: str) -> tuple[str, ...]:
    if not name.startswith(_FULLRAW_LEGACY_PREFIX):
        return (name,)
    suffix = name.removeprefix(_FULLRAW_LEGACY_PREFIX)
    candidates = (*_FULLRAW_SPECIAL_ALIASES.get(name, ()), f"{_FULLRAW_GENERIC_PREFIX}{suffix}", name)
    return tuple(dict.fromkeys(candidates))


def _fullraw_env(name: str, default: str = "") -> str:
    for candidate in _fullraw_env_names(name):
        value = os.environ.get(candidate)
        if value is not None and value != "":
            return value
    return default


def _shard_local_cache_dir() -> Path | None:
    cache_dir_config = _fullraw_env("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_DIR", "").strip()
    return Path(cache_dir_config) if cache_dir_config else None


def _shard_local_cache_max_bytes(cache_dir: Path | None = None) -> int | None:
    raw = _fullraw_env("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES", "").strip()
    if raw.casefold() not in {"auto", "dynamic"}:
        return _positive_int_env("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES")
    target_dir = cache_dir or _shard_local_cache_dir()
    if target_dir is None:
        return None
    probe = target_dir
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    cache_bytes = 0
    if target_dir.exists():
        cache_bytes = sum(
            path.stat().st_size
            for pattern in ("*.sqlite", "*.sqlite-wal", ".*.sqlite.tmp.*")
            for path in target_dir.glob(pattern)
            if path.is_file()
        )
    min_free_bytes = _positive_int_env("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MIN_FREE_BYTES")
    if min_free_bytes is None:
        min_free_gb = _float_or_none(
            _fullraw_env("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MIN_FREE_GB", "")
        )
        min_free_bytes = int(min_free_gb * 1024 * 1024 * 1024) if min_free_gb else usage.total // 20
    return max(0, cache_bytes + usage.free - min_free_bytes)


def _shard_local_cache_health(*, include_dynamic_budget: bool = True) -> dict[str, object]:
    cache_dir = _shard_local_cache_dir()
    raw_max_bytes = _fullraw_env("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES", "")
    health: dict[str, object] = {
        "dir": str(cache_dir) if cache_dir is not None else "",
        "exists": bool(cache_dir and cache_dir.exists()),
        "is_mount": bool(cache_dir and os.path.ismount(cache_dir)),
    }
    if include_dynamic_budget:
        health["max_bytes"] = _shard_local_cache_max_bytes(cache_dir)
    elif raw_max_bytes.strip().casefold() in {"auto", "dynamic"}:
        health["max_bytes_config"] = raw_max_bytes
    else:
        max_bytes = _positive_int_env("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES")
        if max_bytes is not None:
            health["max_bytes"] = max_bytes
    return health


_FULL_COVERAGE_PREFIX_SHARDS = max(1, int(_fullraw_env("V5_MEMO_FULL_RAW_SEARCH_PREFIX_SHARDS", "32")))
_SWEEP_STRATEGY = "profile_relaxed_v11"
_SWEEP_MIN_RESULT_LIMIT = 10
_SHARD_LOCAL_CACHE_LOCK = threading.RLock()
_SHARD_LOCAL_CACHE_IN_PROGRESS: set[Path] = set()
_SHARD_LOCAL_CACHE_RESERVED_BYTES: dict[Path, int] = {}
_DEFAULT_TERM_MAP = (
    ("management", ("management", "manager", "managers", "managerial")),
    ("forecast", ("forecast", "forecasts", "forecasting", "guidance", "estimate", "estimates")),
    ("disclosure", ("disclosure", "disclosures", "disclose", "discloses", "disclosed")),
    ("earnings", ("earnings", "income", "profit", "profits")),
    ("analyst", ("analyst", "analysts", "analysis")),
    ("exercise", ("exercise", "exercises", "training", "trained")),
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


@dataclass(frozen=True, slots=True)
class SweepJob:
    key: str
    query: str
    limit: int
    year_min: int
    year_max: int
    rank_mode: str
    catalog: list[ShardCatalogEntry]
    priority: bool = False


@dataclass(frozen=True, slots=True)
class ShardCacheWarmResult:
    selected_shards: int
    target_ready: int
    ready_shards: int
    warmed_shards: int
    failed_shards: int
    stopped_for_target: bool
    stopped_for_time: bool
    elapsed_seconds: float
    bytes_ready: int
    sources_selected: dict[str, int]
    sources_ready: dict[str, int]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SweepSearchPass:
    role: str
    query: str
    rank_mode: str


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
            rank_mode = _rank_mode(rank_mode)
            order_by = _search_order_by(rank_mode)
            row_limit = max(1, min(limit, 200))
            if rank_mode == "relevance":
                row_limit = min(200, max(row_limit, limit * 5, 50))
            if timeout_seconds is not None and timeout_seconds > 0:
                deadline = time.monotonic() + timeout_seconds
                self._conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
            try:
                rows = self._conn.execute(
                    f"""
                    SELECT
                      p.title, p.abstract, p.doi, p.pmid, p.pmcid,
                      p.openalex_id, p.semantic_scholar_id, p.year,
                      p.journal, p.source, p.url, p.cited_by_count,
                      bm25(paper_fts, 8.0, 3.0, 1.0) AS rank
                    FROM paper_fts
                    JOIN papers p ON p.id = paper_fts.rowid
                    WHERE paper_fts MATCH ?
                      AND (p.year IS NULL OR (p.year >= ? AND p.year <= ?))
                    ORDER BY {order_by}
                    LIMIT ?
                    """,
                    (match_query, year_min, year_max, row_limit),
                ).fetchall()
            finally:
                if timeout_seconds is not None and timeout_seconds > 0:
                    self._conn.set_progress_handler(None, 0)
        hits = [_row_to_hit(row) for row in rows]
        if rank_mode == "relevance":
            hits = [_with_query_fit_score(hit, terms) for hit in hits]
            return _rank_limited_hits(hits, limit=limit)
        return hits[: max(1, min(limit, 200))]

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
    batch_id_offset: int = 0,
) -> list[ShardBatchResult]:
    """Build local shard batches, upload completed batches, then free local disk."""
    if not upload_remote.strip():
        raise ValueError("upload remote is required for build-upload-shards")
    selected_files = files[:max_files] if max_files is not None else files
    results: list[ShardBatchResult] = []
    for batch_index, start in enumerate(range(0, len(selected_files), max(1, batch_files))):
        batch_id = batch_id_offset + batch_index
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


def search_shard_entries_with_receipt(
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
) -> tuple[list[dict[str, object]], dict[str, object]]:
    selected = select_search_shard_entries(entries, query=query)
    hits, completed_paths, timed_out, result_metrics = _search_shard_paths_with_paths_and_receipt(
        [entry.path for entry in selected],
        query,
        limit=limit,
        year_min=year_min,
        year_max=year_max,
        rank_mode=rank_mode,
        workers=workers,
        timeout_seconds=timeout_seconds,
        shard_timeout_seconds=shard_timeout_seconds,
    )
    completed = set(completed_paths)
    searched_entries = [entry for entry in selected if entry.path in completed]
    receipt = shard_coverage_receipt(entries, searched_entries)
    receipt["foreground_selected_shards"] = len(selected)
    receipt["foreground_completed_shards"] = len(searched_entries)
    receipt["foreground_timed_out"] = timed_out
    receipt.update(result_metrics)
    return hits, receipt


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
    hits, _completed_paths, _timed_out, _metrics = _search_shard_paths_with_paths_and_receipt(
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
    hits, completed_paths, timed_out, _metrics = _search_shard_paths_with_paths_and_receipt(
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
    return hits, completed_paths, timed_out


def _search_shard_paths_with_paths_and_receipt(
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
) -> tuple[list[dict[str, object]], list[Path], bool, dict[str, object]]:
    hit_groups: list[list[dict[str, object]]] = []
    completed_paths: list[Path] = []
    timed_out = False
    worker_count = workers if workers is not None else int(_fullraw_env("V5_MEMO_FULL_RAW_SEARCH_WORKERS", "8"))
    worker_count = max(1, min(worker_count, len(paths) or 1))
    search_paths = (
        paths[: min(len(paths), max(worker_count * 4, _FULL_COVERAGE_PREFIX_SHARDS))]
        if timeout_seconds
        else paths
    )
    deadline = time.monotonic() + timeout_seconds if timeout_seconds else None
    per_shard_timeout = shard_timeout_seconds
    start = 0
    while start < len(search_paths):
        if deadline is not None and time.monotonic() >= deadline:
            timed_out = True
            break
        batch = _cache_fit_path_batch(search_paths, start=start, worker_count=worker_count)
        start += len(batch)
        preserve_paths = {
            cache_path
            for path in batch
            if (cache_path := _shard_cache_path(path)) is not None
        }
        pool = ThreadPoolExecutor(max_workers=max(1, min(worker_count, len(batch))))
        timed_out_batch = False
        try:
            futures = {
                pool.submit(
                    _materialize_and_search_one_shard,
                    path,
                    query,
                    limit,
                    year_min,
                    year_max,
                    rank_mode,
                    per_shard_timeout,
                    preserve_paths,
                ): path
                for path in batch
            }
            remaining = None if deadline is None else max(0.05, deadline - time.monotonic())
            for future in as_completed(futures, timeout=remaining):
                path = futures[future]
                try:
                    hits = future.result()
                except (OSError, TimeoutError, sqlite3.Error, subprocess.SubprocessError):
                    continue
                completed_paths.append(path)
                hit_groups.append(hits)
        except FuturesTimeoutError:
            timed_out_batch = True
            timed_out = True
            break
        finally:
            pool.shutdown(wait=not timed_out_batch, cancel_futures=True)
    hits, metrics = _merge_hit_groups_with_receipt(hit_groups, limit=limit)
    return hits, completed_paths, timed_out, metrics


def _materialize_and_search_one_shard(
    path: Path,
    query: str,
    limit: int,
    year_min: int,
    year_max: int,
    rank_mode: str,
    timeout_seconds: float | None,
    preserve: set[Path],
) -> list[dict[str, object]]:
    search_path = _materialized_shard_path(path, preserve=preserve, populate=True)
    return _search_one_shard_for_pool(search_path, query, limit, year_min, year_max, rank_mode, timeout_seconds)


def _cache_fit_path_batch(paths: list[Path], *, start: int, worker_count: int) -> list[Path]:
    batch = paths[start:start + worker_count]
    max_cache_bytes = _shard_local_cache_max_bytes()
    if max_cache_bytes is None:
        return batch
    if max_cache_bytes <= 0:
        return batch
    max_inflight = _positive_int_env("V5_MEMO_FULL_RAW_SWEEP_MAX_INFLIGHT") or 1
    budget = max(1, max_cache_bytes // _sweep_cache_inflight_lanes(max_inflight))
    out: list[Path] = []
    total = 0
    for path in batch:
        try:
            size = max(0, path.stat().st_size)
        except OSError:
            size = 0
        if out and total + size > budget:
            break
        out.append(path)
        total += size
    return out or batch[:1]


def _merge_hit_groups(
    hit_groups: Iterable[Iterable[dict[str, object]]],
    *,
    limit: int,
) -> list[dict[str, object]]:
    hits, _metrics = _merge_hit_groups_with_receipt(hit_groups, limit=limit)
    return hits


def _merge_hit_groups_with_receipt(
    hit_groups: Iterable[Iterable[dict[str, object]]],
    *,
    limit: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    raw_count = 0
    for hits in hit_groups:
        for hit in hits:
            raw_count += 1
            key = _dedupe_key(hit)
            existing = merged.get(key)
            if existing is None or _hit_score(hit) > _hit_score(existing):
                merged[key] = hit
    ranked = _rank_limited_hits(merged.values(), limit=limit)
    return ranked, _hit_diversity_receipt(
        raw_result_count=raw_count,
        unique_result_count=len(merged),
        returned_hits=ranked,
    )


def _hit_diversity_receipt(
    *,
    raw_result_count: int,
    unique_result_count: int,
    returned_hits: list[dict[str, object]],
) -> dict[str, object]:
    duplicate_count = max(0, raw_result_count - unique_result_count)
    citation_counts = [_int_or_none(hit.get("cited_by_count")) or 0 for hit in returned_hits]
    bucket_counts: Counter[str] = Counter(_citation_bucket(count) for count in citation_counts)
    return {
        "result_count_raw": raw_result_count,
        "result_count_unique": unique_result_count,
        "result_count_returned": len(returned_hits),
        "result_duplicate_count": duplicate_count,
        "result_duplicate_rate": round(duplicate_count / raw_result_count, 4) if raw_result_count else 0.0,
        "result_cited_by_range": (
            {"min": min(citation_counts), "max": max(citation_counts)}
            if citation_counts
            else {"min": 0, "max": 0}
        ),
        "result_citation_bucket_counts": dict(sorted(bucket_counts.items())),
        "result_citation_diversity": sum(1 for count in bucket_counts.values() if count > 0),
    }


def _citation_bucket(count: int) -> str:
    if count <= 0:
        return "zero"
    if count < 10:
        return "low"
    if count < 100:
        return "medium"
    if count < 1000:
        return "high"
    return "very_high"


def _rank_limited_hits(
    hits: Iterable[dict[str, object]],
    *,
    limit: int,
) -> list[dict[str, object]]:
    return sorted(
        hits,
        key=lambda hit: (_hit_score(hit), _int_or_none(hit.get("cited_by_count")) or 0),
        reverse=True,
    )[: max(1, min(limit, 200))]


def select_search_shard_paths(paths: list[Path]) -> list[Path]:
    limit = _positive_int_env("V5_MEMO_FULL_RAW_SEARCH_SHARD_LIMIT")
    if limit is None or limit >= len(paths):
        return paths
    order = _fullraw_env("V5_MEMO_FULL_RAW_SEARCH_SHARD_ORDER", "newest").casefold()
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
    min_required = _positive_int_env("V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED")
    if limit is not None and min_required is not None:
        limit = max(limit, min_required)
    if limit is None or limit >= len(entries):
        return _frontload_spread_entries(entries)
    order = _fullraw_env("V5_MEMO_FULL_RAW_SEARCH_SHARD_ORDER", "balanced").casefold()
    if order in {"oldest", "first", "newest", "spread"}:
        paths = select_search_shard_paths([entry.path for entry in entries])
        by_path = {entry.path: entry for entry in entries}
        return [by_path[path] for path in paths if path in by_path]
    selected = _select_balanced_shard_entries(entries, limit, query=query)
    target_ready = min(_FULL_COVERAGE_PREFIX_SHARDS, len(selected))
    return _cache_fit_warm_entries(entries, selected, query=query, target_ready=target_ready)


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
        return _frontload_spread_entries(entries)
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


def _prioritize_sweep_pass_entries(
    entries: list[ShardCatalogEntry],
    prefix_size: int,
    *,
    query: str,
) -> list[ShardCatalogEntry]:
    if prefix_size <= 1 or len(entries) <= 1:
        return entries
    by_source: dict[str, list[ShardCatalogEntry]] = {}
    for entry in entries:
        source = entry.sources[0] if entry.sources else "unknown"
        by_source.setdefault(source, []).append(entry)
    if len(by_source) <= 1:
        return entries
    query_terms = set(_fts_terms(query))

    def candidate_key(entry: ShardCatalogEntry) -> tuple[int, int, int, int, int, str]:
        topic_hits = len(query_terms & set(entry.topic_terms))
        return (
            -topic_hits,
            entry.bytes_used,
            -entry.cited_by_max,
            entry.batch_id,
            entry.shard_id,
            str(entry.path),
        )

    sources = sorted(by_source, key=lambda source: (len(by_source[source]), source))
    ordered_by_source = {source: sorted(by_source[source], key=candidate_key) for source in sources}
    ordered: list[ShardCatalogEntry] = []
    while len(ordered) < len(entries):
        before = len(ordered)
        for source in sources:
            source_entries = ordered_by_source[source]
            if source_entries:
                _extend_unique_entries(ordered, source_entries[:1], len(entries))
                del source_entries[0]
        if len(ordered) == before:
            break
    selected_paths = {entry.path for entry in ordered}
    _extend_unique_entries(ordered, (entry for entry in entries if entry.path not in selected_paths), len(entries))
    return ordered


def _cache_fit_warm_entries(
    entries: list[ShardCatalogEntry],
    selected: list[ShardCatalogEntry],
    *,
    query: str,
    target_ready: int,
) -> list[ShardCatalogEntry]:
    max_cache_bytes = _shard_local_cache_max_bytes()
    if max_cache_bytes is None or max_cache_bytes <= 0 or target_ready <= 0:
        return selected
    target_ready = min(target_ready, len(entries))
    if target_ready <= 0:
        return selected
    query_terms = set(_fts_terms(query))

    def candidate_key(entry: ShardCatalogEntry) -> tuple[int, int, int, int, int, int, str]:
        topic_hits = len(query_terms & set(entry.topic_terms))
        return (
            0 if _cached_materialized_shard_path(entry.path) is not None else 1,
            max(0, entry.bytes_used),
            -topic_hits,
            -entry.cited_by_max,
            entry.batch_id,
            entry.shard_id,
            str(entry.path),
        )

    by_source: dict[str, list[ShardCatalogEntry]] = {}
    for entry in entries:
        source = entry.sources[0] if entry.sources else "unknown"
        by_source.setdefault(source, []).append(entry)
    sources = sorted(by_source, key=lambda source: (len(by_source[source]), source))
    ordered_by_source = {source: sorted(by_source[source], key=candidate_key) for source in sources}
    fit_prefix = sorted(
        (entry for entry in entries if _cached_materialized_shard_path(entry.path) is not None),
        key=candidate_key,
    )
    fit_paths = {entry.path for entry in fit_prefix}
    bytes_used = sum(max(0, entry.bytes_used) for entry in fit_prefix)

    while len(fit_prefix) < target_ready:
        before = len(fit_prefix)
        for source in sources:
            source_entries = ordered_by_source[source]
            while source_entries:
                candidate = source_entries.pop(0)
                if candidate.path in fit_paths:
                    continue
                candidate_bytes = max(0, candidate.bytes_used)
                if _cached_materialized_shard_path(candidate.path) is None and bytes_used + candidate_bytes > max_cache_bytes:
                    continue
                fit_prefix.append(candidate)
                fit_paths.add(candidate.path)
                bytes_used += candidate_bytes
                break
            if len(fit_prefix) >= target_ready:
                break
        if len(fit_prefix) == before:
            break

    if len(fit_prefix) < target_ready:
        remaining = sorted(
            (entry for entry in entries if entry.path not in fit_paths),
            key=candidate_key,
        )
        for candidate in remaining:
            candidate_bytes = max(0, candidate.bytes_used)
            if bytes_used + candidate_bytes > max_cache_bytes:
                continue
            fit_prefix.append(candidate)
            fit_paths.add(candidate.path)
            bytes_used += candidate_bytes
            if len(fit_prefix) >= target_ready:
                break

    if not fit_prefix:
        return selected
    ordered = list(fit_prefix)
    selected_paths = {entry.path for entry in ordered}
    remaining_selected = sorted(
        (entry for entry in selected if entry.path not in selected_paths),
        key=candidate_key,
    )
    _extend_unique_entries(ordered, remaining_selected, max(len(selected), len(ordered)))
    _extend_unique_entries(ordered, entries, max(len(selected), len(ordered)))
    return ordered


def _cache_reuse_sweep_entries(entries: list[ShardCatalogEntry]) -> list[ShardCatalogEntry]:
    def candidate_key(entry: ShardCatalogEntry) -> tuple[int, int, int, int, str]:
        return (
            0 if _cached_materialized_shard_path(entry.path) is not None else 1,
            max(0, entry.bytes_used),
            entry.batch_id,
            entry.shard_id,
            str(entry.path),
        )

    return sorted(entries, key=candidate_key)


def _profile_relaxed_sweep_query(
    query: str,
    entries: list[ShardCatalogEntry],
    *,
    max_terms: int = 3,
) -> str:
    raw_terms = _fts_terms(query)
    filler_terms = _SWEEP_QUERY_FILLER_TERMS
    if any(term in _ALPHA_SWEEP_TERMS for term in raw_terms):
        filler_terms = _SWEEP_QUERY_POPULATION_FILLER_TERMS
    filtered_terms = tuple(term for term in raw_terms if term not in filler_terms)
    terms = filtered_terms if len(filtered_terms) >= 2 else raw_terms
    if len(terms) <= max_terms:
        return " ".join(terms)
    profile_counts: Counter[str] = Counter()
    aliases = {term: _term_aliases(term) for term in terms}
    for entry in entries:
        topic_terms = set(entry.topic_terms)
        if not topic_terms:
            continue
        for term, term_aliases in aliases.items():
            if topic_terms & term_aliases:
                profile_counts[term] += 1
    positive_terms = [term for term in terms if profile_counts[term] > 0]
    if len(positive_terms) < 2:
        positive_terms = [term for term in terms if len(aliases[term]) > 1]
    if len(positive_terms) < 2:
        positive_terms = list(terms)
    ranked = sorted(positive_terms, key=lambda term: (-profile_counts[term], terms.index(term)))
    candidate_terms = set(ranked[: max(max_terms, min(len(ranked), max_terms + 2))])
    first = terms[0]
    if len(first) > 2:
        candidate_terms.add(first)
    rare_terms = [term for term in terms[1:] if profile_counts[term] == 0 and len(term) > 3]
    if rare_terms:
        candidate_terms.add(rare_terms[0])
    ordered = [term for term in terms if term in candidate_terms]
    if len(ordered) > max_terms:
        alpha_terms = [term for term in terms if term in _ALPHA_SWEEP_TERMS and term in ordered]
        if "resistance" in ordered and "training" in terms and "water" not in terms:
            alpha_terms = ["resistance", *alpha_terms]
        if alpha_terms:
            protected = list(dict.fromkeys((first, alpha_terms[0])))
            fill = _spread_terms([term for term in ordered if term not in protected], max_terms - len(protected))
            ordered = [term for term in terms if term in set((*protected, *fill))]
        else:
            ordered = _spread_terms(ordered, max_terms)
    return " ".join(ordered) if len(ordered) >= 2 else " ".join(terms)


def _sweep_search_passes(
    query: str,
    entries: list[ShardCatalogEntry],
    *,
    rank_mode: str,
) -> tuple[SweepSearchPass, ...]:
    focused = _profile_relaxed_sweep_query(query, entries)
    requested_rank_mode = _rank_mode(rank_mode)
    passes = (
        SweepSearchPass("focused", focused, requested_rank_mode),
        SweepSearchPass("citation_heavy", focused, "citation"),
        SweepSearchPass("recency", focused, "recency"),
    )
    return tuple(pass_item for pass_item in passes if _fts_terms(pass_item.query))


def _sweep_cache_query(
    query: str,
    entries: list[ShardCatalogEntry],
    *,
    sweep_shard_limit: int,
    rank_mode: str,
) -> str:
    if entries and sweep_shard_limit >= len(entries):
        passes = _sweep_search_passes(query, entries, rank_mode=rank_mode)
        if passes:
            return passes[0].query
    return query


def _term_aliases(term: str) -> set[str]:
    aliases = {term}
    for canonical, expansions in _DEFAULT_TERM_MAP:
        if term == canonical or term in expansions:
            aliases.add(canonical)
            aliases.update(expansions)
    return aliases


def _spread_terms(terms: list[str], count: int) -> list[str]:
    if count <= 0 or not terms:
        return []
    if count >= len(terms):
        return list(terms)
    if count == 1:
        return [terms[len(terms) // 2]]
    step = (len(terms) - 1) / (count - 1)
    out: list[str] = []
    seen: set[str] = set()
    for index in range(count):
        term = terms[round(index * step)]
        if term not in seen:
            out.append(term)
            seen.add(term)
    return out


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
    if query_terms and not any(query_terms & set(entry.topic_terms) for entry in entries):
        return _spread_entries(rotated, limit)
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


def _frontload_spread_entries(entries: list[ShardCatalogEntry]) -> list[ShardCatalogEntry]:
    prefix_count = min(len(entries), _FULL_COVERAGE_PREFIX_SHARDS)
    out: list[ShardCatalogEntry] = []
    _extend_unique_entries(out, _spread_entries(entries, prefix_count), len(entries))
    _extend_unique_entries(out, entries, len(entries))
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
    index = FullRawFtsIndex(_materialized_shard_path(path), read_only=True)
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


def _search_one_shard_for_pool(
    path: Path,
    query: str,
    limit: int,
    year_min: int,
    year_max: int,
    rank_mode: str,
    timeout_seconds: float | None = None,
) -> list[dict[str, object]]:
    if _fullraw_env("V5_MEMO_FULL_RAW_SEARCH_ISOLATED", "").casefold() in {"1", "true", "yes"}:
        return _search_one_shard_isolated(
            path,
            query,
            limit,
            year_min,
            year_max,
            rank_mode,
            timeout_seconds,
        )
    return _search_one_shard(path, query, limit, year_min, year_max, rank_mode, timeout_seconds)


def _search_one_shard_isolated(
    path: Path,
    query: str,
    limit: int,
    year_min: int,
    year_max: int,
    rank_mode: str,
    timeout_seconds: float | None = None,
) -> list[dict[str, object]]:
    child_timeout = max(0.1, timeout_seconds or 30.0)
    code = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from v5_memo.fullraw_index import _search_one_shard\n"
        "path=Path(sys.argv[1])\n"
        "query=sys.argv[2]\n"
        "limit=int(sys.argv[3])\n"
        "year_min=int(sys.argv[4])\n"
        "year_max=int(sys.argv[5])\n"
        "rank_mode=sys.argv[6]\n"
        "timeout=float(sys.argv[7]) if sys.argv[7] else None\n"
        "print(json.dumps(_search_one_shard(path, query, limit, year_min, year_max, rank_mode, timeout)))\n"
    )
    env = os.environ.copy()
    env["V5_MEMO_FULL_RAW_SEARCH_ISOLATED"] = "0"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            code,
            str(path),
            query,
            str(limit),
            str(year_min),
            str(year_max),
            rank_mode,
            "" if timeout_seconds is None else str(timeout_seconds),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        stdout, stderr = proc.communicate(timeout=child_timeout + 1.0)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        # FUSE-backed SQLite reads can sit in disk wait after SIGKILL. Do not
        # let one stuck child pin the sweep lane forever; the caller records
        # this shard as timed out/deferred and moves on.
        with suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=1.0)
        raise TimeoutError(f"isolated shard search timed out: {path}") from exc
    if proc.returncode != 0:
        raise sqlite3.Error((stderr or stdout or "isolated shard search failed").strip()[:500])
    payload = json.loads(stdout)
    if not isinstance(payload, list):
        raise sqlite3.Error("isolated shard search returned non-list payload")
    return [hit for hit in payload if isinstance(hit, dict)]


def _materialized_shard_path(
    path: Path,
    *,
    preserve: set[Path] | None = None,
    populate: bool = False,
) -> Path:
    cache_path = _shard_cache_path(path)
    if cache_path is None:
        return path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if populate:
        max_cache_bytes = _shard_local_cache_max_bytes(cache_path.parent)
        if max_cache_bytes is not None and max_cache_bytes <= 0:
            return path
    else:
        max_cache_bytes = None
    source_stat = path.stat()
    with _SHARD_LOCAL_CACHE_LOCK:
        if cache_path.exists() and cache_path.stat().st_size == source_stat.st_size:
            os.utime(cache_path, None)
            return cache_path
    if not populate:
        return path
    populate_limit = max_cache_bytes
    if per_worker_bytes := _sweep_worker_cache_bytes():
        populate_limit = per_worker_bytes if populate_limit is None else min(populate_limit, per_worker_bytes)
    if populate_limit is not None and source_stat.st_size > populate_limit:
        return path
    while True:
        with _SHARD_LOCAL_CACHE_LOCK:
            if cache_path.exists() and cache_path.stat().st_size == source_stat.st_size:
                os.utime(cache_path, None)
                return cache_path
            if cache_path not in _SHARD_LOCAL_CACHE_IN_PROGRESS:
                _SHARD_LOCAL_CACHE_IN_PROGRESS.add(cache_path)
                break
        time.sleep(0.05)
    tmp_path = cache_path.with_name(f".{cache_path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    base_preserve = preserve or set()
    with _SHARD_LOCAL_CACHE_LOCK:
        reserved_bytes = sum(_SHARD_LOCAL_CACHE_RESERVED_BYTES.values())
        _evict_shard_cache(
            cache_path.parent,
            required_bytes=reserved_bytes + source_stat.st_size,
            keep=cache_path,
            preserve=base_preserve | _SHARD_LOCAL_CACHE_IN_PROGRESS,
        )
        if max_cache_bytes is not None:
            cache_bytes = _shard_cache_used_bytes(cache_path.parent)
            if cache_bytes + reserved_bytes + source_stat.st_size > max_cache_bytes:
                _SHARD_LOCAL_CACHE_IN_PROGRESS.discard(cache_path)
                return path
        _SHARD_LOCAL_CACHE_RESERVED_BYTES[tmp_path] = source_stat.st_size
        _SHARD_LOCAL_CACHE_IN_PROGRESS.add(tmp_path)
        reserved_bytes = sum(_SHARD_LOCAL_CACHE_RESERVED_BYTES.values())
        _evict_shard_cache(
            cache_path.parent,
            required_bytes=reserved_bytes,
            keep=cache_path,
            preserve=base_preserve | _SHARD_LOCAL_CACHE_IN_PROGRESS,
        )
    try:
        shutil.copy2(path, tmp_path)
        with _SHARD_LOCAL_CACHE_LOCK:
            if cache_path.exists() and cache_path.stat().st_size == source_stat.st_size:
                os.utime(cache_path, None)
                return cache_path
            os.replace(tmp_path, cache_path)
            os.utime(cache_path, None)
            _evict_shard_cache(
                cache_path.parent,
                required_bytes=0,
                keep=cache_path,
                preserve=base_preserve | _SHARD_LOCAL_CACHE_IN_PROGRESS,
            )
            return cache_path
    finally:
        with _SHARD_LOCAL_CACHE_LOCK:
            _SHARD_LOCAL_CACHE_IN_PROGRESS.discard(cache_path)
            _SHARD_LOCAL_CACHE_IN_PROGRESS.discard(tmp_path)
            _SHARD_LOCAL_CACHE_RESERVED_BYTES.pop(tmp_path, None)
        tmp_path.unlink(missing_ok=True)


def _shard_cache_path(path: Path) -> Path | None:
    cache_dir = _shard_local_cache_dir()
    if cache_dir is None:
        return None
    path_abs = path if path.is_absolute() else path.absolute()
    cache_dir_abs = cache_dir if cache_dir.is_absolute() else cache_dir.absolute()
    try:
        if os.path.commonpath((str(path_abs), str(cache_dir_abs))) == str(cache_dir_abs):
            return None
    except ValueError:
        pass
    cache_name = f"{hashlib.sha256(str(path).encode()).hexdigest()[:16]}-{path.name}"
    return cache_dir / cache_name


def _cached_materialized_shard_path(path: Path) -> Path | None:
    cache_path = _shard_cache_path(path)
    if cache_path is None:
        return path if path.exists() else None
    try:
        source_size = path.stat().st_size
        if cache_path.exists() and cache_path.stat().st_size == source_size:
            return cache_path
    except OSError:
        return None
    return None


def _cache_tmp_owner_alive(path: Path) -> bool:
    try:
        pid = int(path.name.rsplit(".tmp.", 1)[1].split(".", 1)[0])
    except (IndexError, ValueError):
        return True
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _evict_shard_cache(
    cache_dir: Path,
    *,
    required_bytes: int,
    keep: Path,
    preserve: set[Path] | None = None,
) -> None:
    max_bytes = _shard_local_cache_max_bytes(cache_dir)
    if max_bytes is None or max_bytes <= 0:
        return
    preserved = {path.resolve() for path in (preserve or set())}
    tmp_ttl_seconds = _float_or_none(
        _fullraw_env("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_TMP_TTL_SECONDS", "3600")
    ) or 3600.0
    now = time.time()
    entries = []
    for pattern in ("*.sqlite", "*.sqlite-wal", ".*.sqlite.tmp.*"):
        for path in cache_dir.glob(pattern):
            if not path.is_file() or path == keep or path.resolve() in preserved:
                continue
            if (
                ".sqlite.tmp." in path.name
                and now - path.stat().st_mtime < tmp_ttl_seconds
                and _cache_tmp_owner_alive(path)
            ):
                continue
            entries.append(path)
    total = sum(path.stat().st_size for path in entries)
    if keep.exists():
        total += keep.stat().st_size
    for path in preserved:
        if path != keep.resolve() and path.exists():
            total += path.stat().st_size
    target = max(0, max_bytes - max(0, required_bytes))
    for path in sorted(entries, key=lambda item: item.stat().st_mtime):
        if total <= target:
            break
        try:
            size = path.stat().st_size
            path.unlink()
            total -= size
        except OSError:
            continue


def _shard_cache_used_bytes(cache_dir: Path) -> int:
    total = 0
    for pattern in ("*.sqlite", "*.sqlite-wal", ".*.sqlite.tmp.*"):
        for path in cache_dir.glob(pattern):
            if not path.is_file():
                continue
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def warm_shard_cache(
    entries: list[ShardCatalogEntry],
    *,
    query: str,
    sweep_shard_limit: int,
    pass_shard_limit: int,
    target_ready: int,
    max_shards: int | None = None,
    max_seconds: float | None = None,
    progress_interval: int = 0,
) -> ShardCacheWarmResult:
    started = time.monotonic()
    selected = select_sweep_shard_entries(entries, query=query, limit=max(1, sweep_shard_limit))
    selected = _prioritize_sweep_pass_entries(
        selected,
        max(1, min(pass_shard_limit, len(selected) or 1)),
        query=query,
    )
    target_ready = max(0, min(target_ready, len(selected)))
    selected = _cache_fit_warm_entries(
        entries,
        selected,
        query=query,
        target_ready=target_ready,
    )
    max_shards = max_shards if max_shards is None else max(0, max_shards)
    max_seconds = max_seconds if max_seconds is None else max(0.0, max_seconds)
    warmed_paths: set[Path] = set()
    failed_paths: set[Path] = set()
    errors: list[str] = []
    stopped_for_time = False

    def ready_entries() -> list[ShardCatalogEntry]:
        return [entry for entry in selected if _cached_materialized_shard_path(entry.path) is not None]

    def ready_cache_paths() -> set[Path]:
        return {
            cache_path
            for entry in selected
            if (cache_path := _cached_materialized_shard_path(entry.path)) is not None
        }

    def ready_bytes() -> int:
        return sum((cache_path.stat().st_size for cache_path in ready_cache_paths()), 0)

    def progress_payload(*, final: bool = False) -> dict[str, object]:
        ready = ready_entries()
        return {
            "event": "warm_shard_cache_done" if final else "warm_shard_cache_progress",
            "query": query,
            "selected_shards": len(selected),
            "target_ready": target_ready,
            "ready_shards": len(ready),
            "warmed_shards": len(warmed_paths),
            "failed_shards": len(failed_paths),
            "sources_ready": _source_counts(ready),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }

    attempted = 0
    for entry in selected:
        if target_ready and len(ready_entries()) >= target_ready:
            break
        if max_shards is not None and attempted >= max_shards:
            break
        if max_seconds is not None and time.monotonic() - started >= max_seconds:
            stopped_for_time = True
            break
        if _cached_materialized_shard_path(entry.path) is not None:
            continue
        max_cache_bytes = _shard_local_cache_max_bytes()
        if max_cache_bytes is not None and ready_bytes() + entry.bytes_used > max_cache_bytes:
            errors.append(
                f"{entry.path}: target_ready exceeds cache budget "
                f"({ready_bytes() + entry.bytes_used} > {max_cache_bytes})"
            )
            break
        attempted += 1
        try:
            _materialized_shard_path(entry.path, preserve=ready_cache_paths(), populate=True)
            if _cached_materialized_shard_path(entry.path) is not None:
                warmed_paths.add(entry.path)
            else:
                failed_paths.add(entry.path)
                errors.append(f"{entry.path}: materialized path missing after copy")
        except OSError as exc:
            failed_paths.add(entry.path)
            errors.append(f"{entry.path}: {exc}")
        if progress_interval > 0 and attempted % progress_interval == 0:
            print(json.dumps(progress_payload(), sort_keys=True), flush=True)

    ready = ready_entries()
    bytes_ready = ready_bytes()
    result = ShardCacheWarmResult(
        selected_shards=len(selected),
        target_ready=target_ready,
        ready_shards=len(ready),
        warmed_shards=len(warmed_paths),
        failed_shards=len(failed_paths),
        stopped_for_target=target_ready > 0 and len(ready) >= target_ready,
        stopped_for_time=stopped_for_time,
        elapsed_seconds=round(time.monotonic() - started, 3),
        bytes_ready=bytes_ready,
        sources_selected=_source_counts(selected),
        sources_ready=_source_counts(ready),
        errors=tuple(errors),
    )
    if progress_interval > 0:
        print(json.dumps({**progress_payload(final=True), **asdict(result)}, sort_keys=True), flush=True)
    return result


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


def _shard_catalog_entry_payload(entry: ShardCatalogEntry) -> dict[str, object]:
    return {
        "path": str(entry.path),
        "batch_id": entry.batch_id,
        "shard_id": entry.shard_id,
        "sources": list(entry.sources),
        "files_completed": entry.files_completed,
        "papers_inserted": entry.papers_inserted,
        "bytes_used": entry.bytes_used,
        "year_min": entry.year_min,
        "year_max": entry.year_max,
        "cited_by_min": entry.cited_by_min,
        "cited_by_max": entry.cited_by_max,
        "cited_by_avg": entry.cited_by_avg,
        "topic_terms": list(entry.topic_terms),
    }


def _shard_catalog_entry_from_payload(payload: dict[str, object]) -> ShardCatalogEntry | None:
    path = _clean(payload.get("path"))
    if not path:
        return None
    batch_id = _int_or_none(payload.get("batch_id"))
    shard_id = _int_or_none(payload.get("shard_id"))
    return ShardCatalogEntry(
        path=Path(path),
        batch_id=batch_id if batch_id is not None else -1,
        shard_id=shard_id if shard_id is not None else -1,
        sources=_string_tuple(payload.get("sources")),
        files_completed=_int_or_none(payload.get("files_completed")) or 0,
        papers_inserted=_int_or_none(payload.get("papers_inserted")) or 0,
        bytes_used=_int_or_none(payload.get("bytes_used")) or 0,
        year_min=_int_or_none(payload.get("year_min")),
        year_max=_int_or_none(payload.get("year_max")),
        cited_by_min=_int_or_none(payload.get("cited_by_min")) or 0,
        cited_by_max=_int_or_none(payload.get("cited_by_max")) or 0,
        cited_by_avg=round(_float_or_none(payload.get("cited_by_avg")) or 0.0, 3),
        topic_terms=_string_tuple(payload.get("topic_terms")),
    )


def load_shard_catalog_cache(path: Path) -> list[ShardCatalogEntry] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    raw_entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(raw_entries, list):
        return None
    entries = [
        entry
        for raw_entry in raw_entries
        if isinstance(raw_entry, dict)
        for entry in [_shard_catalog_entry_from_payload(raw_entry)]
        if entry is not None
    ]
    return entries or None


def _catalog_entries_match_shard_dir(entries: list[ShardCatalogEntry], shard_dir: Path) -> bool:
    if not entries:
        return False
    shard_root = str(shard_dir.absolute())
    for entry in entries:
        try:
            if os.path.commonpath((str(entry.path.absolute()), shard_root)) != shard_root:
                return False
        except ValueError:
            return False
    return True


def _remap_catalog_entries_to_shard_dir(
    entries: list[ShardCatalogEntry],
    shard_dir: Path,
) -> list[ShardCatalogEntry] | None:
    remapped: list[ShardCatalogEntry] = []
    for entry in entries:
        batch_name = entry.path.parent.name
        shard_name = entry.path.name
        if not batch_name.startswith("batch_") or not shard_name.startswith("fullraw_shard_"):
            return None
        remapped.append(ShardCatalogEntry(
            path=shard_dir / batch_name / shard_name,
            batch_id=entry.batch_id,
            shard_id=entry.shard_id,
            sources=entry.sources,
            files_completed=entry.files_completed,
            papers_inserted=entry.papers_inserted,
            bytes_used=entry.bytes_used,
            year_min=entry.year_min,
            year_max=entry.year_max,
            cited_by_min=entry.cited_by_min,
            cited_by_max=entry.cited_by_max,
            cited_by_avg=entry.cited_by_avg,
            topic_terms=entry.topic_terms,
        ))
    return remapped if _catalog_entries_match_shard_dir(remapped, shard_dir) else None


def write_shard_catalog_cache(path: Path, entries: list[ShardCatalogEntry]) -> None:
    _write_json_file(path, {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entries": [_shard_catalog_entry_payload(entry) for entry in entries],
    })


def shard_coverage_receipt(
    entries: list[ShardCatalogEntry],
    selected: list[ShardCatalogEntry],
) -> dict[str, object]:
    sources_total = _source_counts(entries)
    sources_searched = _source_counts(selected)
    return {
        "shards_total": len(entries),
        "shards_searched": len(selected),
        "partial_shard_search": len(entries) > len(selected),
        "sources_total": sources_total,
        "sources_searched": sources_searched,
        "source_count_total": len(sources_total),
        "source_count_searched": len(sources_searched),
        "sources_missing_from_search": tuple(
            source for source in sorted(sources_total) if source not in sources_searched
        ),
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


def _add_planned_sweep_receipt(
    receipt: dict[str, object],
    planned: dict[str, object],
) -> None:
    receipt["sweep_planned_shards"] = planned.get("shards_searched", 0)
    receipt["sweep_planned_sources"] = planned.get("sources_searched", {})
    receipt["sweep_planned_source_count"] = planned.get("source_count_searched", 0)
    receipt["sweep_planned_year_range"] = planned.get(
        "year_range_searched",
        {"min": None, "max": None},
    )
    receipt["sweep_planned_cited_by_range"] = planned.get(
        "cited_by_range_searched",
        {"min": 0, "max": 0},
    )
    receipt["sweep_planned_topic_terms"] = planned.get("topic_terms_searched", ())
    receipt["sweep_planned_papers"] = planned.get("papers_searched", 0)


def _sweep_pass_roles_sufficient(receipt: dict[str, object]) -> bool:
    raw_passes = receipt.get("sweep_search_passes")
    if not isinstance(raw_passes, list | tuple) or not raw_passes:
        return True
    planned_roles = {
        str(pass_item.get("role"))
        for pass_item in raw_passes
        if isinstance(pass_item, dict) and str(pass_item.get("role", "")).strip()
    }
    if not planned_roles:
        return True
    max_passes = _int_or_none(receipt.get("sweep_max_passes")) or len(planned_roles)
    required_roles = min(len(planned_roles), max(1, max_passes))
    completed_roles = set(_string_tuple(receipt.get("sweep_completed_pass_roles")))
    return len(completed_roles & planned_roles) >= required_roles


def sweep_cache_entry_is_terminal(entry: SweepCacheEntry) -> bool:
    return _int_or_none(entry.receipt.get("sweep_remaining_shards")) == 0


def sweep_cache_entry_stopped_no_hits(entry: SweepCacheEntry) -> bool:
    return entry.receipt.get("sweep_stopped_no_hits") is True


def _sweep_cache_entry_should_stop_no_hits(entry: SweepCacheEntry, stop_shards: int) -> bool:
    if sweep_cache_entry_stopped_no_hits(entry):
        return True
    return stop_shards > 0 and not entry.hits and _sweep_cache_entry_progress(entry) >= stop_shards


def _sweep_cache_entry_progress(entry: SweepCacheEntry) -> int:
    return _int_or_none(entry.receipt.get("shards_searched")) or 0


def _prefer_sweep_cache_entry(
    memory_entry: SweepCacheEntry | None,
    disk_entry: SweepCacheEntry | None,
) -> SweepCacheEntry | None:
    if memory_entry is None:
        return disk_entry
    if disk_entry is None:
        return memory_entry
    if sweep_cache_entry_is_terminal(disk_entry) and not sweep_cache_entry_is_terminal(memory_entry):
        return disk_entry
    if _sweep_cache_entry_progress(disk_entry) > _sweep_cache_entry_progress(memory_entry):
        return disk_entry
    return memory_entry


def sweep_cache_entry_is_ready(
    entry: SweepCacheEntry,
    *,
    min_shards_searched: int = 0,
    min_sources_searched: int = 0,
    require_complete_search: bool = False,
    require_complete_sweep: bool = False,
    sweep_strategy: str = "",
) -> bool:
    if not entry.hits:
        return False
    if sweep_strategy and entry.receipt.get("sweep_strategy") != sweep_strategy:
        return False
    if shard_coverage_gate_response(
        entry.receipt,
        min_shards_searched=min_shards_searched,
        min_sources_searched=min_sources_searched,
        require_complete_search=require_complete_search,
    ) is not None:
        return False
    if require_complete_sweep and not _sweep_pass_roles_sufficient(entry.receipt):
        return False
    remaining = _int_or_none(entry.receipt.get("sweep_remaining_shards"))
    return not (require_complete_sweep and remaining is not None and remaining > 0)


def sweep_cache_entry_can_answer_request(
    entry: SweepCacheEntry | None,
    *,
    cache_only: bool = False,
    resume_cached: bool = False,
    min_shards_searched: int = 0,
    min_sources_searched: int = 0,
    require_complete_search: bool = False,
    require_complete_sweep: bool = False,
    sweep_strategy: str = "",
) -> bool:
    return (
        cache_only and entry is not None
        and not resume_cached
        and sweep_cache_entry_is_ready(
            entry,
            min_shards_searched=min_shards_searched,
            min_sources_searched=min_sources_searched,
            require_complete_search=require_complete_search,
            require_complete_sweep=require_complete_sweep,
            sweep_strategy=sweep_strategy,
        )
    )


def _normalize_sweep_cache_query(query: str) -> str:
    return " ".join(_fts_terms(query))


def _sweep_cache_entry_queries(entry: SweepCacheEntry) -> set[str]:
    receipt = entry.receipt
    queries = {
        str(receipt.get("sweep_query") or ""),
        str(receipt.get("sweep_original_query") or ""),
    }
    raw_passes = receipt.get("sweep_search_passes")
    if isinstance(raw_passes, list | tuple):
        queries.update(
            str(pass_item.get("query") or "")
            for pass_item in raw_passes
            if isinstance(pass_item, dict)
        )
    return {query for raw in queries if (query := _normalize_sweep_cache_query(raw))}


def _sweep_cache_entry_matches_request(
    entry: SweepCacheEntry,
    *,
    query: str,
    result_limit: int,
    sweep_shard_limit: int,
    sweep_pass_shard_limit: int,
    sweep_strategy: str,
    sweep_catalog_scope: str = "",
) -> bool:
    receipt = entry.receipt
    if sweep_strategy and receipt.get("sweep_strategy") != sweep_strategy:
        return False
    if sweep_catalog_scope and receipt.get("sweep_catalog_scope") != sweep_catalog_scope:
        return False
    if not _sweep_cache_entry_has_result_limit(entry, result_limit):
        return False
    if _int_or_none(receipt.get("sweep_shard_limit")) != sweep_shard_limit:
        return False
    request_query = _normalize_sweep_cache_query(query)
    cached_queries = _sweep_cache_entry_queries(entry)
    if request_query in cached_queries:
        return True
    if not sweep_cache_entry_is_terminal(entry) or receipt.get("partial_shard_search") is True:
        return False
    return any(_sweep_queries_alias_equivalent(request_query, cached_query) for cached_query in cached_queries)


def _sweep_queries_alias_equivalent(request_query: str, cached_query: str) -> bool:
    request_terms = _fts_terms(request_query)
    cached_terms = _fts_terms(cached_query)
    if not request_terms or not cached_terms or len(request_terms) != len(cached_terms):
        return False
    return all(
        _term_aliases(request_term) & {cached_term}
        for request_term, cached_term in zip(request_terms, cached_terms, strict=True)
    )


def _sweep_cache_entry_has_result_limit(entry: SweepCacheEntry, result_limit: int) -> bool:
    return (_int_or_none(entry.receipt.get("sweep_result_limit")) or len(entry.hits)) >= result_limit


def _should_force_cache_queue(
    *,
    shard_dir_configured: bool,
    require_complete_search: bool,
    sweep_enabled: bool,
) -> bool:
    return shard_dir_configured and require_complete_search and sweep_enabled


def _admit_sweep_key(
    key: str,
    *,
    sweep_inflight: set[str],
    sweep_queued: set[str],
    max_inflight: int,
    priority: bool = False,
    allow_priority_burst: bool = False,
    priority_max_inflight: int = 0,
) -> str:
    if key in sweep_inflight:
        return "running"
    inflight_limit = max(1, max_inflight)
    if len(sweep_inflight) >= inflight_limit:
        priority_limit = _priority_inflight_limit(
            max_inflight,
            allow_priority_burst=allow_priority_burst,
            priority_max_inflight=priority_max_inflight,
        )
        if priority and len(sweep_inflight) < priority_limit:
            sweep_queued.discard(key)
            sweep_inflight.add(key)
            return "queued"
        sweep_queued.add(key)
        return "queued"
    sweep_queued.discard(key)
    sweep_inflight.add(key)
    return "queued"


def _take_next_queued_sweep_job(
    *,
    sweep_inflight: set[str],
    sweep_queued: set[str],
    sweep_queued_jobs: dict[str, SweepJob],
    max_inflight: int,
    allow_priority_burst: bool = False,
    priority_max_inflight: int = 0,
) -> SweepJob | None:
    inflight_limit = max(1, max_inflight)
    priority_limit = _priority_inflight_limit(
        max_inflight,
        allow_priority_burst=allow_priority_burst,
        priority_max_inflight=priority_max_inflight,
    )
    for key in tuple(sweep_queued_jobs):
        job = sweep_queued_jobs.pop(key)
        if key not in sweep_queued:
            continue
        can_burst = job.priority and priority_limit > inflight_limit
        if len(sweep_inflight) >= inflight_limit and (not can_burst or len(sweep_inflight) >= priority_limit):
            sweep_queued_jobs[key] = job
            return None
        sweep_queued.discard(key)
        sweep_inflight.add(key)
        return job
    sweep_queued.clear()
    return None


def _priority_inflight_limit(
    max_inflight: int,
    *,
    allow_priority_burst: bool = False,
    priority_max_inflight: int = 0,
) -> int:
    inflight_limit = max(1, max_inflight)
    if priority_max_inflight > 0:
        return max(inflight_limit, priority_max_inflight)
    return inflight_limit + 1 if allow_priority_burst else inflight_limit


def _queue_sweep_job(sweep_queued_jobs: dict[str, SweepJob], key: str, job: SweepJob) -> None:
    _queue_sweep_job_with_priority(sweep_queued_jobs, key, job, priority=key in sweep_queued_jobs)


def _trim_sweep_queue(
    sweep_queued_jobs: dict[str, SweepJob],
    *,
    sweep_queued: set[str] | None = None,
    max_queue: int = 0,
) -> None:
    while max_queue > 0 and len(sweep_queued_jobs) > max_queue:
        drop_key = next(
            (queued_key for queued_key in reversed(sweep_queued_jobs) if not sweep_queued_jobs[queued_key].priority),
            next(reversed(sweep_queued_jobs)),
        )
        sweep_queued_jobs.pop(drop_key, None)
        if sweep_queued is not None:
            sweep_queued.discard(drop_key)


def _queue_sweep_job_with_priority(
    sweep_queued_jobs: dict[str, SweepJob],
    key: str,
    job: SweepJob,
    *,
    priority: bool,
    sweep_queued: set[str] | None = None,
    max_queue: int = 0,
) -> None:
    if not priority and key not in sweep_queued_jobs:
        sweep_queued_jobs[key] = job
        _trim_sweep_queue(sweep_queued_jobs, sweep_queued=sweep_queued, max_queue=max_queue)
        return
    existing = tuple((queued_key, queued_job) for queued_key, queued_job in sweep_queued_jobs.items() if queued_key != key)
    sweep_queued_jobs.clear()
    sweep_queued_jobs[key] = job
    sweep_queued_jobs.update(existing)
    _trim_sweep_queue(sweep_queued_jobs, sweep_queued=sweep_queued, max_queue=max_queue)


def _sweep_queue_summary(
    sweep_inflight: set[str],
    sweep_queued_jobs: dict[str, SweepJob],
    *,
    max_inflight: int,
    priority_max_inflight: int = 0,
    max_queue: int,
    priority_burst: bool,
    workers: int,
    enabled: bool,
) -> dict[str, object]:
    queued_count = len(sweep_queued_jobs)
    priority_queued = sum(1 for job in sweep_queued_jobs.values() if job.priority)
    return {
        "enabled": enabled,
        "inflight_count": len(sweep_inflight),
        "queued_count": queued_count,
        "priority_queued_count": priority_queued,
        "background_queued_count": max(0, queued_count - priority_queued),
        "max_inflight": max_inflight,
        "priority_max_inflight": _priority_inflight_limit(
            max_inflight,
            allow_priority_burst=priority_burst,
            priority_max_inflight=priority_max_inflight,
        ),
        "max_queue": max_queue,
        "priority_burst": priority_burst,
        "workers": workers,
    }


def _prune_stale_sweep_inflight(
    sweep_inflight: set[str],
    sweep_inflight_started: dict[str, float],
    *,
    now: float,
    stale_after_seconds: float,
) -> tuple[str, ...]:
    if stale_after_seconds <= 0:
        return ()
    stale = tuple(
        key
        for key in tuple(sweep_inflight)
        if now - sweep_inflight_started.get(key, 0.0) > stale_after_seconds
    )
    for key in stale:
        sweep_inflight.discard(key)
        sweep_inflight_started.pop(key, None)
    return stale


def _sweep_watchdog_tick(
    *,
    sweep_inflight: set[str],
    sweep_inflight_started: dict[str, float],
    sweep_queued: set[str],
    sweep_queued_jobs: dict[str, SweepJob],
    max_inflight: int,
    allow_priority_burst: bool,
    priority_max_inflight: int = 0,
    stale_after_seconds: float,
    now: float,
) -> tuple[tuple[str, ...], list[SweepJob]]:
    stale = _prune_stale_sweep_inflight(
        sweep_inflight,
        sweep_inflight_started,
        now=now,
        stale_after_seconds=stale_after_seconds,
    )
    next_jobs: list[SweepJob] = []
    while True:
        next_job = _take_next_queued_sweep_job(
            sweep_inflight=sweep_inflight,
            sweep_queued=sweep_queued,
            sweep_queued_jobs=sweep_queued_jobs,
            max_inflight=max_inflight,
            allow_priority_burst=allow_priority_burst,
            priority_max_inflight=priority_max_inflight,
        )
        if next_job is None:
            break
        sweep_inflight_started[next_job.key] = now
        next_jobs.append(next_job)
    return stale, next_jobs


def shard_coverage_gate_response(
    receipt: dict[str, object],
    *,
    min_shards_searched: int = 0,
    min_sources_searched: int = 0,
    require_complete_search: bool = False,
) -> tuple[int, dict[str, object]] | None:
    failures: list[str] = []
    shards_searched = _int_or_none(receipt.get("shards_searched")) or 0
    sources_searched = receipt.get("sources_searched")
    source_count = len(sources_searched) if isinstance(sources_searched, dict) else 0
    if require_complete_search and receipt.get("partial_shard_search") is True:
        failures.append("partial_shard_search true while complete search is required")
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
            "require_complete_search": require_complete_search,
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


def _filter_raw_files_by_source(files: list[RawFile], source_filter: str) -> list[RawFile]:
    allowed = {
        source.strip().casefold()
        for source in source_filter.split(",")
        if source.strip()
    }
    if not allowed:
        return files
    return [raw_file for raw_file in files if raw_file.source.casefold() in allowed]


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
    host = _fullraw_env("V5_MEMO_FULL_RAW_INDEX_HOST", "127.0.0.1")
    port = int(_fullraw_env("V5_MEMO_FULL_RAW_INDEX_PORT", "9902"))
    index_path = Path(
        _fullraw_env("V5_MEMO_FULL_RAW_INDEX_PATH", "/var/lib/v5-memo/fullraw_index.sqlite")
    )
    shard_dir_config = _fullraw_env("V5_MEMO_FULL_RAW_SHARD_DIR", "").strip()
    shard_dir = Path(shard_dir_config) if shard_dir_config else None
    trust_shard_filenames = _fullraw_env(
        "V5_MEMO_FULL_RAW_SHARD_TRUST_FILENAMES", ""
    ).casefold() in {"1", "true", "yes"}
    shard_manifest_stats = _fullraw_env(
        "V5_MEMO_FULL_RAW_SHARD_MANIFEST_STATS", ""
    ).casefold() in {"1", "true", "yes"}
    fast_health = _fullraw_env("V5_MEMO_FULL_RAW_FAST_HEALTH", "").casefold() in {"1", "true", "yes"}
    shard_catalog_ttl = _float_or_none(
        _fullraw_env("V5_MEMO_FULL_RAW_SHARD_CATALOG_TTL_SECONDS", "")
    ) or 60.0
    min_shards_searched = _positive_int_env("V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED") or 0
    min_sources_searched = _positive_int_env("V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED") or 0
    require_complete_search = _fullraw_env(
        "V5_MEMO_FULL_RAW_REQUIRE_COMPLETE_SEARCH", ""
    ).casefold() in {"1", "true", "yes"}
    sweep_require_complete = _fullraw_env(
        "V5_MEMO_FULL_RAW_SWEEP_REQUIRE_COMPLETE", ""
    ).casefold() in {"1", "true", "yes"}
    sweep_enabled = _fullraw_env("V5_MEMO_FULL_RAW_ASYNC_SWEEP", "").casefold() in {"1", "true", "yes"}
    sweep_ttl = _float_or_none(_fullraw_env("V5_MEMO_FULL_RAW_SWEEP_TTL_SECONDS", "")) or 86400.0
    sweep_max_inflight = _positive_int_env("V5_MEMO_FULL_RAW_SWEEP_MAX_INFLIGHT") or 1
    sweep_workers = _positive_int_env("V5_MEMO_FULL_RAW_SWEEP_WORKERS") or _auto_sweep_workers(sweep_max_inflight)
    sweep_max_queue = _positive_int_env("V5_MEMO_FULL_RAW_SWEEP_MAX_QUEUE") or 0
    sweep_shard_limit = _positive_int_env("V5_MEMO_FULL_RAW_SWEEP_SHARD_LIMIT") or 128
    sweep_pass_shard_limit = _positive_int_env("V5_MEMO_FULL_RAW_SWEEP_PASS_SHARD_LIMIT") or sweep_shard_limit
    sweep_pass_shard_limit = max(1, min(sweep_pass_shard_limit, sweep_shard_limit))
    sweep_no_hit_stop_shards = _positive_int_env("V5_MEMO_FULL_RAW_SWEEP_NO_HIT_STOP_SHARDS") or 0
    sweep_max_passes = _positive_int_env("V5_MEMO_FULL_RAW_SWEEP_MAX_PASSES") or 1
    sweep_max_passes = max(1, min(sweep_max_passes, sweep_shard_limit))
    sweep_priority_burst = _fullraw_env(
        "V5_MEMO_FULL_RAW_SWEEP_PRIORITY_BURST",
        "true",
    ).casefold() in {"1", "true", "yes"}
    sweep_priority_max_inflight = _positive_int_env("V5_MEMO_FULL_RAW_SWEEP_PRIORITY_MAX_INFLIGHT") or 0
    sweep_timeout_seconds = _float_or_none(_fullraw_env("V5_MEMO_FULL_RAW_SWEEP_TIMEOUT_SECONDS", "")) or 300.0
    sweep_timeout_seconds = max(1.0, min(sweep_timeout_seconds, 3600.0))
    sweep_shard_timeout_seconds = _float_or_none(
        _fullraw_env("V5_MEMO_FULL_RAW_SWEEP_SHARD_TIMEOUT_SECONDS", "")
    ) or 10.0
    sweep_shard_timeout_seconds = max(0.1, min(sweep_shard_timeout_seconds, sweep_timeout_seconds))
    search_shard_timeout_seconds = _float_or_none(_fullraw_env("V5_MEMO_FULL_RAW_SEARCH_SUBPROCESS_TIMEOUT", ""))
    shard_catalog_path_config = _fullraw_env("V5_MEMO_FULL_RAW_SHARD_CATALOG_PATH", "").strip()
    shard_catalog_path = Path(shard_catalog_path_config) if shard_catalog_path_config else None
    sweep_cache_dir_config = _fullraw_env("V5_MEMO_FULL_RAW_SWEEP_CACHE_DIR", "").strip()
    sweep_cache_dir = Path(sweep_cache_dir_config) if sweep_cache_dir_config else None
    sweep_catalog_scope = str(shard_dir.absolute()) if shard_dir is not None else ""
    manifest_path = Path(
        _fullraw_env("V5_MEMO_FULL_RAW_MANIFEST", "/var/lib/v5-memo/fullraw_manifest.json")
    )
    rclone_bin = _fullraw_env("V5_MEMO_FULL_RAW_RCLONE", "rclone")
    refresh = _fullraw_env("V5_MEMO_FULL_RAW_REFRESH_MANIFEST", "").casefold() in {"1", "true", "yes"}
    files = load_or_build_manifest(manifest_path, refresh=refresh, rclone_bin=rclone_bin)
    token = (
        _fullraw_env("V5_MEMO_FULL_RAW_INDEX_TOKEN", "")
        or _fullraw_env("V5_MEMO_FULL_RAW_TOKEN", "")
    ).strip()
    index = None if shard_dir else FullRawFtsIndex(index_path)
    if index is not None:
        index.initialize()
    catalog_cache: tuple[float, list[ShardCatalogEntry]] = (0.0, [])
    sweep_cache: dict[str, SweepCacheEntry] = {}
    sweep_inflight: set[str] = set()
    sweep_inflight_started: dict[str, float] = {}
    sweep_queued: set[str] = set()
    sweep_queued_jobs: dict[str, SweepJob] = {}
    sweep_lock = threading.RLock()
    sweep_inflight_stale_seconds = (
        _float_or_none(_fullraw_env("V5_MEMO_FULL_RAW_SWEEP_INFLIGHT_STALE_SECONDS", ""))
        or max(900.0, min(sweep_timeout_seconds + 60.0, 1800.0))
    )

    def sweep_watchdog_tick_locked(now: float) -> tuple[tuple[str, ...], list[SweepJob]]:
        return _sweep_watchdog_tick(
            sweep_inflight=sweep_inflight,
            sweep_inflight_started=sweep_inflight_started,
            sweep_queued=sweep_queued,
            sweep_queued_jobs=sweep_queued_jobs,
            max_inflight=sweep_max_inflight,
            allow_priority_burst=sweep_priority_burst,
            priority_max_inflight=sweep_priority_max_inflight,
            stale_after_seconds=sweep_inflight_stale_seconds,
            now=now,
        )

    def report_sweep_tick(stale: tuple[str, ...], next_jobs: list[SweepJob]) -> None:
        for key in stale:
            print(
                f"fullraw sweep inflight lease expired key={key}",
                file=sys.stderr,
                flush=True,
            )
        for next_job in next_jobs:
            start_sweep_worker(next_job)

    def sweep_queue_summary() -> dict[str, object]:
        stale: tuple[str, ...] = ()
        next_jobs: list[SweepJob] = []
        with sweep_lock:
            if sweep_enabled and shard_dir is not None:
                stale, next_jobs = sweep_watchdog_tick_locked(time.monotonic())
            summary = _sweep_queue_summary(
                sweep_inflight,
                sweep_queued_jobs,
                max_inflight=sweep_max_inflight,
                priority_max_inflight=sweep_priority_max_inflight,
                max_queue=sweep_max_queue,
                priority_burst=sweep_priority_burst,
                workers=sweep_workers,
                enabled=sweep_enabled and shard_dir is not None,
            )
        report_sweep_tick(stale, next_jobs)
        return summary

    def sweep_queue_state(key: str) -> dict[str, object]:
        stale: tuple[str, ...] = ()
        next_jobs: list[SweepJob] = []
        with sweep_lock:
            if sweep_enabled and shard_dir is not None:
                stale, next_jobs = sweep_watchdog_tick_locked(time.monotonic())
            state = {
                **_sweep_queue_summary(
                    sweep_inflight,
                    sweep_queued_jobs,
                    max_inflight=sweep_max_inflight,
                    priority_max_inflight=sweep_priority_max_inflight,
                    max_queue=sweep_max_queue,
                    priority_burst=sweep_priority_burst,
                    workers=sweep_workers,
                    enabled=sweep_enabled and shard_dir is not None,
                ),
                "key_running": key in sweep_inflight,
                "key_queued": key in sweep_queued_jobs,
            }
        report_sweep_tick(stale, next_jobs)
        return state

    def current_catalog() -> list[ShardCatalogEntry]:
        nonlocal catalog_cache
        if shard_dir is None:
            return []
        now = time.monotonic()
        if catalog_cache[1] and now - catalog_cache[0] < shard_catalog_ttl:
            return catalog_cache[1]
        if shard_catalog_path is not None and shard_catalog_path.exists():
            cached_catalog = load_shard_catalog_cache(shard_catalog_path)
            if cached_catalog is not None and _catalog_entries_match_shard_dir(cached_catalog, shard_dir):
                catalog_cache = (now, cached_catalog)
                return cached_catalog
            if cached_catalog is not None:
                remapped_catalog = _remap_catalog_entries_to_shard_dir(cached_catalog, shard_dir)
                if remapped_catalog is not None:
                    write_shard_catalog_cache(shard_catalog_path, remapped_catalog)
                    catalog_cache = (now, remapped_catalog)
                    return remapped_catalog
        catalog = build_shard_catalog(shard_dir, trust_filenames=trust_shard_filenames)
        if shard_catalog_path is not None:
            write_shard_catalog_cache(shard_catalog_path, catalog)
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

    def current_search_with_receipt(
        query: str,
        *,
        limit: int,
        year_min: int,
        year_max: int,
        rank_mode: str,
        timeout_seconds: float | None,
        ) -> tuple[list[dict[str, object]], dict[str, object]]:
        if shard_dir is not None:
            return search_shard_entries_with_receipt(
                current_catalog(),
                query,
                limit=limit,
                year_min=year_min,
                year_max=year_max,
                rank_mode=rank_mode,
                timeout_seconds=timeout_seconds,
                shard_timeout_seconds=search_shard_timeout_seconds or timeout_seconds,
            )
        assert index is not None
        return (
            index.search(query, limit=limit, year_min=year_min, year_max=year_max, rank_mode=rank_mode),
            {},
        )

    def sweep_cache_get(key: str) -> SweepCacheEntry | None:
        with sweep_lock:
            memory_entry = sweep_cache.get(key)
            if memory_entry is not None and sweep_ttl > 0 and time.time() - memory_entry.created_at > sweep_ttl:
                memory_entry = None
            if memory_entry is not None and sweep_cache_entry_is_terminal(memory_entry):
                return memory_entry
        cache_path = _sweep_cache_path(sweep_cache_dir, key)
        disk_entry = None
        if cache_path is not None and cache_path.exists():
            disk_entry = _load_sweep_cache(cache_path, ttl_seconds=sweep_ttl)
        entry = _prefer_sweep_cache_entry(memory_entry, disk_entry)
        if entry is not None and entry is not memory_entry:
            with sweep_lock:
                sweep_cache[key] = entry
        return entry

    def compatible_sweep_cache_get(
        key: str,
        cache_query: str,
        *,
        result_limit: int,
    ) -> SweepCacheEntry | None:
        entry = sweep_cache_get(key)
        if entry is not None and _sweep_cache_entry_matches_request(
            entry,
            query=cache_query,
            result_limit=result_limit,
            sweep_shard_limit=sweep_shard_limit,
            sweep_pass_shard_limit=sweep_pass_shard_limit,
            sweep_strategy=_SWEEP_STRATEGY,
            sweep_catalog_scope=sweep_catalog_scope,
        ):
            return entry
        if sweep_cache_dir is None:
            return None
        best: SweepCacheEntry | None = None
        for path in sweep_cache_dir.glob("*.json"):
            if path.name == f"{key}.json":
                continue
            candidate = _load_sweep_cache(path, ttl_seconds=sweep_ttl)
            if candidate is None or not _sweep_cache_entry_matches_request(
                candidate,
                query=cache_query,
                result_limit=result_limit,
                sweep_shard_limit=sweep_shard_limit,
                sweep_pass_shard_limit=sweep_pass_shard_limit,
                sweep_strategy=_SWEEP_STRATEGY,
                sweep_catalog_scope=sweep_catalog_scope,
            ):
                continue
            best = _prefer_sweep_cache_entry(best, candidate)
        if best is not None:
            sweep_cache_put(key, best, final=sweep_cache_entry_is_terminal(best))
        return best

    def sweep_cache_put(key: str, entry: SweepCacheEntry, *, final: bool = True) -> None:
        with sweep_lock:
            sweep_cache[key] = entry
            if final:
                sweep_inflight.discard(key)
                sweep_inflight_started.pop(key, None)
            elif key in sweep_inflight:
                sweep_inflight_started[key] = time.monotonic()
        cache_path = _sweep_cache_path(sweep_cache_dir, key)
        if cache_path is not None:
            _write_sweep_cache(cache_path, entry)

    def start_sweep_worker(job: SweepJob) -> None:
        def worker() -> None:
            try:
                existing = sweep_cache_get(job.key)
                if (
                    existing is not None
                    and not _sweep_cache_entry_has_result_limit(existing, job.limit)
                    and sweep_cache_entry_is_terminal(existing)
                ):
                    existing = None
                sweep_entries = select_sweep_shard_entries(job.catalog, query=job.query, limit=sweep_shard_limit)
                if len(sweep_entries) >= len(job.catalog):
                    sweep_entries = _cache_reuse_sweep_entries(sweep_entries)
                else:
                    sweep_entries = _prioritize_sweep_pass_entries(
                        sweep_entries,
                        sweep_pass_shard_limit,
                        query=job.query,
                    )
                    sweep_entries = _cache_fit_warm_entries(
                        job.catalog,
                        sweep_entries,
                        query=job.query,
                        target_ready=min_shards_searched,
                    )
                planned_receipt = shard_coverage_receipt(job.catalog, sweep_entries)
                sweep_passes = _sweep_search_passes(job.query, sweep_entries, rank_mode=job.rank_mode)
                completed_path_strings = _sweep_completed_path_strings(existing.receipt if existing else {})
                failed_path_strings = _sweep_failed_path_strings_for_mode(
                    existing.receipt if existing else {},
                    require_complete_sweep=sweep_require_complete,
                )
                deferred_path_strings = set(_string_tuple(
                    (existing.receipt if existing else {}).get("sweep_deferred_paths")
                ))
                merged_hits = list(existing.hits if existing else [])
                previous_passes = _int_or_none((existing.receipt if existing else {}).get("sweep_passes")) or 0
                completed_pass_roles = list(
                    _string_tuple((existing.receipt if existing else {}).get("sweep_completed_pass_roles"))
                )
                receipt: dict[str, object] = existing.receipt if existing else {}
                for pass_index in range(sweep_max_passes):
                    pass_plan = sweep_passes[(previous_passes + pass_index) % len(sweep_passes)]
                    pass_entries, deferred_path_strings = _next_sweep_pass_entries(
                        sweep_entries,
                        completed_path_strings=completed_path_strings,
                        failed_path_strings=failed_path_strings,
                        deferred_path_strings=deferred_path_strings,
                        limit=sweep_pass_shard_limit,
                    )
                    if not pass_entries:
                        break
                    hits, completed_paths, timed_out, pass_metrics = _search_shard_paths_with_paths_and_receipt(
                        [entry.path for entry in pass_entries],
                        pass_plan.query,
                        limit=job.limit,
                        year_min=job.year_min,
                        year_max=job.year_max,
                        rank_mode=pass_plan.rank_mode,
                        workers=sweep_workers,
                        timeout_seconds=sweep_timeout_seconds,
                        shard_timeout_seconds=sweep_shard_timeout_seconds,
                    )
                    completed_pass_roles.append(pass_plan.role)
                    completed_path_strings.update(str(path) for path in completed_paths)
                    missed_path_strings = {
                        str(entry.path)
                        for entry in pass_entries
                        if str(entry.path) not in completed_path_strings
                    }
                    if sweep_require_complete:
                        deferred_path_strings.update(missed_path_strings)
                    if not timed_out:
                        failed_path_strings.update(_sweep_pass_failed_path_strings(
                            pass_entries,
                            completed_path_strings=completed_path_strings,
                            existing_failed_path_strings=failed_path_strings,
                            require_complete_sweep=sweep_require_complete,
                        ))
                    searched_entries = [entry for entry in sweep_entries if str(entry.path) in completed_path_strings]
                    receipt = shard_coverage_receipt(job.catalog, searched_entries)
                    _add_planned_sweep_receipt(receipt, planned_receipt)
                    receipt["sweep_scope"] = "relevant"
                    receipt["sweep_shard_limit"] = sweep_shard_limit
                    receipt["sweep_result_limit"] = job.limit
                    receipt["sweep_selected_shards"] = len(sweep_entries)
                    receipt["sweep_pass_shard_limit"] = sweep_pass_shard_limit
                    receipt["sweep_pass_selected_shards"] = len(pass_entries)
                    receipt["sweep_max_passes"] = sweep_max_passes
                    receipt["sweep_failed_shards"] = len(failed_path_strings)
                    receipt["sweep_failed_paths"] = sorted(failed_path_strings)
                    receipt["sweep_deferred_paths"] = sorted(deferred_path_strings)
                    receipt["sweep_remaining_shards"] = _sweep_remaining_shard_count(
                        selected_shards=len(sweep_entries),
                        completed_shards=len(completed_path_strings),
                        failed_shards=len(failed_path_strings),
                        require_complete_sweep=sweep_require_complete,
                    )
                    receipt["sweep_timed_out"] = timed_out
                    receipt["sweep_timeout_seconds"] = sweep_timeout_seconds
                    receipt["sweep_shard_timeout_seconds"] = sweep_shard_timeout_seconds
                    receipt["sweep_strategy"] = _SWEEP_STRATEGY
                    receipt["sweep_catalog_scope"] = sweep_catalog_scope
                    receipt["sweep_search_passes"] = tuple(asdict(pass_item) for pass_item in sweep_passes)
                    receipt["sweep_completed_pass_roles"] = tuple(completed_pass_roles)
                    receipt["sweep_completed_pass_role_counts"] = dict(sorted(Counter(completed_pass_roles).items()))
                    receipt["sweep_pass_role"] = pass_plan.role
                    receipt["sweep_pass_query"] = pass_plan.query
                    receipt["sweep_pass_rank_mode"] = pass_plan.rank_mode
                    receipt["sweep_pass_result_metrics"] = pass_metrics
                    receipt["sweep_query"] = pass_plan.query
                    receipt["sweep_passes"] = previous_passes + pass_index + 1
                    receipt["sweep_completed_paths"] = sorted(completed_path_strings)
                    if pass_plan.query != job.query:
                        receipt["sweep_original_query"] = job.query
                    merged_hits, result_metrics = _merge_hit_groups_with_receipt([merged_hits, hits], limit=job.limit)
                    receipt.update(result_metrics)
                    required_pass_roles = min(len(sweep_passes), sweep_max_passes)
                    pass_roles_sufficient = len(set(completed_pass_roles)) >= required_pass_roles
                    no_hit_stop = (
                        sweep_no_hit_stop_shards > 0
                        and not merged_hits
                        and len(completed_path_strings) >= sweep_no_hit_stop_shards
                    )
                    if no_hit_stop:
                        receipt["sweep_stopped_no_hits"] = True
                        receipt["sweep_no_hit_stop_shards"] = sweep_no_hit_stop_shards
                    final = (
                        (bool(merged_hits) and receipt_is_sufficient(receipt) and pass_roles_sufficient)
                        or receipt["sweep_remaining_shards"] == 0
                        or no_hit_stop
                        or pass_index + 1 >= sweep_max_passes
                    )
                    sweep_cache_put(job.key, SweepCacheEntry(time.time(), merged_hits, receipt), final=final)
                    if final or (timed_out and not completed_paths and not sweep_require_complete):
                        break
            except Exception as exc:
                print(f"fullraw sweep worker failed key={job.key}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            finally:
                next_job = None
                with sweep_lock:
                    sweep_inflight.discard(job.key)
                    sweep_inflight_started.pop(job.key, None)
                    sweep_queued.discard(job.key)
                    sweep_queued_jobs.pop(job.key, None)
                    next_job = _take_next_queued_sweep_job(
                        sweep_inflight=sweep_inflight,
                        sweep_queued=sweep_queued,
                        sweep_queued_jobs=sweep_queued_jobs,
                        max_inflight=sweep_max_inflight,
                        allow_priority_burst=sweep_priority_burst,
                        priority_max_inflight=sweep_priority_max_inflight,
                    )
                    if next_job is not None:
                        sweep_inflight_started[next_job.key] = time.monotonic()
                if next_job is not None:
                    start_sweep_worker(next_job)

        threading.Thread(target=worker, daemon=True).start()

    def enqueue_sweep(
        *,
        key: str,
        query: str,
        limit: int,
        year_min: int,
        year_max: int,
        rank_mode: str,
        catalog: list[ShardCatalogEntry],
        priority: bool = False,
    ) -> str:
        if not sweep_enabled or shard_dir is None:
            return "disabled"
        result_limit = max(limit, _SWEEP_MIN_RESULT_LIMIT)
        existing = sweep_cache_get(key)
        if (
            existing is not None
            and _sweep_cache_entry_should_stop_no_hits(existing, sweep_no_hit_stop_shards)
            and _sweep_cache_entry_has_result_limit(existing, result_limit)
        ):
            return "stopped_no_hits"
        if existing is not None and (
            _sweep_cache_entry_has_result_limit(existing, result_limit)
            and (
                sweep_entry_is_ready(existing)
                or (sweep_cache_entry_is_terminal(existing) and receipt_is_sufficient(existing.receipt))
            )
        ):
            return "hit"
        job = SweepJob(
            key=key,
            query=query,
            limit=result_limit,
            year_min=year_min,
            year_max=year_max,
            rank_mode=rank_mode,
            catalog=catalog,
            priority=priority,
        )
        with sweep_lock:
            status = _admit_sweep_key(
                key,
                sweep_inflight=sweep_inflight,
                sweep_queued=sweep_queued,
                max_inflight=sweep_max_inflight,
                priority=priority,
                allow_priority_burst=sweep_priority_burst,
                priority_max_inflight=sweep_priority_max_inflight,
            )
            if status == "queued" and key not in sweep_inflight:
                _queue_sweep_job_with_priority(
                    sweep_queued_jobs,
                    key,
                    job,
                    priority=priority,
                    sweep_queued=sweep_queued,
                    max_queue=sweep_max_queue,
                )
                if not priority:
                    return status
                next_job = _take_next_queued_sweep_job(
                    sweep_inflight=sweep_inflight,
                    sweep_queued=sweep_queued,
                    sweep_queued_jobs=sweep_queued_jobs,
                    max_inflight=sweep_max_inflight,
                    allow_priority_burst=sweep_priority_burst,
                    priority_max_inflight=sweep_priority_max_inflight,
                )
                if next_job is None:
                    return status
                job = next_job
            if status != "queued":
                return status
            sweep_inflight_started[job.key] = time.monotonic()
            sweep_queued_jobs.pop(key, None)
        start_sweep_worker(job)
        return "queued"

    def start_sweep_watchdog() -> None:
        if not sweep_enabled or shard_dir is None:
            return

        def watchdog() -> None:
            interval = max(5.0, min(60.0, sweep_inflight_stale_seconds / 3.0))
            while True:
                time.sleep(interval)
                try:
                    with sweep_lock:
                        stale, next_jobs = sweep_watchdog_tick_locked(time.monotonic())
                    report_sweep_tick(stale, next_jobs)
                except Exception as exc:  # pragma: no cover - defensive watchdog guard
                    print(
                        f"fullraw sweep watchdog failed: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

        threading.Thread(target=watchdog, daemon=True).start()

    start_sweep_watchdog()

    def auth_receipt(receipt: dict[str, object]) -> dict[str, object]:
        return {
            **receipt,
            "auth_required": bool(token),
            "authenticated": bool(token),
        }

    def current_receipt(query: str) -> dict[str, object]:
        if shard_dir is None:
            return auth_receipt({})
        catalog = current_catalog()
        return auth_receipt(shard_coverage_receipt(catalog, select_search_shard_entries(catalog, query=query)))

    def coverage_requirements() -> dict[str, int]:
        return {
            "min_shards_searched": min_shards_searched,
            "min_sources_searched": min_sources_searched,
            "require_complete_search": int(require_complete_search),
            "sweep_require_complete": int(sweep_require_complete),
        }

    def shard_cache_health() -> dict[str, object]:
        return _shard_local_cache_health()

    def receipt_is_sufficient(receipt: dict[str, object]) -> bool:
        if shard_coverage_gate_response(
            receipt,
            min_shards_searched=min_shards_searched,
            min_sources_searched=min_sources_searched,
            require_complete_search=require_complete_search,
        ) is not None or (sweep_require_complete and not _sweep_pass_roles_sufficient(receipt)):
            return False
        if sweep_require_complete:
            remaining = _int_or_none(receipt.get("sweep_remaining_shards"))
            if remaining is not None and remaining > 0:
                return False
        return True

    def sweep_entry_is_ready(entry: SweepCacheEntry) -> bool:
        return sweep_cache_entry_is_ready(
            entry,
            min_shards_searched=min_shards_searched,
            min_sources_searched=min_sources_searched,
            require_complete_search=require_complete_search,
            require_complete_sweep=sweep_require_complete,
            sweep_strategy=_SWEEP_STRATEGY,
        )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404)
                return
            if fast_health and shard_dir is not None:
                _write_json(self, 200, {
                    "ok": True,
                    "backend": _BACKEND,
                    "index_path": str(index_path),
                    "shard_dir": str(shard_dir),
                    "fast_health": True,
                    "complete": False,
                    "coverage_requirements": coverage_requirements(),
                    "shard_cache": _shard_local_cache_health(include_dynamic_budget=False),
                    "async_sweep": sweep_queue_summary(),
                })
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
                "shard_cache": shard_cache_health(),
                "async_sweep": sweep_queue_summary(),
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
                queue_if_missing_was_provided = "queue_if_missing" in payload
                queue_if_missing = raw_queue_if_missing is True or (
                    isinstance(raw_queue_if_missing, str)
                    and raw_queue_if_missing.strip().casefold() in {"1", "true", "yes", "on"}
                )
                raw_priority = payload.get("priority")
                priority = raw_priority is True or (
                    isinstance(raw_priority, str)
                    and raw_priority.strip().casefold() in {"1", "true", "yes", "on"}
                )
                raw_allow_partial_results = payload.get("allow_partial_results")
                allow_partial_results = raw_allow_partial_results is True or (
                    isinstance(raw_allow_partial_results, str)
                    and raw_allow_partial_results.strip().casefold() in {"1", "true", "yes", "on"}
                )
                if _should_force_cache_queue(
                    shard_dir_configured=shard_dir is not None,
                    require_complete_search=require_complete_search,
                    sweep_enabled=sweep_enabled,
                ):
                    cache_only = True
                    if not queue_if_missing_was_provided:
                        queue_if_missing = True
            except (TypeError, ValueError, json.JSONDecodeError):
                _write_json(self, 400, {"error": "bad request"})
                return
            if not query:
                _write_json(self, 400, {"error": "query is required"})
                return
            started = time.monotonic()
            catalog = current_catalog() if shard_dir is not None else []
            cache_query = _sweep_cache_query(
                query,
                catalog,
                sweep_shard_limit=sweep_shard_limit,
                rank_mode=rank_mode,
            )
            result_limit = max(limit, _SWEEP_MIN_RESULT_LIMIT)
            cache_key = _sweep_cache_key(
                cache_query,
                limit=result_limit,
                year_min=year_min,
                year_max=year_max,
                rank_mode=rank_mode,
                sweep_shard_limit=sweep_shard_limit,
                sweep_pass_shard_limit=sweep_pass_shard_limit,
                sweep_max_passes=sweep_max_passes,
                sweep_timeout_seconds=sweep_timeout_seconds,
                sweep_shard_timeout_seconds=sweep_shard_timeout_seconds,
                sweep_strategy=_SWEEP_STRATEGY,
                sweep_catalog_scope=sweep_catalog_scope,
            )
            cached = (
                compatible_sweep_cache_get(cache_key, cache_query, result_limit=result_limit)
                if catalog
                else None
            )
            resume_cached = (
                cached is not None
                and cache_only
                and queue_if_missing
                and not (
                    sweep_entry_is_ready(cached)
                    or (sweep_cache_entry_is_terminal(cached) and receipt_is_sufficient(cached.receipt))
                )
            )
            sweep_status = "disabled"
            if sweep_cache_entry_can_answer_request(
                cached,
                cache_only=cache_only,
                resume_cached=resume_cached,
                min_shards_searched=min_shards_searched,
                min_sources_searched=min_sources_searched,
                require_complete_search=require_complete_search,
                require_complete_sweep=sweep_require_complete,
                sweep_strategy=_SWEEP_STRATEGY,
            ):
                assert cached is not None
                hits = cached.hits[:limit]
                receipt = auth_receipt(cached.receipt)
                sweep_status = "hit"
                if cache_only:
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
                                "strategy": _SWEEP_STRATEGY,
                                **sweep_queue_state(cache_key),
                            },
                        },
                        "results": hits,
                    })
                    return
            else:
                if cache_only:
                    hits = []
                    receipt = auth_receipt(cached.receipt) if cached is not None else auth_receipt({})
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
                            priority=priority,
                        )
                        if sweep_status == "hit" and (cached := sweep_cache_get(cache_key)) is not None:
                            receipt = auth_receipt(cached.receipt)
                            if sweep_entry_is_ready(cached) or (
                                sweep_cache_entry_is_terminal(cached)
                                and receipt_is_sufficient(cached.receipt)
                            ):
                                hits = cached.hits[:limit]
                            elif resume_cached:
                                sweep_status = "running"
                        elif resume_cached and cached is not None:
                            receipt = auth_receipt(cached.receipt)
                            if sweep_entry_is_ready(cached) or (
                                sweep_cache_entry_is_terminal(cached)
                                and receipt_is_sufficient(cached.receipt)
                            ):
                                hits = cached.hits[:limit]
                    else:
                        with sweep_lock:
                            sweep_status = "running" if cache_key in sweep_inflight else "miss"
                    if receipt:
                        partial_progress = (
                            cache_only
                            and queue_if_missing
                            and sweep_status in {"busy", "queued", "running"}
                            and not receipt_is_sufficient(receipt)
                        )
                        coverage_gate = shard_coverage_gate_response(
                            receipt,
                            min_shards_searched=min_shards_searched,
                            min_sources_searched=min_sources_searched,
                            require_complete_search=require_complete_search,
                        )
                        if coverage_gate is not None and not partial_progress:
                            status, body = coverage_gate
                            _write_json(self, status, body)
                            return
                        if partial_progress:
                            hits = cached.hits[:limit] if allow_partial_results and cached is not None else []
                    _write_json(self, 200, {
                        "meta": {
                            "count": len(hits),
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                            "backend": _BACKEND,
                            "rank_mode": rank_mode,
                            "shard_receipt": receipt,
                            "cache_only": True,
                            "partial_results": bool(partial_progress and allow_partial_results and hits),
                            "async_sweep": {
                                "enabled": sweep_enabled and bool(catalog),
                                "status": sweep_status,
                                "cache_key": cache_key if sweep_enabled and catalog else "",
                                "scope": "relevant",
                                "shard_limit": sweep_shard_limit,
                                "strategy": _SWEEP_STRATEGY,
                                **sweep_queue_state(cache_key),
                            },
                        },
                        "results": hits,
                    })
                    return
                else:
                    receipt = (
                        auth_receipt(shard_coverage_receipt(catalog, select_search_shard_entries(catalog, query=query)))
                        if catalog
                        else auth_receipt({})
                    )
                    coverage_gate = shard_coverage_gate_response(
                        receipt,
                        min_shards_searched=min_shards_searched,
                        min_sources_searched=min_sources_searched,
                        require_complete_search=require_complete_search,
                    )
                    if coverage_gate is not None:
                        status, body = coverage_gate
                        _write_json(self, status, body)
                        return
                    hits, receipt = current_search_with_receipt(
                        query,
                        limit=limit,
                        year_min=year_min,
                        year_max=year_max,
                        rank_mode=rank_mode,
                        timeout_seconds=timeout_seconds,
                    )
                    receipt = auth_receipt(receipt)
                    sweep_status = enqueue_sweep(
                        key=cache_key,
                        query=query,
                        limit=limit,
                        year_min=year_min,
                        year_max=year_max,
                        rank_mode=rank_mode,
                        catalog=catalog,
                    )
            coverage_gate = shard_coverage_gate_response(
                receipt,
                min_shards_searched=min_shards_searched,
                min_sources_searched=min_sources_searched,
                require_complete_search=require_complete_search,
            )
            if coverage_gate is not None:
                status, body = coverage_gate
                _write_json(self, status, body)
                return
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
                        "strategy": _SWEEP_STRATEGY,
                        **sweep_queue_state(cache_key),
                    },
                },
                "results": hits,
            })

        def log_message(self, fmt: str, *args: object) -> None:
            return

    ThreadingHTTPServer.allow_reuse_address = True
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and serve the V5 fullraw FTS index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--index-path", default=_fullraw_env("V5_MEMO_FULL_RAW_INDEX_PATH", "/var/lib/v5-memo/fullraw_index.sqlite"))
    build.add_argument("--manifest", default=_fullraw_env("V5_MEMO_FULL_RAW_MANIFEST", "/var/lib/v5-memo/fullraw_manifest.json"))
    build.add_argument("--refresh-manifest", action="store_true")
    build.add_argument("--rclone-bin", default=_fullraw_env("V5_MEMO_FULL_RAW_RCLONE", "rclone"))
    build.add_argument("--max-files", type=int)
    build.add_argument("--time-budget-seconds", type=float)
    build.add_argument("--commit-interval", type=int, default=1000)
    build.add_argument(
        "--min-free-gb",
        type=float,
        default=float(_fullraw_env("V5_MEMO_FULL_RAW_INDEX_MIN_FREE_GB", "40")),
    )

    shard_dir_default = _fullraw_env(
        "V5_MEMO_FULL_RAW_SHARD_DIR",
        _fullraw_env("V5_MEMO_FULL_RAW_SHARD_BUILD_DIR", "/var/lib/v5-memo/fullraw-shards"),
    )
    build_shards_parser = subparsers.add_parser("build-shards")
    build_shards_parser.add_argument("--shard-dir", default=shard_dir_default)
    build_shards_parser.add_argument("--manifest", default=_fullraw_env("V5_MEMO_FULL_RAW_MANIFEST", "/var/lib/v5-memo/fullraw_manifest.json"))
    build_shards_parser.add_argument("--refresh-manifest", action="store_true")
    build_shards_parser.add_argument("--rclone-bin", default=_fullraw_env("V5_MEMO_FULL_RAW_RCLONE", "rclone"))
    build_shards_parser.add_argument("--shards", type=int, default=int(_fullraw_env("V5_MEMO_FULL_RAW_SHARDS", "4")))
    build_shards_parser.add_argument("--workers", type=int, default=int(_fullraw_env("V5_MEMO_FULL_RAW_SHARD_WORKERS", "4")))
    build_shards_parser.add_argument("--max-files", type=int)
    build_shards_parser.add_argument("--time-budget-seconds", type=float)
    build_shards_parser.add_argument("--commit-interval", type=int, default=1000)
    build_shards_parser.add_argument(
        "--min-free-gb",
        type=float,
        default=float(_fullraw_env("V5_MEMO_FULL_RAW_INDEX_MIN_FREE_GB", "40")),
    )

    build_upload_parser = subparsers.add_parser("build-upload-shards")
    build_upload_parser.add_argument("--shard-dir", default=shard_dir_default)
    build_upload_parser.add_argument("--upload-remote", default=_fullraw_env("V5_MEMO_FULL_RAW_SHARD_REMOTE", ""))
    build_upload_parser.add_argument("--manifest", default=_fullraw_env("V5_MEMO_FULL_RAW_MANIFEST", "/var/lib/v5-memo/fullraw_manifest.json"))
    build_upload_parser.add_argument("--refresh-manifest", action="store_true")
    build_upload_parser.add_argument("--rclone-bin", default=_fullraw_env("V5_MEMO_FULL_RAW_RCLONE", "rclone"))
    build_upload_parser.add_argument("--batch-files", type=int, default=int(_fullraw_env("V5_MEMO_FULL_RAW_SHARD_BATCH_FILES", "16")))
    build_upload_parser.add_argument("--shards", type=int, default=int(_fullraw_env("V5_MEMO_FULL_RAW_SHARDS", "4")))
    build_upload_parser.add_argument("--workers", type=int, default=int(_fullraw_env("V5_MEMO_FULL_RAW_SHARD_WORKERS", "4")))
    build_upload_parser.add_argument("--max-files", type=int)
    build_upload_parser.add_argument("--source-filter", default=_fullraw_env("V5_MEMO_FULL_RAW_SOURCE_FILTER", ""))
    build_upload_parser.add_argument("--batch-id-offset", type=int, default=int(_fullraw_env("V5_MEMO_FULL_RAW_BATCH_ID_OFFSET", "0")))
    build_upload_parser.add_argument("--commit-interval", type=int, default=1000)
    build_upload_parser.add_argument(
        "--min-free-gb",
        type=float,
        default=float(_fullraw_env("V5_MEMO_FULL_RAW_INDEX_MIN_FREE_GB", "40")),
    )
    build_upload_parser.add_argument("--keep-local", action="store_true")

    search = subparsers.add_parser("search")
    search.add_argument("query")
    search.add_argument("--index-path", default=_fullraw_env("V5_MEMO_FULL_RAW_INDEX_PATH", "/var/lib/v5-memo/fullraw_index.sqlite"))
    search.add_argument("--limit", type=int, default=10)

    search_shards_parser = subparsers.add_parser("search-shards")
    search_shards_parser.add_argument("query")
    search_shards_parser.add_argument("--shard-dir", default=shard_dir_default)
    search_shards_parser.add_argument("--limit", type=int, default=10)
    search_shards_parser.add_argument("--year-min", type=int, default=1900)
    search_shards_parser.add_argument("--year-max", type=int, default=2100)

    explain = subparsers.add_parser("explain")
    explain.add_argument("query")
    explain.add_argument("--index-path", default=_fullraw_env("V5_MEMO_FULL_RAW_INDEX_PATH", "/var/lib/v5-memo/fullraw_index.sqlite"))

    stats = subparsers.add_parser("stats")
    stats.add_argument("--index-path", default=_fullraw_env("V5_MEMO_FULL_RAW_INDEX_PATH", "/var/lib/v5-memo/fullraw_index.sqlite"))
    stats.add_argument("--manifest", default=_fullraw_env("V5_MEMO_FULL_RAW_MANIFEST", "/var/lib/v5-memo/fullraw_manifest.json"))

    stats_shards = subparsers.add_parser("stats-shards")
    stats_shards.add_argument("--shard-dir", default=shard_dir_default)
    stats_shards.add_argument("--manifest", default=_fullraw_env("V5_MEMO_FULL_RAW_MANIFEST", "/var/lib/v5-memo/fullraw_manifest.json"))

    backfill_profiles = subparsers.add_parser("backfill-shard-profiles")
    backfill_profiles.add_argument("--shard-dir", default=shard_dir_default)
    backfill_profiles.add_argument("--max-shards", type=int)
    backfill_profiles.add_argument("--force", action="store_true")
    backfill_profiles.add_argument("--dry-run", action="store_true")
    backfill_profiles.add_argument("--upload-remote", default="")
    backfill_profiles.add_argument("--rclone-bin", default=_fullraw_env("V5_MEMO_FULL_RAW_RCLONE", "rclone"))
    backfill_profiles.add_argument("--verify-sqlite", action="store_true")
    backfill_profiles.add_argument("--progress-interval", type=int, default=25)

    warm_cache = subparsers.add_parser("warm-shard-cache")
    warm_cache.add_argument("--shard-dir", default=shard_dir_default)
    warm_cache.add_argument("--query", required=True)
    warm_cache.add_argument(
        "--sweep-shard-limit",
        type=int,
        default=int(_fullraw_env("V5_MEMO_FULL_RAW_SWEEP_SHARD_LIMIT", "96")),
    )
    warm_cache.add_argument(
        "--pass-shard-limit",
        type=int,
        default=int(_fullraw_env("V5_MEMO_FULL_RAW_SWEEP_PASS_SHARD_LIMIT", "10")),
    )
    warm_cache.add_argument(
        "--target-ready",
        type=int,
        default=int(_fullraw_env("V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED", "50")),
    )
    warm_cache.add_argument("--max-shards", type=int)
    warm_cache.add_argument("--max-seconds", type=float)
    warm_cache.add_argument("--cache-dir", default=_fullraw_env("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_DIR", ""))
    warm_cache.add_argument(
        "--cache-max-gb",
        type=float,
        default=_float_or_none(_fullraw_env("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_GB", "")),
    )
    warm_cache.add_argument("--trust-filenames", action="store_true")
    warm_cache.add_argument("--progress-interval", type=int, default=5)

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
        files = _filter_raw_files_by_source(files, str(args.source_filter))
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
            batch_id_offset=max(0, int(args.batch_id_offset)),
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

    if args.command == "warm-shard-cache":
        if args.cache_dir:
            os.environ["V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_DIR"] = str(args.cache_dir)
        if args.cache_max_gb is not None:
            os.environ["V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES"] = str(
                int(max(0.0, float(args.cache_max_gb)) * 1024**3)
            )
        catalog = build_shard_catalog(Path(args.shard_dir), trust_filenames=bool(args.trust_filenames))
        warm_result = warm_shard_cache(
            catalog,
            query=str(args.query),
            sweep_shard_limit=max(1, int(args.sweep_shard_limit)),
            pass_shard_limit=max(1, int(args.pass_shard_limit)),
            target_ready=max(0, int(args.target_ready)),
            max_shards=args.max_shards,
            max_seconds=args.max_seconds,
            progress_interval=max(0, int(args.progress_interval)),
        )
        print(json.dumps(asdict(warm_result), sort_keys=True))
        if warm_result.failed_shards and not warm_result.stopped_for_target:
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


def _with_query_fit_score(hit: dict[str, object], terms: tuple[str, ...]) -> dict[str, object]:
    if not terms:
        return hit
    title = _clean(hit.get("title"))
    abstract = _clean(hit.get("abstract"))
    title_terms = set(_fts_terms(title))
    abstract_terms = set(_fts_terms(abstract))
    title_hits = sum(1 for term in terms if term in title_terms)
    abstract_hits = sum(1 for term in terms if term in abstract_terms)
    phrase_score = _ordered_window_score(_fts_terms(title), terms) * 2 + _ordered_window_score(
        _fts_terms(abstract),
        terms,
    )
    all_title_bonus = 10 if title_hits == len(terms) else 0
    fit_score = 3 * title_hits + abstract_hits + phrase_score + all_title_bonus
    if fit_score <= 0:
        return hit
    return {**hit, "score": round(_hit_score(hit) + fit_score, 6)}


def _ordered_window_score(text_terms: tuple[str, ...], query_terms: tuple[str, ...]) -> int:
    if len(text_terms) < 2 or len(query_terms) < 2:
        return 0
    text_windows = {
        tuple(text_terms[index:index + size])
        for size in range(2, min(3, len(query_terms)) + 1)
        for index in range(0, len(text_terms) - size + 1)
    }
    return sum(
        len(window)
        for size in range(2, min(3, len(query_terms)) + 1)
        for index in range(0, len(query_terms) - size + 1)
        if (window := tuple(query_terms[index:index + size])) in text_windows
    )


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
    sweep_pass_shard_limit: int = 0,
    sweep_max_passes: int = 0,
    sweep_timeout_seconds: float = 0.0,
    sweep_shard_timeout_seconds: float = 0.0,
    sweep_strategy: str = _SWEEP_STRATEGY,
    sweep_catalog_scope: str = "",
) -> str:
    _ = limit  # Result sufficiency is receipt-gated; the work key should not fragment on it.
    payload = json.dumps(
        {
            "query": query,
            "year_min": year_min,
            "year_max": year_max,
            "rank_mode": _rank_mode(rank_mode),
            "sweep_shard_limit": sweep_shard_limit,
            "sweep_strategy": sweep_strategy,
            "sweep_catalog_scope": sweep_catalog_scope,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sweep_cache_path(cache_dir: Path | None, key: str) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / f"{key}.json"


def _sweep_completed_path_strings(receipt: dict[str, object]) -> set[str]:
    value = receipt.get("sweep_completed_paths")
    if not isinstance(value, list | tuple):
        return set()
    return {str(path) for path in value if str(path)}


def _sweep_failed_path_strings(receipt: dict[str, object]) -> set[str]:
    if receipt.get("sweep_timed_out") is True:
        return set()
    value = receipt.get("sweep_failed_paths")
    if not isinstance(value, list | tuple):
        return set()
    return {str(path) for path in value if str(path)}


def _sweep_failed_path_strings_for_mode(
    receipt: dict[str, object],
    *,
    require_complete_sweep: bool,
) -> set[str]:
    if require_complete_sweep:
        return set()
    return _sweep_failed_path_strings(receipt)


def _sweep_pass_failed_path_strings(
    pass_entries: Iterable[ShardCatalogEntry],
    *,
    completed_path_strings: set[str],
    existing_failed_path_strings: set[str],
    require_complete_sweep: bool,
) -> set[str]:
    if require_complete_sweep:
        return set()
    return {
        str(entry.path)
        for entry in pass_entries
        if str(entry.path) not in completed_path_strings | existing_failed_path_strings
    }


def _next_sweep_pass_entries(
    sweep_entries: list[ShardCatalogEntry],
    *,
    completed_path_strings: set[str],
    failed_path_strings: set[str],
    deferred_path_strings: set[str],
    limit: int,
) -> tuple[list[ShardCatalogEntry], set[str]]:
    blocked = completed_path_strings | failed_path_strings | deferred_path_strings
    remaining_entries = [entry for entry in sweep_entries if str(entry.path) not in blocked]
    if not remaining_entries and deferred_path_strings:
        deferred_path_strings = set()
        blocked = completed_path_strings | failed_path_strings
        remaining_entries = [entry for entry in sweep_entries if str(entry.path) not in blocked]
    return remaining_entries[:limit], deferred_path_strings


def _sweep_remaining_shard_count(
    *,
    selected_shards: int,
    completed_shards: int,
    failed_shards: int,
    require_complete_sweep: bool,
) -> int:
    outstanding = selected_shards - completed_shards
    if not require_complete_sweep:
        outstanding -= failed_shards
    return max(0, outstanding)


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
    lock_path = path.with_name(f"{path.name}.lock")
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with lock_path.open("a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            result_limit = _int_or_none(entry.receipt.get("sweep_result_limit"))
            if result_limit is None and entry.hits:
                result_limit = len(entry.hits)
                entry = SweepCacheEntry(
                    created_at=entry.created_at,
                    hits=entry.hits,
                    receipt={**entry.receipt, "sweep_result_limit": result_limit},
                )
            current = _load_sweep_cache(path, ttl_seconds=0) if path.exists() else None
            if current is not None and result_limit is not None and not _sweep_cache_entry_has_result_limit(
                current,
                result_limit,
            ):
                current = None
            selected = _prefer_sweep_cache_entry(entry, current)
            assert selected is not None
            _write_json_file(tmp_path, {
                "created_at": selected.created_at,
                "hits": selected.hits,
                "receipt": selected.receipt,
            })
            os.replace(tmp_path, path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            with suppress(OSError):
                tmp_path.unlink()


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
    raw = _fullraw_env(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _sweep_cache_inflight_lanes(max_inflight: int) -> int:
    lanes = max(1, max_inflight)
    priority_burst = _fullraw_env("V5_MEMO_FULL_RAW_SWEEP_PRIORITY_BURST", "true").casefold()
    if priority_burst in {"1", "true", "yes"}:
        lanes += 1
    return lanes


def _auto_sweep_workers(max_inflight: int) -> int:
    cpu_workers = max(1, (os.cpu_count() or 1) // max(1, max_inflight))
    max_cache_bytes = _shard_local_cache_max_bytes()
    worker_cache_bytes = _sweep_worker_cache_bytes()
    if max_cache_bytes is None or worker_cache_bytes is None or worker_cache_bytes <= 0:
        return cpu_workers
    cache_workers = max(1, max_cache_bytes // (worker_cache_bytes * _sweep_cache_inflight_lanes(max_inflight)))
    return max(1, min(cpu_workers, cache_workers))


def _sweep_worker_cache_bytes(*, default_gb: float | None = None) -> int | None:
    value = _positive_int_env("V5_MEMO_FULL_RAW_SWEEP_WORKER_CACHE_BYTES")
    if value is not None:
        return value
    raw_gb = _fullraw_env("V5_MEMO_FULL_RAW_SWEEP_WORKER_CACHE_GB", "")
    per_worker_gb = _float_or_none(raw_gb)
    if per_worker_gb is None:
        per_worker_gb = default_gb
    return int(per_worker_gb * 1024 * 1024 * 1024) if per_worker_gb else None


def _float_or_none(value: object) -> float | None:
    if not isinstance(value, int | float | str | bytes | bytearray):
        return None
    try:
        return float(value) if value not in {"", None} else None
    except (TypeError, ValueError):
        return None

if __name__ == "__main__":
    main()
