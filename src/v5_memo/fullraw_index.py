"""Persistent FTS index for the full raw corpus.

The cold scanner proves the raw archive is reachable, but it returns early hits
in file order. This module builds a durable SQLite FTS5 index so memo retrieval
can rank by relevance instead of archive order.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from v5_memo.fullraw_service import RawFile, iter_raw_file_hits, load_or_build_manifest

_WORD = re.compile(r"[A-Za-z0-9]+")
_STOP = {"and", "or", "the", "with", "for", "from", "into", "this", "that"}
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


class FullRawFtsIndex:
    """SQLite FTS5 index over normalized raw-corpus records."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), timeout=60.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()

    def close(self) -> None:
        self._conn.close()

    def initialize(self) -> None:
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
    ) -> list[dict[str, object]]:
        with self._lock:
            self.initialize()
            terms = _fts_terms(query)
            if not terms:
                return []
            match_query = _fts_match_query(self._expanded_term_groups(terms))
            rows = self._conn.execute(
                """
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
                ORDER BY rank ASC, COALESCE(p.cited_by_count, 0) DESC
                LIMIT ?
                """,
                (match_query, year_min, year_max, max(1, min(limit, 200))),
            ).fetchall()
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
            return False
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
) -> list[dict[str, object]]:
    """Search multiple shard DBs and merge ranked, deduped results."""
    merged: dict[str, dict[str, object]] = {}
    for path in index_paths:
        if not path.exists():
            continue
        index = FullRawFtsIndex(path)
        try:
            hits = index.search(query, limit=limit, year_min=year_min, year_max=year_max)
        finally:
            index.close()
        for hit in hits:
            key = _dedupe_key(hit)
            existing = merged.get(key)
            if existing is None or _hit_score(hit) > _hit_score(existing):
                merged[key] = hit
    return sorted(
        merged.values(),
        key=lambda hit: (_hit_score(hit), _int_or_none(hit.get("cited_by_count")) or 0),
        reverse=True,
    )[: max(1, min(limit, 200))]


def discover_shard_paths(shard_dir: Path) -> list[Path]:
    return sorted(path for path in shard_dir.glob("*.sqlite") if path.is_file())


def aggregate_shard_stats(index_paths: list[Path], *, files_total: int = 0) -> IndexStats:
    papers_indexed = 0
    files_indexed = 0
    bytes_used = 0
    for path in index_paths:
        if not path.exists():
            continue
        index = FullRawFtsIndex(path)
        try:
            stats = index.stats(files_total=0)
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
        file_errors=result.file_errors,
    )


def run_server() -> None:
    host = os.environ.get("V5_MEMO_FULL_RAW_INDEX_HOST", "127.0.0.1")
    port = int(os.environ.get("V5_MEMO_FULL_RAW_INDEX_PORT", "9902"))
    index_path = Path(
        os.environ.get("V5_MEMO_FULL_RAW_INDEX_PATH", "/var/lib/v5-memo/fullraw_index.sqlite")
    )
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
    index = FullRawFtsIndex(index_path)
    index.initialize()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404)
                return
            stats = index.stats(files_total=len(files))
            _write_json(self, 200, {
                "ok": True,
                "backend": _BACKEND,
                "index_path": str(index_path),
                "papers_indexed": stats.papers_indexed,
                "files_indexed": stats.files_indexed,
                "files_total": stats.files_total,
                "complete": stats.complete,
                "bytes_used": stats.bytes_used,
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
            except (TypeError, ValueError, json.JSONDecodeError):
                _write_json(self, 400, {"error": "bad request"})
                return
            if not query:
                _write_json(self, 400, {"error": "query is required"})
                return
            started = time.monotonic()
            hits = index.search(query, limit=limit, year_min=year_min, year_max=year_max)
            stats = index.stats(files_total=len(files))
            explain = index.explain_query(query)
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


def _free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(path)
    return int(usage.free)


def _fts_terms(query: str) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in _WORD.findall(query.casefold()):
        if len(token) <= 1 or token in _STOP or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return tuple(terms)


def _first_fts_term(value: str) -> str:
    terms = _fts_terms(value)
    return terms[0] if terms else ""


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
