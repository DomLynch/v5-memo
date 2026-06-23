from __future__ import annotations

import gzip
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pytest import MonkeyPatch

from v5_memo import fullraw_index
from v5_memo.fullraw_index import (
    FullRawFtsIndex,
    ShardBatchResult,
    ShardCatalogEntry,
    SweepScheduler,
    SweepTask,
    aggregate_shard_manifest_stats,
    aggregate_shard_stats,
    backfill_shard_profiles,
    build_shard_catalog,
    build_shards,
    build_upload_shard_batches,
    discover_shard_paths,
    search_shards,
    select_search_shard_entries,
    select_search_shard_paths,
    select_sweep_shard_entries,
    shard_coverage_gate_response,
    shard_coverage_receipt,
    warm_shard_cache,
)
from v5_memo.fullraw_service import RawFile


def _write_jsonl_gzip(path: Path, rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wb") as fh:
        for row in rows:
            fh.write((json.dumps(row) + "\n").encode("utf-8"))


def _raw_file(
    tmp_path: Path,
    name: str,
    rows: list[dict[str, object]],
    *,
    source: str = "openalex",
) -> RawFile:
    path = tmp_path / f"{name}.jsonl.gz"
    _write_jsonl_gzip(path, rows)
    return RawFile(source=source, format="openalex_jsonl", remote=f"file://{path}")


def _corrupt_raw_file(tmp_path: Path, name: str) -> RawFile:
    source = tmp_path / f"{name}.jsonl.gz"
    source.write_bytes(gzip.compress(b'{"display_name":"truncated gzip"}\n')[:-8])
    return RawFile(source="openalex", format="openalex_jsonl", remote=f"file://{source}")


def _write_search_test_batch(
    tmp_path: Path,
    index: int,
    *,
    source: str = "openalex",
    valid: bool = True,
) -> None:
    batch = tmp_path / f"batch_{index:05d}"
    batch.mkdir()
    shard = batch / "fullraw_shard_0000.sqlite"
    profile: dict[str, object] = {
        "year_min": 2024,
        "year_max": 2024,
        "cited_by_min": 10,
        "cited_by_max": 10,
        "cited_by_avg": 10.0,
        "topic_terms": ["management", "forecast", "disclosure"],
    }
    if valid:
        raw_file = _raw_file(tmp_path, f"complete_search_{index}", [{
            "doi": f"https://doi.org/10.example/complete-search-{index}",
            "display_name": f"Management forecast disclosure complete search {index}",
            "abstract": "Management forecast disclosure evidence.",
            "publication_year": 2024,
            "cited_by_count": 10 + index,
        }], source=source)
        index_db = FullRawFtsIndex(shard)
        try:
            index_db.index_files([raw_file], commit_interval=1)
            profile = index_db.profile()
        finally:
            index_db.close()
    else:
        shard.write_text("not sqlite")
    (batch / "complete.json").write_text(json.dumps({
        "batch_id": index,
        "files": [{"source": source}],
        "shards": [{
            "shard_id": 0,
            "files_completed": 1,
            "papers_inserted": 1,
            "bytes_used": shard.stat().st_size,
            **profile,
        }],
    }))


def test_fullraw_index_builds_ranked_queryable_index(tmp_path: Path) -> None:
    raw_file = _raw_file(tmp_path, "openalex", [
        {
            "doi": "https://doi.org/10.2308/tar-9603274096",
            "display_name": "Factors Associated with the Disclosure of Managers' Forecasts",
            "abstract": (
                "Managers disclose forecasts of future earnings when analyst forecast "
                "errors and ownership structure make disclosure useful."
            ),
            "publication_year": 1990,
            "venue": "The Accounting Review",
            "cited_by_count": 110,
        },
        {
            "doi": "https://doi.org/10.2308/tar-4483133",
            "display_name": "Earnings Releases, Anomalies, and the Behavior of Security Returns",
            "abstract": "Earnings forecast error and firm size explain drift variation.",
            "publication_year": 1984,
            "venue": "The Accounting Review",
            "cited_by_count": 200,
        },
        {
            "doi": "https://doi.org/10.noise/ecology",
            "display_name": "Climate space forecasts for island species",
            "abstract": "Vegetation and soil stability forecasts under grazing pressure.",
            "publication_year": 2020,
        },
    ])
    index = FullRawFtsIndex(tmp_path / "fullraw.sqlite")
    try:
        result = index.index_files([raw_file])
        hits = index.search("management forecast disclosure", limit=5)
        explain = index.explain_query("management forecast disclosure")
        stats = index.stats(files_total=1)
    finally:
        index.close()

    assert result.files_completed == 1
    assert result.papers_inserted == 3
    assert stats.papers_indexed == 3
    assert stats.files_indexed == 1
    assert hits[0]["doi"] == "10.2308/tar-9603274096"
    assert hits[0]["source"] == "openalex"
    fts_match = explain["fts_match"]
    assert isinstance(fts_match, str)
    assert "managers" in fts_match
    assert "forecasts" in fts_match
    assert "discloses" in fts_match


def test_fullraw_index_enriches_semantic_scholar_abstract_rows(tmp_path: Path) -> None:
    paper = tmp_path / "s2_paper.jsonl.gz"
    abstract = tmp_path / "s2_abstract.jsonl.gz"
    _write_jsonl_gzip(paper, [{
        "corpusid": 12345,
        "title": "Resveratrol exercise training adaptation",
        "year": 2024,
    }])
    _write_jsonl_gzip(abstract, [{
        "corpusid": 12345,
        "abstract": "Resveratrol exercise training blunted mitochondrial adaptation in older adults.",
    }])
    index = FullRawFtsIndex(tmp_path / "fullraw.sqlite")
    try:
        result = index.index_files([
            RawFile(source="semantic_scholar", format="semantic_scholar_jsonl", remote=f"file://{paper}"),
            RawFile(source="semantic_scholar_abstracts", format="semantic_scholar_jsonl", remote=f"file://{abstract}"),
        ])
        hits = index.search("mitochondrial older adults", limit=5)
    finally:
        index.close()

    assert result.files_completed == 2
    assert result.papers_inserted == 2
    assert hits[0]["semantic_scholar_id"] == "12345"
    assert "blunted mitochondrial adaptation" in str(hits[0]["abstract"])


@pytest.mark.parametrize(
    ("column", "index_name"),
    [
        ("doi", "idx_papers_doi"),
        ("pmid", "idx_papers_pmid"),
        ("pmcid", "idx_papers_pmcid"),
        ("openalex_id", "idx_papers_openalex_id"),
        ("semantic_scholar_id", "idx_papers_semantic_scholar_id"),
    ],
)
def test_fullraw_index_delays_identifier_indexes_until_abstract_merge(
    tmp_path: Path,
    column: str,
    index_name: str,
) -> None:
    index = FullRawFtsIndex(tmp_path / "fullraw.sqlite")
    try:
        index.initialize()
        index_rows = index._conn.execute("PRAGMA index_list('papers')").fetchall()
        index_names = {str(row["name"]) for row in index_rows}
        assert index_name not in index_names
        assert index._update_hit_abstract({
            "abstract": "Identifier indexes should exist before this lookup.",
            column: "identifier-value",
        }) is False
        index_rows = index._conn.execute("PRAGMA index_list('papers')").fetchall()
        index_names = {str(row["name"]) for row in index_rows}
        plan_rows = index._conn.execute(
            f"EXPLAIN QUERY PLAN SELECT id FROM papers WHERE {column} = ? LIMIT 1",
            ("identifier-value",),
        ).fetchall()
    finally:
        index.close()

    assert index_name in index_names
    assert any(index_name in str(row["detail"]) for row in plan_rows)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("doi", "10.example/abstract-merge"),
        ("pmid", "123456"),
        ("pmcid", "PMC123456"),
        ("openalex_id", "https://openalex.org/W123456"),
        ("semantic_scholar_id", "S2123456"),
    ],
)
def test_fullraw_index_updates_empty_abstract_by_any_identifier(
    tmp_path: Path,
    column: str,
    value: str,
) -> None:
    index = FullRawFtsIndex(tmp_path / "fullraw.sqlite")
    try:
        index.initialize()
        hit: dict[str, object] = {
            "title": f"Identifier merge paper {column}",
            "abstract": "",
            "source": "test",
            column: value,
        }
        assert index._insert_hit(hit, source_remote=f"file://paper-{column}.jsonl.gz")
        assert index._update_hit_abstract({
            "abstract": f"Updated abstract through {column} lookup.",
            column: value,
        })
        row = index._conn.execute(
            f"SELECT abstract FROM papers WHERE {column} = ?",
            (value,),
        ).fetchone()
    finally:
        index.close()

    assert row is not None
    assert row["abstract"] == f"Updated abstract through {column} lookup."


def test_fullraw_index_ignores_abstract_rows_without_identifiers(tmp_path: Path) -> None:
    index = FullRawFtsIndex(tmp_path / "fullraw.sqlite")
    try:
        index.initialize()
        assert not index._update_hit_abstract({"abstract": "No stable identifier."})
        count = index._conn.execute("SELECT COUNT(*) AS count FROM papers").fetchone()["count"]
    finally:
        index.close()

    assert count == 0


def test_fullraw_index_uses_persisted_custom_term_map(tmp_path: Path) -> None:
    raw_file = _raw_file(tmp_path, "openalex", [{
        "doi": "https://doi.org/10.example/guidance",
        "display_name": "Management guidance and earnings surprises",
        "abstract": "Managers issue guidance before earnings surprises.",
        "publication_year": 2024,
    }])
    index = FullRawFtsIndex(tmp_path / "fullraw.sqlite")
    try:
        index.index_files([raw_file])
        assert index.search("projection", limit=5) == []

        index.upsert_term_map("projection", ("guidance",), source="test")
        hits = index.search("projection", limit=5)
        explain = index.explain_query("projection")
    finally:
        index.close()

    assert hits[0]["doi"] == "10.example/guidance"
    fts_match = explain["fts_match"]
    assert isinstance(fts_match, str)
    assert "guidance" in fts_match


def test_fullraw_index_is_resumable_and_dedupes_completed_files(tmp_path: Path) -> None:
    raw_file = _raw_file(tmp_path, "openalex", [{
        "doi": "https://doi.org/10.example/one",
        "display_name": "Management forecast disclosure",
        "abstract": "Forecast disclosure and earnings forecast error.",
    }])
    index = FullRawFtsIndex(tmp_path / "fullraw.sqlite")
    try:
        first = index.index_files([raw_file])
        second = index.index_files([raw_file])
        stats = index.stats(files_total=1)
    finally:
        index.close()

    assert first.papers_inserted == 1
    assert second.files_attempted == 0
    assert second.papers_inserted == 0
    assert stats.papers_indexed == 1


@pytest.mark.parametrize(
    ("min_free_bytes", "stopped", "attempted", "inserted"),
    [(10**18, True, 0, 0), (0, False, 1, 1)],
)
def test_fullraw_index_disk_guard(
    tmp_path: Path,
    min_free_bytes: int,
    stopped: bool,
    attempted: int,
    inserted: int,
) -> None:
    raw_file = _raw_file(tmp_path, "openalex", [{
        "doi": "https://doi.org/10.example/one",
        "display_name": "Management forecast disclosure",
        "abstract": "Forecast disclosure and earnings forecast error.",
    }])
    index = FullRawFtsIndex(tmp_path / "fullraw.sqlite")
    try:
        result = index.index_files([raw_file], min_free_bytes=min_free_bytes)
        stats = index.stats(files_total=1)
    finally:
        index.close()

    assert result.stopped_for_budget is stopped
    assert result.files_attempted == attempted
    assert result.papers_inserted == inserted
    assert stats.papers_indexed == inserted


def test_fullraw_index_quarantines_corrupt_source_file(tmp_path: Path) -> None:
    good = _raw_file(tmp_path, "good", [{
        "doi": "https://doi.org/10.example/good",
        "display_name": "Management forecast disclosure",
        "abstract": "Managers disclose forecasts and guidance.",
    }])
    bad = _corrupt_raw_file(tmp_path, "bad")
    index = FullRawFtsIndex(tmp_path / "fullraw.sqlite")
    try:
        result = index.index_files([bad, good], commit_interval=1)
        stats = index.stats(files_total=2)
    finally:
        index.close()

    assert result.files_attempted == 2
    assert result.files_completed == 1
    assert result.files_failed == 1
    assert result.papers_inserted == 1
    assert result.stopped_for_budget is False
    assert "bad.jsonl.gz" in result.file_errors
    assert stats.files_indexed == 1
    assert stats.papers_indexed == 1


def test_disk_guard_checks_index_filesystem(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    checked_paths: list[Path] = []

    def fake_disk_usage(path: Path) -> SimpleNamespace:
        checked_paths.append(path)
        return SimpleNamespace(total=100, used=40, free=60)

    def no_records(_raw_file: RawFile, *, rclone_bin: str = "rclone") -> Iterator[dict[str, object]]:
        del rclone_bin
        yield from ()

    monkeypatch.setattr("v5_memo.fullraw_index.shutil.disk_usage", fake_disk_usage)
    monkeypatch.setattr("v5_memo.fullraw_index.iter_raw_file_hits", no_records)

    index_path = tmp_path / "nested" / "fullraw.sqlite"
    index = FullRawFtsIndex(index_path)
    try:
        index.index_files([RawFile(source="openalex", format="openalex_jsonl", remote="file:///unused.gz")], min_free_bytes=1)
    finally:
        index.close()

    assert checked_paths == [index_path.parent]


def test_parallel_shard_build_and_search(tmp_path: Path) -> None:
    files: list[RawFile] = []
    rows = [
        {
            "doi": "https://doi.org/10.2308/tar-9603274096",
            "display_name": "Factors Associated with the Disclosure of Managers' Forecasts",
            "abstract": "Managers disclose forecasts of future earnings.",
            "publication_year": 1990,
            "cited_by_count": 110,
        },
        {
            "doi": "https://doi.org/10.example/analyst",
            "display_name": "Analyst forecast disclosure and management guidance",
            "abstract": "Analysts use management guidance for forecast disclosure.",
            "publication_year": 2020,
            "cited_by_count": 10,
        },
        {
            "doi": "https://doi.org/10.example/noise",
            "display_name": "Island species forecast ecology",
            "abstract": "Forecasts for climate space under grazing pressure.",
            "publication_year": 2021,
        },
        {
            "doi": "https://doi.org/10.2308/tar-9603274096",
            "display_name": "Duplicate management forecast disclosure",
            "abstract": "Duplicate receipt should be deduped across shards.",
            "publication_year": 1990,
        },
    ]
    for index, row in enumerate(rows):
        files.append(_raw_file(tmp_path, f"openalex_{index}", [row]))

    results = build_shards(
        files,
        shard_dir=tmp_path / "shards",
        shard_count=2,
        workers=2,
        commit_interval=1,
    )
    shard_paths = discover_shard_paths(tmp_path / "shards")
    stats = aggregate_shard_stats(shard_paths, files_total=len(files))
    hits = search_shards(shard_paths, "management forecast disclosure", limit=5)

    assert len(results) == 2
    assert sum(result.files_completed for result in results) == 4
    assert sum(result.papers_inserted for result in results) == 4
    assert len(shard_paths) == 2
    assert stats.files_indexed == 4
    assert stats.papers_indexed == 4
    assert {hit["doi"] for hit in hits} >= {"10.2308/tar-9603274096", "10.example/analyst"}
    assert len([hit for hit in hits if hit["doi"] == "10.2308/tar-9603274096"]) == 1


def test_fullraw_shard_search_returns_partial_hits_on_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = [tmp_path / "fast.sqlite", tmp_path / "slow.sqlite"]
    for path in paths:
        path.touch()

    def fake_search_one_shard(
        path: Path,
        query: str,
            limit: int,
            year_min: int,
            year_max: int,
            rank_mode: str,
            timeout_seconds: float | None = None,
        ) -> list[dict[str, object]]:
        del query, limit, year_min, year_max, rank_mode, timeout_seconds
        if path.name == "slow.sqlite":
            time.sleep(0.2)
        return [{
            "doi": f"10.example/{path.stem}",
            "title": path.stem,
            "score": 1.0,
        }]

    monkeypatch.setattr(fullraw_index, "_search_one_shard", fake_search_one_shard)
    started = time.monotonic()

    hits = fullraw_index._search_shard_paths(
        paths,
        "resveratrol exercise",
        limit=5,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        workers=2,
        timeout_seconds=0.05,
    )

    assert time.monotonic() - started < 0.15
    assert [hit["doi"] for hit in hits] == ["10.example/fast"]


def test_fullraw_shard_search_reports_completed_paths_on_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fast = tmp_path / "fast.sqlite"
    slow = tmp_path / "slow.sqlite"
    fast.touch()
    slow.touch()

    def fake_search_one_shard(
        path: Path,
        query: str,
            limit: int,
            year_min: int,
            year_max: int,
            rank_mode: str,
            timeout_seconds: float | None = None,
        ) -> list[dict[str, object]]:
        del query, limit, year_min, year_max, rank_mode, timeout_seconds
        if path == slow:
            time.sleep(0.2)
        return [{
            "doi": f"10.example/{path.stem}",
            "title": path.stem,
            "score": 1.0,
        }]

    monkeypatch.setattr(fullraw_index, "_search_one_shard", fake_search_one_shard)

    hits, completed_paths, timed_out = fullraw_index._search_shard_paths_with_paths(
        [fast, slow],
        "resveratrol exercise",
        limit=5,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        workers=2,
        timeout_seconds=0.05,
    )

    assert timed_out is True
    assert completed_paths == [fast]
    assert [hit["doi"] for hit in hits] == ["10.example/fast"]


def test_fullraw_shard_search_passes_per_shard_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "fast.sqlite"
    path.touch()
    seen_timeout: list[float | None] = []

    def fake_search_one_shard(
        path: Path,
        query: str,
        limit: int,
        year_min: int,
        year_max: int,
        rank_mode: str,
        timeout_seconds: float | None = None,
    ) -> list[dict[str, object]]:
        del path, query, limit, year_min, year_max, rank_mode
        seen_timeout.append(timeout_seconds)
        return [{"doi": "10.example/fast", "title": "fast", "score": 1.0}]

    monkeypatch.setattr(fullraw_index, "_search_one_shard", fake_search_one_shard)

    fullraw_index._search_shard_paths(
        [path],
        "resveratrol exercise",
        limit=5,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        workers=1,
        shard_timeout_seconds=3.5,
    )

    assert seen_timeout == [3.5]


def test_search_one_shard_materializes_local_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = tmp_path / "remote" / "batch_00000" / "fullraw_shard_0000.sqlite"
    raw_file = _raw_file(tmp_path, "materialized_cache", [{
        "doi": "https://doi.org/10.example/materialized-cache",
        "display_name": "Management forecast disclosure materialized",
        "abstract": "Management forecast disclosure local cache evidence.",
        "publication_year": 2024,
    }])
    index = FullRawFtsIndex(shard)
    try:
        index.index_files([raw_file], commit_interval=1)
    finally:
        index.close()
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_DIR", str(cache_dir))

    hits = fullraw_index._search_one_shard(
        shard,
        "management forecast disclosure",
        5,
        1900,
        2100,
        "relevance",
        1.0,
    )

    cached = list(cache_dir.glob("*.sqlite"))
    assert len(cached) == 1
    assert cached[0].read_bytes() == shard.read_bytes()
    assert [hit["doi"] for hit in hits] == ["10.example/materialized-cache"]


def test_warm_shard_cache_materializes_source_balanced_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries: list[ShardCatalogEntry] = []
    for index, source in enumerate(("openalex", "openalex", "pubmed", "semantic_scholar")):
        shard = tmp_path / f"batch_{index:05d}" / "fullraw_shard_0000.sqlite"
        shard.parent.mkdir()
        shard.write_bytes(f"shard-{index}".encode())
        entries.append(ShardCatalogEntry(
            path=shard,
            batch_id=index,
            shard_id=0,
            sources=(source,),
            files_completed=1,
            papers_inserted=10,
            bytes_used=shard.stat().st_size,
            topic_terms=("pregnancy", "management"),
        ))
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_DIR", str(cache_dir))

    result = warm_shard_cache(
        entries,
        query="pregnancy management",
        sweep_shard_limit=4,
        pass_shard_limit=3,
        target_ready=3,
    )

    assert result.stopped_for_target is True
    assert result.ready_shards == 3
    assert result.warmed_shards == 3
    assert result.sources_ready == {"openalex": 1, "pubmed": 1, "semantic_scholar": 1}
    assert len(list(cache_dir.glob("*.sqlite"))) == 3


def test_warm_shard_cache_cli_sets_cache_and_reports_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, object] = {}

    def fake_build_shard_catalog(shard_dir: Path, *, trust_filenames: bool = False) -> list[ShardCatalogEntry]:
        calls["catalog"] = {"shard_dir": str(shard_dir), "trust_filenames": trust_filenames}
        return []

    def fake_warm_shard_cache(
        entries: list[ShardCatalogEntry],
        *,
        query: str,
        sweep_shard_limit: int,
        pass_shard_limit: int,
        target_ready: int,
        max_shards: int | None = None,
        max_seconds: float | None = None,
        progress_interval: int = 0,
    ) -> fullraw_index.ShardCacheWarmResult:
        calls["warm"] = {
            "entries": entries,
            "query": query,
            "sweep_shard_limit": sweep_shard_limit,
            "pass_shard_limit": pass_shard_limit,
            "target_ready": target_ready,
            "max_shards": max_shards,
            "max_seconds": max_seconds,
            "progress_interval": progress_interval,
        }
        return fullraw_index.ShardCacheWarmResult(
            selected_shards=12,
            target_ready=5,
            ready_shards=5,
            warmed_shards=5,
            failed_shards=0,
            stopped_for_target=True,
            stopped_for_time=False,
            elapsed_seconds=1.25,
            bytes_ready=123,
            sources_selected={"openalex": 6, "semantic_scholar": 6},
            sources_ready={"openalex": 3, "semantic_scholar": 2},
        )

    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(fullraw_index, "build_shard_catalog", fake_build_shard_catalog)
    monkeypatch.setattr(fullraw_index, "warm_shard_cache", fake_warm_shard_cache)
    monkeypatch.setattr(sys, "argv", [
        "fullraw_index.py",
        "warm-shard-cache",
        "--shard-dir",
        str(tmp_path / "shards"),
        "--query",
        "cholestasis pregnancy management",
        "--sweep-shard-limit",
        "12",
        "--pass-shard-limit",
        "3",
        "--target-ready",
        "5",
        "--max-shards",
        "7",
        "--max-seconds",
        "9",
        "--cache-dir",
        str(cache_dir),
        "--cache-max-gb",
        "1",
        "--trust-filenames",
        "--progress-interval",
        "0",
    ])

    fullraw_index.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["ready_shards"] == 5
    assert calls["catalog"] == {"shard_dir": str(tmp_path / "shards"), "trust_filenames": True}
    assert calls["warm"] == {
        "entries": [],
        "query": "cholestasis pregnancy management",
        "sweep_shard_limit": 12,
        "pass_shard_limit": 3,
        "target_ready": 5,
        "max_shards": 7,
        "max_seconds": 9.0,
        "progress_interval": 0,
    }
    assert os.environ["V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_DIR"] == str(cache_dir)
    assert os.environ["V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES"] == str(1024**3)


def test_materialized_shard_cache_evicts_old_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    old = cache_dir / "old.sqlite"
    newer = cache_dir / "newer.sqlite"
    keep = cache_dir / "keep.sqlite"
    old.write_bytes(b"a" * 6)
    newer.write_bytes(b"b" * 6)
    keep.write_bytes(b"c" * 6)
    os.utime(old, (1, 1))
    os.utime(newer, (2, 2))
    os.utime(keep, (3, 3))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES", "12")

    fullraw_index._evict_shard_cache(cache_dir, required_bytes=0, keep=keep)

    assert not old.exists()
    assert newer.exists()
    assert keep.exists()


def test_materialized_shard_cache_preserves_ready_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    old = cache_dir / "old.sqlite"
    preserved = cache_dir / "preserved.sqlite"
    keep = cache_dir / "keep.sqlite"
    old.write_bytes(b"a" * 6)
    preserved.write_bytes(b"b" * 6)
    keep.write_bytes(b"c" * 6)
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES", "12")

    fullraw_index._evict_shard_cache(
        cache_dir,
        required_bytes=0,
        keep=keep,
        preserve={preserved},
    )

    assert not old.exists()
    assert preserved.exists()
    assert keep.exists()


def test_warm_shard_cache_stops_before_eviction_churn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries: list[ShardCatalogEntry] = []
    for index, source in enumerate(("openalex", "pubmed", "semantic_scholar")):
        shard = tmp_path / f"batch_{index:05d}" / "fullraw_shard_0000.sqlite"
        shard.parent.mkdir()
        shard.write_bytes(b"x" * 8)
        entries.append(ShardCatalogEntry(
            path=shard,
            batch_id=index,
            shard_id=0,
            sources=(source,),
            files_completed=1,
            papers_inserted=10,
            bytes_used=shard.stat().st_size,
            topic_terms=("pregnancy",),
        ))
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES", "16")

    result = warm_shard_cache(
        entries,
        query="pregnancy",
        sweep_shard_limit=3,
        pass_shard_limit=3,
        target_ready=3,
    )

    assert result.ready_shards == 2
    assert result.stopped_for_target is False
    assert result.errors
    assert "target_ready exceeds cache budget" in result.errors[0]
    assert len(list(cache_dir.glob("*.sqlite"))) == 2


def test_warm_shard_cache_prefers_cache_fit_source_breadth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [
        ("openalex", 28, ("pregnancy", "management")),
        ("openalex", 8, ("adjacent",)),
        ("pubmed", 9, ("pregnancy",)),
        ("semantic_scholar", 29, ("pregnancy", "management")),
        ("semantic_scholar", 10, ("adjacent",)),
    ]
    entries: list[ShardCatalogEntry] = []
    for index, (source, size, topic_terms) in enumerate(specs):
        shard = tmp_path / f"batch_{index:05d}" / "fullraw_shard_0000.sqlite"
        shard.parent.mkdir()
        shard.write_bytes(b"x" * size)
        entries.append(ShardCatalogEntry(
            path=shard,
            batch_id=index,
            shard_id=0,
            sources=(source,),
            files_completed=1,
            papers_inserted=10,
            bytes_used=size,
            cited_by_max=1000 if "management" in topic_terms else 1,
            topic_terms=topic_terms,
        ))
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES", "27")

    result = warm_shard_cache(
        entries,
        query="pregnancy management",
        sweep_shard_limit=5,
        pass_shard_limit=3,
        target_ready=3,
    )

    cached_files = list(cache_dir.glob("*.sqlite"))
    cached_sizes = sorted(path.stat().st_size for path in cached_files)
    assert result.stopped_for_target is True
    assert result.ready_shards == 3
    assert result.bytes_ready == 27
    assert result.sources_ready == {"openalex": 1, "pubmed": 1, "semantic_scholar": 1}
    assert len(cached_files) == result.ready_shards
    assert all(size > 0 for size in cached_sizes)
    assert sum(cached_sizes) == result.bytes_ready
    assert cached_sizes == [8, 9, 10]


def test_cache_fit_orders_tail_by_smaller_replacements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_large = ShardCatalogEntry(
        path=tmp_path / "old-large.sqlite",
        batch_id=0,
        shard_id=0,
        sources=("openalex",),
        files_completed=1,
        papers_inserted=1,
        bytes_used=100,
        cited_by_max=1000,
        topic_terms=("management",),
    )
    old_medium = ShardCatalogEntry(
        path=tmp_path / "old-medium.sqlite",
        batch_id=1,
        shard_id=0,
        sources=("openalex",),
        files_completed=1,
        papers_inserted=1,
        bytes_used=50,
        cited_by_max=10,
        topic_terms=("management",),
    )
    replacement_small = ShardCatalogEntry(
        path=tmp_path / "replacement-small.sqlite",
        batch_id=2,
        shard_id=0,
        sources=("openalex",),
        files_completed=1,
        papers_inserted=1,
        bytes_used=5,
        cited_by_max=1,
        topic_terms=("adjacent",),
    )
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES", "5")

    ordered = fullraw_index._cache_fit_warm_entries(
        [old_large, old_medium, replacement_small],
        [old_large, old_medium],
        query="management forecast",
        target_ready=1,
    )

    assert ordered == [replacement_small, old_medium]


def test_discover_shard_paths_finds_nested_batch_shards(tmp_path: Path) -> None:
    nested = tmp_path / "batch_00001"
    nested.mkdir()
    incomplete = nested / "fullraw_shard_9999.sqlite"
    incomplete.write_text("")
    shard = nested / "fullraw_shard_0000.sqlite"
    index = FullRawFtsIndex(shard)
    try:
        index.initialize()
    finally:
        index.close()

    assert discover_shard_paths(tmp_path) == [shard]


def test_discover_shard_paths_can_trust_uploaded_filenames(tmp_path: Path) -> None:
    nested = tmp_path / "batch_00001"
    nested.mkdir()
    incomplete = nested / "fullraw_shard_9999.sqlite"
    incomplete.write_text("")

    assert discover_shard_paths(tmp_path, trust_filenames=True) == [incomplete]


def test_read_only_index_searches_existing_shard(tmp_path: Path) -> None:
    shard = tmp_path / "fullraw_shard_0000.sqlite"
    raw_file = _raw_file(tmp_path, "openalex_readonly", [{
        "doi": "https://doi.org/10.example/read-only",
        "display_name": "Management forecast disclosure read only",
        "abstract": "Managers disclose forecasts and guidance.",
        "publication_year": 2024,
    }])
    index = FullRawFtsIndex(shard)
    try:
        index.index_files([raw_file], commit_interval=1)
    finally:
        index.close()

    read_only = FullRawFtsIndex(shard, read_only=True)
    try:
        hits = read_only.search("management forecast disclosure", limit=3)
        stats = read_only.stats(files_total=1)
    finally:
        read_only.close()

    assert [hit["doi"] for hit in hits] == ["10.example/read-only"]
    assert stats.papers_indexed == 1


def test_fullraw_index_profiles_year_citation_and_topics(tmp_path: Path) -> None:
    shard = tmp_path / "fullraw_shard_0000.sqlite"
    raw_file = _raw_file(tmp_path, "openalex_profile", [
        {
            "doi": "https://doi.org/10.example/profile-old",
            "display_name": "Management forecast disclosure",
            "abstract": "Managers disclose forecasts and accounting guidance.",
            "publication_year": 1990,
            "cited_by_count": 110,
        },
        {
            "doi": "https://doi.org/10.example/profile-new",
            "display_name": "Longevity exercise adaptation",
            "abstract": "Exercise adaptation and healthspan resilience.",
            "publication_year": 2024,
            "cited_by_count": 9,
        },
    ])
    index = FullRawFtsIndex(shard)
    try:
        index.index_files([raw_file], commit_interval=1)
        profile = index.profile(topic_limit=6)
    finally:
        index.close()

    assert profile["year_min"] == 1990
    assert profile["year_max"] == 2024
    assert profile["cited_by_min"] == 9
    assert profile["cited_by_max"] == 110
    topic_terms = profile["topic_terms"]
    assert isinstance(topic_terms, tuple)
    assert "forecast" in topic_terms
    assert not {
        "abstract", "of", "in", "by", "to", "was", "study", "data", "used", "or", "however", "should",
    } & set(topic_terms)


def test_fullraw_index_profile_samples_across_shard(tmp_path: Path) -> None:
    shard = tmp_path / "fullraw_profile_spread.sqlite"
    rows = []
    for index in range(30):
        title = f"Generic record {index}"
        abstract = "Background material."
        if index == 21:
            title = "Management forecast disclosure"
            abstract = "Managers disclose forecast guidance."
        rows.append({
            "doi": f"https://doi.org/10.example/spread-{index}",
            "display_name": title,
            "abstract": abstract,
            "publication_year": 1990 + index,
            "cited_by_count": index,
        })
    raw_file = _raw_file(tmp_path, "openalex_profile_spread", rows)
    fts_index = FullRawFtsIndex(shard)
    try:
        fts_index.index_files([raw_file], commit_interval=10)
        profile = fts_index.profile(topic_limit=8, sample_limit=5)
    finally:
        fts_index.close()

    assert profile["profile_sample_size"] == 5
    assert profile["year_min"] == 1990
    assert profile["year_max"] == 2019
    topic_terms = profile["topic_terms"]
    assert isinstance(topic_terms, tuple)
    assert "forecast" in topic_terms


def test_fullraw_index_search_supports_citation_and_recency_rank_modes(tmp_path: Path) -> None:
    shard = tmp_path / "fullraw_rank_modes.sqlite"
    raw_file = _raw_file(tmp_path, "openalex_rank_modes", [
        {
            "doi": "https://doi.org/10.example/high-cited-old",
            "display_name": "Management forecast disclosure evidence",
            "abstract": "Management forecast disclosure evidence.",
            "publication_year": 1990,
            "cited_by_count": 500,
        },
        {
            "doi": "https://doi.org/10.example/recent-low-cited",
            "display_name": "Management forecast disclosure evidence",
            "abstract": "Management forecast disclosure evidence.",
            "publication_year": 2025,
            "cited_by_count": 5,
        },
    ])
    index = FullRawFtsIndex(shard)
    try:
        index.index_files([raw_file], commit_interval=1)
        citation_hits = index.search("management forecast disclosure", limit=2, rank_mode="citation")
        recency_hits = index.search("management forecast disclosure", limit=2, rank_mode="recency")
    finally:
        index.close()

    assert citation_hits[0]["doi"] == "10.example/high-cited-old"
    assert recency_hits[0]["doi"] == "10.example/recent-low-cited"


def test_catalog_reads_manifest_and_manifest_stats(tmp_path: Path) -> None:
    batch = tmp_path / "batch_00001"
    batch.mkdir()
    for shard_id in range(2):
        (batch / f"fullraw_shard_{shard_id:04d}.sqlite").write_text("")
    (batch / "complete.json").write_text(json.dumps({
        "batch_id": 1,
        "files": [
            {"source": "openalex"},
            {"source": "semantic_scholar"},
        ],
        "shards": [
            {
                "shard_id": 0,
                "files_completed": 3,
                "papers_inserted": 42,
                "bytes_used": 1234,
                "year_min": 1980,
                "year_max": 2024,
                "cited_by_min": 0,
                "cited_by_max": 500,
                "cited_by_avg": 22.5,
                "topic_terms": ["management", "forecast"],
            },
            {
                "shard_id": 1,
                "files_completed": 4,
                "papers_inserted": 99,
                "bytes_used": 4567,
                "year_min": 2001,
                "year_max": 2026,
                "cited_by_min": 1,
                "cited_by_max": 40,
                "topic_terms": ["longevity"],
            },
        ],
    }))

    catalog = build_shard_catalog(tmp_path, trust_filenames=True)
    stats = aggregate_shard_manifest_stats(tmp_path, files_total=10)

    assert [(entry.batch_id, entry.shard_id) for entry in catalog] == [(1, 0), (1, 1)]
    assert catalog[0].sources == ("openalex", "semantic_scholar")
    assert catalog[1].papers_inserted == 99
    assert catalog[0].year_min == 1980
    assert catalog[0].cited_by_max == 500
    assert catalog[0].topic_terms == ("management", "forecast")
    assert stats.files_indexed == 7
    assert stats.papers_indexed == 141
    assert stats.bytes_used == 5801


def test_backfill_shard_profiles_updates_existing_manifest(tmp_path: Path) -> None:
    batch = tmp_path / "batch_00000"
    batch.mkdir()
    shard = batch / "fullraw_shard_0000.sqlite"
    raw_file = _raw_file(tmp_path, "openalex_backfill", [
        {
            "doi": "https://doi.org/10.example/backfill-old",
            "display_name": "Management forecast disclosure",
            "abstract": "Managers disclose forecast guidance.",
            "publication_year": 1990,
            "cited_by_count": 99,
        },
        {
            "doi": "https://doi.org/10.example/backfill-new",
            "display_name": "Management forecast disclosure recency",
            "abstract": "Recent managers disclose forecast guidance.",
            "publication_year": 2025,
            "cited_by_count": 7,
        },
    ])
    index = FullRawFtsIndex(shard)
    try:
        index.index_files([raw_file], commit_interval=1)
    finally:
        index.close()
    (batch / "complete.json").write_text(json.dumps({
        "batch_id": 0,
        "files": [{"source": "openalex"}],
        "shards": [{
            "shard_id": 0,
            "files_completed": 1,
            "papers_inserted": 2,
            "bytes_used": shard.stat().st_size,
        }],
    }))

    result = backfill_shard_profiles(tmp_path)
    catalog = build_shard_catalog(tmp_path, trust_filenames=True)
    manifest = json.loads((batch / "complete.json").read_text())

    assert result["shards_profiled"] == 1
    assert result["batches_updated"] == 1
    assert catalog[0].year_min == 1990
    assert catalog[0].year_max == 2025
    assert catalog[0].cited_by_max == 99
    assert "forecast" in catalog[0].topic_terms
    assert manifest["shards"][0]["year_min"] == 1990


def test_backfill_shard_profiles_flushes_each_batch(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for batch_id in range(2):
        batch = tmp_path / f"batch_{batch_id:05d}"
        batch.mkdir()
        shard = batch / f"fullraw_shard_{batch_id:04d}.sqlite"
        shard.touch()
        (batch / "complete.json").write_text(json.dumps({
            "batch_id": batch_id,
            "files": [{"source": "openalex"}],
            "shards": [{
                "shard_id": batch_id,
                "files_completed": 1,
                "papers_inserted": 1,
                "bytes_used": 1,
            }],
        }))

    def fake_profile(path: Path) -> dict[str, object]:
        return {
            "year_min": 2000,
            "year_max": 2024,
            "cited_by_min": 1,
            "cited_by_max": 5,
            "cited_by_avg": 3.0,
            "topic_terms": (path.parent.name,),
        }

    monkeypatch.setattr(fullraw_index, "_profile_shard_path", fake_profile)

    result = backfill_shard_profiles(tmp_path, progress_interval=99)
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]

    assert result["shards_profiled"] == 2
    assert result["batches_updated"] == 2
    assert [event["event"] for event in events] == [
        "profile_backfill_batch_flushed",
        "profile_backfill_batch_flushed",
    ]
    assert [event["batch"] for event in events] == ["batch_00000", "batch_00001"]
    first_manifest = json.loads((tmp_path / "batch_00000" / "complete.json").read_text())
    assert first_manifest["shards"][0]["year_min"] == 2000


def test_backfill_shard_profiles_cli_reports_json(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    batch = tmp_path / "batch_00000"
    batch.mkdir()
    shard = batch / "fullraw_shard_0000.sqlite"
    raw_file = _raw_file(tmp_path, "openalex_backfill_cli", [{
        "doi": "https://doi.org/10.example/backfill-cli",
        "display_name": "Management forecast disclosure CLI",
        "abstract": "Managers disclose forecast guidance.",
        "publication_year": 2024,
    }])
    index = FullRawFtsIndex(shard)
    try:
        index.index_files([raw_file], commit_interval=1)
    finally:
        index.close()
    (batch / "complete.json").write_text(json.dumps({
        "batch_id": 0,
        "files": [{"source": "openalex"}],
        "shards": [{
            "shard_id": 0,
            "files_completed": 1,
            "papers_inserted": 1,
            "bytes_used": shard.stat().st_size,
        }],
    }))
    monkeypatch.setattr(sys, "argv", ["fullraw_index.py", "backfill-shard-profiles", "--shard-dir", str(tmp_path)])

    fullraw_index.main()

    assert json.loads(capsys.readouterr().out)["shards_profiled"] == 1


def test_server_async_sweep_caches_all_shard_results(tmp_path: Path) -> None:
    for index, cited in enumerate((10, 20)):
        batch = tmp_path / f"batch_{index:05d}"
        batch.mkdir()
        shard = batch / "fullraw_shard_0000.sqlite"
        raw_file = _raw_file(tmp_path, f"async_sweep_{index}", [{
            "doi": f"https://doi.org/10.example/async-{index}",
            "display_name": f"Management forecast disclosure async {index}",
            "abstract": "Management forecast disclosure async evidence.",
            "publication_year": 2024,
            "cited_by_count": cited,
        }])
        index_db = FullRawFtsIndex(shard)
        try:
            index_db.index_files([raw_file], commit_interval=1)
            profile = index_db.profile()
        finally:
            index_db.close()
        (batch / "complete.json").write_text(json.dumps({
            "batch_id": index,
            "files": [{"source": "openalex"}],
            "shards": [{
                "shard_id": 0,
                "files_completed": 1,
                "papers_inserted": 1,
                "bytes_used": shard.stat().st_size,
                **profile,
            }],
        }))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": []}))
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd() / "src"),
        "V5_MEMO_FULL_RAW_INDEX_PORT": str(port),
        "V5_MEMO_FULL_RAW_MANIFEST": str(manifest),
        "V5_MEMO_FULL_RAW_SHARD_DIR": str(tmp_path),
        "V5_MEMO_FULL_RAW_SHARD_TRUST_FILENAMES": "1",
        "V5_MEMO_FULL_RAW_SHARD_MANIFEST_STATS": "1",
        "V5_MEMO_FULL_RAW_SEARCH_SHARD_LIMIT": "1",
        "V5_MEMO_FULL_RAW_SEARCH_WORKERS": "1",
        "V5_MEMO_FULL_RAW_ASYNC_SWEEP": "1",
        "V5_MEMO_FULL_RAW_SWEEP_CACHE_DIR": str(tmp_path / "cache"),
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "v5_memo.fullraw_index", "serve"],
        cwd=Path.cwd(),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(50):
            try:
                urllib.request.urlopen(base + "/health", timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            _, stderr = proc.communicate(timeout=1)
            raise AssertionError(f"server did not start: {stderr}")

        def post_search(
            *,
            cache_only: bool = False,
            queue_if_missing: bool = False,
        ) -> dict[str, object]:
            request = urllib.request.Request(
                base + "/search",
                data=json.dumps({
                    "query": "voluntary management earnings forecast disclosure",
                    "top_k": 5,
                    "cache_only": cache_only,
                    "queue_if_missing": queue_if_missing,
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode())
            assert isinstance(payload, dict)
            return payload

        first = post_search(cache_only=True)
        first_meta = first["meta"]
        assert isinstance(first_meta, dict)
        assert first_meta["async_sweep"]["status"] == "miss"
        assert first_meta["cache_only"] is True
        assert first["results"] == []
        assert first_meta["shard_receipt"] == {}

        queued = post_search(cache_only=True, queue_if_missing=True)
        queued_meta = queued["meta"]
        assert isinstance(queued_meta, dict)
        assert queued_meta["async_sweep"]["status"] in {"queued", "hit"}
        assert queued_meta["cache_only"] is True
        if queued_meta["async_sweep"]["status"] != "hit":
            assert queued["results"] == []
            assert queued_meta["shard_receipt"] == {}
        cached = first
        for _ in range(50):
            cached = post_search(cache_only=True)
            meta = cached["meta"]
            assert isinstance(meta, dict)
            if meta["async_sweep"]["status"] == "hit":
                break
            time.sleep(0.1)
        meta = cached["meta"]
        assert isinstance(meta, dict)
        receipt = meta["shard_receipt"]
        assert isinstance(receipt, dict)
        results = cached["results"]
        assert isinstance(results, list)
        assert meta["async_sweep"]["status"] == "hit"
        assert receipt["shards_searched"] == 2
        assert receipt["sweep_strategy"] == fullraw_index._SWEEP_STRATEGY
        assert receipt["sweep_query"] == "management forecast disclosure"
        assert receipt["sweep_original_query"] == "voluntary management earnings forecast disclosure"
        assert receipt["sweep_completed_pass_roles"] == ["focused"]
        search_passes = receipt["sweep_search_passes"]
        assert isinstance(search_passes, list)
        assert [item["role"] for item in search_passes] == [
            "focused",
            "broad",
            "adjacent_field",
            "falsifier",
            "citation_heavy",
            "recency",
        ]
        assert receipt["result_count_raw"] == 2
        assert receipt["result_count_unique"] == 2
        assert receipt["result_duplicate_rate"] == 0.0
        assert receipt["result_citation_diversity"] >= 1
        assert len(results) == 2
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_server_rejects_narrow_cached_sweep_receipt(tmp_path: Path) -> None:
    batch = tmp_path / "batch_00000"
    batch.mkdir()
    shard = batch / "fullraw_shard_0000.sqlite"
    raw_file = _raw_file(tmp_path, "narrow_cached_sweep", [{
        "doi": "https://doi.org/10.example/narrow-cache",
        "display_name": "Management forecast disclosure cached",
        "abstract": "Management forecast disclosure evidence.",
        "publication_year": 2024,
    }])
    index_db = FullRawFtsIndex(shard)
    try:
        index_db.index_files([raw_file], commit_interval=1)
        profile = index_db.profile()
    finally:
        index_db.close()
    (batch / "complete.json").write_text(json.dumps({
        "batch_id": 0,
        "files": [{"source": "openalex"}],
        "shards": [{
            "shard_id": 0,
            "files_completed": 1,
            "papers_inserted": 1,
            "bytes_used": shard.stat().st_size,
            **profile,
        }],
    }))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": []}))
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd() / "src"),
        "V5_MEMO_FULL_RAW_INDEX_PORT": str(port),
        "V5_MEMO_FULL_RAW_MANIFEST": str(manifest),
        "V5_MEMO_FULL_RAW_SHARD_DIR": str(tmp_path),
        "V5_MEMO_FULL_RAW_SHARD_TRUST_FILENAMES": "1",
        "V5_MEMO_FULL_RAW_SHARD_MANIFEST_STATS": "1",
        "V5_MEMO_FULL_RAW_ASYNC_SWEEP": "1",
        "V5_MEMO_FULL_RAW_SWEEP_CACHE_DIR": str(tmp_path / "cache"),
        "V5_MEMO_FULL_RAW_SWEEP_SHARD_LIMIT": "1",
        "V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED": "2",
        "V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "v5_memo.fullraw_index", "serve"],
        cwd=Path.cwd(),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(50):
            try:
                urllib.request.urlopen(base + "/health", timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            _, stderr = proc.communicate(timeout=1)
            raise AssertionError(f"server did not start: {stderr}")

        def post_search(*, queue_if_missing: bool = False) -> dict[str, object]:
            request = urllib.request.Request(
                base + "/search",
                data=json.dumps({
                    "query": "voluntary management earnings forecast disclosure",
                    "top_k": 5,
                    "cache_only": True,
                    "queue_if_missing": queue_if_missing,
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode())
            assert isinstance(payload, dict)
            return payload

        queued = post_search(queue_if_missing=True)
        meta = queued["meta"]
        assert isinstance(meta, dict)
        assert meta["async_sweep"]["status"] in {"queued", "hit"}
        for _ in range(50):
            try:
                post_search()
            except urllib.error.HTTPError as exc:
                assert exc.code == 422
                body = json.loads(exc.read().decode())
                assert body["error"] == "coverage_too_narrow"
                assert body["requirements"] == {
                    "min_shards_searched": 2,
                    "min_sources_searched": 1,
                }
                assert body["shard_receipt"]["shards_searched"] == 1
                assert (
                    body["shard_receipt"]["sweep_original_query"]
                    == "voluntary management earnings forecast disclosure"
                )
                assert body["shard_receipt"]["sweep_query"] == "management forecast disclosure"
                assert body["shard_receipt"]["sweep_completed_pass_roles"] == ["focused"]
                break
            time.sleep(0.1)
        else:
            raise AssertionError("narrow cached sweep was not rejected")
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_server_resumes_narrow_cached_sweep_receipt(tmp_path: Path) -> None:
    for index in range(3):
        batch = tmp_path / f"batch_{index:05d}"
        batch.mkdir()
        shard = batch / "fullraw_shard_0000.sqlite"
        raw_file = _raw_file(tmp_path, f"resume_cached_sweep_{index}", [{
            "doi": f"https://doi.org/10.example/resume-cache-{index}",
            "display_name": f"Management forecast disclosure resume {index}",
            "abstract": "Management forecast disclosure evidence.",
            "publication_year": 2024,
            "cited_by_count": 10 + index,
        }])
        index_db = FullRawFtsIndex(shard)
        try:
            index_db.index_files([raw_file], commit_interval=1)
            profile = index_db.profile()
        finally:
            index_db.close()
        (batch / "complete.json").write_text(json.dumps({
            "batch_id": index,
            "files": [{"source": "openalex"}],
            "shards": [{
                "shard_id": 0,
                "files_completed": 1,
                "papers_inserted": 1,
                "bytes_used": shard.stat().st_size,
                **profile,
            }],
        }))
    cache_dir = tmp_path / "cache"
    catalog = fullraw_index.build_shard_catalog(tmp_path, trust_filenames=True)
    selected = select_sweep_shard_entries(catalog, query="management forecast disclosure", limit=3)
    partial_receipt = shard_coverage_receipt(catalog, [selected[0]])
    partial_receipt.update({
        "sweep_scope": "relevant",
        "sweep_shard_limit": 3,
        "sweep_selected_shards": 3,
        "sweep_pass_shard_limit": 1,
        "sweep_pass_selected_shards": 1,
        "sweep_remaining_shards": 2,
        "sweep_timed_out": True,
        "sweep_timeout_seconds": 300.0,
        "sweep_shard_timeout_seconds": 10.0,
        "sweep_strategy": fullraw_index._SWEEP_STRATEGY,
        "sweep_query": "management forecast disclosure",
        "sweep_passes": 1,
        "sweep_completed_paths": [str(selected[0].path)],
    })
    cache_key = fullraw_index._sweep_cache_key(
        "management forecast disclosure",
        limit=5,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        sweep_shard_limit=3,
    )
    fullraw_index._write_sweep_cache(
        cache_dir / f"{cache_key}.json",
        fullraw_index.SweepCacheEntry(
            time.time(),
            [{
                "doi": "10.example/resume-cache-0",
                "title": "Management forecast disclosure resume 0",
                "score": 1.0,
            }],
            partial_receipt,
        ),
    )
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": []}))
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd() / "src"),
        "V5_MEMO_FULL_RAW_INDEX_PORT": str(port),
        "V5_MEMO_FULL_RAW_MANIFEST": str(manifest),
        "V5_MEMO_FULL_RAW_SHARD_DIR": str(tmp_path),
        "V5_MEMO_FULL_RAW_SHARD_TRUST_FILENAMES": "1",
        "V5_MEMO_FULL_RAW_SHARD_MANIFEST_STATS": "1",
        "V5_MEMO_FULL_RAW_ASYNC_SWEEP": "1",
        "V5_MEMO_FULL_RAW_SWEEP_CACHE_DIR": str(cache_dir),
        "V5_MEMO_FULL_RAW_SWEEP_SHARD_LIMIT": "3",
        "V5_MEMO_FULL_RAW_SWEEP_PASS_SHARD_LIMIT": "1",
        "V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED": "2",
        "V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "v5_memo.fullraw_index", "serve"],
        cwd=Path.cwd(),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(50):
            try:
                urllib.request.urlopen(base + "/health", timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            _, stderr = proc.communicate(timeout=1)
            raise AssertionError(f"server did not start: {stderr}")

        def post_search() -> dict[str, object]:
            request = urllib.request.Request(
                base + "/search",
                data=json.dumps({
                    "query": "management forecast disclosure",
                    "top_k": 5,
                    "cache_only": True,
                    "queue_if_missing": True,
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode())
            assert isinstance(payload, dict)
            return payload

        cached = post_search()
        first_meta = cached["meta"]
        assert isinstance(first_meta, dict)
        if first_meta["async_sweep"]["status"] != "hit":
            first_receipt = first_meta["shard_receipt"]
            assert isinstance(first_receipt, dict)
            assert first_receipt["shards_searched"] == 1
            assert cached["results"] == []
        for _ in range(50):
            meta = cached["meta"]
            assert isinstance(meta, dict)
            if meta["async_sweep"]["status"] == "hit":
                break
            time.sleep(0.1)
            cached = post_search()
        else:
            raise AssertionError("narrow cached sweep did not resume")
        meta = cached["meta"]
        assert isinstance(meta, dict)
        receipt = meta["shard_receipt"]
        assert isinstance(receipt, dict)
        assert receipt["shards_searched"] == 2
        assert receipt["sweep_passes"] == 2
        assert receipt["sweep_pass_selected_shards"] == 1
        assert receipt["sweep_remaining_shards"] == 1
        assert receipt["sweep_planned_shards"] == 3
        assert receipt["sweep_planned_sources"] == {"openalex": 3}
        assert receipt["sweep_planned_source_count"] == 1
        assert receipt["sweep_planned_year_range"] == {"min": 2024, "max": 2024}
        assert receipt["sweep_planned_cited_by_range"] == {"min": 10, "max": 12}
        assert receipt["sweep_planned_papers"] == 3
        completed_paths = receipt["sweep_completed_paths"]
        assert isinstance(completed_paths, list)
        assert len(completed_paths) == 2
        results = cached["results"]
        assert isinstance(results, list)
        assert len(results) == 2
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_server_auto_continues_sweep_until_receipt_is_sufficient(tmp_path: Path) -> None:
    for index in range(3):
        batch = tmp_path / f"batch_{index:05d}"
        batch.mkdir()
        shard = batch / "fullraw_shard_0000.sqlite"
        raw_file = _raw_file(tmp_path, f"auto_continue_sweep_{index}", [{
            "doi": f"https://doi.org/10.example/auto-continue-{index}",
            "display_name": f"Management forecast disclosure auto continue {index}",
            "abstract": "Management forecast disclosure evidence.",
            "publication_year": 2024,
            "cited_by_count": 10 + index,
        }])
        index_db = FullRawFtsIndex(shard)
        try:
            index_db.index_files([raw_file], commit_interval=1)
            profile = index_db.profile()
        finally:
            index_db.close()
        (batch / "complete.json").write_text(json.dumps({
            "batch_id": index,
            "files": [{"source": "openalex"}],
            "shards": [{
                "shard_id": 0,
                "files_completed": 1,
                "papers_inserted": 1,
                "bytes_used": shard.stat().st_size,
                **profile,
            }],
        }))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": []}))
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd() / "src"),
        "V5_MEMO_FULL_RAW_INDEX_PORT": str(port),
        "V5_MEMO_FULL_RAW_MANIFEST": str(manifest),
        "V5_MEMO_FULL_RAW_SHARD_DIR": str(tmp_path),
        "V5_MEMO_FULL_RAW_SHARD_TRUST_FILENAMES": "1",
        "V5_MEMO_FULL_RAW_SHARD_MANIFEST_STATS": "1",
        "V5_MEMO_FULL_RAW_ASYNC_SWEEP": "1",
        "V5_MEMO_FULL_RAW_SWEEP_CACHE_DIR": str(tmp_path / "cache"),
        "V5_MEMO_FULL_RAW_SWEEP_SHARD_LIMIT": "3",
        "V5_MEMO_FULL_RAW_SWEEP_PASS_SHARD_LIMIT": "1",
        "V5_MEMO_FULL_RAW_SWEEP_MAX_PASSES": "3",
        "V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED": "3",
        "V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "v5_memo.fullraw_index", "serve"],
        cwd=Path.cwd(),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(50):
            try:
                urllib.request.urlopen(base + "/health", timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            _, stderr = proc.communicate(timeout=1)
            raise AssertionError(f"server did not start: {stderr}")

        def post_search(*, queue_if_missing: bool = False) -> tuple[int, dict[str, object]]:
            request = urllib.request.Request(
                base + "/search",
                data=json.dumps({
                    "query": "management forecast disclosure",
                    "top_k": 5,
                    "cache_only": True,
                    "queue_if_missing": queue_if_missing,
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    payload = json.loads(response.read().decode())
                    assert isinstance(payload, dict)
                    return response.status, payload
            except urllib.error.HTTPError as exc:
                payload = json.loads(exc.read().decode())
                assert isinstance(payload, dict)
                return exc.code, payload

        status, queued = post_search(queue_if_missing=True)
        assert status == 200
        meta = queued["meta"]
        assert isinstance(meta, dict)
        assert meta["async_sweep"]["status"] in {"queued", "hit"}
        final_body = queued
        for _ in range(50):
            status, body = post_search()
            final_body = body
            if status == 200 and body.get("results"):
                break
            time.sleep(0.1)
        else:
            raise AssertionError("auto-continued sweep did not satisfy coverage")
        meta = final_body["meta"]
        assert isinstance(meta, dict)
        receipt = meta["shard_receipt"]
        assert isinstance(receipt, dict)
        assert receipt["shards_searched"] == 3
        assert receipt["sweep_passes"] == 3
        assert receipt["sweep_max_passes"] == 3
        assert receipt["sweep_remaining_shards"] == 0
        assert receipt["sweep_planned_shards"] == 3
        assert receipt["sweep_planned_sources"] == {"openalex": 3}
        assert receipt["sweep_planned_source_count"] == 1
        assert receipt["sweep_planned_year_range"] == {"min": 2024, "max": 2024}
        assert receipt["sweep_planned_cited_by_range"] == {"min": 10, "max": 12}
        assert receipt["sweep_planned_papers"] == 3
        assert len(receipt["sweep_completed_paths"]) == 3
        results = final_body["results"]
        assert isinstance(results, list)
        assert len(results) == 3
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_server_exhaustive_sweep_continues_after_minimum_receipt(tmp_path: Path) -> None:
    for index in range(3):
        batch = tmp_path / f"batch_{index:05d}"
        batch.mkdir()
        shard = batch / "fullraw_shard_0000.sqlite"
        raw_file = _raw_file(tmp_path, f"exhaustive_sweep_{index}", [{
            "doi": f"https://doi.org/10.example/exhaustive-{index}",
            "display_name": f"Management forecast disclosure exhaustive {index}",
            "abstract": "Management forecast disclosure evidence.",
            "publication_year": 2024,
            "cited_by_count": 10 + index,
        }])
        index_db = FullRawFtsIndex(shard)
        try:
            index_db.index_files([raw_file], commit_interval=1)
            profile = index_db.profile()
        finally:
            index_db.close()
        (batch / "complete.json").write_text(json.dumps({
            "batch_id": index,
            "files": [{"source": "openalex"}],
            "shards": [{
                "shard_id": 0,
                "files_completed": 1,
                "papers_inserted": 1,
                "bytes_used": shard.stat().st_size,
                **profile,
            }],
        }))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": []}))
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd() / "src"),
        "V5_MEMO_FULL_RAW_INDEX_PORT": str(port),
        "V5_MEMO_FULL_RAW_MANIFEST": str(manifest),
        "V5_MEMO_FULL_RAW_SHARD_DIR": str(tmp_path),
        "V5_MEMO_FULL_RAW_SHARD_TRUST_FILENAMES": "1",
        "V5_MEMO_FULL_RAW_SHARD_MANIFEST_STATS": "1",
        "V5_MEMO_FULL_RAW_ASYNC_SWEEP": "1",
        "V5_MEMO_FULL_RAW_SWEEP_CACHE_DIR": str(tmp_path / "cache"),
        "V5_MEMO_FULL_RAW_SWEEP_SHARD_LIMIT": "3",
        "V5_MEMO_FULL_RAW_SWEEP_PASS_SHARD_LIMIT": "1",
        "V5_MEMO_FULL_RAW_SWEEP_MAX_PASSES": "1",
        "V5_MEMO_FULL_RAW_SWEEP_REQUIRE_COMPLETE": "1",
        "V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED": "1",
        "V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "v5_memo.fullraw_index", "serve"],
        cwd=Path.cwd(),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(50):
            try:
                urllib.request.urlopen(base + "/health", timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            _, stderr = proc.communicate(timeout=1)
            raise AssertionError(f"server did not start: {stderr}")

        def post_search() -> tuple[int, dict[str, object]]:
            request = urllib.request.Request(
                base + "/search",
                data=json.dumps({
                    "query": "management forecast disclosure",
                    "top_k": 5,
                    "cache_only": True,
                    "queue_if_missing": True,
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode())
                assert isinstance(payload, dict)
                return response.status, payload

        final_body: dict[str, object] = {}
        for _ in range(80):
            status, body = post_search()
            final_body = body
            meta = body.get("meta")
            receipt = meta.get("shard_receipt") if isinstance(meta, dict) else None
            if status == 200 and isinstance(receipt, dict) and receipt.get("sweep_remaining_shards") == 0:
                break
            time.sleep(0.1)
        else:
            raise AssertionError("exhaustive sweep did not complete all selected shards")
        meta = final_body["meta"]
        assert isinstance(meta, dict)
        receipt = meta["shard_receipt"]
        assert isinstance(receipt, dict)
        assert receipt["shards_searched"] == 3
        assert receipt["sweep_remaining_shards"] == 0
        assert len(receipt["sweep_completed_paths"]) == 3
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_server_source_scoped_exhaustive_sweep_reports_excluded_sources(tmp_path: Path) -> None:
    _write_search_test_batch(tmp_path, 0, source="openalex")
    _write_search_test_batch(tmp_path, 1, source="pubmed")
    _write_search_test_batch(tmp_path, 2, source="semantic_scholar")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": []}))
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd() / "src"),
        "V5_MEMO_FULL_RAW_INDEX_PORT": str(port),
        "V5_MEMO_FULL_RAW_MANIFEST": str(manifest),
        "V5_MEMO_FULL_RAW_SHARD_DIR": str(tmp_path),
        "V5_MEMO_FULL_RAW_SHARD_TRUST_FILENAMES": "1",
        "V5_MEMO_FULL_RAW_SHARD_MANIFEST_STATS": "1",
        "V5_MEMO_FULL_RAW_ASYNC_SWEEP": "1",
        "V5_MEMO_FULL_RAW_SWEEP_CACHE_DIR": str(tmp_path / "cache"),
        "V5_MEMO_FULL_RAW_SEARCH_SOURCE_FILTER": "openalex,pubmed",
        "V5_MEMO_FULL_RAW_SWEEP_SHARD_LIMIT": "10",
        "V5_MEMO_FULL_RAW_SWEEP_PASS_SHARD_LIMIT": "1",
        "V5_MEMO_FULL_RAW_SWEEP_MAX_PASSES": "2",
        "V5_MEMO_FULL_RAW_SWEEP_REQUIRE_COMPLETE": "1",
        "V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED": "2",
        "V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED": "2",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "v5_memo.fullraw_index", "serve"],
        cwd=Path.cwd(),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(50):
            try:
                urllib.request.urlopen(base + "/health", timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            _, stderr = proc.communicate(timeout=1)
            raise AssertionError(f"server did not start: {stderr}")

        def post_search() -> tuple[int, dict[str, object]]:
            request = urllib.request.Request(
                base + "/search",
                data=json.dumps({
                    "query": "management forecast disclosure",
                    "top_k": 5,
                    "cache_only": True,
                    "queue_if_missing": True,
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode())
                assert isinstance(payload, dict)
                return response.status, payload

        final_body: dict[str, object] = {}
        for _ in range(80):
            status, body = post_search()
            final_body = body
            meta = body.get("meta")
            receipt = meta.get("shard_receipt") if isinstance(meta, dict) else None
            if status == 200 and isinstance(receipt, dict) and receipt.get("sweep_remaining_shards") == 0:
                break
            time.sleep(0.1)
        else:
            raise AssertionError("source-scoped sweep did not complete selected shards")
        meta = final_body["meta"]
        assert isinstance(meta, dict)
        receipt = meta["shard_receipt"]
        assert isinstance(receipt, dict)
        assert receipt["source_scope"] == ["openalex", "pubmed"]
        assert receipt["shards_total"] == 2
        assert receipt["shards_searched"] == 2
        assert receipt["partial_shard_search"] is False
        assert receipt["sources_total"] == {"openalex": 1, "pubmed": 1}
        assert receipt["sources_searched"] == {"openalex": 1, "pubmed": 1}
        assert receipt["all_shards_total"] == 3
        assert receipt["all_sources_total"] == {
            "openalex": 1,
            "pubmed": 1,
            "semantic_scholar": 1,
        }
        assert receipt["sources_excluded_by_scope"] == ["semantic_scholar"]
        assert receipt["shards_excluded_by_scope"] == 1
        assert receipt["sweep_remaining_shards"] == 0
        results = final_body["results"]
        assert isinstance(results, list)
        assert {result["source"] for result in results} == {"openalex", "pubmed"}
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_server_complete_search_receipt_requires_completed_shards(tmp_path: Path) -> None:
    _write_search_test_batch(tmp_path, 0, valid=True)
    _write_search_test_batch(tmp_path, 1, valid=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": []}))
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd() / "src"),
        "V5_MEMO_FULL_RAW_INDEX_PORT": str(port),
        "V5_MEMO_FULL_RAW_MANIFEST": str(manifest),
        "V5_MEMO_FULL_RAW_SHARD_DIR": str(tmp_path),
        "V5_MEMO_FULL_RAW_SHARD_TRUST_FILENAMES": "1",
        "V5_MEMO_FULL_RAW_SHARD_MANIFEST_STATS": "1",
        "V5_MEMO_FULL_RAW_SEARCH_SHARD_LIMIT": "2",
        "V5_MEMO_FULL_RAW_REQUIRE_COMPLETE_SEARCH": "1",
        "V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED": "1",
        "V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "v5_memo.fullraw_index", "serve"],
        cwd=Path.cwd(),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(50):
            try:
                urllib.request.urlopen(base + "/health", timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            _, stderr = proc.communicate(timeout=1)
            raise AssertionError(f"server did not start: {stderr}")
        request = urllib.request.Request(
            base + "/search",
            data=json.dumps({
                "query": "management forecast disclosure",
                "top_k": 5,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request, timeout=10)
        assert raised.value.code == 422
        body = json.loads(raised.value.read().decode())
        assert body["error"] == "shard coverage incomplete"
        receipt = body["shard_receipt"]
        assert receipt["shards_total"] == 2
        assert receipt["shards_searched"] == 1
        assert receipt["partial_shard_search"] is True
        assert receipt["search_selected_shards"] == 2
        assert receipt["search_completed_shards"] == 1
        assert receipt["search_failed_shards"] == 1
        assert body["coverage_requirements"]["require_complete_search"] is True
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_server_complete_search_receipt_passes_when_all_selected_shards_complete(tmp_path: Path) -> None:
    _write_search_test_batch(tmp_path, 0, source="openalex")
    _write_search_test_batch(tmp_path, 1, source="pubmed")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": []}))
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd() / "src"),
        "V5_MEMO_FULL_RAW_INDEX_PORT": str(port),
        "V5_MEMO_FULL_RAW_MANIFEST": str(manifest),
        "V5_MEMO_FULL_RAW_SHARD_DIR": str(tmp_path),
        "V5_MEMO_FULL_RAW_SHARD_TRUST_FILENAMES": "1",
        "V5_MEMO_FULL_RAW_SHARD_MANIFEST_STATS": "1",
        "V5_MEMO_FULL_RAW_SEARCH_SHARD_LIMIT": "2",
        "V5_MEMO_FULL_RAW_REQUIRE_COMPLETE_SEARCH": "1",
        "V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED": "2",
        "V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED": "2",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "v5_memo.fullraw_index", "serve"],
        cwd=Path.cwd(),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(50):
            try:
                urllib.request.urlopen(base + "/health", timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            _, stderr = proc.communicate(timeout=1)
            raise AssertionError(f"server did not start: {stderr}")
        request = urllib.request.Request(
            base + "/search",
            data=json.dumps({
                "query": "management forecast disclosure",
                "top_k": 5,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            body = json.loads(response.read().decode())
        assert response.status == 200
        receipt = body["meta"]["shard_receipt"]
        assert receipt["shards_total"] == 2
        assert receipt["shards_searched"] == 2
        assert receipt["partial_shard_search"] is False
        assert receipt["sources_searched"] == {"openalex": 1, "pubmed": 1}
        assert receipt["search_selected_shards"] == 2
        assert receipt["search_completed_shards"] == 2
        assert receipt["search_failed_shards"] == 0
        assert body["meta"]["count"] == 2
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_search_shard_paths_can_use_subprocess_timeouts(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    _write_search_test_batch(tmp_path, 0)
    path = tmp_path / "batch_00000" / "fullraw_shard_0000.sqlite"
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SEARCH_SUBPROCESS_TIMEOUT", "1")

    hits, completed_paths, timed_out, metrics = fullraw_index._search_shard_paths_with_paths_and_receipt(
        [path],
        "management forecast disclosure",
        limit=5,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        timeout_seconds=10,
        shard_timeout_seconds=10,
    )

    assert completed_paths == [path]
    assert timed_out is False
    assert len(hits) == 1
    assert metrics["result_count_returned"] == 1


def test_server_exhaustive_sweep_defers_failed_paths_without_final_hit(tmp_path: Path) -> None:
    invalid_batch = tmp_path / "batch_00000"
    invalid_batch.mkdir()
    invalid_shard = invalid_batch / "fullraw_shard_0000.sqlite"
    invalid_shard.write_text("not sqlite")
    (invalid_batch / "complete.json").write_text(json.dumps({
        "batch_id": 0,
        "files": [{"source": "openalex"}],
        "shards": [{
            "shard_id": 0,
            "files_completed": 1,
            "papers_inserted": 1,
            "bytes_used": invalid_shard.stat().st_size,
            "topic_terms": ["management", "forecast", "disclosure"],
        }],
    }))

    valid_batch = tmp_path / "batch_00001"
    valid_batch.mkdir()
    valid_shard = valid_batch / "fullraw_shard_0000.sqlite"
    raw_file = _raw_file(tmp_path, "exhaustive_defer_valid", [{
        "doi": "https://doi.org/10.example/exhaustive-defer",
        "display_name": "Management forecast disclosure deferred failure",
        "abstract": "Management forecast disclosure evidence.",
        "publication_year": 2024,
    }])
    index_db = FullRawFtsIndex(valid_shard)
    try:
        index_db.index_files([raw_file], commit_interval=1)
        profile = index_db.profile()
    finally:
        index_db.close()
    (valid_batch / "complete.json").write_text(json.dumps({
        "batch_id": 1,
        "files": [{"source": "openalex"}],
        "shards": [{
            "shard_id": 0,
            "files_completed": 1,
            "papers_inserted": 1,
            "bytes_used": valid_shard.stat().st_size,
            **profile,
        }],
    }))

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": []}))
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd() / "src"),
        "V5_MEMO_FULL_RAW_INDEX_PORT": str(port),
        "V5_MEMO_FULL_RAW_MANIFEST": str(manifest),
        "V5_MEMO_FULL_RAW_SHARD_DIR": str(tmp_path),
        "V5_MEMO_FULL_RAW_SHARD_TRUST_FILENAMES": "1",
        "V5_MEMO_FULL_RAW_SHARD_MANIFEST_STATS": "1",
        "V5_MEMO_FULL_RAW_ASYNC_SWEEP": "1",
        "V5_MEMO_FULL_RAW_SWEEP_CACHE_DIR": str(tmp_path / "cache"),
        "V5_MEMO_FULL_RAW_SWEEP_SHARD_LIMIT": "2",
        "V5_MEMO_FULL_RAW_SWEEP_PASS_SHARD_LIMIT": "1",
        "V5_MEMO_FULL_RAW_SWEEP_MAX_PASSES": "2",
        "V5_MEMO_FULL_RAW_SWEEP_REQUIRE_COMPLETE": "1",
        "V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED": "1",
        "V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "v5_memo.fullraw_index", "serve"],
        cwd=Path.cwd(),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(50):
            try:
                urllib.request.urlopen(base + "/health", timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            _, stderr = proc.communicate(timeout=1)
            raise AssertionError(f"server did not start: {stderr}")

        def post_search() -> dict[str, object]:
            request = urllib.request.Request(
                base + "/search",
                data=json.dumps({
                    "query": "management forecast disclosure",
                    "top_k": 5,
                    "cache_only": True,
                    "queue_if_missing": True,
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode())
                assert isinstance(payload, dict)
                return payload

        final_body: dict[str, object] = {}
        for _ in range(80):
            body = post_search()
            meta = body.get("meta")
            receipt = meta.get("shard_receipt") if isinstance(meta, dict) else None
            if (
                isinstance(receipt, dict)
                and receipt.get("shards_searched") == 1
                and receipt.get("sweep_deferred_shards") == 1
            ):
                final_body = body
                break
            time.sleep(0.1)
        else:
            raise AssertionError("exhaustive sweep did not defer failed shard")

        meta = final_body["meta"]
        assert isinstance(meta, dict)
        receipt = meta["shard_receipt"]
        assert isinstance(receipt, dict)
        assert final_body["results"] == []
        assert receipt["sweep_failed_shards"] == 0
        assert receipt["sweep_remaining_shards"] == 1
        assert len(receipt["sweep_completed_paths"]) == 1
        assert len(receipt["sweep_deferred_paths"]) == 1
        async_sweep = meta["async_sweep"]
        assert isinstance(async_sweep, dict)
        assert async_sweep["status"] != "hit"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_server_sweep_requires_pass_roles_after_shard_gate(tmp_path: Path) -> None:
    for index in range(4):
        batch = tmp_path / f"batch_{index:05d}"
        batch.mkdir()
        shard = batch / "fullraw_shard_0000.sqlite"
        raw_file = _raw_file(tmp_path, f"pass_role_gate_{index}", [{
            "doi": f"https://doi.org/10.example/pass-role-{index}",
            "display_name": f"Management forecast disclosure pass role {index}",
            "abstract": "Management forecast disclosure evidence with adjacent operations signal.",
            "publication_year": 2024,
            "cited_by_count": 10 + index,
        }])
        index_db = FullRawFtsIndex(shard)
        try:
            index_db.index_files([raw_file], commit_interval=1)
            profile = index_db.profile()
        finally:
            index_db.close()
        (batch / "complete.json").write_text(json.dumps({
            "batch_id": index,
            "files": [{"source": "openalex"}],
            "shards": [{
                "shard_id": 0,
                "files_completed": 1,
                "papers_inserted": 1,
                "bytes_used": shard.stat().st_size,
                **profile,
            }],
        }))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": []}))
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd() / "src"),
        "V5_MEMO_FULL_RAW_INDEX_PORT": str(port),
        "V5_MEMO_FULL_RAW_MANIFEST": str(manifest),
        "V5_MEMO_FULL_RAW_SHARD_DIR": str(tmp_path),
        "V5_MEMO_FULL_RAW_SHARD_TRUST_FILENAMES": "1",
        "V5_MEMO_FULL_RAW_SHARD_MANIFEST_STATS": "1",
        "V5_MEMO_FULL_RAW_ASYNC_SWEEP": "1",
        "V5_MEMO_FULL_RAW_SWEEP_CACHE_DIR": str(tmp_path / "cache"),
        "V5_MEMO_FULL_RAW_SWEEP_SHARD_LIMIT": "4",
        "V5_MEMO_FULL_RAW_SWEEP_PASS_SHARD_LIMIT": "1",
        "V5_MEMO_FULL_RAW_SWEEP_MAX_PASSES": "3",
        "V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED": "2",
        "V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "v5_memo.fullraw_index", "serve"],
        cwd=Path.cwd(),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(50):
            try:
                urllib.request.urlopen(base + "/health", timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            _, stderr = proc.communicate(timeout=1)
            raise AssertionError(f"server did not start: {stderr}")

        def post_search(*, queue_if_missing: bool = False) -> dict[str, object]:
            request = urllib.request.Request(
                base + "/search",
                data=json.dumps({
                    "query": "management forecast disclosure",
                    "top_k": 5,
                    "cache_only": True,
                    "queue_if_missing": queue_if_missing,
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode())
            assert isinstance(payload, dict)
            return payload

        cached = post_search(queue_if_missing=True)
        for _ in range(50):
            meta = cached["meta"]
            assert isinstance(meta, dict)
            if meta["async_sweep"]["status"] == "hit":
                break
            time.sleep(0.1)
            cached = post_search(queue_if_missing=True)
        else:
            raise AssertionError("sweep did not finish")
        meta = cached["meta"]
        assert isinstance(meta, dict)
        receipt = meta["shard_receipt"]
        assert isinstance(receipt, dict)

        assert receipt["shards_searched"] == 3
        assert receipt["sweep_passes"] == 3
        assert receipt["sweep_completed_pass_roles"] == ["focused", "broad", "adjacent_field"]
        assert receipt["sweep_remaining_shards"] == 1
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_server_async_sweep_uses_cache_fit_order_for_gate(tmp_path: Path) -> None:
    specs = [
        ("openalex", 28, ("management", "forecast")),
        ("openalex", 8, ("adjacent",)),
        ("pubmed", 9, ("management",)),
        ("semantic_scholar", 29, ("management", "forecast")),
        ("semantic_scholar", 10, ("adjacent",)),
    ]
    for index, (source, manifest_size, topic_terms) in enumerate(specs):
        batch = tmp_path / f"batch_{index:05d}"
        batch.mkdir()
        shard = batch / "fullraw_shard_0000.sqlite"
        raw_file = _raw_file(tmp_path, f"cache_fit_sweep_{index}", [{
            "doi": f"https://doi.org/10.example/cache-fit-{index}",
            "display_name": f"Management forecast disclosure cache fit {index}",
            "abstract": "Management forecast disclosure cache fit evidence.",
            "publication_year": 2024,
            "cited_by_count": 100 if "forecast" in topic_terms else 1,
        }])
        index_db = FullRawFtsIndex(shard)
        try:
            index_db.index_files([raw_file], commit_interval=1)
            profile = index_db.profile()
        finally:
            index_db.close()
        shard_meta = {
            **profile,
            "shard_id": 0,
            "files_completed": 1,
            "papers_inserted": 1,
            "bytes_used": manifest_size,
            "topic_terms": list(topic_terms),
        }
        (batch / "complete.json").write_text(json.dumps({
            "batch_id": index,
            "files": [{"source": source}],
            "shards": [shard_meta],
        }))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": []}))
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd() / "src"),
        "V5_MEMO_FULL_RAW_INDEX_PORT": str(port),
        "V5_MEMO_FULL_RAW_MANIFEST": str(manifest),
        "V5_MEMO_FULL_RAW_SHARD_DIR": str(tmp_path),
        "V5_MEMO_FULL_RAW_SHARD_TRUST_FILENAMES": "1",
        "V5_MEMO_FULL_RAW_SHARD_MANIFEST_STATS": "1",
        "V5_MEMO_FULL_RAW_ASYNC_SWEEP": "1",
        "V5_MEMO_FULL_RAW_SWEEP_CACHE_DIR": str(tmp_path / "cache"),
        "V5_MEMO_FULL_RAW_SWEEP_SHARD_LIMIT": "5",
        "V5_MEMO_FULL_RAW_SWEEP_PASS_SHARD_LIMIT": "1",
        "V5_MEMO_FULL_RAW_SWEEP_MAX_PASSES": "3",
        "V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED": "3",
        "V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED": "3",
        "V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES": "27",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "v5_memo.fullraw_index", "serve"],
        cwd=Path.cwd(),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(50):
            try:
                urllib.request.urlopen(base + "/health", timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            _, stderr = proc.communicate(timeout=1)
            raise AssertionError(f"server did not start: {stderr}")

        def post_search(*, queue_if_missing: bool = False) -> tuple[int, dict[str, object]]:
            request = urllib.request.Request(
                base + "/search",
                data=json.dumps({
                    "query": "management forecast disclosure",
                    "top_k": 5,
                    "cache_only": True,
                    "queue_if_missing": queue_if_missing,
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    payload = json.loads(response.read().decode())
                    assert isinstance(payload, dict)
                    return response.status, payload
            except urllib.error.HTTPError as exc:
                payload = json.loads(exc.read().decode())
                assert isinstance(payload, dict)
                return exc.code, payload

        status, _queued = post_search(queue_if_missing=True)
        assert status == 200
        final_body: dict[str, object] = {}
        for _ in range(50):
            status, body = post_search(queue_if_missing=True)
            final_body = body
            if status == 200 and body.get("results"):
                break
            time.sleep(0.1)
        else:
            raise AssertionError("cache-fit async sweep did not satisfy coverage")
        meta = cast(dict[str, object], final_body["meta"])
        receipt = cast(dict[str, object], meta["shard_receipt"])
        raw_completed_paths = receipt["sweep_completed_paths"]
        assert isinstance(raw_completed_paths, list)
        completed_paths = tuple(str(path) for path in raw_completed_paths)

        assert receipt["shards_searched"] == 3
        assert receipt["sources_searched"] == {"openalex": 1, "pubmed": 1, "semantic_scholar": 1}
        assert any("batch_00001" in path for path in completed_paths)
        assert any("batch_00002" in path for path in completed_paths)
        assert any("batch_00004" in path for path in completed_paths)
        assert not any("batch_00000" in path for path in completed_paths)
        assert not any("batch_00003" in path for path in completed_paths)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_server_auto_sweep_skips_failed_shard_passes(tmp_path: Path) -> None:
    invalid_batch = tmp_path / "batch_00000"
    invalid_batch.mkdir()
    invalid_shard = invalid_batch / "fullraw_shard_0000.sqlite"
    invalid_shard.write_text("not sqlite")
    (invalid_batch / "complete.json").write_text(json.dumps({
        "batch_id": 0,
        "files": [{"source": "openalex"}],
        "shards": [{
            "shard_id": 0,
            "files_completed": 1,
            "papers_inserted": 1,
            "bytes_used": invalid_shard.stat().st_size,
            "topic_terms": ["management", "forecast", "disclosure"],
        }],
    }))
    for index in (1, 2):
        batch = tmp_path / f"batch_{index:05d}"
        batch.mkdir()
        shard = batch / "fullraw_shard_0000.sqlite"
        raw_file = _raw_file(tmp_path, f"auto_skip_failed_{index}", [{
            "doi": f"https://doi.org/10.example/auto-skip-{index}",
            "display_name": f"Management forecast disclosure skip failed {index}",
            "abstract": "Management forecast disclosure evidence.",
            "publication_year": 2024,
        }])
        index_db = FullRawFtsIndex(shard)
        try:
            index_db.index_files([raw_file], commit_interval=1)
            profile = index_db.profile()
        finally:
            index_db.close()
        (batch / "complete.json").write_text(json.dumps({
            "batch_id": index,
            "files": [{"source": "openalex"}],
            "shards": [{
                "shard_id": 0,
                "files_completed": 1,
                "papers_inserted": 1,
                "bytes_used": shard.stat().st_size,
                **profile,
            }],
        }))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": []}))
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd() / "src"),
        "V5_MEMO_FULL_RAW_INDEX_PORT": str(port),
        "V5_MEMO_FULL_RAW_MANIFEST": str(manifest),
        "V5_MEMO_FULL_RAW_SHARD_DIR": str(tmp_path),
        "V5_MEMO_FULL_RAW_SHARD_TRUST_FILENAMES": "1",
        "V5_MEMO_FULL_RAW_SHARD_MANIFEST_STATS": "1",
        "V5_MEMO_FULL_RAW_ASYNC_SWEEP": "1",
        "V5_MEMO_FULL_RAW_SWEEP_CACHE_DIR": str(tmp_path / "cache"),
        "V5_MEMO_FULL_RAW_SWEEP_SHARD_LIMIT": "3",
        "V5_MEMO_FULL_RAW_SWEEP_PASS_SHARD_LIMIT": "1",
        "V5_MEMO_FULL_RAW_SWEEP_MAX_PASSES": "3",
        "V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED": "2",
        "V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "v5_memo.fullraw_index", "serve"],
        cwd=Path.cwd(),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(50):
            try:
                urllib.request.urlopen(base + "/health", timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            _, stderr = proc.communicate(timeout=1)
            raise AssertionError(f"server did not start: {stderr}")

        def post_search(*, queue_if_missing: bool = False) -> tuple[int, dict[str, object]]:
            request = urllib.request.Request(
                base + "/search",
                data=json.dumps({
                    "query": "management forecast disclosure",
                    "top_k": 5,
                    "cache_only": True,
                    "queue_if_missing": queue_if_missing,
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    payload = json.loads(response.read().decode())
                    assert isinstance(payload, dict)
                    return response.status, payload
            except urllib.error.HTTPError as exc:
                payload = json.loads(exc.read().decode())
                assert isinstance(payload, dict)
                return exc.code, payload

        status, _queued = post_search(queue_if_missing=True)
        assert status == 200
        final_body: dict[str, object] = {}
        for _ in range(50):
            status, final_body = post_search()
            if status == 200 and final_body.get("results"):
                break
            time.sleep(0.1)
        else:
            raise AssertionError("auto sweep did not skip failed shard")
        meta = final_body["meta"]
        assert isinstance(meta, dict)
        receipt = meta["shard_receipt"]
        assert isinstance(receipt, dict)
        assert receipt["shards_searched"] == 2
        assert receipt["sweep_failed_shards"] == 1
        assert receipt["sweep_remaining_shards"] == 0
        assert len(receipt["sweep_failed_paths"]) == 1
        assert len(receipt["sweep_completed_paths"]) == 2
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_server_sweep_expands_selection_after_failed_shards(tmp_path: Path) -> None:
    for index in range(4):
        batch = tmp_path / f"batch_{index:05d}"
        batch.mkdir()
        invalid_shard = batch / "fullraw_shard_0000.sqlite"
        invalid_shard.write_text("not sqlite")
        (batch / "complete.json").write_text(json.dumps({
            "batch_id": index,
            "files": [{"source": "openalex"}],
            "shards": [{
                "shard_id": 0,
                "files_completed": 1,
                "papers_inserted": 1,
                "bytes_used": invalid_shard.stat().st_size,
                "cited_by_max": 1000 - index,
                "topic_terms": ["management", "forecast", "disclosure"],
            }],
        }))
    for index in range(4, 10):
        batch = tmp_path / f"batch_{index:05d}"
        batch.mkdir()
        shard = batch / "fullraw_shard_0000.sqlite"
        raw_file = _raw_file(tmp_path, f"expand_after_failed_{index}", [{
            "doi": f"https://doi.org/10.example/expand-failed-{index}",
            "display_name": f"Management forecast disclosure risk replacement {index}",
            "abstract": "Management forecast disclosure risk replacement evidence.",
            "publication_year": 2024,
            "cited_by_count": index,
        }])
        index_db = FullRawFtsIndex(shard)
        try:
            index_db.index_files([raw_file], commit_interval=1)
            profile = index_db.profile()
        finally:
            index_db.close()
        (batch / "complete.json").write_text(json.dumps({
            "batch_id": index,
            "files": [{"source": "openalex"}],
            "shards": [{
                "shard_id": 0,
                "files_completed": 1,
                "papers_inserted": 1,
                "bytes_used": shard.stat().st_size,
                **profile,
            }],
        }))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": []}))
    env = {
        **os.environ,
        "PYTHONPATH": str(Path.cwd() / "src"),
        "V5_MEMO_FULL_RAW_INDEX_PORT": str(port),
        "V5_MEMO_FULL_RAW_MANIFEST": str(manifest),
        "V5_MEMO_FULL_RAW_SHARD_DIR": str(tmp_path),
        "V5_MEMO_FULL_RAW_SHARD_TRUST_FILENAMES": "1",
        "V5_MEMO_FULL_RAW_SHARD_MANIFEST_STATS": "1",
        "V5_MEMO_FULL_RAW_ASYNC_SWEEP": "1",
        "V5_MEMO_FULL_RAW_SWEEP_CACHE_DIR": str(tmp_path / "cache"),
        "V5_MEMO_FULL_RAW_SWEEP_SHARD_LIMIT": "8",
        "V5_MEMO_FULL_RAW_SWEEP_PASS_SHARD_LIMIT": "2",
        "V5_MEMO_FULL_RAW_SWEEP_MAX_PASSES": "5",
        "V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED": "5",
        "V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "v5_memo.fullraw_index", "serve"],
        cwd=Path.cwd(),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(50):
            try:
                urllib.request.urlopen(base + "/health", timeout=1).close()
                break
            except OSError:
                time.sleep(0.1)
        else:
            _, stderr = proc.communicate(timeout=1)
            raise AssertionError(f"server did not start: {stderr}")

        def post_search(*, queue_if_missing: bool = False) -> tuple[int, dict[str, object]]:
            request = urllib.request.Request(
                base + "/search",
                data=json.dumps({
                    "query": "management forecast disclosure",
                    "top_k": 5,
                    "cache_only": True,
                    "queue_if_missing": queue_if_missing,
                }).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    payload = json.loads(response.read().decode())
                    assert isinstance(payload, dict)
                    return response.status, payload
            except urllib.error.HTTPError as exc:
                payload = json.loads(exc.read().decode())
                assert isinstance(payload, dict)
                return exc.code, payload

        status, _queued = post_search(queue_if_missing=True)
        assert status == 200
        final_body: dict[str, object] = {}
        for _ in range(200):
            status, final_body = post_search(queue_if_missing=True)
            if status == 200 and final_body.get("results"):
                break
            time.sleep(0.1)
        else:
            raise AssertionError("sweep did not replace failed selected shards")
        meta = final_body["meta"]
        assert isinstance(meta, dict)
        receipt = meta["shard_receipt"]
        assert isinstance(receipt, dict)
        assert receipt["shards_searched"] >= 5
        assert receipt["sweep_failed_shards"] == 4
        assert receipt["sweep_selected_shards"] > 8
        assert len(receipt["sweep_completed_paths"]) >= 5
        results = final_body["results"]
        assert isinstance(results, list)
        assert len(results) == 5
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_select_search_shard_entries_balances_sources_and_batches(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    entries: list[ShardCatalogEntry] = []
    for index, source in enumerate(("openalex", "openalex", "pubmed", "pubmed", "semantic_scholar", "semantic_scholar")):
        entries.append(ShardCatalogEntry(
            path=tmp_path / f"batch_{index:05d}" / "fullraw_shard_0000.sqlite",
            batch_id=index,
            shard_id=0,
            sources=(source,),
            files_completed=1,
            papers_inserted=100 + index,
            bytes_used=1000,
        ))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SEARCH_SHARD_LIMIT", "3")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SEARCH_SHARD_ORDER", "balanced")

    selected = select_search_shard_entries(entries)
    receipt = shard_coverage_receipt(entries, selected)

    assert {entry.sources[0] for entry in selected} == {"openalex", "pubmed", "semantic_scholar"}
    assert receipt["shards_total"] == 6
    assert receipt["shards_searched"] == 3
    assert receipt["partial_shard_search"] is True
    assert receipt["sources_searched"] == {"openalex": 1, "pubmed": 1, "semantic_scholar": 1}
    assert receipt["source_count_total"] == 3
    assert receipt["source_count_searched"] == 3
    assert receipt["sources_missing_from_search"] == ()


def test_select_search_shard_entries_rotates_by_query_variant(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    entries: list[ShardCatalogEntry] = []
    for index in range(12):
        source = ("openalex", "semantic_scholar", "pubmed")[index % 3]
        entries.append(ShardCatalogEntry(
            path=tmp_path / f"batch_{index:05d}" / "fullraw_shard_0000.sqlite",
            batch_id=index,
            shard_id=0,
            sources=(source,),
            files_completed=1,
            papers_inserted=100 + index,
            bytes_used=1000,
        ))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SEARCH_SHARD_LIMIT", "6")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SEARCH_SHARD_ORDER", "balanced")

    first = select_search_shard_entries(entries, query="management forecast disclosure")
    second = select_search_shard_entries(entries, query="management forecast")
    first_receipt = shard_coverage_receipt(entries, first)
    second_receipt = shard_coverage_receipt(entries, second)

    assert {entry.sources[0] for entry in first} == {"openalex", "pubmed", "semantic_scholar"}
    assert {entry.sources[0] for entry in second} == {"openalex", "pubmed", "semantic_scholar"}
    assert {entry.path for entry in first} != {entry.path for entry in second}
    assert first_receipt["shards_searched"] == 6
    assert second_receipt["shards_searched"] == 6
    assert first_receipt["sources_searched"] == {"openalex": 2, "pubmed": 2, "semantic_scholar": 2}
    assert second_receipt["sources_searched"] == {"openalex": 2, "pubmed": 2, "semantic_scholar": 2}
    assert first_receipt["batch_range_searched"] != {"min": None, "max": None}
    assert second_receipt["batch_range_searched"] != {"min": None, "max": None}


def test_select_search_shard_entries_uses_profile_diversity(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    entries = [
        ShardCatalogEntry(
            path=tmp_path / f"batch_{index:05d}" / "fullraw_shard_0000.sqlite",
            batch_id=index,
            shard_id=0,
            sources=("openalex",),
            files_completed=1,
            papers_inserted=100,
            bytes_used=1000,
            year_min=year_min,
            year_max=year_max,
            cited_by_min=cited_min,
            cited_by_max=cited_max,
            topic_terms=topic_terms,
        )
        for index, (year_min, year_max, cited_min, cited_max, topic_terms) in enumerate([
            (1975, 1985, 0, 3, ("management",)),
            (2019, 2026, 2, 400, ("forecast", "disclosure")),
            (1995, 2000, 0, 1, ("unrelated",)),
            (2008, 2012, 1, 40, ("longevity",)),
        ])
    ]
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SEARCH_SHARD_LIMIT", "3")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SEARCH_SHARD_ORDER", "balanced")

    selected = select_search_shard_entries(entries, query="management forecast disclosure")
    receipt = shard_coverage_receipt(entries, selected)

    assert entries[1] in selected
    assert receipt["year_range_searched"] == {"min": 1975, "max": 2026}
    assert receipt["cited_by_range_searched"] == {"min": 0, "max": 400}
    topic_terms_searched = receipt["topic_terms_searched"]
    assert isinstance(topic_terms_searched, tuple)
    assert "forecast" in topic_terms_searched


def test_source_scope_excludes_mixed_disallowed_source_shards(tmp_path: Path) -> None:
    entries = [
        ShardCatalogEntry(
            path=tmp_path / "batch_00000" / "fullraw_shard_0000.sqlite",
            batch_id=0,
            shard_id=0,
            sources=("openalex",),
            files_completed=1,
            papers_inserted=100,
            bytes_used=1000,
        ),
        ShardCatalogEntry(
            path=tmp_path / "batch_00001" / "fullraw_shard_0000.sqlite",
            batch_id=1,
            shard_id=0,
            sources=("openalex", "semantic_scholar"),
            files_completed=2,
            papers_inserted=200,
            bytes_used=2000,
        ),
        ShardCatalogEntry(
            path=tmp_path / "batch_00002" / "fullraw_shard_0000.sqlite",
            batch_id=2,
            shard_id=0,
            sources=("pubmed",),
            files_completed=1,
            papers_inserted=300,
            bytes_used=3000,
        ),
    ]

    scoped = fullraw_index._filter_shard_catalog_by_source(entries, ("openalex", "pubmed"))
    receipt = shard_coverage_receipt(scoped, scoped)
    fullraw_index._add_source_scope_receipt(
        receipt,
        all_entries=entries,
        scoped_entries=scoped,
        source_scope=("openalex", "pubmed"),
    )

    assert [entry.batch_id for entry in scoped] == [0, 2]
    assert receipt["sources_total"] == {"openalex": 1, "pubmed": 1}
    assert receipt["sources_excluded_by_scope"] == ("semantic_scholar",)
    assert receipt["all_sources_total"] == {"openalex": 2, "pubmed": 1, "semantic_scholar": 1}


def test_select_search_shard_entries_prefers_ready_cache_with_source_diversity(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    entries: list[ShardCatalogEntry] = []
    for index, source in enumerate(("openalex", "openalex", "pubmed", "pubmed", "semantic_scholar", "semantic_scholar")):
        path = tmp_path / f"batch_{index:05d}" / "fullraw_shard_0000.sqlite"
        path.parent.mkdir()
        path.write_bytes((f"shard-{index}").encode())
        entries.append(ShardCatalogEntry(
            path=path,
            batch_id=index,
            shard_id=0,
            sources=(source,),
            files_completed=1,
            papers_inserted=100,
            bytes_used=path.stat().st_size,
            topic_terms=("management", "forecast"),
        ))
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SEARCH_SHARD_LIMIT", "3")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SEARCH_SHARD_ORDER", "balanced")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES", "1000000")
    cached_entry = entries[1]
    cached_path = fullraw_index._shard_cache_path(cached_entry.path)
    assert cached_path is not None
    cached_path.parent.mkdir()
    cached_path.write_bytes(cached_entry.path.read_bytes())

    selected = select_search_shard_entries(entries, query="management forecast")
    receipt = shard_coverage_receipt(entries, selected)

    assert selected[0] == cached_entry
    assert receipt["sources_searched"] == {"openalex": 1, "pubmed": 1, "semantic_scholar": 1}
    assert receipt["sources_missing_from_search"] == ()


def test_fullraw_search_contract_preserves_breadth_and_depth_requirements(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    entries: list[ShardCatalogEntry] = []
    sources = ("openalex", "pubmed", "semantic_scholar")
    for source_index, source in enumerate(sources):
        entries.append(ShardCatalogEntry(
            path=tmp_path / f"batch_{source_index:05d}" / "fullraw_shard_0000.sqlite",
            batch_id=source_index,
            shard_id=0,
            sources=(source,),
            files_completed=1,
            papers_inserted=1_000,
            bytes_used=1_000,
            year_min=1975,
            year_max=1985,
            cited_by_min=0,
            cited_by_max=5,
            topic_terms=("management", "forecast"),
        ))
        entries.append(ShardCatalogEntry(
            path=tmp_path / f"batch_{source_index + 10:05d}" / "fullraw_shard_0000.sqlite",
            batch_id=source_index + 10,
            shard_id=0,
            sources=(source,),
            files_completed=1,
            papers_inserted=2_000,
            bytes_used=1_000,
            year_min=2022,
            year_max=2026,
            cited_by_min=250,
            cited_by_max=5_000,
            topic_terms=("forecast", "disclosure"),
        ))
    entries.extend([
        ShardCatalogEntry(
            path=tmp_path / "batch_00050" / "fullraw_shard_0000.sqlite",
            batch_id=50,
            shard_id=0,
            sources=("openalex",),
            files_completed=1,
            papers_inserted=500,
            bytes_used=1_000,
            year_min=2010,
            year_max=2015,
            cited_by_min=10,
            cited_by_max=100,
            topic_terms=("supply", "chain"),
        ),
        ShardCatalogEntry(
            path=tmp_path / "batch_00051" / "fullraw_shard_0000.sqlite",
            batch_id=51,
            shard_id=0,
            sources=("semantic_scholar",),
            files_completed=1,
            papers_inserted=500,
            bytes_used=1_000,
            year_min=2016,
            year_max=2020,
            cited_by_min=10,
            cited_by_max=100,
            topic_terms=("longevity", "exercise"),
        ),
    ])
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SEARCH_SHARD_LIMIT", "6")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SEARCH_SHARD_ORDER", "balanced")

    foreground = select_search_shard_entries(entries, query="management forecast disclosure")
    foreground_receipt = shard_coverage_receipt(entries, foreground)
    sweep = select_sweep_shard_entries(entries, query="management forecast disclosure", limit=8)
    passes = fullraw_index._sweep_search_passes(
        "management forecast disclosure",
        sweep,
        rank_mode="relevance",
    )
    _hits, result_receipt = fullraw_index._merge_hit_groups_with_receipt(
        [
            [
                {
                    "doi": "10.example/openalex-zero",
                    "source": "openalex",
                    "cited_by_count": 0,
                    "score": 1.0,
                },
                {
                    "doi": "10.example/pubmed-mid",
                    "source": "pubmed",
                    "cited_by_count": 50,
                    "score": 1.0,
                },
                {
                    "doi": "10.example/duplicate",
                    "source": "openalex",
                    "cited_by_count": 10,
                    "score": 1.0,
                },
            ],
            [
                {
                    "doi": "10.example/duplicate",
                    "source": "openalex",
                    "cited_by_count": 5_000,
                    "score": 2.0,
                },
                {
                    "doi": "10.example/s2-high",
                    "source": "semantic_scholar",
                    "cited_by_count": 5_000,
                    "score": 1.0,
                },
            ],
        ],
        limit=5,
    )

    assert foreground_receipt["sources_searched"] == {
        "openalex": 2,
        "pubmed": 2,
        "semantic_scholar": 2,
    }
    assert foreground_receipt["year_range_searched"] == {"min": 1975, "max": 2026}
    assert foreground_receipt["cited_by_range_searched"] == {"min": 0, "max": 5000}
    assert foreground_receipt["papers_searched"] == sum(
        entry.papers_inserted for entry in foreground
    )
    assert foreground_receipt["sources_missing_from_search"] == ()
    assert [pass_item.role for pass_item in passes] == [
        "focused",
        "broad",
        "adjacent_field",
        "falsifier",
        "citation_heavy",
        "recency",
    ]
    assert passes[4].rank_mode == "citation"
    assert passes[5].rank_mode == "recency"
    duplicate_rate = result_receipt["result_duplicate_rate"]
    citation_diversity = result_receipt["result_citation_diversity"]
    assert isinstance(duplicate_rate, float)
    assert isinstance(citation_diversity, int)
    assert duplicate_rate > 0
    assert citation_diversity >= 3
    returned_sources = result_receipt["result_sources_returned"]
    assert isinstance(returned_sources, dict)
    assert set(returned_sources) == {"openalex", "pubmed", "semantic_scholar"}


def test_select_sweep_shard_entries_expands_relevant_scope(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    entries: list[ShardCatalogEntry] = []
    sources = ("openalex", "semantic_scholar", "pubmed")
    for index in range(18):
        topic_terms = ("management", "forecast") if index % 4 == 0 else ("longevity",)
        entries.append(ShardCatalogEntry(
            path=tmp_path / f"batch_{index:05d}" / "fullraw_shard_0000.sqlite",
            batch_id=index,
            shard_id=0,
            sources=(sources[index % len(sources)],),
            files_completed=1,
            papers_inserted=100 + index,
            bytes_used=1000,
            cited_by_max=index,
            topic_terms=topic_terms,
        ))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SEARCH_SHARD_LIMIT", "3")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SEARCH_SHARD_ORDER", "balanced")

    foreground = select_search_shard_entries(entries, query="management forecast disclosure")
    sweep = select_sweep_shard_entries(entries, query="management forecast disclosure", limit=9)
    receipt = shard_coverage_receipt(entries, sweep)

    assert len(foreground) == 3
    assert len(sweep) == 9
    assert {entry.sources[0] for entry in sweep} == {"openalex", "semantic_scholar", "pubmed"}
    assert sum(1 for entry in sweep if "forecast" in entry.topic_terms) >= 3
    assert receipt["partial_shard_search"] is True
    assert receipt["shards_searched"] == 9


def test_sweep_pass_prefix_prioritizes_source_breadth(tmp_path: Path) -> None:
    entries: list[ShardCatalogEntry] = []
    for index in range(12):
        entries.append(ShardCatalogEntry(
            path=tmp_path / f"batch_{index:05d}" / "fullraw_shard_0000.sqlite",
            batch_id=index,
            shard_id=0,
            sources=("openalex",),
            files_completed=1,
            papers_inserted=1000,
            bytes_used=10_000_000 + index,
            cited_by_max=1000 + index,
            topic_terms=("management", "forecast"),
        ))
    entries.append(ShardCatalogEntry(
        path=tmp_path / "batch_90000" / "fullraw_shard_0000.sqlite",
        batch_id=90000,
        shard_id=0,
        sources=("pubmed",),
        files_completed=1,
        papers_inserted=14_214,
        bytes_used=43_184_128,
        cited_by_max=25_322,
        topic_terms=("cholestasis", "pregnancy"),
    ))
    entries.append(ShardCatalogEntry(
        path=tmp_path / "batch_00140" / "fullraw_shard_0000.sqlite",
        batch_id=140,
        shard_id=0,
        sources=("semantic_scholar",),
        files_completed=1,
        papers_inserted=800,
        bytes_used=20_000_000,
        cited_by_max=200,
        topic_terms=("pregnancy",),
    ))

    selected = select_sweep_shard_entries(entries, query="cholestasis pregnancy management", limit=9)
    prioritized = fullraw_index._prioritize_sweep_pass_entries(
        selected,
        3,
        query="cholestasis pregnancy management",
    )
    first_pass = prioritized[:3]
    receipt = shard_coverage_receipt(entries, first_pass)

    assert {entry.sources[0] for entry in first_pass} == {"openalex", "pubmed", "semantic_scholar"}
    assert receipt["sources_missing_from_search"] == ()
    assert receipt["sources_searched"] == {"openalex": 1, "pubmed": 1, "semantic_scholar": 1}


def test_sweep_pass_order_stays_source_balanced_after_first_prefix(tmp_path: Path) -> None:
    entries: list[ShardCatalogEntry] = []
    sources = ("openalex", "pubmed", "semantic_scholar")
    for index in range(12):
        source = sources[index % len(sources)]
        entries.append(ShardCatalogEntry(
            path=tmp_path / f"batch_{index:05d}" / "fullraw_shard_0000.sqlite",
            batch_id=index,
            shard_id=0,
            sources=(source,),
            files_completed=1,
            papers_inserted=1000,
            bytes_used=(12 - index) * 1_000_000,
            cited_by_max=index,
            topic_terms=("pregnancy",) if index % 2 == 0 else ("management",),
        ))

    selected = select_sweep_shard_entries(entries, query="pregnancy management", limit=12)
    prioritized = fullraw_index._prioritize_sweep_pass_entries(
        selected,
        3,
        query="pregnancy management",
    )
    first_six_receipt = shard_coverage_receipt(entries, prioritized[:6])

    assert first_six_receipt["sources_searched"] == {
        "openalex": 2,
        "pubmed": 2,
        "semantic_scholar": 2,
    }
    assert {entry.path for entry in prioritized} == {entry.path for entry in selected}


def test_expanded_sweep_reorders_replacements_before_old_pending_tail(tmp_path: Path) -> None:
    attempted = ShardCatalogEntry(
        path=tmp_path / "attempted.sqlite",
        batch_id=0,
        shard_id=0,
        sources=("openalex",),
        files_completed=1,
        papers_inserted=1,
        bytes_used=1,
    )
    old_large = ShardCatalogEntry(
        path=tmp_path / "old-large.sqlite",
        batch_id=1,
        shard_id=0,
        sources=("openalex",),
        files_completed=1,
        papers_inserted=1,
        bytes_used=9_000_000_000,
    )
    replacement_small = ShardCatalogEntry(
        path=tmp_path / "replacement-small.sqlite",
        batch_id=2,
        shard_id=0,
        sources=("openalex",),
        files_completed=1,
        papers_inserted=1,
        bytes_used=1,
    )

    reordered = fullraw_index._reorder_expanded_sweep_entries(
        [attempted, old_large],
        [attempted, replacement_small, old_large],
        attempted_paths={str(attempted.path)},
        limit=3,
    )

    assert reordered == [attempted, replacement_small, old_large]


def test_profile_relaxed_sweep_query_uses_shard_topics(tmp_path: Path) -> None:
    entries = [
        ShardCatalogEntry(
            path=tmp_path / f"batch_{index:05d}" / "fullraw_shard_0000.sqlite",
            batch_id=index,
            shard_id=0,
            sources=("openalex",),
            files_completed=1,
            papers_inserted=100,
            bytes_used=1000,
            topic_terms=topic_terms,
        )
        for index, topic_terms in enumerate([
            ("management", "forecast"),
            ("forecast", "disclosure"),
            ("management", "disclosure"),
            ("longevity", "exercise"),
        ])
    ]

    relaxed = fullraw_index._profile_relaxed_sweep_query(
        "voluntary management earnings forecast disclosure",
        entries,
    )

    assert relaxed == "management forecast disclosure"


def test_profile_relaxed_sweep_query_falls_back_to_term_map_aliases(tmp_path: Path) -> None:
    entries = [
        ShardCatalogEntry(
            path=tmp_path / "batch_00000" / "fullraw_shard_0000.sqlite",
            batch_id=0,
            shard_id=0,
            sources=("openalex",),
            files_completed=1,
            papers_inserted=100,
            bytes_used=1000,
            topic_terms=("system", "development", "time"),
        )
    ]

    relaxed = fullraw_index._profile_relaxed_sweep_query(
        "voluntary management earnings forecast disclosure",
        entries,
    )

    assert relaxed == "management forecast disclosure"


def test_sweep_search_passes_cover_required_retrieval_modes(tmp_path: Path) -> None:
    entries = [
        ShardCatalogEntry(
            path=tmp_path / f"batch_{index:05d}" / "fullraw_shard_0000.sqlite",
            batch_id=index,
            shard_id=0,
            sources=("openalex",),
            files_completed=1,
            papers_inserted=100,
            bytes_used=1000,
            topic_terms=topic_terms,
        )
        for index, topic_terms in enumerate([
            ("management", "forecast"),
            ("forecast", "disclosure"),
            ("supply", "chain"),
            ("longevity", "exercise"),
        ])
    ]

    passes = fullraw_index._sweep_search_passes(
        "voluntary management earnings forecast disclosure",
        entries,
        rank_mode="relevance",
    )

    assert [pass_item.role for pass_item in passes] == [
        "focused",
        "broad",
        "adjacent_field",
        "falsifier",
        "citation_heavy",
        "recency",
    ]
    assert passes[0].query == "management forecast disclosure"
    assert passes[0].rank_mode == "relevance"
    assert passes[2].query == "supply chain longevity"
    assert passes[3].query == "management risk"
    assert passes[4].rank_mode == "citation"
    assert passes[5].rank_mode == "recency"


def test_hit_diversity_receipt_reports_duplicates_and_citation_buckets() -> None:
    hits, receipt = fullraw_index._merge_hit_groups_with_receipt(
        [
            [
                {"doi": "10.example/one", "score": 1.0, "cited_by_count": 0},
                {"doi": "10.example/two", "score": 1.0, "cited_by_count": 25},
            ],
            [
                {"doi": "10.example/one", "score": 2.0, "cited_by_count": 1200},
                {"doi": "10.example/three", "score": 1.0, "cited_by_count": 250},
            ],
        ],
        limit=5,
    )

    assert [hit["doi"] for hit in hits] == [
        "10.example/one",
        "10.example/three",
        "10.example/two",
    ]
    assert receipt["result_count_raw"] == 4
    assert receipt["result_count_unique"] == 3
    assert receipt["result_duplicate_count"] == 1
    assert receipt["result_duplicate_rate"] == 0.25
    assert receipt["result_cited_by_range"] == {"min": 25, "max": 1200}
    assert receipt["result_citation_bucket_counts"] == {
        "high": 1,
        "medium": 1,
        "very_high": 1,
    }
    assert receipt["result_citation_diversity"] == 3
    assert receipt["result_sources_returned"] == {}
    assert receipt["result_source_count"] == 0


def test_hit_merge_keeps_lower_ranked_source_diversity() -> None:
    hits, receipt = fullraw_index._merge_hit_groups_with_receipt(
        [
            [
                {
                    "doi": "10.example/openalex-high",
                    "source": "openalex",
                    "score": 10.0,
                    "cited_by_count": 500,
                    "abstract": "OpenAlex high score.",
                },
                {
                    "doi": "10.example/openalex-mid",
                    "source": "openalex",
                    "score": 9.0,
                    "cited_by_count": 400,
                    "abstract": "OpenAlex mid score.",
                },
            ],
            [
                {
                    "doi": "10.example/semantic-low",
                    "source": "semantic_scholar",
                    "score": 1.0,
                    "cited_by_count": 2,
                    "abstract": "Semantic Scholar lower score.",
                },
                {
                    "doi": "10.example/pubmed-low",
                    "source": "pubmed",
                    "score": 0.5,
                    "cited_by_count": 0,
                    "abstract": "PubMed lower score.",
                },
            ],
        ],
        limit=3,
    )

    assert [hit["source"] for hit in hits] == ["openalex", "semantic_scholar", "pubmed"]
    assert receipt["result_sources_returned"] == {
        "openalex": 1,
        "pubmed": 1,
        "semantic_scholar": 1,
    }
    assert receipt["result_source_count"] == 3
    assert receipt["result_abstract_count"] == 3


def test_sweep_pass_roles_sufficient_requires_planned_roles() -> None:
    planned = [
        {"role": "focused"},
        {"role": "broad"},
        {"role": "adjacent_field"},
        {"role": "falsifier"},
        {"role": "citation_heavy"},
        {"role": "recency"},
    ]

    assert not fullraw_index._sweep_pass_roles_sufficient({
        "sweep_search_passes": planned,
        "sweep_completed_pass_roles": ["focused", "broad", "adjacent_field", "falsifier", "citation_heavy"],
        "sweep_max_passes": 10,
    })
    assert fullraw_index._sweep_pass_roles_sufficient({
        "sweep_search_passes": planned,
        "sweep_completed_pass_roles": ["focused", "broad", "adjacent_field", "falsifier", "citation_heavy", "recency"],
        "sweep_max_passes": 10,
    })
    assert fullraw_index._sweep_pass_roles_sufficient({
        "sweep_search_passes": planned,
        "sweep_completed_pass_roles": ["focused", "broad", "adjacent_field"],
        "sweep_max_passes": 3,
    })


def _sweep_task(key: str) -> SweepTask:
    return SweepTask(
        key=key,
        query=key,
        limit=5,
        year_min=1900,
        year_max=2100,
        rank_mode="default",
        catalog=[],
    )


def test_sweep_scheduler_queues_when_inflight_full() -> None:
    scheduler = SweepScheduler(max_inflight=1, max_pending=2)

    assert scheduler.enqueue(_sweep_task("a")) == ("queued", True)
    assert scheduler.status("a") == "running"
    assert scheduler.enqueue(_sweep_task("b")) == ("queued", False)
    assert scheduler.status("b") == "queued"
    assert scheduler.enqueue(_sweep_task("b")) == ("queued", False)

    next_task = scheduler.finish("a")

    assert next_task is not None
    assert next_task.key == "b"
    assert scheduler.status("b") == "running"
    assert scheduler.finish("b") is None
    assert scheduler.status("b") == "miss"


def test_sweep_scheduler_reports_busy_when_pending_full() -> None:
    scheduler = SweepScheduler(max_inflight=1, max_pending=1)

    assert scheduler.enqueue(_sweep_task("a")) == ("queued", True)
    assert scheduler.enqueue(_sweep_task("b")) == ("queued", False)
    assert scheduler.enqueue(_sweep_task("c")) == ("busy", False)


def test_sweep_cache_key_includes_sweep_shard_limit() -> None:
    small = fullraw_index._sweep_cache_key(
        "management forecast disclosure",
        limit=5,
        year_min=1900,
        year_max=2100,
        rank_mode="citation",
        sweep_shard_limit=32,
    )
    large = fullraw_index._sweep_cache_key(
        "management forecast disclosure",
        limit=5,
        year_min=1900,
        year_max=2100,
        rank_mode="citation",
        sweep_shard_limit=128,
    )

    assert small != large


def test_sweep_cache_key_includes_sweep_strategy() -> None:
    old_strategy = fullraw_index._sweep_cache_key(
        "management forecast disclosure",
        limit=5,
        year_min=1900,
        year_max=2100,
        rank_mode="citation",
        sweep_shard_limit=48,
        sweep_strategy="old",
    )
    current_strategy = fullraw_index._sweep_cache_key(
        "management forecast disclosure",
        limit=5,
        year_min=1900,
        year_max=2100,
        rank_mode="citation",
        sweep_shard_limit=48,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
    )

    assert old_strategy != current_strategy


def test_sweep_cache_key_includes_source_scope() -> None:
    all_sources = fullraw_index._sweep_cache_key(
        "management forecast disclosure",
        limit=5,
        year_min=1900,
        year_max=2100,
        rank_mode="citation",
        sweep_shard_limit=128,
        source_scope=(),
    )
    scoped = fullraw_index._sweep_cache_key(
        "management forecast disclosure",
        limit=5,
        year_min=1900,
        year_max=2100,
        rank_mode="citation",
        sweep_shard_limit=128,
        source_scope=("openalex", "pubmed"),
    )

    assert all_sources != scoped


def test_shard_coverage_gate_response_rejects_too_narrow_search() -> None:
    status, body = shard_coverage_gate_response(
        {
            "shards_searched": 12,
            "sources_searched": {"openalex": 12},
        },
        min_shards_searched=50,
        min_sources_searched=2,
    ) or (0, {})

    assert status == 422
    assert body["error"] == "coverage_too_narrow"
    assert body["requirements"] == {
        "min_shards_searched": 50,
        "min_sources_searched": 2,
    }
    assert body["shard_receipt"] == {
        "shards_searched": 12,
        "sources_searched": {"openalex": 12},
    }


def test_shard_coverage_gate_response_allows_sufficient_search() -> None:
    assert shard_coverage_gate_response(
        {
            "shards_searched": 50,
            "sources_searched": {"openalex": 25, "semantic_scholar": 25},
        },
        min_shards_searched=50,
        min_sources_searched=2,
    ) is None


def test_select_search_shard_paths_can_spread(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    paths = [tmp_path / f"batch_{index:05d}" / "fullraw_shard_0000.sqlite" for index in range(5)]
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SEARCH_SHARD_LIMIT", "3")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SEARCH_SHARD_ORDER", "spread")

    assert select_search_shard_paths(paths) == [paths[0], paths[2], paths[4]]


def test_build_upload_shard_batches_uploads_and_deletes_local_batches(tmp_path: Path) -> None:
    files: list[RawFile] = []
    for index in range(4):
        files.append(_raw_file(tmp_path, f"openalex_batch_{index}", [{
            "doi": f"https://doi.org/10.example/batch-{index}",
            "display_name": f"Management forecast disclosure batch {index}",
            "abstract": "Managers disclose forecasts and guidance.",
            "publication_year": 2024,
        }]))

    local_build = tmp_path / "local-build"
    remote = tmp_path / "remote"
    results = build_upload_shard_batches(
        files,
        shard_dir=local_build,
        upload_remote=f"file://{remote}",
        batch_files=2,
        shard_count=2,
        workers=2,
        commit_interval=1,
        delete_local=True,
    )
    repeated = build_upload_shard_batches(
        files,
        shard_dir=local_build,
        upload_remote=f"file://{remote}",
        batch_files=2,
        shard_count=2,
        workers=2,
        commit_interval=1,
        delete_local=True,
    )

    assert len(results) == 2
    assert sum(result.files_completed for result in results) == 4
    assert sum(result.papers_inserted for result in results) == 4
    assert all(result.uploaded for result in results)
    assert all(result.deleted_local for result in results)
    assert not any((local_build / f"batch_{index:05d}").exists() for index in range(2))
    assert (remote / "batch_00000" / "complete.json").exists()
    assert (remote / "batch_00001" / "complete.json").exists()
    assert len(list((remote / "batch_00000").glob("*.sqlite"))) == 2
    assert all(result.skipped for result in repeated)


def test_build_upload_shard_batches_resumes_partial_local_batch(tmp_path: Path) -> None:
    raw_file = _raw_file(tmp_path, "resume_batch", [{
        "doi": "https://doi.org/10.example/resume-batch",
        "display_name": "Management forecast disclosure resume batch",
        "abstract": "Managers disclose forecasts and guidance.",
        "publication_year": 2024,
    }])
    local_build = tmp_path / "local-build"
    remote = tmp_path / "remote"
    shard_path = local_build / "batch_00000" / "fullraw_shard_0000.sqlite"
    index = FullRawFtsIndex(shard_path)
    try:
        index.initialize()
        assert index._insert_hit(
            {
                "doi": "https://doi.org/10.example/stale-partial",
                "title": "Stale partial row",
                "abstract": "This row should be removed before resume.",
                "source": "openalex",
            },
            source_remote=raw_file.remote,
        )
        index._bump_papers(1)
        index._mark_file(raw_file, status="running", docs_seen=1, docs_indexed=1)
        index._conn.commit()
    finally:
        index.close()

    results = build_upload_shard_batches(
        [raw_file],
        shard_dir=local_build,
        upload_remote=f"file://{remote}",
        batch_files=1,
        shard_count=1,
        workers=1,
        commit_interval=1,
        delete_local=False,
    )

    resumed = FullRawFtsIndex(shard_path, read_only=True)
    try:
        stats = resumed.stats(files_total=1)
        hits = resumed.search("resume batch", limit=5)
    finally:
        resumed.close()

    assert len(results) == 1
    assert results[0].uploaded is True
    assert results[0].files_completed == 1
    assert results[0].papers_inserted == 1
    assert stats.papers_indexed == 1
    assert stats.files_indexed == 1
    assert [hit["doi"] for hit in hits] == ["10.example/resume-batch"]
    assert (remote / "batch_00000" / "complete.json").exists()


def test_build_upload_shard_batches_uses_batch_id_offset(tmp_path: Path) -> None:
    files = [
        _raw_file(tmp_path, "offset_a", [{
            "doi": "https://doi.org/10.example/offset-a",
            "display_name": "Offset batch evidence A",
        }]),
        _raw_file(tmp_path, "offset_b", [{
            "doi": "https://doi.org/10.example/offset-b",
            "display_name": "Offset batch evidence B",
        }]),
    ]
    remote = tmp_path / "remote"

    results = build_upload_shard_batches(
        files,
        shard_dir=tmp_path / "local-build",
        upload_remote=f"file://{remote}",
        batch_files=1,
        shard_count=1,
        workers=1,
        commit_interval=1,
        delete_local=True,
        batch_id_offset=90000,
    )

    assert [result.batch_id for result in results] == [90000, 90001]
    assert (remote / "batch_90000" / "complete.json").exists()
    assert (remote / "batch_90001" / "complete.json").exists()
    assert not (remote / "batch_00000").exists()


def test_build_upload_shard_batches_skips_completed_raw_remotes(tmp_path: Path) -> None:
    files = [
        _raw_file(tmp_path, "skip_a", [{
            "doi": "https://doi.org/10.example/skip-a",
            "display_name": "Skip completed remote evidence A",
        }]),
        _raw_file(tmp_path, "skip_b", [{
            "doi": "https://doi.org/10.example/skip-b",
            "display_name": "Skip completed remote evidence B",
        }]),
        _raw_file(tmp_path, "skip_c", [{
            "doi": "https://doi.org/10.example/skip-c",
            "display_name": "Skip completed remote evidence C",
        }]),
    ]
    completed = tmp_path / "completed"
    (completed / "batch_90000").mkdir(parents=True)
    (completed / "batch_90000" / "complete.json").write_text(json.dumps({
        "files": [{"remote": files[1].remote}],
    }))
    remote = tmp_path / "remote"

    results = build_upload_shard_batches(
        files,
        shard_dir=tmp_path / "local-build",
        upload_remote=f"file://{remote}",
        batch_files=10,
        shard_count=1,
        workers=1,
        commit_interval=1,
        delete_local=True,
        completed_shard_dir=completed,
    )

    manifest = json.loads((remote / "batch_00000" / "complete.json").read_text())
    assert len(results) == 1
    assert results[0].files_completed == 2
    assert [item["remote"] for item in manifest["files"]] == [files[0].remote, files[2].remote]


def test_build_upload_shard_batches_preserves_ids_when_skipping_completed_remotes(tmp_path: Path) -> None:
    files = [
        _raw_file(tmp_path, "preserve_a", [{
            "doi": "https://doi.org/10.example/preserve-a",
            "display_name": "Preserve batch id evidence A",
        }]),
        _raw_file(tmp_path, "preserve_b", [{
            "doi": "https://doi.org/10.example/preserve-b",
            "display_name": "Preserve batch id evidence B",
        }]),
        _raw_file(tmp_path, "preserve_c", [{
            "doi": "https://doi.org/10.example/preserve-c",
            "display_name": "Preserve batch id evidence C",
        }]),
    ]
    remote = tmp_path / "remote"
    completed = remote
    (remote / "batch_00000").mkdir(parents=True)
    (remote / "batch_00000" / "complete.json").write_text(json.dumps({
        "files": [{"remote": files[0].remote}],
    }))

    results = build_upload_shard_batches(
        files,
        shard_dir=tmp_path / "local-build",
        upload_remote=f"file://{remote}",
        batch_files=1,
        shard_count=1,
        workers=1,
        commit_interval=1,
        delete_local=True,
        completed_shard_dir=completed,
    )

    first_new_manifest = json.loads((remote / "batch_00001" / "complete.json").read_text())
    assert [result.batch_id for result in results] == [0, 1, 2]
    assert results[0].skipped is True
    assert first_new_manifest["files"][0]["remote"] == files[1].remote


def test_completed_raw_remotes_reads_rclone_remote(monkeypatch: MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(args: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(tuple(args))
        if args[:3] == ["rclone", "lsf", "-R"]:
            return SimpleNamespace(returncode=0, stdout="batch_91000/complete.json\n", stderr="")
        if args[:2] == ["rclone", "cat"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"files": [{"remote": "sb:raw/pubmed26n1336.xml.gz"}]}),
                stderr="",
            )
        return SimpleNamespace(returncode=1, stdout="", stderr="unexpected")

    monkeypatch.setattr("v5_memo.fullraw_index.subprocess.run", fake_run)

    remotes = fullraw_index._completed_raw_remotes(None, shard_remote="sb:index/fullraw-fts")

    assert remotes == {"sb:raw/pubmed26n1336.xml.gz"}
    assert calls == [
        ("rclone", "lsf", "-R", "sb:index/fullraw-fts"),
        ("rclone", "cat", "sb:index/fullraw-fts/batch_91000/complete.json"),
    ]


def test_fullraw_readiness_counts_remaining_manifest_files(tmp_path: Path) -> None:
    completed = tmp_path / "completed" / "batch_00000"
    completed.mkdir(parents=True)
    (completed / "complete.json").write_text(json.dumps({
        "files": [
            {"remote": "sb:raw/openalex/a.gz"},
            {"remote": "sb:raw/pubmed/a.gz"},
        ],
    }))
    files = [
        RawFile(source="openalex", format="openalex_jsonl", remote="sb:raw/openalex/a.gz"),
        RawFile(source="pubmed", format="pubmed_xml", remote="sb:raw/pubmed/a.gz"),
        RawFile(source="pubmed", format="pubmed_xml", remote="sb:raw/pubmed/b.gz"),
        RawFile(source="semantic_scholar", format="semantic_scholar_jsonl", remote="sb:raw/s2/a.gz"),
    ]

    readiness = fullraw_index.fullraw_readiness(files, completed_shard_dir=tmp_path / "completed")

    assert readiness["ready"] is False
    assert readiness["ready_scope"] == "searchable_files"
    assert readiness["files_total"] == 4
    assert readiness["files_completed"] == 2
    assert readiness["files_remaining"] == 2
    assert readiness["completed_by_source"] == {"openalex": 1, "pubmed": 1}
    assert readiness["remaining_by_source"] == {"pubmed": 1, "semantic_scholar": 1}
    assert readiness["searchable_files_remaining"] == 2
    assert readiness["enrichment_files_remaining"] == 0


def test_fullraw_readiness_is_ready_when_all_manifest_files_are_completed(tmp_path: Path) -> None:
    completed = tmp_path / "completed" / "batch_00000"
    completed.mkdir(parents=True)
    (completed / "complete.json").write_text(json.dumps({
        "files": [
            {"remote": "sb:raw/openalex/a.gz"},
            {"remote": "sb:raw/pubmed/a.gz"},
        ],
    }))
    files = [
        RawFile(source="openalex", format="openalex_jsonl", remote="sb:raw/openalex/a.gz"),
        RawFile(source="pubmed", format="pubmed_xml", remote="sb:raw/pubmed/a.gz"),
    ]

    readiness = fullraw_index.fullraw_readiness(files, completed_shard_dir=tmp_path / "completed")

    assert readiness["ready"] is True
    assert readiness["files_total"] == 2
    assert readiness["files_completed"] == 2
    assert readiness["files_remaining"] == 0
    assert readiness["remaining_by_source"] == {}


def test_fullraw_readiness_reports_enrichment_backlog_without_blocking_searchable_ready(
    tmp_path: Path,
) -> None:
    completed = tmp_path / "completed" / "batch_00000"
    completed.mkdir(parents=True)
    (completed / "complete.json").write_text(json.dumps({
        "files": [{"remote": "sb:raw/semantic_scholar/paper-a.gz"}],
    }))
    files = [
        RawFile(
            source="semantic_scholar",
            format="semantic_scholar_jsonl",
            remote="sb:raw/semantic_scholar/paper-a.gz",
        ),
        RawFile(
            source="semantic_scholar_abstracts",
            format="semantic_scholar_jsonl",
            remote="sb:raw/semantic_scholar/abstract-a.gz",
        ),
    ]

    readiness = fullraw_index.fullraw_readiness(files, completed_shard_dir=tmp_path / "completed")
    strict = fullraw_index.fullraw_readiness(
        files,
        completed_shard_dir=tmp_path / "completed",
        require_enrichment=True,
    )

    assert readiness["ready"] is True
    assert readiness["ready_scope"] == "searchable_files"
    assert readiness["files_remaining"] == 1
    assert readiness["searchable_files_remaining"] == 0
    assert readiness["enrichment_files_remaining"] == 1
    assert readiness["enrichment_remaining_by_source"] == {"semantic_scholar_abstracts": 1}
    assert strict["ready"] is False
    assert strict["ready_scope"] == "all_files"


def test_readiness_cli_fails_closed_when_manifest_is_incomplete(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "files": [
            {"source": "openalex", "format": "openalex_jsonl", "remote": "sb:raw/openalex/a.gz"},
            {"source": "pubmed", "format": "pubmed_xml", "remote": "sb:raw/pubmed/a.gz"},
        ],
    }))
    completed = tmp_path / "completed" / "batch_00000"
    completed.mkdir(parents=True)
    (completed / "complete.json").write_text(json.dumps({
        "files": [{"remote": "sb:raw/openalex/a.gz"}],
    }))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fullraw_index.py",
            "readiness",
            "--manifest",
            str(manifest),
            "--completed-shard-dir",
            str(tmp_path / "completed"),
            "--fail-if-not-ready",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        fullraw_index.main()

    assert exc.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert payload["files_completed"] == 1
    assert payload["files_remaining"] == 1
    assert payload["remaining_by_source"] == {"pubmed": 1}


def test_readiness_cli_can_require_enrichment_backlog(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "files": [
            {
                "source": "semantic_scholar",
                "format": "semantic_scholar_jsonl",
                "remote": "sb:raw/semantic_scholar/paper-a.gz",
            },
            {
                "source": "semantic_scholar_abstracts",
                "format": "semantic_scholar_jsonl",
                "remote": "sb:raw/semantic_scholar/abstract-a.gz",
            },
        ],
    }))
    completed = tmp_path / "completed" / "batch_00000"
    completed.mkdir(parents=True)
    (completed / "complete.json").write_text(json.dumps({
        "files": [{"remote": "sb:raw/semantic_scholar/paper-a.gz"}],
    }))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fullraw_index.py",
            "readiness",
            "--manifest",
            str(manifest),
            "--completed-shard-dir",
            str(tmp_path / "completed"),
            "--require-enrichment",
            "--fail-if-not-ready",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        fullraw_index.main()

    assert exc.value.code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert payload["ready_scope"] == "all_files"
    assert payload["searchable_files_remaining"] == 0
    assert payload["enrichment_files_remaining"] == 1


def test_build_upload_shard_batches_uploads_with_corrupt_file_quarantined(tmp_path: Path) -> None:
    good = _raw_file(tmp_path, "good_batch", [{
        "doi": "https://doi.org/10.example/good-batch",
        "display_name": "Management forecast disclosure batch",
        "abstract": "Managers disclose forecasts and guidance.",
    }])
    bad = _corrupt_raw_file(tmp_path, "bad_batch")
    local_build = tmp_path / "local-build"
    remote = tmp_path / "remote"

    results = build_upload_shard_batches(
        [bad, good],
        shard_dir=local_build,
        upload_remote=f"file://{remote}",
        batch_files=2,
        shard_count=1,
        workers=1,
        commit_interval=1,
        delete_local=False,
    )

    manifest = json.loads((remote / "batch_00000" / "complete.json").read_text())
    assert len(results) == 1
    assert results[0].uploaded is True
    assert results[0].error == ""
    assert results[0].files_completed == 1
    assert results[0].files_failed == 1
    assert results[0].papers_inserted == 1
    assert "bad_batch.jsonl.gz" in results[0].file_errors
    assert manifest["totals"]["files_completed"] == 1
    assert manifest["totals"]["files_failed"] == 1
    assert manifest["totals"]["file_errors"]


def test_build_upload_shard_batches_rejects_zero_value_abstract_batches(tmp_path: Path) -> None:
    abstract_file = _raw_file(tmp_path, "s2_abstract_only", [{
        "corpusid": 12345,
        "abstract": "Standalone abstract with no matching paper row.",
    }])

    results = build_upload_shard_batches(
        [
            RawFile(
                source="semantic_scholar_abstracts",
                format=abstract_file.format,
                remote=abstract_file.remote,
            )
        ],
        shard_dir=tmp_path / "local-build",
        upload_remote=f"file://{tmp_path / 'remote'}",
        batch_files=1,
        shard_count=1,
        workers=1,
        commit_interval=1,
        delete_local=False,
    )

    assert len(results) == 1
    assert results[0].uploaded is False
    assert results[0].files_completed == 1
    assert results[0].papers_inserted == 0
    assert "semantic_scholar_abstracts indexed zero papers" in results[0].error
    assert not (tmp_path / "remote" / "batch_00000" / "complete.json").exists()


def test_build_upload_shard_batches_keeps_all_failed_batch_fatal(tmp_path: Path) -> None:
    bad = _corrupt_raw_file(tmp_path, "all_bad")

    results = build_upload_shard_batches(
        [bad],
        shard_dir=tmp_path / "local-build",
        upload_remote=f"file://{tmp_path / 'remote'}",
        batch_files=1,
        shard_count=1,
        workers=1,
        commit_interval=1,
        delete_local=False,
    )

    assert len(results) == 1
    assert results[0].uploaded is False
    assert results[0].files_completed == 0
    assert results[0].files_failed == 1
    assert "all files failed" in results[0].error
    assert not (tmp_path / "remote" / "batch_00000" / "complete.json").exists()


def test_build_upload_shards_cli_exits_nonzero_on_failed_batch(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sys, "argv", ["fullraw_index.py", "build-upload-shards"])
    monkeypatch.setattr(fullraw_index, "load_or_build_manifest", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        fullraw_index,
        "build_upload_shard_batches",
        lambda *_args, **_kwargs: [
            ShardBatchResult(
                batch_id=137,
                batch_dir=str(tmp_path / "batch_00137"),
                remote_dir="sb:test/batch_00137",
                files_total=16,
                files_completed=10,
                files_failed=0,
                papers_inserted=37_841_334,
                bytes_used=17_475_444_736,
                uploaded=False,
                deleted_local=False,
                skipped=False,
                elapsed_seconds=9027.973,
                error="Compressed file ended before the end-of-stream marker was reached",
            )
        ],
    )

    with pytest.raises(SystemExit) as exc:
        fullraw_index.main()

    assert exc.value.code == 2


def test_build_upload_shards_cli_filters_source_and_offsets_batches(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    files = [
        RawFile(source="openalex", format="openalex_jsonl", remote="file:///openalex.gz"),
        RawFile(source="pubmed", format="pubmed_xml", remote="file:///pubmed-a.gz"),
        RawFile(source="semantic_scholar", format="semantic_scholar_jsonl", remote="file:///s2.gz"),
        RawFile(source="pubmed", format="pubmed_xml", remote="file:///pubmed-b.gz"),
    ]
    seen: dict[str, object] = {}

    def fake_build_upload_shard_batches(
        selected: list[RawFile],
        **kwargs: object,
    ) -> list[ShardBatchResult]:
        seen["files"] = selected
        seen.update(kwargs)
        return []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fullraw_index.py",
            "build-upload-shards",
            "--source-filter",
            "pubmed",
            "--batch-id-offset",
            "90000",
            "--completed-shard-dir",
            str(tmp_path / "completed"),
            "--completed-shard-remote",
            "sb:index/fullraw-fts",
            "--upload-remote",
            f"file://{tmp_path / 'remote'}",
        ],
    )
    monkeypatch.setattr(fullraw_index, "load_or_build_manifest", lambda *_args, **_kwargs: files)
    monkeypatch.setattr(fullraw_index, "build_upload_shard_batches", fake_build_upload_shard_batches)

    fullraw_index.main()

    selected_files = cast(list[RawFile], seen["files"])
    assert [raw_file.remote for raw_file in selected_files] == [
        "file:///pubmed-a.gz",
        "file:///pubmed-b.gz",
    ]
    assert seen["batch_id_offset"] == 90000
    assert seen["completed_shard_dir"] == tmp_path / "completed"
    assert seen["completed_shard_remote"] == "sb:index/fullraw-fts"
