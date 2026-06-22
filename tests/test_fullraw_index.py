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


def _raw_file(tmp_path: Path, name: str, rows: list[dict[str, object]]) -> RawFile:
    source = tmp_path / f"{name}.jsonl.gz"
    _write_jsonl_gzip(source, rows)
    return RawFile(source="openalex", format="openalex_jsonl", remote=f"file://{source}")


def _corrupt_raw_file(tmp_path: Path, name: str) -> RawFile:
    source = tmp_path / f"{name}.jsonl.gz"
    source.write_bytes(gzip.compress(b'{"display_name":"truncated gzip"}\n')[:-8])
    return RawFile(source="openalex", format="openalex_jsonl", remote=f"file://{source}")


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
