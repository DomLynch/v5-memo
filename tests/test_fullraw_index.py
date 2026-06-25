import gzip
import json
import os
import time
from pathlib import Path

import pytest

import v5_memo.fullraw_index as fullraw_index
from v5_memo.fullraw_index import (
    FullRawFtsIndex,
    ShardCatalogEntry,
    build_upload_shard_batches,
    select_search_shard_entries,
    select_sweep_shard_entries,
)
from v5_memo.fullraw_service import RawFile


def _gz(path: Path, rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wb") as fh:
        for row in rows:
            fh.write((json.dumps(row) + "\n").encode())


def _raw(tmp_path: Path, name: str, rows: list[dict[str, object]]) -> RawFile:
    path = tmp_path / f"{name}.jsonl.gz"
    _gz(path, rows)
    return RawFile(source="openalex", format="openalex_jsonl", remote=f"file://{path}")


def _entry(tmp_path: Path, index: int, source: str) -> ShardCatalogEntry:
    path = tmp_path / f"batch_{index:05d}" / "fullraw_shard_0000.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"sqlite")
    return ShardCatalogEntry(
        path=path,
        batch_id=index,
        shard_id=0,
        sources=(source,),
        files_completed=1,
        papers_inserted=10,
        bytes_used=path.stat().st_size,
        cited_by_max=index,
        topic_terms=("resveratrol",) if index % 2 else ("metformin",),
    )


def test_fullraw_index_builds_searchable_ranked_index(tmp_path: Path) -> None:
    index = FullRawFtsIndex(tmp_path / "fullraw.sqlite")
    try:
        result = index.index_files([_raw(tmp_path, "openalex", [
            {"doi": "https://doi.org/10.example/guidance", "display_name": "Management guidance and earnings surprises", "abstract": "Managers issue guidance before earnings surprises.", "publication_year": 2024, "cited_by_count": 20},
            {"doi": "https://doi.org/10.example/noise", "display_name": "Island species forecast ecology", "abstract": "Forecasts for climate space under grazing pressure."},
        ])], commit_interval=1)
        hits = index.search("management guidance earnings", limit=5)
        relaxed_hits = index.search("management forecast earnings guidance", limit=5)
        stats = index.stats(files_total=1)
    finally:
        index.close()
    assert (result.files_completed, result.papers_inserted, stats.papers_indexed) == (1, 2, 2)
    assert hits[0]["doi"] == "10.example/guidance"
    assert relaxed_hits[0]["doi"] == "10.example/guidance"


def test_fullraw_index_enriches_abstract_only_rows(tmp_path: Path) -> None:
    paper, abstract = tmp_path / "paper.jsonl.gz", tmp_path / "abstract.jsonl.gz"
    _gz(paper, [{"corpusid": 123, "title": "Resveratrol exercise training adaptation"}])
    _gz(abstract, [{"corpusid": 123, "abstract": "Resveratrol blunted mitochondrial adaptation."}])
    index = FullRawFtsIndex(tmp_path / "fullraw.sqlite")
    try:
        index.index_files([
            RawFile(source="semantic_scholar", format="semantic_scholar_jsonl", remote=f"file://{paper}"),
            RawFile(source="semantic_scholar_abstracts", format="semantic_scholar_jsonl", remote=f"file://{abstract}"),
        ])
        hits = index.search("mitochondrial adaptation", limit=5)
    finally:
        index.close()
    assert hits[0]["semantic_scholar_id"] == "123"
    assert "blunted mitochondrial" in str(hits[0]["abstract"])


def test_fullraw_index_quarantines_corrupt_source_file(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad.jsonl.gz"
    bad_path.write_bytes(gzip.compress(b'{"display_name":"truncated"}\n')[:-8])
    bad = RawFile(source="openalex", format="openalex_jsonl", remote=f"file://{bad_path}")
    good = _raw(tmp_path, "good", [{
        "doi": "https://doi.org/10.example/good", "display_name": "Management forecast disclosure", "abstract": "Managers disclose forecasts and guidance.",
    }])
    index = FullRawFtsIndex(tmp_path / "fullraw.sqlite")
    try:
        result = index.index_files([bad, good], commit_interval=1)
        stats = index.stats(files_total=2)
    finally:
        index.close()
    assert (result.files_completed, result.files_failed, result.papers_inserted) == (1, 1, 1)
    assert stats.files_indexed == 1


def test_shard_search_returns_partial_hits_on_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fast, slow = tmp_path / "fast.sqlite", tmp_path / "slow.sqlite"
    fast.touch()
    slow.touch()

    def fake_search(path: Path, *args: object, **kwargs: object) -> list[dict[str, object]]:
        del args, kwargs
        if path == slow:
            time.sleep(0.2)
        return [{"doi": f"10.example/{path.stem}", "title": path.stem, "score": 1.0}]
    monkeypatch.setattr(fullraw_index, "_search_one_shard", fake_search)
    hits, paths, timed_out = fullraw_index._search_shard_paths_with_paths(
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
    assert paths == [fast]
    assert [hit["doi"] for hit in hits] == ["10.example/fast"]

    called: list[Path] = []

    def counting_search(path: Path, *args: object, **kwargs: object) -> list[dict[str, object]]:
        del args, kwargs
        called.append(path)
        return []

    monkeypatch.setattr(fullraw_index, "_search_one_shard", counting_search)
    many_paths = [tmp_path / f"many_{idx}.sqlite" for idx in range(200)]
    fullraw_index._search_shard_paths_with_paths(
        many_paths, "metformin", limit=5, year_min=1900, year_max=2100, rank_mode="relevance", timeout_seconds=0.01
    )
    assert len(called) <= fullraw_index._FULL_COVERAGE_PREFIX_SHARDS
    assert many_paths[-1] not in called

def test_select_search_shard_entries_balances_sources_and_rotates_by_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [_entry(tmp_path, idx, "openalex" if idx < 4 else "pubmed") for idx in range(8)]
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SEARCH_SHARD_LIMIT", "4")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SEARCH_SHARD_ORDER", "balanced")
    metformin = select_search_shard_entries(entries, query="metformin exercise")
    resveratrol = select_search_shard_entries(entries, query="resveratrol exercise")
    assert {entry.sources[0] for entry in metformin} == {"openalex", "pubmed"}
    assert [entry.path for entry in metformin] != [entry.path for entry in resveratrol]
    assert fullraw_index._profiled_spread_entries(entries, 4, query="cold immersion") == fullraw_index._spread_entries(fullraw_index._rotate_entries(entries, fullraw_index._query_offset("cold immersion", len(entries))), 4)

def test_search_shard_selection_honors_minimum_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [_entry(tmp_path, idx, "openalex" if idx < 4 else "pubmed") for idx in range(8)]
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SEARCH_SHARD_LIMIT", "3")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED", "6")
    selected = select_search_shard_entries(entries, query="metformin exercise")
    assert len(selected) == 6


def test_search_shard_selection_prefers_cache_fit_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    small = ShardCatalogEntry(tmp_path / "small.sqlite", 1, 0, ("openalex",), 1, 10, 10, topic_terms=("patients",))
    huge = ShardCatalogEntry(tmp_path / "huge.sqlite", 2, 0, ("openalex",), 1, 10_000, 10_000, topic_terms=("patients",))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SEARCH_SHARD_LIMIT", "1")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES", "100")
    assert select_search_shard_entries([huge, small], query="patients")[0] == small
    huge.path.write_bytes(b"x")
    cached = ShardCatalogEntry(tmp_path / "cached.sqlite", 3, 0, ("pubmed",), 1, 20, 20, topic_terms=("patients",))
    cached.path.write_bytes(b"x")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_DIR", str(tmp_path / "cache"))
    fullraw_index._materialized_shard_path(huge.path)
    fullraw_index._materialized_shard_path(cached.path)
    assert set(select_search_shard_entries([huge, small, cached], query="patients")[:2]) == {huge, cached}


def test_full_coverage_shard_selection_frontloads_spread_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [_entry(tmp_path, idx, "openalex") for idx in range(320)]
    monkeypatch.delenv("V5_MEMO_FULL_RAW_SEARCH_SHARD_LIMIT", raising=False)
    selected = select_search_shard_entries(entries, query="metformin")
    swept = select_sweep_shard_entries(entries, query="metformin", limit=len(entries))
    assert len(selected) == len(entries)
    assert selected[:6] == swept[:6]
    assert max(entry.batch_id for entry in selected[:20]) > 30
    assert [entry.batch_id for entry in selected[:6]] != list(range(6))

def test_sweep_passes_do_not_invent_side_queries(tmp_path: Path) -> None:
    entries = [_entry(tmp_path, idx, "openalex") for idx in range(4)]
    passes = fullraw_index._sweep_search_passes("resveratrol exercise training adaptation", entries, rank_mode="relevance")
    assert [item.role for item in passes] == ["focused", "broad", "citation_heavy", "recency"]
    assert all("risk" not in item.query and "patients" not in item.query for item in passes)
    assert fullraw_index._sweep_search_passes("metformin blunts muscle hypertrophy progressive resistance training", entries, rank_mode="relevance")[0].query.startswith("metformin ")

def test_materialized_shard_cache_evicts_old_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    old, newer, keep = (cache_dir / name for name in ("old.sqlite", "newer.sqlite", "keep.sqlite"))
    for path in (old, newer, keep):
        path.write_bytes(b"x" * 6)
    for path, stamp in ((old, 1), (newer, 2), (keep, 3)):
        os.utime(path, (stamp, stamp))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES", "12")
    fullraw_index._evict_shard_cache(cache_dir, required_bytes=0, keep=keep)
    assert not old.exists()
    assert newer.exists()
    assert keep.exists()


def test_build_upload_shard_batches_keeps_all_failed_batch_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fullraw_index, "_remote_complete_exists", lambda *args, **kwargs: False)
    bad_path = tmp_path / "bad.jsonl.gz"
    bad_path.write_bytes(gzip.compress(b'{"display_name":"bad"}\n')[:-8])
    result = build_upload_shard_batches(
        [RawFile(source="openalex", format="openalex_jsonl", remote=f"file://{bad_path}")],
        shard_dir=tmp_path / "shards",
        upload_remote="remote:test",
        batch_files=1,
        shard_count=1,
        workers=1,
        delete_local=False,
    )
    assert result[0].uploaded is False
    assert result[0].files_failed == 1
    assert result[0].error
