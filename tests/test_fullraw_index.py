from __future__ import annotations

import gzip
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest import MonkeyPatch

from v5_memo import fullraw_index
from v5_memo.fullraw_index import (
    FullRawFtsIndex,
    ShardBatchResult,
    ShardCatalogEntry,
    aggregate_shard_manifest_stats,
    aggregate_shard_stats,
    build_shard_catalog,
    build_shards,
    build_upload_shard_batches,
    discover_shard_paths,
    search_shards,
    select_search_shard_entries,
    select_search_shard_paths,
    shard_coverage_receipt,
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
            {"shard_id": 0, "files_completed": 3, "papers_inserted": 42, "bytes_used": 1234},
            {"shard_id": 1, "files_completed": 4, "papers_inserted": 99, "bytes_used": 4567},
        ],
    }))

    catalog = build_shard_catalog(tmp_path, trust_filenames=True)
    stats = aggregate_shard_manifest_stats(tmp_path, files_total=10)

    assert [(entry.batch_id, entry.shard_id) for entry in catalog] == [(1, 0), (1, 1)]
    assert catalog[0].sources == ("openalex", "semantic_scholar")
    assert catalog[1].papers_inserted == 99
    assert stats.files_indexed == 7
    assert stats.papers_indexed == 141
    assert stats.bytes_used == 5801


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

    assert {entry.sources[0] for entry in first} == {"openalex", "pubmed", "semantic_scholar"}
    assert {entry.sources[0] for entry in second} == {"openalex", "pubmed", "semantic_scholar"}
    assert {entry.path for entry in first} != {entry.path for entry in second}


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
