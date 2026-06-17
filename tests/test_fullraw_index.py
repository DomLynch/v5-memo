from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

from pytest import MonkeyPatch

from v5_memo.fullraw_index import (
    FullRawFtsIndex,
    aggregate_shard_stats,
    build_shards,
    build_upload_shard_batches,
    discover_shard_paths,
    search_shards,
)
from v5_memo.fullraw_service import RawFile


def _write_jsonl_gzip(path: Path, rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wb") as fh:
        for row in rows:
            fh.write((json.dumps(row) + "\n").encode("utf-8"))


def test_fullraw_index_builds_ranked_queryable_index(tmp_path: Path) -> None:
    source = tmp_path / "openalex.jsonl.gz"
    _write_jsonl_gzip(
        source,
        [
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
                "abstract": (
                    "Earnings forecast error and firm size explain most post-announcement "
                    "drift variation."
                ),
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
        ],
    )
    index = FullRawFtsIndex(tmp_path / "fullraw.sqlite")
    try:
        result = index.index_files([
            RawFile(source="openalex", format="openalex_jsonl", remote=f"file://{source}")
        ])
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


def test_fullraw_index_uses_persisted_custom_term_map(tmp_path: Path) -> None:
    source = tmp_path / "openalex.jsonl.gz"
    _write_jsonl_gzip(
        source,
        [
            {
                "doi": "https://doi.org/10.example/guidance",
                "display_name": "Management guidance and earnings surprises",
                "abstract": "Managers issue guidance before earnings surprises.",
                "publication_year": 2024,
            }
        ],
    )
    index = FullRawFtsIndex(tmp_path / "fullraw.sqlite")
    try:
        index.index_files([
            RawFile(source="openalex", format="openalex_jsonl", remote=f"file://{source}")
        ])
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
    source = tmp_path / "openalex.jsonl.gz"
    _write_jsonl_gzip(
        source,
        [
            {
                "doi": "https://doi.org/10.example/one",
                "display_name": "Management forecast disclosure",
                "abstract": "Forecast disclosure and earnings forecast error.",
            }
        ],
    )
    index = FullRawFtsIndex(tmp_path / "fullraw.sqlite")
    files = [RawFile(source="openalex", format="openalex_jsonl", remote=f"file://{source}")]
    try:
        first = index.index_files(files)
        second = index.index_files(files)
        stats = index.stats(files_total=1)
    finally:
        index.close()

    assert first.papers_inserted == 1
    assert second.files_attempted == 0
    assert second.papers_inserted == 0
    assert stats.papers_indexed == 1


def test_fullraw_index_stops_when_disk_guard_is_hit(tmp_path: Path) -> None:
    source = tmp_path / "openalex.jsonl.gz"
    _write_jsonl_gzip(
        source,
        [
            {
                "doi": "https://doi.org/10.example/one",
                "display_name": "Management forecast disclosure",
                "abstract": "Forecast disclosure and earnings forecast error.",
            }
        ],
    )
    index = FullRawFtsIndex(tmp_path / "fullraw.sqlite")
    try:
        result = index.index_files(
            [RawFile(source="openalex", format="openalex_jsonl", remote=f"file://{source}")],
            min_free_bytes=10**18,
        )
        stats = index.stats(files_total=1)
    finally:
        index.close()

    assert result.stopped_for_budget is True
    assert result.files_attempted == 0
    assert result.papers_inserted == 0
    assert stats.papers_indexed == 0


def test_fullraw_index_proceeds_when_disk_guard_allows(tmp_path: Path) -> None:
    source = tmp_path / "openalex.jsonl.gz"
    _write_jsonl_gzip(
        source,
        [
            {
                "doi": "https://doi.org/10.example/one",
                "display_name": "Management forecast disclosure",
                "abstract": "Forecast disclosure and earnings forecast error.",
            }
        ],
    )
    index = FullRawFtsIndex(tmp_path / "fullraw.sqlite")
    try:
        result = index.index_files(
            [RawFile(source="openalex", format="openalex_jsonl", remote=f"file://{source}")],
            min_free_bytes=0,
        )
    finally:
        index.close()

    assert result.stopped_for_budget is False
    assert result.files_completed == 1
    assert result.papers_inserted == 1


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
        index.index_files(
            [RawFile(source="openalex", format="openalex_jsonl", remote="file:///unused.gz")],
            min_free_bytes=1,
        )
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
        source = tmp_path / f"openalex_{index}.jsonl.gz"
        _write_jsonl_gzip(source, [row])
        files.append(RawFile(source="openalex", format="openalex_jsonl", remote=f"file://{source}"))

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


def test_build_upload_shard_batches_uploads_and_deletes_local_batches(tmp_path: Path) -> None:
    files: list[RawFile] = []
    for index in range(4):
        source = tmp_path / f"openalex_batch_{index}.jsonl.gz"
        _write_jsonl_gzip(
            source,
            [
                {
                    "doi": f"https://doi.org/10.example/batch-{index}",
                    "display_name": f"Management forecast disclosure batch {index}",
                    "abstract": "Managers disclose forecasts and guidance.",
                    "publication_year": 2024,
                }
            ],
        )
        files.append(RawFile(source="openalex", format="openalex_jsonl", remote=f"file://{source}"))

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
