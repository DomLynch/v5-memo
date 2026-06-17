from __future__ import annotations

import gzip
import json
from pathlib import Path

from v5_memo.fullraw_index import FullRawFtsIndex
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
        hits = index.search("forecast disclosure", limit=5)
        stats = index.stats(files_total=1)
    finally:
        index.close()

    assert result.files_completed == 1
    assert result.papers_inserted == 3
    assert stats.papers_indexed == 3
    assert stats.files_indexed == 1
    assert hits[0]["doi"] == "10.2308/tar-9603274096"
    assert hits[0]["source"] == "openalex"


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
