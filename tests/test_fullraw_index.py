import gzip
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import (
    CancelledError,
)
from concurrent.futures import (
    TimeoutError as FuturesTimeoutError,
)
from dataclasses import replace
from pathlib import Path
from typing import BinaryIO, NamedTuple, cast

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


def test_fullraw_server_env_prefers_generic_researka_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARKA_FULLRAW_SWEEP_MAX_INFLIGHT", "2")
    monkeypatch.setenv("RESEARKA_FULLRAW_MIN_SHARDS_SEARCHED", "1525")
    monkeypatch.delenv("V5_MEMO_FULL_RAW_SWEEP_MAX_INFLIGHT", raising=False)
    monkeypatch.delenv("V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED", raising=False)

    assert fullraw_index._positive_int_env("V5_MEMO_FULL_RAW_SWEEP_MAX_INFLIGHT") == 2
    assert fullraw_index._positive_int_env("V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED") == 1525


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
        stats = index.stats(files_total=1)
    finally:
        index.close()
    assert (result.files_completed, result.papers_inserted, stats.papers_indexed) == (1, 2, 2)
    assert hits[0]["doi"] == "10.example/guidance"


def test_fullraw_index_round_trips_publication_integrity_metadata(tmp_path: Path) -> None:
    index = FullRawFtsIndex(tmp_path / "fullraw.sqlite")
    try:
        index.index_files([_raw(tmp_path, "integrity", [{
            "doi": "https://doi.org/10.example/retracted",
            "display_name": "Ordinary article title",
            "abstract": "Evidence text for retrieval.",
            "type": "journal-article",
            "publication_types": ["Retracted Publication"],
            "is_retracted": True,
            "is_withdrawn": True,
            "correction_status": "RetractionIn",
        }])])
        hits = index.search("evidence retrieval", limit=1)
    finally:
        index.close()

    assert hits[0]["document_type"] == "journal-article"
    assert hits[0]["publication_types"] == ("Retracted Publication",)
    assert hits[0]["is_retracted"] is True
    assert hits[0]["retraction_status_known"] is True
    assert hits[0]["is_withdrawn"] is True
    assert hits[0]["withdrawal_status_known"] is True
    assert hits[0]["correction_status"] == "RetractionIn"


def test_fullraw_duplicate_ingestion_conservatively_merges_unsafe_status(
    tmp_path: Path,
) -> None:
    safe = _raw(tmp_path, "safe", [{
        "doi": "10.example/duplicate",
        "display_name": "Duplicate evidence article",
        "abstract": "Duplicate evidence retrieval text.",
        "type": "article",
        "is_retracted": False,
    }])
    unsafe = _raw(tmp_path, "unsafe", [{
        "doi": "10.example/duplicate",
        "display_name": "Duplicate evidence article",
        "abstract": "Duplicate evidence retrieval text.",
        "type": "retraction-notice",
        "is_retracted": True,
        "is_withdrawn": True,
        "correction_status": "RetractionIn",
    }])
    index = FullRawFtsIndex(tmp_path / "fullraw.sqlite")
    try:
        index.index_files([safe, unsafe])
        hits = index.search("duplicate evidence retrieval", limit=1)
    finally:
        index.close()

    assert hits[0]["is_retracted"] is True
    assert hits[0]["retraction_status_known"] is True
    assert hits[0]["is_withdrawn"] is True
    publication_types = hits[0]["publication_types"]
    assert isinstance(publication_types, tuple)
    assert "retraction-notice" in publication_types
    assert hits[0]["correction_status"] == "RetractionIn"


def test_cross_shard_dedupe_preserves_lower_score_unsafe_status() -> None:
    safe = {
        "doi": "10.example/cross-shard",
        "title": "Preferred content",
        "abstract": "Evidence text.",
        "score": 10.0,
        "document_type": "article",
        "is_retracted": False,
        "retraction_status_known": True,
    }
    unsafe = {
        "doi": "10.example/cross-shard",
        "title": "Lower-ranked content",
        "abstract": "Evidence text.",
        "score": 1.0,
        "document_type": "retraction-notice",
        "is_retracted": True,
        "correction_status": "RetractionIn",
    }

    merged = fullraw_index._merge_hit_groups([[safe], [unsafe]], limit=1)[0]

    assert merged["title"] == "Preferred content"
    assert merged["score"] == 10.0
    assert merged["is_retracted"] is True
    publication_types = merged["publication_types"]
    assert isinstance(publication_types, tuple)
    assert "retraction-notice" in publication_types


def test_read_only_legacy_shard_defaults_missing_integrity_columns(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE papers (
          id INTEGER PRIMARY KEY, source_key TEXT NOT NULL UNIQUE, title TEXT NOT NULL,
          abstract TEXT NOT NULL, doi TEXT, pmid TEXT, pmcid TEXT, openalex_id TEXT,
          semantic_scholar_id TEXT, year INTEGER, journal TEXT, source TEXT NOT NULL,
          source_remote TEXT NOT NULL DEFAULT '', url TEXT, cited_by_count INTEGER, raw_score REAL
        );
        CREATE VIRTUAL TABLE paper_fts USING fts5(
          title, abstract, journal, content='papers', content_rowid='id'
        );
        INSERT INTO papers(id, source_key, title, abstract, source)
        VALUES (1, 'legacy:1', 'Legacy intervention paper', 'Evidence retrieval result.', 'legacy');
        INSERT INTO paper_fts(rowid, title, abstract, journal)
        VALUES (1, 'Legacy intervention paper', 'Evidence retrieval result.', '');
        """
    )
    conn.commit()
    conn.close()

    index = FullRawFtsIndex(path, read_only=True)
    try:
        hits = index.search("evidence retrieval", limit=1)
    finally:
        index.close()

    assert hits[0]["document_type"] == ""
    assert hits[0]["publication_types"] == ()
    assert hits[0]["is_retracted"] is None
    assert hits[0]["retraction_status_known"] is False
    assert hits[0]["is_withdrawn"] is None
    assert hits[0]["withdrawal_status_known"] is False


def test_writable_legacy_index_migrates_integrity_columns_in_place(tmp_path: Path) -> None:
    path = tmp_path / "legacy-writable.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE papers (
          id INTEGER PRIMARY KEY, source_key TEXT NOT NULL UNIQUE, title TEXT NOT NULL,
          abstract TEXT NOT NULL, doi TEXT, pmid TEXT, pmcid TEXT, openalex_id TEXT,
          semantic_scholar_id TEXT, year INTEGER, journal TEXT, source TEXT NOT NULL,
          source_remote TEXT NOT NULL DEFAULT '', url TEXT, cited_by_count INTEGER, raw_score REAL
        )
        """
    )
    conn.commit()
    conn.close()

    index = FullRawFtsIndex(path)
    try:
        index.initialize()
        columns = {
            str(row["name"])
            for row in index._conn.execute("PRAGMA table_info(papers)").fetchall()
        }
    finally:
        index.close()

    assert {
        "document_type",
        "publication_types_json",
        "is_retracted",
        "retraction_status_known",
        "is_withdrawn",
        "withdrawal_status_known",
        "correction_status",
    } <= columns


def test_fullraw_relevance_prefers_construct_fit_over_loose_token_overlap(tmp_path: Path) -> None:
    index = FullRawFtsIndex(tmp_path / "fullraw.sqlite")
    try:
        index.index_files([_raw(tmp_path, "semantic_scholar", [
            {
                "doi": "https://doi.org/10.example/water-resistance",
                "title": "Effects of aquatic exercise training using water-resistance equipment in elderly",
                "abstract": "Cold immersion protocols differ from water resistance training.",
                "cited_by_count": 200,
            },
            {
                "doi": "https://doi.org/10.example/cwi-recovery",
                "title": "Cold Water Immersion and Contrast Water Therapy Do Not Improve Short-Term Recovery Following Resistance Training",
                "abstract": "Cold water immersion was tested after resistance training.",
                "cited_by_count": 3,
            },
        ])], commit_interval=1)
        hits = index.search("cold water immersion resistance training", limit=2)
    finally:
        index.close()

    assert [hit["doi"] for hit in hits] == [
        "10.example/cwi-recovery",
        "10.example/water-resistance",
    ]


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
    timeouts: list[object] = []

    def counting_search(path: Path, *args: object, **kwargs: object) -> list[dict[str, object]]:
        del kwargs
        called.append(path)
        timeouts.append(args[-1])
        return []

    monkeypatch.setattr(fullraw_index, "_search_one_shard", counting_search)
    many_paths = [tmp_path / f"many_{idx}.sqlite" for idx in range(200)]
    fullraw_index._search_shard_paths_with_paths(
        many_paths, "metformin", limit=5, year_min=1900, year_max=2100, rank_mode="relevance", timeout_seconds=0.01, shard_timeout_seconds=30
    )
    assert len(called) <= fullraw_index._FULL_COVERAGE_PREFIX_SHARDS
    assert many_paths[-1] not in called
    assert timeouts == [30] * len(called)


def test_shard_search_cancels_and_drains_running_pool_after_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_calls: list[tuple[bool, bool]] = []
    cancel_events: list[threading.Event] = []

    class FakePool:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def submit(self, *args: object, **kwargs: object) -> object:
            del kwargs
            cancel_event = args[-1]
            assert isinstance(cancel_event, threading.Event)
            cancel_events.append(cancel_event)
            return object()

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            shutdown_calls.append((wait, cancel_futures))

    def fake_as_completed(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise FuturesTimeoutError()

    shard = tmp_path / "one.sqlite"
    shard.touch()
    monkeypatch.setattr(fullraw_index, "ThreadPoolExecutor", FakePool)
    monkeypatch.setattr(fullraw_index, "as_completed", fake_as_completed)

    _hits, _paths, timed_out = fullraw_index._search_shard_paths_with_paths(
        [shard],
        "metformin",
        limit=5,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        workers=1,
        timeout_seconds=0.01,
    )

    assert timed_out is True
    assert len(cancel_events) == 1
    assert cancel_events[0].is_set()
    assert shutdown_calls == [(True, True)]


def test_shard_search_unexpected_failure_cancels_siblings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sibling_started = threading.Event()
    sibling_cancelled = threading.Event()
    bad = tmp_path / "bad.sqlite"
    sibling = tmp_path / "sibling.sqlite"

    def fake_worker(path: Path, *args: object, **_kwargs: object) -> list[dict[str, object]]:
        cancel_event = args[-1]
        assert isinstance(cancel_event, threading.Event)
        if path == bad:
            assert sibling_started.wait(1)
            raise ValueError("unexpected worker failure")
        sibling_started.set()
        assert cancel_event.wait(1)
        sibling_cancelled.set()
        raise CancelledError()

    monkeypatch.setattr(fullraw_index, "_materialize_and_search_one_shard", fake_worker)

    with pytest.raises(ValueError, match="unexpected worker failure"):
        fullraw_index._search_shard_paths_with_paths_and_receipt(
            [bad, sibling],
            "metformin",
            limit=5,
            year_min=1900,
            year_max=2100,
            rank_mode="relevance",
            workers=2,
            timeout_seconds=5,
        )

    assert sibling_cancelled.is_set()


def test_shard_search_propagates_progress_heartbeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = tmp_path / "one.sqlite"
    heartbeats: list[None] = []

    def fake_materialized(
        path: Path,
        *,
        preserve: set[Path] | None = None,
        populate: bool = False,
        cancel_event: threading.Event | None = None,
        progress_callback: object = None,
    ) -> Path:
        del preserve, populate, cancel_event
        assert callable(progress_callback)
        progress_callback()
        return path

    monkeypatch.setattr(fullraw_index, "_materialized_shard_path", fake_materialized)
    monkeypatch.setattr(fullraw_index, "_search_one_shard_for_pool", lambda *_args: [])

    _hits, completed, timed_out, _metrics = fullraw_index._search_shard_paths_with_paths_and_receipt(
        [shard],
        "metformin",
        limit=5,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        workers=1,
        timeout_seconds=5,
        progress_callback=lambda: heartbeats.append(None),
    )

    assert completed == [shard]
    assert timed_out is False
    assert len(heartbeats) >= 4


def test_isolated_shard_search_requests_restart_when_child_cannot_be_reaped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        returncode = None
        killed = False
        waits: list[float | None]

        def __init__(self) -> None:
            self.waits = []

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float | None = None) -> None:
            self.waits.append(timeout)
            if timeout is not None:
                raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)

    fake = FakeProcess()
    restart_reasons: list[str] = []
    monkeypatch.setattr("v5_memo.fullraw_index.subprocess.Popen", lambda *args, **kwargs: fake)
    monkeypatch.setattr(fullraw_index, "_request_process_restart", restart_reasons.append)

    with pytest.raises(OSError, match="could not be reaped"):
        fullraw_index._search_one_shard_isolated(
            tmp_path / "stuck.sqlite",
            "metformin",
            5,
            1900,
            2100,
            "relevance",
            0.01,
        )

    assert fake.killed is True
    assert fake.waits == [1.0]
    assert restart_reasons == ["isolated search child remained alive after SIGKILL"]


def test_process_restart_request_targets_current_process(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: signals.append((pid, sig)))

    fullraw_index._request_process_restart("stalled test worker")

    assert signals == [(os.getpid(), signal.SIGTERM)]
    assert "stalled test worker" in capsys.readouterr().err


def test_write_json_ignores_disconnected_client() -> None:
    class BrokenWriter:
        def write(self, data: bytes) -> None:
            del data
            raise BrokenPipeError

    class FakeHandler:
        wfile = BrokenWriter()

        def send_response(self, status: int) -> None:
            assert status == 200

        def send_header(self, key: str, value: str) -> None:
            assert key
            assert value

        def end_headers(self) -> None:
            return None

    fullraw_index._write_json(FakeHandler(), 200, {"ok": True})  # type: ignore[arg-type]


def test_foreground_receipt_counts_only_completed_shards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [_entry(tmp_path, idx, "openalex") for idx in range(4)]

    def partial_search(path: Path, *args: object, **kwargs: object) -> list[dict[str, object]]:
        del args, kwargs
        if path in {entries[0].path, entries[1].path}:
            return [{"title": "Metformin diabetes", "source": "openalex"}]
        raise OSError("timed out")

    monkeypatch.setattr(fullraw_index, "_search_one_shard", partial_search)
    hits, receipt = fullraw_index.search_shard_entries_with_receipt(
        entries,
        "metformin",
        limit=5,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
    )

    assert hits
    assert receipt["foreground_selected_shards"] == 4
    assert receipt["foreground_completed_shards"] == 2
    assert receipt["shards_searched"] == 2
    assert receipt["partial_shard_search"] is True

def test_sweep_cache_entry_ready_rejects_insufficient_partial_hits() -> None:
    receipt = {
        "shards_searched": 32,
        "shards_total": 1525,
        "partial_shard_search": True,
        "sources_searched": {
            "biorxiv": 7,
            "openalex": 7,
            "pubmed": 7,
            "semantic_scholar": 13,
            "semantic_scholar_abstracts": 5,
        },
        "sweep_remaining_shards": 1493,
    }
    entry = fullraw_index.SweepCacheEntry(
        created_at=time.time(),
        hits=[{"title": "Rapamycin formulation for oral administration"}],
        receipt=receipt,
    )

    assert not fullraw_index.sweep_cache_entry_is_ready(entry, min_shards_searched=512, min_sources_searched=5)
    assert not fullraw_index.sweep_cache_entry_is_ready(entry, min_sources_searched=5, require_complete_search=True)
    assert not fullraw_index.sweep_cache_entry_is_ready(entry, min_sources_searched=5, require_complete_sweep=True)
    receipt.update({
        "sweep_search_passes": ({"role": "focused"}, {"role": "broad"}),
        "sweep_completed_pass_roles": ("focused",),
    })
    assert fullraw_index.sweep_cache_entry_is_ready(entry, min_sources_searched=5)
    assert not fullraw_index.sweep_cache_entry_is_ready(entry, min_sources_searched=5, require_complete_sweep=True)

    receipt.update({
        "shards_searched": 1525,
        "partial_shard_search": False,
        "sweep_remaining_shards": 0,
        "sweep_completed_pass_roles": ("focused", "broad"),
    })
    assert fullraw_index.sweep_cache_entry_is_ready(entry, min_shards_searched=512, min_sources_searched=5, require_complete_search=True, require_complete_sweep=True)


def test_sweep_cache_entry_ready_rejects_stale_strategy() -> None:
    entry = fullraw_index.SweepCacheEntry(
        created_at=time.time(),
        hits=[{"title": "Cold water immersion after resistance training"}],
        receipt={
            "shards_searched": 1525,
            "shards_total": 1525,
            "partial_shard_search": False,
            "sweep_remaining_shards": 0,
            "sweep_failed_shards": 0,
            "sources_searched": {
                "biorxiv": 1,
                "openalex": 1,
                "pubmed": 1,
                "semantic_scholar": 1,
                "semantic_scholar_abstracts": 1,
            },
            "sweep_strategy": "old_profile",
        },
    )

    assert fullraw_index.sweep_cache_entry_is_ready(
        entry,
        min_shards_searched=1525,
        min_sources_searched=5,
        require_complete_search=True,
        require_complete_sweep=True,
    )
    assert not fullraw_index.sweep_cache_entry_is_ready(
        entry,
        min_shards_searched=1525,
        min_sources_searched=5,
        require_complete_search=True,
        require_complete_sweep=True,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
    )
    assert not fullraw_index.sweep_cache_entry_can_answer_request(
        entry,
        cache_only=True,
        min_shards_searched=1525,
        min_sources_searched=5,
        require_complete_search=True,
        require_complete_sweep=True,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
    )


def test_sweep_cache_only_can_answer_agent_poll() -> None:
    entry = fullraw_index.SweepCacheEntry(
        created_at=time.time(),
        hits=[{"title": "Metformin for Longevity and Sarcopenia"}],
        receipt={
            "shards_searched": 169,
            "shards_total": 1525,
            "partial_shard_search": True,
            "sources_searched": {
                "biorxiv": 13,
                "openalex": 64,
                "pubmed": 44,
                "semantic_scholar": 56,
                "semantic_scholar_abstracts": 5,
            },
        },
    )

    assert not fullraw_index.sweep_cache_entry_can_answer_request(
        entry,
        min_shards_searched=150,
        min_sources_searched=5,
    )
    assert fullraw_index.sweep_cache_entry_can_answer_request(
        entry,
        cache_only=True,
        min_shards_searched=150,
        min_sources_searched=5,
    )
    assert not fullraw_index.sweep_cache_entry_can_answer_request(
        entry,
        resume_cached=True,
        min_shards_searched=150,
        min_sources_searched=5,
    )
    assert not fullraw_index.sweep_cache_entry_can_answer_request(
        entry,
        min_shards_searched=512,
        min_sources_searched=5,
    )


def test_complete_shard_service_forces_cache_queue() -> None:
    assert fullraw_index._should_force_cache_queue(
        shard_dir_configured=True,
        require_complete_search=True,
        sweep_enabled=True,
    )
    assert not fullraw_index._should_force_cache_queue(
        shard_dir_configured=True,
        require_complete_search=False,
        sweep_enabled=True,
    )


def test_sweep_cache_key_ignores_result_limits() -> None:
    first = fullraw_index._sweep_cache_key(
        "cold water immersion",
        limit=3,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        sweep_shard_limit=1525,
    )
    second = fullraw_index._sweep_cache_key(
        "cold water immersion",
        limit=10,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        sweep_shard_limit=1525,
    )
    larger = fullraw_index._sweep_cache_key(
        "cold water immersion",
        limit=25,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        sweep_shard_limit=1525,
    )

    assert first == second
    assert second == larger


def test_sweep_cache_key_ignores_term_order() -> None:
    first = fullraw_index._sweep_cache_key(
        "creatine trial resistance",
        limit=25,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        sweep_shard_limit=1525,
    )
    second = fullraw_index._sweep_cache_key(
        "creatine resistance trial",
        limit=25,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        sweep_shard_limit=1525,
    )

    assert first == second


def test_sweep_cache_key_ignores_runtime_knobs() -> None:
    shorter_runtime = fullraw_index._sweep_cache_key(
        "metformin resistance training",
        limit=10,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        sweep_shard_limit=1525,
        sweep_pass_shard_limit=8,
        sweep_max_passes=1525,
        sweep_timeout_seconds=3600.0,
        sweep_shard_timeout_seconds=90.0,
    )
    longer_runtime = fullraw_index._sweep_cache_key(
        "metformin resistance training",
        limit=10,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        sweep_shard_limit=1525,
        sweep_pass_shard_limit=32,
        sweep_max_passes=1525,
        sweep_timeout_seconds=120.0,
        sweep_shard_timeout_seconds=20.0,
    )

    assert shorter_runtime == longer_runtime


def test_sweep_cache_key_changes_when_research_contract_changes() -> None:
    current = fullraw_index._sweep_cache_key(
        "metformin resistance training",
        limit=10,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        sweep_shard_limit=1525,
    )
    different_rank = fullraw_index._sweep_cache_key(
        "metformin resistance training",
        limit=10,
        year_min=1900,
        year_max=2100,
        rank_mode="recency",
        sweep_shard_limit=1525,
    )
    different_coverage = fullraw_index._sweep_cache_key(
        "metformin resistance training",
        limit=10,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        sweep_shard_limit=512,
    )
    different_catalog = fullraw_index._sweep_cache_key(
        "metformin resistance training",
        limit=10,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        sweep_shard_limit=1525,
        sweep_catalog_scope="/var/lib/v5-memo/v5-isolated-fullraw-fts-remote",
    )

    assert current != different_rank
    assert current != different_coverage
    assert current != different_catalog


def test_full_sweep_cache_query_uses_canonical_pass(tmp_path: Path) -> None:
    entries = [
        replace(_entry(tmp_path, idx, "openalex"), topic_terms=("cold", "immersion", "training"))
        for idx in range(4)
    ]

    verbose = fullraw_index._sweep_cache_query(
        "cold water immersion training adaptation",
        entries,
        sweep_shard_limit=4,
        rank_mode="relevance",
    )
    compact = fullraw_index._sweep_cache_query(
        "cold immersion training",
        entries,
        sweep_shard_limit=4,
        rank_mode="relevance",
    )

    assert verbose == compact


def test_full_sweep_cache_query_preserves_outcome_for_compound_intervention(tmp_path: Path) -> None:
    entries = [
        replace(_entry(tmp_path, 0, "openalex"), topic_terms=("creatine", "trial", "resistance")),
        replace(_entry(tmp_path, 1, "openalex"), topic_terms=("adults", "older", "muscle")),
        replace(_entry(tmp_path, 2, "openalex"), topic_terms=("strength", "training", "resistance")),
        replace(_entry(tmp_path, 3, "openalex"), topic_terms=("creatine", "resistance", "trial")),
    ]

    verbose = fullraw_index._sweep_cache_query(
        "creatine resistance training older adults muscle strength trial",
        entries,
        sweep_shard_limit=4,
        rank_mode="relevance",
    )
    compact = fullraw_index._sweep_cache_query(
        "creatine resistance training trial",
        entries,
        sweep_shard_limit=4,
        rank_mode="relevance",
    )

    assert verbose == "creatine resistance strength trial"
    assert compact == "creatine resistance trial"


def test_full_sweep_cache_query_preserves_only_outcome_axis(tmp_path: Path) -> None:
    entries = [
        replace(
            _entry(tmp_path, idx, "openalex"),
            topic_terms=("omega", "muscle", "strength", "trial"),
        )
        for idx in range(4)
    ]

    query = fullraw_index._sweep_cache_query(
        "omega muscle strength trial",
        entries,
        sweep_shard_limit=4,
        rank_mode="relevance",
    )
    population_only = fullraw_index._sweep_cache_query(
        "omega older adults trial",
        entries,
        sweep_shard_limit=4,
        rank_mode="relevance",
    )
    compound = fullraw_index._sweep_cache_query(
        "quercetin dasatinib senolytic muscle strength trial",
        entries,
        sweep_shard_limit=4,
        rank_mode="relevance",
    )
    compact_compound = fullraw_index._sweep_cache_query(
        "quercetin senolytic muscle strength trial",
        entries,
        sweep_shard_limit=4,
        rank_mode="relevance",
    )

    assert query == "omega strength trial"
    assert population_only == "omega trial"
    assert compound == "quercetin senolytic strength trial"
    assert compact_compound == "quercetin senolytic strength trial"


def test_sweep_cache_matcher_accepts_compatible_pass_query() -> None:
    entry = fullraw_index.SweepCacheEntry(
        created_at=time.time(),
        hits=[{"title": "Cold water immersion after training"}],
        receipt={
            "sweep_shard_limit": 1525,
            "sweep_pass_shard_limit": 32,
            "sweep_strategy": fullraw_index._SWEEP_STRATEGY,
            "sweep_original_query": "cold water immersion training adaptation",
            "sweep_search_passes": (
                {"role": "focused", "query": "cold immersion training"},
            ),
        },
    )

    assert fullraw_index._sweep_cache_entry_matches_request(
        entry,
        query="cold immersion training",
        result_limit=1,
        sweep_shard_limit=1525,
        sweep_pass_shard_limit=32,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
    )
    assert not fullraw_index._sweep_cache_entry_matches_request(
        entry,
        query="resveratrol exercise adaptation",
        result_limit=1,
        sweep_shard_limit=1525,
        sweep_pass_shard_limit=32,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
    )
    assert fullraw_index._sweep_cache_entry_matches_request(
        entry,
        query="cold immersion training",
        result_limit=1,
        sweep_shard_limit=1525,
        sweep_pass_shard_limit=4,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
    )


def test_sweep_cache_matcher_accepts_partial_with_reordered_terms() -> None:
    entry = fullraw_index.SweepCacheEntry(
        created_at=time.time(),
        hits=[{"title": "Creatine resistance training trial"}],
        receipt={
            "sweep_result_limit": 25,
            "sweep_shard_limit": 1525,
            "sweep_strategy": fullraw_index._SWEEP_STRATEGY,
            "sweep_query": "creatine trial resistance",
            "partial_shard_search": True,
            "sweep_remaining_shards": 437,
        },
    )

    assert fullraw_index._sweep_cache_entry_matches_request(
        entry,
        query="creatine resistance trial",
        result_limit=25,
        sweep_shard_limit=1525,
        sweep_pass_shard_limit=32,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
    )


def test_sweep_cache_matcher_accepts_only_completed_alias_equivalent_query() -> None:
    entry = fullraw_index.SweepCacheEntry(
        created_at=time.time(),
        hits=[{"title": "Resveratrol and training adaptation"}],
        receipt={
            "sweep_result_limit": 10,
            "sweep_shard_limit": 1525,
            "sweep_strategy": fullraw_index._SWEEP_STRATEGY,
            "sweep_query": "resveratrol training adaptation",
            "partial_shard_search": False,
            "sweep_remaining_shards": 0,
        },
    )
    partial_entry = fullraw_index.SweepCacheEntry(
        created_at=entry.created_at,
        hits=entry.hits,
        receipt={**entry.receipt, "partial_shard_search": True, "sweep_remaining_shards": 12},
    )

    assert fullraw_index._sweep_cache_entry_matches_request(
        entry,
        query="resveratrol exercise adaptation",
        result_limit=10,
        sweep_shard_limit=1525,
        sweep_pass_shard_limit=32,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
    )
    assert not fullraw_index._sweep_cache_entry_matches_request(
        entry,
        query="resveratrol cancer adaptation",
        result_limit=10,
        sweep_shard_limit=1525,
        sweep_pass_shard_limit=32,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
    )
    assert not fullraw_index._sweep_cache_entry_matches_request(
        partial_entry,
        query="resveratrol exercise adaptation",
        result_limit=10,
        sweep_shard_limit=1525,
        sweep_pass_shard_limit=32,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
    )


def test_completed_sweep_cache_with_unsaturated_limit_answers_higher_limit() -> None:
    entry = fullraw_index.SweepCacheEntry(
        created_at=time.time(),
        hits=[{"title": "Urolithin runner recovery"} for _ in range(2)],
        receipt={
            "sweep_result_limit": 10,
            "result_count_raw": 2,
            "sweep_shard_limit": 1525,
            "sweep_strategy": fullraw_index._SWEEP_STRATEGY,
            "sweep_query": "urolithin trained runners",
            "partial_shard_search": False,
            "sweep_remaining_shards": 0,
        },
    )

    assert fullraw_index._sweep_cache_entry_matches_request(
        entry,
        query="urolithin trained runners",
        result_limit=50,
        sweep_shard_limit=1525,
        sweep_pass_shard_limit=32,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
    )


def test_completed_sweep_cache_with_saturated_limit_rejects_higher_limit() -> None:
    entry = fullraw_index.SweepCacheEntry(
        created_at=time.time(),
        hits=[{"title": f"Creatine trial {index}"} for index in range(10)],
        receipt={
            "sweep_result_limit": 10,
            "result_count_raw": 10,
            "sweep_shard_limit": 1525,
            "sweep_strategy": fullraw_index._SWEEP_STRATEGY,
            "sweep_query": "creatine resistance training",
            "partial_shard_search": False,
            "sweep_remaining_shards": 0,
        },
    )

    assert not fullraw_index._sweep_cache_entry_matches_request(
        entry,
        query="creatine resistance training",
        result_limit=50,
        sweep_shard_limit=1525,
        sweep_pass_shard_limit=32,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
    )


def test_completed_sweep_cache_matches_original_query_when_active_query_changed() -> None:
    entry = fullraw_index.SweepCacheEntry(
        created_at=time.time(),
        hits=[{"title": "Urolithin runner recovery"} for _ in range(2)],
        receipt={
            "sweep_result_limit": 10,
            "result_count_raw": 2,
            "sweep_shard_limit": 1525,
            "sweep_strategy": fullraw_index._SWEEP_STRATEGY,
            "sweep_query": "urolithin trained runners",
            "sweep_original_query": "urolithin muscle trained runners placebo trial",
            "partial_shard_search": False,
            "sweep_remaining_shards": 0,
        },
    )

    assert fullraw_index._sweep_cache_entry_matches_active_or_completed_original_query(
        entry,
        active_query="urolithin runners trial",
        original_query="urolithin muscle trained runners placebo trial",
        result_limit=50,
        sweep_shard_limit=1525,
        sweep_pass_shard_limit=32,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
    )


def test_completed_sweep_cache_rejects_old_query_normalizer_strategy() -> None:
    entry = fullraw_index.SweepCacheEntry(
        created_at=time.time(),
        hits=[{"title": "Irrelevant omega trial"} for _ in range(2)],
        receipt={
            "sweep_result_limit": 25,
            "result_count_raw": 2,
            "sweep_shard_limit": 1525,
            "sweep_strategy": "profile_relaxed_v12",
            "sweep_query": "omega trial",
            "sweep_original_query": "omega muscle strength trial",
            "partial_shard_search": False,
            "sweep_remaining_shards": 0,
        },
    )

    assert not fullraw_index._sweep_cache_entry_matches_active_or_completed_original_query(
        entry,
        active_query="omega strength trial",
        original_query="omega muscle strength trial",
        result_limit=25,
        sweep_shard_limit=1525,
        sweep_pass_shard_limit=32,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
    )


def test_partial_sweep_cache_does_not_match_changed_original_query() -> None:
    entry = fullraw_index.SweepCacheEntry(
        created_at=time.time(),
        hits=[{"title": "Urolithin runner recovery"}],
        receipt={
            "sweep_result_limit": 10,
            "result_count_raw": 1,
            "sweep_shard_limit": 1525,
            "sweep_strategy": fullraw_index._SWEEP_STRATEGY,
            "sweep_query": "urolithin trained runners",
            "sweep_original_query": "urolithin muscle trained runners placebo trial",
            "partial_shard_search": True,
            "sweep_remaining_shards": 120,
        },
    )

    assert not fullraw_index._sweep_cache_entry_matches_active_or_completed_original_query(
        entry,
        active_query="urolithin runners trial",
        original_query="urolithin muscle trained runners placebo trial",
        result_limit=10,
        sweep_shard_limit=1525,
        sweep_pass_shard_limit=32,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
    )


def test_sweep_cache_matcher_rejects_stale_catalog_scope() -> None:
    entry = fullraw_index.SweepCacheEntry(
        created_at=time.time(),
        hits=[{"title": "Resveratrol exercise training"}],
        receipt={
            "sweep_result_limit": 10,
            "sweep_shard_limit": 1525,
            "sweep_pass_shard_limit": 32,
            "sweep_strategy": fullraw_index._SWEEP_STRATEGY,
            "sweep_catalog_scope": "/var/lib/v5-memo/fullraw-fts-remote",
            "sweep_query": "resveratrol exercise training",
        },
    )

    assert not fullraw_index._sweep_cache_entry_matches_request(
        entry,
        query="resveratrol exercise training",
        result_limit=10,
        sweep_shard_limit=1525,
        sweep_pass_shard_limit=32,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
        sweep_catalog_scope="/var/lib/v5-memo/v5-isolated-fullraw-fts-remote",
    )


def test_sweep_cache_matcher_rejects_low_limit_legacy_cache() -> None:
    legacy_entry = fullraw_index.SweepCacheEntry(
        created_at=time.time(),
        hits=[{"title": f"Cold immersion training {index}"} for index in range(3)],
        receipt={
            "sweep_shard_limit": 1525,
            "sweep_strategy": fullraw_index._SWEEP_STRATEGY,
            "sweep_query": "cold immersion training",
        },
    )
    marked_entry = fullraw_index.SweepCacheEntry(
        created_at=time.time(),
        hits=[{"title": f"Cold immersion training {index}"} for index in range(3)],
        receipt={
            "sweep_result_limit": 10,
            "sweep_shard_limit": 1525,
            "sweep_pass_shard_limit": 32,
            "sweep_strategy": fullraw_index._SWEEP_STRATEGY,
            "sweep_query": "cold immersion training",
        },
    )

    assert not fullraw_index._sweep_cache_entry_matches_request(
        legacy_entry,
        query="cold immersion training",
        result_limit=10,
        sweep_shard_limit=1525,
        sweep_pass_shard_limit=32,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
    )
    assert fullraw_index._sweep_cache_entry_matches_request(
        marked_entry,
        query="cold immersion training",
        result_limit=10,
        sweep_shard_limit=1525,
        sweep_pass_shard_limit=32,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
    )


def test_completed_disk_sweep_cache_beats_stale_memory_partial() -> None:
    memory_entry = fullraw_index.SweepCacheEntry(
        created_at=time.time(),
        hits=[],
        receipt={
            "shards_searched": 889,
            "shards_total": 1525,
            "partial_shard_search": True,
            "sweep_remaining_shards": 636,
        },
    )
    disk_entry = fullraw_index.SweepCacheEntry(
        created_at=time.time() - 3600,
        hits=[{"title": "Metformin longevity"}],
        receipt={
            "shards_searched": 1525,
            "shards_total": 1525,
            "partial_shard_search": False,
            "sweep_remaining_shards": 0,
            "sweep_failed_shards": 0,
            "source_count_searched": 5,
        },
    )

    selected = fullraw_index._prefer_sweep_cache_entry(memory_entry, disk_entry)

    assert selected is disk_entry
    assert selected.receipt["partial_shard_search"] is False
    assert selected.receipt["shards_searched"] == 1525


def test_cache_only_completed_alias_sweep_hit_does_not_aggregate_remote_stats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_path = tmp_path / "catalog.json"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    shard_dir = tmp_path / "remote-shards"
    shard_dir.mkdir()
    entries = [_entry(shard_dir, idx, "openalex" if idx else "pubmed") for idx in range(2)]
    fullraw_index.write_shard_catalog_cache(catalog_path, entries)
    sweep_catalog_scope = str(shard_dir.absolute())
    cached_query = "metformin training longevity"
    key = fullraw_index._sweep_cache_key(
        cached_query,
        limit=10,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        sweep_shard_limit=2,
        sweep_pass_shard_limit=2,
        sweep_max_passes=1,
        sweep_timeout_seconds=300.0,
        sweep_shard_timeout_seconds=10.0,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
        sweep_catalog_scope=sweep_catalog_scope,
    )
    cache_path = fullraw_index._sweep_cache_path(cache_dir, key)
    assert cache_path is not None
    fullraw_index._write_sweep_cache(
        cache_path,
        fullraw_index.SweepCacheEntry(
            created_at=time.time(),
            hits=[{"title": "Metformin for Longevity and Sarcopenia"}],
            receipt={
                "shards_searched": 2,
                "shards_total": 2,
                "partial_shard_search": False,
                "sweep_remaining_shards": 0,
                "sweep_failed_shards": 0,
                "sweep_result_limit": 10,
                "sweep_shard_limit": 2,
                "sweep_query": cached_query,
                "sources_searched": {"openalex": 1, "pubmed": 1},
                "sweep_search_passes": (
                    {"role": "focused"},
                    {"role": "citation_heavy"},
                    {"role": "recency"},
                ),
                "sweep_completed_pass_roles": ("focused", "citation_heavy", "recency"),
                "sweep_strategy": fullraw_index._SWEEP_STRATEGY,
                "sweep_catalog_scope": sweep_catalog_scope,
            },
        ),
    )

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    def fail_stats(*_args: object, **_kwargs: object) -> fullraw_index.IndexStats:
        raise AssertionError("cache-only hit should not aggregate remote shard stats")

    monkeypatch.setenv("V5_MEMO_FULL_RAW_INDEX_HOST", "127.0.0.1")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_INDEX_PORT", str(port))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_INDEX_TOKEN", "test-token")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_DIR", str(shard_dir))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_CATALOG_PATH", str(catalog_path))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SWEEP_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_ASYNC_SWEEP", "1")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_REQUIRE_COMPLETE_SEARCH", "1")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SWEEP_REQUIRE_COMPLETE", "1")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SWEEP_SHARD_LIMIT", "2")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED", "2")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED", "2")
    monkeypatch.setattr(fullraw_index, "load_or_build_manifest", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(fullraw_index, "aggregate_shard_stats", fail_stats)
    thread = threading.Thread(target=fullraw_index.run_server, daemon=True)
    thread.start()

    payload = json.dumps(
        {
            "query": "metformin exercise longevity",
            "limit": 10,
            "rank_mode": "relevance",
            "cache_only": True,
            "queue_if_missing": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/search",
        data=payload,
        headers={"Authorization": "Bearer test-token", "Content-Type": "application/json"},
        method="POST",
    )
    for _ in range(50):
        try:
            with urllib.request.urlopen(req, timeout=2) as resp:
                body = json.loads(resp.read().decode())
            break
        except OSError:
            time.sleep(0.05)
    else:  # pragma: no cover - defensive test guard
        raise AssertionError("server did not start")

    assert body["meta"]["async_sweep"]["status"] == "hit"
    assert body["meta"]["shard_receipt"]["shards_searched"] == 2
    assert body["results"][0]["title"] == "Metformin for Longevity and Sarcopenia"


def test_cache_only_can_expose_partial_hits_for_discovery_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_path = tmp_path / "catalog.json"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    shard_dir = tmp_path / "remote-shards"
    shard_dir.mkdir()
    entries = [_entry(shard_dir, idx, "openalex" if idx else "pubmed") for idx in range(2)]
    fullraw_index.write_shard_catalog_cache(catalog_path, entries)
    sweep_catalog_scope = str(shard_dir.absolute())
    query = "caffeine exercise null performance"
    cached_query = fullraw_index._sweep_cache_query(
        query,
        entries,
        sweep_shard_limit=2,
        rank_mode="relevance",
    )
    key = fullraw_index._sweep_cache_key(
        cached_query,
        limit=10,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        sweep_shard_limit=2,
        sweep_pass_shard_limit=2,
        sweep_max_passes=1,
        sweep_timeout_seconds=300.0,
        sweep_shard_timeout_seconds=10.0,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
        sweep_catalog_scope=sweep_catalog_scope,
    )
    cache_path = fullraw_index._sweep_cache_path(cache_dir, key)
    assert cache_path is not None
    fullraw_index._write_sweep_cache(
        cache_path,
        fullraw_index.SweepCacheEntry(
            created_at=time.time(),
            hits=[{"title": "Caffeine exercise null performance trial"}],
            receipt={
                "shards_searched": 1,
                "shards_total": 2,
                "partial_shard_search": True,
                "sweep_remaining_shards": 1,
                "sweep_failed_shards": 0,
                "sweep_result_limit": 10,
                "sweep_shard_limit": 2,
                "sweep_query": cached_query,
                "sweep_original_query": query,
                "sources_searched": {"openalex": 1},
                "sweep_strategy": fullraw_index._SWEEP_STRATEGY,
                "sweep_catalog_scope": sweep_catalog_scope,
            },
        ),
    )

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    monkeypatch.setenv("V5_MEMO_FULL_RAW_INDEX_HOST", "127.0.0.1")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_INDEX_PORT", str(port))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_INDEX_TOKEN", "test-token")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_DIR", str(shard_dir))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_CATALOG_PATH", str(catalog_path))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SWEEP_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_ASYNC_SWEEP", "1")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_REQUIRE_COMPLETE_SEARCH", "1")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SWEEP_REQUIRE_COMPLETE", "1")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SWEEP_SHARD_LIMIT", "2")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED", "2")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED", "2")
    monkeypatch.setattr(fullraw_index, "load_or_build_manifest", lambda *_args, **_kwargs: [])
    thread = threading.Thread(target=fullraw_index.run_server, daemon=True)
    thread.start()

    payload = json.dumps(
        {
            "query": query,
            "limit": 10,
            "rank_mode": "relevance",
            "cache_only": True,
            "queue_if_missing": True,
            "allow_partial_results": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/search",
        data=payload,
        headers={"Authorization": "Bearer test-token", "Content-Type": "application/json"},
        method="POST",
    )
    for _ in range(50):
        try:
            with urllib.request.urlopen(req, timeout=2) as resp:
                body = json.loads(resp.read().decode())
            break
        except OSError:
            time.sleep(0.05)
    else:  # pragma: no cover - defensive test guard
        raise AssertionError("server did not start")

    assert body["meta"]["partial_results"] is True
    assert body["meta"]["shard_receipt"]["partial_shard_search"] is True
    assert body["results"][0]["title"] == "Caffeine exercise null performance trial"


def test_strict_cache_only_probe_respects_no_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_path = tmp_path / "catalog.json"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    shard_dir = tmp_path / "remote-shards"
    shard_dir.mkdir()
    entries = [_entry(shard_dir, idx, "openalex" if idx else "pubmed") for idx in range(2)]
    fullraw_index.write_shard_catalog_cache(catalog_path, entries)

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    monkeypatch.setenv("V5_MEMO_FULL_RAW_INDEX_HOST", "127.0.0.1")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_INDEX_PORT", str(port))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_INDEX_TOKEN", "test-token")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_DIR", str(shard_dir))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_CATALOG_PATH", str(catalog_path))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SWEEP_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_ASYNC_SWEEP", "1")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_REQUIRE_COMPLETE_SEARCH", "1")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SWEEP_REQUIRE_COMPLETE", "1")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SWEEP_SHARD_LIMIT", "2")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED", "2")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED", "2")
    monkeypatch.setattr(fullraw_index, "load_or_build_manifest", lambda *_args, **_kwargs: [])
    thread = threading.Thread(target=fullraw_index.run_server, daemon=True)
    thread.start()

    payload = json.dumps(
        {
            "query": "metformin longevity",
            "limit": 10,
            "rank_mode": "relevance",
            "cache_only": True,
            "queue_if_missing": False,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/search",
        data=payload,
        headers={"Authorization": "Bearer test-token", "Content-Type": "application/json"},
        method="POST",
    )
    body: dict[str, object] | None = None
    for _ in range(50):
        try:
            with urllib.request.urlopen(req, timeout=2) as resp:
                body = json.loads(resp.read().decode())
            break
        except urllib.error.HTTPError as exc:
            assert exc.code == 422
            body = json.loads(exc.read().decode())
            break
        except OSError:
            time.sleep(0.05)
    else:  # pragma: no cover - defensive test guard
        raise AssertionError("server did not start")

    assert body is not None
    assert body["error"] == "coverage_too_narrow"


def test_sweep_cache_entry_marks_no_hit_stop_without_becoming_ready() -> None:
    entry = fullraw_index.SweepCacheEntry(
        created_at=time.time(),
        hits=[],
        receipt={
            "shards_searched": 128,
            "shards_total": 1525,
            "partial_shard_search": True,
            "sweep_remaining_shards": 1397,
            "sweep_stopped_no_hits": True,
            "sweep_no_hit_stop_shards": 128,
            "sweep_result_limit": 25,
            "sweep_strategy": fullraw_index._SWEEP_STRATEGY,
        },
    )

    assert fullraw_index.sweep_cache_entry_stopped_no_hits(entry)
    assert fullraw_index._sweep_cache_entry_should_stop_no_hits(entry, 128)
    assert not fullraw_index.sweep_cache_entry_is_ready(
        entry,
        min_shards_searched=1525,
        min_sources_searched=5,
        require_complete_search=True,
        require_complete_sweep=True,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
    )


def test_partial_zero_hit_sweep_cache_honors_no_hit_stop_after_restart() -> None:
    entry = fullraw_index.SweepCacheEntry(
        created_at=time.time(),
        hits=[],
        receipt={
            "shards_searched": 151,
            "shards_total": 1525,
            "partial_shard_search": True,
            "sweep_remaining_shards": 1374,
            "sweep_result_limit": 25,
            "sweep_strategy": fullraw_index._SWEEP_STRATEGY,
        },
    )

    assert fullraw_index._sweep_cache_entry_should_stop_no_hits(entry, 128)
    assert not fullraw_index._sweep_cache_entry_should_stop_no_hits(entry, 200)

    entry_with_hits = fullraw_index.SweepCacheEntry(
        created_at=time.time(),
        hits=[{"title": "signal"}],
        receipt={
            "shards_searched": 151,
            "shards_total": 1525,
            "partial_shard_search": True,
            "sweep_remaining_shards": 1374,
            "sweep_result_limit": 25,
            "sweep_strategy": fullraw_index._SWEEP_STRATEGY,
        },
    )

    assert not fullraw_index._sweep_cache_entry_should_stop_no_hits(entry_with_hits, 128)


def test_fast_health_reports_async_sweep_queue_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_dir = tmp_path / "remote-shards"
    shard_dir.mkdir()

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    monkeypatch.setenv("RESEARKA_FULLRAW_INDEX_HOST", "127.0.0.1")
    monkeypatch.setenv("RESEARKA_FULLRAW_INDEX_PORT", str(port))
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_DIR", str(shard_dir))
    monkeypatch.setenv("RESEARKA_FULLRAW_FAST_HEALTH", "1")
    monkeypatch.setenv("RESEARKA_FULLRAW_ASYNC_SWEEP", "1")
    monkeypatch.setenv("RESEARKA_FULLRAW_SWEEP_MAX_INFLIGHT", "2")
    monkeypatch.setenv("RESEARKA_FULLRAW_SWEEP_MAX_QUEUE", "16")
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MAX_BYTES", "64")
    monkeypatch.setattr(fullraw_index, "load_or_build_manifest", lambda *_args, **_kwargs: [])

    thread = threading.Thread(target=fullraw_index.run_server, daemon=True)
    thread.start()

    body: dict[str, object] | None = None
    for _ in range(50):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                body = json.loads(resp.read().decode())
            break
        except OSError:
            time.sleep(0.05)
    else:  # pragma: no cover - defensive test guard
        raise AssertionError("server did not start")

    assert body is not None
    assert body["fast_health"] is True
    shard_cache = body["shard_cache"]
    assert isinstance(shard_cache, dict)
    assert {
        key: shard_cache[key]
        for key in (
            "dir",
            "exists",
            "is_mount",
            "max_bytes",
            "copy_timeout_seconds",
            "copy_max_inflight",
            "copy_inflight",
            "copy_waiting",
        )
    } == {
        "dir": str(tmp_path / "cache"),
        "exists": False,
        "is_mount": False,
        "max_bytes": 64,
        "copy_timeout_seconds": 180.0,
        "copy_max_inflight": 2,
        "copy_inflight": 0,
        "copy_waiting": 0,
    }
    assert isinstance(shard_cache["copy_timeouts_total"], int)
    assert isinstance(shard_cache["copy_failures_total"], int)
    assert body["async_sweep"] == {
        "enabled": True,
        "inflight_count": 0,
        "queued_count": 0,
        "priority_queued_count": 0,
        "background_queued_count": 0,
        "max_inflight": 2,
        "priority_max_inflight": 3,
        "max_queue": 16,
        "priority_burst": True,
        "workers": fullraw_index._auto_sweep_workers(2),
        "worker_thread_count": 0,
    }
    fullraw_index._set_shard_cache_copy_max_inflight(None)


def test_sweep_queue_summary_counts_priority_and_background_jobs() -> None:
    queued_jobs = {
        "priority": fullraw_index.SweepJob(
            "priority",
            "priority query",
            10,
            1900,
            2100,
            "relevance",
            [],
            priority=True,
        ),
        "background": fullraw_index.SweepJob(
            "background",
            "background query",
            10,
            1900,
            2100,
            "relevance",
            [],
        ),
    }

    assert fullraw_index._sweep_queue_summary(
        {"running"},
        queued_jobs,
        max_inflight=2,
        max_queue=16,
        priority_burst=True,
        priority_max_inflight=4,
        workers=8,
        enabled=True,
    ) == {
        "enabled": True,
        "inflight_count": 1,
        "queued_count": 2,
        "priority_queued_count": 1,
        "background_queued_count": 1,
        "max_inflight": 2,
        "priority_max_inflight": 4,
        "max_queue": 16,
        "priority_burst": True,
        "workers": 8,
    }


def test_stale_sweep_inflight_prune_releases_only_expired_keys() -> None:
    inflight = {"fresh", "live", "orphan", "stale"}
    started = {"fresh": 95.0, "live": 10.0, "stale": 10.0}

    stale = fullraw_index._prune_stale_sweep_inflight(
        inflight,
        started,
        now=100.0,
        stale_after_seconds=60.0,
        live_keys={"live"},
    )

    assert set(stale) == {"orphan", "stale"}
    assert inflight == {"fresh", "live"}
    assert started == {"fresh": 95.0, "live": 10.0}


def test_sweep_watchdog_tick_reclaims_stale_slot_and_promotes_queue() -> None:
    inflight = {"stale"}
    started = {"stale": 10.0}
    queued = {"next"}
    job = fullraw_index.SweepJob("next", "next query", 10, 1900, 2100, "relevance", [])
    queued_jobs = {"next": job}

    stale, next_jobs = fullraw_index._sweep_watchdog_tick(
        sweep_inflight=inflight,
        sweep_inflight_started=started,
        sweep_queued=queued,
        sweep_queued_jobs=queued_jobs,
        max_inflight=1,
        allow_priority_burst=False,
        stale_after_seconds=60.0,
        now=100.0,
    )

    assert stale == ("stale",)
    assert next_jobs == [job]
    assert inflight == {"next"}
    assert started == {"next": 100.0}
    assert queued == set()
    assert queued_jobs == {}


def test_sweep_watchdog_does_not_replace_live_stale_worker() -> None:
    inflight = {"live"}
    started = {"live": 10.0}
    queued = {"next"}
    job = fullraw_index.SweepJob("next", "next query", 10, 1900, 2100, "relevance", [])
    queued_jobs = {"next": job}

    stale, next_jobs = fullraw_index._sweep_watchdog_tick(
        sweep_inflight=inflight,
        sweep_inflight_started=started,
        sweep_queued=queued,
        sweep_queued_jobs=queued_jobs,
        max_inflight=1,
        allow_priority_burst=False,
        stale_after_seconds=60.0,
        now=100.0,
        live_keys={"live"},
    )

    assert stale == ()
    assert next_jobs == []
    assert inflight == {"live"}
    assert started == {"live": 10.0}
    assert queued == {"next"}
    assert queued_jobs == {"next": job}


def test_sweep_watchdog_classifies_only_live_workers_without_heartbeats() -> None:
    release = threading.Event()
    stale_owner = threading.Thread(target=release.wait)
    fresh_owner = threading.Thread(target=release.wait)
    dead_owner = threading.Thread(target=lambda: None)
    stale_owner.start()
    fresh_owner.start()
    dead_owner.start()
    dead_owner.join(1)
    try:
        expired = fullraw_index._expired_live_sweep_workers(
            {"stale", "fresh", "dead", "orphan"},
            {"stale": 10.0, "fresh": 95.0, "dead": 10.0},
            {"stale": stale_owner, "fresh": fresh_owner, "dead": dead_owner},
            now=100.0,
            stale_after_seconds=60.0,
        )
        assert expired == (("stale", stale_owner),)
        assert fullraw_index._revalidate_expired_live_sweep_workers(
            expired,
            {"stale": 10.0},
            {"stale": stale_owner},
            now=100.0,
            stale_after_seconds=60.0,
        ) == ("stale",)

        assert fullraw_index._revalidate_expired_live_sweep_workers(
            expired,
            {"stale": 95.0},
            {"stale": stale_owner},
            now=100.0,
            stale_after_seconds=60.0,
        ) == ()
        assert fullraw_index._revalidate_expired_live_sweep_workers(
            expired,
            {"stale": 10.0},
            {"stale": threading.Thread(target=lambda: None)},
            now=100.0,
            stale_after_seconds=60.0,
        ) == ()
    finally:
        release.set()
        stale_owner.join(1)
        fresh_owner.join(1)


def test_registered_unstarted_sweep_generation_keeps_inflight_ownership() -> None:
    owner = threading.Thread(target=lambda: None)
    inflight = {"job"}
    last_progress = {"job": 10.0}

    released = fullraw_index._release_sweep_inflight_if_unowned(
        "job",
        sweep_inflight=inflight,
        sweep_inflight_started=last_progress,
        sweep_worker_threads={"job": owner},
    )

    assert released is False
    assert inflight == {"job"}
    assert last_progress == {"job": 10.0}


def test_failed_sweep_thread_start_restores_job_at_front_without_clobbering_owner() -> None:
    failed_job = fullraw_index.SweepJob(
        "failed", "failed query", 10, 1900, 2100, "relevance", [], priority=False
    )
    priority_job = fullraw_index.SweepJob(
        "priority", "priority query", 10, 1900, 2100, "relevance", [], priority=True
    )
    background_job = fullraw_index.SweepJob(
        "background", "background query", 10, 1900, 2100, "relevance", []
    )
    failed_owner = threading.Thread(target=lambda: None)
    workers = {"failed": failed_owner}
    inflight = {"failed"}
    started = {"failed": 10.0}
    queued = {"priority", "background"}
    queued_jobs = {"priority": priority_job, "background": background_job}

    assert fullraw_index._requeue_sweep_after_start_failure(
        job=failed_job,
        failed_owner=failed_owner,
        sweep_worker_threads=workers,
        sweep_inflight=inflight,
        sweep_inflight_started=started,
        sweep_queued=queued,
        sweep_queued_jobs=queued_jobs,
        max_queue=2,
    ) is True
    assert workers == {}
    assert inflight == set()
    assert started == {}
    assert queued == {"priority", "failed"}
    assert list(queued_jobs) == ["priority", "failed"]

    newer_owner = threading.Thread(target=lambda: None)
    workers["failed"] = newer_owner
    assert fullraw_index._requeue_sweep_after_start_failure(
        job=failed_job,
        failed_owner=failed_owner,
        sweep_worker_threads=workers,
        sweep_inflight={"failed"},
        sweep_inflight_started={"failed": 20.0},
        sweep_queued=queued,
        sweep_queued_jobs=queued_jobs,
        max_queue=2,
    ) is False
    assert workers == {"failed": newer_owner}
def test_sweep_cache_write_preserves_highest_progress(tmp_path: Path) -> None:
    cache_path = tmp_path / "sweeps" / "metformin.json"
    older_partial = fullraw_index.SweepCacheEntry(
        created_at=time.time() - 60,
        hits=[],
        receipt={
            "shards_searched": 301,
            "shards_total": 1525,
            "partial_shard_search": True,
            "sweep_remaining_shards": 1224,
        },
    )
    newer_partial = fullraw_index.SweepCacheEntry(
        created_at=time.time(),
        hits=[],
        receipt={
            "shards_searched": 512,
            "shards_total": 1525,
            "partial_shard_search": True,
            "sweep_remaining_shards": 1013,
        },
    )

    fullraw_index._write_sweep_cache(cache_path, newer_partial)
    fullraw_index._write_sweep_cache(cache_path, older_partial)

    selected = fullraw_index._load_sweep_cache(cache_path, ttl_seconds=0)

    assert selected is not None
    assert selected.receipt["shards_searched"] == 512
    assert selected.receipt["sweep_remaining_shards"] == 1013


def test_sweep_cache_write_replaces_incompatible_low_hit_terminal_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "sweeps" / "cold-immersion.json"
    low_hit_terminal = fullraw_index.SweepCacheEntry(
        created_at=time.time() - 60,
        hits=[{"title": f"Cold immersion training {index}"} for index in range(3)],
        receipt={
            "shards_searched": 1525,
            "shards_total": 1525,
            "partial_shard_search": False,
            "sweep_remaining_shards": 0,
        },
    )
    mining_partial = fullraw_index.SweepCacheEntry(
        created_at=time.time(),
        hits=[{"title": f"Cold immersion training {index}"} for index in range(10)],
        receipt={
            "shards_searched": 301,
            "shards_total": 1525,
            "partial_shard_search": True,
            "sweep_remaining_shards": 1224,
        },
    )

    fullraw_index._write_sweep_cache(cache_path, low_hit_terminal)
    fullraw_index._write_sweep_cache(cache_path, mining_partial)

    selected = fullraw_index._load_sweep_cache(cache_path, ttl_seconds=0)

    assert selected is not None
    assert len(selected.hits) == 10
    assert selected.receipt["shards_searched"] == 301
    assert selected.receipt["sweep_result_limit"] == 10


def test_sweep_admission_queues_without_exceeding_inflight_limit() -> None:
    inflight = {"active"}
    queued: set[str] = set()

    assert fullraw_index._admit_sweep_key(
        "cold-water",
        sweep_inflight=inflight,
        sweep_queued=queued,
        max_inflight=1,
    ) == "queued"
    assert inflight == {"active"}
    assert queued == {"cold-water"}
    assert fullraw_index._admit_sweep_key(
        "priority-cold-water",
        sweep_inflight=inflight,
        sweep_queued=queued,
        max_inflight=1,
        priority=True,
        allow_priority_burst=True,
    ) == "queued"
    assert inflight == {"active", "priority-cold-water"}
    assert "priority-cold-water" not in queued
    assert fullraw_index._admit_sweep_key(
        "active",
        sweep_inflight=inflight,
        sweep_queued=queued,
        max_inflight=1,
    ) == "running"

    inflight.clear()
    assert fullraw_index._admit_sweep_key(
        "cold-water",
        sweep_inflight=inflight,
        sweep_queued=queued,
        max_inflight=1,
    ) == "queued"
    assert inflight == {"cold-water"}
    assert queued == set()


def test_queued_sweep_job_promotes_when_lane_frees() -> None:
    inflight: set[str] = set()
    queued = {"cold-water"}
    job = fullraw_index.SweepJob(
        key="cold-water",
        query="cold water immersion",
        limit=10,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        catalog=[],
    )
    queued_jobs = {"cold-water": job}

    next_job = fullraw_index._take_next_queued_sweep_job(
        sweep_inflight=inflight,
        sweep_queued=queued,
        sweep_queued_jobs=queued_jobs,
        max_inflight=1,
    )

    assert next_job == job
    assert inflight == {"cold-water"}
    assert queued == set()
    assert queued_jobs == {}


def test_repolled_queued_sweep_job_keeps_fifo_position() -> None:
    older = fullraw_index.SweepJob("older", "older query", 10, 1900, 2100, "relevance", [])
    target_old = fullraw_index.SweepJob("target", "target old", 10, 1900, 2100, "relevance", [])
    later = fullraw_index.SweepJob("later", "later query", 10, 1900, 2100, "relevance", [])
    target_new = fullraw_index.SweepJob("target", "target fresh", 10, 1900, 2100, "relevance", [])
    queued = {"older", "target", "later"}
    queued_jobs = {"older": older, "target": target_old, "later": later}

    fullraw_index._queue_sweep_job(queued_jobs, "target", target_new)
    next_job = fullraw_index._take_next_queued_sweep_job(
        sweep_inflight=set(),
        sweep_queued=queued,
        sweep_queued_jobs=queued_jobs,
        max_inflight=1,
    )

    assert next_job == older
    assert list(queued_jobs) == ["target", "later"]
    assert queued_jobs["target"] == target_new


def test_full_lane_does_not_rotate_oldest_queued_sweep_job() -> None:
    first = fullraw_index.SweepJob(
        "first", "first query", 10, 1900, 2100, "relevance", [], priority=True
    )
    second = fullraw_index.SweepJob(
        "second", "second query", 10, 1900, 2100, "relevance", [], priority=True
    )
    queued_jobs = {"first": first, "second": second}

    next_job = fullraw_index._take_next_queued_sweep_job(
        sweep_inflight={"running"},
        sweep_queued={"first", "second"},
        sweep_queued_jobs=queued_jobs,
        max_inflight=1,
    )

    assert next_job is None
    assert list(queued_jobs) == ["first", "second"]


def test_priority_sweep_jobs_remain_fifo_ahead_of_background() -> None:
    first = fullraw_index.SweepJob(
        "first", "first query", 10, 1900, 2100, "relevance", [], priority=True
    )
    second = fullraw_index.SweepJob(
        "second", "second query", 10, 1900, 2100, "relevance", [], priority=True
    )
    third = fullraw_index.SweepJob(
        "third", "third query", 10, 1900, 2100, "relevance", [], priority=True
    )
    background = fullraw_index.SweepJob(
        "background", "background query", 10, 1900, 2100, "relevance", []
    )
    queued_jobs = {"first": first, "second": second, "background": background}

    fullraw_index._queue_sweep_job_with_priority(
        queued_jobs,
        "third",
        third,
        priority=True,
    )
    fullraw_index._queue_sweep_job_with_priority(
        queued_jobs,
        "first",
        first,
        priority=True,
    )

    assert list(queued_jobs) == ["first", "second", "third", "background"]


def test_priority_sweep_job_gets_next_lane_before_background_queue() -> None:
    older = fullraw_index.SweepJob("older", "older query", 10, 1900, 2100, "relevance", [])
    target = fullraw_index.SweepJob(
        "target", "target query", 10, 1900, 2100, "relevance", [], priority=True
    )
    later = fullraw_index.SweepJob("later", "later query", 10, 1900, 2100, "relevance", [])
    queued = {"older", "target", "later"}
    queued_jobs = {"older": older, "later": later}

    fullraw_index._queue_sweep_job_with_priority(queued_jobs, "target", target, priority=True)
    next_job = fullraw_index._take_next_queued_sweep_job(
        sweep_inflight=set(),
        sweep_queued=queued,
        sweep_queued_jobs=queued_jobs,
        max_inflight=1,
    )

    assert next_job == target
    assert list(queued_jobs) == ["older", "later"]


def test_repolled_background_job_cannot_jump_ahead_of_priority_queue() -> None:
    target = fullraw_index.SweepJob(
        "target", "target query", 10, 1900, 2100, "relevance", [], priority=True
    )
    background_old = fullraw_index.SweepJob(
        "background", "old query", 10, 1900, 2100, "relevance", []
    )
    background_new = fullraw_index.SweepJob(
        "background", "fresh query", 10, 1900, 2100, "relevance", []
    )
    later = fullraw_index.SweepJob("later", "later query", 10, 1900, 2100, "relevance", [])
    queued = {"target", "background", "later"}
    queued_jobs = {"target": target, "background": background_old, "later": later}

    fullraw_index._queue_sweep_job_with_priority(
        queued_jobs,
        "background",
        background_new,
        priority=False,
    )
    next_job = fullraw_index._take_next_queued_sweep_job(
        sweep_inflight=set(),
        sweep_queued=queued,
        sweep_queued_jobs=queued_jobs,
        max_inflight=1,
    )

    assert next_job == target
    assert list(queued_jobs) == ["background", "later"]
    assert queued_jobs["background"] == background_new


def test_background_repoll_cannot_downgrade_existing_priority_job() -> None:
    target = fullraw_index.SweepJob(
        "target", "target query", 10, 1900, 2100, "relevance", [], priority=True
    )
    background_repoll = fullraw_index.SweepJob(
        "target", "target query", 10, 1900, 2100, "relevance", []
    )
    queued_jobs = {"target": target}

    fullraw_index._queue_sweep_job_with_priority(
        queued_jobs,
        "target",
        background_repoll,
        priority=False,
    )

    assert queued_jobs == {"target": target}


def test_sweep_queue_cap_keeps_priority_before_background() -> None:
    older = fullraw_index.SweepJob("older", "older query", 10, 1900, 2100, "relevance", [])
    middle = fullraw_index.SweepJob("middle", "middle query", 10, 1900, 2100, "relevance", [])
    later = fullraw_index.SweepJob("later", "later query", 10, 1900, 2100, "relevance", [])
    target = fullraw_index.SweepJob("target", "target query", 10, 1900, 2100, "relevance", [], priority=True)
    queued = {"older", "middle", "later", "target"}
    queued_jobs = {"older": older, "middle": middle, "later": later}

    fullraw_index._queue_sweep_job_with_priority(
        queued_jobs,
        "target",
        target,
        priority=True,
        sweep_queued=queued,
        max_queue=3,
    )

    assert list(queued_jobs) == ["target", "older", "middle"]
    assert queued == {"target", "older", "middle"}


def test_sweep_queue_cap_drops_new_background_before_priority() -> None:
    target = fullraw_index.SweepJob("target", "target query", 10, 1900, 2100, "relevance", [], priority=True)
    older = fullraw_index.SweepJob("older", "older query", 10, 1900, 2100, "relevance", [])
    later = fullraw_index.SweepJob("later", "later query", 10, 1900, 2100, "relevance", [])
    queued = {"target", "older", "later"}
    queued_jobs = {"target": target, "older": older}

    fullraw_index._queue_sweep_job_with_priority(
        queued_jobs,
        "later",
        later,
        priority=False,
        sweep_queued=queued,
        max_queue=2,
    )

    assert list(queued_jobs) == ["target", "older"]
    assert queued == {"target", "older"}


def test_priority_sweep_job_waits_without_burst_lane() -> None:
    inflight = {"background"}
    queued = {"target"}
    target = fullraw_index.SweepJob("target", "target query", 10, 1900, 2100, "relevance", [], priority=True)
    queued_jobs = {"target": target}

    next_job = fullraw_index._take_next_queued_sweep_job(
        sweep_inflight=inflight,
        sweep_queued=queued,
        sweep_queued_jobs=queued_jobs,
        max_inflight=1,
    )

    assert next_job is None
    assert inflight == {"background"}
    assert queued == {"target"}
    assert queued_jobs == {"target": target}


def test_priority_sweep_job_can_use_opt_in_burst_lane() -> None:
    inflight = {"background"}
    queued = {"target"}
    target = fullraw_index.SweepJob("target", "target query", 10, 1900, 2100, "relevance", [], priority=True)
    queued_jobs = {"target": target}

    next_job = fullraw_index._take_next_queued_sweep_job(
        sweep_inflight=inflight,
        sweep_queued=queued,
        sweep_queued_jobs=queued_jobs,
        max_inflight=1,
        allow_priority_burst=True,
    )

    assert next_job == target
    assert inflight == {"background", "target"}
    assert queued == set()
    assert queued_jobs == {}


def test_background_sweep_job_cannot_use_priority_burst_lane() -> None:
    inflight = {"background"}
    queued = {"target"}
    target = fullraw_index.SweepJob("target", "target query", 10, 1900, 2100, "relevance", [])
    queued_jobs = {"target": target}

    next_job = fullraw_index._take_next_queued_sweep_job(
        sweep_inflight=inflight,
        sweep_queued=queued,
        sweep_queued_jobs=queued_jobs,
        max_inflight=1,
        allow_priority_burst=True,
    )

    assert next_job is None
    assert inflight == {"background"}
    assert queued == {"target"}
    assert queued_jobs == {"target": target}


def test_priority_burst_lane_is_bounded() -> None:
    inflight = {"background", "first-priority"}
    queued = {"target"}
    target = fullraw_index.SweepJob("target", "target query", 10, 1900, 2100, "relevance", [], priority=True)
    queued_jobs = {"target": target}

    next_job = fullraw_index._take_next_queued_sweep_job(
        sweep_inflight=inflight,
        sweep_queued=queued,
        sweep_queued_jobs=queued_jobs,
        max_inflight=1,
    )

    assert next_job is None
    assert inflight == {"background", "first-priority"}
    assert queued == {"target"}
    assert queued_jobs == {"target": target}


def test_priority_sweep_job_uses_configured_priority_inflight_ceiling() -> None:
    inflight = {"background", "first-priority"}
    queued = {"target"}
    target = fullraw_index.SweepJob("target", "target query", 10, 1900, 2100, "relevance", [], priority=True)
    queued_jobs = {"target": target}

    next_job = fullraw_index._take_next_queued_sweep_job(
        sweep_inflight=inflight,
        sweep_queued=queued,
        sweep_queued_jobs=queued_jobs,
        max_inflight=1,
        priority_max_inflight=3,
    )

    assert next_job == target
    assert inflight == {"background", "first-priority", "target"}
    assert queued == set()
    assert queued_jobs == {}


def test_configured_priority_inflight_ceiling_remains_bounded() -> None:
    inflight = {"background", "first-priority", "second-priority"}
    queued = {"target"}
    target = fullraw_index.SweepJob("target", "target query", 10, 1900, 2100, "relevance", [], priority=True)
    queued_jobs = {"target": target}

    next_job = fullraw_index._take_next_queued_sweep_job(
        sweep_inflight=inflight,
        sweep_queued=queued,
        sweep_queued_jobs=queued_jobs,
        max_inflight=1,
        priority_max_inflight=3,
    )

    assert next_job is None
    assert inflight == {"background", "first-priority", "second-priority"}
    assert queued == {"target"}
    assert queued_jobs == {"target": target}


def test_full_sweep_order_reuses_cache_without_query_bias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entries = [
        ShardCatalogEntry(tmp_path / "large-metformin.sqlite", 2, 0, ("openalex",), 1, 10, 200, topic_terms=("metformin",)),
        ShardCatalogEntry(tmp_path / "small-cwi.sqlite", 1, 0, ("pubmed",), 1, 10, 10, topic_terms=("immersion",)),
        ShardCatalogEntry(tmp_path / "cached-resveratrol.sqlite", 3, 0, ("biorxiv",), 1, 10, 100, topic_terms=("resveratrol",)),
    ]
    cached = {entries[2].path}

    monkeypatch.setattr(
        fullraw_index,
        "_cached_materialized_shard_path",
        lambda path: path if path in cached else None,
    )

    assert [entry.path.name for entry in fullraw_index._cache_reuse_sweep_entries(entries)] == [
        "cached-resveratrol.sqlite",
        "small-cwi.sqlite",
        "large-metformin.sqlite",
    ]


def test_complete_sweep_retries_failed_shards_until_coverage_is_complete() -> None:
    receipt = {
        "sweep_failed_paths": ("shard_a.sqlite", "shard_b.sqlite"),
        "sweep_remaining_shards": 4,
    }

    assert fullraw_index._sweep_failed_path_strings_for_mode(
        receipt,
        require_complete_sweep=True,
    ) == set()
    assert fullraw_index._sweep_failed_path_strings_for_mode(
        receipt,
        require_complete_sweep=False,
    ) == {"shard_a.sqlite", "shard_b.sqlite"}
    assert fullraw_index._sweep_remaining_shard_count(
        selected_shards=10,
        completed_shards=6,
        failed_shards=2,
        require_complete_sweep=True,
    ) == 4
    assert fullraw_index._sweep_remaining_shard_count(
        selected_shards=10,
        completed_shards=6,
        failed_shards=2,
        require_complete_sweep=False,
    ) == 2


def test_complete_sweep_defers_timed_out_front_shards(tmp_path: Path) -> None:
    entries = [_entry(tmp_path, idx, "openalex") for idx in range(4)]

    first, deferred = fullraw_index._next_sweep_pass_entries(
        entries,
        completed_path_strings=set(),
        failed_path_strings=set(),
        deferred_path_strings={str(entries[0].path), str(entries[1].path)},
        limit=2,
    )
    second, reset_deferred = fullraw_index._next_sweep_pass_entries(
        entries,
        completed_path_strings={str(entries[2].path), str(entries[3].path)},
        failed_path_strings=set(),
        deferred_path_strings=deferred,
        limit=2,
    )

    assert [entry.path for entry in first] == [entries[2].path, entries[3].path]
    assert second == entries[:2]
    assert reset_deferred == set()


def test_complete_sweep_does_not_poison_missed_pass_entries(tmp_path: Path) -> None:
    pass_entries = [_entry(tmp_path, index, "openalex") for index in range(3)]
    completed = {str(pass_entries[0].path)}

    assert fullraw_index._sweep_pass_failed_path_strings(
        pass_entries,
        completed_path_strings=completed,
        existing_failed_path_strings=set(),
        require_complete_sweep=True,
    ) == set()
    assert fullraw_index._sweep_pass_failed_path_strings(
        pass_entries,
        completed_path_strings=completed,
        existing_failed_path_strings=set(),
        require_complete_sweep=False,
    ) == {str(pass_entries[1].path), str(pass_entries[2].path)}


def test_shard_search_materializes_before_isolated_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = tmp_path / "remote.sqlite"
    materialized = tmp_path / "local.sqlite"
    searched: list[Path] = []

    populate_seen: list[bool] = []

    def fake_materialized(
        path: Path,
        *,
        preserve: set[Path] | None = None,
        populate: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> Path:
        del cancel_event
        assert path == original
        assert preserve == set()
        populate_seen.append(populate)
        return materialized

    def fake_search(path: Path, *_args: object) -> list[dict[str, object]]:
        searched.append(path)
        return [{"title": "Metformin longevity", "score": 1.0}]

    monkeypatch.setattr(fullraw_index, "_materialized_shard_path", fake_materialized)
    monkeypatch.setattr(fullraw_index, "_search_one_shard_for_pool", fake_search)

    hits, completed_paths, timed_out, _metrics = fullraw_index._search_shard_paths_with_paths_and_receipt(
        [original],
        "metformin longevity",
        limit=1,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        workers=1,
        timeout_seconds=5,
    )

    assert searched == [materialized]
    assert completed_paths == [original]
    assert hits[0]["title"] == "Metformin longevity"
    assert populate_seen == [True]
    assert timed_out is False


def test_shard_search_preserves_materialized_batch_before_worker_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    remote_a = tmp_path / "remote-a.sqlite"
    remote_b = tmp_path / "remote-b.sqlite"
    local_a = tmp_path / "local-a.sqlite"
    local_b = tmp_path / "local-b.sqlite"
    live_cache_paths: set[Path] = set()
    preserve_seen: list[set[Path]] = []

    populate_seen: list[bool] = []

    def fake_materialized(
        path: Path,
        *,
        preserve: set[Path] | None = None,
        populate: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> Path:
        del cancel_event
        populate_seen.append(populate)
        preserve = preserve or set()
        preserve_seen.append(set(preserve))
        if path == remote_a:
            live_cache_paths.add(local_a)
            return local_a
        if path == remote_b:
            live_cache_paths.add(local_b)
            return local_b
        raise AssertionError(path)

    def fake_search(path: Path, *_args: object) -> list[dict[str, object]]:
        if path not in live_cache_paths:
            raise OSError(f"evicted before search: {path}")
        return [{"title": path.stem, "score": 1.0}]

    monkeypatch.setattr(fullraw_index, "_materialized_shard_path", fake_materialized)
    monkeypatch.setattr(fullraw_index, "_search_one_shard_for_pool", fake_search)

    hits, completed_paths, timed_out, _metrics = fullraw_index._search_shard_paths_with_paths_and_receipt(
        [remote_a, remote_b],
        "metformin longevity",
        limit=2,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        workers=2,
        timeout_seconds=5,
    )

    assert completed_paths == [remote_a, remote_b]
    assert {hit["title"] for hit in hits} == {"local-a", "local-b"}
    assert populate_seen == [True, True]
    assert preserve_seen == [set(), set()]
    assert timed_out is False


def test_shard_search_materializes_batch_concurrently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    remotes = [tmp_path / f"remote-{index}.sqlite" for index in range(3)]
    locals_by_remote = {path: tmp_path / f"local-{path.stem}.sqlite" for path in remotes}
    barrier = threading.Barrier(len(remotes))
    materialized: list[Path] = []

    def fake_materialized(
        path: Path,
        *,
        preserve: set[Path] | None = None,
        populate: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> Path:
        del preserve, cancel_event
        assert populate is True
        materialized.append(path)
        barrier.wait(timeout=1.0)
        return locals_by_remote[path]

    monkeypatch.setattr(fullraw_index, "_materialized_shard_path", fake_materialized)
    monkeypatch.setattr(
        fullraw_index,
        "_search_one_shard_for_pool",
        lambda path, *_args: [{"title": path.stem, "score": 1.0}],
    )

    hits, completed_paths, timed_out, _metrics = fullraw_index._search_shard_paths_with_paths_and_receipt(
        remotes,
        "metformin longevity",
        limit=3,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        workers=3,
        timeout_seconds=5,
    )

    assert set(materialized) == set(remotes)
    assert set(completed_paths) == set(remotes)
    assert {hit["title"] for hit in hits} == {path.stem for path in locals_by_remote.values()}
    assert timed_out is False


def test_shard_search_caps_worker_batch_to_cache_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    remotes = [tmp_path / f"remote-{idx}.sqlite" for idx in range(4)]
    for path in remotes:
        path.write_bytes(b"x" * 6)
    preserve_sizes: list[int] = []

    populate_seen: list[bool] = []

    def fake_materialized(
        path: Path,
        *,
        preserve: set[Path] | None = None,
        populate: bool = False,
        cancel_event: threading.Event | None = None,
    ) -> Path:
        del cancel_event
        preserve_sizes.append(len(preserve or set()))
        populate_seen.append(populate)
        return tmp_path / f"local-{path.stem}.sqlite"

    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES", "24")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SWEEP_MAX_INFLIGHT", "2")
    monkeypatch.setattr(fullraw_index, "_materialized_shard_path", fake_materialized)
    monkeypatch.setattr(
        fullraw_index,
        "_search_one_shard_for_pool",
        lambda path, *_args: [{"title": path.stem, "score": 1.0}],
    )

    _hits, completed_paths, timed_out, _metrics = fullraw_index._search_shard_paths_with_paths_and_receipt(
        remotes,
        "metformin longevity",
        limit=4,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        workers=4,
        timeout_seconds=5,
    )

    assert preserve_sizes == [0, 0, 0, 0]
    assert populate_seen == [True, True, True, True]
    assert completed_paths == remotes
    assert timed_out is False

    for path, size in zip(remotes, (13, 1, 1, 1), strict=True):
        path.write_bytes(b"x" * size)
    preserve_sizes.clear()
    _hits, completed_paths, timed_out, _metrics = fullraw_index._search_shard_paths_with_paths_and_receipt(
        remotes,
        "metformin longevity",
        limit=4,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        workers=4,
        timeout_seconds=5,
    )

    assert preserve_sizes == [0, 0, 0, 0]
    assert set(completed_paths) == set(remotes)
    assert timed_out is False


def test_cache_fit_batch_only_reserves_burst_lane_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    remotes = [tmp_path / f"remote-{idx}.sqlite" for idx in range(3)]
    for path in remotes:
        path.write_bytes(b"x" * 5)
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES", "20")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SWEEP_MAX_INFLIGHT", "2")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SWEEP_PRIORITY_BURST", "0")
    monkeypatch.delenv("RESEARKA_FULLRAW_SWEEP_PRIORITY_BURST", raising=False)

    assert fullraw_index._cache_fit_path_batch(remotes, start=0, worker_count=3) == remotes[:2]

    monkeypatch.setenv("V5_MEMO_FULL_RAW_SWEEP_PRIORITY_BURST", "1")

    assert fullraw_index._cache_fit_path_batch(remotes, start=0, worker_count=3) == remotes[:1]


def test_cache_fit_batch_keeps_parallel_batch_when_cache_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    remotes = [tmp_path / f"remote-{idx}.sqlite" for idx in range(3)]
    for path in remotes:
        path.write_bytes(b"x" * 5)
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MAX_BYTES", "auto")
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MIN_FREE_BYTES", "500")
    monkeypatch.setattr(
        "v5_memo.fullraw_index.shutil.disk_usage",
        lambda _path: _FakeDiskUsage(total=1000, used=800, free=200),
    )

    assert fullraw_index._cache_fit_path_batch(remotes, start=0, worker_count=3) == remotes


def test_cache_fit_batch_uses_worker_cache_as_scheduler_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    remotes = [tmp_path / f"remote-{idx}.sqlite" for idx in range(3)]
    for path, size in zip(remotes, (30, 6, 6), strict=True):
        path.write_bytes(b"x" * size)
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MAX_BYTES", "100")
    monkeypatch.setenv("RESEARKA_FULLRAW_SWEEP_WORKER_CACHE_BYTES", "6")
    monkeypatch.setenv("RESEARKA_FULLRAW_SWEEP_MAX_INFLIGHT", "1")
    monkeypatch.setenv("RESEARKA_FULLRAW_SWEEP_PRIORITY_BURST", "0")

    assert fullraw_index._cache_fit_path_batch(remotes, start=0, worker_count=3) == remotes[:1]
    assert fullraw_index._cache_fit_path_batch(remotes, start=1, worker_count=2) == remotes[1:]


def test_cache_fit_warm_entries_defers_oversized_remote_shards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    entries = [
        ShardCatalogEntry(
            tmp_path / "huge.sqlite",
            0,
            0,
            ("openalex",),
            1,
            10,
            20,
            topic_terms=("platform",),
        ),
        *[
            ShardCatalogEntry(
                tmp_path / f"small-{idx}.sqlite",
                idx + 1,
                0,
                ("openalex",),
                1,
                10,
                6,
                topic_terms=("platform",),
            )
            for idx in range(4)
        ],
    ]
    for entry in entries:
        entry.path.write_bytes(b"x" * entry.bytes_used)
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MAX_BYTES", "10")

    ordered = fullraw_index._cache_fit_warm_entries(
        entries,
        entries,
        query="platform strategy",
        target_ready=len(entries),
    )

    assert ordered[-1] == entries[0]
    assert {entry.path for entry in ordered} == {entry.path for entry in entries}


class _FakeDiskUsage(NamedTuple):
    total: int
    used: int
    free: int


def test_shard_local_cache_auto_budget_uses_free_space(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "ready.sqlite").write_bytes(b"x" * 10)
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES", "auto")
    monkeypatch.setattr(
        "v5_memo.fullraw_index.shutil.disk_usage",
        lambda _path: _FakeDiskUsage(total=1000, used=600, free=400),
    )

    assert fullraw_index._shard_local_cache_max_bytes() == 360


def test_shard_local_cache_auto_budget_honors_min_free_gb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "ready.sqlite").write_bytes(b"x" * 10)
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MAX_BYTES", "auto")
    monkeypatch.delenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MIN_FREE_BYTES", raising=False)
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MIN_FREE_GB", "0.0000002")
    monkeypatch.setattr(
        "v5_memo.fullraw_index.shutil.disk_usage",
        lambda _path: _FakeDiskUsage(total=1000, used=600, free=400),
    )

    assert fullraw_index._shard_local_cache_max_bytes() == 196


def test_materialized_shard_path_does_not_recache_local_cache_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached = cache_dir / "abcd-fullraw_shard_0001.sqlite"
    cached.write_text("already local")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_DIR", str(cache_dir))

    assert fullraw_index._shard_cache_path(cached) is None
    assert fullraw_index._materialized_shard_path(cached) == cached


def test_shard_cache_copy_command_bypasses_configured_mount(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mount_root = tmp_path / "mounted-shards"
    source = mount_root / "batch_00007" / "fullraw_shard_0003.sqlite"
    target = tmp_path / "cache" / "shard.sqlite"
    monkeypatch.setenv(
        "RESEARKA_FULLRAW_SHARD_REMOTE",
        "sb:researka-database/index/v5/fullraw-fts",
    )
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_DIR", str(mount_root))

    command = fullraw_index._shard_cache_copy_command(source, target)

    assert command[:4] == [
        "rclone",
        "copyto",
        "sb:researka-database/index/v5/fullraw-fts/batch_00007/fullraw_shard_0003.sqlite",
        str(target),
    ]
    assert "--inplace" in command
    assert command[-4:] == ["--retries", "1", "--low-level-retries", "2"]


def test_shard_cache_copy_command_falls_back_outside_configured_mount(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mount_root = tmp_path / "mounted-shards"
    source = tmp_path / "outside" / "fullraw_shard_0003.sqlite"
    target = tmp_path / "cache" / "shard.sqlite"
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_REMOTE", "sb:fullraw-fts")
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_DIR", str(mount_root))

    command = fullraw_index._shard_cache_copy_command(source, target)

    assert command[:3] == [
        sys.executable,
        "-c",
        "import shutil,sys; shutil.copy2(sys.argv[1], sys.argv[2])",
    ]


def test_shard_cache_copy_timeout_kills_stalled_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StalledCopy:
        def __init__(self) -> None:
            self.killed = False
            self.waits = 0

        def wait(self, *, timeout: float) -> int:
            self.waits += 1
            if not self.killed:
                raise subprocess.TimeoutExpired(cmd="copy", timeout=timeout)
            return -9

        def kill(self) -> None:
            self.killed = True

    stalled = StalledCopy()

    def fake_popen(*_args: object, **_kwargs: object) -> StalledCopy:
        return stalled

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monotonic = iter((0.0, 0.0, 0.02))
    monkeypatch.setattr("v5_memo.fullraw_index.time.monotonic", lambda: next(monotonic))

    before = fullraw_index._shard_local_cache_health()
    before_timeouts = before["copy_timeouts_total"]
    before_failures = before["copy_failures_total"]
    assert isinstance(before_timeouts, int)
    assert isinstance(before_failures, int)
    with pytest.raises(TimeoutError, match="shard cache copy made no progress"):
        fullraw_index._copy2_with_timeout(
            Path("remote.sqlite"),
            Path("local.sqlite"),
            timeout_seconds=0.01,
            attempts=1,
        )

    assert stalled.killed
    assert stalled.waits == 2
    after = fullraw_index._shard_local_cache_health()
    assert after["copy_inflight"] == 0
    assert after["copy_timeouts_total"] == before_timeouts + 1
    assert after["copy_failures_total"] == before_failures + 1


def test_shard_cache_copy_timeout_resets_when_copy_progresses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "local.sqlite"

    class ProgressingCopy:
        waits = 0

        def wait(self, *, timeout: float) -> int:
            del timeout
            self.waits += 1
            if self.waits < 3:
                target.write_bytes(b"x" * self.waits)
                raise subprocess.TimeoutExpired(cmd="copy", timeout=0.01)
            return 0

        def kill(self) -> None:
            raise AssertionError("progressing copy must not be killed")

    progressing = ProgressingCopy()
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: progressing)
    monotonic = iter((0.0, 0.0, 0.02, 0.02, 0.04, 0.04))
    monkeypatch.setattr("v5_memo.fullraw_index.time.monotonic", lambda: next(monotonic))

    before = fullraw_index._shard_local_cache_health()
    fullraw_index._copy2_with_timeout(
        Path("remote.sqlite"),
        target,
        timeout_seconds=0.01,
    )
    after = fullraw_index._shard_local_cache_health()

    assert progressing.waits == 3
    assert after["copy_inflight"] == 0
    assert after["copy_timeouts_total"] == before["copy_timeouts_total"]
    assert after["copy_failures_total"] == before["copy_failures_total"]


def test_shard_cache_copy_retries_transient_stall_without_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CopyAttempt:
        def __init__(self, *, stalls: bool) -> None:
            self.stalls = stalls
            self.killed = False

        def wait(self, *, timeout: float) -> int:
            if self.stalls and not self.killed:
                raise subprocess.TimeoutExpired(cmd="copy", timeout=timeout)
            return -9 if self.killed else 0

        def kill(self) -> None:
            self.killed = True

    stalled = CopyAttempt(stalls=True)
    successful = CopyAttempt(stalls=False)
    copies = iter((stalled, successful))
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: next(copies))
    monotonic = iter((0.0, 0.0, 0.02, 0.03, 0.03))
    monkeypatch.setattr("v5_memo.fullraw_index.time.monotonic", lambda: next(monotonic))
    before = fullraw_index._shard_local_cache_health()
    before_timeouts = before["copy_timeouts_total"]
    before_failures = before["copy_failures_total"]
    assert isinstance(before_timeouts, int)
    assert isinstance(before_failures, int)

    fullraw_index._copy2_with_timeout(
        Path("remote.sqlite"),
        Path("local.sqlite"),
        timeout_seconds=0.01,
        attempts=2,
    )

    after = fullraw_index._shard_local_cache_health()
    assert stalled.killed
    assert not successful.killed
    assert after["copy_inflight"] == 0
    assert after["copy_timeouts_total"] == before_timeouts + 1
    assert after["copy_failures_total"] == before_failures


def test_shard_cache_copy_retries_nonzero_exit_and_cleans_partial_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "local.sqlite"
    rclone_partial = target.with_name(f"{target.name}.deadbeef.partial")
    returncodes = iter((1, 0))
    targets_seen_before_spawn: list[bool] = []
    partials_seen_before_spawn: list[bool] = []

    class CopyAttempt:
        def __init__(self, returncode: int, stderr: BinaryIO) -> None:
            self.returncode = returncode
            self.stderr = stderr

        def wait(self, *, timeout: float) -> int:
            del timeout
            target.write_bytes(b"partial" if self.returncode else b"complete")
            if self.returncode:
                rclone_partial.write_bytes(b"rclone partial")
                self.stderr.write(b"connection reset")
            return self.returncode

        def kill(self) -> None:
            raise AssertionError("completed copy must not be killed")

    def create_copy(*_args: object, **kwargs: object) -> CopyAttempt:
        targets_seen_before_spawn.append(target.exists())
        partials_seen_before_spawn.append(rclone_partial.exists())
        return CopyAttempt(next(returncodes), cast(BinaryIO, kwargs["stderr"]))

    monkeypatch.setattr(subprocess, "Popen", create_copy)
    before = fullraw_index._shard_local_cache_health()

    fullraw_index._copy2_with_timeout(
        Path("remote.sqlite"),
        target,
        timeout_seconds=10,
        attempts=2,
    )

    after = fullraw_index._shard_local_cache_health()
    assert targets_seen_before_spawn == [False, False]
    assert partials_seen_before_spawn == [False, False]
    assert target.read_bytes() == b"complete"
    assert not rclone_partial.exists()
    assert after["copy_failures_total"] == before["copy_failures_total"]


def test_shard_cache_copy_reports_terminal_rclone_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "local.sqlite"

    class FailedCopy:
        def __init__(self, stderr: BinaryIO) -> None:
            self.stderr = stderr

        def wait(self, *, timeout: float) -> int:
            del timeout
            target.write_bytes(b"partial")
            self.stderr.write(b"connection reset by peer")
            return 1

        def kill(self) -> None:
            raise AssertionError("completed copy must not be killed")

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **kwargs: FailedCopy(cast(BinaryIO, kwargs["stderr"])),
    )
    before = fullraw_index._shard_local_cache_health()
    before_failures = before["copy_failures_total"]
    assert isinstance(before_failures, int)

    with pytest.raises(OSError, match="connection reset by peer"):
        fullraw_index._copy2_with_timeout(
            Path("remote.sqlite"),
            target,
            timeout_seconds=10,
            attempts=1,
        )

    after = fullraw_index._shard_local_cache_health()
    assert not target.exists()
    assert "connection reset by peer" in capsys.readouterr().err
    assert after["copy_failures_total"] == before_failures + 1


def test_shard_cache_copy_requests_restart_instead_of_overlapping_stuck_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CopyAttempt:
        def __init__(self, *, stalls: bool) -> None:
            self.stalls = stalls
            self.killed = False
            self.wait_timeouts: list[float | None] = []

        def wait(self, *, timeout: float | None = None) -> int:
            self.wait_timeouts.append(timeout)
            if self.stalls and not self.killed:
                raise subprocess.TimeoutExpired(cmd="copy", timeout=timeout or 0.0)
            if self.stalls and timeout is not None:
                raise subprocess.TimeoutExpired(cmd="copy", timeout=timeout)
            return -9 if self.killed else 0

        def kill(self) -> None:
            self.killed = True

    stalled = CopyAttempt(stalls=True)
    restart_reasons: list[str] = []
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: stalled)
    monkeypatch.setattr(fullraw_index, "_request_process_restart", restart_reasons.append)
    monotonic = iter((0.0, 0.0, 0.02))
    monkeypatch.setattr("v5_memo.fullraw_index.time.monotonic", lambda: next(monotonic))

    with pytest.raises(OSError, match="could not be reaped"):
        fullraw_index._copy2_with_timeout(
            Path("remote.sqlite"),
            Path("local.sqlite"),
            timeout_seconds=0.01,
            attempts=2,
        )

    assert stalled.killed
    assert stalled.wait_timeouts[-1:] == [1.0]
    assert restart_reasons == ["copy child remained alive after SIGKILL"]


def test_shard_cache_copy_cancellation_kills_copy_without_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancel_event = threading.Event()

    class ActiveCopy:
        killed = False

        def wait(self, *, timeout: float) -> int:
            if not self.killed:
                cancel_event.set()
                raise subprocess.TimeoutExpired(cmd="copy", timeout=timeout)
            return -9

        def kill(self) -> None:
            self.killed = True

    active = ActiveCopy()
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: active)
    before = fullraw_index._shard_local_cache_health()

    with pytest.raises(CancelledError, match="shard cache copy cancelled"):
        fullraw_index._copy2_with_timeout(
            Path("remote.sqlite"),
            Path("local.sqlite"),
            timeout_seconds=180,
            cancel_event=cancel_event,
        )

    after = fullraw_index._shard_local_cache_health()
    assert active.killed
    assert after["copy_inflight"] == 0
    assert after["copy_timeouts_total"] == before["copy_timeouts_total"]
    assert after["copy_failures_total"] == before["copy_failures_total"]


def test_shard_cache_copy_prestart_cancellation_spawns_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cancel_event = threading.Event()
    cancel_event.set()

    def fail_spawn(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("pre-cancelled copy must not spawn")

    monkeypatch.setattr(subprocess, "Popen", fail_spawn)
    with pytest.raises(CancelledError, match="cancelled before start"):
        fullraw_index._copy2_with_timeout(
            Path("remote.sqlite"),
            Path("local.sqlite"),
            timeout_seconds=180,
            cancel_event=cancel_event,
        )

    health = fullraw_index._shard_local_cache_health()
    assert health["copy_inflight"] == 0
    assert health["copy_waiting"] == 0


def test_shard_cache_copy_limit_serializes_all_copy_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    errors: list[BaseException] = []
    created = 0

    class CopyAttempt:
        def __init__(self, *, first: bool) -> None:
            self.first = first

        def wait(self, *, timeout: float) -> int:
            if not self.first:
                second_started.set()
                return 0
            first_started.set()
            if not release_first.wait(timeout):
                raise subprocess.TimeoutExpired(cmd="copy", timeout=timeout)
            return 0

        def kill(self) -> None:
            raise AssertionError("healthy bounded copy must not be killed")

    def create_copy(*_args: object, **_kwargs: object) -> CopyAttempt:
        nonlocal created
        created += 1
        return CopyAttempt(first=created == 1)

    def copy(source: str) -> None:
        try:
            fullraw_index._copy2_with_timeout(
                Path(source),
                Path(f"{source}.local"),
                timeout_seconds=10,
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    monkeypatch.setattr(fullraw_index, "_SHARD_LOCAL_CACHE_COPY_MAX_INFLIGHT", 1)
    monkeypatch.setattr(subprocess, "Popen", create_copy)
    first = threading.Thread(target=copy, args=("first.sqlite",))
    second = threading.Thread(target=copy, args=("second.sqlite",))

    first.start()
    assert first_started.wait(1)
    second.start()
    assert not second_started.wait(0.1)
    assert fullraw_index._shard_local_cache_health()["copy_inflight"] == 1
    assert fullraw_index._shard_local_cache_health()["copy_waiting"] == 1

    release_first.set()
    first.join(1)
    second.join(1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not errors
    assert second_started.is_set()
    assert fullraw_index._shard_local_cache_health()["copy_inflight"] == 0
    assert fullraw_index._shard_local_cache_health()["copy_waiting"] == 0


def test_shard_cache_copy_limit_also_bounds_copy_without_explicit_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    errors: list[BaseException] = []

    created = 0

    class CopyAttempt:
        def __init__(self, *, first: bool) -> None:
            self.first = first

        def wait(self, *, timeout: float) -> int:
            if self.first:
                first_started.set()
                assert release_first.wait(timeout)
            else:
                second_started.set()
            return 0

        def kill(self) -> None:
            raise AssertionError("healthy copy must not be killed")

    def create_copy(*_args: object, **_kwargs: object) -> CopyAttempt:
        nonlocal created
        created += 1
        return CopyAttempt(first=created == 1)

    def copy(source: str) -> None:
        try:
            fullraw_index._copy2_with_timeout(
                Path(source),
                Path(f"{source}.local"),
                timeout_seconds=None,
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    monkeypatch.setattr(fullraw_index, "_SHARD_LOCAL_CACHE_COPY_MAX_INFLIGHT", 1)
    monkeypatch.setattr(subprocess, "Popen", create_copy)
    first = threading.Thread(target=copy, args=("first.sqlite",))
    second = threading.Thread(target=copy, args=("second.sqlite",))

    first.start()
    assert first_started.wait(1)
    second.start()
    assert not second_started.wait(0.1)
    release_first.set()
    first.join(1)
    second.join(1)

    assert not errors
    assert second_started.is_set()
    assert fullraw_index._shard_local_cache_health()["copy_inflight"] == 0


def test_shard_cache_copy_limit_holds_four_callers_to_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()
    three_started = threading.Event()
    created = 0
    created_lock = threading.Lock()
    errors: list[BaseException] = []

    class CopyAttempt:
        def wait(self, *, timeout: float) -> int:
            if not release.wait(timeout):
                raise subprocess.TimeoutExpired(cmd="copy", timeout=timeout)
            return 0

        def kill(self) -> None:
            raise AssertionError("healthy bounded copy must not be killed")

    def create_copy(*_args: object, **_kwargs: object) -> CopyAttempt:
        nonlocal created
        with created_lock:
            created += 1
            if created == 3:
                three_started.set()
        return CopyAttempt()

    def copy(index: int) -> None:
        try:
            fullraw_index._copy2_with_timeout(
                Path(f"remote-{index}.sqlite"),
                Path(f"local-{index}.sqlite"),
                timeout_seconds=10,
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    monkeypatch.setattr(fullraw_index, "_SHARD_LOCAL_CACHE_COPY_MAX_INFLIGHT", 3)
    monkeypatch.setattr(subprocess, "Popen", create_copy)
    threads = [threading.Thread(target=copy, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()

    assert three_started.wait(1)
    time.sleep(0.1)
    health = fullraw_index._shard_local_cache_health()
    assert created == 3
    assert health["copy_max_inflight"] == 3
    assert health["copy_inflight"] == 3
    assert health["copy_waiting"] == 1

    release.set()
    for thread in threads:
        thread.join(1)

    assert not errors
    assert created == 4
    assert fullraw_index._shard_local_cache_health()["copy_inflight"] == 0


def test_shard_cache_copy_waiter_cancels_without_taking_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    cancel_waiter = threading.Event()
    errors: list[BaseException] = []
    created = 0

    class ActiveCopy:
        def wait(self, *, timeout: float) -> int:
            first_started.set()
            if not release_first.wait(timeout):
                raise subprocess.TimeoutExpired(cmd="copy", timeout=timeout)
            return 0

        def kill(self) -> None:
            raise AssertionError("healthy copy must not be killed")

    def create_copy(*_args: object, **_kwargs: object) -> ActiveCopy:
        nonlocal created
        created += 1
        return ActiveCopy()

    def first_copy() -> None:
        fullraw_index._copy2_with_timeout(
            Path("first.sqlite"),
            Path("first.local.sqlite"),
            timeout_seconds=10,
        )

    def waiting_copy() -> None:
        try:
            fullraw_index._copy2_with_timeout(
                Path("waiting.sqlite"),
                Path("waiting.local.sqlite"),
                timeout_seconds=10,
                cancel_event=cancel_waiter,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    monkeypatch.setattr(fullraw_index, "_SHARD_LOCAL_CACHE_COPY_MAX_INFLIGHT", 1)
    monkeypatch.setattr(subprocess, "Popen", create_copy)
    first = threading.Thread(target=first_copy)
    waiter = threading.Thread(target=waiting_copy)
    first.start()
    assert first_started.wait(1)
    waiter.start()
    for _ in range(20):
        if fullraw_index._shard_local_cache_health()["copy_waiting"] == 1:
            break
        time.sleep(0.01)
    else:  # pragma: no cover - defensive test guard
        raise AssertionError("copy waiter did not block")

    cancel_waiter.set()
    waiter.join(1)
    assert len(errors) == 1
    assert isinstance(errors[0], CancelledError)
    assert created == 1
    assert fullraw_index._shard_local_cache_health()["copy_inflight"] == 1
    assert fullraw_index._shard_local_cache_health()["copy_waiting"] == 0

    release_first.set()
    first.join(1)
    assert fullraw_index._shard_local_cache_health()["copy_inflight"] == 0


def test_shard_cache_copy_waiters_start_fifo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_first = threading.Event()
    first_started = threading.Event()
    start_order: list[str] = []
    errors: list[BaseException] = []

    class CopyAttempt:
        def __init__(self, source: str) -> None:
            self.source = source

        def wait(self, *, timeout: float) -> int:
            if self.source == "first.sqlite":
                first_started.set()
                assert release_first.wait(timeout)
            return 0

        def kill(self) -> None:
            raise AssertionError("healthy FIFO copy must not be killed")

    def create_copy(*args: object, **_kwargs: object) -> CopyAttempt:
        command = args[0]
        assert isinstance(command, list)
        source = Path(str(command[-2])).name
        start_order.append(source)
        return CopyAttempt(source)

    def copy(source: str) -> None:
        try:
            fullraw_index._copy2_with_timeout(
                Path(source),
                Path(f"{source}.local"),
                timeout_seconds=10,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    monkeypatch.setattr(fullraw_index, "_SHARD_LOCAL_CACHE_COPY_MAX_INFLIGHT", 1)
    monkeypatch.setattr(subprocess, "Popen", create_copy)
    first = threading.Thread(target=copy, args=("first.sqlite",))
    older = threading.Thread(target=copy, args=("older.sqlite",))
    newer = threading.Thread(target=copy, args=("newer.sqlite",))
    first.start()
    assert first_started.wait(1)
    older.start()
    for _ in range(50):
        if fullraw_index._shard_local_cache_health()["copy_waiting"] == 1:
            break
        time.sleep(0.01)
    newer.start()
    for _ in range(50):
        if fullraw_index._shard_local_cache_health()["copy_waiting"] == 2:
            break
        time.sleep(0.01)
    release_first.set()
    for thread in (first, older, newer):
        thread.join(1)

    assert errors == []
    assert start_order == ["first.sqlite", "older.sqlite", "newer.sqlite"]
    assert fullraw_index._shard_local_cache_health()["copy_waiting"] == 0


def test_materialized_shard_waiter_cancels_without_touching_owner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    remote = tmp_path / "remote" / "fullraw_shard_0001.sqlite"
    remote.parent.mkdir()
    remote.write_bytes(b"remote shard")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES", "1024")
    cache_path = fullraw_index._shard_cache_path(remote)
    assert cache_path is not None
    cancel_event = threading.Event()
    errors: list[BaseException] = []

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("cancelled same-path waiter must not copy")

    def wait_for_owner() -> None:
        try:
            fullraw_index._materialized_shard_path(
                remote,
                populate=True,
                cancel_event=cancel_event,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    monkeypatch.setattr(fullraw_index, "_copy2_with_timeout", fail_copy)
    with fullraw_index._SHARD_LOCAL_CACHE_LOCK:
        fullraw_index._SHARD_LOCAL_CACHE_IN_PROGRESS.add(cache_path)
    waiter = threading.Thread(target=wait_for_owner)
    try:
        waiter.start()
        time.sleep(0.05)
        assert waiter.is_alive()
        cancel_event.set()
        waiter.join(1)
        assert not waiter.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], CancelledError)
        with fullraw_index._SHARD_LOCAL_CACHE_LOCK:
            assert cache_path in fullraw_index._SHARD_LOCAL_CACHE_IN_PROGRESS
            assert fullraw_index._SHARD_LOCAL_CACHE_RESERVED_BYTES == {}
        assert fullraw_index._shard_local_cache_health()["copy_inflight"] == 0
    finally:
        with fullraw_index._SHARD_LOCAL_CACHE_LOCK:
            fullraw_index._SHARD_LOCAL_CACHE_IN_PROGRESS.discard(cache_path)


def test_materialized_shard_path_does_not_copy_missing_cache_without_populate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    remote = tmp_path / "remote" / "fullraw_shard_0001.sqlite"
    remote.parent.mkdir()
    remote.write_text("remote shard", encoding="utf-8")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_DIR", str(cache_dir))

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("search hot path should not copy shard files")

    monkeypatch.setattr("v5_memo.fullraw_index.shutil.copy2", fail_copy)

    assert fullraw_index._materialized_shard_path(remote) == remote
    assert fullraw_index._cached_materialized_shard_path(remote) is None


def test_materialized_shard_path_skips_populate_when_cache_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cached = cache_dir / "old.sqlite"
    cached.write_bytes(b"old cache")
    remote = tmp_path / "remote" / "fullraw_shard_0001.sqlite"
    remote.parent.mkdir()
    remote.write_text("remote shard", encoding="utf-8")
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MAX_BYTES", "auto")
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MIN_FREE_BYTES", "500")
    monkeypatch.setattr(
        "v5_memo.fullraw_index.shutil.disk_usage",
        lambda _path: _FakeDiskUsage(total=1000, used=800, free=200),
    )

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("cache budget is exhausted; should search remote shard directly")

    monkeypatch.setattr("v5_memo.fullraw_index.shutil.copy2", fail_copy)

    assert fullraw_index._materialized_shard_path(remote, populate=True) == remote
    assert fullraw_index._cached_materialized_shard_path(remote) is None
    assert not cached.exists()


def test_materialized_shard_path_evicts_for_existing_reservations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    remote = tmp_path / "remote" / "fullraw_shard_0001.sqlite"
    remote.parent.mkdir()
    remote.write_bytes(b"data")
    cache_dir.mkdir()
    old_cache = cache_dir / "old.sqlite"
    old_cache.write_bytes(b"xxxx")
    reserved = cache_dir / ".reserved.sqlite.tmp.1.1"
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MAX_BYTES", "10")
    with fullraw_index._SHARD_LOCAL_CACHE_LOCK:
        fullraw_index._SHARD_LOCAL_CACHE_IN_PROGRESS.clear()
        fullraw_index._SHARD_LOCAL_CACHE_RESERVED_BYTES.clear()
        fullraw_index._SHARD_LOCAL_CACHE_IN_PROGRESS.add(reserved)
        fullraw_index._SHARD_LOCAL_CACHE_RESERVED_BYTES[reserved] = 4
    try:
        materialized = fullraw_index._materialized_shard_path(remote, populate=True)
    finally:
        with fullraw_index._SHARD_LOCAL_CACHE_LOCK:
            fullraw_index._SHARD_LOCAL_CACHE_IN_PROGRESS.clear()
            fullraw_index._SHARD_LOCAL_CACHE_RESERVED_BYTES.clear()

    assert materialized != remote
    assert materialized.exists()
    assert not old_cache.exists()


def test_materialized_shard_path_avoids_remote_stat_when_cache_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    remote = tmp_path / "remote" / "fullraw_shard_0001.sqlite"
    remote.parent.mkdir()
    remote.write_text("remote shard", encoding="utf-8")
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MAX_BYTES", "auto")
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MIN_FREE_BYTES", "500")
    monkeypatch.setattr(
        "v5_memo.fullraw_index.shutil.disk_usage",
        lambda _path: _FakeDiskUsage(total=1000, used=800, free=200),
    )
    original_stat = Path.stat

    def guarded_stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if path == remote:
            raise AssertionError("remote stat should be skipped when cache budget is exhausted")
        return original_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", guarded_stat)

    assert fullraw_index._materialized_shard_path(remote, populate=True) == remote


def test_materialized_shard_path_populates_when_worker_cache_cap_exceeded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    remote = tmp_path / "remote" / "fullraw_shard_0001.sqlite"
    remote.parent.mkdir()
    remote.write_bytes(b"large")
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MAX_BYTES", str(1024))
    monkeypatch.setenv("RESEARKA_FULLRAW_SWEEP_WORKER_CACHE_BYTES", "2")

    materialized = fullraw_index._materialized_shard_path(remote, populate=True)

    assert materialized != remote
    assert materialized.exists()
    assert materialized.read_bytes() == b"large"
    assert fullraw_index._cached_materialized_shard_path(remote) == materialized


def test_materialized_shard_path_skips_populate_when_available_cache_too_small(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    remote = tmp_path / "remote" / "fullraw_shard_0001.sqlite"
    remote.parent.mkdir()
    remote.write_bytes(b"large")
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MAX_BYTES", "2")
    monkeypatch.delenv("RESEARKA_FULLRAW_SWEEP_WORKER_CACHE_BYTES", raising=False)
    monkeypatch.delenv("RESEARKA_FULLRAW_SWEEP_WORKER_CACHE_GB", raising=False)

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("shard larger than cache budget should be searched remotely")

    monkeypatch.setattr("v5_memo.fullraw_index.shutil.copy2", fail_copy)

    assert fullraw_index._materialized_shard_path(remote, populate=True) == remote


def test_materialized_shard_path_populates_without_worker_cache_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    remote = tmp_path / "remote" / "fullraw_shard_0001.sqlite"
    remote.parent.mkdir()
    remote.write_bytes(b"remote shard")
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MAX_BYTES", raising=False)
    monkeypatch.delenv("RESEARKA_FULLRAW_SWEEP_WORKER_CACHE_BYTES", raising=False)
    monkeypatch.delenv("RESEARKA_FULLRAW_SWEEP_WORKER_CACHE_GB", raising=False)

    materialized = fullraw_index._materialized_shard_path(remote, populate=True)

    assert materialized != remote
    assert materialized.exists()
    assert materialized.read_bytes() == b"remote shard"


def test_materialized_shard_path_serializes_same_target_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    remote = tmp_path / "remote" / "fullraw_shard_0001.sqlite"
    remote.parent.mkdir()
    remote.write_bytes(b"remote shard")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_DIR", str(cache_dir))
    with fullraw_index._SHARD_LOCAL_CACHE_LOCK:
        fullraw_index._SHARD_LOCAL_CACHE_IN_PROGRESS.clear()

    copy_entered = threading.Event()
    release_copy = threading.Event()
    copied: list[Path] = []
    errors: list[BaseException] = []
    results: list[Path] = []

    def slow_copy(source: Path, target: Path, **_kwargs: object) -> None:
        copied.append(target)
        copy_entered.set()
        assert release_copy.wait(timeout=2), "copy was not released"
        target.write_bytes(source.read_bytes())

    def materialize() -> None:
        try:
            results.append(fullraw_index._materialized_shard_path(remote, populate=True))
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    monkeypatch.setattr(fullraw_index, "_copy2_with_timeout", slow_copy)

    first = threading.Thread(target=materialize)
    second = threading.Thread(target=materialize)
    first.start()
    assert copy_entered.wait(timeout=2)
    second.start()
    time.sleep(0.05)
    assert len(copied) == 1
    release_copy.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert len(copied) == 1
    assert len(results) == 2
    assert results[0] == results[1]
    assert results[0].read_bytes() == b"remote shard"


def test_materialized_shard_path_respects_active_copy_reservations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache"
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    first_remote = remote_dir / "fullraw_shard_0001.sqlite"
    second_remote = remote_dir / "fullraw_shard_0002.sqlite"
    first_remote.write_bytes(b"a" * 8)
    second_remote.write_bytes(b"b" * 8)
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES", "10")
    monkeypatch.delenv("V5_MEMO_FULL_RAW_SWEEP_WORKER_CACHE_BYTES", raising=False)
    monkeypatch.delenv("RESEARKA_FULLRAW_SWEEP_WORKER_CACHE_BYTES", raising=False)
    with fullraw_index._SHARD_LOCAL_CACHE_LOCK:
        fullraw_index._SHARD_LOCAL_CACHE_IN_PROGRESS.clear()
        fullraw_index._SHARD_LOCAL_CACHE_RESERVED_BYTES.clear()

    copy_entered = threading.Event()
    release_copy = threading.Event()
    copied: list[Path] = []
    errors: list[BaseException] = []
    results: list[Path] = []

    def slow_copy(source: Path, target: Path, **_kwargs: object) -> None:
        copied.append(source)
        if source == second_remote:
            raise AssertionError("second shard should not be copied past the cache cap")
        copy_entered.set()
        assert release_copy.wait(timeout=2), "copy was not released"
        target.write_bytes(source.read_bytes())

    def materialize_first() -> None:
        try:
            results.append(fullraw_index._materialized_shard_path(first_remote, populate=True))
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    monkeypatch.setattr(fullraw_index, "_copy2_with_timeout", slow_copy)

    first = threading.Thread(target=materialize_first)
    first.start()
    assert copy_entered.wait(timeout=2)

    assert fullraw_index._materialized_shard_path(second_remote, populate=True) == second_remote

    release_copy.set()
    first.join(timeout=2)

    assert not first.is_alive()
    assert errors == []
    assert copied == [first_remote]
    assert len(results) == 1
    assert results[0] != first_remote
    assert results[0].read_bytes() == b"a" * 8
    with fullraw_index._SHARD_LOCAL_CACHE_LOCK:
        assert fullraw_index._SHARD_LOCAL_CACHE_RESERVED_BYTES == {}


def test_shard_catalog_cache_round_trips_entries(tmp_path: Path) -> None:
    entry = ShardCatalogEntry(
        path=tmp_path / "batch_00001" / "fullraw_shard_0000.sqlite",
        batch_id=1,
        shard_id=0,
        sources=("openalex", "pubmed"),
        files_completed=3,
        papers_inserted=42,
        bytes_used=1024,
        year_min=1999,
        year_max=2025,
        cited_by_min=0,
        cited_by_max=100,
        cited_by_avg=12.5,
        topic_terms=("metformin", "longevity"),
    )
    cache_path = tmp_path / "catalog.json"

    fullraw_index.write_shard_catalog_cache(cache_path, [entry])
    loaded = fullraw_index.load_shard_catalog_cache(cache_path)

    assert loaded == [entry]


def test_shard_catalog_cache_rejects_entries_outside_current_root(tmp_path: Path) -> None:
    shared_root = tmp_path / "shared"
    isolated_root = tmp_path / "isolated"
    shared_entry = ShardCatalogEntry(
        path=shared_root / "batch_00001" / "fullraw_shard_0000.sqlite",
        batch_id=1,
        shard_id=0,
        sources=("openalex",),
        files_completed=1,
        papers_inserted=10,
        bytes_used=1024,
    )
    isolated_entry = replace(shared_entry, path=isolated_root / "batch_00001" / "fullraw_shard_0000.sqlite")

    assert fullraw_index._catalog_entries_match_shard_dir([isolated_entry], isolated_root)
    assert not fullraw_index._catalog_entries_match_shard_dir([shared_entry], isolated_root)


def test_shard_catalog_cache_remaps_entries_to_current_root(tmp_path: Path) -> None:
    shared_root = tmp_path / "shared"
    isolated_root = tmp_path / "isolated"
    shared_entry = ShardCatalogEntry(
        path=shared_root / "batch_00001" / "fullraw_shard_0002.sqlite",
        batch_id=1,
        shard_id=2,
        sources=("openalex",),
        files_completed=3,
        papers_inserted=42,
        bytes_used=2048,
        year_min=2001,
        year_max=2024,
        cited_by_min=1,
        cited_by_max=50,
        cited_by_avg=8.5,
        topic_terms=("resveratrol",),
    )

    remapped = fullraw_index._remap_catalog_entries_to_shard_dir([shared_entry], isolated_root)

    assert remapped == [
        replace(shared_entry, path=isolated_root / "batch_00001" / "fullraw_shard_0002.sqlite")
    ]


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
    fullraw_index._materialized_shard_path(huge.path, populate=True)
    fullraw_index._materialized_shard_path(cached.path, populate=True)
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
    assert [item.role for item in passes] == ["focused", "citation_heavy", "recency"]
    assert all("risk" not in item.query and "patients" not in item.query for item in passes)
    assert fullraw_index._sweep_search_passes("resveratrol blunts exercise training", entries, rank_mode="relevance")[0].query == "resveratrol blunts training"
    metformin_adaptation_query = fullraw_index._sweep_search_passes("metformin resistance training adaptation", entries, rank_mode="relevance")[0].query
    assert "resistance" in metformin_adaptation_query
    assert metformin_adaptation_query != "metformin training adaptation"
    metformin_query = fullraw_index._sweep_search_passes("metformin blunts muscle hypertrophy progressive resistance training", entries, rank_mode="relevance")[0].query
    assert metformin_query.startswith("metformin ")
    assert "blunts" in metformin_query
    cwi_entries = [
        ShardCatalogEntry(
            path=(tmp_path / f"cwi_{idx}.sqlite"),
            batch_id=idx,
            shard_id=0,
            sources=("openalex",),
            files_completed=1,
            papers_inserted=10,
            bytes_used=6,
            topic_terms=("water", "resistance", "training"),
        )
        for idx in range(4)
    ]
    cwi_passes = fullraw_index._sweep_search_passes("cold water immersion resistance training", cwi_entries, rank_mode="relevance")
    assert {item.query for item in cwi_passes} == {"cold immersion training"}
    assert all(item.query != "water resistance training" for item in cwi_passes)
    protein_entries = [
        ShardCatalogEntry(
            path=(tmp_path / f"protein_{idx}.sqlite"),
            batch_id=idx,
            shard_id=0,
            sources=("openalex",),
            files_completed=1,
            papers_inserted=10,
            bytes_used=6,
            topic_terms=terms,
        )
        for idx, terms in enumerate((
            ("protein", "distribution", "synthesis"),
            ("timing", "muscle", "protein"),
        ))
    ]
    protein_passes = fullraw_index._sweep_search_passes("protein timing distribution muscle synthesis", protein_entries, rank_mode="relevance")
    assert {item.query for item in protein_passes} == {"protein timing muscle"}

def test_materialized_shard_cache_evicts_old_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    old, newer, keep = (cache_dir / name for name in (".old.sqlite.tmp.1.1", "newer.sqlite", "keep.sqlite"))
    for path in (old, newer, keep):
        path.write_bytes(b"x" * 6)
    for path, stamp in ((old, 1), (newer, 2), (keep, 3)):
        os.utime(path, (stamp, stamp))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES", "12")
    fullraw_index._evict_shard_cache(cache_dir, required_bytes=0, keep=keep)
    assert not old.exists()
    assert newer.exists()
    assert keep.exists()


def test_materialized_shard_cache_preserves_fresh_temp_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    fresh_tmp = cache_dir / ".fresh.sqlite.tmp.1.1"
    stale_tmp = cache_dir / ".stale.sqlite.tmp.1.1"
    keep = cache_dir / "keep.sqlite"
    for path in (fresh_tmp, stale_tmp, keep):
        path.write_bytes(b"x" * 6)
    now = time.time()
    os.utime(fresh_tmp, (now, now))
    os.utime(stale_tmp, (1, 1))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES", "6")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_TMP_TTL_SECONDS", "3600")

    fullraw_index._evict_shard_cache(cache_dir, required_bytes=0, keep=keep)

    assert fresh_tmp.exists()
    assert not stale_tmp.exists()
    assert keep.exists()


def test_materialized_shard_cache_evicts_dead_pid_temp_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    dead_tmp = cache_dir / ".dead.sqlite.tmp.999999.1"
    keep = cache_dir / "keep.sqlite"
    for path in (dead_tmp, keep):
        path.write_bytes(b"x" * 6)
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES", "100")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_TMP_TTL_SECONDS", "3600")

    def fake_kill(pid: int, sig: int) -> None:
        del sig
        if pid == 999999:
            raise ProcessLookupError

    monkeypatch.setattr(os, "kill", fake_kill)

    fullraw_index._evict_shard_cache(cache_dir, required_bytes=0, keep=keep)

    assert not dead_tmp.exists()
    assert keep.exists()


def test_materialized_shard_cache_preserves_live_pid_temp_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    live_tmp = cache_dir / ".live.sqlite.tmp.123.1"
    keep = cache_dir / "keep.sqlite"
    for path in (live_tmp, keep):
        path.write_bytes(b"x" * 6)
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES", "6")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_TMP_TTL_SECONDS", "3600")
    monkeypatch.setattr(os, "kill", lambda pid, sig: None)

    fullraw_index._evict_shard_cache(cache_dir, required_bytes=0, keep=keep)

    assert live_tmp.exists()
    assert keep.exists()


def test_auto_sweep_workers_scales_by_inflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    monkeypatch.delenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES", raising=False)
    monkeypatch.delenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MAX_BYTES", raising=False)
    monkeypatch.delenv("V5_MEMO_FULL_RAW_SWEEP_WORKER_CACHE_BYTES", raising=False)
    monkeypatch.delenv("RESEARKA_FULLRAW_SWEEP_WORKER_CACHE_BYTES", raising=False)
    monkeypatch.delenv("V5_MEMO_FULL_RAW_SWEEP_WORKER_CACHE_GB", raising=False)
    monkeypatch.delenv("RESEARKA_FULLRAW_SWEEP_WORKER_CACHE_GB", raising=False)

    assert fullraw_index._auto_sweep_workers(1) == 16
    assert fullraw_index._auto_sweep_workers(2) == 8
    assert fullraw_index._auto_sweep_workers(0) == 16


def test_auto_sweep_inflight_preserves_cache_worker_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1024 * 1024 * 1024
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MAX_BYTES", str(31 * gib))
    monkeypatch.setenv("RESEARKA_FULLRAW_SWEEP_WORKER_CACHE_BYTES", str(8 * gib))
    monkeypatch.setenv("RESEARKA_FULLRAW_SWEEP_PRIORITY_BURST", "0")

    assert fullraw_index._auto_sweep_max_inflight() == 3
    assert fullraw_index._auto_sweep_workers(3) == 1

    monkeypatch.setenv("RESEARKA_FULLRAW_SWEEP_PRIORITY_BURST", "1")

    assert fullraw_index._auto_sweep_max_inflight() == 2


def test_configured_sweep_inflight_honors_auto_and_explicit_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARKA_FULLRAW_SWEEP_MAX_INFLIGHT", "auto")
    monkeypatch.setattr(fullraw_index, "_auto_sweep_max_inflight", lambda: 3)

    assert fullraw_index._configured_sweep_max_inflight() == 3

    monkeypatch.setenv("RESEARKA_FULLRAW_SWEEP_MAX_INFLIGHT", "2")

    assert fullraw_index._configured_sweep_max_inflight() == 2


def test_auto_sweep_workers_ignores_cache_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MAX_BYTES", str(8 * 1024 * 1024 * 1024))
    monkeypatch.delenv("RESEARKA_FULLRAW_SWEEP_WORKER_CACHE_BYTES", raising=False)
    monkeypatch.delenv("RESEARKA_FULLRAW_SWEEP_WORKER_CACHE_GB", raising=False)

    assert fullraw_index._auto_sweep_workers(2) == 8


def test_auto_sweep_workers_respects_cache_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MAX_BYTES", str(8 * 1024 * 1024 * 1024))
    monkeypatch.setenv("RESEARKA_FULLRAW_SWEEP_WORKER_CACHE_GB", "1")
    monkeypatch.setenv("RESEARKA_FULLRAW_SWEEP_PRIORITY_BURST", "1")

    assert fullraw_index._auto_sweep_workers(2) == 2

    monkeypatch.setenv("RESEARKA_FULLRAW_SWEEP_PRIORITY_BURST", "0")

    assert fullraw_index._auto_sweep_workers(2) == 4


def test_auto_sweep_workers_uses_cpu_workers_when_cache_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MAX_BYTES", "auto")
    monkeypatch.setenv("RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MIN_FREE_BYTES", "500")
    monkeypatch.setattr(
        "v5_memo.fullraw_index.shutil.disk_usage",
        lambda _path: _FakeDiskUsage(total=1000, used=800, free=200),
    )

    assert fullraw_index._auto_sweep_workers(2) == 8


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
