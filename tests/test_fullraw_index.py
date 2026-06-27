import gzip
import json
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
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
        stats = index.stats(files_total=1)
    finally:
        index.close()
    assert (result.files_completed, result.papers_inserted, stats.papers_indexed) == (1, 2, 2)
    assert hits[0]["doi"] == "10.example/guidance"


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


def test_isolated_shard_search_kills_timed_out_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        returncode = None
        killed = False
        waited = False

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)

        def kill(self) -> None:
            self.killed = True

        def wait(self, *, timeout: float) -> None:
            del timeout
            self.waited = True
            raise subprocess.TimeoutExpired(cmd="fake", timeout=1)

    fake = FakeProcess()
    monkeypatch.setattr("v5_memo.fullraw_index.subprocess.Popen", lambda *args, **kwargs: fake)

    with pytest.raises(TimeoutError):
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
    assert fake.waited is True


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


def test_sweep_cache_key_ignores_result_limit() -> None:
    first = fullraw_index._sweep_cache_key(
        "cold water immersion",
        limit=10,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        sweep_shard_limit=1525,
    )
    second = fullraw_index._sweep_cache_key(
        "cold water immersion",
        limit=25,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        sweep_shard_limit=1525,
    )

    assert first == second


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


def test_cache_only_completed_sweep_hit_does_not_aggregate_remote_stats(
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
    key = fullraw_index._sweep_cache_key(
        "metformin longevity",
        limit=10,
        year_min=1900,
        year_max=2100,
        rank_mode="relevance",
        sweep_shard_limit=2,
        sweep_strategy=fullraw_index._SWEEP_STRATEGY,
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
                "sources_searched": {"openalex": 1, "pubmed": 1},
                "sweep_search_passes": (
                    {"role": "focused"},
                    {"role": "citation_heavy"},
                    {"role": "recency"},
                ),
                "sweep_completed_pass_roles": ("focused", "citation_heavy", "recency"),
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
            "query": "metformin longevity",
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


def test_repolled_queued_sweep_job_gets_next_lane_without_reordering_rest() -> None:
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

    assert next_job == target_new
    assert list(queued_jobs) == ["older", "later"]


def test_priority_sweep_job_gets_next_lane_before_background_queue() -> None:
    older = fullraw_index.SweepJob("older", "older query", 10, 1900, 2100, "relevance", [])
    target = fullraw_index.SweepJob("target", "target query", 10, 1900, 2100, "relevance", [])
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


def test_complete_sweep_retries_failed_shards() -> None:
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


def test_shard_search_materializes_before_isolated_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = tmp_path / "remote.sqlite"
    materialized = tmp_path / "local.sqlite"
    searched: list[Path] = []

    def fake_materialized(path: Path, *, preserve: set[Path] | None = None) -> Path:
        assert path == original
        assert preserve == set()
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
    preserve_seen_for_b: set[Path] | None = None

    def fake_materialized(path: Path, *, preserve: set[Path] | None = None) -> Path:
        nonlocal preserve_seen_for_b
        preserve = preserve or set()
        if path == remote_a:
            live_cache_paths.add(local_a)
            return local_a
        if path == remote_b:
            preserve_seen_for_b = set(preserve)
            if local_a not in preserve:
                live_cache_paths.discard(local_a)
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

    assert preserve_seen_for_b == {local_a}
    assert completed_paths == [remote_a, remote_b]
    assert {hit["title"] for hit in hits} == {"local-a", "local-b"}
    assert timed_out is False


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
    assert [item.role for item in passes] == ["focused", "citation_heavy", "recency"]
    assert all("risk" not in item.query and "patients" not in item.query for item in passes)
    assert fullraw_index._sweep_search_passes("metformin blunts muscle hypertrophy progressive resistance training", entries, rank_mode="relevance")[0].query.startswith("metformin ")
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
