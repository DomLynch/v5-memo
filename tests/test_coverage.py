from __future__ import annotations

import json
from typing import Any, cast

import pytest
from pytest import MonkeyPatch

from v5_memo.coverage import (
    current_search_coverage,
    full_raw_corpus_readiness,
    require_full_raw_corpus,
    require_full_raw_corpus_ready,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_coverage_does_not_claim_raw_450m_without_explicit_service(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", raising=False)

    coverage = current_search_coverage()

    assert coverage.openalex_full_corpus_api is True
    assert coverage.full_raw_local_corpus is False
    assert "not configured/searchable" in coverage.summary


def test_require_full_raw_corpus_fails_closed(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", raising=False)

    with pytest.raises(RuntimeError, match="Full local raw 450M"):
        require_full_raw_corpus()


def test_require_full_raw_corpus_accepts_explicit_service(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9999/search")

    require_full_raw_corpus()


def test_coverage_reports_configured_full_raw_endpoint(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9999/search")

    coverage = current_search_coverage()

    assert coverage.full_raw_local_corpus is True
    assert "configured through http://127.0.0.1:9999/search" in coverage.summary


def test_full_raw_readiness_fails_until_index_is_complete(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9902/search")

    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        del timeout
        url = cast(Any, request).full_url
        if url.endswith("/health"):
            return _FakeResponse({
                "ok": True,
                "complete": False,
                "files_indexed": 2348,
                "files_total": 3917,
            })
        return _FakeResponse({
            "meta": {
                "count": 8,
                "shard_receipt": {
                    "shards_searched": 1129,
                    "shards_total": 1129,
                    "sources_searched": {
                        "openalex": 988,
                        "pubmed": 1,
                        "semantic_scholar": 148,
                    },
                    "sweep_remaining_shards": 0,
                    "sweep_failed_shards": 0,
                },
            },
            "results": [{}],
        })

    monkeypatch.setattr("v5_memo.coverage.urlopen", fake_urlopen)

    readiness = full_raw_corpus_readiness(min_shards_searched=50, min_sources_searched=2)

    assert readiness.ready is False
    assert "full raw index incomplete" in readiness.reason
    assert "files indexed 2348/3917" in readiness.reason
    assert "files=2348/3917" in readiness.summary


def test_full_raw_readiness_accepts_complete_exhaustive_probe(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9902/search")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_TOKEN", "secret")
    seen_authorization: list[str] = []

    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        del timeout
        req = cast(Any, request)
        seen_authorization.append(req.headers.get("Authorization", ""))
        if req.full_url.endswith("/health"):
            return _FakeResponse({
                "ok": True,
                "complete": True,
                "files_indexed": 3917,
                "files_total": 3917,
            })
        return _FakeResponse({
            "meta": {
                "count": 8,
                "shard_receipt": {
                    "shards_searched": 1129,
                    "shards_total": 1129,
                    "sources_searched": {
                        "openalex": 988,
                        "pubmed": 1,
                        "semantic_scholar": 148,
                    },
                    "sweep_remaining_shards": 0,
                    "sweep_failed_shards": 0,
                },
            },
            "results": [{}],
        })

    monkeypatch.setattr("v5_memo.coverage.urlopen", fake_urlopen)

    readiness = full_raw_corpus_readiness(min_shards_searched=50, min_sources_searched=2)

    assert readiness.ready is True
    assert readiness.reason == "ready"
    assert readiness.probe_receipt["shards_searched"] == 1129
    assert seen_authorization == ["Bearer secret", "Bearer secret"]


def test_require_full_raw_ready_fails_closed_on_narrow_probe(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9902/search")

    def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
        del timeout
        if cast(Any, request).full_url.endswith("/health"):
            return _FakeResponse({
                "ok": True,
                "complete": True,
                "files_indexed": 3917,
                "files_total": 3917,
            })
        return _FakeResponse({
            "meta": {
                "count": 8,
                "shard_receipt": {
                    "shards_searched": 12,
                    "shards_total": 1129,
                    "sources_searched": {"openalex": 12},
                },
            },
            "results": [{}],
        })

    monkeypatch.setattr("v5_memo.coverage.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="probe searched 12/50 required shards"):
        require_full_raw_corpus_ready(min_shards_searched=50, min_sources_searched=2)
