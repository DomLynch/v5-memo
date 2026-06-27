from collections.abc import Callable

import pytest
from pytest import MonkeyPatch

from v5_memo.coverage import (
    current_search_coverage,
    full_raw_search_health,
    require_full_raw_corpus,
)


class FakeResponse:
    def __init__(self, body: str) -> None:
        self._body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body.encode("utf-8")


def _health_body(*, complete: bool = True) -> str:
    return (
        '{"ok": true, "backend": "v5-fullraw-fts", "papers_indexed": 123, '
        f'"files_indexed": {4 if complete else 3}, "files_total": 4, '
        f'"complete": {str(complete).lower()}, '
        '"shard_receipt": {"partial_shard_search": false, "sweep_failed_shards": 0, '
        '"shards_searched": 1525, "shards_total": 1525, '
        '"sources_searched": {"openalex": 1, "pubmed": 1, "semantic_scholar": 1, '
        '"semantic_scholar_abstracts": 1, "biorxiv": 1}}}'
    )


def _search_body(*, partial: bool = False, hits: int = 1) -> str:
    return (
        '{"meta": {"count": '
        f"{hits}, "
        '"shard_receipt": {"partial_shard_search": '
        f"{str(partial).lower()}, "
        '"sweep_failed_shards": 0, "shards_searched": 1525, "shards_total": 1525, '
        '"sources_searched": {"openalex": 1, "pubmed": 1, "semantic_scholar": 1, '
        '"semantic_scholar_abstracts": 1, "biorxiv": 1}}}, '
        '"results": [{"title": "Metformin longevity"}]}'
    )


def _fake_fullraw_urlopen(
    *,
    health: str | None = None,
    search: str | None = None,
) -> Callable[[object, float], FakeResponse]:
    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        del timeout
        data = getattr(request, "data", None)
        body = (search or _search_body()) if data else (health or _health_body())
        return FakeResponse(body)

    return fake_urlopen


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


def test_require_full_raw_corpus_accepts_complete_explicit_service(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9999/search")
    monkeypatch.setattr(
        "v5_memo.coverage.urlopen",
        _fake_fullraw_urlopen(search=_search_body()),
    )

    require_full_raw_corpus()


def test_require_full_raw_corpus_rejects_incomplete_service(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9999/search")
    monkeypatch.setattr(
        "v5_memo.coverage.urlopen",
        _fake_fullraw_urlopen(health=_health_body(complete=False), search=_search_body()),
    )

    with pytest.raises(RuntimeError, match="Full local raw 450M"):
        require_full_raw_corpus()


def test_require_full_raw_corpus_accepts_strict_sweep_service(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9999/search")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED", "1525")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED", "5")
    health = (
        '{"ok": true, "backend": "researka-fullraw-indexed-fts5", "fast_health": true, '
        '"complete": false, "coverage_requirements": {"min_shards_searched": 1525, '
        '"min_sources_searched": 5, "require_complete_search": 1, "sweep_require_complete": 1}}'
    )
    monkeypatch.setattr(
        "v5_memo.coverage.urlopen",
        _fake_fullraw_urlopen(health=health, search=_search_body(partial=True)),
    )

    require_full_raw_corpus()


def test_coverage_reports_configured_full_raw_endpoint(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9999/search")
    monkeypatch.setattr(
        "v5_memo.coverage.urlopen",
        _fake_fullraw_urlopen(search=_search_body()),
    )

    coverage = current_search_coverage()

    assert coverage.full_raw_local_corpus is True
    assert "healthy v5-fullraw-fts at http://127.0.0.1:9999/search" in coverage.summary


def test_coverage_rejects_env_only_fullraw_endpoint(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9999/search")

    coverage = current_search_coverage()

    assert coverage.full_raw_local_corpus is False


def test_fullraw_health_uses_search_service_health_endpoint(monkeypatch: MonkeyPatch) -> None:
    seen: list[dict[str, object]] = []

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        seen.append({
            "url": request.full_url,  # type: ignore[attr-defined]
            "timeout": timeout,
            "has_body": bool(getattr(request, "data", None)),
            "authorization": request.get_header("Authorization"),  # type: ignore[attr-defined]
        })
        if getattr(request, "data", None):
            return FakeResponse(_search_body())
        return FakeResponse(_health_body())

    monkeypatch.setenv("V5_MEMO_FULL_RAW_HEALTH_TIMEOUT", "1.5")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_INDEX_TOKEN", "test-token")
    monkeypatch.setattr("v5_memo.coverage.urlopen", fake_urlopen)

    health = full_raw_search_health("http://127.0.0.1:9902/search")

    assert health.ok is True
    assert health.papers_indexed == 123
    assert health.complete is True
    assert health.partial_shard_search is False
    assert health.sweep_failed_shards == 0
    assert health.query_smoke_ok is True
    assert health.source_count == 5
    assert seen == [
        {
            "url": "http://127.0.0.1:9902/health",
            "timeout": 1.5,
            "has_body": False,
            "authorization": "Bearer test-token",
        },
        {
            "url": "http://127.0.0.1:9902/search",
            "timeout": 1.5,
            "has_body": True,
            "authorization": "Bearer test-token",
        },
    ]


def test_fullraw_health_rejects_static_health_without_complete_query_smoke(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "v5_memo.coverage.urlopen",
        _fake_fullraw_urlopen(search=_search_body(partial=True)),
    )

    health = full_raw_search_health("http://127.0.0.1:9902/search")

    assert health.ok is False
    assert health.complete is True
    assert health.query_smoke_ok is False
    assert health.partial_shard_search is True
    assert health.error == "query smoke incomplete or partial"
