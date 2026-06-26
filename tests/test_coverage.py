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
        lambda request, timeout: FakeResponse(
            '{"ok": true, "backend": "v5-fullraw-fts", "papers_indexed": 123, '
            '"files_indexed": 4, "files_total": 4, "complete": true, '
            '"shard_receipt": {"partial_shard_search": false, "sweep_failed_shards": 0}}'
        ),
    )

    require_full_raw_corpus()


def test_require_full_raw_corpus_rejects_incomplete_service(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9999/search")
    monkeypatch.setattr(
        "v5_memo.coverage.urlopen",
        lambda request, timeout: FakeResponse(
            '{"ok": true, "backend": "v5-fullraw-fts", "papers_indexed": 123, '
            '"files_indexed": 3, "files_total": 4, "complete": false}'
        ),
    )

    with pytest.raises(RuntimeError, match="Full local raw 450M"):
        require_full_raw_corpus()


def test_coverage_reports_configured_full_raw_endpoint(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9999/search")
    monkeypatch.setattr(
        "v5_memo.coverage.urlopen",
        lambda request, timeout: FakeResponse(
            '{"ok": true, "backend": "v5-fullraw-fts", "papers_indexed": 123, '
            '"files_indexed": 4, "files_total": 4, "complete": true, '
            '"shard_receipt": {"partial_shard_search": false, "sweep_failed_shards": 0}}'
        ),
    )

    coverage = current_search_coverage()

    assert coverage.full_raw_local_corpus is True
    assert "healthy v5-fullraw-fts at http://127.0.0.1:9999/search" in coverage.summary


def test_coverage_rejects_env_only_fullraw_endpoint(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9999/search")

    coverage = current_search_coverage()

    assert coverage.full_raw_local_corpus is False


def test_fullraw_health_uses_search_service_health_endpoint(monkeypatch: MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        seen["url"] = request.full_url  # type: ignore[attr-defined]
        seen["timeout"] = timeout
        return FakeResponse(
            '{"ok": true, "backend": "v5-fullraw-fts", "papers_indexed": 456, '
            '"files_indexed": 9, "files_total": 9, "complete": true, '
            '"shard_receipt": {"partial_shard_search": false, "sweep_failed_shards": 0, '
            '"sources_searched": {"openalex": 1, "pubmed": 1}}}'
        )

    monkeypatch.setenv("V5_MEMO_FULL_RAW_HEALTH_TIMEOUT", "1.5")
    monkeypatch.setattr("v5_memo.coverage.urlopen", fake_urlopen)

    health = full_raw_search_health("http://127.0.0.1:9902/search")

    assert health.ok is True
    assert health.papers_indexed == 456
    assert health.complete is True
    assert health.partial_shard_search is False
    assert health.sweep_failed_shards == 0
    assert health.source_count == 2
    assert seen == {"url": "http://127.0.0.1:9902/health", "timeout": 1.5}
