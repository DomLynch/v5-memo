import pytest
from pytest import MonkeyPatch

from v5_memo.coverage import current_search_coverage, require_full_raw_corpus


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
