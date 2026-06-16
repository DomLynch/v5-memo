"""Runtime search coverage reporting for V5."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchCoverage:
    """What V5 can search in the current runtime."""

    openalex_full_corpus_api: bool
    researka_corpus_api: bool
    full_raw_local_corpus: bool
    summary: str


def current_search_coverage() -> SearchCoverage:
    """Return a conservative coverage statement.

    The raw 450M+ storage corpus is only treated as searchable when an explicit
    full-raw search service/index URL is configured.
    """
    full_raw_url = os.environ.get("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "").strip()
    researka_url = os.environ.get("RESEARKA_DATABASE_URL", "").strip()
    researka_token = (
        os.environ.get("RESEARKA_DATABASE_TOKEN", "")
        or os.environ.get("RESEARKA_TOKEN", "")
        or os.environ.get("RESEARKA_TOKENS", "")
    ).strip()
    full_raw = bool(full_raw_url)
    researka = bool(researka_url and researka_token)
    summary = (
        "OpenAlex API: searchable full OpenAlex works corpus; "
        "Researka API: searchable when RESEARKA_DATABASE_URL plus token are set "
        "(verified VPS slice: 25,181,785 papers, 1,015,859 embeddings, "
        "24,814,247 Tantivy rows); "
        "local raw 450M+ corpus: "
        + ("configured" if full_raw else "not configured/searchable by V5 yet")
    )
    return SearchCoverage(
        openalex_full_corpus_api=True,
        researka_corpus_api=researka,
        full_raw_local_corpus=full_raw,
        summary=summary,
    )


def require_full_raw_corpus() -> None:
    """Fail loudly if caller requires the full local raw 450M+ corpus."""
    coverage = current_search_coverage()
    if not coverage.full_raw_local_corpus:
        raise RuntimeError(
            "Full local raw 450M+ corpus search is not configured. "
            "Set V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL after building/exposing a real "
            "450M+ search index. Current V5 can use OpenAlex API and, when configured, "
            "the searchable Researka corpus slice."
        )
