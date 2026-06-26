"""Runtime search coverage reporting for V5."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class SearchCoverage:
    """What V5 can search in the current runtime."""

    openalex_full_corpus_api: bool
    researka_corpus_api: bool
    full_raw_local_corpus: bool
    summary: str


@dataclass(frozen=True, slots=True)
class SearchBackendHealth:
    """Runtime proof for an optional search backend."""

    configured: bool
    ok: bool
    url: str = ""
    backend: str = ""
    papers_indexed: int = 0
    files_indexed: int = 0
    files_total: int = 0
    complete: bool = False
    partial_shard_search: bool = False
    sweep_failed_shards: int = 0
    source_count: int = 0
    error: str = ""


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
    full_raw_health = full_raw_search_health(full_raw_url)
    full_raw = full_raw_health.ok
    researka = bool(researka_url and researka_token)
    summary = (
        "OpenAlex API: searchable full OpenAlex works corpus; "
        "Researka API: searchable when RESEARKA_DATABASE_URL plus token are set "
        "(verified VPS slice: 25,181,785 papers, 1,015,859 embeddings, "
        "24,814,247 Tantivy rows); "
        "local raw 450M+ corpus: "
        + (
            (
                f"healthy {full_raw_health.backend} at {full_raw_url} "
                f"({full_raw_health.papers_indexed:,} papers, "
                f"{full_raw_health.files_indexed}/{full_raw_health.files_total} files, "
                f"complete={full_raw_health.complete})"
            )
            if full_raw
            else "not configured/searchable by V5 yet"
        )
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
        health = full_raw_search_health()
        raise RuntimeError(
            "Full local raw 450M+ corpus search is not healthy. "
            "Set V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL to a real fullraw service with "
            f"/health ok=true. Current status: {health.error or 'not configured'}"
        )


def full_raw_search_health(url: str | None = None) -> SearchBackendHealth:
    """Probe the configured fullraw service instead of trusting env presence."""
    search_url = (url if url is not None else os.environ.get("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "")).strip()
    if not search_url:
        return SearchBackendHealth(configured=False, ok=False, error="missing V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL")
    health_url = _health_url(search_url)
    try:
        request = Request(health_url, headers={"User-Agent": "v5-memo/0.1"}, method="GET")
        with urlopen(request, timeout=_health_timeout()) as response:
            data: Any = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        return SearchBackendHealth(configured=True, ok=False, url=search_url, error=str(exc))
    if not isinstance(data, dict):
        return SearchBackendHealth(configured=True, ok=False, url=search_url, error="health response is not an object")
    papers_indexed = _int_value(data.get("papers_indexed"))
    files_indexed = _int_value(data.get("files_indexed"))
    files_total = _int_value(data.get("files_total"))
    backend = str(data.get("backend") or "")
    shard_receipt = data.get("shard_receipt")
    receipt = shard_receipt if isinstance(shard_receipt, dict) else {}
    sources = receipt.get("sources_searched")
    source_count = (
        sum(1 for value in sources.values() if _int_value(value))
        if isinstance(sources, dict)
        else 0
    )
    partial_shard_search = receipt.get("partial_shard_search") is True
    sweep_failed_shards = _int_value(receipt.get("sweep_failed_shards"))
    complete = data.get("complete") is True
    ok = bool(
        data.get("ok") is True
        and backend
        and papers_indexed > 0
        and complete
        and not partial_shard_search
        and sweep_failed_shards == 0
    )
    return SearchBackendHealth(
        configured=True,
        ok=ok,
        url=search_url,
        backend=backend,
        papers_indexed=papers_indexed,
        files_indexed=files_indexed,
        files_total=files_total,
        complete=complete,
        partial_shard_search=partial_shard_search,
        sweep_failed_shards=sweep_failed_shards,
        source_count=source_count,
        error="" if ok else "health incomplete, partial, failed, or missing ok/backend/papers_indexed",
    )


def _health_url(search_url: str) -> str:
    parsed = urlparse(search_url)
    return urlunparse(parsed._replace(path="/health", query="", fragment=""))


def _health_timeout() -> float:
    try:
        return max(0.1, min(float(os.environ.get("V5_MEMO_FULL_RAW_HEALTH_TIMEOUT", "3")), 30.0))
    except ValueError:
        return 3.0


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0
