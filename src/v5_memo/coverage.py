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
    hits: int = 0
    query_smoke_ok: bool = False
    query_smoke: str = ""
    shards_searched: int = 0
    shards_total: int = 0
    error: str = ""


def current_search_coverage() -> SearchCoverage:
    """Return a conservative coverage statement.

    The raw 450M+ storage corpus is only treated as searchable when an explicit
    full-raw search service/index URL is configured.
    """
    full_raw_url = _full_raw_search_url()
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
            "Set RESEARKA_FULLRAW_SEARCH_URL or V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL "
            "to a real fullraw service with "
            f"/health ok=true. Current status: {health.error or 'not configured'}"
        )


def full_raw_search_health(url: str | None = None) -> SearchBackendHealth:
    """Probe the configured fullraw service instead of trusting env presence."""
    search_url = (url if url is not None else _full_raw_search_url()).strip()
    if not search_url:
        return SearchBackendHealth(
            configured=False,
            ok=False,
            error="missing RESEARKA_FULLRAW_SEARCH_URL or V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL",
        )
    health_url = _health_url(search_url)
    try:
        request = Request(health_url, headers=_full_raw_headers(), method="GET")
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
    requirements = data.get("coverage_requirements")
    strict_sweep_ready = (
        isinstance(requirements, dict)
        and _int_value(requirements.get("min_shards_searched")) >= _full_raw_min_shards(0)
        and _int_value(requirements.get("min_sources_searched")) >= _full_raw_min_sources()
        and _int_value(requirements.get("require_complete_search")) == 1
        and _int_value(requirements.get("sweep_require_complete")) == 1
    )
    static_ok = bool(
        data.get("ok") is True
        and backend
        and ((papers_indexed > 0 and complete) or strict_sweep_ready)
        and not partial_shard_search
        and sweep_failed_shards == 0
    )
    smoke = _full_raw_query_smoke(search_url)
    raw_smoke_receipt = smoke.get("receipt")
    smoke_receipt: dict[str, object] = (
        raw_smoke_receipt if isinstance(raw_smoke_receipt, dict) else {}
    )
    smoke_sources = smoke_receipt.get("sources_searched")
    smoke_source_count = (
        sum(1 for value in smoke_sources.values() if _int_value(value))
        if isinstance(smoke_sources, dict)
        else 0
    )
    smoke_failed = _int_value(smoke_receipt.get("sweep_failed_shards"))
    smoke_partial = smoke_receipt.get("partial_shard_search") is True
    smoke_shards = _int_value(smoke_receipt.get("shards_searched"))
    smoke_total = _int_value(smoke_receipt.get("shards_total"))
    min_shards = _full_raw_min_shards(smoke_total)
    min_sources = _full_raw_min_sources()
    hits = _int_value(smoke.get("hits"))
    smoke_ok = bool(
        hits > 0
        and smoke_shards >= min_shards
        and not smoke_partial
        and smoke_failed == 0
        and smoke_source_count >= min_sources
    )
    ok = static_ok and (smoke_ok or strict_sweep_ready)
    error = ""
    if not static_ok:
        error = "health incomplete, partial, failed, or missing ok/backend/papers_indexed"
    elif not smoke_ok and not strict_sweep_ready:
        error = str(smoke.get("error") or "query smoke incomplete or partial")
    return SearchBackendHealth(
        configured=True,
        ok=ok,
        url=search_url,
        backend=backend,
        papers_indexed=papers_indexed,
        files_indexed=files_indexed,
        files_total=files_total,
        complete=complete,
        partial_shard_search=smoke_partial or partial_shard_search,
        sweep_failed_shards=max(sweep_failed_shards, smoke_failed),
        source_count=smoke_source_count or source_count,
        hits=hits,
        query_smoke_ok=smoke_ok,
        query_smoke=str(smoke.get("query") or ""),
        shards_searched=smoke_shards,
        shards_total=smoke_total,
        error=error,
    )


def _health_url(search_url: str) -> str:
    parsed = urlparse(search_url)
    return urlunparse(parsed._replace(path="/health", query="", fragment=""))


def _full_raw_search_url() -> str:
    return (
        os.environ.get("RESEARKA_FULLRAW_SEARCH_URL", "")
        or os.environ.get("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "")
    ).strip()


def _full_raw_headers(*, content_type: bool = False) -> dict[str, str]:
    headers = {"User-Agent": "v5-memo/0.1"}
    if content_type:
        headers["Content-Type"] = "application/json"
    token = (
        os.environ.get("RESEARKA_FULLRAW_INDEX_TOKEN", "")
        or os.environ.get("RESEARKA_FULLRAW_TOKEN", "")
        or os.environ.get("V5_MEMO_FULL_RAW_INDEX_TOKEN", "")
        or os.environ.get("V5_MEMO_FULL_RAW_CORPUS_TOKEN", "")
    )
    token = token.strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _full_raw_query_smoke(search_url: str) -> dict[str, object]:
    query = os.environ.get("V5_MEMO_FULL_RAW_HEALTH_SMOKE_QUERY", "metformin longevity").strip()
    if not query:
        return {"query": "", "hits": 0, "receipt": {}, "error": "missing smoke query"}
    payload = {
        "query": query,
        "limit": 1,
        "rank_mode": "relevance",
        "cache_only": True,
        "queue_if_missing": False,
    }
    headers = _full_raw_headers(content_type=True)
    try:
        request = Request(
            search_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=_health_timeout()) as response:
            data: Any = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        return {"query": query, "hits": 0, "receipt": {}, "error": str(exc)}
    if not isinstance(data, dict):
        return {"query": query, "hits": 0, "receipt": {}, "error": "smoke response is not an object"}
    meta = data.get("meta")
    receipt = meta.get("shard_receipt") if isinstance(meta, dict) else {}
    results = data.get("results")
    return {
        "query": query,
        "hits": len(results) if isinstance(results, list) else 0,
        "receipt": receipt if isinstance(receipt, dict) else {},
        "error": str(data.get("error") or ""),
    }


def _full_raw_min_shards(shards_total: int) -> int:
    configured = _int_env("RESEARKA_FULLRAW_MIN_SHARDS_SEARCHED") or _int_env(
        "V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED",
    )
    if configured > 0:
        return configured
    return shards_total if shards_total > 0 else 1


def _full_raw_min_sources() -> int:
    configured = _int_env("RESEARKA_FULLRAW_MIN_SOURCES_SEARCHED") or _int_env(
        "V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED",
    )
    return configured if configured > 0 else 5


def _health_timeout() -> float:
    try:
        return max(
            0.1,
            min(float(os.environ.get("RESEARKA_FULLRAW_HEALTH_TIMEOUT") or os.environ.get("V5_MEMO_FULL_RAW_HEALTH_TIMEOUT", "3")), 30.0),
        )
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


def _int_env(name: str) -> int:
    try:
        return max(0, int(os.environ.get(name, "0")))
    except ValueError:
        return 0
