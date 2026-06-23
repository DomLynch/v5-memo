"""Runtime search coverage reporting for V5."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True, slots=True)
class SearchCoverage:
    """What V5 can search in the current runtime."""

    openalex_full_corpus_api: bool
    researka_corpus_api: bool
    full_raw_local_corpus: bool
    summary: str


@dataclass(frozen=True, slots=True)
class FullRawCorpusReadiness:
    ready: bool
    reason: str
    health: dict[str, object]
    probe_receipt: dict[str, object]
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
        + (
            f"configured through {full_raw_url}"
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


def full_raw_corpus_readiness(
    *,
    search_url: str | None = None,
    token: str | None = None,
    require_complete: bool = True,
    min_shards_searched: int = 0,
    min_sources_searched: int = 0,
    probe_query: str = "management forecast disclosure",
    timeout: float = 45.0,
) -> FullRawCorpusReadiness:
    """Check whether a configured full-raw service is safe to promote."""
    resolved_search_url = (
        search_url if search_url is not None else os.environ.get("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "")
    ).strip()
    resolved_token = (
        token if token is not None else os.environ.get("V5_MEMO_FULL_RAW_CORPUS_TOKEN", "")
    ).strip()
    if not resolved_search_url:
        return _readiness(False, "missing V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL")

    try:
        health = _request_json(_health_url_from_search_url(resolved_search_url), token=resolved_token, timeout=timeout)
    except RuntimeError as exc:
        return _readiness(False, str(exc))
    if not isinstance(health, dict):
        return _readiness(False, "health response is not an object")

    reasons = _health_readiness_reasons(health, require_complete=require_complete)
    probe_receipt: dict[str, object] = {}
    if probe_query.strip():
        try:
            probe = _request_json(
                resolved_search_url,
                token=resolved_token,
                timeout=timeout,
                payload={
                    "query": probe_query,
                    "limit": 8,
                    "timeout_seconds": timeout,
                },
            )
        except RuntimeError as exc:
            reasons.append(str(exc))
        else:
            probe_receipt = _response_shard_receipt(probe)
            reasons.extend(_probe_readiness_reasons(
                probe,
                probe_receipt,
                require_complete=require_complete,
                min_shards_searched=max(0, min_shards_searched),
                min_sources_searched=max(0, min_sources_searched),
            ))

    ready = not reasons
    reason = "ready" if ready else "; ".join(reasons)
    files_indexed = _int_value(health.get("files_indexed"))
    files_total = _int_value(health.get("files_total"))
    shards = _int_value(probe_receipt.get("shards_searched"))
    shards_total = _int_value(probe_receipt.get("shards_total"))
    summary = (
        f"fullraw readiness: {'ready' if ready else 'not ready'}; "
        f"files={files_indexed or 0}/{files_total or 0}; "
        f"probe_shards={shards or 0}/{shards_total or 0}; "
        f"reason={reason}"
    )
    return FullRawCorpusReadiness(
        ready=ready,
        reason=reason,
        health=dict(health),
        probe_receipt=probe_receipt,
        summary=summary,
    )


def require_full_raw_corpus_ready(
    *,
    search_url: str | None = None,
    token: str | None = None,
    require_complete: bool = True,
    min_shards_searched: int = 0,
    min_sources_searched: int = 0,
    probe_query: str = "management forecast disclosure",
    timeout: float = 45.0,
) -> None:
    readiness = full_raw_corpus_readiness(
        search_url=search_url,
        token=token,
        require_complete=require_complete,
        min_shards_searched=min_shards_searched,
        min_sources_searched=min_sources_searched,
        probe_query=probe_query,
        timeout=timeout,
    )
    if not readiness.ready:
        raise RuntimeError(readiness.summary)


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


def _readiness(
    ready: bool,
    reason: str,
    *,
    health: dict[str, object] | None = None,
    probe_receipt: dict[str, object] | None = None,
) -> FullRawCorpusReadiness:
    return FullRawCorpusReadiness(
        ready=ready,
        reason=reason,
        health=health or {},
        probe_receipt=probe_receipt or {},
        summary=f"fullraw readiness: {'ready' if ready else 'not ready'}; reason={reason}",
    )


def _health_url_from_search_url(search_url: str) -> str:
    stripped = search_url.rstrip("/")
    if stripped.endswith("/search"):
        return stripped.removesuffix("/search") + "/health"
    return stripped + "/health"


def _request_json(
    url: str,
    *,
    token: str,
    timeout: float,
    payload: dict[str, object] | None = None,
) -> Any:
    headers = {"User-Agent": "v5-memo/0.1"}
    data: bytes | None = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, data=data, headers=headers, method="POST" if payload is not None else "GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, OSError, TimeoutError, ValueError) as exc:
        raise RuntimeError(f"{url} failed: {exc}") from exc


def _health_readiness_reasons(health: dict[str, object], *, require_complete: bool) -> list[str]:
    reasons: list[str] = []
    if health.get("ok") is not True:
        reasons.append("health ok is not true")
    files_indexed = _int_value(health.get("files_indexed"))
    files_total = _int_value(health.get("files_total"))
    if require_complete:
        if health.get("complete") is not True:
            reasons.append("full raw index incomplete")
        if files_total is not None and files_indexed is not None and files_indexed < files_total:
            reasons.append(f"files indexed {files_indexed}/{files_total}")
    return reasons


def _probe_readiness_reasons(
    probe: Any,
    receipt: dict[str, object],
    *,
    require_complete: bool,
    min_shards_searched: int,
    min_sources_searched: int,
) -> list[str]:
    reasons: list[str] = []
    meta = probe.get("meta", {}) if isinstance(probe, dict) else {}
    count = _int_value(meta.get("count")) if isinstance(meta, dict) else None
    if count is not None and count <= 0:
        reasons.append("probe returned no results")
    shards = _int_value(receipt.get("shards_searched")) or 0
    if min_shards_searched and shards < min_shards_searched:
        reasons.append(f"probe searched {shards}/{min_shards_searched} required shards")
    source_count = _source_count(receipt.get("sources_searched"))
    if min_sources_searched and source_count < min_sources_searched:
        reasons.append(f"probe searched {source_count}/{min_sources_searched} required sources")
    if require_complete:
        remaining = _int_value(receipt.get("sweep_remaining_shards"))
        failed = _int_value(receipt.get("sweep_failed_shards"))
        if remaining is not None and remaining > 0:
            reasons.append(f"probe sweep has {remaining} remaining shards")
        if failed is not None and failed > 0:
            reasons.append(f"probe sweep has {failed} failed shards")
    return reasons


def _response_shard_receipt(data: Any) -> dict[str, object]:
    if not isinstance(data, dict):
        return {}
    meta = data.get("meta")
    if not isinstance(meta, dict):
        return {}
    receipt = meta.get("shard_receipt")
    return dict(receipt) if isinstance(receipt, dict) else {}


def _source_count(value: object) -> int:
    if not isinstance(value, dict):
        return 0
    return sum(1 for count in value.values() if _int_value(count))


def _int_value(value: object) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            return int(value)
        if isinstance(value, bytes):
            return int(value)
        return None
    except (TypeError, ValueError):
        return None
