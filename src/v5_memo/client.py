"""Small stdlib search clients for OpenAlex, Researka, and full raw corpus search."""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass, replace
from html import unescape
from http.client import RemoteDisconnected
from itertools import combinations
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from v5_memo.schemas import CorpusHit


class SearchBackendError(RuntimeError):
    """Raised when strict search mode cannot reach or parse a backend."""


@dataclass(frozen=True, slots=True)
class FullRawSearchPass:
    name: str
    query: str
    rank_mode: str = "relevance"


_FULLRAW_CORE_DROP = {
    "adaptation",
    "adaptations",
    "effect",
    "effects",
    "mechanism",
    "mechanisms",
    "outcome", "outcomes", "post",
    "response",
    "responses", "exercise", "exercises",
    "result",
    "results", "recovery",
}
_FULLRAW_PAIR_DROP = _FULLRAW_CORE_DROP | {
    "adult",
    "adults",
    "blunt",
    "blunted",
    "human",
    "humans",
    "older",
    "supplement",
    "supplementation",
    "trained",
    "expected",
}
_FULLRAW_RARE_ANCHOR_DROP = {"hypertrophy", "resistance", "strength", "training"}
_FULLRAW_QUERY_FILLER_DROP = {
    "article",
    "articles",
    "clinical",
    "controlled",
    "determine",
    "efficacy",
    "evidence",
    "healthy",
    "paper",
    "papers",
    "participants",
    "randomized",
    "research",
    "study",
    "studies",
}
_FULLRAW_LEGACY_PREFIX = "V5_MEMO_FULL_RAW_"
_FULLRAW_GENERIC_PREFIX = "RESEARKA_FULLRAW_"
_FULLRAW_SPECIAL_ALIASES = {
    "V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL": ("RESEARKA_FULLRAW_SEARCH_URL",),
    "V5_MEMO_FULL_RAW_CORPUS_TOKEN": ("RESEARKA_FULLRAW_TOKEN",),
    "V5_MEMO_FULL_RAW_INDEX_TOKEN": ("RESEARKA_FULLRAW_INDEX_TOKEN", "RESEARKA_FULLRAW_TOKEN"),
}


def _fullraw_env_names(name: str) -> tuple[str, ...]:
    if not name.startswith(_FULLRAW_LEGACY_PREFIX):
        return (name,)
    suffix = name.removeprefix(_FULLRAW_LEGACY_PREFIX)
    candidates = (*_FULLRAW_SPECIAL_ALIASES.get(name, ()), f"{_FULLRAW_GENERIC_PREFIX}{suffix}", name)
    return tuple(dict.fromkeys(candidates))


def _fullraw_env(name: str, default: str = "") -> str:
    for candidate in _fullraw_env_names(name):
        value = os.environ.get(candidate)
        if value is not None and value != "":
            return value
    return default
_DOI_BACKFILL_PRIORITY_TERMS = {
    "attenuate",
    "attenuated",
    "augment",
    "blunt",
    "blunted",
    "expected",
    "hypothesis",
    "impair",
    "impaired",
    "mimic",
    "mimetic",
    "protocol",
    "randomized",
    "reduce",
    "reduced",
    "trial",
}
_FULLRAW_COMPLETED_CACHE_FALLBACK_LIMIT = 10
_UNSAFE_DOI_CHARS = frozenset("()[]{}")


class OpenAlexFullCorpusSearchClient:
    """Synchronous client for OpenAlex works search over the full corpus."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.openalex.org",
        mailto: str = "",
        timeout: float = 20.0,
        year_min: int = 1900,
        year_max: int = 2100,
        max_variants: int = 8,
        strict: bool = False,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._mailto = mailto.strip()
        self._timeout = timeout
        self._year_min = year_min
        self._year_max = year_max
        self._max_variants = max(1, max_variants)
        self._strict = strict

    @classmethod
    def from_env(cls, *, strict: bool = False) -> OpenAlexFullCorpusSearchClient:
        return cls(
            base_url=os.environ.get("V5_MEMO_OPENALEX_URL", "https://api.openalex.org"),
            mailto=os.environ.get("V5_MEMO_OPENALEX_MAILTO", os.environ.get("OPENALEX_MAILTO", "")),
            max_variants=_int_env("V5_MEMO_OPENALEX_MAX_VARIANTS", 1 if strict else 8),
            strict=strict,
        )

    def search(self, query: str, *, limit: int = 25) -> list[CorpusHit]:
        variants = _query_variants(query, limit=self._max_variants)
        if not variants:
            return []
        seed_terms = _query_terms(query)
        per_variant_limit = max(5, min(50, limit))
        best: dict[str, tuple[float, CorpusHit]] = {}
        for variant in variants:
            hits = self._search_works(variant, limit=per_variant_limit)
            variant_terms = _query_terms(variant)
            for rank, hit in enumerate(hits, start=1):
                score = _rerank_score(hit, seed_terms=seed_terms, variant_terms=variant_terms, rank=rank)
                scored = replace(
                    hit,
                    metadata={
                        **hit.metadata,
                        "search_variant": variant,
                        "rerank_score": round(score, 4),
                    },
                )
                current = best.get(scored.source_key)
                if current is None or score > current[0]:
                    best[scored.source_key] = (score, scored)
        return [hit for _, hit in sorted(best.values(), key=lambda item: item[0], reverse=True)[:limit]]

    def _search_works(self, query: str, *, limit: int) -> list[CorpusHit]:
        params = {
            "search": query[:1024],
            "filter": ",".join(
                [
                    "type:article",
                    "has_abstract:true",
                    f"from_publication_date:{self._year_min}-01-01",
                    f"to_publication_date:{self._year_max}-12-31",
                ]
            ),
            "per-page": str(max(1, min(limit, 200))),
            "select": ",".join(
                [
                    "id",
                    "doi",
                    "display_name",
                    "title",
                    "abstract_inverted_index",
                    "publication_year",
                    "primary_location",
                    "cited_by_count",
                ]
            ),
        }
        if self._mailto:
            params["mailto"] = self._mailto
        request = Request(
            f"{self._base_url}/works?{urllib.parse.urlencode(params)}",
            headers={"User-Agent": "v5-memo/0.1"},
            method="GET",
        )
        for attempt in range(2):
            try:
                with urlopen(request, timeout=self._timeout) as response:
                    data: Any = json.loads(response.read().decode("utf-8"))
                return _parse_openalex_response(data)
            except HTTPError as exc:
                if exc.code == 429 and self._strict and attempt == 0:
                    time.sleep(_retry_after_seconds(exc))
                    continue
                if self._strict:
                    raise SearchBackendError(f"OpenAlex search failed: {exc}") from exc
                return []
            except (URLError, TimeoutError, ValueError) as exc:
                if self._strict:
                    raise SearchBackendError(f"OpenAlex search failed: {exc}") from exc
                return []
        return []


class ResearkaSearchClient:
    """Synchronous client for Researka full-paper corpus search."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        timeout: float = 20.0,
        year_min: int = 1900,
        year_max: int = 2100,
        strict: bool = False,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token.strip()
        self._timeout = timeout
        self._year_min = year_min
        self._year_max = year_max
        self._strict = strict

    @classmethod
    def from_env(cls, *, strict: bool = False) -> ResearkaSearchClient:
        return cls(
            base_url=os.environ.get("RESEARKA_DATABASE_URL", "https://database.researka.org"),
            token=_load_researka_token(),
            strict=strict,
        )

    @property
    def configured(self) -> bool:
        return bool(self._base_url and self._token)

    def search(self, query: str, *, limit: int = 25) -> list[CorpusHit]:
        if not self._base_url or not self._token or not query.strip():
            if self._strict and query.strip():
                raise SearchBackendError("Researka search is not configured")
            return []
        return self._search_papers(query, limit=limit)

    def _search_papers(self, query: str, *, limit: int) -> list[CorpusHit]:
        payload = {
            "query": query[:1024],
            "top_k": max(1, min(limit, 200)),
            "year_min": self._year_min,
            "year_max": self._year_max,
        }
        request = Request(
            f"{self._base_url}/api/v1/corpus/search",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "X-Researka-Token": self._token,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                data: Any = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, RemoteDisconnected, ValueError) as exc:
            if self._strict:
                raise SearchBackendError(f"Researka search failed: {exc}") from exc
            return []
        return _parse_corpus_search_response(data)


class FullRawCorpusSearchClient:
    """Client for a real indexed/searchable 450M+ raw corpus service."""

    def __init__(
        self,
        *,
        search_url: str,
        token: str = "",
        timeout: float = 45.0,
        year_min: int = 1900,
        year_max: int = 2100,
        max_variants: int = 16,
        search_budget_seconds: float = 180.0,
        sweep_wait_seconds: float = 0.0,
        sweep_poll_seconds: float = 1.0,
        doi_abstract_backfill_limit: int = 0,
        min_shards_searched: int = 0,
        min_sources_searched: int = 0,
        require_auth: bool = False,
        progress: bool = False,
        strict: bool = False,
    ) -> None:
        self._search_url = search_url.strip()
        self._token = token.strip()
        self._timeout = timeout
        self._year_min = year_min
        self._year_max = year_max
        self._max_variants = max(1, max_variants)
        self._search_budget_seconds = max(0.0, search_budget_seconds)
        self._sweep_wait_seconds = max(0.0, sweep_wait_seconds)
        self._sweep_poll_seconds = max(0.0, sweep_poll_seconds)
        self._doi_abstract_backfill_limit = max(0, doi_abstract_backfill_limit)
        self._min_shards_searched = max(0, min_shards_searched)
        self._min_sources_searched = max(0, min_sources_searched)
        self._require_auth = require_auth or bool(self._token)
        self._progress = progress
        self._strict = strict

    @classmethod
    def from_env(cls, *, strict: bool = False) -> FullRawCorpusSearchClient:
        token = (
            _fullraw_env("V5_MEMO_FULL_RAW_INDEX_TOKEN", "").strip()
            or _fullraw_env("V5_MEMO_FULL_RAW_CORPUS_TOKEN", "").strip()
        )
        search_url = _fullraw_env("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "")
        if not search_url and token:
            search_url = "http://127.0.0.1:9903/search"
        default_min_shards = 1525 if token else 0
        default_min_sources = 5 if token else 0
        search_budget_seconds = min(_float_env("V5_MEMO_FULL_RAW_SEARCH_BUDGET_SECONDS", 180.0), 900.0)
        return cls(
            search_url=search_url,
            token=token,
            timeout=min(_float_env("V5_MEMO_FULL_RAW_QUERY_TIMEOUT", _float_env("V5_MEMO_FULL_RAW_CORPUS_TIMEOUT", 60.0)), 240.0),
            max_variants=min(_int_env("V5_MEMO_FULL_RAW_MAX_VARIANTS", 16), 4),
            search_budget_seconds=search_budget_seconds,
            sweep_wait_seconds=min(_float_env("V5_MEMO_FULL_RAW_FOREGROUND_SWEEP_WAIT_SECONDS", 0.0), search_budget_seconds),
            sweep_poll_seconds=_float_env("V5_MEMO_FULL_RAW_SWEEP_POLL_SECONDS", 1.0),
            doi_abstract_backfill_limit=_int_env(
                "V5_MEMO_FULL_RAW_DOI_ABSTRACT_BACKFILL_LIMIT",
                6,
            ),
            min_shards_searched=_int_env("V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED", default_min_shards),
            min_sources_searched=_int_env("V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED", default_min_sources),
            require_auth=_bool_env("V5_MEMO_FULL_RAW_REQUIRE_AUTH", bool(token)),
            progress=_bool_env("V5_MEMO_FULL_RAW_PROGRESS", False),
            strict=strict,
        )

    @property
    def configured(self) -> bool:
        return bool(self._search_url)

    def search(self, query: str, *, limit: int = 25) -> list[CorpusHit]:
        search_passes = _fullraw_search_passes(query, limit=self._max_variants)
        if not self._search_url or not search_passes:
            return []
        seed_terms = _query_terms(query)
        anchor_terms = tuple(term for term in seed_terms if len(term) >= 4 and term not in _FULLRAW_PAIR_DROP and term not in _FULLRAW_RARE_ANCHOR_DROP)
        per_variant_limit = max(5, min(limit, 50))
        best: dict[str, tuple[float, CorpusHit]] = {}
        total_seen = 0
        duplicate_seen = 0
        passes_run: list[str] = []
        rank_modes_run: list[str] = []
        started = time.monotonic()
        for variant_index, search_pass in enumerate(search_passes, start=1):
            elapsed = time.monotonic() - started
            if self._search_budget_seconds and elapsed >= self._search_budget_seconds:
                self._log_progress(
                    "fullraw search budget reached "
                    f"after {elapsed:.1f}s; variants={variant_index - 1}/{len(search_passes)}"
                )
                break
            expected_variant_seconds = self._timeout
            if (
                self._search_budget_seconds
                and variant_index > 1
                and elapsed + expected_variant_seconds > self._search_budget_seconds
            ):
                self._log_progress(
                    "fullraw search budget would be exceeded "
                    f"after {elapsed:.1f}s; variants={variant_index - 1}/{len(search_passes)}"
                )
                break
            passes_run.append(search_pass.name)
            rank_modes_run.append(search_pass.rank_mode)
            self._log_progress(
                f"fullraw variant {variant_index}/{len(search_passes)} "
                f"start [{search_pass.name}/{search_pass.rank_mode}]: {search_pass.query}"
            )
            variant_started = time.monotonic()
            try:
                hits = self._search_variant(search_pass, limit=per_variant_limit)
            except SearchBackendError as exc:
                if self._uses_cache_sweep_contract() and str(exc).startswith(
                    "Full raw corpus search coverage too narrow"
                ):
                    if best:
                        break
                    if search_pass.query != query and variant_index < len(search_passes):
                        continue
                    raise
                if best:
                    break
                if variant_index < len(search_passes):
                    continue
                raise
            self._log_progress(
                f"fullraw variant {variant_index}/{len(search_passes)} done "
                f"in {time.monotonic() - variant_started:.1f}s; hits={len(hits)}"
            )
            variant_terms = _query_terms(search_pass.query)
            for rank, hit in enumerate(hits, start=1):
                if anchor_terms and not any(term in hit.text.casefold() for term in anchor_terms):
                    continue
                total_seen += 1
                if hit.source_key in best:
                    duplicate_seen += 1
                score = _rerank_score(hit, seed_terms=seed_terms, variant_terms=variant_terms, rank=rank)
                scored = replace(
                    hit,
                    metadata={
                        **hit.metadata,
                        "search_pass": search_pass.name,
                        "search_variant": search_pass.query,
                        "rank_mode": search_pass.rank_mode,
                        "rerank_score": round(score, 4),
                    },
                )
                current = best.get(scored.source_key)
                if current is None or score > current[0]:
                    best[scored.source_key] = (score, scored)
            if len(best) >= limit:
                break
        self._log_progress(f"fullraw query done in {time.monotonic() - started:.1f}s; hits={len(best)}")
        duplicate_rate = round(duplicate_seen / total_seen, 4) if total_seen else 0.0
        auth_receipts: list[dict[str, object]] = []
        for _, hit in best.values():
            raw_receipt = hit.metadata.get("shard_receipt")
            if isinstance(raw_receipt, dict):
                auth_receipts.append(raw_receipt)
        receipt = {
            "duplicate_rate": duplicate_rate,
            "search_passes": tuple(dict.fromkeys(passes_run)),
            "rank_modes": tuple(dict.fromkeys(rank_modes_run)),
            "auth_required": any(receipt.get("auth_required") is True for receipt in auth_receipts),
            "authenticated": any(receipt.get("authenticated") is True for receipt in auth_receipts),
        }
        ranked_hits = [
            replace(hit, metadata={**hit.metadata, "fullraw_search_receipt": receipt})
            for _, hit in sorted(best.values(), key=lambda item: item[0], reverse=True)[:limit]
        ]
        return _backfill_missing_openalex_abstracts(
            ranked_hits,
            limit=self._doi_abstract_backfill_limit,
        )

    def _log_progress(self, message: str) -> None:
        if self._progress:
            print(message, file=sys.stderr, flush=True)

    def _search_variant(self, search_pass: FullRawSearchPass, *, limit: int) -> list[CorpusHit]:
        request_limit = max(1, min(limit, 200))
        payload = {
            "query": search_pass.query[:1024],
            "limit": request_limit,
            "top_k": request_limit,
            "year_min": self._year_min,
            "year_max": self._year_max,
            "corpus": "full_raw_450m_plus",
            "search_pass": search_pass.name,
            "rank_mode": search_pass.rank_mode,
            "timeout_seconds": self._timeout,
        }
        if self._uses_cache_sweep_contract():
            payload.update({"cache_only": True, "queue_if_missing": True, "priority": True})
        initial_error: SearchBackendError | None = None
        try:
            data = self._request_search(payload)
        except SearchBackendError as exc:
            if not self._sweep_wait_seconds:
                raise
            initial_error = exc
            data = {}
        receipt = _full_raw_shard_receipt(data)
        sweep_status = _full_raw_async_sweep_status(data)
        can_wait_for_sweep = initial_error is not None or sweep_status in {"miss", "queued", "running"}
        coverage_error = isinstance(data, dict) and data.get("error") == "coverage_too_narrow"
        if (
            self._sweep_wait_seconds
            and search_pass.name in {"focused", "core"}
            and (not _parse_full_raw_search_response(data) or not self._receipt_is_sufficient(receipt))
            and (can_wait_for_sweep or coverage_error)
        ):
            cached = self._wait_for_sweep_hit(payload)
            if cached is not None:
                data = cached
                receipt = _full_raw_shard_receipt(data)
            elif initial_error is not None:
                raise initial_error
        if not self._receipt_is_sufficient(receipt) and request_limit > _FULLRAW_COMPLETED_CACHE_FALLBACK_LIMIT:
            fallback_payload = {
                **payload,
                "limit": _FULLRAW_COMPLETED_CACHE_FALLBACK_LIMIT,
                "top_k": _FULLRAW_COMPLETED_CACHE_FALLBACK_LIMIT,
                "cache_only": True,
                "queue_if_missing": True,
            }
            try:
                fallback_data = self._request_search(fallback_payload)
            except SearchBackendError:
                fallback_data = {}
            fallback_receipt = _full_raw_shard_receipt(fallback_data)
            if self._receipt_is_sufficient(fallback_receipt):
                self._log_progress("fullraw using completed low-limit exhaustive cache")
                data = fallback_data
                receipt = fallback_receipt
        if not self._receipt_is_sufficient(receipt):
            message = f"Full raw corpus search coverage too narrow: {_full_raw_receipt_summary(receipt)}"
            if self._strict:
                raise SearchBackendError(message)
            self._log_progress(message)
            return []
        return _parse_full_raw_search_response(data)

    def _wait_for_sweep_hit(self, payload: dict[str, object]) -> Any | None:
        deadline = time.monotonic() + self._sweep_wait_seconds
        cache_payload = {**payload, "cache_only": True, "queue_if_missing": True}
        while True:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._log_progress("fullraw async sweep wait expired; status=unknown")
                    return None
                data = self._request_search(cache_payload, timeout=min(self._timeout, remaining))
                status = _full_raw_async_sweep_status(data)
            except SearchBackendError:
                data = {}
                status = "error"
            if status == "hit":
                self._log_progress("fullraw async sweep cache hit")
                return data
            if status == "disabled":
                return None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._log_progress(f"fullraw async sweep wait expired; status={status or 'unknown'}")
                return None
            time.sleep(min(max(self._sweep_poll_seconds, 0.05), remaining))

    def _request_search(self, payload: dict[str, object], *, timeout: float | None = None) -> Any:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "v5-memo/0.1",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = Request(
            self._search_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        request_timeout = self._timeout + 15.0 if timeout is None else max(0.05, timeout)
        for attempt in range(2):
            try:
                with urlopen(request, timeout=request_timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (ConnectionResetError, RemoteDisconnected) as exc:
                if attempt == 0:
                    self._log_progress(f"fullraw remote disconnect; retrying once: {payload.get('query', '')}")
                    continue
                if self._strict:
                    raise SearchBackendError(f"Full raw corpus search failed: {exc}") from exc
                return {}
            except HTTPError as exc:
                body = _read_http_error_json(exc)
                if isinstance(body, dict) and body.get("error") == "coverage_too_narrow":
                    return body
                if self._strict:
                    raise SearchBackendError(f"Full raw corpus search failed: {exc}") from exc
                return {}
            except (URLError, TimeoutError, ValueError) as exc:
                if self._strict:
                    raise SearchBackendError(f"Full raw corpus search failed: {exc}") from exc
                return {}
        return {}

    def _receipt_is_sufficient(self, receipt: dict[str, object]) -> bool:
        if self._require_auth and receipt.get("authenticated") is not True:
            return False
        if not receipt:
            return True
        coverage_required = bool(self._min_shards_searched or self._min_sources_searched)
        if coverage_required and receipt.get("partial_shard_search") is True:
            return False
        if coverage_required and (_int_or_none(receipt.get("sweep_failed_shards")) or 0) > 0:
            return False
        shards = _int_or_none(receipt.get("shards_searched")) or 0
        if self._min_shards_searched and shards < self._min_shards_searched:
            return False
        sources = receipt.get("sources_searched")
        source_count = (
            sum(1 for value in sources.values() if _int_or_none(value))
            if isinstance(sources, dict)
            else 0
        )
        return not (self._min_sources_searched and source_count < self._min_sources_searched)

    def _uses_cache_sweep_contract(self) -> bool:
        return bool(self._require_auth or self._min_shards_searched or self._min_sources_searched)


class HybridCorpusSearchClient:
    """Merge multiple corpus search surfaces behind one searcher contract."""

    def __init__(self, searchers: Sequence[object]) -> None:
        self._searchers = searchers

    def search(self, query: str, *, limit: int = 25) -> list[CorpusHit]:
        seed_terms = _query_terms(query)
        best: dict[str, tuple[float, str, CorpusHit]] = {}
        backend_order: list[str] = []
        for searcher in self._searchers:
            search = getattr(searcher, "search", None)
            if not callable(search):
                continue
            backend = _hybrid_backend_name(searcher)
            try:
                hits = search(query, limit=limit)
            except SearchBackendError:
                continue
            if hits and backend not in backend_order:
                backend_order.append(backend)
            for rank, hit in enumerate(hits, start=1):
                score = _rerank_score(hit, seed_terms=seed_terms, variant_terms=seed_terms, rank=rank)
                current = best.get(hit.source_key)
                if current is None or score > current[0]:
                    best[hit.source_key] = (
                        score,
                        backend,
                        replace(
                            hit,
                            metadata={
                                **hit.metadata,
                                "hybrid_backend": backend,
                                "hybrid_rerank_score": round(score, 4),
                            },
                        ),
                    )
        return _balanced_hybrid_hits(list(best.values()), backend_order, limit=limit)


def _hybrid_backend_name(searcher: object) -> str:
    name = searcher.__class__.__name__.replace("CorpusSearchClient", "")
    return re.sub(r"(?<!^)([A-Z])", r"_\1", name).casefold()


def _balanced_hybrid_hits(
    scored_hits: Sequence[tuple[float, str, CorpusHit]],
    backend_order: Sequence[str],
    *,
    limit: int,
) -> list[CorpusHit]:
    if limit <= 0:
        return []
    by_backend = {
        backend: sorted(
            (item for item in scored_hits if item[1] == backend),
            key=lambda item: item[0],
            reverse=True,
        )
        for backend in backend_order
    }
    selected: dict[str, tuple[float, str, CorpusHit]] = {}
    for backend in backend_order:
        if len(selected) >= limit:
            break
        for item in by_backend.get(backend, ()):
            if item[2].source_key not in selected:
                selected[item[2].source_key] = item
                break
    for item in sorted(scored_hits, key=lambda candidate: candidate[0], reverse=True):
        if len(selected) >= limit:
            break
        selected.setdefault(item[2].source_key, item)
    return [
        hit
        for _, _, hit in sorted(selected.values(), key=lambda item: item[0], reverse=True)
    ][:limit]


def _load_researka_token() -> str:
    direct = os.environ.get("RESEARKA_DATABASE_TOKEN") or os.environ.get("RESEARKA_TOKEN")
    if direct:
        return direct.strip()
    allowlist = os.environ.get("RESEARKA_TOKENS", "")
    for entry in allowlist.split(","):
        token = entry.split(":", 1)[0].strip()
        if token:
            return token
    return ""


def _parse_corpus_search_response(data: Any) -> list[CorpusHit]:
    if not isinstance(data, list):
        return []
    return [hit for item in data if (hit := _parse_paper_hit(item))]


def _parse_full_raw_search_response(data: Any) -> list[CorpusHit]:
    meta: dict[str, Any] = {}
    items: Any = data
    if isinstance(data, dict):
        raw_meta = data.get("meta")
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        items = data.get("results", data.get("hits", []))
    if not isinstance(items, list):
        return []
    match_count = _int_or_none(meta.get("count") or meta.get("total") or meta.get("total_count"))
    shard_receipt = _full_raw_shard_receipt(data)
    hits = [
        hit
        for item in items
        if (
            hit := _parse_full_raw_paper_hit(
                item,
                match_count=match_count,
                shard_receipt=shard_receipt,
            )
        ) is not None
    ]
    return _drop_conflicting_duplicate_doi_year_hits(hits)


def _full_raw_shard_receipt(data: Any) -> dict[str, object]:
    if not isinstance(data, dict):
        return {}
    direct_receipt = data.get("shard_receipt")
    if isinstance(direct_receipt, dict):
        return dict(direct_receipt)
    meta = data.get("meta")
    if not isinstance(meta, dict):
        return {}
    receipt = meta.get("shard_receipt")
    if not isinstance(receipt, dict):
        return {}
    return dict(receipt)


def _full_raw_receipt_summary(receipt: dict[str, object]) -> dict[str, object]:
    return {
        "shards_searched": receipt.get("shards_searched"),
        "shards_total": receipt.get("shards_total"),
        "partial_shard_search": receipt.get("partial_shard_search"),
        "sweep_failed_shards": receipt.get("sweep_failed_shards"),
        "source_count_searched": receipt.get("source_count_searched"),
        "sweep_remaining_shards": receipt.get("sweep_remaining_shards"),
        "sweep_strategy": receipt.get("sweep_strategy"),
        "authenticated": receipt.get("authenticated"),
    }


def _full_raw_async_sweep_status(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    meta = data.get("meta")
    if not isinstance(meta, dict):
        return ""
    async_sweep = meta.get("async_sweep")
    if not isinstance(async_sweep, dict):
        return ""
    status = async_sweep.get("status")
    return status if isinstance(status, str) else ""


def _read_http_error_json(exc: HTTPError) -> Any:
    try:
        return json.loads(exc.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return {}


def _parse_openalex_response(data: Any) -> list[CorpusHit]:
    if not isinstance(data, dict):
        return []
    results = data.get("results")
    if not isinstance(results, list):
        return []
    meta_raw = data.get("meta")
    meta = meta_raw if isinstance(meta_raw, dict) else {}
    match_count = _int_or_none(meta.get("count"))
    return [
        hit
        for item in results
        if (hit := _parse_openalex_work(item, match_count=match_count)) is not None
    ]


def _parse_openalex_work(item: Any, *, match_count: int | None) -> CorpusHit | None:
    if not isinstance(item, dict):
        return None
    title = _clean(item.get("display_name") or item.get("title"), limit=500)
    if not title:
        return None
    raw_doi = _normalize_doi(item.get("doi"))
    doi = _safe_doi(raw_doi)
    openalex_id = _clean(item.get("id"), limit=256)
    primary_location = item.get("primary_location")
    location = primary_location if isinstance(primary_location, dict) else {}
    source_raw = location.get("source")
    source = source_raw if isinstance(source_raw, dict) else {}
    venue = _clean(source.get("display_name"), limit=200) or None
    return CorpusHit(
        hit_id=doi or openalex_id or title,
        title=title,
        abstract=_abstract_from_inverted_index(item.get("abstract_inverted_index")),
        source="openalex:full-corpus",
        year=_int_or_none(item.get("publication_year")),
        url=f"https://doi.org/{doi}" if doi else openalex_id,
        doi=doi,
        venue=venue,
        metadata={
            "openalex_id": openalex_id,
            "raw_doi": raw_doi if raw_doi != doi else "",
            "cited_by_count": _int_or_none(item.get("cited_by_count")),
            "query_match_count": match_count,
        },
    )


def _backfill_missing_openalex_abstracts(
    hits: list[CorpusHit],
    *,
    limit: int,
) -> list[CorpusHit]:
    if limit <= 0:
        return hits
    out: list[CorpusHit | None] = list(hits)
    backfilled = 0
    eligible = [
        (index, hit)
        for index, hit in enumerate(hits)
        if hit.doi
    ]
    eligible.sort(key=lambda item: _doi_backfill_priority(item[1]), reverse=True)
    for index, hit in eligible:
        if backfilled >= limit:
            break
        doi = hit.doi
        if doi is None:
            continue
        enriched = _fetch_openalex_work_by_doi(doi)
        if enriched is None:
            continue
        backfilled += 1
        if enriched.title and not _titles_match(hit.title, enriched.title):
            out[index] = None
            continue
        if hit.abstract or not enriched.abstract:
            continue
        out[index] = replace(
            hit,
            abstract=enriched.abstract,
            year=hit.year or enriched.year,
            venue=hit.venue or enriched.venue,
            metadata={**hit.metadata, "abstract_backfill": "openalex_doi"},
        )
    return [hit for hit in out if hit is not None]


def _doi_backfill_priority(hit: CorpusHit) -> int:
    text = f"{hit.title} {hit.venue or ''}".casefold()
    terms = set(re.findall(r"[a-z][a-z0-9]+", text))
    return len(terms & _DOI_BACKFILL_PRIORITY_TERMS)


def _fetch_openalex_work_by_doi(doi: str) -> CorpusHit | None:
    encoded = urllib.parse.quote(doi, safe="")
    request = Request(
        f"https://api.openalex.org/works/doi:{encoded}",
        headers={"User-Agent": "v5-memo/0.1"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=12.0) as response:
            data: Any = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError):
        return None
    return _parse_openalex_work(data, match_count=None)


def _titles_match(left: str, right: str) -> bool:
    left_terms = _title_terms(left)
    right_terms = _title_terms(right)
    if not left_terms or not right_terms:
        return True
    return len(left_terms & right_terms) / min(len(left_terms), len(right_terms)) >= 0.45


def _title_terms(title: str) -> set[str]:
    stop = {"and", "for", "from", "into", "the", "with", "without", "study", "trial"}
    return {
        raw
        for raw in re.findall(r"[a-z][a-z0-9]{2,}", title.casefold())
        if raw not in stop
    }


def _parse_paper_hit(item: Any) -> CorpusHit | None:
    if not isinstance(item, dict):
        return None
    title = _clean(item.get("title"), limit=500)
    if not title:
        return None
    raw_doi = _normalize_doi(item.get("doi"))
    doi = _safe_doi(raw_doi)
    pmid = _clean(item.get("pmid"), limit=64)
    pmcid = _clean(item.get("pmcid"), limit=64)
    hit_id = doi or pmid or pmcid or title
    return CorpusHit(
        hit_id=hit_id,
        title=title,
        abstract=_clean(item.get("abstract"), limit=4000),
        source="researka:corpus",
        year=_int_or_none(item.get("year")),
        url=f"https://doi.org/{doi}" if doi else "",
        doi=doi,
        venue=_clean(item.get("journal"), limit=200) or None,
        metadata={
            "pmid": pmid,
            "pmcid": pmcid,
            "raw_doi": raw_doi if raw_doi != doi else "",
            "cited_by_count": _int_or_none(item.get("cited_by_count")),
            "similarity_score": _float_or_none(item.get("similarity_score")),
        },
    )


def _parse_full_raw_paper_hit(
    item: Any,
    *,
    match_count: int | None,
    shard_receipt: dict[str, object] | None = None,
) -> CorpusHit | None:
    if not isinstance(item, dict):
        return None
    title = _clean(item.get("title") or item.get("display_name") or item.get("name"), limit=500)
    if not title:
        return None
    raw_doi = _normalize_doi(item.get("doi"))
    doi = _safe_doi(raw_doi)
    year = _int_or_none(item.get("year") or item.get("publication_year"))
    pmid = _clean(item.get("pmid"), limit=64)
    pmcid = _clean(item.get("pmcid"), limit=64)
    openalex_id = _clean(item.get("openalex_id") or item.get("openalex") or item.get("id"), limit=256)
    s2_id = _clean(
        item.get("semantic_scholar_id") or item.get("s2_id") or item.get("corpus_id"),
        limit=128,
    )
    arxiv_id = _clean(item.get("arxiv_id") or item.get("arxiv"), limit=128)
    origin = _clean(item.get("source") or item.get("raw_source") or item.get("provider"), limit=80)
    url = _clean(item.get("url"), limit=512)
    primary_location = item.get("primary_location")
    location = primary_location if isinstance(primary_location, dict) else {}
    source_raw = location.get("source")
    source = source_raw if isinstance(source_raw, dict) else {}
    venue = (
        _clean(item.get("journal") or item.get("venue") or item.get("source_name"), limit=200)
        or _clean(source.get("display_name"), limit=200)
        or None
    )
    abstract = _clean(
        item.get("abstract") or item.get("abstract_text") or item.get("description"),
        limit=4000,
    ) or _abstract_from_inverted_index(item.get("abstract_inverted_index"))
    return CorpusHit(
        hit_id=doi or pmid or pmcid or s2_id or arxiv_id or openalex_id or title,
        title=title,
        abstract=abstract,
        source=f"fullraw:{origin.casefold()}" if origin else "fullraw:450m-plus",
        year=year,
        url=f"https://doi.org/{doi}" if doi else url or openalex_id,
        doi=doi,
        venue=venue,
        metadata={
            "pmid": pmid,
            "pmcid": pmcid,
            "raw_doi": raw_doi if raw_doi != doi else "",
            "openalex_id": openalex_id,
            "semantic_scholar_id": s2_id,
            "arxiv_id": arxiv_id,
            "raw_source": origin,
            "cited_by_count": _int_or_none(item.get("cited_by_count") or item.get("citation_count")),
            "score": _float_or_none(item.get("score") or item.get("search_score")),
            "query_match_count": match_count,
            "shard_receipt": shard_receipt or {},
        },
    )


def _clean(value: object, *, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", unescape(value))
    return " ".join(text.split())[:limit]


def _normalize_doi(value: object) -> str | None:
    doi = _clean(value, limit=256)
    if not doi:
        return None
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.I) or None


def _safe_doi(value: str | None) -> str | None:
    if not value or any(char in value for char in _UNSAFE_DOI_CHARS):
        return None
    return value


def _doi_year_conflicts(doi: str | None, year: int | None) -> bool:
    if not doi or year is None:
        return False
    years = [int(match) for match in re.findall(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", doi)]
    return any(abs(parsed - year) > 3 for parsed in years)


def _drop_conflicting_duplicate_doi_year_hits(hits: list[CorpusHit]) -> list[CorpusHit]:
    by_doi: dict[str, list[CorpusHit]] = {}
    for hit in hits:
        if hit.doi:
            by_doi.setdefault(hit.doi.casefold(), []).append(hit)

    drop_ids: set[int] = set()
    for duplicate_hits in by_doi.values():
        if len(duplicate_hits) < 2:
            continue
        clean = [
            hit
            for hit in duplicate_hits
            if not _doi_year_conflicts(hit.doi, hit.year)
        ]
        if not clean:
            continue
        drop_ids.update(
            id(hit)
            for hit in duplicate_hits
            if _doi_year_conflicts(hit.doi, hit.year)
        )
    return [hit for hit in hits if id(hit) not in drop_ids]


def _abstract_from_inverted_index(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    positioned: list[tuple[int, str]] = []
    for word, positions in value.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for position in positions:
            parsed = _int_or_none(position)
            if parsed is not None and parsed >= 0:
                positioned.append((parsed, word))
    return _clean(" ".join(word for _, word in sorted(positioned)), limit=4000)


def _query_terms(query: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.findall(r"[A-Za-z0-9+]+", query.casefold())
        if len(token) > 1 and token not in {"and", "or", "the", "with", "for", "from", "into"}
    )


def _query_variants(query: str, *, limit: int) -> list[str]:
    terms = _query_terms(query)
    if not terms:
        return []
    candidates: list[tuple[str, ...]] = [terms]
    max_window = min(4, len(terms))
    for size in range(max_window, 1, -1):
        candidates.extend(tuple(terms[start : start + size]) for start in range(0, len(terms) - size + 1))
    if len(terms) > 3:
        candidates.extend(tuple(term for idx, term in enumerate(terms) if idx != drop) for drop in range(len(terms)))

    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = " ".join(candidate)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            return out
    return out


def _fullraw_query_variants(query: str, *, limit: int) -> list[str]:
    terms = _query_terms(query)
    if not terms:
        return []
    out: list[str] = []
    seen: set[str] = set()
    unique_terms = tuple(dict.fromkeys(terms))

    def add(variant: str) -> bool:
        if variant in seen:
            return False
        seen.add(variant)
        out.append(variant)
        return len(out) >= limit

    if add(" ".join(terms)):
        return out
    window_limit = min(limit, max(6, limit // 2))
    for variant in _query_variants(query, limit=window_limit):
        if add(variant):
            return out
    for pair in combinations(unique_terms, 2):
        if min(len(term) for term in pair) < 6:
            continue
        if add(" ".join(pair)):
            return out
    return out


def _fullraw_search_passes(query: str, *, limit: int) -> list[FullRawSearchPass]:
    if limit <= 0:
        return []
    out: list[FullRawSearchPass] = []
    seen: set[tuple[str, str]] = set()

    def add(name: str, variant: str, rank_mode: str = "relevance") -> bool:
        clean = _fullraw_compact_repeated_terms(" ".join(variant.split()))
        if not clean:
            return False
        key = (_fullraw_variant_key(clean), rank_mode)
        if key in seen:
            return False
        seen.add(key)
        out.append(FullRawSearchPass(name=name, query=clean, rank_mode=rank_mode))
        return len(out) >= limit

    raw_terms = _query_terms(query)
    terms = tuple(dict.fromkeys(raw_terms))
    if (
        2 <= len(terms) <= 5
        and len(terms) == len(raw_terms)
        and not any(term in _FULLRAW_QUERY_FILLER_DROP for term in terms)
        and add("focused", " ".join(terms))
    ):
        return out
    core_variant = _fullraw_core_variant(query)
    if core_variant and add("core", core_variant):
        return out
    window_limit = max(1, min(4, limit - 4 if limit > 5 else 1))
    for index, variant in enumerate(_query_variants(query, limit=window_limit)):
        if add("focused" if index == 0 else "broad", variant):
            return out
    first = (terms or ("",))[0]
    anchor = first if (len(first) <= 3 or len(first) >= 6) and first not in _FULLRAW_PAIR_DROP and not first.endswith(("tion", "sion", "ity", "ary", "acy", "ness")) else ""
    if not anchor and len(terms) > 3 and add("broad", " ".join(terms[:3])):
        return out
    pair_limit = max(0, limit - len(out) - (1 if anchor and limit <= 5 else 0))
    pair_added = False
    for variant in _fullraw_pair_variants(query, limit=limit * 2 if anchor else pair_limit):
        if anchor and anchor not in _query_terms(variant):
            continue
        before = len(out)
        if add("broad", variant):
            return out
        if len(out) > before:
            pair_added = True
        if pair_added and anchor:
            break
    if pair_added:
        return out
    if anchor and add("anchor", anchor):
        return out
    if add("adjacent", f"{query} mechanism outcome"):
        return out
    if add("falsifier", f"{query} null adverse conflicting"):
        return out
    if add("citation_heavy", query, "citation"):
        return out
    if add("recency", query, "recency"):
        return out
    for variant in _fullraw_query_variants(query, limit=limit * 2):
        if add("broad", variant):
            return out
    return out


def _fullraw_core_variant(query: str) -> str:
    terms = tuple(dict.fromkeys(_query_terms(query)))
    core = tuple(term for term in terms if term not in _FULLRAW_CORE_DROP and term not in _FULLRAW_QUERY_FILLER_DROP)
    if len(core) >= 2 and (len(core) < len(terms) or len(core) > 6):
        return " ".join(core[:6])
    if len(terms) >= 2 and len(terms) < len(_query_terms(query)):
        return " ".join(terms[:6])
    return ""


def _fullraw_compact_repeated_terms(query: str) -> str:
    terms = _query_terms(query)
    unique = tuple(dict.fromkeys(terms))
    if len(unique) >= 2 and len(unique) * 2 <= len(terms):
        return " ".join(unique)
    return query


def _fullraw_variant_key(query: str) -> str:
    terms = tuple(term for term in dict.fromkeys(_query_terms(query)) if term not in _FULLRAW_QUERY_FILLER_DROP)
    return " ".join(sorted(terms or _query_terms(query)))


def _fullraw_pair_variants(query: str, *, limit: int) -> list[str]:
    if limit <= 0:
        return []
    terms = tuple(
        term
        for term in dict.fromkeys(_query_terms(query))
        if len(term) >= 3 and term not in _FULLRAW_PAIR_DROP
    )
    out: list[str] = []
    seen: set[str] = set()

    def add(left: str, right: str) -> bool:
        if left == right:
            return False
        pair = f"{left} {right}"
        if pair in seen:
            return False
        seen.add(pair)
        out.append(pair)
        return len(out) >= limit

    first = terms[0] if terms else ""
    def term_score(term: str) -> int:
        return len(term) + (8 if term in _DOI_BACKFILL_PRIORITY_TERMS else 0) - (4 if term.endswith(("tion", "sion", "ity", "ary", "acy", "ness")) else 0)
    def specificity(pair: tuple[int, int]) -> tuple[int, int, int]:
        left_idx, right_idx = pair
        left, right = terms[left_idx], terms[right_idx]
        score = term_score(left) + term_score(right) + (6 if left == first else 0) + (4 if right_idx - left_idx == 1 else 0) - (8 if left in _FULLRAW_RARE_ANCHOR_DROP and right in _FULLRAW_RARE_ANCHOR_DROP else 0)
        return (score, left_idx - right_idx, -left_idx)

    pairs = ((left_idx, right_idx) for left_idx in range(len(terms)) for right_idx in range(left_idx + 1, len(terms)))
    for left_idx, right_idx in sorted(pairs, key=specificity, reverse=True):
        if add(terms[left_idx], terms[right_idx]):
            return out
    return out


def _rerank_score(
    hit: CorpusHit,
    *,
    seed_terms: tuple[str, ...],
    variant_terms: tuple[str, ...],
    rank: int,
) -> float:
    text = hit.text.casefold()
    title = hit.title.casefold()
    seed_coverage = _coverage(seed_terms, text)
    variant_coverage = _coverage(variant_terms, text)
    title_coverage = max(_coverage(seed_terms, title), _coverage(variant_terms, title))
    cited = hit.metadata.get("cited_by_count")
    citation_score = math.log10(max(0, cited) + 1) if isinstance(cited, (int, float)) else 0.0
    evidence_bonus = 0.0
    if hit.abstract:
        evidence_bonus += min(12.0, len(hit.abstract) / 250.0)
    if hit.doi:
        evidence_bonus += 4.0
    if hit.abstract and "openalex" in hit.source.casefold():
        evidence_bonus += 4.0
    return (
        seed_coverage * 70.0
        + variant_coverage * 20.0
        + title_coverage * 30.0
        + citation_score * 4.0
        + evidence_bonus
        - rank
    )


def _coverage(terms: tuple[str, ...], text: str) -> float:
    if not terms:
        return 0.0
    text_terms = set(_query_terms(text))
    return sum(1 for term in terms if term in text_terms) / len(terms)


def _int_or_none(value: object) -> int | None:
    if not isinstance(value, int | float | str | bytes | bytearray):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    if not isinstance(value, int | float | str | bytes | bytearray):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_env(name: str, default: float) -> float:
    parsed = _float_or_none(_fullraw_env(name, ""))
    return parsed if parsed is not None else default


def _int_env(name: str, default: int) -> int:
    parsed = _int_or_none(_fullraw_env(name, ""))
    return parsed if parsed is not None else default


def _bool_env(name: str, default: bool) -> bool:
    value = _fullraw_env(name, "")
    if value == "":
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _retry_after_seconds(exc: HTTPError) -> float:
    header = exc.headers.get("Retry-After", "") if exc.headers is not None else ""
    parsed = _float_or_none(header)
    return max(0.0, min(parsed if parsed is not None else 1.0, 5.0))
