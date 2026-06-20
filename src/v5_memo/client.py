"""Small stdlib search clients for OpenAlex, Researka, and full raw corpus search."""
from __future__ import annotations

import json
import math
import os
import re
import urllib.parse
from collections.abc import Sequence
from dataclasses import replace
from html import unescape
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from v5_memo.schemas import CorpusHit


class SearchBackendError(RuntimeError):
    """Raised when strict search mode cannot reach or parse a backend."""


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
            max_variants=_int_env("V5_MEMO_OPENALEX_MAX_VARIANTS", 2 if strict else 8),
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
        try:
            with urlopen(request, timeout=self._timeout) as response:
                data: Any = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            if self._strict:
                raise SearchBackendError(f"OpenAlex search failed: {exc}") from exc
            return []
        return _parse_openalex_response(data)


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
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
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
        strict: bool = False,
    ) -> None:
        self._search_url = search_url.strip()
        self._token = token.strip()
        self._timeout = timeout
        self._year_min = year_min
        self._year_max = year_max
        self._strict = strict

    @classmethod
    def from_env(cls, *, strict: bool = False) -> FullRawCorpusSearchClient:
        return cls(
            search_url=os.environ.get("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", ""),
            token=os.environ.get("V5_MEMO_FULL_RAW_CORPUS_TOKEN", ""),
            timeout=_float_env("V5_MEMO_FULL_RAW_CORPUS_TIMEOUT", 45.0),
            strict=strict,
        )

    @property
    def configured(self) -> bool:
        return bool(self._search_url)

    def search(self, query: str, *, limit: int = 25) -> list[CorpusHit]:
        if not self._search_url or not query.strip():
            return []
        payload = {
            "query": query[:1024],
            "limit": max(1, min(limit, 200)),
            "top_k": max(1, min(limit, 200)),
            "year_min": self._year_min,
            "year_max": self._year_max,
            "corpus": "full_raw_450m_plus",
            "timeout_seconds": self._timeout,
        }
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
        try:
            with urlopen(request, timeout=self._timeout) as response:
                data: Any = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            if self._strict:
                raise SearchBackendError(f"Full raw corpus search failed: {exc}") from exc
            return []
        return _parse_full_raw_search_response(data)


class HybridCorpusSearchClient:
    """Merge multiple corpus search surfaces behind one searcher contract."""

    def __init__(self, searchers: Sequence[object]) -> None:
        self._searchers = searchers

    def search(self, query: str, *, limit: int = 25) -> list[CorpusHit]:
        best: dict[str, CorpusHit] = {}
        for searcher in self._searchers:
            search = getattr(searcher, "search", None)
            if not callable(search):
                continue
            for hit in search(query, limit=limit):
                best.setdefault(hit.source_key, hit)
                if len(best) >= limit:
                    break
        return list(best.values())[:limit]


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
    return [
        hit
        for item in items
        if (hit := _parse_full_raw_paper_hit(item, match_count=match_count)) is not None
    ]


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
    doi = _normalize_doi(item.get("doi"))
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
            "cited_by_count": _int_or_none(item.get("cited_by_count")),
            "query_match_count": match_count,
        },
    )


def _parse_paper_hit(item: Any) -> CorpusHit | None:
    if not isinstance(item, dict):
        return None
    title = _clean(item.get("title"), limit=500)
    if not title:
        return None
    doi = _clean(item.get("doi"), limit=256) or None
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
            "cited_by_count": _int_or_none(item.get("cited_by_count")),
            "similarity_score": _float_or_none(item.get("similarity_score")),
        },
    )


def _parse_full_raw_paper_hit(item: Any, *, match_count: int | None) -> CorpusHit | None:
    if not isinstance(item, dict):
        return None
    title = _clean(item.get("title") or item.get("display_name") or item.get("name"), limit=500)
    if not title:
        return None
    doi = _normalize_doi(item.get("doi"))
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
        year=_int_or_none(item.get("year") or item.get("publication_year")),
        url=f"https://doi.org/{doi}" if doi else url or openalex_id,
        doi=doi,
        venue=venue,
        metadata={
            "pmid": pmid,
            "pmcid": pmcid,
            "openalex_id": openalex_id,
            "semantic_scholar_id": s2_id,
            "arxiv_id": arxiv_id,
            "raw_source": origin,
            "cited_by_count": _int_or_none(item.get("cited_by_count") or item.get("citation_count")),
            "score": _float_or_none(item.get("score") or item.get("search_score")),
            "query_match_count": match_count,
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


def _rerank_score(
    hit: CorpusHit,
    *,
    seed_terms: tuple[str, ...],
    variant_terms: tuple[str, ...],
    rank: int,
) -> float:
    text = hit.text.casefold()
    seed_coverage = _coverage(seed_terms, text)
    variant_coverage = _coverage(variant_terms, text)
    cited = hit.metadata.get("cited_by_count")
    citation_score = math.log10(max(0, cited) + 1) if isinstance(cited, (int, float)) else 0.0
    return (seed_coverage * 70.0) + (variant_coverage * 20.0) + (citation_score * 4.0) - rank


def _coverage(terms: tuple[str, ...], text: str) -> float:
    if not terms:
        return 0.0
    return sum(1 for term in terms if term in text) / len(terms)


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
    parsed = _float_or_none(os.environ.get(name, ""))
    return parsed if parsed is not None else default


def _int_env(name: str, default: int) -> int:
    parsed = _int_or_none(os.environ.get(name, ""))
    return parsed if parsed is not None else default
