"""Search query shapes and a small fullraw client."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from http.client import RemoteDisconnected
from typing import Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from v5_memo.client import FullRawCorpusSearchClient


class HttpResponse(Protocol):
    def __enter__(self) -> HttpResponse: ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...

    def read(self) -> bytes: ...


class RequestOpener(Protocol):
    def __call__(self, request: Request, timeout: float) -> HttpResponse: ...


@dataclass(frozen=True, slots=True)
class Paper:
    paper_id: str
    title: str
    abstract: str
    source: str
    year: int | None = None
    doi: str = ""
    url: str = ""
    venue: str = ""

    @property
    def text(self) -> str:
        return f"{self.title} {self.abstract} {self.venue}"

    @property
    def key(self) -> str:
        if self.doi:
            return f"doi:{self.doi.casefold()}"
        return f"{self.source}:{_norm_title(self.title)}:{self.year or ''}"


@dataclass(frozen=True, slots=True)
class CoverageReceipt:
    hits: int = 0
    shards_searched: int = 0
    shards_total: int = 0
    papers_searched: int = 0
    papers_total: int = 0
    sources_searched: tuple[str, ...] = ()
    partial: bool = False
    error: str = ""


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    papers: tuple[Paper, ...]
    receipt: CoverageReceipt


class V5FullrawSearchClient:
    """Adapter over the proven V5 fullraw search/rerank client."""

    def __init__(self, client: FullRawCorpusSearchClient) -> None:
        self._client = client

    @classmethod
    def from_env(cls) -> V5FullrawSearchClient:
        return cls(FullRawCorpusSearchClient.from_env(strict=False))

    def search(self, query: str, *, limit: int = 25) -> SearchResult:
        hits = self._client.search(query, limit=limit)
        papers = tuple(
            Paper(str(hit.hit_id), hit.title, hit.abstract, hit.source, hit.year, hit.doi or "", hit.url, hit.venue or "")
            for hit in hits
        )
        return SearchResult(
            query=query,
            papers=papers,
            receipt=CoverageReceipt(hits=len(papers), sources_searched=tuple(sorted({hit.source for hit in hits}))),
        )


class FullrawSearchClient:
    """Tiny POST client for the 5TB-backed fullraw search endpoint."""

    def __init__(
        self,
        *,
        search_url: str,
        token: str = "",
        timeout: float = 180.0,
        sweep_wait_seconds: float = 0.0,
        sweep_poll_seconds: float = 10.0,
        opener: RequestOpener | None = None,
    ) -> None:
        self.search_url = search_url.strip()
        self.search_urls = _search_urls(search_url)
        self.token = token.strip()
        self.timeout = timeout
        self.sweep_wait_seconds = sweep_wait_seconds
        self.sweep_poll_seconds = sweep_poll_seconds
        self._opener = opener or cast(RequestOpener, urlopen)

    @classmethod
    def from_env(cls) -> FullrawSearchClient | V5FullrawSearchClient:
        if os.environ.get("V6_FULLRAW_NATIVE") != "1":
            return V5FullrawSearchClient.from_env()
        return cls(
            search_url=os.environ.get(
                "V6_FULLRAW_SEARCH_URL",
                os.environ.get("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", ""),
            ),
            token=os.environ.get(
                "V6_FULLRAW_TOKEN",
                os.environ.get("V5_MEMO_FULL_RAW_CORPUS_TOKEN", ""),
            ),
            timeout=float(os.environ.get("V6_FULLRAW_TIMEOUT", "180")),
            sweep_wait_seconds=float(os.environ.get(
                "V6_FULLRAW_SWEEP_WAIT_SECONDS",
                os.environ.get("V5_MEMO_FULL_RAW_SWEEP_WAIT_SECONDS", "0"),
            )),
            sweep_poll_seconds=float(os.environ.get(
                "V6_FULLRAW_SWEEP_POLL_SECONDS",
                os.environ.get("V5_MEMO_FULL_RAW_SWEEP_POLL_SECONDS", "10"),
            )),
        )

    def search(self, query: str, *, limit: int = 25) -> SearchResult:
        if not self.search_urls:
            raise RuntimeError("V6_FULLRAW_SEARCH_URL or V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL is required")
        last = SearchResult(query=query, papers=(), receipt=CoverageReceipt())
        for variant in _query_variants(query):
            for search_url in self.search_urls:
                try:
                    result = self._search_once(variant, limit=limit, search_url=search_url)
                except (OSError, RemoteDisconnected, TimeoutError, URLError) as exc:
                    last = SearchResult(
                        query=variant,
                        papers=(),
                        receipt=CoverageReceipt(error=f"{type(exc).__name__}: {exc}"),
                    )
                    continue
                last = result
                if result.papers and _result_matches_query(result, variant):
                    return result
        return last

    def _search_once(self, query: str, *, limit: int, search_url: str) -> SearchResult:
        payload = {
            "query": query[:1024],
            "limit": max(1, min(limit, 200)),
            "top_k": max(1, min(limit, 200)),
            "queue_if_missing": True,
            "corpus": "full_raw_5tb",
            "timeout_seconds": self.timeout,
        }
        headers = {"Content-Type": "application/json", "User-Agent": "v6-alpha-memo/0.1"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            data = self._post(search_url, payload, headers)
        except HTTPError as exc:
            data = _http_error_json(exc)
            if not _is_incomplete_coverage(data) or not self.sweep_wait_seconds:
                raise
            cached = self._wait_for_sweep_hit(search_url, payload, headers)
            if cached is None:
                raise
            data = cached
        parsed: list[Paper] = []
        for item in _items(data):
            paper = _parse_paper(item)
            if paper is not None:
                parsed.append(paper)
        papers = tuple(parsed)
        return SearchResult(query=query, papers=papers, receipt=_receipt(data, hits=len(papers)))

    def _post(self, search_url: str, payload: dict[str, object], headers: dict[str, str]) -> object:
        request = Request(
            search_url,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with self._opener(request, timeout=self.timeout + 5) as response:
            return json.loads(response.read().decode())

    def _wait_for_sweep_hit(
        self,
        search_url: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> object | None:
        deadline = time.monotonic() + self.sweep_wait_seconds
        cache_payload = {**payload, "cache_only": True, "queue_if_missing": True}
        while time.monotonic() < deadline:
            data = self._post(search_url, cache_payload, headers)
            status = _async_status(data)
            if status == "hit":
                return data
            if status in {"disabled", "error", "failed"}:
                return None
            time.sleep(min(max(self.sweep_poll_seconds, 0.1), max(deadline - time.monotonic(), 0.1)))
        return None


def query_shapes(seed: str, *, limit: int = 8) -> tuple[str, ...]:
    """Turn a domain/topic seed into targeted novelty-search shapes."""
    seed = " ".join(seed.split())
    words = seed.split()
    templates = (
        "{seed} randomized placebo no effect primary endpoint",
        "{seed} baseline subgroup high low response",
        "{seed} endpoint split randomized trial placebo",
        "{seed} intervention opposite endpoint boundary condition",
        "{seed} field experiment intervention null effect",
        "{seed} benchmark improvement replication failure",
        "{seed} mechanism model human failed translation",
        "{seed} same intervention different modality adaptation",
    )
    base = (
        " ".join(words[:4]),
        " ".join((words[0], "supplementation", "improves", words[-3], "deficiency", "oxidative", "stress")) if len(words) > 4 else seed,
        " ".join(("healthy", "older", "adults", *words[1:3], words[-3], "redox")) if len(words) > 5 else seed,
    )
    queries = [*base, *(template.format(seed=seed) for template in templates if seed)]
    return tuple(dict.fromkeys(queries))[: max(1, limit)]


def _http_error_json(exc: HTTPError) -> object:
    try:
        return json.loads(exc.read().decode())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _is_incomplete_coverage(data: object) -> bool:
    return isinstance(data, dict) and data.get("error") == "shard coverage incomplete"


def _async_status(data: object) -> str:
    if not isinstance(data, dict):
        return ""
    meta = data.get("meta")
    if not isinstance(meta, dict):
        return ""
    sweep = meta.get("async_sweep")
    if not isinstance(sweep, dict):
        return ""
    return str(sweep.get("status") or "")


def merge_results(results: tuple[SearchResult, ...]) -> tuple[Paper, ...]:
    seen: set[str] = set()
    title_index: dict[str, int] = {}
    papers: list[Paper] = []
    for result in results:
        for paper in result.papers:
            if paper.key not in seen:
                title_key = _norm_title(paper.title)
                if title_key in title_index:
                    idx = title_index[title_key]
                    if _paper_rank(paper) > _paper_rank(papers[idx]):
                        seen.discard(papers[idx].key)
                        papers[idx] = paper
                        seen.add(paper.key)
                    continue
                seen.add(paper.key)
                title_index[title_key] = len(papers)
                papers.append(paper)
    return tuple(papers)


def _paper_rank(paper: Paper) -> int:
    text = f"{paper.title} {paper.abstract} {paper.source} {paper.venue} {paper.doi}".casefold()
    score = int(bool(paper.doi)) + int(bool(paper.year)) * 2
    if any(marker in text for marker in ("10.1101/", "arxiv", "biorxiv", "medrxiv", "preprint")):
        score -= 5
    if any(marker in text for marker in ("commentary", "editorial", "in brief", "research highlight")):
        score -= 3
    return score


def _query_variants(query: str) -> tuple[str, ...]:
    raw_words = re.findall(r"[a-z][a-z0-9]{2,}", query.casefold().replace("-", " "))
    words = [word for word in raw_words if word not in _QUERY_DROP]
    context_words = [word for word in raw_words if word in _QUERY_CONTEXT_KEEP or word not in _QUERY_DROP]
    variants = [" ".join(query.split())]
    if context_words:
        variants.append(" ".join(context_words))
    context = [word for word in raw_words if word in _QUERY_CONTEXT_KEEP]
    if context and words:
        variants.append(f"{context[0]} {max(words, key=len)}")
    if words:
        variants.append(" ".join(words))
    if len(words) >= 2:
        variants.append(" ".join(words[:2]))
    if len(words) >= 3:
        variants.append(" ".join(words[:3]))
    return tuple(dict.fromkeys(variant for variant in variants if variant))


def _search_urls(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(url.strip() for url in value.split(",") if url.strip()))


_QUERY_DROP = frozenset({
    "adaptation", "adult", "adults", "aging", "clinical", "condition", "controlled",
    "effect", "endpoint", "expected", "failure", "healthy", "human", "improved",
    "intervention", "mechanism", "mismatch", "model", "modality", "null", "older",
    "opposite", "outcome", "placebo", "protocol", "randomized", "result", "same",
    "subgroup", "translation", "trial",
})
_QUERY_CONTEXT_KEEP = frozenset({"adult", "adults", "healthy", "human", "humans", "older", "participants", "patient", "patients", "workers"})
_PUBMED_BACKFILL_LIMIT = 4


def _result_matches_query(result: SearchResult, query: str) -> bool:
    anchors = frozenset(
        word for word in re.findall(r"[a-z][a-z0-9]{2,}", query.casefold().replace("-", " ")) if word not in _QUERY_DROP
    )
    needed = 1 if len(anchors) < 3 else 2
    return not anchors or any(len(_paper_query_terms(paper) & anchors) >= needed for paper in result.papers[:5])


def _paper_query_terms(paper: Paper) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9]{2,}", paper.text.casefold().replace("-", " ")))


def _items(data: object) -> list[object]:
    if not isinstance(data, dict):
        return []
    raw = data.get("results", data.get("hits", []))
    return raw if isinstance(raw, list) else []


def _parse_paper(item: object) -> Paper | None:
    if not isinstance(item, dict):
        return None
    title = _clean(item.get("title") or item.get("display_name") or item.get("name"))
    if not title:
        return None
    doi = _doi(item.get("doi"))
    paper_id = _clean(item.get("id") or item.get("openalex_id") or doi or title)
    abstract = _clean(item.get("abstract") or item.get("abstract_text") or item.get("description"), limit=4000) or _inverted_abstract(item.get("abstract_inverted_index"))
    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        source=_clean(item.get("source") or item.get("raw_source") or item.get("provider")) or "fullraw",
        year=_int(item.get("year") or item.get("publication_year")),
        doi=doi,
        url=_clean(item.get("url")) or (f"https://doi.org/{doi}" if doi else ""),
        venue=_clean(item.get("venue") or item.get("journal") or item.get("source_name")),
    )


def _receipt(data: object, *, hits: int) -> CoverageReceipt:
    if not isinstance(data, dict):
        return CoverageReceipt(hits=hits)
    meta = data.get("meta")
    meta = meta if isinstance(meta, dict) else {}
    shard = meta.get("shard_receipt")
    shard = shard if isinstance(shard, dict) else {}
    return CoverageReceipt(
        hits=hits,
        shards_searched=_int(shard.get("shards_searched")) or 0,
        shards_total=_int(shard.get("shards_total")) or 0,
        papers_searched=_int(shard.get("papers_searched")) or 0,
        papers_total=_int(shard.get("papers_total")) or 0,
        sources_searched=_sources(shard.get("sources_searched")),
        partial=bool(shard.get("partial_shard_search") or meta.get("partial")),
    )


def _sources(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        return tuple(str(key) for key in value)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    return ()


def _inverted_abstract(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for word, raw_indexes in value.items():
        if isinstance(word, str) and isinstance(raw_indexes, list):
            positions.extend((idx, word) for idx in raw_indexes if isinstance(idx, int))
    return " ".join(word for _, word in sorted(positions))[:4000]


def _clean(value: object, *, limit: int = 500) -> str:
    return " ".join(str(value or "").split())[:limit]


def _doi(value: object) -> str:
    text = _clean(value, limit=250).removeprefix("https://doi.org/").removeprefix("doi:")
    return text.casefold()


def _int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if not isinstance(value, str):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _norm_title(title: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", title.casefold()))
