"""Small stdlib Researka DB search client."""
from __future__ import annotations

import json
import os
import re
from html import unescape
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from v5_memo.schemas import CorpusHit


class ResearkaSearchClient:
    """Synchronous client for Researka corpus search.

    `/api/v1/search` is tried first for paper-level corpus hits. If that surface
    returns no hits, the client falls back to the live Tier2 facts search, which
    is the currently verified Researka evidence path.
    """

    def __init__(
        self, *, base_url: str, token: str, domain: str = "longevity", timeout: float = 20.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token.strip()
        self._domain = domain.strip() or "longevity"
        self._timeout = timeout

    @classmethod
    def from_env(cls, *, domain: str | None = None) -> ResearkaSearchClient:
        return cls(
            base_url=os.environ.get("RESEARKA_DATABASE_URL", "https://database.researka.org"),
            token=os.environ.get("RESEARKA_DATABASE_TOKEN", ""),
            domain=domain or os.environ.get("V5_MEMO_DOMAIN", "longevity"),
        )

    def search(self, query: str, *, limit: int = 25) -> list[CorpusHit]:
        if not self._base_url or not self._token or not query.strip():
            return []
        hits = self._search_papers(query, limit=limit)
        return hits or self._search_tier2_facts(query, limit=limit)

    def _search_papers(self, query: str, *, limit: int) -> list[CorpusHit]:
        payload = {
            "query": query[:1024],
            "established_k": max(1, limit * 4 // 10),
            "discovery_k": max(1, limit // 10),
            "semantic_k": max(1, limit * 5 // 10),
        }
        request = Request(
            f"{self._base_url}/api/v1/search",
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
        except (HTTPError, URLError, TimeoutError, ValueError):
            return []
        return _parse_search_response(data)

    def _search_tier2_facts(self, query: str, *, limit: int) -> list[CorpusHit]:
        payload = {
            "domain": self._domain,
            "query": query[:1024],
            "top_k": limit,
            "min_confidence": "medium",
            "numeric_only": False,
        }
        request = Request(
            f"{self._base_url}/api/v1/tier2/facts/search",
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
        except (HTTPError, URLError, TimeoutError, ValueError):
            return []
        if not isinstance(data, list):
            return []
        return [hit for item in data if (hit := _parse_tier2_fact(item))]


def _parse_search_response(data: Any) -> list[CorpusHit]:
    if isinstance(data, list):
        return [hit for item in data if (hit := _parse_item(item, lane="search"))]
    if not isinstance(data, dict):
        return []
    hits: list[CorpusHit] = []
    for lane in ("established", "discovery", "semantic", "results"):
        items = data.get(lane)
        if not isinstance(items, list):
            continue
        hits.extend(hit for item in items if (hit := _parse_item(item, lane=lane)))
    return hits


def _parse_item(item: Any, *, lane: str) -> CorpusHit | None:
    if not isinstance(item, dict):
        return None
    title = _clean(item.get("title"), limit=500)
    if not title:
        return None
    doi = _clean(item.get("doi") or item.get("paper_id") or item.get("id"), limit=256) or None
    hit_id = doi or _clean(item.get("pmid") or item.get("paper_id") or title, limit=256)
    return CorpusHit(
        hit_id=hit_id,
        title=title,
        abstract=_clean(item.get("abstract") or item.get("summary"), limit=4000),
        source=f"researka:{lane}",
        year=_int_or_none(item.get("year") or item.get("publication_year")),
        url=_clean(item.get("url") or item.get("link"), limit=500),
        doi=doi,
        venue=_clean(item.get("venue") or item.get("journal") or item.get("journal_name"), limit=200)
        or None,
        metadata={"pmid": _clean(item.get("pmid"), limit=64)},
    )


def _parse_tier2_fact(item: Any) -> CorpusHit | None:
    if not isinstance(item, dict):
        return None
    paper_raw = item.get("paper")
    paper = paper_raw if isinstance(paper_raw, dict) else {}
    title = _clean(paper.get("title") or item.get("canonical_phrase"), limit=500)
    if not title:
        return None
    fact_id = _clean(item.get("id") or item.get("fact_id"), limit=256)
    doi = _clean(paper.get("doi") or item.get("paper_id"), limit=256) or None
    claim = _clean(item.get("canonical_phrase"), limit=800)
    excerpt = _clean(item.get("source_excerpt"), limit=2000)
    abstract = " ".join(part for part in (claim, excerpt) if part)
    return CorpusHit(
        hit_id=fact_id or doi or title,
        title=title,
        abstract=abstract,
        source="researka:tier2",
        year=_int_or_none(paper.get("publication_year") or item.get("canonical_year")),
        url=f"https://doi.org/{doi}" if doi else "",
        doi=doi,
        venue=_clean(paper.get("journal_name"), limit=200) or None,
        metadata={
            "fact_id": fact_id,
            "paper_id": _clean(item.get("paper_id"), limit=256),
            "confidence": _clean(item.get("extraction_confidence"), limit=64),
        },
    )


def _clean(value: object, *, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", unescape(value))
    return " ".join(text.split())[:limit]


def _int_or_none(value: object) -> int | None:
    if not isinstance(value, int | float | str | bytes | bytearray):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
