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
    """Synchronous client for Researka full-paper corpus search."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        timeout: float = 20.0,
        year_min: int = 1900,
        year_max: int = 2100,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token.strip()
        self._timeout = timeout
        self._year_min = year_min
        self._year_max = year_max

    @classmethod
    def from_env(cls) -> ResearkaSearchClient:
        return cls(
            base_url=os.environ.get("RESEARKA_DATABASE_URL", "https://database.researka.org"),
            token=os.environ.get("RESEARKA_DATABASE_TOKEN", ""),
        )

    def search(self, query: str, *, limit: int = 25) -> list[CorpusHit]:
        if not self._base_url or not self._token or not query.strip():
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
        except (HTTPError, URLError, TimeoutError, ValueError):
            return []
        return _parse_corpus_search_response(data)


def _parse_corpus_search_response(data: Any) -> list[CorpusHit]:
    if not isinstance(data, list):
        return []
    return [hit for item in data if (hit := _parse_paper_hit(item))]


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


def _float_or_none(value: object) -> float | None:
    if not isinstance(value, int | float | str | bytes | bytearray):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
