"""Small stdlib Researka DB search client."""
from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from v5_memo.schemas import CorpusHit


class ResearkaSearchClient:
    """Synchronous client for `POST /api/v1/search`."""

    def __init__(self, *, base_url: str, token: str, timeout: float = 20.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token.strip()
        self._timeout = timeout

    @classmethod
    def from_env(cls) -> ResearkaSearchClient:
        return cls(
            base_url=os.environ.get("RESEARKA_DATABASE_URL", "https://database.researka.org"),
            token=os.environ.get("RESEARKA_DATABASE_TOKEN", ""),
        )

    def search(self, query: str, *, limit: int = 25) -> list[CorpusHit]:
        if not self._base_url or not self._token or not query.strip():
            return []
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


def _clean(value: object, *, limit: int) -> str:
    return " ".join(value.split())[:limit] if isinstance(value, str) else ""


def _int_or_none(value: object) -> int | None:
    if not isinstance(value, int | float | str | bytes | bytearray):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
