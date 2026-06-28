"""Search fan-out helpers for V5 memo."""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Protocol

from v5_memo.schemas import CorpusHit

_SEED_QUERY_DROP = {
    "a",
    "an",
    "and",
    "article",
    "articles",
    "evidence",
    "or",
    "paper",
    "papers",
    "research",
    "study",
    "studies",
    "the",
}


class CorpusSearcher(Protocol):
    def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]: ...


def collect_seed_hits(
    searcher: CorpusSearcher,
    seed_queries: Sequence[str],
    *,
    per_query_limit: int = 25,
    max_hits: int = 100,
    stop_when: Callable[[Sequence[CorpusHit]], bool] | None = None,
) -> list[CorpusHit]:
    """Search multiple seeds and dedupe before mining insights."""
    seed_queries = _dedupe_seed_queries(seed_queries)
    seen: dict[str, int] = {}
    out: list[CorpusHit] = []
    query_limit = min(per_query_limit, max(1, -(-max_hits // max(1, len(seed_queries)))))
    for query in seed_queries:
        try:
            hits = searcher.search(query, limit=query_limit)
        except RuntimeError as exc:
            if str(exc).startswith("Full raw corpus search coverage too narrow"):
                if out:
                    break
                raise
            continue
        for hit in hits:
            key = hit.source_key
            if key in seen:
                existing = out[seen[key]]
                raw_queries = existing.metadata.get("seed_queries", ())
                queries: tuple[str, ...] = (
                    raw_queries if isinstance(raw_queries, tuple) else ()
                )
                if query not in queries:
                    out[seen[key]] = replace(
                        existing,
                        metadata={**existing.metadata, "seed_queries": (*queries, query)},
                    )
                continue
            seen[key] = len(out)
            out.append(replace(hit, metadata={**hit.metadata, "seed_queries": (query,)}))
            if len(out) >= max_hits:
                return out
        if stop_when is not None and stop_when(out):
            return out
    return out


def _dedupe_seed_queries(seed_queries: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for query in seed_queries:
        key = _seed_query_key(query)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(" ".join(query.split()))
    return out


def _seed_query_key(query: str) -> tuple[str, ...]:
    terms = {
        token
        for token in re.findall(r"[a-z0-9+]+", query.casefold())
        if len(token) > 1 and token not in _SEED_QUERY_DROP
    }
    return tuple(sorted(terms))
