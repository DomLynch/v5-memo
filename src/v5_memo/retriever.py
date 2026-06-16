"""Search fan-out helpers for V5 memo."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from v5_memo.schemas import CorpusHit


class CorpusSearcher(Protocol):
    def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]: ...


def collect_seed_hits(
    searcher: CorpusSearcher,
    seed_queries: Sequence[str],
    *,
    per_query_limit: int = 25,
    max_hits: int = 100,
) -> list[CorpusHit]:
    """Search multiple seeds and dedupe before mining insights."""
    seen: set[str] = set()
    out: list[CorpusHit] = []
    for query in seed_queries:
        for hit in searcher.search(query, limit=per_query_limit):
            key = hit.source_key
            if key in seen:
                continue
            seen.add(key)
            out.append(hit)
            if len(out) >= max_hits:
                return out
    return out
