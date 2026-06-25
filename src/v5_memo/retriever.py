"""Search fan-out helpers for V5 memo."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
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
    seen: dict[str, int] = {}
    out: list[CorpusHit] = []
    query_limit = min(per_query_limit, max(1, -(-max_hits // max(1, len(seed_queries)))))
    for query in seed_queries:
        try:
            hits = searcher.search(query, limit=query_limit)
        except RuntimeError:
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
    return out
