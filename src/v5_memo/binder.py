"""Bind alpha candidates to receipts before prose is rendered."""
from __future__ import annotations

from collections.abc import Sequence

from v5_memo.schemas import CorpusHit, InsightCandidate


def bind_receipts(
    candidate: InsightCandidate,
    hits: Sequence[CorpusHit],
    *,
    min_unique_sources: int = 2,
) -> tuple[CorpusHit, ...]:
    """Return candidate receipts only when source diversity is sufficient."""
    by_id = {hit.hit_id: hit for hit in hits}
    receipts = tuple(
        by_id[hit_id] for hit_id in candidate.receipt_ids if hit_id in by_id
    )
    if len(receipts) != len(candidate.receipt_ids):
        return ()
    if len({hit.source_key for hit in receipts}) < min_unique_sources:
        return ()
    return receipts
