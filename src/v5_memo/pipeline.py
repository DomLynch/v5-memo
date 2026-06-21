"""End-to-end V5 memo pipeline."""
from __future__ import annotations

from collections.abc import Callable, Sequence

from v5_memo.binder import bind_receipts
from v5_memo.gate import meets_publish_bar, no_alpha_failure
from v5_memo.miner import mine_insights, query_anchor_terms
from v5_memo.retriever import CorpusSearcher, collect_seed_hits
from v5_memo.schemas import CorpusHit, InsightCandidate, MemoBuildError, MemoResult
from v5_memo.writer import render_memo

MemoWriter = Callable[[InsightCandidate, Sequence[CorpusHit]], str]
MemoSelector = Callable[
    [Sequence[InsightCandidate], Sequence[CorpusHit]], Sequence[InsightCandidate]
]
def build_alpha_memo(
    *,
    topic: str,
    seed_queries: Sequence[str],
    searcher: CorpusSearcher,
    memo_writer: MemoWriter = render_memo,
    memo_selector: MemoSelector | None = None,
    anchor_queries: Sequence[str] | None = None,
    min_alpha_tier: str = "publishable_alpha",
    per_query_limit: int = 25,
    max_hits: int = 100,
) -> MemoResult:
    """Build the best receipt-bound memo from seed queries."""
    hits = collect_seed_hits(
        searcher,
        seed_queries,
        per_query_limit=per_query_limit,
        max_hits=max_hits,
    )
    anchor_terms = query_anchor_terms(anchor_queries or ())
    if not anchor_terms:
        anchor_terms = query_anchor_terms(seed_queries)
    candidates: Sequence[InsightCandidate] = mine_insights(
        hits,
        topic=topic,
        required_anchor_terms=anchor_terms,
        include_discovery=min_alpha_tier == "discovery_seed",
    )
    candidates = _apply_selector(candidates, hits, memo_selector)
    for candidate in candidates:
        if not meets_publish_bar(candidate, min_alpha_tier):
            continue
        receipts = bind_receipts(candidate, hits)
        if receipts:
            return MemoResult(
                candidate=candidate,
                receipts=receipts,
                markdown=memo_writer(candidate, receipts),
            )
    raise MemoBuildError(
        no_alpha_failure(
            topic=topic,
            hits=hits,
            candidates=candidates,
            min_alpha_tier=min_alpha_tier,
        )
    )


def _apply_selector(
    candidates: Sequence[InsightCandidate],
    hits: Sequence[CorpusHit],
    selector: MemoSelector | None,
) -> Sequence[InsightCandidate]:
    if selector is None:
        return candidates
    by_receipts = {candidate.receipt_ids: candidate for candidate in candidates}
    selected: list[InsightCandidate] = []
    for candidate in selector(candidates, hits):
        original = by_receipts.get(candidate.receipt_ids)
        if original is not None and original not in selected:
            selected.append(original)
    return selected
