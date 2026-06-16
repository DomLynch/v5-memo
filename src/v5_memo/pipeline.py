"""End-to-end V5 memo pipeline."""
from __future__ import annotations

from collections.abc import Callable, Sequence

from v5_memo.binder import bind_receipts
from v5_memo.miner import mine_insights
from v5_memo.retriever import CorpusSearcher, collect_seed_hits
from v5_memo.schemas import CorpusHit, InsightCandidate, MemoResult
from v5_memo.writer import render_memo

MemoWriter = Callable[[InsightCandidate, Sequence[CorpusHit]], str]


def build_alpha_memo(
    *,
    topic: str,
    seed_queries: Sequence[str],
    searcher: CorpusSearcher,
    memo_writer: MemoWriter = render_memo,
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
    candidates = mine_insights(hits, topic=topic)
    for candidate in candidates:
        receipts = bind_receipts(candidate, hits)
        if receipts:
            return MemoResult(
                candidate=candidate,
                receipts=receipts,
                markdown=memo_writer(candidate, receipts),
            )
    raise ValueError("no receipt-bound alpha memo candidate found")
