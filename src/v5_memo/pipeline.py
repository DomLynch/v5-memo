"""End-to-end V5 memo pipeline."""
from __future__ import annotations

from collections.abc import Callable, Sequence

from v5_memo.binder import bind_receipts
from v5_memo.gate import meets_publish_bar, memo_coverage_failure, no_alpha_failure
from v5_memo.miner import _tokens, mine_insights, query_anchor_terms
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
    min_shards_searched: int = 0,
    min_sources_searched: int = 0,
    min_search_passes: int = 0,
) -> MemoResult:
    """Build the best receipt-bound memo from seed queries."""
    hits = collect_seed_hits(
        searcher,
        seed_queries,
        per_query_limit=per_query_limit,
        max_hits=max_hits,
    )
    if anchor_queries is None:
        anchor_terms = _anchor_terms_for_queries(seed_queries)
        primary_anchor_terms = _primary_anchor_terms_for_single_query(seed_queries)
    else:
        anchor_terms = _anchor_terms_for_queries(anchor_queries)
        primary_anchor_terms = _primary_anchor_terms_for_single_query(anchor_queries)
    mined_candidates: list[InsightCandidate] = mine_insights(
        hits,
        topic=topic,
        required_anchor_terms=anchor_terms,
        include_discovery=min_alpha_tier == "discovery_seed",
        max_candidates=25 if memo_selector is not None else 8,
    )
    hits_by_id = {hit.hit_id: hit for hit in hits}
    publishable_candidates = [
        candidate
        for candidate in mined_candidates
        if meets_publish_bar(candidate, min_alpha_tier)
        and _candidate_preserves_primary_anchor(candidate, hits_by_id, primary_anchor_terms)
    ]
    candidates = _apply_selector(publishable_candidates, hits, memo_selector)
    coverage_failures: list[MemoBuildError] = []
    for candidate in candidates:
        receipts = bind_receipts(candidate, hits)
        if receipts:
            coverage_failure = memo_coverage_failure(
                topic=topic,
                receipts=receipts,
                min_shards_searched=min_shards_searched,
                min_sources_searched=min_sources_searched,
                min_search_passes=min_search_passes,
                min_abstract_receipts=1 if min_alpha_tier == "elite_alpha" else 0,
            )
            if coverage_failure is not None:
                coverage_failures.append(MemoBuildError(coverage_failure))
                continue
            return MemoResult(
                candidate=candidate,
                receipts=receipts,
                markdown=memo_writer(candidate, receipts),
            )
    if coverage_failures:
        raise coverage_failures[0]
    raise MemoBuildError(
        no_alpha_failure(
            topic=topic,
            hits=hits,
            candidates=candidates,
            min_alpha_tier=min_alpha_tier,
            mined_candidates=mined_candidates,
        )
    )


def _apply_selector(
    candidates: Sequence[InsightCandidate],
    hits: Sequence[CorpusHit],
    selector: MemoSelector | None,
) -> Sequence[InsightCandidate]:
    if selector is None:
        return candidates
    candidates = _selector_slate(candidates)
    by_receipts = {candidate.receipt_ids: candidate for candidate in candidates}
    selected: list[InsightCandidate] = []
    for candidate in selector(candidates, hits):
        original = by_receipts.get(candidate.receipt_ids)
        if original is not None and original not in selected:
            selected.append(original)
    return selected or candidates


def _selector_slate(candidates: Sequence[InsightCandidate]) -> Sequence[InsightCandidate]:
    """Preserve rank while surfacing one strong candidate per evidence shape."""
    selected: list[InsightCandidate] = []
    seen: set[tuple[str, ...]] = set()

    def add(candidate: InsightCandidate) -> None:
        if candidate.receipt_ids not in seen:
            seen.add(candidate.receipt_ids)
            selected.append(candidate)

    for candidate in candidates[:3]:
        add(candidate)
    shape_keys = (
        "shape:expectation_reversal",
        "shape:promise_outcome_reversal",
        "shape:directional_reversal",
        "shape:boundary_condition",
        "shape:denominator_split",
        "shape:timing_split",
        "shape:measurement_mismatch",
    )
    for shape in shape_keys:
        match = next((candidate for candidate in candidates if shape in candidate.reasons), None)
        if match is not None:
            add(match)
    for candidate in candidates:
        add(candidate)
    return selected


def _anchor_terms_for_queries(queries: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for query in queries:
        for term in query_anchor_terms([query], limit=2):
            if term not in seen:
                seen.add(term)
                out.append(term)
    return tuple(out)


def _primary_anchor_terms_for_single_query(queries: Sequence[str]) -> frozenset[str]:
    if len(queries) != 1:
        return frozenset()
    return frozenset(query_anchor_terms([queries[0]], limit=1))


def _candidate_preserves_primary_anchor(
    candidate: InsightCandidate,
    hits_by_id: dict[str, CorpusHit],
    primary_anchor_terms: frozenset[str],
) -> bool:
    if not primary_anchor_terms:
        return True
    receipts = [
        hits_by_id[receipt_id]
        for receipt_id in candidate.receipt_ids
        if receipt_id in hits_by_id
    ]
    return bool(receipts) and all(
        _tokens(hit.text) & primary_anchor_terms
        for hit in receipts
    )
