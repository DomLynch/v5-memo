"""End-to-end V5 memo pipeline."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from v5_memo.binder import bind_receipts
from v5_memo.gate import (
    candidate_publish_blocker,
    meets_publish_bar,
    memo_coverage_failure,
    no_alpha_failure,
)
from v5_memo.miner import _tokens, mine_insights, query_anchor_terms
from v5_memo.retriever import CorpusSearcher, collect_seed_hits
from v5_memo.schemas import CorpusHit, InsightCandidate, MemoBuildError, MemoResult
from v5_memo.writer import render_memo

MemoWriter = Callable[[InsightCandidate, Sequence[CorpusHit]], str]
MemoSelector = Callable[
    [Sequence[InsightCandidate], Sequence[CorpusHit]], Sequence[InsightCandidate]
]
_PROXY_CONTEXT_OUTCOME_TERMS = frozenset({
    "acute",
    "damage",
    "delayed",
    "early",
    "immediate",
    "inflammation",
    "pain",
    "short",
    "stress",
})
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
    require_publish_quality: bool = False,
) -> MemoResult:
    """Build the best receipt-bound memo from seed queries."""
    if anchor_queries is None:
        anchor_terms = _anchor_terms_for_queries(seed_queries)
        primary_anchor_terms = _primary_anchor_terms_for_single_query(seed_queries)
    else:
        anchor_terms = _anchor_terms_for_queries(anchor_queries)
        primary_anchor_terms = _primary_anchor_terms_for_single_query(anchor_queries)

    def has_publishable_candidate(partial_hits: Sequence[CorpusHit]) -> bool:
        mined = mine_insights(
            partial_hits,
            topic=topic,
            required_anchor_terms=anchor_terms,
            include_discovery=min_alpha_tier == "discovery_seed",
            max_candidates=30 if memo_selector is not None else 12,
        )
        return bool(_publishable_candidates(
            mined,
            partial_hits,
            min_alpha_tier,
            primary_anchor_terms,
            require_publish_quality=require_publish_quality,
        ))

    hits = collect_seed_hits(
        searcher,
        seed_queries,
        per_query_limit=per_query_limit,
        max_hits=max_hits,
        stop_when=has_publishable_candidate if len(seed_queries) > 1 else None,
    )
    mined_candidates: list[InsightCandidate] = mine_insights(
        hits,
        topic=topic,
        required_anchor_terms=anchor_terms,
        include_discovery=min_alpha_tier == "discovery_seed",
        max_candidates=30 if memo_selector is not None else 12,
    )
    publishable_candidates = _publishable_candidates(
        mined_candidates,
        hits,
        min_alpha_tier,
        primary_anchor_terms,
        require_publish_quality=require_publish_quality,
    )
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
            seed_queries=seed_queries,
            anchor_terms=anchor_terms,
        )
    )


def _publishable_candidates(
    mined_candidates: Sequence[InsightCandidate],
    hits: Sequence[CorpusHit],
    min_alpha_tier: str,
    primary_anchor_terms: frozenset[str],
    *,
    require_publish_quality: bool = False,
) -> list[InsightCandidate]:
    hits_by_id = {hit.hit_id: hit for hit in hits}
    out: list[InsightCandidate] = []
    for candidate in mined_candidates:
        if not meets_publish_bar(candidate, min_alpha_tier):
            continue
        if not _candidate_preserves_primary_anchor(candidate, hits_by_id, primary_anchor_terms):
            continue
        if require_publish_quality:
            candidate = _drop_publish_context_receipts(candidate)
            if candidate_publish_blocker(candidate) is not None:
                continue
        out.append(candidate)
    return out


def _drop_publish_context_receipts(candidate: InsightCandidate) -> InsightCandidate:
    candidate = _drop_weak_context_receipts(candidate)
    return _drop_optional_proxy_context_receipts(candidate)


def _drop_weak_context_receipts(candidate: InsightCandidate) -> InsightCandidate:
    blocker = candidate_publish_blocker(candidate)
    if not blocker or blocker.get("error") != "weak_context_receipts":
        return candidate
    raw_receipt_ids = blocker.get("receipt_ids", ())
    if not isinstance(raw_receipt_ids, Sequence) or isinstance(raw_receipt_ids, str):
        return candidate
    drop_ids = {receipt_id for receipt_id in raw_receipt_ids if isinstance(receipt_id, str)}
    if not drop_ids:
        return candidate
    return _drop_receipts(candidate, drop_ids)


def _drop_optional_proxy_context_receipts(candidate: InsightCandidate) -> InsightCandidate:
    drop_ids = {
        card.receipt_id
        for card in candidate.claim_cards
        if card.role == "boundary"
        and (
            card.direction == "proxy"
            or bool(set(card.outcome.split("/")) & _PROXY_CONTEXT_OUTCOME_TERMS)
        )
    }
    if not drop_ids:
        return candidate
    trimmed = _drop_receipts(candidate, drop_ids)
    return trimmed if candidate_publish_blocker(trimmed) is None else candidate


def _drop_receipts(candidate: InsightCandidate, receipt_ids: set[str]) -> InsightCandidate:
    return replace(
        candidate,
        receipt_ids=tuple(receipt_id for receipt_id in candidate.receipt_ids if receipt_id not in receipt_ids),
        receipt_roles=tuple(role for role in candidate.receipt_roles if role.receipt_id not in receipt_ids),
        claim_cards=tuple(card for card in candidate.claim_cards if card.receipt_id not in receipt_ids),
        evidence_graph=tuple(node for node in candidate.evidence_graph if node.receipt_id not in receipt_ids),
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
    selector_choices = list(selector(candidates, hits))
    for candidate in selector_choices:
        original = by_receipts.get(candidate.receipt_ids)
        if original is not None and original not in selected:
            selected.append(original)
    if selected:
        return selected
    return candidates if selector_choices else []


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
