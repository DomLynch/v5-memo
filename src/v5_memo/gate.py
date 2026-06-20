"""Selector gate contracts for V5 memo candidates."""
from __future__ import annotations

from collections.abc import Sequence

from v5_memo.schemas import CorpusHit, InsightCandidate, SearchFailure

_TIER_RANK = {"discovery_seed": 0, "publishable_alpha": 1, "elite_alpha": 2}
_MIN_SCORE_BY_TIER = {
    "discovery_seed": 0,
    "publishable_alpha": 70,
    "elite_alpha": 80,
}


def candidate_alpha_tier(candidate: InsightCandidate) -> str:
    for reason in candidate.reasons:
        if reason.startswith("tier:"):
            return reason.removeprefix("tier:")
    return "discovery_seed"


def meets_min_alpha_tier(candidate: InsightCandidate, min_alpha_tier: str) -> bool:
    return _TIER_RANK[candidate_alpha_tier(candidate)] >= _TIER_RANK[min_alpha_tier]


def meets_publish_bar(candidate: InsightCandidate, min_alpha_tier: str) -> bool:
    return (
        meets_min_alpha_tier(candidate, min_alpha_tier)
        and candidate.score >= _MIN_SCORE_BY_TIER[min_alpha_tier]
    )


def no_alpha_failure(
    *,
    topic: str,
    hits: Sequence[CorpusHit],
    candidates: Sequence[InsightCandidate],
    min_alpha_tier: str,
) -> SearchFailure:
    return SearchFailure(
        code="no_receipt_bound_alpha_candidate",
        message="no receipt-bound alpha memo candidate found",
        details={
            "topic": topic,
            "hit_count": len(hits),
            "candidate_count": len(candidates),
            "min_alpha_tier": min_alpha_tier,
        },
    )
