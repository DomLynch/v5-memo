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
_MIN_NOVELTY_BY_TIER = {
    "discovery_seed": 0,
    "publishable_alpha": 0,
    "elite_alpha": 35,
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
        and candidate.novelty_score >= _MIN_NOVELTY_BY_TIER[min_alpha_tier]
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


def memo_coverage_summary(receipts: Sequence[CorpusHit]) -> dict[str, object]:
    """Summarize fullraw/search coverage visible on bound receipts."""
    years = [hit.year for hit in receipts if hit.year is not None]
    sources_used = sorted({hit.source for hit in receipts if hit.source})
    shard_receipts = [
        receipt
        for hit in receipts
        for key in ("shard_receipt", "fullraw_search_receipt")
        if isinstance((receipt := hit.metadata.get(key)), dict)
    ]
    sources_searched: set[str] = set()
    search_passes: set[str] = set()
    shards_searched = 0
    result_duplicate_rate = 0.0
    result_citation_diversity = 0
    for receipt in shard_receipts:
        shards_searched = max(shards_searched, _int(receipt.get("shards_searched")))
        result_duplicate_rate = max(
            result_duplicate_rate,
            float(receipt.get("result_duplicate_rate") or receipt.get("duplicate_rate") or 0.0),
        )
        result_citation_diversity = max(
            result_citation_diversity,
            _int(receipt.get("result_citation_diversity")),
        )
        raw_sources = receipt.get("sources_searched")
        if isinstance(raw_sources, dict):
            sources_searched.update(str(key) for key, value in raw_sources.items() if _int(value))
        raw_passes = receipt.get("search_passes")
        if isinstance(raw_passes, tuple | list):
            search_passes.update(str(item) for item in raw_passes)
    return {
        "shards_searched": shards_searched,
        "sources_searched": sorted(sources_searched),
        "sources_used": sources_used,
        "year_range": {
            "min": min(years) if years else None,
            "max": max(years) if years else None,
        },
        "abstract_receipt_count": sum(1 for hit in receipts if hit.abstract.strip()),
        "search_passes": sorted(search_passes),
        "result_duplicate_rate": result_duplicate_rate,
        "result_citation_diversity": result_citation_diversity,
    }


def _int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0
