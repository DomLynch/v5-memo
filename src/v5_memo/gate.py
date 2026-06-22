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
    sources: set[str] = set()
    search_passes: set[str] = set()
    shards_searched = 0
    years = [hit.year for hit in receipts if hit.year is not None]
    cited_by_max = 0
    result_duplicate_rate = 0.0
    result_citation_diversity = 0
    abstract_count = 0
    for hit in receipts:
        if hit.abstract.strip():
            abstract_count += 1
        search_pass = hit.metadata.get("search_pass")
        if isinstance(search_pass, str) and search_pass:
            search_passes.add(search_pass)
        cited = hit.metadata.get("cited_by_count")
        if isinstance(cited, int):
            cited_by_max = max(cited_by_max, cited)
        receipt = hit.metadata.get("shard_receipt")
        if not isinstance(receipt, dict):
            continue
        shards_searched = max(shards_searched, _int_value(receipt.get("shards_searched")))
        search_passes.update(_string_values(receipt.get("sweep_completed_pass_roles")))
        result_duplicate_rate = max(
            result_duplicate_rate,
            _float_value(receipt.get("result_duplicate_rate")),
        )
        result_citation_diversity = max(
            result_citation_diversity,
            _int_value(receipt.get("result_citation_diversity")),
        )
        raw_sources = receipt.get("sources_searched")
        if isinstance(raw_sources, dict):
            sources.update(str(source) for source, count in raw_sources.items() if _int_value(count) > 0)
        year_range = receipt.get("year_range_searched")
        if isinstance(year_range, dict):
            years.extend(
                year
                for key in ("min", "max")
                if (year := _int_or_none(year_range.get(key))) is not None
            )
        cited_range = receipt.get("cited_by_range_searched")
        if isinstance(cited_range, dict):
            cited_by_max = max(cited_by_max, _int_value(cited_range.get("max")))
    return {
        "shards_searched": shards_searched,
        "sources_searched": tuple(sorted(sources)),
        "source_count": len(sources),
        "year_range": {"min": min(years), "max": max(years)} if years else {"min": None, "max": None},
        "cited_by_max": cited_by_max,
        "search_passes": tuple(sorted(search_passes)),
        "search_pass_count": len(search_passes),
        "result_duplicate_rate": round(result_duplicate_rate, 4),
        "result_citation_diversity": result_citation_diversity,
        "abstract_receipt_count": abstract_count,
    }


def memo_coverage_failure(
    *,
    topic: str,
    receipts: Sequence[CorpusHit],
    min_shards_searched: int = 0,
    min_sources_searched: int = 0,
    min_search_passes: int = 0,
    min_abstract_receipts: int = 0,
    min_result_citation_diversity: int = 0,
    max_result_duplicate_rate: float | None = None,
) -> SearchFailure | None:
    summary = memo_coverage_summary(receipts)
    failures: list[str] = []
    if min_shards_searched and _int_value(summary["shards_searched"]) < min_shards_searched:
        failures.append("shards_searched")
    if min_sources_searched and _int_value(summary["source_count"]) < min_sources_searched:
        failures.append("sources_searched")
    if min_search_passes and _int_value(summary["search_pass_count"]) < min_search_passes:
        failures.append("search_passes")
    if (
        min_abstract_receipts
        and _int_value(summary["abstract_receipt_count"]) < min_abstract_receipts
    ):
        failures.append("abstract_receipts")
    if (
        min_result_citation_diversity
        and _int_value(summary["result_citation_diversity"]) < min_result_citation_diversity
    ):
        failures.append("result_citation_diversity")
    if (
        max_result_duplicate_rate is not None
        and _float_value(summary["result_duplicate_rate"]) > max_result_duplicate_rate
    ):
        failures.append("result_duplicate_rate")
    if not failures:
        return None
    return SearchFailure(
        code="memo_coverage_too_narrow",
        message="memo evidence coverage too narrow",
        details={
            "topic": topic,
            "failures": tuple(failures),
            "requirements": {
                "min_shards_searched": min_shards_searched,
                "min_sources_searched": min_sources_searched,
                "min_search_passes": min_search_passes,
                "min_abstract_receipts": min_abstract_receipts,
                "min_result_citation_diversity": min_result_citation_diversity,
                "max_result_duplicate_rate": max_result_duplicate_rate,
            },
            "coverage": summary,
        },
    )


def _int_value(value: object) -> int:
    parsed = _int_or_none(value)
    return parsed if parsed is not None else 0


def _float_value(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _string_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value if str(item))


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None
