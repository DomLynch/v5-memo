"""Selector gate contracts for V5 memo candidates."""
from __future__ import annotations

import re
from collections.abc import Sequence

from v5_memo.schemas import ClaimCard, CorpusHit, InsightCandidate, SearchFailure

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
_PRIMARY_SIGNAL_ROLES = frozenset({
    "aggregate_signal",
    "negative_signal",
    "null_signal",
    "positive_signal",
    "tail_risk",
})
_CONTEXT_ROLES = frozenset({"boundary", "consensus", "mechanism", "replication"})
_PROXY_ROLES = frozenset({"boundary"})
_PROXY_OUTCOME_TERMS = frozenset({
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
_METABOLIC_AXIS_TERMS = frozenset({
    "diabetes",
    "diabetic",
    "glucose",
    "glycemic",
    "glycaemic",
    "hyperglycemia",
    "hyperglycemic",
    "insulin",
    "metabolic",
    "prediabetes",
    "t2dm",
})
_MUSCLE_ADAPTATION_AXIS_TERMS = frozenset({
    "atrophy",
    "body composition",
    "hypertrophic",
    "hypertrophy",
    "lean mass",
    "muscle",
    "myofiber",
    "strength",
})
_TOPIC_ANCHOR_STOP = frozenset({
    "adult",
    "adults",
    "effect",
    "effects",
    "function",
    "functions",
    "human",
    "humans",
    "older",
    "outcome",
    "outcomes",
    "performance",
    "study",
    "studies",
    "trial",
    "trials",
})


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


def candidate_publish_blocker(candidate: InsightCandidate) -> dict[str, object] | None:
    claim_cards = tuple(candidate.claim_cards or ())
    if not claim_cards:
        return None
    direct_human = sum(
        1
        for card in claim_cards
        if card.population == "human" and card.support_type == "direct"
    )
    strong_direct_human = sum(
        1
        for card in claim_cards
        if card.population == "human"
        and card.support_type == "direct"
        and card.confidence == "high"
        and card.role != "safety_feasibility"
    )
    indirect_model = sum(
        1
        for card in claim_cards
        if card.population in {"animal", "cell_model"} or card.support_type == "indirect"
    )
    if indirect_model and direct_human == 0:
        return {
            "error": "translational_evidence_too_indirect",
            "direct_human_receipts": direct_human,
            "indirect_model_receipts": indirect_model,
        }
    weak_primary_signals = tuple(
        card.receipt_id
        for card in claim_cards
        if card.role in _PRIMARY_SIGNAL_ROLES
        and not (
            card.population == "human"
            and card.support_type == "direct"
            and card.confidence == "high"
        )
    )
    if weak_primary_signals:
        return {
            "error": "primary_signal_not_strong_direct_human",
            "receipt_ids": weak_primary_signals,
        }
    if direct_human < 2 or strong_direct_human < 2:
        return {
            "error": "insufficient_direct_human_receipts",
            "direct_human_receipts": direct_human,
            "strong_direct_human_receipts": strong_direct_human,
        }
    weak_context = tuple(
        card.receipt_id
        for card in claim_cards
        if card.role in _CONTEXT_ROLES
        and (
            card.confidence == "low"
            or (card.population == "unspecified" and card.support_type == "indirect")
        )
    )
    if weak_context:
        return {
            "error": "weak_context_receipts",
            "receipt_ids": weak_context,
        }
    direction_mismatch = tuple(
        card.receipt_id
        for card in claim_cards
        if card.role in {"promise", "positive_signal"}
        and _positive_role_direction_mismatch(card)
    )
    if direction_mismatch:
        return {
            "error": "positive_role_direction_mismatch",
            "receipt_ids": direction_mismatch,
        }
    mixed_axis = _mixed_metabolic_muscle_axis_receipts(claim_cards)
    if mixed_axis:
        return {
            "error": "mixed_outcome_axis_bundle",
            "receipt_ids": mixed_axis,
        }
    off_topic = _off_topic_primary_receipts(candidate.topic, claim_cards)
    if off_topic:
        return {
            "error": "off_topic_primary_signal",
            "receipt_ids": off_topic,
        }
    off_modality = _off_modality_primary_receipts(candidate.topic, claim_cards)
    if off_modality:
        return {
            "error": "off_modality_primary_signal",
            "receipt_ids": off_modality,
        }
    off_axis_context = _off_axis_direct_context_receipts(candidate.topic, claim_cards)
    if off_axis_context:
        return {
            "error": "off_axis_direct_context",
            "receipt_ids": off_axis_context,
        }
    proxy_receipts = _proxy_boundary_receipts(claim_cards)
    if proxy_receipts and not _has_independent_directional_contrast(claim_cards):
        return {
            "error": "proxy_without_independent_directional_contrast",
            "receipt_ids": proxy_receipts,
        }
    return None


def _positive_role_direction_mismatch(card: ClaimCard) -> bool:
    directions = set(card.direction.split("/"))
    if not directions & {"negative", "null"}:
        return False
    return card.role == "promise" or "positive" not in directions


def _proxy_boundary_receipts(claim_cards: Sequence[ClaimCard]) -> tuple[str, ...]:
    return tuple(
        card.receipt_id
        for card in claim_cards
        if card.role in _PROXY_ROLES
        and (card.direction == "proxy" or bool(set(card.outcome.split("/")) & _PROXY_OUTCOME_TERMS))
        and card.population == "human"
        and card.support_type == "direct"
    )


def _has_independent_directional_contrast(claim_cards: Sequence[ClaimCard]) -> bool:
    directions: set[str] = set()
    for card in claim_cards:
        if card.population != "human":
            continue
        if card.support_type != "direct" or card.confidence != "high":
            continue
        if card.role in _CONTEXT_ROLES or card.direction == "proxy":
            continue
        directions.update(set(card.direction.split("/")) & {"negative", "null", "positive"})
    return len(directions) >= 2


def _mixed_metabolic_muscle_axis_receipts(claim_cards: Sequence[ClaimCard]) -> tuple[str, ...]:
    metabolic_ids: list[str] = []
    muscle_ids: list[str] = []
    for card in claim_cards:
        if card.population != "human" or card.support_type != "direct":
            continue
        text = f"{card.outcome} {card.quote}".casefold()
        if any(term in text for term in _METABOLIC_AXIS_TERMS):
            metabolic_ids.append(card.receipt_id)
        if any(term in text for term in _MUSCLE_ADAPTATION_AXIS_TERMS):
            muscle_ids.append(card.receipt_id)
    if not metabolic_ids or not muscle_ids:
        return ()
    return tuple(dict.fromkeys([*metabolic_ids, *muscle_ids]))


def _off_modality_primary_receipts(
    topic: str,
    claim_cards: Sequence[ClaimCard],
) -> tuple[str, ...]:
    topic_terms = set(re.findall(r"[a-z0-9]+", topic.casefold()))
    if not topic_terms & {"adaptation", "exercise", "resistance", "strength", "training"}:
        return ()
    return tuple(
        card.receipt_id
        for card in claim_cards
        if card.role in _PRIMARY_SIGNAL_ROLES
        and _off_modality_training_quote(card.quote)
    )


def _off_axis_direct_context_receipts(
    topic: str,
    claim_cards: Sequence[ClaimCard],
) -> tuple[str, ...]:
    topic_terms = set(re.findall(r"[a-z0-9]+", topic.casefold()))
    if not (topic_terms & {"adaptation", "exercise", "resistance", "strength", "training"}):
        return ()
    return tuple(
        card.receipt_id
        for card in claim_cards
        if card.role not in _PRIMARY_SIGNAL_ROLES
        and card.population == "human"
        and card.support_type == "direct"
        and _off_topic_quote(card.quote)
    )


def _off_topic_primary_receipts(
    topic: str,
    claim_cards: Sequence[ClaimCard],
) -> tuple[str, ...]:
    topic_terms = set(re.findall(r"[a-z0-9]+", topic.casefold()))
    required_terms = topic_terms & {"resistance", "strength", "training"}
    context_terms = _topic_primary_context_terms(topic)
    anchor_terms = _topic_primary_anchor_terms(topic_terms)
    out: list[str] = []
    for card in claim_cards:
        if card.role not in _PRIMARY_SIGNAL_ROLES:
            continue
        card_terms = set(re.findall(r"[a-z0-9]+", card.quote.casefold()))
        if (
            required_terms
            and _off_topic_quote(card.quote)
            and card.outcome not in {"hypertrophy", "muscle thickness"}
        ):
            out.append(card.receipt_id)
            continue
        if (
            (context_terms and not (context_terms & card_terms))
            or (anchor_terms and not (anchor_terms & card_terms))
        ):
            out.append(card.receipt_id)
    return tuple(out)


def _topic_primary_context_terms(topic: str) -> set[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in re.findall(r"[a-z0-9]+", topic.casefold()):
        if len(raw) < 4 or raw in _TOPIC_ANCHOR_STOP or raw in seen:
            continue
        seen.add(raw)
        ordered.append(raw)
    return set(ordered[1:]) if len(ordered) > 1 else set()


def _topic_primary_anchor_terms(topic_terms: set[str]) -> set[str]:
    return {
        term
        for term in topic_terms
        if len(term) >= 4 and term not in _TOPIC_ANCHOR_STOP
    }


def _off_topic_quote(quote: str) -> bool:
    text = quote.casefold()
    return any(term in text for term in ("accidental", "cold-weather", "hypothermia", "military", "warfighter"))


def _off_modality_training_quote(quote: str) -> bool:
    text = quote.casefold()
    return (
        "post-match" in text
        or "post match" in text
        or ("recovery" in text and "soccer" in text)
    )


def no_alpha_failure(
    *,
    topic: str,
    hits: Sequence[CorpusHit],
    candidates: Sequence[InsightCandidate],
    min_alpha_tier: str,
    mined_candidates: Sequence[InsightCandidate] = (),
    seed_queries: Sequence[str] = (),
    anchor_terms: Sequence[str] = (),
) -> SearchFailure:
    best_mined = max(mined_candidates, key=lambda candidate: candidate.score, default=None)
    publish_quality_blockers = tuple(
        {
            "receipt_ids": candidate.receipt_ids,
            "score": candidate.score,
            "novelty_score": candidate.novelty_score,
            "tier": candidate_alpha_tier(candidate),
            "blocker": blocker,
        }
        for candidate in mined_candidates
        if meets_publish_bar(candidate, min_alpha_tier)
        if (blocker := candidate_publish_blocker(candidate)) is not None
    )
    return SearchFailure(
        code="no_receipt_bound_alpha_candidate",
        message="no receipt-bound alpha memo candidate found",
        details={
            "topic": topic,
            "hit_count": len(hits),
            "candidate_count": len(candidates),
            "mined_candidate_count": len(mined_candidates),
            "best_mined_score": best_mined.score if best_mined is not None else 0,
            "best_mined_novelty": best_mined.novelty_score if best_mined is not None else 0,
            "publish_quality_blocked_count": len(publish_quality_blockers),
            "top_publish_quality_blockers": publish_quality_blockers[:3],
            "min_alpha_tier": min_alpha_tier,
            "queries_used": tuple(seed_queries),
            "anchor_terms": tuple(anchor_terms),
            "top_mined_candidates": tuple(
                {
                    "receipt_ids": candidate.receipt_ids,
                    "score": candidate.score,
                    "novelty_score": candidate.novelty_score,
                    "tier": candidate_alpha_tier(candidate),
                    "reasons": candidate.reasons,
                }
                for candidate in mined_candidates[:3]
            ),
        },
    )


def memo_coverage_summary(receipts: Sequence[CorpusHit]) -> dict[str, object]:
    sources: set[str] = set()
    search_passes: set[str] = set()
    shards_searched = 0
    partial_shard_search = False
    sweep_failed_shards = 0
    years = [hit.year for hit in receipts if hit.year is not None]
    cited_by_max = 0
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
        partial_shard_search = partial_shard_search or receipt.get("partial_shard_search") is True
        sweep_failed_shards += _int_value(receipt.get("sweep_failed_shards"))
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
        "partial_shard_search": partial_shard_search,
        "sweep_failed_shards": sweep_failed_shards,
        "sources_searched": tuple(sorted(sources)),
        "source_count": len(sources),
        "year_range": {"min": min(years), "max": max(years)} if years else {"min": None, "max": None},
        "cited_by_max": cited_by_max,
        "search_passes": tuple(sorted(search_passes)),
        "search_pass_count": len(search_passes),
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
) -> SearchFailure | None:
    summary = memo_coverage_summary(receipts)
    failures: list[str] = []
    min_search_passes = min(min_search_passes, len(receipts))
    if min_shards_searched and _int_value(summary["shards_searched"]) < min_shards_searched:
        failures.append("shards_searched")
    if min_shards_searched and summary["partial_shard_search"] is True:
        failures.append("partial_shard_search")
    if min_shards_searched and _int_value(summary["sweep_failed_shards"]) > 0:
        failures.append("sweep_failed_shards")
    if min_sources_searched and _int_value(summary["source_count"]) < min_sources_searched:
        failures.append("sources_searched")
    if min_search_passes and _int_value(summary["search_pass_count"]) < min_search_passes:
        failures.append("search_passes")
    if (
        min_abstract_receipts
        and _int_value(summary["abstract_receipt_count"]) < min_abstract_receipts
    ):
        failures.append("abstract_receipts")
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
            },
            "coverage": summary,
        },
    )


def _int_value(value: object) -> int:
    parsed = _int_or_none(value)
    return parsed if parsed is not None else 0


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
