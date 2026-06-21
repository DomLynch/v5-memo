"""Pure scoring for cross-corpus alpha candidates."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

_SHAPE_WEIGHTS = {
    "shape:promise_outcome_reversal": 7,
    "shape:expectation_reversal": 6,
    "shape:directional_reversal": 5,
    "shape:boundary_condition": 3,
    "shape:denominator_split": 3,
    "shape:role_inversion": 3,
    "shape:timing_split": 2,
    "shape:measurement_mismatch": 1,
    "shape:expertise_split": 1,
}


@dataclass(frozen=True, slots=True)
class ScoreParts:
    score: int
    novelty_score: int
    evidence_score: int
    reasons: tuple[str, ...]


def score_connection(
    *,
    bridge_terms: tuple[str, ...],
    bridge_doc_counts: Mapping[str, int],
    unique_source_count: int,
    receipt_count: int,
    has_tension: bool,
    shape_score: int = 0,
    shape_reasons: tuple[str, ...] = (),
) -> ScoreParts:
    """Score whether a bridge is interesting enough to draft."""
    if not bridge_terms or receipt_count < 2:
        return ScoreParts(0, 0, 0, ("insufficient_bridge_or_receipts",))

    rarity = sum(1.0 / max(1, bridge_doc_counts.get(term, 1)) for term in bridge_terms)
    novelty = round(100 * min(1.0, rarity / len(bridge_terms)))
    if has_tension and (
        "shape:promise_outcome_reversal" in shape_reasons
        or "shape:expectation_reversal" in shape_reasons
    ):
        novelty = max(novelty, 50)
    evidence = min(100, 35 + 20 * min(unique_source_count, 3) + 5 * min(receipt_count, 3))
    source_bonus = 10 if unique_source_count >= 2 else 0
    tension_bonus = 10 if has_tension else 0
    shape_strength = sum(_SHAPE_WEIGHTS.get(reason, 1) for reason in shape_reasons)
    shape_bonus = min(30, 4 * max(shape_strength, shape_score, 0))
    raw_score = round(0.40 * novelty + 0.35 * evidence + source_bonus + tension_bonus + shape_bonus)
    bridge_cap = 100
    if novelty < 20 and not has_tension:
        bridge_cap = 55
    elif novelty < 20 and not (
        "shape:promise_outcome_reversal" in shape_reasons
        or "shape:expectation_reversal" in shape_reasons
    ):
        bridge_cap = 70
    score = min(100, raw_score, bridge_cap)

    reasons: list[str] = [
        "rare_bridge_terms" if novelty >= 60 else "common_bridge_terms",
        "source_diverse" if unique_source_count >= 2 else "single_source",
    ]
    if has_tension:
        reasons.append("directional_tension")
    reasons.extend(shape_reasons)
    return ScoreParts(score, novelty, evidence, tuple(reasons))
