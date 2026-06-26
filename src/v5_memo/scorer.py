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
    scorecard: Mapping[str, int]


def score_connection(
    *,
    bridge_terms: tuple[str, ...],
    bridge_doc_counts: Mapping[str, int],
    unique_source_count: int,
    receipt_count: int,
    has_tension: bool,
    shape_score: int = 0,
    shape_reasons: tuple[str, ...] = (),
    support_quality: int = 0,
) -> ScoreParts:
    """Score whether a bridge is interesting enough to draft."""
    if not bridge_terms or receipt_count < 2:
        return ScoreParts(
            0,
            0,
            0,
            ("reject: insufficient bridge or receipts",),
            _empty_scorecard(),
        )

    shape_strength = sum(_SHAPE_WEIGHTS.get(reason, 1) for reason in shape_reasons)
    rarity = sum(1.0 / max(1, bridge_doc_counts.get(term, 1)) for term in bridge_terms)
    corpus_rarity = round(100 * min(1.0, rarity / len(bridge_terms)))
    shape_surprise = min(100, 8 * shape_strength + (25 if has_tension else 0))
    scorecard = {
        "retrieval_fit": min(100, 20 + 15 * min(len(bridge_terms), 4) + 5 * shape_strength),
        "construct_match": min(100, 20 + 12 * min(len(bridge_terms), 4) + 4 * shape_strength),
        "directional_contrast": min(100, (65 if has_tension else 15) + 4 * shape_strength),
        "evidence_directness": min(100, 20 + min(45, support_quality) + 8 * min(receipt_count, 4)),
        "method_strength": min(100, 20 + min(35, support_quality) + 4 * shape_strength),
        "independence": 80 if unique_source_count >= 2 else 20,
        "novelty_vs_corpus": max(corpus_rarity, shape_surprise if has_tension else 0),
        "novelty_vs_prior_memos": min(100, 35 + 5 * shape_strength + (15 if has_tension else 0)),
        "falsifiability": min(100, 30 + 6 * shape_strength + (20 if has_tension else 0)),
        "memo_usefulness": min(100, 25 + min(35, support_quality) + 6 * shape_strength),
    }
    novelty = round((scorecard["novelty_vs_corpus"] + scorecard["novelty_vs_prior_memos"]) / 2)
    if has_tension and (
        "shape:promise_outcome_reversal" in shape_reasons
        or "shape:expectation_reversal" in shape_reasons
    ):
        novelty = max(novelty, 55)
    evidence = round((scorecard["evidence_directness"] + scorecard["method_strength"] + scorecard["independence"]) / 3)
    source_bonus = 10 if unique_source_count >= 2 else 0
    tension_bonus = 10 if has_tension else 0
    shape_bonus = min(30, 4 * max(shape_strength, shape_score, 0))
    raw_score = round(sum(scorecard.values()) / len(scorecard) + source_bonus + tension_bonus + shape_bonus)
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
        "rare_bridge_terms" if corpus_rarity >= 60 else "common_bridge_terms",
        "surprising_evidence_shape" if shape_surprise >= 60 else "ordinary_evidence_shape",
        "distant_evidence_clusters" if scorecard["independence"] >= 55 else "nearby_evidence_clusters",
        "source_diverse" if unique_source_count >= 2 else "single_source",
        "strong_claim_support" if support_quality >= 25 else "thin_claim_support",
    ]
    if has_tension:
        reasons.append("directional_tension")
    if scorecard["construct_match"] < 45:
        reasons.append("reject: same keyword, different construct")
    if scorecard["directional_contrast"] < 45:
        reasons.append("reject: no direct outcome contrast")
    if scorecard["novelty_vs_corpus"] >= 60 and scorecard["directional_contrast"] < 45:
        reasons.append("reject: novelty only from rare wording")
    if scorecard["falsifiability"] < 45:
        reasons.append("reject: no falsifier")
    reasons.append("transparent_scorecard")
    reasons.extend(shape_reasons)
    return ScoreParts(score, novelty, evidence, tuple(reasons), scorecard)


def _empty_scorecard() -> Mapping[str, int]:
    return {
        "retrieval_fit": 0,
        "construct_match": 0,
        "directional_contrast": 0,
        "evidence_directness": 0,
        "method_strength": 0,
        "independence": 0,
        "novelty_vs_corpus": 0,
        "novelty_vs_prior_memos": 0,
        "falsifiability": 0,
        "memo_usefulness": 0,
    }
