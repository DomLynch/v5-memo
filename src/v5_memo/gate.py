"""Selector gate contracts for V5 memo candidates."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence

from v5_memo.miner import _HUMAN_POPULATION_TERMS, _norm_token, query_anchor_terms
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
    "outcome",
    "positive_signal",
    "promise",
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
    "blind",
    "blinded",
    "clinical",
    "cohort",
    "controlled",
    "crossover",
    "difference",
    "differences",
    "double",
    "effect",
    "effects",
    "function",
    "functions",
    "human",
    "humans",
    "intervention",
    "label",
    "multicenter",
    "multicentre",
    "observational",
    "open",
    "older",
    "outcome",
    "outcomes",
    "parallel",
    "phase",
    "pilot",
    "performance",
    "placebo",
    "prospective",
    "randomised",
    "randomized",
    "retrospective",
    "single",
    "study",
    "studies",
    "trial",
    "trials",
})
_TOPIC_ENTITY_STOP = (
    _TOPIC_ANCHOR_STOP
    | _METABOLIC_AXIS_TERMS
    | _MUSCLE_ADAPTATION_AXIS_TERMS
    | _PROXY_OUTCOME_TERMS
    | frozenset({
        "adaptation",
        "endurance",
        "exercise",
        "healthy",
        "individual",
        "individuals",
        "mobility",
        "participant",
        "participants",
        "physical",
        "power",
        "recovery",
        "resistance",
        "supplement",
        "supplementation",
        "therapy",
        "treatment",
        "training",
        "was",
    })
)
_TOPIC_CONTEXT_SUFFIXES = (
    "algia",
    "emia",
    "itis",
    "oma",
    "opathy",
    "osis",
    "penia",
)
_TOPIC_ENTITY_BOUNDARY_TERMS = _TOPIC_ENTITY_STOP | frozenset({
    "aging",
    "cardiovascular",
    "cognitive",
    "physical",
})
_CLINICAL_ENTITY_GUARD_TERMS = (
    _HUMAN_POPULATION_TERMS
    | _METABOLIC_AXIS_TERMS
    | _MUSCLE_ADAPTATION_AXIS_TERMS
    | _PROXY_OUTCOME_TERMS
    | frozenset({
    "child",
    "children",
    "clinical",
    "older",
    "placebo",
    "randomised",
    "randomized",
    "trial",
    "trials",
    })
)
_TOPIC_TERM_CANONICAL = {"muscular": "muscle"}
_RECEIPT_CONTEXT_TERMS = frozenset({
    "baseline",
    "community",
    "daily",
    "day",
    "dwelling",
    "eight",
    "eleven",
    "five",
    "followup",
    "four",
    "fourteen",
    "frail",
    "frailty",
    "high",
    "hour",
    "hourly",
    "limitation",
    "low",
    "month",
    "monthly",
    "nine",
    "one",
    "post",
    "pre",
    "seven",
    "severe",
    "six",
    "status",
    "supervised",
    "ten",
    "thirteen",
    "three",
    "twelve",
    "twenty",
    "two",
    "unsupervised",
    "week",
    "weekly",
    "year",
    "yearly",
})
_STRUCTURAL_ENDPOINT_TERMS = frozenset({
    "boundary",
    "context",
    "dose",
    "endpoint",
    "endpoints",
    "modality",
    "outcome",
    "outcomes",
    "population",
    "setting",
})
_EVIDENCE_REANCHOR_BLOCKERS = frozenset({
    "off_modality_primary_signal",
    "off_topic_primary_signal",
})
LEAD_PROPOSAL_SCHEMA = "v5_evidence_lead_proposal_v1"


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
    ambiguous_direction = tuple(
        card.receipt_id
        for card in claim_cards
        if card.role in _PRIMARY_SIGNAL_ROLES
        and _has_ambiguous_direction(card)
        and not _has_endpoint_direction_mapping(card)
    )
    if ambiguous_direction:
        return {
            "error": "ambiguous_direction_without_endpoint_mapping",
            "receipt_ids": ambiguous_direction,
        }
    mixed_axis = _mixed_metabolic_muscle_axis_receipts(claim_cards)
    if mixed_axis:
        return {
            "error": "mixed_outcome_axis_bundle",
            "receipt_ids": mixed_axis,
        }
    off_modality = _off_modality_primary_receipts(candidate.topic, claim_cards)
    if off_modality:
        return {
            "error": "off_modality_primary_signal",
            "receipt_ids": off_modality,
        }
    off_topic = _off_topic_primary_receipts(
        candidate.topic,
        claim_cards,
        bridge_terms=candidate.bridge_terms,
    )
    if off_topic:
        return {
            "error": "off_topic_primary_signal",
            "receipt_ids": off_topic,
        }
    off_axis_context = _off_axis_direct_context_receipts(candidate.topic, claim_cards)
    if off_axis_context:
        return {
            "error": "off_axis_direct_context",
            "receipt_ids": off_axis_context,
        }
    structural_endpoints = _structural_endpoint_receipts(claim_cards)
    if structural_endpoints:
        return {
            "error": "structural_endpoint_without_outcome",
            "receipt_ids": structural_endpoints,
        }
    proxy_receipts = _proxy_boundary_receipts(claim_cards)
    if proxy_receipts and not _has_independent_directional_contrast(claim_cards):
        return {
            "error": "proxy_without_independent_directional_contrast",
            "receipt_ids": proxy_receipts,
        }
    return None


def lead_proposal_fingerprint(
    *,
    schema: str,
    lead: str,
    source_topic: str,
    receipt_ids: Sequence[str],
    source_keys: Sequence[str],
    candidate_score: int,
    candidate_novelty_score: int,
    candidate_tier: str,
    source_blocker: str,
) -> str:
    payload = {
        "schema": schema,
        "lead": lead_proposal_identity(lead),
        "source_topic": lead_proposal_identity(source_topic),
        "receipt_ids": sorted(set(receipt_ids)),
        "source_keys": sorted(set(source_keys)),
        "candidate_score": candidate_score,
        "candidate_novelty_score": candidate_novelty_score,
        "candidate_tier": candidate_tier,
        "source_blocker": source_blocker,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def lead_proposal_identity(value: str) -> str:
    """Return a lossless, Unicode-stable identity for a searched topic."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _proposal_phrase(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.replace("_", " ").casefold()))


def lead_proposal_metadata_valid(
    *,
    candidate_score: int,
    candidate_novelty_score: int,
    candidate_tier: str,
    source_blocker: str,
) -> bool:
    return (
        candidate_tier in {"publishable_alpha", "elite_alpha"}
        and candidate_score >= _MIN_SCORE_BY_TIER[candidate_tier]
        and candidate_novelty_score >= _MIN_NOVELTY_BY_TIER[candidate_tier]
        and source_blocker in _EVIDENCE_REANCHOR_BLOCKERS
    )


def _evidence_lead_proposal(
    *,
    topic: str,
    hits: Sequence[CorpusHit],
    candidates: Sequence[InsightCandidate],
    min_alpha_tier: str,
) -> dict[str, object] | None:
    source_keys_by_identifier: dict[str, set[str]] = {}
    for hit in hits:
        for identifier in {hit.receipt_id, hit.hit_id}:
            source_keys_by_identifier.setdefault(identifier, set()).add(hit.source_key)
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.score,
            -candidate.novelty_score,
            -_TIER_RANK.get(candidate_alpha_tier(candidate), -1),
            -candidate.evidence_score,
            tuple(sorted(candidate.receipt_ids)),
            candidate.topic,
            candidate.thesis,
            tuple(sorted(candidate.bridge_terms)),
            tuple(sorted(candidate.tension_terms)),
            tuple(sorted(candidate.reasons)),
            tuple(sorted(candidate.scorecard.items())),
            tuple(sorted(
                (role.receipt_id, role.role, role.reason)
                for role in candidate.receipt_roles
            )),
            tuple(sorted(
                (
                    card.receipt_id,
                    card.role,
                    card.design,
                    card.population,
                    card.outcome,
                    card.direction,
                    card.support_type,
                    card.confidence,
                    card.quote,
                )
                for card in candidate.claim_cards
            )),
            tuple(sorted(
                (node.receipt_id, node.role, node.reason)
                for node in candidate.evidence_graph
            )),
        ),
    )
    for candidate in ranked:
        if not meets_publish_bar(candidate, min_alpha_tier):
            continue
        blocker = candidate_publish_blocker(candidate)
        if not blocker or blocker.get("error") not in _EVIDENCE_REANCHOR_BLOCKERS:
            continue
        raw_blocker_receipts = blocker.get("receipt_ids")
        blocker_receipts = {
            receipt_id
            for receipt_id in raw_blocker_receipts
            if isinstance(receipt_id, str) and receipt_id
        } if isinstance(raw_blocker_receipts, Sequence) and not isinstance(
            raw_blocker_receipts, str
        ) else set()
        strong_cards = {
            card.receipt_id: card
            for card in candidate.claim_cards
            if card.receipt_id in blocker_receipts
            if card.role in _PRIMARY_SIGNAL_ROLES
            and card.population == "human"
            and card.support_type == "direct"
            and card.confidence == "high"
        }
        receipt_sources: dict[str, str] = {}
        for receipt_id in strong_cards:
            matching_source_keys = source_keys_by_identifier.get(receipt_id, set())
            if len(matching_source_keys) != 1:
                receipt_sources = {}
                break
            receipt_sources[receipt_id] = next(iter(matching_source_keys))
        if len(receipt_sources) < 2 or len(set(receipt_sources.values())) < 2:
            continue
        cards = [strong_cards[receipt_id] for receipt_id in sorted(receipt_sources)]
        bridge_terms = sorted({
            phrase
            for term in candidate.bridge_terms
            if (phrase := _proposal_phrase(term))
        })
        bridge_tokens = {
            token
            for phrase in bridge_terms
            for token in phrase.split()
            if token not in _TOPIC_ANCHOR_STOP
            and token not in _STRUCTURAL_ENDPOINT_TERMS
        }
        if not bridge_tokens:
            continue
        outcomes = sorted({
            phrase
            for card in cards
            if (phrase := _proposal_phrase(card.outcome))
            and phrase not in _STRUCTURAL_ENDPOINT_TERMS
        })
        designs = sorted({
            phrase
            for card in cards
            if (phrase := _proposal_phrase(card.design))
        })
        parts = [*bridge_terms[:3], *outcomes[:2], *designs[:1], "human"]
        lead = " ".join(dict.fromkeys(part for part in parts if part))
        if not outcomes or not lead or _proposal_phrase(topic) == lead:
            continue
        receipt_ids = tuple(sorted(receipt_sources))
        source_keys = tuple(sorted(set(receipt_sources.values())))
        candidate_tier = candidate_alpha_tier(candidate)
        source_blocker = str(blocker["error"])
        return {
            "schema": LEAD_PROPOSAL_SCHEMA,
            "lead": lead,
            "source_topic": topic,
            "receipt_ids": receipt_ids,
            "source_keys": source_keys,
            "candidate_score": candidate.score,
            "candidate_novelty_score": candidate.novelty_score,
            "candidate_tier": candidate_tier,
            "source_blocker": source_blocker,
            "proposal_fingerprint": lead_proposal_fingerprint(
                schema=LEAD_PROPOSAL_SCHEMA,
                lead=lead,
                source_topic=topic,
                receipt_ids=receipt_ids,
                source_keys=source_keys,
                candidate_score=candidate.score,
                candidate_novelty_score=candidate.novelty_score,
                candidate_tier=candidate_tier,
                source_blocker=source_blocker,
            ),
        }
    return None


def _positive_role_direction_mismatch(card: ClaimCard) -> bool:
    directions = set(card.direction.split("/"))
    if not directions & {"negative", "null"}:
        return False
    return card.role == "promise" or "positive" not in directions


def _has_ambiguous_direction(card: ClaimCard) -> bool:
    directions = {part.strip().casefold() for part in card.direction.split("/") if part.strip()}
    return "mixed" in directions or len(directions & {"negative", "null", "positive"}) > 1


def _has_endpoint_direction_mapping(card: ClaimCard) -> bool:
    """Accept an explicit ``endpoint=direction`` mapping without changing the card schema."""
    mapped: dict[str, str] = {}
    for raw_mapping in card.outcome.split("/"):
        if "=" not in raw_mapping:
            return False
        endpoint, direction = (part.strip().casefold() for part in raw_mapping.rsplit("=", 1))
        if not endpoint or direction not in {"negative", "null", "positive"}:
            return False
        mapped[endpoint] = direction
    recorded = {
        part.strip().casefold()
        for part in card.direction.split("/")
        if part.strip().casefold() in {"negative", "null", "positive"}
    }
    return len(mapped) > 1 and set(mapped.values()) == recorded


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
    *,
    bridge_terms: Sequence[str] = (),
) -> tuple[str, ...]:
    topic_terms = set(re.findall(r"[a-z0-9]+", topic.casefold()))
    required_terms = topic_terms & {"resistance", "strength", "training"}
    anchor_terms = _topic_primary_anchor_terms(topic_terms)
    topic_entity_anchors = _topic_entity_terms(topic)
    primary_entity_anchor = topic_entity_anchors[0] if topic_entity_anchors else ""
    topic_entity_terms = set(topic_entity_anchors)
    enforce_entity_identity = bool(topic_terms & _CLINICAL_ENTITY_GUARD_TERMS)
    bridge_anchors = _topic_primary_anchor_terms({
        term
        for bridge in bridge_terms
        for term in re.findall(r"[a-z0-9]+", bridge.casefold())
    })
    axis_terms = anchor_terms - bridge_anchors
    required_overlap = min(2, len(anchor_terms))
    out: list[str] = []
    for card in claim_cards:
        if card.role not in _PRIMARY_SIGNAL_ROLES:
            continue
        card_terms = {
            _TOPIC_TERM_CANONICAL.get(term, term)
            for term in re.findall(r"[a-z0-9]+", card.quote.casefold())
        }
        card_identity_terms = set(_normalized_entity_tokens(card.quote))
        card_entity_terms = set(_specific_entity_terms(card.quote))
        has_competing_entity = bool(
            enforce_entity_identity
            and primary_entity_anchor
            and card_entity_terms
            and (
                primary_entity_anchor not in card_identity_terms
                or (
                    len(topic_entity_anchors) > 1
                    and bool(card_identity_terms - topic_entity_terms)
                    and not set(topic_entity_anchors[1:]) <= card_identity_terms
                )
            )
        )
        if (
            required_terms
            and _off_topic_quote(card.quote)
            and card.outcome not in {"hypertrophy", "muscle thickness"}
        ):
            out.append(card.receipt_id)
            continue
        if (
            has_competing_entity
            or (axis_terms and not (axis_terms & card_terms))
            or (required_overlap and len(anchor_terms & card_terms) < required_overlap)
        ):
            out.append(card.receipt_id)
    return tuple(out)


def _specific_entity_terms(text: str) -> tuple[str, ...]:
    word_count = len(re.findall(r"[a-z0-9]+", text))
    return tuple(
        term
        for term in query_anchor_terms([text], limit=max(8, word_count))
        if term not in _TOPIC_ENTITY_STOP
        and term not in _RECEIPT_CONTEXT_TERMS
        and not term.endswith(_TOPIC_CONTEXT_SUFFIXES)
    )


def _normalized_entity_tokens(text: str) -> tuple[str, ...]:
    return tuple(
        _norm_token(raw)
        for raw in re.findall(r"[a-z0-9]+", text.casefold())
    )


def _topic_entity_terms(topic: str) -> tuple[str, ...]:
    raw_terms = re.findall(r"[a-z0-9]+", topic.casefold())
    boundary = next(
        (
            index
            for index, term in enumerate(raw_terms[1:], start=1)
            if term in _TOPIC_ENTITY_BOUNDARY_TERMS
        ),
        len(raw_terms),
    )
    return tuple(
        term
        for term in _normalized_entity_tokens(" ".join(raw_terms[:boundary]))
        if not term.endswith(_TOPIC_CONTEXT_SUFFIXES)
    )


def _topic_primary_anchor_terms(topic_terms: set[str]) -> set[str]:
    return {
        term
        for term in topic_terms
        if len(term) >= 4 and term not in _TOPIC_ANCHOR_STOP
    }


def _structural_endpoint_receipts(claim_cards: Sequence[ClaimCard]) -> tuple[str, ...]:
    invalid: list[str] = []
    for card in claim_cards:
        if card.population != "human" or card.support_type != "direct" or card.confidence != "high":
            continue
        endpoints = (raw.rsplit("=", 1)[0].strip() for raw in card.outcome.split("/"))
        if any(
            (terms := set(re.findall(r"[a-z0-9]+", endpoint.casefold())))
            and terms <= _STRUCTURAL_ENDPOINT_TERMS
            for endpoint in endpoints
        ):
            invalid.append(card.receipt_id)
    return tuple(invalid)


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
    final_quality_blockers: Sequence[Mapping[str, object]] = (),
) -> SearchFailure:
    best_mined = max(mined_candidates, key=lambda candidate: candidate.score, default=None)
    lead_proposal = _evidence_lead_proposal(
        topic=topic,
        hits=hits,
        candidates=mined_candidates,
        min_alpha_tier=min_alpha_tier,
    )
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
            "final_publish_quality_blocked_count": len(final_quality_blockers),
            "top_final_publish_quality_blockers": tuple(final_quality_blockers[:3]),
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
            **({"lead_proposal": lead_proposal} if lead_proposal is not None else {}),
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
