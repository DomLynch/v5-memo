"""Render V5 memo drafts from already-bound receipts."""
from __future__ import annotations

from collections.abc import Iterable, Sequence

from v5_memo.gate import candidate_alpha_tier
from v5_memo.schemas import ClaimCard, CorpusHit, InsightCandidate

_TITLE_STOPWORDS = frozenset({"during", "after", "before", "following", "under", "with"})


def render_memo(candidate: InsightCandidate, receipts: Sequence[CorpusHit]) -> str:
    if candidate_alpha_tier(candidate) == "discovery_seed":
        return render_discovery_seed(candidate, receipts)
    return render_alpha_memo(candidate, receipts)


def render_alpha_memo(candidate: InsightCandidate, receipts: Sequence[CorpusHit]) -> str:
    """Render a short memo without adding claims outside the candidate receipts."""
    if not receipts:
        raise ValueError("cannot render memo without receipts")
    direct_cards = _direct_human_cards(candidate)
    bridge = ", ".join(_clean_labels(candidate.bridge_terms)) or "unspecified bridge"
    tension = ", ".join(_clean_labels(candidate.tension_terms)) or "not detected"
    hypothesis = _bounded_hypothesis(candidate)
    lines = [
        f"# Alpha memo: {_memo_title(candidate)}",
        "",
        "Hypothesis-level alpha signal; not clinical advice.",
        "",
        f"**Alpha hypothesis:** {hypothesis}",
        "",
        "**Core signal:**",
        *_core_signal_lines(candidate, direct_cards),
        "",
        "**Receipt-level synthesis:**",
        *_synthesis_lines(candidate, direct_cards),
        "",
        "**Limits:**",
        *_limit_lines(candidate, direct_cards),
        "",
        "**What would falsify it:**",
        _falsification_clause(candidate),
        "",
        "**Audit trail:**",
        f"- Signal score: `{candidate.score}` (novelty `{candidate.novelty_score}`, evidence `{candidate.evidence_score}`).",
        f"- Evidence bridge terms: {bridge}.",
        f"- Direction/tension terms: {tension}.",
        "",
        "**Evidence graph:**",
        *_evidence_graph_lines(candidate),
        "",
        "**Receipt roles:**",
        *_receipt_role_lines(candidate),
        "",
        "**Claim ledger:**",
        *_claim_card_lines(candidate),
        "",
        "**Receipts:**",
    ]
    lines.extend(_receipt_line(index, hit) for index, hit in enumerate(receipts, start=1))
    lines.extend([
        "",
        "**Safety note:** This memo is an alpha hypothesis. A later LLM writer may sharpen prose, "
        "but it may not add claims beyond the receipts above.",
    ])
    return "\n".join(lines).strip() + "\n"


def render_discovery_seed(candidate: InsightCandidate, receipts: Sequence[CorpusHit]) -> str:
    """Render a lower-confidence seed without presenting it as publishable alpha."""
    if not receipts:
        raise ValueError("cannot render memo without receipts")
    lines = [
        f"# Discovery seed: {_memo_title(candidate)}",
        "",
        f"**Seed hypothesis:** {candidate.thesis}",
        "",
        "**Status:** Not publishable alpha until selector evidence reaches publishable tier.",
        "",
        "**Evidence graph:**",
        *_evidence_graph_lines(candidate),
        "",
        "**Receipt roles:**",
        *_receipt_role_lines(candidate),
        "",
        "**Claim ledger:**",
        *_claim_card_lines(candidate),
        "",
        "**Receipts:**",
    ]
    lines.extend(_receipt_line(index, hit) for index, hit in enumerate(receipts, start=1))
    return "\n".join(lines).strip() + "\n"


def _receipt_line(index: int, hit: CorpusHit) -> str:
    year = f", {hit.year}" if hit.year else ""
    venue = f", {hit.venue}" if hit.venue else ""
    locator = hit.receipt_id if hit.receipt_id != hit.hit_id else hit.url or hit.hit_id
    return f"{index}. `{hit.hit_id}` {hit.title}{year}{venue}. Source: {hit.source}. ID: {locator}"


def _bounded_hypothesis(candidate: InsightCandidate) -> str:
    direct_cards = _direct_human_cards(candidate)
    if len(direct_cards) < 2:
        return candidate.thesis
    outcomes = _non_generic_labels(card.outcome for card in direct_cards)
    directions = _non_generic_labels(
        direction
        for card in direct_cards
        for direction in card.direction.split("/")
        if direction not in {"proxy", "unclear"}
    )
    designs = _non_generic_labels(card.design for card in direct_cards)
    outcome_text = _join_labels(outcomes[:3]) or "the cited endpoints"
    direction_text = _join_labels(directions[:3]) or "mixed"
    design_text = _join_labels(designs[:2]) or "study"
    return (
        f"In {candidate.topic}, direct human {design_text} receipts support a bounded "
        f"{direction_text} signal across {outcome_text}; treat it as hypothesis-level "
        "until the same population and endpoint are replicated."
    )


def _falsification_clause(candidate: InsightCandidate) -> str:
    if len(_direct_human_cards(candidate)) < 2:
        return (
            "A follow-up search fails to find direct receipts where the bridge "
            "term and both evidence streams appear in the same source-grounded "
            "claim, or the apparent connection collapses to one duplicated source."
        )
    return (
        "A direct human replication in the same population and endpoint shows no "
        "bounded contrast, or the apparent contrast collapses to duplicated, "
        "precursor, or off-axis receipts."
    )


def _core_signal_lines(candidate: InsightCandidate, direct_cards: Sequence[ClaimCard]) -> list[str]:
    thesis = candidate.thesis.strip()
    if len(direct_cards) < 2:
        return [thesis if thesis.endswith(".") else f"{thesis}."]
    outcomes = _non_generic_labels(card.outcome for card in direct_cards)
    directions = _non_generic_labels(
        direction
        for card in direct_cards
        for direction in card.direction.split("/")
        if direction not in {"proxy", "unclear"}
    )
    designs = _non_generic_labels(card.design for card in direct_cards)
    return [
        (
            f"The direct human receipts are {_join_labels(designs[:2]) or 'study'} evidence "
            f"for {_join_labels(outcomes[:3]) or 'the cited endpoints'}."
        ),
        (
            f"Because the recorded directions are {_join_labels(directions[:3]) or 'mixed'}, "
            "this memo treats the bundle as endpoint-specific evidence rather than a pooled clinical effect."
        ),
    ]


def _synthesis_lines(candidate: InsightCandidate, direct_cards: Sequence[ClaimCard]) -> list[str]:
    del candidate
    cards = list(direct_cards)
    if not cards:
        return ["- No direct human claim cards were assigned; keep this as a discovery lead only."]
    return [_claim_summary_line(card) for card in cards]


def _claim_summary_line(card: ClaimCard) -> str:
    outcome = _display_label(card.outcome)
    direction = _display_label(card.direction, fallback=card.direction.replace("_", " "))
    design = _display_label(card.design, fallback=card.design.replace("_", " "))
    quote = card.quote.strip()
    suffix = f" {quote}" if quote else ""
    return f"- `{card.receipt_id}` ({card.role}): {design} in {card.population}; {outcome} is {direction}.{suffix}"


def _limit_lines(candidate: InsightCandidate, direct_cards: Sequence[ClaimCard]) -> list[str]:
    if len(direct_cards) < 2:
        return [
            "Only one direct human receipt is bound, so this should not be read as a mature evidence synthesis.",
            "The next step is another direct receipt in the same population and endpoint family.",
        ]
    outcomes = _non_generic_labels(card.outcome for card in direct_cards)
    directions = _non_generic_labels(
        direction
        for card in direct_cards
        for direction in card.direction.split("/")
        if direction not in {"proxy", "unclear"}
    )
    limits = [
        "The cited receipts should not be pooled unless population, intervention window, and endpoint match.",
        "A same-endpoint human replication would move this from alpha signal toward claim-level evidence.",
    ]
    if len(outcomes) > 1 or len(directions) > 1 or any(card.role == "boundary" for card in direct_cards):
        limits.insert(
            0,
            "The bundle is heterogeneous, so the memo separates endpoint roles instead of presenting one merged effect.",
        )
    if "conference" in " ".join(card.quote.casefold() for card in direct_cards):
        limits.append("At least one receipt appears conference-level; treat that source as boundary evidence.")
    if not candidate.receipt_ids:
        limits.append("No receipt IDs were assigned by the selector.")
    return limits


def _direct_human_cards(candidate: InsightCandidate) -> list[ClaimCard]:
    return [
        card
        for card in candidate.claim_cards
        if card.population == "human"
        and card.support_type == "direct"
        and card.confidence == "high"
        and card.role != "safety_feasibility"
    ]


def _non_generic_labels(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for label in value.split("/"):
            clean = _clean_label(label)
            if not clean or clean in {"long", "outcome", "outcomes", "setting", "unspecified"}:
                continue
            if clean not in out:
                out.append(clean)
    return out


def _clean_labels(values: Iterable[str]) -> list[str]:
    return [_clean_label(value) for value in values if _clean_label(value)]


def _clean_label(value: str) -> str:
    clean = value.replace("_", " ").strip().casefold()
    return {"diabete": "diabetes"}.get(clean, clean)


def _join_labels(labels: Sequence[str]) -> str:
    if len(labels) <= 1:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def _display_label(value: str, *, fallback: str = "unspecified") -> str:
    return _join_labels(_non_generic_labels([value])) or fallback


def _memo_title(candidate: InsightCandidate) -> str:
    if candidate.bridge_terms:
        bridge_terms = _clean_labels(candidate.bridge_terms)
        terms = [term for term in bridge_terms if term not in _TITLE_STOPWORDS]
        return " / ".join((terms or bridge_terms)[:3])
    return candidate.topic


def _receipt_role_lines(candidate: InsightCandidate) -> list[str]:
    if not candidate.receipt_roles:
        return ["- evidence: no selector role assigned"]
    return [
        f"- `{role.receipt_id}`: {role.role} ({role.reason})"
        for role in candidate.receipt_roles
    ]


def _evidence_graph_lines(candidate: InsightCandidate) -> list[str]:
    if not candidate.evidence_graph:
        return ["- graph: no structured graph assigned"]
    return [
        f"- `{node.receipt_id}`: {node.role} ({node.reason})"
        for node in candidate.evidence_graph
    ]


def _claim_card_lines(candidate: InsightCandidate) -> list[str]:
    if not candidate.claim_cards:
        return ["- no structured claim cards assigned"]
    return [
        (
            f"- `{card.receipt_id}`: {card.role}; "
            f"design={_display_label(card.design, fallback=card.design.replace('_', ' '))}; "
            f"population={card.population}; outcome={_display_label(card.outcome)}; "
            f"direction={_display_label(card.direction, fallback=card.direction.replace('_', ' '))}; "
            f"support={card.support_type}/{card.confidence}"
        )
        for card in candidate.claim_cards
    ]
