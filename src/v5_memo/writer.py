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
    bridge = ", ".join(candidate.bridge_terms) or "unspecified bridge"
    tension = ", ".join(candidate.tension_terms) or "not detected"
    hypothesis = _bounded_hypothesis(candidate)
    lines = [
        f"# Alpha memo: {_memo_title(candidate)}",
        "",
        f"**Alpha hypothesis:** {hypothesis}",
        "",
        f"**Signal score:** `{candidate.score}` "
        f"(novelty `{candidate.novelty_score}`, evidence `{candidate.evidence_score}`)",
        "",
        f"**Evidence bridge:** {bridge}.",
        "",
        f"**Tension:** {tension}.",
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
        "**Why it matters:**",
        (
            "This is a receipt-bound lead for investigation: two independent "
            "search hits share a non-obvious bridge term, so the memo can test "
            "whether that bridge explains a boundary condition or a new angle."
        ),
        "",
        "**What would falsify it:**",
        _falsification_clause(candidate),
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
        clean = value.replace("_", " ").strip().casefold()
        if not clean or clean in {"long", "outcome", "outcomes", "unspecified"}:
            continue
        if clean not in out:
            out.append(clean)
    return out


def _join_labels(labels: Sequence[str]) -> str:
    if len(labels) <= 1:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return f"{', '.join(labels[:-1])}, and {labels[-1]}"


def _memo_title(candidate: InsightCandidate) -> str:
    if candidate.bridge_terms:
        terms = [term for term in candidate.bridge_terms if term not in _TITLE_STOPWORDS]
        return " / ".join((terms or list(candidate.bridge_terms))[:3])
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
            f"- `{card.receipt_id}`: {card.role}; design={card.design}; "
            f"population={card.population}; outcome={card.outcome}; "
            f"direction={card.direction}; support={card.support_type}/{card.confidence}"
        )
        for card in candidate.claim_cards
    ]
