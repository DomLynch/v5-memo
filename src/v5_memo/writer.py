"""Render V5 memo drafts from already-bound receipts."""
from __future__ import annotations

from collections.abc import Sequence

from v5_memo.gate import candidate_alpha_tier
from v5_memo.schemas import CorpusHit, InsightCandidate

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
    lines = [
        f"# Alpha memo: {_memo_title(candidate)}",
        "",
        f"**Alpha hypothesis:** {candidate.thesis}",
        "",
        f"**Signal score:** `{candidate.score}` "
        f"(novelty `{candidate.novelty_score}`, evidence `{candidate.evidence_score}`)",
        "",
        "**Scorecard:**",
        *_scorecard_lines(candidate),
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
        (
            "A follow-up search fails to find direct receipts where the bridge "
            "term and both evidence streams appear in the same source-grounded "
            "claim, or the apparent connection collapses to one duplicated source."
        ),
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
    locator = hit.doi or hit.url or hit.hit_id
    return f"{index}. `{hit.hit_id}` {hit.title}{year}{venue}. Source: {hit.source}. ID: {locator}"


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


def _scorecard_lines(candidate: InsightCandidate) -> list[str]:
    if not candidate.scorecard:
        return ["- no structured scorecard assigned"]
    return [f"- {key}: {value}" for key, value in sorted(candidate.scorecard.items())]
