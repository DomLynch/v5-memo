"""Render V5 memo drafts from already-bound receipts."""
from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from v5_memo.gate import candidate_alpha_tier
from v5_memo.schemas import ClaimCard, CorpusHit, InsightCandidate

_TITLE_STOPWORDS = frozenset({"during", "after", "before", "following", "under", "with"})
_NAMED_STUDY_RE = re.compile(r"\(([A-Z][A-Z0-9-]{2,15})\)\s+(?:study|trial)\b")
_REDUCTION_TITLE_RE = re.compile(
    r"\breduction in (?P<endpoint>.+?)(?=[:.;]|\s+(?:among|after|during|following|in (?:adults|patients|participants))\b|$)",
    re.IGNORECASE,
)
_DIRECTIONAL_TITLE_RE = re.compile(
    r"\b(?P<direction>attenuates?|blunts?|reduces?|decreases?|increases?|improves?|worsens?|alters?|modifies?)\s+"
    r"(?P<endpoint>.+?)(?=[:.;]|\s+(?:among|after|during|following|in (?:adults|patients|participants))\b|$)",
    re.IGNORECASE,
)
_DIRECTION_DISPLAY = {
    "alter": "altered",
    "attenuate": "attenuated",
    "blunt": "blunted",
    "decrease": "decreased",
    "improve": "improved",
    "increase": "increased",
    "modify": "modified",
    "reduce": "reduced",
    "worsen": "worsened",
}


def render_memo(candidate: InsightCandidate, receipts: Sequence[CorpusHit]) -> str:
    if candidate_alpha_tier(candidate) == "discovery_seed":
        return render_discovery_seed(candidate, receipts)
    return render_alpha_memo(candidate, receipts)


def render_alpha_memo(candidate: InsightCandidate, receipts: Sequence[CorpusHit]) -> str:
    """Render a short memo without adding claims outside the candidate receipts."""
    if not receipts:
        raise ValueError("cannot render memo without receipts")
    direct_cards = _direct_human_cards(candidate)
    study_family = _study_family_label(receipts)
    bridge = ", ".join(_clean_labels(candidate.bridge_terms)) or "unspecified bridge"
    tension = ", ".join(_clean_labels(candidate.tension_terms)) or "not detected"
    hypothesis = _bounded_hypothesis(candidate, study_family=study_family)
    lines = [
        f"# Alpha memo: {_memo_title(candidate, study_family=study_family)}",
        "",
        "Hypothesis-level alpha signal; not clinical advice.",
        "",
        f"**Alpha hypothesis:** {hypothesis}",
        "",
        "**Core signal:**",
        *_core_signal_lines(candidate, direct_cards, study_family=study_family),
        "",
        "**Receipt-level synthesis:**",
        *_synthesis_lines(candidate, direct_cards, receipts, study_family=study_family),
        "",
        "**Limits:**",
        *_limit_lines(candidate, direct_cards, study_family=study_family),
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


def _bounded_hypothesis(candidate: InsightCandidate, *, study_family: str = "") -> str:
    direct_cards = _direct_human_cards(candidate)
    if len(direct_cards) < 2:
        return candidate.thesis
    if study_family:
        designs = _non_generic_labels(card.design for card in direct_cards)
        design_text = _join_labels(designs[:2]) or "study"
        return (
            f"Within the {study_family} trial program, direct human {design_text} companion analyses "
            "report endpoint-specific findings from one evidence unit; treat them as hypothesis-level "
            "until the same endpoints are independently replicated."
        )
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
    scope = f"Within the {study_family} study program" if study_family else f"In {candidate.topic}"
    evidence_noun = "analyses" if study_family else "receipts"
    return (
        f"{scope}, direct human {design_text} {evidence_noun} support a bounded "
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


def _core_signal_lines(
    candidate: InsightCandidate,
    direct_cards: Sequence[ClaimCard],
    *,
    study_family: str = "",
) -> list[str]:
    thesis = candidate.thesis.strip()
    if len(direct_cards) < 2:
        return [thesis if thesis.endswith(".") else f"{thesis}."]
    if study_family:
        designs = _non_generic_labels(card.design for card in direct_cards)
        return [
            (
                f"The direct human companion analyses are {_join_labels(designs[:2]) or 'study'} evidence "
                f"from the same {study_family} trial program, not independent trials."
            ),
            "Each article is therefore reported by its own endpoint and textual direction, without pooling effects.",
        ]
    outcomes = _non_generic_labels(card.outcome for card in direct_cards)
    directions = _non_generic_labels(
        direction
        for card in direct_cards
        for direction in card.direction.split("/")
        if direction not in {"proxy", "unclear"}
    )
    designs = _non_generic_labels(card.design for card in direct_cards)
    evidence_noun = "companion analyses" if study_family else "receipts"
    return [
        (
            f"The direct human {evidence_noun} are {_join_labels(designs[:2]) or 'study'} evidence "
            f"for {_join_labels(outcomes[:3]) or 'the cited endpoints'}."
        ),
        (
            f"Because the recorded directions are {_join_labels(directions[:3]) or 'mixed'}, "
            "this memo treats the bundle as endpoint-specific evidence rather than a pooled clinical effect."
        ),
    ]


def _synthesis_lines(
    candidate: InsightCandidate,
    direct_cards: Sequence[ClaimCard],
    receipts: Sequence[CorpusHit],
    *,
    study_family: str = "",
) -> list[str]:
    del candidate
    cards = list(direct_cards)
    if not cards:
        return ["- No direct human claim cards were assigned; keep this as a discovery lead only."]
    hits_by_id = {
        receipt_id: hit
        for hit in receipts
        for receipt_id in {hit.hit_id, hit.receipt_id}
    }
    lines: list[str] = []
    if study_family:
        lines.append(
            f"- Evidence unit: the listed articles are endpoint-specific companion analyses from the same "
            f"{study_family} trial program; they count as one study program, not independent replication."
        )
    dated_receipts = [f"`{hit.receipt_id}` ({hit.year})" for hit in receipts if hit.year]
    if dated_receipts:
        lines.append(
            "- Publication metadata: source records report "
            f"{_join_labels(dated_receipts)}; these are article publication years, not trial dates."
        )
    lines.extend(
        _claim_summary_line(card, hit=hits_by_id.get(card.receipt_id), study_family=study_family)
        for card in cards
    )
    return lines


def _claim_summary_line(card: ClaimCard, *, hit: CorpusHit | None = None, study_family: str = "") -> str:
    outcome, direction = _endpoint_finding(card, hit)
    design = _display_label(card.design, fallback=card.design.replace("_", " "))
    receipt_id = hit.receipt_id if hit else card.receipt_id
    family = f"{study_family} companion analysis; " if study_family else ""
    source_title = f" Source title: {hit.title.strip()}" if hit and hit.title.strip() else ""
    return (
        f"- `{receipt_id}`: {family}{design} in {card.population}; endpoint: {outcome}; "
        f"direction: {direction}.{source_title}"
    )


def _endpoint_finding(card: ClaimCard, hit: CorpusHit | None) -> tuple[str, str]:
    fallback_direction = _display_label(card.direction, fallback=card.direction.replace("_", " "))
    if not hit or not hit.title.strip():
        return _display_label(card.outcome), fallback_direction
    title = hit.title.strip().rstrip(".")
    if match := _REDUCTION_TITLE_RE.search(title):
        return _clean_endpoint(match.group("endpoint")), "reduced"
    if match := _DIRECTIONAL_TITLE_RE.search(title):
        raw_direction = match.group("direction").casefold().removesuffix("s")
        return _clean_endpoint(match.group("endpoint")), _DIRECTION_DISPLAY.get(raw_direction, fallback_direction)
    return _display_label(card.outcome, fallback=title), fallback_direction


def _clean_endpoint(value: str) -> str:
    clean = re.sub(r"^(?:a|an|the)\s+", "", value.strip(), flags=re.IGNORECASE)
    clean = re.sub(r"^clinically significant\s+", "", clean, flags=re.IGNORECASE)
    return clean.rstrip(" .:;").casefold()


def _limit_lines(
    candidate: InsightCandidate,
    direct_cards: Sequence[ClaimCard],
    *,
    study_family: str = "",
) -> list[str]:
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
    if study_family:
        limits.insert(
            0,
            f"The {study_family} papers are companion analyses from the same trial program and one evidence unit, "
            "not independent trials.",
        )
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


def _memo_title(candidate: InsightCandidate, *, study_family: str = "") -> str:
    if candidate.bridge_terms:
        bridge_terms = _clean_labels(candidate.bridge_terms)
        terms = [term for term in bridge_terms if term not in _TITLE_STOPWORDS]
        if study_family:
            bridge = _join_labels([term.title() for term in (terms or bridge_terms)[:2]])
            return f"{study_family} Study: Endpoint-Specific {bridge} Findings"
        return " / ".join((terms or bridge_terms)[:3])
    return candidate.topic


def _study_family_label(receipts: Sequence[CorpusHit]) -> str:
    for hit in receipts:
        if match := _NAMED_STUDY_RE.search(hit.text):
            return match.group(1)
    return ""


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
