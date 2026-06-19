"""Render V5 memo drafts from already-bound receipts."""
from __future__ import annotations

from collections.abc import Sequence

from v5_memo.schemas import CorpusHit, InsightCandidate


def render_memo(candidate: InsightCandidate, receipts: Sequence[CorpusHit]) -> str:
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
        f"**Evidence bridge:** {bridge}.",
        "",
        f"**Tension:** {tension}.",
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


def _receipt_line(index: int, hit: CorpusHit) -> str:
    year = f", {hit.year}" if hit.year else ""
    venue = f", {hit.venue}" if hit.venue else ""
    locator = hit.doi or hit.url or hit.hit_id
    return f"{index}. `{hit.hit_id}` {hit.title}{year}{venue}. Source: {hit.source}. ID: {locator}"


def _memo_title(candidate: InsightCandidate) -> str:
    if candidate.bridge_terms:
        return " / ".join(candidate.bridge_terms[:3])
    return candidate.topic
