"""Render V5 memo drafts from already-bound receipts."""
from __future__ import annotations

from collections.abc import Sequence

from v5_memo.gate import candidate_alpha_tier, memo_coverage_summary
from v5_memo.schemas import CorpusHit, InsightCandidate


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
        f"**Evidence bridge:** {bridge}.",
        "",
        f"**Tension:** {tension}.",
        "",
        "**Receipt roles:**",
        *_receipt_role_lines(candidate),
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
    lines.extend(_coverage_receipt_lines(receipts))
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
        "**Receipt roles:**",
        *_receipt_role_lines(candidate),
        "",
        "**Receipts:**",
    ]
    lines.extend(_receipt_line(index, hit) for index, hit in enumerate(receipts, start=1))
    lines.extend(_coverage_receipt_lines(receipts))
    return "\n".join(lines).strip() + "\n"


def _receipt_line(index: int, hit: CorpusHit) -> str:
    year = f", {hit.year}" if hit.year else ""
    venue = f", {hit.venue}" if hit.venue else ""
    locator = hit.doi or hit.url or hit.hit_id
    return f"{index}. `{hit.hit_id}` {hit.title}{year}{venue}. Source: {hit.source}. ID: {locator}"


def _coverage_receipt_lines(receipts: Sequence[CorpusHit]) -> list[str]:
    summary = memo_coverage_summary(receipts)
    years = summary["year_range"]
    year_text = "n/a"
    if isinstance(years, dict) and years.get("min") is not None and years.get("max") is not None:
        year_text = f"{years['min']}-{years['max']}"
    return [
        "",
        "**Coverage receipt:**",
        (
            f"- shards searched: `{summary['shards_searched']}`; "
            f"shard sources: `{_join(summary['sources_searched'])}`"
        ),
        (
            f"- receipt sources: `{_join(summary['sources_used'])}`; "
            f"years: `{year_text}`; abstract receipts: `{summary['abstract_receipt_count']}`"
        ),
        (
            f"- search passes: `{_join(summary['search_passes'])}`; "
            f"duplicate rate: `{summary['result_duplicate_rate']}`; "
            f"citation diversity: `{summary['result_citation_diversity']}`"
        ),
    ]


def _join(value: object) -> str:
    if isinstance(value, list | tuple):
        return ", ".join(str(item) for item in value) or "n/a"
    return str(value) if value else "n/a"


def _memo_title(candidate: InsightCandidate) -> str:
    if candidate.bridge_terms:
        return " / ".join(candidate.bridge_terms[:3])
    return candidate.topic


def _receipt_role_lines(candidate: InsightCandidate) -> list[str]:
    if not candidate.receipt_roles:
        return ["- evidence: no selector role assigned"]
    return [
        f"- `{role.receipt_id}`: {role.role} ({role.reason})"
        for role in candidate.receipt_roles
    ]
