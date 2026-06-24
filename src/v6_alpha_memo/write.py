"""Strict short memo writer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from v6_alpha_memo.score import ScoredPair
from v6_alpha_memo.search import CoverageReceipt, Paper

_MINIMAX_BASE_URL = "https://api.minimax.io/anthropic"


def render_memo(scored: ScoredPair, *, receipt: CoverageReceipt | None = None) -> str:
    pair = scored.pair
    title = _title(scored)
    lines = [
        f"# {title}",
        "",
        f"**One-sentence alpha:** {scored.expectation_update}",
        "",
        f"**Receipt 1:** {_receipt_line(pair.a)}",
        "",
        f"**Receipt 2:** {_receipt_line(pair.b)}",
        "",
        f"**Why this is surprising:** The pair has `{scored.shape}` geometry over "
        f"`{', '.join(pair.anchors[:3])}` rather than a broad literature-summary bridge.",
        "",
        "**Caveats/falsifiers:**",
        "- Reject if the shared anchor is not the same construct/intervention in the full text.",
        "- Reject if later receipts show the apparent reversal is only population, dose, or measurement noise.",
        "- Downgrade to discovery seed if either receipt is a review, case-only report, or keyword-only match.",
    ]
    if receipt is not None:
        lines.extend([
            "",
            "**Search receipt:** "
            f"hits={receipt.hits}; shards={receipt.shards_searched}/{receipt.shards_total}; "
            f"sources={','.join(receipt.sources_searched) or 'unknown'}; "
            f"papers_searched={receipt.papers_searched}; partial={receipt.partial}.",
        ])
    return "\n".join(lines).strip() + "\n"


def render_discovery_seed(scored: ScoredPair, *, receipt: CoverageReceipt | None = None) -> str:
    pair = scored.pair
    lines = [
        f"# Discovery seed: {' / '.join(pair.anchors[:2]) or 'receipt pair'}",
        "",
        "**Status:** Not alpha. The agent found receipt overlap, but no sharp reversal/update pair.",
        "",
        f"**Receipt 1:** {_receipt_line(pair.a)}",
        "",
        f"**Receipt 2:** {_receipt_line(pair.b)}",
        "",
        f"**Why it is only a seed:** The best pair stayed `{scored.shape}` over "
        f"`{', '.join(pair.anchors[:3])}`; it needs a second receipt that forces a real update.",
        "",
        "**Next falsifier/search target:** find a null, failed, endpoint-split, or human-translation receipt on the same intervention/construct.",
    ]
    if receipt is not None:
        lines.extend([
            "",
            "**Search receipt:** "
            f"hits={receipt.hits}; shards={receipt.shards_searched}/{receipt.shards_total}; "
            f"sources={','.join(receipt.sources_searched) or 'unknown'}; "
            f"papers_searched={receipt.papers_searched}; partial={receipt.partial}.",
        ])
    return "\n".join(lines).strip() + "\n"


def render_with_minimax(top_pairs: tuple[ScoredPair, ...], *, receipt: CoverageReceipt | None = None) -> str:
    judged = judge_with_minimax(top_pairs)
    if not judged:
        raise RuntimeError("MiniMax rejected all receipt pairs")
    top_pairs = judged
    api_key = _minimax_key()
    if not api_key:
        return render_memo(top_pairs[0], receipt=receipt)
    payload = {
        "model": os.environ.get("V5_MEMO_MINIMAX_MODEL", "MiniMax-M3"),
        "max_tokens": 900,
        "temperature": 0.2,
        "system": "Pick the strongest receipt pair and write only the required concise memo. Use only supplied receipts.",
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": [{"type": "text", "text": _prompt(top_pairs[:5])}]}],
    }
    base_url = os.environ.get("V5_MEMO_MINIMAX_BASE_URL", _MINIMAX_BASE_URL).rstrip("/")
    request = Request(
        f"{base_url}/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urlopen(request, timeout=float(os.environ.get("V5_MEMO_MINIMAX_TIMEOUT_SECONDS", "60"))) as response:
        data = json.loads(response.read().decode())
    text = _content_text(data).strip()
    return text + ("\n" if text else "")


def judge_with_minimax(top_pairs: tuple[ScoredPair, ...]) -> tuple[ScoredPair, ...]:
    """Return MiniMax-selected top pair, or empty tuple when it rejects all."""
    api_key = _minimax_key()
    if not api_key:
        return top_pairs
    payload = {
        "model": os.environ.get("V5_MEMO_MINIMAX_MODEL", "MiniMax-M3"),
        "max_tokens": 300,
        "temperature": 0.0,
        "system": (
            "You are a strict alpha memo selector. Pick only one receipt pair if it has "
            "sharp novelty and expectation-update geometry. Otherwise reject all. Return only JSON."
        ),
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": [{"type": "text", "text": _judge_prompt(top_pairs[:5])}]}],
    }
    base_url = os.environ.get("V5_MEMO_MINIMAX_BASE_URL", _MINIMAX_BASE_URL).rstrip("/")
    request = Request(
        f"{base_url}/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urlopen(request, timeout=float(os.environ.get("V5_MEMO_MINIMAX_TIMEOUT_SECONDS", "60"))) as response:
        data = json.loads(response.read().decode())
    choice = _parse_choice(_content_text(data))
    if choice is None or choice < 1 or choice > len(top_pairs):
        return ()
    return (top_pairs[choice - 1],)


def _title(scored: ScoredPair) -> str:
    anchor = " / ".join(scored.pair.anchors[:2]) or "receipt pair"
    return f"Alpha memo: {anchor} {scored.shape.replace('_', ' ')}"


def _receipt_line(paper: Paper) -> str:
    bits = [paper.title]
    if paper.year:
        bits.append(str(paper.year))
    if paper.doi:
        bits.append(paper.doi)
    return " | ".join(bits)


def _prompt(pairs: tuple[ScoredPair, ...]) -> str:
    rows = []
    for idx, scored in enumerate(pairs, start=1):
        a, b = scored.pair.a, scored.pair.b
        rows.append(
            {
                "id": idx,
                "score": scored.score,
                "shape": scored.shape,
                "expectation_update": scored.expectation_update,
                "anchors": scored.pair.anchors,
                "receipt_1": _paper_json(a),
                "receipt_2": _paper_json(b),
            }
        )
    return (
        "Return a short memo with: title, one-sentence alpha, receipt 1, receipt 2, "
        "why surprising, caveats/falsifiers. No broad framing beyond receipts.\n"
        + json.dumps(rows, ensure_ascii=False)
    )


def _judge_prompt(pairs: tuple[ScoredPair, ...]) -> str:
    rows = []
    for idx, scored in enumerate(pairs, start=1):
        rows.append({
            "id": idx,
            "score": scored.score,
            "shape": scored.shape,
            "expectation_update": scored.expectation_update,
            "anchors": scored.pair.anchors,
            "receipt_1": _paper_json(scored.pair.a),
            "receipt_2": _paper_json(scored.pair.b),
        })
    return (
        "Choose the one pair that is most likely to make an 8.5/10+ novelty memo. "
        "Reject all weak, obvious, review-like, keyword-only, or broad-title pairs. "
        "Return JSON exactly like {\"choice\": 1, \"reason\": \"...\"} or "
        "{\"choice\": null, \"reason\": \"...\"}.\n"
        + json.dumps(rows, ensure_ascii=False)
    )


def _parse_choice(text: str) -> int | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    choice = data.get("choice") if isinstance(data, dict) else None
    return choice if isinstance(choice, int) else None


def _paper_json(paper: Paper) -> dict[str, object]:
    return {"title": paper.title, "abstract": paper.abstract[:900], "year": paper.year, "doi": paper.doi}


def _minimax_key() -> str:
    for name in ("V6_MINIMAX_API_KEY", "V5_MEMO_MINIMAX_API_KEY", "MINIMAX_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    path = Path.home() / ".codex" / "secrets" / "minimax_api_key"
    return path.read_text().strip() if path.exists() else ""


def _content_text(data: object) -> str:
    if not isinstance(data, dict):
        return ""
    content = data.get("content")
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    return str(data.get("text", ""))
