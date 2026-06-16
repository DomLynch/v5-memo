"""Mine receipt-bound '2 + 2 = 5' alpha hypotheses from corpus hits."""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from itertools import combinations

from v5_memo.schemas import CorpusHit, InsightCandidate
from v5_memo.scorer import score_connection

_WORD = re.compile(r"[a-z][a-z0-9]{2,}")
_STOP = frozenset({
    "about", "advances", "after", "agent", "among", "analysis", "and", "based", "beneficial",
    "between", "can", "cell", "clinical", "comprehensive", "data", "effect",
    "effects", "evidence", "finding", "findings", "from", "group", "groups",
    "human", "impact", "isi", "library", "links", "marker", "markers", "meta", "model",
    "models", "paper", "patients", "predicts", "recent", "reported", "research",
    "response", "results", "review", "shows", "significant", "study", "studies",
    "summary", "systematic", "through", "trial", "using", "with",
})
_POSITIVE = frozenset({"increase", "increased", "raises", "raised", "improve", "improved"})
_NEGATIVE = frozenset({"decrease", "decreased", "reduce", "reduced", "lower", "lowered"})
_NULL = frozenset({"null", "neutral", "unchanged", "failed", "nonsignificant"})


def mine_insights(
    hits: Sequence[CorpusHit],
    *,
    topic: str,
    max_candidates: int = 5,
) -> list[InsightCandidate]:
    """Return ranked alpha candidates from source-diverse hit pairs."""
    clean_hits = _dedupe_hits(hits)
    if len(clean_hits) < 2:
        return []

    topic_tokens = _tokens(topic)
    token_sets = {hit.hit_id: _tokens(hit.text) - topic_tokens for hit in clean_hits}
    doc_counts = Counter(term for terms in token_sets.values() for term in terms)

    candidates: list[InsightCandidate] = []
    for left, right in combinations(clean_hits, 2):
        bridge = _bridge_terms(token_sets[left.hit_id], token_sets[right.hit_id], doc_counts)
        if not bridge:
            continue
        source_keys = {left.source_key, right.source_key}
        if len(source_keys) < 2:
            continue
        tension_terms = _tension_terms(left.text, right.text)
        score = score_connection(
            bridge_terms=bridge,
            bridge_doc_counts=doc_counts,
            unique_source_count=len(source_keys),
            receipt_count=2,
            has_tension=bool(tension_terms),
        )
        candidates.append(InsightCandidate(
            topic=topic,
            thesis=_thesis(topic, bridge, left, right, tension_terms),
            bridge_terms=bridge,
            tension_terms=tension_terms,
            receipt_ids=(left.hit_id, right.hit_id),
            score=score.score,
            novelty_score=score.novelty_score,
            evidence_score=score.evidence_score,
            reasons=score.reasons,
        ))
    return sorted(candidates, key=lambda c: (c.score, c.novelty_score), reverse=True)[
        :max(0, max_candidates)
    ]


def _dedupe_hits(hits: Sequence[CorpusHit]) -> list[CorpusHit]:
    seen: set[str] = set()
    out: list[CorpusHit] = []
    for hit in hits:
        if not hit.hit_id or hit.source_key in seen:
            continue
        seen.add(hit.source_key)
        out.append(hit)
    return out


def _tokens(text: str) -> frozenset[str]:
    return frozenset(t for t in _WORD.findall(text.casefold()) if t not in _STOP)


def _bridge_terms(
    left: frozenset[str], right: frozenset[str], doc_counts: Counter[str],
) -> tuple[str, ...]:
    shared = left & right
    ranked = sorted(shared, key=lambda term: (doc_counts[term], term))
    return tuple(ranked[:4])


def _polarity(text: str) -> frozenset[str]:
    tokens = _tokens(text)
    out: set[str] = set()
    if tokens & _POSITIVE:
        out.add("positive")
    if tokens & _NEGATIVE:
        out.add("negative")
    if tokens & _NULL:
        out.add("null")
    return frozenset(out)


def _tension_terms(left: str, right: str) -> tuple[str, ...]:
    a = _polarity(left)
    b = _polarity(right)
    if not a or not b or a == b:
        return ()
    return tuple(sorted(a | b))


def _axis(hit: CorpusHit, excluded: tuple[str, ...]) -> str:
    excluded_set = set(excluded)
    title_terms = [t for t in _WORD.findall(hit.title.casefold()) if t not in _STOP]
    axis = [t for t in title_terms if t not in excluded_set][:4]
    return " ".join(axis) or (hit.venue or hit.source)


def _thesis(
    topic: str,
    bridge_terms: tuple[str, ...],
    left: CorpusHit,
    right: CorpusHit,
    tension_terms: tuple[str, ...],
) -> str:
    bridge = " / ".join(bridge_terms[:3])
    left_axis = _axis(left, bridge_terms)
    right_axis = _axis(right, bridge_terms)
    if tension_terms:
        return (
            f"{topic} may be hiding a {bridge} boundary condition: "
            f"{left_axis} and {right_axis} point in different directions."
        )
    return f"{topic} may have a {bridge} bridge between {left_axis} and {right_axis}."
