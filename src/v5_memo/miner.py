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
_DENOMINATOR = frozenset({"cohort", "population", "aggregate", "prospective", "longitudinal"})
_TAIL = frozenset({"case", "cases", "fatal", "fatality", "death", "deaths", "risk", "rare"})
_TIMING = frozenset({"acute", "chronic", "short", "long", "early", "late", "immediate", "delayed"})
_ROLE_A = frozenset({"cause", "causes", "driver", "drives", "predict", "predicts", "associated"})
_ROLE_B = frozenset({"confound", "confounds", "selection", "mediates", "moderates", "substitution"})
_METRIC = frozenset({"metric", "score", "benchmark", "accuracy", "performance", "clicks"})
_OUTCOME = frozenset({"outcome", "mortality", "injury", "error", "errors", "dispersion", "quality"})
_EXPERTISE = frozenset({"expert", "experts", "novice", "novices", "nonexpert", "nonexperts"})
_BOUNDARY = frozenset({"boundary", "context", "dose", "endpoint", "modality", "population", "setting"})


def mine_insights(
    hits: Sequence[CorpusHit],
    *,
    topic: str,
    required_anchor_terms: Sequence[str] = (),
    max_candidates: int = 5,
) -> list[InsightCandidate]:
    """Return ranked alpha candidates from source-diverse hit pairs."""
    clean_hits = _dedupe_hits(hits)
    if len(clean_hits) < 2:
        return []

    topic_tokens = _tokens(topic)
    full_token_sets = {hit.hit_id: _tokens(hit.text) for hit in clean_hits}
    token_sets = {hit.hit_id: full_token_sets[hit.hit_id] - topic_tokens for hit in clean_hits}
    anchor_terms = frozenset(required_anchor_terms)
    doc_counts = Counter(term for terms in token_sets.values() for term in terms)

    candidates: list[InsightCandidate] = []
    for left, right in combinations(clean_hits, 2):
        if anchor_terms and not _pair_has_anchor(
            full_token_sets[left.hit_id],
            full_token_sets[right.hit_id],
            anchor_terms,
        ):
            continue
        bridge = _bridge_terms(token_sets[left.hit_id], token_sets[right.hit_id], doc_counts)
        if not bridge:
            continue
        source_keys = {left.source_key, right.source_key}
        if len(source_keys) < 2:
            continue
        tension_terms = _tension_terms(left.text, right.text)
        shape_reasons = _shape_reasons(
            left,
            right,
            bridge_terms=bridge,
            tension_terms=tension_terms,
        )
        if not shape_reasons:
            continue
        score = score_connection(
            bridge_terms=bridge,
            bridge_doc_counts=doc_counts,
            unique_source_count=len(source_keys),
            receipt_count=2,
            has_tension=bool(tension_terms),
            shape_score=len(shape_reasons),
            shape_reasons=shape_reasons,
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


def query_anchor_terms(seed_queries: Sequence[str], *, limit: int = 2) -> tuple[str, ...]:
    """Return ordered anchor terms that chosen receipt pairs must preserve."""
    generic = {
        "angle",
        "condition",
        "exercise",
        "mechanism",
        "response",
        "stress",
    }
    out: list[str] = []
    seen: set[str] = set()
    for query in seed_queries:
        for token in _WORD.findall(query.casefold()):
            if token in _STOP or token in generic or token in seen:
                continue
            seen.add(token)
            out.append(token)
            if len(out) >= limit:
                return tuple(out)
    return tuple(out)


def _pair_has_anchor(
    left_tokens: frozenset[str],
    right_tokens: frozenset[str],
    anchor_terms: frozenset[str],
) -> bool:
    return bool(left_tokens & anchor_terms) and bool(right_tokens & anchor_terms)


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


def _shape_reasons(
    left: CorpusHit,
    right: CorpusHit,
    *,
    bridge_terms: tuple[str, ...],
    tension_terms: tuple[str, ...],
) -> tuple[str, ...]:
    left_tokens = _tokens(left.text)
    right_tokens = _tokens(right.text)
    all_tokens = left_tokens | right_tokens
    reasons: list[str] = []
    if tension_terms:
        reasons.append("shape:directional_reversal")
    if (
        len(bridge_terms) >= 2
        and _axis(left, bridge_terms) != _axis(right, bridge_terms)
        and (tension_terms or all_tokens & _BOUNDARY)
    ):
        reasons.append("shape:boundary_condition")
    if all_tokens & _DENOMINATOR and all_tokens & _TAIL:
        reasons.append("shape:denominator_split")
    if left_tokens & _TIMING and right_tokens & _TIMING and left_tokens != right_tokens:
        reasons.append("shape:timing_split")
    if all_tokens & _ROLE_A and all_tokens & _ROLE_B:
        reasons.append("shape:role_inversion")
    if all_tokens & _METRIC and all_tokens & _OUTCOME:
        reasons.append("shape:measurement_mismatch")
    if all_tokens & _EXPERTISE:
        reasons.append("shape:expertise_split")
    return tuple(dict.fromkeys(reasons))


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
