"""Universal receipt-geometry scorer."""

from __future__ import annotations

import re
from dataclasses import dataclass

from v6_alpha_memo.mine import CandidatePair
from v6_alpha_memo.search import Paper

_WORD_RE = re.compile(r"[a-z][a-z0-9-]{2,}")
_PROMISE = frozenset({
    "activate", "activated", "benefit", "enhance", "enhanced", "improve",
    "improved", "increase", "increased", "mimetic", "mimic", "promote",
    "protect", "protected", "raise", "raised", "superior",
})
_FAILURE = frozenset({
    "attenuate", "attenuated", "blunt", "blunted", "decrease", "decreased",
    "failed", "failure", "impair", "impaired", "lower", "lowered", "null",
    "reduce", "reduced", "worse", "worsened",
})
_MECHANISM = frozenset({
    "animal", "cell", "cells", "in-vitro", "mechanism", "mechanistic", "mice",
    "model", "mouse", "pathway", "preclinical", "rat", "rats",
})
_HUMAN_OUTCOME = frozenset({
    "adult", "adults", "employee", "employees", "field", "firm", "firms",
    "human", "humans", "participants", "patient", "patients", "randomized",
    "trial", "workers",
})
_PROTOCOL = frozenset({"expected", "hypothesis", "intended", "planned", "protocol"})
_RESULT = frozenset({"found", "observed", "result", "results", "showed", "shows"})
_BOUNDARY = frozenset({
    "context", "dose", "endpoint", "endpoints", "market", "modality", "program",
    "selection", "subgroup", "task", "timing",
})


@dataclass(frozen=True, slots=True)
class ScoredPair:
    pair: CandidatePair
    score: int
    shape: str
    expectation_update: str
    reasons: tuple[str, ...]


def score_pairs(
    pairs: tuple[CandidatePair, ...],
    *,
    min_score: int = 55,
    topic_terms: set[str] | frozenset[str] = frozenset(),
) -> tuple[ScoredPair, ...]:
    scoped_terms = frozenset(topic_terms)
    scored = [score_pair(pair, topic_terms=scoped_terms) for pair in pairs]
    kept = [item for item in scored if item.score >= min_score and item.expectation_update]
    kept.sort(key=lambda item: item.score, reverse=True)
    return tuple(kept)


def score_pair(pair: CandidatePair, *, topic_terms: frozenset[str] = frozenset()) -> ScoredPair:
    a, b = pair.a, pair.b
    at, bt = _tokens(a), _tokens(b)
    reasons: list[str] = [f"shared_anchor:{anchor}" for anchor in pair.anchors[:3]]
    score = 20 + min(len(pair.anchors), 4) * 5
    shape = "shared_anchor"
    first, second = a, b

    if _has(at, _PROMISE) and _has(bt, _FAILURE):
        score += 40
        shape = "promise_reversal"
        reasons.append("promise_to_negative_or_null")
    elif _has(bt, _PROMISE) and _has(at, _FAILURE):
        score += 40
        shape = "promise_reversal"
        reasons.append("promise_to_negative_or_null")
        first, second = b, a

    ft, st = _tokens(first), _tokens(second)
    if _has(ft, _MECHANISM) and _has(st, _HUMAN_OUTCOME) and _has(st, _FAILURE):
        score += 30
        shape = "mechanism_to_human_failure"
        reasons.append("mechanism_or_animal_to_human_failure")
    if _has(at, _PROTOCOL) and _has(bt, _RESULT | _FAILURE):
        score += 20
        shape = "protocol_result_mismatch"
        reasons.append("protocol_result_mismatch")
    elif _has(bt, _PROTOCOL) and _has(at, _RESULT | _FAILURE):
        score += 20
        shape = "protocol_result_mismatch"
        reasons.append("protocol_result_mismatch")
        first, second = b, a
    if _has(at | bt, _BOUNDARY) and (_has(at, _PROMISE) != _has(bt, _PROMISE)):
        score += 15
        reasons.append("boundary_or_endpoint_split")
    if a.source.casefold() != b.source.casefold():
        score += 5
        reasons.append("source_diverse")
    if shape == "promise_reversal" and not _role_matches_topic(first, second, pair.anchors, topic_terms):
        score -= 50
        reasons.append("role_mismatch:topic_construct")

    update = _expectation_sentence(first, second, shape)
    return ScoredPair(
        pair=pair,
        score=min(score, 100),
        shape=shape,
        expectation_update=update,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _expectation_sentence(a: Paper, b: Paper, shape: str) -> str:
    if shape == "shared_anchor":
        return ""
    anchor = _short(_best_anchor(a, b))
    return (
        f"{a.title} made us expect {anchor} would travel cleanly as a positive signal; "
        f"{b.title} forces the update that the same anchor can fail, reverse, or split by context."
    )


def _best_anchor(a: Paper, b: Paper) -> str:
    common = _tokens(a) & _tokens(b)
    for word in sorted(common, key=lambda item: (-len(item), item)):
        if word not in _PROMISE and word not in _FAILURE:
            return word
    return "the shared intervention"


def _tokens(paper: Paper) -> set[str]:
    return set(_WORD_RE.findall(paper.text.casefold()))


def _role_matches_topic(a: Paper, b: Paper, anchors: tuple[str, ...], topic_terms: frozenset[str]) -> bool:
    constructs = topic_terms - set(anchors)
    if not constructs:
        return True
    return bool(_loose_tokens(a) & constructs) and bool(_loose_tokens(b) & constructs)


def _loose_tokens(paper: Paper) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9]{2,}", paper.text.casefold()))


def _has(tokens: set[str], needles: frozenset[str]) -> bool:
    return bool(tokens & needles)


def _short(text: str) -> str:
    return text.replace("-", " ")[:60]
