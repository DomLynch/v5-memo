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
_BAD_ANCHOR = frozenset({
    "associated", "background", "combination", "conclusion", "control", "divided",
    "elisa", "significant", "significantly",
})
_CONTEXT_ANCHOR = frozenset({
    "aging", "biomarker", "biomarkers", "biology", "cell", "cells", "disease", "function",
    "functions", "gene", "genes", "health", "human", "humans", "model", "models", "outcome",
    "outcomes", "pathway", "pathways", "protein", "proteins", "trial", "trials",
})
_NONPRIMARY_PHRASES = (
    "case report", "commentary", "dispatch", "editorial", "in brief", "meta-analysis",
    "news and views", "news & views", "perspective", "research highlight", "systematic review",
)
_ANIMAL = frozenset({"mice", "mouse", "rat", "rats"})
_HUMAN_TOPIC = frozenset({
    "adult", "adults", "employee", "employees", "field", "firm", "firms",
    "human", "humans", "men", "participants", "people", "trial", "women", "workers",
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
    anchors = _real_anchors(pair, topic_terms)
    clean_pair = CandidatePair(a=a, b=b, anchors=anchors, reject_reasons=pair.reject_reasons)
    reasons: list[str] = [f"shared_anchor:{anchor}" for anchor in anchors[:3]]
    if not anchors:
        return ScoredPair(clean_pair, 0, "shared_anchor", "", ("reject:no_real_anchor",))
    hygiene_reject = _receipt_hygiene_reject(a, b, anchors)
    if hygiene_reject:
        return ScoredPair(clean_pair, 0, "shared_anchor", "", (*reasons, hygiene_reject))
    score = 20 + min(len(anchors), 4) * 5
    shape = "shared_anchor"
    first, second = a, b

    if _has(at, _PROMISE) and _has(bt, _FAILURE) and _roles_fit("promise_reversal", a, b, topic_terms):
        score += 40
        shape = "promise_reversal"
        reasons.append("promise_to_negative_or_null")
    elif _has(bt, _PROMISE) and _has(at, _FAILURE) and _roles_fit("promise_reversal", b, a, topic_terms):
        score += 40
        shape = "promise_reversal"
        reasons.append("promise_to_negative_or_null")
        first, second = b, a

    if _roles_fit("mechanism_to_human_failure", first, second, topic_terms):
        score += 30
        shape = "mechanism_to_human_failure"
        reasons.append("mechanism_or_animal_to_human_failure")
    if _has(at, _PROTOCOL) and _has(bt, _RESULT | _FAILURE) and _roles_fit("protocol_result_mismatch", a, b, topic_terms):
        score += 20
        shape = "protocol_result_mismatch"
        reasons.append("protocol_result_mismatch")
    elif _has(bt, _PROTOCOL) and _has(at, _RESULT | _FAILURE) and _roles_fit("protocol_result_mismatch", b, a, topic_terms):
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
    if shape != "shared_anchor" and not _role_matches_topic(first, second, topic_terms):
        score = 0
        reasons.append("role_mismatch:topic_construct")
    update = _expectation_sentence(first, second, shape)
    return ScoredPair(
        pair=clean_pair,
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


def _real_anchors(pair: CandidatePair, topic_terms: frozenset[str]) -> tuple[str, ...]:
    title_a = set(_WORD_RE.findall(pair.a.title.casefold()))
    title_b = set(_WORD_RE.findall(pair.b.title.casefold()))
    kept = []
    for anchor in pair.anchors:
        if anchor in _BAD_ANCHOR:
            continue
        if (topic_terms and anchor in topic_terms) or (anchor in title_a and anchor in title_b):
            kept.append(anchor)
    return tuple(dict.fromkeys(kept))[:6]


def _receipt_hygiene_reject(a: Paper, b: Paper, anchors: tuple[str, ...]) -> str:
    if _nonprimary(a) or _nonprimary(b):
        return "reject:non_primary_receipt"
    title_a = set(_WORD_RE.findall(a.title.casefold()))
    title_b = set(_WORD_RE.findall(b.title.casefold()))
    if not any(anchor not in _CONTEXT_ANCHOR and anchor in title_a and anchor in title_b for anchor in anchors):
        return "reject:name_or_context_only_anchor"
    return ""


def _nonprimary(paper: Paper) -> bool:
    text = paper.text.casefold()
    return any(phrase in text for phrase in _NONPRIMARY_PHRASES)


def _tokens(paper: Paper) -> set[str]:
    return set(_WORD_RE.findall(paper.text.casefold()))


def _roles_fit(shape: str, first: Paper, second: Paper, topic_terms: frozenset[str]) -> bool:
    ft, st = _tokens(first), _tokens(second)
    if not _role_matches_topic(first, second, topic_terms):
        return False
    if _human_topic(topic_terms) and not _is_human(second):
        return False
    if shape == "mechanism_to_human_failure":
        return _has(ft, _MECHANISM) and _is_human(second) and _has(st, _FAILURE)
    if shape == "protocol_result_mismatch":
        return _has(ft, _PROTOCOL) and _has(st, _RESULT | _FAILURE)
    if shape == "promise_reversal":
        if _animal_only(second) and (_human_topic(topic_terms) or _is_human(first)):
            return False
        return _has(ft, _PROMISE) and _has(st, _FAILURE)
    return False


def _role_matches_topic(a: Paper, b: Paper, topic_terms: frozenset[str]) -> bool:
    if not topic_terms:
        return True
    left = _loose_tokens(a) & topic_terms
    right = _loose_tokens(b) & topic_terms
    if not left or not right:
        return False
    shared = left & right
    required = 2 if len(topic_terms) >= 3 else 1
    return len(shared) >= required or len(shared) * 2 >= len(left | right)


def _loose_tokens(paper: Paper) -> set[str]:
    tokens = set(re.findall(r"[a-z][a-z0-9]{2,}", paper.text.casefold()))
    return tokens | {word[:-1] for word in tokens if word.endswith("s") and len(word) > 4}


def _has(tokens: set[str], needles: frozenset[str]) -> bool:
    return bool(tokens & needles)


def _human_topic(topic_terms: frozenset[str]) -> bool:
    return bool(topic_terms & _HUMAN_TOPIC)


def _is_human(paper: Paper) -> bool:
    tokens = _tokens(paper)
    return _has(tokens, _HUMAN_OUTCOME) and not _animal_only(paper)


def _animal_only(paper: Paper) -> bool:
    tokens = _tokens(paper)
    return _has(tokens, _ANIMAL) and not _has(tokens, _HUMAN_OUTCOME)


def _short(text: str) -> str:
    return text.replace("-", " ")[:60]
