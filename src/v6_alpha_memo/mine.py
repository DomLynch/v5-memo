"""Mine receipt pairs with real shared anchors."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations

from v6_alpha_memo.search import Paper

_WORD_RE = re.compile(r"[a-z][a-z0-9]{2,}")
_STOP = frozenset({
    "about", "after", "again", "against", "among", "and", "are", "background",
    "based", "between", "both", "but", "case", "cases", "conclusion",
    "conclusions", "control", "controls", "data", "divided", "effect",
    "effects", "elisa", "evidence", "for", "from", "finding", "findings",
    "group", "groups", "has", "have", "human", "humans", "impact", "into",
    "body", "compared", "consumption", "continuous", "day", "days",
    "difference", "end", "exercises", "four", "however", "indices", "level",
    "levels", "male", "may", "measured", "method", "methods", "model",
    "models", "not", "old", "other", "per", "performed",
    "outcome", "outcomes", "paper", "patients", "power", "review", "response",
    "progressive", "protein", "proteins", "received", "recommended",
    "responses", "result", "results", "study", "studies", "system", "systems",
    "the", "therapy", "this", "through", "tissue", "trial", "trials", "using",
    "significant", "significantly", "week", "weeks", "were", "wistar", "with", "within",
})
_BAD_SHARED = _STOP | {
    "association", "analysis", "clinical", "combination", "decreased", "different",
    "increase", "increased", "mechanism", "reduced",
}


@dataclass(frozen=True, slots=True)
class CandidatePair:
    a: Paper
    b: Paper
    anchors: tuple[str, ...]
    reject_reasons: tuple[str, ...] = ()


def mine_pairs(papers: tuple[Paper, ...], *, limit: int = 80) -> tuple[CandidatePair, ...]:
    pairs: list[CandidatePair] = []
    for a, b in combinations(papers, 2):
        reject = _pair_rejects(a, b)
        if reject:
            continue
        anchors = tuple(sorted((_terms(a) & _terms(b)) - _BAD_SHARED, key=lambda word: (-len(word), word)))
        if not anchors:
            continue
        pairs.append(CandidatePair(a=a, b=b, anchors=anchors[:6]))
    pairs.sort(key=lambda pair: (len(pair.anchors), _source_diverse(pair)), reverse=True)
    return tuple(pairs[:limit])


def _pair_rejects(a: Paper, b: Paper) -> tuple[str, ...]:
    reasons: list[str] = []
    if a.key == b.key or _norm(a.title) == _norm(b.title):
        reasons.append("duplicate_receipt")
    if _is_review(a) and _is_review(b):
        reasons.append("review_review")
    if (_is_survey(a) and _is_case(b)) or (_is_case(a) and _is_survey(b)):
        reasons.append("survey_case")
    return tuple(reasons)


def _terms(paper: Paper) -> set[str]:
    return {word for word in _WORD_RE.findall(paper.text.casefold()) if word not in _STOP}


def _source_diverse(pair: CandidatePair) -> int:
    return int(pair.a.source.casefold() != pair.b.source.casefold())


def _is_review(paper: Paper) -> bool:
    text = paper.title.casefold()
    return "review" in text or "meta-analysis" in text or "systematic" in text


def _is_survey(paper: Paper) -> bool:
    return "survey" in paper.title.casefold()


def _is_case(paper: Paper) -> bool:
    return "case study" in paper.title.casefold() or paper.title.casefold().startswith("case ")


def _norm(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.casefold()))
