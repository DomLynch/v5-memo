"""Mine receipt-bound '2 + 2 = 5' alpha hypotheses from corpus hits."""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from itertools import combinations

from v5_memo.schemas import CorpusHit, InsightCandidate, ReceiptRole
from v5_memo.scorer import score_connection

_WORD = re.compile(r"[a-z][a-z0-9]{2,}")
_STOP = frozenset({
    "about", "advances", "after", "agent", "among", "analysis", "and", "based", "beneficial",
    "also", "between", "can", "cell", "clinical", "comprehensive", "data", "effect",
    "effects", "evidence", "finding", "findings", "from", "group", "groups",
    "human", "impact", "isi", "library", "links", "marker", "markers", "meta", "model",
    "models", "paper", "patients", "predicts", "recent", "reported", "research",
    "response", "results", "review", "showed", "shows", "significant", "study", "studies",
    "summary", "systematic", "the", "their", "there", "through", "trial", "using", "with",
})
_BRIDGE_STOP = _STOP | frozenset({
    "action", "compare", "compared", "comparing", "comparative", "comparison",
    "functional", "horse", "impairment", "intermittent", "learning",
    "men", "power", "women",
    "case", "cases", "individual", "individuals", "patient", "people", "per", "person",
    "persons", "such", "thus", "when", "following", "matched",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty",
})
_POSITIVE = frozenset({
    "augment", "augmented", "enhance", "enhanced", "increase", "increased",
    "improve", "improved", "improvement", "raises", "raised",
})
_NEGATIVE = frozenset({
    "attenuate", "attenuated", "blunt", "blunted", "decrease", "decreased",
    "impair", "impaired", "lower", "lowered", "reduce", "reduced",
    "suppress", "suppressed", "suppresses", "worse", "worsened",
})
_ATTENUATE = frozenset({"attenuate", "attenuated"})
_ADVERSE_ENDPOINT = frozenset({
    "acth", "cortisol", "damage", "death", "deaths", "error", "errors",
    "fatal", "fatality", "inflammation", "mortality", "pain", "risk", "stress",
})
_NULL = frozenset({"null", "neutral", "unchanged", "failed", "nonsignificant"})
_NEGATED = frozenset({
    "fail", "failed", "fails", "lack", "lacked", "lacks", "no", "not", "without",
})
_DENOMINATOR = frozenset({"cohort", "population", "aggregate", "prospective", "longitudinal"})
_TAIL = frozenset({"case", "cases", "fatal", "fatality", "death", "deaths", "risk", "rare"})
_TIMING = frozenset({"acute", "chronic", "short", "long", "early", "late", "immediate", "delayed"})
_ROLE_A = frozenset({"cause", "causes", "driver", "drives", "predict", "predicts", "associated"})
_ROLE_B = frozenset({"confound", "confounds", "selection", "mediates", "moderates", "substitution"})
_METRIC = frozenset({"metric", "score", "benchmark", "accuracy", "performance", "clicks"})
_OUTCOME = frozenset({"outcome", "mortality", "injury", "error", "errors", "dispersion", "quality"})
_EXPERTISE = frozenset({"expert", "experts", "novice", "novices", "nonexpert", "nonexperts"})
_BOUNDARY = frozenset({"boundary", "context", "dose", "endpoint", "modality", "population", "setting"})
_INTENT = frozenset({"aim", "aimed", "designed", "expect", "expected", "hypothesis", "intended", "protocol", "theory"})
_OBSERVED = frozenset({"found", "observed", "outcome", "outcomes", "reported", "result", "results", "showed"})
_PROMISE = _INTENT | _ROLE_A | frozenset({
    "activate", "activated", "activates", "activating", "activation",
    "mimetic", "mimetics", "mimic", "mimics",
})
_OUTCOME_ROLE = _OBSERVED | frozenset({
    "cohort", "endpoint", "endpoints", "experiment", "intervention", "randomized",
    "trial", "trials",
}) | _NEGATIVE | _NULL
_PUBLISHABLE_SHAPES = frozenset({
    "shape:promise_outcome_reversal",
    "shape:expectation_reversal",
    "shape:directional_reversal",
    "shape:boundary_condition",
    "shape:denominator_split",
    "shape:role_inversion",
    "shape:timing_split",
})
_ELITE_SHAPES = frozenset({
    "shape:promise_outcome_reversal",
    "shape:expectation_reversal",
})
_SYNTHESIS_TITLE_TERMS = frozenset({
    "candidate", "commentary", "consensus", "guideline", "meta", "perspective",
    "position", "potential", "question", "review", "stand", "strategy", "systematic",
})


def mine_insights(
    hits: Sequence[CorpusHit],
    *,
    topic: str,
    required_anchor_terms: Sequence[str] = (),
    include_discovery: bool = False,
    max_candidates: int = 5,
) -> list[InsightCandidate]:
    """Return ranked alpha candidates from source-diverse hit pairs."""
    clean_hits = _dedupe_hits(hits)
    if len(clean_hits) < 2:
        return []

    full_token_sets = {hit.hit_id: _tokens(hit.text) for hit in clean_hits}
    title_token_sets = {hit.hit_id: _tokens(hit.title) for hit in clean_hits}
    anchor_terms = frozenset(required_anchor_terms)
    doc_counts = Counter(term for terms in full_token_sets.values() for term in terms)

    candidates: list[InsightCandidate] = []
    for left, right in combinations(clean_hits, 2):
        if not anchor_terms and not _shares_seed_query(left, right):
            continue
        if anchor_terms and not _pair_has_anchor(
            full_token_sets[left.hit_id],
            full_token_sets[right.hit_id],
            anchor_terms,
        ):
            continue
        pair_anchor_terms = anchor_terms or _shared_seed_anchor_terms(left, right)
        title_shared = title_token_sets[left.hit_id] & title_token_sets[right.hit_id]
        anchor_bridge = _anchor_bridge_terms(
            full_token_sets[left.hit_id],
            full_token_sets[right.hit_id],
            pair_anchor_terms,
            title_shared,
        )
        title_bridge = _title_bridge_terms(title_shared, doc_counts)
        bridge = (*anchor_bridge, *(term for term in title_bridge if term not in set(anchor_bridge)))[:4]
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
        elite_anchor_bridge = _has_elite_anchor_bridge(
            anchor_bridge,
            pair_anchor_terms,
            shape_reasons,
            tension_terms,
        )
        strong_anchor_bridge = bool(tension_terms and len(anchor_bridge) >= 2)
        if not (strong_anchor_bridge or elite_anchor_bridge) and not _has_title_owned_bridge(
            left,
            right,
            bridge,
            doc_counts=doc_counts,
            total_docs=len(clean_hits),
        ):
            continue
        if set(shape_reasons) == {"shape:directional_reversal"} and len(bridge) < 2:
            continue
        tier = _alpha_tier(shape_reasons, tension_terms)
        if tier == "elite_alpha" and len(bridge) == 1 and len(bridge[0]) < 8:
            continue
        if tier == "discovery_seed" and not include_discovery:
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
            reasons=(*score.reasons, *_direction_cautions(left.text, right.text), f"tier:{tier}"),
            receipt_roles=_receipt_roles(left, right, shape_reasons),
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


def _norm_token(token: str) -> str:
    token = token.casefold()
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 7 and token.endswith("ation"):
        return token[:-5]
    if len(token) > 6 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> frozenset[str]:
    return frozenset(
        token
        for raw in _WORD.findall(text.casefold())
        if (token := _norm_token(raw)) not in _STOP
    )


def query_anchor_terms(seed_queries: Sequence[str], *, limit: int = 3) -> tuple[str, ...]:
    """Return ordered anchor terms that chosen receipt pairs must preserve."""
    generic = {
        "adapt",
        "adaptation",
        "aging",
        "angle",
        "condition",
        "effect",
        "effects",
        "exercise",
        "healthspan",
        "intervention",
        "longevity",
        "mechanism",
        "mimetic",
        "pharmacology",
        "protocol",
        "resistance",
        "response",
        "reversal",
        "stress",
        "train",
        "trained",
        "training",
        "water",
    }
    generic.update(
        _POSITIVE | _NEGATIVE | _OBSERVED | _INTENT | _PROMISE | _OUTCOME_ROLE
        | _BOUNDARY | _TIMING | _METRIC | _OUTCOME | _ROLE_A | _ROLE_B
    )
    out: list[str] = []
    seen: set[str] = set()
    for query in seed_queries:
        for raw in _WORD.findall(query.casefold()):
            token = _norm_token(raw)
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
    return bool(left_tokens & right_tokens & anchor_terms)


def _title_bridge_terms(
    title_shared: frozenset[str],
    doc_counts: Counter[str],
) -> tuple[str, ...]:
    shared = title_shared - _BRIDGE_STOP
    ranked = sorted(shared, key=lambda term: (doc_counts[term], term))
    return tuple(ranked[:4])


def _anchor_bridge_terms(
    left: frozenset[str],
    right: frozenset[str],
    anchor_terms: frozenset[str],
    title_shared: frozenset[str],
) -> tuple[str, ...]:
    shared = (left & right & anchor_terms & title_shared) - _BRIDGE_STOP
    return tuple(sorted(shared)[:2])


def _polarity(text: str) -> frozenset[str]:
    tokens = _tokens(text)
    out: set[str] = set()
    unnegated_positive, negated_positive = _positive_context(text)
    if unnegated_positive:
        out.add("positive")
    if negated_positive:
        out.add("null")
    if tokens & _NEGATIVE:
        if tokens & _ATTENUATE and tokens & _ADVERSE_ENDPOINT and not (tokens & (_NEGATIVE - _ATTENUATE)):
            out.add("positive")
        else:
            out.add("negative")
    if tokens & _NULL:
        out.add("null")
    if len(out) > 1:
        out.add("mixed")
    return frozenset(out)


def _positive_context(text: str) -> tuple[bool, bool]:
    words = [_norm_token(raw) for raw in _WORD.findall(text.casefold())]
    unnegated = False
    negated = False
    for index, word in enumerate(words):
        if word not in _POSITIVE:
            continue
        prefix = set(words[max(0, index - 2):index])
        if prefix & _NEGATED:
            negated = True
        elif prefix & _NEGATIVE:
            continue
        else:
            unnegated = True
    return unnegated, negated


def _tension_terms(left: str, right: str) -> tuple[str, ...]:
    a = _polarity(left) - {"mixed"}
    b = _polarity(right) - {"mixed"}
    if len(a) != 1 or len(b) != 1 or a == b:
        return ()
    return tuple(sorted(a | b))


def _direction_cautions(left: str, right: str) -> tuple[str, ...]:
    if "mixed" in _polarity(left) or "mixed" in _polarity(right):
        return ("caution:mixed_direction",)
    return ()


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
    left_words = _words(left.text)
    right_words = _words(right.text)
    reasons: list[str] = []
    if tension_terms and _has_role_split(left_words, right_words):
        reasons.append("shape:promise_outcome_reversal")
    if tension_terms and _has_cross_receipt_split(left_words, right_words, _INTENT, _OBSERVED):
        reasons.append("shape:expectation_reversal")
    if tension_terms:
        reasons.append("shape:directional_reversal")
    if (
        len(bridge_terms) >= 2
        and _axis(left, bridge_terms) != _axis(right, bridge_terms)
        and (tension_terms or all_tokens & _BOUNDARY)
    ):
        reasons.append("shape:boundary_condition")
    if _has_cross_receipt_split(left_tokens, right_tokens, _DENOMINATOR, _TAIL):
        reasons.append("shape:denominator_split")
    left_timing = left_tokens & _TIMING
    right_timing = right_tokens & _TIMING
    if left_timing and right_timing and left_timing != right_timing:
        reasons.append("shape:timing_split")
    if _has_cross_receipt_split(left_tokens, right_tokens, _ROLE_A, _ROLE_B):
        reasons.append("shape:role_inversion")
    if _has_cross_receipt_split(left_tokens, right_tokens, _METRIC, _OUTCOME):
        reasons.append("shape:measurement_mismatch")
    if all_tokens & _EXPERTISE:
        reasons.append("shape:expertise_split")
    if _is_synthesis_hit(left) or _is_synthesis_hit(right):
        reasons = [reason for reason in reasons if reason not in _ELITE_SHAPES]
    return tuple(dict.fromkeys(reasons))


def _axis(hit: CorpusHit, excluded: tuple[str, ...]) -> str:
    excluded_set = set(excluded)
    title_terms = [
        token
        for raw in _WORD.findall(hit.title.casefold())
        if (token := _norm_token(raw)) not in _STOP
    ]
    axis = [t for t in title_terms if t not in excluded_set][:4]
    return " ".join(axis) or (hit.venue or hit.source)


def _has_title_owned_bridge(
    left: CorpusHit,
    right: CorpusHit,
    bridge_terms: tuple[str, ...],
    *,
    doc_counts: Counter[str],
    total_docs: int,
) -> bool:
    shared_title_terms = _tokens(left.title) & _tokens(right.title)
    max_common_docs = max(4, total_docs // 10)
    return any(
        term in shared_title_terms and doc_counts[term] <= max_common_docs
        for term in bridge_terms
    )


def _has_elite_anchor_bridge(
    anchor_bridge: tuple[str, ...],
    pair_anchor_terms: frozenset[str],
    shape_reasons: tuple[str, ...],
    tension_terms: tuple[str, ...],
) -> bool:
    return (
        bool(tension_terms)
        and bool(set(shape_reasons) & _ELITE_SHAPES)
        and len(anchor_bridge) == 1
        and anchor_bridge[0] in pair_anchor_terms
        and len(anchor_bridge[0]) >= 8
    )


def _shares_seed_query(left: CorpusHit, right: CorpusHit) -> bool:
    left_queries = _seed_queries(left)
    right_queries = _seed_queries(right)
    return not left_queries or not right_queries or bool(set(left_queries) & set(right_queries))


def _shared_seed_anchor_terms(left: CorpusHit, right: CorpusHit) -> frozenset[str]:
    shared_queries = set(_seed_queries(left)) & set(_seed_queries(right))
    anchors: set[str] = set()
    for query in shared_queries:
        anchors.update(query_anchor_terms([query]))
    return frozenset(anchors)


def _seed_queries(hit: CorpusHit) -> tuple[str, ...]:
    raw = hit.metadata.get("seed_queries", ())
    return raw if isinstance(raw, tuple) else ()


def _words(text: str) -> frozenset[str]:
    return _tokens(text)


def _is_synthesis_hit(hit: CorpusHit) -> bool:
    return (
        "?" in hit.title
        or bool({_norm_token(raw) for raw in _WORD.findall(hit.title.casefold())} & _SYNTHESIS_TITLE_TERMS)
    )


def _has_role_split(left_words: frozenset[str], right_words: frozenset[str]) -> bool:
    return (
        bool(left_words & _PROMISE and right_words & _OUTCOME_ROLE)
        or bool(right_words & _PROMISE and left_words & _OUTCOME_ROLE)
    )


def _has_cross_receipt_split(
    left: frozenset[str],
    right: frozenset[str],
    a: frozenset[str],
    b: frozenset[str],
) -> bool:
    return bool(left & a and right & b) or bool(left & b and right & a)


def _alpha_tier(shape_reasons: tuple[str, ...], tension_terms: tuple[str, ...]) -> str:
    reasons = set(shape_reasons)
    if reasons & _ELITE_SHAPES:
        return "elite_alpha"
    if "shape:directional_reversal" in reasons and set(tension_terms) == {"negative", "null"}:
        return "elite_alpha"
    if reasons & _PUBLISHABLE_SHAPES:
        return "publishable_alpha"
    return "discovery_seed"


def _receipt_roles(
    left: CorpusHit,
    right: CorpusHit,
    shape_reasons: tuple[str, ...],
) -> tuple[ReceiptRole, ...]:
    left_words = _words(left.text)
    right_words = _words(right.text)
    if (
        "shape:promise_outcome_reversal" in shape_reasons
        or "shape:expectation_reversal" in shape_reasons
    ):
        left_role = _promise_outcome_role(left_words)
        right_role = _promise_outcome_role(right_words)
        if left_role != right_role:
            return (
                ReceiptRole(left.hit_id, left_role, "promise/outcome split"),
                ReceiptRole(right.hit_id, right_role, "promise/outcome split"),
            )
    if "shape:denominator_split" in shape_reasons:
        return (
            ReceiptRole(left.hit_id, _denominator_role(left_words), "benefit/risk denominator split"),
            ReceiptRole(right.hit_id, _denominator_role(right_words), "benefit/risk denominator split"),
        )
    return (
        ReceiptRole(left.hit_id, _signal_role(left.text), "candidate evidence stream"),
        ReceiptRole(right.hit_id, _signal_role(right.text), "candidate evidence stream"),
    )


def _promise_outcome_role(words: frozenset[str]) -> str:
    if words & _PROMISE and not words & _OUTCOME_ROLE:
        return "promise"
    if words & _OUTCOME_ROLE and not words & _PROMISE:
        return "outcome"
    if words & _PROMISE:
        return "promise"
    return "outcome"


def _denominator_role(words: frozenset[str]) -> str:
    if words & _TAIL:
        return "tail_risk"
    if words & _DENOMINATOR:
        return "aggregate_signal"
    return "evidence"


def _signal_role(text: str) -> str:
    polarity = _polarity(text)
    if len(polarity) == 1:
        return f"{next(iter(polarity))}_signal"
    return "evidence"


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
