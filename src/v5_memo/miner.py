"""Mine receipt-bound '2 + 2 = 5' alpha hypotheses from corpus hits."""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from itertools import combinations

from v5_memo.schemas import ClaimCard, CorpusHit, EvidenceNode, InsightCandidate, ReceiptRole
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
    "men", "muscle", "power", "protein", "synthesi", "women",
    "case", "cases", "disease", "disorder", "disorders", "individual",
    "individuals", "patient", "people", "per", "person", "persons", "risk",
    "such", "syndrome", "thus", "when", "following", "matched",
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
_NULL_PHRASE_RE = re.compile(
    r"\b(?:does|do|did|is|are|was|were)\s+not\b|\bno\s+difference(?:s)?\b|\bwithout\s+difference(?:s)?\b"
)
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
_PROMISE = _INTENT | frozenset({
    "activate", "activated", "activates", "activating", "activation",
    "augment",
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
    "candidate", "commentary", "consensus", "guideline", "meta", "opinion", "opinions",
    "perspective", "position", "potential", "question", "recommendation",
    "recommendations", "review", "stand", "strategy", "systematic",
})
_WEAK_ELITE_SOURCE_TERMS = frozenset({
    "abstract", "conference", "editorial", "poster", "supplement",
})
_NON_PRIMARY_SOURCE_PHRASES = (
    "additional file",
    "faculty opinions recommendation",
    "supplementary file",
    "supplemental file",
    "supplementary material",
    "supplemental material",
    "supplementary data",
    "supplemental data",
    "data sheet",
    "dataset",
    "figshare",
    "dryad",
    "zenodo",
    "conference abstract",
    "meeting abstract",
    "poster abstract",
    "abstract supplement",
)
_SUPPLEMENT_DOI_RE = re.compile(r"(?:^|[-_.])s\d+(?:[-_.])p\d+(?:$|[-_.])")
_TOPIC_CONTEXT_STOP = frozenset({
    "adapt", "adaptation", "aging", "angle", "condition", "effect", "effects",
    "evidence", "healthspan", "human", "intervention", "longevity", "mechanism",
    "mechanisms", "outcome", "outcomes", "pharmacology", "response", "responses",
    "reversal", "study", "trial", "water",
})


def mine_insights(
    hits: Sequence[CorpusHit],
    *,
    topic: str,
    required_anchor_terms: Sequence[str] = (),
    include_discovery: bool = False,
    max_candidates: int = 30,
) -> list[InsightCandidate]:
    """Return ranked alpha candidates from source-diverse hit pairs."""
    clean_hits = _dedupe_hits(hits)
    if len(clean_hits) < 2:
        return []

    full_token_sets = {hit.hit_id: _tokens(hit.text) for hit in clean_hits}
    title_token_sets = {hit.hit_id: _tokens(hit.title) for hit in clean_hits}
    anchor_terms = frozenset(required_anchor_terms)
    topic_context_terms = _topic_context_terms(topic, anchor_terms)
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
        direct_human_pair = _direct_human_receipt(left) and _direct_human_receipt(right)
        if not bridge and direct_human_pair:
            bridge = _full_text_bridge_terms(
                full_token_sets[left.hit_id],
                full_token_sets[right.hit_id],
                doc_counts,
                pair_anchor_terms,
            )
        if not bridge:
            continue
        source_keys = {left.source_key, right.source_key}
        if len(source_keys) < 2:
            continue
        tension_terms = _hit_tension_terms(left, right)
        shape_reasons = _shape_reasons(
            left,
            right,
            bridge_terms=bridge,
            tension_terms=tension_terms,
        )
        if not shape_reasons:
            continue
        if not _pair_has_topic_context(left, right, topic_context_terms, shape_reasons):
            continue
        incomplete_elite_context = bool(
            set(shape_reasons) & _ELITE_SHAPES
            and topic_context_terms
            and not _pair_has_full_topic_context(left, right, topic_context_terms)
        )
        if anchor_terms and set(shape_reasons) & _ELITE_SHAPES and not anchor_bridge:
            continue
        coupling_reasons = _coupling_reasons(
            left,
            right,
            pair_anchor_terms=pair_anchor_terms,
        )
        elite_anchor_bridge = _has_elite_anchor_bridge(
            anchor_bridge,
            pair_anchor_terms,
            shape_reasons,
            tension_terms,
        )
        strong_anchor_bridge = bool(tension_terms and len(anchor_bridge) >= 2)
        direct_human_reversal = direct_human_pair and bool(tension_terms) and len(bridge) >= 2
        if not (strong_anchor_bridge or elite_anchor_bridge) and not _has_title_owned_bridge(
            left,
            right,
            bridge,
            doc_counts=doc_counts,
            total_docs=len(clean_hits),
        ) and not direct_human_reversal:
            continue
        if set(shape_reasons) == {"shape:directional_reversal"} and len(bridge) < 2:
            continue
        tier = _alpha_tier(shape_reasons)
        if tier == "elite_alpha" and (
            incomplete_elite_context or _is_anchor_only_bridge(bridge, pair_anchor_terms)
        ):
            tier = "publishable_alpha"
        if tier == "elite_alpha" and (
            _is_weak_elite_receipt(left) or _is_weak_elite_receipt(right)
        ):
            tier = "publishable_alpha"
        if tier == "discovery_seed" and not include_discovery:
            continue
        receipt_roles = _receipt_roles(left, right, shape_reasons)
        evidence_graph = _evidence_graph(
            clean_hits,
            left,
            right,
            bridge,
            pair_anchor_terms,
            _graph_context_terms(topic, pair_anchor_terms),
            shape_reasons,
            receipt_roles,
        )
        hits_by_id = {hit.hit_id: hit for hit in clean_hits}
        claim_cards = _claim_cards_for_roles(hits_by_id, receipt_roles)
        context_claim_cards = _claim_cards_for_graph(
            hits_by_id,
            tuple(node for node in evidence_graph if node.receipt_id not in {left.hit_id, right.hit_id}),
        )
        if context_claim_cards:
            claim_cards = (*claim_cards, *context_claim_cards)
        evidence_graph, claim_cards = _prioritize_evidence_bundle(
            topic,
            evidence_graph,
            claim_cards,
        )
        graph_receipt_ids = tuple(node.receipt_id for node in evidence_graph)
        score = score_connection(
            bridge_terms=bridge,
            bridge_doc_counts=doc_counts,
            unique_source_count=len(source_keys),
            receipt_count=len(graph_receipt_ids),
            has_tension=bool(tension_terms),
            shape_score=len(shape_reasons),
            shape_reasons=shape_reasons,
            support_quality=_support_quality(claim_cards),
        )
        candidates.append(InsightCandidate(
            topic=topic,
            thesis=_thesis(topic, bridge, left, right, tension_terms),
            bridge_terms=bridge,
            tension_terms=tension_terms,
            receipt_ids=graph_receipt_ids,
            score=score.score,
            novelty_score=score.novelty_score,
            evidence_score=score.evidence_score,
            scorecard=score.scorecard,
            reasons=(
                *score.reasons,
                *coupling_reasons,
                *_direction_cautions(left.text, right.text),
                f"tier:{tier}",
            ),
            receipt_roles=receipt_roles,
            claim_cards=claim_cards,
            evidence_graph=evidence_graph,
        ))
    return sorted(candidates, key=_candidate_rank, reverse=True)[
        :max(0, max_candidates)
    ]


def _prioritize_evidence_bundle(
    topic: str,
    evidence_graph: tuple[EvidenceNode, ...],
    claim_cards: tuple[ClaimCard, ...],
) -> tuple[tuple[EvidenceNode, ...], tuple[ClaimCard, ...]]:
    card_by_id = {card.receipt_id: card for card in claim_cards}
    node_by_id = {node.receipt_id: node for node in evidence_graph}
    order = {node.receipt_id: index for index, node in enumerate(evidence_graph)}
    if not any(
        order.get(card.receipt_id, 0) >= 2 and _strong_direct_human_rct(card)
        for card in claim_cards
    ):
        return evidence_graph, claim_cards
    sorted_ids = sorted(
        order,
        key=lambda receipt_id: _evidence_order_key(
            topic,
            node_by_id[receipt_id],
            card_by_id.get(receipt_id),
            order[receipt_id],
        ),
    )
    sorted_graph = tuple(node_by_id[receipt_id] for receipt_id in sorted_ids)
    sorted_cards = tuple(
        card_by_id[receipt_id]
        for receipt_id in sorted_ids
        if receipt_id in card_by_id
    )
    return sorted_graph, sorted_cards


def _strong_direct_human_rct(card: ClaimCard) -> bool:
    return (
        card.design == "randomized_trial"
        and card.population == "human"
        and card.support_type == "direct"
        and card.confidence == "high"
    )


def _evidence_order_key(
    topic: str,
    node: EvidenceNode,
    card: ClaimCard | None,
    original_index: int,
) -> tuple[int, int]:
    score = 0
    if card is not None:
        score += 30 if card.support_type == "direct" else 0
        score += 20 if card.population == "human" else 0
        score += 10 if card.confidence == "high" else 0
        score += {
            "randomized_trial": 20,
            "intervention_study": 14,
            "cohort": 10,
            "synthesis": 5,
            "mechanistic_model": 3,
        }.get(card.design, 0)
        score += min(10, 2 * len(_topic_overlap_terms(topic, card)))
    score += {"primary": 4, "counter": 4, "replication": 3, "boundary": 2}.get(node.role, 0)
    return (-score, original_index)


def _topic_overlap_terms(topic: str, card: ClaimCard) -> frozenset[str]:
    topic_terms = _expanded_context_terms(_tokens(topic))
    card_terms = _expanded_context_terms(
        _tokens(" ".join((card.outcome, card.quote, card.role)))
    )
    return topic_terms & card_terms


def _candidate_rank(candidate: InsightCandidate) -> tuple[bool, int, int, int, int]:
    return (
        "coupling:named_program" in candidate.reasons,
        candidate.score,
        candidate.novelty_score,
        candidate.evidence_score,
        len(candidate.bridge_terms),
    )


def _topic_context_terms(topic: str, anchor_terms: frozenset[str]) -> frozenset[str]:
    if len(anchor_terms) != 1:
        return frozenset()
    ordered = [
        token
        for raw in _WORD.findall(topic.casefold())
        if (
            token := _norm_token(raw)
        ) not in _STOP
        and token not in _TOPIC_CONTEXT_STOP
    ]
    context = [token for token in ordered if token not in anchor_terms]
    if not context:
        return frozenset()
    return frozenset(context)


def _graph_context_terms(topic: str, anchor_terms: frozenset[str]) -> frozenset[str]:
    if not anchor_terms:
        return frozenset()
    ordered = [
        token
        for raw in _WORD.findall(topic.casefold())
        if (
            token := _norm_token(raw)
        ) not in _STOP
        and token not in _TOPIC_CONTEXT_STOP
    ]
    context = [token for token in ordered if token not in anchor_terms]
    return frozenset(context) if context else _topic_context_terms(topic, anchor_terms)


def _pair_has_topic_context(
    left: CorpusHit,
    right: CorpusHit,
    context_terms: frozenset[str],
    shape_reasons: tuple[str, ...],
) -> bool:
    if not context_terms:
        return True
    left_has = _has_topic_context(left, context_terms)
    right_has = _has_topic_context(right, context_terms)
    if set(shape_reasons) & _ELITE_SHAPES:
        return left_has or right_has
    return left_has and right_has


def _has_topic_context(hit: CorpusHit, context_terms: frozenset[str]) -> bool:
    hit_terms = _expanded_context_terms(_tokens(hit.title) | _tokens(hit.text))
    required = 1 if len(context_terms) == 1 else 2
    return len(hit_terms & context_terms) >= required


def _pair_has_full_topic_context(
    left: CorpusHit,
    right: CorpusHit,
    context_terms: frozenset[str],
) -> bool:
    return _has_topic_context(left, context_terms) and _has_topic_context(right, context_terms)


def _expanded_context_terms(terms: frozenset[str]) -> frozenset[str]:
    expanded = set(terms)
    if "strength" in terms:
        expanded.add("resistance")
    if terms & {"aerobic", "running"}:
        expanded.update({"exercise", "training"})
    if "exercise" in terms:
        expanded.add("training")
    return frozenset(expanded)


def _coupling_reasons(
    left: CorpusHit,
    right: CorpusHit,
    *,
    pair_anchor_terms: frozenset[str],
) -> tuple[str, ...]:
    shared = _raw_title_terms(left.title) & _raw_title_terms(right.title)
    for raw in shared:
        token = _norm_token(raw.casefold())
        if token not in pair_anchor_terms and token not in _BRIDGE_STOP and raw.isupper():
            return ("coupling:named_program",)
    return ()


def _raw_title_terms(title: str) -> frozenset[str]:
    return frozenset(re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", title))


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
        "muscle",
        "protein",
        "resistance",
        "response",
        "reversal",
        "stress",
        "synthesi",
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


def _full_text_bridge_terms(
    left: frozenset[str],
    right: frozenset[str],
    doc_counts: Counter[str],
    anchor_terms: frozenset[str],
) -> tuple[str, ...]:
    shared = (left & right) - _BRIDGE_STOP - anchor_terms
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
    if tokens & _NULL or _NULL_PHRASE_RE.search(text.casefold()):
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


def _hit_tension_terms(left: CorpusHit, right: CorpusHit) -> tuple[str, ...]:
    a = _direction_polarity(left)
    b = _direction_polarity(right)
    if len(a) != 1 or len(b) != 1 or a == b:
        return ()
    return tuple(sorted(a | b))


def _direction_polarity(hit: CorpusHit) -> frozenset[str]:
    if _is_safety_feasibility_pilot(_raw_terms(hit.text)):
        return frozenset()
    title_polarity = _polarity(hit.title) - {"mixed"}
    full_polarity = _polarity(hit.text) - {"mixed"}
    if len(title_polarity) == 1 and len(full_polarity) > 1 and title_polarity & (_NEGATIVE | _NULL):
        return title_polarity
    if len(title_polarity) == 1 and (not full_polarity or full_polarity <= title_polarity):
        return title_polarity
    return full_polarity


def _direct_human_receipt(hit: CorpusHit) -> bool:
    card = _claim_card(hit, ReceiptRole(hit.hit_id, "evidence", "direct-human precheck"))
    return card.population == "human" and card.support_type == "direct" and card.confidence == "high"


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


def _is_anchor_only_bridge(
    bridge_terms: tuple[str, ...],
    pair_anchor_terms: frozenset[str],
) -> bool:
    return len(bridge_terms) == 1 and bridge_terms[0] in pair_anchor_terms


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


def _alpha_tier(shape_reasons: tuple[str, ...]) -> str:
    reasons = set(shape_reasons)
    if reasons & _ELITE_SHAPES:
        return "elite_alpha"
    if reasons & _PUBLISHABLE_SHAPES:
        return "publishable_alpha"
    return "discovery_seed"


def _is_weak_elite_receipt(hit: CorpusHit) -> bool:
    text = " ".join(
        part
        for part in (hit.title, hit.venue or "", hit.doi or "", hit.url)
        if part
    )
    return bool(_tokens(text) & _WEAK_ELITE_SOURCE_TERMS) or _is_non_primary_receipt(hit)


def _is_non_primary_receipt(hit: CorpusHit) -> bool:
    descriptor = " ".join(
        part.casefold()
        for part in (
            hit.title,
            hit.abstract,
            hit.venue or "",
            hit.source,
            hit.doi or "",
            hit.hit_id,
            hit.url,
            " ".join(str(value) for value in hit.metadata.values()),
        )
        if part
    )
    if any(phrase in descriptor for phrase in _NON_PRIMARY_SOURCE_PHRASES):
        return True
    doi = str(hit.doi or hit.hit_id or "").casefold()
    return (
        ("10.1096/fasebj" in doi and ".s1." in doi)
        or doi.startswith("10.3410/f.")
        or doi.startswith("10.1249/01.mss.")
        or bool(_SUPPLEMENT_DOI_RE.search(doi))
    )


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


def _claim_cards(
    left: CorpusHit,
    right: CorpusHit,
    roles: tuple[ReceiptRole, ...],
) -> tuple[ClaimCard, ...]:
    by_id = {left.hit_id: left, right.hit_id: right}
    return _claim_cards_for_roles(by_id, roles)


def _claim_cards_for_roles(
    by_id: dict[str, CorpusHit],
    roles: tuple[ReceiptRole, ...],
) -> tuple[ClaimCard, ...]:
    return tuple(
        _claim_card(by_id[role.receipt_id], role)
        for role in roles
        if role.receipt_id in by_id
    )


def _claim_cards_for_graph(
    by_id: dict[str, CorpusHit],
    graph: tuple[EvidenceNode, ...],
) -> tuple[ClaimCard, ...]:
    return tuple(
        _claim_card(by_id[node.receipt_id], ReceiptRole(node.receipt_id, node.role, node.reason))
        for node in graph
        if node.receipt_id in by_id
    )


def _evidence_graph(
    hits: Sequence[CorpusHit],
    left: CorpusHit,
    right: CorpusHit,
    bridge_terms: tuple[str, ...],
    pair_anchor_terms: frozenset[str],
    topic_context_terms: frozenset[str],
    shape_reasons: tuple[str, ...],
    receipt_roles: tuple[ReceiptRole, ...],
) -> tuple[EvidenceNode, ...]:
    nodes = [
        EvidenceNode(role.receipt_id, _graph_role(role.role), role.reason)
        for role in receipt_roles
    ]
    seen = {node.receipt_id for node in nodes}
    bridge_set = set(bridge_terms) - pair_anchor_terms
    if not bridge_set:
        return tuple(nodes)
    for role in ("mechanism", "boundary", "replication", "consensus"):
        hit = _context_hit(
            hits,
            seen=seen,
            bridge_terms=bridge_set,
            topic_context_terms=topic_context_terms,
            shape_reasons=shape_reasons,
            role=role,
        )
        if hit is None:
            continue
        seen.add(hit.hit_id)
        nodes.append(EvidenceNode(hit.hit_id, role, f"{role} context for evidence graph"))
    return tuple(nodes)


def _graph_role(role: str) -> str:
    if role in {"promise", "positive_signal", "aggregate_signal"}:
        return "primary"
    if role in {"outcome", "negative_signal", "null_signal", "tail_risk"}:
        return "counter"
    return "primary"


def _context_hit(
    hits: Sequence[CorpusHit],
    *,
    seen: set[str],
    bridge_terms: set[str],
    topic_context_terms: frozenset[str],
    shape_reasons: tuple[str, ...],
    role: str,
) -> CorpusHit | None:
    candidates: list[tuple[int, CorpusHit]] = []
    for hit in hits:
        if hit.hit_id in seen:
            continue
        terms = _raw_terms(hit.text)
        overlap = bridge_terms & terms
        if len(overlap) < min(2, len(bridge_terms)):
            continue
        if topic_context_terms and not _has_topic_context(hit, topic_context_terms):
            continue
        if not _hit_matches_graph_role(hit, terms, role, shape_reasons):
            continue
        candidates.append((len(overlap), hit))
    return max(candidates, key=lambda item: (item[0], item[1].year or 0))[1] if candidates else None


def _hit_matches_graph_role(
    hit: CorpusHit,
    terms: frozenset[str],
    role: str,
    shape_reasons: tuple[str, ...],
) -> bool:
    if role == "mechanism":
        return _design_type(terms) == "mechanistic_model" or bool(terms & {"mechanism", "pathway", "signaling"})
    if role == "boundary":
        return bool(terms & (_BOUNDARY | _TIMING | _TAIL))
    if role == "replication":
        return bool(_direction_polarity(hit))
    if role == "consensus":
        return _design_type(terms) == "synthesis" or _is_synthesis_hit(hit)
    return False


def _claim_card(hit: CorpusHit, role: ReceiptRole) -> ClaimCard:
    terms = _raw_terms(hit.text)
    design = _design_type(terms)
    population = _population_type(terms)
    safety_feasibility = _is_safety_feasibility_pilot(terms)
    direction = "unclear" if safety_feasibility else "/".join(sorted(_direction_polarity(hit))) or "unclear"
    direct_designs = {"randomized_trial", "cohort", "intervention_study"}
    non_primary = _is_non_primary_receipt(hit)
    support_type = "direct" if design in direct_designs and population == "human" and not non_primary else "indirect"
    if safety_feasibility:
        support_type = "direct" if population == "human" and not non_primary else "indirect"
    confidence = "high" if support_type == "direct" and direction != "unclear" else "medium" if direction != "unclear" else "low"
    role_name = "safety_feasibility" if safety_feasibility and role.role.endswith("_signal") else role.role
    return ClaimCard(
        receipt_id=hit.hit_id,
        role=role_name,
        design=design,
        population=population,
        outcome=_outcome_label(terms),
        direction=direction,
        support_type=support_type,
        confidence=confidence,
        quote=_claim_quote(hit),
    )


def _design_type(terms: frozenset[str]) -> str:
    if terms & {"review", "meta", "systematic"}:
        return "synthesis"
    if terms & {"randomized", "randomised", "rct", "trial"}:
        return "randomized_trial"
    if terms & {"cohort", "prospective", "longitudinal"}:
        return "cohort"
    if terms & {"intervention", "protocol", "session", "sessions", "training"} and terms & {
        "athlete", "athletes", "participant", "participants", "player", "players", "student", "students",
        "subject", "subjects", "volunteer", "volunteered", "volunteers",
    }:
        return "intervention_study"
    if terms & {"administered", "administ", "supplement", "supplemented"} and terms & {
        "adult", "adults", "human", "humans", "men", "patient", "patients", "women",
    }:
        return "intervention_study"
    if terms & {"mouse", "mice", "rat", "rats", "cell", "cells"}:
        return "mechanistic_model"
    return "unspecified"


def _is_safety_feasibility_pilot(terms: frozenset[str]) -> bool:
    return bool(
        terms & {"pilot", "feasibility", "feasible", "safety"}
        and terms & {"trial", "randomized", "randomised", "rct", "study"}
    )


def _population_type(terms: frozenset[str]) -> str:
    if terms & {
        "athlete", "athletes", "human", "humans", "participant", "participants",
        "patient", "patients", "adult", "adults", "men", "women", "player", "players", "student",
        "students", "subject", "subjects", "volunteer", "volunteered", "volunteers",
    }:
        return "human"
    if terms & {"mouse", "mice", "rat", "rats", "animal", "animals"}:
        return "animal"
    if terms & {"cell", "cells", "cellular"}:
        return "cell_model"
    return "unspecified"


def _outcome_label(terms: frozenset[str]) -> str:
    candidates = sorted(terms & (_OUTCOME | _METRIC | _ADVERSE_ENDPOINT | _BOUNDARY | _TIMING))
    return "/".join(candidates[:3]) if candidates else "unspecified"


def _claim_quote(hit: CorpusHit) -> str:
    text = " ".join((hit.abstract or hit.title).split())
    return text[:180].rstrip()


def _support_quality(cards: tuple[ClaimCard, ...]) -> int:
    quality = 0
    for card in cards:
        quality += {
            "randomized_trial": 12,
            "cohort": 9,
            "synthesis": 5,
            "mechanistic_model": 4,
        }.get(card.design, 2)
        quality += {"human": 8, "animal": 3, "cell_model": 2}.get(card.population, 1)
        if card.direction != "unclear":
            quality += 5
        if card.role in {"promise", "outcome", "tail_risk", "aggregate_signal"}:
            quality += 3
    return min(35, quality)


def _raw_terms(text: str) -> frozenset[str]:
    return frozenset(_norm_token(raw) for raw in _WORD.findall(text.casefold()))


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
