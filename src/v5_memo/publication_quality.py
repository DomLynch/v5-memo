"""Deterministic, receipt-bound quality assessment for public V5 memos."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence

from v5_memo.evidence import (
    has_verified_primary_article_type,
    source_artifact_type,
    source_integrity_issue,
    stable_source_identity,
)
from v5_memo.gate import candidate_publish_blocker
from v5_memo.schemas import ClaimCard, CorpusHit, MemoResult

_P_VALUE_LABEL_PATTERN = r"p(?:[-\u2010-\u2015\u2212]value|\s+value)?"
_P_VALUE_OPERATOR_PATTERN = r"(?:<=|\u2264|=|<)"
_THRESHOLD_LABEL_PATTERN = r"(?:alpha(?:\s+level)?|(?:significance\s+)?threshold)"
_THRESHOLD_CONNECTOR_PATTERN = (
    r"(?:=|of|at|was(?:\s+set)?(?:\s+(?:to|at))?|set\s+(?:to|at))"
)
_STAT_NUMBER_START_PATTERN = r"[-+\u2212]?(?:\d|\.\d)"
_STAT_NUMBER_TOKEN_PATTERN = (
    r"[-+\u2212]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+\u2212]?\d+)?"
)
_STAT_ABBREVIATION_EXPLICIT_SUFFIX_PATTERN = (
    rf"\s*(?:=|:)\s*{_STAT_NUMBER_TOKEN_PATTERN}\b"
    r"(?!\s*(?:bpm|beats?\s+per\s+minute|breaths?\s+per\s+minute|"
    r"(?:beats?|breaths?)\s*/\s*min(?:ute)?s?|/\s*min(?:ute)?s?)\b)"
)
_STAT_ABBREVIATION_SUFFIX_PATTERN = (
    rf"\s*(?:(?:=|:)\s*|(?:(?:was|is|of)\s+))?{_STAT_NUMBER_START_PATTERN}"
)
_STAT_ABBREVIATION_VALUE_PATTERN = (
    rf"\b(?:HR|OR|RR)\b{_STAT_ABBREVIATION_EXPLICIT_SUFFIX_PATTERN}|"
    rf"\b(?:CI|SMD)\b{_STAT_ABBREVIATION_SUFFIX_PATTERN}"
)
_STAT_CONSTRUCT_LABEL_PATTERN = (
    r"(?:(?i:confidence interval|effect size|hazard ratio|odds ratio|relative risk|"
    r"standardized mean difference|cohen'?s d|hedges'? g)|(?:CI|HR|OR|RR|SMD))"
)
_STAT_ANCHOR_RE = re.compile(
    rf"(?i:\b{_P_VALUE_LABEL_PATTERN}\s*{_P_VALUE_OPERATOR_PATTERN}\s*\.?\d+|"
    r"\b95%\s*(?:confidence interval|ci)\b|"
    r"\b(?:confidence interval|effect size|hazard ratio|odds ratio|relative risk|"
    r"standardized mean difference|cohen'?s d|hedges'? g)\b)|"
    rf"{_STAT_ABBREVIATION_VALUE_PATTERN}"
)
_STAT_NUMBER_RE = re.compile(rf"{_STAT_NUMBER_TOKEN_PATTERN}%?")
_NON_SIGNIFICANT_RE = re.compile(
    r"(?i)\b(?:non[- ]?significant|not(?:\s+statistically)?\s+significant|"
    r"(?:statistically\s+)?insignificant|"
    r"(?:did|does)\s+not\s+differ\s+significantly|"
    r"(?:did\s+not|failed\s+to)\s+(?:reach|achieve)(?:\s+statistical)?\s+significance|"
    r"no(?:\s+statistically)?\s+significant"
    r"(?:\s+(?:difference|effect|association|change))?)\b"
)
_P_VALUE_RE = re.compile(
    rf"(?i)\b{_P_VALUE_LABEL_PATTERN}\s*(?P<operator><=|\u2264|=|<)\s*"
    r"(?P<value>(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+\u2212]?\d+)?)"
)
_EXPLICIT_THRESHOLD_RE = re.compile(
    rf"(?i)\b(?P<label>{_THRESHOLD_LABEL_PATTERN})\s*"
    rf"{_THRESHOLD_CONNECTOR_PATTERN}\s*"
    rf"(?P<value>{_STAT_NUMBER_TOKEN_PATTERN})"
)
_THRESHOLD_EXPLANATION_RE = re.compile(
    rf"(?i)\b{_THRESHOLD_LABEL_PATTERN}\s*"
    rf"{_THRESHOLD_CONNECTOR_PATTERN}\s*"
    rf"{_STAT_NUMBER_TOKEN_PATTERN}|"
    r"\bbonferroni(?:[- ]corrected|\s+(?:correction|adjustment))?\b|"
    r"\b(?:false discovery rate|fdr)\b|"
    r"\bmultiple[- ]comparisons?\s+(?:correction|adjustment)\b"
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_INTERPRETATION_CLAUSE_SPLIT_RE = re.compile(
    r"\s*;\s*|\s*,\s*(?=(?:while|whereas|but|although|and)\b)|"
    rf"\s+(?=(?:while|whereas|although|but|yet)\b"
    rf"(?=[^;.!?]*(?:\b{_P_VALUE_LABEL_PATTERN}\s*{_P_VALUE_OPERATOR_PATTERN}|"
    r"\bbaseline\b|\bcontrol\b|\bcomparator\b)))",
    re.IGNORECASE,
)
_NEGATED_THRESHOLD_EXPLANATION_RE = re.compile(
    r"(?i)\b(?:no|without(?:\s+an?))\s+(?:adjusted\s+|prespecified\s+)?threshold\b|"
    r"\bno\s+(?:bonferroni|fdr|false discovery rate)\b"
    r"(?:\s+(?:threshold|correction|adjustment))?(?:\s+was\s+(?:stated|reported|applied))?|"
    r"\bno\s+multiple[- ]comparisons?\s+(?:correction|adjustment)\b|"
    r"\b(?:bonferroni|fdr|false discovery rate)\b(?:\s+\w+){0,2}\s+"
    r"(?:could|can)\s+not\s+be\s+"
    r"(?:calculated|estimated|determined|applied|reported|provided)\b|"
    r"\b(?:threshold|correction|adjustment|bonferroni|fdr|false discovery rate)\b"
    r"(?:\s+\w+){0,3}\s+(?:was\s+)?"
    r"(?:not\s+(?:stated|reported|applied|calculated|available|provided)|"
    r"unstated|unreported|unavailable)\b"
)
_HIGH_RISK_LEVELS = frozenset({"critical", "high", "serious"})
_THRESHOLD_ONLY_TERMS = frozenset({
    "a",
    "adjusted",
    "adjustment",
    "after",
    "alpha",
    "an",
    "at",
    "bonferroni",
    "corrected",
    "correction",
    "discovery",
    "error",
    "false",
    "familywise",
    "fdr",
    "is",
    "level",
    "multiple",
    "nominal",
    "of",
    "prespecified",
    "rate",
    "significance",
    "set",
    "the",
    "threshold",
    "to",
    "unadjusted",
    "was",
})
_DEFAULT_SIGNIFICANCE_THRESHOLD = 0.05
_NON_RESULT_CONTEXT_RE = re.compile(
    rf"\b{_STAT_CONSTRUCT_LABEL_PATTERN}\b[^.!?]{{0,40}}\b"
    r"(?:was|were|is|are)?\s*"
    r"(?:not\s+(?:reported|available|provided|estimated|calculated)|"
    r"unreported|unavailable|absent|missing)\b|"
    rf"\b{_STAT_CONSTRUCT_LABEL_PATTERN}\b"
    r"\s+(?:estimates?\s+)?(?:was|were|is|are|to be|will be)?\s*"
    r"(?i:planned|prespecified)\b|"
    rf"(?i:\bno\s+(?:reported|available|provided|estimated|calculated)\s+)"
    rf"{_STAT_CONSTRUCT_LABEL_PATTERN}\b|"
    rf"(?i:\bno\s+){_STAT_CONSTRUCT_LABEL_PATTERN}\b[^.!?]{{0,24}}\b"
    r"(?i:reported|available|provided|estimated|calculated)\b|"
    r"(?i:\b(?:did|does|do)\s+not\s+"
    r"(?:report|provide|estimate|calculate)\s+(?:an?\s+|the\s+)?)"
    rf"{_STAT_CONSTRUCT_LABEL_PATTERN}\b|(?i:\bpower calculation\b)"
)
_MEASUREMENT_CONTEXT_PATTERNS = (
    ("baseline", re.compile(r"(?i)\b(?:baseline|pre[- ]?intervention|pre[- ]?treatment)\b")),
    ("follow_up", re.compile(r"(?i)\b(?:follow[- ]?up|post[- ]?intervention|post[- ]?treatment)\b")),
)
_FORMAL_RISK_DOMAINS = {
    "rob_2": frozenset({
        "randomization",
        "deviations_from_intervention",
        "missing_outcome_data",
        "outcome_measurement",
        "selective_reporting",
    }),
    "robins_i": frozenset({
        "confounding",
        "participant_selection",
        "intervention_classification",
        "deviations_from_intervention",
        "missing_outcome_data",
        "outcome_measurement",
        "selective_reporting",
    }),
}
_STAT_CONSTRUCT_PATTERNS = (
    (
        "p_value",
        re.compile(
            rf"(?i)\b{_P_VALUE_LABEL_PATTERN}\s*{_P_VALUE_OPERATOR_PATTERN}\s*(?:\d|\.)"
        ),
    ),
    (
        "confidence_interval",
        re.compile(
            rf"(?i:\bconfidence interval\b)|\bCI\b{_STAT_ABBREVIATION_SUFFIX_PATTERN}"
        ),
    ),
    ("effect_size", re.compile(r"(?i)\beffect size\b")),
    (
        "hazard_ratio",
        re.compile(
            rf"(?i:\bhazard ratio\b)|\bHR\b{_STAT_ABBREVIATION_EXPLICIT_SUFFIX_PATTERN}"
        ),
    ),
    (
        "odds_ratio",
        re.compile(
            rf"(?i:\bodds ratio\b)|\bOR\b{_STAT_ABBREVIATION_EXPLICIT_SUFFIX_PATTERN}"
        ),
    ),
    (
        "relative_risk",
        re.compile(
            rf"(?i:\brelative risk\b)|\bRR\b{_STAT_ABBREVIATION_EXPLICIT_SUFFIX_PATTERN}"
        ),
    ),
    (
        "standardized_mean_difference",
        re.compile(
            rf"(?i:\bstandardized mean difference\b)|\bSMD\b{_STAT_ABBREVIATION_SUFFIX_PATTERN}"
        ),
    ),
    ("cohens_d", re.compile(r"(?i)\bcohen'?s d\b")),
    ("hedges_g", re.compile(r"(?i)\bhedges'? g\b")),
)
_TRACE_STOP_TERMS = frozenset({
    "after",
    "before",
    "confidence",
    "effect",
    "from",
    "intervention",
    "non",
    "reported",
    "result",
    "significant",
    "size",
    "statistically",
    "study",
    "that",
    "their",
    "this",
    "trial",
    "were",
    "with",
})


def assess_publication_quality(
    result: MemoResult,
    *,
    public_markdown: str,
) -> dict[str, object]:
    """Build the evidence package and derive—not assert—the publish verdict."""
    receipts = tuple(result.receipts)
    blockers: list[dict[str, object]] = []
    source_checks: list[dict[str, object]] = []
    receipt_index = _receipt_index(receipts)

    for hit in receipts:
        identity = stable_source_identity(hit)
        issue = source_integrity_issue(hit)
        artifact_type = source_artifact_type(hit)
        verified_article_type = has_verified_primary_article_type(hit)
        check: dict[str, object] = {
            "receipt_id": hit.receipt_id,
            "status": "blocked" if issue else "stable_locator",
            "source_type": artifact_type,
            "source_type_verification": (
                "verified_article"
                if artifact_type == "article" and verified_article_type
                else "inferred_article"
                if artifact_type == "article"
                else "blocked"
            ),
            "retraction_status": (
                "retracted"
                if hit.metadata.get("is_retracted") is True
                else "not_retracted"
                if hit.metadata.get("retraction_status_known") is True
                else "unknown"
            ),
            "withdrawal_status": (
                "withdrawn"
                if hit.metadata.get("is_withdrawn") is True
                else "not_withdrawn"
                if hit.metadata.get("withdrawal_status_known") is True
                else "unknown"
            ),
        }
        if (
            hit.metadata.get("retraction_status_known") is True
            or hit.metadata.get("withdrawal_status_known") is True
        ):
            check["status_source"] = hit.source
        if identity is not None:
            check["identity"] = identity
            check["identity_verification"] = (
                "registry"
                if identity["kind"] in {
                    "arxiv",
                    "doi",
                    "openalex",
                    "pmcid",
                    "pmid",
                    "semantic_scholar",
                }
                else "url_only"
            )
        for key in ("document_type", "publication_types", "correction_status"):
            value = hit.metadata.get(key)
            if value not in (None, "", (), [], {}):
                check[key] = value
        if issue is not None:
            check["issue"] = issue["error"]
            blockers.append(issue)
        source_checks.append(check)

    candidate_blocker = candidate_publish_blocker(result.candidate)
    if candidate_blocker is not None:
        blockers.append(candidate_blocker)

    claim_cards = tuple(result.candidate.claim_cards)
    if not claim_cards:
        blockers.append({"error": "missing_claim_cards"})
    claim_ledger, trace_blockers = _claim_evidence_ledger(claim_cards, receipt_index)
    blockers.extend(trace_blockers)

    mapped_receipts = {card.receipt_id for card in claim_cards}
    for hit in receipts:
        if hit.receipt_id not in mapped_receipts and hit.hit_id not in mapped_receipts:
            blockers.append({"error": "unmapped_source_receipt", "receipt_id": hit.receipt_id})

    quantitative_traces, quantitative_blockers = _quantitative_claim_traces(
        public_markdown,
        receipts,
        claim_cards,
    )
    blockers.extend(quantitative_blockers)
    blockers.extend(_statistical_interpretation_blockers(public_markdown))

    risk_assessments = _risk_assessments(claim_cards, receipt_index)
    critical_risks = [
        assessment
        for assessment in risk_assessments
        if _effective_risk_level(assessment) == "critical"
    ]
    high_risks = [
        assessment
        for assessment in risk_assessments
        if _effective_risk_level(assessment) in _HIGH_RISK_LEVELS
    ]
    for assessment in critical_risks:
        blockers.append(
            {
                "error": "critical_risk_of_bias_primary_evidence",
                "receipt_id": assessment["receipt_id"],
                "overall": _effective_risk_level(assessment),
            }
        )
    if risk_assessments and len(high_risks) == len(risk_assessments):
        blockers.append(
            {
                "error": "all_primary_evidence_high_risk",
                "receipt_ids": [assessment["receipt_id"] for assessment in high_risks],
            }
        )

    blockers = _dedupe_blockers(blockers)
    ready = not blockers
    direct_human = sum(
        card.population == "human" and card.support_type == "direct"
        for card in claim_cards
    )
    trace_complete = bool(claim_cards) and len(claim_ledger) == len(claim_cards)
    source_complete = bool(receipts) and all(
        check["status"] == "stable_locator"
        and check["retraction_status"] == "not_retracted"
        and check["withdrawal_status"] == "not_withdrawn"
        and check["source_type_verification"] == "verified_article"
        and check.get("identity_verification") == "registry"
        for check in source_checks
    )
    tier_one = (
        ready
        and direct_human >= 2
        and trace_complete
        and source_complete
        and not high_risks
    )
    formal_risk_appraisals = sum(
        assessment["assessment_scope"] == "full_text" for assessment in risk_assessments
    )
    mature = tier_one and formal_risk_appraisals == len(risk_assessments)
    verdict: dict[str, object] = {
        "decision": "ready_to_publish" if ready else "revise",
        "publish_tier": "TIER_1" if tier_one else "TIER_2",
        "maturity_level": "L5" if mature else "L4",
        "confidence_label": "evidence_backed_signal" if tier_one else "bounded_evidence_brief",
        "blockers": [str(blocker["error"]) for blocker in blockers],
        "axes": {
            "bound_receipts": len(receipts),
            "stable_source_identities": sum("identity" in check for check in source_checks),
            "registry_verified_source_identities": sum(
                check.get("identity_verification") == "registry" for check in source_checks
            ),
            "verified_article_sources": sum(
                check["source_type_verification"] == "verified_article"
                for check in source_checks
            ),
            "verified_not_retracted_sources": sum(
                check["retraction_status"] == "not_retracted" for check in source_checks
            ),
            "verified_not_withdrawn_sources": sum(
                check["withdrawal_status"] == "not_withdrawn" for check in source_checks
            ),
            "claim_traces": len(claim_ledger),
            "claim_cards": len(claim_cards),
            "quantitative_claim_traces": len(quantitative_traces),
            "abstract_risk_screens": len(risk_assessments) - formal_risk_appraisals,
            "risk_of_bias_appraisals": formal_risk_appraisals,
            "formal_risk_of_bias_appraisals": formal_risk_appraisals,
            "direct_human_receipts": direct_human,
        },
    }
    return {
        "publish_verdict": verdict,
        "quality_blockers": blockers,
        "source_integrity": source_checks,
        "claim_evidence_ledger": claim_ledger,
        "quantitative_claim_traces": quantitative_traces,
        "risk_of_bias": risk_assessments,
    }


def quality_blocker(assessment: Mapping[str, object]) -> dict[str, object] | None:
    raw = assessment.get("quality_blockers")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        return None
    first = raw[0]
    return dict(first) if isinstance(first, Mapping) else {"error": "publication_quality_blocked"}


def _receipt_index(receipts: Sequence[CorpusHit]) -> dict[str, CorpusHit]:
    index: dict[str, CorpusHit] = {}
    for hit in receipts:
        for identifier in (hit.hit_id, hit.receipt_id):
            if identifier:
                index[identifier] = hit
                index[identifier.casefold()] = hit
    return index


def _claim_evidence_ledger(
    cards: Sequence[ClaimCard],
    receipt_index: Mapping[str, CorpusHit],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    ledger: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []
    for card in cards:
        hit = receipt_index.get(card.receipt_id) or receipt_index.get(card.receipt_id.casefold())
        if hit is None:
            blockers.append({"error": "claim_receipt_missing", "receipt_id": card.receipt_id})
            continue
        source_text = _source_text(hit)
        quote = " ".join(card.quote.split()).strip()
        start = source_text.find(quote) if quote else -1
        if start < 0:
            blockers.append({"error": "claim_trace_missing", "receipt_id": card.receipt_id})
            continue
        risk = _risk_of_bias(card, hit) if _requires_risk_assessment(card) else None
        item: dict[str, object] = {
            "receipt_id": hit.receipt_id,
            "role": card.role,
            "design": card.design,
            "population": card.population,
            "endpoint": card.outcome,
            "direction": card.direction,
            "directness": card.support_type,
            "confidence": card.confidence,
            "evidence_span": {"start": start, "end": start + len(quote)},
            "evidence_quote": quote,
            "source_text_sha256": _source_hash(source_text),
        }
        if risk is not None:
            item["risk_of_bias"] = risk
        ledger.append(item)
    return ledger, blockers


def _quantitative_claim_traces(
    markdown: str,
    receipts: Sequence[CorpusHit],
    cards: Sequence[ClaimCard],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    traces: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []
    receipt_index = _receipt_index(receipts)
    for claim in _statistical_claims(markdown):
        raw_numbers = _STAT_NUMBER_RE.findall(claim)
        numbers = tuple(dict.fromkeys(_normalize_number(value) for value in raw_numbers))
        claim_terms = _trace_terms(claim)
        claim_constructs = _statistical_constructs(claim)
        claim_pairs = _construct_number_pairs(claim, claim_constructs)
        claim_context = _measurement_context(claim)
        matches: list[tuple[int, ClaimCard, CorpusHit, tuple[int, int]]] = []
        for card in cards:
            hit = receipt_index.get(card.receipt_id) or receipt_index.get(card.receipt_id.casefold())
            if hit is None:
                continue
            evidence_span = _matching_evidence_span(
                _source_text(hit),
                numbers,
                claim_constructs,
                claim_pairs,
                claim_context,
            )
            if evidence_span is None:
                continue
            start, end = evidence_span
            evidence_terms = _trace_terms(_source_text(hit)[start:end])
            endpoint_terms = _trace_terms(card.outcome)
            overlap = len(claim_terms & endpoint_terms & evidence_terms)
            if endpoint_terms and overlap:
                matches.append((overlap, card, hit, evidence_span))
        match = max(matches, key=lambda item: item[0], default=None)
        if match is None:
            blockers.append({"error": "quantitative_claim_untraced", "claim": claim})
            continue
        _, card, hit, (start, end) = match
        source_text = _source_text(hit)
        traces.append(
            {
                "claim": claim,
                "receipt_id": hit.receipt_id,
                "claim_role": card.role,
                "endpoint": card.outcome,
                "direction": card.direction,
                "numbers": list(numbers),
                "evidence_span": {"start": start, "end": end},
                "evidence_quote": source_text[start:end],
                "source_text_sha256": _source_hash(source_text),
            }
        )
    return traces, blockers


def _statistical_claims(markdown: str) -> tuple[str, ...]:
    claims = (
        " ".join(sentence.split()).strip()
        for sentence in _SENTENCE_SPLIT_RE.split(markdown)
        if (
            _STAT_ANCHOR_RE.search(sentence)
            and _STAT_NUMBER_RE.search(sentence)
            and _statistical_constructs(sentence)
        )
    )
    return tuple(dict.fromkeys(claim for claim in claims if claim))


def _statistical_interpretation_blockers(markdown: str) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    for sentence in _statistical_claims(markdown):
        clauses = tuple(
            clause.strip()
            for clause in _INTERPRETATION_CLAUSE_SPLIT_RE.split(sentence)
            if clause.strip()
        )
        for index, clause in enumerate(clauses):
            if not _NON_SIGNIFICANT_RE.search(clause):
                continue
            adjacent_explanations = tuple(
                clauses[adjacent_index]
                for adjacent_index in (index - 1, index + 1)
                if 0 <= adjacent_index < len(clauses)
                if _is_threshold_only_clause(clauses[adjacent_index])
            )
            claim = "; ".join((clause, *adjacent_explanations))
            blockers.extend(_interpretation_blockers_for_claim(claim))
    return blockers


def _interpretation_blockers_for_claim(claim: str) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if not _NON_SIGNIFICANT_RE.search(claim):
        return blockers
    has_explanation = _has_threshold_explanation(claim)
    p_matches = tuple(_P_VALUE_RE.finditer(claim))
    for match in p_matches:
        p_value = float(_normalize_number(match.group("value")))
        operator = match.group("operator")
        explicit_threshold = _applicable_threshold(
            claim,
            p_position=match.start(),
        )
        if explicit_threshold is not None:
            if _p_value_is_significant(p_value, operator, explicit_threshold):
                blockers.append(
                    {
                        "error": "contradictory_statistical_interpretation",
                        "claim": claim,
                        "p_value": p_value,
                        "threshold": explicit_threshold,
                    }
                )
            continue
        if has_explanation:
            continue
        if not _p_value_is_significant(
            p_value,
            operator,
            _DEFAULT_SIGNIFICANCE_THRESHOLD,
        ):
            continue
        blockers.append(
            {
                "error": "unexplained_statistical_threshold",
                "claim": claim,
                "p_value": p_value,
            }
        )
    return blockers


def _is_threshold_only_clause(claim: str) -> bool:
    if not (_EXPLICIT_THRESHOLD_RE.search(claim) or _has_threshold_explanation(claim)):
        return False
    if _P_VALUE_RE.search(claim) or _NON_SIGNIFICANT_RE.search(claim):
        return False
    terms = frozenset(re.findall(r"[a-z]+", claim.casefold()))
    return bool(terms) and terms <= _THRESHOLD_ONLY_TERMS


def _applicable_threshold(claim: str, *, p_position: int) -> float | None:
    """Choose the adjusted threshold governing a p-value, not incidental alpha text."""
    matches = tuple(_EXPLICIT_THRESHOLD_RE.finditer(claim))
    if not matches:
        return None

    def score(match: re.Match[str]) -> tuple[int, int, int]:
        context_start = max(
            claim.rfind(";", 0, match.start()),
            claim.rfind(",", 0, match.start()),
            claim.rfind(" and ", 0, match.start()),
        )
        following = [
            index
            for token in (";", ",", " and ")
            if (index := claim.find(token, match.end())) >= 0
        ]
        context_end = min(following, default=len(claim))
        context = claim[context_start + 1 : context_end].casefold()
        adjusted = bool(
            re.search(
                r"\b(?:adjusted|bonferroni|fdr|false discovery rate|corrected)\b",
                context,
            )
        )
        nominal = bool(re.search(r"\b(?:nominal|unadjusted)\b", context))
        label_priority = int(match.group("label").casefold().endswith("threshold"))
        return (
            2 if adjusted else -2 if nominal else 0,
            label_priority,
            -abs(match.start() - p_position),
        )

    selected = max(matches, key=score)
    return float(_normalize_number(selected.group("value")))


def _p_value_is_significant(p_value: float, operator: str, threshold: float) -> bool:
    """Return whether the reported p-value bound meets an inclusive alpha."""
    if operator in {"<", "<=", "\u2264", "="}:
        return p_value <= threshold
    return False


def _risk_assessments(
    cards: Sequence[ClaimCard],
    receipt_index: Mapping[str, CorpusHit],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen: set[str] = set()
    for card in cards:
        if not _requires_risk_assessment(card) or card.receipt_id in seen:
            continue
        hit = receipt_index.get(card.receipt_id) or receipt_index.get(card.receipt_id.casefold())
        if hit is not None:
            out.append(_risk_of_bias(card, hit))
            seen.add(card.receipt_id)
    return out


def _risk_of_bias(card: ClaimCard, hit: CorpusHit) -> dict[str, object]:
    raw = hit.metadata.get("risk_of_bias")
    supplied_data = raw if isinstance(raw, Mapping) else {}
    supplied = _normalize_risk_level(
        supplied_data.get("overall") if supplied_data else raw
    )
    allowed_levels = {"low", "some_concerns", "serious", "high", "critical"}
    raw_domains = supplied_data.get("domains")
    supplied_scope = str(supplied_data.get("assessment_scope") or "").strip().casefold()
    provenance = str(supplied_data.get("provenance") or "").strip()
    supplied_tool = str(supplied_data.get("tool") or "").strip()
    formal_tool = _formal_risk_tool(supplied_tool)
    normalized_domains = (
        {
            normalized_name: normalized_value
            for key, value in raw_domains.items()
            if (normalized_name := _normalize_risk_domain_name(key))
            and (normalized_value := _normalize_domain_judgment(value))
        }
        if isinstance(raw_domains, Mapping)
        else {}
    )
    required_domains = _FORMAL_RISK_DOMAINS.get(formal_tool, frozenset())
    complete_domains = bool(required_domains) and required_domains <= normalized_domains.keys()
    formal = (
        supplied_scope == "full_text"
        and complete_domains
        and bool(provenance)
        and bool(formal_tool)
        and supplied in allowed_levels
    )
    if formal:
        domains = normalized_domains
        overall = supplied if supplied in allowed_levels else "not_assessed"
        tool = supplied_tool
        screening_flag = ""
    else:
        domains = {
            "allocation_or_confounding": "not_reported",
            "deviations_from_intervention": "not_reported",
            "missing_outcome_data": "not_reported",
            "outcome_measurement": "not_reported",
            "selective_reporting": "not_reported",
        }
        overall = "not_assessed"
        tool = "V5 abstract evidence-risk screen"
        screening_flag = supplied if supplied in _HIGH_RISK_LEVELS else ""
    assessment: dict[str, object] = {
        "receipt_id": hit.receipt_id,
        "tool": tool,
        "assessment_scope": "full_text" if formal else "abstract_only",
        "overall": overall,
        "domains": domains,
        "provenance": (
            provenance
            if formal
            else "unverified provider metadata"
            if raw is not None
            else "receipt title and abstract only"
        ),
    }
    if screening_flag:
        assessment["screening_flag"] = screening_flag
    return assessment


def _effective_risk_level(assessment: Mapping[str, object]) -> str:
    overall = str(assessment.get("overall") or "")
    if overall in _HIGH_RISK_LEVELS:
        return overall
    return str(assessment.get("screening_flag") or "")


def _normalize_risk_level(value: object) -> str:
    text = re.sub(r"[^a-z]+", " ", str(value or "").casefold()).strip()
    if not text or any(phrase in text for phrase in ("not high risk", "not critical risk")):
        return ""
    if "critical" in text:
        return "critical"
    if "serious" in text:
        return "serious"
    if "high" in text:
        return "high"
    if "some concern" in text:
        return "some_concerns"
    if text == "low" or "low risk" in text:
        return "low"
    return ""


def _normalize_domain_judgment(value: object) -> str:
    level = _normalize_risk_level(value)
    if level:
        return level
    text = re.sub(r"[^a-z]+", " ", str(value or "").casefold()).strip()
    if "moderate" in text:
        return "moderate"
    if text in {"no information", "not reported"}:
        return "no_information"
    return ""


def _formal_risk_tool(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    if normalized in {"rob 2", "risk of bias 2"}:
        return "rob_2"
    if normalized in {"robins i", "robins 1"}:
        return "robins_i"
    return ""


def _normalize_risk_domain_name(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()
    aliases = {
        "randomization": "randomization",
        "randomization process": "randomization",
        "bias due to confounding": "confounding",
        "confounding": "confounding",
        "selection of participants": "participant_selection",
        "participant selection": "participant_selection",
        "classification of interventions": "intervention_classification",
        "intervention classification": "intervention_classification",
        "deviations from intended interventions": "deviations_from_intervention",
        "deviations from intervention": "deviations_from_intervention",
        "missing outcome data": "missing_outcome_data",
        "measurement of the outcome": "outcome_measurement",
        "outcome measurement": "outcome_measurement",
        "selection of the reported result": "selective_reporting",
        "selective reporting": "selective_reporting",
    }
    return aliases.get(normalized, "")


def _requires_risk_assessment(card: ClaimCard) -> bool:
    return card.population == "human" and card.support_type == "direct"


def _source_text(hit: CorpusHit) -> str:
    return " ".join(f"{hit.title}. {hit.abstract}".split()).strip()


def _source_hash(source_text: str) -> str:
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()


def _number_positions(text: str, numbers: Sequence[str]) -> tuple[tuple[int, int], ...] | None:
    matches = tuple(
        (match.start(), match.end(), _normalize_number(match.group(0)))
        for match in _STAT_NUMBER_RE.finditer(text)
    )
    positions: list[tuple[int, int]] = []
    for number in numbers:
        position = next(((start, end) for start, end, value in matches if value == number), None)
        if position is None:
            return None
        positions.append(position)
    return tuple(positions)


def _matching_evidence_span(
    text: str,
    numbers: Sequence[str],
    constructs: frozenset[str],
    claim_pairs: frozenset[tuple[str, str]],
    claim_context: frozenset[str],
) -> tuple[int, int] | None:
    """Return one sentence containing every value and statistical construct."""
    start = 0
    boundaries = [*list(_SENTENCE_SPLIT_RE.finditer(text)), None]
    for boundary in boundaries:
        end = boundary.start() if boundary is not None else len(text)
        sentence = text[start:end].strip()
        leading = len(text[start:end]) - len(text[start:end].lstrip())
        sentence_start = start + leading
        if (
            sentence
            and _number_positions(sentence, numbers) is not None
            and constructs <= _statistical_constructs(sentence)
            and claim_pairs <= _construct_number_pairs(sentence, constructs)
            and not _NON_RESULT_CONTEXT_RE.search(sentence)
            and _measurement_context(sentence) <= claim_context
        ):
            return sentence_start, sentence_start + len(sentence)
        start = boundary.end() if boundary is not None else len(text)
    return None


def _statistical_constructs(text: str) -> frozenset[str]:
    return frozenset(
        name
        for name, pattern in _STAT_CONSTRUCT_PATTERNS
        if pattern.search(text) and not _ambiguous_abbreviation(name, text)
    )


def _ambiguous_abbreviation(name: str, text: str) -> bool:
    lowered = text.casefold()
    if name == "hazard_ratio" and "hazard ratio" not in lowered:
        return bool(
            re.search(
                r"(?i)\b(?:heart rate|pulse|bpm)\b|"
                r"\bbeats?(?:\s+per\s+minute|\s*/\s*min(?:ute)?s?)\b",
                text,
            )
        )
    if name == "relative_risk" and "relative risk" not in lowered:
        return bool(
            re.search(
                r"(?i)\b(?:respiratory rate|respiration)\b|"
                r"\bbreaths?(?:\s+per\s+minute|\s*/\s*min(?:ute)?s?)\b|"
                r"/\s*min(?:ute)?s?\b",
                text,
            )
        )
    return False


def _construct_number_pairs(
    text: str,
    constructs: frozenset[str],
) -> frozenset[tuple[str, str]]:
    number_matches = tuple(
        match for match in _STAT_NUMBER_RE.finditer(text)
    )
    pairs: set[tuple[str, str]] = set()
    for name, pattern in _STAT_CONSTRUCT_PATTERNS:
        if name not in constructs or _ambiguous_abbreviation(name, text):
            continue
        for construct in pattern.finditer(text):
            candidates: list[tuple[int, str]] = []
            for number in number_matches:
                overlaps = number.start() < construct.end() and number.end() > construct.start()
                forward_gap = text[construct.end() : number.start()]
                reverse_gap = text[number.end() : construct.start()]
                follows = (
                    0 <= number.start() - construct.end() <= 80
                    and not re.search(
                        r"[;.!?]|,\s*(?:and|but|while|whereas)\b",
                        forward_gap,
                        re.IGNORECASE,
                    )
                )
                immediately_precedes = (
                    0 <= construct.start() - number.end() <= 24
                    and bool(re.fullmatch(r"\s*(?:(?:is|was|as)\s+)?(?:the\s+)?", reverse_gap))
                )
                if overlaps or follows or immediately_precedes:
                    distance = 0 if overlaps else min(
                        abs(number.start() - construct.end()),
                        abs(construct.start() - number.end()),
                    )
                    candidates.append((distance, _normalize_number(number.group(0))))
            if candidates:
                if name == "confidence_interval":
                    pairs.update((name, number) for _, number in candidates)
                else:
                    _, normalized_number = min(candidates, key=lambda item: item[0])
                    pairs.add((name, normalized_number))
    return frozenset(pairs)


def _has_threshold_explanation(claim: str) -> bool:
    non_negated = _NEGATED_THRESHOLD_EXPLANATION_RE.sub("", claim)
    return bool(_THRESHOLD_EXPLANATION_RE.search(non_negated))


def _measurement_context(text: str) -> frozenset[str]:
    return frozenset(
        name for name, pattern in _MEASUREMENT_CONTEXT_PATTERNS if pattern.search(text)
    )


def _normalize_number(value: str) -> str:
    raw = value.strip().rstrip("%").replace("\u2212", "-")
    number = float(raw)
    return f"{number:g}" + ("%" if value.strip().endswith("%") else "")


def _trace_terms(text: str) -> frozenset[str]:
    return frozenset(
        term
        for term in re.findall(r"[a-z][a-z0-9-]{3,}", text.casefold())
        if term not in _TRACE_STOP_TERMS
    )


def _dedupe_blockers(blockers: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen: set[str] = set()
    for blocker in blockers:
        key = json.dumps(blocker, sort_keys=True, separators=(",", ":"), default=str)
        if key not in seen:
            out.append(blocker)
            seen.add(key)
    return out
