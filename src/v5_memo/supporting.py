"""Fail-closed selection of non-core citations for bounded evidence briefs."""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from v5_memo.binder import _evidence_unit_key
from v5_memo.evidence import (
    has_verified_primary_article_type,
    is_non_primary_receipt,
    source_integrity_issue,
    stable_source_identity,
)
from v5_memo.gate import _topic_entity_terms
from v5_memo.miner import _claim_card, _norm_token
from v5_memo.schemas import CorpusHit, ReceiptRole

_DIRECT_DESIGNS = frozenset({"cohort", "intervention_study", "randomized_trial"})
_TOPIC_CONTEXT_STOP = frozenset({
    "adult",
    "adults",
    "blind",
    "blinded",
    "clinical",
    "controlled",
    "double",
    "exercise",
    "human",
    "humans",
    "intervention",
    "older",
    "placebo",
    "program",
    "randomised",
    "randomized",
    "supplement",
    "supplementation",
    "study",
    "studies",
    "therapy",
    "training",
    "treatment",
    "trial",
    "trials",
})
_COMPETING_INTERVENTION_PHRASES = (
    "alone or in combination",
    "alone and in combination",
    "either alone or combined",
    "independent and combined effects",
    "independent or combined effects",
    "combined intervention",
    "multi-arm",
    "multi arm",
)
_INTERVENTION_MARKER_RE = re.compile(
    r"\b(?:exercise|omega-?3|program|supplement(?:ation)?|therapy|training|treatment)\b"
)
_INTERVENTION_LIST_RE = re.compile(r",\s*(?:and|or)\s+(?:an?\s+|the\s+)?")
_COMBINED_SUPPLEMENT_RE = re.compile(
    r"\b[a-z0-9-]+(?:\s+[a-z0-9-]+)?\s+(?:and|plus)\s+"
    r"[a-z0-9-]+(?:\s+[a-z0-9-]+)?\s+supplementation\b"
)
_INTERVENTION_TERM = r"(?:exercise|program|supplement(?:ation)?|therapy|training|treatment)"
_PAIR_CONNECTOR = (
    r"(?:alongside|and|combined\s+with|in\s+combination\s+with|plus|together\s+with|with)"
)
_GENERIC_PAIR_CONNECTOR = (
    r"(?:alongside|and|combined\s+with|in\s+combination\s+with|plus|together\s+with)"
)
_RESULT_DIRECTION_RE = re.compile(
    r"\b(?:decreas(?:e|ed)|did not(?:\s+(?:affect|change|improve|increase|reduce))?|"
    r"higher|improv(?:e|ed)|increas(?:e|ed)|lower|"
    r"no (?:(?:statistically|significant|obvious)\s+){0,2}"
    r"(?:changes?|differences?|effects?)|reduc(?:e|ed)|unchanged)\b"
)
_PREPRINT_DOI_PREFIXES = ("10.1101/",)
_PREPRINT_METADATA_KEYS = (
    "document_type",
    "publication_type",
    "publication_types",
    "subtype",
    "type",
    "work_type",
)
_PREPRINT_MARKERS = ("posted content", "preprint", "working paper")
_HUMAN_TITLE_RE = re.compile(
    r"\b(?:adolescents?|children|elderly|girls?|boys?|people|persons?|residents?|"
    r"schoolchildren|seniors?|women|men|patients?|participants?|adults?|humans?)\b"
)
_POPULATION_MODIFIER_TOKENS = frozenset({
    "community-dwelling",
    "elderly",
    "frail",
    "healthy",
    "hospitalized",
    "institutionalized",
    "older",
    "postmenopausal",
    "premenopausal",
    "pregnant",
    "younger",
})
_RESULT_LINK_TERMS = frozenset({
    "a",
    "an",
    "any",
    "at",
    "by",
    "change",
    "clinically",
    "efficacy",
    "for",
    "in",
    "is",
    "mean",
    "meaningfully",
    "measured",
    "observed",
    "of",
    "on",
    "outcome",
    "endpoint",
    "intervention",
    "significantly",
    "statistically",
    "the",
    "to",
    "was",
    "were",
})


def select_supporting_receipts(
    *,
    topic: str,
    hits: Sequence[CorpusHit],
    core_receipts: Sequence[CorpusHit],
    needed: int,
    min_shards_searched: int = 0,
    min_sources_searched: int = 0,
    min_search_passes: int = 0,
) -> tuple[CorpusHit, ...]:
    """Return exactly ``needed`` deterministic, strict-complete support citations."""
    if needed <= 0:
        return ()
    core_keys = {hit.source_key for hit in core_receipts}
    core_units = {_evidence_unit_key(hit) for hit in core_receipts}
    ranked: list[tuple[int, int, int, str, CorpusHit]] = []
    for hit in hits:
        unit = _evidence_unit_key(hit)
        if (
            hit.source_key in core_keys
            or unit in core_units
            or not _eligible_support(
                hit,
                topic,
                min_shards_searched=min_shards_searched,
                min_sources_searched=min_sources_searched,
                min_search_passes=min_search_passes,
            )
        ):
            continue
        overlap = len(_support_axis_terms(topic) & _normalized_tokens(hit.text))
        cited_by = _int_value(hit.metadata.get("cited_by_count"))
        ranked.append((-overlap, -cited_by, -(hit.year or 0), hit.source_key, hit))
    ranked.sort(key=lambda item: item[:4])
    selected: list[CorpusHit] = []
    seen = set(core_keys)
    seen_units = set(core_units)
    for *_, hit in ranked:
        unit = _evidence_unit_key(hit)
        if hit.source_key in seen or unit in seen_units:
            continue
        seen.add(hit.source_key)
        seen_units.add(unit)
        selected.append(hit)
        if len(selected) == needed:
            break
    return tuple(selected)


def _eligible_support(
    hit: CorpusHit,
    topic: str,
    *,
    min_shards_searched: int,
    min_sources_searched: int,
    min_search_passes: int,
) -> bool:
    identity = stable_source_identity(hit)
    if identity is None or identity.get("kind") not in {"doi", "pmid"}:
        return False
    if _is_preprint(hit, str(identity.get("value") or "")):
        return False
    if (
        source_integrity_issue(hit) is not None
        or is_non_primary_receipt(hit)
        or not has_verified_primary_article_type(hit)
        or (
            str(hit.metadata.get("pmcid") or "").strip()
            and hit.metadata.get("source_type_verification") != "europe_pmc_jats"
        )
    ):
        return False
    if len(re.findall(r"[a-z0-9]+", hit.abstract.casefold())) < 20:
        return False
    if not _strict_complete_receipt(
        hit,
        min_shards_searched=min_shards_searched,
        min_sources_searched=min_sources_searched,
        min_search_passes=min_search_passes,
    ):
        return False
    entity_terms = _topic_entity_terms(topic)
    abstract_terms = tuple(_normalized_tokens_in_order(hit.abstract))
    if not entity_terms or not _contains_sequence(abstract_terms, entity_terms):
        return False
    axis_phrases = _support_axis_phrases(topic)
    if not axis_phrases:
        return False
    if not _has_axis_result_statement(hit.abstract, axis_phrases, entity_terms):
        return False
    if _has_competing_intervention_structure(hit, topic):
        return False
    card = _claim_card(
        hit,
        ReceiptRole(hit.hit_id, "supporting_context", "non-core source-bundle support"),
    )
    human_population = (
        card.population == "human" or _HUMAN_TITLE_RE.search(hit.title) is not None
    )
    return (
        card.design in _DIRECT_DESIGNS
        and human_population
        and card.direction != "unclear"
    )


def _strict_complete_receipt(
    hit: CorpusHit,
    *,
    min_shards_searched: int,
    min_sources_searched: int,
    min_search_passes: int,
) -> bool:
    raw = hit.metadata.get("shard_receipt")
    if not isinstance(raw, Mapping):
        return False
    searched = _int_value(raw.get("shards_searched"))
    total = _int_value(raw.get("shards_total"))
    remaining = raw.get("sweep_remaining_shards")
    raw_sources = raw.get("sources_searched")
    source_count = (
        sum(_int_value(count) > 0 for count in raw_sources.values())
        if isinstance(raw_sources, Mapping)
        else 0
    )
    search_passes = _search_pass_count(hit)
    return (
        total > 0
        and searched >= total
        and searched >= min_shards_searched
        and raw.get("partial_shard_search") is False
        and "sweep_failed_shards" in raw
        and _int_value(raw.get("sweep_failed_shards")) == 0
        and remaining is not None
        and _int_value(remaining) == 0
        and source_count >= min_sources_searched
        and search_passes >= min_search_passes
    )


def _search_pass_count(hit: CorpusHit) -> int:
    raw = hit.metadata.get("fullraw_search_receipt")
    if isinstance(raw, Mapping):
        passes = raw.get("search_passes")
        if isinstance(passes, Sequence) and not isinstance(passes, (str, bytes)):
            return len({str(item) for item in passes if str(item)})
    return int(bool(hit.metadata.get("search_pass")))


def _has_competing_intervention_structure(hit: CorpusHit, topic: str) -> bool:
    text = " ".join(hit.text.casefold().split())
    entity_terms = _topic_entity_terms(topic)
    axis_terms = _support_axis_terms(topic)
    if "factorial" in text or any(
        phrase in text for phrase in _COMPETING_INTERVENTION_PHRASES
    ):
        return True
    title = " ".join(hit.title.casefold().split())
    return (
        _COMBINED_SUPPLEMENT_RE.search(title) is not None
        or _topic_has_paired_intervention(title, entity_terms, axis_terms)
        or _topic_has_paired_intervention(text, entity_terms, axis_terms)
        or (
            _INTERVENTION_LIST_RE.search(title) is not None
            and len(_INTERVENTION_MARKER_RE.findall(title)) >= 2
        )
    )


def _topic_has_paired_intervention(
    title: str,
    entity_terms: Sequence[str],
    axis_terms: frozenset[str],
) -> bool:
    if not entity_terms:
        return False
    entity_pattern = r"\b" + r"\s+".join(map(re.escape, entity_terms)) + r"\b"
    paired_after = re.compile(
        rf"^\s+(?:{_INTERVENTION_TERM}\s+)?{_PAIR_CONNECTOR}\s+"
        rf"(?:[a-z0-9-]+\s+){{0,2}}{_INTERVENTION_TERM}\b"
    )
    generic_after = re.compile(
        rf"^\s+(?:{_INTERVENTION_TERM}\s+)?{_GENERIC_PAIR_CONNECTOR}\s+"
        r"(?!placebo\b|control\b)(?P<candidate>[a-z0-9-]+)\b"
    )
    paired_before = re.compile(
        rf"{_INTERVENTION_TERM}\s+{_PAIR_CONNECTOR}\s*$"
    )
    for match in re.finditer(entity_pattern, title):
        suffix = title[match.end() :]
        generic_match = generic_after.search(suffix)
        if (
            paired_after.search(suffix) is not None
            or paired_before.search(title[: match.start()]) is not None
            or (
                generic_match is not None
                and (
                    candidate := _norm_token(generic_match.group("candidate"))
                ) not in axis_terms
                and candidate not in _TOPIC_CONTEXT_STOP
                and candidate not in _POPULATION_MODIFIER_TOKENS
                and _HUMAN_TITLE_RE.fullmatch(candidate) is None
            )
        ):
            return True
    return False


def _has_axis_result_statement(
    abstract: str,
    axis_phrases: Sequence[Sequence[str]],
    entity_terms: Sequence[str],
) -> bool:
    link_terms = _RESULT_LINK_TERMS | frozenset(entity_terms)
    for sentence in re.split(r"(?<=[.!?])\s+", abstract.casefold()):
        clauses = re.split(
            r";\s*|(?:[,;]\s*)?\b(?:and|but|however|whereas|while)\b\s*",
            sentence,
        )
        for clause in clauses:
            raw_tokens = tuple(re.finditer(r"[a-z0-9]+", clause.casefold()))
            tokens = tuple(_norm_token(match.group()) for match in raw_tokens)
            for axis_phrase in axis_phrases:
                axis_positions = tuple(
                    position
                    for start in range(len(tokens) - len(axis_phrase) + 1)
                    if tuple(tokens[start : start + len(axis_phrase)]) == tuple(axis_phrase)
                    for position in range(start, start + len(axis_phrase))
                )
                if not axis_positions:
                    continue
                for match in _RESULT_DIRECTION_RE.finditer(clause):
                    direction_positions = tuple(
                        index
                        for index, token in enumerate(raw_tokens)
                        if token.start() < match.end() and token.end() > match.start()
                    )
                    if direction_positions and any(
                        _axis_result_is_attached(
                            tokens,
                            axis_position,
                            direction_positions,
                            link_terms,
                        )
                        for axis_position in axis_positions
                    ):
                        return True
    return False


def _support_axis_phrases(topic: str) -> tuple[tuple[str, ...], ...]:
    entity = set(_topic_entity_terms(topic))
    ordered = tuple(
        term
        for term in _normalized_tokens_in_order(topic)
        if term not in entity and term not in _TOPIC_CONTEXT_STOP
    )
    if {"muscle", "strength"} <= set(ordered):
        return (
            ("muscle", "strength"),
            ("muscular", "strength"),
            ("grip", "strength"),
            ("handgrip",),
        )
    return (ordered,) if ordered else ()


def _axis_result_is_attached(
    tokens: Sequence[str],
    axis_position: int,
    direction_positions: Sequence[int],
    link_terms: frozenset[str],
) -> bool:
    direction_start = min(direction_positions)
    direction_end = max(direction_positions)
    if axis_position < direction_start:
        bridge = tokens[axis_position + 1 : direction_start]
        return len(bridge) <= 3 and all(term in link_terms for term in bridge)
    if axis_position > direction_end:
        bridge = tokens[direction_end + 1 : axis_position]
        return len(bridge) <= 5 and all(term in link_terms for term in bridge)
    return True


def _is_preprint(hit: CorpusHit, identity_value: str) -> bool:
    if identity_value.casefold().startswith(_PREPRINT_DOI_PREFIXES):
        return True
    for key in _PREPRINT_METADATA_KEYS:
        raw = hit.metadata.get(key)
        values = raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else (raw,)
        if any(
            marker in str(value or "").casefold()
            for value in values
            for marker in _PREPRINT_MARKERS
        ):
            return True
    return False


def supporting_receipts_are_valid(
    *,
    topic: str,
    supporting_receipts: Sequence[CorpusHit],
    core_receipts: Sequence[CorpusHit],
    min_shards_searched: int,
    min_sources_searched: int = 0,
    min_search_passes: int = 0,
) -> bool:
    """Revalidate final non-core citations before a payload can be submitted."""
    selected = select_supporting_receipts(
        topic=topic,
        hits=supporting_receipts,
        core_receipts=core_receipts,
        needed=len(supporting_receipts),
        min_shards_searched=min_shards_searched,
        min_sources_searched=min_sources_searched,
        min_search_passes=min_search_passes,
    )
    return (
        len(selected) == len(supporting_receipts)
        and {hit.source_key for hit in selected}
        == {hit.source_key for hit in supporting_receipts}
    )


def _support_axis_terms(topic: str) -> frozenset[str]:
    return frozenset(term for phrase in _support_axis_phrases(topic) for term in phrase)


def _normalized_tokens(text: str) -> frozenset[str]:
    return frozenset(_normalized_tokens_in_order(text))


def _normalized_tokens_in_order(text: str) -> tuple[str, ...]:
    return tuple(_norm_token(token) for token in re.findall(r"[a-z0-9]+", text.casefold()))


def _contains_sequence(tokens: Sequence[str], expected: Sequence[str]) -> bool:
    width = len(expected)
    return any(
        tuple(tokens[index:index + width]) == tuple(expected)
        for index in range(len(tokens) - width + 1)
    )


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0
