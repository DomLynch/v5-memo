from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import pytest

from v5_memo.evidence import (
    has_verified_primary_article_type,
    merge_publication_integrity,
    normalize_publication_integrity,
    source_integrity_issue,
    stable_source_identity,
)
from v5_memo.miner import mine_insights
from v5_memo.pipeline import build_alpha_memo
from v5_memo.publication_quality import assess_publication_quality, quality_blocker
from v5_memo.publisher import (
    build_researka_payload,
    publication_quality_blocker,
    submission_readiness_blocker,
)
from v5_memo.schemas import ClaimCard, CorpusHit, InsightCandidate, MemoResult
from v5_memo.supporting import select_supporting_receipts
from v5_memo.writer import render_memo


def _quality_result(*, metadata: dict[str, object] | None = None) -> MemoResult:
    verified_article = {
        "document_type": "Journal Article",
        "is_retracted": False,
        "retraction_status_known": True,
        "is_withdrawn": False,
        "withdrawal_status_known": True,
    }
    receipts = (
        CorpusHit(
            hit_id="trial-a",
            title="Randomized human intervention trial A",
            abstract="The intervention improved the measured endpoint in adults.",
            source="fullraw:pubmed",
            doi="10.1234/trial-a",
            metadata={**verified_article, **(metadata or {})},
        ),
        CorpusHit(
            hit_id="trial-b",
            title="Randomized human intervention trial B",
            abstract="The intervention reduced the measured endpoint in adults.",
            source="fullraw:pubmed",
            doi="10.1234/trial-b",
            metadata=verified_article,
        ),
    )
    cards = (
        ClaimCard(
            receipts[0].receipt_id,
            "positive_signal",
            "randomized_trial",
            "human",
            "measured endpoint",
            "positive",
            "direct",
            "high",
            "Randomized human intervention trial A. The intervention improved the measured endpoint in adults.",
        ),
        ClaimCard(
            receipts[1].receipt_id,
            "negative_signal",
            "randomized_trial",
            "human",
            "measured endpoint",
            "negative",
            "direct",
            "high",
            "Randomized human intervention trial B. The intervention reduced the measured endpoint in adults.",
        ),
    )
    candidate = InsightCandidate(
        topic="intervention outcome",
        thesis="The intervention has a bounded endpoint-dependent signal.",
        bridge_terms=("intervention", "endpoint"),
        tension_terms=("positive", "negative"),
        receipt_ids=tuple(hit.receipt_id for hit in receipts),
        score=90,
        novelty_score=50,
        evidence_score=90,
        reasons=("tier:publishable_alpha", "shape:directional_reversal"),
        claim_cards=cards,
    )
    return MemoResult(candidate=candidate, receipts=receipts, markdown=render_memo(candidate, receipts))


def _strict_support(index: int, *, topic: str = "intervention outcome") -> CorpusHit:
    return CorpusHit(
        hit_id=f"support-{index}",
        title=f"Randomized human {topic} support trial {index}",
        abstract=(
            f"A randomized placebo-controlled trial in human adults tested {topic}. "
            f"The intervention improved the measured {topic} outcome with a prespecified protocol "
            "and reported complete primary endpoint results for all enrolled participants."
        ),
        source="fullraw:pubmed",
        doi=f"10.1234/support-{index}",
        year=2020 + index,
        metadata={
            "document_type": "Journal Article",
            "publication_types": ["Randomized Controlled Trial", "Journal Article"],
            "is_retracted": False,
            "is_withdrawn": False,
            **_strict_retrieval_metadata(),
        },
    )


def _strict_retrieval_metadata() -> dict[str, object]:
    return {
        "shard_receipt": {
            "shards_searched": 1525,
            "shards_total": 1525,
            "partial_shard_search": False,
            "sweep_failed_shards": 0,
            "sweep_remaining_shards": 0,
            "sources_searched": {
                "crossref": 1,
                "openalex": 1,
                "pubmed": 1,
                "semantic_scholar": 1,
                "unpaywall": 1,
            },
        },
        "search_pass": "focused",
        "fullraw_search_receipt": {"search_passes": ["focused"]},
    }


@pytest.mark.parametrize(
    ("hit", "kind"),
    [
        (CorpusHit("doi", "Article", "Evidence text.", "test", doi="10.1234/article"), "doi"),
        (CorpusHit("12345", "Article", "Evidence text.", "test", metadata={"pmid": "12345"}), "pmid"),
        (CorpusHit("pmc", "Article", "Evidence text.", "test", metadata={"pmcid": "PMC12345"}), "pmcid"),
        (CorpusHit("oa", "Article", "Evidence text.", "test", metadata={"openalex_id": "https://openalex.org/W12345"}), "openalex"),
        (CorpusHit("url", "Article", "Evidence text.", "test", url="https://example.org/papers/1"), "url"),
    ],
)
def test_stable_source_identity_accepts_canonical_public_locators(
    hit: CorpusHit,
    kind: str,
) -> None:
    identity = stable_source_identity(hit)

    assert identity is not None
    assert identity["kind"] == kind
    assert source_integrity_issue(hit) is None


def test_source_integrity_rejects_title_only_and_malformed_doi() -> None:
    hit = CorpusHit(
        hit_id="title-only",
        title="Unidentified article",
        abstract="Evidence text.",
        source="test",
        doi="10.bad",
        url="https://doi.org/10.bad",
    )

    assert stable_source_identity(hit) is None
    assert source_integrity_issue(hit) == {
        "error": "missing_stable_source_identity",
        "receipt_id": "10.bad",
    }


@pytest.mark.parametrize(
    "raw",
    [
        {"metadata": {"retracted": True}},
        {"retracted": "retracted"},
        {"retraction_status": "retracted"},
    ],
)
def test_retraction_metadata_shapes_fail_closed(raw: dict[str, object]) -> None:
    metadata = normalize_publication_integrity(raw)
    hit = CorpusHit(
        "retracted",
        "Ordinary article title",
        "Evidence text.",
        "test",
        doi="10.1234/retracted",
        metadata=metadata,
    )

    assert source_integrity_issue(hit) == {
        "error": "source_retracted",
        "receipt_id": "10.1234/retracted",
    }


def test_explicit_not_retracted_status_is_known_and_allowed() -> None:
    metadata = normalize_publication_integrity({"retraction_status": "not_retracted"})
    hit = CorpusHit(
        "current",
        "Ordinary article title",
        "Evidence text.",
        "test",
        doi="10.1234/current",
        metadata=metadata,
    )

    assert metadata["is_retracted"] is False
    assert metadata["retraction_status_known"] is True
    assert source_integrity_issue(hit) is None


@pytest.mark.parametrize("value", ["unknown", "pending", "N/A", {}, [], 2])
def test_unrecognized_retraction_values_remain_unknown(value: object) -> None:
    metadata = normalize_publication_integrity({"is_retracted": value})

    assert metadata["is_retracted"] is None
    assert metadata["retraction_status_known"] is False


@pytest.mark.parametrize("value", ["not_retracted", "notRetracted", "unretracted"])
def test_retraction_negation_shapes_are_known_safe(value: str) -> None:
    metadata = normalize_publication_integrity({"retraction_status": value})

    assert metadata["is_retracted"] is False
    assert metadata["retraction_status_known"] is True


def test_integrity_merge_preserves_unsafe_duplicate_status() -> None:
    merged = merge_publication_integrity(
        {
            "document_type": "Article",
            "is_retracted": False,
            "retraction_status_known": True,
        },
        {
            "document_type": "Retraction Notice",
            "is_retracted": True,
            "correction_status": "RetractionIn",
        },
    )

    assert merged["is_retracted"] is True
    assert "Retraction Notice" in cast(tuple[str, ...], merged["publication_types"])
    assert merged["correction_status"] == "RetractionIn"


@pytest.mark.parametrize(
    ("unsafe_key", "safe_status", "result_key"),
    [
        ("is_retracted", "not_retracted", "is_retracted"),
        ("is_withdrawn", "not_withdrawn", "is_withdrawn"),
    ],
)
def test_explicit_unsafe_boolean_dominates_conflicting_safe_status_text(
    unsafe_key: str,
    safe_status: str,
    result_key: str,
) -> None:
    metadata = normalize_publication_integrity({
        "type": "article",
        unsafe_key: True,
        "retraction_status": safe_status,
    })

    assert metadata[result_key] is True


@pytest.mark.parametrize(
    ("title", "metadata", "error"),
    [
        ("Ordinary article title", {"is_retracted": True}, "source_retracted"),
        ("Ordinary article title", {"correction_status": "RetractionIn"}, "source_retracted"),
        ("Ordinary article title", {"correction_status": "ExpressionOfConcernIn"}, "source_expression_of_concern"),
        ("Expression of concern: ordinary article", {}, "source_expression_of_concern"),
        ("Correction to: ordinary article", {}, "source_correction_notice"),
        ("Withdrawn article", {}, "source_withdrawn"),
        ("Ordinary article title", {"document_type": "Published Erratum"}, "source_correction_notice"),
    ],
)
def test_source_integrity_blocks_explicit_unsafe_publication_status(
    title: str,
    metadata: dict[str, object],
    error: str,
) -> None:
    hit = CorpusHit(
        hit_id="unsafe",
        title=title,
        abstract="Evidence text.",
        source="test",
        doi="10.1234/unsafe",
        metadata=metadata,
    )

    issue = source_integrity_issue(hit)

    assert issue is not None
    assert issue["error"] == error


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ({"update_type": "withdrawal"}, "source_withdrawn"),
        ({"update_type": "correction"}, "source_correction_notice"),
        ({"relation": {"is-retracted-by": "10.1234/notice"}}, "source_retracted"),
        ({"correction_status": "expression_of_concern"}, "source_expression_of_concern"),
        ({"correction_status": "expression-of-concern"}, "source_expression_of_concern"),
    ],
)
def test_provider_status_aliases_normalize_to_integrity_blocks(
    raw: dict[str, object],
    error: str,
) -> None:
    hit = CorpusHit(
        "unsafe-alias",
        "Ordinary article",
        "Evidence text.",
        "fullraw",
        doi="10.1234/unsafe-alias",
        metadata=normalize_publication_integrity(raw),
    )

    issue = source_integrity_issue(hit)

    assert issue is not None
    assert issue["error"] == error


@pytest.mark.parametrize(
    ("title", "error"),
    [
        ("Publisher Correction: ordinary article", "source_correction_notice"),
        ("Retracted: ordinary article", "source_retracted"),
        ("This article has been retracted", "source_retracted"),
        ("Expression-of-concern: ordinary article", "source_expression_of_concern"),
    ],
)
def test_common_notice_title_shapes_are_blocked(title: str, error: str) -> None:
    hit = CorpusHit(
        "notice",
        title,
        "Evidence text.",
        "test",
        doi="10.1234/notice",
    )

    issue = source_integrity_issue(hit)

    assert issue is not None
    assert issue["error"] == error


def test_semantic_scholar_numeric_id_is_not_misclassified_as_pmid() -> None:
    hit = CorpusHit(
        "987654321",
        "Article",
        "Evidence text.",
        "fullraw:semantic_scholar",
        metadata={"semantic_scholar_id": "987654321"},
    )

    identity = stable_source_identity(hit)

    assert identity is not None
    assert identity["kind"] == "semantic_scholar"
    assert identity["value"] == "CorpusID:987654321"


@pytest.mark.parametrize(
    "hit",
    [
        CorpusHit("bad-s2", "Article", "Evidence.", "test", metadata={"semantic_scholar_id": "unknown"}),
        CorpusHit("bare", "Article", "Evidence.", "test", url="https://example.org"),
        CorpusHit("private", "Article", "Evidence.", "test", url="https://127.0.0.1/paper/1"),
        CorpusHit("search", "Article", "Evidence.", "test", url="https://example.org/search?q=paper"),
        CorpusHit("query", "Article", "Evidence.", "test", url="https://example.org/articles?query=paper"),
    ],
)
def test_unverifiable_source_locators_are_not_stable_identities(hit: CorpusHit) -> None:
    assert stable_source_identity(hit) is None


@pytest.mark.parametrize("document_type", ["editorial", "letter", "book-chapter", "peer-review", "banana"])
def test_arbitrary_source_type_cannot_receive_verified_article_status(
    document_type: str,
) -> None:
    hit = CorpusHit(
        "typed",
        "Ordinary source",
        "Evidence text.",
        "test",
        doi="10.1234/typed",
        metadata={"document_type": document_type},
    )

    assert has_verified_primary_article_type(hit) is False


def test_corrected_article_is_flagged_but_not_mistaken_for_correction_notice() -> None:
    hit = CorpusHit(
        hit_id="corrected",
        title="Ordinary randomized trial",
        abstract="Evidence text.",
        source="pubmed",
        doi="10.1234/corrected",
        metadata={
            "document_type": "Journal Article",
            "correction_status": "ErratumIn",
        },
    )

    assert source_integrity_issue(hit) is None


def test_publication_quality_requires_exact_receipt_bound_claim_traces() -> None:
    result = _quality_result()

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    verdict = cast(dict[str, object], assessment["publish_verdict"])
    ledger = cast(list[dict[str, object]], assessment["claim_evidence_ledger"])

    assert quality_blocker(assessment) is None
    assert verdict["decision"] == "ready_to_publish"
    assert verdict["publish_tier"] == "TIER_1"
    assert verdict["maturity_level"] == "L4"
    assert len(ledger) == 2
    assert all(len(cast(str, item["source_text_sha256"])) == 64 for item in ledger)
    assert all(cast(dict[str, int], item["evidence_span"])["end"] > 0 for item in ledger)
    risks = cast(list[dict[str, object]], assessment["risk_of_bias"])
    assert len(risks) == 2
    assert all(risk["tool"] == "V5 abstract evidence-risk screen" for risk in risks)
    assert all(risk["overall"] == "not_assessed" for risk in risks)


def test_unknown_source_status_publishes_only_as_bounded_tier_two() -> None:
    result = _quality_result()
    result = replace(
        result,
        receipts=tuple(replace(hit, metadata={}) for hit in result.receipts),
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    verdict = cast(dict[str, object], assessment["publish_verdict"])

    assert quality_blocker(assessment) is None
    assert verdict["decision"] == "ready_to_publish"
    assert verdict["publish_tier"] == "TIER_2"
    assert verdict["confidence_label"] == "bounded_evidence_brief"


def test_url_only_identity_cannot_receive_tier_one_certification() -> None:
    result = _quality_result()
    receipts = tuple(
        replace(hit, doi=None, url=f"https://example.org/articles/{index}")
        for index, hit in enumerate(result.receipts, start=1)
    )
    cards = tuple(
        replace(card, receipt_id=hit.receipt_id)
        for card, hit in zip(result.candidate.claim_cards, receipts, strict=True)
    )
    result = replace(
        result,
        candidate=replace(
            result.candidate,
            receipt_ids=tuple(hit.receipt_id for hit in receipts),
            claim_cards=cards,
        ),
        receipts=receipts,
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    verdict = cast(dict[str, object], assessment["publish_verdict"])

    assert quality_blocker(assessment) is None
    assert verdict["publish_tier"] == "TIER_2"


def test_mined_claim_cards_pass_the_same_exact_trace_gate() -> None:
    receipts = (
        CorpusHit(
            "a",
            "Randomized human trial finds intervention improves endurance performance",
            "In a randomized placebo-controlled trial in adults, the intervention improved endurance performance and exercise capacity.",
            "fullraw:pubmed",
            doi="10.1234/a",
        ),
        CorpusHit(
            "b",
            "Randomized human trial finds intervention reduces endurance performance",
            "In a randomized placebo-controlled trial in adults, the intervention reduced endurance performance and exercise capacity.",
            "fullraw:pubmed",
            doi="10.1234/b",
        ),
    )
    candidate = mine_insights(receipts, topic="intervention endurance performance")[0]
    result = MemoResult(candidate, receipts, render_memo(candidate, receipts))

    assessment = assess_publication_quality(result, public_markdown=result.markdown)

    assert publication_quality_blocker(result) is None
    assert len(cast(list[object], assessment["claim_evidence_ledger"])) == 2


def test_pipeline_tries_next_candidate_when_selected_memo_fails_final_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _quality_result()
    first = replace(
        first,
        receipts=tuple(replace(hit, hit_id=hit.receipt_id) for hit in first.receipts),
    )
    second_receipts = tuple(
        replace(
            hit,
            hit_id=f"10.1234/fallback-{index}",
            title=hit.title.replace("trial A", f"fallback {index}").replace("trial B", f"fallback {index}"),
            doi=f"10.1234/fallback-{index}",
        )
        for index, hit in enumerate(first.receipts, start=1)
    )
    second_cards = tuple(
        replace(
            card,
            receipt_id=hit.receipt_id,
            quote=f"{hit.title}. {hit.abstract}",
        )
        for card, hit in zip(first.candidate.claim_cards, second_receipts, strict=True)
    )
    second_candidate = replace(
        first.candidate,
        receipt_ids=tuple(hit.receipt_id for hit in second_receipts),
        claim_cards=second_cards,
        thesis="Fallback candidate with exact receipt traces.",
    )
    all_hits = (*first.receipts, *second_receipts)

    class Searcher:
        def search(self, _query: str, *, limit: int = 25) -> tuple[CorpusHit, ...]:
            return tuple(all_hits[:limit])

    monkeypatch.setattr(
        "v5_memo.pipeline.mine_insights",
        lambda *_args, **_kwargs: [first.candidate, second_candidate],
    )

    def writer(candidate: InsightCandidate, receipts: Sequence[CorpusHit]) -> str:
        if candidate.receipt_ids == first.candidate.receipt_ids:
            return "# Alpha memo: selected but untraceable\n\nThe strength effect size was 9.99."
        return render_memo(candidate, receipts)

    result = build_alpha_memo(
        topic=first.candidate.topic,
        seed_queries=("intervention outcome",),
        anchor_queries=(),
        searcher=Searcher(),
        memo_writer=writer,
        memo_selector=lambda candidates, _hits: candidates[:1],
        require_publish_quality=True,
    )

    assert result.candidate.receipt_ids == second_candidate.receipt_ids


def test_pipeline_stops_after_first_fully_quality_approved_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = _quality_result()
    receipts = tuple(
        replace(hit, hit_id=hit.receipt_id)
        for hit in quality.receipts
    )
    calls: list[str] = []

    class Searcher:
        def search(self, query: str, *, limit: int = 25) -> tuple[CorpusHit, ...]:
            del limit
            calls.append(query)
            if query == "unneeded later shape":
                raise AssertionError("later shape should not run after the final gate passes")
            return receipts

    monkeypatch.setattr(
        "v5_memo.pipeline.mine_insights",
        lambda *_args, **_kwargs: [quality.candidate],
    )

    result = build_alpha_memo(
        topic=quality.candidate.topic,
        seed_queries=("intervention outcome", "unneeded later shape"),
        anchor_queries=(),
        searcher=Searcher(),
        require_publish_quality=True,
    )

    assert result.candidate.receipt_ids == quality.candidate.receipt_ids
    assert calls == ["intervention outcome"]


def test_pipeline_waits_for_safe_supports_before_early_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quality = _quality_result()
    core = tuple(
        replace(
            hit,
            hit_id=hit.receipt_id,
            metadata=_strict_retrieval_metadata(),
        )
        for hit in quality.receipts
    )
    supports = tuple(_strict_support(index) for index in range(1, 4))
    calls: list[str] = []

    class Searcher:
        def search(self, query: str, *, limit: int = 25) -> tuple[CorpusHit, ...]:
            del limit
            calls.append(query)
            if query == "core only":
                return core
            if query == "supporting context":
                return supports
            raise AssertionError("pipeline should stop after the citation floor passes")

    monkeypatch.setattr(
        "v5_memo.pipeline.mine_insights",
        lambda *_args, **_kwargs: [quality.candidate],
    )

    result = build_alpha_memo(
        topic=quality.candidate.topic,
        seed_queries=("core only", "supporting context", "unneeded later shape"),
        anchor_queries=(),
        searcher=Searcher(),
        require_publish_quality=True,
        min_shards_searched=1525,
        min_sources_searched=5,
        min_search_passes=1,
    )

    assert [hit.receipt_id for hit in result.supporting_receipts] == [
        "10.1234/support-3",
        "10.1234/support-2",
        "10.1234/support-1",
    ]
    assert calls == ["core only", "supporting context"]


def test_publication_quality_blocks_cross_receipt_quantitative_claim() -> None:
    result = _quality_result()
    receipts = (
        replace(result.receipts[0], abstract="The intervention result was p=0.009 in adults."),
        replace(result.receipts[1], abstract="The standardized effect size was 0.42 in adults."),
    )
    result = MemoResult(
        candidate=result.candidate,
        receipts=receipts,
        markdown="The result was p=0.009 and effect size 0.42.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)

    verdict = cast(dict[str, object], assessment["publish_verdict"])
    assert "quantitative_claim_untraced" in cast(list[str], verdict["blockers"])


def test_quantitative_trace_requires_endpoint_binding_not_numeric_coincidence() -> None:
    result = _quality_result()
    first = replace(result.receipts[0], abstract="The baseline index was 0.42 in adults.")
    first_card = replace(
        result.candidate.claim_cards[0],
        outcome="baseline index",
        quote=f"{first.title}. {first.abstract}",
    )
    result = replace(
        result,
        candidate=replace(
            result.candidate,
            claim_cards=(first_card, result.candidate.claim_cards[1]),
        ),
        receipts=(first, result.receipts[1]),
        markdown="The intervention effect size was 0.42.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    verdict = cast(dict[str, object], assessment["publish_verdict"])

    assert "quantitative_claim_untraced" in cast(list[str], verdict["blockers"])


def test_quantitative_trace_rejects_baseline_value_as_effect_estimate() -> None:
    result = _quality_result()
    first = replace(
        result.receipts[0],
        abstract="Baseline muscle strength was 0.42 before intervention.",
    )
    first_card = replace(
        result.candidate.claim_cards[0],
        outcome="muscle strength",
        quote=f"{first.title}. {first.abstract}",
    )
    result = replace(
        result,
        candidate=replace(
            result.candidate,
            claim_cards=(first_card, result.candidate.claim_cards[1]),
        ),
        receipts=(first, result.receipts[1]),
        markdown="The muscle strength effect size was 0.42.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    verdict = cast(dict[str, object], assessment["publish_verdict"])

    assert "quantitative_claim_untraced" in cast(list[str], verdict["blockers"])


@pytest.mark.parametrize(
    "abstract",
    [
        "Baseline muscle strength was 0.42 before intervention, and effect size estimates were planned for follow-up.",
        "Muscle strength improved. The baseline index effect size was 0.42.",
        "The muscle strength effect size was reported. the measured value was 0.42.",
        "The baseline muscle strength effect size was 0.42 before intervention.",
        "Baseline muscle strength was 0.42 (effect size estimates were planned for follow-up).",
        "Effect size estimates were planned: baseline muscle strength was 0.42.",
    ],
)
def test_quantitative_trace_requires_same_endpoint_construct_and_value_clause(
    abstract: str,
) -> None:
    result = _quality_result()
    first = replace(result.receipts[0], abstract=abstract)
    first_card = replace(
        result.candidate.claim_cards[0],
        outcome="muscle strength",
        quote=f"{first.title}. {first.abstract}",
    )
    result = replace(
        result,
        candidate=replace(
            result.candidate,
            claim_cards=(first_card, result.candidate.claim_cards[1]),
        ),
        receipts=(first, result.receipts[1]),
        markdown="The muscle strength effect size was 0.42.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    verdict = cast(dict[str, object], assessment["publish_verdict"])

    assert "quantitative_claim_untraced" in cast(list[str], verdict["blockers"])


def test_each_distinct_untraced_quantitative_claim_is_reported() -> None:
    result = replace(
        _quality_result(),
        markdown=(
            "The muscle strength effect size was 0.42. "
            "The endurance effect size was 0.73."
        ),
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    blockers = cast(list[dict[str, object]], assessment["quality_blockers"])
    untraced = [
        blocker for blocker in blockers if blocker["error"] == "quantitative_claim_untraced"
    ]
    assert [blocker["claim"] for blocker in untraced] == [
        "The muscle strength effect size was 0.42.",
        "The endurance effect size was 0.73.",
    ]


def test_quantitative_trace_preserves_construct_to_value_pairing() -> None:
    result = _quality_result()
    first = replace(
        result.receipts[0],
        abstract="The measured endpoint had p=0.42 and effect size=0.009.",
    )
    first_card = replace(
        result.candidate.claim_cards[0],
        outcome="measured endpoint",
        quote=f"{first.title}. {first.abstract}",
    )
    result = replace(
        result,
        candidate=replace(
            result.candidate,
            claim_cards=(first_card, result.candidate.claim_cards[1]),
        ),
        receipts=(first, result.receipts[1]),
        markdown="The measured endpoint had p=.009 and effect size=.42.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    verdict = cast(dict[str, object], assessment["publish_verdict"])

    assert "quantitative_claim_untraced" in cast(list[str], verdict["blockers"])

def test_quantitative_trace_binds_number_endpoint_and_receipt() -> None:
    result = _quality_result()
    first = replace(result.receipts[0], abstract="The strength effect size was 0.42 in adults.")
    first_card = replace(
        result.candidate.claim_cards[0],
        outcome="strength",
        quote=f"{first.title}. {first.abstract}",
    )
    result = replace(
        result,
        candidate=replace(
            result.candidate,
            claim_cards=(first_card, result.candidate.claim_cards[1]),
        ),
        receipts=(first, result.receipts[1]),
        markdown="The strength effect size was 0.42.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    traces = cast(list[dict[str, object]], assessment["quantitative_claim_traces"])

    assert quality_blocker(assessment) is None
    assert traces[0]["receipt_id"] == "10.1234/trial-a"
    assert traces[0]["endpoint"] == "strength"


@pytest.mark.parametrize(
    "abstract",
    [
        "The strength effect size: 0.42 in adults.",
        "The planned analysis found a strength effect size of 0.42 in adults.",
        "The per-protocol analysis found a strength effect size of 0.42 in adults.",
        "The strength effect size in the prespecified primary analysis was 0.42.",
    ],
)
def test_valid_result_wording_does_not_create_quantitative_false_block(
    abstract: str,
) -> None:
    result = _quality_result()
    first = replace(result.receipts[0], abstract=abstract)
    first_card = replace(
        result.candidate.claim_cards[0],
        outcome="strength",
        quote=f"{first.title}. {first.abstract}",
    )
    result = replace(
        result,
        candidate=replace(
            result.candidate,
            claim_cards=(first_card, result.candidate.claim_cards[1]),
        ),
        receipts=(first, result.receipts[1]),
        markdown="The strength effect size was 0.42.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)

    assert "quantitative_claim_untraced" not in cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )


@pytest.mark.parametrize(
    "source_spelling",
    [
        "p-value = 0.50",
        "p value = 0.50",
        "p\u2011value = 0.50",
        "p \u2264 0.50",
    ],
)
def test_p_value_spelling_variants_trace_the_same_construct(source_spelling: str) -> None:
    result = _quality_result()
    first = replace(result.receipts[0], abstract=f"The strength {source_spelling} in adults.")
    first_card = replace(
        result.candidate.claim_cards[0],
        outcome="strength",
        quote=f"{first.title}. {first.abstract}",
    )
    result = replace(
        result,
        candidate=replace(
            result.candidate,
            claim_cards=(first_card, result.candidate.claim_cards[1]),
        ),
        receipts=(first, result.receipts[1]),
        markdown="The strength result was p=.50.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)

    assert "quantitative_claim_untraced" not in cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )


@pytest.mark.parametrize("status", ["not reported", "unavailable"])
def test_unreported_construct_cannot_bind_later_colon_value(status: str) -> None:
    result = _quality_result()
    first = replace(
        result.receipts[0],
        abstract=f"The effect size was {status}: muscle strength was 0.42.",
    )
    first_card = replace(
        result.candidate.claim_cards[0],
        outcome="muscle strength",
        quote=f"{first.title}. {first.abstract}",
    )
    result = replace(
        result,
        candidate=replace(
            result.candidate,
            claim_cards=(first_card, result.candidate.claim_cards[1]),
        ),
        receipts=(first, result.receipts[1]),
        markdown="The muscle strength effect size was 0.42.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)

    assert "quantitative_claim_untraced" in cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )


@pytest.mark.parametrize(
    "abstract",
    [
        "No effect size was reported: muscle strength was 0.42.",
        "No reported effect size: muscle strength was 0.42.",
        "The study did not report an effect size: muscle strength was 0.42.",
        "The effect size was absent: muscle strength was 0.42.",
    ],
)
def test_prefix_negated_construct_cannot_bind_later_value(abstract: str) -> None:
    result = _quality_result()
    first = replace(result.receipts[0], abstract=abstract)
    first_card = replace(
        result.candidate.claim_cards[0],
        outcome="muscle strength",
        quote=f"{first.title}. {first.abstract}",
    )
    result = replace(
        result,
        candidate=replace(
            result.candidate,
            claim_cards=(first_card, result.candidate.claim_cards[1]),
        ),
        receipts=(first, result.receipts[1]),
        markdown="The muscle strength effect size was 0.42.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)

    assert "quantitative_claim_untraced" in cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )


def test_unicode_minus_cannot_trace_positive_effect() -> None:
    result = _quality_result()
    first = replace(
        result.receipts[0],
        abstract="The muscle strength effect size was \u22120.42.",
    )
    first_card = replace(
        result.candidate.claim_cards[0],
        outcome="muscle strength",
        quote=f"{first.title}. {first.abstract}",
    )
    result = replace(
        result,
        candidate=replace(
            result.candidate,
            claim_cards=(first_card, result.candidate.claim_cards[1]),
        ),
        receipts=(first, result.receipts[1]),
        markdown="The muscle strength effect size was 0.42.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)

    assert "quantitative_claim_untraced" in cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )


@pytest.mark.parametrize("construct", ["HR", "OR", "RR", "SMD", "CI"])
def test_statistical_abbreviations_are_never_invisible_to_trace_gate(construct: str) -> None:
    result = replace(
        _quality_result(),
        markdown=f"The measured endpoint {construct}=2.0.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)

    assert "quantitative_claim_untraced" in cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )


def test_unreported_abbreviation_cannot_bind_later_value() -> None:
    result = _quality_result()
    first = replace(
        result.receipts[0],
        abstract="HR was not reported: mortality count was 72.",
    )
    first_card = replace(
        result.candidate.claim_cards[0],
        outcome="mortality",
        quote=f"{first.title}. {first.abstract}",
    )
    result = replace(
        result,
        candidate=replace(
            result.candidate,
            claim_cards=(first_card, result.candidate.claim_cards[1]),
        ),
        receipts=(first, result.receipts[1]),
        markdown="The mortality hazard ratio HR=72.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)

    assert "quantitative_claim_untraced" in cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )


def test_logical_or_is_not_classified_as_odds_ratio() -> None:
    result = _quality_result()
    first = replace(
        result.receipts[0],
        abstract="Participants received 1 OR 2 doses.",
    )
    first_card = replace(
        result.candidate.claim_cards[0],
        outcome="dose",
        quote=f"{first.title}. {first.abstract}",
    )
    result = replace(
        result,
        candidate=replace(
            result.candidate,
            claim_cards=(first_card, result.candidate.claim_cards[1]),
        ),
        receipts=(first, result.receipts[1]),
        markdown="Participants received 1 OR 2 doses.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)

    assert cast(list[dict[str, object]], assessment["quantitative_claim_traces"]) == []
    assert "quantitative_claim_untraced" not in cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )


def test_heart_rate_abbreviation_is_not_hazard_ratio_without_explicit_binding() -> None:
    result = _quality_result()
    first = replace(
        result.receipts[0],
        abstract="Heart rate HR=72 bpm.",
    )
    first_card = replace(
        result.candidate.claim_cards[0],
        outcome="heart rate",
        quote=f"{first.title}. {first.abstract}",
    )
    result = replace(
        result,
        candidate=replace(
            result.candidate,
            claim_cards=(first_card, result.candidate.claim_cards[1]),
        ),
        receipts=(first, result.receipts[1]),
        markdown="Heart rate HR=72 bpm.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)

    assert cast(list[dict[str, object]], assessment["quantitative_claim_traces"]) == []
    assert "quantitative_claim_untraced" not in cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )


@pytest.mark.parametrize(
    "statement",
    [
        "Pulse HR=72 beats/min.",
        "Respiration RR=16/min.",
    ],
)
def test_rate_units_disambiguate_clinical_abbreviations(statement: str) -> None:
    result = replace(_quality_result(), markdown=statement)

    assessment = assess_publication_quality(result, public_markdown=result.markdown)

    assert cast(list[dict[str, object]], assessment["quantitative_claim_traces"]) == []
    assert "quantitative_claim_untraced" not in cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )


def test_publication_quality_requires_threshold_explanation() -> None:
    result = _quality_result()
    result = MemoResult(
        candidate=result.candidate,
        receipts=result.receipts,
        markdown="The p=0.009 result was non-significant.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)

    verdict = cast(dict[str, object], assessment["publish_verdict"])
    assert "unexplained_statistical_threshold" in cast(list[str], verdict["blockers"])


def test_negated_threshold_explanation_does_not_bypass_statistical_gate() -> None:
    result = replace(
        _quality_result(),
        markdown=(
            "The p=.009 result was non-significant; "
            "no adjusted threshold was stated."
        ),
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    verdict = cast(dict[str, object], assessment["publish_verdict"])

    assert "unexplained_statistical_threshold" in cast(list[str], verdict["blockers"])


@pytest.mark.parametrize(
    "claim",
    [
        "The adjusted model gave p=.009 but the result was non-significant.",
        "The p=.009 result was non-significant; adjusted alpha was not reported.",
    ],
)
def test_adjustment_words_without_a_threshold_do_not_bypass_gate(claim: str) -> None:
    result = replace(_quality_result(), markdown=claim)

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    verdict = cast(dict[str, object], assessment["publish_verdict"])

    assert "unexplained_statistical_threshold" in cast(list[str], verdict["blockers"])


@pytest.mark.parametrize(
    "claim",
    [
        "The p=.009 result was non-significant; FDR was not reported.",
        "The p=.009 result was non-significant; no FDR was reported.",
        "The p=.009 result was non-significant; Bonferroni was not applied.",
        "The p=.009 result was non-significant; no Bonferroni threshold was stated.",
    ],
)
def test_negated_named_adjustments_do_not_bypass_threshold_gate(claim: str) -> None:
    result = replace(_quality_result(), markdown=claim)

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    verdict = cast(dict[str, object], assessment["publish_verdict"])

    assert "unexplained_statistical_threshold" in cast(list[str], verdict["blockers"])


def test_valid_adjustment_is_not_cancelled_by_unrelated_negation() -> None:
    result = replace(
        _quality_result(),
        markdown=(
            "The p=.009 result was non-significant after FDR adjustment; "
            "Bonferroni was not applied."
        ),
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    blockers = cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )

    assert "unexplained_statistical_threshold" not in blockers
    assert "contradictory_statistical_interpretation" not in blockers


@pytest.mark.parametrize(
    "claim",
    [
        "The p=.009 result was non-significant; FDR was unavailable.",
        "The p=.009 result was non-significant; FDR was not calculated.",
        "The p=.009 result was non-significant because FDR could not be calculated.",
        "The p=.009 result was non-significant because FDR could not be estimated.",
        "The p=.009 result was non-significant because FDR could not be determined.",
    ],
)
def test_unavailable_adjustment_does_not_explain_threshold(claim: str) -> None:
    result = replace(_quality_result(), markdown=claim)

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    blockers = cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )

    assert "unexplained_statistical_threshold" in blockers


@pytest.mark.parametrize(
    "claim",
    [
        "The p value = .009 result was not significant.",
        "The p\u2011value = .009 result did not reach statistical significance.",
        "The p=.009 result did not achieve statistical significance.",
        "The p=.009 result did not differ significantly.",
        "The p=.009 result was statistically insignificant.",
        "The p \u2264 .009 result showed no significant difference.",
        "The p<=.05 result failed to reach statistical significance.",
    ],
)
def test_equivalent_p_value_and_non_significant_forms_cannot_bypass_gate(
    claim: str,
) -> None:
    result = replace(_quality_result(), markdown=claim)

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    blockers = cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )

    assert "unexplained_statistical_threshold" in blockers


def test_all_p_values_in_clause_are_interpreted() -> None:
    result = replace(
        _quality_result(),
        markdown="The measured endpoint p=.50 and p=.009 were non-significant.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    blockers = cast(list[dict[str, object]], assessment["quality_blockers"])

    assert any(
        blocker["error"] == "unexplained_statistical_threshold"
        and blocker["p_value"] == 0.009
        for blocker in blockers
    )


def test_unicode_minus_exponent_is_interpreted_as_a_small_p_value() -> None:
    result = replace(
        _quality_result(),
        markdown="The measured endpoint p=1e\u22123 was non-significant.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    blockers = cast(list[dict[str, object]], assessment["quality_blockers"])

    assert any(
        blocker["error"] == "unexplained_statistical_threshold"
        and blocker["p_value"] == 0.001
        for blocker in blockers
    )


def test_non_significant_baseline_clause_does_not_relabel_result_p_value() -> None:
    result = replace(
        _quality_result(),
        markdown=(
            "The measured endpoint improved at p=.009; "
            "no significant baseline difference was observed."
        ),
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    blockers = cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )

    assert "unexplained_statistical_threshold" not in blockers
    assert "contradictory_statistical_interpretation" not in blockers


@pytest.mark.parametrize(
    "claim",
    [
        (
            "Endpoint A improved significantly (p=.009), while endpoint B "
            "was non-significant (p=.50)."
        ),
        "The endpoint improved at p=.009, but no significant baseline difference was observed.",
        (
            "The endpoint improved at p=.009, although baseline p=.50 "
            "was non-significant."
        ),
        (
            "The endpoint improved at p=.009 although baseline p=.50 "
            "was non-significant."
        ),
        "The endpoint improved at p=.009, and no significant baseline difference was observed.",
        (
            "The endpoint improved at p=.009 but baseline p=.50 "
            "was non-significant."
        ),
        (
            "The endpoint improved at p=.009 yet baseline p=.50 "
            "was non-significant."
        ),
    ],
)
def test_conjunction_clause_does_not_relabel_other_p_value(claim: str) -> None:
    result = replace(_quality_result(), markdown=claim)

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    blockers = cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )

    assert "unexplained_statistical_threshold" not in blockers
    assert "contradictory_statistical_interpretation" not in blockers


def test_explicit_threshold_cannot_explain_a_contradictory_label() -> None:
    result = replace(
        _quality_result(),
        markdown="The measured endpoint p=.009 result was non-significant at alpha=.05.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    blockers = cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )

    assert "contradictory_statistical_interpretation" in blockers


def test_stricter_explicit_threshold_can_explain_non_significant_label() -> None:
    result = replace(
        _quality_result(),
        markdown="The measured endpoint p=.009 result was non-significant at alpha=.001.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    blockers = cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )

    assert "contradictory_statistical_interpretation" not in blockers
    assert "unexplained_statistical_threshold" not in blockers


@pytest.mark.parametrize(
    "claim",
    [
        "Nominal alpha=.05 and adjusted alpha=.005, p=.009 was non-significant.",
        (
            "The p=.009 result was non-significant after Bonferroni correction "
            "at threshold=.001; nominal alpha=.05."
        ),
    ],
)
def test_adjusted_threshold_wins_over_nominal_threshold(claim: str) -> None:
    result = replace(_quality_result(), markdown=claim)

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    blockers = cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )

    assert "contradictory_statistical_interpretation" not in blockers
    assert "unexplained_statistical_threshold" not in blockers


def test_scientific_notation_threshold_is_parsed_numerically() -> None:
    result = replace(
        _quality_result(),
        markdown=(
            "The measured endpoint p=.009 result was non-significant at "
            "adjusted alpha=5e-3."
        ),
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    blockers = cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )

    assert "contradictory_statistical_interpretation" not in blockers
    assert "unexplained_statistical_threshold" not in blockers


def test_adjacent_threshold_clause_explains_preceding_result() -> None:
    result = replace(
        _quality_result(),
        markdown=(
            "The measured endpoint p=.009 result was non-significant; "
            "Bonferroni-adjusted alpha=.001."
        ),
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    blockers = cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )

    assert "contradictory_statistical_interpretation" not in blockers
    assert "unexplained_statistical_threshold" not in blockers


def test_natural_adjacent_threshold_clause_is_attached() -> None:
    result = replace(
        _quality_result(),
        markdown=(
            "The measured endpoint p=.009 result was non-significant; "
            "the adjusted threshold was set to .005."
        ),
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    blockers = cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )

    assert "contradictory_statistical_interpretation" not in blockers
    assert "unexplained_statistical_threshold" not in blockers


@pytest.mark.parametrize(
    "threshold_clause",
    [
        "the adjusted alpha level was set at .005",
        "the significance threshold was .005",
    ],
)
def test_natural_threshold_label_variants_attach(threshold_clause: str) -> None:
    result = replace(
        _quality_result(),
        markdown=(
            "The measured endpoint p=.009 result was non-significant; "
            f"{threshold_clause}."
        ),
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    blockers = cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )

    assert "contradictory_statistical_interpretation" not in blockers
    assert "unexplained_statistical_threshold" not in blockers


@pytest.mark.parametrize(
    "claim",
    [
        (
            "Endpoint A p=.009 was non-significant; "
            "for Endpoint B, Bonferroni-adjusted alpha=.001."
        ),
        "Endpoint A p=.009 was non-significant; Endpoint B used threshold=.005.",
    ],
)
def test_adjacent_threshold_from_another_endpoint_cannot_attach(claim: str) -> None:
    result = replace(_quality_result(), markdown=claim)

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    blockers = cast(
        list[str],
        cast(dict[str, object], assessment["publish_verdict"])["blockers"],
    )

    assert "unexplained_statistical_threshold" in blockers


@pytest.mark.parametrize(("p_value", "expected"), [(".009", 0.009), ("0.009", 0.009)])
def test_p_value_parser_handles_leading_dot_and_zero(
    p_value: str,
    expected: float,
) -> None:
    result = _quality_result()
    result = replace(
        result,
        markdown=f"The p={p_value} result was non-significant without a stated threshold.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    blockers = cast(list[dict[str, object]], assessment["quality_blockers"])
    threshold = next(
        blocker for blocker in blockers if blocker["error"] == "unexplained_statistical_threshold"
    )

    assert threshold["p_value"] == expected


def test_conventional_non_significant_p_value_needs_no_adjustment_explanation() -> None:
    result = replace(
        _quality_result(),
        markdown="The p=0.50 result was non-significant.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    verdict = cast(dict[str, object], assessment["publish_verdict"])

    assert "unexplained_statistical_threshold" not in cast(list[str], verdict["blockers"])


def test_explicit_statistical_threshold_avoids_interpretation_block() -> None:
    result = _quality_result()
    result = replace(
        result,
        markdown="The p=0.50 result was non-significant at the prespecified alpha=0.05 threshold.",
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    verdict = cast(dict[str, object], assessment["publish_verdict"])

    assert "unexplained_statistical_threshold" not in cast(list[str], verdict["blockers"])


def test_payload_serializes_quality_evidence_and_uses_same_blocker() -> None:
    result = _quality_result()

    payload = build_researka_payload(
        result,
        author_agent_id="v5-memo-agent",
        domain_slug="longevity_research",
    )
    evidence = cast(dict[str, object], payload["evidence_bundle"])

    assert publication_quality_blocker(result) is None
    assert cast(dict[str, object], evidence["publish_verdict"])["decision"] == "ready_to_publish"
    assert len(cast(list[object], evidence["claim_evidence_ledger"])) == 2
    assert len(cast(list[object], evidence["risk_of_bias"])) == 2
    assert "fullraw_retrieval_coverage" in evidence
    assert json.loads(json.dumps(payload))["author_agent_id"] == "v5-memo-agent"


def test_bounded_brief_uses_separate_safe_supports_for_five_source_floor() -> None:
    core = _quality_result(metadata={"risk_of_bias": "high"})
    core = replace(
        core,
        receipts=tuple(
            replace(hit, metadata={**hit.metadata, **_strict_retrieval_metadata()})
            for hit in core.receipts
        ),
    )
    before = assess_publication_quality(core, public_markdown=core.markdown)
    supports = tuple(_strict_support(index) for index in range(1, 4))

    blocker = submission_readiness_blocker(core)
    assert blocker is not None
    assert blocker["reason"] == "minimum_citations"

    result = replace(
        core,
        supporting_receipts=supports,
        supporting_min_shards_searched=1525,
        supporting_min_sources_searched=5,
        supporting_min_search_passes=1,
    )
    assert submission_readiness_blocker(result) is None
    payload = build_researka_payload(
        result,
        author_agent_id="v5-memo-agent",
        domain_slug="longevity_research",
    )
    bundle = cast(list[dict[str, object]], payload["source_bundle"])
    after = cast(dict[str, object], payload["evidence_bundle"])

    assert len(bundle) == 5
    assert [entry.get("directness") for entry in bundle] == [
        "direct_core",
        "direct_core",
        "supporting_non_core",
        "supporting_non_core",
        "supporting_non_core",
    ]
    assert cast(dict[str, object], before["publish_verdict"]) == cast(
        dict[str, object], after["publish_verdict"]
    )
    assert len(cast(list[object], after["claim_evidence_ledger"])) == 2
    assert len(cast(list[object], after["risk_of_bias"])) == 2
    assert cast(dict[str, object], payload["metadata"])["supporting_receipt_ids"] == [
        "10.1234/support-1",
        "10.1234/support-2",
        "10.1234/support-3",
    ]


def test_final_submission_gate_revalidates_supporting_sources() -> None:
    core = _quality_result(metadata={"risk_of_bias": "high"})
    core = replace(
        core,
        receipts=tuple(
            replace(hit, metadata={**hit.metadata, **_strict_retrieval_metadata()})
            for hit in core.receipts
        ),
    )
    unsafe = tuple(
        replace(
            _strict_support(index),
            title=f"Correction: unrelated source {index}",
        )
        for index in range(1, 4)
    )

    blocker = submission_readiness_blocker(
        replace(
            core,
            supporting_receipts=unsafe,
            supporting_min_shards_searched=1525,
            supporting_min_sources_searched=5,
            supporting_min_search_passes=1,
        )
    )

    assert blocker == {
        "error": "candidate_publish_blocker",
        "reason": "invalid_supporting_sources",
    }


def test_support_selector_rejects_padding_and_is_deterministic() -> None:
    core = _quality_result().receipts
    topic = "vitamin D muscle strength randomized trial"
    good = [_strict_support(index, topic=topic) for index in range(1, 4)]
    incomplete_receipt = {
        **cast(dict[str, object], good[0].metadata["shard_receipt"]),
        "shards_searched": 1524,
        "sweep_remaining_shards": 1,
    }
    tiny_receipt = {
        **cast(dict[str, object], good[0].metadata["shard_receipt"]),
        "shards_searched": 1,
        "shards_total": 1,
        "sweep_remaining_shards": 0,
    }
    missing_partial = {
        key: value
        for key, value in cast(
            dict[str, object], good[0].metadata["shard_receipt"]
        ).items()
        if key != "partial_shard_search"
    }
    invalid = [
        replace(
            good[0],
            hit_id="incomplete",
            doi="10.1234/incomplete",
            metadata={**good[0].metadata, "shard_receipt": incomplete_receipt},
        ),
        replace(
            good[0],
            hit_id="tiny-complete",
            doi="10.1234/tiny-complete",
            metadata={**good[0].metadata, "shard_receipt": tiny_receipt},
        ),
        replace(
            good[0],
            hit_id="missing-coverage-field",
            doi="10.1234/missing-coverage-field",
            metadata={**good[0].metadata, "shard_receipt": missing_partial},
        ),
        replace(
            good[0],
            hit_id="off-topic",
            doi="10.1234/off-topic",
            title="Vitamin B12 trial",
            abstract=good[0].abstract.replace("vitamin D", "vitamin B12"),
        ),
        replace(
            good[0],
            hit_id="misleading-title",
            doi="10.1234/misleading-title",
            abstract=(
                "A randomized placebo-controlled trial in human adults tested vitamin B12 "
                "for cognition. The intervention improved memory and attention with a "
                "prespecified protocol and complete primary endpoint reporting for all "
                "participants enrolled in the study."
            ),
        ),
        replace(
            good[0],
            hit_id="correction",
            doi="10.1234/correction",
            title="Correction: randomized intervention outcome trial",
        ),
        replace(
            good[0],
            hit_id="combination",
            doi="10.1234/combination",
            abstract=good[0].abstract + " Treatments were tested alone or in combination.",
        ),
        replace(
            good[0],
            hit_id="multi-arm-list",
            doi="10.1234/multi-arm-list",
            title=(
                "Vitamin D supplementation, omega-3 supplementation, or a home exercise "
                "program for muscle strength: randomized trial"
            ),
        ),
        replace(
            good[0],
            hit_id="combined-supplements",
            doi="10.1234/combined-supplements",
            title="Vitamin D and calcium supplementation improves muscle strength",
        ),
        replace(
            good[0],
            hit_id="combined-exercise",
            doi="10.1234/combined-exercise",
            title="Vitamin D supplementation and resistance exercise training improves muscle strength",
        ),
        replace(
            good[0],
            hit_id="mislinked-result",
            doi="10.1234/mislinked-result",
            abstract=(
                "A randomized placebo-controlled trial in human adults tested vitamin D. "
                "Muscle strength was measured as a secondary endpoint, but vitamin D did "
                "not improve bone density. The prespecified protocol reported complete "
                "primary endpoint results for all enrolled participants."
            ),
        ),
        replace(
            good[0],
            hit_id="preprint",
            doi="10.1101/2026.01.01.123456",
        ),
        replace(
            good[0],
            hit_id="url-fallback",
            doi="10.bad",
            url="https://example.org/articles/url-fallback",
        ),
        replace(
            good[0],
            hit_id="duplicate-source",
            doi="10.1234/duplicate-source",
            year=2010,
        ),
    ]

    selected = select_supporting_receipts(
        topic=topic,
        hits=(*reversed(good), *invalid),
        core_receipts=core,
        needed=3,
        min_shards_searched=1525,
        min_sources_searched=5,
        min_search_passes=1,
    )

    assert [hit.receipt_id for hit in selected] == [
        "10.1234/support-3",
        "10.1234/support-2",
        "10.1234/support-1",
    ]


@pytest.mark.parametrize(
    "hit",
    [
        replace(
            _strict_support(1, topic="vitamin D muscle strength randomized trial"),
            title="Vitamin D plus resistance exercise improves muscle strength",
        ),
        replace(
            _strict_support(1, topic="vitamin D muscle strength randomized trial"),
            title="Vitamin D and calcium for muscle strength in adults",
        ),
        replace(
            _strict_support(1, topic="vitamin D muscle strength randomized trial"),
            title=(
                "Vitamin D supplementation and resistance exercise training improves "
                "muscle strength"
            ),
        ),
        replace(
            _strict_support(1, topic="vitamin D muscle strength randomized trial"),
            title="Resistance training combined with vitamin D improves muscle strength",
        ),
        replace(
            _strict_support(1, topic="vitamin D muscle strength randomized trial"),
            title="Vitamin D together with resistance training improves muscle strength",
        ),
        replace(
            _strict_support(1, topic="vitamin D muscle strength randomized trial"),
            title="Vitamin D in combination with resistance training improves muscle strength",
        ),
        replace(
            _strict_support(1, topic="vitamin D muscle strength randomized trial"),
            abstract=(
                "A randomized placebo-controlled trial in human adults tested vitamin D. "
                "Muscle strength was measured as a secondary endpoint, but vitamin D did "
                "not improve bone density. The prespecified protocol reported complete "
                "primary endpoint results for all enrolled participants."
            ),
        ),
        replace(
            _strict_support(1, topic="vitamin D muscle strength randomized trial"),
            abstract=(
                "A randomized placebo-controlled trial in human adults tested vitamin D. "
                "Muscle strength was a secondary endpoint and vitamin D improved bone "
                "density. The prespecified protocol reported complete primary endpoint "
                "results for all enrolled participants."
            ),
        ),
        replace(
            _strict_support(1, topic="vitamin D muscle strength randomized trial"),
            abstract=(
                "A randomized placebo-controlled trial in human adults tested vitamin D. "
                "Muscle mass improved, strength was measured as a secondary endpoint. The "
                "prespecified protocol reported complete primary endpoint results for all "
                "enrolled participants."
            ),
        ),
        replace(
            _strict_support(1, topic="vitamin D muscle strength randomized trial"),
            abstract=(
                "A randomized placebo-controlled trial in human adults tested vitamin D. "
                "Muscle mass was assessed before bone strength improved. The prespecified "
                "protocol reported complete primary endpoint results for all enrolled "
                "participants."
            ),
        ),
        replace(
            _strict_support(1, topic="vitamin D muscle strength randomized trial"),
            abstract=(
                "A randomized placebo-controlled trial in human adults tested vitamin D. "
                "Bone density improved and muscle strength was measured as a secondary "
                "endpoint. The prespecified protocol reported complete primary endpoint "
                "results for all enrolled participants."
            ),
        ),
        replace(
            _strict_support(1, topic="vitamin D muscle strength randomized trial"),
            abstract=(
                "A randomized placebo-controlled trial in human adults tested vitamin D. "
                "Vitamin D improved bone strength in adults. The prespecified protocol "
                "reported complete primary endpoint results for all enrolled participants."
            ),
        ),
        replace(
            _strict_support(1, topic="vitamin D muscle strength randomized trial"),
            title="Randomized trial of muscle strength in adults",
            abstract=(
                "A randomized placebo-controlled trial in human adults tested vitamin D "
                "plus resistance exercise. Muscle strength improved after vitamin D plus "
                "resistance exercise. The prespecified protocol reported complete primary "
                "endpoint results for all enrolled participants."
            ),
        ),
        replace(
            _strict_support(1, topic="vitamin D muscle strength randomized trial"),
            title="Randomized trial of muscle strength in adults",
            abstract=(
                "A randomized trial in human adults investigated vitamin D. Participants "
                "received vitamin D plus resistance exercise training. Muscle strength "
                "improved after the combined program. The prespecified protocol reported "
                "complete primary endpoint results for all enrolled participants."
            ),
        ),
        replace(
            _strict_support(1, topic="vitamin D muscle strength randomized trial"),
            title="Randomized trial of muscle strength in adults",
            abstract=(
                "A randomized trial in human adults tested vitamin D. Participants received "
                "vitamin D plus calcium. Muscle strength improved after treatment. The "
                "prespecified protocol reported complete primary endpoint results for all "
                "enrolled participants."
            ),
        ),
        replace(
            _strict_support(1, topic="vitamin D muscle strength randomized trial"),
            abstract=(
                "A randomized placebo-controlled trial in human adults tested vitamin D. "
                "Muscle strength assessment accompanied improved bone density. The "
                "prespecified protocol reported complete primary endpoint results for all "
                "enrolled participants."
            ),
        ),
        replace(
            _strict_support(1, topic="vitamin D muscle strength randomized trial"),
            abstract=(
                "A randomized placebo-controlled trial in human adults tested vitamin D. "
                "Muscle strength measurement preceded improved bone density. The "
                "prespecified protocol reported complete primary endpoint results for all "
                "enrolled participants."
            ),
        ),
        replace(
            _strict_support(1, topic="vitamin D muscle strength randomized trial"),
            abstract=(
                "A randomized placebo-controlled trial in human adults tested vitamin D. "
                "Muscle strength was recorded before improved bone density. The prespecified "
                "protocol reported complete primary endpoint results for all enrolled "
                "participants."
            ),
        ),
        replace(
            _strict_support(1, topic="vitamin D muscle strength randomized trial"),
            abstract=(
                "A randomized placebo-controlled trial in human adults tested vitamin D. "
                "Bone density improved, and muscle strength was measured as a secondary "
                "endpoint. The prespecified protocol reported complete primary endpoint "
                "results for all enrolled participants."
            ),
        ),
        replace(
            _strict_support(1, topic="heat therapy muscle strength randomized trial"),
            abstract=(
                "A randomized placebo-controlled trial in human adults tested heat therapy. "
                "Muscle strength was measured as a secondary endpoint, while heat therapy "
                "improved skin perfusion. The prespecified protocol reported complete "
                "primary endpoint results for all enrolled participants."
            ),
        ),
    ],
)
def test_support_selector_rejects_competing_or_mislinked_results(hit: CorpusHit) -> None:
    topic = (
        "heat therapy muscle strength randomized trial"
        if "heat therapy" in hit.abstract.casefold()
        else "vitamin D muscle strength randomized trial"
    )

    assert select_supporting_receipts(
        topic=topic,
        hits=(hit,),
        core_receipts=_quality_result().receipts,
        needed=1,
        min_shards_searched=1525,
        min_sources_searched=5,
        min_search_passes=1,
    ) == ()


def test_support_selector_allows_single_intervention_context_with_phrase() -> None:
    topic = "vitamin D muscle strength randomized trial"
    hit = replace(
        _strict_support(1, topic=topic),
        title="Muscle strength in older adults with vitamin D supplementation",
    )

    selected = select_supporting_receipts(
        topic=topic,
        hits=(hit,),
        core_receipts=_quality_result().receipts,
        needed=1,
        min_shards_searched=1525,
        min_sources_searched=5,
        min_search_passes=1,
    )

    assert [receipt.receipt_id for receipt in selected] == ["10.1234/support-1"]


@pytest.mark.parametrize(
    "title",
    [
        "Vitamin D and muscle strength in older adults: a randomized trial",
        "Vitamin D and older adults: a randomized trial of muscle strength",
        "Vitamin D and postmenopausal women: a randomized trial of muscle strength",
    ],
)
def test_support_selector_allows_entity_joined_to_nonintervention_context(
    title: str,
) -> None:
    topic = "vitamin D muscle strength randomized trial"
    hit = replace(
        _strict_support(1, topic=topic),
        title=title,
    )

    selected = select_supporting_receipts(
        topic=topic,
        hits=(hit,),
        core_receipts=_quality_result().receipts,
        needed=1,
        min_shards_searched=1525,
        min_sources_searched=5,
        min_search_passes=1,
    )

    assert [receipt.receipt_id for receipt in selected] == ["10.1234/support-1"]


def test_support_selector_accepts_grip_strength_in_named_human_population() -> None:
    topic = "vitamin D muscle strength randomized trial"
    original = _strict_support(1, topic=topic)
    hit = replace(
        original,
        title=(
            "Can vitamin D improve grip strength in elderly nursing home residents? "
            "A randomized controlled trial"
        ),
        abstract=(
            "A randomized placebo-controlled trial tested vitamin D supplementation in "
            "elderly nursing home residents. Grip strength did not improve in the vitamin D "
            "group compared with the control group after one year. The prespecified protocol "
            "reported complete endpoint results for all enrolled residents."
        ),
    )

    selected = select_supporting_receipts(
        topic=topic,
        hits=(hit,),
        core_receipts=_quality_result().receipts,
        needed=1,
        min_shards_searched=1525,
        min_sources_searched=5,
        min_search_passes=1,
    )

    assert [receipt.receipt_id for receipt in selected] == ["10.1234/support-1"]


def test_support_selector_rejects_metadata_identified_preprint() -> None:
    topic = "vitamin D muscle strength randomized trial"
    original = _strict_support(1, topic=topic)
    preprint = replace(
        original,
        doi="10.21203/rs.3.rs-1234567/v1",
        metadata={
            **original.metadata,
            "document_type": "Preprint",
            "publication_types": ["Preprint"],
        },
    )

    assert select_supporting_receipts(
        topic=topic,
        hits=(preprint,),
        core_receipts=_quality_result().receipts,
        needed=1,
        min_shards_searched=1525,
        min_sources_searched=5,
        min_search_passes=1,
    ) == ()


def test_support_selector_rejects_verified_conference_abstract() -> None:
    topic = "vitamin D muscle strength randomized trial"
    original = _strict_support(1, topic=topic)
    conference_abstract = replace(
        original,
        metadata={
            **original.metadata,
            "document_type": "abstract",
            "publication_types": ["conference abstract"],
        },
    )

    assert select_supporting_receipts(
        topic=topic,
        hits=(conference_abstract,),
        core_receipts=_quality_result().receipts,
        needed=1,
        min_shards_searched=1525,
        min_sources_searched=5,
        min_search_passes=1,
    ) == ()


def test_support_selector_rejects_unverified_pmc_article_type() -> None:
    topic = "vitamin D muscle strength randomized trial"
    original = _strict_support(1, topic=topic)
    unresolved = replace(
        original,
        metadata={
            **original.metadata,
            "pmcid": "PMC9193606",
            "document_type": "article",
            "publication_types": ["article"],
            "source_type_verification": "unavailable",
        },
    )

    assert select_supporting_receipts(
        topic=topic,
        hits=(unresolved,),
        core_receipts=_quality_result().receipts,
        needed=1,
        min_shards_searched=1525,
        min_sources_searched=5,
        min_search_passes=1,
    ) == ()


def test_final_submission_gate_preserves_configured_strict_coverage_floor() -> None:
    tiny_metadata = _strict_retrieval_metadata()
    tiny_receipt = {
        **cast(dict[str, object], tiny_metadata["shard_receipt"]),
        "shards_searched": 10,
        "shards_total": 10,
    }
    tiny_metadata = {**tiny_metadata, "shard_receipt": tiny_receipt}
    core = _quality_result(metadata={"risk_of_bias": "high"})
    core = replace(
        core,
        receipts=tuple(
            replace(hit, metadata={**hit.metadata, **tiny_metadata})
            for hit in core.receipts
        ),
        supporting_receipts=tuple(
            replace(
                _strict_support(index),
                metadata={**_strict_support(index).metadata, **tiny_metadata},
            )
            for index in range(1, 4)
        ),
        supporting_min_shards_searched=1525,
        supporting_min_sources_searched=5,
        supporting_min_search_passes=1,
    )

    assert submission_readiness_blocker(core) == {
        "error": "candidate_publish_blocker",
        "reason": "invalid_supporting_sources",
    }


def test_support_selector_dedupes_after_ranking_independent_of_input_order() -> None:
    topic = "vitamin D muscle strength randomized trial"
    low = replace(
        _strict_support(1, topic=topic),
        doi="10.1234/low",
        year=2010,
        metadata={**_strict_support(1, topic=topic).metadata, "cited_by_count": 1},
    )
    high = replace(
        low,
        hit_id="high",
        doi="10.1234/high",
        year=2025,
        metadata={**low.metadata, "cited_by_count": 100},
    )

    forward = select_supporting_receipts(
        topic=topic,
        hits=(low, high),
        core_receipts=_quality_result().receipts,
        needed=1,
        min_shards_searched=1525,
        min_sources_searched=5,
        min_search_passes=1,
    )
    reverse = select_supporting_receipts(
        topic=topic,
        hits=(high, low),
        core_receipts=_quality_result().receipts,
        needed=1,
        min_shards_searched=1525,
        min_sources_searched=5,
        min_search_passes=1,
    )

    assert [hit.receipt_id for hit in forward] == ["10.1234/high"]
    assert [hit.receipt_id for hit in reverse] == ["10.1234/high"]


def test_one_high_risk_source_downgrades_instead_of_blocking_bounded_output() -> None:
    result = _quality_result(metadata={"risk_of_bias": "high"})

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    verdict = cast(dict[str, object], assessment["publish_verdict"])

    assert publication_quality_blocker(result) is None
    assert verdict["publish_tier"] == "TIER_2"
    assert verdict["maturity_level"] == "L4"
    risks = cast(list[dict[str, object]], assessment["risk_of_bias"])
    assert risks[0]["overall"] == "not_assessed"
    assert risks[0]["screening_flag"] == "high"
    assert risks[0]["provenance"] == "unverified provider metadata"


def test_all_high_risk_primary_sources_block_ready_state() -> None:
    result = _quality_result(metadata={"risk_of_bias": "high"})
    receipts = tuple(
        replace(hit, metadata={"risk_of_bias": "high"}) for hit in result.receipts
    )
    result = replace(result, receipts=receipts)

    blocker = publication_quality_blocker(result)

    assert blocker is not None
    assert blocker["error"] == "candidate_publish_blocker"
    assert blocker["reason"] == "all_primary_evidence_high_risk"


@pytest.mark.parametrize("risk_label", ["high risk", "High risk of bias", "critical risk"])
def test_unstructured_high_risk_phrases_remain_conservative_screens(
    risk_label: str,
) -> None:
    result = _quality_result(metadata={"risk_of_bias": risk_label})

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    risks = cast(list[dict[str, object]], assessment["risk_of_bias"])

    assert risks[0]["assessment_scope"] == "abstract_only"
    assert risks[0]["overall"] == "not_assessed"
    assert risks[0]["screening_flag"] in {"high", "critical"}
    assert risks[0]["provenance"] == "unverified provider metadata"


def test_incomplete_full_text_risk_metadata_cannot_count_as_formal() -> None:
    result = _quality_result()
    incomplete = {
        "assessment_scope": "full_text",
        "domains": {"randomization": "low"},
        "provenance": "provider field",
    }
    result = replace(
        result,
        receipts=tuple(
            replace(hit, metadata={**hit.metadata, "risk_of_bias": incomplete})
            for hit in result.receipts
        ),
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    verdict = cast(dict[str, object], assessment["publish_verdict"])
    risks = cast(list[dict[str, object]], assessment["risk_of_bias"])

    assert verdict["maturity_level"] == "L4"
    assert cast(dict[str, int], verdict["axes"])["formal_risk_of_bias_appraisals"] == 0
    assert all(risk["tool"] == "V5 abstract evidence-risk screen" for risk in risks)
    assert all(risk["overall"] == "not_assessed" for risk in risks)


def test_invalid_full_text_domain_values_cannot_count_as_formal() -> None:
    result = _quality_result()
    invalid = {
        "tool": "RoB 2",
        "assessment_scope": "full_text",
        "overall": "low",
        "domains": {f"domain_{index}": "provider says okay" for index in range(5)},
        "provenance": "full-text review receipt 2026-07-16",
    }
    result = replace(
        result,
        receipts=tuple(
            replace(hit, metadata={**hit.metadata, "risk_of_bias": invalid})
            for hit in result.receipts
        ),
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    verdict = cast(dict[str, object], assessment["publish_verdict"])

    assert verdict["maturity_level"] == "L4"
    assert cast(dict[str, int], verdict["axes"])["formal_risk_of_bias_appraisals"] == 0


def test_unrecognized_risk_tool_cannot_count_as_formal() -> None:
    result = _quality_result()
    invalid = {
        "tool": "Banana",
        "assessment_scope": "full_text",
        "overall": "low",
        "domains": {
            "randomization": "low",
            "deviations_from_intervention": "low",
            "missing_outcome_data": "low",
            "outcome_measurement": "low",
            "selective_reporting": "low",
        },
        "provenance": "full-text review receipt 2026-07-16",
    }
    result = replace(
        result,
        receipts=tuple(
            replace(hit, metadata={**hit.metadata, "risk_of_bias": invalid})
            for hit in result.receipts
        ),
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    verdict = cast(dict[str, object], assessment["publish_verdict"])

    assert verdict["maturity_level"] == "L4"
    assert cast(dict[str, int], verdict["axes"])["formal_risk_of_bias_appraisals"] == 0


def test_provenanced_full_text_risk_appraisals_can_reach_l5() -> None:
    result = _quality_result()
    appraisal = {
        "tool": "RoB 2",
        "assessment_scope": "full_text",
        "overall": "some_concerns",
        "domains": {
            "randomization": "low",
            "deviations_from_intervention": "low",
            "missing_outcome_data": "low",
            "outcome_measurement": "low",
            "selective_reporting": "some_concerns",
        },
        "provenance": "full-text review receipt 2026-07-16",
    }
    result = replace(
        result,
        receipts=tuple(
            replace(hit, metadata={**hit.metadata, "risk_of_bias": appraisal})
            for hit in result.receipts
        ),
    )

    assessment = assess_publication_quality(result, public_markdown=result.markdown)
    verdict = cast(dict[str, object], assessment["publish_verdict"])

    assert verdict["maturity_level"] == "L5"
    assert cast(dict[str, int], verdict["axes"])["formal_risk_of_bias_appraisals"] == 2
