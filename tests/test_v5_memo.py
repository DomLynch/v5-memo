from __future__ import annotations

from collections.abc import Sequence

import pytest

from v5_memo import (
    CorpusHit,
    bind_receipts,
    build_alpha_memo,
    collect_seed_hits,
    mine_insights,
    query_anchor_terms,
    render_memo,
)
from v5_memo.schemas import InsightCandidate
from v5_memo.scorer import score_connection


def _hits() -> list[CorpusHit]:
    return [
        CorpusHit(
            hit_id="h1",
            title="NAD salvage links sleep fragmentation to mitochondrial stress",
            abstract="Sleep fragmentation reduced resilience through NAD salvage and mitochondrial stress.",
            source="researka:semantic",
            year=2025,
            doi="10.1/sleep-nad",
            venue="Aging Cell",
        ),
        CorpusHit(
            hit_id="h2",
            title="NAD salvage predicts exercise response through mitochondrial repair",
            abstract="Exercise improved resilience when NAD salvage and mitochondrial repair markers moved together.",
            source="researka:discovery",
            year=2024,
            doi="10.2/exercise-nad",
            venue="Cell Metabolism",
        ),
        CorpusHit(
            hit_id="h3",
            title="Protein intake changes grip strength in older adults",
            abstract="A nutrition trial measured grip strength without NAD salvage markers.",
            source="researka:established",
            year=2022,
            doi="10.3/protein",
        ),
    ]


def _hit(hit_id: str, title: str, abstract: str) -> CorpusHit:
    return CorpusHit(hit_id=hit_id, title=title, abstract=abstract, source="openalex", doi=f"10.{hit_id}")


def test_mines_bridge_and_renders_receipt_bound_memo() -> None:
    hits = _hits()
    candidate = mine_insights(hits, topic="longevity resilience")[0]
    receipts = bind_receipts(candidate, hits)
    memo = render_memo(candidate, receipts)

    assert memo.startswith("# Alpha memo: mitochondrial / nad")
    assert "mitochondrial" in candidate.bridge_terms
    assert "nad" in candidate.bridge_terms
    assert candidate.score >= 60
    assert "shape:directional_reversal" in candidate.reasons
    assert "Alpha hypothesis" in memo
    assert "longevity resilience may be hiding" in memo
    assert "10.1/sleep-nad" in memo
    assert "10.2/exercise-nad" in memo
    assert "Safety note" in memo


def test_binder_rejects_single_source_candidates() -> None:
    hits = [
        CorpusHit(hit_id="a", title="Same paper A", abstract="shared bridge", source="researka", doi="10.same/x"),
        CorpusHit(hit_id="b", title="Same paper B", abstract="shared bridge", source="researka", doi="10.same/x"),
    ]
    candidate = InsightCandidate(
        topic="topic",
        thesis="topic may have a bridge.",
        bridge_terms=("bridge",),
        tension_terms=(),
        receipt_ids=("a", "b"),
        score=80,
        novelty_score=80,
        evidence_score=80,
        reasons=("source_diverse",),
    )

    assert bind_receipts(candidate, hits) == ()


def test_binder_rejects_duplicate_titles_with_different_ids() -> None:
    hits = [
        CorpusHit(
            hit_id="a",
            title="Review Bot: Automatic Code Review Tool",
            abstract="Developers accepted automated review comments.",
            source="openalex",
            doi="10.1109/icse.2013.6606642",
        ),
        CorpusHit(
            hit_id="b",
            title="Review Bot Automatic Code Review Tool",
            abstract="ACM index for the same automatic review comments paper.",
            source="acm",
            doi="10.5555/2486788.2486915",
        ),
    ]
    candidate = InsightCandidate(
        topic="AI review",
        thesis="AI review may have a trust bridge.",
        bridge_terms=("review", "comments"),
        tension_terms=("positive", "negative"),
        receipt_ids=("a", "b"),
        score=80,
        novelty_score=80,
        evidence_score=80,
        reasons=("source_diverse",),
    )

    assert bind_receipts(candidate, hits) == ()


def test_collect_seed_hits_dedupes_across_seed_queries() -> None:
    class FakeSearch:
        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del limit
            return [
                CorpusHit(
                    hit_id="shared",
                    title="Shared NAD salvage hit",
                    abstract=query,
                    source="researka",
                    doi="10.shared",
                ),
                CorpusHit(
                    hit_id=query,
                    title=f"Unique {query}",
                    abstract="mitochondrial bridge",
                    source="researka",
                    doi=f"10.{query}",
                ),
            ]

    hits = collect_seed_hits(FakeSearch(), ["nad", "mitochondrial"], per_query_limit=2)

    assert [hit.hit_id for hit in hits] == ["shared", "nad", "mitochondrial"]


def test_pipeline_builds_best_memo() -> None:
    class FakeSearch:
        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del query, limit
            return _hits()

    result = build_alpha_memo(
        topic="longevity resilience",
        seed_queries=["sleep nad", "exercise nad"],
        searcher=FakeSearch(),
    )

    assert result.candidate.score >= 60
    assert len(result.receipts) == 2
    assert result.markdown.startswith("# Alpha memo")


def test_pipeline_accepts_custom_memo_writer() -> None:
    class FakeSearch:
        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del query, limit
            return _hits()

    result = build_alpha_memo(
        topic="longevity resilience",
        seed_queries=["sleep nad", "exercise nad"],
        searcher=FakeSearch(),
        memo_writer=lambda candidate, receipts: f"custom: {candidate.topic} / {len(receipts)}",
    )

    assert result.markdown == "custom: longevity resilience / 2"


def test_pipeline_applies_selector_to_existing_candidates() -> None:
    hits = [
        _hit(
            "tail-a",
            "Tool cohort benefit in aggregate outcomes",
            "Tool reduced aggregate outcome risk in a cohort population.",
        ),
        _hit(
            "tail-b",
            "Tool fatality cases in acute sessions",
            "Rare acute tool death cases concentrated in case reports.",
        ),
        _hit(
            "metric-a",
            "Tool improves benchmark accuracy",
            "The tool improved benchmark accuracy score.",
        ),
        _hit(
            "metric-b",
            "Tool increases deployment error outcomes",
            "The tool increased error outcome rates.",
        ),
    ]

    class FakeSearch:
        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del query, limit
            return hits

    deterministic = mine_insights(
        hits,
        topic="AI tool reliability",
        required_anchor_terms=query_anchor_terms(["tool"]),
    )
    chosen = deterministic[-1].receipt_ids

    result = build_alpha_memo(
        topic="AI tool reliability",
        seed_queries=["tool"],
        searcher=FakeSearch(),
        memo_selector=lambda candidates, _hits: [candidates[-1]],
    )

    assert len(deterministic) >= 2
    assert result.candidate.receipt_ids == chosen


def test_pipeline_selector_cannot_invent_receipt_pair() -> None:
    class FakeSearch:
        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del query, limit
            return _hits()

    invented = InsightCandidate(
        topic="longevity resilience",
        thesis="Invented candidate.",
        bridge_terms=("nad",),
        tension_terms=("positive", "negative"),
        receipt_ids=("h1", "missing"),
        score=99,
        novelty_score=99,
        evidence_score=99,
        reasons=("source_diverse",),
    )

    with pytest.raises(ValueError, match="no receipt-bound"):
        build_alpha_memo(
            topic="longevity resilience",
            seed_queries=["sleep nad", "exercise nad"],
            searcher=FakeSearch(),
            memo_selector=lambda _candidates, _hits: [invented],
        )


def test_query_anchor_terms_keep_specific_seed_terms() -> None:
    assert query_anchor_terms([
        "NAD salvage mitochondrial stress exercise response",
        "mitochondrial stress exercise",
    ]) == ("nad", "salvage")


def test_miner_rejects_pairs_without_required_anchor_terms() -> None:
    hits = [
        CorpusHit(
            hit_id="a",
            title="Shear stress increases endothelial mitochondrial biogenesis",
            abstract="Exercise wall shear stress changed vascular mitochondria.",
            source="openalex",
            doi="10.a",
        ),
        CorpusHit(
            hit_id="b",
            title="Shear stress mediates endothelial training adaptation",
            abstract="Handgrip training adaptation depended on shear stress.",
            source="openalex",
            doi="10.b",
        ),
    ]

    assert mine_insights(hits, topic="NAD salvage exercise", required_anchor_terms=("nad", "salvage")) == []


def test_miner_rejects_adjacent_papers_without_alpha_shape() -> None:
    hits = [
        _hit("a", "Retrieval augmented generation evidence pipeline", "Local evidence pipeline reports results."),
        _hit("b", "Retrieval augmented generation evidence review", "Broad evidence review summarizes methods."),
    ]

    assert mine_insights(hits, topic="AI reliability") == []


def test_miner_accepts_non_reversal_alpha_shapes() -> None:
    hits = [
        _hit(
            "a",
            "Sauna bathing and incident hypertension in a prospective cohort",
            "Habitual sauna exposure reduced aggregate hypertension risk in a cohort population.",
        ),
        _hit(
            "b",
            "Sauna alcohol fatality cases in acute hypertension sessions",
            "Rare acute sauna death cases concentrated around alcohol and hypertension risk.",
        ),
    ]

    candidate = mine_insights(hits, topic="longevity sauna cardiovascular risk")[0]

    assert "shape:denominator_split" in candidate.reasons


def test_miner_ranks_shaped_candidates_above_rare_keyword_bridges() -> None:
    hits = [
        _hit("weak-a", "Zorblax workflow evidence summary", "Zorblax workflow evidence appears in one summary."),
        _hit("weak-b", "Zorblax workflow evidence methods", "Zorblax workflow evidence appears in methods."),
        _hit("strong-a", "Tool improves benchmark accuracy", "The tool improved benchmark accuracy score."),
        _hit("strong-b", "Tool increases deployment error outcomes", "The tool increased error outcome rates."),
    ]

    candidate = mine_insights(hits, topic="AI tool reliability")[0]

    assert candidate.receipt_ids == ("strong-a", "strong-b")
    assert "shape:measurement_mismatch" in candidate.reasons


def test_scorer_prefers_asymmetric_alpha_shapes_without_rejecting_seed_shapes() -> None:
    counts = {"bridge": 2}

    seed = score_connection(
        bridge_terms=("bridge",),
        bridge_doc_counts=counts,
        unique_source_count=2,
        receipt_count=2,
        has_tension=False,
        shape_score=1,
        shape_reasons=("shape:measurement_mismatch",),
    )
    alpha = score_connection(
        bridge_terms=("bridge",),
        bridge_doc_counts=counts,
        unique_source_count=2,
        receipt_count=2,
        has_tension=True,
        shape_score=2,
        shape_reasons=("shape:expectation_reversal", "shape:directional_reversal"),
    )

    assert seed.score > 0
    assert alpha.score > seed.score


def test_miner_ranks_expectation_reversal_above_construct_split() -> None:
    hits = [
        _hit(
            "alpha-a",
            "Protocol expected metformin training augmentation",
            "Protocol hypothesis expected metformin would improve strength training response.",
        ),
        _hit(
            "alpha-b",
            "Trial observed metformin training blunting",
            "Outcome trial observed metformin reduced strength training response.",
        ),
        _hit(
            "seed-a",
            "Protein timing chronic hypertrophy endpoint",
            "Protein timing chronic endpoint studied strength hypertrophy outcomes.",
        ),
        _hit(
            "seed-b",
            "Protein distribution acute synthesis endpoint",
            "Protein timing acute endpoint measured muscle protein synthesis outcomes.",
        ),
    ]

    candidate = mine_insights(hits, topic="training adaptation")[0]

    assert candidate.receipt_ids == ("alpha-a", "alpha-b")
    assert "shape:expectation_reversal" in candidate.reasons


def test_pipeline_raises_when_no_receipt_bound_candidate() -> None:
    class EmptySearch:
        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del query, limit
            return []

    with pytest.raises(ValueError, match="no receipt-bound"):
        build_alpha_memo(topic="topic", seed_queries=["x"], searcher=EmptySearch())
