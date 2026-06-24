from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from v5_memo import (
    CorpusHit,
    MemoBuildError,
    bind_receipts,
    build_alpha_memo,
    candidate_alpha_tier,
    collect_seed_hits,
    meets_publish_bar,
    mine_insights,
    query_anchor_terms,
    render_discovery_seed,
    render_memo,
)
from v5_memo.schemas import InsightCandidate
from v5_memo.scorer import score_connection

_FIXTURES = Path(__file__).with_name("fixtures")


def _golden_cases() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (_FIXTURES / "golden_alpha_cases.jsonl").read_text().splitlines()
        if line.strip()
    ]


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
    assert hits[0].metadata["seed_queries"] == ("nad", "mitochondrial")


def test_collect_seed_hits_balances_planned_query_budget() -> None:
    class FakeSearch:
        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            return [
                CorpusHit(
                    hit_id=f"{query}-{index}",
                    title=f"{query} hit {index}",
                    abstract="receipt",
                    source="fullraw",
                    doi=f"10.{query}/{index}",
                )
                for index in range(limit)
            ]

    hits = collect_seed_hits(
        FakeSearch(),
        ["noisy-first-query", "later-reversal-query"],
        per_query_limit=10,
        max_hits=6,
    )

    assert any(hit.hit_id.startswith("later-reversal-query") for hit in hits)


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


def test_pipeline_anchors_cover_late_planned_query_angles() -> None:
    class FakeSearch:
        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del limit
            if "resveratrol" not in query:
                return []
            return [
                CorpusHit(
                    hit_id="promise",
                    title="Resveratrol improves mitochondrial function and exercise performance",
                    abstract="Resveratrol activated a mitochondrial mechanism and improved running performance.",
                    source="openalex",
                    doi="10.promise",
                ),
                CorpusHit(
                    hit_id="outcome",
                    title="Resveratrol exercise training trial blunted cardiovascular adaptation",
                    abstract="Randomized trial observed resveratrol blunted exercise training adaptation.",
                    source="semantic_scholar",
                    doi="10.outcome",
                ),
            ]

    result = build_alpha_memo(
        topic="longevity exercise adaptation supplement reversal",
        seed_queries=[
            "rapamycin mTOR exercise adaptation blunted",
            "NMN NAD skeletal muscle attenuation",
            "resveratrol exercise training adaptation blunted",
        ],
        searcher=FakeSearch(),
        min_alpha_tier="elite_alpha",
    )

    assert result.candidate.receipt_ids == ("promise", "outcome")


def test_pipeline_anchors_to_original_seed_before_planner_drift() -> None:
    hits = [
        _hit(
            "plant-promise",
            "Arabidopsis TOR regulates cotyledon greening",
            "TOR signaling supports cotyledon greening in Arabidopsis.",
        ),
        _hit(
            "plant-outcome",
            "Arabidopsis BIN2 suppresses cotyledon greening",
            "BIN2 altered TOR cotyledon greening outcomes in Arabidopsis.",
        ),
        _hit(
            "promise",
            "Resveratrol mimics exercise mitochondrial biology",
            "Mechanism paper reported resveratrol improved mitochondrial function.",
        ),
        _hit(
            "outcome",
            "Resveratrol blunts exercise training adaptation",
            "Human outcome trial observed resveratrol reduced exercise training benefits.",
        ),
    ]

    class FakeSearch:
        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del query, limit
            return hits

    result = build_alpha_memo(
        topic="resveratrol exercise adaptation",
        seed_queries=["arabidopsis tor cotyledon greening"],
        anchor_queries=["resveratrol exercise training adaptation"],
        searcher=FakeSearch(),
    )

    assert result.candidate.receipt_ids == ("promise", "outcome")
    assert "shape:promise_outcome_reversal" in result.candidate.reasons


def test_pipeline_does_not_fallback_to_planner_anchors_when_seed_is_broad() -> None:
    hits = [
        _hit(
            "promise",
            "Resveratrol mimics exercise mitochondrial biology",
            "Mechanism paper reported resveratrol improved mitochondrial function.",
        ),
        _hit(
            "outcome",
            "Resveratrol blunts exercise training adaptation",
            "Human outcome trial observed resveratrol reduced exercise training benefits.",
        ),
    ]

    class FakeSearch:
        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del query, limit
            return hits

    result = build_alpha_memo(
        topic="longevity exercise adaptation supplement reversal",
        seed_queries=["arabidopsis tor cotyledon greening"],
        anchor_queries=["longevity exercise adaptation intervention reversal"],
        searcher=FakeSearch(),
    )

    assert result.candidate.receipt_ids == ("promise", "outcome")


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
            "Tool safety cohort benefit in aggregate outcomes",
            "Tool safety reduced aggregate outcome risk in a cohort population.",
        ),
        _hit(
            "tail-b",
            "Tool safety fatality cases in acute sessions",
            "Rare acute tool safety death cases concentrated in case reports.",
        ),
        _hit(
            "metric-a",
            "Tool improves benchmark accuracy",
            "The tool improved benchmark accuracy score.",
        ),
        _hit(
            "metric-b",
            "Tool reduces deployment benchmark reliability outcomes",
            "The tool reduced deployment benchmark reliability and worsened error outcome rates.",
        ),
    ]

    class FakeSearch:
        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del query, limit
            return hits

    deterministic = mine_insights(
        hits,
        topic="tool",
        required_anchor_terms=query_anchor_terms(["tool"]),
    )
    chosen = next(
        candidate
        for candidate in reversed(deterministic)
        if meets_publish_bar(candidate, "publishable_alpha")
    ).receipt_ids

    result = build_alpha_memo(
        topic="tool",
        seed_queries=["tool"],
        searcher=FakeSearch(),
        memo_selector=lambda candidates, _hits: [
            next(
                candidate
                for candidate in reversed(candidates)
                if meets_publish_bar(candidate, "publishable_alpha")
            )
        ],
    )

    assert len(deterministic) >= 2
    assert result.candidate.receipt_ids == chosen


def test_pipeline_filters_to_requested_tier_before_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    hits = [
        _hit("low-a", "Tool improves benchmark accuracy", "Tool improved benchmark accuracy."),
        _hit("low-b", "Tool reduces deployment reliability", "Tool reduced reliability."),
        _hit("elite-a", "Protocol expected tool augmentation", "Protocol expected tool augmentation."),
        _hit("elite-b", "Outcome observed tool blunting", "Outcome observed tool blunting."),
    ]
    low = InsightCandidate(
        topic="tool reliability",
        thesis="Low candidate.",
        bridge_terms=("tool",),
        tension_terms=("positive", "negative"),
        receipt_ids=("low-a", "low-b"),
        score=90,
        novelty_score=90,
        evidence_score=90,
        reasons=("tier:publishable_alpha",),
    )
    elite = InsightCandidate(
        topic="tool reliability",
        thesis="Elite candidate.",
        bridge_terms=("tool",),
        tension_terms=("positive", "negative"),
        receipt_ids=("elite-a", "elite-b"),
        score=90,
        novelty_score=90,
        evidence_score=90,
        reasons=("tier:elite_alpha",),
    )
    seen: list[InsightCandidate] = []

    class FakeSearch:
        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del query, limit
            return hits

    def fake_mine(*_args: object, **_kwargs: object) -> list[InsightCandidate]:
        return [low, elite]

    def selector(candidates: Sequence[InsightCandidate], _hits: Sequence[CorpusHit]) -> Sequence[InsightCandidate]:
        seen.extend(candidates)
        return list(candidates)

    monkeypatch.setattr("v5_memo.pipeline.mine_insights", fake_mine)

    result = build_alpha_memo(
        topic="tool reliability",
        seed_queries=["tool"],
        searcher=FakeSearch(),
        memo_selector=selector,
        min_alpha_tier="elite_alpha",
    )

    assert seen == [elite]
    assert result.candidate == elite


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
    ]) == ("nad", "salvage", "mitochondrial")


def test_query_anchor_terms_normalize_light_morphology() -> None:
    assert query_anchor_terms(["forecasts managers supplementation"], limit=3) == (
        "forecast",
        "manager",
        "supplement",
    )


def test_query_anchor_terms_drop_broad_topic_words() -> None:
    assert query_anchor_terms(["longevity aging adaptation healthspan pharmacology resveratrol training"]) == (
        "resveratrol",
    )


def test_query_anchor_terms_do_not_promote_context_to_anchor() -> None:
    assert query_anchor_terms(["resveratrol exercise training adaptation"]) == (
        "resveratrol",
    )


def test_query_anchor_terms_drop_alpha_shape_words() -> None:
    assert query_anchor_terms([
        "exercise adaptation intervention reversal",
        "protocol expected augment observed blunted training adaptation",
    ]) == ()


def test_query_anchor_terms_drop_broad_substrate_and_tissue_words() -> None:
    assert query_anchor_terms([
        "protein timing distribution resistance training muscle protein synthesis hypertrophy",
    ], limit=4) == ("timing", "distribution", "hypertrophy")


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


def test_miner_rejects_pairs_with_only_one_multiword_topic_anchor() -> None:
    hits = [
        _hit(
            "water-a",
            "Water balance improves training response",
            "Water balance improved outcomes.",
        ),
        _hit(
            "water-b",
            "Water status blunts muscle recovery",
            "Water status reduced recovery.",
        ),
    ]
    anchors = query_anchor_terms(["cold water immersion resistance training adaptation"])

    assert mine_insights(
        hits,
        topic="cold water immersion resistance training adaptation",
        required_anchor_terms=anchors,
    ) == []
    assert "water" not in anchors


def test_miner_rejects_asymmetric_anchor_pairs() -> None:
    hits = [
        _hit(
            "statin",
            "Molecular mechanisms of statin intolerance",
            "Patients and cases include rhabdomyolysis reports.",
        ),
        _hit(
            "exercise",
            "Exercise training adaptation in older adults",
            "Patients and cases include exercise response findings.",
        ),
    ]

    assert mine_insights(
        hits,
        topic="statin exercise adaptation",
        required_anchor_terms=("statin", "exercise"),
    ) == []


def test_miner_requires_cross_receipt_shape_ownership() -> None:
    hits = [
        _hit(
            "same-side",
            "Tool cohort fatality case report",
            "One receipt mentions cohort population and rare fatal death cases.",
        ),
        _hit(
            "other-side",
            "Tool safety summary",
            "Second receipt only repeats the tool safety bridge.",
        ),
    ]

    assert mine_insights(hits, topic="tool safety") == []


def test_anchor_only_bridge_requires_elite_shape() -> None:
    hits = [
        _hit(
            "statin-review",
            "Molecular mechanisms of statin intolerance",
            "Statin intolerance reduced tolerability in patient rhabdomyolysis cases.",
        ),
        _hit(
            "lipid-hiv",
            "Response to newly prescribed statin lipid-lowering therapy",
            "Statin lipid-lowering therapy improved LDL response in patients.",
        ),
    ]

    assert mine_insights(
        hits,
        topic="longevity statin exercise adaptation",
        required_anchor_terms=("statin",),
    ) == []


def test_anchor_only_bridge_can_support_elite_reversal() -> None:
    case = next(item for item in _golden_cases() if item["name"] == "resveratrol-real-snippet")
    hits = [
        _hit(str(hit["id"]), str(hit["title"]), str(hit["abstract"]))
        for hit in case["hits"]
    ]

    candidate = mine_insights(
        hits,
        topic=str(case["topic"]),
        required_anchor_terms=tuple(str(anchor) for anchor in case["anchors"]),
    )[0]

    assert candidate.bridge_terms == ("resveratrol",)
    assert "shape:promise_outcome_reversal" in candidate.reasons


def test_title_only_specific_anchor_can_support_elite_promise_outcome_pair() -> None:
    hits = [
        _hit(
            "promise",
            "Resveratrol Improves Mitochondrial Function and Protects against Metabolic Disease by Activating SIRT1 and PGC-1alpha",
            "",
        ),
        _hit(
            "outcome",
            "Resveratrol blunts the positive effects of exercise training on cardiovascular health in aged men",
            "",
        ),
        *[
            _hit(
                f"filler-{index}",
                f"Resveratrol exercise training context {index}",
                "Resveratrol exercise training context.",
            )
            for index in range(10)
        ],
    ]

    candidate = mine_insights(
        hits,
        topic="resveratrol exercise training adaptation reversal",
        required_anchor_terms=("resveratrol",),
    )[0]

    assert candidate.receipt_ids == ("promise", "outcome")
    assert candidate.bridge_terms == ("resveratrol",)
    assert "shape:promise_outcome_reversal" in candidate.reasons
    assert candidate_alpha_tier(candidate) == "elite_alpha"
    assert meets_publish_bar(candidate, "elite_alpha")


def test_mechanism_promise_may_omit_topic_context_when_outcome_has_it() -> None:
    hits = [
        _hit(
            "promise",
            "Resveratrol improves mitochondrial function and protects against metabolic disease",
            "Resveratrol activated SIRT1 PGC-1alpha signaling and improved running performance.",
        ),
        _hit(
            "outcome",
            "Resveratrol blunts the positive effects of exercise training in aged men",
            "The randomized exercise training trial observed resveratrol blunted training adaptation.",
        ),
    ]

    candidate = mine_insights(
        hits,
        topic="resveratrol exercise training adaptation",
        required_anchor_terms=("resveratrol",),
    )[0]

    assert candidate.receipt_ids == ("promise", "outcome")
    assert candidate_alpha_tier(candidate) == "elite_alpha"


def test_topic_context_blocks_same_anchor_injury_drift() -> None:
    hits = [
        _hit(
            "retraction",
            "Retraction: Resveratrol improves mitochondrial biogenesis after brain injury",
            "Retraction notice for resveratrol PGC-1alpha signaling in early brain injury.",
        ),
        _hit(
            "lung",
            "Resveratrol attenuates hyperoxia-induced lung injury via SIRT1 PGC-1alpha",
            "Resveratrol attenuated neonatal rat lung injury through SIRT1 PGC-1alpha signaling.",
        ),
    ]

    candidates = mine_insights(
        hits,
        topic="resveratrol exercise training adaptation",
        required_anchor_terms=("resveratrol",),
        include_discovery=True,
    )

    assert candidates == []


def test_title_only_augment_protocol_can_pair_with_blunted_outcome() -> None:
    hits = [
        _hit(
            "protocol",
            "Metformin to augment strength training effective response in seniors: The MASTERS trial",
            "",
        ),
        _hit(
            "outcome",
            "Metformin blunts muscle hypertrophy in response to progressive resistance exercise training in older adults",
            "The outcome trial observed metformin blunted resistance training hypertrophy.",
        ),
    ]

    candidate = mine_insights(
        hits,
        topic="metformin resistance training adaptation",
        required_anchor_terms=("metformin",),
    )[0]

    assert candidate.receipt_ids == ("protocol", "outcome")
    assert "shape:promise_outcome_reversal" in candidate.reasons
    assert candidate_alpha_tier(candidate) == "elite_alpha"


def test_short_single_word_bridge_is_not_enough_for_elite_shape() -> None:
    hits = [
        _hit("a", "APR improves training", "APR improved training outcomes."),
        _hit("b", "APR blunts recovery", "APR blunted recovery outcomes."),
    ]

    assert mine_insights(hits, topic="APR intervention training", required_anchor_terms=("apr",)) == []


def test_negated_improvement_is_publishable_not_elite_without_promise_outcome() -> None:
    hits = [
        _hit(
            "attenuated",
            "Cold water immersion suppresses muscle recovery after resistance training",
            "Cold water immersion suppresses anabolic recovery mechanisms following resistance training.",
        ),
        _hit(
            "null",
            "Cold water immersion does not improve recovery after resistance training",
            "Cold water immersion did not improve short-term recovery after resistance training.",
        ),
    ]

    candidate = mine_insights(
        hits,
        topic="cold water immersion resistance training adaptation",
        required_anchor_terms=("cold", "water", "immersion"),
    )[0]

    assert candidate.tension_terms == ("negative", "null")
    assert candidate_alpha_tier(candidate) == "publishable_alpha"


def test_common_multiword_anchor_does_not_make_negative_null_pair_elite() -> None:
    hits = [
        _hit(
            "negative",
            "Cold water immersion suppresses muscle recovery after resistance training",
            "Cold water immersion suppresses anabolic recovery mechanisms following resistance training.",
        ),
        _hit(
            "null",
            "Cold water immersion does not improve recovery after resistance training",
            "Cold water immersion did not improve short-term recovery after resistance training.",
        ),
        *[
            _hit(
                f"filler-{index}",
                f"Cold water immersion recovery protocol {index}",
                "Cold water immersion recovery protocol context.",
            )
            for index in range(8)
        ],
    ]

    candidate = mine_insights(
        hits,
        topic="cold water immersion resistance training adaptation",
        required_anchor_terms=("cold", "water", "immersion"),
    )[0]

    assert candidate.receipt_ids == ("negative", "null")
    assert candidate_alpha_tier(candidate) == "publishable_alpha"


def test_weak_source_format_does_not_promote_pair_to_elite() -> None:
    hits = [
        _hit(
            "promise",
            "Resveratrol activates mitochondrial exercise mechanism",
            "Resveratrol improved mitochondrial function and running performance.",
        ),
        _hit(
            "supplement",
            "Conference supplement abstract: Resveratrol blunts exercise training adaptation",
            "Randomized trial observed resveratrol blunted exercise training adaptation.",
        ),
    ]

    candidate = mine_insights(
        hits,
        topic="resveratrol exercise training adaptation",
        required_anchor_terms=("resveratrol",),
    )[0]

    assert "shape:promise_outcome_reversal" in candidate.reasons
    assert candidate_alpha_tier(candidate) == "publishable_alpha"


@pytest.mark.parametrize("case", _golden_cases(), ids=lambda case: case["name"])
def test_miner_golden_alpha_quality_cases(
    case: dict[str, Any],
) -> None:
    topic = str(case["topic"])
    anchors = tuple(str(anchor) for anchor in case["anchors"])
    hits = [
        _hit(str(hit["id"]), str(hit["title"]), str(hit["abstract"]))
        for hit in case["hits"]
    ]
    expected_ids = tuple(str(hit_id) for hit_id in case["expected_ids"])
    expected_shape = str(case["expected_shape"])
    expected_tier = str(case["expected_tier"])
    candidates = mine_insights(hits, topic=topic, required_anchor_terms=anchors)
    if not expected_ids:
        assert candidates == []
        return

    candidate = candidates[0]
    assert candidate.receipt_ids == expected_ids
    assert expected_shape in candidate.reasons
    assert candidate_alpha_tier(candidate) == expected_tier


def test_miner_assigns_selector_receipt_roles() -> None:
    case = next(item for item in _golden_cases() if item["name"] == "metformin")
    hits = [
        _hit(str(hit["id"]), str(hit["title"]), str(hit["abstract"]))
        for hit in case["hits"]
    ]

    candidate = mine_insights(
        hits,
        topic=str(case["topic"]),
        required_anchor_terms=tuple(str(anchor) for anchor in case["anchors"]),
    )[0]

    assert [(role.receipt_id, role.role) for role in candidate.receipt_roles] == [
        ("protocol", "promise"),
        ("outcome", "outcome"),
    ]


def test_miner_does_not_invent_timing_split_from_same_timing_word() -> None:
    hits = [
        _hit("positive", "Acute tool marker improves outcome", "Acute tool marker improved outcome quality."),
        _hit("negative", "Acute tool marker reduces reliability", "Acute tool marker reduced reliability quality."),
    ]

    candidate = mine_insights(hits, topic="tool marker")[0]

    assert "shape:directional_reversal" in candidate.reasons
    assert "shape:timing_split" not in candidate.reasons


def test_render_discovery_seed_downgrades_label() -> None:
    candidate = InsightCandidate(
        topic="topic",
        thesis="Weak bridge only.",
        bridge_terms=("bridge",),
        tension_terms=(),
        receipt_ids=("a", "b"),
        score=40,
        novelty_score=20,
        evidence_score=60,
        reasons=("shape:measurement_mismatch",),
    )

    memo = render_discovery_seed(candidate, [_hit("a", "A title", "A abstract")])

    assert memo.startswith("# Discovery seed:")
    assert "# Alpha memo:" not in memo


def test_pipeline_supports_explicit_discovery_seed_lane() -> None:
    hits = [
        _hit("a", "Audit benchmark metric score", "Audit benchmark metric score compared systems."),
        _hit("b", "Audit deployment outcome errors", "Audit deployment outcome errors compared systems."),
    ]

    class FakeSearch:
        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del query, limit
            return hits

    with pytest.raises(MemoBuildError):
        build_alpha_memo(topic="tool reliability", seed_queries=["audit systems"], searcher=FakeSearch())

    result = build_alpha_memo(
        topic="tool reliability",
        seed_queries=["audit systems"],
        searcher=FakeSearch(),
        min_alpha_tier="discovery_seed",
    )

    assert candidate_alpha_tier(result.candidate) == "discovery_seed"
    assert result.markdown.startswith("# Discovery seed:")


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
    assert candidate_alpha_tier(candidate) == "publishable_alpha"


def test_pipeline_filters_publishable_seed_when_elite_required() -> None:
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

    class FakeSearch:
        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del query, limit
            return hits

    with pytest.raises(MemoBuildError, match="no receipt-bound") as exc:
        build_alpha_memo(
            topic="longevity sauna cardiovascular risk",
            seed_queries=["sauna hypertension"],
            searcher=FakeSearch(),
            min_alpha_tier="elite_alpha",
        )
    assert exc.value.failure.details["min_alpha_tier"] == "elite_alpha"
    assert "hit_count=2" in str(exc.value)
    assert "candidate_count=0" in str(exc.value)
    assert "min_alpha_tier=elite_alpha" in str(exc.value)


def test_publish_bar_blocks_low_score_publishable_shape() -> None:
    candidate = InsightCandidate(
        topic="topic",
        thesis="Weak but shaped.",
        bridge_terms=("common",),
        tension_terms=(),
        receipt_ids=("a", "b"),
        score=55,
        novelty_score=10,
        evidence_score=80,
        reasons=("shape:denominator_split", "tier:publishable_alpha"),
    )

    assert not meets_publish_bar(candidate, "publishable_alpha")


def test_publish_bar_blocks_low_novelty_elite_shape() -> None:
    candidate = InsightCandidate(
        topic="topic",
        thesis="Shaped but generic.",
        bridge_terms=("randomized", "placebo"),
        tension_terms=("negative", "positive"),
        receipt_ids=("a", "b"),
        score=90,
        novelty_score=18,
        evidence_score=90,
        reasons=("shape:promise_outcome_reversal", "tier:elite_alpha"),
    )

    assert not meets_publish_bar(candidate, "elite_alpha")


def test_miner_ranks_shaped_candidates_above_rare_keyword_bridges() -> None:
    hits = [
        _hit("weak-a", "Zorblax workflow evidence summary", "Zorblax workflow evidence appears in one summary."),
        _hit("weak-b", "Zorblax workflow evidence methods", "Zorblax workflow evidence appears in methods."),
        _hit("strong-a", "Tool improves benchmark accuracy", "The tool improved benchmark accuracy score."),
        _hit(
            "strong-b",
            "Tool reduces deployment benchmark reliability outcomes",
            "The tool reduced deployment benchmark reliability while error outcome rates worsened.",
        ),
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


def test_scorer_caps_common_bridge_without_tension() -> None:
    score = score_connection(
        bridge_terms=("bridge",),
        bridge_doc_counts={"bridge": 99},
        unique_source_count=3,
        receipt_count=2,
        has_tension=False,
        shape_score=1,
        shape_reasons=("shape:denominator_split",),
    )

    assert score.score <= 55


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


def test_miner_does_not_turn_mixed_endpoint_packaging_into_reversal() -> None:
    hits = [
        _hit(
            "package-a",
            "Supplement trial reports muscle and mobility endpoints",
            "The supplement improved chair stand but handgrip and SPPB did not differ.",
        ),
        _hit(
            "package-b",
            "Supplement trial reports bone biomarker endpoints",
            "The same supplement improved vitamin D and BMD while lowering CTX.",
        ),
    ]

    candidates = mine_insights(hits, topic="longevity supplement")

    assert all("shape:directional_reversal" not in c.reasons for c in candidates)


def test_miner_rejects_power_word_endpoint_mismatch_as_alpha() -> None:
    hits = [
        _hit(
            "redox",
            "Oxidative stress responses to a graded maximal exercise test in older adults following explosive-type resistance training",
            "Sixteen older adults were randomized to explosive-type resistance training or control. Training attenuated oxidative stress biomarkers after graded maximal exercise.",
        ),
        _hit(
            "hmb",
            "Effects of HMB-free acid supplementation on strength power and hormonal adaptations following resistance training",
            "Sixteen matched healthy men received HMB-FA or placebo. HMB-FA increased peak power and 1RM and produced greater decrements in cortisol and ACTH.",
        ),
    ]

    candidates = mine_insights(
        hits,
        topic="longevity exercise adaptation supplement reversal",
        required_anchor_terms=("training",),
        include_discovery=True,
    )

    assert all(candidate_alpha_tier(candidate) != "publishable_alpha" for candidate in candidates)


def test_miner_rejects_comparative_method_word_as_alpha_bridge() -> None:
    hits = [
        _hit(
            "methylome",
            "The comparative methylome and transcriptome after change of direction compared to straight line running exercise in human skeletal muscle",
            "The mechanism improved skeletal muscle exercise adaptation.",
        ),
        _hit(
            "crispr",
            "A Comparative Study on the Efficiency of CRISPR-Cas9 in Human Embryonic Kidney 293 Cells and Peripheral Blood Mononuclear Cells for Disruption in Programmed Cell Death Protein 1",
            "The randomized trial observed unchanged editing outcomes.",
        ),
    ]

    assert mine_insights(
        hits,
        topic="longevity exercise adaptation",
        include_discovery=True,
    ) == []


def test_miner_rejects_broad_protein_muscle_cachexia_bridge() -> None:
    hits = [
        _hit(
            "timing-review",
            "Revisiting Protein Intake in Fitness Training: New Evidence on Dose, Timing, and Skeletal Muscle Adaptation",
            "Review discusses protein timing and skeletal muscle adaptation in fitness training.",
        ),
        _hit(
            "cachexia",
            "Liver and muscle protein metabolism in cachexia",
            "Cachexia reduced muscle protein metabolism without training timing or distribution endpoints.",
        ),
    ]

    assert mine_insights(
        hits,
        topic="protein timing distribution resistance training muscle protein synthesis hypertrophy",
        required_anchor_terms=query_anchor_terms([
            "protein timing distribution resistance training muscle protein synthesis hypertrophy"
        ]),
        include_discovery=True,
    ) == []


def test_miner_rejects_broad_synthesis_duplicate_bridge() -> None:
    hits = [
        _hit(
            "diphtheria-a",
            "Studies on the mode of action of diphtheria toxin. Protein synthesis in primary heart cell cultures",
            "Diphtheria toxin reduced protein synthesis in cell cultures.",
        ),
        _hit(
            "diphtheria-b",
            "Studies on the mode of action of diphtheria toxin: protein synthesis in primary heart cell cultures",
            "Diphtheria toxin failed to improve protein synthesis in cell cultures.",
        ),
    ]

    assert mine_insights(
        hits,
        topic="protein timing distribution resistance training muscle protein synthesis hypertrophy",
        required_anchor_terms=query_anchor_terms([
            "protein timing distribution resistance training muscle protein synthesis hypertrophy"
        ]),
        include_discovery=True,
    ) == []


def test_miner_rejects_abstract_only_bridge_words_as_alpha() -> None:
    hits = [
        _hit(
            "structure",
            "Malleability of skeletal muscle in overcoming limitations",
            "Skeletal muscle inner membrane adaptation improved close coupling under training.",
        ),
        _hit(
            "mitochondria",
            "Alteration of mitochondrial oxidative phosphorylation in aged muscle",
            "Aged muscle reduced inner membrane coupling under mitochondrial stress.",
        ),
    ]

    assert mine_insights(hits, topic="longevity skeletal muscle adaptation") == []


def test_miner_rejects_pairs_from_unrelated_seed_queries() -> None:
    hits = [
        CorpusHit(
            hit_id="cancer",
            title="mTOR inhibition improves cancer response",
            abstract="mTOR inhibition improved clinical response.",
            source="openalex",
            metadata={"seed_queries": ("mTOR cancer",)},
        ),
        CorpusHit(
            hit_id="aging",
            title="mTOR inhibition worsens cognitive aging signal",
            abstract="mTOR inhibition reduced cognitive aging outcomes.",
            source="openalex",
            metadata={"seed_queries": ("mTOR aging",)},
        ),
    ]

    assert mine_insights(
        hits,
        topic="longevity exercise adaptation",
        required_anchor_terms=(),
        include_discovery=True,
    ) == []


def test_miner_uses_shared_seed_query_terms_as_bridge_anchors() -> None:
    hits = [
        CorpusHit(
            hit_id="a",
            title="Resveratrol activates mitochondrial adaptation",
            abstract="Resveratrol improved mitochondrial adaptation.",
            source="openalex",
            metadata={"seed_queries": ("resveratrol mitochondrial exercise adaptation blunted",)},
        ),
        CorpusHit(
            hit_id="b",
            title="Resveratrol mitochondrial training adaptation blunting",
            abstract="Resveratrol reduced mitochondrial training adaptation.",
            source="openalex",
            metadata={"seed_queries": ("resveratrol mitochondrial exercise adaptation blunted",)},
        ),
    ]

    candidate = mine_insights(
        hits,
        topic="longevity exercise adaptation",
        required_anchor_terms=(),
    )[0]

    assert "resveratrol" in candidate.bridge_terms[:2]


def test_miner_does_not_promote_position_stand_plus_trial_to_elite() -> None:
    hits = [
        _hit(
            "stand",
            "International Society of Sports Nutrition Position Stand: beta-hydroxy-beta-methylbutyrate",
            "The position stand reviewed prior HMB supplementation studies and reported possible positive performance effects.",
        ),
        _hit(
            "trial",
            "Effects of HMB-free acid supplementation on strength power and hormonal adaptations following resistance training",
            "HMB-FA increased peak power and 1RM but produced greater decrements in cortisol and ACTH following resistance training.",
        ),
    ]

    candidates = mine_insights(
        hits,
        topic="longevity exercise adaptation supplement reversal",
        required_anchor_terms=("hmb",),
        include_discovery=True,
    )

    assert all(candidate_alpha_tier(candidate) != "elite_alpha" for candidate in candidates)


def test_miner_does_not_promote_mechanism_adjacent_cwi_pair_to_elite() -> None:
    hits = [
        _hit(
            "adaptation",
            "Cold water immersion attenuates anabolic signalling and skeletal muscle fiber hypertrophy",
            "Cold water immersion attenuated hypertrophy after whole-body resistance training.",
        ),
        _hit(
            "mechanism",
            "Mechanism involved of post-exercise cold water immersion",
            "Blood redistribution and increase in energy expenditure occurred during rewarming.",
        ),
    ]

    candidates = mine_insights(
        hits,
        topic="cold water immersion resistance training adaptation",
        required_anchor_terms=("cold", "water"),
        include_discovery=True,
    )

    assert all(candidate_alpha_tier(candidate) != "elite_alpha" for candidate in candidates)


def test_miner_does_not_promote_potential_strategy_review_to_elite() -> None:
    hits = [
        _hit(
            "hwi-review",
            "Hot water immersion as a potential strategy to manipulate adaptation after resistance training",
            "This review aimed to determine whether hot water immersion could improve acute response and chronic adaptation.",
        ),
        _hit(
            "cwi-trial",
            "Cold water immersion attenuates anabolic signalling and skeletal muscle fiber hypertrophy",
            "Cold water immersion attenuated hypertrophy after whole-body resistance training.",
        ),
    ]

    candidates = mine_insights(
        hits,
        topic="cold water immersion resistance training adaptation",
        required_anchor_terms=("water", "immersion"),
        include_discovery=True,
    )

    assert all(candidate_alpha_tier(candidate) != "elite_alpha" for candidate in candidates)


def test_miner_does_not_promote_question_review_to_elite() -> None:
    hits = [
        _hit(
            "hwi-question",
            "Turning up the heat: can post-exercise hot water immersion be used to manipulate acute physiological responses and chronic adaptation following resistance training?",
            "This narrative review aimed to determine whether hot water immersion could improve acute response and chronic adaptation.",
        ),
        _hit(
            "cwi-trial",
            "Cold water immersion attenuates anabolic signalling and skeletal muscle fiber hypertrophy",
            "Cold water immersion attenuated hypertrophy after whole-body resistance training.",
        ),
    ]

    candidates = mine_insights(
        hits,
        topic="cold water immersion resistance training adaptation",
        required_anchor_terms=("cold", "immersion"),
        include_discovery=True,
    )

    assert all(candidate_alpha_tier(candidate) != "elite_alpha" for candidate in candidates)


def test_miner_requires_elite_bridge_to_preserve_topic_anchor() -> None:
    hits = [
        _hit(
            "metabolic",
            "Effect of pharmacological and physical interventions on diabetic rats",
            "Metformin improved diabetic metabolism in male rats.",
        ),
        _hit(
            "exercise",
            "Physical exercise training but not metformin attenuates diabetic markers",
            "Exercise attenuated diabetic markers while metformin did not.",
        ),
    ]

    candidates = mine_insights(
        hits,
        topic="metformin resistance training adaptation",
        required_anchor_terms=("metformin", "training"),
        include_discovery=True,
    )

    assert all(candidate_alpha_tier(candidate) != "elite_alpha" for candidate in candidates)


def test_miner_does_not_promote_recommendation_proxy_receipts_to_elite() -> None:
    hits = [
        _hit(
            "proxy",
            "Faculty Opinions recommendation of metformin blunts muscle hypertrophy",
            "Recommendation of a metformin resistance training paper.",
        ),
        _hit(
            "protocol",
            "Metformin to augment strength training effective response in seniors",
            "Protocol expected metformin would augment training response.",
        ),
    ]

    candidates = mine_insights(
        hits,
        topic="metformin resistance training adaptation",
        required_anchor_terms=("metformin", "training"),
        include_discovery=True,
    )

    assert all(candidate_alpha_tier(candidate) != "elite_alpha" for candidate in candidates)


def test_miner_prefers_named_program_reversal_over_loose_topic_pair() -> None:
    hits = [
        _hit(
            "masters-protocol",
            "METFORMIN TO AUGMENT STRENGTH TRAINING EFFECTIVE RESPONSE IN SENIORS: THE MASTERS TRIAL",
            "The MASTERS protocol hypothesized metformin would augment strength training.",
        ),
        _hit(
            "masters-outcome",
            "Metformin blunts muscle hypertrophy in response to progressive resistance exercise training: The MASTERS trial",
            "The outcome trial observed metformin blunted resistance training hypertrophy.",
        ),
        _hit(
            "swim",
            "Swim training reduces metformin levels in insulin resistant rats",
            "Swim training reduced metformin levels in insulin resistant rats.",
        ),
        _hit(
            "prediabetes",
            "Independent and combined effects of exercise training and metformin on insulin sensitivity",
            "Exercise training improved insulin sensitivity with metformin.",
        ),
    ]

    candidates = mine_insights(
        hits,
        topic="metformin resistance training adaptation",
        required_anchor_terms=("metformin", "training"),
        include_discovery=True,
    )

    assert candidates[0].receipt_ids == ("masters-protocol", "masters-outcome")
    assert "coupling:named_program" in candidates[0].reasons


def test_miner_rejects_drug_resistance_false_context_for_training_topic() -> None:
    hits = [
        _hit(
            "cancer",
            "Metformin treatment reduces temozolomide resistance of glioblastoma cells",
            "Metformin reduced drug resistance in cancer cells.",
        ),
        _hit(
            "insulin",
            "Independent effects of exercise training and metformin on insulin sensitivity",
            "Exercise training improved insulin sensitivity with metformin.",
        ),
    ]

    candidates = mine_insights(
        hits,
        topic="metformin resistance training adaptation",
        required_anchor_terms=("metformin", "resistance"),
        include_discovery=True,
    )

    assert candidates == []


def test_pipeline_raises_when_no_receipt_bound_candidate() -> None:
    class EmptySearch:
        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del query, limit
            return []

    with pytest.raises(ValueError, match="no receipt-bound"):
        build_alpha_memo(topic="topic", seed_queries=["x"], searcher=EmptySearch())


def test_pipeline_fails_closed_when_memo_coverage_is_too_narrow() -> None:
    class NarrowSearch:
        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del query, limit
            receipt = {
                "shards_searched": 12,
                "sources_searched": {"openalex": 12},
                "year_range_searched": {"min": 2020, "max": 2024},
                "cited_by_range_searched": {"min": 0, "max": 10},
            }
            return [
                CorpusHit(
                    hit_id="narrow-1",
                    title="NAD salvage links sleep fragmentation to mitochondrial stress",
                    abstract="Sleep fragmentation reduced resilience through NAD salvage and mitochondrial stress.",
                    source="fullraw:openalex",
                    year=2024,
                    doi="10.narrow/1",
                    metadata={"shard_receipt": receipt, "search_pass": "focused"},
                ),
                CorpusHit(
                    hit_id="narrow-2",
                    title="NAD salvage predicts exercise response through mitochondrial repair",
                    abstract="Exercise improved resilience when NAD salvage and mitochondrial repair markers moved together.",
                    source="fullraw:openalex",
                    year=2023,
                    doi="10.narrow/2",
                    metadata={"shard_receipt": receipt, "search_pass": "focused"},
                ),
            ]

    with pytest.raises(MemoBuildError, match="coverage too narrow") as exc:
        build_alpha_memo(
            topic="longevity resilience",
            seed_queries=["nad mitochondrial"],
            searcher=NarrowSearch(),
            min_alpha_tier="discovery_seed",
            min_shards_searched=50,
            min_sources_searched=2,
            min_search_passes=2,
        )

    assert exc.value.failure.code == "memo_coverage_too_narrow"
    coverage = exc.value.failure.details["coverage"]
    assert isinstance(coverage, dict)
    assert coverage["shards_searched"] == 12
    assert exc.value.failure.details["failures"] == (
        "shards_searched",
        "sources_searched",
        "search_passes",
    )


def test_pipeline_blocks_title_only_elite_memos() -> None:
    class TitleOnlySearch:
        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del query, limit
            return [
                _hit(
                    "protocol",
                    "Metformin to augment strength training effective response in seniors",
                    "",
                ),
                _hit(
                    "outcome",
                    "Metformin blunts muscle hypertrophy in response to resistance training",
                    "",
                ),
            ]

    with pytest.raises(MemoBuildError, match="coverage too narrow") as exc:
        build_alpha_memo(
            topic="metformin resistance training adaptation",
            seed_queries=["metformin resistance training"],
            searcher=TitleOnlySearch(),
            min_alpha_tier="elite_alpha",
        )

    assert exc.value.failure.details["failures"] == ("abstract_receipts",)


def test_pipeline_accepts_deep_fullraw_memo_coverage() -> None:
    class DeepSearch:
        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del query, limit
            receipt = {
                "shards_total": 100,
                "shards_searched": 48,
                "sources_searched": {"openalex": 24, "semantic_scholar": 24},
                "year_range_searched": {"min": 1990, "max": 2024},
                "cited_by_range_searched": {"min": 0, "max": 1200},
                "sweep_scope": "relevant",
            }
            return [
                CorpusHit(
                    hit_id="deep-1",
                    title="NAD salvage links sleep fragmentation to mitochondrial stress",
                    abstract="Sleep fragmentation reduced resilience through NAD salvage and mitochondrial stress.",
                    source="fullraw:openalex",
                    year=2024,
                    doi="10.deep/1",
                    metadata={"shard_receipt": receipt, "search_pass": "focused"},
                ),
                CorpusHit(
                    hit_id="deep-2",
                    title="NAD salvage predicts exercise response through mitochondrial repair",
                    abstract="Exercise improved resilience when NAD salvage and mitochondrial repair markers moved together.",
                    source="fullraw:semantic_scholar",
                    year=2023,
                    doi="10.deep/2",
                    metadata={"shard_receipt": receipt, "search_pass": "broad"},
                ),
            ]

    result = build_alpha_memo(
        topic="longevity resilience",
        seed_queries=["nad mitochondrial"],
        searcher=DeepSearch(),
        min_alpha_tier="discovery_seed",
        min_shards_searched=48,
        min_sources_searched=2,
        min_search_passes=2,
    )

    assert [hit.doi for hit in result.receipts] == ["10.deep/1", "10.deep/2"]
