import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from email.message import Message
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError

import pytest

from v5_memo.binder import bind_receipts
from v5_memo.evidence import source_artifact_type
from v5_memo.gate import (
    _topic_entity_terms,
    candidate_alpha_tier,
    candidate_publish_blocker,
    memo_coverage_failure,
    no_alpha_failure,
)
from v5_memo.miner import (
    _claim_card,
    _prioritize_evidence_bundle,
    mine_insights,
    query_anchor_terms,
)
from v5_memo.minimax_writer import (
    MemoFormatError,
    build_minimax_prompt,
    validate_minimax_memo,
)
from v5_memo.pipeline import (
    _apply_selector,
    _publishable_candidates,
    _selector_slate,
    build_alpha_memo,
)
from v5_memo.publisher import (
    build_researka_payload,
    set_researka_public_visibility,
    submit_researka,
    wait_researka_decision,
)
from v5_memo.retriever import collect_seed_hits
from v5_memo.schemas import (
    ClaimCard,
    CorpusHit,
    EvidenceNode,
    InsightCandidate,
    MemoBuildError,
    MemoResult,
    ReceiptRole,
)
from v5_memo.scorer import score_connection
from v5_memo.writer import render_memo

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


def _direct_card(receipt_id: str, role: str, outcome: str, direction: str, quote: str) -> ClaimCard:
    return ClaimCard(receipt_id, role, "intervention_study", "human", outcome, direction, "direct", "high", quote)


class _JsonResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "_JsonResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()


def test_submit_researka_retries_429_with_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    sleeps: list[float] = []

    def fake_urlopen(request: object, timeout: float) -> _JsonResponse:
        del timeout
        calls.append(request)
        if len(calls) == 1:
            headers = Message()
            headers["Retry-After"] = "0.25"
            raise HTTPError("https://api.researka.org/submissions", 429, "Too Many Requests", headers, None)
        return _JsonResponse({"submission_id": "sub-retry"})

    monkeypatch.setattr("v5_memo.publisher.urlopen", fake_urlopen)
    monkeypatch.setattr("v5_memo.publisher.time.sleep", lambda seconds: sleeps.append(seconds))

    response = submit_researka(
        {"title": "ok", "author_agent_id": "v5-memo-agent"},
        agent_key="submit-key",
        max_retries=1,
    )

    assert response == {"submission_id": "sub-retry"}
    assert len(calls) == 2
    assert sleeps == [0.25]


def test_submit_researka_does_not_retry_429_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def fake_urlopen(request: object, timeout: float) -> _JsonResponse:
        del timeout
        calls.append(request)
        headers = Message()
        headers["Retry-After"] = "0.25"
        raise HTTPError("https://api.researka.org/submissions", 429, "Too Many Requests", headers, None)

    monkeypatch.setattr("v5_memo.publisher.urlopen", fake_urlopen)

    with pytest.raises(HTTPError):
        submit_researka(
            {"title": "ok", "author_agent_id": "v5-memo-agent"},
            agent_key="submit-key",
        )

    assert len(calls) == 1


def test_wait_researka_decision_waits_for_minted_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decisions: list[dict[str, object]] = [
        {"status": "complete", "decision": "accept"},
        {
            "status": "complete",
            "decision": "accept",
            "publication": {
                "publication_id": "pub-1",
                "doi": "10.17605/OSF.IO/WAIT1",
                "doi_status": "pending",
            },
        },
        {
            "status": "complete",
            "decision": "accept",
            "publication": {
                "publication_id": "pub-1",
                "doi": "10.17605/OSF.IO/WAIT1",
                "doi_status": "minted",
            },
        },
    ]
    calls: list[tuple[str, str]] = []
    sleeps: list[float] = []

    def fake_fetch(submission_id: str, *, api_base: str) -> dict[str, object]:
        calls.append((submission_id, api_base))
        return decisions[len(calls) - 1]

    monkeypatch.setattr("v5_memo.publisher.fetch_researka_decision", fake_fetch)
    monkeypatch.setattr(
        "v5_memo.publisher.time.sleep",
        lambda seconds: sleeps.append(seconds),
    )

    decision = wait_researka_decision(
        "sub-1",
        api_base="https://api.example.test",
        timeout_seconds=10,
        poll_seconds=0.25,
    )

    assert decision["publication"] == {
        "publication_id": "pub-1",
        "doi": "10.17605/OSF.IO/WAIT1",
        "doi_status": "minted",
    }
    assert calls == [("sub-1", "https://api.example.test")] * 3
    assert sleeps == [0.25, 0.25]


def test_wait_researka_decision_marks_accepted_publication_pending_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch(_submission_id: str, *, api_base: str) -> dict[str, object]:
        del api_base
        return {
            "status": "complete",
            "decision": "accept",
            "publication": {"doi_status": "pending"},
        }

    monkeypatch.setattr(
        "v5_memo.publisher.fetch_researka_decision",
        fake_fetch,
    )

    decision = wait_researka_decision("sub-1", timeout_seconds=0)

    assert decision == {
        "status": "timeout",
        "decision": "accept",
        "publication": {"doi_status": "pending"},
        "publication_pending": True,
    }


def test_visibility_403_verifies_already_listed_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def fake_urlopen(request: object, timeout: float) -> _JsonResponse:
        del timeout
        calls.append(request)
        if len(calls) == 1:
            raise HTTPError(
                "https://api.researka.org/ops/publications/pub-1/visibility",
                403,
                "Forbidden",
                Message(),
                None,
            )
        return _JsonResponse({"metadata": {"public_visibility": "listed"}})

    monkeypatch.setattr("v5_memo.publisher.urlopen", fake_urlopen)

    assert set_researka_public_visibility("pub-1", agent_key="agent-key") == {
        "id": "pub-1",
        "public_visibility": "listed",
        "updated": False,
        "verified": True,
    }
    assert len(calls) == 2


class _StaticSearch:
    def __init__(self, hits: Sequence[CorpusHit]) -> None:
        self._hits = hits

    def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
        del query, limit
        return self._hits


class _FunctionSearch:
    def __init__(self, search: Callable[[str, int], Sequence[CorpusHit]]) -> None:
        self._search = search

    def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
        return self._search(query, limit)


def test_mines_bridge_and_renders_receipt_bound_memo() -> None:
    hits = _hits()
    candidate = mine_insights(hits, topic="longevity resilience")[0]
    receipts = bind_receipts(candidate, hits)
    memo = render_memo(candidate, receipts)

    assert memo.startswith("# Alpha memo: mitochondrial / nad")
    assert "mitochondrial" in candidate.bridge_terms
    assert "nad" in candidate.bridge_terms
    assert candidate.score >= 60
    assert candidate.scorecard["construct_match"] >= 50
    assert candidate.scorecard["novelty_vs_corpus"] >= 0
    assert "shape:directional_reversal" in candidate.reasons
    assert "Alpha hypothesis" in memo
    assert "longevity resilience may be hiding" in memo
    assert "10.1/sleep-nad" in memo
    assert "10.2/exercise-nad" in memo
    assert "Safety note" in memo


def test_miner_emits_claim_cards_before_prose() -> None:
    hits = [
        _hit(
            "promise",
            "Protocol expected metformin training augmentation",
            "Protocol hypothesis expected metformin would improve strength training response in older human adults.",
        ),
        _hit(
            "outcome",
            "Trial observed metformin training blunting",
            "Randomized human outcome trial observed metformin reduced strength training response.",
        ),
    ]

    candidate = mine_insights(
        hits,
        topic="metformin resistance training adaptation",
        required_anchor_terms=("metformin", "training"),
    )[0]
    memo = render_memo(candidate, hits)

    by_role = {card.role: card for card in candidate.claim_cards}
    assert by_role["outcome"].design == "randomized_trial"
    assert by_role["outcome"].population == "human"
    assert by_role["outcome"].support_type == "direct"
    assert "**Claim ledger:**" in memo


def test_template_writer_replaces_auto_thesis_with_bounded_claim() -> None:
    candidate = InsightCandidate(
        topic="metformin resistance training",
        thesis=(
            "metformin resistance training may be hiding a metformin / diabetic / "
            "not boundary condition: training modulate humoral inflammatory and "
            "one bout training does point in different directions."
        ),
        bridge_terms=("metformin", "training"),
        tension_terms=("null", "negative"),
        receipt_ids=("glycemic", "hypertrophy"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("tier:publishable_alpha",),
        claim_cards=(
            ClaimCard(
                "glycemic",
                "null_signal",
                "randomized_trial",
                "human",
                "long/glycemic control",
                "null",
                "direct",
                "high",
                "Metformin and resistance training measured glycemic control in human adults.",
            ),
            ClaimCard(
                "hypertrophy",
                "negative_signal",
                "randomized_trial",
                "human",
                "hypertrophy/setting",
                "negative",
                "direct",
                "high",
                "Metformin blunted muscle hypertrophy after resistance training in human adults.",
            ),
        ),
    )
    receipts = [
        _hit("glycemic", "Glycemic control trial", "Human glycemic control trial."),
        _hit("hypertrophy", "Hypertrophy trial", "Human hypertrophy trial."),
    ]

    memo = render_memo(candidate, receipts)

    assert "may be hiding" not in memo
    assert "training modulate humoral inflammatory" not in memo
    assert (
        "**Alpha hypothesis:** In metformin resistance training, direct human "
        "randomized trial receipts support a bounded null and negative signal"
    ) in memo
    assert "long/" not in memo
    assert "/setting" not in memo
    assert "same population and endpoint" in memo


def test_template_writer_leads_with_synthesis_for_boundary_bundle() -> None:
    candidate = InsightCandidate(
        topic="metformin exercise training adaptation older adults trial",
        thesis="Metformin evidence separates exercise adaptation from frailty boundary outcomes.",
        bridge_terms=("metformin", "older", "diabete"),
        tension_terms=("negative", "positive"),
        receipt_ids=("exercise", "frailty"),
        score=100,
        novelty_score=53,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "exercise",
                "negative_signal",
                "randomized_trial",
                "human",
                "cardiorespiratory fitness adaptation",
                "negative",
                "direct",
                "high",
                "Metformin impaired cardiorespiratory fitness adaptation during training.",
            ),
            ClaimCard(
                "frailty",
                "boundary",
                "randomized_trial",
                "human",
                "frailty index",
                "positive",
                "direct",
                "high",
                "Metformin reduced frailty-index progression in a different older-adult endpoint.",
            ),
        ),
    )
    receipts = [
        _hit(
            "exercise",
            "Metformin Impairs the Cardiorespiratory Fitness Adaptation to High-Intensity Power Training",
            "Human randomized trial reported impaired cardiorespiratory fitness adaptation.",
        ),
        _hit(
            "frailty",
            "A Two-Year Trial of Metformin to Reduce Frailty in Older Adults with Glucose Intolerance",
            "Human randomized trial reported reduced frailty-index progression.",
        ),
    ]

    memo = render_memo(candidate, receipts)
    payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown=memo),
        author_agent_id="v5-memo-agent",
        domain_slug="longevity_research",
    )

    assert memo.index("**Core signal:**") < memo.index("**Audit trail:**")
    assert "**Receipt-level synthesis:**" in memo
    assert "The bundle is heterogeneous" in memo
    assert "diabetes" in memo
    assert "diabete" not in memo.replace("diabetes", "")
    assert payload["title"] == "Metformin: Training Adaptation With Boundary Evidence"
    assert "Metformin Impairs the Cardiorespiratory Fitness" not in payload["title"]


def test_template_writer_scopes_companion_analyses_to_named_study() -> None:
    candidate = InsightCandidate(
        topic="intervention exercise training adaptation",
        thesis="Endpoint-specific findings within one study program.",
        bridge_terms=("intervention", "exercise"),
        tension_terms=("negative",),
        receipt_ids=("a", "b"),
        score=100,
        novelty_score=55,
        evidence_score=100,
        reasons=("shape:boundary_condition", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard("a", "negative_signal", "randomized_trial", "human", "unspecified", "negative", "direct", "high", "Primary analysis."),
            ClaimCard("b", "boundary", "randomized_trial", "human", "risk", "negative", "direct", "high", "Companion analysis."),
        ),
    )
    receipts = [
        CorpusHit(
            hit_id="a",
            title="Intervention attenuates metabolic adaptation after exercise training",
            abstract="Secondary analysis of the Example Outcomes (EX-OUT) study.",
            source="fullraw:pubmed",
            year=2026,
            doi="10.1000/a",
        ),
        CorpusHit(
            hit_id="b",
            title="Intervention alters vascular response in adults at elevated risk",
            abstract="The same randomized program reported a vascular endpoint.",
            source="fullraw:pubmed",
            year=2025,
            doi="10.1000/b",
        ),
    ]

    memo = render_memo(candidate, receipts)
    payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown=memo),
        author_agent_id="v5-memo-agent",
        domain_slug="performance",
    )

    assert payload["title"] == "EX-OUT Program: Endpoint-Specific Intervention and Exercise Findings"
    body = cast(str, payload["body_markdown"])
    assert "Within the shared EX-OUT program cohort" in body
    assert "endpoint-dependent rather than uniformly additive" in body
    assert "intervention-exercise interaction attenuated metabolic adaptation and altered vascular response" in body
    assert "10.1000/a and 10.1000/b are companion analyses from the same EX-OUT trial program" in body
    assert body.count("one evidence unit") == 1
    assert "endpoint: metabolic adaptation; direction: attenuated" in body
    assert "endpoint: vascular response; direction: altered" in body
    assert "source records report 10.1000/a (2026) and 10.1000/b (2025)" in body
    assert "`10.1000/a`" not in body
    assert "`10.1000/b`" not in body
    assert "Source-record years 2025 and 2026 may include electronic-first publication dates" in body
    assert "missing protocol details are not inferred across companion articles" in body
    assert "negative_signal" not in body
    assert "(boundary)" not in body
    assert "unspecified is" not in body


def test_template_writer_extracts_reduction_endpoint_without_domain_rules() -> None:
    candidate = InsightCandidate(
        topic="intervention outcome",
        thesis="Endpoint-specific finding.",
        bridge_terms=("intervention", "outcome"),
        tension_terms=("positive",),
        receipt_ids=("a", "b"),
        score=90,
        novelty_score=60,
        evidence_score=90,
        reasons=("tier:publishable_alpha",),
        claim_cards=(
            ClaimCard("a", "outcome", "randomized_trial", "human", "unspecified", "positive", "direct", "high", ""),
            ClaimCard("b", "outcome", "randomized_trial", "human", "unspecified", "positive", "direct", "high", ""),
        ),
    )
    receipts = [
        _hit("a", "Program induces a clinically significant reduction in symptom severity: A trial", "Human randomized trial."),
        _hit("b", "Program improves functional capacity after training", "Human randomized trial."),
    ]

    memo = render_memo(candidate, receipts)

    assert "endpoint: symptom severity; direction: reduced" in memo
    assert "endpoint: functional capacity; direction: improved" in memo


def test_publish_quality_filter_removes_weak_candidates_before_writing() -> None:
    weak = InsightCandidate(
        topic="nicotinamide exercise performance",
        thesis="Weak translational bridge.",
        bridge_terms=("nicotinamide", "exercise"),
        tension_terms=("positive", "negative"),
        receipt_ids=("human", "rat"),
        score=100,
        novelty_score=58,
        evidence_score=90,
        reasons=("tier:publishable_alpha",),
        claim_cards=(
            ClaimCard(
                "human",
                "positive_signal",
                "randomized_trial",
                "human",
                "performance",
                "positive",
                "direct",
                "high",
                "human trial",
            ),
            ClaimCard(
                "rat",
                "boundary",
                "mechanistic_model",
                "animal",
                "performance",
                "negative",
                "indirect",
                "medium",
                "rat model",
            ),
        ),
    )
    strong = InsightCandidate(
        topic="cold immersion training",
        thesis="Two direct human trials create a bounded contrast.",
        bridge_terms=("cold", "immersion"),
        tension_terms=("positive", "negative"),
        receipt_ids=("human-a", "human-b"),
        score=100,
        novelty_score=58,
        evidence_score=90,
        reasons=("tier:publishable_alpha",),
        claim_cards=(
            ClaimCard(
                "human-a",
                "negative_signal",
                "randomized_trial",
                "human",
                "recovery",
                "negative",
                "direct",
                "high",
                "Cold immersion training human trial.",
            ),
            ClaimCard(
                "human-b",
                "positive_signal",
                "intervention_study",
                "human",
                "recovery",
                "positive",
                "direct",
                "high",
                "Cold immersion training human study.",
            ),
        ),
    )

    assert _publishable_candidates(
        [weak, strong],
        [],
        "publishable_alpha",
        frozenset(),
    ) == [weak, strong]
    assert _publishable_candidates(
        [weak, strong],
        [],
        "publishable_alpha",
        frozenset(),
        require_publish_quality=True,
    ) == [strong]


def test_no_alpha_failure_reports_publish_quality_blockers() -> None:
    weak = InsightCandidate(
        topic="nicotinamide exercise performance",
        thesis="Weak translational bridge.",
        bridge_terms=("nicotinamide", "exercise"),
        tension_terms=("positive", "negative"),
        receipt_ids=("human", "rat"),
        score=100,
        novelty_score=58,
        evidence_score=90,
        reasons=("tier:publishable_alpha",),
        claim_cards=(
            ClaimCard(
                "human",
                "positive_signal",
                "randomized_trial",
                "human",
                "performance",
                "positive",
                "direct",
                "high",
                "human trial",
            ),
            ClaimCard(
                "rat",
                "boundary",
                "mechanistic_model",
                "animal",
                "performance",
                "negative",
                "indirect",
                "medium",
                "rat model",
            ),
        ),
    )

    failure = no_alpha_failure(
        topic="nicotinamide exercise performance",
        hits=[],
        candidates=[],
        min_alpha_tier="publishable_alpha",
        mined_candidates=[weak],
        final_quality_blockers=[
            {
                "receipt_ids": ("human", "rat"),
                "blocker": {"error": "claim_trace_missing"},
            }
        ],
    )

    assert failure.details["publish_quality_blocked_count"] == 1
    blockers = failure.details["top_publish_quality_blockers"]
    assert isinstance(blockers, tuple)
    assert blockers[0]["receipt_ids"] == ("human", "rat")
    assert blockers[0]["blocker"] == {
        "error": "insufficient_direct_human_receipts",
        "direct_human_receipts": 1,
        "strong_direct_human_receipts": 1,
    }
    assert failure.details["final_publish_quality_blocked_count"] == 1
    final_blockers = failure.details["top_final_publish_quality_blockers"]
    assert isinstance(final_blockers, tuple)
    assert final_blockers[0]["blocker"] == {"error": "claim_trace_missing"}


def test_no_alpha_failure_proposes_one_receipt_derived_reanchored_lead() -> None:
    hits = [
        CorpusHit(
            hit_id="pressure-a",
            title="Creatine and blood pressure trial A",
            abstract="Randomized human trial found creatine lowered blood pressure.",
            source="fullraw",
            doi="10.1000/pressure-a",
        ),
        CorpusHit(
            hit_id="pressure-b",
            title="Creatine and blood pressure trial B",
            abstract="Randomized human trial found creatine improved blood pressure.",
            source="fullraw",
            doi="10.1000/pressure-b",
        ),
        CorpusHit(
            hit_id="strength",
            title="Creatine and resistance-training strength",
            abstract=(
                "Randomized human trial found creatine improved resistance-training "
                "muscle strength."
            ),
            source="fullraw",
            doi="10.1000/strength",
        ),
    ]
    candidate = InsightCandidate(
        topic="creatine resistance training strength",
        thesis="The direct evidence is cardiovascular rather than resistance-training evidence.",
        bridge_terms=("creatine",),
        tension_terms=("positive",),
        receipt_ids=tuple(hit.receipt_id for hit in hits),
        score=92,
        novelty_score=45,
        evidence_score=96,
        reasons=("tier:publishable_alpha",),
        claim_cards=tuple(
            ClaimCard(
                hit.receipt_id,
                "positive_signal",
                "randomized_trial",
                "human",
                "muscle strength" if hit.hit_id == "strength" else "blood pressure",
                "positive",
                "direct",
                "high",
                hit.abstract,
            )
            for hit in hits
        ),
    )

    failure = no_alpha_failure(
        topic=candidate.topic,
        hits=list(reversed(hits)),
        candidates=[],
        min_alpha_tier="publishable_alpha",
        mined_candidates=[candidate],
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "off_topic_primary_signal",
        "receipt_ids": candidate.receipt_ids[:2],
    }
    proposal = failure.details["lead_proposal"]
    assert isinstance(proposal, Mapping)
    assert proposal["schema"] == "v5_evidence_lead_proposal_v1"
    assert proposal["lead"] == "creatine blood pressure randomized trial human"
    assert proposal["receipt_ids"] == tuple(sorted(candidate.receipt_ids[:2]))
    assert proposal["source_keys"] == (
        "doi:10.1000/pressure-a",
        "doi:10.1000/pressure-b",
    )
    assert proposal["source_blocker"] == "off_topic_primary_signal"
    assert len(proposal["proposal_fingerprint"]) == 64

    tied = replace(candidate, bridge_terms=("betaine",))
    forward = no_alpha_failure(
        topic=candidate.topic,
        hits=hits,
        candidates=[],
        min_alpha_tier="publishable_alpha",
        mined_candidates=[candidate, tied],
    )
    reversed_order = no_alpha_failure(
        topic=candidate.topic,
        hits=list(reversed(hits)),
        candidates=[],
        min_alpha_tier="publishable_alpha",
        mined_candidates=[tied, candidate],
    )
    assert forward.details["lead_proposal"] == reversed_order.details["lead_proposal"]

    elite = replace(candidate, reasons=("tier:elite_alpha",))
    tier_forward = no_alpha_failure(
        topic=candidate.topic,
        hits=hits,
        candidates=[],
        min_alpha_tier="publishable_alpha",
        mined_candidates=[candidate, elite],
    )
    tier_reversed = no_alpha_failure(
        topic=candidate.topic,
        hits=list(reversed(hits)),
        candidates=[],
        min_alpha_tier="publishable_alpha",
        mined_candidates=[elite, candidate],
    )
    tier_proposal = tier_forward.details["lead_proposal"]
    assert isinstance(tier_proposal, Mapping)
    assert tier_proposal == tier_reversed.details["lead_proposal"]
    assert tier_proposal["candidate_tier"] == "elite_alpha"


def test_no_alpha_failure_rejects_ambiguous_receipt_aliases() -> None:
    hits = [
        CorpusHit(
            "primary-a",
            "Primary A",
            "Creatine lowered blood pressure in a randomized human trial.",
            "fullraw",
            doi="10.1000/a",
        ),
        CorpusHit(
            "10.1000/a",
            "Identifier collision",
            "A different human record reused the first DOI as its hit identifier.",
            "fullraw",
        ),
        CorpusHit(
            "primary-b",
            "Primary B",
            "Creatine improved blood pressure in a randomized human trial.",
            "fullraw",
            doi="10.1000/b",
        ),
    ]
    candidate = InsightCandidate(
        topic="creatine resistance training strength",
        thesis="The direct evidence is cardiovascular.",
        bridge_terms=("creatine",),
        tension_terms=("positive",),
        receipt_ids=("10.1000/a", "10.1000/b"),
        score=92,
        novelty_score=45,
        evidence_score=96,
        reasons=("tier:publishable_alpha",),
        claim_cards=tuple(
            ClaimCard(
                receipt_id,
                "positive_signal",
                "randomized_trial",
                "human",
                "blood pressure",
                "positive",
                "direct",
                "high",
                "Creatine changed blood pressure in a randomized human trial.",
            )
            for receipt_id in ("10.1000/a", "10.1000/b")
        ),
    )

    forward = no_alpha_failure(
        topic=candidate.topic,
        hits=hits,
        candidates=[],
        min_alpha_tier="publishable_alpha",
        mined_candidates=[candidate],
    )
    reversed_order = no_alpha_failure(
        topic=candidate.topic,
        hits=list(reversed(hits)),
        candidates=[],
        min_alpha_tier="publishable_alpha",
        mined_candidates=[candidate],
    )

    assert "lead_proposal" not in forward.details
    assert "lead_proposal" not in reversed_order.details


def test_no_alpha_failure_omits_proposal_without_distinct_strong_sources() -> None:
    hits = [
        CorpusHit("a", "Duplicate evidence", "Human trial blood pressure.", "fullraw"),
        CorpusHit("b", "Duplicate evidence", "Human trial blood pressure.", "fullraw"),
    ]
    candidate = InsightCandidate(
        topic="creatine resistance training strength",
        thesis="Duplicate sources cannot seed another lead.",
        bridge_terms=("creatine",),
        tension_terms=("positive",),
        receipt_ids=("a", "b"),
        score=92,
        novelty_score=45,
        evidence_score=96,
        reasons=("tier:publishable_alpha",),
        claim_cards=tuple(
            ClaimCard(
                hit.hit_id,
                "positive_signal",
                "randomized_trial",
                "human",
                "blood pressure",
                "positive",
                "direct",
                "high",
                hit.abstract,
            )
            for hit in hits
        ),
    )

    failure = no_alpha_failure(
        topic=candidate.topic,
        hits=hits,
        candidates=[],
        min_alpha_tier="publishable_alpha",
        mined_candidates=[candidate],
    )

    assert "lead_proposal" not in failure.details
    assert "publish_quality_blocked_count=1" in str(MemoBuildError(failure))


def test_miner_marks_human_intervention_receipts_direct() -> None:
    hits = [
        _hit(
            "negative",
            "Cold-water immersion attenuates strength training adaptation",
            (
                "In a randomized crossover design, 11 participants performed two 8-week "
                "training periods with cold-water immersion after each session."
            ),
        ),
        _hit(
            "positive",
            "Cold-water immersion supports adaptation to strength training",
            (
                "Seventeen trained male students volunteered for a strength training "
                "study with repeated cold-water immersion after training sessions; "
                "strength outcomes improved at retention."
            ),
        ),
    ]

    candidate = mine_insights(
        hits,
        topic="cold water immersion resistance training adaptation",
        required_anchor_terms=("training",),
    )[0]

    assert {(card.design, card.population, card.support_type) for card in candidate.claim_cards} == {
        ("randomized_trial", "human", "direct"),
        ("intervention_study", "human", "direct"),
    }


def test_miner_adds_receipt_backed_evidence_graph_context() -> None:
    hits = [
        _hit(
            "promise",
            "Resveratrol exercise training mechanism improves mitochondrial capacity",
            "Mouse mechanism paper reported resveratrol improved exercise training mitochondrial function.",
        ),
        _hit(
            "outcome",
            "Resveratrol exercise training trial blunted adaptation",
            "Randomized human outcome trial observed resveratrol reduced exercise training adaptation.",
        ),
        _hit(
            "consensus",
            "Systematic review of resveratrol exercise training adaptation",
            "Systematic review summarized resveratrol exercise training adaptation evidence.",
        ),
    ]

    candidate = mine_insights(
        hits,
        topic="resveratrol exercise training adaptation",
        required_anchor_terms=("resveratrol",),
    )[0]
    memo = render_memo(candidate, bind_receipts(candidate, hits))

    assert candidate.receipt_ids == ("promise", "outcome", "consensus")
    assert [node.role for node in candidate.evidence_graph] == ["primary", "counter", "consensus"]
    assert "`consensus`: consensus" in memo
    assert "Evidence graph" in memo


def test_evidence_bundle_promotes_late_direct_rct_before_writer() -> None:
    topic = "cold water immersion resistance training adaptation"
    graph = (
        EvidenceNode("proxy", "primary", "candidate evidence stream"),
        EvidenceNode("soccer", "counter", "candidate evidence stream"),
        EvidenceNode("review", "consensus", "consensus context"),
        EvidenceNode("rct", "replication", "replication context"),
    )
    cards = (
        ClaimCard(
            "proxy",
            "negative_signal",
            "intervention_study",
            "human",
            "muscle swelling",
            "negative",
            "direct",
            "high",
            "Cold-water immersion attenuated muscle swelling after resistance training.",
        ),
        ClaimCard(
            "soccer",
            "null_signal",
            "intervention_study",
            "human",
            "soccer performance",
            "null",
            "direct",
            "high",
            "Soccer players did not improve long-term performance adaptation.",
        ),
        ClaimCard(
            "review",
            "consensus",
            "synthesis",
            "human",
            "adaptation",
            "negative",
            "indirect",
            "medium",
            "Review context only.",
        ),
        ClaimCard(
            "rct",
            "replication",
            "randomized_trial",
            "human",
            "resistance training adaptation",
            "negative",
            "direct",
            "high",
            "Randomized human trial tested cold-water immersion after strength training.",
        ),
    )

    sorted_graph, sorted_cards = _prioritize_evidence_bundle(topic, graph, cards)

    assert [node.receipt_id for node in sorted_graph] == ["rct", "proxy", "soccer", "review"]
    assert sorted_graph[0].role == "primary"
    assert [card.receipt_id for card in sorted_cards] == ["rct", "proxy", "soccer", "review"]
    assert sorted_cards[0].role == "negative_signal"

    receipts = [
        _hit("proxy", "Cold-water immersion muscle swelling proxy", cards[0].quote),
        _hit("soccer", "Cold-water immersion soccer performance", cards[1].quote),
        _hit("review", "Cold-water immersion adaptation review", cards[2].quote),
        _hit("rct", "Does Cold-Water Immersion After Strength Training Attenuate Training Adaptation?", cards[3].quote),
    ]
    candidate = InsightCandidate(
        topic=topic,
        thesis="Cold-water immersion evidence is strongest for resistance-training adaptation.",
        bridge_terms=("cold", "immersion", "training"),
        tension_terms=("negative", "null"),
        receipt_ids=tuple(node.receipt_id for node in sorted_graph),
        score=100,
        novelty_score=80,
        evidence_score=96,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=sorted_cards,
        evidence_graph=sorted_graph,
    )

    bound = bind_receipts(candidate, receipts)
    prompt = build_minimax_prompt(candidate, bound)

    assert bound[0].hit_id == "rct"
    assert "Receipt 1\nID: 10.rct\nTitle: Does Cold-Water Immersion" in prompt
    assert "rct: primary (strongest direct human evidence)" in prompt


def test_evidence_bundle_does_not_promote_weak_late_context() -> None:
    graph = (
        EvidenceNode("primary", "primary", "candidate evidence stream"),
        EvidenceNode("counter", "counter", "candidate evidence stream"),
        EvidenceNode("review", "consensus", "consensus context"),
    )
    cards = (
        ClaimCard("primary", "positive_signal", "intervention_study", "human", "performance", "positive", "direct", "high", "Direct trial."),
        ClaimCard("counter", "negative_signal", "intervention_study", "human", "performance", "negative", "direct", "high", "Direct trial."),
        ClaimCard("review", "consensus", "synthesis", "human", "performance", "negative", "indirect", "medium", "Review only."),
    )

    sorted_graph, sorted_cards = _prioritize_evidence_bundle("training adaptation", graph, cards)

    assert sorted_graph == graph
    assert sorted_cards == cards


def test_evidence_graph_rejects_off_modality_context_receipts() -> None:
    topic = "cold water immersion resistance training adaptation"
    hits = [
        _hit(
            "null",
            "Cold Water Immersion and Contrast Water Therapy Do Not Improve Short-Term Recovery Following Resistance Training",
            "Participants performed resistance training. Cold water immersion did not improve short-term recovery after resistance training.",
        ),
        _hit(
            "attenuate",
            "Post-exercise cold water immersion attenuates acute anabolic signalling and long-term adaptations in muscle to strength training",
            "Cold water immersion attenuated long-term adaptations to strength training.",
        ),
        _hit(
            "cycling",
            "Post-exercise Warm or Cold Water Immersion to Augment the Cardiometabolic Benefits of Exercise Training",
            (
                "Warm or cold water immersion would provide similar or greater benefits. "
                "Long-term exercise training work trial distance increased without differences. "
                "Substituting with cold water immersion does not."
            ),
        ),
        _hit(
            "meta",
            "Effects of post-exercise cold-water immersion on resistance training-induced gains in muscular strength: a meta-analysis",
            "Systematic review of cold water immersion after resistance training adaptation outcomes.",
        ),
    ]

    candidate = mine_insights(hits, topic=topic, required_anchor_terms=query_anchor_terms([topic]))[0]

    assert "cycling" not in candidate.receipt_ids
    assert [node.receipt_id for node in candidate.evidence_graph] == ["null", "attenuate"]


def test_two_receipts_do_not_inflate_evidence_without_support_quality() -> None:
    score = score_connection(
        bridge_terms=("foo", "bar"),
        bridge_doc_counts={"foo": 2, "bar": 2},
        unique_source_count=2,
        receipt_count=2,
        has_tension=False,
    )

    assert score.evidence_score < 85
    assert score.scorecard["evidence_directness"] == score.evidence_score
    assert score.scorecard["directional_contrast"] == 25
    assert "thin_claim_support" in score.reasons


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


def test_binder_dedupes_duplicate_titles_and_preserves_distinct_sources() -> None:
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
        CorpusHit(
            hit_id="c",
            title="Human Oversight of Automated Review Comments",
            abstract="A distinct study measured human oversight of automated comments.",
            source="pubmed",
            doi="10.1000/distinct",
        ),
    ]
    candidate = InsightCandidate(
        topic="AI review",
        thesis="AI review may have a trust bridge.",
        bridge_terms=("review", "comments"),
        tension_terms=("positive", "negative"),
        receipt_ids=("a", "b", "c"),
        score=80,
        novelty_score=80,
        evidence_score=80,
        reasons=("source_diverse",),
    )

    assert [hit.hit_id for hit in bind_receipts(candidate, hits)] == ["a", "c"]


def test_pipeline_prunes_title_duplicates_from_candidate_and_public_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            abstract="A second index entry for the same review-comments paper.",
            source="acm",
            doi="10.5555/2486788.2486915",
        ),
        CorpusHit(
            hit_id="c",
            title="Human Oversight of Automated Review Comments",
            abstract="Human oversight reduced incorrect automated review comments.",
            source="pubmed",
            doi="10.1000/distinct",
        ),
    ]
    candidate = InsightCandidate(
        topic="AI review",
        thesis="Automated review may benefit from human oversight.",
        bridge_terms=("review", "comments"),
        tension_terms=("accepted", "incorrect"),
        receipt_ids=("a", "b", "c"),
        score=90,
        novelty_score=90,
        evidence_score=90,
        reasons=("tier:publishable_alpha",),
        receipt_roles=(
            ReceiptRole("a", "primary", "direct evidence"),
            ReceiptRole("b", "context", "duplicate index entry"),
            ReceiptRole("c", "counter", "distinct boundary evidence"),
        ),
        claim_cards=(
            ClaimCard("a", "positive_signal", "intervention_study", "human", "review", "positive", "direct", "high", "Review Bot: Automatic Code Review Tool. Developers accepted automated review comments."),
            ClaimCard("b", "positive_signal", "intervention_study", "human", "review", "positive", "direct", "high", "Review Bot Automatic Code Review Tool. A second index entry for the same review-comments paper."),
            ClaimCard("c", "negative_signal", "intervention_study", "human", "review", "negative", "direct", "high", "Human Oversight of Automated Review Comments. Human oversight reduced incorrect automated review comments."),
        ),
        evidence_graph=(
            EvidenceNode("a", "primary", "direct evidence"),
            EvidenceNode("b", "context", "duplicate index entry"),
            EvidenceNode("c", "counter", "distinct boundary evidence"),
        ),
    )
    gated_candidates: list[InsightCandidate] = []

    def allow_publish_quality(candidate: InsightCandidate) -> None:
        gated_candidates.append(candidate)
        return None

    monkeypatch.setattr(
        "v5_memo.pipeline.candidate_publish_blocker",
        allow_publish_quality,
    )
    monkeypatch.setattr(
        "v5_memo.pipeline.mine_insights",
        lambda *_args, **_kwargs: [candidate],
    )

    result = build_alpha_memo(
        topic=candidate.topic,
        seed_queries=["AI review"],
        anchor_queries=(),
        searcher=_StaticSearch(hits),
        require_publish_quality=True,
    )
    payload = build_researka_payload(
        result,
        author_agent_id="v5-memo-agent",
        domain_slug="technology",
    )

    assert result.candidate.receipt_ids == ("a", "c")
    assert [card.receipt_id for card in result.candidate.claim_cards] == ["a", "c"]
    assert [role.receipt_id for role in result.candidate.receipt_roles] == ["a", "c"]
    assert [node.receipt_id for node in result.candidate.evidence_graph] == ["a", "c"]
    assert gated_candidates
    assert all(gated.receipt_ids == ("a", "c") for gated in gated_candidates)
    assert all(
        [card.receipt_id for card in gated.claim_cards] == ["a", "c"]
        for gated in gated_candidates
    )
    assert "`b`:" not in result.markdown
    assert "10.5555/2486788.2486915" not in result.markdown
    assert cast(dict[str, object], payload["metadata"])["receipt_ids"] == ["a", "c"]
    assert "10.5555/2486788.2486915" not in json.dumps(payload, sort_keys=True)


def test_collect_seed_hits_dedupes_across_seed_queries() -> None:
    def search(query: str, limit: int) -> Sequence[CorpusHit]:
        del limit
        return [
            CorpusHit("shared", "Shared NAD salvage hit", query, "researka", doi="10.shared"),
            CorpusHit(query, f"Unique {query}", "mitochondrial bridge", "researka", doi=f"10.{query}"),
        ]

    hits = collect_seed_hits(_FunctionSearch(search), ["nad", "mitochondrial"], per_query_limit=2)

    assert [hit.hit_id for hit in hits] == ["shared", "nad", "mitochondrial"]
    assert hits[0].metadata["seed_queries"] == ("nad", "mitochondrial")


def test_collect_seed_hits_balances_planned_query_budget() -> None:
    hits = collect_seed_hits(
        _FunctionSearch(lambda query, limit: [
            CorpusHit(f"{query}-{index}", f"{query} hit {index}", "receipt", "fullraw", doi=f"10.{query}/{index}")
            for index in range(limit)
        ]),
        ["noisy-first-query", "later-reversal-query"],
        per_query_limit=10,
        max_hits=6,
    )

    assert any(hit.hit_id.startswith("later-reversal-query") for hit in hits)


def test_collect_seed_hits_dedupes_near_duplicate_seed_queries() -> None:
    searched: list[tuple[str, int]] = []

    def search(query: str, limit: int) -> Sequence[CorpusHit]:
        searched.append((query, limit))
        return [
            CorpusHit(f"{query}-{index}", f"{query} hit {index}", "receipt", "fullraw")
            for index in range(limit)
        ]

    hits = collect_seed_hits(
        _FunctionSearch(search),
        [
            "urolithin A mitochondrial aging",
            "urolithin mitochondrial aging",
            "urolithin human trial",
        ],
        per_query_limit=10,
        max_hits=3,
    )

    assert searched == [
        ("urolithin A mitochondrial aging", 2),
        ("urolithin human trial", 2),
    ]
    assert len(hits) == 3


def test_collect_seed_hits_skips_late_seed_failure_after_hits() -> None:
    def search(query: str, limit: int) -> Sequence[CorpusHit]:
        if query == "bad" and limit:
            raise RuntimeError("backend miss")
        return [_hit(query, f"{query} title", "receipt")]

    searcher = _FunctionSearch(search)
    assert [hit.hit_id for hit in collect_seed_hits(searcher, ["good", "bad"])] == ["good"]
    assert [hit.hit_id for hit in collect_seed_hits(searcher, ["bad", "good"])] == ["good"]
    assert collect_seed_hits(searcher, ["bad"]) == []


def test_collect_seed_hits_propagates_fullraw_coverage_failure() -> None:
    searched: list[str] = []

    def search(query: str, limit: int) -> Sequence[CorpusHit]:
        del limit
        searched.append(query)
        raise RuntimeError("Full raw corpus search coverage too narrow: {'shards_searched': 32}")

    with pytest.raises(RuntimeError, match="coverage too narrow"):
        collect_seed_hits(_FunctionSearch(search), ["metformin", "metformin blunts"])

    assert searched == ["metformin"]


def test_collect_seed_hits_skips_stopped_no_hit_fullraw_shape() -> None:
    def search(query: str, limit: int) -> Sequence[CorpusHit]:
        del limit
        if query == "dead":
            raise RuntimeError(
                "Full raw corpus search coverage too narrow: "
                "{'shards_searched': 128, 'sweep_stopped_no_hits': True}"
            )
        return [_hit(query, f"{query} title", "full receipt evidence")]

    assert [hit.hit_id for hit in collect_seed_hits(_FunctionSearch(search), ["dead", "good"])] == ["good"]


def test_collect_seed_hits_skips_late_fullraw_coverage_failure_after_hits() -> None:
    searched: list[str] = []

    def search(query: str, limit: int) -> Sequence[CorpusHit]:
        del limit
        searched.append(query)
        if "augment" in query:
            raise RuntimeError("Full raw corpus search coverage too narrow: {'shards_searched': None}")
        return [_hit(query, f"{query} title", "full receipt evidence")]

    hits = collect_seed_hits(
        _FunctionSearch(search),
        [
            "metformin exercise training adaptation",
            "metformin augment exercise training protocol",
            "metformin blunts exercise training",
        ],
    )

    assert [hit.hit_id for hit in hits] == ["metformin exercise training adaptation"]
    assert searched == [
        "metformin exercise training adaptation",
        "metformin augment exercise training protocol",
    ]


def test_pipeline_builds_best_memo() -> None:
    result = build_alpha_memo(
        topic="longevity resilience",
        seed_queries=["sleep nad", "exercise nad"],
        searcher=_StaticSearch(_hits()),
    )

    assert result.candidate.score >= 60
    assert len(result.receipts) == 2
    assert result.markdown.startswith("# Alpha memo")


def test_pipeline_anchors_cover_late_planned_query_angles() -> None:
    def search(query: str, limit: int) -> Sequence[CorpusHit]:
        del limit
        if "resveratrol" not in query:
            return []
        return [
            CorpusHit("promise", "Resveratrol improves mitochondrial function and exercise performance", "Resveratrol activated a mitochondrial mechanism and improved running performance.", "openalex", doi="10.promise"),
            CorpusHit("outcome", "Resveratrol exercise training trial blunted cardiovascular adaptation", "Randomized trial observed resveratrol blunted exercise training adaptation.", "semantic_scholar", doi="10.outcome"),
        ]

    result = build_alpha_memo(
        topic="longevity exercise adaptation supplement reversal",
        seed_queries=[
            "rapamycin mTOR exercise adaptation blunted",
            "NMN NAD skeletal muscle attenuation",
            "resveratrol exercise training adaptation blunted",
        ],
        searcher=_FunctionSearch(search),
        min_alpha_tier="elite_alpha",
    )

    assert result.candidate.receipt_ids == ("promise", "outcome")


def test_miner_accepts_direct_human_reversal_when_anchor_is_full_text_only() -> None:
    hits = [
        _hit(
            "promise",
            "Protocol expected augmentation",
            (
                "Randomized human trial protocol expected metformin to augment "
                "resistance training adaptation."
            ),
        ),
        _hit(
            "outcome",
            "Outcome observed null response",
            (
                "Randomized human trial protocol observed metformin made no difference "
                "to resistance training adaptation."
            ),
        ),
    ]

    candidates = mine_insights(
        hits,
        topic="metformin resistance training adaptation",
        required_anchor_terms=query_anchor_terms(["metformin resistance training adaptation"]),
    )

    assert candidates
    assert candidates[0].receipt_ids == ("promise", "outcome")
    assert tuple(role.role for role in candidates[0].receipt_roles) == ("positive_signal", "null_signal")
    assert candidate_publish_blocker(candidates[0]) is None


def test_miner_rejects_partial_multiword_entity_anchor() -> None:
    topic = "beta alanine resistance training older adults trial"
    hits = [
        _hit(
            "beta-alanine",
            "Beta-Alanine protocol expected resistance training augmentation",
            "Randomized human trial expected beta-alanine to augment resistance training performance in older adults.",
        ),
        _hit(
            "hmb",
            "Beta-hydroxy-beta-methylbutyrate resistance training trial blunted strength",
            "Randomized human trial observed beta-hydroxy-beta-methylbutyrate blunted resistance training performance in older women.",
        ),
        _hit(
            "amyloid",
            "Amyloid beta resistance training trial negative performance",
            "Animal intervention study resistance training reduced amyloid beta and negative performance.",
        ),
    ]

    candidates = mine_insights(
        hits,
        topic=topic,
        required_anchor_terms=query_anchor_terms([topic]),
        include_discovery=True,
    )

    assert candidates == []


def test_pipeline_stops_retrieval_once_publishable_candidate_exists() -> None:
    calls: list[str] = []
    hits = [
        _hit("promise", "Metformin protocol expected training augmentation", "Metformin was expected to augment resistance training adaptation."),
        _hit("outcome", "Metformin trial observed training blunting", "Metformin blunted resistance training adaptation."),
    ]

    def search(query: str, limit: int) -> Sequence[CorpusHit]:
        del limit
        calls.append(query)
        assert query != "slow"
        return hits

    result = build_alpha_memo(
        topic="metformin training",
        seed_queries=["metformin training", "slow"],
        searcher=_FunctionSearch(search),
        min_alpha_tier="elite_alpha",
    )

    assert calls == ["metformin training"]
    assert result.candidate.receipt_ids == ("promise", "outcome")


def test_publish_quality_pipeline_propagates_incomplete_late_shape_coverage() -> None:
    calls: list[str] = []

    def search(query: str, limit: int) -> Sequence[CorpusHit]:
        del limit
        calls.append(query)
        if "augment" in query:
            raise RuntimeError("Full raw corpus search coverage too narrow: {'shards_searched': None}")
        return [_hit("weak", "Metformin resistance training review", "Review evidence.")]

    with pytest.raises(RuntimeError, match="coverage too narrow"):
        build_alpha_memo(
            topic="metformin resistance training adaptation",
            seed_queries=[
                "metformin resistance training adaptation",
                "metformin augment resistance training protocol",
            ],
            searcher=_FunctionSearch(search),
            min_alpha_tier="publishable_alpha",
            require_publish_quality=True,
        )

    assert calls == [
        "metformin resistance training adaptation",
        "metformin augment resistance training protocol",
    ]


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

    result = build_alpha_memo(
        topic="resveratrol exercise adaptation",
        seed_queries=["arabidopsis tor cotyledon greening"],
        anchor_queries=["resveratrol exercise training adaptation"],
        searcher=_StaticSearch(hits),
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

    result = build_alpha_memo(
        topic="longevity exercise adaptation supplement reversal",
        seed_queries=["arabidopsis tor cotyledon greening"],
        anchor_queries=["longevity exercise adaptation intervention reversal"],
        searcher=_StaticSearch(hits),
    )

    assert result.candidate.receipt_ids == ("promise", "outcome")


def test_pipeline_accepts_custom_memo_writer() -> None:
    result = build_alpha_memo(
        topic="longevity resilience",
        seed_queries=["sleep nad", "exercise nad"],
        searcher=_StaticSearch(_hits()),
        memo_writer=lambda candidate, receipts: f"custom: {candidate.topic} / {len(receipts)}",
    )

    assert result.markdown == "custom: longevity resilience / 2"


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

    def fake_mine(*_args: object, **_kwargs: object) -> list[InsightCandidate]:
        return [low, elite]

    def selector(candidates: Sequence[InsightCandidate], _hits: Sequence[CorpusHit]) -> Sequence[InsightCandidate]:
        seen.extend(candidates)
        return list(candidates)

    monkeypatch.setattr("v5_memo.pipeline.mine_insights", fake_mine)

    result = build_alpha_memo(
        topic="tool reliability",
        seed_queries=["tool"],
        searcher=_StaticSearch(hits),
        memo_selector=selector,
        min_alpha_tier="elite_alpha",
    )

    assert seen == [elite]
    assert result.candidate == elite


def test_pipeline_selector_slate_surfaces_diverse_alpha_shapes() -> None:
    def candidate(left: str, right: str, score: int, shape: str) -> InsightCandidate:
        return InsightCandidate(
            topic="tool reliability",
            thesis=shape,
            bridge_terms=("tool", "reliability"),
            tension_terms=("positive", "negative"),
            receipt_ids=(left, right),
            score=score,
            novelty_score=90,
            evidence_score=90,
            reasons=(shape, "tier:elite_alpha"),
        )

    duplicates = [
        candidate(f"dup-{idx}-a", f"dup-{idx}-b", 99 - idx, "shape:measurement_mismatch")
        for idx in range(10)
    ]
    reversal = candidate("rev-a", "rev-b", 80, "shape:expectation_reversal")
    ranked = [*duplicates, reversal]

    assert ranked.index(reversal) == 10
    assert list(_selector_slate(ranked)).index(reversal) == 3


def test_quality_fallbacks_keep_original_miner_rank_after_selector_choice() -> None:
    def candidate(index: int, shape: str = "") -> InsightCandidate:
        return InsightCandidate(
            topic="tool reliability",
            thesis=f"candidate-{index}",
            bridge_terms=("tool", "reliability"),
            tension_terms=("positive", "negative"),
            receipt_ids=(f"{index}-a", f"{index}-b"),
            score=100 - index,
            novelty_score=90,
            evidence_score=90,
            reasons=tuple(value for value in (shape, "tier:elite_alpha") if value),
        )

    ranked = [candidate(index) for index in range(7)]
    selected = candidate(7, "shape:expectation_reversal")
    ranked.insert(4, selected)

    ordered = _apply_selector(
        ranked,
        (),
        lambda slate, _hits: [selected] if selected in slate else [],
        preserve_fallbacks=True,
    )

    assert list(ordered) == [selected, *(candidate for candidate in ranked if candidate != selected)]


def test_pipeline_mines_broader_slate_when_selector_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_mine(*_args: object, **kwargs: object) -> list[InsightCandidate]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr("v5_memo.pipeline.mine_insights", fake_mine)

    with pytest.raises(MemoBuildError):
        build_alpha_memo(
            topic="longevity resilience",
            seed_queries=["sleep nad", "exercise nad"],
            searcher=_StaticSearch(_hits()),
            memo_selector=lambda candidates, _hits: candidates,
        )

    assert captured["max_candidates"] == 30


def test_pipeline_filters_primary_anchor_drift_before_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    hits = [
        _hit("bad-a", "Mitochondrial DNA depletion in childhood muscle", "mtDNA depletion caused myopathy."),
        _hit("bad-b", "Mitochondrial DNA mutator mice show oxidative aging", "mtDNA mutator mice increased hydrogen peroxide."),
        _hit("good-a", "Metformin protocol expected training augmentation", "Metformin was expected to augment training."),
        _hit("good-b", "Metformin outcome observed training blunting", "Metformin blunted resistance training adaptation."),
    ]
    bad = InsightCandidate(
        topic="metformin exercise training mitochondrial adaptation",
        thesis="Drift.",
        bridge_terms=("mitochondrial", "dna"),
        tension_terms=("positive", "negative"),
        receipt_ids=("bad-a", "bad-b"),
        score=99,
        novelty_score=99,
        evidence_score=99,
        reasons=("tier:elite_alpha",),
    )
    good = InsightCandidate(
        topic="metformin exercise training mitochondrial adaptation",
        thesis="Anchored.",
        bridge_terms=("metformin", "training"),
        tension_terms=("positive", "negative"),
        receipt_ids=("good-a", "good-b"),
        score=80,
        novelty_score=80,
        evidence_score=80,
        reasons=("tier:elite_alpha",),
    )
    seen: list[InsightCandidate] = []

    def selector(candidates: Sequence[InsightCandidate], _hits: Sequence[CorpusHit]) -> Sequence[InsightCandidate]:
        seen.extend(candidates)
        return list(candidates)

    monkeypatch.setattr("v5_memo.pipeline.mine_insights", lambda *_args, **_kwargs: [bad, good])

    result = build_alpha_memo(
        topic="metformin exercise training mitochondrial adaptation",
        seed_queries=["metformin exercise training mitochondrial adaptation"],
        searcher=_StaticSearch(hits),
        memo_selector=selector,
        min_alpha_tier="elite_alpha",
    )

    assert seen == [good]
    assert result.candidate == good


@pytest.mark.parametrize(
    ("topic", "primary"),
    [
        ("GlyNAC aging glutathione older adults deficiency", "GlyNAC"),
        ("sodium bicarbonate exercise performance fatigue lactate", "sodium"),
        ("earnings guidance analyst forecast accuracy", "earnings"),
    ],
)
def test_pipeline_primary_anchor_guard_is_topic_agnostic(
    monkeypatch: pytest.MonkeyPatch,
    topic: str,
    primary: str,
) -> None:
    hits = [
        _hit("a", f"{primary} protocol expected improvement", f"{primary} expected improvement."),
        _hit("b", f"{primary} outcome observed blunting", f"{primary} observed blunting."),
    ]
    candidate = InsightCandidate(
        topic=topic,
        thesis="Anchored.",
        bridge_terms=(primary.casefold().rstrip("s"),),
        tension_terms=("positive", "negative"),
        receipt_ids=("a", "b"),
        score=80,
        novelty_score=80,
        evidence_score=80,
        reasons=("tier:publishable_alpha",),
    )

    monkeypatch.setattr("v5_memo.pipeline.mine_insights", lambda *_args, **_kwargs: [candidate])

    result = build_alpha_memo(topic=topic, seed_queries=[topic], searcher=_StaticSearch(hits))

    assert result.candidate == candidate


def test_pipeline_selector_cannot_veto_with_invented_receipt_pair() -> None:
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

    result = build_alpha_memo(
        topic="longevity resilience",
        seed_queries=["sleep nad", "exercise nad"],
        searcher=_StaticSearch(_hits()),
        memo_selector=lambda _candidates, _hits: [invented],
    )

    assert result.candidate.receipt_ids == ("h1", "h2")


def test_pipeline_selector_empty_choice_fails_closed() -> None:
    with pytest.raises(MemoBuildError, match="no receipt-bound") as exc:
        build_alpha_memo(
            topic="longevity resilience",
            seed_queries=["sleep nad", "exercise nad"],
            searcher=_StaticSearch(_hits()),
            memo_selector=lambda _candidates, _hits: [],
    )

    assert exc.value.failure.details["candidate_count"] == 0
    mined_count = exc.value.failure.details["mined_candidate_count"]
    assert isinstance(mined_count, int)
    assert mined_count > 0


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


def test_query_anchor_terms_drop_broad_population_words() -> None:
    assert query_anchor_terms(["metformin exercise training adaptation older adults trial"]) == (
        "metformin",
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


def test_mechanism_promise_without_shared_axis_is_not_elite() -> None:
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
    assert candidate_alpha_tier(candidate) == "publishable_alpha"


def test_metformin_resistance_pair_is_not_elite_without_training_context() -> None:
    hits = [
        _hit("masters", "Metformin blunts muscle hypertrophy in response to progressive resistance exercise training", "Metformin reduced the hypertrophic response to resistance training."),
        _hit("bipolar", "Metformin reversal of insulin resistance improves outcomes in bipolar disorder", "Metformin reversal of insulin resistance improved outcomes in bipolar disorder."),
    ]
    assert mine_insights(
        hits,
        topic="metformin resistance training adaptation",
        required_anchor_terms=("metformin",),
    ) == []


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


def test_miner_rejects_training_pair_when_only_population_anchor_matches() -> None:
    topic = "metformin exercise training adaptation older adults trial"
    hits = [
        _hit(
            "exercise",
            "Metformin impairs cardiorespiratory fitness adaptation to high-intensity power training in older adults",
            "Randomized human trial in older adults with T2D observed metformin impaired VO2peak adaptation during high-intensity power training.",
        ),
        _hit(
            "frailty",
            "A two-year trial of metformin to reduce frailty in older adults with glucose intolerance",
            "Randomized human trial in older adults with glucose intolerance found a significant reduction in frailty progression rate with metformin treatment.",
        ),
    ]

    assert mine_insights(
        hits,
        topic=topic,
        required_anchor_terms=query_anchor_terms([topic]),
        include_discovery=True,
    ) == []


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


def test_claim_card_prefers_directional_title_over_mixed_endpoint_abstract() -> None:
    hit = CorpusHit(
        hit_id="mixed-endpoints",
        title="Intervention attenuates metabolic adaptation after training",
        abstract=(
            "Some endpoints increased, others did not change, and the intervention "
            "attenuated the primary adaptation."
        ),
        source="fullraw:pubmed",
    )

    card = _claim_card(hit, ReceiptRole(hit.hit_id, "negative_signal", "primary endpoint"))

    assert card.direction == "negative"


def test_researka_payload_preserves_memo_and_receipts() -> None:
    candidate = InsightCandidate(
        topic="resveratrol exercise adaptation",
        thesis="Resveratrol may reverse on training.",
        bridge_terms=("resveratrol",),
        tension_terms=("blunted",),
        receipt_ids=("a", "b"),
        score=92,
        novelty_score=90,
        evidence_score=94,
        reasons=("shape:directional_reversal",),
    )
    receipts = [
        _hit("a", "Resveratrol improves mitochondrial function", "Animal mechanism improved capacity."),
        _hit("b", "Resveratrol blunts exercise training", "Human trial blunted adaptation."),
    ]
    markdown = render_memo(candidate, receipts)

    payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown=markdown),
        author_agent_id="v5-alpha",
        domain_slug="longevity",
    )

    assert payload["article_type"] == "alpha_memo"
    body = cast(str, payload["body_markdown"])
    assert "**Seed hypothesis:**" in body
    assert "**Receipts:**" in body
    assert "**Claim ledger:**" not in body
    assert payload["source_bundle"][0]["evidence_type"] == "primary"  # type: ignore[index]


def test_researka_payload_binds_revision_parent() -> None:
    candidate = InsightCandidate(
        topic="revision topic",
        thesis="Revised bounded signal.",
        bridge_terms=("revision", "signal"),
        tension_terms=("negative", "null"),
        receipt_ids=("a", "b"),
        score=90,
        novelty_score=60,
        evidence_score=90,
        reasons=("shape:directional_reversal",),
    )
    receipts = [_hit("a", "Revision trial A", "Negative result."), _hit("b", "Revision trial B", "Null result.")]

    payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown="# Alpha memo: Revised bounded signal"),
        author_agent_id="v5-memo-agent",
        domain_slug="performance",
        parent_submission_id=" parent-submission ",
    )

    assert payload["parent_submission_id"] == "parent-submission"


def test_researka_payload_drops_dangling_clipped_title_subtitle() -> None:
    candidate = InsightCandidate(
        topic="training adaptation",
        thesis="A strong bounded adaptation signal.",
        bridge_terms=("training", "adaptation"),
        tension_terms=("blunted",),
        receipt_ids=("long", "short"),
        score=92,
        novelty_score=90,
        evidence_score=94,
        reasons=("shape:directional_reversal",),
    )
    receipts = [
        _hit(
            "long",
            "Metformin blunts muscle hypertrophy in response to progressive resistance exercise training in older adults: A randomized trial",
            "Human trial reported a bounded adaptation signal.",
        ),
        _hit("short", "Training adaptation response", "Comparator adaptation receipt."),
    ]
    markdown = (
        "# Alpha memo: Metformin blunts muscle hypertrophy in response to progressive resistance exercise training "
        "in older adults: A randomized trial\n\n**Alpha hypothesis:** A strong bounded adaptation signal."
    )

    payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown=markdown),
        author_agent_id="v5-alpha",
        domain_slug="longevity",
    )

    assert payload["title"] == "Metformin blunts muscle hypertrophy in response to progressive resistance exercise training in older adults"
    assert cast(str, payload["body_markdown"]).startswith(
        "# Alpha memo: Metformin blunts muscle hypertrophy in response to progressive resistance exercise training in older adults"
    )


def test_researka_payload_uses_valid_doi_or_pmid_not_empty_doi() -> None:
    candidate = InsightCandidate(
        topic="caffeine exercise",
        thesis="Endpoint-gated caffeine signal.",
        bridge_terms=("caffeine", "exercise"),
        tension_terms=("negative", "positive"),
        receipt_ids=("10.1000/caffeine", "1798317"),
        score=100,
        novelty_score=50,
        evidence_score=85,
        reasons=("shape:directional_reversal",),
    )
    receipts = [
        CorpusHit(
            hit_id="10.1000/caffeine",
            title="Failure of caffeine to affect metabolism during 60 min submaximal exercise.",
            abstract="Caffeine did not change submaximal metabolic endpoints.",
            source="researka",
            doi="10.1000/caffeine",
        ),
        CorpusHit(
            hit_id="1798317",
            title="Caffeine ingestion during exercise to exhaustion in elite distance runners.",
            abstract="Caffeine improved exercise to exhaustion.",
            source="researka",
            metadata={"pmid": "1798317"},
        ),
    ]

    payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown=render_memo(candidate, receipts)),
        author_agent_id="v6-alpha-memo",
        domain_slug="performance",
    )

    source_bundle = cast(list[dict[str, object]], payload["source_bundle"])
    assert source_bundle[0]["doi"] == "10.1000/caffeine"
    assert "doi" not in source_bundle[1]
    assert source_bundle[1]["pmid"] == "1798317"
    assert source_bundle[1]["id"] == "1798317"


def test_researka_payload_submits_human_title_and_plain_doi_citations() -> None:
    candidate = InsightCandidate(
        topic="caffeine exercise",
        thesis="Endpoint-gated caffeine signal.",
        bridge_terms=("caffeine", "exercise"),
        tension_terms=("negative", "positive"),
        receipt_ids=("10.1000/caffeine", "10.1000/runners"),
        score=100,
        novelty_score=50,
        evidence_score=85,
        reasons=("shape:directional_reversal",),
    )
    receipts = [
        CorpusHit(
            hit_id="10.1000/caffeine",
            title="Failure of caffeine to affect metabolism during 60 min submaximal exercise.",
            abstract="Caffeine did not change submaximal metabolic endpoints.",
            source="researka",
            doi="10.1000/caffeine",
        ),
        CorpusHit(
            hit_id="10.1000/runners",
            title="Caffeine ingestion during exercise to exhaustion in elite distance runners.",
            abstract="Caffeine improved exercise to exhaustion.",
            source="researka",
            doi="10.1000/runners",
        ),
    ]

    markdown = "\n".join(
        [
            "# Alpha memo: caffeine / exercise",
            "",
            "**Alpha hypothesis:** Endpoint-gated caffeine signal.",
            "",
            "**Receipts:**",
            "- `10.1000/caffeine`: Failure of caffeine to affect metabolism.",
            "- `10.1000/runners`: Caffeine ingestion during exercise.",
        ]
    )

    payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown=markdown),
        author_agent_id="v5-alpha",
        domain_slug="performance",
    )
    body = cast(str, payload["body_markdown"])

    assert payload["title"] == "Endpoint-gated caffeine signal."
    assert body.startswith("# Alpha memo: Endpoint-gated caffeine signal.")
    assert "Hypothesis-level alpha signal; not clinical advice." in body
    assert "`10.1000/caffeine`" not in body
    assert "`10.1000/runners`" not in body
    assert "10.1000/caffeine:" not in body
    assert "10.1000/runners:" not in body
    assert "10.1000/caffeine" in body
    assert "10.1000/runners" in body


def test_researka_payload_rejects_non_primary_receipts_before_intake() -> None:
    candidate = InsightCandidate(
        topic="nicotinamide riboside exercise performance",
        thesis="Nicotinamide riboside may expose a species and endpoint boundary in exercise performance.",
        bridge_terms=("nicotinamide", "exercise"),
        tension_terms=("negative", "positive"),
        receipt_ids=("supplement", "faseb", "primary"),
        score=90,
        novelty_score=60,
        evidence_score=82,
        reasons=("shape:directional_reversal",),
    )
    receipts = [
        CorpusHit(
            hit_id="supplement",
            title="Additional file 1: of The NAD+ precursor nicotinamide riboside decreases exercise performance in rats",
            abstract="Figshare-hosted supporting data for the nicotinamide riboside exercise study.",
            source="fullraw:openalex",
            doi="10.6084/m9.figshare.c.3601490_d1",
        ),
        CorpusHit(
            hit_id="faseb",
            title="Nicotinamide riboside and exercise performance in healthy volunteers",
            abstract="Conference abstract reported human exercise performance markers.",
            source="fullraw:openalex",
            doi="10.1096/fasebj.2021.35.s1.05282",
            venue="The FASEB Journal",
        ),
        CorpusHit(
            hit_id="jissn-supplement",
            title="Effects of resveratrol supplementation after eccentric exercise",
            abstract="Abstract supplement reported inflammatory markers in runners.",
            source="fullraw:openalex",
            doi="10.1186/1550-2783-8-s1-p15",
            venue="Journal of the International Society of Sports Nutrition",
        ),
        CorpusHit(
            hit_id="primary",
            title="Nicotinamide riboside supplementation and exercise performance in humans",
            abstract="Randomized human trial measured endurance exercise performance after supplementation.",
            source="fullraw:openalex",
            doi="10.1000/nr-human",
        ),
    ]
    markdown = "# Alpha memo: Additional file 1: of The NAD+ precursor nicotinamide riboside decreases exercise performance in rats\n\nBody."

    with pytest.raises(ValueError, match=r"unsupported_source_bundle_artifact_type:.*:supplemental"):
        build_researka_payload(
            MemoResult(candidate=candidate, receipts=receipts, markdown=markdown),
            author_agent_id="v5-memo-agent",
            domain_slug="longevity_research",
        )


def test_miner_excludes_non_primary_artifacts_before_candidate_building() -> None:
    hits = [
        CorpusHit(
            hit_id="supplement",
            title="Additional file 1: Metformin to augment muscle training adaptation",
            abstract="Protocol designed to augment exercise training adaptation in older adults.",
            source="fullraw:openalex",
            doi="10.6084/m9.figshare.c.1_d1",
        ),
        CorpusHit(
            hit_id="trial",
            title="Metformin blunts muscle training adaptation",
            abstract=(
                "Randomized human trial found metformin blunted exercise training adaptation "
                "in older adults."
            ),
            source="fullraw:pubmed",
            doi="10.1000/trial",
        ),
    ]

    candidates = mine_insights(
        hits,
        topic="metformin exercise training adaptation older adults trial",
        required_anchor_terms=("metformin",),
    )

    assert candidates == []


@pytest.mark.parametrize(
    ("title", "doi", "expected"),
    [
        ("Randomized exercise trial", "10.1000/trial", "article"),
        ("Additional file 1: trial checklist", "10.6084/m9.figshare.c.1_d1", "supplemental"),
        ("Exercise trial abstract", "10.1249/01.mss.0000000000000001", "conference_abstract"),
        ("Faculty Opinions recommendation of a trial", "10.3410/f.1", "secondary_commentary"),
    ],
)
def test_source_artifact_policy_is_shared(
    title: str,
    doi: str,
    expected: str,
) -> None:
    hit = CorpusHit(
        hit_id=doi,
        title=title,
        abstract="Evidence descriptor.",
        source="fullraw:test",
        doi=doi,
    )

    assert source_artifact_type(hit) == expected


def test_researka_payload_uses_receipt_title_instead_of_auto_bridge_thesis_title() -> None:
    candidate = InsightCandidate(
        topic="urolithin mitochondrial aging",
        thesis=(
            "urolithin mitochondrial aging may have a mitochondrial / improve bridge "
            "between urolithin provide cardioprotection and muscle function."
        ),
        bridge_terms=("mitochondrial", "improve"),
        tension_terms=("positive",),
        receipt_ids=("primary", "context"),
        score=84,
        novelty_score=70,
        evidence_score=80,
        reasons=("shape:boundary_condition",),
    )
    receipts = [
        CorpusHit(
            hit_id="primary",
            title="Urolithin A provides cardioprotection and improves human cardiovascular health biomarkers",
            abstract="Human supplementation improved a bounded cardiovascular biomarker.",
            source="fullraw:openalex",
            doi="10.1016/j.isci.2025.111814",
        ),
        CorpusHit(
            hit_id="context",
            title="Urolithin A improves mitochondrial health in osteoarthritis models",
            abstract="Mechanistic context receipt.",
            source="fullraw:openalex",
            doi="10.1111/acel.13662",
        ),
    ]
    markdown = "# Alpha memo: mitochondrial improve\n\nBody."

    payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown=markdown),
        author_agent_id="v5-memo-agent",
        domain_slug="longevity_research",
    )

    assert payload["title"] == "Urolithin A provides cardioprotection and improves human cardiovascular health biomarkers"


def test_researka_payload_prefers_bundle_title_for_heterogeneous_direct_endpoints() -> None:
    candidate = InsightCandidate(
        topic="cold immersion training",
        thesis=(
            "cold immersion training may be hiding a cold / immersion / training "
            "boundary condition: acute muscle thickness and performance point in different directions."
        ),
        bridge_terms=("cold", "immersion", "training"),
        tension_terms=("negative", "null"),
        receipt_ids=("thickness", "performance"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            _direct_card(
                "thickness",
                "negative_signal",
                "muscle thickness",
                "negative",
                "Cold-water immersion attenuated elbow flexor muscle thickness after strength training.",
            ),
            _direct_card(
                "performance",
                "null_signal",
                "performance",
                "null",
                "Cold-water immersion training did not improve physical performance.",
            ),
        ),
    )
    receipts = [
        CorpusHit(
            hit_id="thickness",
            title="Effect of Cold-Water Immersion on Elbow Flexors Muscle Thickness After Resistance Training",
            abstract="Human randomized trial measured elbow flexor muscle thickness after resistance training.",
            source="fullraw:semantic_scholar",
            doi="10.1519/JSC.0000000000002322",
        ),
        CorpusHit(
            hit_id="performance",
            title="Cold-water immersion and training performance in human participants",
            abstract="Human intervention study measured training performance.",
            source="fullraw:openalex",
            doi="10.performance",
        ),
    ]
    markdown = "# Alpha memo: cold / immersion / training\n\nBody."

    payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown=markdown),
        author_agent_id="v5-memo-agent",
        domain_slug="longevity_research",
    )

    assert payload["title"] == "Cold Immersion and Training Outcomes in Human Studies"


def test_researka_payload_skips_incomplete_receipt_title() -> None:
    candidate = InsightCandidate(
        topic="resveratrol exercise protocol",
        thesis="Resveratrol exercise evidence splits by timing and endpoint.",
        bridge_terms=("resveratrol", "exercise"),
        tension_terms=("negative", "positive"),
        receipt_ids=("truncated", "complete"),
        score=90,
        novelty_score=58,
        evidence_score=90,
        reasons=("shape:boundary_condition", "tier:publishable_alpha"),
    )
    receipts = [
        CorpusHit(
            hit_id="truncated",
            title="Effects of 14 days of prophylactic resveratrol supplementation in trained endurance runners upon the inflammatory",
            abstract="Human runner intervention measured cytokine response after eccentric exercise.",
            source="fullraw:openalex",
            doi="10.1000/resveratrol-inflammation",
        ),
        CorpusHit(
            hit_id="complete",
            title="Combined exercise training and resveratrol supplementation in older adults with functional limitations",
            abstract="Pilot randomized study assessed safety and feasibility.",
            source="fullraw:openalex",
            doi="10.1016/j.exger.2020.111111",
        ),
    ]
    markdown = "# Alpha memo: Effects of 14 days of prophylactic resveratrol supplementation in trained endurance runners upon the inflammatory\n\nBody."

    payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown=markdown),
        author_agent_id="v5-memo-agent",
        domain_slug="longevity_research",
    )

    assert payload["title"] == "Combined exercise training and resveratrol supplementation in older adults with functional limitations"


def test_claim_card_downgrades_conference_and_supplemental_receipts() -> None:
    conference_hit = CorpusHit(
        hit_id="faseb",
        title="Nicotinamide riboside and exercise performance in healthy volunteers",
        abstract="Randomized human participants trial reported improved performance.",
        source="fullraw:openalex",
        doi="10.1096/fasebj.2021.35.s1.05282",
        venue="The FASEB Journal",
    )
    supplement_hit = CorpusHit(
        hit_id="supplement",
        title="Additional file 1: supporting data for nicotinamide riboside exercise",
        abstract="Human randomized trial supporting data reported improved performance markers.",
        source="fullraw:openalex",
        doi="10.6084/m9.figshare.c.3601490_d1",
    )
    jissn_supplement_hit = CorpusHit(
        hit_id="jissn-supplement",
        title="Effects of resveratrol supplementation after eccentric exercise",
        abstract="Human randomized trial abstract supplement reported inflammatory markers.",
        source="fullraw:openalex",
        doi="10.1186/1550-2783-8-s1-p15",
    )
    faculty_opinions_hit = CorpusHit(
        hit_id="faculty-opinions",
        title="Faculty Opinions recommendation of metformin resistance training trial",
        abstract="Randomized human trial was recommended by Faculty Opinions.",
        source="fullraw:openalex",
        doi="10.3410/f.736671936.793569870",
    )
    acsm_abstract_hit = CorpusHit(
        hit_id="acsm-abstract",
        title="One Bout Of Resistance Exercise Does Not Interfere With Metformin",
        abstract="Randomized human trial abstract reported metabolic syndrome outcomes.",
        source="fullraw:openalex",
        doi="10.1249/01.mss.0000764428.80520.ab",
    )
    ada_abstract_hit = CorpusHit(
        hit_id="ada-abstract",
        title="2267-PUB: Skeletal Muscle Response to Exercise in Patients Taking Metformin",
        abstract="Randomized human trial abstract reported metabolic syndrome outcomes.",
        source="fullraw:openalex",
        doi="10.2337/db19-2267-pub",
    )

    conference = _claim_card(conference_hit, ReceiptRole("faseb", "boundary", "conference abstract"))
    supplement = _claim_card(supplement_hit, ReceiptRole("supplement", "context", "supporting data"))
    jissn_supplement = _claim_card(
        jissn_supplement_hit,
        ReceiptRole("jissn-supplement", "context", "journal supplement abstract"),
    )
    faculty_opinions = _claim_card(
        faculty_opinions_hit,
        ReceiptRole("faculty-opinions", "negative_signal", "recommendation context"),
    )
    acsm_abstract = _claim_card(
        acsm_abstract_hit,
        ReceiptRole("acsm-abstract", "null_signal", "abstract context"),
    )
    ada_abstract = _claim_card(
        ada_abstract_hit,
        ReceiptRole("ada-abstract", "null_signal", "abstract context"),
    )

    assert conference.population == "human"
    assert conference.support_type == "indirect"
    assert conference.confidence == "medium"
    assert supplement.support_type == "indirect"
    assert supplement.confidence == "medium"
    assert jissn_supplement.support_type == "indirect"
    assert jissn_supplement.confidence == "low"
    assert faculty_opinions.support_type == "indirect"
    assert acsm_abstract.support_type == "indirect"
    assert ada_abstract.support_type == "indirect"


def test_claim_card_treats_systematic_review_as_indirect_synthesis() -> None:
    hit = CorpusHit(
        hit_id="10.review",
        title="Systematic review of cold-water immersion after resistance training",
        abstract="Systematic review summarized randomized human trials of cold-water immersion after resistance training.",
        source="fullraw:openalex",
        doi="10.review",
    )

    card = _claim_card(hit, ReceiptRole(hit.hit_id, "mechanism", "review context"))

    assert card.design == "synthesis"
    assert card.support_type == "indirect"
    assert card.confidence == "low"


def test_claim_card_treats_human_supplementation_as_direct_intervention() -> None:
    hit = CorpusHit(
        hit_id="10.1016/j.isci.2025.111814",
        title="Urolithin A improves human cardiovascular health biomarkers",
        abstract=(
            "Preclinically, urolithin A improved mitochondrial quality in aging models. "
            "In humans, UA supplementation for 4 months in healthy older adults significantly "
            "reduced plasma ceramides clinically validated to predict CVD risks."
        ),
        source="fullraw:openalex",
        doi="10.1016/j.isci.2025.111814",
    )

    card = _claim_card(hit, ReceiptRole(hit.hit_id, "tail_risk", "human supplementation biomarker"))

    assert card.design == "intervention_study"
    assert card.population == "human"
    assert card.support_type == "direct"
    assert card.confidence == "high"


def test_claim_card_keeps_beneficial_reductions_positive_for_publish_gate() -> None:
    hit = CorpusHit(
        hit_id="urolithin-positive",
        title="Urolithin A improves muscle endurance in older adults",
        abstract=(
            "A randomized human trial found that urolithin A improved muscle endurance "
            "and reduced fatigue. Exploratory biomarkers were not different and do not "
            "explain the endurance response."
        ),
        source="fullraw:openalex",
        doi="10.1001/jamanetworkopen.2021.44279",
    )

    card = _claim_card(hit, ReceiptRole(hit.hit_id, "positive_signal", "human trial"))
    candidate = InsightCandidate(
        topic="urolithin muscle strength endurance older adults trial",
        thesis="Urolithin direct human receipts should not be blocked by beneficial reduction wording.",
        bridge_terms=("urolithin", "muscle"),
        tension_terms=("positive",),
        receipt_ids=(hit.hit_id, "supporting-positive"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:boundary_condition", "tier:publishable_alpha"),
        claim_cards=(
            card,
            ClaimCard(
                "supporting-positive",
                "positive_signal",
                "randomized_trial",
                "human",
                "muscle strength",
                "positive",
                "direct",
                "high",
                "Urolithin improved muscle strength in a second randomized human trial.",
            ),
        ),
    )

    assert card.direction == "positive"
    assert candidate_publish_blocker(candidate) is None


def test_claim_card_treats_reduced_frailty_progression_as_positive() -> None:
    hit = CorpusHit(
        hit_id="metformin-frailty",
        title="Metformin reduces frailty progression in older adults",
        abstract=(
            "A randomized human trial found a significant reduction in frailty "
            "progression rate with metformin treatment (p=0.0222)."
        ),
        source="fullraw:openalex",
        doi="10.2337/db25-1998-lb",
    )

    card = _claim_card(hit, ReceiptRole(hit.hit_id, "positive_signal", "human frailty outcome"))

    assert card.role == "positive_signal"
    assert card.direction == "positive"
    assert "frailty" in card.outcome


def test_claim_card_treats_noun_reduction_in_frailty_progression_as_positive() -> None:
    hit = CorpusHit(
        hit_id="metformin-frailty-noun",
        title="Metformin frailty outcomes in older adults",
        abstract=(
            "Frailty is a major cause of morbidity and disability. A randomized human "
            "trial found a significant reduction in frailty progression rate with "
            "metformin treatment (p=0.0222)."
        ),
        source="fullraw:openalex",
        doi="10.2337/db25-1998-lb",
    )

    card = _claim_card(hit, ReceiptRole(hit.hit_id, "positive_signal", "human frailty outcome"))

    assert card.direction == "positive"
    assert "frailty" in card.outcome


def test_claim_card_keeps_wistar_rat_receipt_indirect_even_with_human_context() -> None:
    hit = CorpusHit(
        hit_id="resveratrol-rat",
        title="Effect of continuous exercise training on SIRT3 and OGG1 in male Wistar rats",
        abstract=(
            "The introduction discusses human clinical relevance, but the randomized "
            "study used 32 male Wistar rats to measure liver SIRT3 and OGG1 protein."
        ),
        source="fullraw:openalex",
        doi=None,
    )

    card = _claim_card(hit, ReceiptRole(hit.hit_id, "positive_signal", "direct human precheck"))

    assert card.population == "animal"
    assert card.support_type == "indirect"


def test_claim_card_treats_players_training_as_direct_human_intervention() -> None:
    hit = CorpusHit(
        hit_id="10.players",
        title="Cold- and hot-water immersion are not more effective than placebo",
        abstract=(
            "Compared to a placebo, cold-water immersion did not improve post-match "
            "recovery or long-term training adaptations in national level soccer players."
        ),
        source="fullraw:openalex",
        doi="10.players",
    )

    card = _claim_card(hit, ReceiptRole(hit.hit_id, "null_signal", "player training contrast"))

    assert card.design == "intervention_study"
    assert card.population == "human"
    assert card.support_type == "direct"
    assert card.confidence == "high"


def test_claim_card_demotes_acute_proxy_endpoint_signals() -> None:
    proxy_hit = CorpusHit(
        hit_id="10.proxy",
        title="Cold-water immersion changes acute damage and performance markers after resistance training",
        abstract=(
            "Human participants completed resistance training. Cold-water immersion attenuated "
            "acute muscle damage and performance markers after the exercise session."
        ),
        source="fullraw:openalex",
        doi="10.proxy",
    )
    chronic_hit = CorpusHit(
        hit_id="10.chronic",
        title="Cold-water immersion attenuates long-term performance adaptations after resistance training",
        abstract=(
            "Human participants completed resistance training. Cold-water immersion attenuated "
            "long-term performance adaptations after repeated training sessions."
        ),
        source="fullraw:openalex",
        doi="10.chronic",
    )

    proxy_card = _claim_card(proxy_hit, ReceiptRole(proxy_hit.hit_id, "negative_signal", "candidate evidence stream"))
    chronic_card = _claim_card(chronic_hit, ReceiptRole(chronic_hit.hit_id, "negative_signal", "candidate evidence stream"))

    assert proxy_card.role == "boundary"
    assert proxy_card.direction == "proxy"
    assert proxy_card.outcome == "acute/damage/performance"
    assert proxy_card.support_type == "direct"
    assert proxy_card.confidence == "medium"
    assert chronic_card.role == "negative_signal"
    assert chronic_card.direction == "negative"
    assert chronic_card.outcome == "long/performance"


def test_claim_card_marks_acute_muscle_thickness_as_within_arm_context() -> None:
    hit = CorpusHit(
        hit_id="10.1519/JSC.0000000000002322",
        title="Cold water immersion alters acute muscle thickness after resistance training",
        abstract=(
            "Human participants completed resistance training. At 48 h and 72 h, muscle "
            "thickness was reduced with cold-water immersion and higher with passive recovery."
        ),
        source="fullraw:openalex",
        doi="10.1519/JSC.0000000000002322",
    )

    card = _claim_card(hit, ReceiptRole(hit.hit_id, "negative_signal", "candidate evidence stream"))

    assert card.role == "acute_within_arm_signal"
    assert card.direction != "proxy"
    assert card.outcome == "muscle thickness"
    assert card.support_type == "direct"
    assert card.confidence == "medium"


def test_claim_card_does_not_treat_safety_feasibility_pilot_as_positive_efficacy() -> None:
    hit = CorpusHit(
        hit_id="10.1016/j.exger.2020.111111",
        title="Combined exercise training and resveratrol supplementation in older adults",
        abstract=(
            "This pilot randomized trial evaluated safety and feasibility of combining "
            "exercise training with resveratrol supplementation in community-dwelling "
            "older adults with functional limitations."
        ),
        source="fullraw:openalex",
        doi="10.1016/j.exger.2020.111111",
    )

    card = _claim_card(hit, ReceiptRole(hit.hit_id, "positive_signal", "candidate evidence stream"))

    assert card.role == "safety_feasibility"
    assert card.design == "randomized_trial"
    assert card.population == "human"
    assert card.direction == "unclear"
    assert card.support_type == "direct"
    assert card.confidence == "low"


def test_claim_card_marks_comparator_only_benefit_as_null_mixed_direction() -> None:
    hit = CorpusHit(
        hit_id="10.3389/fphys.2021.759240",
        title="Post-exercise Warm or Cold Water Immersion to Augment the Cardiometabolic Benefits of Exercise Training",
        abstract=(
            "Warm or cold water immersion would provide similar or greater benefits. "
            "Work trial distance increased without differences between interventions. "
            "Substituting the second half of exercise with warm water immersion provides "
            "similar cardiometabolic health benefits; however, substituting with cold "
            "water immersion does not."
        ),
        source="fullraw:openalex",
        doi="10.3389/fphys.2021.759240",
    )

    card = _claim_card(hit, ReceiptRole(hit.hit_id, "promise", "promise/outcome split"))

    assert set(card.direction.split("/")) >= {"null", "positive"}


def test_claim_card_preserves_muscle_thickness_outcome() -> None:
    hit = CorpusHit(
        hit_id="10.1123/ijspp.2019-0965",
        title="Does Cold-Water Immersion After Strength Training Attenuate Training Adaptation?",
        abstract=(
            "Cold-water immersion attenuated elbow flexor muscle thickness after strength training, "
            "while 1RM and countermovement jump confidence intervals crossed zero."
        ),
        source="fullraw:semantic_scholar",
        doi="10.1123/ijspp.2019-0965",
    )

    card = _claim_card(hit, ReceiptRole(hit.hit_id, "negative_signal", "candidate evidence stream"))

    assert card.outcome == "muscle thickness"


def test_claim_card_quote_keeps_title_terms_for_publish_gate() -> None:
    hit = CorpusHit(
        hit_id="10.hypothermia",
        title="Prevalence of hypothermia during military cold-water immersion training",
        abstract="Human participants completed a cold-water immersion training assessment.",
        source="fullraw:semantic_scholar",
        doi="10.hypothermia",
    )

    card = _claim_card(hit, ReceiptRole(hit.hit_id, "replication", "context"))

    assert "hypothermia" in card.quote.casefold()
    assert "military" in card.quote.casefold()


def test_publish_blocker_rejects_positive_role_with_null_direction() -> None:
    candidate = InsightCandidate(
        topic="cold water immersion",
        thesis="Comparator-only benefit should not publish as a positive CWI signal.",
        bridge_terms=("cold", "immersion"),
        tension_terms=("negative", "positive"),
        receipt_ids=("receipt-a", "receipt-b"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:promise_outcome_reversal", "tier:elite_alpha"),
        claim_cards=(
            ClaimCard(
                "receipt-a",
                "promise",
                "randomized_trial",
                "human",
                "outcome",
                "null/positive",
                "direct",
                "high",
                "Substituting with cold water immersion does not.",
            ),
            ClaimCard(
                "receipt-b",
                "outcome",
                "randomized_trial",
                "human",
                "outcome",
                "negative",
                "direct",
                "high",
                "Cold water immersion attenuated adaptation.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "positive_role_direction_mismatch",
        "receipt_ids": ("receipt-a",),
    }


def test_publish_blocker_rejects_unmapped_noisy_positive_signal_direction() -> None:
    candidate = InsightCandidate(
        topic="urolithin muscle strength endurance older adults trial",
        thesis="Direct positive human signals should survive incidental reduction wording.",
        bridge_terms=("urolithin", "muscle"),
        tension_terms=("positive",),
        receipt_ids=("strength", "endurance"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:boundary_condition", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "strength",
                "positive_signal",
                "randomized_trial",
                "human",
                "muscle strength",
                "negative/positive",
                "direct",
                "high",
                "Urolithin improved strength while lowering damage markers.",
            ),
            ClaimCard(
                "endurance",
                "positive_signal",
                "randomized_trial",
                "human",
                "muscle endurance",
                "negative/positive",
                "direct",
                "high",
                "Urolithin improved endurance while reducing fatigue.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "ambiguous_direction_without_endpoint_mapping",
        "receipt_ids": ("strength", "endurance"),
    }


def test_publish_blocker_rejects_mixed_metabolic_muscle_axis_bundle() -> None:
    candidate = InsightCandidate(
        topic="metformin resistance training",
        thesis="Glycemic and hypertrophy receipts need a tighter axis before submit.",
        bridge_terms=("metformin", "training"),
        tension_terms=("null", "positive"),
        receipt_ids=("glycemic", "hypertrophy", "body-comp"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "glycemic",
                "null_signal",
                "randomized_trial",
                "human",
                "long",
                "null",
                "direct",
                "high",
                "Resistance exercise with metformin measured glycemic control and insulin sensitivity in T2DM adults.",
            ),
            ClaimCard(
                "hypertrophy",
                "negative_signal",
                "randomized_trial",
                "human",
                "hypertrophy",
                "negative",
                "direct",
                "high",
                "Progressive resistance training had a blunted muscle hypertrophy response with metformin.",
            ),
            ClaimCard(
                "body-comp",
                "replication",
                "intervention_study",
                "human",
                "unspecified",
                "positive",
                "direct",
                "high",
                "Resistance training changed body composition compared with metformin treatment.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "mixed_outcome_axis_bundle",
        "receipt_ids": ("glycemic", "hypertrophy", "body-comp"),
    }


def test_publish_blocker_rejects_weak_primary_signal_receipts() -> None:
    candidate = InsightCandidate(
        topic="cold water immersion",
        thesis="Indirect primary signal should not leave V5 for public review.",
        bridge_terms=("cold", "immersion"),
        tension_terms=("negative", "null"),
        receipt_ids=("indirect", "direct-a", "direct-b"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "indirect",
                "negative_signal",
                "synthesis",
                "human",
                "strength",
                "negative",
                "indirect",
                "medium",
                "Review-level signal.",
            ),
            ClaimCard(
                "direct-a",
                "boundary",
                "randomized_trial",
                "human",
                "strength",
                "negative",
                "direct",
                "high",
                "Direct human signal.",
            ),
            ClaimCard(
                "direct-b",
                "replication",
                "randomized_trial",
                "human",
                "strength",
                "null",
                "direct",
                "high",
                "Direct human signal.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "primary_signal_not_strong_direct_human",
        "receipt_ids": ("indirect",),
    }


def test_publish_blocker_rejects_off_modality_primary_signal() -> None:
    candidate = InsightCandidate(
        topic="cold water immersion resistance training adaptation",
        thesis="Team-sport recovery evidence should not be primary strength-training evidence.",
        bridge_terms=("cold", "immersion"),
        tension_terms=("negative", "null"),
        receipt_ids=("strength", "soccer", "direct"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "strength",
                "negative_signal",
                "randomized_trial",
                "human",
                "muscle thickness",
                "negative",
                "direct",
                "high",
                "Strength-training adaptation was attenuated.",
            ),
            ClaimCard(
                "soccer",
                "null_signal",
                "intervention_study",
                "human",
                "long/performance",
                "null",
                "direct",
                "high",
                "Cold-water immersion recovery did not improve in highly trained soccer players.",
            ),
            ClaimCard(
                "direct",
                "replication",
                "intervention_study",
                "human",
                "strength",
                "negative",
                "direct",
                "high",
                "Direct human replication.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "off_modality_primary_signal",
        "receipt_ids": ("soccer",),
    }


def test_publish_blocker_rejects_indirect_selector_promise() -> None:
    candidate = InsightCandidate(
        topic="intervention training",
        thesis="A protocol cannot carry the primary publish contrast.",
        bridge_terms=("intervention", "training"),
        tension_terms=("positive", "negative"),
        receipt_ids=("protocol", "outcome", "replication"),
        score=100,
        novelty_score=60,
        evidence_score=100,
        reasons=("shape:expectation_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard("protocol", "promise", "randomized_trial", "human", "adaptation", "positive", "indirect", "medium", "Study protocol."),
            ClaimCard("outcome", "outcome", "randomized_trial", "human", "adaptation", "negative", "direct", "high", "Direct outcome."),
            ClaimCard("replication", "replication", "randomized_trial", "human", "adaptation", "negative", "direct", "high", "Direct replication."),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "primary_signal_not_strong_direct_human",
        "receipt_ids": ("protocol",),
    }


def test_publish_blocker_rejects_training_topic_recovery_primary_signal() -> None:
    candidate = InsightCandidate(
        topic="cold immersion training",
        thesis="Recovery receipts should not publish as primary training-adaptation evidence.",
        bridge_terms=("cold", "immersion"),
        tension_terms=("negative", "null"),
        receipt_ids=("strength", "soccer"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "strength",
                "negative_signal",
                "randomized_trial",
                "human",
                "muscle thickness",
                "negative",
                "direct",
                "high",
                "Cold immersion training reduced strength adaptation.",
            ),
            ClaimCard(
                "soccer",
                "null_signal",
                "intervention_study",
                "human",
                "performance",
                "null",
                "direct",
                "high",
                "Cold-water immersion recovery did not improve in highly trained soccer players.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "off_modality_primary_signal",
        "receipt_ids": ("soccer",),
    }


def test_publish_blocker_rejects_off_topic_primary_signal() -> None:
    candidate = InsightCandidate(
        topic="cold water immersion resistance training adaptation",
        thesis="Military cold exposure should not be primary resistance-training evidence.",
        bridge_terms=("cold", "immersion"),
        tension_terms=("negative", "null"),
        receipt_ids=("strength", "military", "direct"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "strength",
                "negative_signal",
                "randomized_trial",
                "human",
                "muscle thickness",
                "negative",
                "direct",
                "high",
                "Strength-training adaptation was attenuated.",
            ),
            ClaimCard(
                "military",
                "negative_signal",
                "intervention_study",
                "human",
                "performance/setting",
                "negative",
                "direct",
                "high",
                "Military cold-water immersion training measured hypothermia and undermined warfighter readiness.",
            ),
            ClaimCard(
                "direct",
                "replication",
                "intervention_study",
                "human",
                "strength",
                "negative",
                "direct",
                "high",
                "Direct human replication.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "off_topic_primary_signal",
        "receipt_ids": ("military",),
    }


def test_publish_blocker_requires_primary_topic_context() -> None:
    candidate = InsightCandidate(
        topic="caffeine cycling performance no difference",
        thesis="Volleyball and sprint receipts should not publish as cycling evidence.",
        bridge_terms=("caffeine", "performance"),
        tension_terms=("positive", "null"),
        receipt_ids=("sprint", "volleyball"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "sprint",
                "positive_signal",
                "randomized_trial",
                "human",
                "performance",
                "positive",
                "direct",
                "high",
                "Caffeine improved repeated sprint performance after napping.",
            ),
            ClaimCard(
                "volleyball",
                "null_signal",
                "randomized_trial",
                "human",
                "performance",
                "null",
                "direct",
                "high",
                "Caffeine mouth rinse did not improve volleyball skill accuracy.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "off_topic_primary_signal",
        "receipt_ids": ("sprint", "volleyball"),
    }


def test_publish_blocker_allows_primary_topic_context() -> None:
    candidate = InsightCandidate(
        topic="urolithin muscle endurance older adults trial",
        thesis="Aligned urolithin muscle-endurance human receipts can pass.",
        bridge_terms=("urolithin", "muscle"),
        tension_terms=("positive", "null"),
        receipt_ids=("strength", "endurance"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "strength",
                "positive_signal",
                "randomized_trial",
                "human",
                "muscle strength",
                "positive",
                "direct",
                "high",
                "Urolithin improved muscle strength and endurance in older adults.",
            ),
            ClaimCard(
                "endurance",
                "null_signal",
                "randomized_trial",
                "human",
                "muscle endurance",
                "null",
                "direct",
                "high",
                "Urolithin muscle endurance response was null in older runners.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) is None


def test_publish_blocker_requires_primary_entity_despite_shared_condition() -> None:
    candidate = InsightCandidate(
        topic="creatine sarcopenia muscle strength older adults randomized trial",
        thesis="A shared condition must not substitute for creatine evidence.",
        bridge_terms=("creatine", "muscle"),
        tension_terms=("positive", "null"),
        receipt_ids=("10.1002/ejsc.70155", "creatine"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "10.1002/ejsc.70155",
                "positive_signal",
                "randomized_trial",
                "human",
                "muscle strength",
                "positive",
                "direct",
                "high",
                "Gliclazide improved muscle strength in adults with sarcopenia.",
            ),
            ClaimCard(
                "creatine",
                "null_signal",
                "randomized_trial",
                "human",
                "muscle strength",
                "null",
                "direct",
                "high",
                "Creatine monohydrate did not improve muscle strength in older adults.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "off_topic_primary_signal",
        "receipt_ids": ("10.1002/ejsc.70155",),
    }


def test_publish_blocker_enforces_entity_for_population_only_clinical_topic() -> None:
    candidate = InsightCandidate(
        topic="creatine muscle strength athletes",
        thesis="A human population signal must activate intervention identity.",
        bridge_terms=("muscle", "strength"),
        tension_terms=("positive", "null"),
        receipt_ids=("gliclazide", "creatine"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "gliclazide", "positive_signal", "randomized_trial", "human",
                "muscle strength", "positive", "direct", "high",
                "Gliclazide improved muscle strength in athletes.",
            ),
            ClaimCard(
                "creatine", "null_signal", "randomized_trial", "human",
                "muscle strength", "null", "direct", "high",
                "Creatine did not improve muscle strength in athletes.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "off_topic_primary_signal",
        "receipt_ids": ("gliclazide",),
    }


@pytest.mark.parametrize(
    "generic_quote",
    [
        "Physical function and muscle strength improved in older adults.",
        "Baseline physical function and muscle strength improved in older adults.",
        "Supervised muscle strength improved in older adults.",
        "Twelve-week muscle strength improved in older adults.",
    ],
)
def test_publish_blocker_allows_generic_axis_quote_without_competing_entity(
    generic_quote: str,
) -> None:
    candidate = InsightCandidate(
        topic="creatine muscle strength older adults randomized trial",
        thesis="A generic axis quote may be contextualized by an aligned receipt.",
        bridge_terms=("creatine", "muscle"),
        tension_terms=("positive", "null"),
        receipt_ids=("generic", "creatine"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "generic",
                "positive_signal",
                "randomized_trial",
                "human",
                "muscle strength",
                "positive",
                "direct",
                "high",
                generic_quote,
            ),
            ClaimCard(
                "creatine",
                "null_signal",
                "randomized_trial",
                "human",
                "muscle strength",
                "null",
                "direct",
                "high",
                "Among frail community dwelling participants with severe baseline "
                "limitations and low functional status, creatine monohydrate improved "
                "muscle strength.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) is None


def test_publish_blocker_normalizes_muscular_to_muscle_axis() -> None:
    candidate = InsightCandidate(
        topic="vitamin d muscle strength older adults randomized trial",
        thesis="The adjectival muscle form remains on the requested axis.",
        bridge_terms=("vitamin", "strength"),
        tension_terms=("positive", "null"),
        receipt_ids=("muscular", "muscle"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "muscular", "positive_signal", "randomized_trial", "human",
                "muscle strength", "positive", "direct", "high",
                "Vitamin D improved muscular strength in older adults.",
            ),
            ClaimCard(
                "muscle", "null_signal", "randomized_trial", "human",
                "muscle strength", "null", "direct", "high",
                "Vitamin D did not improve muscle strength in older adults.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) is None


def test_publish_blocker_rejects_unverified_entity_acronyms() -> None:
    candidate = InsightCandidate(
        topic="nicotinamide mononucleotide sarcopenia muscle strength trial",
        thesis="Receipt-derived acronyms must not authorize their own identity.",
        bridge_terms=("strength", "nad"),
        tension_terms=("positive", "negative", "null"),
        receipt_ids=("nmn", "nad", "expanded"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "nmn", "positive_signal", "randomized_trial", "human",
                "muscle strength", "positive", "direct", "high",
                "NMN improved muscle strength in older adults.",
            ),
            ClaimCard(
                "nad", "negative_signal", "randomized_trial", "human",
                "muscle strength", "negative", "direct", "high",
                "NAD reduced muscle strength in older adults.",
            ),
            ClaimCard(
                "expanded", "null_signal", "randomized_trial", "human",
                "muscle strength", "null", "direct", "high",
                "Nicotinamide mononucleotide did not improve muscle strength.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "off_topic_primary_signal",
        "receipt_ids": ("nmn", "nad"),
    }


def test_publish_blocker_rejects_expanded_entity_with_shared_prefix() -> None:
    candidate = InsightCandidate(
        topic="nicotinamide mononucleotide muscle strength trial",
        thesis="A shared chemical prefix must not substitute for the named molecule.",
        bridge_terms=("strength", "nicotinamide"),
        tension_terms=("positive", "null"),
        receipt_ids=("nad", "nmn"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "nad", "positive_signal", "randomized_trial", "human",
                "muscle strength", "positive", "direct", "high",
                "Nicotinamide adenine dinucleotide improved muscle strength in sarcopenia.",
            ),
            ClaimCard(
                "nmn", "null_signal", "randomized_trial", "human",
                "muscle strength", "null", "direct", "high",
                "Nicotinamide mononucleotide did not improve muscle strength in sarcopenia.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "off_topic_primary_signal",
        "receipt_ids": ("nad",),
    }


def test_publish_blocker_rejects_all_wrong_shared_prefix_entities() -> None:
    candidate = InsightCandidate(
        topic="nicotinamide mononucleotide sarcopenia muscle strength trial",
        thesis="Wrong receipts must not weaken the topic-owned identity.",
        bridge_terms=("strength", "nicotinamide"),
        tension_terms=("positive", "null"),
        receipt_ids=("nad-positive", "nad-null"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "nad-positive", "positive_signal", "randomized_trial", "human",
                "muscle strength", "positive", "direct", "high",
                "Nicotinamide adenine dinucleotide improved muscle strength in sarcopenia.",
            ),
            ClaimCard(
                "nad-null", "null_signal", "randomized_trial", "human",
                "muscle strength", "null", "direct", "high",
                "Nicotinamide adenine dinucleotide did not improve muscle strength in sarcopenia.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "off_topic_primary_signal",
        "receipt_ids": ("nad-positive", "nad-null"),
    }


@pytest.mark.parametrize(
    ("topic", "wrong_quote", "aligned_quote"),
    [
        (
            "vitamin d muscle strength older adults randomized trial",
            "Vitamin B12 improved muscle strength in older adults.",
            "Vitamin D did not improve muscle strength in older adults.",
        ),
        (
            "urolithin a muscle strength older adults randomized trial",
            "Urolithin B improved muscle strength in older adults.",
            "Urolithin A did not improve muscle strength in older adults.",
        ),
        (
            "omega 3 muscle strength older adults randomized trial",
            "Omega 6 improved muscle strength in older adults.",
            "Omega 3 did not improve muscle strength in older adults.",
        ),
    ],
)
def test_publish_blocker_preserves_short_entity_disambiguators(
    topic: str,
    wrong_quote: str,
    aligned_quote: str,
) -> None:
    candidate = InsightCandidate(
        topic=topic,
        thesis="Short entity suffixes remain part of the intervention identity.",
        bridge_terms=("muscle", "strength"),
        tension_terms=("positive", "null"),
        receipt_ids=("wrong", "aligned"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "wrong", "positive_signal", "randomized_trial", "human",
                "muscle strength", "positive", "direct", "high", wrong_quote,
            ),
            ClaimCard(
                "aligned", "null_signal", "randomized_trial", "human",
                "muscle strength", "null", "direct", "high", aligned_quote,
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "off_topic_primary_signal",
        "receipt_ids": ("wrong",),
    }


@pytest.mark.parametrize(
    ("topic", "expected"),
    [
        ("cold water immersion resistance training trial", ("cold", "water", "immersion")),
        ("nicotinamide riboside exercise performance trial", ("nicotinamide", "riboside")),
        ("protein timing distribution resistance training trial", ("protein", "timing", "distribution")),
        ("collagen peptides resistance training trial", ("collagen", "peptide")),
        ("time restricted eating resistance training trial", ("time", "restricted", "eating")),
        ("ketogenic diet resistance training trial", ("ketogenic", "diet")),
        ("sauna bathing cardiovascular aging trial", ("sauna", "bathing")),
        ("beta alanine resistance training trial", ("beta", "alanine")),
        ("vitamin d muscle strength trial", ("vitamin", "d")),
        ("urolithin a muscle strength trial", ("urolithin", "a")),
        ("omega 3 muscle strength trial", ("omega", "3")),
        ("fisetin senolytic muscle strength trial", ("fisetin", "senolytic")),
        (
            "quercetin dasatinib senolytic muscle strength trial",
            ("quercetin", "dasatinib", "senolytic"),
        ),
        ("creatine sarcopenia muscle strength trial", ("creatine",)),
        (
            "nicotinamide mononucleotide sarcopenia muscle strength trial",
            ("nicotinamide", "mononucleotide"),
        ),
    ],
)
def test_topic_entity_terms_preserve_intervention_prefix(
    topic: str,
    expected: tuple[str, ...],
) -> None:
    assert _topic_entity_terms(topic) == expected


def test_publish_blocker_requires_two_core_topic_terms_per_primary_receipt() -> None:
    candidate = InsightCandidate(
        topic="urolithin a muscle strength older adults randomized trial",
        thesis="Generic trial language must not substitute for the requested muscle-strength axis.",
        bridge_terms=("urolithin", "randomized"),
        tension_terms=("positive", "null"),
        receipt_ids=("immune", "heart"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "immune",
                "positive_signal",
                "randomized_trial",
                "human",
                "immune decline",
                "positive",
                "direct",
                "high",
                "Urolithin A randomized placebo-controlled trial in middle-aged adults measured immune decline.",
            ),
            ClaimCard(
                "heart",
                "null_signal",
                "randomized_trial",
                "human",
                "ejection fraction",
                "null",
                "direct",
                "high",
                "Urolithin A randomized placebo-controlled trial in heart-failure patients measured cardiac outcomes.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "off_topic_primary_signal",
        "receipt_ids": ("immune", "heart"),
    }


def test_publish_blocker_does_not_treat_design_terms_as_topic_axis() -> None:
    candidate = InsightCandidate(
        topic="urolithin muscle strength crossover trial",
        thesis="Shared design language must not substitute for the requested outcome axis.",
        bridge_terms=("urolithin", "crossover"),
        tension_terms=("positive", "null"),
        receipt_ids=("immune", "heart"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "immune", "positive_signal", "randomized_trial", "human",
                "immune decline", "positive", "direct", "high",
                "Urolithin crossover study measured immune decline.",
            ),
            ClaimCard(
                "heart", "null_signal", "randomized_trial", "human",
                "ejection fraction", "null", "direct", "high",
                "Urolithin crossover study measured heart failure.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "off_topic_primary_signal",
        "receipt_ids": ("immune", "heart"),
    }


def test_publish_blocker_allows_unknown_but_on_axis_endpoint() -> None:
    candidate = InsightCandidate(
        topic="urolithin ejection fraction heart failure trial",
        thesis="On-axis direct receipts may retain an endpoint outside the parser lexicon.",
        bridge_terms=("urolithin",),
        tension_terms=("positive", "null"),
        receipt_ids=("left", "right"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "left", "positive_signal", "randomized_trial", "human",
                "unspecified", "positive", "direct", "high",
                "Urolithin improved ejection fraction in heart failure.",
            ),
            ClaimCard(
                "right", "null_signal", "randomized_trial", "human",
                "unspecified", "null", "direct", "high",
                "Urolithin did not change ejection fraction in heart failure.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) is None


def test_publish_blocker_rejects_structural_endpoint_labels() -> None:
    candidate = InsightCandidate(
        topic="urolithin muscle strength",
        thesis="Parser structure labels are not biological endpoints.",
        bridge_terms=("urolithin", "muscle"),
        tension_terms=("positive", "null"),
        receipt_ids=("strength", "dose"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "strength",
                "positive_signal",
                "randomized_trial",
                "human",
                "muscle strength",
                "positive",
                "direct",
                "high",
                "Urolithin improved muscle strength in a randomized trial.",
            ),
            ClaimCard(
                "dose",
                "null_signal",
                "randomized_trial",
                "human",
                "dose",
                "null",
                "direct",
                "high",
                "Urolithin muscle dosing did not change strength in a randomized trial.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "structural_endpoint_without_outcome",
        "receipt_ids": ("dose",),
    }


def test_claim_card_excludes_parser_structure_from_endpoint() -> None:
    hit = CorpusHit(
        hit_id="immune",
        title="Effect of urolithin A on age-related immune decline: a randomized trial",
        abstract=(
            "Fifty middle-aged adult participants received a 1,000 mg dose. "
            "The randomized population showed reduced immune decline."
        ),
        source="fullraw:semantic_scholar",
    )

    card = _claim_card(hit, ReceiptRole(hit.hit_id, "positive_signal", "human trial"))

    assert card.outcome == "decline"
    assert "dose" not in card.outcome
    assert "population" not in card.outcome


def test_publish_quality_drops_off_axis_direct_context_receipts() -> None:
    candidate = InsightCandidate(
        topic="cold immersion training",
        thesis="Off-axis operational safety context should not publish inside a training signal bundle.",
        bridge_terms=("cold", "immersion"),
        tension_terms=("negative", "null"),
        receipt_ids=("performance-negative", "performance-null", "10.hypothermia"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            _direct_card(
                "performance-negative",
                "negative_signal",
                "performance",
                "negative",
                "Cold immersion training reduced performance adaptation in human participants.",
            ),
            _direct_card(
                "performance-null",
                "null_signal",
                "performance",
                "null",
                "Cold immersion training did not improve performance in human participants.",
            ),
            _claim_card(
                CorpusHit(
                    hit_id="10.hypothermia",
                    title="Prevalence of hypothermia during military cold-water immersion training",
                    abstract="Human participants completed cold-water immersion training and reduced performance.",
                    source="fullraw:semantic_scholar",
                    doi="10.hypothermia",
                ),
                ReceiptRole("10.hypothermia", "replication", "context"),
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "off_axis_direct_context",
        "receipt_ids": ("10.hypothermia",),
    }

    selected = _publishable_candidates(
        [candidate],
        [
            _hit("performance-negative", "Negative", "Randomized human trial negative signal."),
            _hit("performance-null", "Null", "Human intervention study null signal."),
            _hit("10.hypothermia", "Hypothermia", "Military hypothermia context."),
        ],
        "publishable_alpha",
        frozenset(),
        require_publish_quality=True,
    )

    assert len(selected) == 1
    assert selected[0].receipt_ids == ("performance-negative", "performance-null")
    assert candidate_publish_blocker(selected[0]) is None


def test_publish_blocker_rejects_primary_signal_without_topic_anchor() -> None:
    candidate = InsightCandidate(
        topic="creatine older adults function trial",
        thesis="Off-topic direct human rows should not pass just because they are strong trials.",
        bridge_terms=("trial", "function"),
        tension_terms=("negative", "positive"),
        receipt_ids=("arsenic", "creatine"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "arsenic",
                "tail_risk",
                "randomized_trial",
                "human",
                "chronic/long/outcome",
                "negative",
                "direct",
                "high",
                "Chronic arsenic exposure altered methylation outcomes in adults.",
            ),
            ClaimCard(
                "creatine",
                "positive_signal",
                "randomized_trial",
                "human",
                "performance",
                "positive",
                "direct",
                "high",
                "Creatine supplementation with exercise improved function in older adults.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "off_topic_primary_signal",
        "receipt_ids": ("arsenic",),
    }


def test_publish_blocker_rejects_weak_context_receipts() -> None:
    candidate = InsightCandidate(
        topic="cold water immersion",
        thesis="Weak context rows should not pad a public alpha bundle.",
        bridge_terms=("cold", "immersion"),
        tension_terms=("negative", "null"),
        receipt_ids=("direct-a", "direct-b", "weak-context"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "direct-a",
                "negative_signal",
                "randomized_trial",
                "human",
                "strength",
                "negative",
                "direct",
                "high",
                "Direct human signal.",
            ),
            ClaimCard(
                "direct-b",
                "null_signal",
                "intervention_study",
                "human",
                "strength",
                "null",
                "direct",
                "high",
                "Direct human signal.",
            ),
            ClaimCard(
                "weak-context",
                "consensus",
                "unspecified",
                "unspecified",
                "strength",
                "unclear",
                "indirect",
                "low",
                "Weak context row.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "weak_context_receipts",
        "receipt_ids": ("weak-context",),
    }


def test_publish_blocker_allows_proxy_with_negative_null_directional_contrast() -> None:
    candidate = InsightCandidate(
        topic="cold water immersion resistance training adaptation",
        thesis="A proxy endpoint can be context when direct human receipts already differ.",
        bridge_terms=("cold", "immersion", "training"),
        tension_terms=("negative", "null"),
        receipt_ids=("rct", "proxy", "soccer"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "rct",
                "negative_signal",
                "randomized_trial",
                "human",
                "performance",
                "negative",
                "direct",
                "high",
                "Cold water immersion resistance training chronic negative signal.",
            ),
            ClaimCard(
                "proxy",
                "boundary",
                "intervention_study",
                "human",
                "acute/damage/performance",
                "proxy",
                "direct",
                "high",
                "Acute thickness proxy.",
            ),
            ClaimCard(
                "soccer",
                "null_signal",
                "intervention_study",
                "human",
                "long/performance",
                "null",
                "direct",
                "high",
                "Cold water immersion training adaptation null signal.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) is None


def test_publish_blocker_rejects_directional_acute_proxy_without_independent_contrast() -> None:
    candidate = InsightCandidate(
        topic="cold water immersion resistance training adaptation",
        thesis="Acute damage endpoint should not anchor chronic-adaptation alpha by itself.",
        bridge_terms=("cold", "immersion", "training"),
        tension_terms=("negative", "null"),
        receipt_ids=("rct", "review", "acute"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "rct",
                "negative_signal",
                "randomized_trial",
                "human",
                "performance",
                "negative",
                "direct",
                "high",
                "Cold water immersion resistance training chronic negative signal.",
            ),
            ClaimCard(
                "review",
                "mechanism",
                "synthesis",
                "human",
                "acute/context/damage",
                "negative/null/positive",
                "indirect",
                "medium",
                "Review context.",
            ),
            ClaimCard(
                "acute",
                "boundary",
                "intervention_study",
                "human",
                "acute/damage/performance",
                "negative",
                "direct",
                "high",
                "Acute thickness endpoint.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "proxy_without_independent_directional_contrast",
        "receipt_ids": ("acute",),
    }


def test_publish_blocker_allows_proxy_with_independent_directional_contrast() -> None:
    candidate = InsightCandidate(
        topic="cold water immersion resistance training adaptation",
        thesis="A proxy receipt can be context when direct human receipts already contrast.",
        bridge_terms=("cold", "immersion", "training"),
        tension_terms=("negative", "positive"),
        receipt_ids=("negative", "positive", "proxy"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "negative",
                "negative_signal",
                "randomized_trial",
                "human",
                "performance",
                "negative",
                "direct",
                "high",
                "Cold water immersion training negative signal.",
            ),
            ClaimCard(
                "positive",
                "positive_signal",
                "intervention_study",
                "human",
                "performance",
                "positive",
                "direct",
                "high",
                "Cold water immersion training positive signal.",
            ),
            ClaimCard(
                "proxy",
                "boundary",
                "intervention_study",
                "human",
                "acute/damage/performance",
                "negative",
                "direct",
                "high",
                "Proxy context.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) is None


def test_publish_quality_candidates_drop_weak_context_receipts() -> None:
    candidate = InsightCandidate(
        topic="cold water immersion",
        thesis="Strong direct human signals should not be blocked by weak context padding.",
        bridge_terms=("cold", "immersion"),
        tension_terms=("negative", "null"),
        receipt_ids=("direct-a", "direct-b", "weak-context"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "direct-a",
                "negative_signal",
                "randomized_trial",
                "human",
                "strength",
                "negative",
                "direct",
                "high",
                "Cold water immersion direct negative human signal.",
            ),
            ClaimCard(
                "direct-b",
                "null_signal",
                "intervention_study",
                "human",
                "strength",
                "null",
                "direct",
                "high",
                "Cold water immersion direct null human signal.",
            ),
            ClaimCard(
                "weak-context",
                "consensus",
                "unspecified",
                "unspecified",
                "strength",
                "unclear",
                "indirect",
                "low",
                "Weak context row.",
            ),
        ),
    )

    assert candidate_publish_blocker(candidate) == {
        "error": "weak_context_receipts",
        "receipt_ids": ("weak-context",),
    }

    selected = _publishable_candidates(
        [candidate],
        [
            _hit("direct-a", "Direct A", "Randomized human trial negative signal."),
            _hit("direct-b", "Direct B", "Human intervention study null signal."),
            _hit("weak-context", "Weak context", "Weak context row."),
        ],
        "publishable_alpha",
        frozenset(),
        require_publish_quality=True,
    )

    assert len(selected) == 1
    assert selected[0].receipt_ids == ("direct-a", "direct-b")
    assert candidate_publish_blocker(selected[0]) is None


def test_publish_quality_drops_indirect_graph_context_receipts() -> None:
    candidate = InsightCandidate(
        topic="tool reliability",
        thesis="Direct human signals with optional model context.",
        bridge_terms=("intervention", "training"),
        tension_terms=("negative", "null"),
        receipt_ids=("direct-a", "direct-b", "model-context"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        receipt_roles=(
            ReceiptRole("direct-a", "negative_signal", "direct contrast"),
            ReceiptRole("direct-b", "null_signal", "direct contrast"),
        ),
        claim_cards=(
            ClaimCard("direct-a", "negative_signal", "randomized_trial", "human", "adaptation", "negative", "direct", "high", "Direct tool reliability negative signal."),
            ClaimCard("direct-b", "null_signal", "randomized_trial", "human", "adaptation", "null", "direct", "high", "Direct tool reliability null signal."),
            ClaimCard("model-context", "mechanism", "mechanistic_model", "animal", "adaptation", "positive", "indirect", "medium", "Optional model context."),
        ),
    )

    selected = _publishable_candidates(
        [candidate],
        [
            _hit("direct-a", "Direct A", "Randomized human trial negative signal."),
            _hit("direct-b", "Direct B", "Randomized human trial null signal."),
            _hit("model-context", "Model context", "Animal mechanism signal."),
        ],
        "publishable_alpha",
        frozenset(),
        require_publish_quality=True,
    )

    assert len(selected) == 1
    assert selected[0].receipt_ids == ("direct-a", "direct-b")


def test_researka_payload_strips_internal_diagnostic_sections() -> None:
    candidate = InsightCandidate(
        topic="intervention training",
        thesis="Bounded training signal.",
        bridge_terms=("intervention", "training"),
        tension_terms=("negative", "null"),
        receipt_ids=("a", "b"),
        score=90,
        novelty_score=60,
        evidence_score=90,
        reasons=("shape:directional_reversal",),
    )
    receipts = [_hit("a", "Intervention trial A", "Negative signal."), _hit("b", "Intervention trial B", "Null signal.")]
    markdown = """# Alpha memo: Intervention training signal

**Core signal:**
Public synthesis.

**Audit trail:**
- Internal score.

**Claim ledger:**
- Internal role.

**Receipts:**
1. 10.a Trial A.
2. 10.b Trial B.

**Safety note:** Internal writer instruction.
"""

    payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown=markdown),
        author_agent_id="v5-memo-agent",
        domain_slug="performance",
    )
    body = cast(str, payload["body_markdown"])

    assert "Public synthesis." in body
    assert "**Receipts:**" in body
    assert "Audit trail" not in body
    assert "Claim ledger" not in body
    assert "Safety note" not in body


def test_publish_quality_keeps_negative_null_contrast_after_dropping_weak_context() -> None:
    candidate = InsightCandidate(
        topic="cold water immersion",
        thesis="Strong negative/null direct signals can carry proxy context.",
        bridge_terms=("cold", "immersion"),
        tension_terms=("negative", "null"),
        receipt_ids=("negative", "null", "proxy", "weak-context"),
        score=100,
        novelty_score=58,
        evidence_score=100,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "negative",
                "negative_signal",
                "randomized_trial",
                "human",
                "performance",
                "negative",
                "direct",
                "high",
                "Cold water immersion direct negative human signal.",
            ),
            ClaimCard(
                "null",
                "null_signal",
                "intervention_study",
                "human",
                "long/performance",
                "null",
                "direct",
                "high",
                "Cold water immersion direct null human signal.",
            ),
            ClaimCard(
                "proxy",
                "boundary",
                "intervention_study",
                "human",
                "acute/damage/performance",
                "negative",
                "direct",
                "high",
                "Proxy context.",
            ),
            ClaimCard(
                "weak-context",
                "consensus",
                "unspecified",
                "unspecified",
                "performance",
                "unclear",
                "indirect",
                "low",
                "Weak context row.",
            ),
        ),
    )

    selected = _publishable_candidates(
        [candidate],
        [
            _hit("negative", "Negative", "Randomized human trial negative signal."),
            _hit("null", "Null", "Human intervention study null signal."),
            _hit("proxy", "Proxy", "Human intervention study acute damage signal."),
            _hit("weak-context", "Weak context", "Weak context row."),
        ],
        "publishable_alpha",
        frozenset(),
        require_publish_quality=True,
    )

    assert len(selected) == 1
    assert selected[0].receipt_ids == ("negative", "null")
    assert candidate_publish_blocker(selected[0]) is None


def test_publish_quality_rejects_candidate_when_context_trim_drops_all_receipts() -> None:
    candidate = InsightCandidate(
        topic="cold water immersion",
        thesis="Proxy-only receipts should not become an empty public bundle.",
        bridge_terms=("cold", "immersion"),
        tension_terms=("proxy",),
        receipt_ids=("proxy-a", "proxy-b"),
        score=100,
        novelty_score=70,
        evidence_score=100,
        reasons=("shape:boundary_condition", "tier:elite_alpha"),
        claim_cards=(
            ClaimCard(
                "proxy-a",
                "boundary",
                "intervention_study",
                "human",
                "acute/damage/performance",
                "proxy",
                "direct",
                "high",
                "Cold water immersion acute damage proxy signal.",
            ),
            ClaimCard(
                "proxy-b",
                "boundary",
                "intervention_study",
                "human",
                "acute/damage/performance",
                "proxy",
                "direct",
                "high",
                "Cold water immersion acute performance proxy signal.",
            ),
        ),
    )

    assert _publishable_candidates(
        [candidate],
        [
            _hit("proxy-a", "Proxy A", "Cold water immersion acute damage proxy signal."),
            _hit("proxy-b", "Proxy B", "Cold water immersion acute performance proxy signal."),
        ],
        "publishable_alpha",
        frozenset(),
        require_publish_quality=True,
    ) == []


def test_researka_payload_strips_markdown_wrapped_doi_receipt_labels() -> None:
    candidate = InsightCandidate(
        topic="cold water immersion",
        thesis="Cold immersion adaptation signal.",
        bridge_terms=("cold", "immersion"),
        tension_terms=("attenuated",),
        receipt_ids=("10.1123/ijspp.2019-0965", "10.1519/jsc.0000000000000434"),
        score=100,
        novelty_score=70,
        evidence_score=95,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
    )
    receipts = [
        CorpusHit(
            hit_id="10.1123/ijspp.2019-0965",
            title="Does Cold-Water Immersion After Strength Training Attenuate Training Adaptation?",
            abstract="Cold-water immersion attenuated adaptation after strength training.",
            source="fullraw:semantic_scholar",
            doi="10.1123/ijspp.2019-0965",
        ),
        CorpusHit(
            hit_id="10.1519/jsc.0000000000000434",
            title="Strength Training Adaptations After Cold-Water Immersion",
            abstract="Strength training adaptations were measured after cold-water immersion.",
            source="fullraw:semantic_scholar",
            doi="10.1519/jsc.0000000000000434",
        ),
    ]
    markdown = """
# Alpha memo: Cold-water immersion after strength training

## Receipts
- **10.1123/ijspp.2019-0965**: first receipt.
- **10.1519/jsc.0000000000000434**: second receipt.
""".strip()

    payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown=markdown),
        author_agent_id="v5-alpha",
        domain_slug="performance",
    )
    body = cast(str, payload["body_markdown"])

    assert "10.1123/ijspp.2019-0965**" not in body
    assert "10.1519/jsc.0000000000000434**" not in body
    assert "**10.1123/ijspp.2019-0965**" not in body
    assert "10.1123/ijspp.2019-0965 -" in body
    assert "10.1519/jsc.0000000000000434 -" in body


def test_researka_payload_uses_receipt_title_for_bridge_only_alpha_title() -> None:
    candidate = InsightCandidate(
        topic="cold water immersion resistance training adaptation",
        thesis="Cold immersion strength training may hide a metric-window signal.",
        bridge_terms=("cold", "immersion", "strength", "training"),
        tension_terms=("negative", "positive"),
        receipt_ids=("10.1123/ijspp.2019-0965", "10.1519/jsc.0000000000000434"),
        score=100,
        novelty_score=58,
        evidence_score=96,
        reasons=("shape:directional_reversal",),
    )
    receipts = [
        CorpusHit(
            hit_id="10.1123/ijspp.2019-0965",
            title="Does Cold-Water Immersion After Strength Training Attenuate Training Adaptation?",
            abstract="Cold-water immersion attenuated adaptation after strength training.",
            source="fullraw:semantic_scholar",
            doi="10.1123/ijspp.2019-0965",
        ),
        CorpusHit(
            hit_id="10.1519/jsc.0000000000000434",
            title="Strength Training Adaptations After Cold-Water Immersion",
            abstract="Strength training adaptations were measured after cold-water immersion.",
            source="fullraw:semantic_scholar",
            doi="10.1519/jsc.0000000000000434",
        ),
    ]
    markdown = "# Alpha memo: cold immersion strength training\n\n**Alpha hypothesis:** bounded signal."

    payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown=markdown),
        author_agent_id="v5-alpha",
        domain_slug="performance",
    )

    assert payload["title"] == "Does Cold-Water Immersion After Strength Training Attenuate Training Adaptation?"
    assert cast(str, payload["body_markdown"]).startswith(
        "# Alpha memo: Does Cold-Water Immersion After Strength Training Attenuate Training Adaptation?"
    )


def test_researka_payload_uses_bundle_title_for_heterogeneous_bridge_title() -> None:
    candidate = InsightCandidate(
        topic="cold water immersion resistance training adaptation",
        thesis="Cold immersion training evidence separates structural and performance outcomes.",
        bridge_terms=("cold", "immersion", "training", "water"),
        tension_terms=("negative", "null"),
        receipt_ids=("elbow", "soccer", "strength"),
        score=100,
        novelty_score=58,
        evidence_score=96,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "elbow",
                "boundary",
                "randomized_trial",
                "human",
                "acute/damage/performance",
                "proxy",
                "direct",
                "high",
                "CWI attenuated elbow flexor muscle thickness.",
            ),
            ClaimCard(
                "soccer",
                "replication",
                "intervention_study",
                "human",
                "repeated sprint performance",
                "null",
                "direct",
                "high",
                "CWI did not improve repeated sprint performance.",
            ),
            ClaimCard(
                "strength",
                "boundary",
                "randomized_trial",
                "human",
                "strength adaptation",
                "negative",
                "direct",
                "high",
                "CWI altered strength-training adaptation.",
            ),
        ),
    )
    receipts = [
        CorpusHit(
            hit_id="elbow",
            title="Effect of Cold-Water Immersion on Elbow Flexors Muscle Thickness After Resistance Training",
            abstract="Cold-water immersion affected elbow flexor muscle thickness after training.",
            source="fullraw:openalex",
            doi="10.1000/elbow",
        ),
        CorpusHit(
            hit_id="soccer",
            title="Post-training cold-water immersion in soccer players",
            abstract="Soccer players completed a performance recovery protocol.",
            source="fullraw:openalex",
            doi="10.1000/soccer",
        ),
        CorpusHit(
            hit_id="strength",
            title="Strength training adaptations after cold-water immersion",
            abstract="Strength-training adaptation endpoints were measured.",
            source="fullraw:openalex",
            doi="10.1000/strength",
        ),
    ]
    markdown = "# Alpha memo: cold immersion training water\n\n**Alpha hypothesis:** bounded signal."

    payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown=markdown),
        author_agent_id="v5-alpha",
        domain_slug="performance",
    )

    assert payload["title"] == "Cold Water Immersion: Endpoint Heterogeneity in Acute Proxy vs Chronic Training Adaptation"
    assert payload["title"] != receipts[0].title
    assert cast(str, payload["body_markdown"]).startswith(
        "# Alpha memo: Cold Water Immersion: Endpoint Heterogeneity in Acute Proxy vs Chronic Training Adaptation"
    )
    narrow_payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown=f"# Alpha memo: {receipts[0].title}\n\nBody."),
        author_agent_id="v5-alpha",
        domain_slug="performance",
    )
    assert narrow_payload["title"] == payload["title"]


def test_researka_payload_preserves_negative_ci_signs_in_abstract() -> None:
    candidate = InsightCandidate(
        topic="cold water immersion resistance training adaptation",
        thesis="Cold immersion training evidence separates structural and performance outcomes.",
        bridge_terms=("cold", "immersion", "training", "water"),
        tension_terms=("negative", "null"),
        receipt_ids=("elbow", "soccer"),
        score=100,
        novelty_score=58,
        evidence_score=96,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
    )
    receipts = [
        _hit("elbow", "Cold-water immersion muscle thickness", "Muscle thickness CI crossed zero."),
        _hit("soccer", "Cold-water immersion soccer performance", "Performance CI crossed zero."),
    ]
    markdown = (
        "# Alpha memo: Cold Water Immersion: Muscle Thickness vs Strength Training Adaptation\n\n"
        "Hypothesis-level alpha signal; not clinical advice.\n"
        "## Core signal\n"
        "Muscle thickness was g = 1.20; 95% CI, -0.65 to 1.20, while 1RM was "
        "g = 0.71; 95% CI, -0.30 to 1.72.\n"
    )

    payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown=markdown),
        author_agent_id="v5-alpha",
        domain_slug="performance",
    )

    assert "-0.65 to 1.20" in cast(str, payload["abstract"])
    assert "-0.30 to 1.72" in cast(str, payload["abstract"])


def test_researka_payload_prefers_structural_endpoint_title_over_performance() -> None:
    candidate = InsightCandidate(
        topic="cold water immersion resistance training adaptation",
        thesis="Cold immersion training evidence separates muscle thickness and performance outcomes.",
        bridge_terms=("cold", "immersion", "training", "water"),
        tension_terms=("negative", "null"),
        receipt_ids=("thickness", "performance"),
        score=100,
        novelty_score=58,
        evidence_score=96,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "thickness",
                "negative_signal",
                "randomized_trial",
                "human",
                "muscle thickness",
                "negative",
                "direct",
                "high",
                "CWI attenuated muscle thickness.",
            ),
            ClaimCard(
                "performance",
                "null_signal",
                "intervention_study",
                "human",
                "long/performance",
                "null",
                "direct",
                "high",
                "CWI did not improve long-term performance.",
            ),
        ),
    )
    receipts = [
        _hit("thickness", "Cold-water immersion muscle thickness", "CWI attenuated muscle thickness."),
        _hit("performance", "Cold-water immersion soccer performance", "CWI did not improve performance."),
    ]

    payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown="# Alpha memo: cold immersion training water\n\nBody."),
        author_agent_id="v5-alpha",
        domain_slug="performance",
    )

    assert payload["title"] == "Cold Water Immersion: Muscle Thickness vs Strength Training Adaptation"


def test_researka_payload_narrows_adaptation_title_and_leads_abstract_with_alpha_scope() -> None:
    candidate = InsightCandidate(
        topic="cold water immersion resistance training adaptation",
        thesis="Cold immersion training evidence separates adaptation and recovery endpoints.",
        bridge_terms=("cold", "immersion", "training", "water"),
        tension_terms=("negative", "null"),
        receipt_ids=("recovery", "strength"),
        score=100,
        novelty_score=58,
        evidence_score=96,
        reasons=("shape:directional_reversal", "tier:publishable_alpha"),
        claim_cards=(
            ClaimCard(
                "recovery",
                "replication",
                "intervention_study",
                "human",
                "recovery benefit",
                "null",
                "direct",
                "high",
                "Cold-water immersion did not improve recovery benefit.",
            ),
            ClaimCard(
                "strength",
                "boundary",
                "randomized_trial",
                "human",
                "strength training adaptation",
                "negative",
                "direct",
                "high",
                "Cold-water immersion attenuated strength-training adaptation.",
            ),
        ),
    )
    receipts = [
        _hit("recovery", "Cold-water immersion and recovery benefit", "Human trial reported no recovery benefit."),
        _hit(
            "strength",
            "Strength training adaptations after cold-water immersion",
            "Human trial measured strength-training adaptation.",
        ),
    ]
    markdown = "# Alpha memo: Cold Water Immersion and Training Outcomes in Human Studies\n\nBody."

    payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown=markdown),
        author_agent_id="v5-alpha",
        domain_slug="performance",
    )

    assert payload["title"] == "Cold Water Immersion: Recovery and Strength Training Adaptation"
    assert cast(str, payload["abstract"]).startswith("Hypothesis-level alpha signal; not clinical advice.")
    assert cast(str, payload["body_markdown"]).startswith(
        "# Alpha memo: Cold Water Immersion: Recovery and Strength Training Adaptation"
    )


def test_researka_payload_rewrites_internal_role_label_title() -> None:
    candidate = InsightCandidate(
        topic="resveratrol exercise training",
        thesis="Resveratrol exercise evidence splits by model and adaptation endpoint.",
        bridge_terms=("resveratrol", "exercise", "training"),
        tension_terms=("null", "positive"),
        receipt_ids=("human", "animal"),
        score=100,
        novelty_score=58,
        evidence_score=90,
        reasons=("shape:promise_outcome_reversal", "tier:publishable_alpha"),
    )
    receipts = [
        _hit("human", "Resveratrol supplementation and exercise training in humans", "Human trial reported null adaptation."),
        _hit("animal", "Resveratrol and exercise adaptation in skeletal muscle", "Animal model reported mixed adaptation."),
    ]
    markdown = "# Alpha memo: resveratrol aged skeletal exercise promise outcome\n\n**Alpha hypothesis:** bounded signal."

    payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown=markdown),
        author_agent_id="v5-alpha",
        domain_slug="longevity",
    )

    assert payload["title"] == "Resveratrol exercise evidence splits by model and adaptation endpoint."
    assert cast(str, payload["body_markdown"]).startswith(
        "# Alpha memo: Resveratrol exercise evidence splits by model and adaptation endpoint."
    )


def test_researka_payload_uses_receipt_title_when_thesis_title_is_still_query_like() -> None:
    candidate = InsightCandidate(
        topic="resveratrol exercise adaptation",
        thesis="Resveratrol exercise adaptation.",
        bridge_terms=("resveratrol", "exercise", "adaptation"),
        tension_terms=("null", "positive"),
        receipt_ids=("trial", "mechanism"),
        score=100,
        novelty_score=58,
        evidence_score=90,
        reasons=("shape:promise_outcome_reversal", "tier:publishable_alpha"),
    )
    receipts = [
        _hit("trial", "Resveratrol fails to reduce exercise-training TMAO response", "Human trial reported null TMAO response."),
        _hit("mechanism", "Resveratrol and exercise adaptation in older adults", "Mechanism receipt reported adaptation endpoints."),
    ]
    markdown = "# Alpha memo: resveratrol exercise adaptation promise outcome\n\n**Alpha hypothesis:** bounded signal."

    payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown=markdown),
        author_agent_id="v5-alpha",
        domain_slug="longevity",
    )

    assert payload["title"] == "Resveratrol fails to reduce exercise-training TMAO response"
    assert "Bounded alpha signal" not in cast(str, payload["body_markdown"])


def test_researka_payload_rewrites_slash_bridge_title() -> None:
    candidate = InsightCandidate(
        topic="cold water immersion",
        thesis="Comparator recovery evidence separates cold-water immersion from sports massage.",
        bridge_terms=("cold", "immersion", "water"),
        tension_terms=("negative", "positive"),
        receipt_ids=("review", "trial"),
        score=100,
        novelty_score=58,
        evidence_score=90,
        reasons=("shape:promise_outcome_reversal", "tier:publishable_alpha"),
    )
    receipts = [
        _hit("review", "Cold water immersion recovery review", "Cold water immersion showed mixed recovery evidence."),
        _hit("trial", "Cold water immersion versus massage", "Comparator trial reported sports massage improved ROM."),
    ]
    markdown = "# Alpha memo: cold / immersion / water promise outcome\n\n**Alpha hypothesis:** bounded signal."

    payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown=markdown),
        author_agent_id="v5-alpha",
        domain_slug="longevity",
    )

    assert payload["title"] == "Comparator recovery evidence separates cold-water immersion from sports massage."
    assert " / " not in cast(str, payload["body_markdown"]).splitlines()[0]


def test_minimax_memo_rejects_unsupported_ci_numbers() -> None:
    receipts = [
        CorpusHit(
            hit_id="10.1123/ijspp.2019-0965",
            title="Does Cold-Water Immersion After Strength Training Attenuate Training Adaptation?",
            abstract="Cold-water immersion after strength training reported a control-leg advantage.",
            source="fullraw:semantic_scholar",
            doi="10.1123/ijspp.2019-0965",
        ),
        CorpusHit(
            hit_id="10.1519/jsc.0000000000000434",
            title="Strength Training Adaptations After Cold-Water Immersion",
            abstract="Cold-water immersion and strength training adaptations were compared.",
            source="fullraw:semantic_scholar",
            doi="10.1519/jsc.0000000000000434",
        ),
    ]
    markdown = """
# Alpha memo: Cold-water immersion after strength training

## Core signal
The 1RM effect size confidence interval was -0.42 to 1.04.

## The 2+2=5 angle
The receipts imply a bounded metric-window signal.

## Why this could matter
It keeps the claim hypothesis-level.

## What would break the idea
Direct replication would break it.

## Claim ledger
- 10.1123/ijspp.2019-0965: direct support.
- 10.1519/jsc.0000000000000434: direct support.

## Receipts
- 10.1123/ijspp.2019-0965
- 10.1519/jsc.0000000000000434

## Safety note
Hypothesis only.
""".strip()

    with pytest.raises(MemoFormatError, match="unsupported statistical numbers"):
        validate_minimax_memo(markdown, receipts)


def test_researka_payload_preserves_authenticated_fullraw_coverage() -> None:
    receipt = {
        "shards_total": 100,
        "shards_searched": 48,
        "sources_searched": {"openalex": 24, "semantic_scholar": 24},
        "auth_required": True,
        "authenticated": True,
    }
    candidate = InsightCandidate(
        topic="metformin resistance training adaptation",
        thesis="Metformin protocol and outcome reverse.",
        bridge_terms=("metformin",),
        tension_terms=("blunted",),
        receipt_ids=("a", "b"),
        score=92,
        novelty_score=90,
        evidence_score=94,
        reasons=("shape:promise_outcome_reversal",),
    )
    receipts = [
        CorpusHit(
            hit_id="a",
            title="Metformin augments resistance training protocol",
            abstract="Protocol expected augmentation.",
            source="fullraw:openalex",
            doi="10.a",
            metadata={
                "shard_receipt": receipt,
                "fullraw_search_receipt": {
                    "search_passes": ("focused", "broad"),
                    "auth_required": True,
                    "authenticated": True,
                },
                "search_pass": "focused",
            },
        ),
        CorpusHit(
            hit_id="b",
            title="Metformin blunts resistance training adaptation",
            abstract="Outcome blunted hypertrophy.",
            source="fullraw:semantic_scholar",
            doi="10.b",
            metadata={"shard_receipt": receipt, "search_pass": "broad"},
        ),
    ]
    markdown = render_memo(candidate, receipts)

    payload = build_researka_payload(
        MemoResult(candidate=candidate, receipts=receipts, markdown=markdown),
        author_agent_id="v5-alpha",
        domain_slug="longevity",
    )

    source_bundle = cast(list[dict[str, object]], payload["source_bundle"])
    evidence = cast(dict[str, object], source_bundle[0]["retrieval_evidence"])
    shard_receipt = cast(dict[str, object], evidence["shard_receipt"])
    assert shard_receipt["authenticated"] is True

    evidence_bundle = cast(dict[str, object], payload["evidence_bundle"])
    coverage = cast(dict[str, object], evidence_bundle["fullraw_retrieval_coverage"])
    assert coverage["authenticated"] is True
    assert coverage["auth_required"] is True
    assert coverage["shards_searched"] == 48
    assert coverage["sources_searched"] == ["openalex", "semantic_scholar"]
    assert coverage["search_passes"] == ["broad", "focused"]


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

    with pytest.raises(MemoBuildError, match="no receipt-bound") as exc:
        build_alpha_memo(
            topic="longevity sauna cardiovascular risk",
            seed_queries=["sauna hypertension"],
            searcher=_StaticSearch(hits),
            min_alpha_tier="elite_alpha",
    )
    assert exc.value.failure.details["min_alpha_tier"] == "elite_alpha"
    assert exc.value.failure.details["mined_candidate_count"] == 1
    assert exc.value.failure.details["queries_used"] == ("sauna hypertension",)
    assert exc.value.failure.details["anchor_terms"] == ("sauna", "hypertension")
    top_mined = exc.value.failure.details["top_mined_candidates"]
    assert isinstance(top_mined, tuple)
    assert top_mined[0]["tier"] == "publishable_alpha"
    assert "hit_count=2" in str(exc.value)
    assert "candidate_count=0" in str(exc.value)
    assert "mined_candidate_count=1" in str(exc.value)
    assert "min_alpha_tier=elite_alpha" in str(exc.value)


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


def test_pipeline_fails_closed_on_partial_or_failed_fullraw_coverage() -> None:
    receipt = {
        "shards_total": 1525,
        "shards_searched": 1525,
        "partial_shard_search": True,
        "sweep_failed_shards": 1,
        "sources_searched": {str(idx): 1 for idx in range(5)},
    }
    failure = memo_coverage_failure(
        topic="longevity resilience",
        receipts=[
            CorpusHit("partial-1", "A", "B", "fullraw", metadata={"shard_receipt": receipt}),
            CorpusHit("partial-2", "C", "D", "fullraw", metadata={"shard_receipt": receipt}),
        ],
        min_shards_searched=1525,
        min_sources_searched=5,
    )

    assert failure is not None
    assert failure.details["failures"] == (
        "partial_shard_search",
        "sweep_failed_shards",
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
        min_search_passes=6,
    )

    assert [hit.doi for hit in result.receipts] == ["10.deep/1", "10.deep/2"]
