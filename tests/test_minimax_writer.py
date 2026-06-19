from __future__ import annotations

import json
from urllib.request import Request

import pytest

from v5_memo.minimax_writer import (
    FactVerificationError,
    MiniMaxM3CandidateSelector,
    MiniMaxM3MemoWriter,
    MiniMaxM3SearchPlanner,
    build_minimax_prompt,
    build_minimax_repair_prompt,
    build_minimax_scope_repair_prompt,
    build_minimax_selection_prompt,
    parse_minimax_fact_verdict,
    parse_minimax_queries,
    parse_minimax_selection,
    validate_minimax_memo,
    verify_minimax_memo_claims,
)
from v5_memo.schemas import CorpusHit, InsightCandidate


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class FakeOpener:
    def __init__(self, text: str | list[str]) -> None:
        self.requests: list[Request] = []
        self.timeouts: list[float] = []
        self._texts = [text] if isinstance(text, str) else text

    def __call__(self, request: Request, timeout: float) -> FakeResponse:
        self.timeouts.append(timeout)
        self.requests.append(request)
        text = self._texts[min(len(self.requests) - 1, len(self._texts) - 1)]
        return FakeResponse({"content": [{"type": "text", "text": text}]})


def _candidate(
    *,
    thesis: str = "longevity resilience may have a mitochondrial NAD bridge.",
    score: int = 80,
    novelty_score: int = 70,
    evidence_score: int = 90,
) -> InsightCandidate:
    return InsightCandidate(
        topic="longevity resilience",
        thesis=thesis,
        bridge_terms=("nad", "mitochondrial"),
        tension_terms=("positive", "negative"),
        receipt_ids=("h1", "h2"),
        score=score,
        novelty_score=novelty_score,
        evidence_score=evidence_score,
        reasons=("source_diverse",),
    )


def _receipts() -> list[CorpusHit]:
    return [
        CorpusHit(
            hit_id="h1",
            title="NAD salvage links sleep fragmentation to mitochondrial stress",
            abstract="Sleep fragmentation increased inflammatory tone through NAD salvage.",
            source="openalex:full-corpus",
            doi="10.1/sleep-nad",
        ),
        CorpusHit(
            hit_id="h2",
            title="Exercise response tracks mitochondrial repair",
            abstract="Exercise improved resilience when mitochondrial markers moved together.",
            source="openalex:full-corpus",
            doi="10.2/exercise-nad",
        ),
    ]


def _muscle_candidate() -> InsightCandidate:
    return InsightCandidate(
        topic="longevity protein restriction muscle aging tradeoff",
        thesis="Receipt bridge should be narrowed to muscle biology.",
        bridge_terms=("soleus", "protein"),
        tension_terms=("positive", "negative"),
        receipt_ids=("soleus", "hindlimb"),
        score=80,
        novelty_score=80,
        evidence_score=80,
        reasons=("source_diverse",),
    )


def _muscle_receipts() -> list[CorpusHit]:
    return [
        CorpusHit(
            hit_id="soleus",
            title="Resistance exercise enhances mTORC1 sensitivity to leucine in soleus",
            abstract="The effect was observed in soleus but not tibialis anterior or plantaris.",
            source="openalex",
            doi="10.1016/j.molmet.2022.101615",
        ),
        CorpusHit(
            hit_id="hindlimb",
            title="Protein synthesis versus energy state in contracting muscles",
            abstract="Contractions inhibited protein synthesis in tibialis anterior, gastrocnemius, and plantaris but not soleus.",
            source="openalex",
            doi="10.1152/ajpendo.1984.246.4.e297",
        ),
    ]


def test_minimax_writer_calls_anthropic_endpoint_and_preserves_receipts() -> None:
    text = """# Alpha memo: NAD mitochondrial sleep exercise bridge
## Core signal
NAD and mitochondrial repair may connect the receipts.
## The 2+2=5 angle
Sleep fragmentation and exercise response share the same receipt-bound bridge.
## Why this could matter
The bridge gives a testable resilience hypothesis.
## What would break the idea
The idea breaks if follow-up receipts do not connect the bridge terms.
## Receipts
- 10.1/sleep-nad
- 10.2/exercise-nad
## Safety note
Hypothesis only."""
    opener = FakeOpener([text, json.dumps({"pass": True, "reason": "supported"})])
    writer = MiniMaxM3MemoWriter(api_key="test-key", opener=opener)

    memo = writer.render(_candidate(), _receipts())

    assert "10.1/sleep-nad" in memo
    assert "10.2/exercise-nad" in memo
    request = opener.requests[0]
    assert request.full_url == "https://api.minimax.io/anthropic/v1/messages"
    assert len(opener.requests) == 2
    request_data = request.data
    assert isinstance(request_data, bytes)
    body = json.loads(request_data.decode("utf-8"))
    assert body["model"] == "MiniMax-M3"
    assert body["thinking"] == {"type": "disabled"}


def test_minimax_writer_from_env_uses_v5_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    text = """# Alpha memo: NAD mitochondrial receipt bridge
## Core signal
Signal.
## The 2+2=5 angle
Angle.
## Why this could matter
Matter.
## What would break the idea
Break.
## Receipts
- 10.1/sleep-nad
- 10.2/exercise-nad
## Safety note
Hypothesis only."""
    opener = FakeOpener([text, json.dumps({"pass": True, "reason": "supported"})])
    monkeypatch.setenv("V5_MEMO_MINIMAX_API_KEY", "v5-key")
    monkeypatch.setenv("V5_MEMO_MINIMAX_BASE_URL", "https://example.test/minimax")
    monkeypatch.setenv("V5_MEMO_MINIMAX_MODEL", "MiniMax-M3-Test")
    monkeypatch.setenv("V5_MEMO_MINIMAX_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("V5_MEMO_MINIMAX_MAX_TOKENS", "777")

    writer = MiniMaxM3MemoWriter.from_env(opener=opener)
    writer.render(_candidate(), _receipts())

    request = opener.requests[0]
    assert request.full_url == "https://example.test/minimax/v1/messages"
    assert opener.timeouts == [12.5, 12.5]
    request_data = request.data
    assert isinstance(request_data, bytes)
    body = json.loads(request_data.decode("utf-8"))
    assert body["model"] == "MiniMax-M3-Test"
    assert body["max_tokens"] == 777


def test_minimax_writer_rejects_unsupported_endpoint_claim() -> None:
    memo = """# Alpha memo: fish oil resistance exercise
## Core signal
Fish oil improved chair-rise time in older women.
## The 2+2=5 angle
The endpoint translation is sex-specific.
## Why this could matter
It changes supplement positioning.
## What would break the idea
A replication would break it.
## Receipts
- 10.3945/ajcn.116.140780
## Safety note
Research only."""
    receipt = CorpusHit(
        hit_id="omega",
        title="Sex differences in fish-oil supplementation and resistance exercise",
        abstract=(
            "Chair-rise time did not differ between groups. Maximal isometric torque "
            "and muscle quality improved in older women."
        ),
        source="openalex",
        doi="10.3945/ajcn.116.140780",
    )
    failed = json.dumps({
        "pass": False,
        "claim": "Fish oil improved chair-rise time in older women.",
        "receipt_id": "10.3945/ajcn.116.140780",
        "reason": "chair-rise time did not differ; torque and muscle quality improved",
    })
    opener = FakeOpener([memo, failed, memo, failed, memo, failed])
    writer = MiniMaxM3MemoWriter(api_key="test-key", opener=opener)

    with pytest.raises(ValueError, match="chair-rise time did not differ"):
        writer.render(_candidate(), [receipt])


def test_minimax_writer_repairs_then_accepts_failed_fact_check() -> None:
    bad_memo = """# Alpha memo: fish oil resistance exercise
## Core signal
Fish oil improved chair-rise time in older women.
## The 2+2=5 angle
The endpoint translation is sex-specific.
## Why this could matter
It changes supplement positioning.
## What would break the idea
A replication would break it.
## Receipts
- 10.3945/ajcn.116.140780
## Safety note
Research only."""
    repaired_memo = bad_memo.replace(
        "Fish oil improved chair-rise time in older women.",
        "Fish oil improved maximal isometric torque and muscle quality in older women; chair-rise time did not differ.",
    )
    receipt = CorpusHit(
        hit_id="omega",
        title="Sex differences in fish-oil supplementation and resistance exercise",
        abstract=(
            "Chair-rise time did not differ between groups. Maximal isometric torque "
            "and muscle quality improved in older women."
        ),
        source="openalex",
        doi="10.3945/ajcn.116.140780",
    )
    opener = FakeOpener([
        bad_memo,
        json.dumps({
            "pass": False,
            "claim": "Fish oil improved chair-rise time in older women.",
            "receipt_id": "10.3945/ajcn.116.140780",
            "reason": "chair-rise time did not differ; torque and muscle quality improved",
        }),
        repaired_memo,
        json.dumps({"pass": True, "reason": "supported"}),
    ])
    writer = MiniMaxM3MemoWriter(api_key="test-key", opener=opener)

    memo = writer.render(_candidate(), [receipt])

    assert "maximal isometric torque and muscle quality" in memo
    assert len(opener.requests) == 4
    repair_body = opener.requests[2].data
    assert isinstance(repair_body, bytes)
    repair_prompt = json.loads(repair_body.decode("utf-8"))["messages"][0]["content"][0]["text"]
    assert "Repair this alpha memo" in repair_prompt
    assert "chair-rise time did not differ" in repair_prompt


def test_minimax_writer_repairs_seed_topic_overtitle() -> None:
    bad_memo = """# Alpha memo: longevity protein restriction muscle aging tradeoff
## Core signal
Soleus differs from faster muscles.
## The 2+2=5 angle
The receipts split by muscle type.
## Why this could matter
It is a hypothesis.
## What would break the idea
A direct low-protein receipt would break it.
## Receipts
- 10.1016/j.molmet.2022.101615
- 10.1152/ajpendo.1984.246.4.e297
## Safety note
Preclinical only."""
    repaired_memo = bad_memo.replace(
        "# Alpha memo: longevity protein restriction muscle aging tradeoff",
        "# Alpha memo: soleus leucine mTORC1 protein synthesis split",
    )
    opener = FakeOpener([bad_memo, repaired_memo, json.dumps({"pass": True, "reason": "supported"})])
    writer = MiniMaxM3MemoWriter(api_key="test-key", opener=opener)

    memo = writer.render(_muscle_candidate(), _muscle_receipts())

    assert memo.startswith("# Alpha memo: soleus leucine")
    assert len(opener.requests) == 3
    repair_body = opener.requests[1].data
    assert isinstance(repair_body, bytes)
    repair_prompt = json.loads(repair_body.decode("utf-8"))["messages"][0]["content"][0]["text"]
    assert "title/framing" in repair_prompt
    assert "not supported by receipts" in repair_prompt


def test_minimax_writer_repairs_missing_required_sections() -> None:
    bad_memo = """# Alpha memo: NAD mitochondrial split
## Core signal
The receipts move in different directions.
## Why this could matter
It is a receipt-bound signal.
## What would break the idea
A direct receipt could resolve the split.
## Receipts
- 10.1/sleep-nad
- 10.2/exercise-nad
## Safety note
Research only."""
    repaired_memo = bad_memo.replace(
        "## Why this could matter",
        "## The 2+2=5 angle\nThe point is the bridge, not a broad claim.\n## Why this could matter",
    )
    opener = FakeOpener([bad_memo, repaired_memo, json.dumps({"pass": True, "reason": "supported"})])
    writer = MiniMaxM3MemoWriter(api_key="test-key", opener=opener)

    memo = writer.render(_candidate(), _receipts())

    assert "## The 2+2=5 angle" in memo
    assert len(opener.requests) == 3
    repair_body = opener.requests[1].data
    assert isinstance(repair_body, bytes)
    repair_prompt = json.loads(repair_body.decode("utf-8"))["messages"][0]["content"][0]["text"]
    assert "missing required sections" in repair_prompt


def test_minimax_title_guard_ignores_generic_report_terms() -> None:
    memo = """# Alpha memo: two-report NAD mitochondrial split
## Core signal
The receipts move in different directions.
## The 2+2=5 angle
The point is the bridge, not a broad claim.
## Why this could matter
It is a receipt-bound signal.
## What would break the idea
A direct receipt could resolve the split.
## Receipts
- 10.1/sleep-nad
- 10.2/exercise-nad
## Safety note
Research only."""

    assert validate_minimax_memo(memo, _receipts(), candidate=_candidate()).startswith(
        "# Alpha memo: two-report"
    )


def test_build_minimax_repair_prompt_preserves_required_context() -> None:
    error = FactVerificationError(
        claim="The campaign increased click-through rate.",
        receipt_id="10.biz/test",
        reason="click-through was not reported",
    )
    prompt = build_minimax_repair_prompt("## Receipts\n- 10.biz/test", [
        CorpusHit(
            hit_id="biz",
            title="Ad campaign experiment",
            abstract="Conversion did not differ; click-through was not reported.",
            source="openalex",
            doi="10.biz/test",
        )
    ], error)

    assert "click-through was not reported" in prompt
    assert "Keep every receipt ID exactly as written" in prompt
    assert "Do not add new receipts" in prompt


def test_build_minimax_scope_repair_prompt_forces_receipt_owned_title() -> None:
    prompt = build_minimax_scope_repair_prompt(
        "# Alpha memo: longevity protein restriction\n## Receipts\n- 10.x",
        _receipts(),
        "MiniMax memo title used terms not supported by receipts: longevity",
    )

    assert "Retitle around concepts present in the locked receipt" in prompt
    assert "seed topic as search context only" in prompt
    assert "Do not add new receipts" in prompt


def test_minimax_fact_verifier_rejects_unsupported_business_metric() -> None:
    memo = """# Alpha memo: ad testing
## Core signal
The campaign increased click-through rate.
## The 2+2=5 angle
The market signal moved before conversion.
## Why this could matter
It affects channel allocation.
## What would break the idea
A holdout would break it.
## Receipts
- 10.biz/test
## Safety note
Research only."""
    receipt = CorpusHit(
        hit_id="biz",
        title="Ad campaign experiment",
        abstract="Conversion did not differ between treatment and control; click-through was not reported.",
        source="openalex",
        doi="10.biz/test",
    )
    opener = FakeOpener(json.dumps({
        "pass": False,
        "claim": "The campaign increased click-through rate.",
        "receipt_id": "10.biz/test",
        "reason": "click-through was not reported",
    }))

    with pytest.raises(ValueError, match="click-through was not reported"):
        verify_minimax_memo_claims(memo, [receipt], api_key="test-key", opener=opener)


def test_parse_minimax_fact_verdict_accepts_true_and_rejects_fail() -> None:
    parse_minimax_fact_verdict(json.dumps({"pass": True, "reason": "supported"}))

    with pytest.raises(ValueError, match="unsupported claim"):
        parse_minimax_fact_verdict(json.dumps({"pass": False}))


def test_build_minimax_prompt_bounds_long_abstracts() -> None:
    long_hit = CorpusHit(
        hit_id="long",
        title="Very long receipt",
        abstract=" ".join(["mitochondrial"] * 1000),
        source="openalex:full-corpus",
        doi="10.1/long",
    )

    prompt = build_minimax_prompt(_candidate(), [long_hit, _receipts()[1]])

    assert "... [truncated]" in prompt
    assert "Make the memo read like an insight" in prompt
    assert len(prompt) < 5000


def test_build_minimax_prompt_contains_domain_agnostic_scope_rules() -> None:
    prompt = build_minimax_prompt(_candidate(), _receipts())

    assert "seed topic as search context only" in prompt
    assert "receipt-owned title" in prompt
    assert "Scope every implication to the receipts" in prompt
    assert "population, market" in prompt
    assert "company, channel, model, benchmark" in prompt
    assert "Use source-appropriate descriptors from the receipts" in prompt
    assert "filing/report" in prompt
    assert "case study, market study, campaign" in prompt
    assert "contradiction, boundary condition, inversion" in prompt
    assert "metric mismatch" in prompt
    assert "cross-domain transfer" in prompt


def test_minimax_memo_validation_rejects_dropped_receipt_ids() -> None:
    with pytest.raises(ValueError, match="dropped receipt IDs"):
        validate_minimax_memo(
            """# Alpha memo: x
## Core signal
x
## The 2+2=5 angle
x
## Why this could matter
x
## What would break the idea
x
## Receipts
- 10.1/sleep-nad
## Safety note
x""",
            _receipts(),
        )


def test_minimax_memo_validation_rejects_unreceipted_doi_like_references() -> None:
    with pytest.raises(ValueError, match="unreceipted DOI-like"):
        validate_minimax_memo(
            """# Alpha memo: x
## Core signal
x
## The 2+2=5 angle
x
## Why this could matter
x
## What would break the idea
See 10.5555/not-in-receipts.
## Receipts
- 10.1/sleep-nad
- 10.2/exercise-nad
## Safety note
x""",
            _receipts(),
        )


def test_minimax_memo_validation_allows_markdown_emphasized_receipt_dois() -> None:
    memo = validate_minimax_memo(
        """# Alpha memo: x
## Core signal
x
## The 2+2=5 angle
x
## Why this could matter
x
## What would break the idea
x
## Receipts
- **10.1/sleep-nad**
- **10.2/exercise-nad**
## Safety note
x""",
        _receipts(),
    )

    assert "**10.1/sleep-nad**" in memo


def test_minimax_memo_validation_allows_inline_code_receipt_dois() -> None:
    memo = validate_minimax_memo(
        """# Alpha memo: x
## Core signal
x
## The 2+2=5 angle
x
## Why this could matter
x
## What would break the idea
x
## Receipts
- `10.1/sleep-nad`
- `10.2/exercise-nad`
## Safety note
x""",
        _receipts(),
    )

    assert "`10.1/sleep-nad`" in memo


def test_minimax_memo_validation_rejects_seed_topic_overtitle() -> None:
    with pytest.raises(ValueError, match="longevity"):
        validate_minimax_memo(
            """# Alpha memo: longevity protein restriction muscle aging tradeoff
## Core signal
Soleus differs from faster muscles.
## The 2+2=5 angle
The receipts split by muscle type.
## Why this could matter
It is a hypothesis.
## What would break the idea
A direct low-protein receipt would break it.
## Receipts
- 10.1016/j.molmet.2022.101615
- 10.1152/ajpendo.1984.246.4.e297
## Safety note
Preclinical only.""",
            _muscle_receipts(),
            candidate=_muscle_candidate(),
        )


def test_minimax_memo_validation_rejects_invented_non_seed_title_terms() -> None:
    candidate = InsightCandidate(
        topic="muscle translation",
        thesis="Receipt bridge should be narrowed to muscle biology.",
        bridge_terms=("soleus", "protein"),
        tension_terms=("positive", "negative"),
        receipt_ids=("soleus", "hindlimb"),
        score=80,
        novelty_score=80,
        evidence_score=80,
        reasons=("source_diverse",),
    )

    with pytest.raises(ValueError, match="clinical"):
        validate_minimax_memo(
            """# Alpha memo: clinical mortality soleus protein split
## Core signal
Soleus differs from faster muscles.
## The 2+2=5 angle
The receipts split by muscle type.
## Why this could matter
It is a hypothesis.
## What would break the idea
A direct clinical receipt would break it.
## Receipts
- 10.1016/j.molmet.2022.101615
- 10.1152/ajpendo.1984.246.4.e297
## Safety note
Preclinical only.""",
            _muscle_receipts(),
            candidate=candidate,
        )


def test_minimax_memo_validation_rejects_other_unsupported_seed_terms() -> None:
    candidate = InsightCandidate(
        topic="senescence inflammation healthspan muscle translation",
        thesis="Receipt bridge should be narrowed to muscle biology.",
        bridge_terms=("soleus", "protein"),
        tension_terms=("positive", "negative"),
        receipt_ids=("soleus", "hindlimb"),
        score=80,
        novelty_score=80,
        evidence_score=80,
        reasons=("source_diverse",),
    )

    with pytest.raises(ValueError, match="senescence"):
        validate_minimax_memo(
            """# Alpha memo: senescence inflammation healthspan muscle translation
## Core signal
Soleus differs from faster muscles.
## The 2+2=5 angle
The receipts split by muscle type.
## Why this could matter
It is a hypothesis.
## What would break the idea
A direct senescence receipt would break it.
## Receipts
- 10.1016/j.molmet.2022.101615
- 10.1152/ajpendo.1984.246.4.e297
## Safety note
Preclinical only.""",
            _muscle_receipts(),
            candidate=candidate,
        )


def test_minimax_memo_validation_allows_receipt_owned_retitle() -> None:
    memo = validate_minimax_memo(
        """# Alpha memo: soleus leucine mTORC1 protein synthesis split
## Core signal
Soleus differs from faster muscles.
## The 2+2=5 angle
The receipts split by muscle type.
## Why this could matter
It is a hypothesis.
## What would break the idea
A direct low-protein receipt would break it.
## Receipts
- 10.1016/j.molmet.2022.101615
- 10.1152/ajpendo.1984.246.4.e297
## Safety note
Preclinical only.""",
        _muscle_receipts(),
        candidate=_muscle_candidate(),
    )

    assert memo.startswith("# Alpha memo: soleus leucine")


def test_minimax_planner_returns_json_queries_plus_original_seeds() -> None:
    opener = FakeOpener(
        json.dumps(
            [
                "NAD salvage mitochondrial redox exercise",
                "chronotropic response oxidative stress mitochondria",
            ]
        )
    )
    planner = MiniMaxM3SearchPlanner(api_key="test-key", opener=opener)

    queries = planner.plan(
        topic="NAD salvage, mitochondrial stress, and exercise response",
        seed_queries=["NAD salvage mitochondrial stress"],
        limit=2,
    )

    assert queries == [
        "NAD salvage mitochondrial redox exercise",
        "chronotropic response oxidative stress mitochondria",
        "NAD salvage mitochondrial stress",
    ]
    request = opener.requests[0]
    request_data = request.data
    assert isinstance(request_data, bytes)
    body = json.loads(request_data.decode("utf-8"))
    assert body["model"] == "MiniMax-M3"
    assert "academic corpus search queries" in body["system"]


def test_minimax_planner_from_env_uses_v5_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    opener = FakeOpener(json.dumps(["mitochondrial hormesis exercise response"]))
    monkeypatch.setenv("V5_MEMO_MINIMAX_API_KEY", "v5-key")
    monkeypatch.setenv("V5_MEMO_MINIMAX_BASE_URL", "https://example.test/minimax")
    monkeypatch.setenv("V5_MEMO_MINIMAX_MODEL", "MiniMax-M3-Planner")
    monkeypatch.setenv("V5_MEMO_MINIMAX_TIMEOUT_SECONDS", "9.5")
    monkeypatch.setenv("V5_MEMO_MINIMAX_MAX_TOKENS", "333")

    planner = MiniMaxM3SearchPlanner.from_env(opener=opener)
    queries = planner.plan(topic="exercise response", seed_queries=["NAD salvage"], limit=1)

    assert queries == ["mitochondrial hormesis exercise response", "NAD salvage"]
    request = opener.requests[0]
    assert request.full_url == "https://example.test/minimax/v1/messages"
    assert opener.timeouts == [9.5]
    request_data = request.data
    assert isinstance(request_data, bytes)
    body = json.loads(request_data.decode("utf-8"))
    assert body["model"] == "MiniMax-M3-Planner"
    assert body["max_tokens"] == 333


def test_minimax_selector_picks_existing_alpha_candidate() -> None:
    opener = FakeOpener(
        json.dumps({
            "classification": "alpha_memo",
            "candidate": 2,
            "reason": "second candidate has tighter receipt coupling",
        })
    )
    candidates = [
        _candidate(thesis="adjacent bridge", score=61),
        _candidate(thesis="bounded surprise", score=90),
    ]
    selector = MiniMaxM3CandidateSelector(api_key="test-key", opener=opener)

    selected = selector.select(candidates, _receipts())

    assert selected == [candidates[1]]
    request = opener.requests[0]
    request_data = request.data
    assert isinstance(request_data, bytes)
    body = json.loads(request_data.decode("utf-8"))
    assert "strict research alpha selector" in body["system"]
    assert "Select the strongest alpha memo bridge" in body["messages"][0]["content"][0]["text"]


def test_minimax_selector_downgrades_weak_bridge_to_no_selection() -> None:
    opener = FakeOpener(
        json.dumps({
            "classification": "discovery_seed",
            "candidate": 1,
            "reason": "interesting but indirect",
        })
    )
    selector = MiniMaxM3CandidateSelector(api_key="test-key", opener=opener)

    assert selector.select([_candidate()], _receipts()) == []


def test_minimax_selection_prompt_names_weak_bridge_failures() -> None:
    prompt = build_minimax_selection_prompt([_candidate()], _receipts())

    assert "unsupported domain jump" in prompt
    assert "generic \"evidence is mixed\"" in prompt
    assert "Do not invent candidates or receipt IDs" in prompt


def test_parse_minimax_selection_rejects_invalid_alpha_index() -> None:
    with pytest.raises(ValueError, match="invalid candidate index"):
        parse_minimax_selection(
            json.dumps({"classification": "alpha_memo", "candidate": 99}),
            [_candidate()],
        )


def test_parse_minimax_queries_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        parse_minimax_queries("not json", limit=4)
