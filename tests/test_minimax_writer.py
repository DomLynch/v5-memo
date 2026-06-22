from __future__ import annotations

import json
from email.message import Message
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from v5_memo.minimax_writer import (
    MiniMaxM3CandidateSelector,
    MiniMaxM3MemoWriter,
    MiniMaxM3SearchPlanner,
    build_minimax_prompt,
    build_minimax_search_prompt,
    build_minimax_selection_prompt,
    call_minimax_m3,
    parse_minimax_queries,
    parse_minimax_selection,
    validate_minimax_memo,
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


def test_minimax_call_retries_transient_http_529(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Request] = []

    def opener(request: Request, timeout: float) -> FakeResponse:
        del timeout
        calls.append(request)
        if len(calls) == 1:
            headers: Message[str, str] = Message()
            raise HTTPError(request.full_url, 529, "overloaded", headers, None)
        return FakeResponse({"content": [{"type": "text", "text": "ok"}]})

    monkeypatch.setattr("v5_memo.minimax_writer.time.sleep", lambda _seconds: None)

    assert call_minimax_m3(
        api_key="key",
        prompt="p",
        system="s",
        temperature=0.1,
        max_tokens=10,
        opener=opener,
    ) == "ok"
    assert len(calls) == 2


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
    opener = FakeOpener(text)
    writer = MiniMaxM3MemoWriter(api_key="test-key", opener=opener)

    memo = writer.render(_candidate(), _receipts())

    assert "10.1/sleep-nad" in memo
    assert "10.2/exercise-nad" in memo
    request = opener.requests[0]
    assert request.full_url == "https://api.minimax.io/anthropic/v1/messages"
    assert len(opener.requests) == 1
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
    opener = FakeOpener(text)
    monkeypatch.setenv("V5_MEMO_MINIMAX_API_KEY", "v5-key")
    monkeypatch.setenv("V5_MEMO_MINIMAX_BASE_URL", "https://example.test/minimax")
    monkeypatch.setenv("V5_MEMO_MINIMAX_MODEL", "MiniMax-M3-Test")
    monkeypatch.setenv("V5_MEMO_MINIMAX_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("V5_MEMO_MINIMAX_MAX_TOKENS", "777")

    writer = MiniMaxM3MemoWriter.from_env(opener=opener)
    writer.render(_candidate(), _receipts())

    request = opener.requests[0]
    assert request.full_url == "https://example.test/minimax/v1/messages"
    assert opener.timeouts == [12.5]
    request_data = request.data
    assert isinstance(request_data, bytes)
    body = json.loads(request_data.decode("utf-8"))
    assert body["model"] == "MiniMax-M3-Test"
    assert body["max_tokens"] == 777


def test_minimax_title_guard_ignores_generic_report_terms() -> None:
    memo = """# Alpha memo: two-report abstract readout NAD mitochondrial split
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


def test_minimax_title_guard_ignores_generic_phrasing_terms() -> None:
    memo = """# Alpha memo: NAD mitochondrial status lifts even while suppresses levels leaving
## Core signal
NAD and mitochondrial repair may connect the receipts.
## The 2+2=5 angle
The point is the bridge, not broad phrasing.
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
        "# Alpha memo: NAD mitochondrial"
    )


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
    assert "copy receipt/bridge terms verbatim" in prompt
    assert "Scope every implication to the receipts" in prompt
    assert "population, market" in prompt
    assert "company, channel, model, benchmark" in prompt
    assert "Use source-appropriate descriptors from the receipts" in prompt
    assert "filing/report" in prompt
    assert "case study, market study, campaign" in prompt
    assert "contradiction, boundary condition, inversion" in prompt
    assert "Selector tier:" in prompt
    assert "Receipt roles:" in prompt
    assert "metric mismatch" in prompt
    assert "cross-domain transfer" in prompt


def test_minimax_planner_prompt_prefers_reversal_pairs_not_reviews() -> None:
    prompt = build_minimax_search_prompt(
        topic="exercise adaptation supplement reversal",
        seed_queries=["exercise adaptation"],
        limit=4,
    )

    assert "promise->outcome reversal" in prompt
    assert "observed/blunted/attenuated/impaired/null/reduced" in prompt
    assert "Split those two evidence sides into separate title-like queries" in prompt
    assert "Return adjacent query pairs when possible" in prompt
    assert "must share the same specific intervention" in prompt
    assert "same intervention/construct/program" in prompt
    assert "At least half the queries must name a specific intervention" in prompt
    assert "Avoid broad review, meta-analysis, position-stand" in prompt


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


@pytest.mark.parametrize("receipt_line", ["**10.1/sleep-nad**", "`10.1/sleep-nad`"])
def test_minimax_memo_validation_allows_wrapped_receipt_dois(receipt_line: str) -> None:
    memo = validate_minimax_memo(
        f"""# Alpha memo: x
## Core signal
x
## The 2+2=5 angle
x
## Why this could matter
x
## What would break the idea
x
## Receipts
- {receipt_line}
- `10.2/exercise-nad`
## Safety note
x""",
        _receipts(),
    )

    assert receipt_line in memo


@pytest.mark.parametrize(
    ("topic", "title", "match"),
    [
        ("longevity protein restriction muscle aging tradeoff", "longevity protein restriction muscle aging tradeoff", "longevity"),
        ("muscle translation", "clinical mortality soleus protein split", "clinical"),
        (
            "senescence inflammation healthspan muscle translation",
            "senescence inflammation healthspan muscle translation",
            "senescence",
        ),
    ],
)
def test_minimax_memo_validation_rejects_unreceipted_title_terms(
    topic: str,
    title: str,
    match: str,
) -> None:
    candidate = _muscle_candidate() if "longevity" in topic else InsightCandidate(
        topic=topic,
        thesis="Receipt bridge should be narrowed to muscle biology.",
        bridge_terms=("soleus", "protein"),
        tension_terms=("positive", "negative"),
        receipt_ids=("soleus", "hindlimb"),
        score=80,
        novelty_score=80,
        evidence_score=80,
        reasons=("source_diverse",),
    )

    with pytest.raises(ValueError, match=match):
        validate_minimax_memo(
            f"""# Alpha memo: {title}
## Core signal
Soleus differs from faster muscles.
## The 2+2=5 angle
The receipts split by muscle type.
## Why this could matter
It is a hypothesis.
## What would break the idea
A direct receipt would break it.
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


def test_minimax_title_guard_allows_light_inflection_and_grammar_words() -> None:
    memo = validate_minimax_memo(
        """# Alpha memo: resveratrol blunting while exercise training
## Core signal
Resveratrol may blunt training adaptation.
## The 2+2=5 angle
The title is still receipt-owned despite light grammar.
## Why this could matter
It is a receipt-bound signal.
## What would break the idea
A direct receipt could reverse the direction.
## Receipts
- 10.1113/jphysiol.2013.258061
- 10.1016/j.cell.2006.11.013
## Safety note
Hypothesis only.""",
        [
            CorpusHit(
                hit_id="human",
                title="Resveratrol blunts the positive effects of exercise training",
                abstract="Older men receiving resveratrol had reduced exercise training benefits.",
                source="openalex",
                doi="10.1113/jphysiol.2013.258061",
            ),
            CorpusHit(
                hit_id="mouse",
                title="Resveratrol improves mitochondrial function and exercise performance",
                abstract="Resveratrol improved mitochondrial biology and running endurance in mice.",
                source="openalex",
                doi="10.1016/j.cell.2006.11.013",
            ),
        ],
        candidate=InsightCandidate(
            topic="resveratrol exercise adaptation",
            thesis="Resveratrol promise may reverse in human training.",
            bridge_terms=("resveratrol", "exercise", "training"),
            tension_terms=("positive", "negative"),
            receipt_ids=("human", "mouse"),
            score=90,
            novelty_score=90,
            evidence_score=90,
            reasons=("shape:promise_outcome_reversal",),
        ),
    )

    assert memo.startswith("# Alpha memo: resveratrol blunting")


def test_minimax_title_guard_accepts_fibre_spelling_variant() -> None:
    receipts = [
        CorpusHit(
            hit_id="cwi",
            title="Cold water immersion attenuates skeletal muscle fiber hypertrophy",
            abstract="The trial reported lower skeletal muscle fiber hypertrophy after cold water immersion.",
            source="openalex",
            doi="10.1152/japplphysiol.00127.2019",
        ),
        CorpusHit(
            hit_id="recovery",
            title="Cold water immersion and resistance training recovery",
            abstract="The trial measured resistance training recovery after cold water immersion.",
            source="openalex",
            doi="10.1249/mss.0000000000001269",
        ),
    ]
    candidate = InsightCandidate(
        topic="cold water immersion resistance training adaptation",
        thesis="Cold water immersion may split recovery narrative from adaptation endpoints.",
        bridge_terms=("cold", "water", "immersion"),
        tension_terms=("negative", "null"),
        receipt_ids=("cwi", "recovery"),
        score=90,
        novelty_score=70,
        evidence_score=90,
        reasons=("source_diverse", "tier:elite_alpha"),
    )

    memo = validate_minimax_memo(
        """# Alpha memo: cold water immersion fibre hypertrophy
## Core signal
Cold water immersion changes the adaptation signal.
## The 2+2=5 angle
The receipts split recovery language from fibre hypertrophy.
## Why this could matter
It challenges recovery marketing.
## What would break the idea
A direct trial showing adaptation benefits would break it.
## Receipts
- 10.1152/japplphysiol.00127.2019
- 10.1249/mss.0000000000001269
## Safety note
Receipt-bound only.""",
        receipts,
        candidate=candidate,
    )

    assert "fibre hypertrophy" in memo


def test_minimax_title_guard_allows_alpha_role_terms() -> None:
    receipts = [
        CorpusHit(
            hit_id="cwi",
            title="Cold water immersion attenuates skeletal muscle fiber hypertrophy",
            abstract="The trial reported lower skeletal muscle fiber hypertrophy after cold water immersion.",
            source="openalex",
            doi="10.1152/japplphysiol.00127.2019",
        ),
        CorpusHit(
            hit_id="recovery",
            title="Cold water immersion and resistance training recovery",
            abstract="The trial measured resistance training recovery after cold water immersion.",
            source="openalex",
            doi="10.1249/mss.0000000000001269",
        ),
    ]
    candidate = InsightCandidate(
        topic="cold water immersion resistance training adaptation",
        thesis="Cold water immersion may split recovery narrative from adaptation endpoints.",
        bridge_terms=("cold", "water", "immersion"),
        tension_terms=("negative", "null"),
        receipt_ids=("cwi", "recovery"),
        score=90,
        novelty_score=70,
        evidence_score=90,
        reasons=("shape:promise_outcome_reversal", "tier:elite_alpha"),
    )

    memo = validate_minimax_memo(
        """# Alpha memo: cold water immersion promise outcome split
## Core signal
Cold water immersion changes the adaptation signal.
## The 2+2=5 angle
The receipts split recovery language from fiber hypertrophy.
## Why this could matter
It challenges recovery marketing.
## What would break the idea
A direct trial showing adaptation benefits would break it.
## Receipts
- 10.1152/japplphysiol.00127.2019
- 10.1249/mss.0000000000001269
## Safety note
Receipt-bound only.""",
        receipts,
        candidate=candidate,
    )

    assert "promise outcome split" in memo


def test_minimax_title_guard_allows_temporal_grammar_terms() -> None:
    receipt = CorpusHit(
        hit_id="cwi",
        title="Cold water immersion resistance training adaptation",
        abstract="Cold water immersion changed resistance training adaptation.",
        source="openalex",
        doi="10.5555/cwi",
    )
    candidate = InsightCandidate(
        topic="cold water immersion resistance training adaptation",
        thesis="Cold water immersion may alter adaptation.",
        bridge_terms=("cold", "immersion"),
        tension_terms=("negative",),
        receipt_ids=("cwi",),
        score=90,
        novelty_score=70,
        evidence_score=90,
        reasons=("tier:elite_alpha",),
    )

    memo = validate_minimax_memo(
        """# Alpha memo: cold immersion after resistance training
## Core signal
Cold water immersion changes adaptation.
## The 2+2=5 angle
Temporal grammar should not count as drift.
## Why this could matter
It keeps valid titles from failing.
## What would break the idea
Unsupported domain nouns still fail.
## Receipts
- 10.5555/cwi
## Safety note
Receipt-bound only.""",
        [receipt],
        candidate=candidate,
    )

    assert memo.startswith("# Alpha memo: cold immersion after")


def test_minimax_memo_validation_accepts_displayed_openalex_work_id() -> None:
    receipt = CorpusHit(
        hit_id="https://openalex.org/W7070693264",
        title="Exercise training adaptation response",
        abstract="Training improved adaptation markers.",
        source="fullraw",
    )
    memo = validate_minimax_memo(
        """# Alpha memo: exercise training adaptation
## Core signal
The receipt is cited by source-local ID.
## The 2+2=5 angle
This is only a receipt preservation check.
## Why this could matter
It keeps non-DOI receipts traceable.
## What would break the idea
A missing source-local ID breaks it.
## Receipts
- W7070693264
## Safety note
Traceability only.""",
        [receipt],
        candidate=InsightCandidate(
            topic="exercise adaptation",
            thesis="Receipt preservation check.",
            bridge_terms=("exercise", "training"),
            tension_terms=(),
            receipt_ids=("https://openalex.org/W7070693264",),
            score=70,
            novelty_score=70,
            evidence_score=70,
            reasons=("source_diverse",),
        ),
    )

    assert "W7070693264" in memo


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
    assert "promise/mechanism receipt plus one observed outcome receipt" in prompt
    assert "only commonality is a broad endpoint" in prompt
    assert "Reject review/position-stand/meta-analysis/survey plus one trial" in prompt
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
