from __future__ import annotations

import json
from urllib.request import Request

import pytest

from v5_memo.minimax_writer import (
    MiniMaxM3CandidateSelector,
    MiniMaxM3MemoWriter,
    MiniMaxM3SearchPlanner,
    build_minimax_prompt,
    build_minimax_selection_prompt,
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
    def __init__(self, text: str) -> None:
        self.requests: list[Request] = []
        self.timeouts: list[float] = []
        self._text = text

    def __call__(self, request: Request, timeout: float) -> FakeResponse:
        self.timeouts.append(timeout)
        self.requests.append(request)
        return FakeResponse({"content": [{"type": "text", "text": self._text}]})


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


def test_minimax_writer_calls_anthropic_endpoint_and_preserves_receipts() -> None:
    text = """# Alpha memo: longevity resilience
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
    request_data = request.data
    assert isinstance(request_data, bytes)
    body = json.loads(request_data.decode("utf-8"))
    assert body["model"] == "MiniMax-M3"
    assert body["thinking"] == {"type": "disabled"}


def test_minimax_writer_from_env_uses_v5_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    text = """# Alpha memo: longevity resilience
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
