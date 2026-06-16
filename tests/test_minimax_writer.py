from __future__ import annotations

import json
from urllib.request import Request

import pytest

from v5_memo.minimax_writer import (
    MiniMaxM3MemoWriter,
    MiniMaxM3SearchPlanner,
    parse_minimax_queries,
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
        self._text = text

    def __call__(self, request: Request, timeout: float) -> FakeResponse:
        del timeout
        self.requests.append(request)
        return FakeResponse({"content": [{"type": "text", "text": self._text}]})


def _candidate() -> InsightCandidate:
    return InsightCandidate(
        topic="longevity resilience",
        thesis="longevity resilience may have a mitochondrial NAD bridge.",
        bridge_terms=("nad", "mitochondrial"),
        tension_terms=("positive", "negative"),
        receipt_ids=("h1", "h2"),
        score=80,
        novelty_score=70,
        evidence_score=90,
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


def test_minimax_memo_validation_rejects_dropped_receipt_ids() -> None:
    with pytest.raises(ValueError, match="dropped receipt IDs"):
        validate_minimax_memo(
            "# Alpha memo: x\n## Receipts\n- 10.1/sleep-nad\n## Safety note\nx",
            _receipts(),
        )


def test_minimax_planner_returns_json_queries_plus_original_seeds() -> None:
    opener = FakeOpener(
        json.dumps([
            "NAD salvage mitochondrial redox exercise",
            "chronotropic response oxidative stress mitochondria",
        ])
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


def test_parse_minimax_queries_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        parse_minimax_queries("not json", limit=4)
