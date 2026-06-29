import json
from email.message import Message
from urllib.error import HTTPError, URLError
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
from v5_memo.schemas import ClaimCard, CorpusHit, EvidenceNode, InsightCandidate, ReceiptRole


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "FakeResponse":
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


@pytest.mark.parametrize("error_kind", ["http529", "timeout"])
def test_minimax_call_retries_transient_errors(
    monkeypatch: pytest.MonkeyPatch,
    error_kind: str,
) -> None:
    calls: list[Request] = []

    def opener(request: Request, timeout: float) -> FakeResponse:
        del timeout
        calls.append(request)
        if len(calls) == 1:
            if error_kind == "http529":
                headers: Message[str, str] = Message()
                raise HTTPError(request.full_url, 529, "overloaded", headers, None)
            raise URLError(TimeoutError("handshake timed out"))
        return FakeResponse({"content": [{"type": "text", "text": "ok"}]})

    monkeypatch.setattr("v5_memo.llm.minimax_client.time.sleep", lambda _seconds: None)

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
## Claim ledger
- receipt-bound claim: 10.1/sleep-nad support=direct
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


def test_minimax_writer_retries_once_after_invalid_doi_reference() -> None:
    bad_text = """# Alpha memo: NAD mitochondrial sleep exercise bridge
## Core signal
NAD and mitochondrial repair may connect the receipts plus `10.9999/bad`.
## The 2+2=5 angle
Sleep fragmentation and exercise response share the same receipt-bound bridge.
## Why this could matter
The bridge gives a testable resilience hypothesis.
## What would break the idea
The idea breaks if follow-up receipts do not connect the bridge terms.
## Claim ledger
- receipt-bound claim: 10.1/sleep-nad support=direct
## Receipts
- 10.1/sleep-nad
- 10.2/exercise-nad
## Safety note
Hypothesis only."""
    good_text = bad_text.replace(" plus `10.9999/bad`", "")
    opener = FakeOpener([bad_text, good_text])
    writer = MiniMaxM3MemoWriter(api_key="test-key", opener=opener)

    memo = writer.render(_candidate(), _receipts())

    assert "10.9999/bad" not in memo
    assert len(opener.requests) == 2
    request_data = opener.requests[1].data
    assert isinstance(request_data, bytes)
    body = json.loads(request_data.decode("utf-8"))
    retry_prompt = body["messages"][0]["content"][0]["text"]
    assert "Previous draft failed validation" in retry_prompt
    assert "10.9999/bad" in retry_prompt


def test_minimax_prompt_includes_structured_claim_ledger() -> None:
    candidate = InsightCandidate(
        topic="longevity resilience",
        thesis="receipt-bound claim",
        bridge_terms=("nad",),
        tension_terms=("positive", "negative"),
        receipt_ids=("h1", "h2"),
        score=80,
        novelty_score=70,
        evidence_score=75,
        reasons=("tier:publishable_alpha",),
        scorecard={"directional_contrast": 85, "novelty_vs_corpus": 67},
        claim_cards=(
            ClaimCard(
                receipt_id="h1",
                role="outcome",
                design="randomized_trial",
                population="human",
                outcome="performance",
                direction="negative",
                support_type="direct",
                confidence="high",
                quote="Human trial observed a reduced outcome.",
            ),
        ),
        evidence_graph=(
            EvidenceNode("h1", "primary", "primary direct trial"),
            EvidenceNode("h2", "counter", "counter receipt"),
        ),
    )

    prompt = build_minimax_prompt(candidate, _receipts())

    assert "Evidence graph:" in prompt
    assert "h1: primary" in prompt
    assert "Claim ledger:" in prompt
    assert "Scorecard:" in prompt
    assert "- directional_contrast: 85" in prompt
    assert "- novelty_vs_corpus: 67" in prompt
    assert "design=randomized_trial" in prompt
    assert "support=direct/high" in prompt
    assert "safety_feasibility" in prompt
    assert "do not call it positive efficacy" in prompt
    assert "not directly contradictory" in prompt


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
## Claim ledger
- receipt-bound claim: 10.1/sleep-nad support=direct
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
## Claim ledger
- receipt-bound claim: 10.1/sleep-nad support=direct
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
## Claim ledger
- receipt-bound claim: 10.1/sleep-nad support=direct
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


def test_build_minimax_prompt_omits_unsafe_receipt_doi() -> None:
    unsafe = CorpusHit(
        hit_id="https://openalex.org/W4693",
        title="Cold-water immersion protocol review",
        abstract="Cold-water immersion protocol parameters were reviewed.",
        source="openalex",
        doi="10.31435/ijitss.1(49).2026.4693",
        url="https://openalex.org/W4693",
    )

    prompt = build_minimax_prompt(_candidate(), [unsafe, _receipts()[1]])

    assert unsafe.receipt_id == "https://openalex.org/W4693"
    assert "10.31435/ijitss.1(49).2026.4693" not in prompt
    assert "ID: W4693" in prompt
    assert "Locator: https://openalex.org/W4693" in prompt


def test_build_minimax_prompt_contains_domain_agnostic_scope_rules() -> None:
    prompt = build_minimax_prompt(_candidate(), _receipts())

    assert "seed topic as search context only" in prompt
    assert "receipt-owned title" in prompt
    assert "copy receipt/bridge terms verbatim" in prompt
    assert "The title must be made only from locked receipt title/abstract words" in prompt
    assert "Use this exact receipt-owned title first line: # Alpha memo: nad mitochondrial" in prompt
    assert "Scope every implication to the receipts" in prompt
    assert "State the receipt-owned timing exactly" in prompt
    assert "not a direct contradiction" in prompt
    assert 'Do not say "opposite directions"' in prompt
    assert "quantify the protocol/design gap" in prompt
    assert "label the" in prompt and "mixed/comparator-favored" in prompt
    assert "one concrete next-step uncertainty" in prompt
    assert "Respect receipt roles" in prompt
    assert "observed result or confirmed endpoint" in prompt
    assert "population, market" in prompt
    assert "company, channel, model, benchmark" in prompt
    assert "Use source-appropriate descriptors from the receipts" in prompt
    assert "filing/report" in prompt
    assert "case study, market study, campaign" in prompt
    assert "contradiction, boundary condition, inversion" in prompt
    assert "Selector tier:" in prompt
    assert "Receipt roles:" in prompt
    assert "Evidence graph:" in prompt
    assert "metric mismatch" in prompt
    assert "cross-domain transfer" in prompt


def test_build_minimax_prompt_uses_role_verbs_in_safe_title() -> None:
    candidate = InsightCandidate(
        topic="metformin training",
        thesis="Promise/outcome split.",
        bridge_terms=("metformin", "master", "training"),
        tension_terms=("negative", "positive"),
        receipt_ids=("promise", "outcome"),
        score=100,
        novelty_score=50,
        evidence_score=85,
        reasons=("tier:elite_alpha",),
        receipt_roles=(
            ReceiptRole("promise", "promise", "promise/outcome split"),
            ReceiptRole("outcome", "outcome", "promise/outcome split"),
        ),
    )
    receipts = [
        CorpusHit("promise", "Metformin to augment strength training response", "", "fullraw"),
        CorpusHit("outcome", "Metformin blunts muscle hypertrophy in resistance training", "", "fullraw"),
    ]

    prompt = build_minimax_prompt(candidate, receipts)

    assert "# Alpha memo: metformin augment versus blunt master training" in prompt


def test_build_minimax_prompt_filters_weak_bridge_words_from_title() -> None:
    candidate = InsightCandidate(
        topic="caffeine exercise performance",
        thesis="Endpoint boundary.",
        bridge_terms=("caffeine", "during", "exercise"),
        tension_terms=("negative", "positive"),
        receipt_ids=("metabolism", "exhaustion"),
        score=100,
        novelty_score=50,
        evidence_score=85,
        reasons=("tier:elite_alpha",),
    )
    receipts = [
        CorpusHit("metabolism", "Failure of caffeine to affect metabolism during 60 min submaximal exercise", "", "fullraw"),
        CorpusHit("exhaustion", "Caffeine ingestion during exercise to exhaustion in elite distance runners", "", "fullraw"),
    ]

    prompt = build_minimax_prompt(candidate, receipts)

    assert "# Alpha memo: caffeine exercise" in prompt
    assert "# Alpha memo: caffeine during exercise" not in prompt


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
    assert "may omit downstream endpoint words" in prompt
    assert "upstream-promise query" in prompt
    assert "mechanism queries must still" in prompt
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
## Claim ledger
- receipt-bound claim: 10.1/sleep-nad support=direct
## Receipts
- 10.1/sleep-nad
## Safety note
x""",
            _receipts(),
        )


def test_minimax_memo_validation_rejects_missing_claim_ledger() -> None:
    with pytest.raises(ValueError, match="## Claim ledger"):
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
- 10.2/exercise-nad
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
## Claim ledger
- receipt-bound claim: 10.1/sleep-nad support=direct
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
## Claim ledger
- receipt-bound claim: 10.1/sleep-nad support=direct
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
## Claim ledger
- receipt-bound claim: 10.1016/j.molmet.2022.101615 support=direct
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
## Claim ledger
- receipt-bound claim: 10.1016/j.molmet.2022.101615 support=direct
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
## Claim ledger
- receipt-bound claim: 10.1113/jphysiol.2013.258061 support=direct
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
## Claim ledger
- receipt-bound claim: W7070693264 support=direct
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
    assert body["temperature"] == 0.0
    assert "academic corpus search queries" in body["system"] and "observed endpoints/results" in body["messages"][0]["content"][0]["text"]


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
    assert body["temperature"] == 0.0
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
