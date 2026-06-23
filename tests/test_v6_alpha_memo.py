from __future__ import annotations

import json
from typing import cast
from urllib.request import Request

from v6_alpha_memo import (
    FullrawSearchClient,
    Paper,
    mine_pairs,
    query_shapes,
    render_memo,
    score_pairs,
)
from v6_alpha_memo.run import DemoClient, build_memo
from v6_alpha_memo.search import CoverageReceipt, RequestOpener, SearchResult


def test_query_shapes_are_targeted_but_not_topic_whitelisted() -> None:
    queries = query_shapes("marketing attribution incrementality")

    assert len(queries) >= 6
    assert all("marketing attribution incrementality" in query for query in queries)
    assert any("protocol expected result mismatch" in query for query in queries)
    assert any("replication failure" in query for query in queries)


def test_scores_elite_reversal_geometry_without_topic_hardcoding() -> None:
    papers = (
        Paper(
            paper_id="a",
            title="Tool X improves benchmark accuracy in a mechanistic model",
            abstract="The model showed tool x enhanced accuracy and improved performance.",
            source="openalex",
        ),
        Paper(
            paper_id="b",
            title="Tool X failed to improve human analyst decisions in a randomized field trial",
            abstract="Human analysts using tool x had null results and reduced decision quality.",
            source="semantic_scholar",
        ),
    )

    scored = score_pairs(mine_pairs(papers))

    assert scored
    assert scored[0].score >= 85
    assert scored[0].shape in {"promise_reversal", "mechanism_to_human_failure"}
    assert "made us expect" in scored[0].expectation_update


def test_rejects_review_keyword_overlap_before_writing() -> None:
    papers = (
        Paper("a", "Systematic review of leadership and productivity", "productivity evidence", "openalex"),
        Paper("b", "Review of leadership productivity studies", "productivity evidence", "pubmed"),
    )

    assert mine_pairs(papers) == ()


def test_demo_run_outputs_required_memo_and_trace() -> None:
    run = build_memo("longevity exercise adaptation", client=DemoClient())

    assert run.memo.startswith("# Alpha memo:")
    assert "**One-sentence alpha:**" in run.memo
    assert "**Receipt 1:**" in run.memo
    assert run.top_pairs[0].score >= 85
    assert run.trace["top_pairs"]


def test_anchors_drop_generic_connector_words() -> None:
    run = build_memo("management dashboard forecast accuracy", client=DemoClient())

    assert "and" not in run.top_pairs[0].pair.anchors
    assert "dashboard" in run.top_pairs[0].pair.anchors


def test_fullraw_client_parses_hits_and_coverage_receipt() -> None:
    payload: dict[str, object] = {
        "meta": {
            "shard_receipt": {
                "shards_searched": 965,
                "shards_total": 1397,
                "papers_searched": 648767345,
                "papers_total": 1379119449,
                "sources_searched": {"openalex": 100, "pubmed": 10},
                "partial_shard_search": True,
            }
        },
        "results": [
            {
                "id": "W1",
                "title": "Metformin protects cells from oxidative stress",
                "abstract": "Metformin protected cells in a mechanism model.",
                "source": "openalex",
                "year": 2020,
                "doi": "10.test/metformin",
            }
        ],
    }
    client = FullrawSearchClient(
        search_url="http://fullraw/search",
        token="token",
        opener=_fake_opener(payload),
    )

    result = client.search("metformin oxidative stress", limit=3)

    assert result.receipt.hits == 1
    assert result.receipt.shards_searched == 965
    assert "openalex" in result.receipt.sources_searched
    assert result.papers[0].doi == "10.test/metformin"


def test_fullraw_client_compacts_zero_hit_queries() -> None:
    calls: list[str] = []
    payload: dict[str, object] = {
        "meta": {"shard_receipt": {"shards_searched": 50, "sources_searched": {"openalex": 1}}},
        "results": [],
    }
    hit_payload: dict[str, object] = {
        "meta": {"shard_receipt": {"shards_searched": 50, "sources_searched": {"openalex": 1}}},
        "results": [
            {
                "id": "W1",
                "title": "Metformin blunted exercise adaptation",
                "abstract": "Metformin reduced exercise adaptation in humans.",
                "source": "openalex",
            }
        ],
    }

    def opener(request: Request, timeout: float) -> _Response:
        del timeout
        raw = cast(bytes, request.data or b"{}")
        body = json.loads(raw.decode())
        calls.append(body["query"])
        return _Response(hit_payload if body["query"] == "metformin exercise" else payload)

    client = FullrawSearchClient(search_url="http://fullraw/search", opener=opener)
    result = client.search("metformin exercise adaptation expected improved null outcome randomized trial")

    assert calls[:2] == [
        "metformin exercise adaptation expected improved null outcome randomized trial",
        "metformin exercise",
    ]
    assert result.papers[0].title == "Metformin blunted exercise adaptation"


def test_writer_stays_receipt_owned() -> None:
    run = build_memo("longevity exercise adaptation", client=DemoClient())
    memo = render_memo(run.top_pairs[0])

    assert "longevity/business/AI" not in memo
    assert "Resveratrol" in memo


def test_build_memo_rejects_topic_irrelevant_search_noise() -> None:
    class IrrelevantClient:
        def search(self, query: str, *, limit: int = 25) -> SearchResult:
            del query, limit
            papers = (
                Paper(
                    "a",
                    "Resveratrol activates mitochondrial pathways in mice",
                    "Resveratrol improved endurance in a mouse model.",
                    "openalex",
                ),
                Paper(
                    "b",
                    "Resveratrol blunted human exercise training adaptation",
                    "Resveratrol reduced training gains in human participants.",
                    "pubmed",
                ),
            )
            return SearchResult("noise", papers, CoverageReceipt(hits=2))

    try:
        build_memo("AI retrieval augmented generation factuality", client=IrrelevantClient())
    except RuntimeError as exc:
        assert "no elite receipt-geometry pair" in str(exc)
    else:
        raise AssertionError("irrelevant receipt pair should not pass")


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _fake_opener(payload: dict[str, object]) -> RequestOpener:
    def opener(request: Request, timeout: float) -> _Response:
        assert request.get_header("Authorization") == "Bearer token"
        assert timeout > 0
        return _Response(payload)

    return opener
