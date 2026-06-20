from __future__ import annotations

import json
import urllib.parse
from typing import cast
from urllib.error import URLError
from urllib.request import Request

import pytest
from pytest import MonkeyPatch

from v5_memo.client import (
    FullRawCorpusSearchClient,
    HybridCorpusSearchClient,
    OpenAlexFullCorpusSearchClient,
    ResearkaSearchClient,
    SearchBackendError,
    _parse_corpus_search_response,
    _parse_full_raw_search_response,
    _parse_openalex_response,
    _query_variants,
)
from v5_memo.schemas import CorpusHit


class FakeResponse:
    def __init__(self, payload: object | None = None) -> None:
        self._payload = payload or [
            {
                "pmid": "19587680",
                "doi": "10.1038/nature08221",
                "pmcid": "",
                "title": "Rapamycin fed late in life extends lifespan",
                "abstract": "Full paper abstract text.",
                "year": 2009,
                "journal": "Nature",
                "cited_by_count": 1000,
                "similarity_score": 0.91,
            }
        ]

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_client_posts_to_full_corpus_search(monkeypatch: object) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        captured["url"] = request.full_url
        captured["data"] = request.data
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)  # type: ignore[attr-defined]
    client = ResearkaSearchClient(
        base_url="https://database.example/",
        token="secret",
        timeout=3.0,
        year_min=1950,
        year_max=2030,
    )

    hits = client.search("rapamycin " * 200, limit=250)

    payload = json.loads(cast(bytes, captured["data"]).decode("utf-8"))
    assert captured["url"] == "https://database.example/api/v1/corpus/search"
    assert captured["timeout"] == 3.0
    assert payload == {
        "query": ("rapamycin " * 200)[:1024],
        "top_k": 200,
        "year_min": 1950,
        "year_max": 2030,
    }
    assert hits[0].source == "researka:corpus"


def test_researka_client_loads_first_allowlist_token(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARKA_DATABASE_TOKEN", raising=False)
    monkeypatch.delenv("RESEARKA_TOKEN", raising=False)
    monkeypatch.setenv("RESEARKA_TOKENS", "bot-token:60/m,reader-token:10/m")

    client = ResearkaSearchClient.from_env()

    assert client._token == "bot-token"


def test_researka_strict_mode_rejects_missing_configuration() -> None:
    client = ResearkaSearchClient(base_url="https://database.example", token="", strict=True)

    with pytest.raises(SearchBackendError, match="not configured"):
        client.search("rapamycin", limit=1)


def test_full_raw_client_posts_to_configured_search_service(monkeypatch: object) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        captured["url"] = request.full_url
        captured["data"] = request.data
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        return FakeResponse({
            "meta": {"count": 492361307},
            "results": [
                {
                    "doi": "https://doi.org/10.123/raw",
                    "title": "Raw full corpus NAD exercise signal",
                    "abstract": "NAD salvage and exercise response are linked.",
                    "year": 2024,
                    "journal": "Full Corpus Journal",
                    "source": "semantic_scholar",
                    "score": 17.2,
                }
            ],
        })

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)  # type: ignore[attr-defined]
    client = FullRawCorpusSearchClient(
        search_url="https://search.example/full-raw",
        token="raw-token",
        timeout=7.0,
        year_min=1950,
        year_max=2026,
    )

    hits = client.search("nad exercise " * 200, limit=250)

    payload = json.loads(cast(bytes, captured["data"]).decode("utf-8"))
    headers = cast(dict[str, str], captured["headers"])
    assert captured["url"] == "https://search.example/full-raw"
    assert captured["timeout"] == 7.0
    assert headers["Authorization"] == "Bearer raw-token"
    assert payload == {
        "query": ("nad exercise " * 200)[:1024],
        "limit": 200,
        "top_k": 200,
        "year_min": 1950,
        "year_max": 2026,
        "corpus": "full_raw_450m_plus",
        "timeout_seconds": 7.0,
    }
    assert hits[0].source == "fullraw:semantic_scholar"
    assert hits[0].doi == "10.123/raw"
    assert hits[0].metadata["query_match_count"] == 492361307
    assert hits[0].metadata["score"] == 17.2


def test_full_raw_client_from_env_requires_only_url(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9902/search")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_TOKEN", "secret")

    client = FullRawCorpusSearchClient.from_env()

    assert client.configured is True
    assert client._search_url == "http://127.0.0.1:9902/search"
    assert client._token == "secret"


def test_full_raw_client_loads_timeout_from_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9902/search")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_TIMEOUT", "120")

    client = FullRawCorpusSearchClient.from_env()

    assert client._timeout == 120.0


def test_openalex_strict_mode_raises_backend_errors(monkeypatch: MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        del request, timeout
        raise URLError("offline")

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)
    client = OpenAlexFullCorpusSearchClient(base_url="https://api.example", strict=True)

    with pytest.raises(SearchBackendError, match="OpenAlex search failed"):
        client.search("nad salvage", limit=1)


def test_openalex_lenient_mode_keeps_empty_result_on_backend_errors(monkeypatch: MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        del request, timeout
        raise URLError("offline")

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)
    client = OpenAlexFullCorpusSearchClient(base_url="https://api.example")

    assert client.search("nad salvage", limit=1) == []


def test_hybrid_search_merges_and_dedupes_sources() -> None:
    class StaticSearch:
        def __init__(self, hits: list[CorpusHit]) -> None:
            self._hits = hits

        def search(self, query: str, *, limit: int = 25) -> list[CorpusHit]:
            del query
            return self._hits[:limit]

    shared = CorpusHit(
        hit_id="10.same",
        title="Shared paper",
        abstract="NAD salvage",
        source="researka:corpus",
        doi="10.same",
    )
    openalex_duplicate = CorpusHit(
        hit_id="10.same",
        title="Shared paper duplicate",
        abstract="NAD salvage",
        source="openalex:full-corpus",
        doi="10.same",
    )
    openalex_only = CorpusHit(
        hit_id="10.only",
        title="OpenAlex only",
        abstract="mitochondrial stress",
        source="openalex:full-corpus",
        doi="10.only",
    )

    hits = HybridCorpusSearchClient([
        StaticSearch([shared]),
        StaticSearch([openalex_duplicate, openalex_only]),
    ]).search("nad", limit=5)

    assert [hit.doi for hit in hits] == ["10.same", "10.only"]
    assert hits[0].source == "researka:corpus"


def test_openalex_client_fans_out_dedupes_and_reranks(monkeypatch: object) -> None:
    captured_queries: list[str] = []

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        assert timeout == 3.0
        params = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)
        query = params["search"][0]
        captured_queries.append(query)
        return FakeResponse(
            {
                "meta": {
                    "count": 604
                    if query == "nad salvage mitochondrial stress exercise response"
                    else 8274
                },
                "results": [
                    {
                        "id": f"https://openalex.org/{query}",
                        "doi": "https://doi.org/10.best",
                        "display_name": "NAD salvage and mitochondrial stress improve exercise response",
                        "abstract_inverted_index": {
                            "NAD": [0],
                            "salvage": [1],
                            "mitochondrial": [3],
                            "stress": [4],
                            "exercise": [6],
                            "response": [7],
                        },
                        "publication_year": 2024,
                        "primary_location": {"source": {"display_name": "Aging Cell"}},
                        "cited_by_count": 100,
                    },
                    {
                        "id": f"https://openalex.org/noisy-{query}",
                        "doi": f"https://doi.org/10.noisy/{len(captured_queries)}",
                        "display_name": "Exercise response paper",
                        "abstract_inverted_index": {"exercise": [0], "response": [1]},
                        "publication_year": 2022,
                        "primary_location": {"source": {"display_name": "Journal"}},
                        "cited_by_count": 10,
                    },
                ],
            }
        )

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)  # type: ignore[attr-defined]
    client = OpenAlexFullCorpusSearchClient(
        base_url="https://api.example",
        timeout=3.0,
        year_min=2000,
        year_max=2026,
        max_variants=3,
    )

    hits = client.search("NAD salvage mitochondrial stress exercise response", limit=5)

    assert captured_queries == [
        "nad salvage mitochondrial stress exercise response",
        "nad salvage mitochondrial stress",
        "salvage mitochondrial stress exercise",
    ]
    assert len([hit for hit in hits if hit.doi == "10.best"]) == 1
    assert hits[0].doi == "10.best"
    assert hits[0].source == "openalex:full-corpus"
    assert hits[0].metadata["search_variant"] == "nad salvage mitochondrial stress exercise response"
    assert hits[0].metadata["query_match_count"] == 604


def test_query_variants_do_not_depend_on_one_long_query() -> None:
    assert _query_variants("NAD salvage mitochondrial stress exercise response", limit=5) == [
        "nad salvage mitochondrial stress exercise response",
        "nad salvage mitochondrial stress",
        "salvage mitochondrial stress exercise",
        "mitochondrial stress exercise response",
        "nad salvage mitochondrial",
    ]


def test_parse_openalex_response_reconstructs_abstract() -> None:
    hits = _parse_openalex_response(
        {
            "meta": {"count": 492361307},
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "doi": "https://doi.org/10.123/test",
                    "display_name": "<i>NAD salvage</i>",
                    "abstract_inverted_index": {"NAD": [0], "salvage": [1], "works": [2]},
                    "publication_year": 2026,
                    "primary_location": {"source": {"display_name": "Nature"}},
                    "cited_by_count": 12,
                }
            ],
        }
    )

    hit = hits[0]
    assert hit.title == "NAD salvage"
    assert hit.abstract == "NAD salvage works"
    assert hit.url == "https://doi.org/10.123/test"
    assert hit.venue == "Nature"
    assert hit.metadata["query_match_count"] == 492361307


def test_parse_corpus_search_rejects_non_list_shape() -> None:
    hits = _parse_corpus_search_response({"results": []})
    assert hits == []


def test_parse_full_raw_search_accepts_results_shape_and_openalex_abstract() -> None:
    hits = _parse_full_raw_search_response({
        "meta": {"total": "280000000"},
        "results": [
            {
                "id": "https://openalex.org/W1",
                "doi": "https://doi.org/10.456/full",
                "display_name": "OpenAlex raw storage item",
                "abstract_inverted_index": {"NAD": [0], "repair": [1]},
                "publication_year": 2025,
                "primary_location": {"source": {"display_name": "Nature Aging"}},
                "provider": "openalex",
                "cited_by_count": "42",
            }
        ],
    })

    hit = hits[0]
    assert hit.source == "fullraw:openalex"
    assert hit.doi == "10.456/full"
    assert hit.abstract == "NAD repair"
    assert hit.venue == "Nature Aging"
    assert hit.metadata["query_match_count"] == 280000000
    assert hit.metadata["cited_by_count"] == 42


def test_parse_full_corpus_paper_hit() -> None:
    hits = _parse_corpus_search_response([
        {
            "pmid": "19587680",
            "doi": "10.1038/nature08221",
            "pmcid": "PMC2786175",
            "title": "Rapamycin fed late in life extends lifespan in genetically heterogeneous mice",
            "abstract": "Rapamycin fed late in life extended lifespan in female mice.",
            "authors": [],
            "year": 2009,
            "journal": "Nature",
            "cited_by_count": 1000,
            "similarity_score": 0.91,
            "metadata": {},
        }
    ])

    hit = hits[0]
    assert hit is not None
    assert hit.source == "researka:corpus"
    assert hit.doi == "10.1038/nature08221"
    assert hit.year == 2009
    assert hit.url == "https://doi.org/10.1038/nature08221"
    assert hit.venue == "Nature"
    assert hit.metadata["pmid"] == "19587680"
    assert hit.metadata["pmcid"] == "PMC2786175"
    assert hit.metadata["cited_by_count"] == 1000
    assert hit.metadata["similarity_score"] == 0.91


def test_corpus_parser_strips_html_titles() -> None:
    hits = _parse_corpus_search_response([
        {
            "pmid": "1",
            "doi": "",
            "pmcid": "",
            "title": "<p>The beneficial effects of metformin</p>",
            "abstract": "Metformin reduced cancer risk.",
            "year": 2019,
            "journal": "Aging",
            "cited_by_count": 7,
            "similarity_score": "0.5",
        }
    ])

    hit = hits[0]
    assert hit is not None
    assert hit.title == "The beneficial effects of metformin"
    assert hit.hit_id == "1"
    assert hit.metadata["similarity_score"] == 0.5
