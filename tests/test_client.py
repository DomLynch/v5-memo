import json
import urllib.parse
from email.message import Message
from http.client import RemoteDisconnected
from io import BytesIO
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest
from pytest import MonkeyPatch

from v5_memo.client import (
    FullRawCorpusSearchClient,
    HybridCorpusSearchClient,
    OpenAlexFullCorpusSearchClient,
    ResearkaSearchClient,
    SearchBackendError,
    _fullraw_search_passes,
    _parse_corpus_search_response,
    _parse_full_raw_search_response,
    _parse_openalex_response,
    _rerank_score,
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

    def __enter__(self) -> "FakeResponse":
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

def test_researka_client_reports_missing_token_as_unconfigured() -> None:
    client = ResearkaSearchClient(base_url="https://database.example", token="")

    assert client.configured is False

def test_full_raw_client_posts_to_configured_search_service(monkeypatch: object) -> None:
    captured: dict[str, object] = {}
    payloads: list[dict[str, object]] = []

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        captured["url"] = request.full_url
        captured["data"] = request.data
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        payloads.append(json.loads(cast(bytes, request.data).decode("utf-8")))
        return FakeResponse({
            "meta": {
                "count": 492361307,
                "shard_receipt": {
                    "shards_searched": 24,
                    "sources_searched": {"semantic_scholar": 24},
                    "auth_required": True,
                    "authenticated": True,
                },
            },
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

    headers = cast(dict[str, str], captured["headers"])
    assert captured["url"] == "https://search.example/full-raw"
    assert captured["timeout"] == 22.0
    assert headers["Authorization"] == "Bearer raw-token"
    assert payloads[0] == {
        "query": ("nad exercise " * 200)[:1024],
        "limit": 50,
        "top_k": 50,
        "year_min": 1950,
        "year_max": 2026,
        "corpus": "full_raw_450m_plus",
        "search_pass": "focused",
        "rank_mode": "relevance",
        "timeout_seconds": 7.0,
    }
    assert any(payload["query"] == "nad exercise" for payload in payloads)
    assert hits[0].source == "fullraw:semantic_scholar"
    assert hits[0].doi == "10.123/raw"
    assert hits[0].metadata["query_match_count"] == 492361307
    assert hits[0].metadata["score"] == 17.2
    search_receipt = cast(dict[str, object], hits[0].metadata["fullraw_search_receipt"])
    assert search_receipt["authenticated"] is True

def test_full_raw_client_relaxes_strict_long_queries(monkeypatch: object) -> None:
    requested: list[str] = []

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        del timeout
        payload = json.loads(cast(bytes, request.data).decode("utf-8"))
        query = cast(str, payload["query"])
        requested.append(query)
        if query != "management forecast":
            return FakeResponse({"meta": {"count": 0}, "results": []})
        return FakeResponse({
            "meta": {"count": 1},
            "results": [{
                "doi": "10.123/forecast",
                "title": "Management forecasts and information asymmetry",
                "abstract": "Management forecasts reveal information asymmetry.",
                "year": 2012,
                "source": "openalex",
            }],
        })

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)  # type: ignore[attr-defined]
    client = FullRawCorpusSearchClient(search_url="https://search.example/full-raw")

    hits = client.search("voluntary management earnings forecast accuracy information asymmetry", limit=10)

    assert "management forecast" in requested
    assert hits[0].doi == "10.123/forecast"
    assert hits[0].metadata["search_variant"] == "management forecast"

def test_full_raw_client_preserves_shard_receipt(monkeypatch: object) -> None:
    receipt = {
        "shards_total": 100,
        "shards_searched": 24,
        "partial_shard_search": True,
        "sources_searched": {"openalex": 12, "semantic_scholar": 12},
    }

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        del request, timeout
        return FakeResponse({
            "meta": {"count": 1, "shard_receipt": receipt},
            "results": [{
                "doi": "10.123/receipt",
                "title": "Management forecast disclosure breadth",
                "abstract": "Management forecast disclosure evidence.",
                "year": 2024,
                "source": "openalex",
            }],
        })

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)  # type: ignore[attr-defined]
    client = FullRawCorpusSearchClient(search_url="https://search.example/full-raw", max_variants=1)

    hits = client.search("management forecast disclosure", limit=3)

    assert hits[0].metadata["shard_receipt"] == receipt


def test_full_raw_client_requires_auth_receipt_when_token_configured(monkeypatch: object) -> None:
    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        del request, timeout
        return FakeResponse({
            "meta": {
                "count": 1,
                "shard_receipt": {
                    "shards_searched": 24,
                    "sources_searched": {"openalex": 24},
                },
            },
            "results": [{
                "doi": "10.123/no-auth-proof",
                "title": "Management forecast disclosure breadth",
                "abstract": "Management forecast disclosure evidence.",
                "year": 2024,
                "source": "openalex",
            }],
        })

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)  # type: ignore[attr-defined]
    client = FullRawCorpusSearchClient(
        search_url="https://search.example/full-raw",
        token="raw-token",
        max_variants=1,
    )

    assert client.search("management forecast disclosure", limit=3) == []

def test_full_raw_rerank_prefers_title_owned_hits_over_abstract_only_hits() -> None:
    abstract_only = CorpusHit("a", "Quality child care", "Metformin resistance training adaptation.", "openalex")
    title_owned = CorpusHit("t", "Metformin resistance training adaptation trial", "", "openalex")
    terms = ("metformin", "resistance", "training", "adaptation")
    assert _rerank_score(title_owned, seed_terms=terms, variant_terms=terms, rank=2) > _rerank_score(
        abstract_only,
        seed_terms=terms,
        variant_terms=terms,
        rank=1,
    )

def test_full_raw_client_filters_hits_without_rare_query_anchor(monkeypatch: object) -> None:
    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        del request, timeout
        return FakeResponse({
            "meta": {"count": 2},
            "results": [
                {"doi": "10.123/generic", "title": "Resistance training adaptation", "abstract": "Resistance training adaptation in older adults.", "year": 2024, "source": "openalex"},
                {"doi": "10.123/metformin", "title": "Metformin blunts resistance training adaptation", "abstract": "Metformin altered resistance training adaptation.", "year": 2024, "source": "openalex"},
            ],
        })

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)  # type: ignore[attr-defined]
    client = FullRawCorpusSearchClient(search_url="https://search.example/full-raw", max_variants=1)

    hits = client.search("metformin resistance training adaptation", limit=5)

    assert [hit.doi for hit in hits] == ["10.123/metformin"]

def test_full_raw_client_waits_for_async_sweep_cache_hit(monkeypatch: object) -> None:
    payloads: list[dict[str, object]] = []
    timeouts: list[float] = []

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        timeouts.append(timeout)
        payload = json.loads(cast(bytes, request.data).decode("utf-8"))
        payloads.append(payload)
        if payload.get("cache_only") is True:
            return FakeResponse({
                "meta": {
                    "count": 1,
                    "shard_receipt": {
                        "shards_total": 100,
                        "shards_searched": 48,
                        "sources_searched": {"openalex": 24, "semantic_scholar": 24},
                    },
                    "async_sweep": {"status": "hit"},
                },
                "results": [{
                    "doi": "10.123/deep",
                    "title": "Deep sweep evidence",
                    "abstract": "Management forecast disclosure evidence from a deep sweep.",
                    "year": 2024,
                    "source": "semantic_scholar",
                }],
            })
        if len(payloads) == 3:
            return FakeResponse({
                "meta": {"count": 0, "async_sweep": {"status": "queued"}},
                "results": [],
            })
        return FakeResponse({
            "meta": {
                "count": 1,
                "shard_receipt": {"shards_total": 100, "shards_searched": 12},
                "async_sweep": {"status": "queued"},
            },
            "results": [{
                "doi": "10.123/shallow",
                "title": "Shallow foreground evidence",
                "abstract": "Management forecast disclosure evidence.",
                "year": 2024,
                "source": "openalex",
            }],
        })

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)  # type: ignore[attr-defined]
    client = FullRawCorpusSearchClient(
        search_url="https://search.example/full-raw",
        max_variants=2,
        sweep_wait_seconds=1.0,
        min_shards_searched=48,
        min_sources_searched=2,
    )

    hits = client.search("management forecast disclosure", limit=3)

    assert [payload.get("cache_only") for payload in payloads] == [None, True, None]
    assert payloads[1].get("queue_if_missing") is True
    assert timeouts[0] == 60.0
    assert timeouts[1] <= 1.0
    assert hits[0].doi == "10.123/deep"
    assert hits[0].metadata["shard_receipt"] == {
        "shards_total": 100,
        "shards_searched": 48,
        "sources_searched": {"openalex": 24, "semantic_scholar": 24},
    }


def test_full_raw_client_waits_for_zero_hit_foreground_sweep(monkeypatch: object) -> None:
    payloads: list[dict[str, object]] = []

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        del timeout
        payload = json.loads(cast(bytes, request.data).decode("utf-8"))
        payloads.append(payload)
        if payload.get("cache_only") is True:
            return FakeResponse({
                "meta": {"count": 1, "shard_receipt": {"shards_searched": 48}, "async_sweep": {"status": "hit"}},
                "results": [{"doi": "10.123/sweep", "title": "Cold water immersion adaptation", "source": "openalex"}],
            })
        return FakeResponse({
            "meta": {"count": 0, "shard_receipt": {"shards_searched": 12}, "async_sweep": {"status": "queued"}},
            "results": [],
        })

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)  # type: ignore[attr-defined]
    client = FullRawCorpusSearchClient(search_url="https://search.example/full-raw", max_variants=1, sweep_wait_seconds=1.0)

    hits = client.search("cold water immersion adaptation", limit=3)

    assert [payload.get("cache_only") for payload in payloads] == [None, True]
    assert hits[0].doi == "10.123/sweep"


def test_full_raw_client_keeps_sufficient_foreground_hit(monkeypatch: object) -> None:
    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        del timeout
        payload = json.loads(cast(bytes, request.data).decode("utf-8"))
        assert payload.get("cache_only") is not True
        return FakeResponse({
            "meta": {
                "count": 1,
                "shard_receipt": {"shards_searched": 48, "sources_searched": {"openalex": 24, "pubmed": 24}},
            },
            "results": [{"doi": "10.123/foreground", "title": "Metformin longevity foreground evidence", "source": "openalex"}],
        })

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)  # type: ignore[attr-defined]
    client = FullRawCorpusSearchClient(
        search_url="https://search.example/full-raw",
        max_variants=1,
        sweep_wait_seconds=1.0,
        min_shards_searched=48,
        min_sources_searched=2,
    )

    assert client.search("metformin longevity", limit=3)[0].doi == "10.123/foreground"


def test_full_raw_client_does_not_wait_on_unknown_non_strict_empty_response(
    monkeypatch: object,
) -> None:
    payloads: list[dict[str, object]] = []

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        del timeout
        payloads.append(json.loads(cast(bytes, request.data).decode("utf-8")))
        raise TimeoutError("foreground too slow")

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)  # type: ignore[attr-defined]
    client = FullRawCorpusSearchClient(
        search_url="https://search.example/full-raw",
        max_variants=1,
        sweep_wait_seconds=1.0,
    )

    assert client.search("resveratrol exercise training", limit=3) == []
    assert [payload.get("cache_only") for payload in payloads] == [None]


def test_full_raw_client_recovers_non_strict_coverage_error_from_sweep(
    monkeypatch: object,
) -> None:
    payloads: list[dict[str, object]] = []

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        del timeout
        payload = json.loads(cast(bytes, request.data).decode("utf-8"))
        payloads.append(payload)
        if payload.get("cache_only") is True:
            return FakeResponse({
                "meta": {
                    "count": 1,
                    "shard_receipt": {"shards_searched": 32, "sources_searched": {"openalex": 32}},
                    "async_sweep": {"status": "hit"},
                },
                "results": [{"doi": "10.123/metformin", "title": "Metformin longevity evidence", "source": "openalex"}],
            })
        body = json.dumps({
            "error": "coverage_too_narrow",
            "shard_receipt": {"shards_searched": 0, "sources_searched": {}},
        }).encode("utf-8")
        raise HTTPError(
            url="https://search.example/full-raw",
            code=422,
            msg="coverage_too_narrow",
            hdrs=Message(),
            fp=BytesIO(body),
        )

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)  # type: ignore[attr-defined]
    client = FullRawCorpusSearchClient(
        search_url="https://search.example/full-raw",
        max_variants=1,
        sweep_wait_seconds=1.0,
        min_shards_searched=1,
        min_sources_searched=1,
    )

    hits = client.search("metformin longevity", limit=3)

    assert [payload.get("cache_only") for payload in payloads] == [None, True]
    assert hits[0].doi == "10.123/metformin"


def test_full_raw_client_keeps_prior_hits_when_later_strict_variant_fails(monkeypatch: object) -> None:
    calls = 0

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        del request, timeout
        nonlocal calls
        calls += 1
        if calls > 1:
            raise TimeoutError("later variant failed")
        return FakeResponse({
            "meta": {"count": 1, "shard_receipt": {"shards_searched": 1, "authenticated": True}},
            "results": [{"doi": "10.123/kept", "title": "Metformin longevity kept hit", "source": "openalex"}],
        })

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)  # type: ignore[attr-defined]
    client = FullRawCorpusSearchClient(search_url="https://search.example/full-raw", token="t", max_variants=2, strict=True)

    assert client.search("metformin longevity", limit=3)[0].doi == "10.123/kept"


def test_full_raw_client_uses_cache_only_after_strict_foreground_timeout(
    monkeypatch: object,
) -> None:
    payloads: list[dict[str, object]] = []

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        del timeout
        payload = json.loads(cast(bytes, request.data).decode("utf-8"))
        payloads.append(payload)
        if payload.get("cache_only") is True:
            return FakeResponse({
                "meta": {
                    "count": 1,
                    "shard_receipt": {
                        "shards_total": 100,
                        "shards_searched": 48,
                        "sources_searched": {"openalex": 24, "semantic_scholar": 24},
                    },
                    "async_sweep": {"status": "hit"},
                },
                "results": [{
                    "doi": "10.123/recovered",
                    "title": "Recovered cache-only evidence",
                    "abstract": "Management forecast disclosure recovered from cache-only sweep.",
                    "year": 2024,
                    "source": "semantic_scholar",
                }],
            })
        raise TimeoutError("foreground too slow")

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)  # type: ignore[attr-defined]
    client = FullRawCorpusSearchClient(
        search_url="https://search.example/full-raw",
        max_variants=1,
        sweep_wait_seconds=1.0,
        min_shards_searched=48,
        min_sources_searched=2,
        strict=True,
    )

    hits = client.search("management forecast disclosure", limit=3)

    assert [payload.get("cache_only") for payload in payloads] == [None, True]
    assert payloads[1].get("queue_if_missing") is True
    assert hits[0].doi == "10.123/recovered"

def test_full_raw_client_retries_connection_reset_during_cache_poll(
    monkeypatch: object,
) -> None:
    payloads: list[dict[str, object]] = []
    reset_once = True

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        nonlocal reset_once
        del timeout
        payload = json.loads(cast(bytes, request.data).decode("utf-8"))
        payloads.append(payload)
        if payload.get("cache_only") is not True:
            raise TimeoutError("foreground too slow")
        if reset_once:
            reset_once = False
            raise ConnectionResetError("reset")
        return FakeResponse({
            "meta": {
                "count": 1,
                "shard_receipt": {
                    "shards_total": 100,
                    "shards_searched": 48,
                    "sources_searched": {"openalex": 24, "semantic_scholar": 24},
                },
                "async_sweep": {"status": "hit"},
            },
            "results": [{
                "doi": "10.123/recovered",
                "title": "Recovered cache-only evidence",
                "abstract": "Management forecast disclosure recovered from cache-only sweep.",
                "year": 2024,
                "source": "semantic_scholar",
            }],
        })

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)  # type: ignore[attr-defined]
    client = FullRawCorpusSearchClient(
        search_url="https://search.example/full-raw",
        max_variants=1,
        sweep_wait_seconds=1.0,
        min_shards_searched=48,
        min_sources_searched=2,
        strict=True,
    )

    hits = client.search("management forecast disclosure", limit=3)

    assert [payload.get("cache_only") for payload in payloads] == [None, True, True]
    assert hits[0].doi == "10.123/recovered"

def test_full_raw_client_sends_search_pass_receipts(monkeypatch: object) -> None:
    requested: list[dict[str, object]] = []

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        del timeout
        payload = json.loads(cast(bytes, request.data).decode("utf-8"))
        requested.append(payload)
        search_pass = str(payload["search_pass"])
        rank_mode = str(payload["rank_mode"])
        return FakeResponse({
            "meta": {"count": 1, "shard_receipt": {"shards_searched": 12}},
            "results": [{
                "doi": f"10.123/{search_pass}-{rank_mode}",
                "title": f"{search_pass} evidence",
                "abstract": "Management forecast disclosure evidence.",
                "year": 2024,
                "source": "openalex",
            }],
        })

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)  # type: ignore[attr-defined]
    client = FullRawCorpusSearchClient(search_url="https://search.example/full-raw", max_variants=8)

    hits = client.search("management forecast disclosure", limit=10)

    assert [payload["search_pass"] for payload in requested] == [
        "focused",
        "broad",
        "broad",
        "broad",
        "anchor",
        "adjacent",
        "falsifier",
        "citation_heavy",
    ]
    assert [payload["rank_mode"] for payload in requested] == [
        "relevance",
        "relevance",
        "relevance",
        "relevance",
        "relevance",
        "relevance",
        "relevance",
        "citation",
    ]
    assert {hit.metadata["search_pass"] for hit in hits} >= {
        "focused",
        "broad",
        "anchor",
        "adjacent",
        "falsifier",
        "citation_heavy",
    }
    assert {hit.metadata["rank_mode"] for hit in hits} == {"relevance", "citation"}

def test_full_raw_client_records_duplicate_rate_across_passes(monkeypatch: object) -> None:
    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        del request, timeout
        return FakeResponse({
            "meta": {"count": 1, "shard_receipt": {"shards_searched": 12}},
            "results": [{
                "doi": "10.123/shared",
                "title": "Shared fullraw receipt",
                "abstract": "Management forecast disclosure evidence.",
                "year": 2024,
                "source": "openalex",
            }],
        })

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)  # type: ignore[attr-defined]
    client = FullRawCorpusSearchClient(search_url="https://search.example/full-raw", max_variants=4)

    hits = client.search("management forecast disclosure", limit=10)

    receipt = hits[0].metadata["fullraw_search_receipt"]
    assert isinstance(receipt, dict)
    assert receipt["duplicate_rate"] == 0.75
    assert receipt["search_passes"] == ("focused", "broad", "anchor")
    assert receipt["rank_modes"] == ("relevance",)

def test_full_raw_client_can_fail_closed_on_narrow_shard_receipt(monkeypatch: object) -> None:
    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        del request, timeout
        return FakeResponse({
            "meta": {
                "count": 1,
                "shard_receipt": {
                    "shards_total": 100,
                    "shards_searched": 3,
                    "sources_searched": {"openalex": 3},
                },
            },
            "results": [{
                "doi": "10.123/narrow",
                "title": "Narrow pull",
                "year": 2024,
                "source": "openalex",
            }],
        })

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)  # type: ignore[attr-defined]
    client = FullRawCorpusSearchClient(
        search_url="https://search.example/full-raw",
        max_variants=1,
        min_shards_searched=12,
        min_sources_searched=2,
    )

    assert client.search("management forecast disclosure", limit=3) == []

def test_full_raw_client_from_env_requires_only_url(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9902/search")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_TOKEN", "secret")

    client = FullRawCorpusSearchClient.from_env()

    assert client.configured is True
    assert client._search_url == "http://127.0.0.1:9902/search"
    assert client._token == "secret"
    assert client._require_auth is True

def test_full_raw_client_loads_timeout_from_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9902/search")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_TIMEOUT", "999")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_MAX_VARIANTS", "7")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SEARCH_BUDGET_SECONDS", "7200")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_FOREGROUND_SWEEP_WAIT_SECONDS", "7200")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SWEEP_POLL_SECONDS", "2")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_PROGRESS", "true")

    client = FullRawCorpusSearchClient.from_env()

    assert client._timeout == 30.0
    assert client._max_variants == 4
    assert client._search_budget_seconds == 900.0
    assert client._sweep_wait_seconds == 240.0
    assert client._sweep_poll_seconds == 2.0
    assert client._progress is True

def test_full_raw_client_budget_stops_variant_fanout(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requested: list[str] = []
    ticks = iter([0.0, 0.0, 0.0, 1.0, 31.0, 31.0])

    def fake_monotonic() -> float:
        return next(ticks)

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        del timeout
        payload = json.loads(cast(bytes, request.data).decode("utf-8"))
        requested.append(cast(str, payload["query"]))
        return FakeResponse({
            "meta": {"count": 1},
            "results": [{
                "doi": "10.123/fullraw",
                "title": "Cold water immersion resistance training",
                "abstract": "Cold water immersion attenuates resistance training adaptation.",
                "year": 2020,
                "source": "openalex",
            }],
        })

    monkeypatch.setattr("v5_memo.client.time.monotonic", fake_monotonic)
    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)
    client = FullRawCorpusSearchClient(
        search_url="https://search.example/full-raw",
        search_budget_seconds=30.0,
        max_variants=4,
        progress=True,
    )

    hits = client.search("cold water immersion resistance training", limit=5)

    assert len(requested) == 1
    assert hits[0].doi == "10.123/fullraw"
    err = capsys.readouterr().err
    assert "fullraw variant 1/4 start" in err
    assert "fullraw search budget reached after 31.0s; variants=1/4" in err

def test_full_raw_client_retries_remote_disconnected_once(monkeypatch: MonkeyPatch) -> None:
    calls = 0

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        nonlocal calls
        del request, timeout
        calls += 1
        if calls == 1:
            raise RemoteDisconnected("closed")
        return FakeResponse({
            "results": [{
                "doi": "10.123/fullraw",
                "title": "Resveratrol exercise training adaptation",
                "abstract": "Resveratrol exercise training adaptation changed older human outcomes.",
                "year": 2013,
                "source": "openalex",
            }],
        })

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)
    client = FullRawCorpusSearchClient(search_url="https://search.example/full-raw", max_variants=1)

    hits = client.search("resveratrol exercise training adaptation", limit=5)

    assert calls == 2
    assert hits[0].doi == "10.123/fullraw"

def test_full_raw_client_strict_remote_disconnected_raises_after_retry_cap(
    monkeypatch: MonkeyPatch,
) -> None:
    calls = 0

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        nonlocal calls
        del request, timeout
        calls += 1
        raise RemoteDisconnected("closed")

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)
    client = FullRawCorpusSearchClient(
        search_url="https://search.example/full-raw",
        max_variants=1,
        strict=True,
    )

    with pytest.raises(SearchBackendError, match="Full raw corpus search failed"):
        client.search("resveratrol exercise training adaptation", limit=5)
    assert calls == 2

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

def test_openalex_strict_from_env_uses_bounded_fanout(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("V5_MEMO_OPENALEX_MAX_VARIANTS", raising=False)

    client = OpenAlexFullCorpusSearchClient.from_env(strict=True)

    assert client._max_variants == 1

def test_openalex_retries_once_after_rate_limit(monkeypatch: MonkeyPatch) -> None:
    calls = 0
    sleeps: list[float] = []
    headers = Message()
    headers["Retry-After"] = "0.01"

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        nonlocal calls
        del request, timeout
        calls += 1
        if calls == 1:
            raise HTTPError("https://api.example", 429, "Too Many Requests", headers, None)
        return FakeResponse({
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "display_name": "Resveratrol exercise adaptation",
                    "abstract_inverted_index": {"Resveratrol": [0], "adaptation": [2]},
                }
            ]
        })

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)
    monkeypatch.setattr("v5_memo.client.time.sleep", lambda seconds: sleeps.append(seconds))

    hits = OpenAlexFullCorpusSearchClient(base_url="https://api.example", strict=True).search(
        "resveratrol adaptation",
        limit=1,
    )

    assert calls == 2
    assert sleeps == [0.01]
    assert hits[0].title == "Resveratrol exercise adaptation"

def test_openalex_lenient_rate_limit_returns_empty_without_sleep(monkeypatch: MonkeyPatch) -> None:
    sleeps: list[float] = []
    headers = Message()
    headers["Retry-After"] = "60"

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        del request, timeout
        raise HTTPError("https://api.example", 429, "Too Many Requests", headers, None)

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)
    monkeypatch.setattr("v5_memo.client.time.sleep", lambda seconds: sleeps.append(seconds))

    hits = OpenAlexFullCorpusSearchClient(base_url="https://api.example").search(
        "resveratrol adaptation",
        limit=1,
    )

    assert hits == []
    assert sleeps == []

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

def test_hybrid_search_skips_failed_backend() -> None:
    class FailingSearch:
        def search(self, query: str, *, limit: int = 25) -> list[CorpusHit]:
            del query, limit
            raise SearchBackendError("rate limited")

    class StaticSearch:
        def search(self, query: str, *, limit: int = 25) -> list[CorpusHit]:
            del query, limit
            return [
                CorpusHit(
                    hit_id="10.good",
                    title="Resveratrol exercise adaptation",
                    abstract="Human trial evidence.",
                    source="fullraw:semantic_scholar",
                    doi="10.good",
                )
            ]

    hits = HybridCorpusSearchClient([FailingSearch(), StaticSearch()]).search("resveratrol")

    assert [hit.doi for hit in hits] == ["10.good"]

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

@pytest.mark.parametrize(
    ("query", "limit", "expected"),
    [
        ("cold water immersion attenuates muscle mass strength resistance training", 8, ("attenuates resistance",)),
        ("resveratrol exercise training adaptation", 5, ("resveratrol", "resveratrol training")),
        ("resveratrol sirt1 pgc 1a mitochondrial biogenesis endurance training", 5, ("resveratrol mitochondrial",)),
        ("metformin augment strength training seniors", 5, ("metformin augment",)),
        ("nmn supplementation vo2max adaptation trained cyclists", 5, ("nmn vo2max",)),
        ("metformin resistance training adaptation", 6, ("metformin training", "resistance training")),
        ("metformin expected to augment resistance training hypertrophy protocol", 4, ("metformin augment",)),
    ],
)
def test_fullraw_search_passes_preserve_key_recall_shapes(query: str, limit: int, expected: tuple[str, ...]) -> None:
    queries = [search_pass.query for search_pass in _fullraw_search_passes(query, limit=limit)]

    assert all(item in queries for item in expected)

def test_fullraw_rerank_prefers_abstract_backed_doi_receipts(monkeypatch: MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        del request, timeout
        return FakeResponse({
            "results": [
                {
                    "title": "Resveratrol exercise training older men",
                    "year": 2024,
                    "provider": "semantic_scholar",
                },
                {
                    "doi": "https://doi.org/10.1113/jphysiol.2013.258061",
                    "title": "Resveratrol exercise training older men",
                    "abstract": (
                        "Resveratrol exercise training older men adaptation cardiovascular "
                        "health maximal oxygen uptake blood pressure cholesterol blunted."
                    ),
                    "year": 2013,
                    "provider": "openalex",
                },
            ],
        })

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)
    client = FullRawCorpusSearchClient(search_url="https://fullraw.example/search")

    hits = client.search("resveratrol exercise training older men", limit=2)

    assert hits[0].doi == "10.1113/jphysiol.2013.258061"
    assert hits[0].abstract

def test_fullraw_search_backfills_missing_doi_abstracts(monkeypatch: MonkeyPatch) -> None:
    seen_urls: list[str] = []

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        del timeout
        seen_urls.append(request.full_url)
        if "api.openalex.org/works/doi:" in request.full_url:
            return FakeResponse({
                "id": "https://openalex.org/W123",
                "doi": "https://doi.org/10.1093/geroni/igy023.2009",
                "display_name": "Metformin to augment strength training effective response in seniors",
                "publication_year": 2018,
                "abstract_inverted_index": {
                    "Protocol": [0],
                    "hypothesized": [1],
                    "metformin": [2],
                    "would": [3],
                    "augment": [4],
                    "training": [5],
                },
                "primary_location": {"source": {"display_name": "Innovation in Aging"}},
            })
        return FakeResponse({
            "results": [
                {
                    "doi": "https://doi.org/10.1093/geroni/igy023.2009",
                    "title": "Metformin to augment strength training effective response in seniors",
                    "year": 2018,
                    "provider": "semantic_scholar",
                },
            ],
        })

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)
    client = FullRawCorpusSearchClient(
        search_url="https://fullraw.example/search",
        max_variants=1,
        doi_abstract_backfill_limit=1,
    )

    hits = client.search("metformin augment strength training", limit=1)

    assert hits[0].abstract == "Protocol hypothesized metformin would augment training"
    assert hits[0].metadata["abstract_backfill"] == "openalex_doi"
    assert any("api.openalex.org/works/doi:" in url for url in seen_urls)

def test_fullraw_search_drops_doi_title_mismatch(monkeypatch: MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        del timeout
        if "api.openalex.org/works/doi:" in request.full_url:
            if "10.1016%2fs0008-6363%2895%2900018-6" in request.full_url.casefold():
                return FakeResponse({
                    "id": "https://openalex.org/W1000",
                    "doi": "https://doi.org/10.1016/s0008-6363(95)00018-6",
                    "display_name": "Reactive hyperaemia is impaired in hypertrophied guinea pig hearts",
                    "publication_year": 1995,
                    "abstract_inverted_index": {"Reactive": [0], "hyperaemia": [1]},
                })
            return FakeResponse({
                "id": "https://openalex.org/W999",
                "doi": "https://doi.org/10.1161/01.cir.0000129233.51320.92",
                "display_name": "Omega-3 fatty acids and atrial fibrillation prevention",
                "publication_year": 2004,
                "abstract_inverted_index": {"Omega": [0], "prevention": [1]},
            })
        return FakeResponse({
            "results": [
                {
                    "doi": "https://doi.org/10.1161/01.CIR.0000129233.51320.92",
                    "title": "LVAD recovery correlates with sarcoplasmic reticulum calcium content",
                    "abstract": "LVAD clinical recovery correlated with sarcoplasmic reticulum calcium content.",
                    "year": 2004,
                    "provider": "openalex",
                },
                {
                    "doi": "https://doi.org/10.1016/s0008-6363(95)00018-6",
                    "title": "Reactive hyperaemia is impaired in hypertrophied guinea pig hearts",
                    "abstract": "Reactive hyperaemia was impaired in hypertrophied guinea pig hearts.",
                    "year": 1995,
                    "provider": "openalex",
                },
            ],
        })

    monkeypatch.setattr("v5_memo.client.urlopen", fake_urlopen)
    client = FullRawCorpusSearchClient(
        search_url="https://fullraw.example/search",
        max_variants=1,
        doi_abstract_backfill_limit=5,
    )

    hits = client.search("recovery hypertrophy", limit=2)

    assert [hit.title for hit in hits] == [
        "Reactive hyperaemia is impaired in hypertrophied guinea pig hearts"
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

def test_parse_full_raw_search_rejects_conflicting_doi_year_metadata() -> None:
    hits = _parse_full_raw_search_response({
        "results": [
            {
                "doi": "https://doi.org/10.1152/japplphysiol.00007.2024",
                "title": "Unrelated title with mismatched DOI metadata",
                "abstract": "Exercise adaptation terms appear in corrupted metadata.",
                "year": 2016,
                "provider": "semantic_scholar",
            },
            {
                "doi": "https://doi.org/10.1152/japplphysiol.00007.2024",
                "title": "Exercise adaptation paper",
                "abstract": "Exercise adaptation terms appear in clean metadata.",
                "year": 2024,
                "provider": "semantic_scholar",
            },
        ],
    })

    assert [hit.year for hit in hits] == [2024]

def test_parse_full_raw_search_keeps_valid_doi_article_codes_that_look_like_years() -> None:
    hits = _parse_full_raw_search_response({
        "results": [
            {
                "doi": "https://doi.org/10.1093/GERONI/IGY023.2009",
                "title": "Metformin to augment strength training effective response in seniors",
                "abstract": "The MASTERS trial tested whether metformin augments strength training response.",
                "year": 2018,
                "provider": "openalex",
            },
        ],
    })

    assert [hit.doi for hit in hits] == ["10.1093/GERONI/IGY023.2009"]
    assert hits[0].year == 2018

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
