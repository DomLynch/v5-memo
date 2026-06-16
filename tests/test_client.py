from __future__ import annotations

import json
from typing import cast
from urllib.request import Request

from v5_memo.client import ResearkaSearchClient, _parse_corpus_search_response


class FakeResponse:
    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps([
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
        ]).encode("utf-8")


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


def test_parse_corpus_search_rejects_non_list_shape() -> None:
    hits = _parse_corpus_search_response({"results": []})
    assert hits == []


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
