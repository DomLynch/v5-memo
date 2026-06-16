from __future__ import annotations

from v5_memo.client import _parse_search_response


def test_parse_tier2_fact_response_shape() -> None:
    hits = _parse_search_response({
        "results": [],
    })
    assert hits == []


def test_tier2_fact_parser_via_fallback_method() -> None:
    from v5_memo.client import _parse_tier2_fact

    hit = _parse_tier2_fact({
        "id": "rapamycin/itp/harrison_2009/lifespan_female",
        "paper_id": "10.1038/nature08221",
        "paper": {
            "title": "Rapamycin fed late in life extends lifespan in genetically heterogeneous mice",
            "doi": "10.1038/nature08221",
            "publication_year": 2009,
            "journal_name": "Nature",
        },
        "canonical_phrase": "Rapamycin increased female lifespan by 14%.",
        "source_excerpt": "Rapamycin fed late in life extended lifespan in female mice.",
        "extraction_confidence": "canonical",
    })

    assert hit is not None
    assert hit.source == "researka:tier2"
    assert hit.doi == "10.1038/nature08221"
    assert hit.year == 2009
    assert "female lifespan" in hit.abstract


def test_tier2_parser_strips_html_titles() -> None:
    from v5_memo.client import _parse_tier2_fact

    hit = _parse_tier2_fact({
        "id": "x",
        "paper_id": "10.x/html",
        "paper": {
            "title": "<p>The beneficial effects of metformin</p>",
            "doi": "10.x/html",
            "publication_year": 2019,
        },
        "canonical_phrase": "Metformin reduced cancer risk.",
    })

    assert hit is not None
    assert hit.title == "The beneficial effects of metformin"
