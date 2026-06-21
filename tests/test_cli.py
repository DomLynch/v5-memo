from __future__ import annotations

import sys
from collections.abc import Sequence

import pytest
from pytest import MonkeyPatch

from v5_memo import CorpusHit
from v5_memo.__main__ import main
from v5_memo.client import ResearkaSearchClient


class EmptyFullRaw:
    configured = False


class ResveratrolOpenAlex:
    def __init__(self, *, require_resveratrol_query: bool = False) -> None:
        self._require_resveratrol_query = require_resveratrol_query

    def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
        del limit
        if self._require_resveratrol_query and "resveratrol" not in query:
            return []
        return [
            CorpusHit(
                hit_id="promise",
                title="Resveratrol mimics exercise mitochondrial biology",
                abstract="Mechanism paper reported resveratrol improved mitochondrial function.",
                source="openalex",
                doi="10.promise",
            ),
            CorpusHit(
                hit_id="outcome",
                title="Resveratrol blunts exercise training adaptation",
                abstract="Human outcome trial observed resveratrol reduced exercise training benefits.",
                source="openalex",
                doi="10.outcome",
            ),
        ]


def _patch_smart_sources(monkeypatch: MonkeyPatch, openalex: ResveratrolOpenAlex) -> None:
    monkeypatch.setattr(
        "v5_memo.__main__.FullRawCorpusSearchClient.from_env",
        lambda strict=False: EmptyFullRaw(),
    )
    monkeypatch.setattr(
        "v5_memo.__main__.ResearkaSearchClient.from_env",
        lambda strict=False: ResearkaSearchClient(base_url="https://database.example", token="", strict=strict),
    )
    monkeypatch.setattr(
        "v5_memo.__main__.OpenAlexFullCorpusSearchClient.from_env",
        lambda strict=False: openalex,
    )


def test_demo_cli_renders_alpha_shape(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["v5_memo", "--demo"])

    main()

    captured = capsys.readouterr()
    assert "Alpha memo" in captured.out
    assert "point in different directions" in captured.out


def test_fullraw_searcher_fails_closed_without_endpoint(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v5_memo",
            "--searcher",
            "fullraw",
            "--topic",
            "longevity resilience",
            "--query",
            "NAD mitochondrial stress",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Full local raw 450M+ corpus search is not configured" in captured.err
    assert "V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL" in captured.err


def test_smart_cli_skips_unconfigured_researka_when_openalex_available(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_smart_sources(monkeypatch, ResveratrolOpenAlex())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v5_memo",
            "--searcher",
            "smart",
            "--planner",
            "seed",
            "--writer",
            "template",
            "--selector",
            "deterministic",
            "--topic",
            "resveratrol exercise adaptation",
            "--query",
            "resveratrol exercise adaptation",
        ],
    )

    main()

    captured = capsys.readouterr()
    assert "Alpha memo" in captured.out
    assert "Resveratrol" in captured.out


def test_smart_cli_planner_surfaces_elite_pair_from_broad_seed(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class EmptyFullRaw:
        configured = False

    class FakePlanner:
        def plan(
            self,
            *,
            topic: str,
            seed_queries: Sequence[str],
            limit: int = 8,
        ) -> list[str]:
            del topic, limit
            return ["resveratrol exercise training adaptation", *seed_queries]

    _patch_smart_sources(monkeypatch, ResveratrolOpenAlex(require_resveratrol_query=True))
    monkeypatch.setattr(
        "v5_memo.__main__.MiniMaxM3SearchPlanner.from_env",
        lambda: FakePlanner(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v5_memo",
            "--searcher",
            "smart",
            "--writer",
            "template",
            "--selector",
            "deterministic",
            "--topic",
            "longevity exercise adaptation",
            "--query",
            "longevity exercise adaptation pharmacology",
        ],
    )

    main()

    captured = capsys.readouterr()
    assert "Alpha memo" in captured.out
    assert "resveratrol" in captured.out.casefold()
    assert "different directions" in captured.out


def test_planned_cli_without_user_query_anchors_to_planned_queries(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen_seed_queries: list[Sequence[str]] = []

    class FakePlanner:
        def plan(
            self,
            *,
            topic: str,
            seed_queries: Sequence[str],
            limit: int = 8,
        ) -> list[str]:
            del topic, limit
            seen_seed_queries.append(seed_queries)
            return ["resveratrol exercise training adaptation"]

    monkeypatch.setattr("v5_memo.__main__._require_full_raw_or_exit", lambda: None)
    monkeypatch.setattr(
        "v5_memo.__main__.FullRawCorpusSearchClient.from_env",
        lambda strict=False: ResveratrolOpenAlex(require_resveratrol_query=True),
    )
    monkeypatch.setattr(
        "v5_memo.__main__.MiniMaxM3SearchPlanner.from_env",
        lambda: FakePlanner(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v5_memo",
            "--searcher",
            "fullraw",
            "--planner",
            "minimax",
            "--writer",
            "template",
            "--selector",
            "deterministic",
            "--topic",
            "longevity exercise adaptation",
        ],
    )

    main()

    captured = capsys.readouterr()
    assert "Alpha memo" in captured.out
    assert "resveratrol" in captured.out.casefold()
    assert seen_seed_queries == [["longevity exercise adaptation"]]
