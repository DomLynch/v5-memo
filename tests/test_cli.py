from __future__ import annotations

import sys
from collections.abc import Sequence

import pytest
from pytest import MonkeyPatch

from v5_memo import CorpusHit
from v5_memo.__main__ import main
from v5_memo.client import ResearkaSearchClient


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
    class EmptyFullRaw:
        configured = False

    class StaticOpenAlex:
        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del query, limit
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
        lambda strict=False: StaticOpenAlex(),
    )
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

    class PlannedOpenAlex:
        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del limit
            if "resveratrol" not in query:
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
        lambda strict=False: PlannedOpenAlex(),
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
