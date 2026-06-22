from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest import MonkeyPatch

from v5_memo import CorpusHit
from v5_memo.__main__ import _alpha_shape_queries, _topic_anchored_queries, main
from v5_memo.client import ResearkaSearchClient
from v5_memo.schemas import MemoBuildError


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


def test_seed_planner_uses_topic_outside_demo(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, object] = {}

    def fake_build_alpha_memo(**kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace(markdown="# Alpha memo: ok\n")

    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v5_memo",
            "--planner",
            "seed",
            "--searcher",
            "openalex",
            "--topic",
            "resveratrol exercise training adaptation",
        ],
    )

    main()

    assert "Alpha memo" in capsys.readouterr().out
    assert seen["seed_queries"] == ["resveratrol exercise training adaptation"]


def test_topic_anchored_queries_reject_planner_drift_for_specific_topics() -> None:
    assert _topic_anchored_queries(
        [
            "metformin ampk activation skeletal muscle hypertrophy mechanism",
            "metformin resistance training older adults",
        ],
        "metformin resistance training adaptation",
    ) == ["metformin resistance training older adults"]


def test_alpha_shape_queries_add_universal_promise_and_outcome_probes() -> None:
    assert _alpha_shape_queries("metformin resistance training adaptation") == [
        "metformin expected augment resistance training protocol",
        "metformin blunted impaired attenuated resistance training outcome",
    ]


def test_cli_forwards_memo_coverage_thresholds_from_env(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, object] = {}

    def fake_build_alpha_memo(**kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace(markdown="# Alpha memo: ok\n")

    monkeypatch.setenv("V5_MEMO_MEMO_MIN_SHARDS_SEARCHED", "50")
    monkeypatch.setenv("V5_MEMO_MEMO_MIN_SOURCES_SEARCHED", "2")
    monkeypatch.setenv("V5_MEMO_MEMO_MIN_SEARCH_PASSES", "4")
    monkeypatch.setenv("V5_MEMO_MEMO_MIN_RESULT_CITATION_DIVERSITY", "3")
    monkeypatch.setenv("V5_MEMO_MEMO_MAX_RESULT_DUPLICATE_RATE", "0.2")
    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
    monkeypatch.setattr(sys, "argv", ["v5_memo", "--demo"])

    main()

    assert "Alpha memo" in capsys.readouterr().out
    assert seen["min_shards_searched"] == 50
    assert seen["min_sources_searched"] == 2
    assert seen["min_search_passes"] == 4
    assert seen["min_result_citation_diversity"] == 3
    assert seen["max_result_duplicate_rate"] == 0.2


def test_deploy_env_documents_fail_closed_memo_receipt_gates() -> None:
    env_text = Path("deploy/v5-memo-fullraw-shards.env.example").read_text()

    assert "V5_MEMO_MEMO_MIN_SEARCH_PASSES=6" in env_text
    assert "V5_MEMO_MEMO_MIN_RESULT_CITATION_DIVERSITY=2" in env_text
    assert "V5_MEMO_MEMO_MAX_RESULT_DUPLICATE_RATE=0.2" in env_text


def test_fullraw_cli_inherits_search_service_coverage_thresholds(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, object] = {}

    def fake_build_alpha_memo(**kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace(markdown="# Alpha memo: ok\n")

    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9902/search")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED", "50")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED", "2")
    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v5_memo",
            "--searcher",
            "fullraw",
            "--planner",
            "seed",
            "--writer",
            "template",
            "--topic",
            "management forecast disclosure",
        ],
    )

    main()

    assert "Alpha memo" in capsys.readouterr().out
    assert seen["min_shards_searched"] == 50
    assert seen["min_sources_searched"] == 2


def test_cli_explicit_zero_disables_inherited_coverage_threshold(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, object] = {}

    def fake_build_alpha_memo(**kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace(markdown="# Alpha memo: ok\n")

    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9902/search")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED", "50")
    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v5_memo",
            "--searcher",
            "fullraw",
            "--planner",
            "seed",
            "--writer",
            "template",
            "--min-shards-searched",
            "0",
        ],
    )

    main()

    assert "Alpha memo" in capsys.readouterr().out
    assert seen["min_shards_searched"] == 0


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


def test_planned_cli_drops_automatic_broad_topic_seed(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, list[str]] = {}

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

    class FakeFullRaw:
        configured = True

    def fake_build_alpha_memo(**kwargs: object) -> SimpleNamespace:
        seed_queries = kwargs["seed_queries"]
        anchor_queries = kwargs["anchor_queries"]
        assert isinstance(seed_queries, list)
        assert isinstance(anchor_queries, list)
        seen["seed_queries"] = seed_queries
        seen["anchor_queries"] = anchor_queries
        return SimpleNamespace(markdown="# Alpha memo: ok\n")

    monkeypatch.setattr("v5_memo.__main__._require_full_raw_or_exit", lambda: None)
    monkeypatch.setattr("v5_memo.__main__.FullRawCorpusSearchClient.from_env", lambda strict=False: FakeFullRaw())
    monkeypatch.setattr("v5_memo.__main__.MiniMaxM3SearchPlanner.from_env", lambda: FakePlanner())
    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
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
            "longevity intervention exercise adaptation reversal",
        ],
    )

    main()

    assert "Alpha memo" in capsys.readouterr().out
    assert seen == {
        "seed_queries": ["resveratrol exercise training adaptation"],
        "anchor_queries": ["resveratrol exercise training adaptation"],
    }


def test_planned_cli_drops_queries_that_lose_specific_topic_anchor(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, list[str]] = {}

    class FakePlanner:
        def plan(
            self,
            *,
            topic: str,
            seed_queries: Sequence[str],
            limit: int = 8,
        ) -> list[str]:
            del topic, seed_queries, limit
            return [
                "post activation potentiation cryotherapy attenuate 1rm strength",
                "cryotherapy resistance training hypertrophy blunted adaptation",
                "cold water immersion blunts hypertrophy resistance trained men",
            ]

    class FakeFullRaw:
        configured = True

    def fake_build_alpha_memo(**kwargs: object) -> SimpleNamespace:
        seed_queries = kwargs["seed_queries"]
        anchor_queries = kwargs["anchor_queries"]
        assert isinstance(seed_queries, list)
        assert isinstance(anchor_queries, list)
        seen["seed_queries"] = seed_queries
        seen["anchor_queries"] = anchor_queries
        return SimpleNamespace(markdown="# Alpha memo: ok\n")

    monkeypatch.setattr("v5_memo.__main__._require_full_raw_or_exit", lambda: None)
    monkeypatch.setattr("v5_memo.__main__.FullRawCorpusSearchClient.from_env", lambda strict=False: FakeFullRaw())
    monkeypatch.setattr("v5_memo.__main__.MiniMaxM3SearchPlanner.from_env", lambda: FakePlanner())
    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
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
            "cold water immersion resistance training adaptation",
        ],
    )

    main()

    assert "Alpha memo" in capsys.readouterr().out
    assert seen == {
        "seed_queries": [
            "cold water immersion resistance training adaptation",
            "cold water immersion blunts hypertrophy resistance trained men",
            "cold water immersion expected augment resistance protocol",
            "cold water immersion blunted impaired attenuated resistance outcome",
        ],
        "anchor_queries": ["cold water immersion resistance training adaptation"],
    }


def test_planned_cli_self_corrects_when_first_planned_anchor_drifts(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakePlanner:
        def plan(
            self,
            *,
            topic: str,
            seed_queries: Sequence[str],
            limit: int = 8,
        ) -> list[str]:
            del topic, seed_queries, limit
            return [
                "arabidopsis tor cotyledon greening",
                "resveratrol exercise training adaptation",
            ]

    class FakeFullRaw:
        configured = True

        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del limit
            if "arabidopsis" in query:
                return [
                    CorpusHit(
                        hit_id="plant-a",
                        title="Arabidopsis TOR controls cotyledon greening",
                        abstract="TOR promoted cotyledon greening in Arabidopsis.",
                        source="fullraw",
                        doi="10.plant/a",
                    )
                ]
            if "resveratrol" in query:
                return [
                    CorpusHit(
                        hit_id="promise",
                        title="Resveratrol mimics exercise mitochondrial biology",
                        abstract="Mechanism paper reported resveratrol improved mitochondrial function.",
                        source="fullraw",
                        doi="10.promise",
                    ),
                    CorpusHit(
                        hit_id="outcome",
                        title="Resveratrol blunts exercise training adaptation",
                        abstract="Human outcome trial observed resveratrol reduced exercise training benefits.",
                        source="fullraw",
                        doi="10.outcome",
                    ),
                ]
            return []

    monkeypatch.setattr("v5_memo.__main__._require_full_raw_or_exit", lambda: None)
    monkeypatch.setattr("v5_memo.__main__.FullRawCorpusSearchClient.from_env", lambda strict=False: FakeFullRaw())
    monkeypatch.setattr("v5_memo.__main__.MiniMaxM3SearchPlanner.from_env", lambda: FakePlanner())
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


def test_planned_cli_does_not_rerun_fullraw_after_no_alpha(
    monkeypatch: MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakePlanner:
        def plan(
            self,
            *,
            topic: str,
            seed_queries: Sequence[str],
            limit: int = 8,
        ) -> list[str]:
            del topic, seed_queries, limit
            return ["generic exercise power adaptation"]

    class FakeFullRaw:
        configured = True

        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del limit
            calls.append(query)
            return [
                CorpusHit(
                    hit_id="weak",
                    title="Exercise adaptation review",
                    abstract="Review summarized exercise adaptation literature.",
                    source="fullraw",
                    doi="10.weak/review",
                )
            ]

    monkeypatch.setattr("v5_memo.__main__._require_full_raw_or_exit", lambda: None)
    monkeypatch.setattr("v5_memo.__main__.FullRawCorpusSearchClient.from_env", lambda strict=False: FakeFullRaw())
    monkeypatch.setattr("v5_memo.__main__.MiniMaxM3SearchPlanner.from_env", lambda: FakePlanner())
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

    with pytest.raises(MemoBuildError):
        main()

    assert calls == ["generic exercise power adaptation"]


def test_planned_cli_keeps_full_query_recall_budget(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, int] = {}

    class FakePlanner:
        def plan(
            self,
            *,
            topic: str,
            seed_queries: Sequence[str],
            limit: int = 8,
        ) -> list[str]:
            del topic, seed_queries
            return [f"planned query {index}" for index in range(limit)]

    class FakeFullRaw:
        configured = True

    def fake_build_alpha_memo(**kwargs: object) -> SimpleNamespace:
        per_query_limit = kwargs["per_query_limit"]
        max_hits = kwargs["max_hits"]
        assert isinstance(per_query_limit, int)
        assert isinstance(max_hits, int)
        seen["per_query_limit"] = per_query_limit
        seen["max_hits"] = max_hits
        return SimpleNamespace(markdown="# Alpha memo: ok\n")

    monkeypatch.setattr("v5_memo.__main__._require_full_raw_or_exit", lambda: None)
    monkeypatch.setattr("v5_memo.__main__.FullRawCorpusSearchClient.from_env", lambda strict=False: FakeFullRaw())
    monkeypatch.setattr("v5_memo.__main__.MiniMaxM3SearchPlanner.from_env", lambda: FakePlanner())
    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v5_memo",
            "--searcher",
            "fullraw",
            "--planner",
            "minimax",
            "--planner-limit",
            "8",
            "--writer",
            "template",
            "--selector",
            "deterministic",
            "--topic",
            "longevity exercise adaptation",
        ],
    )

    main()

    assert "Alpha memo" in capsys.readouterr().out
    assert seen == {"per_query_limit": 50, "max_hits": 500}
