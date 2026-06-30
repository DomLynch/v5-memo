import json
import sys
from collections.abc import Sequence
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest
from pytest import MonkeyPatch

from v5_memo.__main__ import (
    _alpha_shape_queries,
    _alpha_shaped_planned_queries,
    _dedupe_queries,
    _publish_blocker,
    _topic_anchored_queries,
    main,
)
from v5_memo.client import ResearkaSearchClient, SearchBackendError
from v5_memo.schemas import (
    ClaimCard,
    CorpusHit,
    InsightCandidate,
    MemoBuildError,
    MemoResult,
    SearchFailure,
)

_COVERAGE_THRESHOLD_ENV = (
    "V5_MEMO_MEMO_MIN_SHARDS_SEARCHED",
    "V5_MEMO_MEMO_MIN_SOURCES_SEARCHED",
    "V5_MEMO_MEMO_MIN_SEARCH_PASSES",
    "V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED",
    "V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED",
)


@pytest.fixture(autouse=True)
def _isolate_cli_coverage_threshold_env(monkeypatch: MonkeyPatch) -> None:
    for name in _COVERAGE_THRESHOLD_ENV:
        monkeypatch.delenv(name, raising=False)


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
        "metformin human trial resistance training",
        "metformin augment resistance training protocol",
        "metformin blunts resistance training",
    ]
    assert _alpha_shape_queries("resveratrol blunts exercise training") == [
        "resveratrol mimics exercise training",
        "resveratrol human trial exercise training",
        "resveratrol augment exercise training protocol",
        "resveratrol blunts exercise training",
    ]
    assert _alpha_shape_queries("protein timing distribution muscle synthesis")[0] == (
        "protein human trial timing distribution"
    )


def test_alpha_shaped_planner_queries_prefer_direct_evidence_language() -> None:
    assert _alpha_shaped_planned_queries([
        "metformin blunts survival benefit caloric restriction mice",
        "metformin attenuated healthspan extension germ-free mice",
        "metformin impairs exercise-induced mitochondrial biogenesis older adults",
    ])[0] == "metformin impairs exercise-induced mitochondrial biogenesis older adults"


def test_alpha_shaped_planner_queries_prefer_high_signal_failure_terms() -> None:
    assert _alpha_shaped_planned_queries([
        "urolithin mitochondrial aging",
        "urolithin primary endpoint failed subgroup",
    ])[0] == "urolithin primary endpoint failed subgroup"


def test_dedupe_queries_collapses_near_duplicate_fullraw_shapes() -> None:
    assert _dedupe_queries([
        "urolithin A mitochondrial aging",
        "urolithin mitochondrial aging",
        "urolithin human trial",
    ]) == [
        "urolithin A mitochondrial aging",
        "urolithin human trial",
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
    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
    monkeypatch.setattr(sys, "argv", ["v5_memo", "--demo"])

    main()

    assert "Alpha memo" in capsys.readouterr().out
    assert seen["min_shards_searched"] == 50
    assert seen["min_sources_searched"] == 2
    assert seen["min_search_passes"] == 4


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
    monkeypatch.setattr("v5_memo.__main__._require_full_raw_or_exit", lambda: None)
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
            "metformin resistance training adaptation",
        ],
    )

    main()

    assert "Alpha memo" in capsys.readouterr().out
    assert seen["seed_queries"] == [
        "metformin resistance training adaptation",
        "metformin human trial resistance training",
        "metformin augment resistance training protocol",
        "metformin blunts resistance training",
    ]
    assert (
        seen["per_query_limit"],
        seen["max_hits"],
        seen["min_shards_searched"],
        seen["min_sources_searched"],
    ) == (25, 75, 50, 2)


def test_fullraw_cli_allows_recall_depth_env_override(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, object] = {}

    def fake_build_alpha_memo(**kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace(markdown="# Alpha memo: ok\n")

    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9902/search")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_PER_QUERY_LIMIT", "40")
    monkeypatch.setenv("V5_MEMO_FULL_RAW_MAX_HITS", "80")
    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
    monkeypatch.setattr("v5_memo.__main__._require_full_raw_or_exit", lambda: None)
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
            "metformin resistance training adaptation",
        ],
    )

    main()

    assert "Alpha memo" in capsys.readouterr().out
    assert (seen["per_query_limit"], seen["max_hits"]) == (40, 80)


def test_fullraw_cli_allows_single_recall_limit_env_override(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, object] = {}

    def fake_build_alpha_memo(**kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace(markdown="# Alpha memo: ok\n")

    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9902/search")
    monkeypatch.delenv("V5_MEMO_FULL_RAW_PER_QUERY_LIMIT", raising=False)
    monkeypatch.delenv("V5_MEMO_FULL_RAW_MAX_HITS", raising=False)
    monkeypatch.setenv("V5_MEMO_FULL_RAW_RECALL_LIMIT", "30")
    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
    monkeypatch.setattr("v5_memo.__main__._require_full_raw_or_exit", lambda: None)
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
            "metformin resistance training adaptation",
        ],
    )

    main()

    assert "Alpha memo" in capsys.readouterr().out
    assert (seen["per_query_limit"], seen["max_hits"]) == (30, 90)


def test_fullraw_cli_uses_wider_recall_for_minimax_planner(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, object] = {}

    class FakeFullRaw:
        configured = True

    def fake_build_alpha_memo(**kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace(markdown="# Alpha memo: ok\n")

    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9915/search")
    monkeypatch.delenv("V5_MEMO_FULL_RAW_PER_QUERY_LIMIT", raising=False)
    monkeypatch.delenv("V5_MEMO_FULL_RAW_RECALL_LIMIT", raising=False)
    monkeypatch.delenv("V5_MEMO_FULL_RAW_MAX_HITS", raising=False)
    monkeypatch.setattr("v5_memo.__main__._require_full_raw_or_exit", lambda: None)
    monkeypatch.setattr("v5_memo.__main__.FullRawCorpusSearchClient.from_env", lambda strict=False: FakeFullRaw())
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
            "metformin resistance training adaptation",
        ],
    )

    main()

    assert "Alpha memo" in capsys.readouterr().out
    assert (seen["per_query_limit"], seen["max_hits"]) == (50, 200)


def test_cli_prints_search_backend_error_without_traceback(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_build_alpha_memo(**_kwargs: object) -> object:
        raise SearchBackendError("Full raw corpus search coverage too narrow: {'shards_searched': 32}")

    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fail_build_alpha_memo)
    monkeypatch.setattr("v5_memo.__main__._require_full_raw_or_exit", lambda: None)
    monkeypatch.setattr(sys, "argv", ["v5_memo", "--searcher", "fullraw", "--topic", "metformin"])

    with pytest.raises(SystemExit) as exc:
        main()

    captured = capsys.readouterr()
    assert exc.value.code == 1
    assert "coverage too narrow" in captured.err
    assert "Traceback" not in captured.err


def test_cli_writes_build_failure_receipt_without_traceback(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "build-error.json"

    def fail_build_alpha_memo(**_kwargs: object) -> object:
        raise MemoBuildError(
            SearchFailure(
                code="no_receipt_bound_alpha_candidate",
                message="no receipt-bound alpha memo candidate found",
                details={"hit_count": 14, "candidate_count": 0},
            )
        )

    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fail_build_alpha_memo)
    monkeypatch.setattr("v5_memo.__main__._require_full_raw_or_exit", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v5_memo",
            "--searcher",
            "fullraw",
            "--topic",
            "metformin",
            "--publish-receipt-path",
            str(receipt_path),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    captured = capsys.readouterr()
    receipt = json.loads(receipt_path.read_text())
    assert exc.value.code == 1
    assert receipt == {
        "details": {"candidate_count": 0, "hit_count": 14},
        "error": "no_receipt_bound_alpha_candidate",
        "message": "no receipt-bound alpha memo candidate found",
    }
    assert "no receipt-bound alpha memo candidate found" in captured.err
    assert "Traceback" not in captured.err


def test_cli_writes_search_backend_error_receipt(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "search-error.json"

    def fail_build_alpha_memo(**_kwargs: object) -> object:
        raise SearchBackendError("Full raw corpus search coverage too narrow: {'shards_searched': 32}")

    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fail_build_alpha_memo)
    monkeypatch.setattr("v5_memo.__main__._require_full_raw_or_exit", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v5_memo",
            "--searcher",
            "fullraw",
            "--topic",
            "metformin",
            "--publish-receipt-path",
            str(receipt_path),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    receipt = json.loads(receipt_path.read_text())
    assert exc.value.code == 1
    assert receipt["error"] == "search_backend_error"
    assert "coverage too narrow" in receipt["message"]


def test_cli_submit_researka_uses_generated_memo(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}
    receipt_path = tmp_path / "submit-receipt.json"

    def fake_build_alpha_memo(**kwargs: object) -> SimpleNamespace:
        seen["require_publish_quality"] = kwargs["require_publish_quality"]
        return SimpleNamespace(markdown="# Alpha memo: ok\n")

    def fake_build_payload(result: SimpleNamespace, *, author_agent_id: str, domain_slug: str) -> dict[str, object]:
        seen["markdown"] = result.markdown
        seen["author_agent_id"] = author_agent_id
        seen["domain_slug"] = domain_slug
        return {"title": "ok"}

    def fake_submit(
        payload: dict[str, object],
        *,
        agent_key: str,
        api_base: str,
        submit_url: str = "",
        timeout: float = 60.0,
    ) -> dict[str, object]:
        seen["payload"] = payload
        seen["agent_key"] = agent_key
        seen["api_base"] = api_base
        seen["submit_url"] = submit_url
        seen["timeout"] = timeout
        return {"submission_id": "sub-1"}

    monkeypatch.setenv("V5_MEMO_RESEARKA_AGENT_KEY", "submit-key")
    monkeypatch.setenv("V5_MEMO_RESEARKA_AGENT_ID", "v5-memo-agent")
    monkeypatch.setenv("V5_MEMO_RESEARKA_DOMAIN_SLUG", "longevity_research")
    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
    monkeypatch.setattr("v5_memo.__main__.build_researka_payload", fake_build_payload)
    monkeypatch.setattr("v5_memo.__main__.submit_researka", fake_submit)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v5_memo",
            "--demo",
            "--submit-researka",
            "--publish-receipt-path",
            str(receipt_path),
        ],
    )

    main()

    captured = capsys.readouterr()
    assert "# Alpha memo: ok" in captured.out
    assert '"submission_id": "sub-1"' in captured.err
    assert seen == {
        "markdown": "# Alpha memo: ok\n",
        "author_agent_id": "v5-memo-agent",
        "domain_slug": "longevity_research",
        "payload": {"title": "ok"},
        "agent_key": "submit-key",
        "api_base": "https://api.researka.org",
        "submit_url": "",
        "timeout": 60.0,
        "require_publish_quality": True,
    }
    assert json.loads(receipt_path.read_text()) == {"submission_id": "sub-1"}


def test_cli_submit_researka_fails_closed_without_agent_key(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "submit-error.json"

    def fake_build_alpha_memo(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(markdown="# Alpha memo: ok\n")

    for name in (
        "V5_MEMO_RESEARKA_AGENT_KEY",
        "V5_MEMO_RESEARKA_API_KEY",
        "RESEARKA_API_KEY_V5",
        "RESEARKA_AGENT_KEY",
        "RESEARKA_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("V5_MEMO_RESEARKA_AGENT_ID", "v5-memo-agent")
    monkeypatch.setenv("V5_MEMO_RESEARKA_DOMAIN_SLUG", "longevity_research")
    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v5_memo",
            "--demo",
            "--submit-researka",
            "--publish-receipt-path",
            str(receipt_path),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 3
    assert "Researka submit requires" in capsys.readouterr().err
    assert json.loads(receipt_path.read_text()) == {
        "error": "missing_researka_submit_config",
        "missing": ["V5_MEMO_RESEARKA_AGENT_KEY"],
    }


def test_cli_output_dir_writes_memo_and_prints_path(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    def fake_build_alpha_memo(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(markdown="# Alpha memo: stored\n\nBody.\n")

    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
    monkeypatch.setattr(sys, "argv", ["v5_memo", "--demo", "--output-dir", str(tmp_path)])

    main()

    written = list(tmp_path.glob("*.md"))
    assert len(written) == 1
    assert capsys.readouterr().out.strip() == str(written[0])
    assert written[0].read_text() == "# Alpha memo: stored\n\nBody.\n"


def test_cli_submit_accepts_v5_versioned_key_and_submit_url(
    monkeypatch: MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_build_alpha_memo(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(markdown="# Alpha memo: ok\n")

    def fake_build_payload(
        _result: SimpleNamespace,
        *,
        author_agent_id: str,
        domain_slug: str,
    ) -> dict[str, object]:
        return {"author_agent_id": author_agent_id, "domain_slug": domain_slug}

    def fake_submit(
        payload: dict[str, object],
        *,
        agent_key: str,
        api_base: str,
        submit_url: str = "",
        timeout: float = 60.0,
    ) -> dict[str, object]:
        del timeout
        seen.update({
            "payload": payload,
            "agent_key": agent_key,
            "api_base": api_base,
            "submit_url": submit_url,
        })
        return {"submission_id": "sub-1"}

    monkeypatch.setenv("RESEARKA_API_KEY_V5", "v5-key")
    monkeypatch.setenv("V5_MEMO_RESEARKA_AGENT_ID", "v5-memo-agent")
    monkeypatch.setenv("V5_MEMO_RESEARKA_DOMAIN_SLUG", "longevity_research")
    monkeypatch.setenv("RESEARKA_SUBMIT_URL", "https://api.researka.org/custom-submit")
    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
    monkeypatch.setattr("v5_memo.__main__.build_researka_payload", fake_build_payload)
    monkeypatch.setattr("v5_memo.__main__.submit_researka", fake_submit)
    monkeypatch.setattr(sys, "argv", ["v5_memo", "--demo", "--publish"])

    main()

    assert seen == {
        "payload": {
            "author_agent_id": "v5-memo-agent",
            "domain_slug": "longevity_research",
        },
        "agent_key": "v5-key",
        "api_base": "https://api.researka.org",
        "submit_url": "https://api.researka.org/custom-submit",
    }


def test_cli_publish_waits_for_accept_and_lists_publication(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}
    receipt_path = tmp_path / "publish-receipt.json"

    def fake_build_alpha_memo(**kwargs: object) -> SimpleNamespace:
        seen["require_publish_quality"] = kwargs["require_publish_quality"]
        return SimpleNamespace(markdown="# Alpha memo: ok\n")

    def fake_build_payload(
        _result: SimpleNamespace,
        *,
        author_agent_id: str,
        domain_slug: str,
    ) -> dict[str, object]:
        return {"author_agent_id": author_agent_id, "domain_slug": domain_slug}

    def fake_submit(
        payload: dict[str, object],
        *,
        agent_key: str,
        api_base: str,
        submit_url: str = "",
        timeout: float = 60.0,
    ) -> dict[str, object]:
        del submit_url, timeout
        seen["submit"] = (payload, agent_key, api_base)
        return {"submission": {"id": "sub-1"}}

    def fake_wait(
        submission_id: str,
        *,
        api_base: str,
        timeout_seconds: float,
        poll_seconds: float,
    ) -> dict[str, object]:
        seen["wait"] = (submission_id, api_base, timeout_seconds, poll_seconds)
        return {
            "status": "complete",
            "decision": "accept",
            "publication": {"publication_id": "pub-1"},
        }

    def fake_list(
        publication_id: str,
        *,
        agent_key: str,
        api_base: str,
        visibility: str = "listed",
    ) -> dict[str, object]:
        seen["list"] = (publication_id, agent_key, api_base, visibility)
        return {"id": publication_id, "public_visibility": visibility, "updated": True}

    monkeypatch.setenv("V5_MEMO_RESEARKA_AGENT_KEY", "submit-key")
    monkeypatch.setenv("V5_MEMO_RESEARKA_AGENT_ID", "v5-memo-agent")
    monkeypatch.setenv("V5_MEMO_RESEARKA_DOMAIN_SLUG", "longevity_research")
    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
    monkeypatch.setattr("v5_memo.__main__.build_researka_payload", fake_build_payload)
    monkeypatch.setattr("v5_memo.__main__.submit_researka", fake_submit)
    monkeypatch.setattr("v5_memo.__main__.wait_researka_decision", fake_wait)
    monkeypatch.setattr("v5_memo.__main__.set_researka_public_visibility", fake_list)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v5_memo",
            "--demo",
            "--publish",
            "--researka-decision-wait-seconds",
            "20",
            "--researka-decision-poll-seconds",
            "2",
            "--researka-list-if-accepted",
            "--publish-receipt-path",
            str(receipt_path),
        ],
    )

    main()

    receipt = json.loads(receipt_path.read_text())
    assert receipt["decision"]["decision"] == "accept"
    assert receipt["visibility"] == {
        "id": "pub-1",
        "public_visibility": "listed",
        "updated": True,
    }
    assert seen == {
        "submit": (
            {"author_agent_id": "v5-memo-agent", "domain_slug": "longevity_research"},
            "submit-key",
            "https://api.researka.org",
        ),
        "wait": ("sub-1", "https://api.researka.org", 20.0, 2.0),
        "list": ("pub-1", "submit-key", "https://api.researka.org", "listed"),
        "require_publish_quality": True,
    }


def test_cli_publish_records_accept_when_visibility_listing_is_forbidden(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}
    receipt_path = tmp_path / "publish-receipt.json"

    def fake_build_alpha_memo(**kwargs: object) -> SimpleNamespace:
        seen["require_publish_quality"] = kwargs["require_publish_quality"]
        return SimpleNamespace(markdown="# Alpha memo: ok\n")

    def fake_build_payload(
        _result: SimpleNamespace,
        *,
        author_agent_id: str,
        domain_slug: str,
    ) -> dict[str, object]:
        return {"author_agent_id": author_agent_id, "domain_slug": domain_slug}

    def fake_submit(
        payload: dict[str, object],
        *,
        agent_key: str,
        api_base: str,
        submit_url: str = "",
        timeout: float = 60.0,
    ) -> dict[str, object]:
        del payload, agent_key, api_base, submit_url, timeout
        return {"submission": {"id": "sub-2"}}

    def fake_wait(
        submission_id: str,
        *,
        api_base: str,
        timeout_seconds: float,
        poll_seconds: float,
    ) -> dict[str, object]:
        del submission_id, api_base, timeout_seconds, poll_seconds
        return {
            "status": "complete",
            "decision": "accept",
            "publication": {"publication_id": "pub-2"},
        }

    def fake_list(
        publication_id: str,
        *,
        agent_key: str,
        api_base: str,
        visibility: str = "listed",
    ) -> dict[str, object]:
        del publication_id, agent_key, api_base, visibility
        raise HTTPError("https://api.researka.org/ops/publications/pub-2/visibility", 403, "Forbidden", Message(), None)

    monkeypatch.setenv("V5_MEMO_RESEARKA_AGENT_KEY", "submit-key")
    monkeypatch.setenv("V5_MEMO_RESEARKA_AGENT_ID", "v5-memo-agent")
    monkeypatch.setenv("V5_MEMO_RESEARKA_DOMAIN_SLUG", "longevity_research")
    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
    monkeypatch.setattr("v5_memo.__main__.build_researka_payload", fake_build_payload)
    monkeypatch.setattr("v5_memo.__main__.submit_researka", fake_submit)
    monkeypatch.setattr("v5_memo.__main__.wait_researka_decision", fake_wait)
    monkeypatch.setattr("v5_memo.__main__.set_researka_public_visibility", fake_list)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v5_memo",
            "--demo",
            "--publish",
            "--researka-decision-wait-seconds",
            "20",
            "--researka-list-if-accepted",
            "--publish-receipt-path",
            str(receipt_path),
        ],
    )

    main()

    receipt = json.loads(receipt_path.read_text())
    assert receipt["decision"]["decision"] == "accept"
    assert receipt["visibility_error"] == {
        "error": "researka_visibility_update_failed",
        "status": 403,
        "reason": "Forbidden",
        "publication_id": "pub-2",
    }
    assert seen == {"require_publish_quality": True}


def test_cli_publish_defaults_to_deterministic_selector_with_minimax_writer(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    def fake_build_alpha_memo(**kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace(
            markdown="# Alpha memo: ok\n",
            candidate=InsightCandidate(
                topic="longevity resilience",
                thesis="ok",
                bridge_terms=("nad",),
                tension_terms=("negative", "null"),
                receipt_ids=("a", "b"),
                score=90,
                novelty_score=60,
                evidence_score=90,
                reasons=("shape:directional_reversal", "tier:publishable_alpha"),
            ),
            receipts=[],
        )

    def fail_selector_from_env(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("publish default should not instantiate MiniMax selector")

    monkeypatch.setenv("V5_MEMO_RESEARKA_AGENT_KEY", "submit-key")
    monkeypatch.setenv("V5_MEMO_RESEARKA_AGENT_ID", "v5-memo-agent")
    monkeypatch.setenv("V5_MEMO_RESEARKA_DOMAIN_SLUG", "longevity_research")
    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
    monkeypatch.setattr(
        "v5_memo.__main__.MiniMaxM3MemoWriter.from_env",
        lambda: SimpleNamespace(render=lambda *_args: "# Alpha memo: ok\n"),
    )
    monkeypatch.setattr(
        "v5_memo.__main__.MiniMaxM3CandidateSelector.from_env",
        fail_selector_from_env,
    )
    monkeypatch.setattr(
        "v5_memo.__main__.build_researka_payload",
        lambda *_args, **_kwargs: {"author_agent_id": "v5-memo-agent"},
    )
    monkeypatch.setattr(
        "v5_memo.__main__.submit_researka",
        lambda *_args, **_kwargs: {"submission": {"id": "sub-1"}},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v5_memo",
            "--demo",
            "--writer",
            "minimax",
            "--publish",
            "--publish-receipt-path",
            str(tmp_path / "receipt.json"),
        ],
    )

    main()

    assert seen["memo_selector"] is None
    assert seen["require_publish_quality"] is True


def test_cli_smart_defaults_to_publishable_tier(monkeypatch: MonkeyPatch) -> None:
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
            "--demo",
            "--searcher",
            "smart",
            "--planner",
            "seed",
            "--writer",
            "template",
            "--selector",
            "deterministic",
        ],
    )

    main()

    assert seen["min_alpha_tier"] == "publishable_alpha"


def test_cli_emit_discovery_on_fail_reruns_as_review_seed(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    discovery = InsightCandidate(
        topic="longevity resilience",
        thesis="Discovery fallback.",
        bridge_terms=("nad",),
        tension_terms=(),
        receipt_ids=("a", "b"),
        score=10,
        novelty_score=10,
        evidence_score=10,
        reasons=("tier:discovery_seed",),
    )

    def fake_build_alpha_memo(**kwargs: object) -> SimpleNamespace:
        calls.append(str(kwargs["min_alpha_tier"]))
        if len(calls) == 1:
            raise MemoBuildError(SearchFailure("no_alpha", "no alpha"))
        return SimpleNamespace(markdown="# Discovery seed: fallback\n", candidate=discovery)

    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v5_memo",
            "--demo",
            "--emit-discovery-on-fail",
            "--output-dir",
            str(tmp_path),
        ],
    )

    main()

    written = list(tmp_path.glob("*.md"))
    assert calls == ["publishable_alpha", "discovery_seed"]
    assert len(written) == 1
    assert capsys.readouterr().out.strip() == str(written[0])
    assert written[0].read_text() == "# Discovery seed: fallback\n"


def test_cli_publish_does_not_submit_discovery_seed(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "submit-error.json"
    discovery = InsightCandidate(
        topic="longevity resilience",
        thesis="Discovery only.",
        bridge_terms=("nad",),
        tension_terms=(),
        receipt_ids=("a", "b"),
        score=10,
        novelty_score=10,
        evidence_score=10,
        reasons=("tier:discovery_seed",),
    )

    def fake_build_alpha_memo(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(markdown="# Discovery seed: ok\n", candidate=discovery)

    def fake_submit(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("discovery seeds must not submit")

    monkeypatch.setenv("V5_MEMO_RESEARKA_AGENT_KEY", "submit-key")
    monkeypatch.setenv("V5_MEMO_RESEARKA_AGENT_ID", "v5-memo-agent")
    monkeypatch.setenv("V5_MEMO_RESEARKA_DOMAIN_SLUG", "longevity_research")
    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
    monkeypatch.setattr("v5_memo.__main__.submit_researka", fake_submit)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v5_memo",
            "--demo",
            "--publish",
            "--publish-receipt-path",
            str(receipt_path),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 4
    assert "Discovery seed output was not submitted" in capsys.readouterr().err
    assert json.loads(receipt_path.read_text()) == {
        "error": "discovery_seed_not_submitted",
        "tier": "discovery_seed",
    }


def test_cli_publish_blocks_zero_human_translational_memo(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "submit-error.json"
    candidate = InsightCandidate(
        topic="resveratrol exercise training",
        thesis="Animal model signal is not enough for a public human alpha claim.",
        bridge_terms=("resveratrol", "training"),
        tension_terms=("null", "positive"),
        receipt_ids=("human", "rat"),
        score=90,
        novelty_score=50,
        evidence_score=80,
        reasons=("tier:publishable_alpha",),
        claim_cards=(
            ClaimCard("human", "promise", "mechanistic_model", "animal", "stress", "null", "indirect", "medium", "mouse model"),
            ClaimCard("rat", "outcome", "mechanistic_model", "animal", "strength", "positive", "indirect", "medium", "rat model"),
        ),
    )

    def fake_build_alpha_memo(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(markdown="# Alpha memo: animal-heavy\n", candidate=candidate)

    def fake_submit(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("animal-heavy translational memo should not submit")

    monkeypatch.setenv("V5_MEMO_RESEARKA_AGENT_KEY", "submit-key")
    monkeypatch.setenv("V5_MEMO_RESEARKA_AGENT_ID", "v5-memo-agent")
    monkeypatch.setenv("V5_MEMO_RESEARKA_DOMAIN_SLUG", "longevity_research")
    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
    monkeypatch.setattr("v5_memo.__main__.submit_researka", fake_submit)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v5_memo",
            "--demo",
            "--publish",
            "--publish-receipt-path",
            str(receipt_path),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 5
    assert "Publish blocked: translational_evidence_too_indirect" in capsys.readouterr().err
    assert json.loads(receipt_path.read_text()) == {
        "direct_human_receipts": 0,
        "error": "translational_evidence_too_indirect",
        "indirect_model_receipts": 2,
    }


def test_publish_blocker_allows_direct_human_plus_context() -> None:
    candidate = InsightCandidate(
        topic="nicotinamide riboside exercise performance",
        thesis="Human signal with mechanistic context should go to Researka review.",
        bridge_terms=("nicotinamide", "exercise"),
        tension_terms=("negative", "positive"),
        receipt_ids=("human-a", "human-b", "rat"),
        score=90,
        novelty_score=50,
        evidence_score=80,
        reasons=("tier:publishable_alpha",),
        claim_cards=(
            ClaimCard("human-a", "positive_signal", "randomized_trial", "human", "performance", "positive", "direct", "high", "human trial"),
            ClaimCard("human-b", "negative_signal", "intervention_study", "human", "performance", "negative", "direct", "high", "human trial"),
            ClaimCard("rat", "boundary", "mechanistic_model", "animal", "performance", "negative", "indirect", "medium", "rat model"),
        ),
    )

    assert _publish_blocker(SimpleNamespace(markdown="# Alpha memo: ok", candidate=candidate, receipts=())) is None


def test_publish_blocker_requires_two_strong_direct_human_receipts() -> None:
    candidate = InsightCandidate(
        topic="nicotinamide riboside exercise performance",
        thesis="One human receipt plus animal context is not enough for publish.",
        bridge_terms=("nicotinamide", "exercise"),
        tension_terms=("negative", "positive"),
        receipt_ids=("human", "rat"),
        score=90,
        novelty_score=50,
        evidence_score=80,
        reasons=("tier:publishable_alpha",),
        claim_cards=(
            ClaimCard("human", "positive_signal", "randomized_trial", "human", "performance", "positive", "direct", "high", "human trial"),
            ClaimCard("rat", "boundary", "mechanistic_model", "animal", "performance", "negative", "indirect", "medium", "rat model"),
        ),
    )

    assert _publish_blocker(SimpleNamespace(markdown="# Alpha memo: ok", candidate=candidate, receipts=())) == {
        "direct_human_receipts": 1,
        "error": "insufficient_direct_human_receipts",
        "strong_direct_human_receipts": 1,
    }


def test_cli_publish_blocks_unbundled_invalid_doi_citation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "submit-error.json"
    bad_doi = "10.31435/ijitss.1(49).2026.4693"
    candidate = InsightCandidate(
        topic="cold water immersion",
        thesis="Comparator evidence must keep source IDs bundle-safe.",
        bridge_terms=("cold", "immersion"),
        tension_terms=("negative", "positive"),
        receipt_ids=(bad_doi, "10.1136/bjsports-2013-092433"),
        score=90,
        novelty_score=50,
        evidence_score=80,
        reasons=("tier:publishable_alpha",),
        claim_cards=(
            ClaimCard(bad_doi, "promise", "cohort", "human", "recovery", "positive", "direct", "high", "human cohort"),
            ClaimCard(
                "10.1136/bjsports-2013-092433",
                "outcome",
                "randomized_trial",
                "human",
                "recovery",
                "negative",
                "direct",
                "high",
                "human trial",
            ),
        ),
    )
    receipts = [
        CorpusHit(
            hit_id=bad_doi,
            title="Cold water immersion teaching and performance",
            abstract="Human cohort context.",
            source="fullraw:openalex",
            doi=bad_doi,
        ),
        CorpusHit(
            hit_id="10.1136/bjsports-2013-092433",
            title="Cold water immersion recovery review",
            abstract="Human trial context.",
            source="fullraw:openalex",
            doi="10.1136/bjsports-2013-092433",
        ),
    ]

    def fake_build_alpha_memo(**_kwargs: object) -> MemoResult:
        return MemoResult(
            candidate=candidate,
            receipts=receipts,
            markdown=f"# Alpha memo: cold water\n\nReceipts: {bad_doi} and 10.1136/bjsports-2013-092433.",
        )

    def fake_submit(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("invalid DOI citations must not submit")

    monkeypatch.setenv("V5_MEMO_RESEARKA_AGENT_KEY", "submit-key")
    monkeypatch.setenv("V5_MEMO_RESEARKA_AGENT_ID", "v5-memo-agent")
    monkeypatch.setenv("V5_MEMO_RESEARKA_DOMAIN_SLUG", "longevity_research")
    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
    monkeypatch.setattr("v5_memo.__main__.submit_researka", fake_submit)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v5_memo",
            "--demo",
            "--publish",
            "--publish-receipt-path",
            str(receipt_path),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 5
    assert "Publish blocked: unbundled_doi_citation" in capsys.readouterr().err
    assert json.loads(receipt_path.read_text()) == {
        "dois": [bad_doi],
        "error": "unbundled_doi_citation",
    }


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
    monkeypatch.setattr("v5_memo.__main__._require_full_raw_or_exit", lambda: None)
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
    assert "Full local raw 450M+ corpus search is not healthy" in captured.err
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


def test_smart_cli_uses_lenient_optional_backends(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[tuple[str, bool]] = []

    def fake_build_alpha_memo(**kwargs: object) -> SimpleNamespace:
        del kwargs
        return SimpleNamespace(markdown="# Alpha memo: ok\n")

    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
    def fullraw_from_env(*, strict: bool = False) -> EmptyFullRaw:
        seen.append(("fullraw", strict))
        return EmptyFullRaw()

    def researka_from_env(*, strict: bool = False) -> ResearkaSearchClient:
        seen.append(("researka", strict))
        return ResearkaSearchClient(base_url="https://database.example", token="", strict=strict)

    def openalex_from_env(*, strict: bool = False) -> ResveratrolOpenAlex:
        seen.append(("openalex", strict))
        return ResveratrolOpenAlex()

    monkeypatch.setattr("v5_memo.__main__.FullRawCorpusSearchClient.from_env", fullraw_from_env)
    monkeypatch.setattr("v5_memo.__main__.ResearkaSearchClient.from_env", researka_from_env)
    monkeypatch.setattr("v5_memo.__main__.OpenAlexFullCorpusSearchClient.from_env", openalex_from_env)
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
        ],
    )

    main()

    assert "Alpha memo" in capsys.readouterr().out
    assert seen == [("fullraw", False), ("researka", False), ("openalex", False)]


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
        ],
    )

    main()

    captured = capsys.readouterr()
    assert "Alpha memo" in captured.out
    assert "resveratrol" in captured.out.casefold()
    assert "different directions" in captured.out


def test_explicit_query_skips_minimax_planner(monkeypatch: MonkeyPatch) -> None:
    seen: dict[str, list[str]] = {}

    class FakePlanner:
        def plan(self, **_kwargs: object) -> list[str]:
            raise AssertionError("explicit --query must not be replaced by planner output")

    def fake_build_alpha_memo(**kwargs: object) -> SimpleNamespace:
        seed_queries = kwargs["seed_queries"]
        assert isinstance(seed_queries, list)
        seen["seed_queries"] = seed_queries
        return SimpleNamespace(markdown="# Alpha memo: ok\n")

    monkeypatch.setattr("v5_memo.__main__.MiniMaxM3SearchPlanner.from_env", lambda: FakePlanner())
    monkeypatch.setattr("v5_memo.__main__.build_alpha_memo", fake_build_alpha_memo)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v5_memo",
            "--searcher",
            "openalex",
            "--planner",
            "minimax",
            "--writer",
            "template",
            "--selector",
            "deterministic",
            "--topic",
            "metformin resistance training adaptation",
            "--query",
            "metformin resistance training adaptation",
        ],
    )

    main()

    assert seen == {"seed_queries": ["metformin resistance training adaptation"]}


def test_explicit_fullraw_publish_preserves_query_slate(
    monkeypatch: MonkeyPatch,
) -> None:
    seen: dict[str, list[str]] = {}

    def fake_build_alpha_memo(**kwargs: object) -> SimpleNamespace:
        seed_queries = kwargs["seed_queries"]
        assert isinstance(seed_queries, list)
        seen["seed_queries"] = seed_queries
        return SimpleNamespace(markdown="# Alpha memo: ok\n")

    monkeypatch.setenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", "http://127.0.0.1:9915/search")
    for name in (
        "V5_MEMO_RESEARKA_AGENT_KEY",
        "V5_MEMO_RESEARKA_AGENT_ID",
        "V5_MEMO_RESEARKA_DOMAIN_SLUG",
        "RESEARKA_AGENT_KEY",
        "RESEARKA_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("v5_memo.__main__._require_full_raw_or_exit", lambda: None)
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
            "--selector",
            "deterministic",
            "--topic",
            "resveratrol exercise training adaptation",
            "--query",
            "resveratrol exercise training adaptation",
            "--publish",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 3
    assert seen == {"seed_queries": ["resveratrol exercise training adaptation"]}


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
            "cold water immersion human trial resistance training",
            "cold water immersion augment resistance training protocol",
            "cold water immersion blunts resistance training",
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


def test_strict_fullraw_drops_unshaped_planner_queries(
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
                "training deconditioning reversal mitochondrial biogenesis",
                "resveratrol blunts exercise training adaptation",
            ]

    class FakeFullRaw:
        configured = True

    def fake_build_alpha_memo(**kwargs: object) -> SimpleNamespace:
        seed_queries = kwargs["seed_queries"]
        assert isinstance(seed_queries, list)
        seen["seed_queries"] = seed_queries
        return SimpleNamespace(markdown="# Alpha memo: ok\n")

    monkeypatch.setenv("V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED", "1525")
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
            "longevity exercise adaptation reversal",
        ],
    )

    main()

    assert "Alpha memo" in capsys.readouterr().out
    assert seen == {"seed_queries": ["resveratrol blunts exercise training adaptation"]}


def test_strict_fullraw_uses_specific_seed_before_planner_sweeps(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, list[str]] = {}

    class FakePlanner:
        def plan(self, **_kwargs: object) -> list[str]:
            raise AssertionError("specific strict-fullraw topics should not planner-fanout first")

    class FakeFullRaw:
        configured = True

    def fake_build_alpha_memo(**kwargs: object) -> SimpleNamespace:
        seed_queries = kwargs["seed_queries"]
        assert isinstance(seed_queries, list)
        seen["seed_queries"] = seed_queries
        return SimpleNamespace(markdown="# Alpha memo: ok\n")

    monkeypatch.setenv("V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED", "1525")
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
            "metformin resistance training adaptation",
        ],
    )

    main()

    assert "Alpha memo" in capsys.readouterr().out
    assert seen == {"seed_queries": [
        "metformin resistance training adaptation",
        "metformin human trial resistance training",
        "metformin augment resistance training protocol",
        "metformin blunts resistance training",
    ]}


def test_strict_fullraw_does_not_planner_fanout_for_broad_one_anchor_seed(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, list[str]] = {}

    class FakePlanner:
        def plan(self, **_kwargs: object) -> list[str]:
            raise AssertionError("broad strict-fullraw seed should fail fast, not planner-fanout")

    class FakeFullRaw:
        configured = True

    def fake_build_alpha_memo(**kwargs: object) -> SimpleNamespace:
        seed_queries = kwargs["seed_queries"]
        assert isinstance(seed_queries, list)
        seen["seed_queries"] = seed_queries
        return SimpleNamespace(markdown="# Alpha memo: ok\n")

    monkeypatch.setenv("V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED", "1525")
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
            "metformin longevity",
        ],
    )

    main()

    assert "Alpha memo" in capsys.readouterr().out
    assert seen == {"seed_queries": ["metformin longevity"]}


def test_strict_fullraw_fails_fast_when_planner_has_no_alpha_shape(
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
            return ["training deconditioning reversal mitochondrial biogenesis"]

    class FakeFullRaw:
        configured = True

        def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
            del limit
            calls.append(query)
            return []

    monkeypatch.setenv("V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED", "1525")
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
            "longevity exercise adaptation reversal",
        ],
    )

    with pytest.raises(MemoBuildError):
        main()

    assert calls == []
