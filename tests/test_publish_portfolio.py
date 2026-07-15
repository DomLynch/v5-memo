from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from urllib.request import Request

from pytest import MonkeyPatch


def _load_portfolio() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "v5_publish_portfolio.py"
    spec = importlib.util.spec_from_file_location("v5_publish_portfolio", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ready_receipt() -> dict[str, object]:
    return {"ready": True, "validation": "publish_quality"}


def _accepted_receipt() -> dict[str, object]:
    return {
        "decision": {
            "decision": "accept",
            "publication": {
                "publication_id": "pub-1",
                "doi": "10.17605/OSF.IO/TEST1",
                "doi_status": "minted",
            },
        },
        "visibility": {"id": "pub-1", "public_visibility": "listed"},
    }


def _cache_env(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    cache_dir = tmp_path / "active-cache"
    shard_dir = tmp_path / "active-shards"
    cache_dir.mkdir()
    shard_dir.mkdir()
    return (
        {
            "RESEARKA_FULLRAW_SWEEP_CACHE_DIR": str(cache_dir),
            "RESEARKA_FULLRAW_SHARD_DIR": str(shard_dir),
            "RESEARKA_FULLRAW_SWEEP_SHARD_LIMIT": "1525",
            "RESEARKA_FULLRAW_SWEEP_PASS_SHARD_LIMIT": "12",
            "RESEARKA_FULLRAW_SWEEP_TTL_SECONDS": "604800",
            "V5_MEMO_FULL_RAW_PER_QUERY_LIMIT": "25",
            "V5_MEMO_FULL_RAW_MAX_HITS": "100",
        },
        cache_dir,
        shard_dir,
    )


def _write_sweep_cache(
    portfolio: ModuleType,
    path: Path,
    *,
    query: str,
    scope: str,
    created_at: float | None = None,
    shards_searched: int = 1525,
    remaining_shards: int = 0,
    failed_shards: int = 0,
    pass_shard_limit: int | None = 12,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "partial_shard_search": shards_searched != 1525,
        "result_count_raw": 25,
        "shards_searched": shards_searched,
        "shards_total": 1525,
        "sweep_catalog_scope": scope,
        "sweep_failed_shards": failed_shards,
        "sweep_original_query": query,
        "sweep_query": query,
        "sweep_remaining_shards": remaining_shards,
        "sweep_result_limit": 25,
        "sweep_shard_limit": 1525,
        "sweep_strategy": portfolio._SWEEP_STRATEGY,
    }
    if pass_shard_limit is not None:
        receipt["sweep_pass_shard_limit"] = pass_shard_limit
    path.write_text(json.dumps({
        "created_at": time.time() if created_at is None else created_at,
        "hits": [{"id": index} for index in range(25)],
        "receipt": receipt,
    }))


def test_build_command_preserves_strict_submit_gate(tmp_path: Path) -> None:
    portfolio = _load_portfolio()
    config = portfolio.RunConfig(
        output_dir=tmp_path,
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner="seed",
        writer="template",
        selector="deterministic",
        min_alpha_tier="publishable",
        submit=True,
        decision_wait_seconds=60,
        decision_poll_seconds=2,
        submit_wait_seconds=10,
        max_leads=1,
        state_path=None,
        lead_file=None,
        auto_discover_leads=False,
        min_open_leads=0,
        discover_count=0,
        blocked_retry_hours=0,
    )

    command = portfolio.build_command(
        "metformin exercise adaptation",
        tmp_path,
        tmp_path / "r.json",
        config,
    )

    assert command[:3] == ["python3", "-m", "v5_memo"]
    assert "--query" not in command
    assert "--require-full-raw-corpus" in command
    assert command[command.index("--searcher") + 1] == "fullraw"
    assert command[command.index("--min-alpha-tier") + 1] == "publishable"
    assert "--submit-researka" in command
    assert "--researka-list-if-accepted" in command
    assert command[command.index("--researka-decision-wait-seconds") + 1] == "60"
    assert command[command.index("--researka-submit-wait-seconds") + 1] == "10"


def test_build_command_preparation_uses_submit_equivalent_quality_gate(tmp_path: Path) -> None:
    portfolio = _load_portfolio()
    config = portfolio.RunConfig(
        output_dir=tmp_path,
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner=None,
        writer=None,
        selector=None,
        min_alpha_tier="publishable",
        submit=False,
        decision_wait_seconds=0,
        decision_poll_seconds=1,
        submit_wait_seconds=0,
        max_leads=1,
        state_path=None,
        lead_file=None,
        auto_discover_leads=False,
        min_open_leads=0,
        discover_count=0,
        blocked_retry_hours=0,
        ready_buffer_size=3,
    )

    command = portfolio.build_command(
        "generic lead",
        tmp_path,
        tmp_path / "r.json",
        config,
    )

    assert "--validate-publish-quality" in command
    assert "--submit-researka" not in command


def test_accept_decision_is_pending_when_visibility_update_warns() -> None:
    portfolio = _load_portfolio()

    status = portfolio.classify_run(
        0,
        {
            "decision": {"decision": "accept"},
            "visibility_error": {"error": "researka_visibility_update_failed"},
        },
        submit=True,
    )

    assert status == "accepted_pending_publication"


def test_accept_decision_requires_minted_listed_publication() -> None:
    portfolio = _load_portfolio()

    status = portfolio.classify_run(0, _accepted_receipt(), submit=True)

    assert status == "accepted"


def test_accept_decision_supports_camel_case_public_visibility() -> None:
    portfolio = _load_portfolio()
    receipt = _accepted_receipt()
    receipt["visibility"] = {"id": "pub-1", "publicVisible": True}

    status = portfolio.classify_run(0, receipt, submit=True)

    assert status == "accepted"


def test_empty_prepare_receipt_does_not_create_false_ready_supply() -> None:
    portfolio = _load_portfolio()

    status = portfolio.classify_run(0, {}, submit=False)

    assert status == "blocked:invalid_publish_quality_receipt"


def test_durable_submission_receipt_wins_even_if_outer_process_times_out() -> None:
    portfolio = _load_portfolio()

    status = portfolio.classify_run(
        124,
        {"submission": {"id": "sub-durable"}},
        submit=True,
    )

    assert status == "submitted"


def test_empty_submission_shape_is_not_misclassified_as_submitted() -> None:
    portfolio = _load_portfolio()

    status = portfolio.classify_run(124, {"submission": None}, submit=True)

    assert status == "failed_no_receipt"


def test_retryable_submit_failure_preserves_prequalified_ready_lead(
    tmp_path: Path,
) -> None:
    portfolio = _load_portfolio()
    state_path = tmp_path / "state.json"
    ready_receipt = tmp_path / "ready.json"
    state: dict[str, object] = {
        "attempted_leads": {
            "ready lead": {
                "receipt_path": str(ready_receipt),
                "status": "ready",
                "updated_at": portfolio._timestamp(),
            }
        }
    }

    portfolio._save_attempted_lead(
        state_path,
        state,
        {
            "lead": "ready lead",
            "receipt_path": str(tmp_path / "deferred.json"),
            "status": "deferred",
        },
        preserve_ready=True,
    )

    saved = json.loads(state_path.read_text())["attempted_leads"]["ready lead"]
    assert saved["status"] == "ready"
    assert saved["last_attempt_status"] == "deferred"
    assert saved["ready_receipt_path"] == str(ready_receipt)

    portfolio._save_attempted_lead(
        state_path,
        state,
        {
            "lead": "ready lead",
            "receipt_path": str(tmp_path / "pending.json"),
            "status": "accepted_pending_publication",
        },
        preserve_ready=True,
    )

    saved = json.loads(state_path.read_text())["attempted_leads"]["ready lead"]
    assert saved["status"] == "ready"
    assert saved["last_attempt_status"] == "accepted_pending_publication"


def test_known_submission_remains_ready_for_idempotent_decision_retry(
    tmp_path: Path,
) -> None:
    portfolio = _load_portfolio()
    state_path = tmp_path / "state.json"
    state: dict[str, object] = {
        "attempted_leads": {
            "ready lead": {
                "receipt_path": str(tmp_path / "ready.json"),
                "status": "ready",
                "updated_at": portfolio._timestamp(),
            }
        }
    }

    portfolio._save_attempted_lead(
        state_path,
        state,
        {
            "lead": "ready lead",
            "receipt_path": str(tmp_path / "submitted.json"),
            "status": "submitted",
        },
        preserve_ready=True,
    )

    saved = json.loads(state_path.read_text())["attempted_leads"]["ready lead"]
    assert saved["status"] == "ready"
    assert saved["last_attempt_status"] == "submitted"


def test_noop_status_writes_auditable_lock_receipt(tmp_path: Path) -> None:
    portfolio = _load_portfolio()
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "attempted_leads": {
            "ready lead": {"status": "ready"},
            "warming lead": {"status": "warming:search_coverage"},
        }
    }))

    code = portfolio.main([
        "--output-dir",
        str(tmp_path / "run"),
        "--state-path",
        str(state_path),
        "--record-noop-status",
        "lock_busy",
    ])

    receipt = json.loads((tmp_path / "run" / "portfolio.json").read_text())
    assert code == 0
    assert receipt["final_status"] == "lock_busy"
    assert receipt["ready_buffer_count_before"] == 1
    assert receipt["ready_buffer_count_after"] == 1
    assert receipt["records"] == []


def test_run_portfolio_continues_after_blocker_until_ready(tmp_path: Path) -> None:
    portfolio = _load_portfolio()
    config = portfolio.RunConfig(
        output_dir=tmp_path,
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner=None,
        writer=None,
        selector=None,
        min_alpha_tier="publishable",
        submit=False,
        decision_wait_seconds=0,
        decision_poll_seconds=1,
        submit_wait_seconds=0,
        max_leads=0,
        state_path=None,
        lead_file=None,
        auto_discover_leads=False,
        min_open_leads=0,
        discover_count=0,
        blocked_retry_hours=0,
    )
    calls: list[list[str]] = []

    def fake_runner(
        command: list[str],
        _env: dict[str, str],
        _cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        receipt = Path(command[command.index("--publish-receipt-path") + 1])
        receipt.parent.mkdir(parents=True, exist_ok=True)
        if len(calls) == 1:
            receipt.write_text(json.dumps({"error": "candidate_publish_blocker"}))
            return subprocess.CompletedProcess(command, 5, stdout="", stderr="blocked")
        receipt.write_text(json.dumps(_ready_receipt()))
        return subprocess.CompletedProcess(command, 0, stdout="ready", stderr="")

    code = portfolio.run_portfolio(
        ["weak lead", "good lead", "unused lead"],
        config,
        runner=fake_runner,
        env={},
        cwd=Path.cwd(),
    )

    summary = json.loads((tmp_path / "portfolio.json").read_text())
    assert code == 0
    assert len(calls) == 2
    assert summary["attempted_leads"] == 2
    assert summary["records"][0]["status"] == "blocked:candidate_publish_blocker"
    assert summary["records"][1]["status"] == "ready"


def test_run_portfolio_times_out_stuck_lead_and_continues(tmp_path: Path) -> None:
    portfolio = _load_portfolio()
    config = portfolio.RunConfig(
        output_dir=tmp_path,
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner=None,
        writer=None,
        selector=None,
        min_alpha_tier="publishable",
        submit=True,
        decision_wait_seconds=0,
        decision_poll_seconds=1,
        submit_wait_seconds=0,
        max_leads=2,
        state_path=None,
        lead_file=None,
        auto_discover_leads=False,
        min_open_leads=0,
        discover_count=0,
        blocked_retry_hours=0,
        lead_timeout_seconds=3,
    )
    calls: list[list[str]] = []

    def fake_runner(
        command: list[str],
        _env: dict[str, str],
        _cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        receipt = Path(command[command.index("--publish-receipt-path") + 1])
        receipt.parent.mkdir(parents=True, exist_ok=True)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, timeout=3, output="partial", stderr="stuck")
        receipt.write_text(json.dumps(_accepted_receipt()))
        return subprocess.CompletedProcess(command, 0, stdout="accepted", stderr="")

    code = portfolio.run_portfolio(
        ["stuck lead", "good lead"],
        config,
        runner=fake_runner,
        env={},
        cwd=Path.cwd(),
    )

    summary = json.loads((tmp_path / "portfolio.json").read_text())
    first_receipt = json.loads(Path(summary["records"][0]["receipt_path"]).read_text())
    assert code == 0
    assert len(calls) == 2
    assert summary["records"][0]["status"] == "blocked:lead_timeout"
    assert summary["records"][0]["returncode"] == 124
    assert summary["records"][1]["status"] == "accepted"
    assert first_receipt["error"] == "lead_timeout"


def test_recent_lead_timeout_remains_retryable(tmp_path: Path) -> None:
    portfolio = _load_portfolio()
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "attempted_leads": {
            "stuck lead": {
                "status": "blocked:lead_timeout",
                "updated_at": portfolio._timestamp(),
            }
        }
    }))
    config = portfolio.RunConfig(
        output_dir=tmp_path / "run",
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner=None,
        writer=None,
        selector=None,
        min_alpha_tier="publishable",
        submit=True,
        decision_wait_seconds=0,
        decision_poll_seconds=1,
        submit_wait_seconds=0,
        max_leads=1,
        state_path=state_path,
        lead_file=None,
        auto_discover_leads=False,
        min_open_leads=0,
        discover_count=0,
        blocked_retry_hours=24,
    )
    calls: list[list[str]] = []

    def fake_runner(
        command: list[str],
        _env: dict[str, str],
        _cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        receipt = Path(command[command.index("--publish-receipt-path") + 1])
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(_accepted_receipt()))
        return subprocess.CompletedProcess(command, 0, stdout="accepted", stderr="")

    code = portfolio.run_portfolio(
        ["stuck lead"],
        config,
        runner=fake_runner,
        env={},
        cwd=Path.cwd(),
    )

    summary = json.loads((tmp_path / "run" / "portfolio.json").read_text())
    assert code == 0
    assert len(calls) == 1
    assert calls[0][calls[0].index("--topic") + 1] == "stuck lead"
    assert summary["skipped_recent_attempts"] == 0


def test_available_leads_prioritizes_warming_and_parks_terminal_revision() -> None:
    portfolio = _load_portfolio()
    state = {
        "attempted_leads": {
            "warming lead": {
                "status": "warming:search_coverage",
                "updated_at": portfolio._timestamp(),
            },
            "revision lead": {
                "status": "decision:revise",
                "updated_at": portfolio._timestamp(),
            },
        },
    }

    available = portfolio._available_leads(
        ["warming lead", "revision lead", "fresh lead"],
        state,
        blocked_retry_hours=24,
        now=portfolio.datetime.now(portfolio.UTC),
    )

    assert available == ["warming lead", "fresh lead"]


def test_prepare_prioritizes_closest_warming_lead_with_resource_limit(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    portfolio = _load_portfolio()
    far_receipt = tmp_path / "far.json"
    near_receipt = tmp_path / "near.json"
    far_receipt.write_text(json.dumps({
        "error": "search_backend_error",
        "message": "Full raw corpus search coverage too narrow: {'sweep_remaining_shards': 1190}",
    }))
    near_receipt.write_text(json.dumps({
        "error": "search_backend_error",
        "message": "Full raw corpus search coverage too narrow: {'sweep_remaining_shards': 93}",
    }))
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "attempted_leads": {
            "far lead": {
                "receipt_path": str(far_receipt),
                "status": "warming:search_coverage",
            },
            "near lead": {
                "receipt_path": str(near_receipt),
                "status": "warming:search_coverage",
            },
        }
    }))

    class HealthResponse:
        def __enter__(self) -> HealthResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({
                "ok": True,
                "async_sweep": {"max_inflight": 1},
            }).encode()

    observed: dict[str, object] = {}

    def health_request(req: Request, *, timeout: float) -> HealthResponse:
        observed["url"] = req.full_url
        observed["authorization"] = req.get_header("Authorization")
        observed["timeout"] = timeout
        return HealthResponse()

    monkeypatch.setattr(portfolio.request, "urlopen", health_request)
    calls: list[list[str]] = []

    def fake_runner(
        command: list[str],
        _env: dict[str, str],
        _cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        receipt = Path(command[command.index("--publish-receipt-path") + 1])
        receipt.write_text(json.dumps(_ready_receipt()))
        return subprocess.CompletedProcess(command, 0, stdout="ready", stderr="")

    config = portfolio.RunConfig(
        output_dir=tmp_path / "run",
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner=None,
        writer=None,
        selector=None,
        min_alpha_tier="publishable",
        submit=False,
        decision_wait_seconds=0,
        decision_poll_seconds=1,
        submit_wait_seconds=0,
        max_leads=3,
        state_path=state_path,
        lead_file=None,
        auto_discover_leads=False,
        min_open_leads=0,
        discover_count=0,
        blocked_retry_hours=0,
        ready_buffer_size=3,
        resource_aware_max_leads=True,
    )

    code = portfolio.run_portfolio(
        ["far lead", "near lead", "fresh lead"],
        config,
        runner=fake_runner,
        env={
            "V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL": "http://127.0.0.1:9915/api/search",
            "V5_MEMO_FULL_RAW_INDEX_TOKEN": "test-token",
        },
        cwd=Path.cwd(),
    )

    summary = json.loads((tmp_path / "run" / "portfolio.json").read_text())
    assert code == 0
    assert len(calls) == 1
    assert calls[0][calls[0].index("--topic") + 1] == "near lead"
    assert summary["effective_max_leads"] == 1
    assert summary["resource_max_inflight"] == 1
    assert summary["resource_limit_fallback"] is False
    assert summary["resource_aware_max_leads"] is True
    assert summary["ready_buffer_count_after"] == 1
    assert observed == {
        "url": "http://127.0.0.1:9915/health",
        "authorization": "Bearer test-token",
        "timeout": 2.0,
    }


def test_resource_aware_prepare_fails_safe_to_one_lead(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    portfolio = _load_portfolio()

    def unavailable(*_args: object, **_kwargs: object) -> object:
        raise OSError("unavailable")

    monkeypatch.setattr(portfolio.request, "urlopen", unavailable)
    config = portfolio.RunConfig(
        output_dir=tmp_path,
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner=None,
        writer=None,
        selector=None,
        min_alpha_tier="publishable",
        submit=False,
        decision_wait_seconds=0,
        decision_poll_seconds=1,
        submit_wait_seconds=0,
        max_leads=3,
        state_path=None,
        lead_file=None,
        auto_discover_leads=False,
        min_open_leads=0,
        discover_count=0,
        blocked_retry_hours=0,
        resource_aware_max_leads=True,
    )

    limit, probed = portfolio._preparation_lead_limit(
        config,
        {"V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL": "http://127.0.0.1:9915/search"},
    )

    assert limit == 1
    assert probed is None


def test_warming_coverage_order_handles_unknowns_and_stable_ties(tmp_path: Path) -> None:
    portfolio = _load_portfolio()
    state = {
        "attempted_leads": {
            "unknown lead": {"status": "warming:search_coverage"},
            "near a": {"status": "warming:search_coverage", "sweep_remaining_shards": 15},
            "near b": {"status": "warming:search_coverage", "sweep_remaining_shards": 15},
            "far lead": {"status": "warming:search_coverage", "sweep_remaining_shards": 100},
        }
    }

    available = portfolio._available_leads(
        ["unknown lead", "near b", "fresh lead", "far lead", "near a"],
        state,
        blocked_retry_hours=24,
        now=portfolio.datetime.now(portfolio.UTC),
    )

    assert available == ["near b", "near a", "far lead", "unknown lead", "fresh lead"]


def test_prepare_prioritizes_eligible_lead_with_complete_first_query_cache(
    tmp_path: Path,
) -> None:
    portfolio = _load_portfolio()
    env, cache_dir, shard_dir = _cache_env(tmp_path)
    cached_lead = "cached compound resistance training older adults trial"
    query, result_limit = portfolio._first_query_cache_specs(
        [cached_lead],
        env,
    )[portfolio._lead_key(cached_lead)]
    assert result_limit == 25
    _write_sweep_cache(
        portfolio,
        cache_dir / "complete.json",
        query=query,
        scope=str(shard_dir.absolute()),
    )
    state = {
        "attempted_leads": {
            cached_lead: {"status": "blocked:no_receipt_bound_alpha_candidate"},
            "warming lead": {
                "status": "warming:search_coverage",
                "sweep_remaining_shards": 1,
            },
        }
    }

    cache_keys = portfolio._complete_first_query_cache_lead_keys(
        ["warming lead", cached_lead],
        env,
        planner=None,
    )
    available = portfolio._available_leads(
        ["warming lead", cached_lead],
        state,
        blocked_retry_hours=0,
        now=portfolio.datetime.now(portfolio.UTC),
        complete_cache_lead_keys=cache_keys,
    )

    assert cache_keys == {portfolio._lead_key(cached_lead)}
    assert available == [cached_lead, "warming lead"]


def test_cache_priority_ignores_stale_wrong_scope_incomplete_and_unconfigured(
    tmp_path: Path,
) -> None:
    portfolio = _load_portfolio()
    env, cache_dir, shard_dir = _cache_env(tmp_path)
    leads = [
        "stale intervention trial",
        "wrong scope trial",
        "incomplete trial",
        "missing pass width trial",
        "wrong pass width trial",
    ]
    specs = portfolio._first_query_cache_specs(leads, env)
    _write_sweep_cache(
        portfolio,
        cache_dir / "stale.json",
        query=specs[portfolio._lead_key(leads[0])][0],
        scope=str(shard_dir.absolute()),
        created_at=0,
    )
    _write_sweep_cache(
        portfolio,
        cache_dir / "wrong-scope.json",
        query=specs[portfolio._lead_key(leads[1])][0],
        scope=str(tmp_path / "different-shards"),
    )
    _write_sweep_cache(
        portfolio,
        cache_dir / "incomplete.json",
        query=specs[portfolio._lead_key(leads[2])][0],
        scope=str(shard_dir.absolute()),
        shards_searched=1524,
        remaining_shards=1,
    )
    _write_sweep_cache(
        portfolio,
        cache_dir / "missing-pass-width.json",
        query=specs[portfolio._lead_key(leads[3])][0],
        scope=str(shard_dir.absolute()),
        pass_shard_limit=None,
    )
    _write_sweep_cache(
        portfolio,
        cache_dir / "wrong-pass-width.json",
        query=specs[portfolio._lead_key(leads[4])][0],
        scope=str(shard_dir.absolute()),
        pass_shard_limit=16,
    )
    legacy_dir = tmp_path / "unconfigured-cache"
    _write_sweep_cache(
        portfolio,
        legacy_dir / "otherwise-valid.json",
        query=specs[portfolio._lead_key(leads[0])][0],
        scope=str(shard_dir.absolute()),
    )

    cache_keys = portfolio._complete_first_query_cache_lead_keys(
        leads,
        env,
        planner=None,
    )

    assert cache_keys == set()


def test_complete_cache_priority_preserves_input_order_among_equals(
    tmp_path: Path,
) -> None:
    portfolio = _load_portfolio()
    env, cache_dir, shard_dir = _cache_env(tmp_path)
    leads = [
        "second cached compound trial",
        "first cached compound trial",
        "warming lead",
    ]
    specs = portfolio._first_query_cache_specs(leads[:2], env)
    for index, lead in enumerate(leads[:2]):
        _write_sweep_cache(
            portfolio,
            cache_dir / f"complete-{index}.json",
            query=specs[portfolio._lead_key(lead)][0],
            scope=str(shard_dir.absolute()),
        )
    state = {
        "attempted_leads": {
            lead: {"status": "blocked:no_receipt_bound_alpha_candidate"}
            for lead in leads[:2]
        }
        | {
            "warming lead": {
                "status": "warming:search_coverage",
                "sweep_remaining_shards": 1,
            }
        }
    }

    cache_keys = portfolio._complete_first_query_cache_lead_keys(
        leads,
        env,
        planner=None,
    )
    available = portfolio._available_leads(
        leads,
        state,
        blocked_retry_hours=0,
        now=portfolio.datetime.now(portfolio.UTC),
        complete_cache_lead_keys=cache_keys,
    )

    assert available == leads


def test_cache_query_limit_ignores_generic_only_limit_variables() -> None:
    portfolio = _load_portfolio()

    specs = portfolio._first_query_cache_specs(
        ["generic compound resistance training trial"],
        {
            "RESEARKA_FULLRAW_PER_QUERY_LIMIT": "7",
            "RESEARKA_FULLRAW_MAX_HITS": "7",
        },
    )

    assert next(iter(specs.values()))[1] == 50


def test_prepare_revalidates_post_quality_submission_failure_before_warming() -> None:
    portfolio = _load_portfolio()
    state = {
        "attempted_leads": {
            "warming lead": {
                "status": "warming:search_coverage",
                "sweep_remaining_shards": 1,
            },
            "submit retry": {
                "status": "blocked:researka_submit_failed",
                "updated_at": portfolio._timestamp(),
            },
        }
    }

    available = portfolio._available_leads(
        ["warming lead", "submit retry"],
        state,
        blocked_retry_hours=24,
        now=portfolio.datetime.now(portfolio.UTC),
        retry_post_quality=True,
    )
    normal_retry = portfolio._available_leads(
        ["warming lead", "submit retry"],
        state,
        blocked_retry_hours=24,
        now=portfolio.datetime.now(portfolio.UTC),
    )

    assert available == ["submit retry", "warming lead"]
    assert normal_retry == ["warming lead"]


def test_post_quality_retry_stays_ahead_of_complete_cache_priority() -> None:
    portfolio = _load_portfolio()
    state = {
        "attempted_leads": {
            "submit retry": {"status": "blocked:researka_submit_failed"},
            "cached lead": {"status": "blocked:no_receipt_bound_alpha_candidate"},
            "warming lead": {
                "status": "warming:search_coverage",
                "sweep_remaining_shards": 1,
            },
        }
    }

    available = portfolio._available_leads(
        ["warming lead", "cached lead", "submit retry"],
        state,
        blocked_retry_hours=0,
        now=portfolio.datetime.now(portfolio.UTC),
        retry_post_quality=True,
        complete_cache_lead_keys={"cached lead", "submit retry"},
    )

    assert available == ["submit retry", "cached lead", "warming lead"]


def test_available_leads_prioritizes_ready_supply_for_submit() -> None:
    portfolio = _load_portfolio()
    state = {
        "attempted_leads": {
            "ready lead": {"status": "ready", "updated_at": portfolio._timestamp()},
            "warming lead": {
                "status": "warming:search_coverage",
                "updated_at": portfolio._timestamp(),
            },
        },
    }

    available = portfolio._available_leads(
        ["warming lead", "fresh lead", "ready lead"],
        state,
        blocked_retry_hours=24,
        now=portfolio.datetime.now(portfolio.UTC),
        prefer_ready=True,
    )

    assert available == ["ready lead", "warming lead", "fresh lead"]


def test_ready_only_submit_consumes_ready_supply_without_cold_fallback(tmp_path: Path) -> None:
    portfolio = _load_portfolio()
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "attempted_leads": {
            "ready lead": {"status": "ready", "updated_at": portfolio._timestamp()},
            "warming lead": {
                "status": "warming:search_coverage",
                "updated_at": portfolio._timestamp(),
            },
        }
    }))
    config = portfolio.RunConfig(
        output_dir=tmp_path / "run",
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner=None,
        writer=None,
        selector=None,
        min_alpha_tier="publishable",
        submit=True,
        decision_wait_seconds=0,
        decision_poll_seconds=1,
        submit_wait_seconds=0,
        max_leads=1,
        state_path=state_path,
        lead_file=None,
        auto_discover_leads=True,
        min_open_leads=40,
        discover_count=20,
        blocked_retry_hours=24,
        ready_only=True,
    )
    calls: list[list[str]] = []

    def fake_runner(
        command: list[str],
        _env: dict[str, str],
        _cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        receipt = Path(command[command.index("--publish-receipt-path") + 1])
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(_accepted_receipt()))
        return subprocess.CompletedProcess(command, 0, stdout="accepted", stderr="")

    code = portfolio.run_portfolio(
        ["warming lead", "cold lead", "ready lead"],
        config,
        runner=fake_runner,
        env={},
        cwd=Path.cwd(),
    )

    summary = json.loads((tmp_path / "run" / "portfolio.json").read_text())
    assert code == 0
    assert len(calls) == 1
    assert calls[0][calls[0].index("--topic") + 1] == "ready lead"
    assert summary["discovered_leads"] == []
    assert summary["ready_only"] is True
    assert summary["final_status"] == "accepted"


def test_ready_only_submit_is_clean_noop_without_ready_supply(tmp_path: Path) -> None:
    portfolio = _load_portfolio()
    config = portfolio.RunConfig(
        output_dir=tmp_path / "run",
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner=None,
        writer=None,
        selector=None,
        min_alpha_tier="publishable",
        submit=True,
        decision_wait_seconds=0,
        decision_poll_seconds=1,
        submit_wait_seconds=0,
        max_leads=1,
        state_path=None,
        lead_file=None,
        auto_discover_leads=False,
        min_open_leads=0,
        discover_count=0,
        blocked_retry_hours=0,
        ready_only=True,
    )

    def fail_runner(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("cold leads must not run in ready-only mode")

    code = portfolio.run_portfolio(
        ["cold lead"],
        config,
        runner=fail_runner,
        env={},
        cwd=Path.cwd(),
    )

    summary = json.loads((tmp_path / "run" / "portfolio.json").read_text())
    assert code == 0
    assert summary["selected_leads"] == 0
    assert summary["attempted_leads"] == 0
    assert summary["final_status"] == "ready_buffer_empty"


def test_prepare_run_fills_only_missing_ready_capacity(tmp_path: Path) -> None:
    portfolio = _load_portfolio()
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "attempted_leads": {
            "ready lead": {"status": "ready", "updated_at": portfolio._timestamp()},
            "orphan ready": {"status": "ready", "updated_at": portfolio._timestamp()},
        }
    }))
    config = portfolio.RunConfig(
        output_dir=tmp_path / "run",
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner=None,
        writer=None,
        selector=None,
        min_alpha_tier="publishable",
        submit=False,
        decision_wait_seconds=0,
        decision_poll_seconds=1,
        submit_wait_seconds=0,
        max_leads=3,
        state_path=state_path,
        lead_file=None,
        auto_discover_leads=False,
        min_open_leads=0,
        discover_count=0,
        blocked_retry_hours=24,
        ready_buffer_size=2,
    )
    calls: list[list[str]] = []

    def fake_runner(
        command: list[str],
        _env: dict[str, str],
        _cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        receipt = Path(command[command.index("--publish-receipt-path") + 1])
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(_ready_receipt()))
        return subprocess.CompletedProcess(command, 0, stdout="ready", stderr="")

    code = portfolio.run_portfolio(
        ["ready lead", "fresh lead", "unused lead"],
        config,
        runner=fake_runner,
        env={},
        cwd=Path.cwd(),
    )

    summary = json.loads((tmp_path / "run" / "portfolio.json").read_text())
    state = json.loads(state_path.read_text())
    assert code == 0
    assert len(calls) == 1
    assert calls[0][calls[0].index("--topic") + 1] == "fresh lead"
    assert summary["ready_buffer_count_before"] == 1
    assert summary["ready_buffer_count_after"] == 2
    assert summary["selected_leads"] == 1
    assert summary["publish_quality_validation"] is True
    assert summary["records"][0]["publish_quality_validation"] is True
    assert state["attempted_leads"]["ready lead"]["status"] == "ready"
    assert state["attempted_leads"]["orphan ready"]["status"] == "ready"
    assert state["attempted_leads"]["fresh lead"]["status"] == "ready"


def test_prepare_run_is_noop_when_ready_buffer_is_full(tmp_path: Path) -> None:
    portfolio = _load_portfolio()
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "attempted_leads": {
            "ready one": {"status": "ready", "updated_at": portfolio._timestamp()},
            "ready two": {"status": "ready", "updated_at": portfolio._timestamp()},
        }
    }))
    config = portfolio.RunConfig(
        output_dir=tmp_path / "run",
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner=None,
        writer=None,
        selector=None,
        min_alpha_tier="publishable",
        submit=False,
        decision_wait_seconds=0,
        decision_poll_seconds=1,
        submit_wait_seconds=0,
        max_leads=1,
        state_path=state_path,
        lead_file=None,
        auto_discover_leads=False,
        min_open_leads=0,
        discover_count=0,
        blocked_retry_hours=24,
        ready_buffer_size=2,
    )

    code = portfolio.run_portfolio(
        ["ready one", "ready two", "fresh lead"],
        config,
        runner=lambda *_args: (_ for _ in ()).throw(AssertionError("must not run")),
        env={},
        cwd=Path.cwd(),
    )

    summary = json.loads((tmp_path / "run" / "portfolio.json").read_text())
    assert code == 0
    assert summary["attempted_leads"] == 0
    assert summary["final_status"] == "ready_buffer_full"
    assert summary["ready_buffer_count_before"] == 2
    assert summary["ready_buffer_count_after"] == 2


def test_run_portfolio_skips_completed_leads_and_saves_success(tmp_path: Path) -> None:
    portfolio = _load_portfolio()
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"completed_leads": {"done lead": {"status": "accepted"}}}))
    config = portfolio.RunConfig(
        output_dir=tmp_path / "run",
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner=None,
        writer=None,
        selector=None,
        min_alpha_tier="publishable",
        submit=True,
        decision_wait_seconds=0,
        decision_poll_seconds=1,
        submit_wait_seconds=0,
        max_leads=0,
        state_path=state_path,
        lead_file=None,
        auto_discover_leads=False,
        min_open_leads=0,
        discover_count=0,
        blocked_retry_hours=0,
    )
    calls: list[list[str]] = []

    def fake_runner(
        command: list[str],
        _env: dict[str, str],
        _cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        receipt = Path(command[command.index("--publish-receipt-path") + 1])
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(_accepted_receipt()))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    code = portfolio.run_portfolio(
        ["done lead", "fresh lead"],
        config,
        runner=fake_runner,
        env={},
        cwd=Path.cwd(),
    )

    state = json.loads(state_path.read_text())
    summary = json.loads((tmp_path / "run" / "portfolio.json").read_text())
    assert code == 0
    assert len(calls) == 1
    assert calls[0][calls[0].index("--topic") + 1] == "fresh lead"
    assert summary["skipped_completed_leads"] == 1
    assert state["completed_leads"]["fresh lead"]["status"] == "accepted"


def test_submitted_without_decision_is_not_completed(tmp_path: Path) -> None:
    portfolio = _load_portfolio()
    state_path = tmp_path / "state.json"
    config = portfolio.RunConfig(
        output_dir=tmp_path / "run",
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner=None,
        writer=None,
        selector=None,
        min_alpha_tier="publishable",
        submit=True,
        decision_wait_seconds=0,
        decision_poll_seconds=1,
        submit_wait_seconds=0,
        max_leads=0,
        state_path=state_path,
        lead_file=None,
        auto_discover_leads=False,
        min_open_leads=0,
        discover_count=0,
        blocked_retry_hours=24,
    )

    def fake_runner(
        command: list[str],
        _env: dict[str, str],
        _cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        receipt = Path(command[command.index("--publish-receipt-path") + 1])
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({"submission_id": "pending-submit"}))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    code = portfolio.run_portfolio(
        ["pending lead"],
        config,
        runner=fake_runner,
        env={},
        cwd=Path.cwd(),
    )

    state = json.loads(state_path.read_text())
    summary = json.loads((tmp_path / "run" / "portfolio.json").read_text())
    assert code == 0
    assert summary["final_status"] == "submitted"
    assert state["attempted_leads"]["pending lead"]["status"] == "submitted"
    assert "pending lead" not in state.get("completed_leads", {})


def test_auto_discovery_appends_unique_leads_when_open_queue_is_low(tmp_path: Path) -> None:
    portfolio = _load_portfolio()
    lead_file = tmp_path / "leads.txt"
    lead_file.write_text("done lead\nurolithin a muscle strength older adults randomized trial\n")
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({"completed_leads": {"done lead": {"status": "accepted"}}}))
    config = portfolio.RunConfig(
        output_dir=tmp_path / "run",
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner=None,
        writer=None,
        selector=None,
        min_alpha_tier="publishable",
        submit=False,
        decision_wait_seconds=0,
        decision_poll_seconds=1,
        submit_wait_seconds=0,
        max_leads=1,
        state_path=state_path,
        lead_file=lead_file,
        auto_discover_leads=True,
        min_open_leads=3,
        discover_count=3,
        blocked_retry_hours=0,
    )

    def fake_runner(
        command: list[str],
        _env: dict[str, str],
        _cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        receipt = Path(command[command.index("--publish-receipt-path") + 1])
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(_ready_receipt()))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    code = portfolio.run_portfolio(
        ["done lead", "urolithin a muscle strength older adults randomized trial"],
        config,
        runner=fake_runner,
        env={},
        cwd=Path.cwd(),
    )

    lead_lines = [line for line in lead_file.read_text().splitlines() if line]
    summary = json.loads((tmp_path / "run" / "portfolio.json").read_text())
    assert code == 0
    assert len(lead_lines) == len(set(lead_lines))
    assert len(summary["discovered_leads"]) == 3
    assert "urolithin a muscle strength older adults randomized trial" not in summary["discovered_leads"]


def test_recent_blocked_leads_cool_down_so_later_leads_can_run(tmp_path: Path) -> None:
    portfolio = _load_portfolio()
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "attempted_leads": {
            "blocked lead": {
                "status": "blocked:candidate_publish_blocker",
                "updated_at": portfolio._timestamp(),
            },
        },
    }))
    config = portfolio.RunConfig(
        output_dir=tmp_path / "run",
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner=None,
        writer=None,
        selector=None,
        min_alpha_tier="publishable",
        submit=False,
        decision_wait_seconds=0,
        decision_poll_seconds=1,
        submit_wait_seconds=0,
        max_leads=1,
        state_path=state_path,
        lead_file=None,
        auto_discover_leads=False,
        min_open_leads=0,
        discover_count=0,
        blocked_retry_hours=24,
    )
    calls: list[list[str]] = []

    def fake_runner(
        command: list[str],
        _env: dict[str, str],
        _cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        receipt = Path(command[command.index("--publish-receipt-path") + 1])
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(_ready_receipt()))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    code = portfolio.run_portfolio(
        ["blocked lead", "fresh lead"],
        config,
        runner=fake_runner,
        env={},
        cwd=Path.cwd(),
    )

    summary = json.loads((tmp_path / "run" / "portfolio.json").read_text())
    assert code == 0
    assert calls[0][calls[0].index("--topic") + 1] == "fresh lead"
    assert summary["skipped_recent_attempts"] == 1


def test_recent_submitted_leads_cool_down_so_later_leads_can_run(tmp_path: Path) -> None:
    portfolio = _load_portfolio()
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "attempted_leads": {
            "pending lead": {
                "status": "submitted",
                "updated_at": portfolio._timestamp(),
            },
        },
    }))
    config = portfolio.RunConfig(
        output_dir=tmp_path / "run",
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner=None,
        writer=None,
        selector=None,
        min_alpha_tier="publishable",
        submit=False,
        decision_wait_seconds=0,
        decision_poll_seconds=1,
        submit_wait_seconds=0,
        max_leads=1,
        state_path=state_path,
        lead_file=None,
        auto_discover_leads=False,
        min_open_leads=0,
        discover_count=0,
        blocked_retry_hours=24,
    )
    calls: list[list[str]] = []

    def fake_runner(
        command: list[str],
        _env: dict[str, str],
        _cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        receipt = Path(command[command.index("--publish-receipt-path") + 1])
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(_ready_receipt()))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    code = portfolio.run_portfolio(
        ["pending lead", "fresh lead"],
        config,
        runner=fake_runner,
        env={},
        cwd=Path.cwd(),
    )

    summary = json.loads((tmp_path / "run" / "portfolio.json").read_text())
    assert code == 0
    assert calls[0][calls[0].index("--topic") + 1] == "fresh lead"
    assert summary["skipped_recent_attempts"] == 1


def test_recent_revise_decision_is_parked_while_fresh_lead_runs(tmp_path: Path) -> None:
    portfolio = _load_portfolio()
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "attempted_leads": {
            "revision lead": {
                "status": "decision:revise",
                "updated_at": portfolio._timestamp(),
            },
        },
    }))
    config = portfolio.RunConfig(
        output_dir=tmp_path / "run",
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner=None,
        writer=None,
        selector=None,
        min_alpha_tier="publishable",
        submit=False,
        decision_wait_seconds=0,
        decision_poll_seconds=1,
        submit_wait_seconds=0,
        max_leads=1,
        state_path=state_path,
        lead_file=None,
        auto_discover_leads=False,
        min_open_leads=0,
        discover_count=0,
        blocked_retry_hours=24,
    )
    calls: list[list[str]] = []

    def fake_runner(
        command: list[str],
        _env: dict[str, str],
        _cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        receipt = Path(command[command.index("--publish-receipt-path") + 1])
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(_ready_receipt()))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    code = portfolio.run_portfolio(
        ["revision lead", "fresh lead"],
        config,
        runner=fake_runner,
        env={},
        cwd=Path.cwd(),
    )

    summary = json.loads((tmp_path / "run" / "portfolio.json").read_text())
    assert code == 0
    assert calls[0][calls[0].index("--topic") + 1] == "fresh lead"
    assert summary["selected_leads"] == 1


def test_terminal_revision_receipt_is_persisted_without_automatic_resubmit(
    tmp_path: Path,
) -> None:
    portfolio = _load_portfolio()
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "attempted_leads": {
            "revision lead": {
                "status": "ready",
                "updated_at": portfolio._timestamp(),
            },
        },
    }))
    config = portfolio.RunConfig(
        output_dir=tmp_path / "submit",
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner=None,
        writer=None,
        selector=None,
        min_alpha_tier="publishable",
        submit=True,
        decision_wait_seconds=0,
        decision_poll_seconds=1,
        submit_wait_seconds=0,
        max_leads=1,
        state_path=state_path,
        lead_file=None,
        auto_discover_leads=False,
        min_open_leads=0,
        discover_count=0,
        blocked_retry_hours=24,
        ready_only=True,
    )
    calls: list[list[str]] = []

    def fake_runner(
        command: list[str],
        _env: dict[str, str],
        _cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        receipt = Path(command[command.index("--publish-receipt-path") + 1])
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({
            "submission": {"id": "parent-submission"},
            "decision": {
                "decision": "revise",
                "required_revisions": ["Narrow the population.", "Deduplicate sources."],
                "review_summary": "The synthesis must be rebuilt.",
                "resubmission": {
                    "allowed": True,
                    "parent_submission_id": "parent-submission",
                },
            },
        }))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    assert portfolio.run_portfolio(
        ["revision lead"],
        config,
        runner=fake_runner,
        env={},
        cwd=Path.cwd(),
    ) == 1

    state = json.loads(state_path.read_text())
    saved = state["attempted_leads"]["revision lead"]
    assert saved["status"] == "decision:revise"
    assert saved["revision"] == {
        "parent_submission_id": "parent-submission",
        "required_revisions": ["Narrow the population.", "Deduplicate sources."],
        "resubmission_allowed": True,
        "review_summary": "The synthesis must be rebuilt.",
    }

    prepare = replace(
        config,
        output_dir=tmp_path / "prepare",
        submit=False,
        ready_only=False,
        ready_buffer_size=1,
    )
    assert portfolio.run_portfolio(
        ["revision lead"],
        prepare,
        runner=fake_runner,
        env={},
        cwd=Path.cwd(),
    ) == 0
    assert len(calls) == 1
    summary = json.loads((tmp_path / "prepare" / "portfolio.json").read_text())
    assert summary["selected_leads"] == 0


def test_search_coverage_warming_continues_past_generic_zero_wait(tmp_path: Path) -> None:
    portfolio = _load_portfolio()
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "attempted_leads": {
            "cold lead": {
                "status": "blocked:search_backend_error",
                "updated_at": portfolio._timestamp(),
            },
        },
    }))
    config = portfolio.RunConfig(
        output_dir=tmp_path / "run",
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner=None,
        writer=None,
        selector=None,
        min_alpha_tier="publishable",
        submit=True,
        decision_wait_seconds=0,
        decision_poll_seconds=1,
        submit_wait_seconds=0,
        max_leads=12,
        state_path=state_path,
        lead_file=None,
        auto_discover_leads=False,
        min_open_leads=0,
        discover_count=0,
        blocked_retry_hours=24,
    )
    calls: list[list[str]] = []

    def fake_runner(
        command: list[str],
        run_env: dict[str, str],
        _cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert run_env["RESEARKA_FULLRAW_FOREGROUND_SWEEP_WAIT_SECONDS"] == "0"
        assert "V5_MEMO_FULL_RAW_FOREGROUND_SWEEP_WAIT_SECONDS" not in run_env
        assert "V5_MEMO_FULL_RAW_SEARCH_BUDGET_SECONDS" not in run_env
        receipt = Path(command[command.index("--publish-receipt-path") + 1])
        receipt.parent.mkdir(parents=True, exist_ok=True)
        if command[command.index("--topic") + 1] == "fresh lead":
            receipt.write_text(json.dumps({"submission": {"id": "submitted"}}))
            return subprocess.CompletedProcess(command, 0, stdout="submitted", stderr="")
        receipt.write_text(json.dumps({
            "error": "search_backend_error",
            "message": "Full raw corpus search coverage too narrow: {'shards_searched': None}",
        }))
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="warming")

    code = portfolio.run_portfolio(
        ["cold lead", "fresh lead"],
        config,
        runner=fake_runner,
        env={"RESEARKA_FULLRAW_FOREGROUND_SWEEP_WAIT_SECONDS": "0"},
        cwd=Path.cwd(),
    )

    state = json.loads(state_path.read_text())
    summary = json.loads((tmp_path / "run" / "portfolio.json").read_text())
    assert code == 0
    assert len(calls) == 2
    assert calls[0][calls[0].index("--topic") + 1] == "cold lead"
    assert calls[1][calls[1].index("--topic") + 1] == "fresh lead"
    assert summary["final_status"] == "submitted"
    assert summary["skipped_recent_attempts"] == 0
    assert state["attempted_leads"]["cold lead"]["status"] == "warming:search_coverage"
    assert state["attempted_leads"]["fresh lead"]["status"] == "submitted"
    assert "cold lead" not in state.get("completed_leads", {})

    calls.clear()
    second_code = portfolio.run_portfolio(
        ["cold lead", "fresh lead"],
        replace(config, output_dir=tmp_path / "run2"),
        runner=fake_runner,
        env={"RESEARKA_FULLRAW_FOREGROUND_SWEEP_WAIT_SECONDS": "0"},
        cwd=Path.cwd(),
    )

    second_summary = json.loads((tmp_path / "run2" / "portfolio.json").read_text())
    assert second_code == 0
    assert len(calls) == 1
    assert calls[0][calls[0].index("--topic") + 1] == "cold lead"
    assert second_summary["skipped_recent_attempts"] == 1


def test_portfolio_injects_fullraw_wait_when_unconfigured(tmp_path: Path) -> None:
    portfolio = _load_portfolio()
    config = portfolio.RunConfig(
        output_dir=tmp_path,
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner=None,
        writer=None,
        selector=None,
        min_alpha_tier="publishable",
        submit=True,
        decision_wait_seconds=0,
        decision_poll_seconds=1,
        submit_wait_seconds=0,
        max_leads=1,
        state_path=None,
        lead_file=None,
        auto_discover_leads=False,
        min_open_leads=0,
        discover_count=0,
        blocked_retry_hours=0,
    )

    run_env = portfolio._portfolio_run_env(config, {})

    assert run_env["V5_MEMO_FULL_RAW_FOREGROUND_SWEEP_WAIT_SECONDS"] == "21600"
    assert run_env["V5_MEMO_FULL_RAW_SEARCH_BUDGET_SECONDS"] == "21600"


def test_portfolio_caps_injected_fullraw_wait_to_lead_timeout(tmp_path: Path) -> None:
    portfolio = _load_portfolio()
    config = portfolio.RunConfig(
        output_dir=tmp_path,
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner=None,
        writer=None,
        selector=None,
        min_alpha_tier="publishable",
        submit=True,
        decision_wait_seconds=0,
        decision_poll_seconds=1,
        submit_wait_seconds=0,
        max_leads=1,
        state_path=None,
        lead_file=None,
        auto_discover_leads=False,
        min_open_leads=0,
        discover_count=0,
        blocked_retry_hours=0,
        lead_timeout_seconds=600,
    )

    run_env = portfolio._portfolio_run_env(config, {})

    assert run_env["V5_MEMO_FULL_RAW_FOREGROUND_SWEEP_WAIT_SECONDS"] == "60"
    assert run_env["V5_MEMO_FULL_RAW_SEARCH_BUDGET_SECONDS"] == "60"


def test_run_env_prefers_explicit_v5_zero_over_generic_default(tmp_path: Path) -> None:
    portfolio = _load_portfolio()
    config = portfolio.RunConfig(
        output_dir=tmp_path,
        python="python3",
        module="v5_memo",
        searcher="fullraw",
        planner=None,
        writer=None,
        selector=None,
        min_alpha_tier="publishable",
        submit=True,
        decision_wait_seconds=0,
        decision_poll_seconds=1,
        submit_wait_seconds=0,
        max_leads=1,
        state_path=None,
        lead_file=None,
        auto_discover_leads=False,
        min_open_leads=0,
        discover_count=0,
        blocked_retry_hours=0,
    )

    run_env = portfolio._portfolio_run_env(
        config,
        {
            "V5_MEMO_FULL_RAW_FOREGROUND_SWEEP_WAIT_SECONDS": "0",
            "RESEARKA_FULLRAW_FOREGROUND_SWEEP_WAIT_SECONDS": "0",
        },
    )

    assert run_env["V5_MEMO_FULL_RAW_FOREGROUND_SWEEP_WAIT_SECONDS"] == "0"
    assert run_env["RESEARKA_FULLRAW_FOREGROUND_SWEEP_WAIT_SECONDS"] == "0"
    assert "V5_MEMO_FULL_RAW_SEARCH_BUDGET_SECONDS" not in run_env
