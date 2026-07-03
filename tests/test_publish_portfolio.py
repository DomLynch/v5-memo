from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType


def _load_portfolio() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "v5_publish_portfolio.py"
    spec = importlib.util.spec_from_file_location("v5_publish_portfolio", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_accept_decision_wins_even_when_visibility_update_warns() -> None:
    portfolio = _load_portfolio()

    status = portfolio.classify_run(
        0,
        {
            "decision": {"decision": "accept"},
            "visibility_error": {"error": "researka_visibility_update_failed"},
        },
        submit=True,
    )

    assert status == "accepted"


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
        receipt.write_text(json.dumps({"memo": "ready"}))
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
        receipt.write_text(json.dumps({"decision": {"decision": "accept"}}))
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
        receipt.write_text(json.dumps({"memo": "ready"}))
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
        receipt.write_text(json.dumps({"memo": "ready"}))
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
        receipt.write_text(json.dumps({"memo": "ready"}))
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


def test_search_coverage_warming_stops_then_uses_daily_cooldown(tmp_path: Path) -> None:
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
        assert run_env["V5_MEMO_FULL_RAW_FOREGROUND_SWEEP_WAIT_SECONDS"] == "14400"
        receipt = Path(command[command.index("--publish-receipt-path") + 1])
        receipt.parent.mkdir(parents=True, exist_ok=True)
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
    assert len(calls) == 1
    assert calls[0][calls[0].index("--topic") + 1] == "cold lead"
    assert summary["final_status"] == "warming:search_coverage"
    assert summary["skipped_recent_attempts"] == 0
    assert state["attempted_leads"]["cold lead"]["status"] == "warming:search_coverage"
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
    assert calls[0][calls[0].index("--topic") + 1] == "fresh lead"
    assert second_summary["skipped_recent_attempts"] == 1
