from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
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


def test_classify_accept_with_visibility_error_is_not_publish_success() -> None:
    portfolio = _load_portfolio()

    status = portfolio.classify_run(
        0,
        {
            "decision": {"decision": "accept"},
            "visibility_error": {"error": "researka_visibility_update_failed"},
        },
        submit=True,
    )

    assert status == "accepted_unlisted"


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
