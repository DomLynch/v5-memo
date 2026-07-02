#!/usr/bin/env python3
"""Try V5 publish leads until one clears the existing strict CLI gate."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_LEADS = (
    "urolithin muscle strength endurance older adults trial",
    "urolithin muscle recovery trained runners placebo trial",
    "metformin exercise training adaptation older adults trial",
    "resveratrol exercise training mitochondrial adaptation trial",
    "cold water immersion resistance training adaptation trial",
)
TAIL_CHARS = 2000

Runner = Callable[
    [Sequence[str], Mapping[str, str], Path],
    subprocess.CompletedProcess[str],
]


@dataclass(frozen=True)
class RunConfig:
    output_dir: Path
    python: str
    module: str
    searcher: str
    planner: str | None
    writer: str | None
    selector: str | None
    min_alpha_tier: str
    submit: bool
    decision_wait_seconds: float
    decision_poll_seconds: float
    submit_wait_seconds: float
    max_leads: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug[:80] or "lead"


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        stripped = value.strip()
        key = stripped.casefold()
        if stripped and key not in seen:
            seen.add(key)
            result.append(stripped)
    return result


def load_leads(args: argparse.Namespace) -> list[str]:
    leads = list(args.lead or [])
    if args.lead_file:
        for line in Path(args.lead_file).read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                leads.append(stripped)
    return _dedupe(leads or list(DEFAULT_LEADS))


def build_command(lead: str, lead_dir: Path, receipt_path: Path, config: RunConfig) -> list[str]:
    command = [
        config.python,
        "-m",
        config.module,
        "--topic",
        lead,
        "--query",
        lead,
        "--searcher",
        config.searcher,
        "--min-alpha-tier",
        config.min_alpha_tier,
        "--require-full-raw-corpus",
        "--output-dir",
        str(lead_dir),
        "--publish-receipt-path",
        str(receipt_path),
    ]
    if config.planner:
        command.extend(["--planner", config.planner])
    if config.writer:
        command.extend(["--writer", config.writer])
    if config.selector:
        command.extend(["--selector", config.selector])
    if config.submit:
        command.extend([
            "--submit-researka",
            "--researka-list-if-accepted",
            "--researka-decision-wait-seconds",
            str(config.decision_wait_seconds),
            "--researka-decision-poll-seconds",
            str(config.decision_poll_seconds),
            "--researka-submit-wait-seconds",
            str(config.submit_wait_seconds),
        ])
    return command


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"error": "invalid_receipt_json"}
    return data if isinstance(data, dict) else {"error": "invalid_receipt_shape"}


def classify_run(returncode: int, receipt: Mapping[str, object], *, submit: bool) -> str:
    raw_decision = receipt.get("decision")
    if isinstance(raw_decision, Mapping):
        decision = str(raw_decision.get("decision") or "")
        if decision == "accept":
            return "accepted"
        if decision in {"reject", "revise"}:
            return f"decision:{decision}"
    if submit and returncode == 0 and ("submission" in receipt or "submission_id" in receipt):
        return "submitted"
    error = receipt.get("error")
    if error == "researka_submit_deferred":
        return "deferred"
    if error:
        return f"blocked:{error}"
    if returncode == 0:
        return "submitted" if submit else "ready"
    return "failed_no_receipt"


def _should_stop(status: str) -> bool:
    return status in {"accepted", "submitted", "ready", "deferred"}


def _run(command: Sequence[str], env: Mapping[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env),
        text=True,
        capture_output=True,
        check=False,
    )


def _env_for_repo(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src = str(repo_root / "src")
    env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else f"{src}{os.pathsep}{env['PYTHONPATH']}"
    return env


def run_portfolio(
    leads: Sequence[str],
    config: RunConfig,
    *,
    runner: Runner = _run,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> int:
    repo_root = cwd or _repo_root()
    run_env = env or _env_for_repo(repo_root)
    lead_limit = config.max_leads if config.max_leads > 0 else len(leads)
    selected = list(leads[:lead_limit])
    records: list[dict[str, object]] = []
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for index, lead in enumerate(selected, start=1):
        lead_dir = config.output_dir / f"{index:02d}-{_slug(lead)}"
        receipt_path = lead_dir / "publish-receipt.json"
        lead_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(lead, lead_dir, receipt_path, config)
        completed = runner(command, run_env, repo_root)
        (lead_dir / "stdout.txt").write_text(completed.stdout or "")
        (lead_dir / "stderr.txt").write_text(completed.stderr or "")
        receipt = _read_json(receipt_path)
        status = classify_run(completed.returncode, receipt, submit=config.submit)
        record = {
            "lead": lead,
            "returncode": completed.returncode,
            "status": status,
            "receipt_path": str(receipt_path),
            "stdout_tail": (completed.stdout or "")[-TAIL_CHARS:],
            "stderr_tail": (completed.stderr or "")[-TAIL_CHARS:],
        }
        error = receipt.get("error")
        if error:
            record["error"] = error
        records.append(record)
        if _should_stop(status):
            break

    summary = {
        "created_at": _timestamp(),
        "submit": config.submit,
        "selected_leads": len(selected),
        "attempted_leads": len(records),
        "final_status": records[-1]["status"] if records else "no_leads",
        "records": records,
    }
    (config.output_dir / "portfolio.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    final_status = str(summary["final_status"])
    if final_status in {"accepted", "submitted", "ready"}:
        return 0
    return 6 if final_status == "deferred" else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lead", action="append", default=[])
    parser.add_argument("--lead-file")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp") / f"v5-portfolio-{_timestamp()}")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--module", default="v5_memo")
    parser.add_argument(
        "--searcher",
        choices=["fullraw", "smart", "hybrid", "openalex", "researka"],
        default="fullraw",
    )
    parser.add_argument("--planner", choices=["seed", "minimax"])
    parser.add_argument("--writer", choices=["template", "minimax"])
    parser.add_argument("--selector", choices=["deterministic", "minimax"])
    parser.add_argument("--min-alpha-tier", choices=["publishable", "elite"], default="publishable")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--decision-wait-seconds", type=float, default=180.0)
    parser.add_argument("--decision-poll-seconds", type=float, default=5.0)
    parser.add_argument("--submit-wait-seconds", type=float, default=0.0)
    parser.add_argument("--max-leads", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    leads = load_leads(args)
    if not leads:
        raise SystemExit("at least one lead is required")
    if args.max_leads < 0:
        raise SystemExit("--max-leads must be >= 0")
    config = RunConfig(
        output_dir=args.output_dir,
        python=args.python,
        module=args.module,
        searcher=args.searcher,
        planner=args.planner,
        writer=args.writer,
        selector=args.selector,
        min_alpha_tier=args.min_alpha_tier,
        submit=args.submit,
        decision_wait_seconds=args.decision_wait_seconds,
        decision_poll_seconds=args.decision_poll_seconds,
        submit_wait_seconds=args.submit_wait_seconds,
        max_leads=args.max_leads,
    )
    return run_portfolio(leads, config)


if __name__ == "__main__":
    raise SystemExit(main())
