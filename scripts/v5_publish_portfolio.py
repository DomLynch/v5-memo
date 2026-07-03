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
from datetime import UTC, datetime, timedelta
from pathlib import Path

DEFAULT_LEADS = (
    "urolithin muscle strength endurance older adults trial",
    "urolithin muscle recovery trained runners placebo trial",
    "metformin exercise training adaptation older adults trial",
    "resveratrol exercise training mitochondrial adaptation trial",
    "cold water immersion resistance training adaptation trial",
)
DISCOVERY_INTERVENTIONS = (
    "urolithin a",
    "nicotinamide riboside",
    "nmn",
    "taurine",
    "creatine",
    "omega 3",
    "vitamin d",
    "collagen peptides",
    "nitrate supplementation",
    "beetroot nitrate",
    "sauna bathing",
    "heat therapy",
    "cold water immersion",
    "metformin",
    "acarbose",
    "rapamycin",
    "fisetin senolytic",
    "quercetin dasatinib senolytic",
    "spermidine",
    "resveratrol",
    "curcumin",
    "ashwagandha",
    "beta alanine",
    "time restricted eating",
    "ketogenic diet",
    "protein timing",
    "leucine supplementation",
)
DISCOVERY_CONTEXTS = (
    "muscle strength older adults randomized trial",
    "resistance training adaptation human trial",
    "exercise recovery inflammation placebo trial",
    "mitochondrial function exercise adaptation human trial",
    "frailty physical function older adults trial",
    "sarcopenia lean mass randomized trial",
    "endurance performance older adults trial",
    "blood pressure vascular aging trial",
    "insulin sensitivity exercise training trial",
    "cognitive aging physical function trial",
    "sleep recovery exercise adaptation trial",
    "bone density resistance training trial",
    "tendon collagen adaptation trial",
    "immune inflammation aging placebo trial",
)
DISCOVERY_ANGLES = (
    "",
    "null result",
    "dose response",
    "sex differences",
    "subgroup response",
    "adverse adaptation",
)
TAIL_CHARS = 2000
STATE_TIME_FORMAT = "%Y%m%dT%H%M%SZ"
PORTFOLIO_SWEEP_WAIT_SECONDS = "14400"
V5_SWEEP_WAIT_ENV = "V5_MEMO_FULL_RAW_FOREGROUND_SWEEP_WAIT_SECONDS"
GENERIC_SWEEP_WAIT_ENV = "RESEARKA_FULLRAW_FOREGROUND_SWEEP_WAIT_SECONDS"
V5_SEARCH_BUDGET_ENV = "V5_MEMO_FULL_RAW_SEARCH_BUDGET_SECONDS"
GENERIC_SEARCH_BUDGET_ENV = "RESEARKA_FULLRAW_SEARCH_BUDGET_SECONDS"

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
    state_path: Path | None
    lead_file: Path | None
    auto_discover_leads: bool
    min_open_leads: int
    discover_count: int
    blocked_retry_hours: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _timestamp() -> str:
    return datetime.now(UTC).strftime(STATE_TIME_FORMAT)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug[:80] or "lead"


def _lead_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        stripped = value.strip()
        key = _lead_key(stripped)
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


def _load_state(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _completed_leads(state: Mapping[str, object]) -> Mapping[str, object]:
    raw = state.get("completed_leads")
    return raw if isinstance(raw, Mapping) else {}


def _attempted_leads(state: Mapping[str, object]) -> Mapping[str, object]:
    raw = state.get("attempted_leads")
    return raw if isinstance(raw, Mapping) else {}


def _state_keys(values: Mapping[str, object]) -> set[str]:
    return {_lead_key(str(value)) for value in values}


def _parse_state_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.strptime(value, STATE_TIME_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        return None


def _positive_float(value: object) -> float | None:
    if not isinstance(value, str | int | float):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _portfolio_run_env(config: RunConfig, base_env: Mapping[str, str]) -> dict[str, str]:
    run_env = dict(base_env)
    if not (config.submit and config.searcher == "fullraw"):
        return run_env
    injected_sweep_wait = False
    if (
        _positive_float(run_env.get(V5_SWEEP_WAIT_ENV)) is None
        and _positive_float(run_env.get(GENERIC_SWEEP_WAIT_ENV)) is None
    ):
        run_env[V5_SWEEP_WAIT_ENV] = PORTFOLIO_SWEEP_WAIT_SECONDS
        injected_sweep_wait = True
    if injected_sweep_wait or (
        _positive_float(run_env.get(V5_SEARCH_BUDGET_ENV)) is None
        and _positive_float(run_env.get(GENERIC_SEARCH_BUDGET_ENV)) is None
    ):
        run_env[V5_SEARCH_BUDGET_ENV] = PORTFOLIO_SWEEP_WAIT_SECONDS
    return run_env


def _attempt_on_cooldown(
    lead: str,
    state: Mapping[str, object],
    *,
    retry_hours: float,
    now: datetime,
) -> bool:
    if retry_hours <= 0:
        return False
    attempts = _attempted_leads(state)
    lead_key = _lead_key(lead)
    for raw_lead, raw_meta in attempts.items():
        if _lead_key(str(raw_lead)) != lead_key or not isinstance(raw_meta, Mapping):
            continue
        status = str(raw_meta.get("status") or "")
        if status in {"accepted", "ready", "blocked:search_backend_error"}:
            return False
        updated_at = _parse_state_time(raw_meta.get("updated_at"))
        if updated_at is None:
            return False
        return now - updated_at < timedelta(hours=retry_hours)
    return False


def _available_leads(
    leads: Sequence[str],
    state: Mapping[str, object],
    *,
    blocked_retry_hours: float,
    now: datetime,
) -> list[str]:
    completed_keys = _state_keys(_completed_leads(state))
    return [
        lead
        for lead in leads
        if _lead_key(lead) not in completed_keys
        and not _attempt_on_cooldown(lead, state, retry_hours=blocked_retry_hours, now=now)
    ]


def discover_leads(
    known_leads: Sequence[str],
    state: Mapping[str, object],
    *,
    count: int,
) -> list[str]:
    if count <= 0:
        return []
    known = {_lead_key(lead) for lead in known_leads}
    known.update(_state_keys(_completed_leads(state)))
    known.update(_state_keys(_attempted_leads(state)))
    out: list[str] = []
    for angle in DISCOVERY_ANGLES:
        for context in DISCOVERY_CONTEXTS:
            for intervention in DISCOVERY_INTERVENTIONS:
                lead = " ".join(part for part in (intervention, context, angle) if part)
                key = _lead_key(lead)
                if key in known:
                    continue
                known.add(key)
                out.append(lead)
                if len(out) >= count:
                    return out
    return out


def _append_leads(path: Path | None, leads: Sequence[str]) -> None:
    if path is None or not leads:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text() if path.exists() else ""
    prefix = "" if not existing or existing.endswith("\n") else "\n"
    path.write_text(f"{existing}{prefix}" + "\n".join(leads) + "\n")


def _save_attempted_lead(
    path: Path | None,
    state: dict[str, object],
    record: Mapping[str, object],
) -> None:
    if path is None:
        return
    raw = state.setdefault("attempted_leads", {})
    if not isinstance(raw, dict):
        raw = {}
        state["attempted_leads"] = raw
    raw[str(record["lead"])] = {
        "receipt_path": record["receipt_path"],
        "status": record["status"],
        "updated_at": _timestamp(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def _save_completed_lead(
    path: Path | None,
    state: dict[str, object],
    record: Mapping[str, object],
) -> None:
    if path is None:
        return
    raw = state.setdefault("completed_leads", {})
    if not isinstance(raw, dict):
        raw = {}
        state["completed_leads"] = raw
    raw[str(record["lead"])] = {
        "receipt_path": record["receipt_path"],
        "status": record["status"],
        "updated_at": _timestamp(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True))


def classify_run(returncode: int, receipt: Mapping[str, object], *, submit: bool) -> str:
    raw_decision = receipt.get("decision")
    if isinstance(raw_decision, Mapping):
        decision = str(raw_decision.get("decision") or "")
        if decision == "accept":
            return "accepted"
        if decision in {"reject", "revise"}:
            return f"decision:{decision}"
    if submit and returncode == 0 and any(key in receipt for key in ("submission", "submission_id", "id")):
        return "submitted"
    error = receipt.get("error")
    if error == "researka_submit_deferred":
        return "deferred"
    if error == "search_backend_error" and "coverage too narrow" in str(receipt.get("message") or "").casefold():
        return "warming:search_coverage"
    if error:
        return f"blocked:{error}"
    if returncode == 0:
        return "submitted" if submit else "ready"
    return "failed_no_receipt"


def _should_stop(status: str) -> bool:
    return status in {"accepted", "submitted", "ready", "deferred"} or status.startswith("warming:")


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
    run_env = _portfolio_run_env(config, env or _env_for_repo(repo_root))
    state = _load_state(config.state_path)
    now = datetime.now(UTC)
    expanded_leads = list(leads)
    initial_completed_keys = _state_keys(_completed_leads(state))
    available_leads = _available_leads(
        expanded_leads,
        state,
        blocked_retry_hours=config.blocked_retry_hours,
        now=now,
    )
    discovered: list[str] = []
    if config.auto_discover_leads and len(available_leads) < config.min_open_leads:
        needed = max(config.discover_count, config.min_open_leads - len(available_leads))
        discovered = discover_leads(expanded_leads, state, count=needed)
        _append_leads(config.lead_file, discovered)
        expanded_leads.extend(discovered)
        available_leads = _available_leads(
            expanded_leads,
            state,
            blocked_retry_hours=config.blocked_retry_hours,
            now=now,
        )
    lead_limit = config.max_leads if config.max_leads > 0 else len(available_leads)
    selected = list(available_leads[:lead_limit])
    skipped_completed_count = sum(
        1 for lead in expanded_leads if _lead_key(lead) in initial_completed_keys
    )
    skipped_recent_count = sum(
        1
        for lead in expanded_leads
        if _lead_key(lead) not in initial_completed_keys
        and _attempt_on_cooldown(
            lead,
            state,
            retry_hours=config.blocked_retry_hours,
            now=now,
        )
    )
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
        visibility_error = receipt.get("visibility_error")
        if visibility_error:
            record["visibility_error"] = visibility_error
        records.append(record)
        _save_attempted_lead(config.state_path, state, record)
        if _should_stop(status):
            if status == "accepted":
                _save_completed_lead(config.state_path, state, record)
            break

    summary = {
        "created_at": _timestamp(),
        "auto_discover_leads": config.auto_discover_leads,
        "available_leads": len(available_leads),
        "discovered_leads": discovered,
        "submit": config.submit,
        "skipped_completed_leads": skipped_completed_count,
        "skipped_recent_attempts": skipped_recent_count,
        "selected_leads": len(selected),
        "attempted_leads": len(records),
        "final_status": records[-1]["status"] if records else "no_new_leads",
        "records": records,
    }
    (config.output_dir / "portfolio.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    final_status = str(summary["final_status"])
    if final_status in {"accepted", "submitted", "ready", "no_new_leads"} or final_status.startswith("warming:"):
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
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--auto-discover-leads", action="store_true")
    parser.add_argument("--min-open-leads", type=int, default=0)
    parser.add_argument("--discover-count", type=int, default=20)
    parser.add_argument("--blocked-retry-hours", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    leads = load_leads(args)
    if not leads:
        raise SystemExit("at least one lead is required")
    if args.max_leads < 0:
        raise SystemExit("--max-leads must be >= 0")
    if args.min_open_leads < 0:
        raise SystemExit("--min-open-leads must be >= 0")
    if args.discover_count < 0:
        raise SystemExit("--discover-count must be >= 0")
    if args.blocked_retry_hours < 0:
        raise SystemExit("--blocked-retry-hours must be >= 0")
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
        state_path=args.state_path,
        lead_file=Path(args.lead_file) if args.lead_file else None,
        auto_discover_leads=args.auto_discover_leads,
        min_open_leads=args.min_open_leads,
        discover_count=args.discover_count,
        blocked_retry_hours=args.blocked_retry_hours,
    )
    return run_portfolio(leads, config)


if __name__ == "__main__":
    raise SystemExit(main())
