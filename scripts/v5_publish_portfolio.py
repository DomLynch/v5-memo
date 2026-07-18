#!/usr/bin/env python3
"""Try V5 publish leads until one clears the existing strict CLI gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib import parse, request

from v5_memo.__main__ import _alpha_shape_queries, _dedupe_queries
from v5_memo.client import _fullraw_search_passes
from v5_memo.fullraw_index import (
    _SWEEP_STRATEGY,
    SweepCacheEntry,
    _load_sweep_cache,
    _sweep_cache_entry_matches_active_or_completed_original_query,
)
from v5_memo.gate import (
    LEAD_PROPOSAL_SCHEMA,
    lead_proposal_fingerprint,
    lead_proposal_identity,
    lead_proposal_metadata_valid,
)
from v5_memo.miner import query_anchor_terms

TAIL_CHARS = 2000
STATE_TIME_FORMAT = "%Y%m%dT%H%M%SZ"
PORTFOLIO_SWEEP_WAIT_SECONDS = "21600"
V5_SWEEP_WAIT_ENV = "V5_MEMO_FULL_RAW_FOREGROUND_SWEEP_WAIT_SECONDS"
GENERIC_SWEEP_WAIT_ENV = "RESEARKA_FULLRAW_FOREGROUND_SWEEP_WAIT_SECONDS"
V5_SEARCH_BUDGET_ENV = "V5_MEMO_FULL_RAW_SEARCH_BUDGET_SECONDS"
GENERIC_SEARCH_BUDGET_ENV = "RESEARKA_FULLRAW_SEARCH_BUDGET_SECONDS"
V5_HEALTH_WAIT_ENV = "V5_MEMO_FULL_RAW_HEALTH_WAIT_SECONDS"
GENERIC_HEALTH_WAIT_ENV = "RESEARKA_FULLRAW_HEALTH_WAIT_SECONDS"
PORTFOLIO_HEALTH_WAIT_SECONDS = 300.0
CACHE_SCAN_LIMIT = 512
CACHE_DERIVED_SOURCE = "complete_sweep_cache"
CACHE_QUEUE_IF_MISSING_ENV = "V5_MEMO_FULL_RAW_QUEUE_IF_MISSING"
CACHE_PER_QUERY_LIMIT_ENV = "V5_MEMO_FULL_RAW_PER_QUERY_LIMIT"
FOCUS_LEASE_ENV = "V5_MEMO_FULL_RAW_FOCUS_LEASE"
PORTFOLIO_SEARCH_URL_ENV = "V5_MEMO_PORTFOLIO_FULL_RAW_CORPUS_SEARCH_URL"

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
    lead_timeout_seconds: float = 0.0
    ready_buffer_size: int = 0
    ready_only: bool = False
    resource_aware_max_leads: bool = False


@dataclass(frozen=True, slots=True)
class ReceiptLeadProposal:
    lead: str
    source_topic: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class CacheLeadProposal:
    lead: str
    cache_key: str
    fingerprint: str
    result_limit: int


@dataclass(frozen=True, slots=True)
class SweepCacheContext:
    cache_dir: Path
    catalog_scope: str
    shard_limit: int
    pass_shard_limit: int
    ttl_seconds: float


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
    return _dedupe(leads)


def build_command(
    lead: str,
    lead_dir: Path,
    receipt_path: Path,
    config: RunConfig,
    *,
    explicit_query: str | None = None,
) -> list[str]:
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
    if explicit_query:
        command.extend(["--query", explicit_query])
    if config.planner:
        command.extend(["--planner", config.planner])
    if config.writer:
        command.extend(["--writer", config.writer])
    if config.selector:
        command.extend(["--selector", config.selector])
    if not config.submit and config.ready_buffer_size > 0:
        command.append("--validate-publish-quality")
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


def _write_state(path: Path, state: Mapping[str, object]) -> None:
    """Replace state atomically so derived provenance cannot be partially saved."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _completed_leads(state: Mapping[str, object]) -> Mapping[str, object]:
    raw = state.get("completed_leads")
    return raw if isinstance(raw, Mapping) else {}


def _attempted_leads(state: Mapping[str, object]) -> Mapping[str, object]:
    raw = state.get("attempted_leads")
    return raw if isinstance(raw, Mapping) else {}


def _derived_leads(state: Mapping[str, object]) -> Mapping[str, object]:
    raw = state.get("derived_leads")
    return raw if isinstance(raw, Mapping) else {}


def _cache_derived_lead_meta(
    state: Mapping[str, object],
    lead: str,
) -> Mapping[str, object] | None:
    lead_key = _lead_key(lead)
    for raw_lead, raw_meta in _derived_leads(state).items():
        if (
            _lead_key(str(raw_lead)) == lead_key
            and isinstance(raw_meta, Mapping)
            and raw_meta.get("source") == CACHE_DERIVED_SOURCE
        ):
            return raw_meta
    return None


def _ready_lead_keys(state: Mapping[str, object]) -> set[str]:
    completed = _state_keys(_completed_leads(state))
    return {
        _lead_key(str(lead))
        for lead, meta in _attempted_leads(state).items()
        if isinstance(meta, Mapping)
        and str(meta.get("status") or "") == "ready"
        and _lead_key(str(lead)) not in completed
    }


def _state_keys(values: Mapping[str, object]) -> set[str]:
    return {_lead_key(str(value)) for value in values}


def _terminal_decision_lead_keys(state: Mapping[str, object]) -> set[str]:
    return {
        _lead_key(str(lead))
        for lead, meta in _attempted_leads(state).items()
        if isinstance(meta, Mapping)
        and str(meta.get("status") or "") in {"decision:reject", "decision:revise"}
    }


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


def _configured_env(env: Mapping[str, str], name: str) -> bool:
    return bool(str(env.get(name, "")).strip())


def _format_seconds(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def _portfolio_sweep_wait_seconds(config: RunConfig) -> str:
    wait_seconds = _positive_float(PORTFOLIO_SWEEP_WAIT_SECONDS) or 0.0
    if config.lead_timeout_seconds > 0:
        wait_seconds = min(wait_seconds, max(30.0, config.lead_timeout_seconds / 10.0))
    return _format_seconds(wait_seconds)


def _portfolio_run_env(config: RunConfig, base_env: Mapping[str, str]) -> dict[str, str]:
    run_env = dict(base_env)
    portfolio_search_url = str(run_env.get(PORTFOLIO_SEARCH_URL_ENV, "")).strip()
    if portfolio_search_url:
        # The shared platform keeps ownership of the canonical route variables.
        # A portfolio-only route is applied to this subprocess environment so a
        # saturated shared queue cannot starve V5 publication.
        run_env["RESEARKA_FULLRAW_SEARCH_URL"] = portfolio_search_url
        run_env["V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL"] = portfolio_search_url
    # Every portfolio command explicitly requires fullraw, including hybrid and
    # smart modes, so all of them need the same bounded readiness grace period.
    if not (
        _configured_env(run_env, V5_HEALTH_WAIT_ENV)
        or _configured_env(run_env, GENERIC_HEALTH_WAIT_ENV)
    ):
        wait_seconds = PORTFOLIO_HEALTH_WAIT_SECONDS
        if config.lead_timeout_seconds > 0:
            wait_seconds = min(
                wait_seconds,
                config.lead_timeout_seconds / 2.0,
            )
        run_env[V5_HEALTH_WAIT_ENV] = _format_seconds(wait_seconds)
    if config.searcher != "fullraw" or not (config.submit or config.ready_buffer_size > 0):
        return run_env
    if not _configured_env(run_env, V5_SWEEP_WAIT_ENV):
        bounded_wait = _positive_float(_portfolio_sweep_wait_seconds(config)) or 0.0
        generic_wait = _positive_float(run_env.get(GENERIC_SWEEP_WAIT_ENV))
        wait_seconds = min(generic_wait, bounded_wait) if generic_wait else bounded_wait
        run_env[V5_SWEEP_WAIT_ENV] = _format_seconds(wait_seconds)
    v5_wait = _positive_float(run_env.get(V5_SWEEP_WAIT_ENV))
    generic_budget = _positive_float(run_env.get(GENERIC_SEARCH_BUDGET_ENV))
    if (
        v5_wait is not None
        and not _configured_env(run_env, V5_SEARCH_BUDGET_ENV)
        and (generic_budget is None or generic_budget < v5_wait)
    ):
        run_env[V5_SEARCH_BUDGET_ENV] = _format_seconds(v5_wait)
    return run_env


def _fullraw_health_url(env: Mapping[str, str]) -> str:
    search_url = next(
        (
            str(env.get(name, "")).strip()
            for name in (
                "V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL",
                "RESEARKA_FULLRAW_SEARCH_URL",
            )
            if str(env.get(name, "")).strip()
        ),
        "",
    )
    if not search_url:
        return ""
    parsed = parse.urlsplit(search_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return parse.urlunsplit((parsed.scheme, parsed.netloc, "/health", "", ""))


def _fullraw_health_headers(env: Mapping[str, str]) -> dict[str, str]:
    headers = {"User-Agent": "v5-memo/0.1"}
    token = next(
        (
            str(env.get(name, "")).strip()
            for name in (
                "RESEARKA_FULLRAW_INDEX_TOKEN",
                "RESEARKA_FULLRAW_TOKEN",
                "V5_MEMO_FULL_RAW_INDEX_TOKEN",
                "V5_MEMO_FULL_RAW_CORPUS_TOKEN",
            )
            if str(env.get(name, "")).strip()
        ),
        "",
    )
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fullraw_max_inflight(env: Mapping[str, str]) -> int | None:
    health_url = _fullraw_health_url(env)
    if not health_url:
        return None
    try:
        health_request = request.Request(
            health_url,
            headers=_fullraw_health_headers(env),
            method="GET",
        )
        with request.urlopen(health_request, timeout=2.0) as response:
            payload = json.loads(response.read())
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if payload.get("ok") is not True:
        return None
    sweep = payload.get("async_sweep")
    if not isinstance(sweep, Mapping):
        return None
    raw_limit = sweep.get("max_inflight")
    return raw_limit if type(raw_limit) is int and raw_limit > 0 else None


def _preparation_lead_limit(
    config: RunConfig,
    env: Mapping[str, str],
) -> tuple[int, int | None]:
    if not config.resource_aware_max_leads or config.searcher != "fullraw":
        return config.max_leads, None
    probed_limit = _fullraw_max_inflight(env)
    resource_limit = probed_limit or 1
    if config.max_leads <= 0:
        return resource_limit, probed_limit
    return min(config.max_leads, resource_limit), probed_limit


def _attempt_on_cooldown(
    lead: str,
    state: Mapping[str, object],
    *,
    retry_hours: float,
    now: datetime,
    retry_post_quality: bool = False,
) -> bool:
    if retry_hours <= 0:
        return False
    attempts = _attempted_leads(state)
    lead_key = _lead_key(lead)
    for raw_lead, raw_meta in attempts.items():
        if _lead_key(str(raw_lead)) != lead_key or not isinstance(raw_meta, Mapping):
            continue
        status = str(raw_meta.get("status") or "")
        if retry_post_quality and _post_quality_status(status):
            return False
        if status.startswith("warming:") or status in {
            "accepted",
            "ready",
            "blocked:search_backend_error",
            "blocked:lead_timeout",
        }:
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
    prefer_ready: bool = False,
    retry_post_quality: bool = False,
    complete_cache_lead_keys: set[str] | None = None,
    warming_fingerprints: Mapping[str, str] | None = None,
) -> list[str]:
    completed_keys = _state_keys(_completed_leads(state))
    terminal_decision_keys = _terminal_decision_lead_keys(state)
    available = [
        lead
        for lead in leads
        if _lead_key(lead) not in completed_keys
        and _lead_key(lead) not in terminal_decision_keys
        and not _attempt_on_cooldown(
            lead,
            state,
            retry_hours=blocked_retry_hours,
            now=now,
            retry_post_quality=retry_post_quality,
        )
    ]
    warming_lease_key = (
        _warming_lease_key(available, state, warming_fingerprints)
        if warming_fingerprints is not None
        else None
    )
    return sorted(
        available,
        key=lambda lead: _attempt_priority(
            lead,
            state,
            prefer_ready=prefer_ready,
            complete_cache_lead_keys=complete_cache_lead_keys,
            warming_fingerprints=warming_fingerprints,
            warming_lease_key=warming_lease_key,
        ),
    )


def _receipt_remaining_shards(meta: Mapping[str, object]) -> int | None:
    raw_remaining = meta.get("sweep_remaining_shards")
    if isinstance(raw_remaining, int) and raw_remaining >= 0:
        return raw_remaining
    raw_path = meta.get("receipt_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    receipt = _read_json(Path(raw_path))
    details = receipt.get("details")
    detail_coverage = details.get("coverage") if isinstance(details, Mapping) else None
    for container in (
        receipt,
        details,
        detail_coverage,
        receipt.get("coverage"),
    ):
        if not isinstance(container, Mapping):
            continue
        raw_remaining = container.get("sweep_remaining_shards")
        if isinstance(raw_remaining, int) and raw_remaining >= 0:
            return raw_remaining
    message = str(receipt.get("message") or "")
    match = re.search(r"['\"]sweep_remaining_shards['\"]\s*:\s*(\d+)", message)
    return int(match.group(1)) if match else None


def _search_warming_status(status: str) -> bool:
    return status.startswith("warming:")


def _warming_lease_key(
    leads: Sequence[str],
    state: Mapping[str, object],
    warming_fingerprints: Mapping[str, str],
) -> str:
    eligible_keys = {_lead_key(lead) for lead in leads}
    exact: list[tuple[float, int, int, str]] = []
    legacy: list[tuple[float, int, int, str]] = []
    for order, (raw_lead, raw_meta) in enumerate(_attempted_leads(state).items()):
        if not isinstance(raw_meta, Mapping):
            continue
        status = str(raw_meta.get("status") or "")
        if not _search_warming_status(status):
            continue
        lead_key = _lead_key(str(raw_lead))
        if lead_key not in eligible_keys:
            continue
        current_fingerprint = warming_fingerprints.get(lead_key)
        if not current_fingerprint:
            continue
        updated_at = _parse_state_time(raw_meta.get("updated_at"))
        updated_rank = -updated_at.timestamp() if updated_at is not None else float("inf")
        remaining = _receipt_remaining_shards(raw_meta)
        rank = (
            updated_rank,
            remaining if remaining is not None else sys.maxsize,
            order,
            lead_key,
        )
        stored_fingerprint = raw_meta.get("warming_fingerprint")
        if stored_fingerprint == current_fingerprint:
            exact.append(rank)
        elif stored_fingerprint is None or stored_fingerprint == "":
            legacy.append(rank)
    candidates = exact or legacy
    return min(candidates)[-1] if candidates else ""


def _attempt_priority(
    lead: str,
    state: Mapping[str, object],
    *,
    prefer_ready: bool = False,
    complete_cache_lead_keys: set[str] | None = None,
    warming_fingerprints: Mapping[str, str] | None = None,
    warming_lease_key: str | None = None,
) -> tuple[int, int]:
    lead_key = _lead_key(lead)
    for raw_lead, raw_meta in _attempted_leads(state).items():
        if _lead_key(str(raw_lead)) != lead_key or not isinstance(raw_meta, Mapping):
            continue
        status = str(raw_meta.get("status") or "")
        if status == "ready":
            return (-1 if prefer_ready else 4, 0)
        if _post_quality_status(status):
            return (0, -1)
        if complete_cache_lead_keys and lead_key in complete_cache_lead_keys:
            return (0, 0)
        if _search_warming_status(status):
            remaining = _receipt_remaining_shards(raw_meta)
            if warming_fingerprints is None:
                return (0, remaining if remaining is not None else sys.maxsize)
            if lead_key != warming_lease_key:
                current_fingerprint = warming_fingerprints.get(lead_key)
                if (
                    current_fingerprint
                    and raw_meta.get("warming_fingerprint") == current_fingerprint
                ):
                    return (
                        1,
                        remaining if remaining is not None else sys.maxsize,
                    )
                return (3, 0)
            return (
                0,
                remaining + 1 if remaining is not None else sys.maxsize,
            )
        if status in {"blocked:lead_timeout", "blocked:researka_submit_failed"}:
            return (1, 0)
        if status.startswith("decision:"):
            return (3, 0)
        return (4, 0)
    if complete_cache_lead_keys and lead_key in complete_cache_lead_keys:
        return (0, 0)
    return (2, 0)


def _env_text(env: Mapping[str, str], *names: str) -> str:
    return next(
        (
            str(env.get(name, "")).strip()
            for name in names
            if str(env.get(name, "")).strip()
        ),
        "",
    )


def _consistent_env_text(env: Mapping[str, str], *names: str) -> str | None:
    values = [str(env.get(name, "")).strip() for name in names]
    configured = [value for value in values if value]
    if len(set(configured)) > 1:
        return None
    return configured[0] if configured else ""


def _positive_int_text(value: str | None) -> int | None:
    try:
        parsed = int(value or "")
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _positive_float_text(value: str | None) -> float | None:
    try:
        parsed = float(value or "")
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _positive_int_from_env(env: Mapping[str, str], *names: str) -> int | None:
    return _positive_int_text(_env_text(env, *names))


def _positive_float_from_env(env: Mapping[str, str], *names: str) -> float | None:
    return _positive_float_text(_env_text(env, *names))


def _sweep_cache_context(
    env: Mapping[str, str],
    *,
    planner: str | None,
) -> SweepCacheContext | None:
    if planner not in {None, "seed"}:
        return None
    cache_dir_text = _consistent_env_text(
        env,
        "RESEARKA_FULLRAW_SWEEP_CACHE_DIR",
        "V5_MEMO_FULL_RAW_SWEEP_CACHE_DIR",
    )
    shard_dir_text = _consistent_env_text(
        env,
        "RESEARKA_FULLRAW_SHARD_DIR",
        "V5_MEMO_FULL_RAW_SHARD_DIR",
    )
    raw_shard_limit = _consistent_env_text(
        env,
        "RESEARKA_FULLRAW_SWEEP_SHARD_LIMIT",
        "V5_MEMO_FULL_RAW_SWEEP_SHARD_LIMIT",
    )
    raw_pass_limit = _consistent_env_text(
        env,
        "RESEARKA_FULLRAW_SWEEP_PASS_SHARD_LIMIT",
        "V5_MEMO_FULL_RAW_SWEEP_PASS_SHARD_LIMIT",
    )
    raw_ttl = _consistent_env_text(
        env,
        "RESEARKA_FULLRAW_SWEEP_TTL_SECONDS",
        "V5_MEMO_FULL_RAW_SWEEP_TTL_SECONDS",
    )
    shard_limit = _positive_int_text(raw_shard_limit)
    if (
        not cache_dir_text
        or not shard_dir_text
        or shard_limit is None
        or raw_pass_limit is None
        or raw_ttl is None
    ):
        return None
    cache_dir = Path(cache_dir_text)
    if not cache_dir.is_dir():
        return None
    return SweepCacheContext(
        cache_dir=cache_dir,
        catalog_scope=str(Path(shard_dir_text).absolute()),
        shard_limit=shard_limit,
        pass_shard_limit=_positive_int_text(raw_pass_limit) or shard_limit,
        ttl_seconds=_positive_float_text(raw_ttl) or 86400.0,
    )


def _sweep_cache_paths(cache_dir: Path) -> list[Path]:
    try:
        with os.scandir(cache_dir) as entries:
            paths = sorted(
                (
                    Path(entry.path)
                    for entry in entries
                    if entry.is_file(follow_symlinks=False)
                    and entry.name.endswith(".json")
                ),
                key=lambda path: path.name,
            )
    except OSError:
        return []
    return paths[:CACHE_SCAN_LIMIT]


def _sweep_cache_source_query(receipt: Mapping[str, object]) -> str:
    original_query = str(receipt.get("sweep_original_query") or "").strip()
    if original_query:
        return original_query
    # The current cache writer omits sweep_original_query only when the active
    # pass query is identical to the job query, so sweep_query is exact here.
    return str(receipt.get("sweep_query") or "").strip()


def _strict_complete_sweep_cache_entry(
    entry: SweepCacheEntry,
    context: SweepCacheContext,
    *,
    require_source_query: bool,
) -> bool:
    receipt = entry.receipt
    source_query = _sweep_cache_source_query(receipt)
    return bool(
        len(entry.hits) >= 2
        and receipt.get("partial_shard_search") is False
        and receipt.get("shards_searched") == context.shard_limit
        and receipt.get("shards_total") == context.shard_limit
        and receipt.get("sweep_shard_limit") == context.shard_limit
        and receipt.get("sweep_pass_shard_limit") == context.pass_shard_limit
        and receipt.get("sweep_remaining_shards") == 0
        and receipt.get("sweep_failed_shards", 0) == 0
        and receipt.get("sweep_strategy") == _SWEEP_STRATEGY
        and receipt.get("sweep_catalog_scope") == context.catalog_scope
        and receipt.get("sweep_timed_out") is not True
        and receipt.get("sweep_stopped_no_hits") is not True
        and not receipt.get("sweep_deferred_paths")
        and (not require_source_query or source_query)
    )


def _strict_complete_sweep_cache_entries(
    context: SweepCacheContext,
    *,
    require_source_query: bool,
) -> list[tuple[Path, SweepCacheEntry]]:
    result: list[tuple[Path, SweepCacheEntry]] = []
    for path in _sweep_cache_paths(context.cache_dir):
        entry = _load_sweep_cache(path, ttl_seconds=context.ttl_seconds)
        if entry is not None and _strict_complete_sweep_cache_entry(
            entry,
            context,
            require_source_query=require_source_query,
        ):
            result.append((path, entry))
    return result


def _first_query_cache_specs(
    leads: Sequence[str],
    env: Mapping[str, str],
) -> dict[str, tuple[str, int]]:
    per_query_limit = _positive_int_from_env(
        env,
        "V5_MEMO_FULL_RAW_PER_QUERY_LIMIT",
        "V5_MEMO_FULL_RAW_RECALL_LIMIT",
    ) or 50
    specs: dict[str, tuple[str, int]] = {}
    for lead in leads:
        seed_queries = _dedupe_queries([lead, *_alpha_shape_queries(lead)])
        if not seed_queries:
            continue
        max_hits = _positive_int_from_env(
            env,
            "V5_MEMO_FULL_RAW_MAX_HITS",
        ) or per_query_limit * max(2, min(4, len(seed_queries)))
        result_limit = min(
            per_query_limit,
            max(1, -(-max_hits // len(seed_queries))),
        )
        passes = _fullraw_search_passes(seed_queries[0], limit=1)
        if passes:
            specs[_lead_key(lead)] = (passes[0].query, result_limit)
    return specs


def _warming_fingerprints(
    leads: Sequence[str],
    env: Mapping[str, str],
    *,
    planner: str | None,
) -> dict[str, str]:
    if planner not in {None, "seed"}:
        return {}
    search_url = _consistent_env_text(
        env,
        "RESEARKA_FULLRAW_SEARCH_URL",
        "V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL",
    )
    if not search_url:
        return {}
    backend = {
        "search_url": search_url.rstrip("/"),
        "shard_dir": _consistent_env_text(
            env,
            "RESEARKA_FULLRAW_SHARD_DIR",
            "V5_MEMO_FULL_RAW_SHARD_DIR",
        ),
        "sweep_cache_dir": _consistent_env_text(
            env,
            "RESEARKA_FULLRAW_SWEEP_CACHE_DIR",
            "V5_MEMO_FULL_RAW_SWEEP_CACHE_DIR",
        ),
        "sweep_shard_limit": _positive_int_text(
            _consistent_env_text(
                env,
                "RESEARKA_FULLRAW_SWEEP_SHARD_LIMIT",
                "V5_MEMO_FULL_RAW_SWEEP_SHARD_LIMIT",
            )
        ),
        "sweep_pass_shard_limit": _positive_int_text(
            _consistent_env_text(
                env,
                "RESEARKA_FULLRAW_SWEEP_PASS_SHARD_LIMIT",
                "V5_MEMO_FULL_RAW_SWEEP_PASS_SHARD_LIMIT",
            )
        ),
        "sweep_ttl_seconds": _positive_float_text(
            _consistent_env_text(
                env,
                "RESEARKA_FULLRAW_SWEEP_TTL_SECONDS",
                "V5_MEMO_FULL_RAW_SWEEP_TTL_SECONDS",
            )
        ),
        "sweep_strategy": _SWEEP_STRATEGY,
    }
    if any(value in {None, ""} for value in backend.values()):
        return {}
    out: dict[str, str] = {}
    for lead_key, (query, result_limit) in _first_query_cache_specs(leads, env).items():
        payload = {
            "backend": backend,
            "query": " ".join(query.casefold().split()),
            "result_limit": result_limit,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        out[lead_key] = hashlib.sha256(encoded).hexdigest()
    return out


def _complete_first_query_cache_lead_keys(
    leads: Sequence[str],
    env: Mapping[str, str],
    *,
    planner: str | None,
) -> set[str]:
    context = _sweep_cache_context(env, planner=planner)
    if context is None:
        return set()
    specs = _first_query_cache_specs(leads, env)
    if not specs:
        return set()

    matched: set[str] = set()
    for _, entry in _strict_complete_sweep_cache_entries(
        context,
        require_source_query=False,
    ):
        for lead_key, (query, result_limit) in specs.items():
            if lead_key in matched:
                continue
            if _sweep_cache_entry_matches_active_or_completed_original_query(
                entry,
                active_query=query,
                original_query=query,
                result_limit=result_limit,
                sweep_shard_limit=context.shard_limit,
                sweep_pass_shard_limit=context.pass_shard_limit,
                sweep_strategy=_SWEEP_STRATEGY,
                sweep_catalog_scope=context.catalog_scope,
            ):
                matched.add(lead_key)
    return matched


def _cache_relevance_score(
    candidate: str,
    known_leads: Sequence[str],
) -> tuple[int, int, int] | None:
    candidate_terms = {
        term for term in _lead_key(candidate).split() if len(term) >= 3
    }
    candidate_anchors = set(query_anchor_terms([candidate], limit=8))
    known_terms_and_anchors = [
        (
            {term for term in _lead_key(lead).split() if len(term) >= 3},
            set(query_anchor_terms([lead], limit=8)),
        )
        for lead in _dedupe(known_leads)
    ]
    known_terms_and_anchors = [
        (terms, anchors)
        for terms, anchors in known_terms_and_anchors
        if terms and anchors
    ]
    if (
        len(candidate_terms) < 2
        or not candidate_anchors
        or not known_terms_and_anchors
    ):
        return None
    scores: list[tuple[int, int, int]] = []
    for terms, anchors in known_terms_and_anchors:
        shared = candidate_terms & terms
        shared_anchors = candidate_anchors & anchors
        if len(shared) < 2 or not shared_anchors:
            continue
        overlap_ratio = 1000 * len(shared) // max(len(candidate_terms), len(terms))
        scores.append((len(shared_anchors), len(shared), overlap_ratio))
    return max(scores) if scores else None


def _cache_proposal_fingerprint(
    *,
    lead: str,
    cache_key: str,
    created_at: float,
    result_limit: int,
    context: SweepCacheContext,
) -> str:
    payload = {
        "cache_key": cache_key,
        "catalog_scope": context.catalog_scope,
        "created_at": created_at,
        "lead": _lead_key(lead),
        "pass_shard_limit": context.pass_shard_limit,
        "result_limit": result_limit,
        "shard_limit": context.shard_limit,
        "source": CACHE_DERIVED_SOURCE,
        "strategy": _SWEEP_STRATEGY,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _discover_complete_cache_lead_proposal(
    known_leads: Sequence[str],
    state: Mapping[str, object],
    env: Mapping[str, str],
    *,
    planner: str | None,
    count: int,
) -> CacheLeadProposal | None:
    if count <= 0:
        return None
    context = _sweep_cache_context(env, planner=planner)
    if context is None:
        return None
    known_keys = {_lead_key(lead) for lead in known_leads}
    known_keys.update(_state_keys(_completed_leads(state)))
    known_keys.update(_state_keys(_attempted_leads(state)))
    known_keys.update(_state_keys(_derived_leads(state)))
    consumed_fingerprints = {
        str(meta.get("proposal_fingerprint") or "")
        for meta in _derived_leads(state).values()
        if isinstance(meta, Mapping)
    }
    ranked: list[tuple[tuple[int, int, int], float, str, CacheLeadProposal]] = []
    for path, entry in _strict_complete_sweep_cache_entries(
        context,
        require_source_query=True,
    ):
        lead = " ".join(_sweep_cache_source_query(entry.receipt).split())
        lead_key = _lead_key(lead)
        if not lead_key or lead_key in known_keys:
            continue
        specs = _first_query_cache_specs([lead], env)
        spec = specs.get(lead_key)
        if spec is None:
            continue
        query, result_limit = spec
        if _lead_key(query) != lead_key:
            continue
        if not _sweep_cache_entry_matches_active_or_completed_original_query(
            entry,
            active_query=query,
            original_query=lead,
            result_limit=result_limit,
            sweep_shard_limit=context.shard_limit,
            sweep_pass_shard_limit=context.pass_shard_limit,
            sweep_strategy=_SWEEP_STRATEGY,
            sweep_catalog_scope=context.catalog_scope,
        ):
            continue
        relevance = _cache_relevance_score(lead, known_leads)
        if relevance is None:
            continue
        fingerprint = _cache_proposal_fingerprint(
            lead=lead,
            cache_key=path.name,
            created_at=entry.created_at,
            result_limit=result_limit,
            context=context,
        )
        if fingerprint in consumed_fingerprints:
            continue
        ranked.append((
            relevance,
            entry.created_at,
            lead_key,
            CacheLeadProposal(
                lead=lead,
                cache_key=path.name,
                fingerprint=fingerprint,
                result_limit=result_limit,
            ),
        ))
    ranked.sort(
        key=lambda item: (
            -item[0][0],
            -item[0][1],
            -item[0][2],
            -item[1],
            item[2],
            item[3].cache_key,
        )
    )
    return ranked[0][3] if ranked else None


def _post_quality_status(status: str) -> bool:
    return status in {
        "accepted_pending_publication",
        "deferred",
        "submitted",
    } or status.startswith("blocked:researka_")


def discover_leads(
    known_leads: Sequence[str],
    state: Mapping[str, object],
    *,
    count: int,
) -> list[str]:
    proposal = _discover_lead_proposal(known_leads, state, count=count)
    return [proposal.lead] if proposal is not None else []


def _discover_lead_proposal(
    known_leads: Sequence[str],
    state: Mapping[str, object],
    *,
    count: int,
) -> ReceiptLeadProposal | None:
    if count <= 0:
        return None
    known = {_lead_key(lead) for lead in known_leads}
    known.update(_state_keys(_completed_leads(state)))
    known.update(_state_keys(_attempted_leads(state)))
    derived_keys = _state_keys(_derived_leads(state))
    known.update(derived_keys)
    consumed_fingerprints = {
        str(meta.get("proposal_fingerprint") or "")
        for meta in _derived_leads(state).values()
        if isinstance(meta, Mapping)
    }
    attempts = sorted(
        _attempted_leads(state).items(),
        key=lambda item: _parse_state_time(
            item[1].get("updated_at") if isinstance(item[1], Mapping) else None
        )
        or datetime.min.replace(tzinfo=UTC),
        reverse=True,
    )
    for raw_lead, raw_meta in attempts:
        if not isinstance(raw_meta, Mapping):
            continue
        if (
            _lead_key(str(raw_lead)) in derived_keys
            or raw_meta.get("status") != "blocked:no_receipt_bound_alpha_candidate"
        ):
            continue
        receipt_path = raw_meta.get("receipt_path")
        if not isinstance(receipt_path, str) or not receipt_path:
            continue
        proposal = _receipt_lead_proposal(
            _read_json(Path(receipt_path)),
            expected_source_topic=str(raw_lead),
        )
        if (
            proposal is not None
            and _lead_key(proposal.lead) not in known
            and proposal.fingerprint not in consumed_fingerprints
        ):
            return proposal
    return None


def _proposal_strings(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return None
    if any(not isinstance(item, str) or not item or item != item.strip() for item in value):
        return None
    strings = tuple(value)
    return strings if len(strings) == len(set(strings)) else None


def _receipt_lead_proposal(
    receipt: Mapping[str, object],
    *,
    expected_source_topic: str,
) -> ReceiptLeadProposal | None:
    if receipt.get("error") != "no_receipt_bound_alpha_candidate":
        return None
    details = receipt.get("details")
    raw_proposal = details.get("lead_proposal") if isinstance(details, Mapping) else None
    if not isinstance(raw_proposal, Mapping):
        return None
    lead = str(raw_proposal.get("lead") or "").strip()
    source_topic = str(raw_proposal.get("source_topic") or "").strip()
    receipt_ids = _proposal_strings(raw_proposal.get("receipt_ids"))
    source_keys = _proposal_strings(raw_proposal.get("source_keys"))
    fingerprint = str(raw_proposal.get("proposal_fingerprint") or "").strip()
    candidate_score = raw_proposal.get("candidate_score")
    candidate_novelty = raw_proposal.get("candidate_novelty_score")
    candidate_tier = str(raw_proposal.get("candidate_tier") or "")
    source_blocker = str(raw_proposal.get("source_blocker") or "")
    if (
        raw_proposal.get("schema") != LEAD_PROPOSAL_SCHEMA
        or not lead
        or not source_topic
        or lead_proposal_identity(source_topic)
        != lead_proposal_identity(expected_source_topic)
        or lead_proposal_identity(lead) == lead_proposal_identity(source_topic)
        or receipt_ids is None
        or source_keys is None
        or len(receipt_ids) < 2
        or len(source_keys) < 2
        or type(candidate_score) is not int
        or type(candidate_novelty) is not int
        or not lead_proposal_metadata_valid(
            candidate_score=candidate_score,
            candidate_novelty_score=candidate_novelty,
            candidate_tier=candidate_tier,
            source_blocker=source_blocker,
        )
    ):
        return None
    expected = lead_proposal_fingerprint(
        schema=LEAD_PROPOSAL_SCHEMA,
        lead=lead,
        source_topic=source_topic,
        receipt_ids=receipt_ids,
        source_keys=source_keys,
        candidate_score=candidate_score,
        candidate_novelty_score=candidate_novelty,
        candidate_tier=candidate_tier,
        source_blocker=source_blocker,
    )
    if fingerprint != expected:
        return None
    return ReceiptLeadProposal(lead, source_topic, fingerprint)


def _derived_supply_open(state: Mapping[str, object]) -> bool:
    completed = _state_keys(_completed_leads(state))
    attempts = _attempted_leads(state)
    for raw_lead in _derived_leads(state):
        lead_key = _lead_key(str(raw_lead))
        if lead_key in completed:
            continue
        attempt = next(
            (
                meta
                for lead, meta in attempts.items()
                if _lead_key(str(lead)) == lead_key and isinstance(meta, Mapping)
            ),
            None,
        )
        if attempt is None:
            return True
        status = str(attempt.get("status") or "")
        if (
            _search_warming_status(status)
            or status in {"accepted", "ready", "blocked:lead_timeout", "blocked:search_backend_error"}
            or _post_quality_status(status)
        ):
            return True
    return False


def _save_derived_lead(
    path: Path | None,
    state: dict[str, object],
    proposal: ReceiptLeadProposal | CacheLeadProposal,
) -> None:
    raw = state.setdefault("derived_leads", {})
    if not isinstance(raw, dict):
        raw = {}
        state["derived_leads"] = raw
    entry: dict[str, object] = {
        "proposal_fingerprint": proposal.fingerprint,
        "updated_at": _timestamp(),
    }
    if isinstance(proposal, ReceiptLeadProposal):
        entry["source_topic"] = proposal.source_topic
    else:
        entry.update({
            "cache_key": proposal.cache_key,
            "cache_result_limit": proposal.result_limit,
            "source": CACHE_DERIVED_SOURCE,
        })
    raw[proposal.lead] = entry
    if path is not None:
        _write_state(path, state)


def _save_attempted_lead(
    path: Path | None,
    state: dict[str, object],
    record: Mapping[str, object],
    *,
    preserve_ready: bool = False,
) -> None:
    if path is None:
        return
    raw = state.setdefault("attempted_leads", {})
    if not isinstance(raw, dict):
        raw = {}
        state["attempted_leads"] = raw
    lead = str(record["lead"])
    previous = raw.get(lead)
    previous_meta = previous if isinstance(previous, Mapping) else {}
    record_status = str(record["status"])
    keep_ready = (
        preserve_ready
        and str(previous_meta.get("status") or "") == "ready"
        and _submission_status_is_retryable(record_status)
    )
    entry: dict[str, object] = {
        "receipt_path": record["receipt_path"],
        "status": "ready" if keep_ready else record_status,
        "updated_at": _timestamp(),
    }
    remaining = record.get("sweep_remaining_shards")
    if isinstance(remaining, int) and remaining >= 0:
        entry["sweep_remaining_shards"] = remaining
    warming_fingerprint = record.get("warming_fingerprint")
    if (
        _search_warming_status(record_status)
        and not keep_ready
        and isinstance(warming_fingerprint, str)
        and warming_fingerprint
    ):
        entry["warming_fingerprint"] = warming_fingerprint
    if keep_ready:
        entry["last_attempt_status"] = record_status
        entry["ready_receipt_path"] = previous_meta.get(
            "ready_receipt_path",
            previous_meta.get("receipt_path"),
        )
    revision = record.get("revision")
    if isinstance(revision, Mapping):
        entry["revision"] = dict(revision)
    raw[lead] = entry
    _write_state(path, state)


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
    _write_state(path, state)


def _listed_minted_publication(receipt: Mapping[str, object]) -> bool:
    if receipt.get("visibility_error"):
        return False
    raw_decision = receipt.get("decision")
    if not isinstance(raw_decision, Mapping):
        return False
    raw_publication = raw_decision.get("publication")
    if not isinstance(raw_publication, Mapping):
        return False
    publication_id = raw_publication.get("publication_id") or raw_publication.get("id")
    doi = raw_publication.get("doi")
    doi_status = raw_publication.get("doi_status") or raw_publication.get("doiStatus")
    raw_visibility = receipt.get("visibility")
    if not isinstance(raw_visibility, Mapping):
        return False
    visibility = (
        raw_visibility.get("public_visibility")
        or raw_visibility.get("publicVisibility")
        or raw_visibility.get("visibility")
    )
    listed = (
        raw_visibility.get("public_visible") is True
        or raw_visibility.get("publicVisible") is True
        or str(visibility or "").casefold() == "listed"
    )
    return bool(
        publication_id
        and isinstance(doi, str)
        and doi.strip()
        and str(doi_status or "").casefold() == "minted"
        and listed
    )


def _submission_status_is_retryable(status: str) -> bool:
    # Researka's duplicate response returns the original submission ID. Keeping
    # these leads ready makes the next unattended run poll that same submission
    # through DOI minting/listing instead of stranding it.
    return status.startswith("warming:") or (
        status in {
            "accepted_pending_publication",
            "blocked:search_backend_error",
            "blocked:lead_timeout",
            "deferred",
            "failed_no_receipt",
            "submitted",
        }
        or status.startswith("blocked:researka_")
    )


def _submission_id(receipt: Mapping[str, object]) -> str:
    for key in ("submission_id", "id"):
        raw_id = receipt.get(key)
        if isinstance(raw_id, str) and raw_id.strip():
            return raw_id.strip()
    raw_submission = receipt.get("submission")
    if isinstance(raw_submission, Mapping):
        for key in ("id", "submission_id"):
            raw_id = raw_submission.get(key)
            if isinstance(raw_id, str) and raw_id.strip():
                return raw_id.strip()
    return ""


def _revision_context(receipt: Mapping[str, object]) -> dict[str, object]:
    raw_decision = receipt.get("decision")
    if not isinstance(raw_decision, Mapping):
        return {}
    if str(raw_decision.get("decision") or "") != "revise":
        return {}

    raw_resubmission = raw_decision.get("resubmission")
    resubmission = raw_resubmission if isinstance(raw_resubmission, Mapping) else {}
    raw_parent_id = resubmission.get("parent_submission_id")
    parent_id = (
        raw_parent_id.strip()
        if isinstance(raw_parent_id, str) and raw_parent_id.strip()
        else _submission_id(receipt)
    )
    raw_required = raw_decision.get("required_revisions")
    required_revisions = (
        [
            revision.strip()
            for revision in raw_required
            if isinstance(revision, str) and revision.strip()
        ]
        if isinstance(raw_required, list)
        else []
    )
    context: dict[str, object] = {
        "required_revisions": required_revisions,
        "resubmission_allowed": resubmission.get("allowed") is True,
    }
    if parent_id:
        context["parent_submission_id"] = parent_id
    review_summary = raw_decision.get("review_summary")
    if isinstance(review_summary, str) and review_summary.strip():
        context["review_summary"] = review_summary.strip()
    return context


def write_noop_portfolio(
    output_dir: Path,
    *,
    status: str,
    state_path: Path | None = None,
) -> int:
    state = _load_state(state_path)
    ready_count = len(_ready_lead_keys(state))
    summary = {
        "created_at": _timestamp(),
        "attempted_leads": 0,
        "final_status": status,
        "ready_buffer_count_after": ready_count,
        "ready_buffer_count_before": ready_count,
        "records": [],
        "selected_leads": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "portfolio.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    return 0


def classify_run(returncode: int, receipt: Mapping[str, object], *, submit: bool) -> str:
    raw_decision = receipt.get("decision")
    if isinstance(raw_decision, Mapping):
        decision = str(raw_decision.get("decision") or "")
        if decision == "accept":
            return "accepted" if _listed_minted_publication(receipt) else "accepted_pending_publication"
        if decision in {"reject", "revise"}:
            return f"decision:{decision}"
    if submit and _submission_id(receipt):
        return "submitted"
    error = receipt.get("error")
    if error == "researka_submit_deferred":
        return "deferred"
    if error == "search_backend_error":
        if (
            receipt.get("stage") == "fullraw_preflight"
            and receipt.get("retryable") is True
        ):
            return "warming:search_backend_unavailable"
        if "coverage too narrow" in str(receipt.get("message") or "").casefold():
            return "warming:search_coverage"
    if error:
        return f"blocked:{error}"
    if returncode == 0 and not submit:
        if receipt.get("ready") is True and receipt.get("validation") == "publish_quality":
            return "ready"
        return "blocked:invalid_publish_quality_receipt"
    if returncode == 0:
        return "failed_no_receipt"
    return "failed_no_receipt"


def _should_stop(status: str) -> bool:
    return status in {
        "accepted",
        "accepted_pending_publication",
        "submitted",
        "ready",
        "deferred",
    }


def _portfolio_exit_code(final_status: str, *, preparing: bool) -> int:
    # A retryable coverage warm-up is healthy, but an unreachable fullraw
    # backend must fail the systemd run so the outage cannot look successful.
    if final_status == "warming:search_backend_unavailable":
        return 1
    if final_status in {
        "accepted",
        "accepted_pending_publication",
        "submitted",
        "ready",
        "ready_buffer_full",
        "ready_buffer_empty",
        "no_new_leads",
    } or final_status.startswith("warming:"):
        return 0
    if preparing and final_status in {
        "blocked:candidate_publish_blocker",
        "blocked:no_receipt_bound_alpha_candidate",
    }:
        return 0
    return 6 if final_status == "deferred" else 1


def _run(
    command: Sequence[str],
    env: Mapping[str, str],
    cwd: Path,
    *,
    timeout_seconds: float = 0.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds or None,
    )


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def _run_lead(
    runner: Runner,
    command: Sequence[str],
    env: Mapping[str, str],
    cwd: Path,
    receipt_path: Path,
    *,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    try:
        if runner is _run:
            return _run(command, env, cwd, timeout_seconds=timeout_seconds)
        return runner(command, env, cwd)
    except subprocess.TimeoutExpired as exc:
        message = f"lead timed out after {timeout_seconds:g}s"
        if not receipt_path.exists():
            receipt_path.write_text(json.dumps({
                "error": "lead_timeout",
                "message": message,
                "timeout_seconds": timeout_seconds,
            }))
        stderr = _timeout_text(exc.stderr)
        stderr = f"{stderr.rstrip()}\n{message}\n" if stderr else f"{message}\n"
        return subprocess.CompletedProcess(
            list(command),
            124,
            stdout=_timeout_text(exc.stdout),
            stderr=stderr,
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
    preparing = not config.submit and config.ready_buffer_size > 0
    expanded_leads = _dedupe([
        *leads,
        *(str(lead) for lead in _derived_leads(state)),
    ])
    initial_completed_keys = _state_keys(_completed_leads(state))
    complete_cache_lead_keys = (
        _complete_first_query_cache_lead_keys(
            expanded_leads,
            run_env,
            planner=config.planner,
        )
        if preparing and config.searcher == "fullraw"
        else set()
    )
    warming_fingerprints = (
        _warming_fingerprints(
            expanded_leads,
            run_env,
            planner=config.planner,
        )
        if preparing and config.searcher == "fullraw"
        else None
    ) or None
    available_leads = _available_leads(
        expanded_leads,
        state,
        blocked_retry_hours=config.blocked_retry_hours,
        now=now,
        prefer_ready=config.submit,
        retry_post_quality=preparing,
        complete_cache_lead_keys=complete_cache_lead_keys,
        warming_fingerprints=warming_fingerprints,
    )
    eligible_keys = {_lead_key(lead) for lead in expanded_leads}
    ready_keys = _ready_lead_keys(state) & eligible_keys
    if config.ready_only:
        available_leads = [
            lead for lead in available_leads if _lead_key(lead) in ready_keys
        ]
    attempted_keys = _state_keys(_attempted_leads(state))
    discovered: list[str] = []
    if (
        config.auto_discover_leads
        and preparing
        and not config.ready_only
        and len(ready_keys) < config.ready_buffer_size
    ):
        proposal: ReceiptLeadProposal | CacheLeadProposal | None = None
        if not any(
            _lead_key(lead) in complete_cache_lead_keys
            and _cache_derived_lead_meta(state, lead) is not None
            for lead in available_leads
        ):
            proposal = _discover_complete_cache_lead_proposal(
                expanded_leads,
                state,
                run_env,
                planner=config.planner,
                count=min(1, config.discover_count),
            )
        if (
            proposal is None
            and not _derived_supply_open(state)
            and not any(
                _lead_key(lead) not in attempted_keys for lead in available_leads
            )
        ):
            proposal = _discover_lead_proposal(
                expanded_leads,
                state,
                count=min(1, config.discover_count),
            )
        discovered = [proposal.lead] if proposal is not None else []
        if proposal is not None:
            _save_derived_lead(config.state_path, state, proposal)
        expanded_leads = _dedupe([*discovered, *expanded_leads])
        complete_cache_lead_keys = (
            _complete_first_query_cache_lead_keys(
                expanded_leads,
                run_env,
                planner=config.planner,
            )
            if preparing and config.searcher == "fullraw"
            else set()
        )
        warming_fingerprints = (
            _warming_fingerprints(
                expanded_leads,
                run_env,
                planner=config.planner,
            )
            if preparing and config.searcher == "fullraw"
            else None
        ) or None
        available_leads = _available_leads(
            expanded_leads,
            state,
            blocked_retry_hours=config.blocked_retry_hours,
            now=now,
            prefer_ready=config.submit,
            retry_post_quality=preparing,
            complete_cache_lead_keys=complete_cache_lead_keys,
            warming_fingerprints=warming_fingerprints,
        )
        eligible_keys = {_lead_key(lead) for lead in expanded_leads}
        ready_keys = _ready_lead_keys(state) & eligible_keys
    ready_before = len(ready_keys)
    if preparing:
        available_leads = [
            lead for lead in available_leads if _lead_key(lead) not in ready_keys
        ]
    warming_lease_lead_key = (
        _warming_lease_key(available_leads, state, warming_fingerprints)
        if warming_fingerprints is not None
        else ""
    )
    if preparing:
        configured_lead_limit, resource_max_inflight = _preparation_lead_limit(
            config,
            run_env,
        )
    else:
        configured_lead_limit, resource_max_inflight = config.max_leads, None
    lead_limit = configured_lead_limit if configured_lead_limit > 0 else len(available_leads)
    if preparing:
        lead_limit = min(lead_limit, max(0, config.ready_buffer_size - ready_before))
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
            retry_post_quality=preparing,
        )
    )
    records: list[dict[str, object]] = []
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for index, lead in enumerate(selected, start=1):
        lead_dir = config.output_dir / f"{index:02d}-{_slug(lead)}"
        receipt_path = lead_dir / "publish-receipt.json"
        lead_dir.mkdir(parents=True, exist_ok=True)
        cache_meta = _cache_derived_lead_meta(state, lead)
        lead_env = dict(run_env)
        lead_env[FOCUS_LEASE_ENV] = (
            "1" if _lead_key(lead) == warming_lease_lead_key else "0"
        )
        explicit_query = None
        if cache_meta is not None:
            lead_env[CACHE_QUEUE_IF_MISSING_ENV] = "0"
            cache_result_limit = cache_meta.get("cache_result_limit")
            if type(cache_result_limit) is int and cache_result_limit > 0:
                lead_env[CACHE_PER_QUERY_LIMIT_ENV] = str(cache_result_limit)
            explicit_query = lead
        command = build_command(
            lead,
            lead_dir,
            receipt_path,
            config,
            explicit_query=explicit_query,
        )
        completed = _run_lead(
            runner,
            command,
            lead_env,
            repo_root,
            receipt_path,
            timeout_seconds=config.lead_timeout_seconds,
        )
        (lead_dir / "stdout.txt").write_text(completed.stdout or "")
        (lead_dir / "stderr.txt").write_text(completed.stderr or "")
        receipt = _read_json(receipt_path)
        status = classify_run(completed.returncode, receipt, submit=config.submit)
        record = {
            "lead": lead,
            "returncode": completed.returncode,
            "status": status,
            "publish_quality_validation": "--validate-publish-quality" in command,
            "receipt_path": str(receipt_path),
            "stdout_tail": (completed.stdout or "")[-TAIL_CHARS:],
            "stderr_tail": (completed.stderr or "")[-TAIL_CHARS:],
        }
        if cache_meta is not None:
            record.update({
                "cache_derived": True,
                "queue_if_missing": False,
            })
        error = receipt.get("error")
        if error:
            record["error"] = error
        remaining = _receipt_remaining_shards({"receipt_path": str(receipt_path)})
        if remaining is not None:
            record["sweep_remaining_shards"] = remaining
        current_warming_fingerprint = (
            warming_fingerprints.get(_lead_key(lead))
            if warming_fingerprints is not None
            else None
        )
        if _search_warming_status(status) and current_warming_fingerprint:
            record["warming_fingerprint"] = current_warming_fingerprint
        visibility_error = receipt.get("visibility_error")
        if visibility_error:
            record["visibility_error"] = visibility_error
        revision = _revision_context(receipt)
        if revision:
            record["revision"] = revision
        records.append(record)
        _save_attempted_lead(
            config.state_path,
            state,
            record,
            preserve_ready=config.submit,
        )
        if _should_stop(status) and not preparing:
            if status == "accepted":
                _save_completed_lead(config.state_path, state, record)
            break

    ready_after = len(_ready_lead_keys(state) & eligible_keys)
    no_attempt_status = (
        "ready_buffer_full"
        if preparing and ready_before >= config.ready_buffer_size
        else "ready_buffer_empty"
        if config.ready_only
        else "no_new_leads"
    )

    summary = {
        "created_at": _timestamp(),
        "auto_discover_leads": config.auto_discover_leads,
        "available_leads": len(available_leads),
        "discovered_leads": discovered,
        "submit": config.submit,
        "ready_only": config.ready_only,
        "preparing": preparing,
        "publish_quality_validation": preparing,
        "resource_aware_max_leads": config.resource_aware_max_leads,
        "resource_max_inflight": resource_max_inflight,
        "resource_limit_fallback": bool(
            preparing
            and config.resource_aware_max_leads
            and config.searcher == "fullraw"
            and resource_max_inflight is None
        ),
        "effective_max_leads": lead_limit,
        "complete_cache_priority_leads": [
            lead
            for lead in available_leads
            if _lead_key(lead) in complete_cache_lead_keys
        ],
        "cache_derived_selected_leads": [
            lead for lead in selected if _cache_derived_lead_meta(state, lead) is not None
        ],
        "warming_lease_lead": next(
            (
                lead
                for lead in available_leads
                if _lead_key(lead) == warming_lease_lead_key
            ),
            None,
        ),
        "ready_buffer_size": config.ready_buffer_size,
        "ready_buffer_count_before": ready_before,
        "ready_buffer_count_after": ready_after,
        "skipped_completed_leads": skipped_completed_count,
        "skipped_recent_attempts": skipped_recent_count,
        "selected_leads": len(selected),
        "attempted_leads": len(records),
        "final_status": records[-1]["status"] if records else no_attempt_status,
        "records": records,
    }
    (config.output_dir / "portfolio.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return _portfolio_exit_code(str(summary["final_status"]), preparing=preparing)


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
    parser.add_argument("--decision-wait-seconds", type=float, default=600.0)
    parser.add_argument("--decision-poll-seconds", type=float, default=5.0)
    parser.add_argument("--submit-wait-seconds", type=float, default=0.0)
    parser.add_argument("--max-leads", type=int, default=0)
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--auto-discover-leads", action="store_true")
    parser.add_argument("--min-open-leads", type=int, default=0)
    parser.add_argument("--discover-count", type=int, default=20)
    parser.add_argument("--blocked-retry-hours", type=float, default=0.0)
    parser.add_argument("--lead-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--ready-buffer-size", type=int, default=0)
    parser.add_argument("--ready-only", action="store_true")
    parser.add_argument("--resource-aware-max-leads", action="store_true")
    parser.add_argument("--record-noop-status", choices=["lock_busy"])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.record_noop_status:
        return write_noop_portfolio(
            args.output_dir,
            status=args.record_noop_status,
            state_path=args.state_path,
        )
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
    if args.lead_timeout_seconds < 0:
        raise SystemExit("--lead-timeout-seconds must be >= 0")
    if args.ready_buffer_size < 0:
        raise SystemExit("--ready-buffer-size must be >= 0")
    if args.ready_only and not args.submit:
        raise SystemExit("--ready-only requires --submit")
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
        lead_timeout_seconds=args.lead_timeout_seconds,
        ready_buffer_size=args.ready_buffer_size,
        ready_only=args.ready_only,
        resource_aware_max_leads=args.resource_aware_max_leads,
    )
    return run_portfolio(leads, config)


if __name__ == "__main__":
    raise SystemExit(main())
