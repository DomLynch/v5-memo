#!/usr/bin/env python3
"""Run one exhaustive fullraw sweep and persist the normal sweep cache."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from v5_memo.fullraw_index import (
    _SWEEP_STRATEGY,
    SweepCacheEntry,
    _add_planned_sweep_receipt,
    _int_or_none,
    _load_sweep_cache,
    _merge_hit_groups_with_receipt,
    _positive_int_env,
    _prioritize_sweep_pass_entries,
    _search_shard_paths_with_paths_and_receipt,
    _sweep_cache_key,
    _sweep_cache_path,
    _sweep_completed_path_strings,
    _sweep_failed_path_strings_for_mode,
    _sweep_remaining_shard_count,
    _sweep_search_passes,
    _write_sweep_cache,
    load_shard_catalog_cache,
    select_sweep_shard_entries,
    shard_coverage_receipt,
)


def _json_event(**payload: object) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _install_signal_logging() -> None:
    def handle_signal(signum: int, _frame: object) -> None:
        _json_event(event="signal", signal=signal.Signals(signum).name)
        raise SystemExit(128 + signum)

    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, handle_signal)


def main() -> int:
    _install_signal_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=os.environ.get("QUERY", "").strip())
    parser.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", "5")))
    parser.add_argument("--rank-mode", default=os.environ.get("RANK_MODE", "relevance"))
    parser.add_argument("--year-min", type=int, default=int(os.environ.get("YEAR_MIN", "1900")))
    parser.add_argument("--year-max", type=int, default=int(os.environ.get("YEAR_MAX", "2100")))
    args = parser.parse_args()

    if not args.query:
        raise SystemExit("--query or QUERY is required")

    catalog_path = Path(os.environ["V5_MEMO_FULL_RAW_SHARD_CATALOG_PATH"])
    cache_dir = Path(os.environ["V5_MEMO_FULL_RAW_SWEEP_CACHE_DIR"])
    catalog = load_shard_catalog_cache(catalog_path)
    if catalog is None:
        raise SystemExit(f"catalog cache unreadable: {catalog_path}")

    sweep_shard_limit = _positive_int_env("V5_MEMO_FULL_RAW_SWEEP_SHARD_LIMIT") or len(catalog)
    sweep_shard_limit = min(sweep_shard_limit, len(catalog))
    workers = _positive_int_env("V5_MEMO_FULL_RAW_SWEEP_WORKERS") or 8
    pass_shard_limit = _positive_int_env("V5_MEMO_FULL_RAW_SWEEP_PASS_SHARD_LIMIT") or max(workers * 4, 50)
    pass_shard_limit = max(1, min(pass_shard_limit, sweep_shard_limit))
    max_passes = _positive_int_env("V5_MEMO_FULL_RAW_SWEEP_MAX_PASSES") or sweep_shard_limit
    timeout_seconds = _env_float("V5_MEMO_FULL_RAW_SWEEP_TIMEOUT_SECONDS", 900.0)
    shard_timeout_seconds = _env_float("V5_MEMO_FULL_RAW_SWEEP_SHARD_TIMEOUT_SECONDS", 60.0)

    cache_key = _sweep_cache_key(
        args.query,
        limit=args.limit,
        year_min=args.year_min,
        year_max=args.year_max,
        rank_mode=args.rank_mode,
        sweep_shard_limit=sweep_shard_limit,
        sweep_strategy=_SWEEP_STRATEGY,
    )
    cache_path = _sweep_cache_path(cache_dir, cache_key)
    assert cache_path is not None
    existing = _load_sweep_cache(cache_path, ttl_seconds=0) if cache_path.exists() else None

    selected = select_sweep_shard_entries(catalog, query=args.query, limit=sweep_shard_limit)
    selected = _prioritize_sweep_pass_entries(selected, pass_shard_limit, query=args.query)
    planned_receipt = shard_coverage_receipt(catalog, selected)
    sweep_passes = _sweep_search_passes(args.query, selected, rank_mode=args.rank_mode)
    if not sweep_passes:
        raise SystemExit("no sweep passes generated")

    completed_path_strings = _sweep_completed_path_strings(existing.receipt if existing else {})
    failed_path_strings = _sweep_failed_path_strings_for_mode(
        existing.receipt if existing else {},
        require_complete_sweep=True,
    )
    merged_hits = list(existing.hits if existing else [])
    previous_passes = 0
    completed_pass_roles: list[str] = []
    if existing is not None:
        previous_passes = _int_or_none(existing.receipt.get("sweep_passes")) or 0
        raw_roles = existing.receipt.get("sweep_completed_pass_roles") or ()
        if isinstance(raw_roles, list | tuple):
            completed_pass_roles = [str(role) for role in raw_roles if str(role)]

    _json_event(
        cache_key=cache_key,
        completed=len(completed_path_strings),
        event="start",
        pass_shards=pass_shard_limit,
        query=args.query,
        selected=len(selected),
        workers=workers,
    )

    no_progress_passes = 0
    try:
        for pass_offset in range(max_passes):
            remaining_entries = [
                entry
                for entry in selected
                if str(entry.path) not in completed_path_strings | failed_path_strings
            ]
            if not remaining_entries:
                break
            pass_entries = remaining_entries[:pass_shard_limit]
            pass_plan = sweep_passes[(previous_passes + pass_offset) % len(sweep_passes)]
            started = time.monotonic()
            hits, completed_paths, timed_out, pass_metrics = _search_shard_paths_with_paths_and_receipt(
                [entry.path for entry in pass_entries],
                pass_plan.query,
                limit=args.limit,
                year_min=args.year_min,
                year_max=args.year_max,
                rank_mode=pass_plan.rank_mode,
                workers=workers,
                timeout_seconds=timeout_seconds,
                shard_timeout_seconds=shard_timeout_seconds,
            )
            before_completed = len(completed_path_strings)
            completed_path_strings.update(str(path) for path in completed_paths)
            progress = len(completed_path_strings) - before_completed
            no_progress_passes = no_progress_passes + 1 if progress <= 0 else 0
            completed_pass_roles.append(pass_plan.role)
            searched_entries = [
                entry for entry in selected if str(entry.path) in completed_path_strings
            ]
            receipt = shard_coverage_receipt(catalog, searched_entries)
            _add_planned_sweep_receipt(receipt, planned_receipt)
            receipt.update(
                {
                    "sweep_scope": "relevant",
                    "sweep_shard_limit": sweep_shard_limit,
                    "sweep_selected_shards": len(selected),
                    "sweep_pass_shard_limit": pass_shard_limit,
                    "sweep_pass_selected_shards": len(pass_entries),
                    "sweep_max_passes": max_passes,
                    "sweep_failed_shards": len(failed_path_strings),
                    "sweep_failed_paths": sorted(failed_path_strings),
                    "sweep_remaining_shards": _sweep_remaining_shard_count(
                        selected_shards=len(selected),
                        completed_shards=len(completed_path_strings),
                        failed_shards=len(failed_path_strings),
                        require_complete_sweep=True,
                    ),
                    "sweep_timed_out": timed_out,
                    "sweep_timeout_seconds": timeout_seconds,
                    "sweep_shard_timeout_seconds": shard_timeout_seconds,
                    "sweep_strategy": _SWEEP_STRATEGY,
                    "sweep_search_passes": tuple(asdict(pass_item) for pass_item in sweep_passes),
                    "sweep_completed_pass_roles": tuple(completed_pass_roles),
                    "sweep_pass_role": pass_plan.role,
                    "sweep_pass_query": pass_plan.query,
                    "sweep_pass_rank_mode": pass_plan.rank_mode,
                    "sweep_pass_result_metrics": pass_metrics,
                    "sweep_query": pass_plan.query,
                    "sweep_passes": previous_passes + pass_offset + 1,
                    "sweep_completed_paths": sorted(completed_path_strings),
                }
            )
            if pass_plan.query != args.query:
                receipt["sweep_original_query"] = args.query
            merged_hits, result_metrics = _merge_hit_groups_with_receipt(
                [merged_hits, hits],
                limit=args.limit,
            )
            receipt.update(result_metrics)
            _write_sweep_cache(cache_path, SweepCacheEntry(time.time(), merged_hits, receipt))
            _json_event(
                completed=len(completed_path_strings),
                elapsed=round(time.monotonic() - started, 3),
                event="pass_done",
                hits=len(merged_hits),
                pass_number=previous_passes + pass_offset + 1,
                progress=progress,
                remaining=receipt["sweep_remaining_shards"],
                role=pass_plan.role,
                timed_out=timed_out,
            )
            if receipt["sweep_remaining_shards"] == 0:
                break
            if no_progress_passes >= 3:
                _json_event(event="no_progress", remaining=receipt["sweep_remaining_shards"])
                return 2
    except Exception as exc:  # pragma: no cover - operational safety net
        _json_event(event="error", error=str(exc), traceback=traceback.format_exc())
        return 1

    final = _load_sweep_cache(cache_path, ttl_seconds=0)
    receipt = final.receipt if final else {}
    _json_event(
        event="done",
        hits=len(final.hits if final else []),
        remaining=receipt.get("sweep_remaining_shards"),
        shards_searched=receipt.get("shards_searched"),
        shards_total=receipt.get("shards_total"),
        partial=receipt.get("partial_shard_search"),
    )
    return 0 if receipt.get("sweep_remaining_shards") == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
