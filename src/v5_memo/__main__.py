"""CLI for offline demo or live full-corpus memo generation."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError

from v5_memo.client import (
    FullRawCorpusSearchClient,
    HybridCorpusSearchClient,
    OpenAlexFullCorpusSearchClient,
    ResearkaSearchClient,
    SearchBackendError,
)
from v5_memo.coverage import current_search_coverage, require_full_raw_corpus
from v5_memo.gate import candidate_publish_blocker
from v5_memo.miner import query_anchor_terms
from v5_memo.minimax_writer import (
    MiniMaxM3CandidateSelector,
    MiniMaxM3MemoWriter,
    MiniMaxM3SearchPlanner,
)
from v5_memo.pipeline import build_alpha_memo
from v5_memo.publisher import (
    build_researka_payload,
    load_researka_submit_config,
    researka_publication_id,
    researka_submission_id,
    set_researka_public_visibility,
    submit_researka,
    wait_researka_decision,
)
from v5_memo.retriever import CorpusSearcher, _seed_query_key
from v5_memo.schemas import CorpusHit, MemoBuildError, MemoResult
from v5_memo.writer import render_memo

_TOPIC_TERM_RE = re.compile(r"[a-z][a-z0-9]{2,}")
_TOPIC_FILTER_DROP = frozenset(
    (  # noqa: SIM905
        "adaptation adaptations adult adults aging effect effects evidence healthspan human "
        "humans intervention longevity mechanism mechanisms older outcome outcomes pharmacology "
        "response responses reversal study studies supplement supplementation trial trials"
    ).split()
)
_SHAPE_CONTEXT_TERMS = frozenset({"exercise", "resistance", "strength", "training"})
_ALPHA_QUERY_TERMS = frozenset({
    "activate", "activates", "activated", "augment", "augments", "augmented",
    "blunted", "blunts", "designed", "expected", "impair", "impaired", "impairs",
    "failed", "failure", "mimic", "mimics",
    "null", "observed", "placebo", "primary", "endpoint", "protocol", "randomized",
    "reduced", "reduces", "replication", "subgroup",
    "attenuate", "attenuated", "attenuates", "unchanged",
})
_NEGATIVE_ALPHA_QUERY_TERMS = frozenset({
    "attenuate", "attenuated", "attenuates", "blunted", "blunts", "failed",
    "failure", "impair", "impaired", "impairs", "null", "reduced", "reduces",
    "unchanged",
})
_DIRECT_EVIDENCE_QUERY_TERMS = frozenset({
    "adult", "adults", "clinical", "cohort", "human", "humans", "older",
    "patient", "patients", "randomized", "trial",
})
_MODEL_ONLY_QUERY_TERMS = frozenset({"germ", "mice", "mouse", "murine"})
_UNSAFE_DOI_CHARS = frozenset("()")
_DEFAULT_FULLRAW_RECALL_LIMIT = 25


class DemoSearch:
    def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
        del limit
        hits = {
            "sleep": CorpusHit(
                hit_id="demo-sleep",
                title="NAD salvage links sleep fragmentation to mitochondrial stress",
                abstract=(
                    "Sleep fragmentation reduced resilience through NAD salvage "
                    "and mitochondrial stress."
                ),
                source="demo",
                year=2025,
                doi="10.demo/sleep-nad",
                venue="Aging Cell",
            ),
            "exercise": CorpusHit(
                hit_id="demo-exercise",
                title="NAD salvage predicts exercise response through mitochondrial repair",
                abstract=(
                    "Exercise improved resilience when NAD salvage and mitochondrial "
                    "repair markers moved together."
                ),
                source="demo",
                year=2024,
                doi="10.demo/exercise-nad",
                venue="Cell Metabolism",
            ),
        }
        return [hit for key, hit in hits.items() if key in query.casefold()] or list(hits.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--topic", default="longevity resilience")
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--coverage-report", action="store_true")
    parser.add_argument("--require-full-raw-corpus", action="store_true")
    parser.add_argument("--planner", choices=["seed", "minimax"])
    parser.add_argument("--planner-limit", type=int, default=4)
    parser.add_argument(
        "--searcher",
        choices=["openalex", "researka", "fullraw", "hybrid", "smart"],
        default="openalex",
    )
    parser.add_argument("--writer", choices=["template", "minimax"])
    parser.add_argument("--selector", choices=["deterministic", "minimax"])
    parser.add_argument("--min-alpha-tier", choices=["discovery", "publishable", "elite"])
    parser.add_argument("--min-shards-searched", type=int)
    parser.add_argument("--min-sources-searched", type=int)
    parser.add_argument("--min-search-passes", type=int)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--emit-discovery-on-fail", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--submit-researka", action="store_true")
    parser.add_argument("--publish-receipt-path", default="")
    parser.add_argument(
        "--researka-decision-wait-seconds",
        type=float,
        default=float(os.environ.get("V5_MEMO_RESEARKA_DECISION_WAIT_SECONDS", "0") or 0),
    )
    parser.add_argument(
        "--researka-decision-poll-seconds",
        type=float,
        default=float(os.environ.get("V5_MEMO_RESEARKA_DECISION_POLL_SECONDS", "5") or 5),
    )
    parser.add_argument("--researka-list-if-accepted", action="store_true")
    parser.add_argument("--researka-agent-id", default=os.environ.get("V5_MEMO_RESEARKA_AGENT_ID", ""))
    parser.add_argument("--researka-domain-slug", default=os.environ.get("V5_MEMO_RESEARKA_DOMAIN_SLUG", ""))
    parser.add_argument(
        "--researka-parent-submission-id",
        default=os.environ.get("V5_MEMO_RESEARKA_PARENT_SUBMISSION_ID", ""),
    )
    parser.add_argument("--researka-api-base", default=os.environ.get("V5_MEMO_RESEARKA_API_BASE", "https://api.researka.org"))
    parser.add_argument("--researka-submit-url", default=os.environ.get("V5_MEMO_RESEARKA_SUBMIT_URL", ""))
    parser.add_argument(
        "--researka-submit-wait-seconds",
        type=float,
        default=float(os.environ.get("V5_MEMO_RESEARKA_SUBMIT_WAIT_SECONDS", "0") or 0),
    )
    args = parser.parse_args()
    fullraw_backed = args.searcher in {"fullraw", "hybrid", "smart"}
    args.min_shards_searched = _coverage_threshold(
        args.min_shards_searched,
        primary="V5_MEMO_MEMO_MIN_SHARDS_SEARCHED",
        fallback="V5_MEMO_FULL_RAW_MIN_SHARDS_SEARCHED",
        allow_fallback=fullraw_backed,
    )
    args.min_sources_searched = _coverage_threshold(
        args.min_sources_searched,
        primary="V5_MEMO_MEMO_MIN_SOURCES_SEARCHED",
        fallback="V5_MEMO_FULL_RAW_MIN_SOURCES_SEARCHED",
        allow_fallback=fullraw_backed,
    )
    args.min_search_passes = _coverage_threshold(
        args.min_search_passes,
        primary="V5_MEMO_MEMO_MIN_SEARCH_PASSES",
    )

    if args.coverage_report:
        print(current_search_coverage().summary)
        return
    if args.require_full_raw_corpus:
        _require_full_raw_or_exit()

    searcher_mode = "hybrid" if args.searcher == "smart" else args.searcher
    planner_mode = args.planner or ("minimax" if args.searcher == "smart" else "seed")
    writer_mode = args.writer or ("minimax" if args.searcher == "smart" else "template")
    selector_mode = args.selector or (
        "deterministic" if args.submit_researka or args.publish
        else "minimax" if writer_mode == "minimax"
        else "deterministic"
    )
    alpha_tier = args.min_alpha_tier or "publishable"
    min_alpha_tier = "discovery_seed" if alpha_tier == "discovery" else f"{alpha_tier}_alpha"

    searcher: CorpusSearcher
    if args.demo:
        searcher = DemoSearch()
    elif searcher_mode == "fullraw":
        _require_full_raw_or_exit()
        searcher = FullRawCorpusSearchClient.from_env(strict=True)
    elif searcher_mode == "researka":
        searcher = ResearkaSearchClient.from_env()
    elif searcher_mode == "hybrid":
        searchers: list[CorpusSearcher] = []
        full_raw = FullRawCorpusSearchClient.from_env(strict=False)
        if full_raw.configured:
            searchers.append(full_raw)
        researka = ResearkaSearchClient.from_env(strict=False)
        if researka.configured:
            searchers.append(researka)
        searchers.extend([
            OpenAlexFullCorpusSearchClient.from_env(strict=False),
        ])
        searcher = HybridCorpusSearchClient(searchers)
    else:
        searcher = OpenAlexFullCorpusSearchClient.from_env(strict=False)
    memo_writer = render_memo
    if writer_mode == "minimax":
        memo_writer = MiniMaxM3MemoWriter.from_env().render
    memo_selector = None
    if selector_mode == "minimax":
        memo_selector = MiniMaxM3CandidateSelector.from_env().select
    explicit_queries = bool(args.query)
    if args.query:
        base_queries = args.query
    elif planner_mode == "minimax":
        base_queries = [args.topic]
    elif args.demo:
        base_queries = [
            "sleep NAD salvage mitochondrial stress",
            "exercise NAD salvage mitochondrial repair",
        ]
    else:
        base_queries = [args.topic]
    queries = base_queries
    base_anchor_terms = query_anchor_terms(base_queries)
    shape_queries = _alpha_shape_queries(args.topic)
    strict_fullraw_auto = fullraw_backed and not explicit_queries
    if planner_mode == "minimax" and not explicit_queries and not (
        strict_fullraw_auto
        and base_anchor_terms
        and (shape_queries or len(base_anchor_terms) == 1)
    ):
        queries = MiniMaxM3SearchPlanner.from_env().plan(
            topic=args.topic,
            seed_queries=base_queries,
            limit=args.planner_limit,
        )
        if not explicit_queries:
            planned_queries = [query for query in queries if query not in set(base_queries)]
            planned_queries = _topic_anchored_queries(planned_queries, args.topic)
            if fullraw_backed and args.min_shards_searched >= 512:
                planned_queries = _alpha_shaped_planned_queries(planned_queries)
            topic_has_anchors = bool(query_anchor_terms(base_queries))
            if topic_has_anchors:
                first_anchor = set(query_anchor_terms(base_queries, limit=1))
                planned = [query for query in planned_queries if not fullraw_backed or first_anchor <= set(_topic_filter_terms(query))]
                queries = _dedupe_queries([*base_queries, *(planned[:2] or shape_queries if fullraw_backed else [*shape_queries, *planned])])
            else:
                queries = planned_queries or ([] if fullraw_backed else base_queries)
    if fullraw_backed and not explicit_queries:
        queries = _dedupe_queries([*queries, *shape_queries])
    anchor_queries = base_queries
    if not explicit_queries and not query_anchor_terms(base_queries):
        anchor_queries = queries
    wider_recall = planner_mode == "minimax" or selector_mode == "minimax"
    if fullraw_backed:
        wider_fullraw_recall = wider_recall or args.submit_researka or args.publish
        default_recall_limit = 50 if wider_fullraw_recall else _DEFAULT_FULLRAW_RECALL_LIMIT
        per_query_limit = (
            _int_env("V5_MEMO_FULL_RAW_PER_QUERY_LIMIT")
            or _int_env("V5_MEMO_FULL_RAW_RECALL_LIMIT")
            or default_recall_limit
        )
        max_query_multiplier = 4 if wider_fullraw_recall else 3
        max_hits = _int_env("V5_MEMO_FULL_RAW_MAX_HITS") or per_query_limit * max(
            2,
            min(max_query_multiplier, len(queries)),
        )
    else:
        per_query_limit = 50 if wider_recall else 25
        max_hits = 500 if wider_recall else 100
    build_kwargs = {
        "topic": args.topic,
        "seed_queries": queries,
        "searcher": searcher,
        "memo_writer": memo_writer,
        "memo_selector": memo_selector,
        "anchor_queries": anchor_queries,
        "min_alpha_tier": min_alpha_tier,
        "per_query_limit": per_query_limit,
        "max_hits": max_hits,
        "min_shards_searched": args.min_shards_searched,
        "min_sources_searched": args.min_sources_searched,
        "min_search_passes": args.min_search_passes,
        "require_publish_quality": args.submit_researka or args.publish,
    }
    try:
        result = build_alpha_memo(**build_kwargs)
    except MemoBuildError as exc:
        if args.emit_discovery_on_fail:
            result = build_alpha_memo(
                topic=args.topic,
                seed_queries=queries,
                searcher=searcher,
                memo_writer=render_memo,
                memo_selector=None,
                anchor_queries=anchor_queries,
                min_alpha_tier="discovery_seed",
                per_query_limit=per_query_limit,
                max_hits=max_hits,
                min_shards_searched=args.min_shards_searched,
                min_sources_searched=args.min_sources_searched,
                min_search_passes=args.min_search_passes,
            )
        elif not (args.publish_receipt_path or args.submit_researka or args.publish):
            raise
        else:
            error: dict[str, object] = {
                "error": exc.failure.code,
                "message": exc.failure.message,
                "details": exc.failure.details,
            }
            _write_json(args.publish_receipt_path, error)
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc
    except SearchBackendError as exc:
        error = {
            "error": "search_backend_error",
            "message": str(exc),
        }
        _write_json(args.publish_receipt_path, error)
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    memo_path = _write_memo(args.output_dir, result) if args.output_dir else None
    if args.submit_researka or args.publish:
        config = load_researka_submit_config(
            agent_id=args.researka_agent_id,
            domain_slug=args.researka_domain_slug,
            api_base=args.researka_api_base,
            submit_url=args.researka_submit_url,
        )
        if config.missing:
            error = {"error": "missing_researka_submit_config", "missing": config.missing}
            _write_json(args.publish_receipt_path, error)
            print(f"Researka submit requires {', '.join(config.missing)}", file=sys.stderr)
            raise SystemExit(3)
        if _is_discovery_seed(result):
            error = {"error": "discovery_seed_not_submitted", "tier": "discovery_seed"}
            _write_json(args.publish_receipt_path, error)
            print("Discovery seed output was not submitted to Researka", file=sys.stderr)
            raise SystemExit(4)
        if blocker := _publish_blocker(result):
            _write_json(args.publish_receipt_path, blocker)
            print(f"Publish blocked: {blocker['error']}", file=sys.stderr)
            raise SystemExit(5)
        if args.researka_parent_submission_id.strip():
            payload = build_researka_payload(
                result,
                author_agent_id=config.agent_id,
                domain_slug=config.domain_slug,
                parent_submission_id=args.researka_parent_submission_id,
            )
        else:
            payload = build_researka_payload(
                result,
                author_agent_id=config.agent_id,
                domain_slug=config.domain_slug,
            )
        response, fail_receipt = _submit_researka_with_cooldown(
            payload,
            agent_key=config.agent_key,
            api_base=config.api_base,
            submit_url=config.submit_url,
            wait_seconds=args.researka_submit_wait_seconds,
        )
        if fail_receipt:
            _write_json(args.publish_receipt_path, fail_receipt)
            if fail_receipt.get("error") == "researka_submit_deferred":
                print(f"Researka submit deferred for {fail_receipt['retry_after']}s", file=sys.stderr)
            else:
                print(f"Researka submit failed: HTTP {fail_receipt['status']} {fail_receipt['reason']}", file=sys.stderr)
            raise SystemExit(6)
        receipt: dict[str, object] = dict(response)
        should_wait = args.researka_decision_wait_seconds > 0 or args.researka_list_if_accepted
        submission_id = researka_submission_id(response)
        if should_wait and submission_id:
            decision = wait_researka_decision(
                submission_id,
                api_base=config.api_base,
                timeout_seconds=max(args.researka_decision_wait_seconds, 1.0),
                poll_seconds=args.researka_decision_poll_seconds,
            )
            receipt["decision"] = decision
            publication_id = researka_publication_id(decision)
            if args.researka_list_if_accepted and decision.get("decision") == "accept" and publication_id:
                try:
                    receipt["visibility"] = set_researka_public_visibility(
                        publication_id,
                        agent_key=config.agent_key,
                        api_base=config.api_base,
                        visibility="listed",
                    )
                except HTTPError as exc:
                    receipt["visibility_error"] = {
                        "error": "researka_visibility_update_failed",
                        "status": exc.code,
                        "reason": exc.reason,
                        "publication_id": publication_id,
                    }
        _write_json(args.publish_receipt_path, receipt)
        print(json.dumps(receipt, sort_keys=True), file=sys.stderr)
    print(memo_path if memo_path is not None else result.markdown)


def _require_full_raw_or_exit() -> None:
    try:
        require_full_raw_corpus()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


def _researka_submit_cooldown() -> tuple[float, Mapping[str, object]] | None:
    path = _researka_submit_cooldown_path()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, Mapping):
        return None
    until = _float_value(raw.get("until"))
    remaining = until - time.time()
    return (remaining, raw) if remaining > 0 else None


def _submit_researka_with_cooldown(
    payload: dict[str, object],
    *,
    agent_key: str,
    api_base: str,
    submit_url: str,
    wait_seconds: float,
) -> tuple[dict[str, object], dict[str, object]]:
    deadline = time.time() + max(0.0, wait_seconds)
    while True:
        if cooldown := _researka_submit_cooldown():
            remaining, state = cooldown
            if time.time() + remaining > deadline:
                return {}, _submit_defer_receipt(remaining, state)
            print(f"Researka submit waiting for cooldown: {int(remaining + 0.999)}s", file=sys.stderr)
            time.sleep(max(0.0, remaining))
        try:
            return submit_researka(
                payload,
                agent_key=agent_key,
                api_base=api_base,
                submit_url=submit_url,
            ), {}
        except HTTPError as exc:
            response_body = _http_error_body(exc)
            if duplicate_response := _duplicate_submission_response(exc, response_body):
                return duplicate_response, {}
            response_headers = _rate_limit_headers(exc)
            cooldown_state = _record_researka_submit_cooldown(exc, response_body=response_body)
            fail_receipt = _submit_failed_receipt(
                exc,
                cooldown_state,
                response_body=response_body,
                response_headers=response_headers,
            )
            if exc.code != 429:
                return {}, fail_receipt
            cooldown = _researka_submit_cooldown()
            if not cooldown:
                return {}, fail_receipt
            remaining, state = cooldown
            if time.time() + remaining > deadline:
                return {}, fail_receipt
            print(f"Researka submit retrying after cooldown: {int(remaining + 0.999)}s", file=sys.stderr)
            time.sleep(max(0.0, remaining))


def _duplicate_submission_response(exc: HTTPError, response_body: str) -> dict[str, object]:
    if exc.code != 409 or not response_body:
        return {}
    try:
        payload = json.loads(response_body)
    except json.JSONDecodeError:
        return {}
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if not isinstance(detail, dict) or detail.get("error") != "duplicate_submission":
        return {}
    submission_id = detail.get("submission_id")
    if not isinstance(submission_id, str) or not submission_id.strip():
        return {}
    return {"submission": {"id": submission_id.strip()}, "duplicate_submission": True}


def _submit_defer_receipt(remaining: float, state: Mapping[str, object]) -> dict[str, object]:
    return {
        "error": "researka_submit_deferred",
        "retry_after": int(remaining + 0.999),
        "cooldown_until": state.get("until_iso", ""),
        "previous_status": state.get("status", ""),
        "previous_reason": state.get("reason", ""),
        "attempts": state.get("attempts", ""),
    }


def _submit_failed_receipt(
    exc: HTTPError,
    cooldown_state: Mapping[str, object],
    *,
    response_body: str | None = None,
    response_headers: Mapping[str, str] | None = None,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "error": "researka_submit_failed",
        "status": exc.code,
        "reason": exc.reason,
        "retry_after": exc.headers.get("Retry-After", "") if exc.headers is not None else "",
        "cooldown_until": cooldown_state.get("until_iso", "") if cooldown_state else "",
        "attempts": cooldown_state.get("attempts", "") if cooldown_state else "",
    }
    if limit_kind := cooldown_state.get("limit_kind"):
        receipt["limit_kind"] = limit_kind
    body = _http_error_body(exc) if response_body is None else response_body
    if body:
        receipt["response_body"] = body
    headers = _rate_limit_headers(exc) if response_headers is None else dict(response_headers)
    if headers:
        receipt["response_headers"] = headers
    return receipt


def _http_error_body(exc: HTTPError, *, limit: int = 1000) -> str:
    try:
        raw = exc.read(limit + 1)
    except (OSError, ValueError):
        return ""
    if not raw:
        return ""
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    return text[:limit]


def _rate_limit_headers(exc: HTTPError) -> dict[str, str]:
    if exc.headers is None:
        return {}
    out: dict[str, str] = {}
    for key in ("Retry-After", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"):
        if value := exc.headers.get(key):
            out[key] = value
    return out


def _record_researka_submit_cooldown(exc: HTTPError, *, response_body: str = "") -> Mapping[str, object]:
    if exc.code != 429:
        return {}
    retry_after = _float_value(exc.headers.get("Retry-After", "") if exc.headers is not None else "")
    path = _researka_submit_cooldown_path()
    previous = _read_json_mapping(path)
    attempts = int(_float_value(previous.get("attempts"))) + 1 if previous else 1
    limit_kind = "daily" if _is_daily_limit_response(response_body) else "rate"
    if retry_after <= 0:
        if limit_kind == "daily":
            retry_after = _daily_limit_retry_after_seconds()
        else:
            base = _float_value(os.environ.get("V5_MEMO_RESEARKA_SUBMIT_COOLDOWN_SECONDS", "300")) or 300.0
            retry_after = base * (2 ** min(attempts - 1, 4))
    max_cooldown = 86400.0 if limit_kind == "daily" else 3600.0
    until = time.time() + max(1.0, min(retry_after, max_cooldown))
    payload: dict[str, object] = {
        "attempts": attempts,
        "limit_kind": limit_kind,
        "until": until,
        "until_iso": datetime.fromtimestamp(until, UTC).isoformat(),
        "status": exc.code,
        "reason": exc.reason,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True))
    return payload


def _is_daily_limit_response(response_body: str) -> bool:
    return "daily_limit_exceeded" in response_body.casefold()


def _daily_limit_retry_after_seconds() -> float:
    override = _float_value(os.environ.get("V5_MEMO_RESEARKA_DAILY_LIMIT_COOLDOWN_SECONDS", ""))
    if override > 0:
        return override
    now = time.time()
    today = datetime.fromtimestamp(now, UTC).date()
    reset_day = today + timedelta(days=1)
    reset = datetime(reset_day.year, reset_day.month, reset_day.day, tzinfo=UTC) + timedelta(minutes=5)
    return max(300.0, reset.timestamp() - now)


def _researka_submit_cooldown_path() -> Path:
    return Path(os.environ.get("V5_MEMO_RESEARKA_SUBMIT_COOLDOWN_PATH", "/tmp/v5-memo-researka-submit-cooldown.json"))


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, Mapping) else {}


def _float_value(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        return 0.0
    try:
        return float(value) if value not in (None, "") else 0.0
    except ValueError:
        return 0.0


def _topic_anchored_queries(queries: Sequence[str], topic: str) -> list[str]:
    if not query_anchor_terms([topic], limit=4):
        return list(queries)
    topic_anchors = set(_topic_filter_terms(topic))
    if not topic_anchors:
        return list(queries)
    required_overlap = min(2, len(topic_anchors))
    filtered = [
        query
        for query in queries
        if len(topic_anchors & set(_topic_filter_terms(query))) >= required_overlap
    ]
    return filtered


def _alpha_shaped_planned_queries(queries: Sequence[str]) -> list[str]:
    shaped = [
        query for query in queries
        if set(_topic_filter_terms(query)) & _ALPHA_QUERY_TERMS
    ]
    return sorted(shaped, key=_alpha_planned_query_rank, reverse=True)


def _alpha_planned_query_rank(query: str) -> tuple[int, int, int]:
    terms = set(_TOPIC_TERM_RE.findall(query.casefold()))
    return (
        len(terms & _DIRECT_EVIDENCE_QUERY_TERMS) - len(terms & _MODEL_ONLY_QUERY_TERMS),
        len(terms & _ALPHA_QUERY_TERMS),
        len(terms),
    )


def _alpha_shape_queries(topic: str) -> list[str]:
    terms = list(_topic_filter_terms(topic))
    has_negative_shape = bool(set(terms) & _NEGATIVE_ALPHA_QUERY_TERMS)
    cleaned_terms = [term for term in terms if term not in _ALPHA_QUERY_TERMS]
    terms = cleaned_terms or terms
    if len(terms) < 2:
        return []
    split_at = next(
        (index for index, term in enumerate(terms[1:], start=1) if term in _SHAPE_CONTEXT_TERMS),
        1,
    )
    anchor = " ".join(terms[:split_at])
    rest = " ".join(terms[split_at:])
    direct_rest = " ".join(terms[split_at:split_at + 2])
    queries = [
        f"{anchor} human trial {direct_rest}".strip(),
        f"{anchor} augment {rest} protocol",
        f"{anchor} blunts {rest}",
    ]
    if has_negative_shape and set(terms) & _SHAPE_CONTEXT_TERMS:
        queries.insert(0, f"{anchor} mimics {rest}")
    return queries


def _dedupe_queries(queries: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for query in queries:
        clean = " ".join(query.split())
        key = _seed_query_key(clean)
        if clean and key and key not in seen:
            seen.add(key)
            out.append(clean)
    return out


def _topic_filter_terms(topic: str) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in _TOPIC_TERM_RE.findall(topic.casefold()):
        if raw in _TOPIC_FILTER_DROP or raw in seen:
            continue
        seen.add(raw)
        out.append(raw)
    return tuple(out[:5])


def _int_env(name: str) -> int:
    return _optional_int_env(name) or 0


def _coverage_threshold(
    explicit: int | None,
    *,
    primary: str,
    fallback: str = "",
    allow_fallback: bool = False,
) -> int:
    if explicit is not None:
        return max(0, explicit)
    primary_value = _optional_int_env(primary)
    if primary_value is not None:
        return primary_value
    if allow_fallback and fallback:
        return _optional_int_env(fallback) or 0
    return 0


def _optional_int_env(name: str) -> int | None:
    try:
        raw = os.environ.get(name)
        if raw is None or raw.strip() == "":
            return None
        return max(0, int(raw))
    except ValueError:
        return None


def _write_memo(output_dir: str, result: MemoResult) -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    heading = next((line[2:] for line in result.markdown.splitlines() if line.startswith("# ")), "v5 memo")
    slug = re.sub(r"[^a-z0-9]+", "-", heading.casefold()).strip("-")[:90] or "v5-memo"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = target_dir / f"{stamp}-{slug}.md"
    path.write_text(result.markdown.strip() + "\n", encoding="utf-8")
    return path


def _write_json(path: str, payload: Mapping[str, object]) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _is_discovery_seed(result: object) -> bool:
    candidate = getattr(result, "candidate", None)
    reasons = getattr(candidate, "reasons", ())
    return any(reason == "tier:discovery_seed" for reason in reasons)


def _publish_blocker(result: object) -> dict[str, object] | None:
    markdown = getattr(result, "markdown", "")
    receipts = tuple(getattr(result, "receipts", ()) or ())
    unsafe_dois = sorted(
        doi
        for hit in receipts
        if (doi := str(getattr(hit, "doi", "") or "").strip())
        and any(char in doi for char in _UNSAFE_DOI_CHARS)
        and doi in markdown
    )
    if unsafe_dois:
        return {"error": "unbundled_doi_citation", "dois": unsafe_dois}
    candidate = getattr(result, "candidate", None)
    return candidate_publish_blocker(candidate) if candidate is not None else None


if __name__ == "__main__":
    main()
