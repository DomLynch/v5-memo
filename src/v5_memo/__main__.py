"""CLI for offline demo or live full-corpus memo generation."""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from v5_memo.client import (
    FullRawCorpusSearchClient,
    HybridCorpusSearchClient,
    OpenAlexFullCorpusSearchClient,
    ResearkaSearchClient,
)
from v5_memo.coverage import current_search_coverage, require_full_raw_corpus
from v5_memo.miner import query_anchor_terms
from v5_memo.minimax_writer import (
    MiniMaxM3CandidateSelector,
    MiniMaxM3MemoWriter,
    MiniMaxM3SearchPlanner,
)
from v5_memo.pipeline import build_alpha_memo
from v5_memo.retriever import CorpusSearcher
from v5_memo.schemas import CorpusHit
from v5_memo.writer import render_memo


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
    parser.add_argument("--min-shards-searched", type=int, default=_int_env("V5_MEMO_MEMO_MIN_SHARDS_SEARCHED"))
    parser.add_argument("--min-sources-searched", type=int, default=_int_env("V5_MEMO_MEMO_MIN_SOURCES_SEARCHED"))
    parser.add_argument("--min-search-passes", type=int, default=_int_env("V5_MEMO_MEMO_MIN_SEARCH_PASSES"))
    args = parser.parse_args()

    if args.coverage_report:
        print(current_search_coverage().summary)
        return
    if args.require_full_raw_corpus:
        _require_full_raw_or_exit()

    searcher_mode = "hybrid" if args.searcher == "smart" else args.searcher
    planner_mode = args.planner or ("minimax" if args.searcher == "smart" else "seed")
    writer_mode = args.writer or ("minimax" if args.searcher == "smart" else "template")
    selector_mode = args.selector or ("minimax" if writer_mode == "minimax" else "deterministic")
    alpha_tier = args.min_alpha_tier or (
        "elite" if args.searcher == "smart" or selector_mode == "minimax" else "publishable"
    )
    min_alpha_tier = "discovery_seed" if alpha_tier == "discovery" else f"{alpha_tier}_alpha"

    searcher: CorpusSearcher
    if args.demo:
        searcher = DemoSearch()
    elif searcher_mode == "fullraw":
        _require_full_raw_or_exit()
        searcher = FullRawCorpusSearchClient.from_env(strict=args.searcher == "smart")
    elif searcher_mode == "researka":
        searcher = ResearkaSearchClient.from_env()
    elif searcher_mode == "hybrid":
        searchers: list[CorpusSearcher] = []
        full_raw = FullRawCorpusSearchClient.from_env(strict=args.searcher == "smart")
        if full_raw.configured:
            searchers.append(full_raw)
        researka = ResearkaSearchClient.from_env(strict=args.searcher == "smart")
        if researka.configured:
            searchers.append(researka)
        searchers.extend([
            OpenAlexFullCorpusSearchClient.from_env(strict=args.searcher == "smart"),
        ])
        searcher = HybridCorpusSearchClient(searchers)
    else:
        searcher = OpenAlexFullCorpusSearchClient.from_env(strict=args.searcher == "smart")
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
    else:
        base_queries = [
            "sleep NAD salvage mitochondrial stress",
            "exercise NAD salvage mitochondrial repair",
        ]
    queries = base_queries
    if planner_mode == "minimax":
        queries = MiniMaxM3SearchPlanner.from_env().plan(
            topic=args.topic,
            seed_queries=base_queries,
            limit=args.planner_limit,
        )
        if not explicit_queries:
            planned_queries = [query for query in queries if query not in set(base_queries)]
            planned_queries = _topic_anchored_queries(planned_queries, args.topic)
            queries = planned_queries or base_queries
    anchor_queries = base_queries
    if not explicit_queries and not query_anchor_terms(base_queries):
        anchor_queries = queries
    wider_recall = planner_mode == "minimax" or selector_mode == "minimax"
    result = build_alpha_memo(
        topic=args.topic,
        seed_queries=queries,
        searcher=searcher,
        memo_writer=memo_writer,
        memo_selector=memo_selector,
        anchor_queries=anchor_queries,
        min_alpha_tier=min_alpha_tier,
        per_query_limit=50 if wider_recall else 25,
        max_hits=500 if wider_recall else 100,
        min_shards_searched=args.min_shards_searched,
        min_sources_searched=args.min_sources_searched,
        min_search_passes=args.min_search_passes,
    )
    print(result.markdown)


def _require_full_raw_or_exit() -> None:
    try:
        require_full_raw_corpus()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


def _topic_anchored_queries(queries: Sequence[str], topic: str) -> list[str]:
    topic_anchors = set(query_anchor_terms([topic], limit=4))
    if not topic_anchors:
        return list(queries)
    required_overlap = min(2, len(topic_anchors))
    filtered = [
        query
        for query in queries
        if len(topic_anchors & set(query_anchor_terms([query], limit=4))) >= required_overlap
    ]
    return filtered


def _int_env(name: str) -> int:
    try:
        return max(0, int(os.environ.get(name, "0")))
    except ValueError:
        return 0


if __name__ == "__main__":
    main()
