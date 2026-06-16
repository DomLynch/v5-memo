"""CLI for offline demo or live Researka DB memo generation."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

from v5_memo.client import ResearkaSearchClient
from v5_memo.pipeline import build_alpha_memo
from v5_memo.schemas import CorpusHit


class DemoSearch:
    def search(self, query: str, *, limit: int = 25) -> Sequence[CorpusHit]:
        del limit
        hits = {
            "sleep": CorpusHit(
                hit_id="demo-sleep",
                title="NAD salvage links sleep fragmentation to mitochondrial stress",
                abstract=(
                    "Sleep fragmentation increased inflammatory tone through NAD salvage "
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
    parser.add_argument("--domain", default="longevity")
    parser.add_argument("--query", action="append", default=[])
    args = parser.parse_args()

    searcher = (
        DemoSearch()
        if args.demo
        else ResearkaSearchClient.from_env(domain=args.domain)
    )
    queries = args.query or [
        "sleep NAD salvage mitochondrial stress",
        "exercise NAD salvage mitochondrial repair",
    ]
    result = build_alpha_memo(topic=args.topic, seed_queries=queries, searcher=searcher)
    print(result.markdown)


if __name__ == "__main__":
    main()
