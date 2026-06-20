"""Independent V5 alpha memo writer."""
from v5_memo.binder import bind_receipts
from v5_memo.client import (
    FullRawCorpusSearchClient,
    HybridCorpusSearchClient,
    OpenAlexFullCorpusSearchClient,
    ResearkaSearchClient,
)
from v5_memo.coverage import SearchCoverage, current_search_coverage, require_full_raw_corpus
from v5_memo.gate import candidate_alpha_tier, meets_min_alpha_tier
from v5_memo.miner import mine_insights, query_anchor_terms
from v5_memo.minimax_writer import (
    MiniMaxM3CandidateSelector,
    MiniMaxM3MemoWriter,
    MiniMaxM3SearchPlanner,
)
from v5_memo.pipeline import build_alpha_memo
from v5_memo.retriever import collect_seed_hits
from v5_memo.schemas import (
    CorpusHit,
    InsightCandidate,
    MemoBuildError,
    MemoResult,
    ReceiptRole,
    SearchFailure,
)
from v5_memo.writer import render_alpha_memo, render_discovery_seed, render_memo

__all__ = [
    "CorpusHit",
    "FullRawCorpusSearchClient",
    "HybridCorpusSearchClient",
    "InsightCandidate",
    "MemoBuildError",
    "MemoResult",
    "MiniMaxM3CandidateSelector",
    "MiniMaxM3MemoWriter",
    "MiniMaxM3SearchPlanner",
    "OpenAlexFullCorpusSearchClient",
    "ReceiptRole",
    "ResearkaSearchClient",
    "SearchCoverage",
    "SearchFailure",
    "bind_receipts",
    "build_alpha_memo",
    "candidate_alpha_tier",
    "collect_seed_hits",
    "current_search_coverage",
    "meets_min_alpha_tier",
    "mine_insights",
    "query_anchor_terms",
    "render_alpha_memo",
    "render_discovery_seed",
    "render_memo",
    "require_full_raw_corpus",
]
