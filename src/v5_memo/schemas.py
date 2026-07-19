"""Typed contracts for search hits, insight candidates, and rendered memos."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

_UNSAFE_DOI_CHARS = frozenset("()[]{}")


def _usable_doi(value: str | None) -> str:
    doi = (value or "").strip()
    return "" if not doi or any(char in doi for char in _UNSAFE_DOI_CHARS) else doi


@dataclass(frozen=True, slots=True)
class CorpusHit:
    """One searchable corpus item that can become a memo receipt."""

    hit_id: str
    title: str
    abstract: str
    source: str
    year: int | None = None
    url: str = ""
    doi: str | None = None
    venue: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def receipt_id(self) -> str:
        return _usable_doi(self.doi) or self.hit_id

    @property
    def source_key(self) -> str:
        if doi := _usable_doi(self.doi):
            return f"doi:{doi.casefold()}"
        pmid = self.metadata.get("pmid")
        if isinstance(pmid, str) and pmid:
            return f"pmid:{pmid}"
        for key in ("pmcid", "openalex_id", "semantic_scholar_id", "arxiv_id"):
            value = self.metadata.get(key)
            if isinstance(value, str) and value:
                return f"{key}:{value.casefold()}"
        return f"{self.source}:{self.title.casefold()}:{self.year or ''}"

    @property
    def text(self) -> str:
        return f"{self.title} {self.abstract} {self.venue or ''}"


@dataclass(frozen=True, slots=True)
class ClaimCard:
    """Receipt-local claim shape used before prose generation."""

    receipt_id: str
    role: str
    design: str
    population: str
    outcome: str
    direction: str
    support_type: str
    confidence: str
    quote: str


@dataclass(frozen=True, slots=True)
class EvidenceNode:
    """One receipt's role in the candidate evidence graph."""

    receipt_id: str
    role: str
    reason: str


@dataclass(frozen=True, slots=True)
class InsightCandidate:
    """Receipt-addressed alpha hypothesis candidate."""

    topic: str
    thesis: str
    bridge_terms: tuple[str, ...]
    tension_terms: tuple[str, ...]
    receipt_ids: tuple[str, ...]
    score: int
    novelty_score: int
    evidence_score: int
    reasons: tuple[str, ...]
    scorecard: Mapping[str, int] = field(default_factory=dict)
    receipt_roles: tuple[ReceiptRole, ...] = ()
    claim_cards: tuple[ClaimCard, ...] = ()
    evidence_graph: tuple[EvidenceNode, ...] = ()


@dataclass(frozen=True, slots=True)
class ReceiptRole:
    """Selector-owned receipt role used by writers and judges."""

    receipt_id: str
    role: str
    reason: str


@dataclass(frozen=True, slots=True)
class MemoResult:
    """Rendered memo plus the evidence used to render it."""

    candidate: InsightCandidate
    receipts: Sequence[CorpusHit]
    markdown: str
    supporting_receipts: Sequence[CorpusHit] = ()
    supporting_min_shards_searched: int = 0
    supporting_min_sources_searched: int = 0
    supporting_min_search_passes: int = 0


@dataclass(frozen=True, slots=True)
class SearchFailure:
    """Structured reason an alpha memo could not be built."""

    code: str
    message: str
    details: Mapping[str, object] = field(default_factory=dict)


class MemoBuildError(ValueError):
    """Raised when the pipeline cannot build a receipt-bound memo."""

    def __init__(self, failure: SearchFailure) -> None:
        self.failure = failure
        detail = ", ".join(
            f"{key}={value}"
            for key, value in failure.details.items()
            if key in {
                "hit_count",
                "candidate_count",
                "mined_candidate_count",
                "best_mined_score",
                "best_mined_novelty",
                "publish_quality_blocked_count",
                "min_alpha_tier",
            }
        )
        super().__init__(f"{failure.message} ({detail})" if detail else failure.message)
