"""Typed contracts for search hits, insight candidates, and rendered memos."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


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
        return self.doi or self.hit_id

    @property
    def source_key(self) -> str:
        if self.doi:
            return f"doi:{self.doi.casefold()}"
        pmid = self.metadata.get("pmid")
        if isinstance(pmid, str) and pmid:
            return f"pmid:{pmid}"
        return f"{self.source}:{self.title.casefold()}:{self.year or ''}"

    @property
    def text(self) -> str:
        return f"{self.title} {self.abstract} {self.venue or ''}"


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
    receipt_roles: tuple[ReceiptRole, ...] = ()


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
        super().__init__(failure.message)
