"""Shared source-artifact admissibility policy."""
from __future__ import annotations

import re
from typing import Literal

from v5_memo.schemas import CorpusHit

SourceArtifactType = Literal[
    "article",
    "conference_abstract",
    "secondary_commentary",
    "supplemental",
]

_SUPPLEMENTAL_PHRASES = (
    "additional file",
    "data sheet",
    "dataset",
    "dryad",
    "figshare",
    "supplemental data",
    "supplemental file",
    "supplemental material",
    "supplementary data",
    "supplementary file",
    "supplementary material",
    "zenodo",
)
_CONFERENCE_PHRASES = (
    "abstract supplement",
    "conference abstract",
    "meeting abstract",
    "poster abstract",
)
_SECONDARY_PHRASES = (
    "comment on",
    "corrigendum",
    "erratum",
    "faculty opinions recommendation",
    "reply to",
)
_SUPPLEMENT_DOI_RE = re.compile(r"(?:^|[-_.])s\d+(?:[-_.])p\d+(?:$|[-_.])")
_NUMBERED_ABSTRACT_TITLE_RE = re.compile(r"\b\d{2,5}-pub:")


def source_artifact_type(hit: CorpusHit) -> SourceArtifactType:
    """Classify whether a corpus hit is admissible as article-level evidence."""
    descriptor = " ".join(
        part.casefold()
        for part in (
            hit.title,
            hit.abstract,
            hit.venue or "",
            hit.source,
            hit.doi or "",
            hit.hit_id,
            hit.url,
            " ".join(str(value) for value in hit.metadata.values()),
        )
        if part
    )
    if any(phrase in descriptor for phrase in _SUPPLEMENTAL_PHRASES):
        return "supplemental"
    doi = str(hit.doi or hit.hit_id or "").casefold()
    if (
        any(phrase in descriptor for phrase in _CONFERENCE_PHRASES)
        or ("10.1096/fasebj" in doi and ".s1." in doi)
        or doi.startswith("10.1249/01.mss.")
        or bool(_SUPPLEMENT_DOI_RE.search(doi))
        or bool(_NUMBERED_ABSTRACT_TITLE_RE.search(descriptor))
    ):
        return "conference_abstract"
    if any(phrase in descriptor for phrase in _SECONDARY_PHRASES) or doi.startswith("10.3410/f."):
        return "secondary_commentary"
    return "article"


def is_non_primary_receipt(hit: CorpusHit) -> bool:
    return source_artifact_type(hit) != "article"
