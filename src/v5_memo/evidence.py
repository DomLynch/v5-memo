"""Shared source-artifact admissibility policy."""
from __future__ import annotations

import ipaddress
import re
from collections.abc import Mapping, Sequence
from typing import Literal
from urllib.parse import parse_qsl, urlparse

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
    "correction notice",
    "corrigendum",
    "erratum",
    "expression of concern",
    "faculty opinions recommendation",
    "retraction",
    "reply to",
    "withdrawal notice",
    "withdrawn",
)
_SUPPLEMENT_DOI_RE = re.compile(r"(?:^|[-_.])s\d+(?:[-_.])p\d+(?:$|[-_.])")
_NUMBERED_ABSTRACT_TITLE_RE = re.compile(r"\b\d{2,5}-pub:")
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_PMCID_RE = re.compile(r"^PMC\d+$", re.IGNORECASE)
_OPENALEX_RE = re.compile(r"^(?:https?://openalex\.org/)?W\d+$", re.IGNORECASE)
_ARXIV_RE = re.compile(r"^(?:arXiv:)?(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})(?:v\d+)?$", re.IGNORECASE)
_CORRECTION_TITLE_RE = re.compile(
    r"^(?:(?:author|publisher)\s+)?(?:correction|corrigendum|erratum)\b|"
    r"\bcorrection\s+(?:notice|to)\b",
    re.IGNORECASE,
)
_RETRACTION_TITLE_RE = re.compile(
    r"^(?:retraction|retracted)(?::|\b)|\bretraction\s+(?:notice|of)\b|"
    r"\b(?:this\s+)?article\s+has\s+been\s+retracted\b",
    re.IGNORECASE,
)
_CONCERN_TITLE_RE = re.compile(r"^expression[-\s]+of[-\s]+concern\b", re.IGNORECASE)
_WITHDRAWN_TITLE_RE = re.compile(
    r"^(?:withdrawn|withdrawal\s+notice)\b|\bwithdrawn\s+article\b",
    re.IGNORECASE,
)
_INTEGRITY_METADATA_KEYS = (
    "correction_status",
    "document_type",
    "is_retracted",
    "is_withdrawn",
    "publication_type",
    "publication_types",
    "relation",
    "relations",
    "retracted",
    "retraction_status",
    "type",
    "update_type",
    "withdrawn",
    "work_type",
)
_PRIMARY_ARTICLE_TYPES = frozenset({
    "article",
    "clinical trial",
    "comparative study",
    "controlled clinical trial",
    "cross sectional study",
    "evaluation study",
    "interventional study",
    "journal article",
    "multicenter study",
    "observational study",
    "original article",
    "pragmatic clinical trial",
    "randomized controlled trial",
    "research article",
    "validation study",
})
_NON_PRIMARY_DOCUMENT_TYPES = frozenset({
    "abstract",
    "book chapter",
    "comment",
    "conference abstract",
    "correction",
    "editorial",
    "erratum",
    "expression of concern",
    "letter",
    "meeting abstract",
    "peer review",
    "poster abstract",
    "published erratum",
    "retraction notice",
    "review",
    "withdrawal notice",
})


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
    secondary_descriptor = " ".join(
        part.casefold()
        for part in (
            hit.title,
            hit.venue or "",
            hit.source,
            hit.doi or "",
            hit.hit_id,
            _source_type_metadata_text(hit),
        )
        if part
    )
    if (
        bool(_source_type_values(hit) & _NON_PRIMARY_DOCUMENT_TYPES)
        or
        any(phrase in secondary_descriptor for phrase in _SECONDARY_PHRASES)
        or doi.startswith("10.3410/f.")
    ):
        return "secondary_commentary"
    return "article"


def is_non_primary_receipt(hit: CorpusHit) -> bool:
    return source_artifact_type(hit) != "article"


def has_verified_primary_article_type(hit: CorpusHit) -> bool:
    """Require a recognized provider type before granting top-tier source status."""
    types = _source_type_values(hit)
    return bool(types & _PRIMARY_ARTICLE_TYPES) and not bool(
        types & _NON_PRIMARY_DOCUMENT_TYPES
    )


def normalize_publication_integrity(item: Mapping[str, object]) -> dict[str, object]:
    """Normalize provider-specific publication status into one shared contract."""
    nested_raw = item.get("metadata")
    nested = nested_raw if isinstance(nested_raw, Mapping) else {}
    document_type = _clean_metadata_value(
        item.get("document_type")
        or item.get("work_type")
        or item.get("type_crossref")
        or item.get("type")
        or nested.get("document_type")
        or nested.get("documentType")
        or nested.get("publication_type")
        or nested.get("publicationType")
    )
    publication_types = _metadata_text_tuple(
        item.get("publication_types")
        or item.get("publication_type")
        or item.get("publicationtypes")
        or item.get("publicationTypes")
        or nested.get("publication_types")
        or nested.get("publication_type")
        or nested.get("publicationtypes")
        or nested.get("publicationTypes")
    )
    raw_retracted = item.get("is_retracted")
    if raw_retracted is None:
        raw_retracted = nested.get("is_retracted")
    if raw_retracted is None:
        raw_retracted = item.get("retracted")
    if raw_retracted is None:
        raw_retracted = nested.get("retracted")
    raw_withdrawn = item.get("is_withdrawn")
    if raw_withdrawn is None:
        raw_withdrawn = nested.get("is_withdrawn")
    if raw_withdrawn is None:
        raw_withdrawn = item.get("withdrawn")
    if raw_withdrawn is None:
        raw_withdrawn = nested.get("withdrawn")
    explicit_status = _clean_metadata_value(
        item.get("correction_status")
        or item.get("retraction_status")
        or item.get("update_type")
        or nested.get("correction_status")
        or nested.get("retraction_status")
        or nested.get("update_type")
    )
    relation_status = _metadata_status_text(
        item.get("relation")
        or item.get("relations")
        or nested.get("relation")
        or nested.get("relations")
    )
    correction_status = " ".join(
        value for value in (explicit_status, relation_status) if value
    )[:500]
    status_text = _normalize_status_text(
        " ".join((document_type, " ".join(publication_types), correction_status))
    )
    parsed_retracted = _metadata_bool(raw_retracted)
    parsed_withdrawn = _metadata_bool(raw_withdrawn)
    is_retracted = (
        True
        if parsed_retracted is True or _status_says_retracted(status_text)
        else False
        if _status_says_not_retracted(status_text)
        else parsed_retracted
    )
    is_withdrawn = (
        True
        if parsed_withdrawn is True or _status_says_withdrawn(status_text)
        else False
        if _status_says_not_withdrawn(status_text)
        else parsed_withdrawn
    )
    return {
        "document_type": document_type,
        "publication_types": publication_types,
        "is_retracted": is_retracted,
        "retraction_status_known": (
            parsed_retracted is not None or _status_mentions_retraction(status_text)
        ),
        "is_withdrawn": is_withdrawn,
        "withdrawal_status_known": (
            parsed_withdrawn is not None or _status_mentions_withdrawal(status_text)
        ),
        "correction_status": correction_status,
    }


def merge_publication_integrity(
    preferred: Mapping[str, object],
    observed: Mapping[str, object],
) -> dict[str, object]:
    """Merge duplicate-source status conservatively while content/rank stays preferred."""
    left = normalize_publication_integrity(preferred)
    right = normalize_publication_integrity(observed)
    publication_types = tuple(dict.fromkeys(
        (
            *(_metadata_text_tuple(left.get("publication_types"))),
            *(_metadata_text_tuple(right.get("publication_types"))),
            *(
                value
                for value in (
                    _clean_metadata_value(left.get("document_type")),
                    _clean_metadata_value(right.get("document_type")),
                )
                if value
            ),
        )
    ))
    return {
        "document_type": left.get("document_type") or right.get("document_type") or "",
        "publication_types": publication_types,
        "is_retracted": _merge_unsafe_bool(left.get("is_retracted"), right.get("is_retracted")),
        "retraction_status_known": bool(
            left.get("retraction_status_known") or right.get("retraction_status_known")
        ),
        "is_withdrawn": _merge_unsafe_bool(left.get("is_withdrawn"), right.get("is_withdrawn")),
        "withdrawal_status_known": bool(
            left.get("withdrawal_status_known") or right.get("withdrawal_status_known")
        ),
        "correction_status": " ".join(dict.fromkeys(
            value
            for value in (
                _clean_metadata_value(left.get("correction_status")),
                _clean_metadata_value(right.get("correction_status")),
            )
            if value
        ))[:500],
    }


def stable_source_identity(hit: CorpusHit) -> dict[str, str] | None:
    """Return a canonical public locator, without trusting a title-only receipt."""
    doi = _valid_doi(hit.doi)
    if doi:
        return {"kind": "doi", "value": doi, "url": f"https://doi.org/{doi}"}

    pmid = _digits(hit.metadata.get("pmid"))
    if not pmid and "pubmed" in hit.source.casefold() and hit.hit_id.isdigit():
        pmid = hit.hit_id
    if pmid:
        return {
            "kind": "pmid",
            "value": pmid,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }

    pmcid = str(hit.metadata.get("pmcid") or "").strip()
    if _PMCID_RE.fullmatch(pmcid):
        normalized = pmcid.upper()
        return {
            "kind": "pmcid",
            "value": normalized,
            "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{normalized}/",
        }

    openalex = str(hit.metadata.get("openalex_id") or "").strip()
    if not openalex and _OPENALEX_RE.fullmatch(hit.hit_id):
        openalex = hit.hit_id
    if _OPENALEX_RE.fullmatch(openalex):
        work_id = openalex.rsplit("/", 1)[-1].upper()
        return {
            "kind": "openalex",
            "value": work_id,
            "url": f"https://openalex.org/{work_id}",
        }

    arxiv = str(hit.metadata.get("arxiv_id") or "").strip()
    if _ARXIV_RE.fullmatch(arxiv):
        value = arxiv.removeprefix("arXiv:")
        return {"kind": "arxiv", "value": value, "url": f"https://arxiv.org/abs/{value}"}

    semantic_scholar = _semantic_scholar_id(hit)
    if semantic_scholar:
        return {
            "kind": "semantic_scholar",
            "value": semantic_scholar,
            "url": f"https://www.semanticscholar.org/paper/{semantic_scholar}",
        }

    url = _stable_http_url(hit.url)
    if url:
        return {"kind": "url", "value": url, "url": url}
    return None


def source_integrity_issue(hit: CorpusHit) -> dict[str, object] | None:
    """Fail closed for non-articles, unsafe publication status, or no stable identity."""
    title = " ".join(hit.title.split())
    metadata_text = _integrity_metadata_text(hit)
    correction_status = str(hit.metadata.get("correction_status") or "")
    if (
        _truthy_metadata(hit, "is_retracted", "retracted")
        or _RETRACTION_TITLE_RE.search(title)
        or any(term in metadata_text for term in ("retracted publication", "retraction of publication"))
        or _status_says_retracted(correction_status)
    ):
        return {"error": "source_retracted", "receipt_id": hit.receipt_id}
    if (
        _truthy_metadata(hit, "is_withdrawn", "withdrawn")
        or _WITHDRAWN_TITLE_RE.search(title)
        or _status_says_withdrawn(metadata_text)
    ):
        return {"error": "source_withdrawn", "receipt_id": hit.receipt_id}
    if (
        _CONCERN_TITLE_RE.search(title)
        or _status_says_expression_of_concern(metadata_text)
        or _status_says_expression_of_concern(correction_status)
    ):
        return {"error": "source_expression_of_concern", "receipt_id": hit.receipt_id}
    if _CORRECTION_TITLE_RE.search(title) or _status_says_correction(correction_status):
        return {"error": "source_correction_notice", "receipt_id": hit.receipt_id}
    document_types = " ".join(
        str(hit.metadata.get(key) or "").casefold()
        for key in ("document_type", "publication_type", "publication_types", "type", "work_type")
    )
    if any(
        term in document_types
        for term in ("correction", "corrigendum", "erratum", "retraction notice")
    ):
        return {"error": "source_correction_notice", "receipt_id": hit.receipt_id}
    artifact_type = source_artifact_type(hit)
    if artifact_type != "article":
        return {
            "error": "source_not_article",
            "receipt_id": hit.receipt_id,
            "artifact_type": artifact_type,
        }
    if stable_source_identity(hit) is None:
        return {"error": "missing_stable_source_identity", "receipt_id": hit.receipt_id}
    return None


def _valid_doi(value: object) -> str:
    doi = str(value or "").strip().removeprefix("https://doi.org/").rstrip(".,;")
    return doi if _DOI_RE.fullmatch(doi) and not any(char in doi for char in "()[]{}") else ""


def _digits(value: object) -> str:
    text = str(value or "").strip()
    return text if text.isdigit() else ""


def _stable_http_url(value: object) -> str:
    text = str(value or "").strip()
    parsed = urlparse(text)
    hostname = parsed.hostname or ""
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or "." not in hostname:
        return ""
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        return ""
    if hostname.casefold() == "localhost" or hostname.casefold().endswith(".local"):
        return ""
    path_parts = tuple(part.casefold() for part in parsed.path.split("/") if part)
    query_keys = {key.casefold() for key, _ in parse_qsl(parsed.query)}
    if (
        not path_parts
        or any(part in {"search", "query", "results"} for part in path_parts)
        or query_keys & {"q", "query", "search", "keywords"}
    ):
        return ""
    if hostname in {"doi.org", "dx.doi.org"} and not _valid_doi(parsed.path.lstrip("/")):
        return ""
    return text


def _integrity_metadata_text(hit: CorpusHit) -> str:
    return _normalize_status_text(" ".join(
        str(hit.metadata.get(key, ""))
        for key in _INTEGRITY_METADATA_KEYS
        if hit.metadata.get(key) not in (None, "", False, (), [], {})
    ))


def _source_type_metadata_text(hit: CorpusHit) -> str:
    return " ".join(
        str(hit.metadata.get(key, "")).casefold()
        for key in ("document_type", "publication_type", "publication_types", "type", "work_type")
        if hit.metadata.get(key) not in (None, "", False, (), [], {})
    )


def _source_type_values(hit: CorpusHit) -> frozenset[str]:
    values: list[str] = []
    for key in ("document_type", "publication_type", "publication_types", "type", "work_type"):
        raw = hit.metadata.get(key)
        values.extend(_metadata_text_tuple(raw))
    return frozenset(_normalize_status_text(value) for value in values if value)


def _truthy_metadata(hit: CorpusHit, *keys: str) -> bool:
    for key in keys:
        value = hit.metadata.get(key)
        if value is True:
            return True
        if isinstance(value, str) and value.strip().casefold() in {
            "1",
            "retracted",
            "true",
            "withdrawn",
            "yes",
        }:
            return True
    return False


def _clean_metadata_value(value: object) -> str:
    return " ".join(value.split())[:200] if isinstance(value, str) else ""


def _metadata_text_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        clean = _clean_metadata_value(value)
        return (clean,) if clean else ()
    if not isinstance(value, Sequence):
        return ()
    return tuple(clean for item in value if (clean := _clean_metadata_value(item)))


def _metadata_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1 if value in {0, 1} else None
    if not isinstance(value, str):
        return None
    normalized = _normalize_status_text(value)
    if normalized in {"1", "retracted", "true", "withdrawn", "yes"}:
        return True
    if normalized in {"0", "false", "no", "not retracted", "not withdrawn", "unretracted"}:
        return False
    return None


def _status_says_retracted(value: str) -> bool:
    normalized = _normalize_status_text(value)
    return "retract" in normalized and not any(
        term in normalized for term in ("not retract", "not retracted", "unretracted")
    )


def _status_mentions_retraction(value: str) -> bool:
    normalized = _normalize_status_text(value)
    return "retract" in normalized or "unretracted" in normalized


def _status_says_not_retracted(value: str) -> bool:
    normalized = _normalize_status_text(value)
    return any(term in normalized for term in ("not retract", "not retracted", "unretracted"))


def _status_says_withdrawn(value: str) -> bool:
    normalized = _normalize_status_text(value)
    return "withdraw" in normalized and not any(
        term in normalized for term in ("not withdraw", "not withdrawn", "unwithdrawn")
    )


def _status_mentions_withdrawal(value: str) -> bool:
    normalized = _normalize_status_text(value)
    return "withdraw" in normalized or "unwithdrawn" in normalized


def _status_says_not_withdrawn(value: str) -> bool:
    normalized = _normalize_status_text(value)
    return any(term in normalized for term in ("not withdraw", "not withdrawn", "unwithdrawn"))


def _status_says_expression_of_concern(value: str) -> bool:
    return "expression of concern" in _normalize_status_text(value)


def _status_says_correction(value: str) -> bool:
    normalized = _normalize_status_text(value)
    if any(
        relation in normalized
        for relation in ("erratum in", "correction in", "corrected in", "is corrected by")
    ):
        return False
    return any(term in normalized.split() for term in ("correction", "corrigendum", "erratum"))


def _normalize_status_text(value: object) -> str:
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(value or ""))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _metadata_status_text(value: object) -> str:
    if isinstance(value, Mapping):
        return " ".join(
            part
            for key, nested in value.items()
            for part in (_clean_metadata_value(str(key)), _metadata_status_text(nested))
            if part
        )[:500]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return " ".join(_metadata_status_text(item) for item in value)[:500]
    return _clean_metadata_value(value)


def _merge_unsafe_bool(left: object, right: object) -> bool | None:
    values = (left, right)
    if True in values:
        return True
    if False in values:
        return False
    return None


def _semantic_scholar_id(hit: CorpusHit) -> str:
    raw = str(hit.metadata.get("semantic_scholar_id") or "").strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", raw):
        return raw.casefold()
    prefixed = re.fullmatch(r"(?i)corpusid:(\d+)", raw)
    corpus_id = prefixed.group(1) if prefixed is not None else raw
    if corpus_id.isdigit() and int(corpus_id) > 0 and (
        prefixed is not None or "semantic_scholar" in hit.source.casefold()
    ):
        return f"CorpusID:{corpus_id}"
    return ""
