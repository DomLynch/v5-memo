import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from v5_memo.schemas import CorpusHit, MemoResult

_RETRIEVAL_EVIDENCE_KEYS = (
    "shard_receipt",
    "fullraw_search_receipt",
    "search_pass",
    "search_variant",
    "rank_mode",
)
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_CODE_DOI_RE = re.compile(r"`(10\.\d{4,9}/[^`\s]+)`", re.IGNORECASE)
_MARKED_DOI_RE = re.compile(
    r"(?P<wrap>`|\*\*|__)(?P<doi>10\.\d{4,9}/[^\s`<>()\[\]{}\"']+?)(?P=wrap)",
    re.IGNORECASE,
)
_DOI_LABEL_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'\])}>,;:`]+):", re.IGNORECASE)
_TITLE_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_SENTENCE_END = re.compile(r"([.!?])(?:\s|$)")
_TITLE_ROLE_TERMS = frozenset({"boundary", "context", "mechanism", "outcome", "promise", "receipt"})
_NON_ARTICLE_TITLE_PHRASES = (
    "additional file",
    "supplementary file",
    "supplemental file",
    "supplementary material",
    "supplemental material",
    "supplementary data",
    "supplemental data",
    "data sheet",
    "dataset",
    "appendix",
    "corrigendum",
    "erratum",
    "comment on",
    "reply to",
)
_SUPPLEMENTAL_RECEIPT_PHRASES = (
    *_NON_ARTICLE_TITLE_PHRASES,
    "figshare",
    "dryad",
    "zenodo",
)
_CONFERENCE_RECEIPT_PHRASES = (
    "conference abstract",
    "meeting abstract",
    "poster abstract",
    "abstract supplement",
)
_SUBMIT_KEY_ENV_NAMES = (
    "V5_MEMO_RESEARKA_AGENT_KEY",
    "V5_MEMO_RESEARKA_API_KEY",
    "RESEARKA_API_KEY_V5",
    "RESEARKA_AGENT_KEY",
    "RESEARKA_API_KEY",
)


@dataclass(frozen=True, slots=True)
class ResearkaSubmitConfig:
    agent_key: str
    agent_id: str
    domain_slug: str
    api_base: str
    submit_url: str

    @property
    def missing(self) -> tuple[str, ...]:
        out: list[str] = []
        if not self.agent_key:
            out.append("V5_MEMO_RESEARKA_AGENT_KEY")
        if not self.agent_id:
            out.append("V5_MEMO_RESEARKA_AGENT_ID")
        if not self.domain_slug:
            out.append("V5_MEMO_RESEARKA_DOMAIN_SLUG")
        return tuple(out)


def load_researka_submit_config(
    *,
    agent_id: str = "",
    domain_slug: str = "",
    api_base: str = "",
    submit_url: str = "",
    environ: Mapping[str, str] | None = None,
) -> ResearkaSubmitConfig:
    env = os.environ if environ is None else environ
    agent_key = next(
        (value.strip() for name in _SUBMIT_KEY_ENV_NAMES if (value := env.get(name, "")).strip()),
        "",
    )
    return ResearkaSubmitConfig(
        agent_key=agent_key,
        agent_id=agent_id.strip() or env.get("V5_MEMO_RESEARKA_AGENT_ID", "").strip(),
        domain_slug=domain_slug.strip() or env.get("V5_MEMO_RESEARKA_DOMAIN_SLUG", "").strip(),
        api_base=(api_base.strip() or env.get("V5_MEMO_RESEARKA_API_BASE", "").strip() or "https://api.researka.org"),
        submit_url=(submit_url.strip() or env.get("V5_MEMO_RESEARKA_SUBMIT_URL", "").strip() or env.get("RESEARKA_SUBMIT_URL", "").strip()),
    )


def build_researka_payload(result: MemoResult, *, author_agent_id: str, domain_slug: str) -> dict[str, object]:
    body = _submission_markdown(result.markdown.strip())
    candidate = result.candidate
    heading = next((line[2:] for line in body.splitlines() if line.startswith("# ")), "Untitled alpha memo")
    title = _submission_title(result, heading)
    body = _replace_heading(body, title)
    body = _append_alpha_disclaimer(body)
    abstract = _abstract_from_markdown(body)
    source_bundle = [_source_bundle_entry(hit) for hit in result.receipts]
    fullraw_coverage = _fullraw_retrieval_coverage(result.receipts)
    verdict = {"decision": "ready_to_publish", "publish_tier": "TIER_1", "maturity_level": "L5", "confidence_label": "evidence_backed_signal", "blockers": [], "axes": {"bound_receipts": len(source_bundle)}}
    return {
        "title": title,
        "abstract": abstract,
        "author_agent_id": author_agent_id,
        "author_agent_slug": author_agent_id,
        "article_type": "alpha_memo",
        "artifact_type": "alpha_memo",
        "domain_slug": domain_slug,
        "body_markdown": body,
        "source_bundle": source_bundle,
        "evidence_bundle": {"publish_verdict": verdict, "fullraw_retrieval_coverage": fullraw_coverage},
        "metadata": {"receipt_ids": list(candidate.receipt_ids), "score": candidate.score},
    }


def _submission_markdown(markdown: str) -> str:
    plain = _MARKED_DOI_RE.sub(lambda match: match.group("doi").rstrip(".,;:*_`"), markdown)
    plain = _CODE_DOI_RE.sub(lambda match: match.group(1).rstrip(".,;:"), plain)
    return _DOI_LABEL_RE.sub(r"\1 -", plain)


def _replace_heading(markdown: str, title: str) -> str:
    lines = markdown.splitlines()
    if lines and lines[0].startswith("# Alpha memo:"):
        lines[0] = f"# Alpha memo: {title}"
        return "\n".join(lines).strip()
    return markdown


def _append_alpha_disclaimer(markdown: str) -> str:
    if not markdown.startswith("# Alpha memo:") or "not clinical advice" in markdown.casefold():
        return markdown
    lines = markdown.splitlines()
    lines.insert(2, "Hypothesis-level alpha signal; not clinical advice.")
    return "\n".join(lines).strip()


def _submission_title(result: MemoResult, heading: str) -> str:
    raw = heading.replace("Alpha memo: ", "", 1).strip()
    if _query_like_title(raw) or _non_article_title(raw):
        raw = _first_sentence(result.candidate.thesis) or result.candidate.topic
    if _bridge_only_title(raw, result.candidate.bridge_terms):
        raw = _receipt_title(result) or raw
    if _query_like_title(raw) or _non_article_title(raw):
        raw = _receipt_title(result) or result.candidate.topic
    return _clip_title(raw)


def _query_like_title(title: str) -> bool:
    tokens = _TITLE_TOKEN_RE.findall(title.casefold())
    return len(tokens) < 4 or title.count("/") >= 2 or len(set(tokens) & _TITLE_ROLE_TERMS) >= 2


def _bridge_only_title(title: str, bridge_terms: Sequence[str]) -> bool:
    tokens = set(_TITLE_TOKEN_RE.findall(title.casefold()))
    bridge = {term.casefold() for term in bridge_terms}
    return 0 < len(tokens) <= 4 and tokens <= bridge


def _non_article_title(title: str) -> bool:
    clean = " ".join(title.casefold().split())
    return any(phrase in clean for phrase in _NON_ARTICLE_TITLE_PHRASES)


def _receipt_title(result: MemoResult) -> str:
    for hit in result.receipts:
        title = " ".join(hit.title.split()).strip(" .")
        if title and not _non_article_title(title):
            return title
    return ""


def _first_sentence(text: str) -> str:
    clean = " ".join(text.split()).strip()
    if not clean:
        return ""
    for match in _SENTENCE_END.finditer(clean):
        return clean[: match.end(1)].strip()
    return clean


def _clip_title(title: str) -> str:
    clean = " ".join(title.split()).strip(" #`*_")
    if len(clean) <= 120:
        return clean
    clipped = clean[:120].rsplit(" ", 1)[0].rstrip(" ,;:")
    return clipped or clean[:120].rstrip(" ,;:")


def _retrieval_evidence(hit: CorpusHit) -> dict[str, object]:
    return {
        key: value
        for key in _RETRIEVAL_EVIDENCE_KEYS
        if (value := hit.metadata.get(key)) not in (None, "", {}, ())
    }


def _abstract_from_markdown(markdown: str) -> str:
    plain = " ".join(markdown.translate(str.maketrans("#*_`>-", "      ")).split())
    if len(plain) <= 900:
        return plain
    clipped = plain[:900].rstrip()
    last_end = 0
    for match in _SENTENCE_END.finditer(clipped):
        last_end = match.end(1)
    return clipped[:last_end].strip() if last_end >= 80 else clipped.rstrip(" ,;:") + "."


def _source_bundle_entry(hit: CorpusHit) -> dict[str, object]:
    evidence_type = _source_evidence_type(hit)
    entry: dict[str, object] = {
        "title": hit.title,
        "url": hit.url,
        "source": hit.source,
        "source_type": _source_type(hit, evidence_type),
        "year": hit.year,
        "evidence_type": evidence_type,
        "excerpt": _source_excerpt(hit),
        "retrieval_evidence": _retrieval_evidence(hit),
    }
    doi = _valid_doi(hit.doi)
    if doi:
        entry["doi"] = doi
    elif pmid := _pmid(hit):
        entry["pmid"] = pmid
        entry["id"] = pmid
    return entry


def _source_evidence_type(hit: CorpusHit) -> str:
    text = _receipt_descriptor(hit)
    if any(phrase in text for phrase in _SUPPLEMENTAL_RECEIPT_PHRASES):
        return "supplemental"
    if any(phrase in text for phrase in _CONFERENCE_RECEIPT_PHRASES):
        return "conference_abstract"
    doi = str(hit.doi or hit.hit_id or "").casefold()
    if "10.6084/m9.figshare" in doi:
        return "supplemental"
    if "10.1096/fasebj" in doi and ".s1." in doi:
        return "conference_abstract"
    return "primary"


def _source_type(hit: CorpusHit, evidence_type: str) -> str:
    if evidence_type != "primary":
        return evidence_type[:40]
    raw = hit.source.split(":", 1)[-1] or hit.source or "openalex"
    clean = re.sub(r"[^a-z0-9_-]+", "_", raw.casefold()).strip("_")
    return (clean or "openalex")[:40]


def _source_excerpt(hit: CorpusHit) -> str:
    text = " ".join((hit.abstract or hit.title).split()).strip()
    if len(text) >= 20:
        return text[:5000]
    fallback = " ".join(part for part in (hit.title, hit.venue or "", hit.source) if part).strip()
    return (fallback or hit.receipt_id)[:5000]


def _receipt_descriptor(hit: CorpusHit) -> str:
    return " ".join(
        part.casefold()
        for part in (
            hit.title,
            hit.abstract,
            hit.venue or "",
            hit.source,
            str(hit.doi or ""),
            str(hit.hit_id or ""),
            " ".join(str(value) for value in hit.metadata.values()),
        )
        if part
    )


def _valid_doi(value: object) -> str:
    doi = str(value or "").strip().rstrip(".,;")
    return doi if _DOI_RE.match(doi) and "(" not in doi and ")" not in doi else ""


def _pmid(hit: CorpusHit) -> str:
    raw = hit.metadata.get("pmid")
    pmid = str(raw or (hit.hit_id if hit.hit_id.isdigit() else "")).strip()
    return pmid if pmid.isdigit() else ""


def _fullraw_retrieval_coverage(receipts: Sequence[CorpusHit]) -> dict[str, object]:
    shards_searched = 0
    sources: set[str] = set()
    search_passes: set[str] = set()
    auth_required = False
    authenticated = False
    partial_shard_search = False
    sweep_failed_shards = 0
    fullraw_count = 0
    for hit in receipts:
        if hit.source.startswith("fullraw:"):
            fullraw_count += 1
        raw_receipt = hit.metadata.get("shard_receipt")
        receipt = raw_receipt if isinstance(raw_receipt, Mapping) else {}
        shards_searched = max(shards_searched, _int_value(receipt.get("shards_searched")))
        partial_shard_search = partial_shard_search or receipt.get("partial_shard_search") is True
        sweep_failed_shards += _int_value(receipt.get("sweep_failed_shards"))
        auth_required = auth_required or receipt.get("auth_required") is True
        authenticated = authenticated or receipt.get("authenticated") is True
        raw_sources = receipt.get("sources_searched")
        if isinstance(raw_sources, Mapping):
            sources.update(str(source) for source, count in raw_sources.items() if _int_value(count) > 0)
        raw_pass = hit.metadata.get("search_pass")
        if isinstance(raw_pass, str) and raw_pass:
            search_passes.add(raw_pass)
        raw_search_receipt = hit.metadata.get("fullraw_search_receipt")
        search_receipt = raw_search_receipt if isinstance(raw_search_receipt, Mapping) else {}
        raw_passes = search_receipt.get("search_passes")
        if isinstance(raw_passes, Sequence) and not isinstance(raw_passes, str):
            search_passes.update(str(item) for item in raw_passes if str(item))
    return {
        "receipt_count": fullraw_count,
        "auth_required": auth_required,
        "authenticated": authenticated,
        "shards_searched": shards_searched,
        "partial_shard_search": partial_shard_search,
        "sweep_failed_shards": sweep_failed_shards,
        "sources_searched": sorted(sources),
        "search_passes": sorted(search_passes),
    }


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def submit_researka(
    payload: dict[str, object],
    *,
    agent_key: str,
    api_base: str = "https://api.researka.org",
    submit_url: str = "",
    timeout: float = 60.0,
) -> dict[str, object]:
    headers = {"Content-Type": "application/json", "x-api-key": agent_key, "Authorization": f"Bearer {agent_key}"}
    agent_slug = payload.get("author_agent_slug") or payload.get("author_agent_id")
    if isinstance(agent_slug, str) and agent_slug.strip():
        headers["X-Agent-Slug"] = agent_slug.strip()
    url = submit_url.strip() or f"{api_base.rstrip('/')}/submissions"
    req = Request(url, data=json.dumps(payload).encode(), method="POST", headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return cast(dict[str, object], data) if isinstance(data, dict) else {"response": data}


def wait_researka_decision(
    submission_id: str,
    *,
    api_base: str = "https://api.researka.org",
    timeout_seconds: float = 300.0,
    poll_seconds: float = 5.0,
) -> dict[str, object]:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    last: dict[str, object] = {"status": "pending"}
    while True:
        try:
            last = fetch_researka_decision(submission_id, api_base=api_base)
        except HTTPError as exc:
            if exc.code != 404:
                raise
            last = {"status": "pending", "http_status": 404}
        if last.get("status") == "complete" or last.get("decision") in {"accept", "reject", "revise"}:
            return last
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {**last, "status": "timeout"}
        time.sleep(min(max(0.1, poll_seconds), remaining))


def fetch_researka_decision(
    submission_id: str,
    *,
    api_base: str = "https://api.researka.org",
    timeout: float = 30.0,
) -> dict[str, object]:
    url = f"{api_base.rstrip('/')}/submissions/{submission_id}/decision"
    req = Request(url, method="GET", headers={"Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return cast(dict[str, object], data) if isinstance(data, dict) else {"response": data}


def set_researka_public_visibility(
    publication_id: str,
    *,
    agent_key: str,
    api_base: str = "https://api.researka.org",
    visibility: str = "listed",
    timeout: float = 30.0,
) -> dict[str, object]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "x-api-key": agent_key,
        "Authorization": f"Bearer {agent_key}",
    }
    url = f"{api_base.rstrip('/')}/ops/publications/{publication_id}/visibility"
    req = Request(
        url,
        data=json.dumps({"visibility": visibility}).encode(),
        method="POST",
        headers=headers,
    )
    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return cast(dict[str, object], data) if isinstance(data, dict) else {"response": data}


def researka_submission_id(response: Mapping[str, object]) -> str:
    raw_submission = response.get("submission")
    if isinstance(raw_submission, Mapping):
        raw_id = raw_submission.get("id")
        if isinstance(raw_id, str):
            return raw_id
    for key in ("submission_id", "id"):
        raw_id = response.get(key)
        if isinstance(raw_id, str):
            return raw_id
    raw_job = response.get("job")
    if isinstance(raw_job, Mapping):
        raw_id = raw_job.get("target_object_id")
        if isinstance(raw_id, str):
            return raw_id
    return ""


def researka_publication_id(decision: Mapping[str, object]) -> str:
    raw_publication = decision.get("publication")
    if isinstance(raw_publication, Mapping):
        raw_id = raw_publication.get("publication_id") or raw_publication.get("id")
        if isinstance(raw_id, str):
            return raw_id
    for key in ("publication_id", "publicationId"):
        raw_id = decision.get(key)
        if isinstance(raw_id, str):
            return raw_id
    return ""
