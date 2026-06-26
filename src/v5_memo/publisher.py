import json
import re
from collections.abc import Mapping, Sequence
from typing import cast
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
_SENTENCE_END = re.compile(r"([.!?])(?:\s|$)")


def build_researka_payload(result: MemoResult, *, author_agent_id: str, domain_slug: str) -> dict[str, object]:
    body = result.markdown.strip()
    candidate = result.candidate
    heading = next((line[2:] for line in body.splitlines() if line.startswith("# ")), "Untitled alpha memo")
    abstract = _abstract_from_markdown(body)
    source_bundle = [_source_bundle_entry(hit) for hit in result.receipts]
    fullraw_coverage = _fullraw_retrieval_coverage(result.receipts)
    verdict = {"decision": "ready_to_publish", "publish_tier": "TIER_1", "maturity_level": "L5", "confidence_label": "evidence_backed_signal", "blockers": [], "axes": {"bound_receipts": len(source_bundle)}}
    return {
        "title": heading.replace("Alpha memo: ", "", 1).strip(),
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
    entry: dict[str, object] = {
        "title": hit.title,
        "url": hit.url,
        "source": hit.source,
        "year": hit.year,
        "evidence_type": "primary",
        "retrieval_evidence": _retrieval_evidence(hit),
    }
    doi = _valid_doi(hit.doi)
    if doi:
        entry["doi"] = doi
    elif pmid := _pmid(hit):
        entry["pmid"] = pmid
        entry["id"] = pmid
    return entry


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


def submit_researka(payload: dict[str, object], *, agent_key: str, api_base: str = "https://api.researka.org", timeout: float = 60.0) -> dict[str, object]:
    headers = {"Content-Type": "application/json", "x-api-key": agent_key, "Authorization": f"Bearer {agent_key}"}
    agent_slug = payload.get("author_agent_slug") or payload.get("author_agent_id")
    if isinstance(agent_slug, str) and agent_slug.strip():
        headers["X-Agent-Slug"] = agent_slug.strip()
    req = Request(f"{api_base.rstrip('/')}/submissions", data=json.dumps(payload).encode(), method="POST", headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return cast(dict[str, object], data) if isinstance(data, dict) else {"response": data}
