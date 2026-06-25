import json
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


def build_researka_payload(result: MemoResult, *, author_agent_id: str, domain_slug: str) -> dict[str, object]:
    body = result.markdown.strip()
    candidate = result.candidate
    heading = next((line[2:] for line in body.splitlines() if line.startswith("# ")), "Untitled alpha memo")
    abstract = " ".join(body.translate(str.maketrans("#*_`>-", "      ")).split()[:180])
    source_bundle = [
        {
            "title": h.title,
            "doi": h.doi or "",
            "url": h.url,
            "source": h.source,
            "year": h.year,
            "evidence_type": "primary",
            "retrieval_evidence": _retrieval_evidence(h),
        }
        for h in result.receipts
    ]
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


def _fullraw_retrieval_coverage(receipts: Sequence[CorpusHit]) -> dict[str, object]:
    shards_searched = 0
    sources: set[str] = set()
    search_passes: set[str] = set()
    auth_required = False
    authenticated = False
    fullraw_count = 0
    for hit in receipts:
        if hit.source.startswith("fullraw:"):
            fullraw_count += 1
        raw_receipt = hit.metadata.get("shard_receipt")
        receipt = raw_receipt if isinstance(raw_receipt, Mapping) else {}
        shards_searched = max(shards_searched, _int_value(receipt.get("shards_searched")))
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
    req = Request(f"{api_base.rstrip('/')}/submissions", data=json.dumps(payload).encode(), method="POST", headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return cast(dict[str, object], data) if isinstance(data, dict) else {"response": data}
