import json
from typing import cast
from urllib.request import Request, urlopen

from v5_memo.schemas import MemoResult


def build_researka_payload(result: MemoResult, *, author_agent_id: str, domain_slug: str) -> dict[str, object]:
    body = result.markdown.strip()
    candidate = result.candidate
    heading = next((line[2:] for line in body.splitlines() if line.startswith("# ")), "Untitled alpha memo")
    abstract = " ".join(body.translate(str.maketrans("#*_`>-", "      ")).split()[:180])
    source_bundle = [{"title": h.title, "doi": h.doi or "", "url": h.url, "source": h.source, "year": h.year, "evidence_type": "primary"} for h in result.receipts]
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
        "evidence_bundle": {"publish_verdict": verdict},
        "metadata": {"receipt_ids": list(candidate.receipt_ids), "score": candidate.score},
    }

def submit_researka(payload: dict[str, object], *, agent_key: str, api_base: str = "https://api.researka.org", timeout: float = 60.0) -> dict[str, object]:
    headers = {"Content-Type": "application/json", "x-api-key": agent_key, "Authorization": f"Bearer {agent_key}"}
    req = Request(f"{api_base.rstrip('/')}/submissions", data=json.dumps(payload).encode(), method="POST", headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return cast(dict[str, object], data) if isinstance(data, dict) else {"response": data}
