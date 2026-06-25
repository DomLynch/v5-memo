import json
from typing import cast
from urllib.request import Request, urlopen

from v5_memo.schemas import MemoResult


def build_researka_payload(result: MemoResult, *, author_agent_id: str, domain_slug: str) -> dict[str, object]:
    body = result.markdown.strip()
    candidate = result.candidate
    heading = next((line[2:] for line in body.splitlines() if line.startswith("# ")), "Untitled alpha memo")
    abstract = " ".join(body.translate(str.maketrans("#*_`>-", "      ")).split()[:180])
    source_bundle = [{"title": hit.title, "doi": hit.doi or "", "url": hit.url, "source": hit.source} for hit in result.receipts]
    return {
        "title": heading.replace("Alpha memo: ", "", 1).strip(),
        "abstract": abstract,
        "author_agent_id": author_agent_id,
        "author_agent_slug": author_agent_id,
        "agent_id": author_agent_id,
        "article_type": "alpha_memo",
        "artifact_type": "alpha_memo",
        "domain_slug": domain_slug,
        "topic": candidate.topic,
        "body_markdown": body,
        "markdown": body,
        "citations": source_bundle,
        "source_bundle": source_bundle,
        "novelty_score": candidate.novelty_score,
        "confidence_score": candidate.evidence_score,
        "metadata": {"receipt_ids": list(candidate.receipt_ids), "score": candidate.score, "domain_slug": domain_slug},
    }

def submit_researka(
    payload: dict[str, object],
    *,
    agent_key: str,
    api_base: str = "https://api.researka.org",
    submit_url: str = "",
    timeout: float = 60.0,
) -> dict[str, object]:
    headers = {"Content-Type": "application/json", "X-Agent-Key": agent_key}
    url = submit_url or f"{api_base.rstrip('/')}/submissions"
    req = Request(url, data=json.dumps(payload).encode(), method="POST", headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return cast(dict[str, object], data) if isinstance(data, dict) else {"response": data}
