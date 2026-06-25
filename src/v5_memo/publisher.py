from __future__ import annotations

import json
from typing import cast
from urllib.request import Request, urlopen

from v5_memo.schemas import MemoResult


def build_researka_payload(result: MemoResult, *, author_agent_id: str, domain_slug: str) -> dict[str, object]:
    body = result.markdown.strip()
    candidate = result.candidate
    heading = next((line[2:] for line in body.splitlines() if line.startswith("# ")), "Untitled alpha memo")
    abstract = " ".join(body.translate(str.maketrans("#*_`>-", "      ")).split()[:180])
    return {
        "title": heading.replace("Alpha memo: ", "", 1).strip(),
        "abstract": abstract,
        "author_agent_id": author_agent_id,
        "author_agent_slug": author_agent_id,
        "article_type": "alpha_memo",
        "artifact_type": "alpha_memo",
        "domain_slug": domain_slug,
        "body_markdown": body,
        "source_bundle": [{"title": hit.title, "doi": hit.doi or "", "url": hit.url, "source": hit.source} for hit in result.receipts],
        "metadata": {"receipt_ids": list(candidate.receipt_ids), "score": candidate.score},
    }

def submit_researka(payload: dict[str, object], *, agent_key: str, api_base: str = "https://api.researka.org", timeout: float = 60.0) -> dict[str, object]:
    headers = {"Content-Type": "application/json", "X-Agent-Key": agent_key}
    req = Request(f"{api_base.rstrip('/')}/v1/research-objects", data=json.dumps(payload).encode(), method="POST", headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return cast(dict[str, object], data) if isinstance(data, dict) else {"response": data}
