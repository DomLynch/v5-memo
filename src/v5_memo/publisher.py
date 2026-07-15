import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from v5_memo.evidence import source_artifact_type
from v5_memo.schemas import ClaimCard, CorpusHit, MemoResult

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
_BOLD_SECTION_RE = re.compile(r"^\*\*(?P<heading>[^*]+):\*\*(?:\s|$)")
_NAMED_STUDY_TITLE_RE = re.compile(r"^[A-Z][A-Z0-9-]{2,15}\s+(?:Study|Program):")
_TITLE_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_DANGLING_TITLE_TAIL_RE = re.compile(
    r":\s*(?:a|an|the|and|or|of|in|on|to|for|with|during|after|before|upon)?$",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"([.!?])(?:\s|$)")
_TITLE_ROLE_TERMS = frozenset({"boundary", "context", "mechanism", "outcome", "promise", "receipt"})
_TOPIC_CONTEXT_TERMS = frozenset({
    "adaptation",
    "adaptations",
    "adult",
    "adults",
    "aging",
    "exercise",
    "human",
    "humans",
    "older",
    "outcome",
    "outcomes",
    "performance",
    "resistance",
    "response",
    "responses",
    "strength",
    "studies",
    "study",
    "training",
    "trial",
    "trials",
})
_GENERIC_OUTCOME_TERMS = frozenset({"", "outcome", "outcomes", "unspecified"})
_AUTO_THESIS_TITLE_PHRASES = (
    " may have a ",
    " may be hiding a ",
    " bridge between ",
    " boundary condition:",
)
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
_RESEARKA_EVIDENCE_TYPES = frozenset({"primary", "review"})
_INTERNAL_SUBMISSION_SECTIONS = frozenset({
    "audit trail",
    "claim ledger",
    "evidence graph",
    "receipt roles",
    "safety note",
})
_INCOMPLETE_TITLE_ENDINGS = frozenset({
    "and",
    "or",
    "of",
    "in",
    "on",
    "to",
    "for",
    "with",
    "during",
    "after",
    "before",
    "upon",
    "the",
    "a",
    "an",
    "acute",
    "adaptive",
    "cardiovascular",
    "chronic",
    "clinical",
    "inflammatory",
    "metabolic",
    "mitochondrial",
    "oxidative",
    "skeletal",
})
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


def build_researka_payload(
    result: MemoResult,
    *,
    author_agent_id: str,
    domain_slug: str,
    parent_submission_id: str = "",
) -> dict[str, object]:
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
    payload: dict[str, object] = {
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
    if parent_submission_id.strip():
        payload["parent_submission_id"] = parent_submission_id.strip()
    return payload


def _submission_markdown(markdown: str) -> str:
    plain = _MARKED_DOI_RE.sub(lambda match: match.group("doi").rstrip(".,;:*_`"), markdown)
    plain = _CODE_DOI_RE.sub(lambda match: match.group(1).rstrip(".,;:"), plain)
    return _strip_internal_submission_sections(_DOI_LABEL_RE.sub(r"\1 -", plain))


def _strip_internal_submission_sections(markdown: str) -> str:
    lines: list[str] = []
    skipping = False
    for line in markdown.splitlines():
        stripped = line.strip()
        bold_heading = _BOLD_SECTION_RE.match(stripped)
        heading = (
            bold_heading.group("heading").casefold()
            if bold_heading
            else stripped.removeprefix("## ").rstrip(":").casefold()
        )
        is_heading = stripped.startswith("## ") or bold_heading is not None
        if is_heading:
            skipping = heading in _INTERNAL_SUBMISSION_SECTIONS
        if not skipping:
            lines.append(line)
    return "\n".join(lines).strip()


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
    if _NAMED_STUDY_TITLE_RE.match(raw):
        return _clip_title(raw)
    direct_human = _direct_human_claim_cards(result)
    endpoint_title = _endpoint_heterogeneity_title(result, direct_human)
    if endpoint_title:
        raw = endpoint_title
    elif _query_like_title(raw) or _bridge_only_title(raw, result.candidate.bridge_terms):
        raw = _bundle_title(result) or raw
    if _incomplete_title(raw):
        raw = _receipt_title(result) or _first_sentence(result.candidate.thesis) or result.candidate.topic
    if _query_like_title(raw) or _non_article_title(raw):
        raw = _first_sentence(result.candidate.thesis) or result.candidate.topic
    if _bridge_only_title(raw, result.candidate.bridge_terms):
        raw = _bundle_title(result) or _receipt_title(result) or raw
    if _generic_scope_title(raw, result.candidate.topic):
        raw = _bundle_title(result) or raw
    if (
        _query_like_title(raw)
        or _non_article_title(raw)
        or _auto_thesis_title(raw)
        or _incomplete_title(raw)
    ):
        raw = _bundle_title(result) or _receipt_title(result) or result.candidate.topic
    return _clip_title(raw)


def _query_like_title(title: str) -> bool:
    tokens = _TITLE_TOKEN_RE.findall(title.casefold())
    return len(tokens) < 4 or title.count("/") >= 2 or len(set(tokens) & _TITLE_ROLE_TERMS) >= 2


def _bridge_only_title(title: str, bridge_terms: Sequence[str]) -> bool:
    tokens = set(_TITLE_TOKEN_RE.findall(title.casefold()))
    bridge = {term.casefold() for term in bridge_terms}
    return 0 < len(tokens) <= 4 and tokens <= bridge


def _generic_scope_title(title: str, topic: str) -> bool:
    clean = " ".join(title.casefold().split())
    topic_tokens = set(_TITLE_TOKEN_RE.findall(topic.casefold()))
    return (
        "training outcomes in human studies" in clean
        and ("adaptation" in topic_tokens or "adaptations" in topic_tokens)
    )


def _bundle_title(result: MemoResult) -> str:
    direct_human = _direct_human_claim_cards(result)
    endpoint_title = _endpoint_heterogeneity_title(result, direct_human)
    if endpoint_title:
        return endpoint_title
    boundary_title = _boundary_bundle_title(result, direct_human)
    outcomes: set[str] = set()
    for card in direct_human:
        outcome = " ".join(_TITLE_TOKEN_RE.findall(card.outcome.casefold()))
        if outcome not in _GENERIC_OUTCOME_TERMS:
            outcomes.add(outcome)
    if len(direct_human) < 2 or len(outcomes) < 2:
        if boundary_title:
            return boundary_title
        return ""
    intervention = _topic_intervention_title(result.candidate.topic)
    if not intervention:
        return ""
    topic_tokens = set(_TITLE_TOKEN_RE.findall(result.candidate.topic.casefold()))
    training_terms = {"exercise", "training", "resistance", "strength"}
    if "adaptation" in topic_tokens or "adaptations" in topic_tokens:
        outcome_tokens = {
            token
            for outcome in outcomes
            for token in _TITLE_TOKEN_RE.findall(outcome)
        }
        outcome_label = (
            "Strength Training Adaptation"
            if topic_tokens & {"resistance", "strength"}
            else "Training Adaptation"
        )
        if outcome_tokens & {"hypertrophy", "thickness"}:
            contrast_label = "Muscle Thickness" if "thickness" in outcome_tokens else "Hypertrophy"
            return f"{intervention}: {contrast_label} vs {outcome_label}"
        if outcome_tokens & {"performance", "recovery"}:
            contrast_label = "Recovery" if "recovery" in outcome_tokens else "Performance"
            return f"{intervention}: {contrast_label} and {outcome_label}"
        if boundary_title:
            return boundary_title
        return f"{intervention} and {outcome_label}"
    outcome_label = "Training Outcomes" if topic_tokens & training_terms else "Outcomes"
    return f"{intervention} and {outcome_label} in Human Studies"


def _boundary_bundle_title(result: MemoResult, direct_human: Sequence[ClaimCard]) -> str:
    if len(direct_human) < 2 or not any(card.role == "boundary" for card in direct_human):
        return ""
    intervention = _topic_intervention_title(result.candidate.topic)
    if not intervention:
        return ""
    topic_tokens = set(_TITLE_TOKEN_RE.findall(result.candidate.topic.casefold()))
    if topic_tokens & {"adaptation", "adaptations", "exercise", "resistance", "strength", "training"}:
        return f"{intervention}: Training Adaptation With Boundary Evidence"
    return f"{intervention}: Boundary Evidence Across Human Receipts"


def _direct_human_claim_cards(result: MemoResult) -> list[ClaimCard]:
    return [
        card
        for card in result.candidate.claim_cards
        if card.population.casefold() == "human" and card.support_type.casefold() == "direct"
    ]


def _endpoint_heterogeneity_title(result: MemoResult, direct_human: Sequence[ClaimCard]) -> str:
    proxy_terms = {"acute", "damage", "delayed", "early", "immediate", "inflammation", "pain", "short", "stress"}
    has_proxy = False
    has_directional_endpoint = False
    for card in result.candidate.claim_cards:
        if card not in direct_human:
            continue
        outcome_terms = set(_TITLE_TOKEN_RE.findall(card.outcome.casefold()))
        direction = card.direction.casefold()
        if direction == "proxy" or (card.role == "boundary" and outcome_terms & proxy_terms):
            has_proxy = True
        elif direction not in {"proxy", "unclear"}:
            has_directional_endpoint = True
    if not has_proxy or not has_directional_endpoint:
        return ""
    intervention = _topic_intervention_title(result.candidate.topic)
    if not intervention:
        return ""
    topic_tokens = set(_TITLE_TOKEN_RE.findall(result.candidate.topic.casefold()))
    adaptation_label = (
        "Chronic Training Adaptation"
        if topic_tokens & {"adaptation", "adaptations", "exercise", "resistance", "strength", "training"}
        else "Primary Outcome"
    )
    return f"{intervention}: Endpoint Heterogeneity in Acute Proxy vs {adaptation_label}"


def _topic_intervention_title(topic: str) -> str:
    tokens = _TITLE_TOKEN_RE.findall(topic)
    kept: list[str] = []
    for token in tokens:
        if token.casefold() in _TOPIC_CONTEXT_TERMS and kept:
            break
        kept.append(token)
        if len(kept) >= 5:
            break
    return _title_case_tokens(kept)


def _title_case_tokens(tokens: Sequence[str]) -> str:
    small = {"and", "for", "in", "of", "on", "the", "to", "with"}
    words: list[str] = []
    for index, token in enumerate(tokens):
        word = token.upper() if token.isupper() and len(token) <= 5 else token.capitalize()
        if index > 0 and token.casefold() in small:
            word = token.casefold()
        words.append(word)
    return " ".join(words).strip()


def _non_article_title(title: str) -> bool:
    clean = " ".join(title.casefold().split())
    return any(phrase in clean for phrase in _NON_ARTICLE_TITLE_PHRASES)


def _auto_thesis_title(title: str) -> bool:
    clean = " ".join(title.casefold().split())
    return any(phrase in clean for phrase in _AUTO_THESIS_TITLE_PHRASES)


def _incomplete_title(title: str) -> bool:
    tokens = _TITLE_TOKEN_RE.findall(title.casefold())
    if not tokens:
        return True
    return tokens[-1] in _INCOMPLETE_TITLE_ENDINGS


def _receipt_title(result: MemoResult) -> str:
    for hit in result.receipts:
        title = " ".join(hit.title.split()).strip(" .")
        if title and not _non_article_title(title) and not _incomplete_title(title):
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
    if len(clean) > 120:
        clean = clean[:120].rsplit(" ", 1)[0].rstrip(" ,;:")
    trimmed = _DANGLING_TITLE_TAIL_RE.sub("", clean).rstrip(" ,;:")
    return trimmed or clean


def _retrieval_evidence(hit: CorpusHit) -> dict[str, object]:
    return {
        key: value
        for key in _RETRIEVAL_EVIDENCE_KEYS
        if (value := hit.metadata.get(key)) not in (None, "", {}, ())
    }


def _abstract_from_markdown(markdown: str) -> str:
    plain = " ".join(markdown.translate(str.maketrans("#*_`>", "     ")).split())
    plain = _alpha_disclaimer_first(plain)
    if len(plain) <= 900:
        return plain
    clipped = plain[:900].rstrip()
    last_end = 0
    for match in _SENTENCE_END.finditer(clipped):
        last_end = match.end(1)
    return clipped[:last_end].strip() if last_end >= 80 else clipped.rstrip(" ,;:") + "."


def _alpha_disclaimer_first(text: str) -> str:
    disclaimer = "Hypothesis-level alpha signal; not clinical advice."
    normalized = "Hypothesis level alpha signal; not clinical advice."
    if disclaimer not in text and normalized not in text:
        return text
    clean = " ".join(text.split()).strip()
    if clean.startswith(disclaimer):
        return clean
    without_disclaimer = clean.replace(disclaimer, "", 1).replace(normalized, "", 1).strip()
    return f"{disclaimer} {without_disclaimer}"


def _source_bundle_entry(hit: CorpusHit) -> dict[str, object]:
    artifact_type = source_artifact_type(hit)
    if artifact_type != "article":
        raise ValueError(
            f"unsupported_source_bundle_artifact_type:{hit.receipt_id}:{artifact_type}"
        )
    evidence_type = _source_evidence_type(hit)
    if evidence_type not in _RESEARKA_EVIDENCE_TYPES:
        raise ValueError(
            f"unsupported_source_bundle_evidence_type:{hit.receipt_id}:{evidence_type}"
        )
    entry: dict[str, object] = {
        "title": hit.title,
        "url": hit.url,
        "source": hit.source,
        "source_type": _source_type(hit),
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
    text = f"{hit.title} {hit.metadata.get('evidence_type', '')}".casefold()
    return "review" if any(term in text for term in ("review", "meta-analysis", "systematic")) else "primary"


def _source_type(hit: CorpusHit) -> str:
    raw = hit.source.split(":", 1)[-1] or hit.source or "openalex"
    clean = re.sub(r"[^a-z0-9_-]+", "_", raw.casefold()).strip("_")
    return (clean or "openalex")[:40]


def _source_excerpt(hit: CorpusHit) -> str:
    text = " ".join((hit.abstract or hit.title).split()).strip()
    if len(text) >= 20:
        return text[:5000]
    fallback = " ".join(part for part in (hit.title, hit.venue or "", hit.source) if part).strip()
    if len(fallback) < 20:
        fallback = f"Source receipt {hit.receipt_id} from {hit.source}."
    return fallback[:5000]


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
    max_retries: int = 0,
) -> dict[str, object]:
    headers = {"Content-Type": "application/json", "x-api-key": agent_key, "Authorization": f"Bearer {agent_key}"}
    agent_slug = payload.get("author_agent_slug") or payload.get("author_agent_id")
    if isinstance(agent_slug, str) and agent_slug.strip():
        headers["X-Agent-Slug"] = agent_slug.strip()
    url = submit_url.strip() or f"{api_base.rstrip('/')}/submissions"
    body = json.dumps(payload).encode()
    attempts = max(0, max_retries) + 1
    for attempt in range(attempts):
        req = Request(url, data=body, method="POST", headers=headers)
        try:
            with urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            break
        except HTTPError as exc:
            if exc.code != 429 or attempt >= attempts - 1:
                raise
            time.sleep(_retry_after_seconds(exc))
    return cast(dict[str, object], data) if isinstance(data, dict) else {"response": data}


def _retry_after_seconds(exc: HTTPError) -> float:
    header = exc.headers.get("Retry-After", "") if exc.headers is not None else ""
    try:
        parsed = float(header)
    except ValueError:
        parsed = 1.0
    return max(0.0, min(parsed, 10.0))


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
        decision = str(last.get("decision") or "")
        if decision in {"reject", "revise"}:
            return last
        if decision == "accept" and researka_publication_is_minted(last):
            return last
        if last.get("status") == "complete" and not decision:
            return last
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = {**last, "status": "timeout"}
            if decision == "accept":
                timed_out["publication_pending"] = True
            return timed_out
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
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except HTTPError as exc:
        if exc.code != 403:
            raise
        publication = fetch_researka_publication(
            publication_id,
            api_base=api_base,
            timeout=timeout,
        )
        if researka_publication_visibility(publication) != "listed":
            raise
        return {
            "id": publication_id,
            "public_visibility": "listed",
            "updated": False,
            "verified": True,
        }
    return cast(dict[str, object], data) if isinstance(data, dict) else {"response": data}


def fetch_researka_publication(
    publication_id: str,
    *,
    api_base: str = "https://api.researka.org",
    timeout: float = 30.0,
) -> dict[str, object]:
    url = f"{api_base.rstrip('/')}/publications/{publication_id}"
    req = Request(url, method="GET", headers={"Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return cast(dict[str, object], data) if isinstance(data, dict) else {"response": data}


def researka_publication_visibility(publication: Mapping[str, object]) -> str:
    nested = publication.get("publication")
    record = nested if isinstance(nested, Mapping) else publication
    if record.get("publicVisible") is True or record.get("public_visible") is True:
        return "listed"
    metadata = record.get("metadata")
    metadata_record = metadata if isinstance(metadata, Mapping) else {}
    for source in (record, metadata_record):
        for key in ("publicVisibility", "public_visibility", "visibility"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().casefold()
    return ""


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


def researka_publication_is_minted(decision: Mapping[str, object]) -> bool:
    raw_publication = decision.get("publication")
    if not isinstance(raw_publication, Mapping):
        return False
    doi = raw_publication.get("doi")
    doi_status = raw_publication.get("doi_status") or raw_publication.get("doiStatus")
    return bool(
        researka_publication_id(decision)
        and isinstance(doi, str)
        and doi.strip()
        and str(doi_status or "").casefold() == "minted"
    )
