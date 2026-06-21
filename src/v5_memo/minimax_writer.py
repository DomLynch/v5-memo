"""MiniMax-M3 planner and memo writer constrained by corpus receipts."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.request import Request, urlopen

from v5_memo.gate import candidate_alpha_tier
from v5_memo.schemas import CorpusHit, InsightCandidate

MINIMAX_BASE_URL = "https://api.minimax.io/anthropic"
MINIMAX_MODEL = "MiniMax-M3"
MINIMAX_KEY_ENV = "MINIMAX_API_KEY"
MINIMAX_V5_KEY_ENV = "V5_MEMO_MINIMAX_API_KEY"
MINIMAX_BASE_URL_ENV = "V5_MEMO_MINIMAX_BASE_URL"
MINIMAX_MODEL_ENV = "V5_MEMO_MINIMAX_MODEL"
MINIMAX_TIMEOUT_ENV = "V5_MEMO_MINIMAX_TIMEOUT_SECONDS"
MINIMAX_MAX_TOKENS_ENV = "V5_MEMO_MINIMAX_MAX_TOKENS"
MINIMAX_KEY_FILE = Path.home() / ".codex" / "secrets" / "minimax_api_key"
RECEIPT_ABSTRACT_CHAR_LIMIT = 1400
_REQUIRED_MEMO_SECTIONS = (
    "# Alpha memo:",
    "## Core signal",
    "## The 2+2=5 angle",
    "## Why this could matter",
    "## What would break the idea",
    "## Receipts",
    "## Safety note",
)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s<>()\[\]{}\"']+", re.IGNORECASE)
_DOI_TRAILING_PUNCTUATION = ".,;:*_`"
_TITLE_WORD_RE = re.compile(r"[a-z][a-z0-9]{2,}")
_TITLE_STOPWORDS = frozenset({
    "alpha", "memo", "and", "for", "from", "into", "may", "not", "the", "with",
    "without", "between", "versus", "under", "over", "through", "across",
    "one", "two", "three", "both", "same", "abstract", "readout", "readouts", "report", "reports",
    "signal", "signals", "effect", "effects", "tradeoff", "tradeoffs",
    "boundary", "condition", "conditions", "hypothesis", "discovery", "seed",
    "split", "splits", "diverge", "diverges", "divergence", "outlier",
    "bridge", "bridges", "receipt", "receipts",
    "while",
})


class HttpResponse(Protocol):
    def __enter__(self) -> HttpResponse: ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None: ...

    def read(self) -> bytes: ...


class RequestOpener(Protocol):
    def __call__(self, request: Request, timeout: float) -> HttpResponse: ...


class MemoScopeError(ValueError):
    """A memo title/framing used concepts outside the supplied receipts."""


class MemoFormatError(ValueError):
    """A memo omitted the required Markdown contract."""


class MiniMaxM3MemoWriter:
    """Anthropic-compatible MiniMax writer with receipt-preservation checks."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = MINIMAX_BASE_URL,
        model: str = MINIMAX_MODEL,
        timeout: float = 60.0,
        max_tokens: int = 1200,
        opener: RequestOpener | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("MINIMAX_API_KEY is required for MiniMax writer")
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._opener = opener or cast(RequestOpener, urlopen)

    @classmethod
    def from_env(cls, *, opener: RequestOpener | None = None) -> MiniMaxM3MemoWriter:
        return cls(
            api_key=load_minimax_api_key(),
            base_url=_env_string([MINIMAX_BASE_URL_ENV, "MINIMAX_BASE_URL"], MINIMAX_BASE_URL),
            model=_env_string([MINIMAX_MODEL_ENV, "MINIMAX_MODEL"], MINIMAX_MODEL),
            timeout=_env_float([MINIMAX_TIMEOUT_ENV], 60.0),
            max_tokens=_env_int([MINIMAX_MAX_TOKENS_ENV], 1200),
            opener=opener,
        )

    def render(self, candidate: InsightCandidate, receipts: Sequence[CorpusHit]) -> str:
        prompt = build_minimax_prompt(candidate, receipts)
        markdown = self._write(prompt, temperature=0.35)
        return validate_minimax_memo(markdown, receipts, candidate=candidate)

    def _write(self, prompt: str, *, temperature: float) -> str:
        return _strip_markdown_fence(
            call_minimax_m3(
                api_key=self._api_key,
                prompt=prompt,
                system=(
                    "You write concise research alpha memos. Use only the supplied receipts. "
                    "Do not add uncited mechanisms, claims, facts, statistics, or references."
                ),
                temperature=temperature,
                max_tokens=self._max_tokens,
                base_url=self._base_url,
                model=self._model,
                timeout=self._timeout,
                opener=self._opener,
            )
        )


class MiniMaxM3SearchPlanner:
    """Use MiniMax-M3 to propose sharper corpus search angles."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = MINIMAX_BASE_URL,
        model: str = MINIMAX_MODEL,
        timeout: float = 45.0,
        max_tokens: int = 700,
        opener: RequestOpener | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("MINIMAX_API_KEY is required for MiniMax planner")
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._opener = opener or cast(RequestOpener, urlopen)

    @classmethod
    def from_env(cls, *, opener: RequestOpener | None = None) -> MiniMaxM3SearchPlanner:
        return cls(
            api_key=load_minimax_api_key(),
            base_url=_env_string([MINIMAX_BASE_URL_ENV, "MINIMAX_BASE_URL"], MINIMAX_BASE_URL),
            model=_env_string([MINIMAX_MODEL_ENV, "MINIMAX_MODEL"], MINIMAX_MODEL),
            timeout=_env_float([MINIMAX_TIMEOUT_ENV], 45.0),
            max_tokens=_env_int([MINIMAX_MAX_TOKENS_ENV], 700),
            opener=opener,
        )

    def plan(self, *, topic: str, seed_queries: Sequence[str], limit: int = 8) -> list[str]:
        prompt = build_minimax_search_prompt(topic=topic, seed_queries=seed_queries, limit=limit)
        text = call_minimax_m3(
            api_key=self._api_key,
            prompt=prompt,
            system=(
                "You design high-recall, high-signal academic corpus search queries. "
                "Return only valid JSON."
            ),
            temperature=0.35,
            max_tokens=self._max_tokens,
            base_url=self._base_url,
            model=self._model,
            timeout=self._timeout,
            opener=self._opener,
        )
        planned = parse_minimax_queries(text, limit=limit)
        return _dedupe_queries([*planned, *seed_queries])


class MiniMaxM3CandidateSelector:
    """Use MiniMax-M3 as a taste judge over deterministic candidate bridges."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = MINIMAX_BASE_URL,
        model: str = MINIMAX_MODEL,
        timeout: float = 45.0,
        max_tokens: int = 700,
        opener: RequestOpener | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("MINIMAX_API_KEY is required for MiniMax selector")
        self._api_key = api_key.strip()
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._max_tokens = max_tokens
        self._opener = opener or cast(RequestOpener, urlopen)

    @classmethod
    def from_env(cls, *, opener: RequestOpener | None = None) -> MiniMaxM3CandidateSelector:
        return cls(
            api_key=load_minimax_api_key(),
            base_url=_env_string([MINIMAX_BASE_URL_ENV, "MINIMAX_BASE_URL"], MINIMAX_BASE_URL),
            model=_env_string([MINIMAX_MODEL_ENV, "MINIMAX_MODEL"], MINIMAX_MODEL),
            timeout=_env_float([MINIMAX_TIMEOUT_ENV], 45.0),
            max_tokens=_env_int([MINIMAX_MAX_TOKENS_ENV], 700),
            opener=opener,
        )

    def select(
        self,
        candidates: Sequence[InsightCandidate],
        hits: Sequence[CorpusHit],
    ) -> list[InsightCandidate]:
        if not candidates:
            return []
        prompt = build_minimax_selection_prompt(candidates, hits)
        text = call_minimax_m3(
            api_key=self._api_key,
            prompt=prompt,
            system=(
                "You are a strict research alpha selector. Pick only tight, receipt-bound "
                "bridges. Return only valid JSON."
            ),
            temperature=0.2,
            max_tokens=self._max_tokens,
            base_url=self._base_url,
            model=self._model,
            timeout=self._timeout,
            opener=self._opener,
        )
        return parse_minimax_selection(text, candidates)


def call_minimax_m3(
    *,
    api_key: str,
    prompt: str,
    system: str,
    temperature: float,
    max_tokens: int,
    base_url: str = MINIMAX_BASE_URL,
    model: str = MINIMAX_MODEL,
    timeout: float = 60.0,
    opener: RequestOpener | None = None,
) -> str:
    body = json.dumps(
        {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        }
    ).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/v1/messages",
        data=body,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    request_opener = opener or cast(RequestOpener, urlopen)
    with request_opener(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return _anthropic_text(data)


def load_minimax_api_key() -> str:
    for env_name in (MINIMAX_V5_KEY_ENV, MINIMAX_KEY_ENV):
        key = os.environ.get(env_name, "").strip()
        if key:
            return key
    if MINIMAX_KEY_FILE.exists():
        return MINIMAX_KEY_FILE.read_text().strip()
    return ""


def build_minimax_prompt(candidate: InsightCandidate, receipts: Sequence[CorpusHit]) -> str:
    if not receipts:
        raise ValueError("cannot write MiniMax memo without receipts")
    receipt_block = "\n\n".join(
        _receipt_block(index, hit) for index, hit in enumerate(receipts, start=1)
    )
    return f"""Write a sharper alpha memo from the locked evidence below.

Hard rules:
- Use only the supplied receipts.
- Keep every receipt ID exactly as written.
- Do not invent mechanisms, clinical advice, causal certainty, new papers, or new numbers.
- If a connection is uncertain, say it is a hypothesis.
- Treat the seed topic as search context only; do not use broad seed-topic words in
  the title unless those words appear in the locked receipt titles/abstracts.
- Title the memo around receipt-owned concepts, not around the user's seed query.
- In the title, copy receipt/bridge terms verbatim; do not use synonyms or paraphrases.
- Scope every implication to the receipts: state the specific population, market,
  company, channel, model, benchmark, timeframe, geography, or source type only when
  the receipts provide it.
- Use source-appropriate descriptors from the receipts, not generic prestige labels:
  trial/protocol, filing/report, benchmark, case study, market study, campaign, interview,
  dataset, or model card.
- Make the memo read like an insight, not a literature summary: surface the non-obvious bridge,
  contradiction, boundary condition, inversion, neglected proxy, metric mismatch, or
  cross-domain transfer.
- Avoid generic phrases such as "more research is needed" unless tied to a receipt-specific test.
- Output Markdown only.
- Keep it under 450 words.

Required structure:
# Alpha memo: <receipt-owned title>
## Core signal
## The 2+2=5 angle
## Why this could matter
## What would break the idea
## Receipts
## Safety note

Candidate thesis:
{candidate.thesis}

Bridge terms:
{", ".join(candidate.bridge_terms) or "none"}

Tension terms:
{", ".join(candidate.tension_terms) or "none"}

Scores:
signal={candidate.score}, novelty={candidate.novelty_score}, evidence={candidate.evidence_score}

Selector tier:
{candidate_alpha_tier(candidate)}

Receipt roles:
{_role_block(candidate)}

Locked receipts:
{receipt_block}
"""


def build_minimax_search_prompt(*, topic: str, seed_queries: Sequence[str], limit: int) -> str:
    seeds = "\n".join(f"- {query}" for query in seed_queries) or "- none"
    return f"""Create {limit} search queries for finding non-obvious, receipt-worthy papers.

Goal:
Find papers that could reveal surprising cross-links, boundary conditions, contradictions,
mechanisms, or underused proxies for this topic:
{topic}

Existing seed queries:
{seeds}

Rules:
- Search is over a massive academic corpus, so use specific scientific terms.
- Do not make one huge query; make diverse 3-7 term queries.
- Include synonym/adjacent-mechanism angles, not just restatements.
- Prefer queries likely to surface real papers, not essay phrases.
- Return JSON only: an array of strings.
"""


def build_minimax_selection_prompt(
    candidates: Sequence[InsightCandidate],
    hits: Sequence[CorpusHit],
) -> str:
    by_id = {hit.hit_id: hit for hit in hits}
    blocks = "\n\n".join(
        _candidate_block(index, candidate, by_id)
        for index, candidate in enumerate(candidates, start=1)
    )
    return f"""Select the strongest alpha memo bridge from these deterministic candidates.

Rules:
- Prefer one bounded surprise that appears only when the receipts are read together.
- Strong: same intervention/construct/program, direct reversal, endpoint split, or negative/null boundary.
- Strongest: one promise/mechanism receipt plus one observed outcome receipt sharing the same core construct.
- Weak: adjacent papers, broad survey plus case study, unsupported domain jump, or generic "evidence is mixed".
- If no candidate is tight enough, classify as "discovery_seed".
- Do not invent candidates or receipt IDs.
- Return JSON only:
  {{"classification":"alpha_memo|discovery_seed|reject","candidate":1,"reason":"short reason"}}

Candidates:
{blocks}
"""


def parse_minimax_selection(
    text: str,
    candidates: Sequence[InsightCandidate],
) -> list[InsightCandidate]:
    stripped = _strip_markdown_fence(text)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("MiniMax selector did not return valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("MiniMax selector must return a JSON object")
    classification = str(data.get("classification", "")).casefold()
    if classification != "alpha_memo":
        return []
    index = data.get("candidate")
    if not isinstance(index, int) or not 1 <= index <= len(candidates):
        raise ValueError("MiniMax selector returned an invalid candidate index")
    return [candidates[index - 1]]


def parse_minimax_queries(text: str, *, limit: int) -> list[str]:
    stripped = _strip_markdown_fence(text)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ValueError("MiniMax planner did not return valid JSON") from exc
    if not isinstance(data, list):
        raise ValueError("MiniMax planner must return a JSON array")
    queries: list[str] = []
    for item in data:
        if not isinstance(item, str):
            continue
        query = " ".join(item.split())
        if 4 <= len(query) <= 160:
            queries.append(query)
    queries = _dedupe_queries(queries)
    if not queries:
        raise ValueError("MiniMax planner returned no usable queries")
    return queries[: max(1, limit)]


def validate_minimax_memo(
    markdown: str,
    receipts: Sequence[CorpusHit],
    *,
    candidate: InsightCandidate | None = None,
) -> str:
    text = markdown.strip()
    if not text:
        raise MemoFormatError("MiniMax returned an empty memo")
    missing_sections = [section for section in _REQUIRED_MEMO_SECTIONS if section not in text]
    if missing_sections:
        raise MemoFormatError(
            f"MiniMax memo missing required sections: {', '.join(missing_sections)}"
        )
    missing = [hit.receipt_id for hit in receipts if hit.receipt_id not in text]
    if missing:
        raise ValueError(f"MiniMax memo dropped receipt IDs: {', '.join(missing)}")
    allowed_dois = _receipt_dois(receipts)
    extra_dois = sorted(_extract_dois(text) - allowed_dois)
    if extra_dois:
        raise ValueError(
            f"MiniMax memo included unreceipted DOI-like references: {', '.join(extra_dois)}"
        )
    if candidate is not None:
        _validate_receipt_owned_title(text, receipts, candidate)
    return text + "\n"


def _receipt_block(index: int, hit: CorpusHit) -> str:
    year = str(hit.year) if hit.year is not None else "unknown"
    venue = hit.venue or "unknown venue"
    locator = hit.doi or hit.url or hit.hit_id
    abstract = _truncate_receipt_text(hit.abstract, RECEIPT_ABSTRACT_CHAR_LIMIT)
    return (
        f"Receipt {index}\n"
        f"ID: {hit.receipt_id}\n"
        f"Title: {hit.title}\n"
        f"Year: {year}\n"
        f"Venue: {venue}\n"
        f"Source: {hit.source}\n"
        f"Locator: {locator}\n"
        f"Abstract: {abstract}"
    )


def _candidate_block(
    index: int,
    candidate: InsightCandidate,
    hits_by_id: Mapping[str, CorpusHit],
) -> str:
    receipts = "\n".join(
        _selection_receipt(hit)
        for receipt_id in candidate.receipt_ids
        if (hit := hits_by_id.get(receipt_id)) is not None
    )
    return (
        f"Candidate {index}\n"
        f"Thesis: {candidate.thesis}\n"
        f"Bridge terms: {', '.join(candidate.bridge_terms) or 'none'}\n"
        f"Tension terms: {', '.join(candidate.tension_terms) or 'none'}\n"
        f"Reasons: {', '.join(candidate.reasons) or 'none'}\n"
        f"Tier: {candidate_alpha_tier(candidate)}\n"
        f"Receipt roles: {_inline_roles(candidate)}\n"
        f"Score: {candidate.score}\n"
        f"Receipts:\n{receipts}"
    )


def _selection_receipt(hit: CorpusHit) -> str:
    abstract = _truncate_receipt_text(hit.abstract, 500)
    year = f" ({hit.year})" if hit.year is not None else ""
    return f"- {hit.hit_id}: {hit.title}{year}; {abstract}"


def _role_block(candidate: InsightCandidate) -> str:
    if not candidate.receipt_roles:
        return "- none assigned"
    return "\n".join(
        f"- {role.receipt_id}: {role.role} ({role.reason})"
        for role in candidate.receipt_roles
    )


def _inline_roles(candidate: InsightCandidate) -> str:
    if not candidate.receipt_roles:
        return "none assigned"
    return "; ".join(f"{role.receipt_id}={role.role}" for role in candidate.receipt_roles)


def _anthropic_text(data: Mapping[str, Any]) -> str:
    content = data.get("content")
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if (
            isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        ):
            parts.append(item["text"])
    return "\n".join(parts).strip()


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        first = lines[0].strip().casefold()
        if first in {"```", "```json", "```markdown", "```md"}:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _dedupe_queries(queries: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for query in queries:
        key = query.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(query)
    return deduped


def _env_string(names: Sequence[str], default: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def _env_float(names: Sequence[str], default: float) -> float:
    value = _env_string(names, "")
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{names[0]} must be a number") from exc


def _env_int(names: Sequence[str], default: int) -> int:
    value = _env_string(names, "")
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{names[0]} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{names[0]} must be positive")
    return parsed


def _truncate_receipt_text(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        return "No abstract supplied."
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 16)].rstrip() + " ... [truncated]"


def _receipt_dois(receipts: Sequence[CorpusHit]) -> set[str]:
    allowed: set[str] = set()
    for hit in receipts:
        for value in (hit.doi, hit.receipt_id):
            if value:
                allowed.update(_extract_dois(value))
    return allowed


def _extract_dois(text: str) -> set[str]:
    return {_normalize_doi(match.group(0)) for match in _DOI_RE.finditer(text)}


def _normalize_doi(value: str) -> str:
    return value.strip().rstrip(_DOI_TRAILING_PUNCTUATION).casefold()


def _validate_receipt_owned_title(
    markdown: str,
    receipts: Sequence[CorpusHit],
    candidate: InsightCandidate,
) -> None:
    first_line = markdown.splitlines()[0] if markdown.splitlines() else ""
    if not first_line.casefold().startswith("# alpha memo:"):
        return
    title_terms = _title_terms(first_line)
    supported_terms = _receipt_terms(receipts) | _title_terms(" ".join(candidate.bridge_terms))
    unsupported_terms = sorted(title_terms - supported_terms)
    if unsupported_terms:
        raise MemoScopeError(
            "MiniMax memo title used terms not supported by receipts: "
            + ", ".join(unsupported_terms)
        )


def _title_terms(text: str) -> set[str]:
    return {
        _normalize_title_term(term)
        for term in _TITLE_WORD_RE.findall(text.casefold())
        if term not in _TITLE_STOPWORDS
    }


def _receipt_terms(receipts: Sequence[CorpusHit]) -> set[str]:
    terms: set[str] = set()
    for hit in receipts:
        terms.update(_title_terms(hit.text))
    return terms


def _normalize_title_term(term: str) -> str:
    if len(term) > 6 and term.endswith("ing"):
        return term[:-3]
    if len(term) > 5 and term.endswith("ed"):
        return term[:-2]
    if len(term) > 4 and term.endswith("s") and not term.endswith(("ss", "sis")):
        return term[:-1]
    return term
