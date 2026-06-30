"""MiniMax-M3 planner and memo writer constrained by corpus receipts."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import cast
from urllib.request import urlopen

from v5_memo.gate import candidate_alpha_tier
from v5_memo.llm.minimax_client import (
    MINIMAX_BASE_URL,
    MINIMAX_BASE_URL_ENV,
    MINIMAX_MAX_TOKENS_ENV,
    MINIMAX_MODEL,
    MINIMAX_MODEL_ENV,
    MINIMAX_TIMEOUT_ENV,
    RequestOpener,
)
from v5_memo.llm.minimax_client import (
    call_minimax_m3 as call_minimax_m3,
)
from v5_memo.llm.minimax_client import (
    load_minimax_api_key as load_minimax_api_key,
)
from v5_memo.schemas import CorpusHit, InsightCandidate

RECEIPT_ABSTRACT_CHAR_LIMIT = 120
_REQUIRED_MEMO_SECTIONS = (
    "# Alpha memo:",
    "## Core signal",
    "## The 2+2=5 angle",
    "## Why this could matter",
    "## What would break the idea",
    "## Claim ledger",
    "## Receipts",
    "## Safety note",
)
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s<>()\[\]{}\"']+", re.IGNORECASE)
_DOI_TRAILING_PUNCTUATION = ".,;:*_`"
_STAT_CONTEXT_RE = re.compile(r"(?i)(?:confidence interval|\bci\b|effect size|cohen'?s d|hedges'? g|standardized mean difference).{0,120}")
_STAT_ANCHOR_RE = re.compile(r"(?i)\b(?:p\s*=\s*\.?\d+|g\s*=\s*[-+]?\d+(?:\.\d+)?|95%\s*(?:confidence interval|\bci\b)|confidence interval)")
_LIMIT_ANCHOR_RE = re.compile(r"(?i)\b\d+\s+(?:adults|athletes|men|participants|patients|players|subjects|volunteers|women)\b")
_STAT_NUMBER_RE = re.compile(r"[-+]?\d+\.\d+%?|[-+]?\d+%")
_ADVICE_RE = re.compile(
    r"(?i)\b(?:athletes?|clinicians?|companies?|investors?|managers?|patients?|practitioners?)\b"
    r"[^.\n]{0,120}\bshould\b|\bshould\s+(?:avoid|buy|invest|prescribe|prioriti[sz]e|sell|take|treat|use)\b"
)
_MARKET_FRAMING_RE = re.compile(r"(?i)\b(?:market\s+for|reframe\s+the\s+market|commercial\s+market|investment)\b")
_CONVERSION_OVERCLAIM_RE = re.compile(r"(?is)\bconverts?\b.{0,160}\binto\b")
_TITLE_WORD_RE = re.compile(r"[a-z][a-z0-9]{2,}")
_TITLE_STOPWORDS = frozenset({
    "alpha", "memo", "and", "for", "from", "into", "may", "not", "the", "with",
    "without", "between", "versus", "under", "over", "through", "across",
    "after", "before", "during", "following",
    "one", "two", "three", "both", "same", "abstract", "readout", "readouts", "report", "reports",
    "signal", "signals", "effect", "effects", "tradeoff", "tradeoffs",
    "boundary", "condition", "conditions", "hypothesis", "discovery", "seed",
    "split", "splits", "diverge", "diverges", "divergence", "outlier",
    "promise", "outcome", "outcomes",
    "bridge", "bridges", "receipt", "receipts",
    "while", "even", "yet", "leave", "leaves", "leaving", "level", "levels", "lift", "lifts",
    "elevate", "elevates", "elevated", "elevating",
    "status", "statuses", "suppress", "suppresses", "suppressed", "suppressing",
})
_BODY_SITE_TERMS = frozenset({
    "ankle", "arm", "calf", "elbow", "flexor", "glute", "hamstring", "hip",
    "knee", "leg", "limb", "quadricep", "soleus", "tendon", "thigh",
    "vastus", "medialis",
})


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
        last_error: ValueError | None = None
        for attempt in range(2):
            retry_note = (
                ""
                if attempt == 0
                else "\n\nPrevious draft failed validation: "
                f"{last_error}. Rewrite from the locked receipts only and remove any unlisted DOI-like references."
            )
            markdown = self._write(prompt + retry_note, temperature=0.35 if attempt == 0 else 0.0)
            try:
                return validate_minimax_memo(markdown, receipts, candidate=candidate)
            except ValueError as exc:
                last_error = exc
                if attempt == 1:
                    raise
        raise AssertionError("unreachable MiniMax writer retry loop")

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
            temperature=0.0,
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
            temperature=0.0,
            max_tokens=self._max_tokens,
            base_url=self._base_url,
            model=self._model,
            timeout=self._timeout,
            opener=self._opener,
        )
        return parse_minimax_selection(text, candidates)


def build_minimax_prompt(candidate: InsightCandidate, receipts: Sequence[CorpusHit]) -> str:
    if not receipts:
        raise ValueError("cannot write MiniMax memo without receipts")
    receipt_block = "\n\n".join(
        _receipt_block(index, hit) for index, hit in enumerate(receipts, start=1)
    )
    title = _safe_alpha_title(candidate, receipts)
    return f"""Write a sharper alpha memo from the locked evidence below.

Hard rules:
- Use only the supplied receipts.
- Keep every receipt ID exactly as written.
- Do not invent mechanisms, clinical advice, causal certainty, new papers, or new numbers.
- Do not make market, product, investment, practitioner, patient, or action claims unless receipts say those concepts.
- If a connection is uncertain, say it is a hypothesis.
- Treat the seed topic as search context only; use broad seed words in the title only if receipts contain them.
- Title the memo around receipt-owned concepts, not around the user's seed query.
- In the title, copy receipt/bridge terms verbatim; do not use synonyms or paraphrases.
- The title must be made only from locked receipt title/abstract words or listed bridge terms.
- Scope every implication to the receipts: population, market, company, channel, model, benchmark, timeframe, geography, source type.
- State the receipt-owned timing exactly; do not turn pre-exercise, prior-to-use,
  or post-intervention exposure into "during" unless a receipt says during.
- If receipts split by endpoint or metric, say it is not a direct contradiction
  unless both receipts measure the same endpoint family.
- Do not say "opposite directions" unless protocol/design/population/endpoint match;
  otherwise quantify the protocol/design gap and frame a bounded contrast.
- If overall gains coexist with a control/comparator-favored contrast, label the
  claim ledger mixed/comparator-favored, not simply positive.
- If claim-card role is safety_feasibility or receipt says pilot/safety/feasibility, do not call it positive efficacy.
- Never call a receipt "feasibility/safety-adjacent" unless its claim-card role is safety_feasibility.
- Respect receipt roles: promise/protocol/intent/mechanism means expected/designed/hypothesized/framed, not observed result or confirmed endpoint.
- Anchor on the strongest direct human evidence; put weaker context/proxy receipts after it.
- In Core signal, name sample size/statistical context for the strongest receipt; frame one small RCT/cohort as one receipt, not settled consensus.
- Do not equate acute swelling, soreness, thickness, or damage proxies with chronic adaptation
  unless the receipt says adaptation, hypertrophy, or strength changed.
- Proxy/boundary receipts are secondary; do not make them co-equal anchors for chronic-adaptation claims.
- If a proxy/boundary receipt sits beside chronic or long-term adaptation receipts, frame the core signal as endpoint heterogeneity.
- If a systematic review or synthesis receipt has its own negative/null/positive direction,
  state convergence or context only.
- If receipts use different modalities/populations/endpoints, name the split, frame as unresolved endpoint heterogeneity,
  and do not claim one protocol condition converts one result into another.
- Use the 2+2=5 section to state the bounded contrast; if the receipts are heterogeneous
  rather than contradictory, explicitly say they are not directly contradictory.
- Use source-appropriate descriptors from the receipts.
- Make the memo read like an insight: contradiction, boundary, inversion, proxy, metric mismatch, or transfer.
- In "Why this could matter", give one falsifiable hypothesis, not a list.
- In "What would break the idea", name one concrete next-step uncertainty or study design.
- Include a concise Claim ledger section before Receipts. Each claim must use the
  supplied claim-card receipt ID and support type; do not invent unsupported claims.
- Use this exact receipt-owned title first line: # Alpha memo: {title}
- Output Markdown only.
- Keep it under 450 words.

Required structure:
# Alpha memo: {title}
## Core signal
## The 2+2=5 angle
## Why this could matter
## What would break the idea
## Claim ledger
## Receipts
## Safety note

Safety note: concrete limitations only; include sample size, sex if stated or "sex not stated", training status, population, protocol.

Candidate thesis:
{candidate.thesis}

Bridge terms:
{", ".join(candidate.bridge_terms) or "none"}

Tension terms:
{", ".join(candidate.tension_terms) or "none"}

Scores:
signal={candidate.score}, novelty={candidate.novelty_score}, evidence={candidate.evidence_score}

Scorecard:
{_scorecard_block(candidate)}

Selector tier:
{candidate_alpha_tier(candidate)}

Receipt roles:
{_role_block(candidate)}

Evidence graph:
{_evidence_graph_block(candidate)}

Claim ledger:
{_claim_card_block(candidate)}

Locked receipts:
{receipt_block}
"""


def _safe_alpha_title(candidate: InsightCandidate, receipts: Sequence[CorpusHit]) -> str:
    by_id = {_receipt_display_id(hit): hit for hit in receipts} | {hit.hit_id: hit for hit in receipts}
    promise_terms: list[str] = []
    outcome_terms: list[str] = []
    for role in candidate.receipt_roles:
        hit = by_id.get(role.receipt_id)
        if hit is None:
            continue
        title_terms = _title_terms(hit.title)
        if role.role == "promise":
            promise_terms.extend(term for term in ("augment", "mimic", "protocol", "expected") if term in title_terms)
        elif role.role == "outcome":
            outcome_terms.extend(term for term in ("blunt", "impair", "attenuate", "null", "reduce") if term in title_terms)
    terms = [term for term in candidate.bridge_terms if term and term not in _TITLE_STOPWORDS]
    if promise_terms and outcome_terms:
        terms = [terms[0], promise_terms[0], "versus", outcome_terms[0], *terms[1:3]]
    else:
        roles = {role.role for role in candidate.receipt_roles}
        if {"promise", "outcome"} <= roles:
            terms.extend(["promise", "outcome"])
    return " ".join(dict.fromkeys(terms)) or "receipt-bound alpha"


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
- Prefer promise->outcome reversal searches: protocol/expected/designed/mimic/augment
  paired with observed/blunted/attenuated/impaired/null/reduced.
- Split those two evidence sides into separate title-like queries. Do not pack the
  promise/mechanism terms and the blunted/null outcome terms into one huge query.
- Return adjacent query pairs when possible: query N should search the promise/mechanism
  side, query N+1 should search the observed/null/blunted outcome side, and both
  must share the same specific intervention, construct, product, model, or program.
- The promise/mechanism-side query may omit downstream endpoint words when those words
  would hide the seminal mechanism paper; keep the shared intervention/construct.
- Include at least one 2-4 term upstream-promise query shaped like: shared intervention/construct
  + improves/activates/augments + specific mechanism/output. Do not include the
  downstream application terms in that query unless they belong to the promise paper.
- For intervention topics, include at least one promise-side title query using words
  like expected/designed/protocol/augment/mimic and one outcome-side title query using
  words like blunts/null/reduced/impaired/attenuated for the same intervention.
- Outcome-side queries should name observed endpoints/results, not pathway-only proxies unless the seed topic names that proxy.
- Do not spend most queries on pathway-only mechanisms; mechanism queries must still
  include the shared intervention, construct, product, model, or program.
- Prefer same intervention/construct/program across evidence objects.
- At least half the queries must name a specific intervention, construct, product,
  model, program, or mechanism; do not rely on generic words like intervention,
  evidence, aging, exercise, market, business, AI, or older adults.
- Avoid broad review, meta-analysis, position-stand, guideline, or survey queries unless
  the seed topic explicitly asks for synthesis evidence.
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
- Reject shared-vocabulary bridges where the only commonality is a broad endpoint,
  sample size, method word, or generic intervention class.
- Reject review/position-stand/meta-analysis/survey plus one trial as alpha unless
  the synthesis paper is itself the object of contradiction.
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
    missing = [
        _receipt_display_id(hit)
        for hit in receipts
        if _receipt_display_id(hit) not in text
    ]
    if missing:
        raise ValueError(f"MiniMax memo dropped receipt IDs: {', '.join(missing)}")
    allowed_dois = _receipt_dois(receipts)
    extra_dois = sorted(_extract_dois(text) - allowed_dois)
    if extra_dois:
        raise ValueError(
            f"MiniMax memo included unreceipted DOI-like references: {', '.join(extra_dois)}"
        )
    _validate_supported_stat_numbers(text, receipts)
    _validate_receipt_owned_body_terms(text, receipts)
    _validate_no_conversion_overclaim(text)
    _validate_public_alpha_framing(text, receipts)
    if candidate is not None:
        _validate_receipt_owned_title(text, receipts, candidate)
        _validate_claim_ledger(text, candidate)
    return text + "\n"


def _validate_supported_stat_numbers(markdown: str, receipts: Sequence[CorpusHit]) -> None:
    receipt_numbers = {
        _normalize_stat_number(match.group(0))
        for hit in receipts
        for match in _STAT_NUMBER_RE.finditer(hit.text)
    }
    unsupported: list[str] = []
    for context in _STAT_CONTEXT_RE.finditer(markdown):
        for number in _STAT_NUMBER_RE.findall(context.group(0)):
            if _looks_like_doi_prefix(number, context.group(0)):
                continue
            normalized = _normalize_stat_number(number)
            if normalized not in receipt_numbers:
                unsupported.append(number)
    if unsupported:
        raise MemoFormatError(
            "MiniMax memo included unsupported statistical numbers: "
            + ", ".join(dict.fromkeys(unsupported))
        )


def _normalize_stat_number(value: str) -> str:
    raw = value.strip().rstrip("%")
    try:
        number = float(raw)
    except ValueError:
        return value.strip()
    return f"{number:g}" + ("%" if value.strip().endswith("%") else "")


def _looks_like_doi_prefix(number: str, context: str) -> bool:
    raw = number.strip().rstrip("%")
    if not raw.startswith("10."):
        return False
    return re.search(rf"(?<![\d.]){re.escape(raw)}(?:/|[A-Za-z])", context) is not None


def _validate_public_alpha_framing(markdown: str, receipts: Sequence[CorpusHit]) -> None:
    if _ADVICE_RE.search(markdown):
        raise MemoScopeError("MiniMax memo included advice/action framing")
    receipt_text = " ".join(hit.text for hit in receipts).casefold()
    if "market" not in receipt_text and _MARKET_FRAMING_RE.search(markdown):
        raise MemoScopeError("MiniMax memo included unreceipted market framing")


def _validate_no_conversion_overclaim(markdown: str) -> None:
    if _CONVERSION_OVERCLAIM_RE.search(markdown):
        raise MemoScopeError("MiniMax memo claimed one receipt condition converts another result")


def _validate_receipt_owned_body_terms(markdown: str, receipts: Sequence[CorpusHit]) -> None:
    unsupported = sorted((_title_terms(markdown) & _BODY_SITE_TERMS) - _receipt_terms(receipts))
    if unsupported:
        raise MemoScopeError(
            "MiniMax memo used body-site terms not supported by receipts: "
            + ", ".join(unsupported)
        )


def _validate_claim_ledger(markdown: str, candidate: InsightCandidate) -> None:
    if "## Claim ledger" not in markdown:
        raise MemoFormatError("MiniMax memo missing required claim ledger")
    if not candidate.claim_cards:
        return
    ledger = markdown.split("## Claim ledger", 1)[1].split("## Receipts", 1)[0]
    missing = [
        card.receipt_id
        for card in candidate.claim_cards
        if card.receipt_id not in ledger or card.support_type not in ledger
    ]
    if missing:
        raise MemoFormatError(
            "MiniMax memo claim ledger missing receipt/support entries: "
            + ", ".join(dict.fromkeys(missing))
        )


def _receipt_block(index: int, hit: CorpusHit) -> str:
    year = str(hit.year) if hit.year is not None else "unknown"
    venue = hit.venue or "unknown venue"
    receipt_id = _receipt_display_id(hit)
    locator = _receipt_locator(hit)
    abstract = _truncate_receipt_text(hit.abstract, RECEIPT_ABSTRACT_CHAR_LIMIT)
    stat_context = _receipt_stat_context(hit.abstract)
    stat_line = f"\nStatistics/context: {stat_context}" if stat_context else ""
    return (
        f"Receipt {index}\n"
        f"ID: {receipt_id}\n"
        f"Title: {hit.title}\n"
        f"Year: {year}\n"
        f"Venue: {venue}\n"
        f"Source: {hit.source}\n"
        f"Locator: {locator}\n"
        f"Abstract: {abstract}{stat_line}"
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
        f"Evidence graph: {_inline_evidence_graph(candidate)}\n"
        f"Claim cards: {_inline_claim_cards(candidate)}\n"
        f"Score: {candidate.score}\n"
        f"Scorecard: {_inline_scorecard(candidate)}\n"
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


def _evidence_graph_block(candidate: InsightCandidate) -> str:
    if not candidate.evidence_graph:
        return "- none assigned"
    return "\n".join(
        f"- {node.receipt_id}: {node.role} ({node.reason})"
        for node in candidate.evidence_graph
    )


def _inline_evidence_graph(candidate: InsightCandidate) -> str:
    if not candidate.evidence_graph:
        return "none assigned"
    return "; ".join(f"{node.receipt_id}={node.role}" for node in candidate.evidence_graph)


def _scorecard_block(candidate: InsightCandidate) -> str:
    if not candidate.scorecard:
        return "- none assigned"
    return "\n".join(
        f"- {key}: {value}"
        for key, value in sorted(candidate.scorecard.items())
    )


def _inline_scorecard(candidate: InsightCandidate) -> str:
    if not candidate.scorecard:
        return "none assigned"
    return "; ".join(f"{key}={value}" for key, value in sorted(candidate.scorecard.items()))


def _claim_card_block(candidate: InsightCandidate) -> str:
    if not candidate.claim_cards:
        return "- none assigned"
    return "\n".join(
        (
            f"- {card.receipt_id}: role={card.role}; design={card.design}; "
            f"population={card.population}; outcome={card.outcome}; "
            f"direction={card.direction}; support={card.support_type}/{card.confidence}; "
            f"quote={card.quote}"
        )
        for card in candidate.claim_cards
    )


def _inline_claim_cards(candidate: InsightCandidate) -> str:
    if not candidate.claim_cards:
        return "none assigned"
    return "; ".join(
        f"{card.receipt_id}:{card.role}/{card.design}/{card.population}/{card.direction}"
        for card in candidate.claim_cards
    )


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


def _receipt_stat_context(text: str) -> str:
    normalized = " ".join(text.split())
    snippets: list[str] = []
    seen: set[str] = set()
    for match in _STAT_ANCHOR_RE.finditer(normalized):
        start = max(0, match.start() - 120)
        end = min(len(normalized), match.end() + 180)
        snippet = normalized[start:end].strip(" ,.;")
        key = snippet.casefold()
        if key in seen:
            continue
        seen.add(key)
        snippets.append(snippet)
        if len(snippets) >= 8:
            break
    for match in _LIMIT_ANCHOR_RE.finditer(normalized):
        start = max(0, match.start() - 100)
        end = min(len(normalized), match.end() + 160)
        snippet = normalized[start:end].strip(" ,.;")
        key = snippet.casefold()
        if key not in seen:
            seen.add(key)
            snippets.append(snippet)
        if len(snippets) >= 8:
            break
    if not snippets:
        return ""
    outcome_terms = {"adaptation", "adaptations", "jump", "muscle", "performance", "strength", "thickness"}
    snippets.sort(
        key=lambda value: (
            bool(set(_TITLE_WORD_RE.findall(value.casefold())) & outcome_terms),
            len(_STAT_NUMBER_RE.findall(value)),
        ),
        reverse=True,
    )
    return _truncate_receipt_text(" | ".join(snippets[:2]), 360)


def _receipt_dois(receipts: Sequence[CorpusHit]) -> set[str]:
    allowed: set[str] = set()
    for hit in receipts:
        for value in (hit.receipt_id,):
            if value:
                allowed.update(_extract_dois(value))
    return allowed


def _receipt_display_id(hit: CorpusHit) -> str:
    if hit.receipt_id != hit.hit_id:
        return hit.receipt_id
    match = re.search(r"\bW\d+\b", hit.hit_id, re.IGNORECASE)
    return match.group(0) if match else hit.hit_id


def _receipt_locator(hit: CorpusHit) -> str:
    if hit.receipt_id != hit.hit_id:
        return hit.receipt_id
    return hit.url or hit.hit_id


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
    terms: set[str] = set()
    for raw in _TITLE_WORD_RE.findall(text.casefold()):
        term = _normalize_title_term(raw)
        if raw not in _TITLE_STOPWORDS and term not in _TITLE_STOPWORDS:
            terms.add(term)
    return terms


def _receipt_terms(receipts: Sequence[CorpusHit]) -> set[str]:
    terms: set[str] = set()
    for hit in receipts:
        terms.update(_title_terms(hit.text))
    return terms


def _normalize_title_term(term: str) -> str:
    if len(term) > 6 and term.endswith("sses"):
        term = term[:-2]
    elif len(term) > 6 and term.endswith("ing"):
        term = term[:-3]
    elif len(term) > 5 and term.endswith("ed"):
        term = term[:-2]
    elif len(term) > 4 and term.endswith("s") and not term.endswith(("ss", "sis")):
        term = term[:-1]
    return {"fibre": "fiber"}.get(term, term)
