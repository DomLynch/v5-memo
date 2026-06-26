"""Small golden-case evaluator for V5 alpha candidate mining."""
from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from v5_memo.gate import candidate_alpha_tier
from v5_memo.miner import mine_insights
from v5_memo.schemas import CorpusHit


@dataclass(frozen=True, slots=True)
class EvalCaseResult:
    name: str
    passed: bool
    expected_positive: bool
    expected_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    expected_tier: str
    selected_tier: str
    expected_shape: str
    selected_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvalReport:
    cases: int
    passed: int
    failed: int
    positive_cases: int
    negative_cases: int
    false_positive_count: int
    missed_positive_count: int
    results: tuple[EvalCaseResult, ...]


def evaluate_golden_cases(path: Path) -> EvalReport:
    results = tuple(_evaluate_case(case) for case in _load_cases(path))
    passed = sum(1 for result in results if result.passed)
    positive_cases = sum(1 for result in results if result.expected_positive)
    false_positive_count = sum(
        1 for result in results if not result.expected_positive and result.selected_ids
    )
    missed_positive_count = sum(
        1 for result in results if result.expected_positive and not result.selected_ids
    )
    return EvalReport(
        cases=len(results),
        passed=passed,
        failed=len(results) - passed,
        positive_cases=positive_cases,
        negative_cases=len(results) - positive_cases,
        false_positive_count=false_positive_count,
        missed_positive_count=missed_positive_count,
        results=results,
    )


def _load_cases(path: Path) -> Iterable[Mapping[str, Any]]:
    for line in path.read_text().splitlines():
        if line.strip():
            raw = json.loads(line)
            if isinstance(raw, dict):
                yield raw


def _evaluate_case(case: Mapping[str, Any]) -> EvalCaseResult:
    expected_ids = _string_tuple(case.get("expected_ids"))
    expected_shape = str(case.get("expected_shape") or "")
    expected_tier = str(case.get("expected_tier") or "")
    candidates = mine_insights(
        _hits(case.get("hits")),
        topic=str(case.get("topic") or ""),
        required_anchor_terms=_string_tuple(case.get("anchors")),
        max_candidates=25,
    )
    selected = candidates[0] if candidates else None
    selected_ids = selected.receipt_ids if selected else ()
    selected_tier = candidate_alpha_tier(selected) if selected else ""
    selected_reasons = selected.reasons if selected else ()
    return EvalCaseResult(
        name=str(case.get("name") or ""),
        passed=(
            selected_ids == expected_ids
            and selected_tier == expected_tier
            and (not expected_shape or expected_shape in selected_reasons)
        ),
        expected_positive=bool(expected_ids),
        expected_ids=expected_ids,
        selected_ids=selected_ids,
        expected_tier=expected_tier,
        selected_tier=selected_tier,
        expected_shape=expected_shape,
        selected_reasons=selected_reasons,
    )


def _hits(raw_hits: object) -> list[CorpusHit]:
    if not isinstance(raw_hits, list):
        return []
    hits: list[CorpusHit] = []
    for raw in raw_hits:
        if not isinstance(raw, dict):
            continue
        hit_id = str(raw.get("id") or raw.get("hit_id") or "")
        if not hit_id:
            continue
        hits.append(
            CorpusHit(
                hit_id=hit_id,
                title=str(raw.get("title") or ""),
                abstract=str(raw.get("abstract") or ""),
                source=str(raw.get("source") or "fixture"),
                doi=str(raw.get("doi") or f"10.fixture/{hit_id}"),
            )
        )
    return hits


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    return tuple(str(item) for item in value if str(item))


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/golden_alpha_cases.jsonl")
    report = evaluate_golden_cases(path)
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    if report.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
