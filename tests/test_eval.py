from pathlib import Path

from v5_memo.eval import evaluate_golden_cases

_FIXTURES = Path(__file__).with_name("fixtures")


def test_golden_eval_harness_scores_all_cases() -> None:
    report = evaluate_golden_cases(_FIXTURES / "golden_alpha_cases.jsonl")

    assert report.failed == 0
    assert report.passed == report.cases
    assert report.cases >= 5
    assert report.positive_cases >= 3
    assert report.negative_cases >= 3
    assert report.false_positive_count == 0
    assert report.missed_positive_count == 0
