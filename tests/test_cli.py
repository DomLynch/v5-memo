from __future__ import annotations

import sys

import pytest
from pytest import MonkeyPatch

from v5_memo.__main__ import main


def test_fullraw_searcher_fails_closed_without_endpoint(
    monkeypatch: MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v5_memo",
            "--searcher",
            "fullraw",
            "--topic",
            "longevity resilience",
            "--query",
            "NAD mitochondrial stress",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Full local raw 450M+ corpus search is not configured" in captured.err
    assert "V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL" in captured.err
