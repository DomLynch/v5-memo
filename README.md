# V5 Memo

Independent alpha memo writer for finding short, receipt-bound research insights from full-corpus OpenAlex search.

This repo is separate from v3 and v4. First slice:

1. Fan out each seed into related OpenAlex full-corpus searches.
2. Dedupe hits.
3. Locally rerank merged hits by term coverage, source rank, and citation signal.
4. Mine source-diverse bridge candidates.
5. Score novelty/evidence/tension.
6. Bind receipts.
7. Render a short memo.

Offline demo:

```bash
PYTHONPATH=src python -m v5_memo --demo
```

Quality gate:

```bash
python -m pytest -q
python -m ruff check src tests
python -m mypy src tests
```

Live full-corpus use needs no token:

```bash
PYTHONPATH=src python -m v5_memo \
  --topic "longevity resilience" \
  --query "NAD salvage mitochondrial stress" \
  --query "exercise response mitochondrial repair"
```
