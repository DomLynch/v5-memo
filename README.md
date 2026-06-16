# V5 Memo

Independent alpha memo writer for finding short, receipt-bound research insights from full-corpus OpenAlex search.

This repo is separate from v3 and v4. First slice:

1. Optionally ask MiniMax-M3 to plan sharper seed queries.
2. Fan out each seed into related OpenAlex full-corpus searches.
3. Dedupe hits.
4. Locally rerank merged hits by term coverage, source rank, and citation signal.
5. Mine source-diverse bridge candidates.
6. Score novelty/evidence/tension.
7. Bind receipts.
8. Render a short memo with the deterministic template, or optionally ask MiniMax-M3
   to rewrite the memo inside the locked receipts.

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

MiniMax-M3 writer pass:

```bash
MINIMAX_API_KEY=... PYTHONPATH=src python -m v5_memo \
  --planner minimax \
  --writer minimax \
  --topic "longevity resilience" \
  --query "NAD salvage mitochondrial stress" \
  --query "exercise response mitochondrial repair"
```

The MiniMax planner proposes search angles. OpenAlex retrieval, dedupe, scoring,
and receipt binding stay deterministic; the writer must preserve every locked
receipt ID.
