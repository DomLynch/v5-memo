# V5 Memo

Independent alpha memo writer for finding short, receipt-bound research insights from corpus-scale Researka search.

This repo is separate from v3 and v4. First slice:

1. Search corpus seeds.
2. Dedupe hits.
3. Mine source-diverse bridge candidates.
4. Score novelty/evidence/tension.
5. Bind receipts.
6. Render a short memo.

Offline demo:

```bash
PYTHONPATH=src python -m v5_memo --demo
```

Quality gate:

```bash
python -m pytest -q
python -m ruff check src tests
python -m mypy src
```

Live DB use needs `RESEARKA_DATABASE_TOKEN`:

```bash
PYTHONPATH=src RESEARKA_DATABASE_TOKEN=... \
python -m v5_memo \
  --topic "longevity resilience" \
  --query "NAD salvage mitochondrial stress" \
  --query "exercise response mitochondrial repair"
```
