# V5 Memo

Small, independent alpha memo writer. It searches corpus surfaces, mines
receipt-bound tensions, binds receipts, then writes a short memo.

This repo is separate from v3 and v4.

## How To Run

Prerequisites:

- Python 3.11+
- Optional `MINIMAX_API_KEY` or `V5_MEMO_MINIMAX_API_KEY` for MiniMax planning/writing
- Optional `V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL` for the full raw indexed search endpoint
- Optional `RESEARKA_DATABASE_URL` and `RESEARKA_TOKENS` for the Researka corpus API

Install and test:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests
python -m mypy src tests
```

Offline demo:

```bash
PYTHONPATH=src python -m v5_memo --demo
```

Current best memo path:

```bash
MINIMAX_API_KEY=... \
PYTHONPATH=src python -m v5_memo \
  --searcher smart \
  --topic "longevity resilience" \
  --query "NAD salvage mitochondrial stress"
```

`--searcher smart` means:

1. MiniMax proposes better search angles.
2. V5 searches hybrid corpus surfaces.
3. V5 dedupes, scores, mines, and binds receipts deterministically.
4. MiniMax rewrites only inside the locked receipts.

## Search Modes

OpenAlex, no token:

```bash
PYTHONPATH=src python -m v5_memo \
  --topic "longevity resilience" \
  --query "NAD salvage mitochondrial stress"
```

Researka corpus:

```bash
RESEARKA_DATABASE_URL=http://127.0.0.1:8810 \
RESEARKA_TOKENS=... \
PYTHONPATH=src python -m v5_memo \
  --searcher researka \
  --topic "longevity resilience" \
  --query "NAD salvage mitochondrial stress"
```

Full raw indexed corpus:

```bash
V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL=http://127.0.0.1:9902/search \
V5_MEMO_FULL_RAW_CORPUS_TOKEN=... \
MINIMAX_API_KEY=... \
PYTHONPATH=src python -m v5_memo \
  --searcher fullraw \
  --planner minimax \
  --writer minimax \
  --topic "longevity resilience" \
  --query "NAD salvage mitochondrial stress"
```

Coverage truth:

```bash
PYTHONPATH=src python -m v5_memo --coverage-report
PYTHONPATH=src python -m v5_memo --require-full-raw-corpus
```

`--searcher fullraw` and `--require-full-raw-corpus` fail closed unless a real
full-raw search service is configured.

## Full Raw Index

Build and serve the local FTS index:

```bash
PYTHONPATH=src python -m v5_memo.fullraw_index build
PYTHONPATH=src python -m v5_memo.fullraw_index serve
```

Build immutable shard batches for the full 470M+ corpus:

```bash
PYTHONPATH=src python -m v5_memo.fullraw_index build-upload-shards
PYTHONPATH=src python -m v5_memo.fullraw_index stats-shards
PYTHONPATH=src python -m v5_memo.fullraw_index search-shards "management forecast disclosure"
```

Serve shard-backed search:

```bash
V5_MEMO_FULL_RAW_SHARD_DIR=/mnt/fullraw-shards \
PYTHONPATH=src python -m v5_memo.fullraw_index serve
```

Default indexed endpoint: `127.0.0.1:9902`.

The index stores a persistent `term_map` table for query expansion. Inspect the
actual expansion used for a query:

```bash
PYTHONPATH=src python -m v5_memo.fullraw_index explain "management forecast disclosure"
```

Run the golden insight-quality harness:

```bash
PYTHONPATH=src python -m v5_memo.eval
```

Storage Box shard layout details live in
[`docs/architecture/fullraw-storage-box-shards.md`](docs/architecture/fullraw-storage-box-shards.md).
