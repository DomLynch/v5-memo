# V5 Memo

Independent alpha memo writer for finding short, receipt-bound research insights from corpus-scale search.

This repo is separate from v3 and v4. First slice:

1. Optionally ask MiniMax-M3 to plan sharper seed queries.
2. Search OpenAlex, Researka corpus, full raw corpus service, or a hybrid of them.
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
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests
python -m mypy src tests
```

The `dev` extra pins the test/lint/type tool versions. Use the project venv
rather than system-wide tools; the VPS global Python intentionally does not have
`ruff` or `mypy` installed.

OpenAlex full-corpus use needs no token:

```bash
PYTHONPATH=src python -m v5_memo \
  --topic "longevity resilience" \
  --query "NAD salvage mitochondrial stress" \
  --query "exercise response mitochondrial repair"
```

Researka corpus use searches the live Researka corpus API. Verified on the VPS:
25,181,785 paper rows, 1,015,859 embedded rows, and a 24,814,247-row Tantivy
index. This is not yet the full raw 450M+ storage corpus.

```bash
RESEARKA_DATABASE_URL=http://127.0.0.1:8810 \
RESEARKA_TOKENS=... \
PYTHONPATH=src python -m v5_memo \
  --searcher researka \
  --topic "longevity resilience" \
  --query "NAD salvage mitochondrial stress"
```

Full raw corpus mode is the strict 450M+ path. It requires a real indexed search
service over the cold raw archive:

```bash
V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL=http://127.0.0.1:9901/search \
V5_MEMO_FULL_RAW_CORPUS_TOKEN=... \
MINIMAX_API_KEY=... \
PYTHONPATH=src python -m v5_memo \
  --searcher fullraw \
  --planner minimax \
  --writer minimax \
  --topic "longevity resilience" \
  --query "NAD salvage mitochondrial stress"
```

The full raw service contract is a `POST` endpoint. V5 sends:

```json
{"query":"...","limit":25,"top_k":25,"year_min":1900,"year_max":2100,"corpus":"full_raw_450m_plus"}
```

The endpoint should return either a JSON array of papers or
`{"meta":{"count":...},"results":[...]}`. Each paper can use standard fields:
`title`, `abstract`, `doi`, `pmid`, `pmcid`, `openalex_id`, `semantic_scholar_id`,
`arxiv_id`, `year`, `journal`/`venue`, `source`, `url`, `cited_by_count`, `score`.

V5 includes a localhost cold-scan backend that can expose the rclone raw archive
without a large local index:

```bash
PYTHONPATH=src python -m v5_memo.fullraw_service
```

Default bind: `127.0.0.1:9901`. Default archive roots:
OpenAlex, Semantic Scholar papers, Semantic Scholar abstracts, PubMed, and
bioRxiv/medRxiv under `sb:researka-database/raw`. This backend streams gzipped
raw files and returns the first receipt candidates it finds. It enables the LLM
layer to use the full raw archive surface now; it is not a fast exhaustive
Tantivy/DuckDB index.

Production fullraw search should use the indexed FTS backend:

```bash
PYTHONPATH=src python -m v5_memo.fullraw_index build
PYTHONPATH=src python -m v5_memo.fullraw_index serve
```

Default bind: `127.0.0.1:9902`. Default index path:
`/var/lib/v5-memo/fullraw_index.sqlite`. The builder reads the same raw manifest
as the cold scanner, normalizes title/abstract/DOI/source fields, writes a
resumable SQLite FTS5 index, and dedupes completed files. The search service
answers the same `POST /search` contract as the cold scanner, but returns BM25
ranked hits from the persistent inverted index instead of first matches in file
order.

The indexed backend also stores a persistent `term_map` table for query
expansion. Defaults cover high-value memo language such as
`management -> manager/managers/managerial`, `forecast -> forecasts/guidance`,
and `disclosure -> disclosures/disclose/disclosed`. This is why a query like
`management forecast disclosure` can match a paper titled
`Factors Associated with the Disclosure of Managers' Forecasts` even though the
exact word `management` is absent.

Inspect the actual expansion used for a query:

```bash
PYTHONPATH=src python -m v5_memo.fullraw_index explain \
  "management forecast disclosure"
```

The HTTP `/search` response also includes `meta.expanded_query` and
`meta.term_groups` for auditability.

On the VPS, install and start the systemd units in `deploy/`:

```bash
cp deploy/v5-memo-fullraw-index*.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now v5-memo-fullraw-index.service
systemctl enable --now v5-memo-fullraw-index-build.service
```

Then point V5 at the indexed endpoint:

```bash
V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL=http://127.0.0.1:9902/search
```

Coverage note: until `v5-memo.fullraw_index stats` reports all manifest files
complete, the indexed endpoint is fast and ranked over the indexed subset. Once
complete, it is the intended full 470M+ raw-corpus retrieval path.

For the full 470M+ build, use Storage Box-backed immutable shards rather than a
single live SQLite database on a remote mount. See
`docs/architecture/fullraw-storage-box-shards.md` and
`deploy/v5-memo-fullraw-shards.env.example`.

Hybrid mode searches the full raw service first when configured, then Researka,
then OpenAlex, and dedupes receipts:

```bash
PYTHONPATH=src python -m v5_memo \
  --searcher hybrid \
  --planner minimax \
  --writer minimax \
  --topic "longevity resilience" \
  --query "NAD salvage mitochondrial stress"
```

Smart mode is the shortest command for the current best path: MiniMax plans
queries, V5 searches hybrid corpus surfaces, then MiniMax writes from locked
receipts. If `V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL` is set, smart mode includes
the full raw corpus first.

```bash
PYTHONPATH=src python -m v5_memo \
  --searcher smart \
  --topic "longevity resilience" \
  --query "NAD salvage mitochondrial stress"
```

Coverage truth:

```bash
PYTHONPATH=src python -m v5_memo --coverage-report
PYTHONPATH=src python -m v5_memo --require-full-raw-corpus
```

`--require-full-raw-corpus` and `--searcher fullraw` fail unless a real 450M+
local raw-corpus search service/index is configured through
`V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL`.

MiniMax-M3 writer pass:

```bash
MINIMAX_API_KEY=... PYTHONPATH=src python -m v5_memo \
  --planner minimax \
  --writer minimax \
  --topic "longevity resilience" \
  --query "NAD salvage mitochondrial stress" \
  --query "exercise response mitochondrial repair"
```

The MiniMax planner proposes search angles. Retrieval, dedupe, scoring, and
receipt binding stay deterministic; the writer must preserve every locked receipt
ID.

Optional MiniMax overrides:

```bash
V5_MEMO_MINIMAX_API_KEY=...
V5_MEMO_MINIMAX_BASE_URL=https://api.minimax.io/anthropic
V5_MEMO_MINIMAX_MODEL=MiniMax-M3
V5_MEMO_MINIMAX_TIMEOUT_SECONDS=60
V5_MEMO_MINIMAX_MAX_TOKENS=1200
```
