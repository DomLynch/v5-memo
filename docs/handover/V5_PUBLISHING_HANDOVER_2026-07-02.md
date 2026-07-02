# V5 Publishing Handover - 2026-07-02

## Scope

This handover is V5 Memo only.

Do not edit, restart, or borrow success from v3, v4, or v6. They are independent lanes.

Repo:
- GitHub: https://github.com/DomLynch/v5-memo
- Branch: `codex/019ed112/main`
- Local/VPS path: `/Users/domininclynch/Desktop/Business/V5 Memo` and `/opt/v5-memo`

## Current Answer

V5 is not publishing today.

Latest verified public state:
- Public endpoint: `https://researka.org/api/publications?surface=alpha&limit=100`
- Total alpha items returned: `93`
- V5 public items: `7`
- V5 public items dated `2026-07-02`: `0`
- Latest V5 public item:
  - `createdAt`: `2026-06-30T22:42:28.278981+04:00`
  - `agentId`: `v5-memo-agent`
  - `artifactId`: `894750f0-2e4a-4214-9f4e-f641edf929ff`
  - `title`: `Cold Water Immersion: Performance and Strength Training Adaptation`
  - `decision`: `accept`
  - `doi`: `10.17605/OSF.IO/E296K`

Current blocker:
- V5 has a plausible active lead, but the strict 5TB fullraw receipt is still partial.
- Active lead: `urolithin muscle endurance older adults trial`
- Latest checked receipt before this handover:
  - `shards_searched`: `844/1525`
  - `partial_shard_search`: `true`
  - `sweep_failed_shards`: `0`
  - searched sources: `4`
  - trusted results: `0` because partial results are blocked for publish.

## Publish Rule

Do not submit a V5 memo unless the fullraw receipt shows:

```text
shards_searched = 1525
partial_shard_search = false
sweep_failed_shards = 0
source_count_searched >= 5
```

Then run the V5 publish-quality/A-grade gate. Only submit if that passes.

## Why V5 Has Not Published Today

The blocker is upstream of writing:

1. V5 is correctly using the isolated 5TB fullraw search service on `127.0.0.1:9915`.
2. The active lead is still warming through the 1525-shard gate.
3. The lead has partial visible hits, but partial results are not trusted for public publishing.
4. Completed older caches were mostly CWI/resveratrol/metformin/protein; dry-run gates found no safe fresh non-duplicate publish.
5. Therefore V5 is correctly not submitting today.

This is not currently a Researka platform outage and not a prose/writer problem.

## Current Live V5 State

Last verified sync before writing this file:

```text
MacBook HEAD = 7565bec93b9ead9f76fe0e7b1fa0244193f1e911
GitHub branch codex/019ed112/main = 7565bec93b9ead9f76fe0e7b1fa0244193f1e911
VPS /opt/v5-memo HEAD = 7565bec93b9ead9f76fe0e7b1fa0244193f1e911
VPS dirt = 0
V5 service = active
V5 PID = 2925139
```

Live V5 health receipt:

```text
HEALTH ok=true fast=true
cache=/dev/shm/v5-memo-shard-cache-5tb
max_bytes=12884901888
min_shards=1525
min_sources=5
require_complete=1
inflight=2
queued=0
max_inflight=1
workers=6
```

Latest active lead receipt:

```text
LEAD query=urolithin muscle endurance older adults trial
status=running
shards=844/1525
remaining=681
partial=true
failed=0
sources=4
hits=0
```

## Active Lead Evidence

Partial diagnostic with `allow_partial_results=true` previously showed these direct-looking urolithin hits:

```text
Effect of Urolithin A Supplementation on Muscle Endurance and Mitochondrial Health in Older Adults: A Randomized Clinical Trial.
doi=10.1001/jamanetworkopen.2021.44279

Urolithin A improves muscle strength, exercise performance, and biomarkers of mitochondrial health in a randomized trial in middle-aged adults
doi=10.1016/j.xcrm.2022.100633

Evaluating the Impact of Urolithin A Supplementation on Running Performance, Recovery, and Mitochondrial Biomarkers in Highly Trained Male Distance Runners.
doi=10.1007/s40279-025-02292-5
```

These make the lead worth finishing, but not worth submitting before strict coverage completes.

## Main Files To Audit

CLI / submit path:
- https://github.com/DomLynch/v5-memo/blob/codex/019ed112/main/src/v5_memo/__main__.py
- Important anchors:
  - args and publish flags: `src/v5_memo/__main__.py:107`
  - fullraw wider recall and build kwargs: `src/v5_memo/__main__.py:261`
  - MemoBuildError receipt write: `src/v5_memo/__main__.py:291`
  - Researka submit config / blocker / submit: `src/v5_memo/__main__.py:329`
  - final publish blocker wrapper: `src/v5_memo/__main__.py:727`

Fullraw search service:
- https://github.com/DomLynch/v5-memo/blob/codex/019ed112/main/src/v5_memo/fullraw_index.py
- Important anchors:
  - materialize and search one shard: `src/v5_memo/fullraw_index.py:1263`
  - cache-fit batch sizing: `src/v5_memo/fullraw_index.py:1277`
  - strict shard coverage receipt fields: `src/v5_memo/fullraw_index.py:2547`
  - ready gate: `src/v5_memo/fullraw_index.py:2645`
  - HTTP `/search` handler: `src/v5_memo/fullraw_index.py:3706`
  - cache-only queue / partial-progress response: `src/v5_memo/fullraw_index.py:3834`

Publish-quality gate:
- https://github.com/DomLynch/v5-memo/blob/codex/019ed112/main/src/v5_memo/gate.py
- Important anchors:
  - minimum tier/score/novelty: `src/v5_memo/gate.py:94`
  - candidate publish blocker: `src/v5_memo/gate.py:102`
  - failure diagnostics: `src/v5_memo/gate.py:342`

Memo pipeline:
- https://github.com/DomLynch/v5-memo/blob/codex/019ed112/main/src/v5_memo/pipeline.py
- Important anchors:
  - `build_alpha_memo`: `src/v5_memo/pipeline.py:34`
  - publish-quality candidate filtering: `src/v5_memo/pipeline.py:131`
  - context receipt dropping: `src/v5_memo/pipeline.py:156`

Publisher:
- https://github.com/DomLynch/v5-memo/blob/codex/019ed112/main/src/v5_memo/publisher.py

V5 isolated service config:
- https://github.com/DomLynch/v5-memo/blob/codex/019ed112/main/deploy/v5-memo-isolated-fullraw-search.service
- Important anchors:
  - isolated port `9915`: `deploy/v5-memo-isolated-fullraw-search.service:12`
  - strict gates: `deploy/v5-memo-isolated-fullraw-search.service:16`
  - full 1525 sweep: `deploy/v5-memo-isolated-fullraw-search.service:21`
  - pass size: `deploy/v5-memo-isolated-fullraw-search.service:22`
  - max inflight: `deploy/v5-memo-isolated-fullraw-search.service:25`
  - tmpfs cache: `deploy/v5-memo-isolated-fullraw-search.service:32`

V5 isolated env example:
- https://github.com/DomLynch/v5-memo/blob/codex/019ed112/main/deploy/v5-memo-isolated-fullraw.env.example
- Important anchors:
  - V5 search URL: `deploy/v5-memo-isolated-fullraw.env.example:7`
  - shard source path: `deploy/v5-memo-isolated-fullraw.env.example:14`
  - tmpfs cache path: `deploy/v5-memo-isolated-fullraw.env.example:17`
  - strict gates: `deploy/v5-memo-isolated-fullraw.env.example:23`
  - sweep settings: `deploy/v5-memo-isolated-fullraw.env.example:29`

Tests likely relevant to audit:
- `tests/test_fullraw_index.py`
- `tests/test_fullraw_service.py`
- `tests/test_coverage.py`
- `tests/test_cli.py`
- `tests/test_v5_memo.py`

## Recent Fixes Already In V5

Current deployed commit before this handover:

```text
7565bec Keep V5 shard cache local under reservations
```

What it changed:
- Fixed shard-cache eviction accounting so existing in-flight reservations are included before materializing the next shard.
- Prevents V5 falling back to slow remote shard search just because tmpfs cache accounting ignored already reserved bytes.
- Added regression test:
  - `tests/test_fullraw_index.py::test_materialized_shard_path_evicts_for_existing_reservations`

Verification for that commit when made:

```text
16 passed in 0.10s
ruff: All checks passed
mypy: Success: no issues found in 3 source files
```

Operational fixes already applied:
- V5 isolated search is on port `9915`.
- V5 cache is under `/dev/shm/v5-memo-shard-cache-5tb`.
- Old V5-only root cache `/var/lib/v5-memo/v5-shard-cache-5tb` was pruned.
- Shared/non-V5 cache paths were not touched.

## Known Challenges

1. Full coverage is slow.
   - Current pass rate observed around one 31-32 shard batch per several minutes.
   - Strict gate still needs all `1525` shards.

2. V5 is configured with `RESEARKA_FULLRAW_SWEEP_MAX_INFLIGHT=1`.
   - This avoids choking the 5TB mount but limits throughput.
   - Raising this may speed up coverage but risks I/O contention and restarts.

3. Service restarts have historically reset or fragmented warm-up progress.
   - Do not restart V5 unless a code/config deploy requires it.
   - If the service is already live on the target commit, prefer leaving it running.

4. Partial hits are promising but not publish-safe.
   - `allow_partial_results=true` can reveal lead quality.
   - `allow_partial_results=false` must remain the publish path.

5. Candidate quality after full coverage is still unknown for the current urolithin lead.
   - Once full coverage lands, run the A-grade gate before submit.
   - Do not assume direct human hits mean the publish-quality gate will pass.

## Commands For GPT Pro Audit

Check public V5 state:

```bash
python3 - <<'PY'
import json, urllib.request
url='https://researka.org/api/publications?surface=alpha&limit=100'
with urllib.request.urlopen(url, timeout=20) as r:
    data=json.load(r)
items=data if isinstance(data,list) else data.get('items') or data.get('publications') or data.get('data') or []
v5=[x for x in items if 'v5' in str(x.get('agentId','')).lower()]
v5_sorted=sorted(v5, key=lambda x: x.get('createdAt',''))
latest=v5_sorted[-1] if v5_sorted else {}
today=[x for x in v5 if str(x.get('createdAt','')).startswith('2026-07-02')]
print('TOTAL_ALPHA=', len(items))
print('V5_COUNT=', len(v5))
print('TODAY_V5_JULY2_COUNT=', len(today))
print(json.dumps({k: latest.get(k) for k in ['createdAt','agentId','artifactId','title','decision','doi']}, indent=2))
PY
```

Check V5 VPS sync:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 -i ~/.ssh/binance_futures_tool root@100.96.74.1 \
  'cd /opt/v5-memo && git rev-parse HEAD && git status --porcelain=v1 && systemctl is-active v5-memo-isolated-fullraw-search.service'
```

Check V5 fullraw health and current lead:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 -i ~/.ssh/binance_futures_tool root@100.96.74.1 'bash -s' <<'SH'
set -euo pipefail
PID=$(systemctl show v5-memo-isolated-fullraw-search.service -p MainPID --value)
TOKEN=$(tr '\0' '\n' < /proc/$PID/environ | awk -F= '/^(RESEARKA_FULLRAW_INDEX_TOKEN|V5_MEMO_FULL_RAW_INDEX_TOKEN)=/{print $2; exit}')
curl -fsS http://127.0.0.1:9915/health | jq '{ok,fast_health,coverage_requirements,shard_cache,async_sweep}'
curl -fsS -X POST http://127.0.0.1:9915/search \
  -H "Authorization: Bearer ${TOKEN}" \
  -H 'Content-Type: application/json' \
  -d '{"query":"urolithin muscle endurance older adults trial","limit":10,"rank_mode":"relevance","cache_only":true,"queue_if_missing":true,"priority":true,"allow_partial_results":false}' \
| jq '{meta:{shard_receipt:.meta.shard_receipt, async_sweep:.meta.async_sweep}, result_count:(.results|length)}'
SH
```

If and only if receipt becomes full, run a no-massage gate check first:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 -i ~/.ssh/binance_futures_tool root@100.96.74.1 'bash -s' <<'SH'
set -euo pipefail
cd /opt/v5-memo
OUT=/tmp/v5-urolithin-full-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
python3 -m v5_memo \
  --topic "urolithin muscle endurance older adults trial" \
  --query "urolithin muscle endurance older adults trial" \
  --searcher fullraw \
  --planner seed \
  --selector deterministic \
  --writer template \
  --min-alpha-tier publishable \
  --require-full-raw-corpus \
  --output-dir "$OUT" \
  --publish-receipt-path "$OUT/publish-receipt.json"
cat "$OUT/publish-receipt.json"
find "$OUT" -maxdepth 1 -type f -print -exec wc -c {} \;
SH
```

Only submit if the gate produces a publishable memo and no blocker receipt.

## Audit Questions

1. Is `SWEEP_MAX_INFLIGHT=1` the correct safety/performance tradeoff for this VPS and 5TB rclone mount?
2. Can V5 safely increase throughput without reintroducing root-disk pressure or service restarts?
3. Is `PASS_SHARD_LIMIT=32` optimal, or should V5 use a smaller batch to commit progress more frequently?
4. Should V5 persist partial sweep progress more granularly so restarts lose less work?
5. Is the urolithin lead likely to pass publish-quality after full coverage, or should V5 also warm one backup lead?
6. Are the publish-quality blockers too strict, too lenient, or correctly aligned with Researka's current alpha acceptance gate?
7. Does the submit path correctly fail closed on missing credentials, cooldown, duplicate topics, partial receipts, and quality blockers?

## Strongest Current Next Step

Do not restart V5.

Let the active urolithin sweep finish to `1525/1525`, then run the no-massage gate check above. If it passes, submit with `--submit-researka --researka-list-if-accepted` and verify the public feed.

If the sweep stalls for more than two consecutive 10-minute windows with no shard progress and no temp-file growth, inspect worker threads and tmpfs files before changing config.

