# Fullraw Search: Storage Box Sharded Index Plan

## Decision

Use the Hetzner Storage Box for completed fullraw search index shards, not as the
live filesystem for one huge SQLite FTS database.

## Current Evidence

Verified on the VPS:

```text
rclone about sb:
Total:   1 TiB
Used:    895.064 GiB
Free:    128.936 GiB

rclone lsd sb:researka-database/raw:
biorxiv
openalex
pubmed
semantic_scholar

V5 indexed endpoint:
V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL=http://127.0.0.1:9902/search
V5_MEMO_FULL_RAW_INDEX_PATH=/var/lib/v5-memo/fullraw_index.sqlite
```

The current local index is a single SQLite FTS5 file. It is fast for local disk
reads, but it is not a good fit for direct remote Storage Box writes because FTS
does many small random reads/writes.

## Why Not One SQLite File On Storage Box

The Storage Box is cheap, large remote storage. It is good for archive files and
finished shard files. It is not ideal as a live random-write database volume.

Bad path:

```text
/mnt/storagebox/fullraw_index.sqlite
```

Why this is bad:

- SQLite FTS does many small random writes.
- Remote mounts add latency and disconnect risk.
- A mid-write network issue can corrupt or stall the live index.
- Query latency becomes remote-filesystem latency.

## Safe Architecture

Build local shard, upload finished shard, query with local cache.

```text
Storage Box raw corpus
  sb:researka-database/raw/openalex/...
  sb:researka-database/raw/pubmed/...
  sb:researka-database/raw/semantic_scholar/...
  sb:researka-database/raw/biorxiv/...

Builder on VPS
  reads raw .gz files from Storage Box
  builds one local SQLite FTS shard at a time
  verifies shard health
  uploads immutable shard to Storage Box

Storage Box index shards
  sb:researka-database/index/v5/fullraw-fts/openalex/updated_date=2025-07-23.sqlite
  sb:researka-database/index/v5/fullraw-fts/pubmed/baseline_2026_01.sqlite
  sb:researka-database/index/v5/fullraw-fts/semantic_scholar/part_000123.sqlite

Search service on VPS
  keeps hot shards in /var/cache/v5-memo/fullraw-shards
  downloads missing relevant shards from Storage Box
  searches local cached shards
  merges and reranks results
```

## Proposed Environment

```bash
V5_MEMO_FULL_RAW_SHARD_REMOTE=sb:researka-database/index/v5/fullraw-fts
V5_MEMO_FULL_RAW_SHARD_CACHE=/var/cache/v5-memo/fullraw-shards
V5_MEMO_FULL_RAW_SHARD_CACHE_MAX_GB=60
V5_MEMO_FULL_RAW_SHARD_BUILD_DIR=/var/lib/v5-memo/shard-build
V5_MEMO_FULL_RAW_SHARD_MANIFEST=/var/lib/v5-memo/fullraw_shard_manifest.json
```

Keep the existing live endpoint:

```bash
V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL=http://127.0.0.1:9902/search
```

## Build Flow

1. Pick one raw manifest partition.
2. Build one local FTS shard in `V5_MEMO_FULL_RAW_SHARD_BUILD_DIR`.
3. Run `PRAGMA integrity_check`.
4. Write shard metadata:
   - source
   - raw remote paths covered
   - paper count
   - byte size
   - term count
   - created commit SHA
5. Upload shard and metadata to `V5_MEMO_FULL_RAW_SHARD_REMOTE`.
6. Delete local build shard after upload unless it is in the hot cache.

## Query Flow

1. Expand query through the persisted `term_map`.
2. Use shard metadata to select likely shards.
3. Ensure those shards exist in the local cache.
4. Search cached shards locally with SQLite FTS5.
5. Merge BM25-ranked results across shards.
6. Return the same `POST /search` API shape V5 already uses.

## Files That Need Code Changes

- `src/v5_memo/fullraw_index.py`
  - split single-index class into single-shard primitive plus shard coordinator
  - add shard manifest read/write
  - add rclone upload/download helpers
  - merge results from multiple shard files

- `deploy/v5-memo-fullraw-index.service`
  - add shard cache env vars

- `deploy/v5-memo-fullraw-index-build.service`
  - build shards, not one monolithic `/var/lib/v5-memo/fullraw_index.sqlite`

- `tests/test_fullraw_index.py`
  - add tests for shard creation, manifest, cache fetch, multi-shard merge

## Storage Requirement

The current Storage Box has only `128.936 GiB` free. That is not enough for a
complete 470M+ FTS index. Increase the Storage Box first.

Practical target:

```text
minimum: 2 TB Storage Box
better: 5 TB Storage Box
```

Reason: the index can plausibly be hundreds of GB to over 1 TB, and shard
metadata/cache/temporary build files need headroom.

## Safe Next Implementation

Do not mount the Storage Box as the live SQLite path. Implement:

```text
local build -> verified shard -> rclone copy to Storage Box -> local cache search
```

This gives cheap storage without turning every search into remote filesystem
latency.
