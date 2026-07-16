import gzip
import json
import textwrap
from pathlib import Path

import pytest

from v5_memo import fullraw_index
from v5_memo.fullraw_service import RawFile, iter_raw_file_hits


def _write_gzip(path: Path, text: str) -> None:
    with gzip.open(path, "wb") as fh:
        fh.write(text.encode("utf-8"))


def test_iter_raw_file_hits_reads_local_jsonl_fixture(tmp_path: Path) -> None:
    source = tmp_path / "openalex.jsonl.gz"
    _write_gzip(
        source,
        json.dumps({
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.raw/one",
            "display_name": "NAD mitochondrial exercise adaptation",
            "abstract_inverted_index": {"NAD": [0], "mitochondrial": [1], "repair": [2]},
            "publication_year": 2025,
            "cited_by_count": 10,
            "type": "article",
            "is_retracted": True,
        }) + "\n",
    )

    hits = list(iter_raw_file_hits(RawFile(source="openalex", format="openalex_jsonl", remote=f"file://{source}")))

    assert hits[0]["doi"] == "10.raw/one"
    assert hits[0]["source"] == "openalex"
    assert hits[0]["abstract"] == "NAD mitochondrial repair"
    assert hits[0]["document_type"] == "article"
    assert hits[0]["is_retracted"] is True
    assert hits[0]["retraction_status_known"] is True


@pytest.mark.parametrize("field", ["publicationtypes", "publicationTypes"])
def test_semantic_scholar_publication_type_aliases_are_preserved(
    tmp_path: Path,
    field: str,
) -> None:
    source = tmp_path / f"semantic-{field}.jsonl.gz"
    _write_gzip(
        source,
        json.dumps({
            "corpusid": 987,
            "title": "Human intervention evidence",
            "abstract": "The intervention changed the endpoint.",
            field: ["JournalArticle", "ClinicalTrial"],
        }) + "\n",
    )

    hits = list(iter_raw_file_hits(RawFile(
        source="semantic_scholar",
        format="semantic_scholar_jsonl",
        remote=f"file://{source}",
    )))

    assert hits[0]["publication_types"] == ("JournalArticle", "ClinicalTrial")


def test_iter_raw_file_hits_skips_invalid_jsonl_rows(tmp_path: Path) -> None:
    source = tmp_path / "openalex.jsonl.gz"
    _write_gzip(
        source,
        "{bad json}\n" + json.dumps({
            "doi": "10.raw/real",
            "display_name": "NAD repair response",
            "abstract": "NAD was measured directly.",
            "publication_year": 2025,
        }) + "\n",
    )

    hits = list(iter_raw_file_hits(RawFile(source="openalex", format="openalex_jsonl", remote=f"file://{source}")))

    assert [hit["doi"] for hit in hits] == ["10.raw/real"]


def test_iter_raw_file_hits_tolerates_semantic_scholar_truncated_gzip_after_valid_rows(tmp_path: Path) -> None:
    source = tmp_path / "semantic.jsonl.gz"
    source.write_bytes(gzip.compress(json.dumps({
        "corpusid": 123,
        "title": "NAD mitochondrial repair",
        "abstract": "NAD changed mitochondrial repair.",
    }).encode("utf-8") + b"\n")[:-8])

    hits = list(iter_raw_file_hits(RawFile(
        source="semantic_scholar",
        format="semantic_scholar_jsonl",
        remote=f"file://{source}",
    )))

    assert [hit["semantic_scholar_id"] for hit in hits] == ["123"]


def test_iter_raw_file_hits_keeps_semantic_scholar_abstract_only_rows(tmp_path: Path) -> None:
    source = tmp_path / "s2_abstracts.jsonl.gz"
    _write_gzip(
        source,
        json.dumps({
            "corpusid": 12345,
            "openaccessinfo": {"externalids": {"DOI": "10.raw/s2"}},
            "abstract": "Resveratrol exercise training changed mitochondrial adaptation.",
        }) + "\n",
    )

    hits = list(iter_raw_file_hits(RawFile(
        source="semantic_scholar_abstracts",
        format="semantic_scholar_jsonl",
        remote=f"file://{source}",
    )))

    assert hits[0]["title"] == ""
    assert hits[0]["doi"] == "10.raw/s2"
    assert hits[0]["semantic_scholar_id"] == "12345"
    assert hits[0]["abstract"] == "Resveratrol exercise training changed mitochondrial adaptation."


def test_iter_raw_file_hits_reads_local_pubmed_xml_fixture(tmp_path: Path) -> None:
    source = tmp_path / "pubmed.xml.gz"
    _write_gzip(
        source,
        textwrap.dedent(
            """\
            <?xml version="1.0" encoding="utf-8"?>
            <PubmedArticleSet>
              <PubmedArticle>
                <MedlineCitation>
                  <PMID>123</PMID>
                  <Article>
                    <Journal><Title>Test Journal</Title></Journal>
                    <ArticleTitle>NAD mitochondrial stress response</ArticleTitle>
                    <ELocationID EIdType="doi">10.raw/pubmed</ELocationID>
                    <Abstract><AbstractText>NAD repair changed after exercise.</AbstractText></Abstract>
                    <PublicationTypeList><PublicationType>Retracted Publication</PublicationType></PublicationTypeList>
                    <Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
                  </Article>
                  <CommentsCorrections RefType="RetractionIn"><RefSource>Retraction notice</RefSource></CommentsCorrections>
                </MedlineCitation>
              </PubmedArticle>
            </PubmedArticleSet>
            """
        ),
    )

    hits = list(iter_raw_file_hits(RawFile(source="pubmed", format="pubmed_xml", remote=f"file://{source}")))

    assert hits[0]["pmid"] == "123"
    assert hits[0]["doi"] == "10.raw/pubmed"
    assert hits[0]["source"] == "pubmed"
    assert hits[0]["publication_types"] == ("Retracted Publication",)
    assert hits[0]["is_retracted"] is True
    assert hits[0]["retraction_status_known"] is True
    assert hits[0]["correction_status"] == "RetractionIn"


def test_strict_5tb_service_keeps_secret_env_file() -> None:
    deploy_dir = Path(__file__).resolve().parents[1] / "deploy"
    config = deploy_dir / "v5-memo-fullraw-index-strict-5tb.conf"
    env_example = (deploy_dir / "v5-memo-fullraw-shards.env.example").read_text()
    env_files = [line for line in config.read_text().splitlines() if line.startswith("EnvironmentFile")]

    assert env_files == [
        "EnvironmentFile=",
        "EnvironmentFile=/etc/researka-fullraw.env",
        "EnvironmentFile=-/etc/researka-fullraw-overrides.env",
    ]
    assert "/etc/v5-memo/" not in config.read_text()
    assert "TimeoutStopSec=120" in config.read_text()
    assert "TimeoutStopFailureMode=kill" in config.read_text()
    assert "KillMode=control-group" in config.read_text()
    assert "SendSIGKILL=yes" in config.read_text()
    assert "Environment=RESEARKA_FULLRAW_SEARCH_ISOLATED=1" in config.read_text()
    assert "Environment=RESEARKA_FULLRAW_SWEEP_PASS_SHARD_LIMIT=32" in config.read_text()
    assert "Environment=RESEARKA_FULLRAW_SWEEP_WORKERS=8" in config.read_text()
    assert "Environment=RESEARKA_FULLRAW_SWEEP_MAX_INFLIGHT=" not in config.read_text()
    assert "Environment=RESEARKA_FULLRAW_SWEEP_PRIORITY_BURST=0" in config.read_text()
    assert "Environment=RESEARKA_FULLRAW_SWEEP_MAX_QUEUE=4" in config.read_text()
    assert "Environment=RESEARKA_FULLRAW_SEARCH_BUDGET_SECONDS=7200" in config.read_text()
    assert "Environment=RESEARKA_FULLRAW_FOREGROUND_SWEEP_WAIT_SECONDS=7200" in config.read_text()
    assert "Environment=RESEARKA_FULLRAW_SWEEP_TIMEOUT_SECONDS=900" in config.read_text()
    assert "Environment=RESEARKA_FULLRAW_SWEEP_SHARD_TIMEOUT_SECONDS=20" in config.read_text()
    assert "Environment=RESEARKA_FULLRAW_SEARCH_PREFIX_SHARDS=128" in config.read_text()
    assert "Environment=RESEARKA_FULLRAW_SWEEP_CACHE_DIR=/var/lib/v5-memo/fullraw-sweep-cache" in config.read_text()
    assert "Environment=RESEARKA_FULLRAW_SHARD_CATALOG_PATH=/var/lib/v5-memo/fullraw-shard-catalog.json" in config.read_text()
    assert "Environment=RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_DIR=/var/lib/v5-memo/shard-cache-5tb" in config.read_text()
    assert "Environment=RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MAX_BYTES=auto" in config.read_text()
    assert "RESEARKA_FULLRAW_SEARCH_ISOLATED=1" in env_example
    assert "RESEARKA_FULLRAW_SWEEP_PASS_SHARD_LIMIT=32" in env_example
    assert "RESEARKA_FULLRAW_INDEX_PATH=/var/lib/v5-memo/index/fullraw_index.sqlite" in env_example
    assert "RESEARKA_FULLRAW_SHARD_DIR=/var/lib/v5-memo/fullraw-fts-remote" in env_example
    assert "RESEARKA_FULLRAW_SWEEP_CACHE_DIR=/var/lib/v5-memo/fullraw-sweep-cache" in env_example
    assert "RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MAX_BYTES=auto" in env_example
    assert "RESEARKA_FULLRAW_SHARD_CATALOG_PATH=/var/lib/v5-memo/fullraw-shard-catalog.json" in env_example
    assert "RESEARKA_FULLRAW_SEARCH_BUDGET_SECONDS=7200" in env_example
    assert "RESEARKA_FULLRAW_FOREGROUND_SWEEP_WAIT_SECONDS=7200" in env_example
    assert "RESEARKA_FULLRAW_SWEEP_TIMEOUT_SECONDS=900" in env_example
    assert "RESEARKA_FULLRAW_SWEEP_SHARD_TIMEOUT_SECONDS=20" in env_example
    assert "RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_DIR=/var/lib/v5-memo/shard-cache-5tb" in env_example
    assert "RESEARKA_FULLRAW_SWEEP_WORKERS=8" in env_example
    assert "RESEARKA_FULLRAW_SWEEP_MAX_INFLIGHT=2" in env_example
    assert "RESEARKA_FULLRAW_SWEEP_PRIORITY_BURST=0" in env_example
    assert "RESEARKA_FULLRAW_SWEEP_MAX_QUEUE=4" in env_example


def test_v5_isolated_fullraw_service_uses_v5_lane() -> None:
    deploy_dir = Path(__file__).resolve().parents[1] / "deploy"
    config = (deploy_dir / "v5-memo-isolated-fullraw-search.service").read_text()

    assert "v5-memo-fullraw-shard-cache-mount.service" not in config
    assert "v5-memo-fullraw-fts-root-mount.service" not in config
    assert "v5-memo-isolated-fullraw-fts-mount.service" in config
    assert "EnvironmentFile=/etc/v5-memo/env" in config
    assert "EnvironmentFile=/etc/v5-memo/isolated-fullraw.env" in config
    assert "Environment=RESEARKA_FULLRAW_INDEX_PORT=9915" in config
    assert "Environment=V5_MEMO_FULL_RAW_INDEX_PORT=9915" in config
    assert "Environment=RESEARKA_FULLRAW_SEARCH_ISOLATED=1" in config
    assert "Environment=RESEARKA_FULLRAW_SWEEP_CACHE_DIR=/var/lib/v5-memo/v5-fullraw-sweep-cache" in config
    assert "Environment=RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_DIR=/var/lib/v5-memo/v5-shard-cache-5tb" in config
    assert "same three-copy resource ceiling" in config
    assert "Environment=RESEARKA_FULLRAW_SWEEP_PASS_SHARD_LIMIT=12" in config
    assert "Environment=RESEARKA_FULLRAW_SWEEP_NO_HIT_STOP_SHARDS=128" in config
    assert "Environment=RESEARKA_FULLRAW_SWEEP_MAX_INFLIGHT=auto" in config
    assert "Environment=RESEARKA_FULLRAW_SWEEP_PRIORITY_BURST=0" in config
    assert "Environment=RESEARKA_FULLRAW_SWEEP_MAX_QUEUE=64" in config
    assert "Environment=RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MAX_BYTES=auto" in config
    assert "Environment=RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MIN_FREE_GB" not in config
    assert "Auto workers = dynamic local cache budget / per-worker cache budget" in config
    assert "Environment=RESEARKA_FULLRAW_SWEEP_WORKER_CACHE_BYTES=8589934592" in config
    assert "Environment=RESEARKA_FULLRAW_SWEEP_WORKER_CACHE_GB" not in config
    assert "Environment=RESEARKA_FULLRAW_SWEEP_SHARD_TIMEOUT_SECONDS=180" in config
    assert "Environment=RESEARKA_FULLRAW_MAX_VARIANTS=1" in config
    assert "Environment=RESEARKA_FULLRAW_DOI_ABSTRACT_BACKFILL_LIMIT=16" in config
    assert "Environment=RESEARKA_FULLRAW_SWEEP_WORKERS=" not in config
    assert "/etc/researka-fullraw.env" not in config
    assert "/etc/researka-fullraw-overrides.env" not in config


def test_v5_isolated_fullraw_env_overrides_shared_shard_dir() -> None:
    deploy_dir = Path(__file__).resolve().parents[1] / "deploy"
    env_example = (deploy_dir / "v5-memo-isolated-fullraw.env.example").read_text()

    assert "RESEARKA_FULLRAW_SHARD_DIR=/var/lib/v5-memo/v5-isolated-fullraw-fts-remote" in env_example
    assert "RESEARKA_FULLRAW_SWEEP_CACHE_DIR=/var/lib/v5-memo/v5-fullraw-sweep-cache" in env_example
    assert "RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_DIR=/var/lib/v5-memo/v5-shard-cache-5tb" in env_example
    assert "RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MAX_BYTES=auto" in env_example
    assert "Auto workers = dynamic local cache budget / per-worker cache budget" in env_example
    assert "RESEARKA_FULLRAW_SWEEP_WORKER_CACHE_BYTES=8589934592" in env_example
    assert "RESEARKA_FULLRAW_SWEEP_WORKER_CACHE_GB" not in env_example
    assert "RESEARKA_FULLRAW_FOREGROUND_SWEEP_WAIT_SECONDS=0" in env_example
    assert "same three-copy resource ceiling" in env_example
    assert "RESEARKA_FULLRAW_SWEEP_PASS_SHARD_LIMIT=12" in env_example
    assert "RESEARKA_FULLRAW_SWEEP_NO_HIT_STOP_SHARDS=128" in env_example
    assert "RESEARKA_FULLRAW_SWEEP_MAX_INFLIGHT=auto" in env_example
    assert "RESEARKA_FULLRAW_SWEEP_TIMEOUT_SECONDS=900" in env_example
    assert "RESEARKA_FULLRAW_SWEEP_SHARD_TIMEOUT_SECONDS=180" in env_example
    assert "RESEARKA_FULLRAW_SWEEP_PRIORITY_BURST=0" in env_example
    assert "RESEARKA_FULLRAW_SWEEP_MAX_QUEUE=64" in env_example
    assert "RESEARKA_FULLRAW_INDEX_PORT=9915" in env_example


def test_v5_portfolio_publisher_keeps_strict_sweep_batch_focused() -> None:
    deploy_dir = Path(__file__).resolve().parents[1] / "deploy"
    config = (deploy_dir / "v5-memo-portfolio-publish.service").read_text()
    isolation = (deploy_dir / "zzz-v5-portfolio-isolated-fullraw.conf").read_text()
    isolation_installer = (deploy_dir / "install-v5-portfolio-isolation.sh").read_text()
    timer = (deploy_dir / "v5-memo-portfolio-publish.timer").read_text()
    prepare_config = (deploy_dir / "v5-memo-portfolio-prepare.service").read_text()
    prepare_timer = (deploy_dir / "v5-memo-portfolio-prepare.timer").read_text()
    catchup_config = (deploy_dir / "v5-memo-portfolio-catchup.service").read_text()
    catchup_timer = (deploy_dir / "v5-memo-portfolio-catchup.timer").read_text()

    assert "TimeoutStartSec=150min" in config
    assert "Publish only prequalified supply" in config
    assert "Environment=V5_MEMO_PORTFOLIO_MAX_LEADS=1" in config
    assert "Environment=V5_MEMO_PORTFOLIO_LEAD_TIMEOUT_SECONDS=1200" in config
    assert "Environment=V5_MEMO_PORTFOLIO_DECISION_WAIT_SECONDS=600" in config
    assert "--submit --ready-only" in config
    assert "--auto-discover-leads" not in config
    assert '--max-leads "${V5_MEMO_PORTFOLIO_MAX_LEADS:-1}"' in config
    assert '--lead-timeout-seconds "${V5_MEMO_PORTFOLIO_LEAD_TIMEOUT_SECONDS:-1200}"' in config
    assert '--decision-wait-seconds "${V5_MEMO_PORTFOLIO_DECISION_WAIT_SECONDS:-600}"' in config
    assert "OnCalendar=*-*-* 00/8:20:00" in timer
    assert "Environment=V5_MEMO_READY_BUFFER_SIZE=3" in prepare_config
    assert "Environment=V5_MEMO_PREPARE_MAX_LEADS=3" in prepare_config
    assert '--ready-buffer-size "${V5_MEMO_READY_BUFFER_SIZE:-3}"' in prepare_config
    assert "--resource-aware-max-leads" in prepare_config
    assert "--validate-publish-quality" not in prepare_config
    assert "--submit" not in prepare_config
    assert "--state-path /var/lib/v5-memo/portfolio-runs/state.json" in prepare_config
    assert "/usr/bin/flock -n 9" in prepare_config
    assert "/usr/bin/flock -w 900 9" in config
    assert "--record-noop-status lock_busy" in prepare_config
    assert "--record-noop-status lock_busy" in config
    assert "OnCalendar=*-*-* *:00,15,30,45:00" in prepare_timer
    assert "RandomizedDelaySec=2min" in prepare_timer
    assert "Unit=v5-memo-portfolio-prepare.service" in prepare_timer
    assert "--submit --ready-only" in catchup_config
    assert "--max-leads 1" in catchup_config
    assert "--auto-discover-leads" not in catchup_config
    assert "TimeoutStartSec=35min" in prepare_config
    assert "TimeoutStartSec=45min" in catchup_config
    assert "/usr/bin/flock -w 900 9" in catchup_config
    assert "--lead-timeout-seconds 1200" in catchup_config
    assert "--record-noop-status lock_busy" in catchup_config
    assert "OnCalendar=*-*-* *:10,25,40,55:00" in catchup_timer
    assert "Unit=v5-memo-portfolio-catchup.service" in catchup_timer
    assert "researka-fullraw-search.service" in isolation
    assert "Wants=network-online.target researka-fullraw-search.service" in isolation
    assert "After=network-online.target researka-fullraw-search.service" in isolation
    assert "RESEARKA_FULLRAW_SEARCH_URL=http://127.0.0.1:9903/search" in isolation
    assert "RESEARKA_FULL_RAW" not in isolation
    assert "V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL=http://127.0.0.1:9903/search" in isolation
    assert "V5_MEMO_FULL_RAW_INDEX_PORT=9903" in isolation
    assert "9915" not in isolation
    assert "v5-memo-portfolio-prepare.service" in isolation_installer
    assert "v5-memo-portfolio-catchup.service" in isolation_installer
    assert "v5-memo-portfolio-publish.service" in isolation_installer
    assert '"$deploy_dir/$unit"' in isolation_installer
    assert '"$unit_dir/$unit"' in isolation_installer
    assert "rm -f" not in isolation_installer
    assert "systemctl daemon-reload" in isolation_installer


def test_v5_isolated_fullraw_mount_uses_separate_vfs_cache() -> None:
    deploy_dir = Path(__file__).resolve().parents[1] / "deploy"
    config = (deploy_dir / "v5-memo-isolated-fullraw-fts-mount.service").read_text()

    assert "sb:researka-database/index/v5/fullraw-fts" in config
    assert "/var/lib/v5-memo/v5-isolated-fullraw-fts-remote" in config
    assert "/var/cache/v5-memo/v5-isolated-rclone-vfs-cache" in config
    assert "--vfs-cache-mode=minimal" in config
    assert "--vfs-cache-max-size=2G" in config
    assert "--vfs-cache-max-age=30m" in config
    assert "--vfs-read-chunk-streams=4" in config
    assert "--vfs-read-chunk-size=16Mi" in config
    assert "--vfs-read-chunk-size-limit=off" in config
    assert "/var/lib/v5-memo/fullraw-fts-remote" not in config


def test_v5_writable_shard_cache_mount_caps_root_vfs_cache() -> None:
    deploy_dir = Path(__file__).resolve().parents[1] / "deploy"
    config = (deploy_dir / "v5-memo-fullraw-shard-cache-mount.service").read_text()

    assert "sb:researka-database/index/v5/fullraw-shard-cache-5tb" in config
    assert "/mnt/HC_Volume_106011525/v5-memo/fullraw-shard-cache-remote" in config
    assert "mountpoint -q /mnt/HC_Volume_106011525/v5-memo/fullraw-shard-cache-remote" in config
    assert "--vfs-cache-mode=writes" in config
    assert "--vfs-cache-max-size=8G" in config
    assert "--vfs-cache-max-age=30m" in config


def test_fast_shard_cache_health_skips_dynamic_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("V5_MEMO_FULL_RAW_SHARD_LOCAL_CACHE_MAX_BYTES", "auto")

    def fail_dynamic_budget(cache_dir: Path | None = None) -> int | None:
        del cache_dir
        raise AssertionError("fast health must not scan cache budget")

    monkeypatch.setattr(fullraw_index, "_shard_local_cache_max_bytes", fail_dynamic_budget)

    health = fullraw_index._shard_local_cache_health(include_dynamic_budget=False)

    assert health["dir"] == str(tmp_path)
    assert health["exists"] is True
    assert health["max_bytes_config"] == "auto"
    assert "max_bytes" not in health
