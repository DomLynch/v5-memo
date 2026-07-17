import gzip
import json
import os
import shutil
import subprocess
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
    assert "RESEARKA_FULLRAW_MAX_VARIANTS=1" in env_example
    assert "RESEARKA_FULLRAW_INDEX_PORT=9915" in env_example


def test_v5_portfolio_publisher_keeps_strict_sweep_batch_focused() -> None:
    deploy_dir = Path(__file__).resolve().parents[1] / "deploy"
    config = (deploy_dir / "v5-memo-portfolio-publish.service").read_text()
    isolation = (deploy_dir / "zzz-v5-portfolio-isolated-fullraw.conf").read_text()
    dedicated_profile = (deploy_dir / "v5-portfolio-publish-fullraw.conf").read_text()
    owned_profile = (deploy_dir / "v5-memo-publish-fullraw-owned.conf").read_text()
    shared_sidecar_profile = (
        deploy_dir / "v5-memo-publish-fullraw-shared.conf"
    ).read_text()
    shared_profile = (deploy_dir / "v5-portfolio-shared-fullraw.conf").read_text()
    isolation_installer = (deploy_dir / "install-v5-portfolio-isolation.sh").read_text()
    shared_env = (deploy_dir / "v5-memo-portfolio-shared-fullraw.env").read_text()
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
    assert "Environment=V5_MEMO_PREPARE_MAX_LEADS=1" in prepare_config
    assert "one strict candidate at a time" in prepare_config
    assert "--auto-discover-leads" in prepare_config
    assert "--min-open-leads" not in prepare_config
    assert "--discover-count 1" in prepare_config
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
    assert "EnvironmentFile=/etc/v5-memo/portfolio-shared-fullraw.env" in isolation
    assert "Wants=v5-memo-publish-fullraw-search.service" in dedicated_profile
    assert "After=v5-memo-publish-fullraw-search.service" in dedicated_profile
    assert "V5_MEMO_PORTFOLIO_FULL_RAW_CORPUS_SEARCH_URL=http://127.0.0.1:9935/search" in dedicated_profile
    assert "EnvironmentFile=" not in dedicated_profile
    assert "not legacy fullraw sidecars" in owned_profile
    assert "ConditionPathExists=" in owned_profile
    assert "Shared 9903 owns fullraw search" in shared_sidecar_profile
    assert "ConditionPathExists=/run/researka-fullraw-allow-legacy-sidecars" in (
        shared_sidecar_profile
    )
    assert "Wants=network-online.target researka-fullraw-search.service" in shared_profile
    assert "UnsetEnvironment=V5_MEMO_PORTFOLIO_FULL_RAW_CORPUS_SEARCH_URL" in shared_profile
    assert "RESEARKA_FULLRAW_SEARCH_URL=http://127.0.0.1:9903/search" in shared_env
    assert "V5_MEMO_FULL_RAW_CORPUS_SEARCH_URL=http://127.0.0.1:9903/search" in shared_env
    assert "V5_MEMO_FULL_RAW_INDEX_PORT=9903" in shared_env
    shared_values = dict(
        line.split("=", 1)
        for line in shared_env.splitlines()
        if line and not line.startswith("#")
    )
    expected_shared_values = {
        "RESEARKA_FULLRAW_SHARD_DIR": "/var/lib/researka-fullraw/fullraw-fts",
        "RESEARKA_FULLRAW_SWEEP_CACHE_DIR": "/var/lib/researka-fullraw/sweep-cache",
        "RESEARKA_FULLRAW_SWEEP_SHARD_LIMIT": "1525",
        "RESEARKA_FULLRAW_SWEEP_PASS_SHARD_LIMIT": "32",
        "RESEARKA_FULLRAW_SWEEP_TTL_SECONDS": "604800",
        "V5_MEMO_FULL_RAW_SHARD_DIR": "/var/lib/researka-fullraw/fullraw-fts",
        "V5_MEMO_FULL_RAW_SWEEP_CACHE_DIR": "/var/lib/researka-fullraw/sweep-cache",
        "V5_MEMO_FULL_RAW_SWEEP_SHARD_LIMIT": "1525",
        "V5_MEMO_FULL_RAW_SWEEP_PASS_SHARD_LIMIT": "32",
        "V5_MEMO_FULL_RAW_SWEEP_TTL_SECONDS": "604800",
    }
    assert expected_shared_values.items() <= shared_values.items()
    assert "/var/lib/v5-memo/v5-fullraw-sweep-cache" not in shared_env
    assert "/var/lib/v5-memo/v5-isolated-fullraw-fts-remote" not in shared_env
    assert "9915" not in shared_env
    assert "v5-memo-portfolio-prepare.service" in isolation_installer
    assert "v5-memo-portfolio-catchup.service" in isolation_installer
    assert "v5-memo-portfolio-publish.service" in isolation_installer
    assert "v5-memo-publish-fullraw-fts-mount.service" in isolation_installer
    assert "v5-memo-publish-fullraw-search.service" in isolation_installer
    assert "dropin=zzzzz-v5-portfolio-fullraw-route.conf" in isolation_installer
    assert "V5_MEMO_PORTFOLIO_SEARCH_ROUTE" in isolation_installer
    assert "must be dedicated or shared" in isolation_installer
    assert 'route=${V5_MEMO_PORTFOLIO_SEARCH_ROUTE:-}' in isolation_installer
    assert "The marker is the durable operator opt-in" in isolation_installer
    assert "dedicated V5 fullraw requires V5_MEMO_ALLOW_DEDICATED_FULLRAW=1" in (
        isolation_installer
    )
    assert 'requires $dedicated_marker' in isolation_installer
    assert 'systemctl enable "$mount_unit" "$search_unit"' in isolation_installer
    assert 'mountpoint -q "$publish_mount"' in isolation_installer
    assert 'systemctl restart "$search_unit"' in isolation_installer
    assert 'systemctl disable --now "$search_unit" "$mount_unit"' in isolation_installer
    assert '"$deploy_dir/$unit"' in isolation_installer
    assert '"$unit_dir/$unit"' in isolation_installer
    assert '"$deploy_dir/$selected_profile"' in isolation_installer
    assert 'install_sidecar_profile "$owned_profile"' in isolation_installer
    assert 'install_sidecar_profile "$shared_sidecar_profile"' in isolation_installer
    assert "owned_dropin=zzzzzzzz-v5-publish-fullraw-owned.conf" in isolation_installer
    assert '"$unit_dir/$unit.d/$owned_dropin"' in isolation_installer
    assert '"$config_dir/portfolio-shared-fullraw.env"' in isolation_installer
    assert "rm -f" not in isolation_installer
    assert "systemctl daemon-reload" in isolation_installer


def test_v5_publish_fullraw_service_is_bounded_and_not_legacy() -> None:
    deploy_dir = Path(__file__).resolve().parents[1] / "deploy"
    search = (deploy_dir / "v5-memo-publish-fullraw-search.service").read_text()
    mount = (deploy_dir / "v5-memo-publish-fullraw-fts-mount.service").read_text()
    env = (deploy_dir / "v5-memo-publish-fullraw.env").read_text()

    assert "v5-memo-isolated-fullraw-search.service" not in search
    assert "BindsTo=v5-memo-publish-fullraw-fts-mount.service" in search
    assert "ExecStartPre=/usr/bin/mountpoint -q" in search
    assert "EnvironmentFile=/etc/v5-memo/env" in search
    assert "EnvironmentFile=/etc/v5-memo/publish-fullraw.env" in search
    assert search.index("EnvironmentFile=/etc/v5-memo/env") < search.index(
        "EnvironmentFile=/etc/v5-memo/publish-fullraw.env",
    )
    assert "RESEARKA_FULLRAW_INDEX_PORT=9935" in env
    assert "V5_MEMO_FULL_RAW_INDEX_PORT=9935" in env
    assert "RESEARKA_FULLRAW_SWEEP_WORKERS=1" in env
    assert "RESEARKA_FULLRAW_SWEEP_MAX_INFLIGHT=1" in env
    assert "RESEARKA_FULLRAW_SWEEP_MAX_QUEUE=4" in env
    assert "RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MAX_BYTES=auto" in env
    assert "RESEARKA_FULLRAW_SHARD_LOCAL_CACHE_MIN_FREE_GB=42" in env
    assert "RESEARKA_FULLRAW_SWEEP_WORKER_CACHE_BYTES=4294967296" in env
    assert "RESEARKA_FULLRAW_FAST_HEALTH=0" in env
    assert "RESEARKA_FULLRAW_SHARD_MANIFEST_STATS=1" in env
    assert "MemoryMax=8G" in search
    assert "MemorySwapMax=1G" in search
    assert "ConditionPathExists=" not in search
    assert "9915" not in search
    assert "--vfs-cache-max-size=1G" in mount
    assert "--vfs-read-chunk-streams=2" in mount
    assert "v5-publish-fullraw-fts-remote" in mount
    assert "9915" not in mount


def test_portfolio_route_installer_switches_without_touching_shared_unit(tmp_path: Path) -> None:
    deploy_dir = Path(__file__).resolve().parents[1] / "deploy"
    unit_dir = tmp_path / "systemd"
    config_dir = tmp_path / "config"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    systemctl_log = tmp_path / "systemctl.log"
    systemctl_state = tmp_path / "systemctl.state"
    (fake_bin / "systemctl").write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$SYSTEMCTL_LOG\"\n"
        "case \"$1\" in\n"
        "  enable|restart) printf '%s\\n' enabled > \"$SYSTEMCTL_STATE\" ;;\n"
        "  disable) printf '%s\\n' disabled > \"$SYSTEMCTL_STATE\" ;;\n"
        "  is-active|is-enabled) ! grep -q '^disabled$' \"$SYSTEMCTL_STATE\" ;;\n"
        "esac\n",
    )
    (fake_bin / "mountpoint").write_text("#!/bin/sh\nexit 0\n")
    (fake_bin / "flock").write_text("#!/bin/sh\nexit 0\n")
    (fake_bin / "install").write_text(
        "#!/bin/sh\n"
        "if [ -n \"${FAIL_INSTALL_MATCH:-}\" ]; then\n"
        "  case \"$*\" in *\"$FAIL_INSTALL_MATCH\"*) exit 1 ;; esac\n"
        "fi\n"
        "exec \"$REAL_INSTALL\" \"$@\"\n",
    )
    (fake_bin / "curl").write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *http_code*) printf '%s' 400 ;;\n"
        "  *) printf '%s\\n' "
        "'{\"ok\":true,\"backend\":\"researka-fullraw-indexed-fts5\",\"shard_dir\":\"/var/lib/v5-memo/v5-publish-fullraw-fts-remote\",\"shard_receipt\":{\"shards_total\":1525},\"coverage_requirements\":{\"min_shards_searched\":1525,\"require_complete_search\":1,\"sweep_require_complete\":1},\"async_sweep\":{\"max_inflight\":1,\"workers\":1}}' ;;\n"
        "esac\n",
    )
    (fake_bin / "sleep").write_text("#!/bin/sh\nexit 0\n")
    for command in ("systemctl", "mountpoint", "flock", "install", "curl", "sleep"):
        (fake_bin / command).chmod(0o755)
    sentinel = unit_dir / "researka-fullraw-search.service"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("platform-owned\n")
    platform_break_glass = tmp_path / "allow-legacy-sidecars"
    platform_break_glass.touch()
    competing_name = "zzzzzzzzzzzzzzzz-researka-shared-disabled.conf"
    competing_overrides = []
    for unit in (
        "v5-memo-publish-fullraw-fts-mount.service",
        "v5-memo-publish-fullraw-search.service",
    ):
        competing_override = unit_dir / f"{unit}.d" / competing_name
        competing_override.parent.mkdir(parents=True)
        competing_override.write_text(
            "[Unit]\nConditionPathExists=/run/researka-fullraw-allow-legacy-sidecars\n",
        )
        competing_overrides.append(competing_override)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SYSTEMCTL_LOG": str(systemctl_log),
        "SYSTEMCTL_STATE": str(systemctl_state),
        "REAL_INSTALL": shutil.which("install") or "/usr/bin/install",
        "SYSTEMD_UNIT_DIR": str(unit_dir),
        "V5_MEMO_CONFIG_DIR": str(config_dir),
        "V5_MEMO_PORTFOLIO_LOCK_PATH": str(tmp_path / "portfolio.lock"),
        "V5_MEMO_PLATFORM_BREAK_GLASS_PATH": str(platform_break_glass),
        "V5_MEMO_PUBLISH_MOUNT_PATH": str(tmp_path / "mount"),
        "V5_MEMO_PUBLISH_CATALOG_PATH": str(tmp_path / "catalog.json"),
        "V5_MEMO_PORTFOLIO_SEARCH_ROUTE": "dedicated",
        "V5_MEMO_ALLOW_DEDICATED_FULLRAW": "1",
    }
    config_dir.mkdir()
    (config_dir / "env").write_text("RESEARKA_FULLRAW_INDEX_TOKEN=test-token\n")
    dedicated_marker = config_dir / "allow-dedicated-fullraw"
    dedicated_marker.touch()
    catalog_shard = tmp_path / "mount" / "batch_00000" / "fullraw_shard_0000.sqlite"
    catalog_shard.parent.mkdir(parents=True)
    catalog_shard.touch()
    (tmp_path / "catalog.json").write_text(
        json.dumps({
            "entries": [
                {"path": "/old/root/batch_00000/fullraw_shard_0000.sqlite"}
                for _ in range(1525)
            ],
        }),
    )

    subprocess.run(
        ["/bin/sh", str(deploy_dir / "install-v5-portfolio-isolation.sh")],
        check=True,
        env=env,
    )

    route = unit_dir / "v5-memo-portfolio-prepare.service.d" / "zzzzz-v5-portfolio-fullraw-route.conf"
    search_override = (
        unit_dir
        / "v5-memo-publish-fullraw-search.service.d"
        / "zzzzzzzz-v5-publish-fullraw-owned.conf"
    )
    mount_override = (
        unit_dir
        / "v5-memo-publish-fullraw-fts-mount.service.d"
        / "zzzzzzzz-v5-publish-fullraw-owned.conf"
    )
    assert search_override.name < competing_name
    assert mount_override.name < competing_name
    assert sorted(path.name for path in search_override.parent.glob("*.conf"))[-1] == (
        competing_name
    )
    assert sorted(path.name for path in mount_override.parent.glob("*.conf"))[-1] == (
        competing_name
    )
    assert "V5_MEMO_PORTFOLIO_FULL_RAW_CORPUS_SEARCH_URL" in route.read_text()
    assert "ConditionPathExists=" in search_override.read_text()
    assert "ConditionPathExists=" in mount_override.read_text()
    assert all("allow-legacy-sidecars" in path.read_text() for path in competing_overrides)
    assert "enable v5-memo-publish-fullraw-fts-mount.service" in systemctl_log.read_text()
    assert "restart v5-memo-publish-fullraw-search.service" in systemctl_log.read_text()
    assert "stop v5-memo-portfolio-prepare.timer" in systemctl_log.read_text()
    assert "start v5-memo-portfolio-prepare.timer" in systemctl_log.read_text()
    assert "stop v5-memo-portfolio-prepare.service" in systemctl_log.read_text()
    assert sentinel.read_text() == "platform-owned\n"

    systemctl_log.write_text("")
    platform_break_glass.unlink()
    env.pop("V5_MEMO_PORTFOLIO_SEARCH_ROUTE")
    env.pop("V5_MEMO_ALLOW_DEDICATED_FULLRAW")
    platform_rejected = subprocess.run(
        ["/bin/sh", str(deploy_dir / "install-v5-portfolio-isolation.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert platform_rejected.returncode == 2
    assert "platform break-glass marker" in platform_rejected.stderr
    assert systemctl_log.read_text() == ""

    platform_break_glass.touch()
    subprocess.run(
        ["/bin/sh", str(deploy_dir / "install-v5-portfolio-isolation.sh")],
        check=True,
        env=env,
    )
    durable_default_log = systemctl_log.read_text()
    assert "enable v5-memo-publish-fullraw-fts-mount.service" in durable_default_log
    assert "restart v5-memo-publish-fullraw-search.service" in durable_default_log
    env["V5_MEMO_PORTFOLIO_SEARCH_ROUTE"] = "dedicated"
    env["V5_MEMO_ALLOW_DEDICATED_FULLRAW"] = "1"

    (fake_bin / "curl").write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' "
        "'{\"ok\":true,\"corpus_complete\":true,\"async_sweep\":{\"max_inflight\":1,\"workers\":1}}'\n",
    )
    env["FAIL_INSTALL_MATCH"] = (
        "v5-memo-portfolio-catchup.service.d/zzzzz-v5-portfolio-fullraw-route.conf"
    )
    failed_rollback = subprocess.run(
        ["/bin/sh", str(deploy_dir / "install-v5-portfolio-isolation.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    catchup_route = (
        unit_dir
        / "v5-memo-portfolio-catchup.service.d"
        / "zzzzz-v5-portfolio-fullraw-route.conf"
    )
    assert failed_rollback.returncode != 0
    assert "rollback failed" in failed_rollback.stderr
    assert "V5_MEMO_PORTFOLIO_FULL_RAW_CORPUS_SEARCH_URL" in catchup_route.read_text()
    env.pop("FAIL_INSTALL_MATCH")

    env["V5_MEMO_PORTFOLIO_SEARCH_ROUTE"] = "shared"
    subprocess.run(
        ["/bin/sh", str(deploy_dir / "install-v5-portfolio-isolation.sh")],
        check=True,
        env=env,
    )

    assert "UnsetEnvironment=V5_MEMO_PORTFOLIO_FULL_RAW_CORPUS_SEARCH_URL" in route.read_text()
    assert "disable --now v5-memo-publish-fullraw-search.service" in systemctl_log.read_text()
    assert "ConditionPathExists=/run/researka-fullraw-allow-legacy-sidecars" in (
        search_override.read_text()
    )
    assert "ConditionPathExists=/run/researka-fullraw-allow-legacy-sidecars" in (
        mount_override.read_text()
    )
    assert sentinel.read_text() == "platform-owned\n"

    systemctl_log.write_text("")
    dedicated_marker.unlink()
    env["V5_MEMO_PORTFOLIO_SEARCH_ROUTE"] = "dedicated"
    env["V5_MEMO_ALLOW_DEDICATED_FULLRAW"] = "1"
    marker_rejected = subprocess.run(
        ["/bin/sh", str(deploy_dir / "install-v5-portfolio-isolation.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert marker_rejected.returncode == 2
    assert "allow-dedicated-fullraw" in marker_rejected.stderr
    assert systemctl_log.read_text() == ""

    assert "UnsetEnvironment=V5_MEMO_PORTFOLIO_FULL_RAW_CORPUS_SEARCH_URL" in route.read_text()
    assert systemctl_state.read_text() == "disabled\n"
    assert sentinel.read_text() == "platform-owned\n"

    systemctl_log.write_text("")
    env.pop("V5_MEMO_PORTFOLIO_SEARCH_ROUTE")
    env.pop("V5_MEMO_ALLOW_DEDICATED_FULLRAW")
    subprocess.run(
        ["/bin/sh", str(deploy_dir / "install-v5-portfolio-isolation.sh")],
        check=True,
        env=env,
    )
    default_log = systemctl_log.read_text()
    assert "disable --now v5-memo-publish-fullraw-search.service" in default_log
    assert "enable v5-memo-publish-fullraw-fts-mount.service" not in default_log

    systemctl_log.write_text("")
    env["V5_MEMO_PORTFOLIO_SEARCH_ROUTE"] = "dedicated"
    rejected = subprocess.run(
        ["/bin/sh", str(deploy_dir / "install-v5-portfolio-isolation.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert rejected.returncode == 2
    assert "V5_MEMO_ALLOW_DEDICATED_FULLRAW=1" in rejected.stderr
    assert systemctl_log.read_text() == ""


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
