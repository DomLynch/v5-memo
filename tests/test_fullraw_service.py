import gzip
import json
import textwrap
from pathlib import Path

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
        }) + "\n",
    )

    hits = list(iter_raw_file_hits(RawFile(source="openalex", format="openalex_jsonl", remote=f"file://{source}")))

    assert hits[0]["doi"] == "10.raw/one"
    assert hits[0]["source"] == "openalex"
    assert hits[0]["abstract"] == "NAD mitochondrial repair"


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
                    <Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
                  </Article>
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


def test_strict_5tb_service_keeps_secret_env_file() -> None:
    deploy_dir = Path(__file__).resolve().parents[1] / "deploy"
    config = deploy_dir / "v5-memo-fullraw-index-strict-5tb.conf"
    env_example = (deploy_dir / "v5-memo-fullraw-shards.env.example").read_text()
    env_files = [line for line in config.read_text().splitlines() if line.startswith("EnvironmentFile")]

    assert env_files[:3] == [
        "EnvironmentFile=",
        "EnvironmentFile=/etc/v5-memo/env",
        "EnvironmentFile=/etc/v5-memo/fullraw-effective.env",
    ]
    assert "TimeoutStopSec=120" in config.read_text()
    assert "TimeoutStopFailureMode=kill" in config.read_text()
    assert "KillMode=process" in config.read_text()
    assert "SendSIGKILL=yes" in config.read_text()
    assert "Environment=V5_MEMO_FULL_RAW_SEARCH_ISOLATED=0" in config.read_text()
    assert "Environment=V5_MEMO_FULL_RAW_SWEEP_PASS_SHARD_LIMIT=128" in config.read_text()
    assert "Environment=V5_MEMO_FULL_RAW_SWEEP_WORKERS=8" in config.read_text()
    assert "Environment=V5_MEMO_FULL_RAW_SWEEP_MAX_INFLIGHT=1" in config.read_text()
    assert "Environment=V5_MEMO_FULL_RAW_SWEEP_PRIORITY_BURST=0" in config.read_text()
    assert "Environment=V5_MEMO_FULL_RAW_SEARCH_PREFIX_SHARDS=128" in config.read_text()
    assert "V5_MEMO_FULL_RAW_SEARCH_ISOLATED=0" in env_example
    assert "V5_MEMO_FULL_RAW_SWEEP_PASS_SHARD_LIMIT=128" in env_example
    assert "V5_MEMO_FULL_RAW_INDEX_PATH=/mnt/HC_Volume_106011525/v5-memo/index/fullraw_index.sqlite" in env_example
    assert "V5_MEMO_FULL_RAW_SHARD_DIR=/mnt/HC_Volume_106011525/v5-memo/fullraw-fts-remote" in env_example
