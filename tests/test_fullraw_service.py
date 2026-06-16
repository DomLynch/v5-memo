from __future__ import annotations

import gzip
import json
import textwrap
from pathlib import Path

from v5_memo.fullraw_service import RawCorpusScanner, RawFile


def _write_gzip(path: Path, text: str) -> None:
    with gzip.open(path, "wb") as fh:
        fh.write(text.encode("utf-8"))


def test_raw_scanner_reads_local_jsonl_fixture(tmp_path: Path) -> None:
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

    result = RawCorpusScanner([
        RawFile(source="openalex", format="openalex_jsonl", remote=f"file://{source}")
    ]).search("NAD mitochondrial", limit=5, timeout_seconds=5)

    assert result.complete is True
    assert result.files_scanned == 1
    assert result.hits[0]["doi"] == "10.raw/one"
    assert result.hits[0]["source"] == "openalex"
    assert result.hits[0]["abstract"] == "NAD mitochondrial repair"


def test_raw_scanner_reads_local_pubmed_xml_fixture(tmp_path: Path) -> None:
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

    result = RawCorpusScanner([
        RawFile(source="pubmed", format="pubmed_xml", remote=f"file://{source}")
    ]).search("NAD mitochondrial", limit=5, timeout_seconds=5)

    assert result.complete is True
    assert result.hits[0]["pmid"] == "123"
    assert result.hits[0]["doi"] == "10.raw/pubmed"
    assert result.hits[0]["source"] == "pubmed"
