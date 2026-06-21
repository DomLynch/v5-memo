"""Raw corpus manifest and record-normalization helpers for the fullraw index."""
from __future__ import annotations

import gzip
import json
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any

DEFAULT_SOURCE_SPECS = (
    ("openalex", "openalex_jsonl", "sb:researka-database/raw/openalex/works"),
    ("semantic_scholar", "semantic_scholar_jsonl", "sb:researka-database/raw/semantic_scholar/papers"),
    (
        "semantic_scholar_abstracts",
        "semantic_scholar_jsonl",
        "sb:researka-database/raw/semantic_scholar/abstracts",
    ),
    ("pubmed", "pubmed_xml", "sb:researka-database/raw/pubmed"),
    ("biorxiv", "biorxiv_jsonl", "sb:researka-database/raw/biorxiv"),
)


@dataclass(frozen=True, slots=True)
class RawFile:
    source: str
    format: str
    remote: str

def iter_raw_file_hits(
    raw_file: RawFile,
    *,
    rclone_bin: str = "rclone",
) -> Iterator[dict[str, object]]:
    """Yield normalized records from one raw corpus file without search filtering."""
    if raw_file.format == "pubmed_xml":
        yield from _iter_pubmed_xml(raw_file, rclone_bin=rclone_bin)
        return
    yield from _iter_jsonl(raw_file, rclone_bin=rclone_bin)


def load_or_build_manifest(path: Path, *, refresh: bool = False, rclone_bin: str = "rclone") -> list[RawFile]:
    if path.exists() and not refresh:
        return [
            RawFile(source=item["source"], format=item["format"], remote=item["remote"])
            for item in json.loads(path.read_text()).get("files", [])
        ]
    files = build_manifest(rclone_bin=rclone_bin)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": [
            {"source": file.source, "format": file.format, "remote": file.remote}
            for file in files
        ],
    }, indent=2) + "\n")
    return files


def build_manifest(*, rclone_bin: str = "rclone") -> list[RawFile]:
    files: list[RawFile] = []
    for source, file_format, root in _source_specs():
        try:
            listed = subprocess.run(
                [rclone_bin, "lsf", "--files-only", "-R", root, "--include", "*.gz"],
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if listed.returncode != 0:
            continue
        for relpath in listed.stdout.splitlines():
            relpath = relpath.strip()
            if relpath:
                files.append(RawFile(source=source, format=file_format, remote=f"{root.rstrip('/')}/{relpath}"))
    return files


def _source_specs() -> tuple[tuple[str, str, str], ...]:
    configured = os.environ.get("V5_MEMO_FULL_RAW_SOURCES", "").strip()
    if not configured:
        return DEFAULT_SOURCE_SPECS
    specs: list[tuple[str, str, str]] = []
    for chunk in configured.split(";"):
        parts = [part.strip() for part in chunk.split("|", 2)]
        if len(parts) == 3 and all(parts):
            specs.append((parts[0], parts[1], parts[2]))
    return tuple(specs) or DEFAULT_SOURCE_SPECS


def _iter_jsonl(
    raw_file: RawFile,
    *,
    rclone_bin: str,
) -> Iterator[dict[str, object]]:
    with _open_gzip_stream(raw_file.remote, rclone_bin=rclone_bin) as stream:
        for raw_line in stream:
            line = raw_line.decode("utf-8", errors="replace")
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            hit = _normalize_json_hit(item, raw_file.source)
            if hit:
                yield hit


def _iter_pubmed_xml(
    raw_file: RawFile,
    *,
    rclone_bin: str,
) -> Iterator[dict[str, object]]:
    with _open_gzip_stream(raw_file.remote, rclone_bin=rclone_bin) as stream:
        try:
            for _event, elem in ET.iterparse(stream, events=("end",)):
                if _strip_ns(elem.tag) != "PubmedArticle":
                    continue
                hit = _normalize_pubmed_hit(elem)
                elem.clear()
                if hit:
                    yield hit
        except ET.ParseError:
            return


class _GzipStream:
    def __init__(self, remote: str, *, rclone_bin: str) -> None:
        self._remote = remote
        self._rclone_bin = rclone_bin
        self._proc: subprocess.Popen[bytes] | None = None
        self._stream: gzip.GzipFile | None = None

    def __enter__(self) -> gzip.GzipFile:
        if self._remote.startswith("file://"):
            self._stream = gzip.open(self._remote.removeprefix("file://"), "rb")
            return self._stream
        self._proc = subprocess.Popen(
            [self._rclone_bin, "cat", self._remote],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if self._proc.stdout is None:
            raise OSError("rclone stdout unavailable")
        self._stream = gzip.GzipFile(fileobj=self._proc.stdout)
        return self._stream

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._stream is not None:
            self._stream.close()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def _open_gzip_stream(remote: str, *, rclone_bin: str) -> _GzipStream:
    return _GzipStream(remote, rclone_bin=rclone_bin)


def _normalize_json_hit(item: Any, source: str) -> dict[str, object] | None:
    if not isinstance(item, dict):
        return None
    open_access = item.get("openaccessinfo") or item.get("openAccessInfo")
    open_access_ids = (
        open_access.get("externalids") or open_access.get("externalIds")
        if isinstance(open_access, dict)
        else None
    )
    external = item.get("externalids") or item.get("externalIds") or item.get("ids") or open_access_ids
    external_ids = external if isinstance(external, dict) else {}
    title = _clean(item.get("title") or item.get("display_name"))
    doi = _normalize_doi(item.get("doi") or external_ids.get("DOI") or external_ids.get("doi"))
    pmid = _clean(external_ids.get("PubMed") or item.get("pmid"))
    pmcid = _clean(external_ids.get("PubMedCentral") or item.get("pmcid"))
    s2_id = _clean(item.get("corpusid") or item.get("corpusId") or external_ids.get("CorpusId"))
    openalex_id = _clean(item.get("id") if str(item.get("id", "")).startswith("https://openalex.org/") else "")
    abstract = (
        _clean(item.get("abstract") or item.get("abstract_text"))
        or _abstract_from_inverted_index(item.get("abstract_inverted_index"))
        or _clean((item.get("tldr") or {}).get("text") if isinstance(item.get("tldr"), dict) else "")
    )
    if not title and not (source == "semantic_scholar_abstracts" and abstract and (doi or pmid or pmcid or s2_id)):
        return None
    venue = _clean(
        item.get("venue")
        or item.get("journal")
        or ((item.get("journal") or {}).get("name") if isinstance(item.get("journal"), dict) else "")
    )
    return {
        "title": title,
        "abstract": abstract,
        "doi": doi,
        "pmid": pmid,
        "pmcid": pmcid,
        "openalex_id": openalex_id,
        "semantic_scholar_id": s2_id,
        "year": _int_or_none(item.get("year") or item.get("publication_year")),
        "journal": venue,
        "source": source,
        "url": _clean(item.get("url")) or (f"https://doi.org/{doi}" if doi else openalex_id),
        "cited_by_count": _int_or_none(item.get("citationcount") or item.get("cited_by_count")),
        "score": 1.0,
    }


def _normalize_pubmed_hit(elem: ET.Element) -> dict[str, object] | None:
    title = _clean(_find_text(elem, ".//ArticleTitle"))
    if not title:
        return None
    doi = ""
    for node in elem.findall(".//ELocationID"):
        if node.attrib.get("EIdType", "").casefold() == "doi":
            doi = _normalize_doi(node.text)
            break
    return {
        "title": title,
        "abstract": _clean(" ".join(node.text or "" for node in elem.findall(".//AbstractText"))),
        "doi": doi,
        "pmid": _clean(_find_text(elem, ".//PMID")),
        "pmcid": "",
        "year": _int_or_none(_find_text(elem, ".//PubDate/Year") or _find_text(elem, ".//DateCompleted/Year")),
        "journal": _clean(_find_text(elem, ".//Journal/Title")),
        "source": "pubmed",
        "url": f"https://doi.org/{doi}" if doi else "",
        "score": 1.0,
    }


def _find_text(elem: ET.Element, path: str) -> str:
    node = elem.find(path)
    return "" if node is None or node.text is None else node.text


def _clean(value: object) -> str:
    if value is None:
        return ""
    return " ".join(re.sub(r"<[^>]+>", " ", unescape(str(value))).split())


def _normalize_doi(value: object) -> str:
    doi = _clean(value)
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.I)


def _abstract_from_inverted_index(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    positioned: list[tuple[int, str]] = []
    for word, positions in value.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for position in positions:
            parsed = _int_or_none(position)
            if parsed is not None:
                positioned.append((parsed, word))
    return _clean(" ".join(word for _pos, word in sorted(positioned)))


def _int_or_none(value: object) -> int | None:
    if not isinstance(value, int | float | str | bytes | bytearray):
        return None
    try:
        return int(value) if value not in {"", None} else None
    except (TypeError, ValueError):
        return None


def _dedupe_key(hit: dict[str, object]) -> str:
    for key in ("doi", "pmid", "pmcid", "semantic_scholar_id", "openalex_id"):
        value = _clean(hit.get(key))
        if value:
            return f"{key}:{value.casefold()}"
    return f"title:{_clean(hit.get('title')).casefold()}:{hit.get('year') or ''}"


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
