"""Local HTTP service for cold raw-corpus search.

This is a low-footprint bridge over the rclone raw archive. It is not a fast
Tantivy/DuckDB index; it streams compressed source files and returns the first
matching receipt candidates.
"""
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_WORD = re.compile(r"[A-Za-z0-9+]+")
_STOP = {"and", "or", "the", "with", "for", "from", "into", "this", "that"}

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


@dataclass(frozen=True, slots=True)
class SearchResult:
    hits: list[dict[str, object]]
    files_scanned: int
    files_total: int
    complete: bool
    elapsed_seconds: float


class RawCorpusScanner:
    def __init__(self, files: list[RawFile], *, rclone_bin: str = "rclone") -> None:
        self._files = files
        self._rclone_bin = rclone_bin

    def search(self, query: str, *, limit: int, timeout_seconds: float) -> SearchResult:
        started = time.monotonic()
        terms = _query_terms(query)
        hits: list[dict[str, object]] = []
        seen: set[str] = set()
        files_scanned = 0
        complete = True
        deadline = started + max(1.0, timeout_seconds)
        for raw_file in self._files:
            if time.monotonic() >= deadline:
                complete = False
                break
            files_scanned += 1
            for hit in self._scan_file(raw_file, terms, deadline=deadline):
                key = _dedupe_key(hit)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(hit)
                if len(hits) >= limit:
                    complete = False
                    return SearchResult(
                        hits=hits,
                        files_scanned=files_scanned,
                        files_total=len(self._files),
                        complete=complete,
                        elapsed_seconds=round(time.monotonic() - started, 3),
                    )
        return SearchResult(
            hits=hits,
            files_scanned=files_scanned,
            files_total=len(self._files),
            complete=complete,
            elapsed_seconds=round(time.monotonic() - started, 3),
        )

    def _scan_file(self, raw_file: RawFile, terms: tuple[str, ...], *, deadline: float) -> list[dict[str, object]]:
        if raw_file.format == "pubmed_xml":
            return list(_scan_pubmed_xml(raw_file, terms, deadline=deadline, rclone_bin=self._rclone_bin))
        return list(_scan_jsonl(raw_file, terms, deadline=deadline, rclone_bin=self._rclone_bin))


def iter_raw_file_hits(
    raw_file: RawFile,
    *,
    rclone_bin: str = "rclone",
) -> Iterator[dict[str, object]]:
    """Yield normalized records from one raw corpus file without search filtering."""
    if raw_file.format == "pubmed_xml":
        yield from _iter_pubmed_xml(raw_file, terms=(), deadline=None, rclone_bin=rclone_bin)
        return
    yield from _iter_jsonl(raw_file, terms=(), deadline=None, rclone_bin=rclone_bin)


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


def run_server() -> None:
    host = os.environ.get("V5_MEMO_FULL_RAW_HOST", "127.0.0.1")
    port = int(os.environ.get("V5_MEMO_FULL_RAW_PORT", "9901"))
    manifest_path = Path(
        os.environ.get("V5_MEMO_FULL_RAW_MANIFEST", "/opt/v5-memo/state/fullraw_manifest.json")
    )
    rclone_bin = os.environ.get("V5_MEMO_FULL_RAW_RCLONE", "rclone")
    refresh = os.environ.get("V5_MEMO_FULL_RAW_REFRESH_MANIFEST", "").casefold() in {"1", "true", "yes"}
    files = load_or_build_manifest(manifest_path, refresh=refresh, rclone_bin=rclone_bin)
    scanner = RawCorpusScanner(files, rclone_bin=rclone_bin)
    token = os.environ.get("V5_MEMO_FULL_RAW_TOKEN", "").strip()
    default_timeout = float(os.environ.get("V5_MEMO_FULL_RAW_TIMEOUT_SECONDS", "45"))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404)
                return
            _write_json(self, 200, {
                "ok": True,
                "backend": "v5-fullraw-cold-scan",
                "files_total": len(files),
                "manifest": str(manifest_path),
            })

        def do_POST(self) -> None:
            if self.path != "/search":
                self.send_error(404)
                return
            if token and self.headers.get("Authorization", "") != f"Bearer {token}":
                _write_json(self, 401, {"error": "unauthorized"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                query = str(payload.get("query", "")).strip()
                limit = max(1, min(int(payload.get("limit") or payload.get("top_k") or 25), 200))
                timeout_seconds = float(payload.get("timeout_seconds") or default_timeout)
            except (TypeError, ValueError, json.JSONDecodeError):
                _write_json(self, 400, {"error": "bad request"})
                return
            if not query:
                _write_json(self, 400, {"error": "query is required"})
                return
            result = scanner.search(query, limit=limit, timeout_seconds=timeout_seconds)
            _write_json(self, 200, {
                "meta": {
                    "count": len(result.hits),
                    "files_scanned": result.files_scanned,
                    "files_total": result.files_total,
                    "complete": result.complete,
                    "elapsed_seconds": result.elapsed_seconds,
                    "backend": "v5-fullraw-cold-scan",
                },
                "results": result.hits,
            })

        def log_message(self, fmt: str, *args: object) -> None:
            return

    ThreadingHTTPServer((host, port), Handler).serve_forever()


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


def _scan_jsonl(
    raw_file: RawFile,
    terms: tuple[str, ...],
    *,
    deadline: float,
    rclone_bin: str,
) -> list[dict[str, object]]:
    return list(_iter_jsonl(raw_file, terms=terms, deadline=deadline, rclone_bin=rclone_bin))


def _scan_pubmed_xml(
    raw_file: RawFile,
    terms: tuple[str, ...],
    *,
    deadline: float,
    rclone_bin: str,
) -> list[dict[str, object]]:
    return list(_iter_pubmed_xml(raw_file, terms=terms, deadline=deadline, rclone_bin=rclone_bin))


def _iter_jsonl(
    raw_file: RawFile,
    *,
    terms: tuple[str, ...],
    deadline: float | None,
    rclone_bin: str,
) -> Iterator[dict[str, object]]:
    with _open_gzip_stream(raw_file.remote, rclone_bin=rclone_bin) as stream:
        for raw_line in stream:
            if deadline is not None and time.monotonic() >= deadline:
                break
            line = raw_line.decode("utf-8", errors="replace")
            if terms and not _contains_terms(line.casefold(), terms):
                continue
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
    terms: tuple[str, ...],
    deadline: float | None,
    rclone_bin: str,
) -> Iterator[dict[str, object]]:
    with _open_gzip_stream(raw_file.remote, rclone_bin=rclone_bin) as stream:
        try:
            for _event, elem in ET.iterparse(stream, events=("end",)):
                if deadline is not None and time.monotonic() >= deadline:
                    break
                if _strip_ns(elem.tag) != "PubmedArticle":
                    continue
                hit = _normalize_pubmed_hit(elem)
                elem.clear()
                if not hit:
                    continue
                folded = f"{hit.get('title', '')} {hit.get('abstract', '')}".casefold()
                if not terms or _contains_terms(folded, terms):
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
    external = item.get("externalids") or item.get("externalIds") or item.get("ids")
    external_ids = external if isinstance(external, dict) else {}
    title = _clean(item.get("title") or item.get("display_name"))
    if not title:
        return None
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


def _query_terms(query: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _WORD.findall(query.casefold())
        if len(token) > 1 and token not in _STOP
    )


def _contains_terms(text: str, terms: tuple[str, ...]) -> bool:
    return all(
        re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None
        for term in terms
    )


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


def _write_json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, object]) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


if __name__ == "__main__":
    run_server()
