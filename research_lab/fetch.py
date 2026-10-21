"""Bounded development-only downloads and strictly offline immutable cache reads.

Only ``fetch_manifest`` performs network requests. Source objects and receipts are
published without replacement; a changed source requires a new catalog version
or a separate cache, never a silent refresh during an experiment.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time
import urllib.request

from research_lab.catalog import official_url, select_records, validate_record
from research_lab.integrity import file_hash, json_hash, read_json, require

CHUNK_BYTES = 64 * 1024
MAX_REDIRECTS = 3


def _safe_path(value):
    path = Path(value).expanduser().absolute()
    require(not any(part.is_symlink() for part in (path, *path.parents)), "Symlink cache paths are unsupported")
    return path


def _paths(record, cache):
    validate_record(record)
    cache = _safe_path(cache)
    receipt = _safe_path(cache / "receipts" / record["dataset_id"] / (record["version"] + ".json"))
    return cache, receipt


def _csv_columns(raw, record):
    require(raw and b"\0" not in raw, "Empty/binary source is not CSV")
    require(not raw.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08", b"\x1f\x8b", b"BZh", b"\xfd7zXZ", b"7z\xbc\xaf\x27\x1c"))
            and raw[257:262] != b"ustar", "Archive downloads are rejected, including ZIP/tar/compressed payloads")
    try:
        text = raw.decode(record["adapter"]["encoding"])
        prefix = text.lstrip("\ufeff \r\n\t").lower()
        require(not prefix.startswith(("<!doctype", "<html", "<?xml", "<head", "<body", "{")),
                "HTML/login/metadata payload is not CSV")
        rows = csv.reader(io.StringIO(text), delimiter=",", strict=True)
        columns = next(rows)
        require(1 < len(columns) <= 1024 and all(columns) and len(set(columns)) == len(columns), "Invalid CSV header")
        require(record["target"] in columns, "Downloaded CSV lacks the declared target")
        count = 0
        for row in rows:
            require(len(row) == len(columns), "Ragged CSV row")
            count += 1
        require(count > 0, "CSV has no data rows")
        return columns
    except (UnicodeError, csv.Error, StopIteration) as error:
        raise ValueError("Source is not the declared UTF-8 CSV") from error


class _OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        official_url(newurl, download=True)
        count = getattr(req, "lab_redirects", 0) + 1
        require(count <= MAX_REDIRECTS, "Too many source redirects")
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        require(result is not None, "Unsupported source redirect")
        result.lab_redirects = count
        return result


class _DeadlineHTTPS(urllib.request.HTTPSHandler):
    def __init__(self, deadline):
        super().__init__()
        self.deadline = deadline

    def https_open(self, request):
        remaining = self.deadline - time.monotonic()
        require(remaining > 0, "Source download exceeded total timeout")
        request.timeout = min(request.timeout, remaining)
        return super().https_open(request)


def _open_url(url, timeout):
    # Development downloads inherit the platform's normal proxy transport.
    # Solver environments are sanitized separately; proxy settings are never
    # copied into receipts or logs. Redirects and TLS verification remain checked.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(), _OfficialRedirect(),
                                         _DeadlineHTTPS(time.monotonic() + timeout))
    request = urllib.request.Request(url, headers={"User-Agent": "alfa-research-lab/1", "Accept": "text/csv", "Accept-Encoding": "identity"})
    return opener.open(request, timeout=timeout)


def _download(record, max_bytes, timeout):
    official_url(record["download_url"], download=True)
    deadline = time.monotonic() + timeout
    with _open_url(record["download_url"], timeout) as response:
        require(response.status == 200, "Source did not return HTTP 200")
        final_url = official_url(response.geturl(), download=True)
        encoding = response.headers.get("Content-Encoding", "identity").lower()
        require(encoding in ("", "identity"), "Compressed HTTP payloads are unsupported")
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        # Official UCI static CSV responses can omit Content-Type. An absent
        # hint is accepted only because the full payload must still pass strict
        # UTF-8 CSV/target/header validation and archive/HTML rejection below.
        require(content_type in ("", "text/csv", "application/csv", "text/plain", "application/octet-stream"),
                "Source Content-Type is not CSV/plain bytes")
        length = response.headers.get("Content-Length")
        if length is not None:
            require(length.isdigit() and 0 < int(length) <= max_bytes, "Source Content-Length exceeds byte bound")
        disposition = response.headers.get("Content-Disposition", "")
        filename = re.search(r"filename\*?\s*=\s*[\"']?([^\"';]+)", disposition, re.I)
        if filename:
            name = filename.group(1)
            require("/" not in name and "\\" not in name and ".." not in name and name.lower().endswith(".csv"),
                    "Unsafe or archived Content-Disposition filename")
        blocks, size = [], 0
        while True:
            remaining = deadline - time.monotonic()
            require(remaining > 0, "Source download exceeded total timeout")
            # urllib's HTTPResponse exposes the connected TLS socket here. Apply
            # the remaining total deadline to each blocking read, rather than
            # granting a new per-read timeout. In-memory fixtures have no socket.
            socket = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
            if socket is not None:
                socket.settimeout(remaining)
            block = response.read(min(CHUNK_BYTES, max_bytes - size + 1))
            require(time.monotonic() <= deadline, "Source download exceeded total timeout")
            if not block:
                break
            size += len(block)
            require(size <= max_bytes, "Source download exceeded byte bound")
            blocks.append(block)
        require(length is None or size == int(length), "Source Content-Length mismatch")
    raw = b"".join(blocks)
    columns = _csv_columns(raw, record)
    return raw, final_url, columns


def _publish_immutable(path, raw):
    path = _safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _safe_path(path.parent)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".fetch-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError:
            require(path.is_file() and not path.is_symlink() and path.read_bytes() == raw,
                    "Immutable cache destination already contains different bytes")
    finally:
        temporary.unlink(missing_ok=True)


def cached_member(record, cache):
    """Return verified raw CSV + receipt. Missing/corrupt cache is never repaired."""
    cache, receipt_path = _paths(record, cache)
    require(record["status"] == "ready", "Dataset is ineligible for cached execution")
    receipt = read_json(receipt_path)
    require(receipt.get("schema_version") == 1 and receipt.get("dataset_id") == record["dataset_id"]
            and receipt.get("version") == record["version"] and receipt.get("record_sha256") == json_hash(record),
            "Cache receipt does not match the catalog record")
    sha = receipt.get("sha256")
    require(isinstance(sha, str) and re.fullmatch(r"[a-f0-9]{64}", sha), "Invalid receipt source hash")
    require(receipt.get("source_sha256") == sha and receipt.get("member_sha256") == sha
            and receipt.get("object_path") == "objects/" + sha and receipt.get("format") == "csv",
            "Unsafe or inconsistent cache object receipt")
    require(receipt.get("download_url") == record["download_url"], "Cache source URL drift")
    official_url(receipt.get("final_url"), download=True)
    retrieved = receipt.get("retrieved_at_utc")
    require(isinstance(retrieved, str), "Missing retrieval timestamp")
    try:
        timestamp = datetime.fromisoformat(retrieved)
        require(timestamp.utcoffset() is not None and timestamp.utcoffset().total_seconds() == 0,
                "Retrieval timestamp must be UTC")
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid retrieval timestamp") from error
    source_bytes = receipt.get("source_bytes")
    require(type(source_bytes) is int and 0 < source_bytes <= 64 * 1024 * 1024, "Invalid cached source byte count")
    obj = _safe_path(cache / "objects" / sha)
    require(obj.is_file() and obj.stat().st_size == source_bytes, "Missing or truncated cached source")
    require(file_hash(obj) == sha, "Cached source hash mismatch; cache is corrupt")
    for key, actual in (("source_sha256", sha), ("member_sha256", sha), ("source_bytes", source_bytes)):
        require(record[key] is None or record[key] == actual, "Source disagrees with pinned catalog evidence: " + key)
    require(record.get("publisher_sha256") is None or record["publisher_sha256"] == sha,
            "Source disagrees with publisher checksum")
    columns = _csv_columns(obj.read_bytes(), record)
    require(receipt.get("columns") == columns, "Cached CSV header receipt mismatch")
    return obj, receipt


def _fetch_one(record, cache, max_bytes, timeout):
    cache, receipt_path = _paths(record, cache)
    if receipt_path.exists():
        path, receipt = cached_member(record, cache)
        return {"dataset_id": record["dataset_id"], "status": "cached", "sha256": receipt["sha256"],
                "source_bytes": receipt["source_bytes"], "receipt": str(receipt_path), "object": str(path)}
    raw, final_url, columns = _download(record, max_bytes, timeout)
    sha = hashlib.sha256(raw).hexdigest()
    for key, actual in (("source_sha256", sha), ("member_sha256", sha), ("source_bytes", len(raw))):
        require(record[key] is None or record[key] == actual, "Downloaded source disagrees with pinned " + key)
    require(record.get("publisher_sha256") is None or record["publisher_sha256"] == sha,
            "Downloaded source disagrees with publisher checksum")
    receipt = {"schema_version": 1, "dataset_id": record["dataset_id"], "version": record["version"],
               "record_sha256": json_hash(record), "download_url": record["download_url"], "final_url": final_url,
               "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "sha256": sha, "source_sha256": sha,
               "member_sha256": sha, "source_bytes": len(raw), "format": "csv", "columns": columns,
               "object_path": "objects/" + sha, "hash_origin": "computed_from_downloaded_bytes_not_publisher_signature"}
    _publish_immutable(cache / "objects" / sha, raw)
    _publish_immutable(receipt_path, (json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
    path, receipt = cached_member(record, cache)
    return {"dataset_id": record["dataset_id"], "status": "fetched", "sha256": sha, "source_bytes": len(raw),
            "receipt": str(receipt_path), "object": str(path)}


def fetch_manifest(manifest, cache, profile):
    """Explicit development download operation; no model imports or package installs."""
    records = select_records(manifest, profile)
    results = []
    for record in records:
        if record["status"] != "ready":
            results.append({"dataset_id": record["dataset_id"], "status": "skipped", "reason": record["reason"]})
            continue
        # A pre-existing receipt is verified outside the network-error summary:
        # corrupted evidence blocks the operation and is never overwritten.
        _, receipt_path = _paths(record, cache)
        if receipt_path.exists():
            cached_member(record, cache)
        try:
            results.append(_fetch_one(record, cache, manifest["max_download_bytes"], manifest["download_timeout_seconds"]))
        except (OSError, ValueError) as error:
            results.append({"dataset_id": record["dataset_id"], "status": "unavailable", "reason": str(error)})
    return {"schema_version": 1, "profile": profile, "records": results,
            "counts": {status: sum(row["status"] == status for row in results)
                       for status in ("fetched", "cached", "skipped", "unavailable")},
            "source_bytes": sum(row.get("source_bytes", 0) for row in results)}
