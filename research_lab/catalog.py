"""Verified public source metadata. Loading a catalog never performs I/O on the network."""
from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit

from research_lab.integrity import read_json, require

OFFICIAL_HOST = "archive.ics.uci.edu"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
MAX_DOWNLOAD_SECONDS = 60
REQUIRED = {
    "dataset_id", "version", "source_record", "source_url", "download_url", "retrieved_at_utc",
    "license", "license_url", "attribution", "source_sha256", "source_bytes", "member_sha256",
    "adapter_version", "adapter_sha256", "target", "task_type", "metric", "direction", "id_rule",
    "label_rule", "positive_class", "feature_exclusions", "group_rule", "time_rule", "split_rule",
    "seeds", "row_counts", "split_hashes", "role", "prior_exposure", "status", "reason", "adapter",
}


def official_url(value, *, download=False):
    """Only allow verified UCI HTTPS endpoints, never credentials or traversal."""
    require(isinstance(value, str), "Source URL must be a string")
    parsed = urlsplit(value)
    require(parsed.scheme == "https" and parsed.hostname == OFFICIAL_HOST
            and parsed.port in (None, 443) and parsed.username is None and parsed.password is None,
            "Only official UCI HTTPS source URLs are supported")
    path = unquote(parsed.path)
    require(not parsed.query and not parsed.fragment and "\\" not in path
            and not any(part in (".", "..") for part in path.split("/")), "Unsafe source URL path")
    if download:
        require(re.fullmatch(r"/static/public/[1-9][0-9]*/data\.csv", path) is not None,
                "Only official unarchived CSV downloads are supported; ZIP/tar paths are rejected")
    return value


def validate_record(record):
    require(isinstance(record, dict) and REQUIRED <= record.keys(), "Incomplete dataset record")
    for name in ("dataset_id", "version"):
        require(isinstance(record[name], str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", record[name]),
                "Unsafe dataset identity/version")
    require(record["status"] in {"ready", "unverified", "unavailable", "ineligible"}, "Unknown dataset status")
    require(isinstance(record["reason"], str) and bool(record["reason"]), "Dataset status needs a reason")
    require(record["role"] in {"development", "confirmation", "sanity"}, "Unknown dataset role")
    require(isinstance(record["prior_exposure"], str) and record["prior_exposure"], "Prior exposure must be explicit")
    if record["status"] == "ready":
        official_url(record["source_url"])
        official_url(record["download_url"], download=True)
        require(record["license"] == "CC BY 4.0" and record["license_url"] == LICENSE_URL
                and isinstance(record["attribution"], str) and record["attribution"], "Unverified license/attribution")
    require(record["task_type"] in {"classification", "regression"}, "Unsupported task type")
    require(record["direction"] in {"minimize", "maximize"}, "Unsupported metric direction")
    require(isinstance(record["target"], str) and record["target"], "Missing target")
    require(isinstance(record["adapter"], dict) and record["adapter"].get("format") == "csv",
            "Only CSV adapters are supported; archives are rejected")
    require(record["adapter"].get("encoding") in ("utf-8", "utf-8-sig")
            and record["adapter"].get("delimiter") == ",", "Unsupported CSV encoding/delimiter")
    for key in ("source_sha256", "member_sha256", "adapter_sha256"):
        require(record[key] is None or isinstance(record[key], str)
                and re.fullmatch(r"[a-f0-9]{64}", record[key]), "Invalid checksum field: " + key)
    require(record.get("publisher_sha256") is None or isinstance(record["publisher_sha256"], str)
            and re.fullmatch(r"[a-f0-9]{64}", record["publisher_sha256"]), "Invalid publisher checksum")
    require(record["source_bytes"] is None or type(record["source_bytes"]) is int
            and 0 < record["source_bytes"] <= MAX_DOWNLOAD_BYTES, "Invalid source byte count")
    require(isinstance(record["seeds"], list) and record["seeds"]
            and all(type(seed) is int and 0 <= seed < 2**32 for seed in record["seeds"]), "Invalid seeds")
    require(isinstance(record["feature_exclusions"], list), "Feature exclusions must be explicit")
    require(record["target"] not in record.get("numeric", [])
            and record["target"] not in record.get("categorical", []), "Target listed as a feature")
    return record


def validate_manifest(manifest):
    require(isinstance(manifest, dict) and manifest.get("schema_version") == 1, "Unsupported catalog schema")
    records = manifest.get("datasets")
    require(isinstance(records, list) and 1 <= len(records) <= 100, "Invalid dataset list")
    seen = set()
    for record in records:
        validate_record(record)
        require(record["dataset_id"] not in seen, "Duplicate dataset ID")
        seen.add(record["dataset_id"])
    for key, limit in (("max_download_bytes", MAX_DOWNLOAD_BYTES), ("download_timeout_seconds", MAX_DOWNLOAD_SECONDS)):
        require(type(manifest.get(key)) is int and 1 <= manifest[key] <= limit, "Invalid download bound: " + key)
    return manifest


def load_manifest(path):
    return validate_manifest(read_json(path))


def select_records(manifest, profile):
    validate_manifest(manifest)
    require(profile in {"pilot", "suite", "all", "confirmation"}, "Unknown dataset profile")
    records = manifest["datasets"]
    if profile == "pilot":
        return [record for record in records if record.get("pilot") is True]
    if profile == "confirmation":
        return [record for record in records if record["role"] == "confirmation" and not record.get("known_public_sanity", False)]
    return list(records)
