"""No network, models or downloads: adversarial provenance/cache fixtures only."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.request

from research_lab.catalog import load_manifest, official_url, select_records, validate_manifest
from research_lab.fetch import _download, _OfficialRedirect, _open_url, _publish_immutable, cached_member, fetch_manifest

CATALOG = Path(__file__).resolve().parents[1] / "research_lab" / "datasets.json"
CSV = b"feature,y\n1,no\n2,yes\n"


class Response(io.BytesIO):
    def __init__(self, raw=CSV, *, headers=None, url="https://archive.ics.uci.edu/static/public/222/data.csv", status=200):
        super().__init__(raw)
        self.headers = {"Content-Type": "text/csv", **(headers or {})}
        self.status = status
        self.url = url

    def geturl(self):
        return self.url


class PublicCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lab cache Тест ")
        self.addCleanup(self.temporary.cleanup)
        self.cache = Path(self.temporary.name) / "cache"
        original = load_manifest(CATALOG)
        self.record = deepcopy(original["datasets"][0])
        self.record["dataset_id"] = "fixture"
        self.record["version"] = "v1"
        self.record["numeric"] = ["feature"]
        self.record["categorical"] = []
        self.manifest = {**original, "datasets": [self.record]}

    def fetch(self, raw=CSV, headers=None):
        with patch("research_lab.fetch._open_url", return_value=Response(raw, headers=headers)) as opener:
            result = fetch_manifest(self.manifest, self.cache, "pilot")
        return result, opener

    def receipt_path(self):
        return self.cache / "receipts" / "fixture" / "v1.json"

    def mutate_receipt(self, **changes):
        path = self.receipt_path()
        value = json.loads(path.read_text(encoding="utf-8"))
        value.update(changes)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_catalog_independent_roles_and_unknown_hashes(self):
        manifest = load_manifest(CATALOG)
        self.assertEqual(10, len(select_records(manifest, "all")))
        self.assertEqual(["bank_marketing", "wdbc", "wine_quality_red", "auto_mpg"],
                         [row["dataset_id"] for row in select_records(manifest, "pilot")])
        confirmation = select_records(manifest, "confirmation")
        self.assertEqual(5, len(confirmation))
        self.assertNotIn("iris", [row["dataset_id"] for row in confirmation])
        for row in manifest["datasets"]:
            self.assertEqual("CC BY 4.0", row["license"])
            self.assertIsNone(row["source_sha256"])
            self.assertIsNone(row["source_bytes"])
            self.assertIsNone(row["adapter_sha256"])
            self.assertIn("unknown", row["prior_exposure"])

    def test_null_source_facts_become_actual_immutable_receipt(self):
        result, opener = self.fetch(headers={"Content-Length": str(len(CSV))})
        self.assertEqual("fetched", result["records"][0]["status"])
        self.assertEqual(1, opener.call_count)
        path, receipt = cached_member(self.record, self.cache)
        sha = hashlib.sha256(CSV).hexdigest()
        self.assertEqual(CSV, path.read_bytes())
        self.assertEqual(self.cache / "objects" / sha, path)
        self.assertEqual(sha, receipt["source_sha256"])
        self.assertEqual(sha, receipt["member_sha256"])
        self.assertEqual(len(CSV), receipt["source_bytes"])
        self.assertTrue(receipt["retrieved_at_utc"].endswith("+00:00"))
        self.assertIsNone(self.record["source_sha256"])
        prior = self.receipt_path().read_bytes()
        with patch("research_lab.fetch._open_url", side_effect=AssertionError("unexpected network")):
            repeated = fetch_manifest(self.manifest, self.cache, "pilot")
            cached_member(self.record, self.cache)
        self.assertEqual("cached", repeated["records"][0]["status"])
        self.assertEqual(prior, self.receipt_path().read_bytes())

    def test_missing_cache_is_read_only_failure(self):
        with patch("research_lab.fetch._open_url", side_effect=AssertionError("unexpected network")):
            with self.assertRaises(ValueError):
                cached_member(self.record, self.cache)
        self.assertFalse(self.cache.exists())

    def test_same_length_source_corruption_blocks_and_never_refreshes(self):
        self.fetch()
        path, _ = cached_member(self.record, self.cache)
        path.write_bytes(CSV.replace(b"1,no", b"9,no"))
        with patch("research_lab.fetch._open_url", side_effect=AssertionError("unexpected network")):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                cached_member(self.record, self.cache)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                fetch_manifest(self.manifest, self.cache, "pilot")

    def test_changed_record_and_receipt_path_are_rejected(self):
        self.fetch()
        changed = deepcopy(self.record)
        changed["feature_exclusions"].append("feature")
        with self.assertRaisesRegex(ValueError, "catalog record"):
            cached_member(changed, self.cache)
        self.mutate_receipt(object_path="../../private.csv")
        with self.assertRaisesRegex(ValueError, "object receipt"):
            cached_member(self.record, self.cache)

    def test_receipt_header_and_timestamp_tampering_fail(self):
        self.fetch()
        self.mutate_receipt(columns=["y", "feature"])
        with self.assertRaisesRegex(ValueError, "header receipt"):
            cached_member(self.record, self.cache)
        self.mutate_receipt(columns=["feature", "y"], retrieved_at_utc="2026-10-21T00:00:00")
        with self.assertRaisesRegex(ValueError, "timestamp"):
            cached_member(self.record, self.cache)

    def test_declared_checksum_failure_publishes_no_receipt(self):
        self.record["source_sha256"] = "0" * 64
        result, _ = self.fetch()
        self.assertEqual("unavailable", result["records"][0]["status"])
        self.assertFalse(self.receipt_path().exists())

    def test_all_unavailable_results_are_retained(self):
        second = deepcopy(self.record)
        second.update(dataset_id="skipped", status="unverified", reason="No verified license")
        self.manifest["datasets"].append(second)
        with patch("research_lab.fetch._open_url", side_effect=OSError("offline")):
            result = fetch_manifest(self.manifest, self.cache, "all")
        self.assertEqual(["unavailable", "skipped"], [row["status"] for row in result["records"]])
        self.assertEqual(1, result["counts"]["unavailable"])
        self.assertEqual(1, result["counts"]["skipped"])

    def test_download_size_checks_declared_and_actual_bytes(self):
        for headers, bound in [({"Content-Length": "100000"}, 100), ({}, 8), ({"Content-Length": "2"}, 100)]:
            with self.subTest(headers=headers, bound=bound):
                with patch("research_lab.fetch._open_url", return_value=Response(headers=headers)):
                    with self.assertRaisesRegex(ValueError, "byte bound|Content-Length"):
                        _download(self.record, bound, 10)

    def test_total_deadline_checks_read_including_finalization(self):
        with patch("research_lab.fetch._open_url", return_value=Response()), patch(
                "research_lab.fetch.time.monotonic", side_effect=[0, 0, 11]):
            with self.assertRaisesRegex(ValueError, "total timeout"):
                _download(self.record, 1024, 10)

    def test_archives_binary_login_and_ragged_payloads_never_cached(self):
        payloads = [b"PK\x03\x04garbage", b"\x1f\x8bgarbage", b"x" * 257 + b"ustar" + b"x" * 50,
                    b"<!doctype html><html>login</html>", b"feature,y\n1,yes,extra\n", b"feature,y\n",
                    b"feature,y\n1,\0\n", b"wrong,header\n1,yes\n"]
        for raw in payloads:
            with self.subTest(raw=raw[:20]):
                result, _ = self.fetch(raw)
                self.assertEqual("unavailable", result["records"][0]["status"])
                self.assertFalse(self.receipt_path().exists())

    def test_malicious_http_headers_and_official_final_url(self):
        for headers in [{"Content-Type": "text/html"}, {"Content-Encoding": "gzip"},
                        {"Content-Disposition": 'attachment; filename="../../data.csv"'},
                        {"Content-Disposition": 'attachment; filename="source.zip"'}]:
            with self.subTest(headers=headers):
                result, _ = self.fetch(headers=headers)
                self.assertEqual("unavailable", result["records"][0]["status"])
        with patch("research_lab.fetch._open_url", return_value=Response(url="https://example.com/data.csv")):
            with self.assertRaisesRegex(ValueError, "official"):
                _download(self.record, 1024, 10)

    def test_official_response_without_mime_still_requires_strict_csv(self):
        response = Response()
        response.headers.pop("Content-Type")
        with patch("research_lab.fetch._open_url", return_value=response):
            raw, _, columns = _download(self.record, 1024, 10)
        self.assertEqual(CSV, raw)
        self.assertEqual(["feature", "y"], columns)
        response = Response(b"<!doctype html><html>Login</html>")
        response.headers.pop("Content-Type")
        with patch("research_lab.fetch._open_url", return_value=response):
            with self.assertRaisesRegex(ValueError, "HTML"):
                _download(self.record, 1024, 10)

    def test_https_path_credentials_archive_and_traversal_rejected(self):
        urls = ["http://archive.ics.uci.edu/static/public/9/data.csv",
                "https://attacker.invalid/static/public/9/data.csv",
                "https://user:secret@archive.ics.uci.edu/static/public/9/data.csv",
                "https://archive.ics.uci.edu/static/public/9/../9/data.csv",
                "https://archive.ics.uci.edu/static/public/9/%2e%2e/9/data.csv",
                "https://archive.ics.uci.edu/static/public/9/auto.zip",
                "https://archive.ics.uci.edu/static/public/9/data.csv?redirect=1"]
        for url in urls:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    official_url(url, download=True)
        self.record["dataset_id"] = "../fixture"
        with self.assertRaisesRegex(ValueError, "identity/version"):
            fetch_manifest(self.manifest, self.cache, "all")

    def test_redirect_official_https_and_count_bound(self):
        handler = _OfficialRedirect()
        req = urllib.request.Request(self.record["download_url"])
        with self.assertRaises(ValueError):
            handler.redirect_request(req, None, 302, "Found", {}, "http://archive.ics.uci.edu/static/public/222/data.csv")
        with self.assertRaises(ValueError):
            handler.redirect_request(req, None, 302, "Found", {}, "https://other.invalid/data.csv")
        req.lab_redirects = 3
        with self.assertRaisesRegex(ValueError, "Too many"):
            handler.redirect_request(req, None, 302, "Found", {}, self.record["download_url"])

    def test_development_fetch_inherits_normal_proxy_transport(self):
        with patch("research_lab.fetch.urllib.request.ProxyHandler") as proxy, patch(
                "research_lab.fetch.urllib.request.build_opener") as build:
            _open_url(self.record["download_url"], 10)
        proxy.assert_called_once_with()
        self.assertEqual(1, build.return_value.open.call_count)

    def test_immutable_publication_never_replaces_existing_bytes(self):
        path = self.cache / "test"
        _publish_immutable(path, b"first")
        _publish_immutable(path, b"first")
        with self.assertRaisesRegex(ValueError, "Immutable"):
            _publish_immutable(path, b"second")
        self.assertEqual(b"first", path.read_bytes())

    def test_symlink_cache_and_object_are_rejected(self):
        target = Path(self.temporary.name) / "target"
        target.mkdir()
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.cache.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation unavailable on this host")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            fetch_manifest(self.manifest, self.cache, "pilot")
        self.cache.unlink()
        self.fetch()
        obj, _ = cached_member(self.record, self.cache)
        saved = target / "source.csv"
        saved.write_bytes(obj.read_bytes())
        obj.unlink()
        obj.symlink_to(saved)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            cached_member(self.record, self.cache)

    def test_unverified_license_cannot_claim_ready(self):
        self.record["license"] = "unknown"
        with self.assertRaisesRegex(ValueError, "license"):
            validate_manifest(self.manifest)


if __name__ == "__main__":
    unittest.main()
