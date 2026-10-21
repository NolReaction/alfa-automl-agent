"""Review and optionally delete the pinned stale branches of this repository.

Dry-run is the default. Both modes read remote refs and may fetch immutable
objects for graph verification; only --execute can push. Configure your normal
Git authentication beforehand. No credentials, remote URLs or Git stderr are
printed. The archive commit must be pinned in the manifest before execution.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from urllib.parse import urlsplit


REPOSITORY = "NolReaction/alfa-automl-agent"
DEFAULT_MANIFEST = Path(__file__).resolve().parent / "docs/branch_cleanup_20261021.json"
DECISIONS = {"keep", "delete_merged", "archive_first", "delete_after_merge", "transfer_then_delete"}
AFTER_MERGE = {
    "work/murad-history-schema-v1", "work/history-auto-v1", "work/frequency-candidate-v1",
}
SHA = re.compile(r"[0-9a-f]{40}\Z")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*\Z")


class CleanupError(RuntimeError):
    """A bounded diagnostic that never embeds raw Git output."""


def valid_branch(value):
    return (isinstance(value, str) and len(value) <= 200 and NAME.fullmatch(value)
            and not any(token in value for token in ("..", "//", "@{"))
            and not value.startswith("refs/") and not value.endswith(("/", ".", ".lock"))
            and all(not part.startswith(".") for part in value.split("/")))


def valid_path(value):
    return (isinstance(value, str) and len(value) <= 300 and NAME.fullmatch(value)
            and not PurePosixPath(value).is_absolute() and ".." not in value.split("/")
            and "//" not in value and not value.endswith("/"))


def expected_remote(url):
    """Accept the expected GitHub owner/repo without returning a credential URL."""
    value = url.strip()
    if "://" in value:
        try:
            parsed = urlsplit(value)
            if parsed.scheme not in {"https", "ssh"} or parsed.hostname != "github.com":
                return False
            if parsed.port not in ({None, 443} if parsed.scheme == "https" else {None, 22}):
                return False
        except ValueError:
            return False
        if parsed.query or parsed.fragment:
            return False
        path = parsed.path.strip("/")
    else:
        match = re.fullmatch(r"[^\s/@]+@github\.com:([^\s]+)", value)
        if not match:
            return False
        path = match.group(1).strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return path.casefold() == REPOSITORY.casefold()


def validate_manifest(value):
    if not isinstance(value, dict) or value.get("version") != 1 or value.get("repository") != REPOSITORY:
        raise CleanupError("Manifest does not identify the expected repository/version.")
    archive = value.get("archive")
    if (not isinstance(archive, dict) or not valid_branch(archive.get("branch"))
            or not archive["branch"].startswith("archive/")):
        raise CleanupError("Manifest archive branch is invalid.")
    if archive.get("commit") is not None and not SHA.fullmatch(str(archive["commit"])):
        raise CleanupError("Manifest archive commit must be null or a full SHA-1.")
    rows = value.get("branches")
    if not isinstance(rows, list) or len(rows) > 100:
        raise CleanupError("Manifest branch list is invalid.")
    seen = set()
    for row in rows:
        if (not isinstance(row, dict) or not valid_branch(row.get("name"))
                or row["name"] in seen or not SHA.fullmatch(str(row.get("expected_sha", "")))
                or row.get("decision") not in DECISIONS
                or type(row.get("require_main_ancestor", False)) is not bool):
            raise CleanupError("Manifest contains an invalid or duplicate branch entry.")
        seen.add(row["name"])
        transfers = row.get("required_transfers", [])
        if not isinstance(transfers, list) or len(transfers) > 100:
            raise CleanupError("Manifest document transfer list is invalid.")
        if row["decision"] == "transfer_then_delete" and not transfers:
            raise CleanupError("Document branches require explicit transfer paths.")
        for item in transfers:
            if (not isinstance(item, dict) or not valid_path(item.get("source_path"))
                    or not valid_path(item.get("main_path"))):
                raise CleanupError("Manifest document transfer path is invalid.")
    return value


class Git:
    def __init__(self, repo, runner=None):
        self.repo = Path(repo).resolve()
        self.runner = runner or subprocess.run

    def command(self, args, *, allowed=(0,)):
        env = dict(os.environ)
        env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
        try:
            result = self.runner(["git", "-C", str(self.repo), *args],
                                 capture_output=True, text=True, stdin=subprocess.DEVNULL,
                                 env=env, timeout=60, check=False)
        except (OSError, subprocess.TimeoutExpired):
            raise CleanupError("Git could not complete the requested operation; no raw output is displayed.") from None
        if result.returncode not in allowed:
            text = (result.stderr or "").casefold()
            if any(token in text for token in ("authenticat", "authoriz", "credential", "username", "password",
                                               "permission denied", "access denied", "terminal prompts disabled", "http 401", "http 403")):
                raise CleanupError("Git authentication is unavailable or was rejected; configure existing repository access and retry.")
            raise CleanupError(f"Git operation failed (exit {result.returncode}); no raw output is displayed.")
        return result

    def verify_origin(self):
        if not expected_remote(self.command(["remote", "get-url", "origin"]).stdout):
            raise CleanupError("Origin is not the expected GitHub owner/repository; refusing cleanup.")
        pushes = self.command(["remote", "get-url", "--push", "--all", "origin"]).stdout.splitlines()
        # Multiple destinations make a single push's outcome ambiguous, even
        # when they are different transports to the same repository.
        if len(pushes) != 1 or not expected_remote(pushes[0]):
            raise CleanupError("Origin must have exactly one push destination for the expected GitHub owner/repository.")

    def heads(self):
        output = self.command(["ls-remote", "--heads", "origin"]).stdout
        heads = {}
        for line in output.splitlines():
            parts = line.split()
            if len(parts) != 2 or not SHA.fullmatch(parts[0]) or not parts[1].startswith("refs/heads/"):
                raise CleanupError("Remote returned an invalid head listing.")
            name = parts[1][len("refs/heads/"):]
            if name in heads:
                raise CleanupError("Remote returned duplicate heads.")
            heads[name] = parts[0]
        return heads

    def checked_out(self):
        output = self.command(["worktree", "list", "--porcelain"]).stdout
        return {line[len("branch refs/heads/"):] for line in output.splitlines()
                if line.startswith("branch refs/heads/")}

    def current_branch(self):
        result = self.command(["symbolic-ref", "--quiet", "--short", "HEAD"], allowed=(0, 1))
        return result.stdout.strip() if result.returncode == 0 else None

    def ancestor(self, earlier, later):
        return self.command(["merge-base", "--is-ancestor", earlier, later], allowed=(0, 1)).returncode == 0

    def blob(self, commit, path):
        result = self.command(["rev-parse", "--verify", f"{commit}:{path}"], allowed=(0, 1, 128))
        output = result.stdout.strip()
        return output if result.returncode == 0 and SHA.fullmatch(output) else None

    def delete(self, branch, expected_sha):
        ref = "refs/heads/" + branch
        self.command(["push", "--porcelain", f"--force-with-lease={ref}:{expected_sha}", "origin", ":" + ref])


def hard_protected(branch, archive_branch, checked_out):
    lower = branch.casefold()
    return (branch in {"main", archive_branch} or branch in checked_out
            or lower.startswith(("archive/", "work/nika", "nika/")))


def cleanup(repo, manifest, *, execute=False, runner=None):
    """Verify the entire plan before the first push, then use a per-ref lease."""
    if type(execute) is not bool:
        raise CleanupError("Execution must be an explicit boolean request.")
    manifest = validate_manifest(manifest)
    git = Git(repo, runner=runner)
    git.verify_origin()
    checked_out = git.checked_out()
    heads = git.heads()
    archive = manifest["archive"]
    archive_sha, archive_branch = archive.get("commit"), archive["branch"]
    report = {"mode": "execute" if execute else "dry_run", "repository": REPOSITORY,
              "status": "blocked", "kept": [], "skipped": [], "blocked": [],
              "would_delete": [], "deleted": []}
    if execute and git.current_branch() != "main":
        report["blocked"].append({"branch": None, "reason": "execute_requires_main_checkout"})
        return report
    candidates = []
    for row in manifest["branches"]:
        name = row["name"]
        if row["decision"] == "keep":
            report["kept"].append(name)
        elif hard_protected(name, archive_branch, checked_out):
            report["blocked"].append({"branch": name, "reason": "protected_or_checked_out"})
        elif name not in heads:
            report["skipped"].append({"branch": name, "reason": "remote_ref_missing"})
        elif heads[name] != row["expected_sha"]:
            report["blocked"].append({"branch": name, "reason": "remote_head_moved"})
        else:
            candidates.append(row)
    if not candidates and not report["blocked"]:
        report["status"] = "nothing_to_delete"
        return report
    main_sha = heads.get("main")
    if main_sha is None:
        report["blocked"].append({"branch": "main", "reason": "remote_main_missing"})
    if archive_sha is None:
        report["blocked"].append({"branch": archive_branch, "reason": "archive_commit_not_pinned"})
    elif heads.get(archive_branch) != archive_sha:
        report["blocked"].append({"branch": archive_branch, "reason": "archive_missing_or_moved"})
    if report["blocked"]:
        return report
    # Fetch immutable objects, never update local branch refs or FETCH_HEAD.
    git.command(["fetch", "--no-tags", "--no-write-fetch-head", "origin", archive_sha, main_sha])
    for row in candidates:
        name, expected = row["name"], row["expected_sha"]
        if not git.ancestor(expected, archive_sha):
            report["blocked"].append({"branch": name, "reason": "tip_not_in_pinned_archive"})
        transfers = row.get("required_transfers", [])
        needs_main = (name in AFTER_MERGE or row["decision"] in {"delete_merged", "delete_after_merge"}
                      or row.get("require_main_ancestor", False))
        in_main = git.ancestor(expected, main_sha) if needs_main or transfers else False
        if needs_main and not in_main:
            report["blocked"].append({"branch": name, "reason": "tip_not_merged_into_main"})
        # A merged tip preserves every original blob in main's history, even
        # when a current document was subsequently edited. For squash/cherry-
        # picked documentation, require an exact current blob as fallback.
        for item in transfers if not in_main else []:
            source = git.blob(expected, item["source_path"])
            target = git.blob(main_sha, item["main_path"])
            if source is None or target != source:
                report["blocked"].append({"branch": name, "reason": "document_not_transferred_exactly",
                                           "source_path": item["source_path"], "main_path": item["main_path"]})
    if report["blocked"]:
        return report
    report["would_delete"] = [{"branch": row["name"], "expected_sha": row["expected_sha"]} for row in candidates]
    if not execute:
        report["status"] = "dry_run_ready"
        return report
    for row in candidates:
        # A preceding successful deletion does not authorize a moved next ref.
        try:
            git.verify_origin()
            if (git.current_branch() != "main"
                    or hard_protected(row["name"], archive_branch, git.checked_out())):
                report["blocked"].append({"branch": row["name"], "reason": "local_protection_changed_before_push"})
                report["status"] = "partial" if report["deleted"] else "blocked"
                return report
            live = git.heads()
            if (live.get(archive_branch) != archive_sha or live.get("main") != main_sha
                    or live.get(row["name"]) != row["expected_sha"]):
                report["blocked"].append({"branch": row["name"], "reason": "remote_refs_changed_before_push"})
                report["status"] = "partial" if report["deleted"] else "blocked"
                return report
            git.delete(row["name"], row["expected_sha"])
        except CleanupError as error:
            report["blocked"].append({"branch": row["name"], "reason": "pre_push_check_or_push_failed", "error": str(error)})
            report["status"] = "partial" if report["deleted"] else "blocked"
            return report
        report["deleted"].append({"branch": row["name"], "expected_sha": row["expected_sha"]})
    report["status"] = "completed"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--execute", action="store_true", help="Explicitly allow the verified leased deletes")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        report = cleanup(args.repo, manifest, execute=args.execute)
    except (CleanupError, OSError, ValueError, TypeError):
        # Deliberately omit exception/raw subprocess text that could carry a URL.
        import sys
        error = sys.exc_info()[1]
        message = str(error) if isinstance(error, CleanupError) else "Could not read or validate the cleanup manifest."
        report = {"status": "blocked", "mode": "execute" if args.execute else "dry_run", "error": message}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"dry_run_ready", "nothing_to_delete", "completed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
