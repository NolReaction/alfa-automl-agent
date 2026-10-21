"""Safety checks with fabricated Git responses; no network or branch deletion."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cleanup_branches as cleanup


MAIN, ARCHIVE, TIP, OTHER, BLOB = (character * 40 for character in "12345")
ARCHIVE_BRANCH = "archive/2026-10-21"
STALE = "work/retired-test"


def entry(name=STALE, sha=TIP, decision="archive_first", transfers=(), main_required=False):
    return {"name": name, "expected_sha": sha, "decision": decision,
            "require_main_ancestor": main_required, "required_transfers": list(transfers)}


def manifest(rows=None):
    return {"version": 1, "repository": cleanup.REPOSITORY,
            "archive": {"branch": ARCHIVE_BRANCH, "commit": ARCHIVE},
            "branches": rows if rows is not None else [entry()]}


class FakeGit:
    def __init__(self):
        self.calls = []
        self.url = "https://github.com/NolReaction/alfa-automl-agent.git"
        self.push_urls = None
        self.push_url_sequence = []
        self.remote_heads = {"main": MAIN, ARCHIVE_BRANCH: ARCHIVE, STALE: TIP}
        self.head_sequence = []
        self.checked_out = {"main"}
        self.current_branch = "main"
        self.current_sequence = []
        self.worktree_sequence = []
        self.archived, self.merged = {TIP, OTHER}, {TIP, OTHER}
        self.blobs = {}
        self.fail_operation = None
        self.failure_stderr = "an internal error"
        self.timeout_operation = None
        self.auth_prompt_flags = []

    @property
    def pushes(self):
        return [args for args in self.calls if args[0] == "push"]

    def __call__(self, argv, **kwargs):
        # No real subprocess is called. Do not retain the full environment.
        args = argv[3:]
        self.calls.append(args)
        self.auth_prompt_flags.append((kwargs["env"]["GIT_TERMINAL_PROMPT"], kwargs["env"]["GCM_INTERACTIVE"]))
        assert kwargs["capture_output"] and kwargs["text"] and kwargs["check"] is False
        assert kwargs["stdin"] == subprocess.DEVNULL and kwargs["timeout"] == 60
        operation = args[0]
        if operation == self.timeout_operation:
            raise subprocess.TimeoutExpired(argv, 60)
        if operation == self.fail_operation:
            return SimpleNamespace(returncode=128, stdout="", stderr=self.failure_stderr)
        output, code = "", 0
        if operation == "remote":
            if args == ["remote", "get-url", "origin"]:
                output = self.url + "\n"
            else:
                assert args == ["remote", "get-url", "--push", "--all", "origin"]
                urls = self.push_url_sequence.pop(0) if self.push_url_sequence else self.push_urls
                output = "\n".join(urls if urls is not None else [self.url]) + "\n"
        elif operation == "worktree":
            checked = self.worktree_sequence.pop(0) if self.worktree_sequence else self.checked_out
            output = "\n".join(f"worktree /fake/{index}\nHEAD {MAIN}\nbranch refs/heads/{name}\n"
                               for index, name in enumerate(sorted(checked)))
        elif operation == "symbolic-ref":
            assert args == ["symbolic-ref", "--quiet", "--short", "HEAD"]
            branch = self.current_sequence.pop(0) if self.current_sequence else self.current_branch
            output, code = (branch + "\n", 0) if branch else ("", 1)
        elif operation == "ls-remote":
            assert args == ["ls-remote", "--heads", "origin"]
            heads = self.head_sequence.pop(0) if self.head_sequence else self.remote_heads
            output = "".join(f"{sha}\trefs/heads/{name}\n" for name, sha in heads.items())
        elif operation == "fetch":
            assert args == ["fetch", "--no-tags", "--no-write-fetch-head", "origin", ARCHIVE, MAIN]
        elif operation == "merge-base":
            assert args[:2] == ["merge-base", "--is-ancestor"]
            allowed = self.archived if args[3] == ARCHIVE else self.merged
            code = 0 if args[2] in allowed else 1
        elif operation == "rev-parse":
            assert args[:2] == ["rev-parse", "--verify"]
            value = self.blobs.get(args[2])
            output, code = (value + "\n", 0) if value else ("", 128)
        elif operation == "push":
            pass
        else:
            raise AssertionError("Unexpected fake Git operation: " + operation)
        return SimpleNamespace(returncode=code, stdout=output, stderr="")


class BranchCleanupTests(unittest.TestCase):
    def setUp(self):
        self.git = FakeGit()

    def run_plan(self, plan=None, *, execute=False):
        return cleanup.cleanup(Path("/fake/repo"), plan or manifest(), execute=execute, runner=self.git)

    def assert_blocked_without_push(self, plan, reason):
        report = self.run_plan(plan, execute=True)
        self.assertEqual(report["status"], "blocked")
        self.assertIn(reason, [item["reason"] for item in report["blocked"]])
        self.assertEqual(self.git.pushes, [])
        return report

    def test_default_dry_run_verifies_plan_without_any_push(self):
        report = self.run_plan()
        self.assertEqual(report["status"], "dry_run_ready")
        self.assertEqual(report["mode"], "dry_run")
        self.assertEqual(report["would_delete"], [{"branch": STALE, "expected_sha": TIP}])
        self.assertEqual(report["deleted"], [])
        self.assertEqual(self.git.pushes, [])
        self.assertTrue(all(pair == ("0", "never") for pair in self.git.auth_prompt_flags))

    def test_execute_uses_exact_ref_and_expected_head_lease(self):
        report = self.run_plan(execute=True)
        self.assertEqual(report["status"], "completed")
        ref = "refs/heads/" + STALE
        self.assertEqual(self.git.pushes, [["push", "--porcelain", f"--force-with-lease={ref}:{TIP}", "origin", ":" + ref]])
        self.assertEqual(report["deleted"], [{"branch": STALE, "expected_sha": TIP}])
        self.assertNotIn("--force", self.git.pushes[0])

    def test_moved_ref_blocks_entire_plan_before_deleting_another_valid_ref(self):
        second = "work/another-retired"
        self.git.remote_heads[second] = OTHER
        self.git.remote_heads[STALE] = MAIN
        self.assert_blocked_without_push(manifest([entry(second, OTHER), entry()]), "remote_head_moved")
        self.assertFalse(any(args[0] == "fetch" for args in self.git.calls))

    def test_missing_ref_is_skipped_without_deletion_or_fetch(self):
        del self.git.remote_heads[STALE]
        report = self.run_plan(execute=True)
        self.assertEqual(report["status"], "nothing_to_delete")
        self.assertEqual(report["skipped"], [{"branch": STALE, "reason": "remote_ref_missing"}])
        self.assertEqual(self.git.pushes, [])

    def test_missing_archive_pin_is_never_executable(self):
        plan = manifest()
        plan["archive"]["commit"] = None
        self.assert_blocked_without_push(plan, "archive_commit_not_pinned")

    def test_remote_archive_missing_or_moved_blocks_before_fetch(self):
        for missing in (True, False):
            with self.subTest(missing=missing):
                self.git = FakeGit()
                if missing:
                    del self.git.remote_heads[ARCHIVE_BRANCH]
                else:
                    self.git.remote_heads[ARCHIVE_BRANCH] = OTHER
                self.assert_blocked_without_push(manifest(), "archive_missing_or_moved")
                self.assertFalse(any(args[0] == "fetch" for args in self.git.calls))

    def test_archive_must_reach_each_exact_tip(self):
        self.git.archived.clear()
        self.assert_blocked_without_push(manifest(), "tip_not_in_pinned_archive")

    def test_pr21_23_24_require_main_ancestry_even_if_manifest_weakens_flag(self):
        for name in sorted(cleanup.AFTER_MERGE):
            with self.subTest(branch=name):
                self.git = FakeGit()
                self.git.remote_heads[name] = TIP
                self.git.merged.clear()
                self.assert_blocked_without_push(manifest([entry(name)]), "tip_not_merged_into_main")

    def test_claimed_merged_branch_requires_actual_main_ancestry(self):
        self.git.merged.clear()
        self.assert_blocked_without_push(manifest([entry(decision="delete_merged")]), "tip_not_merged_into_main")

    def test_main_nika_archive_and_checked_out_branches_are_hard_protected(self):
        for name in ("main", "work/nika-research-policy", "work/nika-next", "nika/experiment", ARCHIVE_BRANCH,
                     "archive/older-release", STALE):
            with self.subTest(branch=name):
                self.git = FakeGit()
                self.git.remote_heads[name] = TIP
                if name == STALE:
                    self.git.checked_out.add(name)
                self.assert_blocked_without_push(manifest([entry(name)]), "protected_or_checked_out")

    def test_other_linked_worktree_branch_is_also_protected(self):
        self.git.checked_out.update({"work/unrelated-active", STALE})
        self.assert_blocked_without_push(manifest(), "protected_or_checked_out")

    def test_new_checkout_or_new_linked_worktree_is_rechecked_before_push(self):
        for change in ("current_branch", "linked_worktree"):
            with self.subTest(change=change):
                self.git = FakeGit()
                if change == "current_branch":
                    self.git.current_sequence = ["main", "work/new-active"]
                else:
                    self.git.worktree_sequence = [{"main"}, {"main", STALE}]
                self.assert_blocked_without_push(manifest(), "local_protection_changed_before_push")

    def test_execute_requires_main_checkout_but_dry_run_does_not(self):
        for branch in ("work/current-research", None):
            with self.subTest(branch=branch):
                self.git = FakeGit()
                self.git.current_branch = branch
                self.assert_blocked_without_push(manifest(), "execute_requires_main_checkout")
                self.assertEqual(self.run_plan()["status"], "dry_run_ready")

    def test_keep_rows_do_not_become_delete_candidates_when_remote_heads_advance(self):
        plan = manifest([entry("main", MAIN, "keep"), entry("work/nika-research-policy", TIP, "keep")])
        self.git.remote_heads["work/nika-research-policy"] = OTHER
        report = self.run_plan(plan, execute=True)
        self.assertEqual(report["status"], "nothing_to_delete")
        self.assertEqual(report["kept"], ["main", "work/nika-research-policy"])
        self.assertEqual(self.git.pushes, [])

    def test_rechecks_target_archive_and_main_immediately_before_push(self):
        for changed in (STALE, ARCHIVE_BRANCH, "main"):
            with self.subTest(changed=changed):
                self.git = FakeGit()
                before = dict(self.git.remote_heads)
                later = {**before, changed: OTHER}
                self.git.head_sequence = [before, later]
                self.assert_blocked_without_push(manifest(), "remote_refs_changed_before_push")

    def test_a_later_ref_race_reports_partial_and_does_not_delete_moved_tip(self):
        second = "work/second-retired"
        self.git.remote_heads[second] = OTHER
        before = dict(self.git.remote_heads)
        self.git.head_sequence = [before, before, {**before, second: MAIN}]
        report = self.run_plan(manifest([entry(), entry(second, OTHER)]), execute=True)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["deleted"], [{"branch": STALE, "expected_sha": TIP}])
        self.assertEqual(len(self.git.pushes), 1)

    def test_document_transfer_must_match_blob_before_any_push(self):
        source_path, main_path = "docs/old/report.md", "docs/archive/report.md"
        plan = manifest([entry(decision="transfer_then_delete", transfers=[{
            "source_path": source_path, "main_path": main_path}])])
        for target in (None, OTHER):
            with self.subTest(target=target):
                self.git = FakeGit()
                self.git.merged.clear()
                self.git.blobs = {f"{TIP}:{source_path}": BLOB}
                if target:
                    self.git.blobs[f"{MAIN}:{main_path}"] = target
                self.assert_blocked_without_push(plan, "document_not_transferred_exactly")
        self.git = FakeGit()
        self.git.merged.clear()
        self.git.blobs = {f"{TIP}:{source_path}": BLOB, f"{MAIN}:{main_path}": BLOB}
        self.assertEqual(self.run_plan(plan, execute=True)["status"], "completed")

    def test_merged_document_head_preserves_original_history_after_current_doc_edit(self):
        source_path = "docs/research_report.md"
        plan = manifest([entry(decision="transfer_then_delete", transfers=[{
            "source_path": source_path, "main_path": source_path}])])
        self.git.blobs = {f"{TIP}:{source_path}": BLOB, f"{MAIN}:{source_path}": OTHER}
        report = self.run_plan(plan, execute=True)
        self.assertEqual(report["status"], "completed")
        self.assertFalse(any(args[0] == "rev-parse" for args in self.git.calls))

    def test_wrong_origin_is_rejected_without_exposing_credential_url(self):
        self.git.url = "https://user:FAKE_SECRET@github.com/another/repo.git"
        with self.assertRaises(cleanup.CleanupError) as error:
            self.run_plan(execute=True)
        self.assertNotIn("FAKE_SECRET", str(error.exception))
        self.assertNotIn(self.git.url, str(error.exception))
        self.assertEqual(self.git.calls, [["remote", "get-url", "origin"]])

    def test_expected_credentialed_origin_is_never_printed_in_a_report(self):
        self.git.url = "https://user:FAKE_SECRET@github.com/NolReaction/alfa-automl-agent.git"
        report = self.run_plan()
        self.assertEqual(report["status"], "dry_run_ready")
        self.assertNotIn("FAKE_SECRET", json.dumps(report))
        self.assertNotIn(self.git.url, json.dumps(report))

    def test_foreign_or_multiple_effective_push_urls_are_rejected_before_fetch(self):
        for urls in (["https://github.com/another/fork.git"],
                     [self.git.url, "https://github.com/another/fork.git"],
                     [self.git.url, "git@github.com:NolReaction/alfa-automl-agent.git"]):
            with self.subTest(urls=urls):
                self.git = FakeGit()
                self.git.push_urls = urls
                with self.assertRaises(cleanup.CleanupError):
                    self.run_plan(execute=True)
                self.assertEqual(self.git.pushes, [])
                self.assertFalse(any(args[0] in {"fetch", "ls-remote"} for args in self.git.calls))

    def test_push_destination_is_rechecked_after_preflight(self):
        self.git.push_url_sequence = [[self.git.url], ["https://github.com/another/fork.git"]]
        report = self.run_plan(execute=True)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(self.git.pushes, [])
        self.assertNotIn("https://", json.dumps(report))

    def test_auth_failure_is_clear_without_git_stderr_or_remote_url(self):
        self.git.fail_operation = "push"
        self.git.failure_stderr = "Authentication failed for https://user:FAKE_SECRET@github.com/owner/repo"
        report = self.run_plan(execute=True)
        self.assertEqual(report["status"], "blocked")
        self.assertIn("authentication", report["blocked"][0]["error"].lower())
        self.assertNotIn("FAKE_SECRET", json.dumps(report))
        self.assertNotIn("https://", json.dumps(report))
        self.assertEqual(report["deleted"], [])

    def test_fetch_timeout_never_reaches_push_or_echoes_process_arguments(self):
        self.git.timeout_operation = "fetch"
        with self.assertRaises(cleanup.CleanupError) as error:
            self.run_plan(execute=True)
        self.assertEqual(self.git.pushes, [])
        self.assertNotIn("origin", str(error.exception))

    def test_invalid_manifest_is_rejected_before_subprocesses(self):
        for field, value in (("name", "../main"), ("name", "--delete"), ("expected_sha", "short"),
                             ("decision", "force"), ("require_main_ancestor", "yes")):
            with self.subTest(field=field, value=value):
                self.git = FakeGit()
                plan = manifest()
                plan["branches"][0][field] = value
                with self.assertRaises(cleanup.CleanupError):
                    self.run_plan(plan, execute=True)
                self.assertEqual(self.git.calls, [])

    def test_duplicate_manifest_ref_is_rejected_before_subprocesses(self):
        with self.assertRaises(cleanup.CleanupError):
            self.run_plan(manifest([entry(), entry()]), execute=True)
        self.assertEqual(self.git.calls, [])

    def test_execution_argument_cannot_be_a_truthy_string(self):
        with self.assertRaises(cleanup.CleanupError):
            self.run_plan(execute="yes")
        self.assertEqual(self.git.calls, [])

    def test_expected_owner_repo_validation_for_https_and_ssh(self):
        accepted = ["https://github.com/NolReaction/alfa-automl-agent.git", "git@github.com:NolReaction/alfa-automl-agent.git",
                    "ssh://git@github.com/NolReaction/alfa-automl-agent.git", "https://github.com/nolreaction/alfa-automl-agent"]
        rejected = ["https://github.com.evil.invalid/NolReaction/alfa-automl-agent.git", "https://github.com/NolReaction/other.git",
                    "https://github.com:444/NolReaction/alfa-automl-agent.git", "https://github.com/[", "ssh://[",
                    "https://github.com/NolReaction/alfa-automl-agent.git?token=FAKE_SECRET"]
        for value in accepted:
            self.assertTrue(cleanup.expected_remote(value))
        for value in rejected:
            self.assertFalse(cleanup.expected_remote(value))

    def test_cli_without_execute_never_pushes_and_flag_explicitly_enables_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest()), encoding="utf-8")
            args = ["--repo", directory, "--manifest", str(path)]
            for execute in (False, True):
                with self.subTest(execute=execute):
                    self.git = FakeGit()
                    output = io.StringIO()
                    with patch.object(cleanup.subprocess, "run", side_effect=self.git), redirect_stdout(output):
                        status = cleanup.main([*args, *(["--execute"] if execute else [])])
                    self.assertEqual(status, 0)
                    self.assertEqual(len(self.git.pushes), int(execute))
                    self.assertEqual(json.loads(output.getvalue())["mode"], "execute" if execute else "dry_run")


if __name__ == "__main__":
    unittest.main()
