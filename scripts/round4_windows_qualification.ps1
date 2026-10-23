[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{40}$')]
    [string]$ReleaseCommit,
    [Parameter(Mandatory = $true)]
    [string]$PublicTask,
    [string]$RunRoot = ("C:\aq-r4-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [string]$Python = ".\.venv\Scripts\python.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$qualificationRepo = Split-Path -Parent $PSScriptRoot
$qualificationPython = (Resolve-Path -LiteralPath $Python).Path
$qualificationTask = (Resolve-Path -LiteralPath $PublicTask).Path
$qualificationRoot = [System.IO.Path]::GetFullPath($RunRoot)

# This is a small caller of the existing supervisor, not another search runner.
# Native CLI remains at 10500s. The outer 10530s watchdog is not solver time.
$qualificationCode = @'
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

repo, expected_sha, public_task, output = sys.argv[1:]
repo, public_task, output = map(Path, (repo, public_task, output))
sys.path.insert(0, str(repo))
import psutil
from agent.task_spec import load_task
from research_lab.integrity import check_identity, file_hash, source_identity, write_json
from research_lab.process import coordinator_lock, machine_lock_path, supervise
from validation import validate_predictions

if os.name != "nt" or sys.version_info[:2] != (3, 12):
    raise SystemExit("Use Windows and Python 3.12 for this qualification.")
identity = source_identity(repo)
if identity["sha"].lower() != expected_sha.lower() or identity["dirty"]:
    raise SystemExit("Release SHA must match a clean checkout; no checkout is changed by this script.")
if output.exists() or output.is_relative_to(repo):
    raise SystemExit("RunRoot must be a new directory outside the checkout.")
versions = {item.metadata["Name"].lower(): item.version for item in metadata.distributions()}
for line in (repo / "requirements.txt").read_text(encoding="utf-8").splitlines():
    if line.strip() and not line.lstrip().startswith("#"):
        name, expected = line.strip().split("==", 1)
        if metadata.version(name) != expected:
            raise SystemExit("Pinned runtime dependency mismatch: " + name)
memory = psutil.virtual_memory()
if memory.total < 30_000_000_000:
    raise SystemExit("This sequence requires the intended 32 GB workstation; record smaller probes separately.")
spec = load_task(public_task)
if spec.classification or spec.metric.name != "RMSE":
    raise SystemExit("The first long qualification is one public RMSE regression task.")

output.mkdir(parents=True, exist_ok=False)
task = output / "task"
task.mkdir()
shutil.copyfile(public_task, task / "task.md")
relative_data = Path(spec.data_dir).relative_to(public_task.parent)
data = task / relative_data
data.mkdir(parents=True, exist_ok=True)
for name in ("train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv"):
    shutil.copyfile(Path(spec.data_dir) / name, data / name)
inputs = {str(path.relative_to(task)): file_hash(path) for path in task.rglob("*") if path.is_file()}
for name in ("requirements.txt", "requirements.platform.txt"):
    shutil.copyfile(repo / name, output / name)
plan = {"release": identity, "python": sys.version, "platform": platform.platform(),
        "packages": versions, "physical_ram_bytes": memory.total, "available_ram_bytes": memory.available,
        "logical_cpus": os.cpu_count(), "threads": 8, "sampled_tree_rss_limit_bytes": 28 * 1024**3,
        "native_cli_seconds": 10500, "external_watchdog_seconds": 10530,
        "watchdog_note": "30s is a hang watchdog margin, not extra native training time or deadline proof.",
        "input_sha256": inputs, "wrapper_sha256": file_hash(repo / "scripts/round4_windows_qualification.ps1"),
        "requirements_sha256": {name: file_hash(output / name) for name in
                                ("requirements.txt", "requirements.platform.txt")},
        "scope": "Known public TRAIN development; not hidden TEST quality; Windows resource qualification pending review."}
write_json(output / "plan.json", plan)
with (output / "build.log").open("w", encoding="utf-8") as log:
    subprocess.run([sys.executable, str(repo / "build_submission.py"), "--output", str(output / "solution.zip")],
                   cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
flat = output / "flat"
with zipfile.ZipFile(output / "solution.zip") as archive:
    if any(Path(name).name != name for name in archive.namelist()):
        raise SystemExit("Expected the organizer's flat ZIP.")
    archive.extractall(flat)
plan["zip_sha256"] = file_hash(output / "solution.zip")
plan["flat_sha256"] = {path.name: file_hash(path) for path in flat.iterdir() if path.is_file()}
write_json(output / "plan.json", plan)
argv = [sys.executable, "-u", str(flat / "solution.py"), str(task / "task.md"),
        "--mode", "offline", "--time-budget", "10500"]
with coordinator_lock(machine_lock_path()):
    execution = supervise(argv, task, output / "execution", 10530, 28 * 1024**3, threads=8)
write_json(output / "execution.json", execution)
reports = list(task.glob(".agent_runs/*/run.json"))
native = json.loads(reports[0].read_text(encoding="utf-8")) if len(reports) == 1 else {}
budget, elapsed = native.get("time_budget_seconds"), native.get("elapsed_seconds")
numeric = lambda value: type(value) in (int, float)
checks = {"root_exit_zero": execution.get("exit_code") == 0,
          "watchdog_not_reached": execution.get("external_timeout") is False,
          "memory_limit_not_reached": execution.get("memory_limit_exceeded") is False,
          "tree_resources_available": execution.get("resources", {}).get("available") is True,
          "owned_cleanup_complete": execution.get("cleanup", {}).get("complete") is True,
          "one_native_report": len(reports) == 1,
          "native_deadline_reported_within_budget": numeric(budget) and numeric(elapsed)
              and 0 < budget <= 10500 and 0 <= elapsed <= budget,
          "api_attempts_zero": native.get("api", {}).get("attempts") == 0,
          "api_responses_zero": native.get("api", {}).get("responses") == 0,
          "input_hashes_unchanged": inputs == {name: file_hash(task / name) for name in inputs},
          "flat_hashes_unchanged": plan["flat_sha256"] == {name: file_hash(flat / name)
                                                         for name in plan["flat_sha256"]}}
try:
    check_identity(identity)
    checks["release_sources_unchanged"] = True
except (OSError, ValueError):
    checks["release_sources_unchanged"] = False
validation = validate_predictions(task)
checks["output_structure_valid"] = validation.get("valid") is True
summary = {"qualification": "requires_review", "checks": checks,
           "automatic_mechanism_checks_passed": all(checks.values()), "output_validation": validation,
           "native_report": str(reports[0]) if len(reports) == 1 else None,
           "native_budget_seconds": budget, "native_elapsed_seconds": elapsed,
           "native_status": native.get("status"), "native_reason": native.get("reason"),
           "final_check": native.get("evaluations", {}).get("final_check"),
           "publication": native.get("publication"), "published_recipe": native.get("published_recipe"),
           "refit_execution": native.get("refit_execution"),
           "submission_sha256": file_hash(task / "submission.csv") if (task / "submission.csv").is_file() else None,
           "note": "No complete receipt/full-refit or hidden-score audit is claimed. Watchdog margin cannot certify native deadline."}
write_json(output / "qualification.json", summary)
print("Evidence:", output)
print("Automatic mechanism checks:", summary["automatic_mechanism_checks_passed"], "; qualification requires review.")
raise SystemExit(0 if summary["automatic_mechanism_checks_passed"] else 2)
'@

$qualificationDriver = Join-Path ([System.IO.Path]::GetTempPath()) (([System.IO.Path]::GetRandomFileName()) + ".py")
try {
    # A file avoids Windows PowerShell 5.1's native -c quote rewriting.
    [System.IO.File]::WriteAllText($qualificationDriver, $qualificationCode, (New-Object System.Text.UTF8Encoding($false)))
    & $qualificationPython $qualificationDriver $qualificationRepo $ReleaseCommit $qualificationTask $qualificationRoot
    $qualificationExitCode = $LASTEXITCODE
}
finally {
    Remove-Item -LiteralPath $qualificationDriver -ErrorAction SilentlyContinue
}
if ($qualificationExitCode -ne 0) {
    throw "Qualification stopped (exit $qualificationExitCode). Preserve RunRoot and inspect its reports; do not start another ML run."
}
