[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{40}$')]
    [string]$ReleaseCommit,
    [Parameter(Mandatory = $true)]
    [string]$RegressionTask,
    [Parameter(Mandatory = $true)]
    [string]$ClassificationTask,
    [string]$RunRoot = ("C:\aq-queue-" + (Get-Date -Format "yyyyMMdd-HHmmss")),
    [string]$Python = ".\.venv\Scripts\python.exe",
    [ValidateRange(301, 10500)]
    [int]$TimeBudget = 10500
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$queueRepo = Split-Path -Parent $PSScriptRoot
$queuePython = (Resolve-Path -LiteralPath $Python).Path
$queueRegression = (Resolve-Path -LiteralPath $RegressionTask).Path
$queueClassification = (Resolve-Path -LiteralPath $ClassificationTask).Path
$queueOutput = [System.IO.Path]::GetFullPath($RunRoot)

Write-Host "Three serial jobs: regression OFF, regression ON, classification."
Write-Host "Each native budget is at most $TimeBudget seconds; early completion is allowed."
Write-Host "Keep this PowerShell window open and disable automatic sleep for the run."
& $queuePython -u (Join-Path $queueRepo "experiment_round5_overnight.py") `
    --release-commit $ReleaseCommit --regression-task $queueRegression `
    --classification-task $queueClassification --output $queueOutput --time-budget $TimeBudget
if ($LASTEXITCODE -ne 0) {
    throw "Queue stopped. Preserve $queueOutput, including queue.json, audit.json and evidence.zip."
}
Write-Host "Finished. Review $queueOutput\evidence.zip and keep the full run directory."
