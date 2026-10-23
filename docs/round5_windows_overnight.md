# Windows: three serial public native runs

This developer queue prepares three fresh task copies before any training:

| Order | Task | Native CLI difference | Maximum native budget |
| --- | --- | --- | --- |
| 1 | Public regression | `--no-rescue-regression` | 10500 s |
| 2 | Same regression input bytes | `--rescue-regression` | 10500 s |
| 3 | Public classification | `--no-rescue-regression` | 10500 s |

All use the same clean release, frozen flat package, selected Python 3.12 environment,
offline/adaptive/legacy route and CPU lane (8 threads, CUDA hidden). Total native
caps are 8 h 45 min. They are ceilings, not promised durations. Native search may
finish earlier. The queue adds no duplicate fits or waiting merely to fill the night.
Preparation, audits, cleanup and a 30 s external hang watchdog per job are separate;
the watchdog never extends the native training allowance. Imports and observers
consume the original job deadline. Rescue ON is a request: eligibility and remaining
budget may cause the host to skip it. `audit.json` records its actual status.

## Launch

Use a clean checkout of the accepted queue commit. Keep your existing venv; do not
reinstall it when the pinned preflight passes. Download/validate both public tasks:

```powershell
& $python .\prepare_data.py all --data-dir C:\aq-public
if ($LASTEXITCODE -ne 0) { throw "Public data preparation failed" }
& .\scripts\round5_windows_overnight.ps1 -ReleaseCommit $queueCommit `
    -Python $python -RegressionTask C:\aq-public\regression\task.md `
    -ClassificationTask C:\aq-public\classification\task.md
```

`$python` must be an absolute path to the existing venv's Python, resolved before
changing directories. `$queueCommit` must be the exact clean checkout SHA. The
runner requires Windows, Python 3.12, all 15 runtime pins, at least 30 billion bytes
physical RAM, 15 GiB free disk and a new output directory outside the checkout.
The complete installed package versions are recorded; this does not certify that
all transitive packages match the platform. Keep PowerShell open and prevent
automatic sleep. Do not start another training tool concurrently.

The per-user OS lock covers preparation and the entire queue, including audits.
It also excludes other tools using `research_lab.process.machine_lock_path`.
Unrelated applications do not honor this lock. Three solver jobs never overlap.

## Completion and stopping

`queue.json` is written atomically before and after each job. Its states are
`pending`, `running`, `completed`, `failed`, `interrupted`; the overall queue is
`running`, `stopped`, or `completed`. Ctrl+C stops the queue and preserves evidence.
If Windows/power kills the coordinator, a stale `running` state is interruption,
not success. There is deliberately no automatic training resume or reuse of an
old output directory. Inspect the old evidence before explicitly starting fresh.

Required automatic checks include exact current-returned report/hash, zero API,
original deadline, root process-tree sampling and cleanup, unchanged inputs,
flat package and source identity, selected outer/refit execution resources,
fresh publication and independent receipt/context/input validation. Outer metrics
are independently recalculated from saved predictions and TRAIN labels. Ordinary
rejected research trials do not by themselves stop the queue. Missing, unknown
or failed required evidence stops it before another training job. A successful
`requires_review` result may continue when `continuation_allowed` is true.
Resource observation is sampled, not an OS sandbox or proof of unsampled orphans.

Logs: `jobs\<profile>\execution\stdout.txt` and `stderr.txt` are bounded supervisor
tails finalized after execution. `worker_evidence.json` is updated during the run.
Inspect progress from another PowerShell window:

```powershell
Get-Content C:\aq-queue-<timestamp>\queue.json -Raw
Get-Content C:\aq-queue-<timestamp>\jobs\regression_off\worker_evidence.json -Raw
```

At completion or a caught failure, `evidence.zip` contains diagnostic JSON/logs,
candidate source and predictions with an included-file SHA256 manifest. It is capped
at 256 MiB uncompressed and lists skipped files. Large input copies and models stay
in the original directory: preserve that directory for complete receipt/input review.
This diagnostic bundle alone is not a fully portable re-audit package.

`comparison.json` admits the predeclared OFF/ON pair only after both technical audits
pass and their outer split hashes match. It reports RMSE ON minus OFF. These are
known public TRAIN development scores used in final selection; they are neither
unbiased estimates nor hidden competition TEST scores. One pair establishes no
general improvement and never promotes a model/default. The already finished older
qualification run is preserved separately, not retroactively inserted into this pair.

The coordinator/worker/scripts are development files. They do not enter the flat
competition ZIP, alter native model recipes, or activate the pending OOF work.
The same coordinator is intended to become the local app's execution adapter.
