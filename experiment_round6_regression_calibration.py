"""Serial, fixed round-6 regression development; never a production router.

Only external periods 2, 3 and 4 are prepared. Held labels stay in this parent;
run_candidate receives supplied fit labels and prediction features. Original
full-TRAIN/TEST and external Year5 invocations are forbidden in this round.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import uuid
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import zipfile
import time

import numpy as np
import pandas as pd

from agent.task_spec import load_task
from experiment_rescue_regression_screen import CSV, digest, read, score_csv, write

ROOT = Path(__file__).resolve().parent
BASE_SHA = "114fc2c18595898e0f6ec099e42795eec4af0be9"
PLAN_DOCUMENT = ROOT / "docs/round6_regression_plan.md"
PLAN_RECIPES = ROOT / "docs/round6_regression_plan.json"
ARMS = ("I", "S20", "S50", "A20")
SCREEN_SECONDS, CONFIRM_SECONDS = 180, 360
TOTAL_SECONDS = 2 * 60 * 60
MEMORY_BYTES = 8 * 1024**3
DEVELOPMENT = (2, 3, 4)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def json_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_identity(require_clean=True):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    dirty = bool(git("status", "--porcelain", "--untracked-files=normal"))
    if require_clean and dirty:
        raise ValueError("Commit source and preregistered plan before measurements")
    dependencies = [*sorted((ROOT / "agent").glob("*.py")),
                    ROOT / "offline_baseline.py", ROOT / "validation.py",
                    ROOT / "experiment_rescue_regression_screen.py",
                    ROOT / "experiment_rescue_regression_diagnostics.py",
                    ROOT / "experiment_round4_regression.py", ROOT / "experiment_round5_regression.py", Path(__file__),
                    ROOT / "requirements.txt", ROOT / "requirements.platform.txt"]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in dependencies}
    return {"head": git("rev-parse", "HEAD"), "dirty": dirty,
            "dependency_hashes": hashes, "dependency_sha256": json_hash(hashes)}


def environment():
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "catboost", "lightgbm", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.system(),
            "packages": packages, "cpu_threads": 8, "device": "CPU"}


def _checked_source(frozen):
    identity = source_identity()
    if identity != frozen["source"]:
        raise ValueError("Measured source differs from prepared immutable implementation")
    return identity


def _plan_sealed():
    if not PLAN_DOCUMENT.is_file():
        raise ValueError("Preregister and commit docs/round6_regression_plan.md first")
    for path in (PLAN_DOCUMENT, PLAN_RECIPES):
        checked = subprocess.check_output(["git", "-C", str(ROOT), "show",
            "HEAD:" + str(path.relative_to(ROOT))])
        if hashlib.sha256(checked).hexdigest() != digest(path):
            raise ValueError("Preregistered plan must equal its committed bytes")


def publication_identity(published_commit):
    """Require the exact published commit object and identical clean local tree."""
    if not isinstance(published_commit, str) or len(published_commit) != 40 or any(
            value not in "0123456789abcdef" for value in published_commit):
        raise ValueError("Supply the full fetched published implementation commit SHA")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    local = source_identity()
    published_type = git("cat-file", "-t", published_commit)
    published_tree = git("rev-parse", published_commit + "^{tree}")
    local_tree = git("rev-parse", "HEAD^{tree}")
    if published_type != "commit" or published_tree != local_tree:
        raise ValueError("Published and clean measured implementation trees differ")
    return {"published_sha": published_commit, "measured_local_sha": local["head"],
            "published_tree_sha": published_tree, "measured_tree_sha": local_tree,
            "tree_equality_verified": True}


def prepare(task, output, period_column, size_column, published_commit=None):
    started = time.monotonic()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory")
    output.mkdir(parents=True)
    write(output / "attempts.json", {"version": 1, "ceiling_seconds": TOTAL_SECONDS, "attempts": []})
    ordinal = _reserve(output, kind="prepare", label="development_inputs", cap=60, started=started)
    try:
        result = _prepare(task, output, period_column, size_column, published_commit)
        record = {"version": 1, "kind": "prepare", "status": "completed", "physical_fit_count": 0,
                  "completed_fit_count": 0, "source": result["source"], "publication": result["publication"],
                  "budget_seconds": 60, "host_total_seconds": time.monotonic() - started}
        finalize_deadline(record, started=started, cap=60)
        path = output / "preparation_summary.json"
        write(path, record)
        _finish_attempt(output, ordinal, record["host_total_seconds"], record["status"], path)
        if record["status"] != "completed":
            raise TimeoutError("Initial preparation exceeded its original allowance")
        return result
    except (ValueError, OSError, TimeoutError, subprocess.SubprocessError) as error:
        if read(output / "attempts.json")["attempts"][ordinal - 1]["status"] == "running":
            _preparation_failure(output, output / "preflight" / "prepare", ordinal, started,
                                 "prepare", None, None, 60, error)
        raise


def _prepare(task, output, period_column, size_column, published_commit):
    """Write only the three allowed development folds; no Year5 artifacts."""
    _plan_sealed()
    identity = source_identity()
    output, task = Path(output).resolve(), Path(task).resolve()
    publication = publication_identity(published_commit)
    preregistered = read(PLAN_RECIPES)
    if preregistered["base_commit"] != BASE_SHA or tuple(preregistered["arms"]) != ARMS:
        raise ValueError("Preregistered base or fixed arm order differs")
    for name, expected in preregistered["input_files"].items():
        path = task.parent / name
        if not path.is_file() or digest(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError("Inputs differ from the preregistered public input hashes")
    for name, expected in preregistered["dependency_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("An unchanged preregistered dependency was modified")
    if period_column != preregistered["period_column"] or size_column != preregistered["size_column"]:
        raise ValueError("Declared development roles differ from preregistration")
    spec = load_task(task)
    if spec.metric.name != "RMSE" or period_column not in spec.numeric or size_column not in spec.numeric:
        raise ValueError("Require RMSE and explicitly declared numeric period and size features")
    train = pd.read_csv(spec.data_dir / "train.csv", **CSV)
    periods = pd.to_numeric(train[period_column], errors="raise")
    if not np.isfinite(periods.to_numpy(dtype=float)).all():
        raise ValueError("Development period values must be finite")
    relative = spec.data_dir.relative_to(task.parent)
    entries = {}
    for period in DEVELOPMENT:
        fitting, held = train.loc[periods < period].copy(), train.loc[periods == period].copy()
        if fitting.empty or held.empty:
            raise ValueError("A fixed development fold is empty")
        case = output / "inputs" / f"p{period}"
        data = case / relative
        data.mkdir(parents=True)
        shutil.copyfile(task, case / "task.md")
        shutil.copyfile(spec.data_dir / "data_dictionary.csv", data / "data_dictionary.csv")
        fitting.to_csv(data / "train.csv", index=False)
        held.drop(columns=spec.target_column).to_csv(data / "test.csv", index=False)
        pd.DataFrame({spec.id_column: held[spec.id_column], spec.target_column: "0"}).to_csv(
            data / "sample_submission.csv", index=False)
        answers = output / "host_only" / f"p{period}.csv"
        answers.parent.mkdir(exist_ok=True)
        held.loc[:, [spec.id_column, spec.target_column]].to_csv(answers, index=False)
        entries[str(period)] = {"fit_rows": len(fitting), "predict_rows": len(held),
            "fit_positions_sha256": json_hash(np.flatnonzero((periods < period).to_numpy()).tolist()),
            "held_positions_sha256": json_hash(np.flatnonzero((periods == period).to_numpy()).tolist()),
            "fit_ids_sha256": json_hash(fitting[spec.id_column].tolist()),
            "held_ids_sha256": json_hash(held[spec.id_column].tolist()),
            "files": {str(path.relative_to(output)): digest(path)
                      for path in [*sorted(case.rglob("*.csv")), case / "task.md", answers]}}
    record = {"version": 1, "round": 6, "base_sha": BASE_SHA,
        "created_at": datetime.now(timezone.utc).isoformat(), "source": identity, "publication": publication,
        "period_column": period_column, "size_column": size_column,
        "source_task": str(task), "source_files": {path.name: digest(path)
            for path in [task, *sorted(spec.data_dir.glob("*.csv"))]},
        "plan_document_sha256": digest(PLAN_DOCUMENT), "plan_recipes_sha256": digest(PLAN_RECIPES), "folds": entries,
        "screen": {"period": 4, "arms": list(ARMS), "cap_seconds": SCREEN_SECONDS},
        "confirmation": {"periods": list(DEVELOPMENT), "maximum_finalists": 2, "cap_seconds": CONFIRM_SECONDS},
        "known5_evaluation": "forbidden_in_round6", "total_heavy_seconds": TOTAL_SECONDS,
        "history": {"scope": "host_inner_fit", "max_cell_visits": 8000000,
                    "max_checked_configs": 48, "extended": False},
        "comparison": "fixed_current_rescue_control_not_full_native_scheduler",
        "scope": "known_public_development_not_blind_evaluation"}
    write(output / "plan.json", record)
    write(output / "environment.json", environment())
    return record


def plan(output):
    output = Path(output).resolve()
    value = read(output / "plan.json")
    if value.get("round") != 6 or value.get("base_sha") != BASE_SHA or set(value["folds"]) != {"2", "3", "4"}:
        raise ValueError("Not the fixed round6 development plan")
    if digest(PLAN_DOCUMENT) != value["plan_document_sha256"] or digest(PLAN_RECIPES) != value["plan_recipes_sha256"]:
        raise ValueError("Preregistered plan changed")
    for entry in value["folds"].values():
        for name, expected in entry["files"].items():
            if digest(output / name) != expected:
                raise ValueError("Frozen development input changed")
    return value


@contextmanager
def _ledger_guard(output):
    """Persistent OS advisory lock; its pathname is never deleted or recovered."""
    path = Path(output) / "queue.guard"
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    acquired = False
    try:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        yield
    finally:
        if acquired:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _used_seconds(ledger):
    total = 0.0
    for row in ledger["attempts"]:
        elapsed = row.get("elapsed_seconds")
        value = row["cap_seconds"] if elapsed is None else elapsed
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid consumed time in original serial ledger")
        total += value
    return total


def _reserve(output, *, kind, label, cap, retry_reason=None, started=None):
    path = Path(output) / "attempts.json"
    if type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0:
        raise ValueError("Reservation requires a positive finite original cap")
    with _ledger_guard(output):
        ledger = read(path)
        if any(row["status"] == "running" for row in ledger["attempts"]):
            raise ValueError("An unfinished attempt still owns the original computation queue")
        if _used_seconds(ledger) + cap > TOTAL_SECONDS:
            raise ValueError("Original two-hour heavy-computation allowance exhausted")
        if retry_reason is not None:
            if not isinstance(retry_reason, str) or not 10 <= len(retry_reason) <= 500:
                raise ValueError("Technical retry requires a bounded explicit explanation")
            if any(row.get("retry_reason") is not None for row in ledger["attempts"]):
                raise ValueError("Only one documented technical retry is permitted for the series")
        row = {"ordinal": len(ledger["attempts"]) + 1, "kind": kind, "label": label,
               "cap_seconds": cap, "started_at": datetime.now(timezone.utc).isoformat(),
               "started_monotonic": time.monotonic() if started is None else started,
               "status": "running", "elapsed_seconds": None, "retry_reason": retry_reason}
        ledger["attempts"].append(row)
        atomic(path, ledger)
        return row["ordinal"]


def _finish_attempt(output, ordinal, elapsed, status, summary):
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Attempt elapsed time must be observed, finite and nonnegative")
    path = Path(output) / "attempts.json"
    with _ledger_guard(output):
        ledger = read(path)
        row = ledger["attempts"][ordinal - 1]
        if row["ordinal"] != ordinal or row["status"] != "running":
            raise ValueError("Attempt ledger identity mismatch")
        row.update(status=status, elapsed_seconds=elapsed,
                   summary=str(Path(summary).relative_to(output)), summary_sha256=digest(summary))
        atomic(path, ledger)


def record_preflight_failure(output, *, stage, period, arm, started, cap, error):
    """A blocked reservation has zero fits and an immutable, separately typed row."""
    output = Path(output).resolve()
    if not (output / "attempts.json").is_file():
        raise error
    elapsed = time.monotonic() - started
    directory = output / "preflight" / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    record = {"version": 1, "round": 6, "stage": stage, "period": period, "arm": arm,
        "kind": "preflight", "status": "preflight_failed", "budget_seconds": cap,
        "host_total_seconds": elapsed, "physical_fit_count": 0, "completed_fit_count": 0,
        "physical_fits": [], "metrics": None, "quality_metric": None,
        "stop_reason": "blocked_before_model_invocation", "error_type": type(error).__name__, "api_calls": 0}
    write(directory / "summary.json", record)
    (directory / "local_failure.txt").write_text(str(error), encoding="utf-8")
    # Concurrent active invocations remain protected by their existing reservation.
    # The short advisory lock serializes this zero-fit row with completion updates.
    with _ledger_guard(output):
        ledger = read(output / "attempts.json")
        row = {"ordinal": len(ledger["attempts"]) + 1, "kind": "preflight",
            "label": f"{stage}/p{period}/{arm}", "cap_seconds": cap,
            "started_at": datetime.now(timezone.utc).isoformat(), "started_monotonic": started,
            "status": "preflight_failed", "elapsed_seconds": elapsed, "retry_reason": None,
            "summary": str((directory / "summary.json").relative_to(output)),
            "summary_sha256": digest(directory / "summary.json")}
        ledger["attempts"].append(row)
        atomic(output / "attempts.json", ledger)
    return record


def finalize_deadline(record, *, started, cap, now=None):
    """Last writer guard after diagnostics/identity checks and before success publication."""
    end = time.monotonic() if now is None else now
    elapsed = end - started
    record["host_total_seconds"] = elapsed
    record["original_start_monotonic"] = started
    record["original_deadline_monotonic"] = started + cap
    record["deadline_checked_monotonic"] = end
    finite = type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0
    record["deadline_admissible"] = finite and elapsed <= cap
    if record.get("status") == "completed" and not record["deadline_admissible"]:
        record.update(status="failed", stop_reason="host_deadline_after_diagnostics",
                      error_type="TimeoutError", metrics=None, quality_metric=None)
    return record


def admit_completed(record, expected_cap=None):
    """Reader/auditor independently rejects forged or legacy over-cap completed rows."""
    if record.get("status") != "completed":
        raise ValueError("Only a completed attempt can be admitted")
    cap, elapsed = record.get("budget_seconds"), record.get("host_total_seconds")
    if (type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0 or
            type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0 or elapsed > cap):
        raise ValueError("Completed attempt exceeds its finite original deadline")
    if expected_cap is not None and cap != expected_cap:
        raise ValueError("Completed attempt uses a different preregistered cap")
    if record.get("deadline_admissible") is not True:
        raise ValueError("Completed attempt lacks its final post-diagnostics deadline check")
    if record.get("kind") == "real_flat_package_toy_smoke_not_quality_comparison":
        smoke_elapsed = record.get("smoke_elapsed_seconds")
        if (type(smoke_elapsed) not in (int, float) or not math.isfinite(smoke_elapsed) or
                smoke_elapsed < 0 or smoke_elapsed > cap or record.get("evidence_binding_verified") is not True):
            raise ValueError("Smoke receipt or frozen archive/event binding is inadmissible")
    if "physical_fit_count" in record:
        began, completed = record["physical_fit_count"], record.get("completed_fit_count")
        if type(began) is not int or began < 0 or type(completed) is not int or completed != began:
            raise ValueError("Completed physical fit counts do not reconcile")
        if record.get("physical_fits") is not None and len(record["physical_fits"]) != began:
            raise ValueError("Physical fit event ledger differs from its count")
    if "actual_analytic_fit_count" in record:
        began, completed = record["actual_analytic_fit_count"], record.get("completed_analytic_fit_count")
        if type(began) is not int or began < 0 or type(completed) is not int or completed != began:
            raise ValueError("Completed analytic fit counts do not reconcile")
    if "original_start_monotonic" in record:
        start, deadline, checked = (record[key] for key in
            ("original_start_monotonic", "original_deadline_monotonic", "deadline_checked_monotonic"))
        if not all(type(value) in (int, float) and math.isfinite(value) for value in (start, deadline, checked)):
            raise ValueError("Nonfinite monotonic deadline evidence")
        if not math.isclose(deadline - start, cap, rel_tol=0, abs_tol=1e-6) or not math.isclose(
                checked - start, elapsed, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Deadline timestamps do not match recorded total")
    return True


def case_frames(output, period):
    if period not in DEVELOPMENT:
        raise ValueError("Only development periods 2, 3 and 4 are permitted")
    output = Path(output)
    spec = load_task(output / "inputs" / f"p{period}" / "task.md")
    fit = pd.read_csv(spec.data_dir / "train.csv", **CSV)
    held = pd.read_csv(spec.data_dir / "test.csv", **CSV)
    answers = pd.read_csv(output / "host_only" / f"p{period}.csv", **CSV)
    if held[spec.id_column].tolist() != answers[spec.id_column].tolist():
        raise ValueError("Frozen answer positions differ from prediction input")
    return spec, fit, held, answers


def discover(output, period):
    started = time.monotonic()
    output = Path(output).resolve()
    if period not in DEVELOPMENT:
        raise ValueError("Only development periods 2, 3 and 4 are permitted")
    directory = output / "discovery" / f"p{period}"
    try:
        if directory.exists():
            raise ValueError("History discovery occurs once per transition")
        ordinal = _reserve(output, kind="discovery", label=f"p{period}", cap=8, started=started)
    except (ValueError, OSError) as error:
        return record_preflight_failure(output, stage="discovery", period=period, arm=None,
                                        started=started, cap=8, error=error)
    try:
        from agent.regression_discovery import discover_regression_schema_bounded
        frozen = plan(output)
        identity = _checked_source(frozen)
        spec, fit, _, _ = case_frames(output, period)
        values = pd.to_numeric(fit[frozen["period_column"]], errors="raise")
        mask = values < values.max()
        inner = fit.loc[mask, list(spec.features)].copy()
        report = discover_regression_schema_bounded(inner, feature_columns=list(spec.features),
            numeric_columns=list(spec.numeric), period_column=frozen["period_column"],
            output_dir=directory, deadline=started + 8, extended=False)
        record = {"version": 1, "period": period, "period_column": frozen["period_column"],
            "source": identity, "outer_fit_rows": len(fit), "host_inner_fit_rows": len(inner),
            "host_inner_fit_ids_sha256": json_hash(fit.loc[mask, spec.id_column].tolist()),
            "fit_input_sha256": report["fit_input_sha256"],
            "discovery_report_sha256": digest(directory / "discovery_report.json"),
            "selected_config": report["selected_config"], "status": "completed",
            "discovery_status": report["status"], "budget_seconds": 8,
            "physical_fit_count": 0, "completed_fit_count": 0,
            "reason": report["reason"], "max_cell_visits": 8000000, "max_checked_configs": 48,
            "history_assessment": "not_assessable" if report["selected_config"] is None else "frozen_schema_available",
            "nesting": "outer development t; host INNER-fit excludes latest supplied period; candidate calibration independently inside supplied fit"}
        if source_identity() != identity:
            raise ValueError("Source changed during discovery")
        finalize_deadline(record, started=started, cap=8)
        record["elapsed_seconds"] = record["host_total_seconds"]
        write(directory / "host_schema.json", record)
        _finish_attempt(output, ordinal, record["elapsed_seconds"], record["status"], directory / "host_schema.json")
        return record
    except (ValueError, OSError, TimeoutError) as error:
        return _preparation_failure(output, directory, ordinal, started, "discovery", period, None, 8, error)


def schema(output, period):
    if period not in DEVELOPMENT:
        raise ValueError("No Year5 discovery is available")
    directory = Path(output) / "discovery" / f"p{period}"
    record = read(directory / "host_schema.json")
    admit_completed(record, expected_cap=8)
    _ledger_summary(output, directory / "host_schema.json")
    frozen = plan(output)
    if record["source"] != frozen["source"]:
        raise ValueError("Schema discovery source differs from the prepared implementation")
    report = read(directory / "discovery_report.json")
    if digest(directory / "discovery_report.json") != record["discovery_report_sha256"]:
        raise ValueError("Frozen discovery report changed")
    if report["selected_config"] != record["selected_config"] or report["max_cell_visits"] != 8000000 or report["max_checked_configs"] != 48:
        raise ValueError("Only unchanged original 8M/48 discovery may be used")
    if report["fit_input_sha256"] is not None and digest(directory / "fit_features.csv") != report["fit_input_sha256"]:
        raise ValueError("Frozen discovery features changed")
    return record


def instrument_source(original):
    """Same actual-fit observer for incumbent and new arms; persists before fit."""
    return original + r'''

_round6_original_fit_predict = fit_predict

def fit_predict(X_fit, y_fit, X_predict, context):
    import json,time
    from pathlib import Path
    from catboost import CatBoostRegressor
    from agent.features import FeatureTransformer
    ledger,analytic,events=[],[],[]
    path=Path("physical_fits.json")
    started=time.monotonic()
    def component():
        try:
            from agent.round6_regression_candidate import active_component
            value=active_component()
            return dict(value) if value else {"id":"incumbent","engine":"catboost","feature_subset":"all_declared"}
        except ImportError:
            return {"id":"incumbent","engine":"catboost","feature_subset":"all_declared"}
    def persist():
        temporary=path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"version":1,"original_deadline_monotonic":context["deadline_monotonic"],
            "fits":ledger,"analytic_fits":analytic,"stages":events},allow_nan=False),encoding="utf-8")
        temporary.replace(path)
    def observe(original,engine,kind):
        def wrapped(model,*args,**kwargs):
            X=args[0] if args else kwargs.get("X",kwargs.get("data"))
            count=X.shape[0] if hasattr(X,"shape") else (X.num_row() if hasattr(X,"num_row") else None)
            info=component()
            phase=info.get("phase") or (("full_refit" if count==len(X_fit) else "internal_selection") if kind=="model_fit" else "incumbent_combined_stage")
            item={"component":info,"engine":engine,"phase":phase,"kind":kind,
                "rows":int(count) if count is not None else None,"started":True,"completed":False,
                "deadline_monotonic":info.get("deadline_monotonic",context["deadline_monotonic"]),
                "started_monotonic":time.monotonic(),
                "start_offset_seconds":time.monotonic()-started,"elapsed_seconds":None,"exception_type":None}
            target=ledger if kind=="model_fit" else events
            target.append(item);persist();begin=time.monotonic()
            try:
                result=original(model,*args,**kwargs)
                item["completed"]=True
                return result
            except BaseException as error:
                item["exception_type"]=type(error).__name__
                raise
            finally:
                item["elapsed_seconds"]=time.monotonic()-begin;persist()
        return wrapped
    methods=[(CatBoostRegressor,"fit","catboost","model_fit"),
             (CatBoostRegressor,"predict","catboost","prediction"),
             (FeatureTransformer,"fit_transform","preprocessing","fit_transform"),
             (FeatureTransformer,"transform","preprocessing","transform")]
    originals=[];persist()
    for cls,name,engine,kind in methods:
        previous=getattr(cls,name);originals.append((cls,name,previous))
        setattr(cls,name,observe(previous,engine,kind))
    from agent import round6_regression_candidate as candidate_module
    analytic_original=candidate_module.fit_correction
    def analytic_fit(arm,y,p):
        begin=time.monotonic();info=component()
        item={"component":info,"engine":"analytic_residual","phase":"correction_fit_A",
            "kind":"analytic_fit","rows":len(y),"started":True,"completed":False,
            "deadline_monotonic":info.get("deadline_monotonic",context["deadline_monotonic"]),
            "started_monotonic":begin,"start_offset_seconds":begin-started,
            "elapsed_seconds":None,"exception_type":None}
        analytic.append(item);persist()
        try:
            result=analytic_original(arm,y,p);item["completed"]=True;return result
        except BaseException as error:
            item["exception_type"]=type(error).__name__;raise
        finally:
            item["elapsed_seconds"]=time.monotonic()-begin;persist()
    candidate_module.fit_correction=analytic_fit
    try:
        return _round6_original_fit_predict(X_fit,y_fit,X_predict,context)
    finally:
        candidate_module.fit_correction=analytic_original
        for cls,name,previous in originals:setattr(cls,name,previous)
        persist()
'''


def diagnostics(fit, held, answers, predictions, spec, frozen, config):
    from experiment_round4_regression import diagnostics as existing_diagnostics
    result = existing_diagnostics(fit, held, answers, predictions, spec, frozen, config)
    # The legacy generic diagnostic invents absent past rows without a schema.
    # Those proxies cannot be evidence of actual cold/history status.
    if config is None:
        result = [row for row in result if row["partition"] not in {"past_rows", "recency"}]
    return result


def _execution_public(execution):
    if execution is None:
        return None
    fields = ("exit_code", "timed_out", "memory_limit_exceeded", "host_deadline_exceeded",
              "elapsed_seconds", "host_elapsed_seconds", "resources")
    result = {key: execution.get(key) for key in fields}
    for stream in ("stdout", "stderr"):
        result[stream + "_sha256"] = text_hash(execution.get(stream, ""))
    return result


def _recipe(arm, period_column, config):
    if arm == "I":
        from agent.regression_candidate import candidate_source, make_regression_recipe
        return candidate_source(), make_regression_recipe(period_column, history_config=config)
    from agent.round6_regression_candidate import candidate_source, make_round6_recipe
    return candidate_source(), make_round6_recipe(period_column, arm, history_config=config)


def _run(output, directory, fit, held, spec, frozen, recipe, original, cap, *, stage, period=None, arm=None, answers=None, retry_reason=None, started=None, ordinal=None):
    from agent.research_candidate import CandidateExecutionError, freeze_candidate, run_candidate
    started = time.monotonic() if started is None else started
    identity = _checked_source(frozen)
    if cap not in (SCREEN_SECONDS, CONFIRM_SECONDS):
        raise ValueError("Use only the fixed 180/360-second invocation caps")
    directory.mkdir(parents=True, exist_ok=False)
    if ordinal is None:
        ordinal = _reserve(output, kind=stage, label=f"p{period}/{arm}", cap=cap, retry_reason=retry_reason)
    expires = started + cap
    record = {"version": 1, "round": 6, "status": "running", "stage": stage,
        "period": period, "arm": arm, "source": identity, "base_sha": BASE_SHA,
        "budget_seconds": cap, "original_deadline_monotonic": expires,
        "started_at": datetime.now(timezone.utc).isoformat(), "recipe": recipe,
        "original_source_sha256": text_hash(original), "fit_rows": len(fit), "predict_rows": len(held),
        "physical_fit_count": None, "completed_fit_count": None, "metrics": None,
        "quality_scope": "known_development" if answers is not None else "unlabelled_test_contract_only",
        "environment": environment(), "api_calls": 0,
        "history_gate": "unknown_schema_unavailable" if recipe["history_config"] is None else "assessable",
        "split": frozen["folds"].get(str(period)) if period is not None else None,
        "input_hashes": frozen["source_files"], "retry_reason": retry_reason}
    path = directory / "source.py"
    path.write_text(instrument_source(original), encoding="utf-8")
    (directory / "original_source.py").write_text(original, encoding="utf-8")
    candidate = freeze_candidate(path, recipe, directory / "frozen")
    record.update(code_sha256=candidate.code_hash, recipe_sha256=candidate.recipe_hash,
                  dependency_sha256=identity["dependency_sha256"])
    write(directory / "attempt_started.json", record)
    execution = None
    saved_env = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    os.environ.update({key: "8" for key in saved_env})
    try:
        result = run_candidate(candidate, spec=spec, fit_frame=fit, predict_frame=held,
            output_dir=directory, phase="full_refit", timeout=cap, deadline=expires,
            max_memory_bytes=MEMORY_BYTES)
        result.verify()
        execution = result.execution
        candidate_host_completed_seconds = time.monotonic() - started
        record["candidate_host_completed_seconds"] = candidate_host_completed_seconds
        record["host_preparation_seconds"] = candidate_host_completed_seconds - execution.get("host_elapsed_seconds", 0)
        if time.monotonic() >= expires:
            raise TimeoutError("Original invocation deadline exhausted before host receipt verification")
        scoring_started = time.monotonic()
        result.copy_output(directory / "submission.csv")
        work = result.output_path.parent
        evidence = read(work / "regression_evidence.json")
        ablation_path = work / "regression_ablation.csv"
        if digest(ablation_path) != evidence["ablation_sha256"]:
            raise ValueError("Candidate ablation changed after prediction")
        shutil.copyfile(work / "regression_evidence.json", directory / "regression_evidence.json")
        shutil.copyfile(ablation_path, directory / "regression_ablation.csv")
        alternatives = pd.read_csv(ablation_path, **CSV)
        if len(alternatives) != len(held) or not {"base", "mean", "calibrated"} <= set(alternatives):
            raise ValueError("Candidate ablation contract differs")
        if arm == "I":
            alternatives["identity"] = alternatives["base"]
        if "identity" not in alternatives:
            raise ValueError("Correction candidate must retain its aligned identity ablation")
        if any(name not in {"identity", "base", "mean", "calibrated", "corrected"} for name in alternatives):
            raise ValueError("Unexpected ablation column")
        predictions, scores = {}, {}
        for name in alternatives:
            values = pd.to_numeric(alternatives[name], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite ablation predictions")
            frame = spec.prediction_frame(held[spec.id_column].tolist(), values)
            target = directory / (name + ".csv")
            frame.to_csv(target, index=False)
            predictions[name] = values
            if answers is not None:
                scores[name] = score_csv(spec.task_path, answers, target)
        public = pd.read_csv(directory / "submission.csv", **CSV)
        selected = "base" if recipe["history_policy"] == "off" else recipe["history_policy"]
        if not np.array_equal(pd.to_numeric(public[spec.target_column]), predictions[selected]):
            raise ValueError("Published predictions differ from the frozen history policy")
        if answers is not None:
            answer_frame = pd.read_csv(answers, **CSV)
            segments = diagnostics(fit, held, answer_frame, predictions, spec, frozen, recipe["history_config"])
            write(directory / "segments.json", {"rows": segments,
                "history_assessment": record["history_gate"],
                "note": "Partitions overlap; never sum shares across partitions. Missing schema is not cold membership."})
            record["segments_sha256"] = digest(directory / "segments.json")
        shutil.copyfile(result.receipt_path, directory / "receipt.json")
        record.update(status="completed", evidence=evidence, metrics=scores if answers is not None else None,
            receipt_sha256=result.receipt_hash, output_sha256=result.output_hash, ids_exact_order=True, finite=True,
            quality_metric=None if answers is None else scores[selected]["rmse"],
            selected_ablation=selected, native_admission=False,
            post_prediction_diagnostics_seconds=time.monotonic() - scoring_started)
        correction = evidence.get("correction", {})
        record["correction_status"] = "incumbent_control" if arm == "I" else correction.get("status", "unknown")
        record["correction_reason"] = "unchanged_control" if arm == "I" else correction.get("reason", "unknown")
    except CandidateExecutionError as error:
        execution = error.execution
        record.update(status="failed", error_type=type(error).__name__,
                      stop_reason="candidate_process_failed", quality_metric=None)
        (directory / "local_failure.txt").write_text(str(error), encoding="utf-8")
    except (ValueError, OSError, TimeoutError) as error:
        record.update(status="failed", error_type=type(error).__name__,
                      stop_reason="candidate_contract_or_deadline_failure", quality_metric=None)
        (directory / "local_failure.txt").write_text(str(error), encoding="utf-8")
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    ledgers = sorted(directory.glob("full_refit_*/physical_fits.json"))
    if len(ledgers) == 1:
        ledger = read(ledgers[0])
        shutil.copyfile(ledgers[0], directory / "physical_fits.json")
        record.update(physical_fits=ledger["fits"], analytic_fits=ledger.get("analytic_fits", []), stages=ledger["stages"],
            physical_fit_count=len(ledger["fits"]), completed_fit_count=sum(row["completed"] for row in ledger["fits"]),
            physical_engine_fit_count=len(ledger["fits"]),
            actual_analytic_fit_count=len(ledger.get("analytic_fits", [])),
            completed_analytic_fit_count=sum(row["completed"] for row in ledger.get("analytic_fits", [])),
            total_fit_operations=len(ledger["fits"]) + len(ledger.get("analytic_fits", [])),
            physical_fits_sha256=digest(ledgers[0]), physical_fits_reason=None,
            timing_note="Phase/stage timers can be nested; do not sum. host_total_seconds is authoritative wall time")
    else:
        record.update(physical_fits=None, stages=None, physical_fits_reason="observer_unavailable_before_model_import_or_worker_termination")
    record.update(execution=_execution_public(execution), completed_at=datetime.now(timezone.utc).isoformat())
    try:
        final_identity = source_identity()
    except (ValueError, OSError, subprocess.SubprocessError):
        final_identity = None
    if final_identity != identity:
        record.update(status="failed", stop_reason="source_changed_during_measurement", metrics=None, quality_metric=None)
    if record["status"] == "completed" and (record.get("physical_fit_count") is None or
            record["physical_fit_count"] < 1 or record["completed_fit_count"] != record["physical_fit_count"]):
        record.update(status="failed", stop_reason="actual_fit_observer_incomplete", metrics=None, quality_metric=None)
    # Must be the last potentially expensive computation before success publication.
    # Diagnostics, receipts, physical ledgers and source hashes have all finished.
    finalize_deadline(record, started=started, cap=cap)
    write(directory / "summary.json", record)
    _finish_attempt(output, ordinal, record["host_total_seconds"], record["status"], directory / "summary.json")
    return record


def _run_path(output, stage, period, arm, retry=False):
    return Path(output) / "runs" / stage / f"p{period}" / (arm + ("_technical_retry" if retry else ""))


def _ledger_summary(output, path):
    output, path = Path(output), Path(path)
    ledger = read(output / "attempts.json")
    relative = str(path.relative_to(output))
    matches = [row for row in ledger["attempts"] if row.get("summary") == relative]
    if len(matches) != 1 or digest(path) != matches[0]["summary_sha256"]:
        raise ValueError("Attempt summary does not match its recorded immutable digest")
    return matches[0]


def _record(output, stage, period, arm):
    output = Path(output)
    retry = _run_path(output, stage, period, arm, True) / "summary.json"
    path = retry if retry.is_file() else _run_path(output, stage, period, arm) / "summary.json"
    record = read(path)
    ledger_row = _ledger_summary(output, path)
    if ledger_row["status"] != record["status"]:
        raise ValueError("Attempt summary/ledger terminal status differs")
    frozen = plan(output)
    if record["status"] == "completed":
        admit_completed(record, expected_cap=SCREEN_SECONDS if stage == "screen" else CONFIRM_SECONDS)
        if record["source"] != frozen["source"] or record["input_hashes"] != frozen["source_files"] or record["split"] != frozen["folds"][str(period)]:
            raise ValueError("Paired attempt source/input/split identity differs")
        if record["arm"] != arm or record["stage"] != stage or record["period"] != period:
            raise ValueError("Attempt label does not match its immutable location")
        expected_cap = SCREEN_SECONDS if stage == "screen" else CONFIRM_SECONDS
        if record["budget_seconds"] != expected_cap:
            raise ValueError("Paired attempt has a different cap")
        config = schema(output, period)["selected_config"]
        original, recipe = _recipe(arm, frozen["period_column"], config)
        if (record["recipe"] != recipe or record["recipe_sha256"] != json_hash(recipe)
                or record["original_source_sha256"] != text_hash(original)
                or record["code_sha256"] != text_hash(instrument_source(original))):
            raise ValueError("Attempt frozen algorithm differs from the declared arm")
        for name, field in (("segments.json", "segments_sha256"), ("receipt.json", "receipt_sha256"),
                            ("physical_fits.json", "physical_fits_sha256"), ("submission.csv", "output_sha256")):
            if digest(path.parent / name) != record[field]:
                raise ValueError("Attempt evidence changed after measurement")
        fits = read(path.parent / "physical_fits.json")["fits"]
        if (len(fits) != record["physical_fit_count"] or not fits or
                sum(row["completed"] for row in fits) != record["completed_fit_count"] or
                not all(row["started"] and row["completed"] for row in fits)):
            raise ValueError("Actual started/completed physical fits do not reconcile")
        analytic = read(path.parent / "physical_fits.json").get("analytic_fits", [])
        if (len(analytic) != record["actual_analytic_fit_count"] or
                sum(row["completed"] for row in analytic) != record["completed_analytic_fit_count"] or
                not all(row["started"] and row["completed"] for row in analytic) or
                record["total_fit_operations"] != len(fits) + len(analytic)):
            raise ValueError("Actual analytic correction fits do not reconcile")
    return record, path


def _preparation_failure(output, directory, ordinal, started, stage, period, arm, cap, error):
    directory.mkdir(parents=True, exist_ok=True)
    ledgers = sorted(directory.glob("full_refit_*/physical_fits.json"))
    observed = read(ledgers[0]) if len(ledgers) == 1 else None
    fits = observed["fits"] if observed else []
    analytic = observed.get("analytic_fits", []) if observed else []
    status = "failed" if fits else "preflight_failed"
    record = {"version": 1, "round": 6, "status": status, "stage": stage,
        "period": period, "arm": arm, "budget_seconds": cap,
        "original_deadline_monotonic": started + cap,
        "host_total_seconds": time.monotonic() - started, "metrics": None,
        "quality_metric": None, "physical_fit_count": len(fits), "completed_fit_count": sum(row["completed"] for row in fits),
        "physical_fits": fits, "physical_fits_reason": None if observed else "no_observed_model_start_before_preflight_failure",
        "analytic_fits": analytic, "actual_analytic_fit_count": len(analytic),
        "completed_analytic_fit_count": sum(row["completed"] for row in analytic),
        "total_fit_operations": len(fits) + len(analytic),
        "stop_reason": "invocation_preparation_failed", "error_type": type(error).__name__, "api_calls": 0}
    (directory / "local_failure.txt").write_text(str(error), encoding="utf-8")
    if not (directory / "summary.json").exists():
        write(directory / "summary.json", record)
        _finish_attempt(output, ordinal, record["host_total_seconds"], status, directory / "summary.json")
    return record


def run(output, stage, period, arm, retry_reason=None):
    started = time.monotonic()
    output = Path(output).resolve()
    if arm not in ARMS or stage not in {"screen", "confirmation"} or period not in DEVELOPMENT:
        raise ValueError("Unknown preregistered stage/period/arm")
    if stage == "screen" and period != 4:
        raise ValueError("Screen uses only external period4")
    directory = _run_path(output, stage, period, arm, retry_reason is not None)
    cap = SCREEN_SECONDS if stage == "screen" else CONFIRM_SECONDS
    try:
        if directory.exists():
            raise ValueError("Attempt directory already exists; preserve it unchanged")
        ordinal = _reserve(output, kind=stage, label=f"p{period}/{arm}", cap=cap,
                           retry_reason=retry_reason, started=started)
    except (ValueError, OSError) as error:
        return record_preflight_failure(output, stage=stage, period=period, arm=arm,
                                        started=started, cap=cap, error=error)
    try:
        frozen = plan(output)
        _checked_source(frozen)
        if stage == "confirmation":
            promotion = _checked_promotion(output)
            if arm not in {"I", *promotion["finalists"]}:
                raise ValueError("Confirmation is restricted to incumbent and frozen finalists")
            if arm != "I" and sum(_fold_failure(output, earlier, arm) for earlier in DEVELOPMENT if earlier < period) >= 2:
                raise ValueError("Finalist stopped after two material distinct-period failures")
        discovered = schema(output, period)
        spec, fit, held, _ = case_frames(output, period)
        original, recipe = _recipe(arm, frozen["period_column"], discovered["selected_config"])
        if recipe["history_config"] != discovered["selected_config"]:
            raise ValueError("Recipe lost the one frozen host INNER-fit history configuration")
        if retry_reason is not None:
            previous = read(_run_path(output, stage, period, arm) / "summary.json")
            if previous["status"] != "failed" or previous["recipe"] != recipe or previous["original_source_sha256"] != text_hash(original):
                raise ValueError("Retry requires a failed identical original recipe/source")
        return _run(output, directory, fit, held, spec, frozen, recipe, original, cap,
            stage=stage, period=period, arm=arm, answers=output / "host_only" / f"p{period}.csv",
            retry_reason=retry_reason, started=started, ordinal=ordinal)
    except (ValueError, OSError, TimeoutError, subprocess.SubprocessError) as error:
        return _preparation_failure(output, directory, ordinal, started, stage, period, arm, cap, error)


def _selected_metrics(record):
    return record["metrics"][record.get("selected_ablation", "calibrated")]


def promote(output, *, persist=True):
    output = Path(output).resolve()
    _checked_source(plan(output))
    control, _ = _record(output, "screen", 4, "I")
    if control["status"] != "completed":
        raise ValueError("A completed incumbent screen is required")
    reference = _selected_metrics(control)["rmse"]
    candidates, considered = [], []
    for arm in ARMS[1:]:
        record, path = _record(output, "screen", 4, arm)
        item = {"arm": arm, "summary_sha256": digest(path), "status": record["status"], "relative_rmse_change": None}
        if record["status"] == "completed":
            relative = _selected_metrics(record)["rmse"] / reference - 1
            item["relative_rmse_change"] = relative
            if relative <= -.002:
                candidates.append((arm, record))
        considered.append(item)
    # Greedy fixed ranking; practically tied RMSE favors fewer physical fits,
    # then the predeclared arm order. No data-dependent new mixtures appear.
    finalists = []
    while candidates and len(finalists) < 2:
        best = min(_selected_metrics(record)["rmse"] for _, record in candidates)
        tied = [(arm, record) for arm, record in candidates if _selected_metrics(record)["rmse"] < best * 1.002]
        chosen = min(tied, key=lambda item: (item[1]["physical_fit_count"], ARMS.index(item[0])))
        finalists.append(chosen[0])
        candidates = [item for item in candidates if item[0] != chosen[0]]
    result = {"version": 1, "source": source_identity(), "finalists": finalists, "screen": considered,
        "incumbent_summary_sha256": digest(_record(output, "screen", 4, "I")[1]),
        "status": "promoted" if finalists else "retain_incumbent_no_screen_improvement",
        "rule": "at_least_0.2pct_RMSE_gain; top_two; practical_ties_fewer_fits_then_fixed_arm_order"}
    if persist:
        path = output / "promotion.json"
        if path.exists() and read(path) != result:
            raise ValueError("Preserve an existing differing promotion; do not rewrite evidence")
        if not path.exists():
            write(path, result)
    return result


def _checked_promotion(output):
    stored = read(Path(output) / "promotion.json")
    expected = promote(output, persist=False)
    if stored != expected:
        raise ValueError("Promotion differs from the frozen ranking and measured screen hashes")
    return stored


def _segments_gate(control, candidate, control_path, candidate_path):
    before = read(control_path.parent / "segments.json")["rows"]
    after = read(candidate_path.parent / "segments.json")["rows"]
    before = {(row["partition"], row["segment"]): row for row in before if row["arm"] == control["selected_ablation"]}
    after = {(row["partition"], row["segment"]): row for row in after if row["arm"] == candidate["selected_ablation"]}
    total = _selected_metrics(control)["sse"]
    minimum_rows = max(100, math.ceil(.01 * _selected_metrics(control)["rows"]))
    failures = []
    for key, earlier in before.items():
        current = after[key]
        if current["n"] != earlier["n"]:
            raise ValueError("Paired segment membership changed")
        meaningful = earlier["n"] >= minimum_rows and earlier["sse"] >= .01 * total
        if not meaningful or not earlier["sse"]:
            continue
        delta = current["sse"] - earlier["sse"]
        relative = delta / earlier["sse"]
        if (key[0] == "history" and key[1] != "schema_unavailable" and relative > .10) or (
                (key[0] == "category_novelty" or key[0].startswith("category:")) and relative > .10 and delta > .01 * total):
            failures.append({"partition": key[0], "segment": key[1], "rows": earlier["n"],
                             "control_sse": earlier["sse"], "candidate_sse": current["sse"],
                             "relative_sse_change": relative, "absolute_sse_change": delta})
    return {"status": "failed" if failures else "passed_observed_segments", "failures": failures,
            "cold_history_gate": "unknown" if candidate["recipe"]["history_config"] is None else
                ("failed" if any(row["partition"] == "history" for row in failures) else "passed"),
            "meaningful_rows_minimum": minimum_rows, "meaningful_control_sse_fraction": .01}


def _fold_failure(output, period, arm):
    try:
        candidate, path = _record(output, "confirmation", period, arm)
        control, control_path = _record(output, "confirmation", period, "I")
    except FileNotFoundError:
        return 0
    if candidate["status"] != "completed" or control["status"] != "completed":
        return 1
    relative = _selected_metrics(candidate)["rmse"] / _selected_metrics(control)["rmse"] - 1
    return int(relative >= .002 or _segments_gate(control, candidate, control_path, path)["status"] == "failed")


def evaluate(output):
    output = Path(output).resolve()
    frozen = plan(output)
    _checked_source(frozen)
    promotion = _checked_promotion(output)
    findings = []
    for arm in promotion["finalists"]:
        folds, wins, total_control, total_candidate = [], 0, 0., 0.
        complete, fold_cap_ok, segments_ok = True, True, True
        configs = []
        for period in DEVELOPMENT:
            try:
                control, control_path = _record(output, "confirmation", period, "I")
                candidate, path = _record(output, "confirmation", period, arm)
            except FileNotFoundError:
                complete = False
                folds.append({"period": period, "status": "not_executed", "gates": "unknown"})
                continue
            if candidate["status"] != "completed" or control["status"] != "completed":
                complete = False
                folds.append({"period": period, "status": "failed", "gates": "unknown",
                    "control_summary_sha256": digest(control_path), "candidate_summary_sha256": digest(path)})
                continue
            if candidate["budget_seconds"] != CONFIRM_SECONDS or control["budget_seconds"] != CONFIRM_SECONDS:
                raise ValueError("Confirmation cannot mix 180-second and 360-second evidence")
            a, b = _selected_metrics(control), _selected_metrics(candidate)
            relative = b["rmse"] / a["rmse"] - 1
            wins += relative <= -.002
            total_control += a["sse"]
            total_candidate += b["sse"]
            fold_cap_ok &= relative <= .002
            segments = _segments_gate(control, candidate, control_path, path)
            segments_ok &= segments["status"] != "failed"
            configs.append(candidate["recipe"]["history_config"])
            folds.append({"period": period, "status": "completed", "control": a, "candidate": b,
                "relative_rmse_change": relative, "segments": segments,
                "control_summary_sha256": digest(control_path), "candidate_summary_sha256": digest(path)})
        same_config = len(configs) == 3 and all(value == configs[0] for value in configs)
        pooled_change = math.sqrt(total_candidate / total_control) - 1 if complete and total_control > 0 else None
        passed = complete and wins >= 2 and pooled_change is not None and pooled_change <= -.002 and fold_cap_ok and segments_ok and same_config
        findings.append({"arm": arm, "folds": folds, "complete": complete, "practical_wins": wins,
            "control_sse": total_control if complete else None, "candidate_sse": total_candidate if complete else None,
            "pooled_relative_rmse_change": pooled_change,
            "complexity_rank": ARMS.index(arm),
            "every_fold_within_0_2pct": fold_cap_ok if complete else None,
            "significant_segment_gate": segments_ok if complete else None,
            "consistent_development_schema": same_config,
            "frozen_history_config": configs[0] if same_config else None,
            "history_gate": "unknown" if same_config and configs[0] is None else "assessable" if same_config else "unresolved_configurations",
            "eligible_for_freeze": passed, "status": "qualified_development_only" if passed else "retain_incumbent_or_stopped"})
    eligible = [row for row in findings if row["eligible_for_freeze"]]
    if eligible:
        best_sse = min(row["candidate_sse"] for row in eligible)
        tied = [row for row in eligible if math.sqrt(row["candidate_sse"] / best_sse) < 1.002]
        winner = min(tied, key=lambda row: (row["complexity_rank"], ARMS.index(row["arm"])))
    else:
        winner = None
    result = {"version": 1, "source": source_identity(), "findings": findings,
        "recommended_arm": winner["arm"] if winner else "I",
        "decision": "freeze_development_candidate" if winner else "retain_incumbent",
        "native_admission": False, "known5_compatibility": "not_executed_forbidden", "history_admission": "not_established"}
    atomic(output / "evaluation.json", result)
    return result


def freeze(output):
    output = Path(output).resolve()
    frozen = plan(output)
    evaluation = evaluate(output)
    winner = evaluation["recommended_arm"]
    if winner == "I":
        raise ValueError("No candidate qualified; retain incumbent")
    findings = next(row for row in evaluation["findings"] if row["arm"] == winner)
    identity = source_identity()
    template = None
    for period in DEVELOPMENT:
        record, _ = _record(output, "confirmation", period, winner)
        if record["source"] != identity:
            raise ValueError("Freeze requires exactly the measured clean source")
        if template is None:
            template = record["recipe"]
        elif template != record["recipe"]:
            raise ValueError("Different development history configurations need captain review, not an untested replacement")
    original, expected = _recipe(winner, frozen["period_column"], findings["frozen_history_config"])
    if expected != template or text_hash(original) != record["original_source_sha256"]:
        raise ValueError("Freeze source or recipe differs from measured development")
    (output / "frozen_winner_source.py").write_text(original, encoding="utf-8")
    write(output / "frozen_winner_recipe.json", template)
    decision = {"version": 1, "source": identity, "winner": winner, "recipe": template,
        "original_source_sha256": text_hash(original), "recipe_sha256": json_hash(template),
        "evaluation_sha256": digest(output / "evaluation.json"), "history_config": findings["frozen_history_config"],
        "native_admission": False, "history_admission": "unknown" if findings["frozen_history_config"] is None else "not_native_admitted",
        "known5_compatibility": "not_executed_forbidden", "status": "frozen_development_algorithm"}
    write(output / "frozen_decision.json", decision)
    return decision


def validation_source_hashes():
    """Bytes binding a no-fit receipt, independent of publication commit metadata."""
    hashes = source_identity(require_clean=False)["dependency_hashes"]
    paths = [PLAN_DOCUMENT, PLAN_RECIPES, *sorted((ROOT / "tests").glob("test_round6_regression_*.py"))]
    hashes.update({str(path.relative_to(ROOT)): digest(path) for path in paths})
    return hashes


def register_checks(output, receipt_path):
    started = time.monotonic()
    output, receipt_path = Path(output).resolve(), Path(receipt_path).resolve()
    _checked_source(plan(output))
    receipt = read(receipt_path)
    required = {"version", "kind", "label", "elapsed_seconds", "exit_code", "command", "source_hashes"}
    if set(receipt) != required or receipt["version"] != 1 or receipt["kind"] not in {"model_free_checks", "audit_only_checks"}:
        raise ValueError("Use the exact model-free/audit-only check receipt schema")
    elapsed = receipt["elapsed_seconds"]
    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0 or
            type(receipt["exit_code"]) is not int or not isinstance(receipt["label"], str) or
            not 1 <= len(receipt["label"]) <= 120 or not isinstance(receipt["command"], str) or
            not 1 <= len(receipt["command"]) <= 3000):
        raise ValueError("Invalid explicit check receipt")
    if receipt["source_hashes"] != validation_source_hashes():
        raise ValueError("Check receipt source bytes differ from frozen implementation/tests/plan")
    directory = output / "checks" / digest(receipt_path)
    if directory.exists():
        raise ValueError("This check receipt is already charged; preserve its original record")
    cap = elapsed + 30
    ordinal = _reserve(output, kind=receipt["kind"], label=receipt["label"], cap=cap,
                       started=started - elapsed)
    directory.mkdir(parents=True)
    shutil.copyfile(receipt_path, directory / "receipt.json")
    result = {"version": 1, "kind": receipt["kind"], "label": receipt["label"],
        "source": source_identity(), "physical_fit_count": 0, "completed_fit_count": 0,
        "status": "completed" if receipt["exit_code"] == 0 else "failed", "budget_seconds": cap,
        "external_check_seconds": elapsed, "receipt_sha256": digest(directory / "receipt.json"), "api_calls": 0}
    finalize_deadline(result, started=started - elapsed, cap=cap)
    result["elapsed_seconds"] = result["host_total_seconds"]
    write(directory / "summary.json", result)
    _finish_attempt(output, ordinal, result["elapsed_seconds"], result["status"], directory / "summary.json")
    return result


def queue(output):
    """Exactly four screen arms and fresh gated confirmation; safely resumable."""
    output = Path(output).resolve()
    frozen = plan(output)
    _checked_source(frozen)
    validation = read(output / "validation/flat_smoke/summary.json")
    admit_completed(validation, expected_cap=195)
    _ledger_summary(output, output / "validation/flat_smoke/summary.json")
    if validation["source"] != frozen["source"]:
        raise ValueError("Registered smoke belongs to a different frozen source")
    validation_directory = output / "validation/flat_smoke"
    for name, expected in validation["artifact_files"].items():
        path = validation_directory / name
        if path.is_symlink() or not path.resolve().is_relative_to(validation_directory) or digest(path) != expected:
            raise ValueError("Registered smoke evidence changed before the measurement queue")
    if any(row["status"] == "running" for row in read(output / "attempts.json")["attempts"]):
        raise ValueError("Another unfinished reservation owns the serial queue")
    def discovery_once(period):
        if (output / "discovery" / f"p{period}" / "host_schema.json").exists():
            return schema(output, period)
        row = discover(output, period)
        if row["status"] != "completed":
            raise ValueError("Discovery did not finish; retain evidence and stop queue")
        return row
    def once(stage, period, arm):
        if (_run_path(output, stage, period, arm) / "summary.json").exists():
            row, _ = _record(output, stage, period, arm)
        else:
            row = run(output, stage, period, arm)
        print(json.dumps({key: row.get(key) for key in ("stage", "period", "arm", "status", "quality_metric", "correction_status", "physical_fit_count")}), flush=True)
        if row["status"] == "preflight_failed":
            raise ValueError("A preflight failure stopped the queue without repeating any model")
        return row
    discovery_once(4)
    for arm in ARMS:
        once("screen", 4, arm)
    promoted = promote(output)
    stops = []
    if promoted["finalists"]:
        for period in DEVELOPMENT:
            discovery_once(period)
            once("confirmation", period, "I")
            for arm in promoted["finalists"]:
                failures = sum(_fold_failure(output, earlier, arm) for earlier in DEVELOPMENT if earlier < period)
                if failures >= 2:
                    stops.append({"stage": "confirmation", "period": period, "arm": arm,
                        "status": "not_executed", "reason": "two_material_distinct_period_failures", "physical_fit_count": 0})
                    continue
                once("confirmation", period, arm)
    write(output / "stopped.json", {"rows": stops})
    return evaluate(output)


def _close_number(actual, expected, label):
    if type(actual) not in (int, float) or not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-6):
        raise ValueError("Independent audit mismatch: " + label)


def audit(output):
    """No fitting: independently recompute every completed CSV metric and segment."""
    output = Path(output).resolve()
    if (output / "audit.json").exists():
        _ledger_summary(output, output / "audit.json")
        raise ValueError("Preserve the original audit; any later review needs a separate registered supplement")
    started = time.monotonic()
    ordinal = _reserve(output, kind="audit", label="all_completed_predictions", cap=120, started=started)
    result = {"version": 1, "kind": "audit", "status": "completed", "budget_seconds": 120,
        "physical_fit_count": 0, "completed_fit_count": 0, "api_calls": 0, "checks": [], "metrics_tables": 0}
    try:
        frozen = plan(output)
        result["source"] = _checked_source(frozen)
        for path in sorted((output / "runs").glob("*/*/*/summary.json")):
            raw = read(path)
            if raw["status"] != "completed":
                _ledger_summary(output, path)
                result["checks"].append({"summary": str(path.relative_to(output)), "status": raw["status"],
                    "admitted": False, "physical_fit_count": raw.get("physical_fit_count")})
                continue
            record, original_path = _record(output, raw["stage"], raw["period"], raw["arm"])
            if original_path != path:
                # A retained failed original and a successful identical technical retry
                # are never confused; completed duplicates are not permitted.
                raise ValueError("Completed original/retry ambiguity")
            spec, fit, held, answers = case_frames(output, record["period"])
            y = pd.to_numeric(answers[spec.target_column], errors="raise").to_numpy(dtype=float)
            predictions = {}
            for name, expected in record["metrics"].items():
                csv_path = path.parent / (name + ".csv")
                if digest(csv_path) != expected["prediction_sha256"]:
                    raise ValueError("A raw ablation prediction digest changed")
                frame = pd.read_csv(csv_path, **CSV)
                if frame[spec.id_column].tolist() != answers[spec.id_column].tolist():
                    raise ValueError("Independent audit IDs differ from held positions")
                values = pd.to_numeric(frame[spec.target_column], errors="raise").to_numpy(dtype=float)
                if len(values) != len(y) or not np.isfinite(values).all():
                    raise ValueError("Independent audit found nonfinite/wrong-length predictions")
                error = y - values
                for key, value in {"rmse": float(np.sqrt(np.mean(error * error))),
                        "mae": float(np.mean(np.abs(error))), "sse": float(np.sum(error * error))}.items():
                    _close_number(expected[key], value, name + "/" + key)
                if expected["rows"] != len(y):
                    raise ValueError("Metric row count differs")
                predictions[name] = values
                result["metrics_tables"] += 1
            stored = read(path.parent / "segments.json")["rows"]
            recomputed = diagnostics(fit, held, answers, predictions, spec, frozen, record["recipe"]["history_config"])
            keys = lambda row: (row["arm"], row["partition"], row["segment"])
            before, after = {keys(row): row for row in stored}, {keys(row): row for row in recomputed}
            if set(before) != set(after):
                raise ValueError("Independent segment membership differs")
            for key in before:
                for field in before[key]:
                    a, b = before[key][field], after[key][field]
                    if isinstance(a, float) and isinstance(b, (int, float)):
                        _close_number(a, b, str(key) + "/" + field)
                    elif a != b:
                        raise ValueError("Independent segment field differs: " + str(key) + "/" + field)
            result["checks"].append({"summary": str(path.relative_to(output)), "summary_sha256": digest(path),
                "status": "verified", "metrics_tables": len(predictions), "segment_rows": len(stored),
                "physical_fit_count": record["physical_fit_count"], "correction_status": record.get("correction_status")})
        if source_identity() != result["source"]:
            raise ValueError("Audit source changed")
    except (ValueError, OSError, TimeoutError, KeyError, subprocess.SubprocessError) as error:
        result.update(status="failed", error_type=type(error).__name__, reason=str(error))
    finalize_deadline(result, started=started, cap=120)
    write(output / "audit.json", result)
    _finish_attempt(output, ordinal, result["host_total_seconds"], result["status"], output / "audit.json")
    return result


def reserve_validation(output):
    started = time.monotonic()
    output = Path(output).resolve()
    _checked_source(plan(output))
    ordinal = _reserve(output, kind="validation", label="flat_smoke", cap=195, started=started)
    return {"status": "reserved", "ordinal": ordinal, "cap_seconds": 195,
            "original_deadline_monotonic": started + 195}


def _smoke_binding_errors(smoke, receipt):
    """Bind the flat bytes and probe to the frozen checkout, retaining failures."""
    from build_submission import archive_files
    errors = []
    try:
        expected = {name: hashlib.sha256(content).hexdigest() for name, content in archive_files().items()}
        if read(smoke / "packaged_dependency_hashes.json") != expected:
            errors.append("packaged_dependency_hashes_differ_from_frozen_source")
        for name, field in (("flat_package.zip", "archive_sha256"),
                            ("smoke_probe.py", "probe_sha256"),
                            ("packaged_dependency_hashes.json", "packaged_dependency_hashes_sha256")):
            if digest(smoke / name) != receipt.get(field):
                errors.append("smoke_receipt_digest_mismatch:" + name)
        with zipfile.ZipFile(smoke / "flat_package.zip") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(expected) or archive.testzip() is not None:
                errors.append("flat_archive_entry_manifest_mismatch")
            elif any(hashlib.sha256(archive.read(name)).hexdigest() != checksum for name, checksum in expected.items()):
                errors.append("flat_archive_bytes_differ_from_frozen_source")
        if any(digest(smoke / "package" / name) != checksum for name, checksum in expected.items()):
            errors.append("extracted_package_bytes_differ_from_frozen_source")
        native = ast.parse((ROOT / "tests/test_round6_regression_native.py").read_text(encoding="utf-8"))
        probes = [ast.literal_eval(node.value) for node in native.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "PROBE" for target in node.targets)]
        if len(probes) != 1 or text_hash(probes[0]) != digest(smoke / "smoke_probe.py"):
            errors.append("smoke_probe_differs_from_frozen_test")
    except (ValueError, OSError, KeyError, SyntaxError, zipfile.BadZipFile) as error:
        errors.append("smoke_binding_evidence_unavailable:" + type(error).__name__)
    return errors


def register_validation(output, smoke_directory):
    """Import the explicitly reserved two-invocation flat smoke, even on failure."""
    registration_started = time.monotonic()
    output, smoke = Path(output).resolve(), Path(smoke_directory).resolve()
    _checked_source(plan(output))
    ledger = read(output / "attempts.json")
    pending = [row for row in ledger["attempts"] if row["status"] == "running"]
    if len(pending) != 1 or pending[0]["kind"] != "validation" or pending[0]["label"] != "flat_smoke":
        raise ValueError("Reserve the fixed flat-smoke allowance before executing it")
    receipt = read(smoke / "smoke_receipt.json")
    elapsed = receipt["elapsed_seconds"]
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Smoke elapsed time is invalid")
    directory = output / "validation" / "flat_smoke"
    directory.mkdir(parents=True, exist_ok=False)
    files = [smoke / name for name in ("smoke_receipt.json", "flat_package.zip", "packaged_dependency_hashes.json", "smoke_probe.py")]
    package = smoke / "package"
    files += [package / name for name in ("flat_smoke_summary.json", "frozen_candidate.py", "frozen_recipe.json")]
    events, evidence_errors = [], []
    for invocation in sorted(package.glob("invocation_*")):
        for name in ("predictions.csv", "regression_evidence.json", "regression_ablation.csv", "native_fit_events.jsonl", "analytic_fit_events.jsonl"):
            path = invocation / name
            if path.is_file():
                files.append(path)
                if name in {"native_fit_events.jsonl", "analytic_fit_events.jsonl"}:
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            try:
                                event = json.loads(line)
                                if (not isinstance(event, dict) or not {"engine", "fit_ordinal", "component", "phase", "rows", "event"} <= set(event)
                                        or event["engine"] not in {"catboost", "analytic_residual"}
                                        or type(event["fit_ordinal"]) is not int or event["fit_ordinal"] < 1
                                        or event["event"] not in {"started", "completed", "failed"}):
                                    raise ValueError("Malformed physical fit event")
                                if event["event"] != "started" and (
                                        type(event.get("elapsed_seconds")) not in (int, float)
                                        or not math.isfinite(event["elapsed_seconds"]) or event["elapsed_seconds"] < 0):
                                    raise ValueError("Malformed physical fit timing")
                                if event["event"] == "failed" and not isinstance(event.get("exception_type"), str):
                                    raise ValueError("Malformed failed physical fit event")
                                events.append(event)
                            except (ValueError, TypeError):
                                evidence_errors.append("malformed_fit_event:" + name)
    for path in files:
        if path.is_file():
            relative = path.relative_to(smoke)
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    pairs, event_counts = {}, {}
    for event in events:
        key = (event["engine"], event["fit_ordinal"])
        counts = event_counts.setdefault(key, {"started": 0, "completed": 0, "failed": 0})
        counts[event["event"]] += 1
        item = pairs.setdefault(key, {"ordinal": event["fit_ordinal"], "engine": event["engine"], "component": event["component"],
            "phase": event["phase"], "rows": event["rows"], "started": False, "completed": False,
            "elapsed_seconds": None, "exception_type": None})
        if event["event"] == "started":
            item["started"] = True
        elif event["event"] == "completed":
            item.update(completed=True, elapsed_seconds=event["elapsed_seconds"])
        elif event["event"] == "failed":
            item.update(elapsed_seconds=event["elapsed_seconds"], exception_type=event["exception_type"])
        if any(item[field] != event[field] for field in ("engine", "component", "phase", "rows")):
            evidence_errors.append("fit_event_metadata_changed")
    for counts in event_counts.values():
        if counts["started"] != 1 or counts["completed"] + counts["failed"] != 1:
            evidence_errors.append("fit_event_start_or_terminal_count_mismatch")
    evidence_errors.extend(_smoke_binding_errors(smoke, receipt))
    artifact_files = {str(path.relative_to(directory)): digest(path) for path in directory.rglob("*") if path.is_file()}
    engine_fits = [row for row in pairs.values() if row["engine"] != "analytic_residual"]
    analytic_fits = [row for row in pairs.values() if row["engine"] == "analytic_residual"]
    result = {"version": 1, "kind": "real_flat_package_toy_smoke_not_quality_comparison",
        "artifact_files": artifact_files,
        "evidence_binding_verified": not evidence_errors, "evidence_binding_errors": evidence_errors,
        "source": source_identity(), "smoke_elapsed_seconds": elapsed, "budget_seconds": 195,
        "status": "completed" if receipt["returncode"] == 0 else "failed", "receipt_sha256": digest(smoke / "smoke_receipt.json"),
        "physical_fits": engine_fits, "physical_fit_count": sum(row["started"] for row in engine_fits),
        "completed_fit_count": sum(row["completed"] for row in engine_fits),
        "analytic_fits": analytic_fits, "actual_analytic_fit_count": sum(row["started"] for row in analytic_fits),
        "completed_analytic_fit_count": sum(row["completed"] for row in analytic_fits),
        "total_fit_operations": sum(row["started"] for row in pairs.values()),
        "physical_fits_reason": None if events else "no_model_fit_events_observed", "api_calls": 0}
    if result["status"] == "completed" and (not engine_fits or not all(row["completed"] for row in pairs.values())):
        result.update(status="failed", stop_reason="smoke_actual_fit_observer_incomplete")
    if result["status"] == "completed" and elapsed > 195:
        result.update(status="failed", stop_reason="smoke_receipt_exceeds_original_cap")
    if result["status"] == "completed" and evidence_errors:
        result.update(status="failed", stop_reason="smoke_source_or_event_binding_failed")
    # Reservation, flat packaging/extraction, two runs, and registration all
    # spend the same original allowance; invoke these commands back-to-back.
    finalize_deadline(result, started=pending[0]["started_monotonic"], cap=195)
    result["registration_seconds"] = time.monotonic() - registration_started
    result["elapsed_seconds"] = result["host_total_seconds"]
    write(directory / "summary.json", result)
    _finish_attempt(output, pending[0]["ordinal"], result["elapsed_seconds"], result["status"], directory / "summary.json")
    return result


def export(output, destination):
    """Verified data-free ZIP from an explicit allowlist, never input trees."""
    output, destination = Path(output).resolve(), Path(destination).resolve()
    frozen = plan(output)
    _checked_source(frozen)
    audited = read(output / "audit.json")
    admit_completed(audited, expected_cap=120)
    _ledger_summary(output, output / "audit.json")
    if audited["source"] != frozen["source"]:
        raise ValueError("Audit belongs to a different measured source")
    files = {}
    for name in ("environment.json", "attempts.json", "preparation_summary.json", "audit.json", "stopped.json", "promotion.json", "evaluation.json", "frozen_decision.json",
                 "frozen_winner_source.py", "frozen_winner_recipe.json"):
        path = output / name
        if path.is_file():
            files[name] = path
    for directory in sorted((output / "runs").glob("*/*/*")):
        if not directory.is_dir():
            continue
        for name in ("summary.json", "attempt_started.json", "segments.json", "physical_fits.json",
                     "source.py", "original_source.py", "receipt.json", "submission.csv", "identity.csv", "base.csv", "mean.csv", "calibrated.csv", "corrected.csv",
                     "regression_evidence.json", "regression_ablation.csv"):
            path = directory / name
            if path.is_file():
                files[str(path.relative_to(output))] = path
        for name in ("candidate.py", "recipe.json"):
            path = directory / "frozen" / name
            if path.is_file():
                files[str(path.relative_to(output))] = path
    for path in sorted((output / "discovery").glob("*/host_schema.json")):
        files[str(path.relative_to(output))] = path
    for prefix in ("preflight", "checks"):
        for directory in (output / prefix).glob("*"):
            for name in ("summary.json", "receipt.json"):
                path = directory / name
                if path.is_file():
                    files[str(path.relative_to(output))] = path
    validation = output / "validation" / "flat_smoke"
    if validation.is_dir():
        registration = read(validation / "summary.json")
        files[str((validation / "summary.json").relative_to(output))] = validation / "summary.json"
        for name, expected in registration["artifact_files"].items():
            path = validation / name
            if path.is_symlink() or not path.resolve().is_relative_to(validation) or digest(path) != expected:
                raise ValueError("Registered validation evidence changed")
            files[str(path.relative_to(output))] = path
    for path in [*sorted((ROOT / "agent").glob("round6_regression_*.py")),
                 *sorted((ROOT / "tests").glob("test_round6_regression_*.py")), Path(__file__),
                 ROOT / "requirements.txt", ROOT / "requirements.platform.txt", PLAN_DOCUMENT, PLAN_RECIPES]:
        files["implementation/" + str(path.relative_to(ROOT))] = path
    if destination.exists() or destination.suffix.lower() != ".zip":
        raise ValueError("Use a fresh .zip evidence destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "base_sha": BASE_SHA, "source": source_identity(),
        "publication": frozen["publication"],
        "input_hashes": frozen["source_files"], "folds": frozen["folds"],
        "plan_document_sha256": frozen["plan_document_sha256"], "data_free": True,
        "excludes": ["raw_TRAIN", "raw_TEST", "held_answers", "models", "fit.csv", "predict.csv", "context.json", "native_logs"],
        "files": {name: {"sha256": digest(path), "bytes": path.stat().st_size} for name, path in sorted(files.items())}}
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False))
        for name, path in sorted(files.items()):
            archive.write(path, arcname=name)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != {"manifest.json", *files}:
            raise ValueError("Evidence archive CRC or entry manifest failed")
        for name, expected in manifest["files"].items():
            payload = archive.read(name)
            if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise ValueError("Evidence archive content differs from manifest")
    return {"version": 1, "archive_sha256": digest(destination), "archive_bytes": destination.stat().st_size,
            "entries": len(files) + 1, "data_free": True, "crc_and_manifest_verified": True}


def compact(output, destination):
    output, destination = Path(output).resolve(), Path(destination).resolve()
    frozen = plan(output)
    _checked_source(frozen)
    attempts = read(output / "attempts.json")
    records = []
    for path in sorted((output / "runs").glob("*/*/*/summary.json")):
        row = read(path)
        _ledger_summary(output, path)
        if row["status"] == "completed":
            admit_completed(row, SCREEN_SECONDS if row["stage"] == "screen" else CONFIRM_SECONDS)
        records.append(row)
    identities = {}
    for record in records:
        identity = record.get("source")
        if identity is not None:
            key = identity["dependency_sha256"]
            identities[key] = identity
            record["source"] = {"head": identity["head"], "dirty": identity["dirty"], "dependency_sha256": key}
    validations = [read(path) for path in sorted((output / "validation").glob("*/summary.json"))]
    auxiliary = [read(path) for prefix in ("checks", "preflight") for path in sorted((output / prefix).glob("*/summary.json"))]
    payload = {"version": 1, "base_sha": BASE_SHA, "source": source_identity(), "source_identities": identities,
        "publication": frozen["publication"], "validation": validations, "auxiliary_checks": auxiliary,
        "environment": read(output / "environment.json"), "attempts": attempts, "records": records,
        "known5": "not_executed_forbidden", "full_train": "not_executed_forbidden", "native_admission": False}
    for name in ("evaluation", "audit", "stopped"):
        if (output / (name + ".json")).is_file():
            payload[name] = read(output / (name + ".json"))
    payload["correction_decisions"] = [{key: row.get(key) for key in
        ("stage", "period", "arm", "status", "correction_status", "correction_reason", "physical_fit_count", "actual_analytic_fit_count")}
        for row in records]
    payload["total_observed_seconds"] = _used_seconds(attempts)
    destination.mkdir(parents=True, exist_ok=True)
    write(destination / "round6_regression_results.json", payload)
    columns = ("ordinal", "kind", "label", "stage", "period", "arm", "status", "budget_seconds",
        "host_total_seconds", "fit_rows", "predict_rows", "physical_fit_count", "completed_fit_count",
        "actual_analytic_fit_count", "completed_analytic_fit_count", "total_fit_operations",
        "correction_status", "correction_reason", "quality_metric", "code_sha256", "recipe_sha256",
        "output_sha256", "summary_sha256")
    rows = []
    for attempt in attempts["attempts"]:
        row = {key: attempt.get(key) for key in ("ordinal", "kind", "label", "status", "summary_sha256")}
        row.update(budget_seconds=attempt["cap_seconds"], host_total_seconds=attempt["elapsed_seconds"])
        if attempt.get("summary"):
            summary = read(output / attempt["summary"])
            row.update({key: summary.get(key) for key in columns if key in summary})
        rows.append(row)
    with (destination / "round6_regression_results.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (destination / "round6_regression_correction_decisions.csv").open("x", encoding="utf-8", newline="") as stream:
        columns = ("stage", "period", "arm", "status", "correction_status", "correction_reason", "physical_fit_count", "actual_analytic_fit_count")
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(payload["correction_decisions"])
    return {"records": len(records), "ledger_rows": len(rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--task", type=Path, required=True)
    p.add_argument("--period-column", required=True)
    p.add_argument("--size-column", required=True)
    p.add_argument("--published-commit", required=True)
    p = commands.add_parser("discover")
    p.add_argument("--period", type=int, choices=DEVELOPMENT, required=True)
    p = commands.add_parser("run")
    p.add_argument("--stage", choices=("screen", "confirmation"), required=True)
    p.add_argument("--period", type=int, choices=DEVELOPMENT, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--technical-retry-reason")
    p = commands.add_parser("register-checks")
    p.add_argument("--receipt", type=Path, required=True)
    p = commands.add_parser("register-validation")
    p.add_argument("--smoke-directory", type=Path, required=True)
    for name in ("promote", "evaluate", "freeze", "reserve-validation", "queue", "audit"):
        commands.add_parser(name)
    for name in ("export", "compact"):
        p = commands.add_parser(name)
        p.add_argument("--destination", type=Path, required=True)
    for p in commands.choices.values():
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.task, args.output, args.period_column, args.size_column, args.published_commit)
    elif args.command == "discover":
        result = discover(args.output, args.period)
    elif args.command == "run":
        result = run(args.output, args.stage, args.period, args.arm, args.technical_retry_reason)
    elif args.command == "reserve-validation":
        result = reserve_validation(args.output)
    elif args.command == "register-checks":
        result = register_checks(args.output, args.receipt)
    elif args.command == "register-validation":
        result = register_validation(args.output, args.smoke_directory)
    elif args.command == "promote":
        result = promote(args.output)
    elif args.command == "evaluate":
        result = evaluate(args.output)
    elif args.command == "freeze":
        result = freeze(args.output)
    elif args.command == "queue":
        result = queue(args.output)
    elif args.command == "audit":
        result = audit(args.output)
    elif args.command == "export":
        result = export(args.output, args.destination)
    else:
        result = compact(args.output, args.destination)
    print(json.dumps({key: result.get(key) for key in ("status", "stage", "period", "arm", "quality_metric", "finalists", "decision", "recommended_arm") if key in result}, allow_nan=False), flush=True)
    return 1 if result.get("status") in {"failed", "preflight_failed"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
