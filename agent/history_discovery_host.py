"""Bounded, opt-in discovery using only raw INNER-fit feature rows."""

from __future__ import annotations

import csv
import hashlib
import json
from time import monotonic as _now

from agent.automl import _file_hash
from agent.execution import run_python
from agent.history_runtime import normalize_history_options
from agent.llm import ROOT, safe_error_message


DISCOVERY_SECONDS = 8.0
DISCOVERY_MEMORY_BYTES = 1024 ** 3
MAX_ROWS = 250_000
MAX_FEATURES = 128
MAX_CELLS = 8_000_000
MAX_CSV_BYTES = 64 * 1024 ** 2
MAX_VALUE_CHARS = 512
MAX_RESULT_BYTES = 64 * 1024


class _DiscoveryLimit(RuntimeError):
    pass


class _BoundedCSV:
    def __init__(self, stream, deadline):
        self.stream, self.deadline, self.bytes = stream, deadline, 0

    def write(self, text):
        if _now() >= self.deadline:
            raise _DiscoveryLimit("preparation_time_limit")
        size = len(text.encode("utf-8"))
        if self.bytes + size > MAX_CSV_BYTES:
            raise _DiscoveryLimit("input_byte_limit")
        self.bytes += size
        return self.stream.write(text)


def _source(root=ROOT):
    return (
        "import sys,json\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "import pandas as pd\nfrom agent.history_discovery import discover_history_schema\n"
        "request=json.loads(Path('request.json').read_text(encoding='utf-8'))\n"
        "frame=pd.read_csv('fit_features.csv',dtype=object,keep_default_na=False,na_filter=False)\n"
        "result=discover_history_schema(frame,feature_columns=request['feature_columns'],"
        "numeric_columns=request['numeric_columns'],period_column=request['period_column'])\n"
        "Path('discovery.json').write_text(json.dumps(result,ensure_ascii=False,"
        "allow_nan=False,separators=(',',':')),encoding='utf-8')\n"
    )


def _eligible(state):
    gate = state.gate
    if state.report.get("stage") != "inner_research" or state.dialogue_closed:
        return "research_not_open"
    if gate.manifest.get("task_semantics", {}).get("adaptive_research_allowed") is False:
        return "task_semantics_unavailable"
    if gate.spec.metric.task_type != "regression":
        return "regression_required"
    if gate.choice.get("strategy") != "time":
        return "time_validation_required"
    inner = gate.inner_split
    if not gate.fit_rows or not gate.held_rows or not isinstance(inner, dict):
        return "inner_validation_unavailable"
    if inner.get("strategy") != "time" or inner.get("time_column") != gate.choice.get("time_column"):
        return "matching_inner_time_required"
    if gate.choice.get("time_column") not in gate.spec.features:
        return "period_not_in_features"
    if state.baseline is None or state.baseline_inner is None:
        return "measured_baseline_required"
    if state.research_timeout() <= 2:
        return "finalization_reserve"
    return None


def _reject_constant(_value):
    raise ValueError("Non-finite discovery JSON")


def _read_result(directory, features, numeric, period):
    path = directory / "discovery.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RESULT_BYTES:
        raise ValueError("Discovery requires a bounded regular JSON result")
    result = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if (not isinstance(result, dict) or result.get("version") != 1
            or result.get("scope") != "inner_fit_features_only"
            or result.get("identity_verified") is not False
            or result.get("status") not in {"proposed", "abstained", "limited"}):
        raise ValueError("Invalid discovery result contract")
    if result.get("reason") is not None and not isinstance(result["reason"], str):
        raise ValueError("Invalid discovery reason")
    # Also reject nested non-finite values such as 1e999, not just NaN tokens.
    json.dumps(result, allow_nan=False)
    if result["status"] != "proposed":
        if result.get("selected_config") is not None:
            raise ValueError("Incomplete discovery must not select a history config")
        return result, None
    options = normalize_history_options(result.get("selected_config"), 0.5)
    if options is None:
        raise ValueError("Discovery did not freeze a selected configuration")
    config = options["history_config"]
    if (config["period_column"] != period or not set(config["key_columns"]) <= set(features) - {period}
            or not set(config["numeric_columns"]) <= set(numeric)
            or config["positive_columns"] or config["ambiguity_policy"] != "past_unique_only"):
        raise ValueError("Discovery selected columns or semantics outside the host allowance")
    proposal = result.get("proposer_result")
    proposals = proposal.get("proposals", []) if isinstance(proposal, dict) else []
    if (not isinstance(proposal, dict) or proposal.get("status") != "proposed"
            or proposal.get("scope") != "inner_fit_features_only" or proposal.get("auto_activate") is not False
            or not isinstance(proposals, list) or len(proposals) != 1
            or not isinstance(proposals[0], dict) or proposals[0].get("config") != config
            or proposals[0].get("identity_verified") is not False):
        raise ValueError("Selected discovery config has no complete top-one proposal")
    return result, options


def discover_history_options(state):
    """Return one explicit history recipe option, or preserve normal research.

    The discovery worker receives a feature-only CSV of exactly INNER-fit rows.
    This is trusted-code subprocess isolation, not an OS security sandbox.
    Input serialization and worker imports use the same discovery allowance;
    outer checks and final refit keep the original host budget and reserve.
    """
    started = _now()
    report = {"version": 1, "policy": "auto_schema_v1", "requested": True,
        "scope": "inner_fit_features_only", "status": "skipped", "reason": None,
        "identity_verified": False, "blend_weight": 0.5, "selected_config": None,
        "time_limit_seconds": DISCOVERY_SECONDS, "worker_memory_limit_bytes": DISCOVERY_MEMORY_BYTES,
        "fit_rows": 0, "feature_columns": [], "period_column": None,
        "fit_input_sha256": None, "fit_ids_sha256": None, "worker_source_sha256": None,
        "worker_source_text_sha256": None,
        "execution": None, "result": None, "elapsed_seconds": 0.0}
    state.report["history_discovery"] = report

    def finish(status, reason, options=None):
        report.update(status=status, reason=reason, elapsed_seconds=max(0, _now() - started))
        state.report["extensions"]["history_discovery"] = status
        state.event("history_discovery", status=status, reason=reason)
        return options

    reason = _eligible(state)
    if reason:
        return finish("skipped", reason)
    gate, baseline = state.gate, state.baseline
    identity = (baseline.code_hash, baseline.recipe_hash)
    gate.check_inputs()
    state.check_source()
    baseline.verify()
    options = None
    try:
        features, numeric = list(gate.spec.features), list(gate.spec.numeric)
        period = gate.choice["time_column"]
        rows = len(gate.fit_rows)
        report.update(fit_rows=rows, feature_columns=features, period_column=period)
        if (rows > MAX_ROWS or len(features) > MAX_FEATURES or rows * len(features) > MAX_CELLS):
            raise _DiscoveryLimit("input_shape_limit")
        if {gate.spec.id_column, gate.spec.target_column} & set(features):
            raise ValueError("Discovery features contain a target or technical ID")
        deadline = min(started + DISCOVERY_SECONDS, _now() + state.research_timeout())
        directory = state.run_dir / "history_discovery"
        directory.mkdir(exist_ok=False)
        frame = gate.research_train.iloc[gate.fit_rows]
        ids = frame[gate.spec.id_column].tolist()
        report["fit_ids_sha256"] = hashlib.sha256(json.dumps(ids, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        with (directory / "fit_features.csv").open("w", encoding="utf-8", newline="") as stream:
            bounded = _BoundedCSV(stream, deadline)
            writer = csv.writer(bounded)
            writer.writerow(features)
            for values in frame.loc[:, features].itertuples(index=False, name=None):
                for value in values:
                    if value is not None and (not isinstance(value, str) or len(value) > MAX_VALUE_CHARS):
                        raise _DiscoveryLimit("raw_value_limit")
                writer.writerow(values)
        report["fit_input_sha256"] = _file_hash(directory / "fit_features.csv")
        request = {"feature_columns": features, "numeric_columns": numeric, "period_column": period}
        (directory / "request.json").write_text(json.dumps(request, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        timeout = min(deadline - _now(), state.research_timeout())
        if timeout <= 0:
            raise _DiscoveryLimit("preparation_time_limit")
        source = _source()
        report["worker_source_text_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
        execution = run_python(source, directory, directory / "execution", 0,
            timeout=timeout, max_memory_bytes=DISCOVERY_MEMORY_BYTES)
        report["execution"] = execution
        if execution.get("timed_out") or _now() >= deadline:
            raise _DiscoveryLimit("worker_time_limit")
        if execution.get("memory_limit_exceeded"):
            raise _DiscoveryLimit("worker_memory_limit")
        if execution.get("exit_code") != 0 or execution.get("error"):
            raise RuntimeError("Discovery worker failed")
        script = directory / "execution" / "experiment_000.py"
        if script.is_symlink() or not script.is_file():
            raise ValueError("Discovery worker source artifact is missing")
        report["worker_source_sha256"] = _file_hash(script)
        result, options = _read_result(directory, features, numeric, period)
        report["result"] = result
        report["selected_config"] = options["history_config"] if options else None
        status, reason = ("selected", "one_schema_frozen") if options else (result["status"], result["reason"])
    except _DiscoveryLimit as error:
        status, reason = "limited", str(error)
    except (OSError, ValueError, RuntimeError, TimeoutError, TypeError, KeyError, UnicodeError) as error:
        status, reason = "failed", "discovery_failed"
        report["error"] = safe_error_message(error)
    # Integrity failures are not converted into a benign discovery refusal.
    gate.check_inputs()
    state.check_source()
    if state.baseline is not baseline or (baseline.code_hash, baseline.recipe_hash) != identity:
        raise ValueError("Discovery changed the frozen baseline identity")
    baseline.verify()
    return finish(status, reason, options)
