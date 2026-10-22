"""Opt-in bounded schema discovery for the isolated regression candidate.

The caller supplies exactly INNER-fit raw features. Neither labels, prediction
features nor IDs enter the worker. The returned config must be frozen by the
caller and reused for INNER, outer and full refit; the candidate never calls
discovery again. Equal keys remain a heuristic, not verified entity identity.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import time

import pandas as pd

from agent import history_discovery as _history_discovery_module
from agent import history_schema_proposer as _history_schema_proposer_module
from agent import project_root
from agent.history_discovery import (MAX_FEATURES, MAX_ROWS,
                                     MAX_CELL_VISITS as DEFAULT_CELL_VISITS,
                                     MAX_CHECKED_CONFIGS as DEFAULT_CHECKED_CONFIGS,
                                     MAX_VALUE_CHARS, discover_history_schema)


MAX_CELL_VISITS = 12_000_000
MAX_CHECKED_CONFIGS = 192
DISCOVERY_SECONDS = 8.0
DISCOVERY_MEMORY_BYTES = 1024 ** 3


def discover_regression_schema(X_inner, *, feature_columns, numeric_columns, period_column):
    """Pure feature-only search; a caller must impose its time/memory deadline."""
    return discover_history_schema(X_inner, feature_columns=feature_columns,
        numeric_columns=numeric_columns, period_column=period_column,
        max_cell_visits=MAX_CELL_VISITS, max_checked_configs=MAX_CHECKED_CONFIGS,
        adaptive_checks=True)


def _worker_source(extended):
    root = project_root()
    module, function = (("agent.regression_discovery", "discover_regression_schema") if extended
                        else ("agent.history_discovery", "discover_history_schema"))
    return ("import json,sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(root)!r})\n"
        "import pandas as pd\n"
        f"from {module} import {function} as discover\n"
        "request=json.loads(Path('request.json').read_text(encoding='utf-8'))\n"
        "frame=pd.read_csv('fit_features.csv',dtype=object,keep_default_na=False,na_filter=False)\n"
        "result=discover(frame,**request)\n"
        "Path('discovery.json').write_text(json.dumps(result,allow_nan=False,"
        "ensure_ascii=False,separators=(',',':')),encoding='utf-8')\n")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def discover_regression_schema_bounded(X_inner, *, feature_columns, numeric_columns,
                                       period_column, output_dir, deadline=None, extended=True):
    """Serialize and search within one 8s allowance and sampled 1GiB child cap.

    ``output_dir`` must be fresh. ``deadline`` is an optional earlier monotonic
    caller deadline; it cannot extend the fixed allowance. Contract violations
    raise ValueError. Resource limits/worker failures return no selected config
    so the caller can continue with its base candidate. Sampled RSS is not an
    OS memory sandbox; unknown telemetry remains unknown in ``execution``.
    """
    from agent.execution import run_python
    from agent.history_discovery_host import _BoundedCSV, _DiscoveryLimit, _read_result

    started = time.monotonic()
    if type(extended) is not bool:
        raise ValueError("extended must be a boolean")
    work_limit = MAX_CELL_VISITS if extended else DEFAULT_CELL_VISITS
    check_limit = MAX_CHECKED_CONFIGS if extended else DEFAULT_CHECKED_CONFIGS
    if deadline is not None and (type(deadline) not in {int, float} or not math.isfinite(deadline)):
        raise ValueError("deadline must be a finite monotonic time or None")
    expires = min(started + DISCOVERY_SECONDS, deadline) if deadline is not None else started + DISCOVERY_SECONDS
    if not isinstance(X_inner, pd.DataFrame) or X_inner.columns.has_duplicates:
        raise ValueError("INNER-fit requires a DataFrame with unique columns")
    for names in (feature_columns, numeric_columns):
        if (not isinstance(names, (list, tuple)) or any(not isinstance(n, str) or not n.strip() for n in names)
                or len(names) != len(set(names))):
            raise ValueError("Feature roles must contain unique nonempty names")
    if (not feature_columns or set(X_inner.columns) != set(feature_columns)
            or len(X_inner.columns) != len(feature_columns)
            or not set(numeric_columns) <= set(feature_columns)
            or not isinstance(period_column, str) or period_column not in feature_columns):
        raise ValueError("INNER-fit columns must equal the declared features and include the period")
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    # The contest builder flattens modules to _alfa_agent__*.py. Resolve the
    # files actually imported by Python instead of assuming source filenames.
    module_sources = {
        "regression_discovery.py": Path(__file__).resolve(),
        "history_discovery.py": Path(_history_discovery_module.__file__).resolve(),
        "history_schema_proposer.py": Path(_history_schema_proposer_module.__file__).resolve(),
    }
    module_hashes = {name: _hash(path) for name, path in module_sources.items()}
    report = {"version": 1, "policy": "regression_bounded_discovery_v1" if extended else "original_discovery_48",
        "scope": "inner_fit_features_only", "status": "skipped", "reason": None,
        "selected_config": None, "identity_verified": False, "result": None,
        "fit_rows": len(X_inner), "feature_columns": list(feature_columns),
        "period_column": period_column, "fit_input_sha256": None,
        "request_sha256": None, "worker_source_sha256": None, "result_sha256": None,
        "implementation_sha256": module_hashes,
        "time_limit_seconds": DISCOVERY_SECONDS, "worker_memory_limit_bytes": DISCOVERY_MEMORY_BYTES,
        "max_cell_visits": work_limit, "max_checked_configs": check_limit,
        "execution": None, "elapsed_seconds": 0.0}
    try:
        if len(X_inner) > MAX_ROWS or len(feature_columns) > MAX_FEATURES:
            raise _DiscoveryLimit("input_shape_limit")
        if 2 * len(X_inner) * len(feature_columns) > work_limit:
            raise _DiscoveryLimit("work_limit")
        if time.monotonic() >= expires:
            raise _DiscoveryLimit("preparation_time_limit")
        with (directory / "fit_features.csv").open("x", encoding="utf-8", newline="") as stream:
            bounded = _BoundedCSV(stream, expires)
            writer = csv.writer(bounded)
            writer.writerow(feature_columns)
            for values in X_inner.loc[:, list(feature_columns)].itertuples(index=False, name=None):
                for value in values:
                    if value is not None and not isinstance(value, str):
                        raise ValueError("INNER-fit raw cells must be str or None")
                    if isinstance(value, str) and len(value) > MAX_VALUE_CHARS:
                        raise _DiscoveryLimit("value_length_limit")
                writer.writerow(values)
        report["fit_input_sha256"] = _hash(directory / "fit_features.csv")
        request = {"feature_columns": list(feature_columns), "numeric_columns": list(numeric_columns),
                   "period_column": period_column}
        (directory / "request.json").write_text(json.dumps(request, allow_nan=False), encoding="utf-8")
        report["request_sha256"] = _hash(directory / "request.json")
        source = _worker_source(extended)
        report["worker_source_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
        remaining = expires - time.monotonic()
        if remaining <= 0:
            raise _DiscoveryLimit("preparation_time_limit")
        execution = run_python(source, directory, directory / "execution", 0,
                               timeout=remaining, max_memory_bytes=DISCOVERY_MEMORY_BYTES)
        report["execution"] = execution
        if (report["fit_input_sha256"] != _hash(directory / "fit_features.csv")
                or report["request_sha256"] != _hash(directory / "request.json")
                or any(_hash(module_sources[name]) != sha for name, sha in module_hashes.items())):
            raise RuntimeError("Discovery worker changed its frozen inputs")
        if execution.get("timed_out") or time.monotonic() >= expires:
            raise _DiscoveryLimit("worker_time_limit")
        if execution.get("memory_limit_exceeded"):
            raise _DiscoveryLimit("worker_memory_limit")
        if execution.get("exit_code") != 0 or execution.get("error"):
            report.update(status="failed", reason="worker_failed")
        else:
            try:
                result, options = _read_result(directory, feature_columns, numeric_columns, period_column)
            except (ValueError, TypeError, KeyError) as error:
                raise RuntimeError("Invalid discovery worker result") from error
            report.update(status=result["status"], reason=result["reason"], result=result,
                selected_config=options["history_config"] if options else None,
                result_sha256=_hash(directory / "discovery.json"))
    except _DiscoveryLimit as error:
        report.update(status="limited", reason=str(error))
    except (OSError, RuntimeError, UnicodeError, TimeoutError) as error:
        report.update(status="failed", reason=type(error).__name__)
    report["elapsed_seconds"] = time.monotonic() - started
    with (directory / "discovery_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return report
