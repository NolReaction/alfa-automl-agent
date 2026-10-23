"""Opt-in early frozen OOF invocation, with paid cost admission and no TPE feedback.

The unchanged Round4 factories own the model. This module owns scheduling,
bounded fit-only preflight, dependency seals and host invocation accounting.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import time

from agent.offline_research import canonical_recipe_key
from agent.research_candidate import CandidateExecutionError, freeze_candidate
from agent.round4_classification_candidate import candidate_source, checked_recipe, file_hash, json_hash, make_recipe
from agent.round4_classification_probability import fixed_incumbent_recipe

PROBE_SECONDS = 120
OOF_SECONDS = 600
PREFLIGHT_SECONDS = 60
DEPENDENCIES = tuple("agent." + name for name in (
    "round5_classification_host", "round4_classification_candidate", "round4_classification_probability",
    "round4_classification_features", "oof_classification", "classification_decision",
    "models", "features", "model_config", "metrics", "splits", "execution", "resources",
    "file_io", "task_spec", "research_candidate", "research_v3", "research_budget",
    "research_workspace", "offline_research", "research_extensions", "search", "adaptive_search",
    "frequency_candidate", "target_transform", "task_contract")) + ("offline_baseline", "validation", "agent")
DISTRIBUTIONS = ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl", "lightgbm", "psutil")


def dependency_snapshot():
    sources = {}
    for name in DEPENDENCIES:
        origin = importlib.util.find_spec(name).origin
        if not origin or not Path(origin).is_file():
            raise ValueError("Round5 dependency source unavailable: " + name)
        sources[name] = {"path": str(Path(origin).resolve()), "sha256": file_hash(origin)}
    return {"sources": sources, "versions": {name: importlib.metadata.version(name) for name in DISTRIBUTIONS}}


def _integrity_error(state, reason):
    state._local_integrity_failed = True
    raise ValueError(reason)


def verify_dependencies(state):
    snapshot = getattr(state, "_round5_snapshot", None)
    if snapshot is None:
        return
    if getattr(state, "_local_integrity_failed", False):
        raise ValueError("Round5 integrity previously failed")
    try:
        if dependency_snapshot() != snapshot["dependencies"]:
            _integrity_error(state, "Round5 dependency snapshot changed")
        for candidate in snapshot["candidates"]:
            candidate.verify()
    except (OSError, ValueError) as error:
        _integrity_error(state, str(error))


def cheap_eligibility(state):
    """No subset materialization, exports, raw hashing or augmentation here."""
    if state.dialogue_closed or state.report.get("stage") != "inner_research":
        return "research_closed"
    if not getattr(state, "adaptive_search", False):
        return "adaptive_required"
    gate, spec = state.gate, state.gate.research_spec
    if (spec.metric.task_type != "classification" or spec.metric.name != "Macro F1"
            or spec.prediction_kind != "labels"):
        return "macro_f1_labels_required"
    if gate.manifest.get("task_semantics", {}).get("adaptive_research_allowed") is False:
        return "task_semantics_unavailable"
    if state.baseline is None or state.baseline_inner is None:
        return "measured_baseline_required"
    if not gate.fit_rows or not gate.held_rows:
        return "inner_validation_unavailable"
    if (gate.choice.get("strategy") != "grouped" or not isinstance(gate.inner_split, dict)
            or gate.inner_split.get("strategy") != "grouped"):
        return "grouped_policy_required"
    classes = list(spec.classes)
    if (not 2 <= len(classes) <= 50 or len(set(classes)) != len(classes)
            or any(not isinstance(value, str) or not value.strip() for value in classes)
            or classes != list(gate.spec.classes)):
        return "class_order_invalid"
    features, numeric, categorical = list(spec.features), list(spec.numeric), list(spec.categorical)
    if (not features or len(set(features)) != len(features)
            or len(set(numeric)) != len(numeric) or len(set(categorical)) != len(categorical)
            or set(numeric) & set(categorical) or set(numeric) | set(categorical) != set(features)
            or {spec.id_column, spec.target_column} & set(features)
            or features != list(gate.features) or numeric != list(gate.numeric)
            or categorical != list(gate.categorical)):
        return "feature_roles_invalid"
    # Check every eventual phase without touching outer values or labels.
    shapes = ((len(gate.fit_rows), len(gate.held_rows)),
              (len(gate.outer_fit_frame), len(gate.outer_holdout_frame)),
              (len(gate.train), len(gate.test)))
    if any(fit > 100_000 or predict > 500_000 for fit, predict in shapes):
        return "row_cap_exceeded"
    if len(features) > 512 or any(max(shape) * len(features) > 10_000_000 for shape in shapes):
        return "feature_matrix_cap_exceeded"
    if state.research_timeout(PROBE_SECONDS) < PROBE_SECONDS:
        return "probe_reserve_unavailable"
    return None


def semantic_slot(recipes, seen, ordinary_code_hash):
    known = set()
    for record in seen:
        try:
            known.add(canonical_recipe_key(record["recipe"], record["code_hash"]))
        except (KeyError, TypeError, ValueError):
            continue
    for recipe in recipes:
        config = recipe.get("model_config", {})
        params = config.get("parameters", {})
        if (recipe.get("action") == "catboost" and config.get("model") == "CatBoostClassifier"
                and params.get("iterations") == 400 and params.get("depth") == 10
                and params.get("learning_rate") == .03
                and canonical_recipe_key(recipe, ordinary_code_hash) not in known):
            return recipe
    return None


def _support_reason(partition, classes):
    if not partition.get("available"):
        return partition.get("reason", "grouped_split_unavailable")
    if len(partition["tuning_rows"]) < 100 or len(partition["selection_rows"]) < 60:
        return "ab_insufficient_rows"
    support = partition["support"]
    if any(support[part][label]["rows"] < 5 for part in ("A", "B") for label in classes):
        return "ab_insufficient_class_support"
    if len(support["folds"]) != 3 or any(fold["fit"][label]["rows"] < 1
                                          for fold in support["folds"] for label in classes):
        return "fold_missing_class"
    return None


def preflight_worker(request_path, output_path):
    """Structural checks use only INNER-fit features and labels, never fits."""
    import joblib
    from agent.round4_classification_candidate import coverage
    request = joblib.load(request_path)
    partition = coverage(request["X"], request["y"], request["context"])
    if partition["available"]:
        groups = partition["groups"]
        A, B = partition["tuning_rows"], partition["selection_rows"]
        if set(groups[i] for i in A) & set(groups[i] for i in B):
            raise ValueError("Preflight A/B raw-group overlap")
    result = {"request_sha256": file_hash(request_path), "partition": partition,
              "reason": _support_reason(partition, request["context"]["classes"])}
    Path(output_path).write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def bounded_preflight(state, context):
    import joblib
    from agent import project_root
    from agent.execution import run_python
    started = time.monotonic()
    allowance = state.research_timeout(PREFLIGHT_SECONDS)
    if allowance <= 0:
        return {"reason": "preflight_reserve_unavailable", "elapsed_seconds": 0}
    directory = state.run_dir / "round5_classification" / "preflight"
    directory.mkdir(parents=True, exist_ok=False)
    request, output = directory / "request.joblib", directory / "result.json"
    gate = state.gate
    fitting = gate.research_train.iloc[gate.fit_rows]
    joblib.dump({"X": fitting.loc[:, context["features"]],
                 "y": fitting[gate.research_spec.target_column], "context": context}, request)
    digest = file_hash(request)
    remaining = min(started + allowance, state.budget.deadline - state.budget.publication_buffer) - time.monotonic()
    if remaining <= 0:
        return {"reason": "preflight_staging_timeout", "elapsed_seconds": time.monotonic() - started}
    code = (f"import sys\nsys.path.insert(0, {str(project_root())!r})\n"
            "from agent.round5_classification_host import preflight_worker\n"
            f"preflight_worker({str(request)!r}, {str(output)!r})\n")
    execution = run_python(code, directory, directory / "logs", 0, timeout=remaining,
                           max_memory_bytes=6 * 1024 ** 3)
    if file_hash(request) != digest:
        _integrity_error(state, "Preflight request changed")
    result = {"elapsed_seconds": time.monotonic() - started, "execution": execution,
              "request_sha256": digest, "result_path": str(output)}
    if (execution.get("exit_code") != 0 or execution.get("timed_out")
            or execution.get("memory_limit_exceeded") or execution.get("error")
            or time.monotonic() >= started + allowance):
        return {**result, "reason": "preflight_failed"}
    value = json.loads(output.read_text(encoding="utf-8"))
    if value["request_sha256"] != digest:
        _integrity_error(state, "Preflight receipt mismatch")
    partition = value["partition"]
    return {**result, "reason": value["reason"], "split_sha256": partition["split_sha256"],
            "support": partition["support"], "structurally_available": partition["available"]}


def invocation_evidence(directory):
    """Include partial/cancelled child events even when the parent has failed."""
    directory = Path(directory)
    events = []
    parse_errors = []
    for path in sorted(directory.glob("round4_work/*/events.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get("status") not in {"started", "completed", "cancelled_before_fit"}:
                    raise ValueError("Unknown fit event")
                events.append({**event, "path": str(path)})
            except (ValueError, TypeError, AttributeError):
                parse_errors.append({"path": str(path), "line": number})
    cancelled = sum(event["status"] == "cancelled_before_fit" for event in events)
    result = {"directory": str(directory), "events": events,
              "physical_fits_started": sum(event["status"] == "started" for event in events) - cancelled,
              "physical_fits_completed": sum(event["status"] == "completed" for event in events),
              "physical_fits_cancelled_before_fit": cancelled}
    if parse_errors:
        result["event_parse_errors"] = parse_errors
        result["observed_fit_counts_lower_bounds"] = {key: result[key] for key in (
            "physical_fits_started", "physical_fits_completed", "physical_fits_cancelled_before_fit")}
        for key in result["observed_fit_counts_lower_bounds"]:
            result[key] = None
    for name in ("context.json", "fit.csv", "predict.csv", "receipt.json", "round4_evidence.json"):
        if (directory / name).is_file():
            result.setdefault("hashes", {})[name] = file_hash(directory / name)
    return result


def verify_result(state, candidate, result):
    snapshot = state._round5_snapshot
    if candidate.code_hash != snapshot["source_hash"]:
        return
    if candidate.recipe_hash not in snapshot["recipe_hashes"]:
        _integrity_error(state, "Round5 unsealed recipe")
    try:
        result.verify()
        directory = result.receipt_path.parent
        context = json.loads((directory / "context.json").read_text(encoding="utf-8"))
        evidence = json.loads((directory / "round4_evidence.json").read_text(encoding="utf-8"))
        for key in ("nonce", "phase", "code_hash", "recipe_hash"):
            if evidence[key] != context[key]:
                raise ValueError("Round5 receipt identity mismatch")
        if (context["code_hash"] != candidate.code_hash or context["recipe_hash"] != candidate.recipe_hash
                or context["classes"] != snapshot["classes"] or evidence["classes"] != snapshot["classes"]
                or checked_recipe(context["recipe"]) != candidate.recipe
                or evidence["status"] != "completed" or evidence["fresh_full_refit"] is not True):
            raise ValueError("Round5 result identity mismatch")
        counts = invocation_evidence(directory)
        cap = candidate.recipe["caps"]["max_physical_fits"]
        if (not 2 <= counts["physical_fits_completed"] <= counts["physical_fits_started"] <= cap
                or any(counts[key] != evidence[key] for key in ("physical_fits_started", "physical_fits_completed"))):
            raise ValueError("Round5 physical fit receipt mismatch")
        final_verified = False
        for child in evidence["child_runs"]:
            meta = child["metadata"]
            subdir = directory / "round4_work" / ("f" + str(child["fit_index"]))
            if meta is None:
                execution = child["execution"]
                if (child["fit_index"] == "final" or not (execution.get("exit_code") != 0
                        or execution.get("timed_out") or execution.get("memory_limit_exceeded")
                        or execution.get("error"))):
                    raise ValueError("Round5 successful child lacks metadata")
                continue
            if (file_hash(subdir / "request.joblib") != meta["request_sha256"]
                    or file_hash(subdir / "probabilities.npy") != meta["probabilities_sha256"]):
                raise ValueError("Round5 child receipt mismatch")
            if child["fit_index"] == "final":
                final_verified = (meta["fit_rows"] == evidence["full_fit_rows"]
                                  and meta["predict_rows"] == evidence["predict_rows"])
        if not final_verified:
            raise ValueError("Round5 fresh full-row final child missing")
        state.report["classification_round4"].setdefault("invocations", []).append({
            **counts, "phase": result.phase, "code_hash": result.code_hash, "recipe_hash": result.recipe_hash,
            "receipt_path": str(result.receipt_path), "receipt_sha256": result.receipt_hash,
            "decision": evidence["decision"], "fresh_full_refit": True,
            "fit_rows": evidence["full_fit_rows"], "predict_rows": evidence["predict_rows"]})
    except (OSError, ValueError, KeyError, TypeError) as error:
        _integrity_error(state, str(error))


def stage_limit(state, candidate, timeout, stage):
    if candidate.recipe_hash == state._round5_snapshot["oof_recipe_hash"] and candidate.code_hash == state._round5_snapshot["source_hash"]:
        entry = state.report["final_stage_budgets"][stage]
        entry["allocator_timeout_seconds"] = timeout
        timeout = min(OOF_SECONDS, timeout)
        entry["timeout_seconds"] = timeout
    return timeout


def run_round5_host(state, recipes, seen, ordinary_code_hash, assess, track):
    report = {"requested_mode": "on", "reached_mode": "off", "status": "skipped", "reason": None,
              "comparison_type": "model_plus_scheduling_plus_paid_probe", "used_invocations": 0,
              "probe": None, "oof": None, "ordinary_slot_removed": False}
    state.report["classification_round4"] = report
    used = 0

    def finish(reason):
        report.update(reason=reason, used_invocations=used, ordinary_remaining_trial_cap=64 - used)
        state.save()
        return recipes, seen, used

    reason = cheap_eligibility(state)
    if reason:
        return finish(reason)
    slot = semantic_slot(recipes, seen, ordinary_code_hash)
    if slot is None:
        return finish("ordinary_cb_slot_absent_or_seen")
    report["replaced_ordinary_key"] = canonical_recipe_key(slot, ordinary_code_hash)
    spec = state.gate.research_spec
    classes = list(spec.classes)
    context = {"classes": classes, "features": list(spec.features), "numeric": list(spec.numeric),
               "categorical": list(spec.categorical)}
    directory = state.run_dir / "round5_classification"
    directory.mkdir(parents=True, exist_ok=False)
    source = directory / "candidate.py"
    source.write_text(candidate_source(), encoding="utf-8")
    oof_recipe = checked_recipe(make_recipe(fixed_incumbent_recipe(), classes, variant="oof"))
    identity_recipe = checked_recipe(make_recipe(fixed_incumbent_recipe(), classes, variant="identity"))
    oof = freeze_candidate(source, oof_recipe, directory / "oof_frozen")
    identity = freeze_candidate(source, identity_recipe, directory / "identity_frozen")
    dependencies = dependency_snapshot()
    snapshot = {"dependencies": dependencies, "candidates": (oof, identity), "source_hash": oof.code_hash,
                "classes": classes, "recipe_hashes": (oof.recipe_hash, identity.recipe_hash),
                "oof_recipe_hash": oof.recipe_hash}
    state._round5_snapshot = snapshot
    manifest = {"dependencies": dependencies, "classes": classes, "source_hash": oof.code_hash,
                "oof_recipe_hash": oof.recipe_hash, "identity_recipe_hash": identity.recipe_hash,
                "oof_recipe": oof.recipe, "identity_recipe": identity.recipe}
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report["manifest"] = str(directory / "manifest.json")
    try:
        report["preflight"] = bounded_preflight(state, context)
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        report["preflight"] = {"reason": "preflight_failed", "error_type": type(error).__name__}
    state.check_source()
    state.gate.check_inputs()
    if report["preflight"]["reason"]:
        return finish(report["preflight"]["reason"])
    known = {canonical_recipe_key(row["recipe"], row["code_hash"]) for row in seen}
    if canonical_recipe_key(oof.recipe, oof.code_hash) in known:
        return finish("custom_already_seen")
    if canonical_recipe_key(identity.recipe, identity.code_hash) in known:
        return finish("identity_seen_without_compatible_cost_receipt")
    if state.research_timeout(PROBE_SECONDS) < PROBE_SECONDS:
        return finish("probe_reserve_unavailable")
    track(state, identity)
    probe = {"status": "started", "cap_seconds": PROBE_SECONDS, "cost_only": True,
             "code_hash": identity.code_hash, "recipe_hash": identity.recipe_hash}
    report["probe"] = probe
    used += 1
    report["used_invocations"] = used
    state.report["local_experiments"] = state.report.get("local_experiments", 0) + 1
    seen = [*seen, {"recipe": identity.recipe, "code_hash": identity.code_hash}]
    state.tried_recipes.append(seen[-1])
    before = set((state.run_dir / "inner_evaluations").glob("*"))
    started = time.monotonic()
    state.save()
    try:
        result = state.measure_inner(identity, maximum=PROBE_SECONDS)
        probe.update(status="completed", receipt_path=str(result.receipt_path), receipt_sha256=result.receipt_hash)
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        probe.update(status="failed", error_type=type(error).__name__)
        if isinstance(error, CandidateExecutionError):
            probe["execution"] = error.execution
            if any(word in str(error).lower() for word in ("receipt", "changed", "mismatch")):
                state._local_integrity_failed = True
    finally:
        probe["elapsed_seconds"] = time.monotonic() - started
        probe["attempts"] = [invocation_evidence(path) for path in sorted(
            set((state.run_dir / "inner_evaluations").glob("*")) - before) if path.is_dir()]
        state.save()
    state.check_source()
    state.gate.check_inputs()
    if probe["status"] != "completed":
        return finish("identity_probe_failed")
    p = probe["elapsed_seconds"]
    estimate = 4 * p * 1.5 + max(120, 2 * p)
    available = state.research_timeout(OOF_SECONDS)
    report["admission"] = {"probe_wall_seconds": p, "estimated_oof_seconds": estimate,
                            "available_research_seconds": available, "guaranteed": False}
    if not math.isfinite(p) or p <= 0 or p > PROBE_SECONDS or estimate > available:
        return finish("oof_cost_admission_failed")
    track(state, oof)
    used += 1
    report["used_invocations"] = used
    state.report["local_experiments"] = state.report.get("local_experiments", 0) + 1
    # Consume the semantic ordinary slot permanently once OOF is attempted.
    recipes = [recipe for recipe in recipes if canonical_recipe_key(recipe, ordinary_code_hash) != report["replaced_ordinary_key"]]
    seen = [*seen, {"recipe": slot, "code_hash": ordinary_code_hash}, {"recipe": oof.recipe, "code_hash": oof.code_hash}]
    report.update(status="attempted", reached_mode="on", ordinary_slot_removed=True)
    report["oof"] = {"code_hash": oof.code_hash, "recipe_hash": oof.recipe_hash, "cap_seconds": OOF_SECONDS}
    before = set((state.run_dir / "inner_evaluations").glob("*"))
    started = time.monotonic()
    state.save()
    try:
        feedback = assess(state, oof, {"hypothesis": "Frozen Round4 OOF early classification bundle."},
                          origin="local", maximum=OOF_SECONDS)
        report["oof"]["feedback"] = feedback
        report["status"] = "completed"
        state.event("round5_classification_experiment", result=feedback)
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        report["oof"].update(error_type=type(error).__name__)
        report["status"] = "failed"
        if isinstance(error, CandidateExecutionError):
            report["oof"]["execution"] = error.execution
            if any(word in str(error).lower() for word in ("receipt", "changed", "mismatch")):
                state._local_integrity_failed = True
    finally:
        report["oof"]["elapsed_seconds"] = time.monotonic() - started
        report["oof"]["attempts"] = [invocation_evidence(path) for path in sorted(
            set((state.run_dir / "inner_evaluations").glob("*")) - before) if path.is_dir()]
        state.save()
    state.check_source()
    state.gate.check_inputs()
    return finish("oof_attempted")
