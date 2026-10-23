"""Round 6: is the actual host INNER fit ready for the frozen round 4 OOF candidate?

Offline developer tool. It builds the real ResearchWorkspace exactly as the v3
host does (agent/research_v3.py:265 prepare_research: automatic outer split,
its research fit, then the actual INNER fit/held) on a temporary copy of the
task, checks that this INNER fit is the table PR #51 hands to its preflight,
identity probe and measure_inner, and audits it with the accepted round 5
audit_oof_fit. It trains no model, starts no identity probe or training
subprocess and makes no API request. Structural readiness does not mean that
OOF improves Macro F1.

    python experiment_round6_inner_readiness.py --task-md data/classification/task.md
        --output-dir ../round6-reports/classification --control-repeat
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import time
from unittest import mock

import numpy as np
import pandas as pd

from agent import research_workspace
from agent.research_workspace import ResearchWorkspace
from agent.round4_classification_candidate import make_recipe
from agent.round4_classification_probability import fixed_incumbent_recipe
import experiment_round4_feature_passport as passport
import experiment_round5_data_readiness as readiness
from offline_baseline import find_data


SCHEMA_VERSION = "round6-inner-readiness/1"
SCOPE = "actual_host_inner_fit_only_oof_structure"
FILES = ("round6_inner_readiness_report.json", "round6_inner_readiness_support.csv",
         "round6_inner_readiness_summary.md")
DECISIONS = ("refused", "inner_unavailable", *readiness.FIT_DECISIONS)
SCRATCH_PREFIX = "r6-inner-copy-"  # temporary task copy and host workspace; never exported
FROZEN_PUBLIC_RECIPE_SHA256 = "cde4de874b1b236c9dc485d25b4be01a59788666b91ffe8f4341069525fa9dfc"
FROZEN_PUBLIC_CLASSES = ("Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic", "Normal",
                         "Reconnaissance", "Shellcode", "Worms")
MIN_GROUPS_PER_CLASS = 2  # agent/splits.py:89: a grouped split needs every class in two raw-feature groups
PR51 = {"branch": "origin/work/round5-oof-host-v1", "commit": "25e5967",
        "file": "agent/round5_classification_host.py", "use": "read-only source; not imported or run"}
# PR #51 :97-103 checks these shapes for the INNER, outer and full-refit phases before its preflight.
PR51_CAPS = {"fit_rows": 100_000, "predict_rows": 500_000, "features": 512, "cells": 10_000_000}

STATEMENT = ("Фактический INNER fit канонического host v3 на временной копии задачи: авто outer, его research fit, "
             "INNER fit/held. Это не INNER-контекст для Qwen и не допуск модели. Модели не обучаются, identity probe "
             "не запускается, API не вызывается. Структурная готовность не означает, что OOF повысит Macro F1.")
SOURCES = {
    "host_path": "agent/research_v3.py:264-265 prepare_research -> ResearchWorkspace(task_path, workspace)",
    "task_and_data": "agent/research_workspace.py:290-297 load_task, data_fingerprint, load_task_data",
    "outer_split": "agent/research_workspace.py:299-301 choose_validation, split_rows; agent/automl.py:158-163",
    "outer_fields": "agent/research_workspace.py:323-338 outer_fit_rows, outer_holdout_rows, research_train",
    "inner_fields": "agent/research_workspace.py:340-349 inner_split_rows -> fit_rows, held_rows; agent/search.py:19-36",
    "pr51_preflight_table": "PR #51 25e5967 agent/round5_classification_host.py:169-171 "
                            "gate.research_train.iloc[gate.fit_rows]",
    "pr51_probe_and_oof": "PR #51 25e5967 agent/round5_classification_host.py:360, :397 -> "
                          "agent/research_v3.py:231-238 measure_inner(fit_frame=research_train.iloc[fit_rows])",
    "candidate_view": "agent/research_candidate.py:299-300 fit.csv; driver :214 dtype=str keep_default_na=False, :224",
    "recipe": "PR #51 25e5967 :318 checked_recipe(make_recipe(fixed_incumbent_recipe(), classes, variant='oof'))",
    "audit": "experiment_round5_data_readiness.py:655 audit_oof_fit (PR #52)",
    "coverage": "agent/round4_classification_candidate.py:84-120 coverage",
    "gates": "agent/oof_classification.py:296-320; agent/classification_decision.py:32-39 DEFAULT_GUARD_OPTIONS",
    "raw_groups": "agent/splits.py:83 pandas.util.hash_pandas_object(train[features], index=False)",
}
TEST_READS = [
    "load_task -> load_data открывает и проверяет test.csv (agent/task_spec.py:285-286, offline_baseline.py:128-145).",
    "load_task_data снова читает test.csv; data_fingerprint хеширует его (agent/research_workspace.py:292-294, :438).",
    "choose_validation для классификации смотрит только на дубли признаков TRAIN (agent/automl.py:158-163); "
    "для регрессии она сравнивает периоды TEST (agent/automl.py:164-182).",
    "Ограничения PR #51 по форме полного refit используют число строк TEST (PR #51 :97-103).",
    "audit_oof_fit не читает файлов и TEST: только переданные X_fit, y_fit, context.",
]
LIMITATIONS = [
    "Группа — хеш всех raw-признаков строки, как в agent.splits.split_rows; копии строки с любыми labels "
    "остаются в одной группе.",
    "Время вложенных вызовов внутри конструктора ResearchWorkspace и audit_oof_fit измерено обёртками вызова "
    "без изменения кода; этапы эксклюзивны, их сумма не превышает общего времени.",
    "Время одного прогона на этом компьютере. Стартовый baseline-поиск host, LLM-диалог, dependency_snapshot и "
    "subprocess preflight PR #51 не входят: они обучают модели или запускают процессы.",
    "Память — пик этого процесса за весь прогон, если платформа его отдаёт; это не RSS дерева процессов кандидата.",
]
NOT_EVALUATED_EXTRA = [
    {"id": "identity_probe_seconds", "status": "unknown", "requirement": "identity probe ≤ 120 с (PR #51 :21, :360)",
     "reason": "identity probe обучает модель и здесь не запускается"},
    {"id": "cost_admission", "status": "unknown",
     "requirement": "4·probe·1.5 + max(120, 2·probe) ≤ доступного research-времени (PR #51 :377-383)",
     "reason": "нужен измеренный wall time identity probe"},
    {"id": "startup_baseline", "status": "unknown", "requirement": "измеренный baseline до OOF (PR #51 :76-77)",
     "reason": "стартовый поиск host обучает модели (agent/research_v3.py prepare_baseline)"},
    {"id": "pr51_preflight_subprocess", "status": "not_evaluated",
     "requirement": "preflight PR #51 ≤ 60 с в отдельном процессе (PR #51 :22, :157-194)",
     "reason": "код PR #51 не запускается; та же coverage измерена в этом процессе"},
    {"id": "platform_admission", "status": "unknown", "requirement": "приём решения платформой",
     "reason": "платформа не вызывается"},
]


class Stopwatch:
    """Wall time of named stages and of wrapped calls, plus the last result of each wrapped call."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.results: dict[str, object] = {}

    def add(self, name: str, started: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - started

    @contextlib.contextmanager
    def stage(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, started)

    @contextlib.contextmanager
    def watching(self, module, labels: dict[str, str]):
        """Time calls a module makes through its own globals; arguments and results pass unchanged."""
        def timed(label, function):
            def wrapper(*args, **kwargs):
                started = time.perf_counter()
                try:
                    result = function(*args, **kwargs)
                finally:
                    self.add(label, started)
                self.results[label] = result
                return result
            return wrapper

        with contextlib.ExitStack() as stack:
            for attribute, label in labels.items():
                stack.enter_context(mock.patch.object(module, attribute, timed(label, getattr(module, attribute))))
            yield


# ---------------------------------------------------------------- inputs

def task_files(task_md: Path) -> tuple[Path, list[Path]]:
    """task.md and every file of the data folder the host would find (offline_baseline.find_data)."""
    task_md = Path(task_md).resolve()
    if not task_md.is_file():
        raise ValueError("task.md не найден.")
    data_dir = find_data(task_md.parent)
    return task_md.parent, [task_md, *sorted(path for path in data_dir.rglob("*") if path.is_file())]


def file_hashes(root: Path, files: list[Path]) -> dict[str, dict]:
    return {path.relative_to(root).as_posix(): {"sha256": readiness.file_sha256(path), "bytes": path.stat().st_size}
            for path in files}


def copy_task(task_md: Path, destination: Path) -> tuple[Path, dict]:
    """A byte copy with the same relative layout; the source is only read."""
    task_dir, files = task_files(task_md)
    for path in files:
        target = destination / path.relative_to(task_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    copied = [destination / path.relative_to(task_dir) for path in files]
    return destination / Path(task_md).name, file_hashes(destination, copied)


def row_hashes(frame: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<u8").tobytes()).hexdigest()


# ---------------------------------------------------------------- host INNER

def build_host_workspace(task_path: Path, workspace: Path, watch: Stopwatch) -> ResearchWorkspace:
    """The v3 host call itself (agent/research_v3.py:265); nested calls are timed, never replaced."""
    labels = {"load_task": "task_spec_load", "load_task_data": "data_load",
              "choose_validation": "outer_choose_validation", "split_rows": "outer_split_rows",
              "inner_split_rows": "inner_split_rows"}
    with watch.watching(research_workspace, labels), watch.stage("workspace_constructor"):
        return ResearchWorkspace(task_path, workspace)


def fit_contract(gate: ResearchWorkspace, scratch: Path, watch: Stopwatch) -> dict:
    """The INNER fit as the candidate receives it, checked against the PR #51 preflight table."""
    spec = gate.research_spec
    features, target, id_column = list(spec.features), spec.target_column, spec.id_column
    with watch.stage("fit_table_extraction"):
        # PR #51 :169-171; its identity probe (:360) and OOF (:397) use the same rows via measure_inner.
        preflight = gate.research_train.iloc[gate.fit_rows]
        staging = Path(tempfile.mkdtemp(prefix="fit-", dir=scratch))
        path = staging / "fit.csv"
        # agent/research_candidate.py:299-300 and its driver (:214, :224).
        preflight.loc[:, [id_column, *features, target]].to_csv(path, index=False)
        fit_csv_sha256 = readiness.file_sha256(path)
        fit = pd.read_csv(path, dtype=str, keep_default_na=False)
        shutil.rmtree(staging)
        X, y = fit.loc[:, features].copy(), fit[target].copy()
    with watch.stage("fit_table_equivalence"):
        hashes = row_hashes(X)
        record = {
            "source": SOURCES["pr51_preflight_table"], "candidate_view": SOURCES["candidate_view"],
            "rows": int(len(X)), "feature_count": len(features), "features_sha256": passport.digest(features),
            "row_hashes_sha256": array_sha256(hashes), "labels_sha256": passport.digest(y.tolist()),
            "ids_sha256": passport.digest(fit[id_column].tolist()), "fit_csv_sha256": fit_csv_sha256,
            "same_features": list(X.columns) == features == list(preflight.loc[:, features].columns),
            "same_rows_order_and_values": (len(X) == len(preflight)
                                           and np.array_equal(hashes, row_hashes(preflight.loc[:, features]))
                                           and X.equals(preflight.loc[:, features].reset_index(drop=True))),
            "same_labels": y.tolist() == preflight[target].tolist(),
            "same_ids": fit[id_column].tolist() == preflight[id_column].tolist(),
        }
    return {"X": X, "y": y, "record": record}


def fit_context(gate: ResearchWorkspace) -> dict:
    """The context PR #51 builds (:310-313, :318) and run_candidate passes, with the frozen OOF recipe."""
    spec, host = gate.research_spec, gate.spec
    classes = list(spec.classes)
    if classes != list(host.classes):
        raise ValueError("Отказ: порядок классов research_spec не совпадает с TaskSpec host.")
    if ((list(spec.features), list(spec.numeric), list(spec.categorical))
            != (list(gate.features), list(gate.numeric), list(gate.categorical))):
        raise ValueError("Отказ: признаки и роли research_spec не совпадают с данными host.")
    return {"features": list(spec.features), "numeric": list(spec.numeric), "categorical": list(spec.categorical),
            "classes": classes, "task_type": spec.metric.task_type, "prediction_kind": spec.prediction_kind,
            "metric": spec.metric.to_dict(), "id_column": spec.id_column, "target_column": spec.target_column,
            "recipe": make_recipe(fixed_incumbent_recipe(), classes, variant="oof")}


def _positions(values, size: int) -> np.ndarray | None:
    """Integer positions inside range(size) without repeats, or None."""
    try:
        rows = np.asarray(values, dtype=np.int64)
    except (TypeError, ValueError, OverflowError):
        return None
    if rows.ndim != 1 or (len(rows) and ((rows < 0).any() or (rows >= size).any())) or len(np.unique(rows)) != len(rows):
        return None
    return rows


def _shared(groups: np.ndarray, first: np.ndarray, second: np.ndarray) -> int:
    return len(np.intersect1d(groups[first], groups[second]))


def nested_boundaries(gate: ResearchWorkspace, contract: dict, partition: dict | None) -> list[dict]:
    """Recount every nested boundary on actual TRAIN positions and raw-feature groups; nothing is trusted."""
    checks: list[dict] = []

    def check(check_id: str, status: str, **counts) -> None:
        checks.append({"id": check_id, "status": status, **counts})

    def verdict(ok: bool) -> str:
        return "passed" if ok else "violated"

    train, n = gate.train, len(gate.train)
    groups = row_hashes(train.loc[:, list(gate.features)])
    outer_fit, outer_held = _positions(gate.outer_fit_rows, n), _positions(gate.outer_holdout_rows, n)
    if outer_fit is None or outer_held is None:
        check("outer_partition", "violated", reason="invalid_positions")
        return checks
    union = np.union1d(outer_fit, outer_held)
    check("outer_partition", verdict(not len(np.intersect1d(outer_fit, outer_held)) and len(union) == n),
          fit_rows=int(len(outer_fit)), holdout_rows=int(len(outer_held)),
          shared_rows=int(len(np.intersect1d(outer_fit, outer_held))), uncovered_rows=int(n - len(union)))
    research = gate.research_train
    same = len(research) == len(outer_fit) and np.array_equal(row_hashes(research), row_hashes(train.iloc[outer_fit]))
    check("research_is_outer_fit", verdict(same), rows=int(len(research)))
    grouped_outer = gate.choice.get("strategy") == "grouped"
    check("outer_groups_disjoint", verdict(_shared(groups, outer_fit, outer_held) == 0) if grouped_outer
          else "not_applicable", shared_groups=_shared(groups, outer_fit, outer_held),
          strategy=gate.choice.get("strategy"))
    inner_fit, inner_held = _positions(gate.fit_rows, len(research)), _positions(gate.held_rows, len(research))
    if inner_fit is None or inner_held is None or not same:
        check("inner_partition", "violated", reason="invalid_positions_or_research_view")
        return checks
    inner_union = np.union1d(inner_fit, inner_held)
    check("inner_partition", verdict(not len(np.intersect1d(inner_fit, inner_held)) and len(inner_union) == len(research)),
          fit_rows=int(len(inner_fit)), held_rows=int(len(inner_held)),
          shared_rows=int(len(np.intersect1d(inner_fit, inner_held))),
          uncovered_rows=int(len(research) - len(inner_union)))
    fit_train, held_train = outer_fit[inner_fit], outer_fit[inner_held]
    check("inner_outside_outer_holdout",
          verdict(not len(np.intersect1d(fit_train, outer_held)) and not len(np.intersect1d(held_train, outer_held))),
          fit_rows_in_holdout=int(len(np.intersect1d(fit_train, outer_held))),
          held_rows_in_holdout=int(len(np.intersect1d(held_train, outer_held))))
    grouped_inner = isinstance(gate.inner_split, dict) and gate.inner_split.get("strategy") == "grouped"
    check("inner_groups_disjoint", verdict(_shared(groups, fit_train, held_train) == 0) if grouped_inner
          else "not_applicable", shared_groups=_shared(groups, fit_train, held_train),
          strategy=gate.inner_split.get("strategy") if isinstance(gate.inner_split, dict) else None)
    X, y = contract["X"], contract["y"]
    target = gate.spec.target_column
    audited = (len(X) == len(fit_train) and np.array_equal(row_hashes(X), groups[fit_train])
               and y.tolist() == train[target].iloc[fit_train].tolist())
    if partition is not None:
        audited = audited and np.array_equal(np.asarray(partition["groups"], dtype=np.uint64), groups[fit_train])
    check("audit_input_is_inner_fit", verdict(audited), rows=int(len(X)))
    if partition is None or not partition.get("available"):
        check("ab_and_folds", "not_reached", reason="grouped_split_unavailable" if partition else "audit_not_run")
        return checks
    local = len(X)
    A, B = _positions(partition["tuning_rows"], local), _positions(partition["selection_rows"], local)
    if A is None or B is None:
        check("ab_partition", "violated", reason="invalid_positions")
        return checks
    A_train, B_train = fit_train[A], fit_train[B]
    ab_union = np.union1d(A, B)
    check("ab_partition", verdict(not len(np.intersect1d(A, B)) and len(ab_union) == local
                                  and _shared(groups, A_train, B_train) == 0),
          A_rows=int(len(A)), B_rows=int(len(B)), shared_rows=int(len(np.intersect1d(A, B))),
          uncovered_rows=int(local - len(ab_union)), shared_groups=_shared(groups, A_train, B_train))
    held_times = np.zeros(local, dtype=np.int64)
    folds_ok = len(partition["folds"]) == 3
    for index, fold in enumerate(partition["folds"], 1):
        fit, held = _positions(fold["fit_rows"], local), _positions(fold["held_rows"], local)
        if fit is None or held is None:
            check(f"fold_{index}", "violated", reason="invalid_positions")
            folds_ok = False
            continue
        np.add.at(held_times, held, 1)
        outside = int(len(np.setdiff1d(fit, A)) + len(np.setdiff1d(held, A)))
        shared_rows, shared_groups = int(len(np.intersect1d(fit, held))), _shared(groups, fit_train[fit], fit_train[held])
        ok = not outside and not shared_rows and not shared_groups
        folds_ok = folds_ok and ok
        check(f"fold_{index}", verdict(ok), fit_rows=int(len(fit)), held_rows=int(len(held)),
              rows_outside_A=outside, shared_rows=shared_rows, shared_groups=shared_groups)
    once = bool((held_times[A] == 1).all()) and not held_times[B].any()
    check("folds_hold_A_once", verdict(folds_ok and once), fold_count=len(partition["folds"]),
          A_rows_held_once=int((held_times[A] == 1).sum()), B_rows_held=int((held_times[B] > 0).sum()))
    return checks


# ---------------------------------------------------------------- aggregation

def support_table(audit: dict) -> list[dict]:
    """Rows and raw groups per class and part, with the binding threshold of that part and its margin."""
    options = audit["recipe"]["guard_options"]
    rules = {"allowed": ("groups", MIN_GROUPS_PER_CLASS, "agent/splits.py:89"),
             "A": ("rows", options["min_tuning_class_count"], "agent/oof_classification.py:303 min_tuning_class_count"),
             "B": ("rows", options["min_selection_class_count"],
                   "agent/oof_classification.py:319 min_selection_class_count"),
             "fit": ("rows", 1, "agent/oof_classification.py:305 fit_missing_class")}
    result = []
    for part, counts in readiness.support_parts(audit):
        kind, threshold, source = rules.get("fit" if part.endswith("_fit") else part, (None, None, None))
        for label in audit["classes"]:
            observed = counts[label]
            row = {"part": part, "class": label, "rows": observed["rows"], "groups": observed["groups"],
                   "threshold_on": kind, "threshold": threshold, "source": source,
                   "margin": None if kind is None else observed[kind] - threshold}
            result.append(row)
    return result


def minimum_margin(rows: list[dict]) -> dict | None:
    bounded = [row for row in rows if row["margin"] is not None]
    if not bounded:
        return None
    lowest = min(row["margin"] for row in bounded)
    return {"margin": lowest, "at": [{key: row[key] for key in ("part", "class", "threshold_on", "threshold",
                                                                "rows", "groups")}
                                     for row in bounded if row["margin"] == lowest]}


def host_prerequisites(gate: ResearchWorkspace) -> list[dict]:
    """Structural parts of PR #51 cheap_eligibility (:64-106), recomputed here; runtime parts stay unknown."""
    spec = gate.research_spec
    classes, features = list(spec.classes), list(spec.features)
    shapes = {"inner": [len(gate.fit_rows), len(gate.held_rows)],
              "outer": [len(gate.outer_fit_frame), len(gate.outer_holdout_frame)],
              "full_refit": [len(gate.train), len(gate.test)]}
    largest = max(max(shape) for shape in shapes.values())
    items = [
        ("macro_f1_labels", spec.metric.task_type == "classification" and spec.metric.name == "Macro F1"
         and spec.prediction_kind == "labels", spec.metric.name, ":71-73"),
        ("task_semantics", gate.manifest.get("task_semantics", {}).get("adaptive_research_allowed") is not False,
         gate.manifest.get("task_semantics", {}).get("adaptive_research_allowed"), ":74-75"),
        ("inner_available", bool(gate.fit_rows and gate.held_rows), bool(gate.held_rows), ":78-79"),
        ("grouped_policy", gate.choice.get("strategy") == "grouped" and isinstance(gate.inner_split, dict)
         and gate.inner_split.get("strategy") == "grouped",
         [gate.choice.get("strategy"), gate.inner_split.get("strategy") if isinstance(gate.inner_split, dict) else None],
         ":80-82"),
        ("class_count_and_order", 2 <= len(classes) <= 50 and len(set(classes)) == len(classes)
         and classes == list(gate.spec.classes), len(classes), ":83-87"),
        ("feature_roles", features == list(gate.features) and list(spec.numeric) == list(gate.numeric)
         and list(spec.categorical) == list(gate.categorical) and set(spec.numeric) | set(spec.categorical) == set(features)
         and not set(spec.numeric) & set(spec.categorical), len(features), ":88-95"),
        ("row_caps", all(fit <= PR51_CAPS["fit_rows"] and predict <= PR51_CAPS["predict_rows"]
                         for fit, predict in shapes.values()), shapes, ":97-101"),
        ("feature_matrix_caps", len(features) <= PR51_CAPS["features"] and largest * len(features) <= PR51_CAPS["cells"],
         {"features": len(features), "largest_rows": largest, "cells": largest * len(features)}, ":102-103"),
    ]
    result = [{"id": name, "status": "passed" if ok else "violated", "observed": observed,
               "source": f"PR #51 {PR51['commit']} {PR51['file']}{lines}"} for name, ok, observed, lines in items]
    result += [{"id": name, "status": "unknown", "observed": None, "source": f"PR #51 {PR51['commit']} {PR51['file']}{lines}"}
               for name, lines in (("adaptive_search_on", ":68-69"), ("measured_baseline", ":76-77"),
                                   ("probe_reserve", ":104-105, :345-346"), ("ordinary_catboost_slot", ":306-308"),
                                   ("identity_probe_completed", ":359-376"), ("cost_admission", ":377-383"))]
    return result


def first_fit(decision: str, reasons: list[str], audit: dict | None, prerequisites: list[dict],
              gate: ResearchWorkspace) -> dict:
    """Which check decides whether the first OOF callback (fold 1 fit) would be reached."""
    host = "violated" if any(item["status"] == "violated" for item in prerequisites) else "passed"
    runtime = [item["id"] for item in prerequisites if item["status"] == "unknown"]
    base = {"host_prerequisites": host, "runtime": "unknown", "runtime_conditions": runtime}
    if decision == "refused":
        return {**base, "structural": "not_reached", "deciding_check": reasons[0] if reasons else None,
                "source": "nested boundaries / fit table equivalence / contract"}
    if decision == "inner_unavailable":
        return {**base, "structural": "not_reached", "deciding_check": "inner_split_rows",
                "reason": gate.inner_unavailable_reason, "source": SOURCES["inner_fields"]}
    blocking = next((item for item in audit["checks"] if item["status"] != "passed"), None)
    if blocking is not None:
        return {**base, "structural": "blocked", "deciding_check": blocking["id"], "reason": blocking["reason"],
                "source": blocking["source"]}
    return {**base, "structural": "starts", "deciding_check": "oof_support_gates",
            "reason": "first_fit_reached_without_training",
            "source": "agent/oof_classification.py:296-320; последний gate перед первым callback — класс в B ≥ "
                      "min_selection_class_count (:319)"}


def recipe_record(audit: dict) -> dict:
    sha, classes = audit["recipe"]["sha256"], list(audit["classes"])
    same_classes = tuple(classes) == FROZEN_PUBLIC_CLASSES
    matches = sha == FROZEN_PUBLIC_RECIPE_SHA256
    explanation = None
    if not matches:
        explanation = ("Состав или порядок классов отличается от публичной классификации; recipe зависит только от "
                       "классов (make_recipe(fixed_incumbent_recipe(), classes)). Рецепт не подменяется."
                       if not same_classes else
                       "Классы совпадают с публичной классификацией, но hash другой: код recipe изменился после заморозки.")
    return {"variant": audit["recipe"]["variant"], "sha256": sha, "frozen_public_sha256": FROZEN_PUBLIC_RECIPE_SHA256,
            "matches_frozen_public": matches, "classes_equal_frozen_public": same_classes,
            "explanation": explanation, "source": SOURCES["recipe"]}


def audit_contract(gate: ResearchWorkspace, contract: dict | None, watch: Stopwatch) -> dict:
    """Everything about the frozen fit contract; reads no file and no TEST."""
    if contract is None:
        return {"decision": "inner_unavailable", "reasons": [f"inner_split_rows:{gate.inner_unavailable_reason}"],
                "recipe": None, "audit": None, "boundaries": nested_boundaries_without_inner(gate),
                "support": [], "minimum_margin": None}
    reasons = [f"fit_table:{key}" for key in ("same_features", "same_rows_order_and_values", "same_labels", "same_ids")
               if not contract["record"][key]]
    audit = partition = None
    try:
        context = fit_context(gate)
        watch.results.pop("coverage", None)
        with watch.watching(readiness, {"coverage": "coverage"}), watch.stage("audit_oof_fit"):
            audit = readiness.audit_oof_fit(contract["X"], contract["y"], context)
        partition = watch.results.get("coverage")
    except ValueError as error:
        reasons.append(f"contract:{error}")
    with watch.stage("boundary_checks"):
        boundaries = nested_boundaries(gate, contract, partition)
    reasons += [f"boundary:{item['id']}" for item in boundaries if item["status"] == "violated"]
    if audit is not None and any(check["status"] == "violated" and check["id"] in {"ab_disjoint", "folds_disjoint",
                                                                                   "held_cover_A_once"}
                                 for check in audit["checks"]):
        reasons.append("audit:partition_layout")
    if reasons or audit is None:
        decision = "refused"
    else:
        decision, reasons = audit["decision"], list(audit["reasons"])
    support = support_table(audit) if audit is not None else []
    return {"decision": decision, "reasons": reasons, "recipe": recipe_record(audit) if audit else None,
            "audit": audit, "boundaries": boundaries, "support": support, "minimum_margin": minimum_margin(support)}


def nested_boundaries_without_inner(gate: ResearchWorkspace) -> list[dict]:
    """Outer checks still apply when the host has no INNER."""
    empty = {"X": pd.DataFrame(columns=list(gate.features)), "y": pd.Series([], dtype=object)}
    return [item for item in nested_boundaries(gate, empty, None)
            if item["id"] in {"outer_partition", "research_is_outer_fit", "outer_groups_disjoint"}]


# ---------------------------------------------------------------- one preparation

def inspect(task_md: Path, scratch: Path, watch: Stopwatch) -> dict:
    """One model-free host preparation on a fresh copy; returns the deterministic report content."""
    task_dir, files = task_files(task_md)
    with watch.stage("task_copy_and_hashes"):
        inputs = file_hashes(task_dir, files)
        copy_path, copied = copy_task(task_md, scratch / "task")
    gate = build_host_workspace(copy_path, scratch / "workspace", watch)
    contract = fit_contract(gate, scratch, watch) if gate.fit_rows and gate.held_rows else None
    result = audit_contract(gate, contract, watch)
    with watch.stage("source_checks"):
        try:
            gate.check_source()
            copy_unchanged = True
        except (OSError, ValueError):
            copy_unchanged = False
        unchanged = file_hashes(task_dir, files) == inputs
    prerequisites = host_prerequisites(gate)
    return {
        "schema_version": SCHEMA_VERSION, "scope": SCOPE, "statement": STATEMENT,
        "source_commit": passport.source_commit(), "pr51_source": PR51,
        "inputs": inputs, "inputs_unchanged": unchanged, "copy_matches_inputs": copied == inputs,
        "copy_unchanged_during_run": copy_unchanged,
        "host": {"path": SOURCES["host_path"],
                 "outer": {"strategy": gate.choice.get("strategy"), "validation_reason": gate.validation_reason,
                           "split": gate.split, "fit_rows": len(gate.outer_fit_rows),
                           "holdout_rows": len(gate.outer_holdout_rows)},
                 "inner": {"available": bool(gate.held_rows), "unavailable_reason": gate.inner_unavailable_reason,
                           "split": gate.inner_split, "fit_rows": len(gate.fit_rows), "held_rows": len(gate.held_rows)},
                 "test_reads": TEST_READS},
        "fit_table": contract["record"] if contract else None,
        "decision": result["decision"], "reasons": result["reasons"],
        "first_fit": first_fit(result["decision"], result["reasons"], result["audit"], prerequisites, gate),
        "recipe": result["recipe"], "host_prerequisites": prerequisites, "boundaries": result["boundaries"],
        "support": result["support"], "minimum_margin": result["minimum_margin"],
        "part_constraints": result["audit"]["constraints"] if result["audit"] else [],
        "audit": result["audit"],
        "not_evaluated": [*(item for item in (result["audit"] or {}).get("not_evaluated", [])
                            if item["id"] != "host_inner_equivalence"), *NOT_EVALUATED_EXTRA],
        "sources": SOURCES, "limitations": LIMITATIONS,
    }


def stages(watch: Stopwatch) -> dict:
    """Exclusive stage times: nested wrapped calls are subtracted from the stage that contains them."""
    s = watch.seconds.get
    load = s("task_spec_load", 0.0) + s("data_load", 0.0)
    constructor = s("workspace_constructor", 0.0)
    inner_split = s("inner_split_rows", 0.0)
    coverage = s("coverage", 0.0)
    result = {
        "task_copy_and_hashes": s("task_copy_and_hashes", 0.0),
        "task_spec_and_data_load": load,
        "outer_workspace": constructor - load - inner_split,
        "inner_extraction": inner_split + s("fit_table_extraction", 0.0),
        "fit_table_equivalence": s("fit_table_equivalence", 0.0),
        "coverage": coverage,
        "guard_cap_audit": s("audit_oof_fit", 0.0) - coverage,
        "boundary_checks": s("boundary_checks", 0.0),
        "source_checks": s("source_checks", 0.0),
    }
    detail = {"outer_choose_validation": s("outer_choose_validation", 0.0),
              "outer_split_rows": s("outer_split_rows", 0.0), "inner_split_rows": inner_split,
              "fit_table_csv_staging": s("fit_table_extraction", 0.0)}
    host = constructor + coverage  # the constructor already contains the load and the INNER split
    return {"seconds": {key: round(value, 3) for key, value in result.items()},
            "of_which": {key: round(value, 3) for key, value in detail.items()},
            "host_preparation_model_free_seconds": round(host, 3),
            "host_preparation_scope": "TaskSpec/data load + ResearchWorkspace (outer, INNER split, файлы workspace) + "
                                      "coverage на INNER fit; без стартового baseline, LLM, dependency_snapshot и "
                                      "subprocess preflight PR #51"}


# ---------------------------------------------------------------- rendering

def render_support_csv(report: dict) -> str:
    return readiness.render_csv(("part", "class", "rows", "groups", "threshold_on", "threshold", "margin", "source"), [
        [row["part"], row["class"], row["rows"], row["groups"], row["threshold_on"], row["threshold"], row["margin"],
         row["source"]] for row in report["support"]])


def render_summary(report: dict) -> str:
    md, table, plain, fmt = passport.md, passport.table, readiness.plain, readiness.fmt
    host, fit, first = report["host"], report["fit_table"], report["first_fit"]
    lines = [f"# Готовность host INNER к OOF: {report['decision']}", "", report["statement"], ""]
    rows = [["scope", report["scope"]], ["Решение", report["decision"]],
            ["Причины", ", ".join(plain(reason) for reason in report["reasons"]) or "—"],
            ["Первый OOF fit (структурно)", f"{first['structural']} — {first['deciding_check']}"],
            ["Предусловия PR #51 / runtime", f"{first['host_prerequisites']} / {first['runtime']}"],
            ["Outer", f"{host['outer']['strategy']}: fit {host['outer']['fit_rows']}, holdout {host['outer']['holdout_rows']}"],
            ["INNER", f"fit {host['inner']['fit_rows']}, held {host['inner']['held_rows']}"
                      + (f"; недоступен: {plain(host['inner']['unavailable_reason'])}" if not host["inner"]["available"] else "")],
            ["Входы не изменились", fmt(report["inputs_unchanged"])], ["content_sha256", f"`{report['content_sha256']}`"]]
    if report["recipe"]:
        rows.insert(5, ["recipe_sha256", f"`{report['recipe']['sha256']}` (замороженный: "
                                         f"{fmt(report['recipe']['matches_frozen_public'])})"])
    lines += table(["Поле", "Значение"], rows)
    if report["recipe"] and report["recipe"]["explanation"]:
        lines += [f"Recipe: {plain(report['recipe']['explanation'])}", ""]
    if fit:
        lines += ["## Точный fit", "", f"Источник: {plain(fit['source'])}. Вид кандидата: {plain(fit['candidate_view'])}.", ""]
        lines += table(["Поле", "Значение"], [
            ["Строк / признаков", f"{fit['rows']} / {fit['feature_count']}"],
            ["Совпадение с таблицей PR #51: признаки / строки и порядок / метки / ID",
             " / ".join(fmt(fit[key]) for key in ("same_features", "same_rows_order_and_values", "same_labels", "same_ids"))],
            ["row_hashes_sha256", f"`{fit['row_hashes_sha256']}`"], ["fit_csv_sha256", f"`{fit['fit_csv_sha256']}`"]])
    lines += ["## Вложенные границы", ""]
    lines += table(["Проверка", "Статус", "Счётчики"], [
        [item["id"], item["status"], plain(", ".join(f"{key}={value}" for key, value in item.items()
                                                    if key not in {"id", "status"}) or "—")]
        for item in report["boundaries"]])
    if report["support"]:
        margin = report["minimum_margin"]
        lines += ["## Минимальный запас", "", f"Запас {margin['margin']}: " + ", ".join(
            f"{item['part']} / {md(item['class'])} ({item['threshold_on']} ≥ {item['threshold']}, строк {item['rows']}, "
            f"групп {item['groups']})" for item in margin["at"]), ""]
        lines += ["## Поддержка классов: строки / группы (запас)", ""]
        parts = [part for part in dict.fromkeys(row["part"] for row in report["support"]) if not part.endswith("_held")]
        index = {(row["part"], row["class"]): row for row in report["support"]}
        lines += table(["Класс", *parts], [[md(label), *(
            f"{index[(part, label)]['rows']} / {index[(part, label)]['groups']}"
            + (f" ({index[(part, label)]['margin']})" if index[(part, label)]["margin"] is not None else "")
            for part in parts)] for label in report["audit"]["classes"]])
        lines += ["Пороги: allowed — групп ≥ 2; A, B — строк ≥ 5; fold fit — строк ≥ 1. Held-части folds — в CSV.", ""]
        lines += ["## Ограничения кандидата", ""]
        lines += table(["Ограничение", "Порог", "Наблюдение", "Выполнено"], [
            [item["id"], f"{item['limit']} {item['threshold']}", str(item["observed"]), fmt(item["satisfied"])]
            for item in report["part_constraints"]])
    if report["audit"]:
        lines += ["## Проверки audit_oof_fit", ""]
        lines += table(["Проверка", "Статус", "Причина"], [[check["id"], check["status"], plain(check["reason"] or "—")]
                                                         for check in report["audit"]["checks"]])
        duplicates = report["audit"]["duplicates"]
        lines += [f"Дубли raw-признаков: групп с копиями {duplicates['duplicate_groups']}, групп с разными labels "
                  f"{duplicates['conflicting_label_groups']} (строк {duplicates['rows_in_conflicting_groups']}).", ""]
    lines += ["## Предусловия host из PR #51", ""]
    lines += table(["Условие", "Статус", "Источник"], [[item["id"], item["status"], plain(item["source"])]
                                                     for item in report["host_prerequisites"]])
    timing = report["generated"]["timing"]
    lines += ["## Время (с, эксклюзивно)", ""]
    lines += table(["Этап", "Секунды"], [[name, str(value)] for name, value in timing["seconds"].items()])
    lines += [f"Модель-free подготовка host до identity probe: {timing['host_preparation_model_free_seconds']} с "
              f"({plain(timing['host_preparation_scope'])}). Рендер экспорта: {timing['export_render_seconds']} с, "
              "запись трёх файлов на диск не входит.", ""]
    memory = report["generated"]["memory"]
    lines += [f"Пик памяти: {memory['peak_bytes'] if memory['peak_bytes'] is not None else 'unknown'} байт "
              f"({memory['method'] or 'нет счётчика'}, scope {memory['scope']}).", ""]
    repeat = report["generated"].get("control_repeat")
    if repeat:
        lines += [f"Контрольный повтор: content совпал — {fmt(repeat['content_equal'])}, подготовка host "
                  f"{repeat['timing']['host_preparation_model_free_seconds']} с. Исходный результат сохранён.", ""]
    lines += ["## TEST", ""] + [f"- {text}" for text in report["host"]["test_reads"]] + [""]
    lines += ["## Не оценено без обучения", ""]
    lines += table(["Пункт", "Требование", "Статус", "Почему"], [
        [item["id"], plain(item["requirement"]), item["status"], plain(item["reason"])] for item in report["not_evaluated"]])
    lines += ["## Ограничения", ""] + [f"- {text}" for text in report["limitations"]] + [""]
    return "\n".join(lines)


# ---------------------------------------------------------------- export

def prepare(task_md: Path) -> tuple[dict, dict]:
    started = time.perf_counter()
    watch = Stopwatch()
    with tempfile.TemporaryDirectory(prefix=SCRATCH_PREFIX, ignore_cleanup_errors=True) as scratch:
        content = inspect(task_md, Path(scratch), watch)
    timing = stages(watch)
    timing["total_seconds"] = round(time.perf_counter() - started, 3)
    return content, timing


def export(task_md: Path, output_dir: Path, *, control_repeat: bool = False) -> dict:
    task_dir, _ = task_files(task_md)
    output = readiness.check_output(output_dir, [task_dir], FILES)
    report, timing = prepare(task_md)
    report["config"] = {"schema_version": SCHEMA_VERSION, "frozen_public_recipe_sha256": FROZEN_PUBLIC_RECIPE_SHA256,
                        "min_groups_per_class": MIN_GROUPS_PER_CLASS, "pr51_caps": PR51_CAPS,
                        "guard_options": report["audit"]["recipe"]["guard_options"] if report["audit"] else None}
    report["config_sha256"] = passport.digest(report["config"])
    report["content_sha256"] = passport.digest(report)
    generated = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "timing": timing}
    if control_repeat:
        repeat, repeat_timing = prepare(task_md)
        repeat["config"], repeat["config_sha256"] = report["config"], report["config_sha256"]
        generated["control_repeat"] = {"content_equal": passport.digest(repeat) == report["content_sha256"],
                                       "timing": repeat_timing}
    generated["memory"] = readiness.peak_memory()
    report["generated"] = generated
    # The export cost is the CSV render and the JSON serialization; the summary that
    # quotes it and the three small disk writes follow and are not included.
    started = time.perf_counter()
    texts = {"round6_inner_readiness_support.csv": render_support_csv(report)}
    passport.to_json(report)
    generated["timing"]["export_render_seconds"] = round(time.perf_counter() - started, 3)
    texts["round6_inner_readiness_summary.md"] = render_summary(report)
    readiness.write_outputs(output, FILES, report, texts)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-md", type=Path, required=True, help="task.md задачи; данные ищутся рядом, как в host")
    parser.add_argument("--output-dir", type=Path, required=True, help="Папка для трёх файлов, вне папки задачи")
    parser.add_argument("--control-repeat", action="store_true",
                        help="Один контрольный повтор модель-free подготовки на новой копии")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Cyrillic output on Windows consoles and pipes
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        report = export(args.task_md, args.output_dir, control_repeat=args.control_repeat)
    except (OSError, ValueError) as error:
        print(f"Экспорт не выполнен: {error}", file=sys.stderr)
        return 1
    fit = report["fit_table"]
    print(f"INNER OOF: {report['decision']}, fit {fit['rows'] if fit else '—'} строк, первый fit "
          f"{report['first_fit']['structural']}; content_sha256 {report['content_sha256'][:12]}…, "
          f"файлы в {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
