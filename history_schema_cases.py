"""Synthetic catalog for the history schema proposer: small tables with a hidden true object id.

A case holds raw text columns (str or None, as the host passes them), the keyword arguments of the proposer call,
the expected outcome and the true object id of every row (Case.truth). The truth stays in the case and is never
passed to the proposer; target values do not exist at all. Column names are neutral and unrelated to public data.

    family 1  clean_panel, clean_panel_renamed     a panel of objects; the renamed copy has other names and order
    family 2  composite_key                        two parts that collide alone and identify the objects together
    family 3  long_explicit_key                    a five-part key that only explicit_key_sets can reach
    family 4  no_allowed_period, single_period, unique_objects        must abstain
    family 5  coarse_key, technical_counter        constant/coarse parts with collisions; a row counter like time
    family 6  missing_and_literals                 missing parts, literal "NA" and "?", numeric spellings, periods
                                                   that differ only in the 21st decimal
    family 7  repeated_records                     keys repeated inside past periods
    family 8  deceptive_one_object, deceptive_two_objects   the same X, a different hidden truth

The oracle functions below count the evidence by brute force over pairs of rows, independently of the proposer;
adapter_evidence gets the same counts from the accepted HistoryEncoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import random

import numpy as np
import pandas as pd

from agent.building_history import history_keys, numeric_value
from agent.history_candidate import HistoryConfig


@dataclass(frozen=True)
class Case:
    name: str
    family: int
    raw_columns: dict           # column -> list of str or None
    call: dict                  # keyword arguments of propose_history_schemas except raw_columns
    truth: list                 # hidden object id per row; never passed to the proposer
    expect: str                 # "propose" or "abstain"
    admissible: tuple = ()      # ((period column, key columns), ...) that identify the objects of the design
    reason: str | None = None   # expected reason of an abstention
    renamed_from: dict = field(default_factory=dict)  # new name -> name in the source case (family 1)


def _call(columns, periods, keys, *, explicit=(), numeric=(), positive=(), task_type="regression") -> dict:
    return {"task_type": task_type, "feature_columns": list(columns), "allowed_period_columns": list(periods),
            "allowed_key_columns": list(keys), "explicit_key_sets": [list(names) for names in explicit],
            "numeric_key_columns": list(numeric), "positive_key_columns": list(positive)}


def _table(records: list[dict], columns) -> tuple[dict, list]:
    return {name: [record[name] for record in records] for name in columns}, [record["_truth"] for record in records]


def _panel(rng: random.Random, objects: list, periods: list, presence: float = 0.8) -> list[dict]:
    """objects: (truth, {column: text}); each object is seen in two periods or more (if there are), rows shuffled."""
    records = []
    for truth, parts in objects:
        seen = [period for period in periods if rng.random() < presence]
        if len(seen) < min(2, len(periods)):
            seen = sorted(rng.sample(periods, min(2, len(periods))), key=periods.index)
        for period in seen:
            records.append({**parts, "period": period, "reading": f"{rng.uniform(0, 1):.6f}", "_truth": truth})
    rng.shuffle(records)
    return records


def _sites(rng: random.Random, count: int) -> list:
    return [(f"obj{index}", {"site": f"S{index:03d}", "zone": f"Z{index % 3}",
                             "size": str(rng.choice((120, 450, 900, 1500, 3000)))}) for index in range(count)]


PERIODS = ["2001", "2002", "2003", "2004", "2005"]
SITE_COLUMNS = ("site", "zone", "size", "reading", "period")


def clean_panel(rng: random.Random) -> Case:
    raw, truth = _table(_panel(rng, _sites(rng, 40), PERIODS), SITE_COLUMNS)
    call = _call(SITE_COLUMNS, ["period"], ["site", "zone", "size", "reading"], numeric=["size"], positive=["size"])
    return Case("clean_panel", 1, raw, call, truth, "propose", (("period", ("site",)),))


RENAME = {"site": "q_k", "zone": "b_z", "size": "m_s", "reading": "a_r", "period": "t_n"}


def clean_panel_renamed(rng: random.Random) -> Case:
    """The same table as clean_panel for the same seed, with other names and another column order."""
    source = clean_panel(rng)
    order = ["reading", "period", "size", "zone", "site"]
    raw = {RENAME[name]: list(source.raw_columns[name]) for name in order}
    call = _call([RENAME[name] for name in order], [RENAME["period"]],
                 [RENAME[name] for name in ("reading", "size", "zone", "site")],
                 numeric=[RENAME["size"]], positive=[RENAME["size"]])
    return Case("clean_panel_renamed", 1, raw, call, list(source.truth), "propose",
                ((RENAME["period"], (RENAME["site"],)),), renamed_from={new: old for old, new in RENAME.items()})


def composite_key(rng: random.Random) -> Case:
    pairs = [(a, b) for a in range(8) for b in range(6)]
    objects = [(f"obj{a}_{b}", {"left": f"L{a}", "right": f"R{b}", "parity": f"P{a % 2}"}) for a, b in pairs]
    columns = ("left", "right", "parity", "reading", "period")
    raw, truth = _table(_panel(rng, objects, PERIODS), columns)
    call = _call(columns, ["period"], ["left", "right", "parity", "reading"])
    return Case("composite_key", 2, raw, call, truth, "propose", (("period", ("left", "right")),))


def long_explicit_key(rng: random.Random) -> Case:
    parts = ["p1", "p2", "p3", "p4", "p5"]
    codes = rng.sample([(a, b, c, d, e) for a in range(3) for b in range(3) for c in range(3) for d in range(3)
                        for e in range(3)], 60)
    objects = [("obj" + "".join(map(str, code)), {name: f"v{value}" for name, value in zip(parts, code)})
               for code in codes]
    columns = (*parts, "reading", "period")
    raw, truth = _table(_panel(rng, objects, PERIODS), columns)
    call = _call(columns, ["period"], [*parts, "reading"], explicit=[parts])
    return Case("long_explicit_key", 3, raw, call, truth, "propose", (("period", tuple(parts)),))


def no_allowed_period(rng: random.Random) -> Case:
    raw, truth = _table(_panel(rng, _sites(rng, 40), PERIODS), SITE_COLUMNS)
    call = _call(SITE_COLUMNS, [], ["site", "zone", "size"], numeric=["size"])
    return Case("no_allowed_period", 4, raw, call, truth, "abstain", reason="no_allowed_period_column")


def single_period(rng: random.Random) -> Case:
    raw, truth = _table(_panel(rng, _sites(rng, 40), ["12"], presence=1.0), SITE_COLUMNS)
    call = _call(SITE_COLUMNS, ["period"], ["site", "zone", "size"], numeric=["size"])
    return Case("single_period", 4, raw, call, truth, "abstain", reason="single_period")


def unique_objects(rng: random.Random) -> Case:
    records = [{"site": f"S{index:03d}", "zone": f"Z{index % 3}", "period": rng.choice(PERIODS),
                "reading": f"{rng.uniform(0, 1):.6f}", "size": "450", "_truth": f"obj{index}"} for index in range(80)]
    raw, truth = _table(records, SITE_COLUMNS)
    call = _call(SITE_COLUMNS, ["period"], ["site", "reading"])
    return Case("unique_objects", 4, raw, call, truth, "abstain", reason="no_past_coverage")


def coarse_key(rng: random.Random) -> Case:
    objects = [(f"obj{index}", {"site": "same", "zone": f"Z{index % 3}", "size": "450"}) for index in range(40)]
    raw, truth = _table(_panel(rng, objects, PERIODS, presence=1.0), SITE_COLUMNS)
    call = _call(SITE_COLUMNS, ["period"], ["site", "zone", "size"], numeric=["size"])
    return Case("coarse_key", 5, raw, call, truth, "abstain", reason="no_past_coverage")


def technical_counter(rng: random.Random) -> Case:
    records = sorted(_panel(rng, _sites(rng, 40), PERIODS), key=lambda record: record["period"])
    for index, record in enumerate(records):
        record["sequence"] = str(index + 1)  # grows with the period, unique per row
    rng.shuffle(records)
    columns = (*SITE_COLUMNS, "sequence")
    raw, truth = _table(records, columns)
    call = _call(columns, ["period"], ["site", "zone", "sequence"], numeric=["sequence"])
    return Case("technical_counter", 5, raw, call, truth, "propose", (("period", ("site",)),))


CLOSE_PERIODS = ["1." + "0" * 20 + str(step) for step in (1, 2, 3, 4)]  # equal as float, distinct as Decimal


def missing_and_literals(rng: random.Random) -> Case:
    labels = ["NA", "?", "alpha", "beta"]
    objects = [(f"obj{label}{code}", {"label": label, "code": str(61240 + code)}) for label in labels
               for code in range(10)]
    records = _panel(rng, objects, CLOSE_PERIODS, presence=0.9)
    for record in records:
        draw = rng.random()
        if draw < 0.06:
            record["label"] = rng.choice([None, "", "   "])  # missing: never one common object
        elif draw < 0.12:
            record["code"] = rng.choice([None, " ", "n/a"])  # "n/a" is not a number: missing in a numeric part
        elif draw < 0.4:
            record["code"] = rng.choice([record["code"] + ".0", " " + record["code"] + ".00 "])  # same number
    columns = ("label", "code", "reading", "period")
    raw, truth = _table(records, columns)
    call = _call(columns, ["period"], ["label", "code"], numeric=["code"])
    return Case("missing_and_literals", 6, raw, call, truth, "propose", (("period", ("code", "label")),))


def repeated_records(rng: random.Random) -> Case:
    records = _panel(rng, _sites(rng, 40), PERIODS)
    duplicates = [dict(record, reading=f"{rng.uniform(0, 1):.6f}") for record in records if rng.random() < 0.1]
    records += duplicates
    rng.shuffle(records)
    raw, truth = _table(records, SITE_COLUMNS)
    call = _call(SITE_COLUMNS, ["period"], ["site", "zone", "size"], numeric=["size"], positive=["size"])
    return Case("repeated_records", 7, raw, call, truth, "propose", (("period", ("site",)),))


def _deceptive(rng: random.Random, two_objects: bool, name: str) -> Case:
    """30 observed keys in four periods; with two_objects the key alternates between two hidden objects."""
    records = []
    for index in range(30):
        for step, period in enumerate(PERIODS[:4]):
            truth = f"obj{index}" + (f"_{step % 2}" if two_objects else "")
            records.append({"site": f"K{index:02d}", "zone": f"Z{index % 2}", "period": period,
                            "reading": f"{rng.uniform(0, 1):.6f}", "size": "900", "_truth": truth})
    rng.shuffle(records)
    raw, truth = _table(records, SITE_COLUMNS)
    call = _call(SITE_COLUMNS, ["period"], ["site", "zone"])
    return Case(name, 8, raw, call, truth, "propose", (("period", ("site",)),))


def deceptive_one_object(rng: random.Random) -> Case:
    return _deceptive(rng, False, "deceptive_one_object")


def deceptive_two_objects(rng: random.Random) -> Case:
    return _deceptive(rng, True, "deceptive_two_objects")


CATALOG = (clean_panel, clean_panel_renamed, composite_key, long_explicit_key, no_allowed_period, single_period,
           unique_objects, coarse_key, technical_counter, missing_and_literals, repeated_records,
           deceptive_one_object, deceptive_two_objects)


def build_catalog(seed: int) -> list[Case]:
    """Every case of the catalog; each case draws from its own generator seeded by (seed, case name)."""
    return [make(random.Random(f"{seed}:{make.__name__.removesuffix('_renamed')}")) for make in CATALOG]


def scaling_case(rows: int, seed: int) -> Case:
    """A panel of about `rows` rows: ten periods, three key parts that identify objects together, two noise columns."""
    rng = random.Random(f"{seed}:scaling:{rows}")
    periods = [str(2000 + step) for step in range(10)]
    objects = [(f"obj{index}", {"left": f"L{index % 97}", "right": f"R{index // 97}", "zone": f"Z{index % 5}"})
               for index in range(max(2, rows // 8))]
    records = _panel(rng, objects, periods)[:rows]
    for record in records:
        record["noise"] = str(rng.randint(0, 10 ** 6))
    columns = ("left", "right", "zone", "noise", "reading", "period")
    raw, truth = _table(records, columns)
    call = _call(columns, ["period"], ["left", "right", "zone", "noise", "reading"])
    return Case(f"scaling_{rows}", 0, raw, call, truth, "propose", (("period", ("left", "right")),))


# ---------------------------------------------------------------- independent oracle (pairs of rows)

def _oracle_value(text, numeric: bool, positive: bool):
    if text is None or text.strip() == "":
        return None
    if not numeric:
        return text.strip()
    try:
        number = Decimal(text.strip())
    except InvalidOperation:
        return None
    if not number.is_finite() or (positive and number <= 0):
        return None
    return number


def oracle_rows(raw_columns: dict, config: dict) -> list:
    """(key or None, period or None) of every row under a config, parsed without the proposer's code."""
    numeric, positive = set(config["numeric_columns"]), set(config["positive_columns"])
    result = []
    for index in range(len(raw_columns[config["period_column"]])):
        period = _oracle_value(raw_columns[config["period_column"]][index], True, False)
        key = tuple(_oracle_value(raw_columns[name][index], name in numeric, name in positive)
                    for name in config["key_columns"])
        result.append((None if any(part is None for part in key) else key, period))
    return result


def oracle_evidence(raw_columns: dict, config: dict, truth=None) -> dict:
    """Evidence by brute force; with truth also the matches with other true objects (audit only)."""
    rows = oracle_rows(raw_columns, config)
    eligible = [index for index, (key, period) in enumerate(rows) if key is not None and period is not None]
    counts = {"rows": len(rows), "eligible_rows": len(eligible), "strict_past_history_rows": 0,
              "past_unambiguous_history_rows": 0, "past_ambiguous_history_rows": 0,
              "distinct_keys": len({rows[index][0] for index in eligible}),
              "distinct_periods": len({rows[index][1] for index in eligible})}
    audit = {"false_match_rows": 0, "false_applied_rows": 0, "correct_applied_rows": 0, "true_past_rows": 0}
    for i in eligible:
        key, period = rows[i]
        earlier = [j for j in eligible if rows[j][0] == key and rows[j][1] < period]
        if truth is not None and any(truth[j] == truth[i] and rows[j][1] is not None and rows[j][1] < period
                                     for j in range(len(rows))):
            audit["true_past_rows"] += 1
        if not earlier:
            continue
        counts["strict_past_history_rows"] += 1
        ambiguous = any(sum(1 for k in earlier if rows[k][1] == rows[j][1]) > 1 for j in earlier)
        counts["past_ambiguous_history_rows" if ambiguous else "past_unambiguous_history_rows"] += 1
        if truth is not None:
            false = any(truth[j] != truth[i] for j in earlier)
            audit["false_match_rows"] += false
            if not ambiguous:
                audit["false_applied_rows" if false else "correct_applied_rows"] += 1
    return {**counts, **audit} if truth is not None else counts


def adapter_evidence(raw: dict, config: dict) -> dict:
    """The same counts from the accepted HistoryEncoder (fit on the eligible rows, transform them)."""
    parsed = HistoryConfig.from_json(config)
    frame = pd.DataFrame(raw, dtype=object)
    periods = [numeric_value(value) for value in frame[parsed.period_column]]
    keys = history_keys(frame, parsed.key_columns, parsed.numeric_columns, parsed.positive_columns)
    usable = np.array([key is not None and period is not None for key, period in zip(keys, periods)])
    X = frame.loc[usable, parsed.columns]
    features = parsed.encoder().fit_transform(X, np.zeros(len(X)))
    present = features["history_present"].to_numpy() == 1
    ambiguous = features["history_ambiguous"].to_numpy() == 1
    return {"rows": len(frame), "eligible_rows": int(usable.sum()), "strict_past_history_rows": int(present.sum()),
            "past_unambiguous_history_rows": int((present & ~ambiguous).sum()),
            "past_ambiguous_history_rows": int((present & ambiguous).sum()),
            "distinct_keys": len({key for key, ok in zip(keys, usable) if ok}),
            "distinct_periods": len({period for period, ok in zip(periods, usable) if ok})}
