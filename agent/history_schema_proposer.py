"""Bounded proposer of history schemas: which allowed columns could serve as a history key and a period.

Contract v1 (task docs/team_round3/MURAD_HISTORY_SCHEMA.md, report docs/history_schema_proposer_report.md):

    result = propose_history_schemas(raw_columns, task_type="regression", feature_columns=features,
                                     allowed_period_columns=["p"], allowed_key_columns=["a", "b", "c"])
    for proposal in result["proposals"]:
        config = HistoryConfig.from_json(proposal["config"])   # the host measures it on INNER splits itself

The proposer sees only the raw INNER-fit features: never the target, test or outer rows, paths, the metric or an
API. It uses the standard library only: no models, files, environment, network or clock. It returns at most
max_proposals structurally eligible configs for a later INNER measurement by the host; it never activates one
(auto_activate is False), chooses a blend weight or claims an improvement. identity_verified is always False: equal
observed key values are not proof of one object, and the same X can hide different true objects. The policy
"past_unique_only" is fixed beforehand; comparing it with "allow" belongs to the host's measurement.

raw_columns maps a column name to a sequence of str or None, one value per row, as read before preprocessing.
Every name of raw_columns must be in feature_columns (the host removes the target and technical ids first); every
allowed, explicit, numeric and positive name must be in feature_columns and in raw_columns. Only the host decides
which columns may be a period or a key part: neither a name like "year", numbers nor monotonicity prove time.

Values (the semantics of agent/history_candidate.py):
    period and numeric key parts: exact Decimal(text.strip()) ("61242.0" equals "61242", nothing is rounded or passed
        through float); a value that is not a finite number is missing; in positive key columns a value <= 0 is
        missing too; only numeric_key_columns are compared as numbers;
    text key parts: text.strip(), compared exactly; None and blank text are missing, literal "NA" or "?" is a value;
    a missing key part or period makes the row ineligible: it neither has nor gives history.

Evidence of a config (key K, period p) over the eligible rows (full key and valid period):
    strict_past_history_rows       rows of period t whose key occurs in some period < t
    past_ambiguous_history_rows    of those, rows whose key occurred more than once inside one period < t
    past_unambiguous_history_rows  the rest: the rows the adapter blends under "past_unique_only"
Rows of the same period, later periods and the composition of a query batch never count as history.

Search (deterministic and independent of the order of rows, columns, explicit sets and dictionaries):
    1. explicit_key_sets with every eligible period, in the order (width, sorted names), unchanged;
    2. automatic keys of up to max_auto_key_width allowed key columns, best first. Every key column is screened once
       per period: its own past coverage is an exact upper bound for every key that contains it, because a wider
       key can only lose coverage. The candidate with the best optimistic rank (bound as unambiguous rows, no
       ambiguity) is checked next; after a check its one-column extensions enter the pool with the bound
       min(coverage of the key, coverage of the added column);
    3. the search is complete when no unchecked candidate could still enter the top max_proposals; it stops earlier
       after max_checked_configs full checks (status "limited", the counts show what was left).
Rank, best first: more past_unambiguous_history_rows, fewer past_ambiguous_history_rows, fewer key columns,
explicit before automatic, then the period name and the sorted key names. A proposal needs at least one past
unambiguous history row. A wider key that groups the rows exactly as a better-ranked subset of it (same period,
same eligible rows, same number of keys) is skipped: its history is identical. Coverage is a structural score,
never evidence of the right object.

Result statuses: "proposed"; "abstained" with a reason (classification_task, unknown_task_type, no_rows,
no_allowed_period_column, no_allowed_key_columns, no_valid_period, single_period, no_past_coverage); "limited" with a
reason (period_columns_over_limit, key_columns_over_limit, explicit_key_width_over_limit, row_limit, work_limit,
check_limit) and the counts in "limits". Over a column, row or work cap no value is read: rows are never sampled.
Malformed input (types, duplicate or unknown names, ragged columns, bool or non-positive limits) raises ValueError
without echoing data values. The inputs are never changed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from itertools import accumulate, compress, filterfalse, repeat
from operator import add, gt, mul


VERSION = 1
SCOPE = "inner_fit_features_only"
POLICY = "past_unique_only"
PROPOSAL_REASON = "structurally_eligible_for_inner_measurement"
DEFAULT_LIMITS = {
    "max_rows": 250_000,            # rows of raw_columns
    "max_work": 30_000_000,         # planned row-column visits: parsing, screening and every possible check
    "max_period_columns": 2,
    "max_key_columns": 12,
    "max_auto_key_width": 3,
    "max_explicit_key_width": 6,
    "max_checked_configs": 24,      # full evidence computations, explicit and automatic together
    "max_proposals": 4,
}


# ---------------------------------------------------------------- input checks

def _names(value, field: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list of column names")
    if any(not isinstance(name, str) or not name.strip() for name in value):
        raise ValueError(f"{field} must contain non-empty strings only")
    duplicates = sorted(name for name, count in Counter(value).items() if count > 1)
    if duplicates:
        raise ValueError(f"{field} contains duplicate column names: {duplicates}")
    return tuple(value)


def _row_count(raw_columns) -> int:
    if not isinstance(raw_columns, Mapping) or not raw_columns:
        raise ValueError("raw_columns must be a non-empty mapping of column name to a sequence of values")
    lengths = set()
    for name, values in raw_columns.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("raw_columns names must be non-empty strings")
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise ValueError(f"raw_columns[{name!r}] must be a sequence of str or None")
        lengths.add(len(values))
    if len(lengths) != 1:
        raise ValueError("raw_columns must be rectangular: every column needs the same number of values")
    return lengths.pop()


def _check_limits(limits) -> dict:
    if limits is None:
        return dict(DEFAULT_LIMITS)
    if not isinstance(limits, Mapping):
        raise ValueError("limits must be a mapping or None")
    unknown = sorted(repr(name) for name in limits if name not in DEFAULT_LIMITS)
    if unknown:
        raise ValueError(f"Unknown limits: {unknown}; allowed: {list(DEFAULT_LIMITS)}")
    result = dict(DEFAULT_LIMITS)
    for name, value in limits.items():
        if type(value) is not int or value < 1:  # bool and float are rejected on purpose
            raise ValueError(f"Limit {name!r} must be a positive int")
        result[name] = value
    return result


def _subset(names, allowed, field: str, target: str) -> None:
    outside = sorted(set(names) - set(allowed))
    if outside:
        raise ValueError(f"{field} must be {target}; not: {outside}")


# ---------------------------------------------------------------- values

MISSING = float("inf")  # code of a missing key part: inf * radix + code stays inf, so a key with a gap is inf
NOT_STR = "raw_columns values must be str or None"


def _text(value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(NOT_STR)
    value = value.strip()
    return value or None


def _number(value) -> Decimal | None:
    text = _text(value)
    if text is None:
        return None
    try:
        number = Decimal(text)  # construction from text is exact and ignores the context precision
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _unique(values) -> dict:
    """The distinct values in order of first appearance, which keeps memory access close to the column order."""
    try:
        return dict.fromkeys(values)
    except TypeError:  # an unhashable value is not str or None
        raise ValueError(NOT_STR) from None


def _text_codes(values) -> tuple[list, int]:
    """Dense int codes of stripped text values and their number; None and blank text are MISSING.

    The codes only group rows, so their numbering does not matter.
    """
    try:
        stripped = list(map(str.strip, values))
    except TypeError:  # None, or a value that is not str
        try:
            stripped = list(map(str.strip, ["" if value is None else value for value in values]))
        except TypeError:
            raise ValueError(NOT_STR) from None
    unique = dict.fromkeys(stripped)
    unique.pop("", None)
    code = dict(zip(unique, range(len(unique))))
    code[""] = MISSING
    return list(map(code.__getitem__, stripped)), len(unique)


def _number_codes(values, positive: bool) -> tuple[list, int]:
    """Dense int codes of exact numbers and their number; a non-number, or a value <= 0 if positive, is MISSING.

    Every distinct raw value is parsed once; equal numbers (Decimal 61242 and 61242.0) share a code.
    """
    index, code = {}, {}
    for raw in _unique(values):
        number = _number(raw)
        if number is None or (positive and number <= 0):
            code[raw] = MISSING
        else:
            code[raw] = index.setdefault(number, len(index))
    return list(map(code.__getitem__, values)), len(index)


def _ranks(values) -> tuple[list, int]:
    """Order-preserving int ranks of the exact Decimal periods and their number; a missing period gets that number.

    The Decimals are only compared (in sorted), never computed with, so the context precision cannot round them.
    """
    parsed = {raw: _number(raw) for raw in _unique(values)}
    order = {value: position for position, value in enumerate(sorted(v for v in set(parsed.values()) if v is not None))}
    missing = len(order)
    rank = {raw: missing if value is None else order[value] for raw, value in parsed.items()}
    return list(map(rank.__getitem__, values)), missing


def _period_order(ranks: list, size: int) -> tuple[list, list]:
    """Positions of the rows with a valid period sorted by period, and the end position of every period.

    A missing period has the rank size, so those rows sort last and are cut off: they never have or give history.
    """
    order = sorted(range(len(ranks)), key=ranks.__getitem__)
    sizes = Counter(ranks)
    ends = list(accumulate(map(sizes.__getitem__, range(size))))
    del order[ends[-1]:]
    return order, ends


# ---------------------------------------------------------------- evidence

def _evidence(key_codes: list[tuple[list, int]], ends: list, rows: int) -> dict:
    """Structural evidence of one key (codes of its columns) and period over the eligible rows.

    key_codes hold only the rows with a valid period, sorted by period; ends are the end positions of the periods.
    The key parts are packed into one int per row (mixed radix; MISSING if a part is missing). The periods are visited
    in ascending order with the keys seen in earlier periods and the keys repeated inside one earlier period, so a row
    never takes history from its own period or a later one. The rows of the new keys of a period are the growth of
    the seen keys plus the extra rows of new keys that repeat inside this period.
    """
    combined = key_codes[0][0]
    for codes, size in key_codes[1:]:
        combined = list(map(add, map(mul, combined, repeat(max(size, 1))), codes))
    seen, repeated = set(), set()
    eligible = past = ambiguous = periods = start = 0
    for end in ends:
        counts = Counter(combined[start:end])
        start = end
        counts.pop(MISSING, None)
        if not counts:
            continue
        total = sum(counts.values())
        multi = list(compress(counts, map(gt, counts.values(), repeat(1)))) if total > len(counts) else []
        new_multi = list(filterfalse(seen.__contains__, multi))
        if len(repeated) <= len(counts):  # rows of this period whose key repeated inside an earlier period
            ambiguous += sum(map(counts.get, repeated, repeat(0)))
        else:
            ambiguous += sum(map(counts.__getitem__, repeated.intersection(counts)))
        before = len(seen)
        seen.update(counts)
        past += total - (len(seen) - before) - (sum(map(counts.__getitem__, new_multi)) - len(new_multi))
        repeated.update(multi)  # only for the later periods
        eligible += total
        periods += 1
    return {"rows": rows, "eligible_rows": eligible, "strict_past_history_rows": past,
            "past_unambiguous_history_rows": past - ambiguous, "past_ambiguous_history_rows": ambiguous,
            "distinct_keys": len(seen), "distinct_periods": periods}


def _rank(candidate: dict) -> tuple:
    evidence = candidate["evidence"]
    return (-evidence["past_unambiguous_history_rows"], evidence["past_ambiguous_history_rows"],
            len(candidate["key"]), 0 if candidate["explicit"] else 1, candidate["period"], tuple(sorted(candidate["key"])))


def _same_partition(narrow: dict, wide: dict) -> bool:
    """True when the wider key splits the rows exactly as the narrower one, so its history is identical.

    A key with more columns has no more eligible rows and refines the groups of its subset; with the same eligible
    rows and the same number of keys the groups are the same.
    """
    a, b = narrow["evidence"], wide["evidence"]
    return (narrow["period"] == wide["period"] and set(narrow["key"]) < set(wide["key"])
            and a["eligible_rows"] == b["eligible_rows"] and a["distinct_keys"] == b["distinct_keys"])


def _optimistic(entry: tuple, bound: int) -> tuple:
    period, key = entry
    return (-bound, 0, len(key), 1, period, tuple(sorted(key)))


# ---------------------------------------------------------------- search

def _search(values: dict, periods: dict, period_order: list, key_columns: tuple, explicit: list, caps: dict,
            rows: int, counts: dict) -> tuple[list, bool]:
    """Checked candidates in rank order and whether the search is complete."""
    budget, top, width = caps["max_checked_configs"], caps["max_proposals"], caps["max_auto_key_width"]
    checked = {}  # (period, frozenset of key columns) -> candidate
    screened = {}  # evidence of one-column keys from the screening, reused (and still counted) by their check
    sorted_codes = {}  # (period, column) -> codes of the rows with a valid period, sorted by that period

    def evidence_of(period: str, key) -> dict:
        order, ends = periods[period]
        parts = []
        for name in key:
            if (period, name) not in sorted_codes:
                codes, count = values[name]
                sorted_codes[period, name] = list(map(codes.__getitem__, order)), count
            parts.append(sorted_codes[period, name])
        return _evidence(parts, ends, rows)

    def check(period: str, key: tuple, is_explicit: bool) -> dict:
        evidence = screened.pop((period, frozenset(key)), None)
        if evidence is None:
            evidence = evidence_of(period, key)
        candidate = {"period": period, "key": key, "explicit": is_explicit, "evidence": evidence}
        checked[period, frozenset(key)] = candidate
        counts["checked_configs"] += 1
        return candidate

    def leaders() -> list:
        kept = []
        for candidate in sorted(checked.values(), key=_rank):
            if len(kept) == top or candidate["evidence"]["past_unambiguous_history_rows"] == 0:
                break
            if not any(_same_partition(other, candidate) for other in kept):
                kept.append(candidate)
        return kept

    for key in sorted(explicit, key=lambda names: (len(names), sorted(names))):
        for period in period_order:
            if period not in key:
                if counts["checked_configs"] == budget:
                    counts["unchecked_explicit"] += 1
                else:
                    check(period, key, True)
    if counts["unchecked_explicit"]:
        return leaders(), False

    usable, pool, expanded = {}, {}, set()
    for period in period_order:
        usable[period] = {}
        for name in key_columns:
            if name != period:
                counts["screened_columns"] += 1
                single = (period, frozenset((name,)))
                if single in checked:  # an explicit one-column key
                    evidence = checked[single]["evidence"]
                else:
                    evidence = screened[single] = evidence_of(period, (name,))
                bound = evidence["strict_past_history_rows"]  # the past coverage of this column alone
                if bound > 0:
                    usable[period][name] = bound
                    pool[period, frozenset((name,))] = bound

    complete = True
    while pool:
        entry = min(pool, key=lambda item: _optimistic(item, pool[item]))
        best = leaders()
        if len(best) == top and _optimistic(entry, pool[entry]) > _rank(best[-1]):
            break  # no other pool entry and no extension of one can rank better than this entry
        period, names = entry
        candidate = checked.get(entry)  # an explicit set equal to this key is reused without a new check
        if candidate is None:
            if counts["checked_configs"] == budget:
                complete = False
                break
            candidate = check(period, tuple(sorted(names)), False)
        del pool[entry]
        expanded.add(entry)
        if len(names) < width:
            coverage = candidate["evidence"]["strict_past_history_rows"]
            for name, bound in usable[period].items():
                child = (period, names | {name})
                bound = min(coverage, bound)
                if name not in names and child not in expanded and bound > 0:
                    pool[child] = min(pool.get(child, bound), bound)
    counts["unchecked_pool"] = len(pool)
    return leaders(), complete


# ---------------------------------------------------------------- public

def propose_history_schemas(
    raw_columns, *, task_type, feature_columns,
    allowed_period_columns, allowed_key_columns,
    explicit_key_sets=(), numeric_key_columns=(),
    positive_key_columns=(), limits=None,
) -> dict:
    """Up to max_proposals history configs for a later INNER measurement; see the module docstring."""
    rows = _row_count(raw_columns)
    if not isinstance(task_type, str):
        raise ValueError("task_type must be a string")
    features = _names(feature_columns, "feature_columns")
    _subset(raw_columns, features, "raw_columns", "feature columns (the host removes the target and ids first)")
    periods_allowed = _names(allowed_period_columns, "allowed_period_columns")
    keys_allowed = _names(allowed_key_columns, "allowed_key_columns")
    numeric = _names(numeric_key_columns, "numeric_key_columns")
    positive = _names(positive_key_columns, "positive_key_columns")
    if isinstance(explicit_key_sets, str) or not isinstance(explicit_key_sets, (list, tuple)):
        raise ValueError("explicit_key_sets must be a list of lists of column names")
    explicit = [_names(names, "explicit_key_sets item") for names in explicit_key_sets]
    if any(not names for names in explicit):
        raise ValueError("explicit_key_sets must not contain an empty key")
    if len({frozenset(names) for names in explicit}) != len(explicit):
        raise ValueError("explicit_key_sets contains the same key twice")
    for field, names in (("allowed_period_columns", periods_allowed), ("allowed_key_columns", keys_allowed)):
        _subset(names, features, field, "feature columns")
        _subset(names, raw_columns, field, "columns of raw_columns")
    _subset([name for names in explicit for name in names], keys_allowed, "explicit_key_sets", "allowed key columns")
    _subset(numeric, keys_allowed, "numeric_key_columns", "allowed key columns")
    _subset(positive, numeric, "positive_key_columns", "numeric key columns")
    caps = _check_limits(limits)

    counts = {"rows": rows, "work_planned": 0, "eligible_period_columns": 0, "screened_columns": 0,
              "checked_configs": 0, "unchecked_explicit": 0, "unchecked_pool": 0, "search_complete": False}

    def result(status: str, reason: str | None, proposals=()) -> dict:
        return {"version": VERSION, "status": status, "reason": reason, "scope": SCOPE, "auto_activate": False,
                "proposals": [_proposal(candidate, numeric, positive) for candidate in proposals],
                "limits": {"caps": dict(caps), "counts": dict(counts)}}

    if task_type == "classification":
        return result("abstained", "classification_task")
    if task_type != "regression":
        return result("abstained", "unknown_task_type")
    if rows == 0:
        return result("abstained", "no_rows")
    if not periods_allowed:
        return result("abstained", "no_allowed_period_column")
    if not keys_allowed:
        return result("abstained", "no_allowed_key_columns")
    if len(periods_allowed) > caps["max_period_columns"]:
        return result("limited", "period_columns_over_limit")
    if len(keys_allowed) > caps["max_key_columns"]:
        return result("limited", "key_columns_over_limit")
    if any(len(names) > caps["max_explicit_key_width"] for names in explicit):
        return result("limited", "explicit_key_width_over_limit")
    widest = max([caps["max_auto_key_width"], *map(len, explicit)])
    parsed = set(periods_allowed) | set(keys_allowed)
    counts["work_planned"] = rows * (len(parsed) + len(keys_allowed) * len(periods_allowed)
                                     + caps["max_checked_configs"] * (widest + 1))
    if rows > caps["max_rows"]:
        return result("limited", "row_limit")
    if counts["work_planned"] > caps["max_work"]:
        return result("limited", "work_limit")

    ranked = {name: _ranks(raw_columns[name]) for name in periods_allowed}
    values = {name: _number_codes(raw_columns[name], name in positive) if name in numeric
              else _text_codes(raw_columns[name]) for name in keys_allowed}
    distinct = {name: size for name, (_, size) in ranked.items()}
    valid = {name: len(ranks) - ranks.count(size) for name, (ranks, size) in ranked.items()}
    period_order = sorted((name for name in periods_allowed if distinct[name] >= 2),
                          key=lambda name: (-distinct[name], -valid[name], name))
    counts["eligible_period_columns"] = len(period_order)
    if not period_order:
        return result("abstained", "single_period" if any(valid.values()) else "no_valid_period")

    periods = {name: _period_order(*ranked[name]) for name in period_order}
    proposals, complete = _search(values, periods, period_order, keys_allowed, explicit, caps, rows, counts)
    counts["search_complete"] = complete
    if not complete:
        return result("limited", "check_limit", proposals)
    if not proposals:
        return result("abstained", "no_past_coverage")
    return result("proposed", None, proposals)


def _proposal(candidate: dict, numeric: tuple, positive: tuple) -> dict:
    key = list(candidate["key"])
    return {"config": {"key_columns": key, "period_column": candidate["period"],
                       "numeric_columns": [name for name in key if name in numeric],
                       "positive_columns": [name for name in key if name in positive],
                       "ambiguity_policy": POLICY},
            "evidence": dict(candidate["evidence"]),
            "identity_verified": False,
            "reason": PROPOSAL_REASON}
