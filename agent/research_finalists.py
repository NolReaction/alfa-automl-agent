"""A bounded shortlist of host-measured canonical INNER recipes.

This module neither trains candidates nor reads data or validation answers.
The caller records only successful canonical INNER measurements after its own
stage-cost and finalization-feasibility checks. Repeat scores never replace a
recipe's canonical score. Baseline and incumbent remain caller-owned, so pool
eviction cannot remove either from the final shortlist.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import groupby
import math
import re

from agent.metrics import get_metric


MAX_POOL_ENTRIES = 6
MAX_FINALISTS = 3
_METRIC_FIELDS = {"name", "value", "higher_is_better"}
_HASH = re.compile(r"[0-9a-f]{64}")


def _identity(candidate):
    if candidate is None or not callable(getattr(candidate, "verify", None)):
        raise ValueError("A finalist must be a frozen candidate")
    candidate.verify()
    values = (getattr(candidate, "code_hash", None), getattr(candidate, "recipe_hash", None))
    if any(not isinstance(value, str) or _HASH.fullmatch(value) is None for value in values):
        raise ValueError("A finalist requires frozen SHA-256 code and recipe identities")
    return ":".join(values)


def _metric(metric, expected=None):
    if not isinstance(metric, dict) or set(metric) != _METRIC_FIELDS:
        raise ValueError("A finalist requires a canonical host metric with exactly three fields")
    name, value, higher = metric["name"], metric["value"], metric["higher_is_better"]
    spec = get_metric(name)
    if name != spec.name or type(higher) is not bool or higher != spec.higher_is_better:
        raise ValueError("Finalist metric name and direction must match the canonical registry")
    try:
        finite = type(value) in {int, float} and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("A finalist canonical score must be a finite real number")
    if expected is not None and (name, higher) != expected:
        raise ValueError("Finalists must share the same canonical metric")
    return deepcopy(metric)


def _family(candidate, supplied=None):
    if supplied is None:
        recipe = getattr(candidate, "recipe", None)
        if not isinstance(recipe, dict):
            raise ValueError("A finalist requires a frozen recipe or an explicit family")
        supplied = recipe.get("family") or recipe.get("action") or "custom"
    if (not isinstance(supplied, str) or not supplied.strip() or supplied != supplied.strip()
            or len(supplied) > 120):
        raise ValueError("A finalist family must be a nonempty bounded string")
    return supplied


def _ranked(entries, *, families=()):
    """Rank by score; family diversity breaks equal-score ties only.

    Frozen identity is the last tie-breaker, independent of arrival order.
    Diversity never promotes a worse canonical score over a better one.
    """
    def score(entry):
        metric = entry["metric"]
        return -metric["value"] if metric["higher_is_better"] else metric["value"]

    seen = set(families)
    ordered = sorted(entries, key=lambda entry: (score(entry), entry["identity"]))
    for _, values in groupby(ordered, key=score):
        tied = list(values)
        while tied:
            item = min(tied, key=lambda entry: (entry["family"] in seen, entry["identity"]))
            tied.remove(item)
            seen.add(item["family"])
            yield item


class FinalistPool:
    """Keep six measured recipes and return at most three distinct finalists.

    ``record`` is a host-only boundary: its score must come from the canonical
    INNER split, and the caller must first enforce finite per-stage costs no
    greater than 1800 seconds and its current finalization-feasibility policy.
    No metric written by candidate code is admissible here. The pool cannot
    establish that provenance from a numeric score alone.
    """

    def __init__(self, max_candidates=MAX_FINALISTS):
        if type(max_candidates) is not int or not 2 <= max_candidates <= MAX_FINALISTS:
            raise ValueError("max_candidates must be an integer of two or three")
        self.max_candidates = max_candidates
        self._entries = {}
        self._metric_identity = None

    def __len__(self):
        return len(self._entries)

    def _verify(self):
        for key, entry in self._entries.items():
            if _identity(entry["candidate"]) != key:
                raise ValueError("A retained finalist changed its frozen identity")

    def record(self, candidate, metric, *, family=None):
        """Record a verified canonical score; return whether a new entry stays.

        Repeated identical observations are neutral. A different score for an
        already retained identity is rejected, preventing a repeat/outer score
        from silently replacing the canonical one. Eviction deletes no files.
        """
        self._verify()
        key = _identity(candidate)
        checked = _metric(metric, self._metric_identity)
        chosen_family = _family(candidate, family)
        previous = self._entries.get(key)
        if previous is not None:
            if previous["metric"] != checked:
                raise ValueError("A finalist's canonical score cannot be replaced")
            if previous["family"] != chosen_family:
                raise ValueError("A finalist's family cannot be replaced")
            return False
        entry = {"identity": key, "candidate": candidate, "metric": checked, "family": chosen_family}
        retained = list(_ranked([*self._entries.values(), entry]))[:MAX_POOL_ENTRIES]
        self._entries = {item["identity"]: item for item in retained}
        self._metric_identity = checked["name"], checked["higher_is_better"]
        return key in self._entries

    def shortlist(self, baseline, baseline_metric, selected=None, selected_metric=None):
        """Retain baseline/incumbent, then the strongest distinct alternative.

        A missing candidate and metric must both be ``None``. A supplied frozen
        candidate requires a verified canonical score; unmeasured baselines
        must remain on the caller's existing non-shortlist recovery path.
        """
        self._verify()
        if baseline is None:
            if baseline_metric is not None or selected is not None or selected_metric is not None:
                raise ValueError("A finalist shortlist requires a measured frozen baseline")
            return []
        expected = self._metric_identity
        chosen, used = [], set()
        for identifier, candidate, metric in (("baseline", baseline, baseline_metric),
                                               ("candidate", selected, selected_metric)):
            if candidate is None:
                if metric is not None:
                    raise ValueError("A canonical score requires its frozen finalist")
                continue
            key = _identity(candidate)
            checked = _metric(metric, expected)
            expected = checked["name"], checked["higher_is_better"]
            previous = self._entries.get(key)
            if previous is not None and previous["metric"] != checked:
                raise ValueError("Shortlist score differs from the retained canonical score")
            if key in used:
                original = next(item for item in chosen if item["identity"] == key)
                if original["metric"] != checked:
                    raise ValueError("Identical finalists cannot have different canonical scores")
                continue
            family = previous["family"] if previous is not None else _family(candidate)
            chosen.append({"id": identifier, "identity": key, "candidate": candidate,
                           "metric": checked, "family": family})
            used.add(key)
        if len(chosen) < self.max_candidates:
            alternatives = _ranked((entry for key, entry in self._entries.items() if key not in used),
                                   families=(entry["family"] for entry in chosen))
            alternative = next(alternatives, None)
            if alternative is not None:
                chosen.append({"id": "alternative", **alternative})
        return [{"id": entry["id"], "candidate": entry["candidate"],
                 "metric": deepcopy(entry["metric"]), "family": entry["family"]}
                for entry in chosen]
