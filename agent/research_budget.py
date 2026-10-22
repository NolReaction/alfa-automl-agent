"""A measured, bounded time plan for the v3 research lifecycle.

Row-based costs are conservative heuristics, not a guarantee that training time
scales linearly (or that a model will fit its estimate).  All child timeouts
remain bounded by the original deadline.  No model-written timing is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time


STAGE_ORDER = ("baseline_outer", "candidate_outer", "baseline_refit", "candidate_refit")
SAFETY_FACTOR = 1.5
MAX_ROW_SCALE = 16.0
MAX_RECENT_MEASUREMENTS = 4
ADAPTIVE_RESERVE_MIN_BUDGET = 300.0


def _number(value, name, *, positive=False):
    if (type(value) not in {int, float} or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'non-negative'} number")
    return float(value)


def _rows(value, name):
    if type(value) is not int or not 1 <= value <= 1_000_000_000:
        raise ValueError(f"{name} must be an integer from 1 to 1000000000")
    return value


def _key(value, *, optional=False):
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("recipe_key must be a non-empty string of at most 256 characters")
    return value


@dataclass(frozen=True)
class Measurement:
    elapsed: float
    fit_rows: int
    predict_rows: int
    measured_at: float


@dataclass(frozen=True)
class FinalizationEstimate:
    stage_seconds: dict[str, float]
    estimated_seconds: float
    reserve_seconds: float
    diagnostics: dict
    deadline: float
    measurement_generation: int


class BudgetPolicy:
    def __init__(self, time_budget_seconds, *, started=None, clock=time.monotonic,
                 adaptive_reserve=False):
        self.time_budget_seconds = _number(time_budget_seconds, "time_budget_seconds", positive=True)
        if self.time_budget_seconds > 10500:
            raise ValueError("Research budget cannot exceed 10500 seconds")
        if not callable(clock):
            raise ValueError("clock must be callable")
        if type(adaptive_reserve) is not bool:
            raise ValueError("adaptive_reserve must be a boolean")
        self.adaptive_reserve = adaptive_reserve
        self._clock = clock
        now = _number(clock(), "clock")
        self.started = now if started is None else _number(started, "started")
        if self.started > now:
            raise ValueError("Budget start cannot be in the future")
        self._deadline = self.started + self.time_budget_seconds
        if not math.isfinite(self._deadline) or self._deadline <= self.started:
            raise ValueError("Budget deadline is not representable")
        self._last_now = now
        self.minimum_final_reserve = min(self.time_budget_seconds * .5,
                                        max(15.0, self.time_budget_seconds * .30))
        self.publication_buffer = min(2.0, self.time_budget_seconds * .05)
        self._measurements: dict[str, list[Measurement]] = {}
        self._generation = 0

    @property
    def deadline(self):
        return self._deadline

    def _now(self):
        now = _number(self._clock(), "clock")
        if now < self._last_now:
            raise ValueError("Budget clock moved backwards")
        self._last_now = now
        return now

    def remaining(self):
        return max(0.0, self.deadline - self._now())

    def record_measurement(self, recipe_key, elapsed, fit_rows, predict_rows, *, measured_at=None):
        """Record host-measured completed execution for this exact frozen recipe.

        Keys should include both code and recipe hashes.  Measurements from a
        different recipe are never borrowed.  Earlier runs and out-of-order
        timestamps are rejected rather than silently overriding a newer sample.
        """
        recipe_key = _key(recipe_key)
        elapsed = _number(elapsed, "elapsed", positive=True)
        fit_rows, predict_rows = _rows(fit_rows, "fit_rows"), _rows(predict_rows, "predict_rows")
        now = self._now()
        measured_at = now if measured_at is None else _number(measured_at, "measured_at")
        if not self.started <= measured_at <= now:
            raise ValueError("Measurement timestamp is outside this run")
        if elapsed > measured_at - self.started + .05:
            raise ValueError("Measurement duration exceeds the elapsed run time")
        previous = self._measurements.get(recipe_key, [])
        if previous and measured_at <= previous[-1].measured_at:
            raise ValueError("Stale or duplicate recipe measurement")
        sample = Measurement(elapsed, fit_rows, predict_rows, measured_at)
        self._measurements[recipe_key] = [*previous, sample][-MAX_RECENT_MEASUREMENTS:]
        self._generation += 1

    def _estimate(self, recipe_key, fit_rows, predict_rows, default):
        samples = self._measurements.get(recipe_key, ())
        if not samples:
            return min(self.time_budget_seconds, default), {"basis": "unmeasured_default", "samples": 0}
        costs, clipped = [], False
        for sample in samples:
            ratio = max(1.0, fit_rows / sample.fit_rows, predict_rows / sample.predict_rows)
            # Mild superlinear headroom; neither this exponent nor its cap is
            # a statement about any estimator's actual complexity.
            scale = min(MAX_ROW_SCALE, ratio ** 1.25)
            raw = sample.elapsed * SAFETY_FACTOR * scale + .25
            clipped = clipped or scale == MAX_ROW_SCALE or raw > self.time_budget_seconds
            costs.append(min(self.time_budget_seconds, max(.5, raw)))
        return max(costs), {"basis": "host_measurement_row_heuristic", "samples": len(samples),
                            "bounded": clipped}

    def finalization_estimate(self, *, baseline_key, candidate_key=None, outer_fit_rows,
                              outer_predict_rows, full_fit_rows, test_rows):
        baseline_key, candidate_key = _key(baseline_key, optional=True), _key(candidate_key, optional=True)
        outer_fit_rows = _rows(outer_fit_rows, "outer_fit_rows")
        outer_predict_rows = _rows(outer_predict_rows, "outer_predict_rows")
        full_fit_rows, test_rows = _rows(full_fit_rows, "full_fit_rows"), _rows(test_rows, "test_rows")
        if full_fit_rows < outer_fit_rows:
            raise ValueError("Full fit cannot contain fewer rows than outer fit")
        self._now()
        stages, details = {}, {}
        for prefix, key in (("baseline", baseline_key), ("candidate", candidate_key)):
            if key is None:
                continue
            for suffix, fitting, predicting, default in (
                    ("outer", outer_fit_rows, outer_predict_rows, 2.0),
                    ("refit", full_fit_rows, test_rows, 3.0)):
                stage = prefix + "_" + suffix
                stages[stage], details[stage] = self._estimate(key, fitting, predicting, default)
        stages = {stage: stages[stage] for stage in STAGE_ORDER if stage in stages}
        estimated = sum(stages.values()) + self.publication_buffer
        reserve_floor = self.minimum_final_reserve
        reserve_policy = "fixed_fraction"
        # Release the large discovery-stage floor only after every pending
        # recipe has its OWN completed host measurement. Unknown candidates
        # never inherit the recovery baseline's cheap timing. Short smoke
        # budgets deliberately retain the historical conservative allocation.
        if (self.adaptive_reserve and self.time_budget_seconds > ADAPTIVE_RESERVE_MIN_BUDGET
                and baseline_key is not None and details
                and all(item["basis"] == "host_measurement_row_heuristic" for item in details.values())):
            reserve_floor = min(reserve_floor, max(15.0, min(120.0, self.time_budget_seconds * .02)))
            reserve_policy = "measured_pending_recipes"
        reserve = min(self.time_budget_seconds, max(reserve_floor, estimated))
        diagnostics = {
            "kind": "heuristic", "guaranteed": False, "safety_factor": SAFETY_FACTOR,
            "row_scale_limit": MAX_ROW_SCALE, "stage_estimates": details,
            "fits_original_budget_estimate": estimated <= self.time_budget_seconds,
        }
        if self.adaptive_reserve:
            diagnostics.update(reserve_policy=reserve_policy,
                               minimum_reserve_applied_seconds=reserve_floor)
        return FinalizationEstimate(stages, estimated, reserve, diagnostics,
                                    self.deadline, self._generation)

    def estimate_trial_cost(self, recipe_key, fit_rows, predict_rows):
        """Host-timing estimate for admitting a complete paired inner round."""
        recipe_key = _key(recipe_key)
        fit_rows, predict_rows = _rows(fit_rows, "fit_rows"), _rows(predict_rows, "predict_rows")
        self._now()
        return self._estimate(recipe_key, fit_rows, predict_rows, default=2.0)[0]

    def _check_plan(self, estimate):
        if (not isinstance(estimate, FinalizationEstimate) or estimate.deadline != self.deadline
                or estimate.measurement_generation != self._generation):
            raise ValueError("Stale or foreign finalization estimate; recompute after measurements")
        if (set(estimate.stage_seconds) - set(STAGE_ORDER)
                or not math.isfinite(estimate.reserve_seconds) or estimate.reserve_seconds < 0
                or any(type(value) not in {int, float} or not math.isfinite(value) or value <= 0
                       for value in estimate.stage_seconds.values())):
            raise ValueError("Invalid finalization estimate")

    def research_timeout(self, estimate, *, maximum=1800, minimum=1, expected_seconds=None):
        """Return zero when another trial threatens the measured final reserve."""
        self._check_plan(estimate)
        maximum, minimum = _number(maximum, "maximum", positive=True), _number(minimum, "minimum", positive=True)
        if minimum > maximum:
            raise ValueError("minimum timeout cannot exceed maximum")
        expected = minimum if expected_seconds is None else _number(expected_seconds, "expected_seconds")
        spare = max(0.0, self.remaining() - estimate.reserve_seconds)
        if spare < max(minimum, expected):
            return 0.0
        return min(maximum, spare)

    def finalization_timeouts(self, estimate, *, completed=(), maximum=1800, minimum=.25):
        """Allocate ALL pending stages together; their sum cannot exceed remaining.

        Recompute before each stage, marking already completed/abandoned stages.
        Fund every pending estimate when they all fit. Otherwise estimated
        baseline outer/refit costs receive priority over candidate work, so a
        costly candidate cannot consume the whole recovery budget.
        """
        self._check_plan(estimate)
        maximum, minimum = _number(maximum, "maximum", positive=True), _number(minimum, "minimum", positive=True)
        if minimum > maximum:
            raise ValueError("minimum timeout cannot exceed maximum")
        if (not isinstance(completed, (tuple, list, set, frozenset))
                or any(stage not in STAGE_ORDER for stage in completed)
                or len(set(completed)) != len(completed)):
            raise ValueError("completed must contain unique known stage names")
        pending = [stage for stage in STAGE_ORDER if stage in estimate.stage_seconds and stage not in completed]
        allocations = {stage: 0.0 for stage in pending}
        available = max(0.0, self.remaining() - self.publication_buffer)
        if not pending or available <= 0:
            return allocations
        floors = {stage: min(maximum, max(minimum, estimate.stage_seconds[stage])) for stage in pending}
        required = sum(floors.values())
        if required > available:
            # Subtracting absolute monotonic timestamps can lose a few ulps
            # even at the exact reserve boundary. Cover only that rounding
            # gap, leaving the correction unspent rather than creating time.
            tolerance = min(1e-6, 4 * math.ulp(self.deadline))
            shortfall = required - available
            largest = max(floors, key=floors.get)
            corrected = floors[largest] - shortfall - tolerance
            if shortfall <= tolerance and corrected >= minimum:
                allocations.update(floors)
                allocations[largest] = corrected
                return allocations
        if required <= available:
            # A feasible finalization plan must fund every stage's estimate
            # before distributing surplus. Otherwise the fallback receives
            # both its floor and a weighted share of the candidate's floor.
            allocations.update(floors)
            available -= required
            active = pending
        else:
            protected = [stage for stage in pending if stage.startswith("baseline_")]
            protected_cost = sum(floors[stage] for stage in protected)
            if protected_cost >= available:
                for stage in protected:
                    allocations[stage] = available * floors[stage] / protected_cost
                return {stage: value if value >= minimum else 0.0 for stage, value in allocations.items()}
            allocations.update({stage: floors[stage] for stage in protected})
            available -= protected_cost
            candidates = [stage for stage in pending if stage not in protected]
            # With too little time for a candidate's remaining stages, keep its
            # shares at zero and use the spare time for the fallback instead.
            active = pending if available >= len(candidates) * minimum else protected
            for stage in candidates:
                if stage in active:
                    allocations[stage] = minimum
                    available -= minimum
        while available > 1e-9:
            active = [stage for stage in active if allocations[stage] < maximum]
            if not active:
                break
            weight = sum(estimate.stage_seconds[stage] for stage in active)
            portion = available
            for stage in active:
                extra = min(maximum - allocations[stage], portion * estimate.stage_seconds[stage] / weight)
                allocations[stage] += extra
                available -= extra
        return {stage: value if value >= minimum else 0.0 for stage, value in allocations.items()}

    def report(self, estimate=None):
        if estimate is not None:
            self._check_plan(estimate)
        result = {"version": 1, "kind": "heuristic", "guaranteed": False,
                  "time_budget_seconds": self.time_budget_seconds, "remaining_seconds": self.remaining(),
                  "minimum_final_reserve_seconds": self.minimum_final_reserve,
                  "publication_buffer_seconds": self.publication_buffer,
                  "measurement_generation": self._generation,
                  "measurements": {key: [{"elapsed_seconds": item.elapsed,
                      "fit_rows": item.fit_rows, "predict_rows": item.predict_rows,
                      "at_seconds": item.measured_at - self.started} for item in values]
                      for key, values in self._measurements.items()}}
        if self.adaptive_reserve:
            result.update(adaptive_reserve=True,
                          elapsed_seconds=max(0.0, self.time_budget_seconds - result["remaining_seconds"]))
        if estimate is not None:
            result.update(estimated_finalization_seconds=estimate.estimated_seconds,
                          final_reserve_seconds=estimate.reserve_seconds,
                          stage_seconds=dict(estimate.stage_seconds), diagnostics=estimate.diagnostics)
        return result
