"""The frozen plateau criterion, as pure arithmetic over already-recorded rows.

Nothing here imports torch, numpy, a world or a model. Every function takes plain data and
returns plain data, so the whole criterion is exercised by pure tests.

The criterion for one run (one arm and one seed, never pooled across arms or seeds):

* one window is 480 episodes = 40 scene cycles of 12 = 120 PPO updates;
* every check looks at the last three non-overlapping 480-episode windows;
* for all three unordered pairs of windows, the 90% block-bootstrap interval of the
  difference of mean on-time success rates must sit inside [-0.025, +0.025];
* the range of the three point means must be at most 0.025 -- the interval and the point
  range guard different things and one never stands in for the other;
* the OLS slope of the three window means against x = (0, 1, 2) must have |slope| <= 0.0125
  per full window. The 40 in-window cycle points are not fitted;
* the mean charged fraction of budget may vary by at most 0.01 across the three windows. This is
  a required additional convergence gate on cost-curve stability: exceeding it means the run has
  not converged yet, never that the history is corrupt. A saturated budget is reported as
  saturation and is legal;
* the three windows must also be complete and consistent: 480 distinct episodes each, the
  registered scene for that episode index, 24 generated goals, 1000 steps, charged bytes never
  above the 58600 budget, and 120 recorded updates whose optimizer counts agree and whose
  diagnostic fields are finite.

Bootstrap: 10000 resamples over the 40 cycles of a window, paired cycle by cycle. The random
stream is derived from SHA256(label|arm|seed|end_episode|pair) and consumed only through
stats.index_for; the training RNGs are never read, written or advanced.

A run is converged only after two consecutive passing checks; the second check rests on 480
further episodes. Each run may reach that state at its own check episode and stops there.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

from . import constants as C
from . import stats

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_INSUFFICIENT = "INSUFFICIENT_HISTORY"
STATUS_INVALID = "INVALID_HISTORY"

#: Diagnostic fields every update of the window must carry and which must be finite.
DIAGNOSTIC_FIELDS = ("policy_surrogate", "value_loss_task", "value_loss_cost", "entropy",
                     "approx_kl", "gradient_norm_mean")
#: Row fields read from both parent schemas (they share these names).
BASE_ROW_FIELDS = ("arm", "seed", "episode_index", "scene", "on_time", "generated_goals",
                   "charged_bytes", "steps")


def derive_seed(arm, seed, end_episode, pair):
    """Deterministic bootstrap seed for one (run, check, pair); no live RNG is involved."""
    text = "%s|%s|%d|%d|%d:%d" % (C.CONVERGENCE_SEED_LABEL, arm, int(seed), int(end_episode),
                                  int(pair[0]), int(pair[1]))
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def provenance_of(start, stop):
    """Annotate the three explicit source intervals plus this stage; arithmetic is unchanged."""
    start, stop = int(start), int(stop)
    if start < 0:
        raise RuntimeError("a convergence window starts before episode 0: %d" % (start,))
    labels = [source["source_id"].upper() + "_READONLY" for source in C.HISTORY_SOURCES
              if start < source["episodes"][1] and stop > source["episodes"][0]]
    if stop > C.STAGE_START_EPISODE:
        labels.append("NEW_STAGE_%d" % C.STAGE_INDEX)
    if not labels:
        raise RuntimeError("empty or unassigned history window %d..%d" % (start, stop))
    return labels[0] if len(labels) == 1 else "MIXED_" + "_AND_".join(labels)


def window_bounds(end_episode, windows=None, width=None):
    """The last N non-overlapping blocks that end at end_episode."""
    count = int(C.CONVERGENCE_WINDOWS if windows is None else windows)
    size = int(C.CONVERGENCE_WINDOW_EPISODES if width is None else width)
    end = int(end_episode)
    bounds = []
    for index in range(count):
        start = end - (count - index) * size
        bounds.append({"index": index, "start": start, "end": start + size,
                       "width": size, "provenance": provenance_of(start, start + size)})
    return bounds


def history_needed(end_episode):
    """First episode index a check at end_episode requires."""
    return int(end_episode) - C.CONVERGENCE_WINDOWS * C.CONVERGENCE_WINDOW_EPISODES


def cycle_means(values):
    """Mean of every consecutive 12-episode scene cycle; a partial cycle is an error."""
    values = [float(v) for v in values]
    if len(values) % C.SCENE_CYCLE != 0:
        raise RuntimeError("a window of %d episodes is not a whole number of %d-scene cycles"
                           % (len(values), C.SCENE_CYCLE))
    means = []
    for start in range(0, len(values), C.SCENE_CYCLE):
        block = values[start:start + C.SCENE_CYCLE]
        means.append(sum(block) / len(block))
    return means


def ols_slope(values):
    """Slope of values against x = 0, 1, 2, ...; the criterion uses three window means."""
    values = [float(v) for v in values]
    n = len(values)
    if n < 2:
        raise RuntimeError("a slope needs at least two points, got %d" % (n,))
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    numerator = sum((index - mean_x) * (value - mean_y)
                    for index, value in enumerate(values))
    denominator = sum((index - mean_x) ** 2 for index in range(n))
    return numerator / denominator


def _finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _integer(value, label, problems):
    """A metadata integer, never truncating a fraction and never accepting a bool."""
    if isinstance(value, bool):
        problems.append("%s is a bool, not an integer" % (label,))
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    problems.append("%s is not an integer: %r" % (label, value))
    return None


def pair_interval(left, right, *, seed, reps=None, lower_order=None, upper_order=None):
    """Paired cycle-block bootstrap of mean(left) - mean(right)."""
    left = [float(v) for v in left]
    right = [float(v) for v in right]
    if len(left) != len(right):
        raise RuntimeError("paired windows must have the same cycle count")
    n = len(left)
    total_reps = int(C.CONVERGENCE_BOOTSTRAP_REPS if reps is None else reps)
    low = int(C.CONVERGENCE_LOWER_ORDER if lower_order is None else lower_order)
    high = int(C.CONVERGENCE_UPPER_ORDER if upper_order is None else upper_order)
    base = sum(left) / n - sum(right) / n
    deltas = []
    for replicate in range(total_reps):
        total = 0.0
        for draw in range(n):
            index = stats.index_for(seed, replicate, draw, n)
            total += left[index] - right[index]
        deltas.append(total / n)
    deltas.sort()
    return {"n_cycles": n, "mean_delta": base, "lower": deltas[low], "upper": deltas[high],
            "resamples": total_reps, "order_statistics": [low, high], "seed": seed,
            "interval_level": C.CONVERGENCE_INTERVAL_LEVEL, "paired": True,
            "block": "scene cycle of %d episodes" % C.SCENE_CYCLE}


def _window_rows(series, bound, arm, seed, problems):
    """Every row of one window, with the integrity rules applied."""
    label = "window[%d] %d..%d" % (bound["index"], bound["start"], bound["end"] - 1)
    raw_rows = series.get("rows") if isinstance(series, Mapping) else None
    if isinstance(raw_rows, Mapping):
        candidates = list(raw_rows.items())
    elif isinstance(raw_rows, (list, tuple)):
        candidates = list(enumerate(raw_rows))
    else:
        problems.append("%s: rows is not a mapping or list" % (label,))
        return []
    by_index = {}
    for key, row in candidates:
        try:
            index = int(key)
        except (TypeError, ValueError, OverflowError):
            problems.append("%s: row key %r is not an episode index" % (label, key))
            continue
        by_index.setdefault(index, []).append(row)
    rows = []
    for index in range(bound["start"], bound["end"]):
        matches = by_index.get(index, [])
        if not matches:
            problems.append("%s: episode %d is missing" % (label, index))
            continue
        if len(matches) != 1:
            problems.append("%s: episode %d appears %d times"
                            % (label, index, len(matches)))
            continue
        row = matches[0]
        if not isinstance(row, Mapping):
            problems.append("%s: episode %d is not a row mapping" % (label, index))
            continue
        row_index = _integer(row.get("episode_index", index), "%s episode_index" % (label,),
                             problems)
        if row_index is None:
            continue
        if row_index != index:
            problems.append("%s: row key %d carries episode_index %d"
                            % (label, index, row_index))
            continue
        if str(row.get("arm")) != arm:
            problems.append("%s: episode %d arm %r != %r"
                            % (label, index, row.get("arm"), arm))
            continue
        row_seed = _integer(row.get("seed"), "%s seed" % (label,), problems)
        if row_seed is None:
            continue
        if row_seed != int(seed):
            problems.append("%s: episode %d seed %r != %d"
                            % (label, index, row.get("seed"), int(seed)))
            continue
        expected_scene = C.scene_for(seed, index)
        row_scene = _integer(row.get("scene"), "%s scene" % (label,), problems)
        if row_scene is None:
            continue
        if row_scene != int(expected_scene):
            problems.append("%s: episode %d scene %r != registered %d"
                            % (label, index, row.get("scene"), expected_scene))
            continue
        generated = _integer(row.get("generated_goals"), "%s generated_goals" % (label,),
                             problems)
        if generated is None:
            continue
        if generated != C.NUM_GOALS:
            problems.append("%s: episode %d generated %d goals, expected %d"
                            % (label, index, generated, C.NUM_GOALS))
            continue
        steps = _integer(row.get("steps"), "%s steps" % (label,), problems)
        if steps is None:
            continue
        if steps != C.STEPS:
            problems.append("%s: episode %d ran %r steps, expected %d"
                            % (label, index, row.get("steps"), C.STEPS))
            continue
        on_time = _integer(row.get("on_time"), "%s on_time" % (label,), problems)
        if on_time is None:
            continue
        if on_time < 0 or on_time > generated:
            problems.append("%s: episode %d on_time %d outside [0, %d]"
                            % (label, index, on_time, generated))
            continue
        charged = row.get("charged_bytes")
        if (not _finite(charged)) or float(charged) < 0.0 or float(charged) > C.BUDGET_BYTES:
            problems.append("%s: episode %d charged %r outside [0, %d]"
                            % (label, index, charged, C.BUDGET_BYTES))
            continue
        rows.append(row)
    if len(rows) != bound["width"]:
        problems.append("%s: %d usable rows, expected %d"
                        % (label, len(rows), bound["width"]))
    return rows


def _window_metrics(bound, rows, problems):
    label = "window[%d]" % (bound["index"],)
    success = [float(row["on_time"]) / float(row["generated_goals"]) for row in rows]
    spent = [float(row["charged_bytes"]) / float(C.BUDGET_BYTES) for row in rows]
    if not success:
        problems.append("%s: no usable row to describe" % (label,))
        return None
    if len(rows) != bound["width"]:
        return None
    for value in success + spent:
        if not _finite(value):
            problems.append("%s: non-finite success or budget fraction" % (label,))
            return None
    return {"index": bound["index"], "start": bound["start"], "end": bound["end"],
            "provenance": bound["provenance"], "episodes": len(rows),
            "cycles": len(success) // C.SCENE_CYCLE,
            "success_rate": sum(success) / len(success),
            "budget_mean": sum(spent) / len(spent),
            "budget_max": max(spent),
            "budget_saturated": max(spent) >= 0.999999,
            "cycle_success": cycle_means(success),
            "cycle_budget": cycle_means(spent)}


def _update_metrics(series, bound, problems):
    """The window's PPO diagnostics: 120 updates, matching counts, finite fields."""
    first = bound["start"] // C.EPISODES_PER_UPDATE
    last = bound["end"] // C.EPISODES_PER_UPDATE
    label = "window[%d] updates" % (bound["index"],)
    window_updates = []
    updates = series.get("updates") if isinstance(series, Mapping) else None
    if not isinstance(updates, Mapping):
        problems.append("%s: updates is not a mapping" % (label,))
        return {"updates": 0, "episodes": bound["width"], "optimizer_steps": 0}
    for update_index in range(first, last):
        record = updates.get(update_index)
        if record is None:
            record = updates.get(str(update_index))
        if record is None:
            problems.append("%s: update %d is missing" % (label, update_index))
            continue
        if not isinstance(record, Mapping):
            problems.append("%s: update %d is not a mapping" % (label, update_index))
            continue
        steps = _integer(record.get("optimizer_steps"),
                         "%s update optimizer_steps" % (label,), problems)
        minibatches = _integer(record.get("minibatches"),
                               "%s update minibatches" % (label,), problems)
        if steps is None or minibatches is None:
            continue
        if steps <= 0:
            problems.append("%s: update %d performed %d optimizer steps"
                            % (label, update_index, steps))
            continue
        if steps != minibatches:
            problems.append("%s: update %d records %d steps for %d minibatches"
                            % (label, update_index, steps, minibatches))
            continue
        for field in DIAGNOSTIC_FIELDS:
            if field not in record:
                problems.append("%s: update %d has no %r" % (label, update_index, field))
            elif not _finite(record[field]):
                problems.append("%s: update %d %s=%r is not finite"
                                % (label, update_index, field, record[field]))
        window_updates.append(record)
    expected = C.CONVERGENCE_UPDATES_PER_WINDOW
    if len(window_updates) != expected:
        problems.append("%s: %d updates recorded, expected %d"
                        % (label, len(window_updates), expected))
    return {"updates": len(window_updates), "episodes": bound["width"],
            "optimizer_steps": sum(int(item.get("optimizer_steps", 0))
                                   for item in window_updates)}


def evaluate(series, *, arm, seed, end_episode):
    """One convergence certificate for one run at one check episode. Pure arithmetic."""
    arm = str(arm)
    seed = int(seed)
    end_episode = int(end_episode)
    certificate = {
        "schema": C.CONVERGENCE_SCHEMA,
        "experiment_id": C.RUN_ID,
        "arm": arm, "seed": seed, "end_episode": end_episode,
        "window_episodes": C.CONVERGENCE_WINDOW_EPISODES,
        "windows": C.CONVERGENCE_WINDOWS,
        "history_needed": history_needed(end_episode),
        "status": STATUS_FAIL,
        "passed": False,
        "problems": [],
        "checks": {},
        "criterion": {
            "pair_bound": C.CONVERGENCE_PAIR_BOUND,
            "mean_range_bound": C.CONVERGENCE_MEAN_RANGE_BOUND,
            "slope_bound": C.CONVERGENCE_SLOPE_BOUND,
            "slope_basis": C.CONVERGENCE_SLOPE_BASIS,
            "budget_range_bound": C.CONVERGENCE_BUDGET_RANGE_BOUND,
            "budget_range_role": C.CONVERGENCE_BUDGET_RANGE_ROLE,
            "interval_level": C.CONVERGENCE_INTERVAL_LEVEL,
            "resamples": C.CONVERGENCE_BOOTSTRAP_REPS,
            "required_consecutive": C.CONVERGENCE_REQUIRED_CONSECUTIVE,
        },
    }
    problems = certificate["problems"]
    history_start = history_needed(end_episode)
    if history_start < 0:
        problems.append("a check at %d needs episode %d, which does not exist"
                        % (end_episode, history_start))
        certificate["status"] = STATUS_INSUFFICIENT
        return certificate
    bounds = window_bounds(end_episode)
    windows = []
    for bound in bounds:
        rows = _window_rows(series, bound, arm, seed, problems)
        metrics = _window_metrics(bound, rows, problems)
        if metrics is None:
            certificate["status"] = STATUS_INVALID
            return certificate
        metrics["updates"] = _update_metrics(series, bound, problems)
        windows.append(metrics)
    certificate["window_detail"] = windows
    if problems:
        certificate["status"] = STATUS_INVALID
        return certificate

    rates = [window["success_rate"] for window in windows]
    budgets = [window["budget_mean"] for window in windows]
    intervals = []
    pair_ok = True
    for pair in C.CONVERGENCE_PAIRS:
        left = windows[pair[0]]["cycle_success"]
        right = windows[pair[1]]["cycle_success"]
        interval = pair_interval(left, right, seed=derive_seed(arm, seed, end_episode, pair))
        inside = (interval["lower"] >= -C.CONVERGENCE_PAIR_BOUND
                  and interval["upper"] <= C.CONVERGENCE_PAIR_BOUND)
        interval["pair"] = [pair[0], pair[1]]
        interval["inside_bound"] = bool(inside)
        if not inside:
            pair_ok = False
            problems.append("pair %s: interval [%.6f, %.6f] leaves [%.3f, %.3f]"
                            % (list(pair), interval["lower"], interval["upper"],
                               -C.CONVERGENCE_PAIR_BOUND, C.CONVERGENCE_PAIR_BOUND))
        intervals.append(interval)
    certificate["intervals"] = intervals

    mean_range = max(rates) - min(rates)
    if mean_range <= 1e-12:
        mean_range = 0.0
    mean_range_ok = mean_range <= C.CONVERGENCE_MEAN_RANGE_BOUND
    if not mean_range_ok:
        problems.append("point-mean range %.6f exceeds %.3f"
                        % (mean_range, C.CONVERGENCE_MEAN_RANGE_BOUND))

    slope = ols_slope(rates)
    slope_ok = abs(slope) <= C.CONVERGENCE_SLOPE_BOUND
    if not slope_ok:
        problems.append("window-mean slope %.6f leaves +-%.4f per window"
                        % (slope, C.CONVERGENCE_SLOPE_BOUND))

    budget_range = max(budgets) - min(budgets)
    budget_ok = budget_range <= C.CONVERGENCE_BUDGET_RANGE_BOUND
    if not budget_ok:
        problems.append("budget mean range %.6f exceeds %.3f"
                        % (budget_range, C.CONVERGENCE_BUDGET_RANGE_BOUND))

    certificate["checks"] = {
        "window_success_means": rates,
        "pair_intervals_inside": bool(pair_ok),
        "mean_range": mean_range,
        "mean_range_ok": bool(mean_range_ok),
        "window_mean_slope": slope,
        "slope_ok": bool(slope_ok),
        "budget_means": budgets,
        "budget_mean_range": budget_range,
        "budget_range_ok": bool(budget_ok),
        "budget_saturated_windows": [window["index"] for window in windows
                                     if window["budget_saturated"]],
        "history_provenance": [window["provenance"] for window in windows],
    }
    passed = bool(pair_ok and mean_range_ok and slope_ok and budget_ok and not problems)
    certificate["passed"] = passed
    certificate["status"] = STATUS_PASS if passed else STATUS_FAIL
    certificate["note"] = (
        "equivalence over three 480-episode windows; this is not a global error rate, not a "
        "proof of optimality, and a stable low plateau is still a low plateau")
    return certificate


def consecutive_passes(checks):
    """Trailing consecutive passes, each exactly one check step after the previous one."""
    ordered = sorted(checks, key=lambda item: int(item["end_episode"]))
    count = 0
    previous = None
    for item in ordered:
        episode = int(item["end_episode"])
        if not item.get("passed"):
            count = 0
            previous = episode
            continue
        if previous is not None and episode - previous != C.CONVERGENCE_CHECK_STEP:
            count = 0
        count += 1
        previous = episode
    return count


def is_converged(checks):
    """A run is converged once the trailing consecutive passes reach the required number."""
    return consecutive_passes(checks) >= C.CONVERGENCE_REQUIRED_CONSECUTIVE

