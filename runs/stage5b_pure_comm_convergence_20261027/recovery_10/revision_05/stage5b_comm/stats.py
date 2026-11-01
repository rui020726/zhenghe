"""Scene-paired statistics with the archived deterministic SHA256 bootstrap."""

from __future__ import annotations

import hashlib
import math

from . import constants as C


def index_for(seed, replicate, draw, n):
    """Deterministic index: first four digest bytes, big endian, modulo n."""
    text = str(int(seed)) + "|" + str(int(replicate)) + "|" + str(int(draw))
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % int(n)


def bootstrap_ci(values, seed=C.BOOTSTRAP_SEED, reps=C.BOOTSTRAP_RESAMPLES,
                 lower_order=C.BOOTSTRAP_LOWER_ORDER, upper_order=C.BOOTSTRAP_UPPER_ORDER):
    values = [float(v) for v in values]
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "lower": None, "upper": None,
                "resamples": int(reps), "seed": int(seed),
                "order_statistics": [int(lower_order), int(upper_order)]}
    means = []
    for replicate in range(int(reps)):
        total = 0.0
        for draw in range(n):
            total += values[index_for(seed, replicate, draw, n)]
        means.append(total / n)
    means.sort()
    return {"n": n, "mean": sum(values) / n, "lower": means[int(lower_order)],
            "upper": means[int(upper_order)], "resamples": int(reps), "seed": int(seed),
            "order_statistics": [int(lower_order), int(upper_order)]}


def scene_means(rows, key):
    """Mean over seeds inside one scene; repeated seeds are not extra scene samples."""
    buckets = {}
    for row in rows:
        buckets.setdefault(int(row["scene"]), []).append(float(row[key]))
    return {scene: sum(values) / len(values) for scene, values in sorted(buckets.items())}


def paired_scene_deltas(rows, reference_rows, key):
    """Per-scene deltas on scene means; mismatch of scene sets fails closed."""
    left = scene_means(rows, key)
    right = scene_means(reference_rows, key)
    if set(left) != set(right):
        raise RuntimeError("scene sets differ: %s vs %s" % (sorted(left), sorted(right)))
    return [left[scene] - right[scene] for scene in sorted(left)]


def arm_summary(rows, arm, key):
    means = scene_means(rows, key)
    return {"arm": arm, "scenes": len(means),
            "mean": (sum(means.values()) / len(means) if means else None),
            "per_scene": {str(k): round(v, 9) for k, v in means.items()}}


def comparison(rows, reference_rows, key):
    deltas = paired_scene_deltas(rows, reference_rows, key)
    interval = bootstrap_ci(deltas)
    return {"key": key, "per_scene_deltas": [round(v, 9) for v in deltas],
            "mean_delta": interval["mean"], "ci95": [interval["lower"], interval["upper"]],
            "resamples": interval["resamples"], "seed": interval["seed"],
            "order_statistics": interval["order_statistics"], "scene_paired": True}


def energy(on_time, spent_bytes):
    """on_time per 10 kB; null when nothing was spent (never 0.0)."""
    spent = float(spent_bytes)
    if spent <= 0.0:
        return None
    return float(on_time) / (spent / 10000.0)


def finite(value):
    return value is None or (isinstance(value, (int, float)) and math.isfinite(float(value)))

