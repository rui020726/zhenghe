"""Run the four named classical algorithms on the 15-cell x 8-scene benchmark.

Stages
------
``train``     15 cells x 3 *training* scenes with ``feature_log`` (immediate + feature dump).
              Produces one JSONL per world with the 19-dim state at each target's first offer
              and a label read off the lifecycle log (completed inside its own deadline).
``fit``       Ridge regression of the label on the standardised 19-dim state; stores the
              coefficient vector, the feature mean/sd and ``(X'X + lI)^-1`` for the UCB bonus.
``calibrate`` Two *training* scenes x 15 cells at every candidate operating point, plus the
              ``immediate`` reference on the same 30 worlds.  Used only to pick each arm's
              parameter so that the global bytes ratio lands near 0.82; the DEV scenes are
              never touched here.
``choose``    Nearest-to-0.82 point per arm -> ``execution/classic/CHOSEN.json``.
``final``     15 cells x 8 DEV scenes for the four arms -> ``execution/baselines/classic_v1``.

The training and calibration scenes come from the training family (22019xx) and are disjoint
from the eight DEV evaluation scenes (23019xx), so the bandit's parameters never see the
benchmark.  No environment, relay or frozen file is modified; the arms only wrap ``immediate``.

    D:/project_env/python.exe analysis/classic_baselines.py --stage train --workers 8
    D:/project_env/python.exe analysis/classic_baselines.py --stage fit
    D:/project_env/python.exe analysis/classic_baselines.py --stage calibrate --workers 8
    D:/project_env/python.exe analysis/classic_baselines.py --stage choose
    D:/project_env/python.exe analysis/classic_baselines.py --stage final --workers 8
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "execution/baselines/classic_v1"
TRAIN_DIR = BASE / "execution/classic/train"
CALIB_DIR = BASE / "execution/classic/calibration"
MODEL_DIR = BASE / "execution/classic/model"
MODEL_PATH = MODEL_DIR / "linucb.json"
CHOSEN = BASE / "execution/classic/CHOSEN.json"

TRAIN_SCENES = (2201901, 2201902, 2201903)
CALIB_SCENES = (2201904, 2201905)
DEV_SCENES = tuple(range(2301901, 2301909))

ARMS = ("red_aqm", "token_bucket", "ahp_utility", "linucb_bandit")
TARGET_BYTE_RATIO = 0.82

RED_POINTS = tuple({"min_th": 0.0, "max_th": 0.5, "max_p": value}
                   for value in (0.10, 0.20, 0.30, 0.40))
AHP_POINTS = tuple({"threshold": value}
                   for value in (0.18, 0.22, 0.26, 0.29, 0.30, 0.33, 0.36, 0.39, 0.42))
BUCKET_DEPTH = 3.0


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8", newline="\n")
    os.replace(str(tmp), str(path))


def immediate_pages():
    """``{cell_label: emissions per world}`` from the sealed full-broadcast baseline."""
    payload = json.loads((BASE / "execution/baselines/grid_v2/BASELINES.json")
                         .read_text(encoding="utf-8"))
    pages = {}
    for label, entry in (payload.get("per_cell") or {}).items():
        arm = (entry.get("arms") or {}).get("normal_immediate") or {}
        worlds = int(arm.get("worlds") or 0)
        if worlds:
            pages[label] = float(arm["emissions"]) / worlds
    return pages


def token_parameters(cell_label_value, pages, fraction, depth=None):
    """Rate that spends ``fraction`` of ``immediate``'s pages over 1000 steps, after the bucket.

    ``depth="auto"`` sizes the burst tolerance from the reference scale: 60 % of the budget,
    capped at the twelve detections that arrive on the first tick.  It is still a token bucket;
    the depth only says how much of the budget may leave back-to-back.
    """
    if depth == "auto":
        depth = min(12.0, 0.6 * fraction * float(pages))
    depth = float(BUCKET_DEPTH if depth is None else depth)
    rate = max(0.0, (fraction * float(pages) - depth) / 1000.0)
    return {"rate": rate, "depth": depth, "fraction": fraction}


def on_time_targets(directory, window=1000):
    """Targets that completed inside their own deadline -- the same rule as ``waste.py``."""
    with gzip.open(str(Path(directory) / "lifecycle.jsonl.gz"), "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    discovered, completed = {}, {}
    for record in records:
        target = str(record.get("target_id"))
        stamp = float(record.get("timestamp") or 0.0)
        kind = record.get("event_type")
        detail = record.get("detail") or {}
        if kind == "detection" and stamp <= window:
            discovered[target] = stamp + float(detail.get("deadline_offset", 0.0))
        elif kind == "progress" and stamp <= window and str(record.get("status")) == "completed":
            completed.setdefault(target, stamp)
    return {target for target, due in discovered.items()
            if target in completed and completed[target] <= due}


def label_trace(trace_path, world_dir):
    """Append the per-target on-time label to every first-offer row in one trace."""
    on_time = on_time_targets(world_dir)
    path = Path(trace_path)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    header, rows = lines[0], lines[1:]
    out = [header]
    for row in rows:
        row["label"] = int(str(row["target"]) in on_time)
        out.append(row)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True)
                              for row in out) + "\n", encoding="utf-8", newline="\n")
    return sum(row["label"] for row in rows), len(rows)


def worker(task):
    sys.dont_write_bytecode = True
    result = {"cell": list(task["cell"]), "scene": int(task["scene"]), "arm": task["arm"],
              "stage": task["stage"], "tag": task.get("tag"),
              "params": task.get("params")}
    started = time.time()
    try:
        root = Path(task["root"])
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from stage5b_comm import episode, scale_program
        from stage5b_comm.paths import Layout

        layout = Layout(root)
        layout.execution = Path(task["run_dir"])
        layout.world_logs = layout.execution / "world_logs"
        layout.rows = layout.execution / "rows"
        layout.ledger = layout.execution / "ledger"
        layout.receipts = layout.execution / "receipts"
        for path in (layout.world_logs, layout.rows, layout.ledger, layout.receipts):
            path.mkdir(parents=True, exist_ok=True)
        episode.ensure_import_paths(layout)
        episode.native_import_preflight(layout)

        cell = tuple(int(v) for v in task["cell"])
        scene = int(task["scene"])
        params = dict(task.get("params") or {})
        seed = scale_program.eval_seed(scene, cell)
        if task["arm"] == "red_aqm":
            params.setdefault("seed", seed)
        row = episode.run_rule_world(
            layout, task["arm"], scene, phase="eval", cell=cell,
            classic_params=params, model_path=task.get("model_path"),
            search_agents=task.get("search_agents"))
        emissions = list(row.get("emission_sequence") or [])
        distinct = {str(entry[1]) for entry in emissions}
        result.update({
            "status": "ok",
            "on_time": int(row["on_time"]),
            "generated_goals": int(row.get("generated_goals") or 0),
            "emissions": int(row["emissions"]),
            "distinct_targets": len(distinct),
            "duplicate_sends": int(row["emissions"]) - len(distinct),
            "spent_bytes": int(row["spent_bytes"]),
            "bytes_per_on_time": row.get("bytes_per_on_time"),
            "on_time_rate": row.get("on_time_rate"),
            "budget_refused": int(row.get("masked_by_credit_total") or 0),
            "capacity_refused": int(row.get("masked_by_capacity_total") or 0),
            "subscribers_live": row.get("subscribers_live"),
            "rescue_robots": row.get("rescue_robots"),
            "cell_label": row.get("cell_label") or scale_program.cell_label(cell),
            "search_agents": task.get("search_agents"),
            "budget_bytes": int(row.get("budget_bytes") or 0),
            "budget_restored": row.get("budget_authority_restored"),
            "classic_parameters": row.get("rule_parameters"),
        })
        if task["stage"] == "train":
            artefacts = row.get("artefacts") or {}
            world_dir = Path(next(iter(artefacts.values()))).parent
            positives, total = label_trace(params["trace"], world_dir)
            result["trace_rows"] = total
            result["trace_positives"] = positives
    except Exception as error:
        result["status"] = "error"
        result["error"] = "%s: %s" % (type(error).__name__, error)
        result["traceback"] = traceback.format_exc()[-3000:]
    result["elapsed_seconds"] = round(time.time() - started, 2)
    return result


def run_tasks(tasks, workers, progress=True):
    results = []
    with ProcessPoolExecutor(max_workers=int(workers),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(worker, task): task for task in tasks}
        done_count = 0
        while pending:
            finished, _ = wait(set(pending), return_when=FIRST_COMPLETED)
            for future in finished:
                task = pending.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = {"cell": list(task["cell"]), "scene": int(task["scene"]),
                              "arm": task["arm"], "stage": task["stage"],
                              "status": "executor_error",
                              "error": "%s: %s" % (type(error).__name__, error)}
                results.append(result)
                done_count += 1
                if progress:
                    print("  [%d/%d] %-16s s%-7d %-14s on_time=%-3s/%s bytes=%-7s"
                          % (done_count, len(tasks), result["cell"], result["scene"],
                             result["arm"], result.get("on_time"), result.get("generated_goals"),
                             result.get("spent_bytes")), flush=True)
    failures = [row for row in results if row.get("status") != "ok"]
    if failures:
        print("FAILURES: %d" % len(failures))
        for failure in failures[:10]:
            print("  ", failure.get("cell"), failure.get("scene"), failure.get("arm"),
                  failure.get("error"))
    return results


def stage_train(workers):
    from stage5b_comm import scale_program

    traces = TRAIN_DIR / "traces"
    tasks = []
    for cell in scale_program.CELLS:
        label = "%d_%d_%d" % cell
        for scene in TRAIN_SCENES:
            trace = traces / ("%s_s%d.jsonl" % (label, scene))
            tasks.append({"root": str(BASE), "run_dir": str(TRAIN_DIR / "worlds"),
                          "cell": list(cell), "scene": scene, "arm": "feature_log",
                          "stage": "train", "params": {"trace": str(trace)}})
    print("train: %d worlds (%d cells x %d training scenes)"
          % (len(tasks), 15, len(TRAIN_SCENES)))
    results = run_tasks(tasks, workers)
    payload = {"schema": "stage5b-classic-train/1", "scenes": list(TRAIN_SCENES),
               "worlds": len(results), "results": results, "ts": utc()}
    dump(TRAIN_DIR / "TRAIN_WORLDS.json", payload)
    ok = [row for row in results if row.get("status") == "ok"]
    print("train: %d/%d worlds, %d first-offer rows, %d positives"
          % (len(ok), len(results), sum(row["trace_rows"] for row in ok),
             sum(row["trace_positives"] for row in ok)))
    return results


def auc(scores, labels):
    import numpy as np

    scores = np.asarray(scores, dtype="float64")
    labels = np.asarray(labels, dtype="int64")
    positives = labels == 1
    negatives = ~positives
    if positives.sum() == 0 or negatives.sum() == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype="float64")
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    unique = np.unique(scores)
    for value in unique:
        mask = scores == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return float((ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2.0)
                 / (positives.sum() * negatives.sum()))


def stage_fit():
    import numpy as np

    rows = []
    for path in sorted((TRAIN_DIR / "traces").glob("*.jsonl")):
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                 if line]
        rows.extend(line for line in lines[1:] if "state" in line)
    if not rows:
        raise SystemExit("no training rows; run --stage train first")
    states = np.asarray([row["state"] for row in rows], dtype="float64")
    labels = np.asarray([int(row["label"]) for row in rows], dtype="float64")
    mean = states.mean(axis=0)
    sd = states.std(axis=0)
    sd[sd == 0.0] = 1.0
    standard = (states - mean) / sd
    design = np.hstack([np.ones((len(standard), 1)), standard])
    ridge = 1.0
    alpha = 0.5
    matrix = design.T @ design + ridge * np.eye(design.shape[1])
    theta = np.linalg.solve(matrix, design.T @ labels)
    a_inv = np.linalg.inv(matrix)
    scores = design @ theta
    quad = np.einsum("ij,jk,ik->i", design, a_inv, design)
    ucb = scores + alpha * np.sqrt(np.maximum(0.0, quad))
    candidates = [float(np.quantile(ucb, 1.0 - rate)) for rate in (0.70, 0.80, 0.88)]
    model = {
        "schema": "stage5b-classic-linucb/1",
        "theta": theta.tolist(),
        "alpha": alpha,
        "threshold": candidates[1],
        "mean": mean.tolist(),
        "sd": sd.tolist(),
        "a_inv": a_inv.tolist(),
        "threshold_candidates": candidates,
        "training": {
            "scenes": list(TRAIN_SCENES),
            "worlds": len(list((TRAIN_DIR / "traces").glob("*.jsonl"))),
            "samples": int(len(rows)),
            "positives": int(labels.sum()),
            "base_rate": float(labels.mean()),
            "auc_in_sample": auc(ucb, labels),
            "ridge": ridge,
            "ucb_alpha": alpha,
            "state": "19-dim comm-send-state/1, revision scaling, #11 denominator 200",
            "reward": "1 if the target completed inside its deadline under immediate, else 0",
        },
    }
    dump(MODEL_PATH, model)
    print("fit: %d samples, base rate %.4f, AUC(UCB score) %.4f"
          % (len(rows), labels.mean(), auc(ucb, labels) or 0.0))
    print("threshold candidates (admission 70/80/88 %%): %s"
          % ", ".join("%.4f" % value for value in candidates))
    print("wrote", MODEL_PATH)


def calibration_points():
    points = {"normal_immediate": [{}],
              "token_bucket": [{"fraction": TARGET_BYTE_RATIO, "depth": depth}
                               for depth in (3.0, 5.0, 8.0, "auto")],
              "red_aqm": list(RED_POINTS),
              "ahp_utility": list(AHP_POINTS)}
    if MODEL_PATH.exists():
        model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        # The intercept is the score of an average context; the grid is expressed as a margin
        # above it.  The raw training quantiles (0.63--0.72) sit below the intercept and admit
        # every candidate at evaluation, so the calibration grid is the intercept plus
        # 0.05/0.09/0.13/0.17 -- still strictly above the mean row, still one fixed point per
        # run, and chosen without touching a DEV world.
        base = float(model["theta"][0])
        points["linucb_bandit"] = [{"threshold": base + delta}
                                   for delta in (-0.07, -0.05, -0.02, 0.0, 0.02,
                                                 0.05, 0.09, 0.13, 0.17)]
    return points


def point_tag(point):
    def text(value):
        if isinstance(value, str):
            return value
        return ("%.4f" % value).rstrip("0").rstrip(".").replace("-", "m")

    return "_".join("%s%s" % (key, text(value)) for key, value in sorted(point.items()))


def stage_calibrate(workers, only=None):
    from stage5b_comm import scale_program

    pages = immediate_pages()
    points = calibration_points()
    tasks = []
    for arm, arm_points in points.items():
        if only and arm not in only:
            continue
        for point in arm_points:
            tag = point_tag(point) if point else "reference"
            run_dir = CALIB_DIR / arm / tag
            if (run_dir / "RESULTS.json").exists():
                print("calibrate: skip %s %s (already measured)" % (arm, tag))
                continue
            for cell in scale_program.CELLS:
                label = scale_program.cell_label(cell)
                params = dict(point)
                if arm == "token_bucket":
                    params.update(token_parameters(label, pages[label], point["fraction"],
                                                   point.get("depth")))
                for scene in CALIB_SCENES:
                    tasks.append({"root": str(BASE), "run_dir": str(run_dir),
                                  "cell": list(cell), "scene": scene, "arm": arm,
                                  "stage": "calibrate", "params": params,
                                  "tag": tag,
                                  "model_path": str(MODEL_PATH)})
    print("calibrate: %d worlds (%d points x %d cells x %d scenes)"
          % (len(tasks), sum(len(value) for value in points.values()), 15, len(CALIB_SCENES)))
    results = run_tasks(tasks, workers)
    for arm, arm_points in points.items():
        if only and arm not in only:
            continue
        for point in arm_points:
            tag = point_tag(point) if point else "reference"
            if (CALIB_DIR / arm / tag / "RESULTS.json").exists():
                continue
            own = [row for row in results if row["arm"] == arm and row.get("tag") == tag]
            payload = {"schema": "stage5b-classic-calibration/1", "arm": arm,
                       "point": point, "scenes": list(CALIB_SCENES),
                       "worlds": len(own), "results": own, "ts": utc()}
            dump(CALIB_DIR / arm / tag / "RESULTS.json", payload)
    print("calibrate: wrote %d point files" % sum(len(v) for v in points.values()))


def stage_choose():
    points = calibration_points()
    reference = None
    ref_path = CALIB_DIR / "normal_immediate" / "reference" / "RESULTS.json"
    if not ref_path.exists():
        raise SystemExit("no immediate calibration reference; run --stage calibrate")
    reference = json.loads(ref_path.read_text(encoding="utf-8"))["results"]
    base = {}
    for row in reference:
        key = (tuple(row["cell"]), int(row["scene"]))
        base[key] = int(row["spent_bytes"])
    chosen = {}
    for arm, arm_points in points.items():
        if arm == "normal_immediate":
            continue
        best = None
        for point in arm_points:
            tag = point_tag(point)
            path = CALIB_DIR / arm / tag / "RESULTS.json"
            if not path.exists():
                continue
            rows = [row for row in json.loads(path.read_text(encoding="utf-8"))["results"]
                    if row.get("status") == "ok"]
            spent = sum(int(row["spent_bytes"]) for row in rows)
            denominator = sum(base[(tuple(row["cell"]), int(row["scene"]))] for row in rows
                              if (tuple(row["cell"]), int(row["scene"])) in base)
            ratio = spent / denominator if denominator else None
            on_time = sum(int(row["on_time"]) for row in rows)
            goals = sum(int(row["generated_goals"]) for row in rows)
            candidate = {"point": point, "byte_ratio": ratio,
                         "on_time_rate": (on_time / goals) if goals else None,
                         "worlds": len(rows)}
            if ratio is None:
                continue
            if (best is None
                    or abs(ratio - TARGET_BYTE_RATIO)
                    < abs(best["byte_ratio"] - TARGET_BYTE_RATIO)):
                best = candidate
            elif (arm == "token_bucket"
                  and abs(ratio - TARGET_BYTE_RATIO) <= 0.03
                  and abs(best["byte_ratio"] - TARGET_BYTE_RATIO) <= 0.03
                  and candidate["on_time_rate"] > best["on_time_rate"]):
                # The bucket depth is the one parameter the byte target does not pin down.
                # Among points that all hit ~0.82 bytes, keep the one that keeps more tasks.
                best = candidate
        if best is not None:
            chosen[arm] = best
    payload = {"schema": "stage5b-classic-chosen/1", "target_byte_ratio": TARGET_BYTE_RATIO,
               "chosen": chosen, "ts": utc()}
    dump(CHOSEN, payload)
    for arm, best in chosen.items():
        print("%-14s %-40s bytes %.3f on-time %.4f"
              % (arm, json.dumps(best["point"], sort_keys=True), best["byte_ratio"],
                 best["on_time_rate"]))
    print("wrote", CHOSEN)


def stage_final(workers, only=None, override=None):
    from stage5b_comm import scale_program

    chosen = json.loads(CHOSEN.read_text(encoding="utf-8"))["chosen"]
    pages = immediate_pages()
    arms = tuple(arm for arm in ARMS if not only or arm in only)
    logs = OUT / "world_logs"
    logs.mkdir(parents=True, exist_ok=True)
    superseded = logs / "_superseded"
    for arm in arms:
        stale = sorted(logs.glob("eval__%s_*" % arm))
        if stale:
            superseded.mkdir(exist_ok=True)
            for directory in stale:
                directory.rename(superseded / ("%s_%s" % (arm, directory.name)))
            print("final: quarantined %d %s worlds under %s"
                  % (len(stale), arm, superseded))
    tasks = []
    for arm in arms:
        point = dict(override if (override and len(arms) == 1) else chosen[arm]["point"])
        for cell in scale_program.CELLS:
            label = scale_program.cell_label(cell)
            params = dict(point)
            if arm == "token_bucket":
                params.update(token_parameters(label, pages[label], point["fraction"],
                                               point.get("depth")))
            for scene in DEV_SCENES:
                tasks.append({"root": str(BASE), "run_dir": str(OUT), "cell": list(cell),
                              "scene": scene, "arm": arm, "stage": "final",
                              "params": params, "model_path": str(MODEL_PATH)})
    print("final: %d worlds (%d arms x 15 cells x 8 DEV scenes)"
          % (len(tasks), len(arms)))
    results = run_tasks(tasks, workers)
    previous = []
    archive = OUT / "BASELINES.json"
    if archive.exists():
        keep = {arm for arm in ARMS if arm not in arms}
        previous = [row for row in json.loads(archive.read_text(encoding="utf-8"))["results"]
                    if row.get("arm") in keep]
        print("final: kept %d rows of the arms that were not re-run" % len(previous))
    results = previous + results
    ok = [row for row in results if row.get("status") == "ok"]
    per_cell = {}
    for cell in sorted({tuple(row["cell"]) for row in ok}):
        entry = {}
        for arm in sorted({row["arm"] for row in ok}):
            rows = [row for row in ok if row["arm"] == arm and tuple(row["cell"]) == cell]
            on_time = sum(row["on_time"] for row in rows)
            generated = sum(row["generated_goals"] for row in rows)
            entry[arm] = {"worlds": len(rows), "on_time": on_time,
                          "generated_goals": generated,
                          "q": (on_time / generated) if generated else None,
                          "spent_bytes": sum(row["spent_bytes"] for row in rows),
                          "emissions": sum(row["emissions"] for row in rows)}
        per_cell["G%d_R%d_N%d" % cell] = {"cell": list(cell), "arms": entry}
    payload = {
        "schema": "stage5b-revision02-classic/1",
        "root": str(BASE),
        "scenes": list(DEV_SCENES),
        "arms": list(arms),
        "worlds": len(results),
        "worlds_ok": len(ok),
        "chosen": chosen,
        "override": override,
        "per_cell": per_cell,
        "results": sorted(results, key=lambda row: (tuple(row["cell"]), row["arm"],
                                                    row["scene"])),
        "ts": utc(),
    }
    dump(OUT / "BASELINES.json", payload)
    print("final: %d/%d worlds ok" % (len(ok), len(results)))
    print("wrote", OUT / "BASELINES.json")
    print("sha256", sha256(OUT / "BASELINES.json"))


GEXT_GOALS = (54, 60, 72)
GEXT_PAIRS = ((6, 3), (6, 6), (12, 3), (12, 6), (12, 12))
GEXT_FULL_SCENE_PAIRS = ((6, 6), (12, 6))
ANCHOR3 = (2301901, 2301902, 2301903)
DEEP_GOALS = (72, 84, 96, 108, 120, 144)
DEEP_ROBOTS = (6, 12, 18)
DEEP_AGENTS = (1, 2, 4)


def scale_pages(grid):
    """immediate pages per world for every (cell[, S]) of the scale grid being extended."""
    path = BASE / ("execution/g_scaling/normal_immediate/RESULTS.json" if grid == "g"
                   else "execution/g_rs_grid/normal_immediate/RESULTS.json")
    acc = {}
    for row in json.loads(path.read_text(encoding="utf-8"))["results"]:
        if row.get("status") != "ok":
            continue
        key = (tuple(row["cell"]), row.get("search_agents"))
        entry = acc.setdefault(key, [0, 0])
        entry[0] += int(row["emissions"])
        entry[1] += 1
    pages = {key: total / worlds for key, (total, worlds) in acc.items()}
    if grid == "g":
        # G=12/24/48 are archived in the 15-cell baseline, not in the sweep's own file.
        for row in rows_of_baseline():
            key = (tuple(row["cell"]), None)
            entry = acc.setdefault(key, [0, 0])
            entry[0] += int(row["emissions"])
            entry[1] += 1
        pages = {key: total / worlds for key, (total, worlds) in acc.items()}
    return pages


def rows_of_baseline():
    payload = json.loads((BASE / "execution/baselines/grid_v2/BASELINES.json")
                         .read_text(encoding="utf-8"))
    return [row for row in payload.get("results") or []
            if row.get("status") == "ok" and row.get("arm") == "normal_immediate"]


def stage_scale(workers, grid, goals=None):
    """Run the four classical arms, at their frozen operating points, on a scale grid.

    No parameter is re-tuned here: the point chosen on the 15-cell benchmark is carried over
    unchanged, so the run measures whether the operating point *transfers* to larger worlds.
    The token bucket's rate is the one per-cell quantity that must be recomputed, because it
    is defined as a fraction of that cell's own immediate pages.
    """
    from stage5b_comm import scale_program

    chosen = json.loads(CHOSEN.read_text(encoding="utf-8"))["chosen"]
    pages = scale_pages(grid)
    if grid == "g":
        run_goals = tuple(goals) if goals else GEXT_GOALS
        points = [((goal, robots, subscribers), None)
                  for goal in run_goals for robots, subscribers in GEXT_PAIRS]
        out_root = BASE / "execution/g_scaling"
        schema = "stage5b-g-scaling-arm/1"
        meta = {"goals": list(run_goals), "pairs": [list(p) for p in GEXT_PAIRS],
                "archived_goals": [12, 24, 48], "new_goals": list(run_goals)}
    else:
        points = [((goal, robots, 6), agents)
                  for goal in DEEP_GOALS for robots in DEEP_ROBOTS for agents in DEEP_AGENTS]
        out_root = BASE / "execution/g_rs_grid"
        schema = "stage5b-g-rs-grid-arm/1"
        meta = {"goals": list(DEEP_GOALS), "robots": list(DEEP_ROBOTS),
                "search_agents": list(DEEP_AGENTS), "subscribers": 6,
                "scenes": list(ANCHOR3)}
    tasks = []
    for arm in ARMS:
        point = dict(chosen[arm]["point"])
        run_dir = out_root / arm
        for cell, agents in points:
            params = dict(point)
            if arm == "token_bucket":
                key = (cell, agents)
                if key not in pages:
                    raise SystemExit("no immediate pages for %s S=%s" % (cell, agents))
                params.update(token_parameters(scale_program.cell_label(cell), pages[key],
                                               point["fraction"], point.get("depth")))
            scenes = (DEV_SCENES if (grid == "g" and (cell[1], cell[2]) in GEXT_FULL_SCENE_PAIRS)
                      else ANCHOR3)
            for scene in scenes:
                tasks.append({"root": str(BASE), "run_dir": str(run_dir), "cell": list(cell),
                              "scene": scene, "arm": arm, "stage": "scale",
                              "params": params, "search_agents": agents,
                              "model_path": str(MODEL_PATH)})
    print("scale(%s): %d worlds (%d arms)" % (grid, len(tasks), len(ARMS)))
    results = run_tasks(tasks, workers)
    for arm in ARMS:
        own = [row for row in results if row["arm"] == arm]
        target = out_root / arm / "RESULTS.json"
        if target.exists():
            keep_goals = {int(row["cell"][0]) for row in own}
            previous = [row for row in json.loads(target.read_text(encoding="utf-8"))["results"]
                        if int(row["cell"][0]) not in keep_goals]
            print("scale(%s): %s keeps %d earlier worlds" % (grid, arm, len(previous)))
            own = previous + own
        ok = [row for row in own if row.get("status") == "ok"]
        payload = {"schema": schema, "root": str(BASE), "arm": arm,
                   "worlds": len(own), "results": sorted(
                       own, key=lambda row: (tuple(row["cell"]), row.get("search_agents") or 0,
                                             row["scene"])),
                   "failures": [row for row in own if row.get("status") != "ok"],
                   "operating_point": chosen[arm]["point"], "ts": utc()}
        payload.update(meta)
        payload["goals"] = sorted({int(row["cell"][0]) for row in own})
        dump(out_root / arm / "RESULTS.json", payload)
        print("scale(%s): %s %d/%d worlds ok" % (grid, arm, len(ok), len(own)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("train", "fit", "calibrate", "choose", "final", "scale"))
    parser.add_argument("--grid", default="g", choices=("g", "deep"))
    parser.add_argument("--goals", default="", help="comma-separated goal counts for --grid g")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--arms", default="", help="comma-separated arms to calibrate")
    parser.add_argument("--override", default="", help="JSON point for a single-arm re-run")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(BASE))
    if args.stage == "train":
        stage_train(args.workers)
    elif args.stage == "fit":
        stage_fit()
    elif args.stage == "calibrate":
        only = tuple(a for a in args.arms.split(",") if a) or None
        stage_calibrate(args.workers, only)
    elif args.stage == "choose":
        stage_choose()
    elif args.stage == "scale":
        goals = tuple(int(v) for v in args.goals.split(",") if v) or None
        stage_scale(args.workers, args.grid, goals)
    else:
        only = tuple(a for a in args.arms.split(",") if a) or None
        override = json.loads(args.override) if args.override else None
        stage_final(args.workers, only, override)


if __name__ == "__main__":
    main()
