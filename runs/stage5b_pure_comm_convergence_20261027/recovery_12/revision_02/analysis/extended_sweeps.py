"""Five orthogonal probes on the sealed weights: channel, features, budget, logs, latency.

Nothing here trains. Every world is an evaluation world under a registered configuration, and
every comparison arm is either a sealed checkpoint (v4s@300, v4t@100, v4f@200) or a zero-training
rule already used by the paper. The 98 pinned files are untouched.

    I  delivery   comm_delivery_probability in {0.3 .. 1.0} over three cells, six arms
    J1 feature    v4t with one input column forced to a constant, or #11 back to a 50-step clip
    J2 capacity   #16 counterfactually rewritten to 2.0 at the cells that self-throttled
    K  budget     20 %..100 % of immediate's pages, four selectors
    M  latency    wall-clock milliseconds per decision at six goal counts

World counts, with the four cached stages kept out of the totals:

    I:  3 cells x 8 scenes x 7 new probabilities x 6 arms = 1008   (0.7 reuses the archive)
    J1: 15 cells x 3 anchor scenes x 5 variants            =  225
    J2: 6 cells x 3 anchor scenes x 2 (control, probe)     =   36
    K:  2 cells x 8 scenes x 9 fractions x 4 selectors     =  576
    M:  6 cells x 1 scene                                  =    6

Each stage writes ``execution/extended/<stage>/<arm>/RESULTS.json`` and is resumable: a world
already on disk is never re-opened, which is how a partially written arm is retried (ledger #38).

    D:/project_env/python.exe analysis/extended_sweeps.py --stage delivery --workers 12
    D:/project_env/python.exe analysis/extended_sweeps.py --stage features --workers 12
    D:/project_env/python.exe analysis/extended_sweeps.py --stage capacity --workers 12
    D:/project_env/python.exe analysis/extended_sweeps.py --stage budget --workers 12
    D:/project_env/python.exe analysis/extended_sweeps.py --stage latency --workers 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
STAGE_ROOT = BASE.parents[1]
OUT = BASE / "execution/extended"

ANCHOR = (2301901, 2301902, 2301903)
ALL_SCENES = tuple(range(2301901, 2301909))

DELIVERY_CELLS = ((24, 6, 6), (48, 6, 6), (48, 12, 6))
DELIVERY_VALUES = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DELIVERY_ARMS = ("normal_immediate", "dedup_immediate", "event_maxwait",
                 "slack_matched_rate", "v4s@300", "v4t@100")
#: Rule arms (no checkpoint) in the delivery sweep.
DELIVERY_RULE_ARMS = ("normal_immediate", "dedup_immediate", "event_maxwait",
                      "slack_matched_rate")

TRAIN_CELLS = tuple((goals, robots, subscribers)
                    for goals in (12, 24, 48)
                    for robots in (6, 12)
                    for subscribers in (3, 6, 12)
                    if subscribers <= robots)

#: Experiment J1: the five registered mutations of v4t's input row.
FEATURE_VARIANTS = {
    "mask16": {"mask": {"16": 1.0}},
    "mask14": {"mask": {"14": 0.0}},
    "mask2": {"mask": {"2": 0.0}},
    "mask17": {"mask": {"17": 1.0}},
    "horizon50": {"horizon": 50.0},
}

CAPACITY_CELLS = ((48, 12, 3), (48, 12, 6), (60, 12, 3), (60, 12, 6), (72, 12, 3), (72, 12, 6))
CAPACITY_VARIANTS = {"control": None, "cap16x2": {"16": 2.0}}

BUDGET_CELLS = ((48, 6, 6), (48, 12, 6))
BUDGET_FRACTIONS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
BUDGET_SELECTORS = ("v4f", "slack_budget", "fifo_budget", "random_budget")

LATENCY_CELLS = ((12, 6, 6), (24, 6, 6), (48, 6, 6), (72, 6, 6), (108, 6, 6), (144, 6, 6))

LEARNED = {
    "v4f@200": {"tag": "train_v4f_anchor", "episode": 200},
    "v4s@300": {"tag": "train_v4s_cap300", "episode": 300},
    "v4t@100": {"tag": "train_v4t_cap_frozen", "episode": 100},
}


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


def cell_label(cell):
    return "G%d_R%d_N%d" % (int(cell[0]), int(cell[1]), int(cell[2]))


def arm_slug(arm):
    return arm.replace("@", "_")


def arm_path(stage, arm):
    return OUT / stage / arm_slug(arm) / "RESULTS.json"


def identity_sha():
    return sha256(STAGE_ROOT / "recovery_09b/revision_01/protocol"
                  / "RECOVERY_EXPERIMENT_IDENTITY.json")


def sealed_caps():
    """The page budget each sealed checkpoint carries per cell, read from its own state.

    The delivery sweep and J1 must run "the deployed configuration under a different
    condition", so the cap travels with the checkpoint instead of being re-derived from a
    baseline measured under a different channel. v4f has no cap.
    """
    import torch
    caps = {}
    for arm, spec in LEARNED.items():
        path = BASE / "execution" / spec["tag"] / "checkpoints" / (
            "milestone_%d.pth" % spec["episode"])
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        caps[arm] = dict((payload.get("state") or {}).get("cap_by_cell") or {})
    return caps


def archived_immediate_pages():
    """``{cell_label: pages per episode}`` from the sealed baseline plus the G sweep.

    The 15 training cells come from ``grid_v2``; the extended goal counts (60, 72) from the G
    sweep's own ``normal_immediate`` stage, which is the run this revision already published.
    Both are the same arm measured the same way, so the merge is a union of cells, not of two
    conventions.
    """
    totals = {}

    def absorb(rows):
        for row in rows:
            if row.get("arm", "normal_immediate") != "normal_immediate":
                continue
            if row.get("status") != "ok":
                continue
            label = cell_label(tuple(int(value) for value in row["cell"]))
            entry = totals.setdefault(label, [0, 0])
            entry[0] += int(row["emissions"])
            entry[1] += 1

    payload = json.loads((BASE / "execution/baselines/grid_v2/BASELINES.json")
                         .read_text(encoding="utf-8"))
    absorb(payload.get("results") or [])
    sweep = BASE / "execution/g_scaling/normal_immediate/RESULTS.json"
    if sweep.exists():
        absorb(json.loads(sweep.read_text(encoding="utf-8")).get("results") or [])
    return {label: max(1, int(round(pages / worlds)))
            for label, (pages, worlds) in totals.items()}


def matched_caps():
    """The page depth the paper's ``slack_matched_rate`` was matched to, per cell.

    Read from that arm's own archived rows (``throttle_parameters.cap``), not recomputed: the
    arm is defined as "the deadline rule at exactly v4t's pages", and those pages are a property
    of the sealed run. Passing v4t's *ceiling* instead would hand the heuristic a much deeper
    budget (47 pages where v4t spends 23) and the 0.7 point would no longer coincide with the
    archived arm.
    """
    path = BASE / "execution/baselines/paper_slack_matched/BASELINES.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    caps = {}
    for row in payload.get("results") or []:
        if row.get("status") != "ok":
            continue
        label = cell_label(tuple(int(value) for value in row["cell"]))
        cap = (row.get("throttle_parameters") or {}).get("cap")
        if cap is not None:
            caps.setdefault(label, int(cap))
    return caps


def load_tasks(stage, arm):
    path = arm_path(stage, arm)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {(tuple(row["cell"]), int(row["scene"]), str(row.get("variant"))): row
            for row in payload.get("results") or [] if row.get("status") == "ok"}


def write_tasks(stage, arm, rows, extra):
    merged = load_tasks(stage, arm)
    for row in rows:
        if row.get("status") == "ok":
            merged[(tuple(row["cell"]), int(row["scene"]), str(row.get("variant")))] = row
    failures = [row for row in rows if row.get("status") != "ok"]
    payload = {
        "schema": "stage5b-extended-arm/1",
        "stage": stage,
        "arm": arm,
        "root": str(BASE),
        "worlds": len(merged),
        "failures": failures,
        "results": sorted(merged.values(),
                          key=lambda row: (tuple(row["cell"]), int(row["scene"]),
                                           str(row.get("variant")))),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    payload.update(extra)
    dump(arm_path(stage, arm), payload)
    return len(merged), len(failures)


def _layout(root, run_dir):
    from stage5b_comm.paths import Layout
    layout = Layout(root)
    layout.execution = run_dir
    layout.world_logs = run_dir / "world_logs"
    layout.rows = run_dir / "rows"
    layout.ledger = run_dir / "ledger"
    layout.receipts = run_dir / "receipts"
    for path in (layout.world_logs, layout.rows, layout.ledger, layout.receipts):
        path.mkdir(parents=True, exist_ok=True)
    return layout


def _world_readings(row):
    """The four efficiency readings, from the two logs this world just wrote."""
    sys.path.insert(0, str(BASE / "analysis"))
    from comm_efficiency import measure
    archived = list((row.get("artefacts") or {}).values())
    if not archived:
        raise RuntimeError("row carries no artifact paths")
    directory = Path(archived[0]).parent
    readings = measure(directory)
    readings["spent_bytes"] = int(row["spent_bytes"])
    return readings


def _timed_forward(controller, samples):
    """Wrap ``controller.forward`` so every decision's wall-clock is recorded."""
    import torch
    original = controller.forward

    def timed(batch):
        start = time.perf_counter()
        out = original(batch)
        elapsed = (time.perf_counter() - start) * 1000.0
        samples.append({"ms": elapsed, "candidates": int(batch.shape[0])})
        return out

    controller.forward = timed
    return torch


def worker(task):
    sys.dont_write_bytecode = True
    result = {"stage": task["stage"], "arm": task["arm"], "cell": task["cell"],
              "scene": int(task["scene"]), "variant": str(task.get("variant")),
              "cell_label": cell_label(tuple(task["cell"]))}
    started = time.time()
    try:
        root, run_dir = Path(task["root"]), Path(task["run_dir"])
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from stage5b_comm import episode, trainer

        layout = _layout(root, run_dir)
        episode.ensure_import_paths(layout)
        episode.native_import_preflight(layout)
        cell = tuple(int(value) for value in task["cell"])

        if task["kind"] == "rule":
            row = episode.run_rule_world(
                layout, task["arm"], int(task["scene"]), phase="eval", cell=cell,
                immediate_pages=task.get("immediate_pages"),
                matched_pages_by_cell=task.get("matched_pages"),
                delivery_probability=task.get("delivery"),
                variant=task.get("variant") or None, cap=task.get("cap"))
        else:
            controller, _opt, _state = trainer.load_checkpoint(
                Path(task["checkpoint"]), arm="T2", identity_sha256=task["identity_sha"],
                expected_seed=91002, expected_episode=int(task["checkpoint_episode"]))
            samples = []
            if task["stage"] == "latency":
                _timed_forward(controller, samples)
            row = episode.run_eval_world(
                layout, "T2", 91002, int(task["scene"]), controller, cell=cell,
                episode_index=int(task["checkpoint_episode"]),
                cap=task.get("cap"), floor=None,
                delivery_probability=task.get("delivery"),
                feature_mask=task.get("mask"),
                horizon_denominator=task.get("horizon"),
                variant=task.get("variant") or None)
            if samples:
                ordered = sorted(item["ms"] for item in samples)
                index = max(0, min(len(ordered) - 1, int(round(0.99 * (len(ordered) - 1)))))
                result["latency"] = {
                    "decisions": len(ordered),
                    "mean_ms": sum(ordered) / len(ordered),
                    "p50_ms": ordered[len(ordered) // 2],
                    "p99_ms": ordered[index],
                    "candidates_per_decision": (
                        sum(item["candidates"] for item in samples) / float(len(samples))),
                }

        sends = list(row.get("emission_sequence") or [])
        distinct = {str(entry[1]) for entry in sends}
        result.update({
            "status": "ok",
            "on_time": int(row["on_time"]),
            "generated_goals": int(row.get("generated_goals") or 0),
            "emissions": int(row["emissions"]),
            "distinct_targets": len(distinct),
            "duplicate_sends": int(row["emissions"]) - len(distinct),
            "spent_bytes": int(row["spent_bytes"]),
            "subscribers_live": row.get("subscribers_live"),
            "rescue_robots": row.get("rescue_robots"),
            "delivery_probability": row.get("comm_delivery_probability"),
            "transport_route": row.get("transport_assertion_route"),
            "cap": task.get("cap"),
            "scale_audit": {
                "feature_mask": (row.get("scale_invariant") or {}).get("feature_mask"),
                "horizon_denominator": ((row.get("scale_invariant") or {})
                                        .get("horizon_scale") or {}).get(
                                            "installed_denominator"),
            },
            "budget_restored": row.get("budget_authority_restored"),
        })
        if task["stage"] in ("delivery", "budget", "features", "capacity", "nocap"):
            try:
                result.update({("waste_%s" % key): value
                               for key, value in _world_readings(row).items()})
            except Exception as error:  # the reading is auxiliary; the world is still a result
                result["waste_error"] = "%s: %s" % (type(error).__name__, error)
    except Exception as error:  # a failed world is reported, never counted as a result
        result["status"] = "error"
        result["error"] = "%s: %s" % (type(error).__name__, error)
        result["traceback"] = traceback.format_exc()[-2500:]
    result["elapsed_seconds"] = round(time.time() - started, 2)
    return result


def run_tasks(stage, arm, tasks, workers, extra):
    # The caller may hand over a task list that covers several arms (the feature stage builds
    # the five mutations in one pass). Only this call's arm may be submitted: otherwise the
    # next variant re-submits worlds that already exist and every one of them fails with
    # FileExistsError, which is what the first pass of this stage did.
    tasks = [task for task in tasks if task["arm"] == arm]
    pending_tasks = [task for task in tasks
                     if (tuple(task["cell"]), int(task["scene"]), str(task.get("variant")))
                     not in load_tasks(stage, arm)]
    print("%s/%s: %d tasks, %d already on disk" % (stage, arm, len(tasks),
                                                   len(tasks) - len(pending_tasks)), flush=True)
    if not pending_tasks:
        print("%s/%s: nothing to do" % (stage, arm), flush=True)
        return
    started = time.time()
    results, done = [], 0
    with ProcessPoolExecutor(max_workers=int(workers),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(worker, task) for task in pending_tasks}
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                row = future.result()
                results.append(row)
                done += 1
                if row["status"] != "ok":
                    print("  [%d/%d] %s %s s%d FAILED: %s"
                          % (done, len(pending_tasks), row["arm"], row["cell_label"],
                             row["scene"], row.get("error")), flush=True)
                elif done % 50 == 0:
                    print("  [%d/%d] %s" % (done, len(pending_tasks),
                                            time.strftime("%H:%M:%S", time.gmtime())),
                          flush=True)
    worlds, failures = write_tasks(stage, arm, results, extra)
    print("%s/%s: %d worlds on disk, %d failures, %.0fs"
          % (stage, arm, worlds, failures, time.time() - started), flush=True)


# --------------------------------------------------------------------------- stage builders
def delivery_tasks(arm, caps, pages):
    matched = matched_caps()
    tasks = []
    for cell in DELIVERY_CELLS:
        label = cell_label(cell)
        for probability in DELIVERY_VALUES:
            if probability == 0.7:
                continue        # the sealed 0.7 world is reused, never re-run
            for scene in ALL_SCENES:
                task = {"root": str(BASE), "run_dir": str(OUT / "delivery" / arm_slug(arm)),
                        "stage": "delivery", "arm": arm, "cell": list(cell),
                        "scene": int(scene), "variant": "p%02d" % int(round(probability * 100)),
                        "kind": ("rule" if arm in DELIVERY_RULE_ARMS else "learned"),
                        "delivery": float(probability)}
                if arm in DELIVERY_RULE_ARMS:
                    task["immediate_pages"] = int(pages.get(label, 1))
                    if arm == "slack_matched_rate":
                        # The paper's matched depth, held fixed across the channel sweep so the
                        # only thing that moves is p.
                        cap = matched.get(label)
                        if cap is None:
                            raise SystemExit("no archived matched depth for %s" % label)
                        task["matched_pages"] = {label: int(cap)}
                else:
                    spec = LEARNED[arm]
                    task["checkpoint"] = str(BASE / "execution" / spec["tag"] / "checkpoints"
                                             / ("milestone_%d.pth" % spec["episode"]))
                    task["checkpoint_episode"] = spec["episode"]
                    task["identity_sha"] = identity_sha()
                    task["cap"] = caps[arm].get(label)
                tasks.append(task)
    return tasks


def feature_tasks(caps):
    tasks = []
    for variant, mutation in FEATURE_VARIANTS.items():
        for cell in TRAIN_CELLS:
            label = cell_label(cell)
            for scene in ANCHOR:
                task = {"root": str(BASE), "run_dir": str(OUT / "features" / variant),
                        "stage": "features", "arm": variant, "cell": list(cell),
                        "scene": int(scene), "variant": variant, "kind": "learned",
                        "checkpoint": str(BASE / "execution/train_v4t_cap_frozen/checkpoints"
                                          / "milestone_100.pth"),
                        "checkpoint_episode": 100, "identity_sha": identity_sha(),
                        "cap": caps["v4t@100"].get(label)}
                if "mask" in mutation:
                    task["mask"] = dict(mutation["mask"])
                if "horizon" in mutation:
                    task["horizon"] = float(mutation["horizon"])
                tasks.append(task)
    return tasks


def capacity_tasks(caps, pages):
    tasks = []
    for variant, mask in CAPACITY_VARIANTS.items():
        for cell in CAPACITY_CELLS:
            label = cell_label(cell)
            # The control and the probe share one cap: the G sweep's own rule (immediate - 1),
            # so the probe's send count is comparable to the published self-throttling numbers.
            cap = max(1, int(pages.get(label, 1)) - 1)
            for scene in ANCHOR:
                task = {"root": str(BASE), "run_dir": str(OUT / "capacity" / variant),
                        "stage": "capacity", "arm": "v4t_" + variant, "cell": list(cell),
                        "scene": int(scene), "variant": variant, "kind": "learned",
                        "checkpoint": str(BASE / "execution/train_v4t_cap_frozen/checkpoints"
                                          / "milestone_100.pth"),
                        "checkpoint_episode": 100, "identity_sha": identity_sha(),
                        "cap": cap, "mask": dict(mask) if mask else None}
                tasks.append(task)
    return tasks


def budget_tasks(pages):
    tasks = []
    for selector in BUDGET_SELECTORS:
        for cell in BUDGET_CELLS:
            label = cell_label(cell)
            base_pages = int(pages[label])
            for fraction in BUDGET_FRACTIONS:
                cap = max(1, int(round(fraction * base_pages)))
                for scene in ALL_SCENES:
                    variant = "b%03d" % int(round(fraction * 100))
                    task = {"root": str(BASE), "run_dir": str(OUT / "budget" / selector),
                            "stage": "budget", "arm": selector, "cell": list(cell),
                            "scene": int(scene), "variant": variant,
                            "kind": ("learned" if selector == "v4f" else "rule"),
                            "cap": cap}
                    if selector == "v4f":
                        task["checkpoint"] = str(BASE / "execution/train_v4f_anchor"
                                                 / "checkpoints/milestone_200.pth")
                        task["checkpoint_episode"] = 200
                        task["identity_sha"] = identity_sha()
                    else:
                        task["immediate_pages"] = base_pages
                    tasks.append(task)
    return tasks


def nocap_tasks(caps):
    """The same weights as v4t, with **no page table**: 15 cells x 8 scenes.

    This is the deployment question "what happens beyond the table" asked at the cells where
    the table exists, so the two rows differ only by the cap. Passing ``cap=None`` leaves the
    relay's ranked list untruncated, which is exactly what v4f@200 does; the run is therefore
    also a determinism check against the archived v4f evaluation.
    """
    tasks = []
    for cell in TRAIN_CELLS:
        for scene in ALL_SCENES:
            tasks.append({
                "root": str(BASE), "run_dir": str(OUT / "nocap" / "v4t_100"),
                "stage": "nocap", "arm": "v4t@100", "cell": list(cell),
                "scene": int(scene), "variant": "capoff", "kind": "learned",
                "checkpoint": str(BASE / "execution/train_v4t_cap_frozen/checkpoints"
                                  / "milestone_100.pth"),
                "checkpoint_episode": 100, "identity_sha": identity_sha(),
                "cap": None})
    return tasks


def latency_tasks(caps):
    tasks = []
    for cell in LATENCY_CELLS:
        label = cell_label(cell)
        tasks.append({
            "root": str(BASE), "run_dir": str(OUT / "latency" / "v4t"),
            "stage": "latency", "arm": "v4t@100", "cell": list(cell),
            "scene": int(ANCHOR[0]), "variant": "lat", "kind": "learned",
            "checkpoint": str(BASE / "execution/train_v4t_cap_frozen/checkpoints"
                              / "milestone_100.pth"),
            "checkpoint_episode": 100, "identity_sha": identity_sha(),
            "cap": caps["v4t@100"].get(label)})
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=["delivery", "features", "capacity", "budget", "latency", "nocap"])
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    sys.dont_write_bytecode = True

    caps = sealed_caps()
    pages = archived_immediate_pages()
    extra = {"checkpoint_caps": caps, "immediate_pages": pages,
             "identity_sha256": identity_sha()}

    if args.stage == "delivery":
        for arm in DELIVERY_ARMS:
            run_tasks("delivery", arm, delivery_tasks(arm, caps, pages), args.workers, extra)
    elif args.stage == "features":
        # The reference arm is the sealed v4t 45-world anchor evaluation, which the paper table
        # already prints; it is pointed at here rather than re-run.
        for variant in FEATURE_VARIANTS:
            run_tasks("features", variant, feature_tasks(caps), args.workers,
                      dict(extra, variant=variant))
    elif args.stage == "capacity":
        for variant in CAPACITY_VARIANTS:
            run_tasks("capacity", "v4t_" + variant, capacity_tasks(caps, pages),
                      args.workers, dict(extra, variant=variant))
    elif args.stage == "budget":
        for selector in BUDGET_SELECTORS:
            run_tasks("budget", selector, budget_tasks(pages), args.workers, extra)
    elif args.stage == "nocap":
        run_tasks("nocap", "v4t@100", nocap_tasks(caps), args.workers, extra)
    else:
        run_tasks("latency", "v4t@100", latency_tasks(caps), args.workers, extra)


if __name__ == "__main__":
    main()
