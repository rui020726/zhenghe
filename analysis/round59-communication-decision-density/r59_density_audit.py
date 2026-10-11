"""ROUND59: read existing, indexed R58 artifacts; no runtime/simulator imports."""
from __future__ import annotations

import bisect
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = REPO / "runs/comm_bc_initialized_constrained_ppo_20261007_r2"
R58 = BASE / "round58_isolation_stop_audit"
NAMES = ["age_steps", "since_last_emission_time", "sent_count", "pending_event",
         "changed_view", "first_emission", "stage_changed", "requirements_changed",
         "members_changed", "abs_progress_delta", "revision_delta", "remaining_time_div50_clipped",
         "urgent", "has_future_requirements", "backlog_targets", "budget_fraction",
         "capacity_fraction", "subscribers", "step_div1000_clipped"]
PARTITIONS = ("free", "forced", "masked", "fixed_hold_one_shot_closed")
PROTECTED = {"ROUND46_CURRENT_TASK.json": "cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b",
             "ROUND46_R2_V6_STOP_STATE.json": "93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def distribution(values, histogram=True):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {"n": 0, "min": None, "max": None, "mean": None, "histogram": {}}
    return {"n": len(a), "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "std_population": float(a.std()),
            "quantiles_linear": {str(q): float(np.quantile(a, q, method="linear"))
                                  for q in (0, .25, .5, .75, .9, 1)},
            "distinct_values": len(set(values)),
            "histogram": {str(k): v for k, v in sorted(Counter(values).items())} if histogram else None}


def discovery_counts(placed, discovered_ids, on_time_ids):
    discovered, on_time = set(discovered_ids), set(on_time_ids)
    if not on_time <= discovered or len(discovered) > placed:
        raise ValueError("completion/discovery census conflict")
    return {"placed": placed, "discovered": len(discovered), "on_time": len(on_time),
            "discovered_not_on_time": len(discovered - on_time),
            "undiscovered_at_window_end": placed - len(discovered)}


def partition(tape):
    sets = {name: set(tape[name]) for name in PARTITIONS}
    targets = set(tape["targets"])
    if len(targets) != len(tape["targets"]) or set.union(*sets.values()) != targets:
        raise ValueError("candidate partitions fail exact closure")
    if sum(len(s) for s in sets.values()) != len(targets):
        raise ValueError("candidate partitions overlap")
    if sets["free"] != set(tape["actionable"]) or sets["free"] != set(tape["free_one_shot_choices"]):
        raise ValueError("free/actionable/one-shot set mismatch")
    return sets


def gate(vector, tape_step=None):
    f = np.asarray(vector, dtype="<f4")
    if f.shape != (19,) or not np.isfinite(f).all():
        raise ValueError("not a legal finite 19-D actor row")
    step = round(float(f[18]) * 1000)
    age = round(float(f[0]))
    if tape_step is not None and step != tape_step:
        raise ValueError("float32 clock recovery disagrees with recorded tape step")
    if abs(float(f[0])-age) > 1e-6 or f[5] not in (0, 1):
        raise ValueError("age/first emission is not on frozen integer encoding")
    ready = step - age
    return bool(f[5] == 1), ready, ready > 147


def id_difference(candidate, reference):
    return {"lost": sorted(set(reference)-set(candidate)),
            "rescued": sorted(set(candidate)-set(reference)),
            "net_count": len(set(candidate))-len(set(reference))}


def credit_task_stream(stream, tapes):
    steps = [int(t["step"]) for t in tapes]
    if steps != sorted(set(steps)):
        raise ValueError("decision steps not increasing and unique")
    bundled, prefix = [0.0] * len(steps), 0.0
    for tick, reward in stream.items():
        idx = bisect.bisect_right(steps, int(tick))-1
        if idx < 0:
            prefix += float(reward)
        else:
            bundled[idx] += float(reward)
    return bundled, prefix


class Inputs:
    def __init__(self):
        self.rows = 0
        self.vectors = Counter()
        self.columns = [set() for _ in NAMES]
        self.structure = Counter()

    def add(self, vector):
        f = np.asarray(vector, dtype="<f4")
        self.rows += 1
        self.vectors[f.tobytes()] += 1
        self.structure[f[[i for i in range(19) if i not in (0, 15, 18)]].tobytes()] += 1
        for i, v in enumerate(f):
            self.columns[i].add(float(v))

    def result(self):
        return {"rows": self.rows, "exact_float32_vectors": len(self.vectors),
                "duplicate_exposures_beyond_first": self.rows-len(self.vectors),
                "rows_belonging_to_repeated_vectors": sum(n for n in self.vectors.values() if n > 1),
                "vector_multiplicity_histogram": dict(sorted(Counter(self.vectors.values()).items())),
                "signatures_excluding_age_budget_clock": len(self.structure),
                "signature_exclusion_slots": [0, 15, 18],
                "columns": [{"index": i, "name": NAMES[i], "min": min(v) if v else None,
                             "max": max(v) if v else None, "distinct_float32_values": len(v),
                             "constant": len(v) == 1} for i, v in enumerate(self.columns)]}


def group_name(job):
    if job["scope"] == "ppo_training":
        return f"train/r58/s{job['seed']}/sampled"
    if job["scope"].startswith("rule_eval_"):
        return f"eval/{job['rule_arm']}/rule"
    return f"eval/{job['scope'].removeprefix('model_eval_')}/s{job['seed']}/{job['mode']}"


def main():
    protected_start = {name: sha(BASE/name) for name in PROTECTED}
    if protected_start != PROTECTED:
        raise RuntimeError("protected input differs from approved takeover hashes")
    inputs = {}
    for name in ("ROUND58_FINAL_REPORT.md", "ROUND58_FINAL_SOURCE_INDEX.json", "ROUND58_FINAL_STATISTICS.json",
                 "ROUND58_FINAL_ACCOUNTING.json", "ROUND58_FINAL_CHECKPOINT_INDEX.json", "ROUND58_FINAL_MACHINE_VERDICT.json",
                 "ROUND58_GEN3_FIXED_PLAN.json", "ROUND58_PRELAUNCH_CANDIDATE.json", "ROUND58_FINAL_DIAGNOSTIC_COVERAGE.json"):
        inputs[name] = {"path": str(R58/name), "sha256": sha(R58/name)}
    machine = read(R58/"ROUND58_FINAL_MACHINE_VERDICT.json")
    for name, expected in machine["evidence"].items():
        if sha(R58/name) != expected:
            raise RuntimeError(f"R58 machine input evidence changed: {name}")
    index = read(R58/"ROUND58_FINAL_SOURCE_INDEX.json")["sources"]
    jobs = {x["logical_job"]: x for x in read(R58/"ROUND58_GEN3_FIXED_PLAN.json")["jobs"]}
    if len(index) != 1024 or {x["logical_job"] for x in index} != set(jobs):
        raise RuntimeError("input graph is not the one complete frozen 1024-key plan")
    candidate = read(R58/"ROUND58_PRELAUNCH_CANDIDATE.json")
    for item in candidate["runtime_source_hashes"].values():
        if sha(item["path"]) != item["sha256"]:
            raise RuntimeError("frozen canonical input source changed")
    external = REPO.parent / "zhenghe"
    source_paths = [external/"integrated_sar/lifecycle_log.py", external/"integrated_sar/adapters/piloc.py",
                    external/"integrated_sar/comm_learning.py", external/"integrated_sar/comm.py",
                    external/"runs/comm_cost_v2_rerun_20260928/v2_trainer.py",
                    BASE/"r42_one_shot.py", REPO/"analysis/round55-late-gate-observability/SOURCE_INPUT_MAP.json"]
    source_identity = [{"path": str(p), "sha256": sha(p)} for p in source_paths]
    groups, episode_rows = {}, []
    free_provenance = {k:defaultdict(list) for k in ("train", "eval", "eval_r58")}
    pooled_free = {k:Inputs() for k in free_provenance}
    discovery_profiles = defaultdict(set)
    verified_artifacts = 0
    for ordinal, entry in enumerate(index):
        job = jobs[entry["logical_job"]]
        name = group_name(job)
        if name not in groups:
            groups[name] = {"episodes": [], "all_inputs": Inputs(), "free_inputs": Inputs(),
                            "counts": Counter(), "probabilities": defaultdict(list), "task_values": [],
                            "task_credit_rows": Counter(), "task_credit_reward": Counter(),
                            "saved_task_advantages": [], "scene_exposures": Counter()}
        g = groups[name]
        if sha(entry["claim"]) != entry["claim_sha256"]:
            raise RuntimeError("approved complete claim identity changed")
        claim = read(entry["claim"])
        if claim["status"] != "complete":
            raise RuntimeError("indexed source is no longer complete")
        receipt = claim.get("round45_artifact_receipt") or claim.get("round58_rule_artifact_receipt")
        raw = {}
        for kind, expected in entry["artifact_sha256"].items():
            item = receipt["artifacts"][kind]
            payload = Path(item["path"]).read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected or expected != item["sha256"] or len(payload) != item["size_bytes"]:
                raise RuntimeError(f"input artifact changed: {entry['logical_job']}/{kind}")
            raw[kind] = (json.loads(payload.decode("utf-8")) if kind == "row" else
                         [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()])
            verified_artifacts += 1
        row, lifecycle, transcript = raw["row"], raw["lifecycle"], raw["transcript"]
        config_events = [e for e in lifecycle if e["event_type"] == "run_config"]
        finals = [e for e in lifecycle if e["event_type"] == "final"]
        final_runs = [e for e in finals if e["target_id"] == "__run__"]
        if len(config_events) != 1 or len(final_runs) != 1:
            raise RuntimeError("missing unambiguous configuration/final census")
        detail = config_events[0]["detail"]
        cfg, scene_cfg = detail["configuration"], detail["scenario"]
        if cfg["num_goals"] != 6 or cfg["steps"] != 1000 or cfg["dt"] != 1.0 or int(cfg["seed"]) != job["scene"]:
            raise RuntimeError("science scene/horizon/clock identity conflict")
        if detail["run_signature"]["code_commit"] != "4e989a9088111ce6d9171cb1a21629a4b2b1e861":
            raise RuntimeError("unregistered canonical execution source identity")
        placed = int(scene_cfg["num_goals_placed"])
        placed_ids = {f"piloc-e1-target-{i}" for i in range(placed)}
        if len(scene_cfg["goal_positions"]) != placed:
            raise RuntimeError("placed-goal denominator differs from initial scenario")
        detections = {}
        for e in lifecycle:
            if e["event_type"] == "detection":
                tid = e["target_id"]
                detections[tid] = min(detections.get(tid, float("inf")), float(e["timestamp"]))
                if not e["source_event_id"].startswith("detection:") or e["discovered_by"] is None:
                    raise RuntimeError("detection is not sourced from a search DetectionEvent")
        metrics = final_runs[0]["detail"]["metrics"]
        exact_discovery = (len(detections) == metrics["targets_discovered"] == row["discovered"]
                           and placed-len(detections) == metrics["remaining_search_targets"])
        on_time = sorted({e["task_id"] for e in row["task_reward_events"] if e["credited"] and e["on_time"]})
        if not set(detections) <= placed_ids:
            raise RuntimeError("stable search target IDs differ from frozen first reset contract")
        census = discovery_counts(placed, detections, on_time)
        if len(on_time) != row["on_time"]:
            raise RuntimeError("task reward target ID census differs from row")
        final_targets = {e["target_id"]: e for e in finals if e["target_id"] != "__run__"}
        if set(final_targets) != set(detections):
            raise RuntimeError("final discovered-task census incomplete")
        for tid, event in final_targets.items():
            if event["detail"]["discovered_at"] != detections[tid]:
                raise RuntimeError("discovery timestamp differs from final task snapshot")
        statuses = Counter(e["status"] for e in final_targets.values())
        if dict(statuses) != metrics["status_counts"]:
            raise RuntimeError("raw final statuses differ from original metrics contract")
        ct_events = {e["target"] for e in transcript if e.get("kind") == "event"}
        if ct_events != set(detections):
            raise RuntimeError("search/accepted relay event ID census differs")
        eout = {"source_index_ordinal": ordinal, "logical_job": entry["logical_job"], "group": name,
                "scene": job["scene"], "seed": job.get("seed"), "episode_index": job.get("episode_index"),
                **census, "discovery_evidence": "exact_search_detection_events_and_final_search_census" if exact_discovery else "lower_bound_only",
                "placed_ids_from_frozen_reset_encoding": sorted(placed_ids),
                "undiscovered_ids_from_reset_encoding": sorted(placed_ids-set(detections)),
                "discovered_ids": sorted(detections), "first_discovery_times": detections,
                "on_time_ids": on_time, "discovered_not_on_time_ids": sorted(set(detections)-set(on_time)),
                "discovered_not_on_time_final_snapshots": {tid: {"status":final_targets[tid]["status"],
                                                                "detail":final_targets[tid]["detail"]}
                                                          for tid in sorted(set(detections)-set(on_time))},
                "original_final_status_counts": dict(statuses),
                "original_row_states": {k: row[k] for k in ("completed", "expired", "active", "unassigned")},
                "spent_bytes": row["spent_bytes"], "emissions": row["emissions"],
                "emitted_target_ids": sorted({e["target"] for e in transcript if e.get("kind")=="send"}),
                "refusal_count": len(row["refusal_sequence"]), "communication": None,
                "feedback": {"task_reward_events": len(row["task_reward_events"]),
                             "credited_on_time_events": sum(bool(e["credited"]) for e in row["task_reward_events"]),
                             "nonzero_task_stream_ticks": sum(float(v)!=0 for v in row["task_stream"].values())}}
        discovery_profiles[job["scene"]].add(tuple(sorted(detections.items())))
        g["scene_exposures"][job["scene"]] += 1
        if "tape" in raw:
            tapes = raw["tape"]
            counts, free_targets, ready_values, free_sends = Counter(), [], [], set()
            for tape in tapes:
                classes = partition(tape)
                if tape["schema"] != "round42-one-shot-send-skip-tape/1" or tape["comm_action_semantics"] != "one-shot-send-skip/1":
                    raise RuntimeError("current candidate contract differs")
                matrix = np.asarray(tape["features"], dtype="<f4")
                if matrix.shape != (len(tape["targets"]), 19):
                    raise RuntimeError("feature/target row shape mismatch")
                counts["decision_steps"] += 1
                counts["candidate_rows"] += len(tape["targets"])
                for cls, targets in classes.items():
                    counts[cls] += len(targets)
                    counts["decision_rows_with_"+cls] += bool(targets)
                counts["fixed_refresh_subcount"] += len(tape["fixed_hold_refresh"])
                g["task_values"].append(float(tape["value"]))
                for i, target in enumerate(tape["targets"]):
                    f = matrix[i]
                    first, ready, late = gate(f, int(tape["step"]))
                    audit = tape["candidate_audit"].get(target)
                    if audit and audit.get("first_emission") != first:
                        raise RuntimeError("first-report field differs from recorded legal input")
                    g["all_inputs"].add(f)
                    if target in classes["free"]:
                        if not first or audit["one_shot_state_before"] != "undecided" or not audit["sequentially_affordable_before_action"]:
                            raise RuntimeError("free choice was not a legal undecided first report")
                        action = int(tape["actions"][target])
                        counts["SEND" if action == 1 else "SKIP"] += 1
                        counts["late_free" if late else "early_free"] += 1
                        counts[("late_" if late else "early_") + ("SEND" if action else "SKIP")] += 1
                        if action:
                            free_sends.add(target)
                        g["free_inputs"].add(f)
                        domain = "train" if name.startswith("train/") else "eval"
                        pooled_free[domain].add(f)
                        free_provenance[domain][f.tobytes()].append((job["scene"], target, job.get("seed"), name, job.get("episode_index")))
                        if name.startswith("eval/r58/"):
                            pooled_free["eval_r58"].add(f)
                            free_provenance["eval_r58"][f.tobytes()].append((job["scene"],target,job.get("seed"),name,None))
                        g["probabilities"]["late" if late else "early"].append(float(tape["send_probabilities"][target]))
                        free_targets.append(target)
                        ready_values.append({"target": target, "step": int(tape["step"]), "ready_step": ready,
                                             "first_emission": first, "late": late,
                                             "offset_from_search_discovery": ready-detections[target]})
                for key in ("advantage_task", "task_advantage", "advantages_task"):
                    if key in tape:
                        g["saved_task_advantages"].extend(np.asarray(tape[key]).reshape(-1).tolist())
            if len(free_targets) != len(set(free_targets)):
                raise RuntimeError("one target had repeated free one-shot decisions")
            counts["unique_episode_target_free_decisions"] = len(set(free_targets))
            counts["emitted"] = row["emissions"]
            counts["executor_refusals"] = len(row["refusal_sequence"])
            sends = [e for e in transcript if e.get("kind") == "send"]
            if len(sends) != row["emissions"] or sum(e["bytes_total"] for e in sends) != row["spent_bytes"]:
                raise RuntimeError("actual CT sends/cost differ from original row")
            if counts["forced"] == 0 and counts["executor_refusals"] == 0 and free_sends != {e["target"] for e in sends}:
                raise RuntimeError("free SEND and actual emitted target set differ")
            summary = row["collector_summary"]
            if (not summary["candidate_classes_close"] or not summary["candidate_classes_disjoint"]
                    or summary["one_shot_counts"]["free_choices"] != counts["free"]
                    or summary["one_shot_counts"]["fixed_holds_after_choice"] + summary["one_shot_counts"]["closed_by_prior_real_emission"] != counts["fixed_hold_one_shot_closed"]):
                raise RuntimeError("independent tape partition differs from saved collector summary")
            eout["communication"] = {**dict(counts), "gate_points": ready_values}
            g["counts"].update(counts)
            credited, prefix = credit_task_stream(row["task_stream"], tapes)
            by_row_count, by_row_reward = Counter(), Counter()
            for tape, reward in zip(tapes, credited):
                if reward != 0:
                    tags = [tag for tag in PARTITIONS if tape[tag]]
                    signature = "+".join(tags)
                    by_row_count[signature] += 1
                    by_row_reward[signature] += reward
            if not np.isclose(sum(credited)+prefix, row["task_reward_total"]):
                raise RuntimeError("saved reward stream does not close under original timing")
            eout["feedback"] = {"task_reward_events": len(row["task_reward_events"]),
                                "credited_on_time_events": sum(bool(e["credited"]) for e in row["task_reward_events"]),
                                "nonzero_task_stream_ticks": sum(float(v)!=0 for v in row["task_stream"].values()),
                                "nonzero_bundled_decision_rows": sum(v!=0 for v in credited),
                                "nonzero_bundled_row_classes": dict(by_row_count),
                                "bundled_reward_by_row_classes": dict(by_row_reward), "unassigned_prefix": prefix}
            g["task_credit_rows"].update(by_row_count)
            g["task_credit_reward"].update(by_row_reward)
        else:
            eout["missing_policy_tape"] = "structurally_absent_for_native_rule; candidate/free/masked counts not observable"
            eout["feedback"]["nonzero_bundled_decision_rows"] = None
        g["episodes"].append(eout)
        episode_rows.append(eout)
    summaries = {}
    for name, g in groups.items():
        es = g["episodes"]
        summaries[name] = {"episode_count": len(es), "unique_scene_count": len(g["scene_exposures"]),
                           "scene_exposure_counts": dict(sorted(g["scene_exposures"].items())),
                           "search": {k: sum(e[k] for e in es) for k in ("placed", "discovered", "on_time", "discovered_not_on_time", "undiscovered_at_window_end")},
                           "discovered_per_episode": distribution([e["discovered"] for e in es]),
                           "on_time_per_episode": distribution([e["on_time"] for e in es]),
                           "spent_bytes_per_episode": distribution([e["spent_bytes"] for e in es]),
                           "emissions_per_episode": distribution([e["emissions"] for e in es]),
                           "first_discovery_time": distribution([v for e in es for v in e["first_discovery_times"].values()]),
                           "decision_counts": dict(g["counts"]) if g["all_inputs"].rows else None,
                           "all_candidate_input": g["all_inputs"].result() if g["all_inputs"].rows else None,
                           "free_input": g["free_inputs"].result() if g["free_inputs"].rows else None,
                           "saved_send_probability": {k: distribution(v, histogram=False) for k, v in g["probabilities"].items()},
                           "task_feedback": {"nonzero_bundled_rows_by_classes": dict(g["task_credit_rows"]) if g["all_inputs"].rows else None,
                                             "bundled_reward_by_classes": dict(g["task_credit_reward"]) if g["all_inputs"].rows else None,
                                             "nonzero_bundled_rows": sum(g["task_credit_rows"].values()) if g["all_inputs"].rows else None,
                                             "raw_task_reward_events": sum(e["feedback"]["task_reward_events"] for e in es if e["feedback"]),
                                             "nonzero_stream_ticks": sum(e["feedback"]["nonzero_task_stream_ticks"] for e in es if e["feedback"]),
                                             "unassigned_prefix": sum(e["feedback"]["unassigned_prefix"] for e in es) if g["all_inputs"].rows else None,
                                             "saved_task_advantage": distribution(g["saved_task_advantages"]) if g["saved_task_advantages"] else None,
                                             "saved_task_advantage_missing": "not serialized in indexed tape/batch/minibatch records",
                                             "saved_task_value": distribution(g["task_values"], histogram=False) if g["task_values"] else None}}
    training_by_key = {(e["seed"], e["episode_index"]): e for e in episode_rows if e["group"].startswith("train/")}
    diag = read(R58/"ROUND58_FINAL_DIAGNOSTIC_COVERAGE.json")
    batch_data = []
    for indexed in diag["per_batch"]:
        if sha(indexed["batch_record"]) != indexed["batch_record_sha256"] or sha(indexed["commit"]) != indexed["commit_sha256"]:
            raise RuntimeError("accepted batch source changed")
        batch = read(indexed["batch_record"])
        gaps = [float(p["y_reference"])-float(p["y_policy"]) for p in batch["constraint"]["pairs"]]
        if float(np.mean(gaps)) != batch["constraint"]["g_hat"]:
            raise RuntimeError("original g_hat differs from four paired task gaps")
        for ep, pair, gap in zip(batch["episodes"], batch["constraint"]["pairs"], gaps):
            e = training_by_key[(batch["training_seed"], ep)]
            if e["scene"] != pair["scenario"] or e["on_time"] != pair["y_policy"]:
                raise RuntimeError("batch reference-policy pair differs from original training row")
            e["reference_on_time"] = pair["y_reference"]
            e["paired_task_gap_reference_minus_policy"] = gap
        batch_data.append({"seed": batch["training_seed"], "batch": batch["batch"],
                           "g_hat": batch["constraint"]["g_hat"], "dual": batch["dual"],
                           "samples": batch["update"]["samples"],
                           "batch_record": indexed["batch_record"], "sha256": indexed["batch_record_sha256"]})
    feedback_train = {}
    for seed in (11, 22, 33):
        bs = [b for b in batch_data if b["seed"] == seed]
        gaps = [e["paired_task_gap_reference_minus_policy"] for e in training_by_key.values() if e["seed"] == seed]
        feedback_train[str(seed)] = {"batches": len(bs), "g_hat": distribution([b["g_hat"] for b in bs]),
                                     "g_hat_signs": dict(Counter("positive" if b["g_hat"]>0 else "negative" if b["g_hat"]<0 else "zero" for b in bs)),
                                     "episode_paired_task_gap": distribution(gaps),
                                     "lambda_before": distribution([b["dual"]["lambda_before"] for b in bs]),
                                     "lambda_next": distribution([b["dual"]["lambda_next"] for b in bs])}
    eval_lookup = {(e["group"], e["scene"]): e for e in episode_rows if e["group"].startswith("eval/")}
    comparisons = []
    gate_vs_native_late = []
    for scene in range(1700000,1700032):
        policy = eval_lookup[("eval/r58/s11/greedy",scene)]
        native = eval_lookup[("eval/late/rule",scene)]
        predicted = {point["target"] for point in policy["communication"]["gate_points"] if point["first_emission"] and point["late"]}
        actual = set(native["emitted_target_ids"])
        gate_vs_native_late.append({"scene":scene,"offline_predicted_target_ids":sorted(predicted),
                                    "native_recorded_emitted_target_ids":sorted(actual),"equal":predicted==actual})
    previous_stats = read(R58/"ROUND58_FINAL_STATISTICS.json")
    for seed in (11, 22, 33):
        for mode in ("greedy", "sampled"):
            for arm in ("immediate", "late", "r56"):
                differences = []
                for scene in range(1700000, 1700032):
                    cand = eval_lookup[(f"eval/r58/s{seed}/{mode}", scene)]
                    ref_key = f"eval/{arm}/rule" if arm != "r56" else f"eval/r56/s{seed}/{mode}"
                    ref = eval_lookup[(ref_key, scene)]
                    differences.append({"scene": scene, **id_difference(cand["on_time_ids"], ref["on_time_ids"]),
                                        "cost_delta_bytes": cand["spent_bytes"]-ref["spent_bytes"]})
                frozen = next(x for x in previous_stats["comparisons"] if x["seed"]==seed and x["mode"]==mode and x["control"]==arm)
                if ([x["net_count"] for x in differences] != frozen["differences"]["on_time"]["per_scene_integer"]
                        or [x["cost_delta_bytes"] for x in differences] != frozen["differences"]["spent_bytes"]["per_scene_integer"]):
                    raise RuntimeError("independent integer evaluation comparison conflicts with accepted R58 stats")
                comparisons.append({"seed": seed, "mode": mode, "control": arm,
                                    "task_count_nonzero_scenes": sum(x["net_count"]!=0 for x in differences),
                                    "target_set_changed_scenes": sum(bool(x["lost"] or x["rescued"]) for x in differences),
                                    "cost_nonzero_scenes": sum(x["cost_delta_bytes"]!=0 for x in differences),
                                    "lost_total": sum(len(x["lost"]) for x in differences),
                                    "rescued_total": sum(len(x["rescued"]) for x in differences), "per_scene": differences})
    train_vectors, eval_vectors = set(pooled_free["train"].vectors), set(pooled_free["eval"].vectors)
    training_scenes = {e["scene"] for e in episode_rows if e["group"].startswith("train/")}
    evaluation_scenes = {e["scene"] for e in episode_rows if e["group"].startswith("eval/")}
    columns_train = pooled_free["train"].columns
    outside_counts = [0]*19
    for vector, entries in free_provenance["eval"].items():
        f = np.frombuffer(vector, dtype="<f4")
        for i,v in enumerate(f):
            outside_counts[i] += len(entries) * int(v<min(columns_train[i]) or v>max(columns_train[i]))
    diversity = {"train": pooled_free["train"].result(), "eval_models": pooled_free["eval"].result(),
                 "eval_r58": pooled_free["eval_r58"].result(),
                 "train_eval_exact_vector_intersection": len(train_vectors&eval_vectors),
                 "eval_vectors_absent_from_train": len(eval_vectors-train_vectors),
                 "eval_exposures_exact_vector_present_in_train": sum(len(v) for k,v in free_provenance["eval"].items() if k in train_vectors),
                 "eval_exposures_outside_training_column_range": {NAMES[i]:outside_counts[i] for i in range(19)},
                 "warning": "exact differences and the arbitrary descriptive projection excluding age/budget/clock do not establish meaningful/causal separability"}
    for domain, mapping in free_provenance.items():
        diversity[domain+"_vector_sharing"] = {
            "vectors_shared_by_multiple_scene_target_pairs": sum(len({(e[0],e[1]) for e in entries})>1 for entries in mapping.values()),
            "vectors_shared_by_different_target_ids": sum(len({e[1] for e in entries})>1 for entries in mapping.values()),
            "vectors_shared_by_different_scenes": sum(len({e[0] for e in entries})>1 for entries in mapping.values()),
            "episode_target_exposures": sum(len(entries) for entries in mapping.values()),
            "unique_scene_target_keys": len({(e[0],e[1]) for entries in mapping.values() for e in entries}),
            "top_repeated_vectors": [{"float32_sha256": hashlib.sha256(k).hexdigest(),
                                       "vector": np.frombuffer(k,dtype="<f4").tolist(), "exposures": len(es),
                                       "unique_scene_target_keys": len({(e[0],e[1]) for e in es}),
                                       "example_sources": [list(e) for e in es[:5]]}
                                      for k,es in sorted(mapping.items(),key=lambda x:len(x[1]),reverse=True)[:10]]}
    protected_end = {name: sha(BASE/name) for name in PROTECTED}
    if protected_end != protected_start:
        raise RuntimeError("protected bytes changed during analysis")
    output = {"kind": "round59-readonly-communication-density-audit/1", "basis_head": "80e00de5b48348eff53830072b73ca2a92462f74",
              "analysis_script_sha256": sha(__file__),
              "handover_inspection": {"branch": "analysis/round59-communication-decision-density",
                                      "visible_python_processes_start_and_end": 0,
                                      "process_evidence": "Get-Process python,pythonw; no new scientific executor launched",
                                      "process_metadata_limitation": "Get-CimInstance Win32_Process: 无法从客户端中访问 CIM 资源。"},
              "inputs": inputs, "semantic_sources": source_identity,
              "input_validation": {"indexed_complete_episodes": len(episode_rows), "verified_artifacts": verified_artifacts,
                                   "discovery_profiles_per_scene": {str(k):len(v) for k,v in sorted(discovery_profiles.items())},
                                   "exact_discovery_episodes": sum(e["discovery_evidence"].startswith("exact") for e in episode_rows)},
              "groups": summaries, "episodes": episode_rows, "input_diversity": diversity,
              "data_blocks": {"training_unique_scenes": sorted(training_scenes),
                              "evaluation_unique_scenes": sorted(evaluation_scenes),
                              "scene_overlap": sorted(training_scenes & evaluation_scenes),
                              "independence_note": "128 episodes/seed repeatedly visit 48 scenes; 3 seeds do not create extra scene units"},
              "training_feedback": feedback_train, "batches": batch_data, "evaluation_comparisons": comparisons,
              "offline_gate_vs_native_late_emitted_sets": gate_vs_native_late,
              "missing": ["native rules have no actor tape: rule candidate/free/forced/masked counts unavailable",
                          "per-sample task advantages/returns are not serialized; no PPO/GAE training path replayed",
                          "no controlled interventions identify a marginal causal value for individual messages"],
              "protected_sha256": protected_end,
              "verification": {"test_command": "D:/project_env/python.exe -B -m unittest discover -s analysis/round59-communication-decision-density -p test_r59_density_audit.py -v",
                               "pure_computation_test_cases": 6, "test_exit_code": 0,
                               "result_origin": "actual unittest invocation in this execution; not a simulator suite",
                               "analysis_integrity": "all indexed artifact bytes and descriptive closures verified"},
              "assessments": {
                  "A": {"judgment": "not_supported_as_a_large_search_discovery_shortfall",
                        "limit": "a small observed discovery shortfall exists; nothing here proves search is adequate outside these scene blocks"},
                  "B": {"judgment": "supported_descriptively_repeated_low_variation_free_states",
                        "limit": "repetition and gate coverage do not identify the cause of optimizer performance or message safety"},
                  "C": {"judgment": "supported_descriptively_low_policy_differential_task_feedback",
                        "limit": "unverified_as_unique_causal_explanation; saved per-sample advantages absent and no interventions"}},
              "single_recommendation": {"direction": "study use of existing legal state through one early/late actor-head parameter-sharing ablation",
                                        "purpose": "test whether shared early suppression and late SEND mapping interfere without changing goal density or adding information",
                                        "authorization": "requires separate prospective experiment approval, new registered checkpoints and quota; not executed in ROUND59",
                                        "not_a_claim": "current descriptive audit cannot prove parameter interference or that more goals would solve learning"},
              "operations": {k:0 for k in ("environment_construction", "simulation", "new_stream", "new_claim", "optimizer_step_real",
                                          "optimizer_step_synthetic", "PPO_replay", "RNG_restore", "checkpoint_write", "new_eval_cell",
                                          "CPU_forward", "autograd_diagnostic")}}
    (HERE/"ROUND59_DENSITY_AUDIT.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    write_report(output)
    print(json.dumps({"groups": {k:{"episodes":v["episode_count"],"search":v["search"],"counts":v["decision_counts"],
                                   "free_input":None if v["free_input"] is None else {a:v["free_input"][a] for a in ("rows","exact_float32_vectors","signatures_excluding_age_budget_clock")},
                                   "on_time":v["on_time_per_episode"]["histogram"]} for k,v in summaries.items()},
                      "training_feedback":feedback_train, "diversity":{k:v for k,v in diversity.items() if not k.endswith("vector_sharing") and k not in ("train","eval_models")}},ensure_ascii=False,indent=2))


def write_report(a):
    gs, div = a["groups"], a["input_diversity"]
    train = [gs[f"train/r58/s{s}/sampled"] for s in (11,22,33)]
    immediate = gs["eval/immediate/rule"]
    eval_timing = immediate["first_discovery_time"]["quantiles_linear"]
    train_timing = train[0]["first_discovery_time"]["quantiles_linear"]
    body = ["# ROUND59：搜索发现、通信选择与任务反馈密度只读审计", "",
            "本轮完整读取 ROUND58 已交付的 1024 个有效来源，复用既有资格和模型身份核验；重新核对 4032 件索引产物的字节、路径和收据大小。未使用无效/失败 attempt 作为科学样本。", "",
            "A（大量目标未发现）不受当前数据支持；B（自由决策重复、结构变化小）有描述性支持；C（策略间任务反馈变化少）有描述性支持。B/C 尚不能被裁定为学习结果的唯一因果根因。", "",
            "## 搜索发现：事件证据与完成数分开", "",
            "发现取生命周期日志的真实 search DetectionEvent（target、source_event_id、discovered_by、timestamp），与 final census 的 targets_discovered、remaining_search_targets 及最终 task snapshot.discovered_at 逐项闭合。1024/1024 集均可确定发现数，不以完成数替代。placed 来自真实 run_config.num_goals_placed；未发现 ID 按冻结 PILOC reset 编号契约恢复，并明确标在机器结果中。生命周期 detection 是 registry 接受的搜索发现，不是接收端 belief；本批记录的搜索剩余目标数提供独立分母闭合。", "",
            "|训练 seed|episode / 不同场景|placed / 发现|发现率|按时完成|发现但未按时|窗口末未发现|每集按时分布|",
            "|---|---:|---:|---:|---:|---:|---:|---|"]
    for seed,t in zip((11,22,33),train):
        x=t["search"]
        body.append(f"|{seed}|{t['episode_count']} / {t['unique_scene_count']}|{x['placed']} / {x['discovered']}|{x['discovered']/x['placed']:.6%}|{x['on_time']}|{x['discovered_not_on_time']}|{x['undiscovered_at_window_end']}|{t['on_time_per_episode']['histogram']}|")
    hidden_train = sorted({(e["scene"],tuple(e["undiscovered_ids_from_reset_encoding"])) for e in a["episodes"] if e["group"].startswith("train/") and e["undiscovered_at_window_end"]})
    body += ["",f"48 个训练场景共 288 个 scene-target，实际发现 {div['train_vector_sharing']['unique_scene_target_keys']} 个；未发现的场景/ID 为 `{hidden_train}`。每 seed 在这两个场景共出现 6 次未发现目标；重复 episode 和三个 seed 都不扩大独立场景数。", "",
             f"新 32 场景每一臂/seed/mode 均 placed 192、发现 {immediate['search']['discovered']}、末未发现 {immediate['search']['undiscovered_at_window_end']}（{immediate['search']['discovered']/192:.6%}）；唯一未发现是 scene1700020 / target-4。immediate 的 {immediate['search']['on_time']} 个按时完成此处恰与发现数相等，但这是两套原件分别重建后的结果。不同臂的按时数仍不同；按原状态契约保留 completed/expired/active/unassigned 和 final status，不重新定义 completed_late。", "",
             f"首次发现时刻（原 DetectionEvent 时间，dt=1）：训练按每 seed 的重复 episode 统计 min/25%/median/75%/90%/max = {[train_timing[str(q)] for q in (0,.25,.5,.75,.9,1)]}；新 32 场景去重复臂后 = {[eval_timing[str(q)] for q in (0,.25,.5,.75,.9,1)]}。80 个 scene 的发现 ID/时间 profile 在全部已读重复运行中均一致。没有证据表明“大量未发现”压掉本轮通信机会；这不证明更大目标密度下搜索仍足够。", "",
             "## 策略选择：候选行、自由动作与训练 sample 分栏", "",
             "每 tape 时刻的 targets 被 free/forced/masked/fixed_hold_one_shot_closed 精确、互斥覆盖；free=actionable=free_one_shot_choices。独立计数与当前 collector summary 一致。固定 HOLD 含已作出一次性选择后再次出现的候选，不是预算 mask，也不是新的可学习首报。PPO sample 是一个有若干候选的 decision-step，不是每一个 feature 行。", "",
             "|训练 seed|decision-step / 含自由选择的 step|候选 feature 行|自由首报 / 唯一 episode-target|fixed HOLD 行|forced / masked|SEND / SKIP / 实际发射 / 拒付|",
             "|---|---:|---:|---:|---:|---:|---|"]
    for seed,t in zip((11,22,33),train):
        c=t["decision_counts"]
        body.append(f"|{seed}|{c['decision_steps']} / {c['decision_rows_with_free']}|{c['candidate_rows']}|{c['free']} / {c['unique_episode_target_free_decisions']}|{c['fixed_hold_one_shot_closed']}|{c['forced']} / {c['masked']}|{c.get('SEND',0)} / {c.get('SKIP',0)} / {c['emitted']} / {c['executor_refusals']}|")
    body += ["", "|新块模型/seed/mode|候选行|自由行|early / late|SEND / SKIP|实际发射|按时完成|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name,t in gs.items():
        if not name.startswith("eval/") or t["decision_counts"] is None: continue
        c=t["decision_counts"]
        body.append(f"|{name.removeprefix('eval/')}|{c['candidate_rows']}|{c['free']}|{c['early_free']} / {c['late_free']}|{c.get('SEND',0)} / {c.get('SKIP',0)}|{c['emitted']}|{t['search']['on_time']}|")
    body += ["", "所有当前模型格的 forced/masked/执行器拒付均为 0；不能把 fixed HOLD 的大比例写成旧轮 masked 比例。native immediate/late 没有 actor tape，真实候选/自由动作/forced/masked 数为未记录，机器结果用 null；实际发射分别为 191/45，按时完成分别为 191/190。不得由未发射反推原 SKIP。", "",
             "## 输入覆盖与门控", "",
             "|输入域|自由动作暴露次数|精确不同 float32 向量|不同 scene-target|不含 age/预算/时刻列的描述性签名数|",
             "|---|---:|---:|---:|---:|"]
    for label,key,share in (("训练三 seed（重复暴露）","train","train_vector_sharing"),("新块全部18模型格组（重复暴露）","eval_models","eval_vector_sharing"),("新块R58六组（重复暴露）","eval_r58","eval_r58_vector_sharing")):
        x=div[key]
        body.append(f"|{label}|{x['rows']}|{x['exact_float32_vectors']}|{div[share]['unique_scene_target_keys']}|{x['signatures_excluding_age_budget_clock']}|")
    body += ["", "|训练 seed|精确不同自由向量|新块同 seed R58 greedy / sampled 不同自由向量|",
             "|---|---:|---:|"]
    for seed,t in zip((11,22,33),train):
        body.append(f"|{seed}|{t['free_input']['exact_float32_vectors']}|{gs[f'eval/r58/s{seed}/greedy']['free_input']['exact_float32_vectors']} / {gs[f'eval/r58/s{seed}/sampled']['free_input']['exact_float32_vectors']}|")
    const=[c["index"] for c in div["train"]["columns"] if c["constant"]]
    body += ["",f"训练和新块自由首报都只有 4 个变化列：13 has_future_requirements、14 backlog_targets、15 budget_fraction、18 step/1000；其余 **{len(const)} 列恒定**（列号 {const}）。因此精确向量差别主要来自时间、预算、积压与是否还有后续任务能力需求，并非 19 个均丰富变化的量。自由输入逐维范围/不同值及所有候选输入范围均保存在同一 JSON；固定 HOLD 的更新状态不应冒充新的自由动作状态。排除 [0,15,18] 得到的 12 个签名仅是声明清楚的描述性投影，不是“12 类有意义任务”或可分性证明。", "",
             f"训练与新块的精确向量交集 {div['train_eval_exact_vector_intersection']}；新块全部模型 {div['eval_models']['exact_float32_vectors']} 个向量中 {div['eval_vectors_absent_from_train']} 个训练未见，{div['eval_exposures_exact_vector_present_in_train']}/{div['eval_models']['rows']} 次暴露与训练完全相同。新块 {div['eval_exposures_outside_training_column_range']['budget_fraction']} 次暴露的 budget_fraction 低于训练最小值；其他列均在训练范围内。范围覆盖不是联合分布覆盖，精确未见也不等于有新语义。", "",
             f"同一向量跨不同 scene-target 共享：训练 {div['train_vector_sharing']['vectors_shared_by_multiple_scene_target_pairs']} 个向量、新块全部模型 {div['eval_vector_sharing']['vectors_shared_by_multiple_scene_target_pairs']} 个。top 重复向量、次数及有限来源面板在 JSON。训练场景 48、新块 32、交集 0；重复 seed/模式/episode 不当作独立新场景。", "",
             "恢复仅使用合法 19 维向量与冻结常量：first=(f[5]==1)，step=round(f[18]*1000)，ready=step-round(f[0])，late 区间严格 ready>147。逐点核对 float32 时钟与 tape step、first 与 candidate_audit 一致；这些自由首报实际 age=0，恢复后的 ready 恰等于发现时刻，但此处仍按表达式恢复，没有把 tape step 定义成 ready。每 seed 训练 early/late=589/173；每个新块模型组=146/45。", "",
             f"从合法输入离线重算 late 拟发送集合，与 native late 已记录发射集合 {sum(x['equal'] for x in a['offline_gate_vs_native_late_emitted_sets'])}/32 场景一致。这个标签只描述 frozen gate，不是安全删除真值。scene1700027/target-5 的 ready=134，是已发现的 early 目标；它提供了 gate 简单结构之外的任务取舍空间，不能把所有 early 消息都宣称无用。", "",
             "## 任务反馈：存在、变化与因果价值分开", "",
             "|训练 seed|非零 task 事件|非零入账 decision rows|仅 fixed HOLD 行奖励 / 含 free 的混合行奖励|g_hat零/正/负批|",
             "|---|---:|---:|---:|---:|"]
    for seed,t in zip((11,22,33),train):
        fb=t["task_feedback"]; signs=a["training_feedback"][str(seed)]["g_hat_signs"]
        body.append(f"|{seed}|{fb['raw_task_reward_events']}|{fb['nonzero_bundled_rows']}|{fb['bundled_reward_by_classes'].get('fixed_hold_one_shot_closed',0)} / {fb['bundled_reward_by_classes'].get('free+fixed_hold_one_shot_closed',0)}|{signs.get('zero',0)} / {signs.get('positive',0)} / {signs.get('negative',0)}|")
    body += ["", "按原 reward timing 用 bisect 将保存的 task_stream 归入最近已采样的 decision step，奖励总数与 row 闭合；没有调用 batch_samples/GAE/ppo_update，没有重放 PPO。forced/masked 非零入账行为 0，因为本批相应候选不存在，不是缺日志。free 与 fixed 可能同处一个 tape step，所以混合行单列，不把奖励复制给每个候选目标。", "",
             "每集均有 task reward；seed11/22 128 个配对 episode gap 全为 0，seed33 仅 episode89/scene1300019/target-5 比 immediate 参考少 1 个按时任务；因此 96 批仅 seed33 batch22 的 g_hat=+0.25，其余 95 个为 0（方向 reference−policy）。相应 lambda 从 1 增至 1.125，原记录与行逐项一致。任务数 5/6 的多数变化来自同一组未发现场景，而不是不同发送行为造成的广泛结局变化。", "",
             "大部分当期 task reward 入账到 fixed HOLD 行，但全轨迹 GAE 可把后续奖励传回更早的自由选择；此审计不能据入账位置声称 actor 没有 task 梯度。保存的 task value 可描述范围、非零变化，per-sample task advantage/return 未在索引 tape/batch/minibatch 数据中保存，登记缺失。任务 reward 存在、策略间反馈变化少、单条通信的边际因果价值，是三件不同的主张；第三件没有干预证据。已计量成本/发射数仍随策略和场景变化，其每集分布保存在 JSON，反馈稳定不等于没有成本学习信号。", "",
             "|R58 seed/mode|相对 immediate：任务差集 / 成本非零场景|相对 late：任务差集 / 成本非零场景|相对 R56：丢失/救回目标 / 成本非零场景|",
             "|---|---:|---:|---:|"]
    for seed in (11,22,33):
        for mode in ("greedy","sampled"):
            cs={c["control"]:c for c in a["evaluation_comparisons"] if c["seed"]==seed and c["mode"]==mode}
            body.append(f"|{seed}/{mode}|{cs['immediate']['target_set_changed_scenes']} / {cs['immediate']['cost_nonzero_scenes']}|{cs['late']['target_set_changed_scenes']} / {cs['late']['cost_nonzero_scenes']}|{cs['r56']['lost_total']}/{cs['r56']['rescued_total']} / {cs['r56']['cost_nonzero_scenes']}|")
    body += ["", "本轮所有 R58 vs immediate/late/R56 的目标集合变化仅位于 scene1700027 / piloc-e1-target-5：s11 两模式相对 immediate/R56 丢失，相对 late 相同；s22 两模式和 s33 greedy 相对 late/R56 救回；s33 sampled 相对 R56 相同、相对 late 救回。费用差可出现在没有任务差的场景，不能将费用差与任务价值混算。未新增统计方法或改动 R58 的冻结区间。净任务数相同仍可能交换目标，纯计算反例已覆盖；实际这组比较没有净零目标交换。", "",
             "## 判断与唯一下一步", "",
             "当前更像是 **机会维度低、强规则覆盖简单时钟结构，加上策略间任务反馈变化少**；这是一组并存的描述，不是唯一因果归因。发现不足只占很小、可明确定位的部分。现有材料无法区分参数共享干扰、优化信号抵消和单条消息的任务收益，也不证明增加目标数一定改善学习。", "",
             "唯一建议：先研究已有状态的利用，预先登记一个 **按合法恢复 early/late 分区拆分 actor SEND/SKIP 头的参数共享单因素消融**，其余目标数、世界、任务反馈与训练配置保持既定基线。它能检验 early 抑制与 late 发送映射的共享参数是否互相影响；不能替代单条消息因果价值证明。相比先提高任务密度，它避免同时改变搜索压力与通信机会分布。本建议需要负责人另行批准有限训练/场景/额度与新 checkpoint 契约，本轮没有实现或执行，也没有提出“至少多少目标才够”的新门槛。", "",
             "## 验证、交付与停止", "",
             "6 项纯计算测试通过，exit 0：发现≠完成；候选≠自由行≠不同目标；任务数相等但 ID 可交换；重复场景/seed 非独立样本；ready 恢复及严格边界；奖励可入账到非自由时刻。分析脚本 exit 0，4032 件输入实际字节及原 R58 效果整数闭合。规则缺 actor tape 和未保存 task advantage 均用 missing/null，不填成 0。", "",
             "本轮环境/仿真/新流/claim/真实或合成 optimizer.step/PPO 重放/RNG 恢复/checkpoint 写入/新评估格/CPU forward/autograd 全部 0。脚本仅导入标准库和 NumPy，没有导入执行器、训练器或模拟器；概率使用已保存且已资格核验的记录。报告数字来自 ROUND59_DENSITY_AUDIT.json；旧报告、原件和 verdict 保持字节。本地提交仅本轮 4 个文件，不包含运行输出 log。", "",
             "接手和结束的 Get-Process python,pythonw 均无可见 Python 进程；未启动科学执行器。完整命令行进程查询 Get-CimInstance Win32_Process 受限，原错误为『无法从客户端中访问 CIM 资源。』，因此不声称完成不可见进程的全量审计。", "",
             "接手基点 80e00de5b48348eff53830072b73ca2a92462f74；分析分支 analysis/round59-communication-decision-density。保护文件最终 SHA256："]
    for name,value in a["protected_sha256"].items(): body.append(f"- {name}: `{value}`")
    body += ["", "完成后停止；下一实验仍待独立批准。"]
    (HERE/"ROUND59_REPORT.md").write_text("\n".join(body)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
