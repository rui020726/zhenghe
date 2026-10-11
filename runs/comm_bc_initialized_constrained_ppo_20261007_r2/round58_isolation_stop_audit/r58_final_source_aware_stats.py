"""Read-only, source-aware ROUND58 final matrix and frozen paired statistics.

No archived statistics function is imported. All rows are selected by the
single complete claim for each frozen logical key, not by a glob or a version
directory. Every recorded receipt artifact is checked against its raw bytes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = REPO / "runs/comm_bc_initialized_constrained_ppo_20261007_r2"
RAW = BASE / "round58_gen3/raw"
SCENES = tuple(range(1_700_000, 1_700_032))
SEEDS = (11, 22, 33)
MODES = ("greedy", "sampled")
ARMS = ("r58", "r53", "r56", "immediate", "late")


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt_check(claim: dict[str, Any], row: dict[str, Any], *, rule: bool) -> dict[str, Any]:
    key = "round58_rule_artifact_receipt" if rule else "round45_artifact_receipt"
    receipt = claim.get(key)
    expected_kind = ("round58-native-rule-three-artifact-receipt/1" if rule
                     else "round45-r2-artifact-receipt/2")
    expected_names = ({"row", "lifecycle", "transcript"} if rule
                      else {"row", "lifecycle", "transcript", "tape"})
    if not isinstance(receipt, dict) or receipt.get("kind") != expected_kind:
        raise RuntimeError(f"{claim['logical_job']}: missing qualified {key}")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected_names:
        raise RuntimeError(f"{claim['logical_job']}: wrong artifact set")
    for name, item in artifacts.items():
        path = Path(item["path"])
        if (not path.is_file() or str(path.resolve()) != str(Path(row["artefacts"][name]).resolve())
                or sha(path) != item["sha256"] or path.stat().st_size != int(item["size_bytes"])
                or Path(claim["round45_artifact_paths"][name]).resolve() != path.resolve()):
            raise RuntimeError(f"{claim['logical_job']}: {name} raw artifact identity mismatch")
    if rule:
        if (claim.get("round45_artifact_receipt") is not None
                or row.get("collector_summary") is not None
                or Path(row["artefacts"]["tape"]).exists()
                or receipt.get("actor_tape") != "structurally_absent_for_native_rule"):
            raise RuntimeError("native rule incorrectly claims a learned actor tape")
    return {name: item["sha256"] for name, item in artifacts.items()}


def metrics(row: dict[str, Any]) -> dict[str, Any]:
    ids = sorted({str(event["task_id"]) for event in row.get("task_reward_events", [])
                  if event.get("on_time") and event.get("credited")})
    spent = row.get("spent_bytes")
    if (type(spent) is not int or spent < 0
            or spent != int(row.get("airtime_tap_total", -1))
            or int(row.get("on_time", -1)) != len(ids)
            or int(row.get("emissions", -1)) != len(row.get("emission_sequence", []))):
        raise RuntimeError("row cost, credited-target or emission integer closure failed")
    firsts = sum(1 for emission in row["emission_sequence"]
                 if len(emission) >= 6 and emission[5] is True)
    return {"on_time": len(ids), "spent_bytes": spent,
            "emissions": len(row["emission_sequence"]), "first_reports": firsts,
            "on_time_task_ids": ids}


def paired_bootstrap(values: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    if values.shape != (32,) or indices.shape != (10_000, 32):
        raise ValueError("frozen 32-scene paired bootstrap unit/shape changed")
    resampled_means = values[indices].mean(axis=1)
    lo, hi = np.quantile(resampled_means, [0.025, 0.975], method="linear")
    return {"mean": float(values.mean()), "ci95_linear": [float(lo), float(hi)],
            "strictly_positive": bool(lo > 0), "strictly_negative": bool(hi < 0)}


def main() -> None:
    plan = read(HERE / "ROUND58_GEN3_FIXED_PLAN.json")
    jobs = {job["logical_job"]: job for job in plan["jobs"]}
    if len(jobs) != 1024 or set(plan["eval_scenes_new"]) != set(SCENES):
        raise RuntimeError("frozen 1024-key/32-scene plan identity changed")
    source_map = read(HERE / "ROUND58_GEN3_APPROVED_RESUME_SOURCES_V35.json")
    candidate_inputs = read(HERE / "ROUND58_CANDIDATE_INPUT_IDENTITY.json")
    expected_ckpt = {
        ("r58", int(seed)): info["checkpoint_sha256"]
        for seed, info in source_map["completed_training_sources"].items()}
    for seed in SEEDS:
        for arm in ("r53", "r56"):
            expected_ckpt[(arm, seed)] = candidate_inputs["models"][str(seed)][
                f"{arm.upper()}_ep128"]["actual_sha256"]
    claim_paths = sorted((RAW / "launch_claims").glob("*/*.json"))
    all_claims: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path in claim_paths:
        claim = read(path)
        if claim.get("logical_job") not in jobs:
            raise RuntimeError(f"unplanned raw claim {path}")
        all_claims.setdefault(claim["logical_job"], []).append((path, claim))
    by_arm: dict[tuple[str, int, str], dict[int, dict[str, Any]]] = {}
    sources = []
    train_counts = {str(seed): 0 for seed in SEEDS}
    for logical, job in jobs.items():
        complete = [(path, claim) for path, claim in all_claims.get(logical, [])
                    if claim.get("status") == "complete"]
        if len(complete) != 1:
            raise RuntimeError(f"{logical}: expected exactly one complete claim, got {len(complete)}")
        claim_path, claim = complete[0]
        row_path = Path(claim["round45_artifact_paths"]["row"])
        if not row_path.is_file() or BASE.resolve() not in row_path.resolve().parents:
            raise RuntimeError(f"{logical}: completed row is missing/outside R58 source graph")
        row = read(row_path)
        if (int(row.get("scenario_seed", -1)) != int(job["scene"])
                or int(row.get("attempt", -1)) != int(claim["attempt"])
                or row.get("training_version") != claim.get("training_version")
                or int(row.get("steps", -1)) != 1000):
            raise RuntimeError(f"{logical}: row/claim scenario/version/step mismatch")
        scope = job["scope"]
        rule = scope.startswith("rule_eval_")
        artifact_hashes = receipt_check(claim, row, rule=rule)
        met = metrics(row)
        if scope == "ppo_training":
            seed = int(job["seed"])
            if int(row.get("training_seed", -1)) != seed or int(row.get("episode_index", -1)) != int(job["episode_index"]):
                raise RuntimeError("training row seed/cursor differs from frozen job")
            train_counts[str(seed)] += 1
        else:
            if rule:
                arm, seed, mode = job["rule_arm"], 0, "rule"
                if row.get("model_arm") != arm.upper() + "_RULE":
                    raise RuntimeError("rule row arm differs from frozen job")
            else:
                arm = scope.removeprefix("model_eval_")
                seed, mode = int(job["seed"]), str(job["mode"])
                expected_label = "NEW" if arm == "r58" else arm.upper()
                if (row.get("model_arm") != expected_label
                        or int(row.get("training_seed", -1)) != seed
                        or row.get("eval_mode") != mode):
                    raise RuntimeError("model row arm/seed/mode differs from frozen job")
                ident = row.get("checkpoint_load_identity") or row.get("checkpoint_identity") or {}
                recorded_ckpt = ident.get("checkpoint_sha256") or ident.get("file_sha256")
                if recorded_ckpt != expected_ckpt[(arm, seed)]:
                    raise RuntimeError(f"{logical}: fixed ep128 checkpoint byte identity differs")
            key = (arm, seed, mode)
            scene = int(job["scene"])
            if scene in by_arm.setdefault(key, {}):
                raise RuntimeError(f"duplicate scene {key} {scene}")
            by_arm[key][scene] = met
        sources.append({"logical_job": logical, "claim": str(claim_path),
                        "claim_sha256": sha(claim_path), "row": str(row_path),
                        "row_sha256": artifact_hashes["row"],
                        "artifact_sha256": artifact_hashes,
                        "source_root": str(row_path.parent.parent),
                        "attempt": int(claim["attempt"]), "category": claim["category"],
                        "receipt_kind": ("rule_three_real_artifacts" if rule else "R45_four_artifacts")})
    if train_counts != {str(seed): 128 for seed in SEEDS}:
        raise RuntimeError(f"training source graph has wrong seed counts: {train_counts}")
    expected_keys = {(arm, seed, mode) for arm in ("r58", "r53", "r56")
                     for seed in SEEDS for mode in MODES}
    expected_keys |= {(arm, 0, "rule") for arm in ("immediate", "late")}
    if set(by_arm) != expected_keys or any(set(rows) != set(SCENES) for rows in by_arm.values()):
        raise RuntimeError("frozen 576 model + 64 rule matrix is incomplete")
    rng = np.random.Generator(np.random.PCG64(20261008))
    indices = rng.integers(0, 32, size=(10_000, 32))
    comparisons = []
    absolute = []
    for (arm, seed, mode), rows in sorted(by_arm.items()):
        absolute.append({"arm": arm, "seed": seed, "mode": mode,
                         "on_time_total": sum(rows[s]["on_time"] for s in SCENES),
                         "spent_bytes_total": sum(rows[s]["spent_bytes"] for s in SCENES),
                         "spent_bytes_mean": sum(rows[s]["spent_bytes"] for s in SCENES) / 32,
                         "first_reports_total": sum(rows[s]["first_reports"] for s in SCENES),
                         "emissions_total": sum(rows[s]["emissions"] for s in SCENES),
                         "per_scene": [{"scene": s, **rows[s]} for s in SCENES]})
    for seed in SEEDS:
        for mode in MODES:
            candidate = by_arm[("r58", seed, mode)]
            for control in ("r56", "r53", "immediate", "late"):
                reference = by_arm[(control, seed, mode)] if control in ("r56", "r53") else by_arm[(control, 0, "rule")]
                deltas = {metric: np.array([candidate[s][metric] - reference[s][metric]
                                            for s in SCENES], dtype=np.int64)
                          for metric in ("spent_bytes", "on_time", "first_reports", "emissions")}
                changes = []
                for scene in SCENES:
                    cand_ids = set(candidate[scene]["on_time_task_ids"])
                    ref_ids = set(reference[scene]["on_time_task_ids"])
                    changes.append({"scene": scene, "lost": sorted(ref_ids - cand_ids),
                                    "rescued": sorted(cand_ids - ref_ids),
                                    "identical": cand_ids == ref_ids})
                comparisons.append({"seed": seed, "mode": mode, "candidate": "r58",
                                    "control": control, "direction": "r58-minus-control",
                                    "differences": {metric: {**paired_bootstrap(vals, indices),
                                                             "per_scene_integer": vals.tolist()}
                                                    for metric, vals in deltas.items()},
                                    "target_id_changes": changes,
                                    "identical_on_time_id_scenes": sum(change["identical"] for change in changes)})
    output = {"kind": "round58-source-aware-final-statistics/1",
              "status": "complete_development_block_not_deployment",
              "scenes": list(SCENES), "training_counts": train_counts,
              "source_root_counts": {root: sum(src["source_root"] == root for src in sources)
                                     for root in sorted({src["source_root"] for src in sources})},
              "resampling": {"rng": "NumPy Generator PCG64", "seed": 20261008,
                             "draws": 10000, "scene_unit": 32,
                             "index_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                             "quantile": "numpy linear 0.025/0.975"},
              "absolute": absolute, "comparisons": comparisons,
              "not_established": ["formal_noninferiority", "overall_zero_task_loss", "deployment_success"]}
    (HERE / "ROUND58_FINAL_SOURCE_INDEX.json").write_text(
        json.dumps({"kind": "round58-final-source-index/1", "sources": sources}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    (HERE / "ROUND58_FINAL_STATISTICS.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"training": train_counts, "source_roots": output["source_root_counts"],
                      "absolute": [{k: v for k, v in row.items() if k != "per_scene"} for row in absolute],
                      "primary": [row for row in comparisons if row["control"] == "r56"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
