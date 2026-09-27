"""Stage C 启动前冻结：配置/模型 manifest、scene 注册表、seed 碰撞审计、启动台账。

只读仓库与磁盘，不启动任何环境。冻结条件（任一不满足即退出非 0）：
- 生产三棵树相对 Stage C 分支起点 6ebefbe 零 diff；
- 启动台账为 0；
- seed 碰撞审计通过（两个新区块与全项目已claim区间/既有 artefact 无交集）；
- 合成测试已另行运行通过（本脚本不跑测试，只登记套件名）。

    /d/project_env/python.exe runs/comm_cost_stage_c_20260927/sc_freeze.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import sc_common as common

PRODUCTION_TREES = ("integrated_sar", "task_allocation", "piloc")
BRANCH_START = "6ebefbe4e7a81182281807333d64d60cadf03198"

#: The ranges this project has already claimed or sealed, as registered in the rounds' own
#: modules and manifests. The audit is run against this table and against the artefact scan.
CLAIMED_RANGES = (
    ("sealed_test_set", 800000, 800399),
    ("learned_model_reported_stream", 900000, 900199),
    ("systemloop_20260918", 31000, 31059),
    ("task_pressure_g6", 1000000, 1000015),
    ("task_pressure_g8", 1000100, 1000115),
    ("task_pressure_g10", 1000200, 1000215),
    ("task_pressure_g12", 1000300, 1000315),
    ("D1_20260923", 1100000, 1100059),
    ("search_trace_and_policy_20260925", 1200000, 1200059),
)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=common.ROOT, capture_output=True, text=True,
                          encoding="utf-8").stdout.strip()


def audit_collisions() -> dict[str, Any]:
    """The two new blocks against every claimed range and every artefact on disk."""
    findings: list[dict[str, Any]] = []
    blocks = {"training": common.TRAINING_SCENARIOS, "development": common.DEVELOPMENT_SCENARIOS}
    for label, block in blocks.items():
        low, high = int(block[0]), int(block[-1])
        for name, claimed_low, claimed_high in CLAIMED_RANGES:
            if not (high < claimed_low or low > claimed_high):
                findings.append({"block": label, "claimed": name,
                                 "block_span": [low, high], "claimed_span": [claimed_low,
                                                                            claimed_high]})
    # Any file whose name embeds a seed of either block, or any JSON artefact carrying one.
    # This round's own directory is excluded: the audit asks whether *other* rounds already
    # claimed these scenarios, not whether this round has started writing its own artefacts.
    pattern = re.compile(r"13(?:00(?:0[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-7])|10(?:0[0-9]|1[0-5]))")
    embedded = []
    for root, _dirs, files in __import__("os").walk(common.ROOT / "runs"):
        if Path(root) == common.HERE or common.HERE in Path(root).parents:
            continue
        for name in files:
            if pattern.search(name):
                embedded.append(str(Path(root, name).relative_to(common.ROOT)))
    seeded = []
    for path in sorted((common.ROOT / "runs").glob("**/*.json")):
        if common.HERE in path.parents or path.parent == common.HERE:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if re.search(r'"seed"\s*:\s*1300(?:0[0-4][0-9]|1[0-5])\b', text) or re.search(
                r'"seed"\s*:\s*13010(?:0[0-9]|1[0-5])\b', text):
            seeded.append(str(path.relative_to(common.ROOT)))
    return {"claimed_ranges": [{"name": name, "low": low, "high": high}
                               for name, low, high in CLAIMED_RANGES],
            "new_blocks": {"training": [int(common.TRAINING_SCENARIOS[0]),
                                        int(common.TRAINING_SCENARIOS[-1])],
                           "development": [int(common.DEVELOPMENT_SCENARIOS[0]),
                                           int(common.DEVELOPMENT_SCENARIOS[-1])]},
            "range_overlaps": findings,
            "artefacts_embedding_new_seeds": embedded,
            "artefacts_with_new_seeds": seeded,
            "note": "the round-19 staged-approval documents proposed placeholder blocks "
                    "1300000..1300001/1300100..1300219 conditionally ('subject to the collision "
                    "audit'); they were never executed by any run, and ROUND22 §4 registers the "
                    "two blocks above. Registered here rather than silently ignored.",
            "clean": not findings and not embedded and not seeded}


def behaviour_fields() -> dict[str, Any]:
    """The resolved behaviour configuration every arm shares (identity fields excluded)."""
    fields, _substituted = common.stage_b_inherited_config(int(common.TRAINING_SCENARIOS[0]))
    identity = {"output", "comm_transcript", "info_probe", "search_probe", "decision_trace",
                "record_stream", "replay_stream", "seed", "comm_policy", "comm_policy_params"}
    return {key: value for key, value in sorted(fields.items()) if key not in identity}


def configuration_hash(fields: dict[str, Any]) -> str:
    import hashlib

    payload = {"setting": common.STAGE_B_INHERITED_OVERRIDES,
               "inherited_behaviour": fields,
               "hyperparameters": {
                   "clip": common.CLIP, "value_coefficient": common.VALUE_COEFFICIENT,
                   "entropy_bonus": common.ENTROPY_BONUS, "learning_rate": common.LEARNING_RATE,
                   "max_grad_norm": common.MAX_GRAD_NORM, "epochs_per_batch": common.EPOCHS_PER_BATCH,
                   "minibatch_steps": common.MINIBATCH_STEPS,
                   "episodes_per_update": common.EPISODES_PER_UPDATE,
                   "gamma": common.GAMMA, "gae_lambda": common.GAE_LAMBDA,
                   "b0_bytes": common.B0_BYTES, "delta": common.DELTA,
                   "lambda_init": common.LAMBDA_INIT, "eta_lambda": common.ETA_LAMBDA,
                   "data_order_seed": common.DATA_ORDER_SEED},
               "train_seeds": list(common.TRAIN_SEEDS),
               "episodes_per_seed": common.EPISODES_PER_SEED}
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    diff = git("diff", "--name-only", BRANCH_START, "--", *PRODUCTION_TREES)
    changed = [line for line in diff.splitlines() if line]
    audit = audit_collisions()
    if changed:
        raise SystemExit(f"production trees changed since the branch start: {changed}")
    report = common.ledger_report()
    failures_only = bool(report["results"]) and all(not entry["ok"] for entry in report["results"])
    if report["used"] and not failures_only:
        raise SystemExit(f"launch ledger already holds successful starts: {report['used']}")
    if not audit["clean"]:
        raise SystemExit(f"seed collision audit failed: {json.dumps(audit, ensure_ascii=False)}")

    identity = common.code_identity()
    fields = behaviour_fields()
    manifest = {
        "round": "22", "phase": "Stage C learning pilot freeze",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head_at_freeze": git("rev-parse", "HEAD"),
        "branch_start": BRANCH_START,
        "spec": {"file": "ROUND22_COMM_COST_LEARNING_PILOT.md",
                 "source_commit": "f73ebdf8cc3ab60a6a866fdbe0e299b122245cf7",
                 "sha256": common.sha256_file(common.ROOT / "ROUND22_COMM_COST_LEARNING_PILOT.md")},
        "production_diff": {"branch_start": BRANCH_START, "trees": list(PRODUCTION_TREES),
                            "changed_files": changed, "clean": True},
        "code_identity": identity,
        "configuration_hash": configuration_hash(fields),
        "inherited_behaviour_fields": fields,
        "wiring_constants": {
            "b0_bytes": common.B0_BYTES, "delta": common.DELTA,
            "lambda_init": common.LAMBDA_INIT, "eta_lambda": common.ETA_LAMBDA,
            "gamma": common.GAMMA, "gae_lambda": common.GAE_LAMBDA,
            "data_order_seed": common.DATA_ORDER_SEED,
            "log_prob_tolerance": common.LOG_PROB_TOLERANCE,
            "note": "delta=0 is this round's experimental training target, not a deployment "
                    "zero-degradation guarantee (spec §2)"},
        "ppo_hyperparameters": {
            "clip": common.CLIP, "value_coefficient_per_head": common.VALUE_COEFFICIENT,
            "entropy_bonus": common.ENTROPY_BONUS, "learning_rate": common.LEARNING_RATE,
            "max_grad_norm": common.MAX_GRAD_NORM, "epochs_per_batch": common.EPOCHS_PER_BATCH,
            "minibatch_steps": common.MINIBATCH_STEPS,
            "episodes_per_update": common.EPISODES_PER_UPDATE,
            "provenance": "reused from the pilot round (runs/comm_budget_pilot_20260919/"
                          "train_pilot.py); every value is stated rather than inherited by "
                          "silence (spec §2)",
            "registered_differences_from_old_trainer": [
                "no advantage standardisation (old flatten divided by std per episode)",
                "two critics (task + cost) with VALUE_COEFFICIENT each",
                "gamma=1 (old 0.99), gae_lambda=0.95",
                "entropy remains an extra regulariser at the pilot's coefficient; its effect on "
                "the pure cost objective is registered, not hidden"]},
        "scene_registry": {
            "training_block": [int(common.TRAINING_SCENARIOS[0]),
                               int(common.TRAINING_SCENARIOS[-1])],
            "development_block": [int(common.DEVELOPMENT_SCENARIOS[0]),
                                  int(common.DEVELOPMENT_SCENARIOS[-1])],
            "training_scenarios": list(common.TRAINING_SCENARIOS),
            "development_scenarios": list(common.DEVELOPMENT_SCENARIOS),
            "collision_audit": audit},
        "rule_arms": {"reference": common.REFERENCE_ARM,
                      "rules": common.RULE_ARMS,
                      "send_once_alias": common.SEND_ONCE_ALIAS_NOTE},
        "launch_plan": {
            "normal_launches": common.NORMAL_LAUNCHES, "reserve_launches": common.RESERVE_LAUNCHES,
            "hard_cap": common.LAUNCH_CAP,
            "breakdown": {"training": 3 * common.EPISODES_PER_SEED,
                          "immediate_reference": 48 + 16,
                          "learning_dev_eval": 3 * 3 * 16,
                          "rule_dev_eval": 5 * 16},
            "execution_order": ["64 immediate references (48 training + 16 development scenarios)",
                                "seed 11 first 4 training episodes + correctness gate (counted)",
                                "seed 11 remaining 124 episodes",
                                "seed 22 128 episodes",
                                "seed 33 128 episodes",
                                "learning-model dev evaluation at 0/64/128 episodes x 16 scenes",
                                "five rule arms x 16 dev scenes"],
            "eval_timepoints": ["init", "64", "128"]},
        "synthetic_tests": {"suites": ["runs/comm_cost_stage_c_20260927/test_sc_math.py"],
                            "launches": 0},
    }
    manifest["pre_launch_failures"] = {
        "count": len(report["results"]),
        "entries": report["results"],
        "note": "technical failures spent before any successful launch; the fix that followed is "
                "registered as this freeze's new code identity, and the failures stay in the "
                "ledger rather than being deducted"}
    common.write_json("CONFIG_AND_MODEL_MANIFEST.json", manifest)
    common.write_json("SCENE_REGISTRY.json", {
        "training_block": manifest["scene_registry"]["training_block"],
        "development_block": manifest["scene_registry"]["development_block"],
        "training_scenarios": manifest["scene_registry"]["training_scenarios"],
        "development_scenarios": manifest["scene_registry"]["development_scenarios"],
        "collision_audit": audit})
    common.write_json("START_LEDGER.json", {
        "plan": manifest["launch_plan"], "live": {"used": common.ledger_total()}})
    print(json.dumps({"head": manifest["head_at_freeze"],
                      "production_clean": True, "collision_clean": audit["clean"],
                      "configuration_hash": manifest["configuration_hash"],
                      "pre_launch_failures": len(report["results"]),
                      "normal": common.NORMAL_LAUNCHES, "cap": common.LAUNCH_CAP},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
