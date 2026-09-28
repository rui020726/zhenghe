"""ROUND24 v2 重跑启动前冻结：配置/模型 manifest、scene 注册表、seed 碰撞审计、启动台账。

只读仓库与磁盘，不启动任何环境。冻结条件（任一不满足即退出非 0）：
- 起点等价证据存在且通过（本地修复 tip 与远端修复头 64cddcf 内容一致，见 START_EQUIVALENCE.json）；
- 生产三棵树相对已核验修复头零 diff；
- 启动台账为 0、无任何已持久化的本轮 v2 训练 checkpoint（含里程碑）；
- 场景块碰撞审计通过（与旧 Stage C 同名块登记为 intentional_development_reuse）；
- 合成门禁结果已落盘（SYNTHETIC_GATE.json，实际计数而非历史字符串）；
- 与旧 Stage C 机读设置的逐项 diff 只剩已登记差异（输出隔离、harness 命名、评估安排）。

    /d/project_env/python.exe runs/comm_cost_v2_rerun_20260928/v2_freeze.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import v2_common as common

PRODUCTION_TREES = ("integrated_sar", "task_allocation", "piloc")
#: 本地修复 tip（树等价于远端修复头 64cddcffc6f8796f132df8b8e003f005d0bdd749）。
BRANCH_START = "b7217d18b4e7f4e301d0b62a810892967adc6566"
REMOTE_START_HEAD = "64cddcffc6f8796f132df8b8e003f005d0bdd749"
#: 同一场景块的上一轮使用者（ROUND24 §2.6：登记为迭代开发复用，不宣称全新未见）。
PRIOR_BLOCK_USER_DIR = common.STAGE_C_DIR

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


def start_equivalence() -> dict[str, Any]:
    """The verified start point: local fix tip versus the remote fix head (ROUND24 §0).

    The remote head was verified through the authorised GitHub channel by commit SHA; the
    per-entry blob/tree SHAs are recorded in START_EQUIVALENCE.json (git was unreachable this
    session). This function re-verifies every recorded SHA against the local tree and refuses
    to freeze without that evidence.
    """
    path = common.OUTPUT_DIR / "START_EQUIVALENCE.json"
    if not path.exists():
        raise RuntimeError("START_EQUIVALENCE.json is missing; the start point is unverified")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("remote_head") != REMOTE_START_HEAD:
        raise RuntimeError("start-equivalence evidence names a different remote head")
    if not payload.get("verified"):
        raise RuntimeError("start-equivalence evidence is not marked verified")
    local_head = git("rev-parse", "HEAD")
    local_tree = git("rev-parse", f"{BRANCH_START}^{{tree}}")
    if payload.get("local_head") != BRANCH_START or payload.get("local_tree") != local_tree:
        raise RuntimeError("start-equivalence evidence does not match this repository state")
    entries = payload.get("remote_entries") or {}
    if not entries:
        raise RuntimeError("start-equivalence evidence carries no entries")
    declared = {name: value["sha"] if isinstance(value, dict) else value
                for name, value in entries.items()}
    actual = {}
    for line in git("ls-tree", BRANCH_START).splitlines():
        meta, name = line.split("\t", 1)
        actual[name] = meta.split()[2]
    if declared != actual:
        raise RuntimeError("remote head entries and the local tree differ; stopping before freeze")
    return {"remote_head": REMOTE_START_HEAD, "local_head": BRANCH_START,
            "local_tree": local_tree, "head_at_freeze": local_head,
            "entries_compared": len(declared),
            "note": payload.get("note")}


def audit_collisions() -> dict[str, Any]:
    """The two reused blocks against every claimed range and every artefact on disk.

    ROUND24 §2.6: the blocks were registered and used by ROUND22/Stage C. They are registered
    here as intentional_development_reuse -- the prior round's own directory is excluded from
    the scan (and counted as evidence of that reuse), but no other round's artefacts may carry
    these seeds, and the reuse is never relabelled as a brand-new unseen block.
    """
    findings: list[dict[str, Any]] = []
    blocks = {"training": common.TRAINING_SCENARIOS, "development": common.DEVELOPMENT_SCENARIOS}
    for label, block in blocks.items():
        low, high = int(block[0]), int(block[-1])
        for name, claimed_low, claimed_high in CLAIMED_RANGES:
            if not (high < claimed_low or low > claimed_high):
                findings.append({"block": label, "claimed": name,
                                 "block_span": [low, high], "claimed_span": [claimed_low,
                                                                            claimed_high]})
    pattern = re.compile(r"13(?:00(?:0[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-7])|10(?:0[0-9]|1[0-5]))")
    embedded = []
    prior_round_artefacts = 0
    for root, _dirs, files in __import__("os").walk(common.ROOT / "runs"):
        here = Path(root)
        if (here == common.HERE or common.HERE in here.parents
                or here == common.OUTPUT_DIR or common.OUTPUT_DIR in here.parents):
            continue
        if here == PRIOR_BLOCK_USER_DIR or PRIOR_BLOCK_USER_DIR in here.parents:
            # Registered intentional reuse: the previous round's own outputs under the same
            # registered blocks; counted, not treated as a new collision.
            prior_round_artefacts += sum(1 for name in files if pattern.search(name))
            continue
        for name in files:
            if pattern.search(name):
                embedded.append(str(here.relative_to(common.ROOT) / name))
    seeded = []
    for path in sorted((common.ROOT / "runs").glob("**/*.json")):
        if (common.HERE in path.parents or path.parent == common.HERE
                or common.OUTPUT_DIR in path.parents or path.parent == common.OUTPUT_DIR
                or PRIOR_BLOCK_USER_DIR in path.parents or path.parent == PRIOR_BLOCK_USER_DIR):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if re.search(r'"seed"\s*:\s*1300(?:0[0-4][0-9]|1[0-5])\b', text) or re.search(
                r'"seed"\s*:\s*13010(?:0[0-5]\b)', text):
            seeded.append(str(path.relative_to(common.ROOT)))
    return {"claimed_ranges": [{"name": name, "low": low, "high": high}
                               for name, low, high in CLAIMED_RANGES],
            "new_blocks": {"training": [int(common.TRAINING_SCENARIOS[0]),
                                        int(common.TRAINING_SCENARIOS[-1])],
                           "development": [int(common.DEVELOPMENT_SCENARIOS[0]),
                                           int(common.DEVELOPMENT_SCENARIOS[-1])]},
            "reuse_registration": {
                "status": "intentional_development_reuse",
                "registered_by": "ROUND22_COMM_COST_LEARNING_PILOT.md §4",
                "prior_user": "runs/comm_cost_stage_c_20260927 (ROUND22 execution, 677 launches)",
                "prior_round_artefacts_embedding_block_seeds": prior_round_artefacts,
                "note": "the same development blocks are reused deliberately; they are not "
                        "claimed as new or unseen, and the sealed block stays sealed"},
            "range_overlaps": findings,
            "artefacts_embedding_block_seeds_outside_registered_use": embedded,
            "artefacts_with_block_seeds_outside_registered_use": seeded,
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


# ------------------------------------------------------------------ 与旧 Stage C 的逐项 diff -------------
def stage_c_difference_audit(new_configuration_hash: str) -> dict[str, Any]:
    """ROUND24 §2.4: item-by-item diff against the old Stage C machine-readable settings.

    Only the registered differences may remain: the four correctness repairs (already in the
    verified fix chain, with their own evidence), output isolation (directory and artefact
    naming), and this spec's evaluation arrangement. Anything else -- LR, epochs, entropy,
    clip, width, features, world settings -- stops the freeze.
    """
    old_path = PRIOR_BLOCK_USER_DIR / "CONFIG_AND_MODEL_MANIFEST.json"
    if not old_path.exists():
        raise RuntimeError("the old Stage C manifest is missing; the diff baseline is unreadable")
    old = json.loads(old_path.read_text(encoding="utf-8"))
    diffs: list[dict[str, Any]] = []

    def compare(label: str, old_value: Any, new_value: Any, *, registered: bool) -> None:
        if old_value != new_value:
            diffs.append({"field": label, "old": old_value, "new": new_value,
                          "registered": registered})

    for key, value in sorted(old.get("wiring_constants", {}).items()):
        new_value = {"b0_bytes": common.B0_BYTES, "delta": common.DELTA,
                     "lambda_init": common.LAMBDA_INIT, "eta_lambda": common.ETA_LAMBDA,
                     "gamma": common.GAMMA, "gae_lambda": common.GAE_LAMBDA,
                     "data_order_seed": common.DATA_ORDER_SEED,
                     "log_prob_tolerance": common.LOG_PROB_TOLERANCE}.get(key)
        if key == "note":
            continue
        compare(f"wiring_constants.{key}", value, new_value, registered=False)
    old_ppo = old.get("ppo_hyperparameters", {})
    new_ppo = {"clip": common.CLIP, "value_coefficient_per_head": common.VALUE_COEFFICIENT,
               "entropy_bonus": common.ENTROPY_BONUS, "learning_rate": common.LEARNING_RATE,
               "max_grad_norm": common.MAX_GRAD_NORM, "epochs_per_batch": common.EPOCHS_PER_BATCH,
               "minibatch_steps": common.MINIBATCH_STEPS,
               "episodes_per_update": common.EPISODES_PER_UPDATE}
    for key, value in sorted(old_ppo.items()):
        if key in ("provenance", "registered_differences_from_old_trainer"):
            continue
        compare(f"ppo_hyperparameters.{key}", value, new_ppo.get(key), registered=False)
    old_registry = old.get("scene_registry", {})
    compare("scene_registry.training_block", old_registry.get("training_block"),
            [int(common.TRAINING_SCENARIOS[0]), int(common.TRAINING_SCENARIOS[-1])],
            registered=False)
    compare("scene_registry.development_block", old_registry.get("development_block"),
            [int(common.DEVELOPMENT_SCENARIOS[0]), int(common.DEVELOPMENT_SCENARIOS[-1])],
            registered=False)
    compare("rule_arms", old.get("rule_arms"),
            {"reference": common.REFERENCE_ARM, "rules": common.RULE_ARMS,
             "send_once_alias": common.SEND_ONCE_ALIAS_NOTE}, registered=False)
    compare("configuration_hash", old.get("configuration_hash"), new_configuration_hash,
            registered=False)
    unregistered = [diff for diff in diffs if not diff["registered"]]
    registered_differences = [
        {"difference": "four trainer correctness repairs (cost sign, cost-head optimizer "
                       "lifecycle, GAE next-value, reward attribution)",
         "where": "inherited from the verified fix chain 8970ce9..b7217d1 (= remote 64cddcf "
                  "tree); evidence: TRAINER_CORRECTNESS_FINDINGS.md, "
                  "CORRECTNESS_REGRESSION_EVIDENCE.json"},
        {"difference": "output isolation",
         "where": "runs/comm_cost_v2_rerun_20260928 instead of any earlier directory; "
                  "artefact stems v2_ instead of sc_; old round directory read-only"},
        {"difference": "evaluation arrangement (ROUND24 §5)",
         "where": "init_greedy / final_greedy / final_sampled replace the old 0/64/128 greedy "
                  "blocks; no 64-episode development evaluation; milestones ep0/64/128 saved "
                  "and only ep128 evaluated; eval action RNG registered per (seed, scene, mode)"},
    ]
    return {"old_manifest": str(old_path.relative_to(common.ROOT)),
            "diffs": diffs,
            "registered_differences": registered_differences,
            "clean": not unregistered,
            "note": "the four repairs are already inside the verified fix-chain code; this "
                    "diff is against the old round-22 freeze to show no NEW parameter moved"}


def main() -> int:
    local_head = git("rev-parse", "HEAD")
    equivalence = start_equivalence()
    diff = git("diff", "--name-only", BRANCH_START, "--", *PRODUCTION_TREES)
    changed = [line for line in diff.splitlines() if line]
    audit = audit_collisions()
    if changed:
        raise SystemExit(f"production trees changed since the branch start: {changed}")
    report = common.ledger_report()
    persisted = sorted(common.CHECKPOINTS.glob("v2_train_s*.pt")) if common.CHECKPOINTS.is_dir() else []
    if persisted:
        raise SystemExit("a persisted v2 training checkpoint exists under an earlier code "
                         "identity; refusing to re-freeze (the rerun would mix two code states)")
    training_started = any(str(entry.get("label", "")).startswith("train_s")
                           for entry in report["entries"])
    if training_started:
        raise SystemExit("training launches are already in this round's ledger; refusing to re-freeze")
    if not audit["clean"]:
        raise SystemExit(f"seed collision audit failed: {json.dumps(audit, ensure_ascii=False)}")

    synthetic_path = common.OUTPUT_DIR / "SYNTHETIC_GATE.json"
    if not synthetic_path.exists():
        raise SystemExit("SYNTHETIC_GATE.json is missing; run the zero-environment synthetic "
                         "gate first (ROUND24 §2.1) and register its actual numbers")
    synthetic = json.loads(synthetic_path.read_text(encoding="utf-8"))
    if not synthetic.get("passed"):
        raise SystemExit("the synthetic gate did not pass; refusing to freeze")
    code_hashes = common.code_identity()
    if synthetic.get("freeze_sha256") != code_hashes["freeze_sha256"]:
        raise SystemExit("the synthetic gate ran under a different code identity "
                         f"({synthetic.get('freeze_sha256')} != {code_hashes['freeze_sha256']}); "
                         "re-run it under the code that is about to be frozen")

    identity = common.code_identity()
    fields = behaviour_fields()
    config_hash = configuration_hash(fields)
    difference_audit = stage_c_difference_audit(config_hash)
    if not difference_audit["clean"]:
        raise SystemExit("unregistered differences from the old Stage C settings: "
                         + json.dumps(difference_audit["diffs"], ensure_ascii=False))
    manifest = {
        "round": "24", "phase": "ROUND24 correctness-v2 rerun freeze (start from initialisation)",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head_at_freeze": local_head,
        "branch_start": BRANCH_START,
        "remote_start_head": REMOTE_START_HEAD,
        "start_equivalence": equivalence,
        "spec": {"file": "ROUND24_COMM_COST_V2_RERUN.md",
                 "source_commit": "86cff0226e5552c66c799cc7d210fa6006d058e0",
                 "source_branch": "docs/comm-cost-v2-rerun-plan",
                 "blob_sha1": "5d8ae2a556636c747043dca648a19d2efe3c82bc",
                 "sha256": common.sha256_file(
                     common.ROOT / "ROUND24_COMM_COST_V2_RERUN.md"),
                 "note": "git port unreachable this session; the document was fetched by the "
                         "authorised GitHub channel at the fixed commit and saved byte-for-byte"},
        "production_diff": {"branch_start": BRANCH_START, "trees": list(PRODUCTION_TREES),
                            "changed_files": changed, "clean": True},
        "code_identity": identity,
        "configuration_hash": config_hash,
        "inherited_behaviour_fields": fields,
        "stage_c_difference_audit": difference_audit,
        "wiring_constants": {
            "b0_bytes": common.B0_BYTES, "delta": common.DELTA,
            "lambda_init": common.LAMBDA_INIT, "eta_lambda": common.ETA_LAMBDA,
            "gamma": common.GAMMA, "gae_lambda": common.GAE_LAMBDA,
            "data_order_seed": common.DATA_ORDER_SEED,
            "log_prob_tolerance": common.LOG_PROB_TOLERANCE,
            "note": "delta=0 is this round's experimental training target, not a deployment "
                    "zero-degradation guarantee (spec §3)"},
        "ppo_hyperparameters": {
            "clip": common.CLIP, "value_coefficient_per_head": common.VALUE_COEFFICIENT,
            "entropy_bonus": common.ENTROPY_BONUS, "learning_rate": common.LEARNING_RATE,
            "max_grad_norm": common.MAX_GRAD_NORM, "epochs_per_batch": common.EPOCHS_PER_BATCH,
            "minibatch_steps": common.MINIBATCH_STEPS,
            "episodes_per_update": common.EPISODES_PER_UPDATE,
            "provenance": "reused from the pilot round (runs/comm_budget_pilot_20260919/"
                          "train_pilot.py); every value is stated rather than inherited by "
                          "silence (spec §3)",
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
            "reuse_registration": audit["reuse_registration"],
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
                                "init_greedy / final_greedy / final_sampled development "
                                "evaluation, 16 scenes each, per seed",
                                "five rule arms x 16 dev scenes"],
            "eval_blocks": {name: dict(spec) for name, spec in common.EVAL_BLOCKS.items()},
            "milestones": list(common.MILESTONE_EPISODES),
            "note": "the old 677 launches and all old artefacts stay untouched; this ledger is "
                    "an independent new budget (spec §4)"},
        "synthetic_gate": {
            "path": "runs/comm_cost_v2_rerun_20260928/SYNTHETIC_GATE.json",
            "recorded": synthetic,
            "note": "actual counts from the re-run ROUND23 counterexample suite under the "
                    "frozen code identity; the historical 20/20 string is not reused"},
    }
    manifest["pre_launch_history"] = {
        "count": len(report["results"]),
        "entries": report["results"],
        "note": "launches spent before this freeze in this round's ledger"}
    common.write_json("CONFIG_AND_MODEL_MANIFEST.json", manifest)
    common.write_json("SCENE_REGISTRY.json", {
        "training_block": manifest["scene_registry"]["training_block"],
        "development_block": manifest["scene_registry"]["development_block"],
        "training_scenarios": manifest["scene_registry"]["training_scenarios"],
        "development_scenarios": manifest["scene_registry"]["development_scenarios"],
        "reuse_registration": audit["reuse_registration"],
        "collision_audit": audit})
    common.write_json("START_LEDGER.json", {
        "plan": manifest["launch_plan"], "live": {"used": common.ledger_total()}})
    print(json.dumps({"head": manifest["head_at_freeze"],
                      "remote_start_head": REMOTE_START_HEAD,
                      "production_clean": True, "collision_clean": audit["clean"],
                      "configuration_hash": manifest["configuration_hash"],
                      "stage_c_diff_clean": difference_audit["clean"],
                      "synthetic_tests": synthetic.get("tests_run"),
                      "pre_launch_launches": len(report["results"]),
                      "normal": common.NORMAL_LAUNCHES, "cap": common.LAUNCH_CAP},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
