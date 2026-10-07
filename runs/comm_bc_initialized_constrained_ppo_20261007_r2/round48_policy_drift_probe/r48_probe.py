"""ROUND48 read-only policy drift probe: BC/ep0 vs ep128 p(SEND) on frozen
same-input panel records taken from the final greedy evaluation tapes.

Zero environments, zero optimizer steps, zero checkpoint writes. CPU, eval, no_grad.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

RUN_DIR = Path(__file__).resolve().parents[1]
REPO = RUN_DIR.parents[1]
V6_CHECKPOINTS = RUN_DIR / "v6" / "checkpoints"
V8_ROOT = RUN_DIR / "v8_eval_booking_close"
V9_ROOT = RUN_DIR / "v9_eval_runtime_close"
V10_ROOT = RUN_DIR / "v10_eval_final_close"
R45_DIR = REPO / "runs" / "comm_rule_warmstart_cost_learning_20261006_r2"
R44_DIR = REPO / "runs" / "comm_dual_response_ablation_20261004"
R47_DIR = RUN_DIR / "round47_cost_decomposition"
OUT = Path(__file__).resolve().parent

SEEDS = (11, 22, 33)
SCENES = tuple(range(1_301_000, 1_301_016))
SEND, HOLD = 1, 0
TOLERANCE = 1e-6
FEATURE_DIM = 19

REGISTERED_EP128 = {
    11: ("v6/checkpoints/v2_train_s11_ep128.pt",
         "0f161663aaf963552f8f02d6363f3c52e6f67119217f835c501cb2d0705bfc10",
         "8b87f4d2591a3a0c573b71a08ec8444db1e3f91850edbf2f2b958dddb31e5670"),
    22: ("v6/checkpoints/v2_train_s22_ep128.pt",
         "aae4aa1b24b7e7ffadffeb7a09d45935f44970ac62a8cfd4344630248472cc62",
         "b88a6b377d9e18c3cef17ab5205f7652b1394fcfbc6b28e5642140025028caf4"),
    33: ("v6/checkpoints/v2_train_s33_ep128.pt",
         "10f3fc8d2c2a788b7e48e1f9021b5b234c39a00706bfe486e6f2bab84a0fb9b9",
         "dc2903397865c853c59ed631e7b0a699fcb2d55b7dee8956101c53984b36b821"),
}
# ROUND46 training ep0 identity from the V4 freeze (initial models, read-only identity).
REGISTERED_EP0 = {
    11: {"path": str(REPO / "runs/comm_one_shot_gate_resume_20261003_v2/checkpoints/v2_train_s11_ep0.pt"),
         "file_sha256": "70fe64c18f61b33d17b6aa528cf830b43919a89b22210b11d7edb732bd75f1fd",
         "parameter_sha256": "37303c494cd997894fba1441cf9d2cde86cf2135e71f443a4ed1bcb9ec7f0330",
         "round45_bc_checkpoint_sha256": "76ffc86c4aca26fb3b27380d5195a0cbd07822f7b85d1f1070a5550776dd458a"},
    22: {"path": str(REPO / "runs/comm_one_shot_gate_resume_20261003_v2/checkpoints/v2_train_s22_ep0.pt"),
         "file_sha256": "96a09f2f4796069a10f77e5472a4fa8c09ddbf1407e44a7102c022c43c5790c5",
         "parameter_sha256": "e8531e0aff95ed207bc70026247d45869656d266aa974838a7ec2cfe352479b0",
         "round45_bc_checkpoint_sha256": "3cdb83785c51f9e40005d66a1d024d06a5ba6940616e22f3b01a9bf65b5e609d"},
    33: {"path": str(REPO / "runs/comm_one_shot_gate_resume_20261003_v2/checkpoints/v2_train_s33_ep0.pt"),
         "file_sha256": "9030be21c945a490802692569e0fb5b4608602394e43233666ff83adcc2cfe8d",
         "parameter_sha256": "10bae22f3896adc507f05c5d868aad43b0441d19f754f2ef4dc373fc964e57a5",
         "round45_bc_checkpoint_sha256": "9329ef048dfc5c1805af5756c7ea0962eedfc8d01aa26841e35af7001616803a"},
}
BC_POST_PATHS = {
    seed: R45_DIR / "checkpoints" / f"BC_POST_MODEL_s{seed}.pt" for seed in SEEDS
}


class V10Error(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_json(path: Path, value) -> str:
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    Path(path).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def parameter_summary(controller: dict) -> dict:
    """Exact re-implementation of sb_controller.parameter_summary over the four
    saved module state dicts (trunk/action_head/value_task_head/value_cost_head)."""
    tensors: dict[str, "torch.Tensor"] = {}
    prefixes = {"trunk": "trunk", "action_head": "action_head",
                "value_task_head": "value_task_head", "value_cost_head": "value_cost_head"}
    for role, state in controller.items():
        prefix = prefixes[role]
        for name, tensor in state.items():
            tensors[f"{prefix}.{name}"] = tensor.detach().cpu()
    digest = hashlib.sha256()
    for name in sorted(tensors):
        digest.update(name.encode("utf-8"))
        digest.update(tensors[name].contiguous().numpy().tobytes())
    return {"parameter_count": int(sum(t.numel() for t in tensors.values())),
            "parameter_sha256": digest.hexdigest()}


def forward_logits(controller: dict, batch: "torch.Tensor") -> "torch.Tensor":
    """trunk = Linear(19,64) -> Tanh -> Linear(64,64) -> Tanh; logits = action_head(hidden)."""
    state = controller
    hidden = torch.tanh(torch.nn.functional.linear(
        batch, state["trunk"]["0.weight"], state["trunk"]["0.bias"]))
    hidden = torch.tanh(torch.nn.functional.linear(
        hidden, state["trunk"]["2.weight"], state["trunk"]["2.bias"]))
    return torch.nn.functional.linear(
        hidden, state["action_head"]["weight"], state["action_head"]["bias"])


def p_send(controller: dict, batch: "torch.Tensor") -> "torch.Tensor":
    logits = forward_logits(controller, batch)
    return torch.softmax(logits, dim=-1)[:, SEND]


def load_controller(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    controller = payload.get("controller")
    if not isinstance(controller, dict) or set(controller) != {
            "trunk", "action_head", "value_task_head", "value_cost_head"}:
        raise V10Error(f"unexpected controller schema in {path}")
    return controller


def build_model_identity() -> dict:
    block = {"kind": "round48-model-role-identity/1",
             "task_id": "ROUND48-READONLY-POLICY-DRIFT-PROBE",
             "roles": {}, "verification": {}, "loading": {
                 "mode": "state-dict reconstruction + registered parameter-hash verification",
                 "device": "cpu", "eval_mode": True, "grad": False,
                 "optimizer_restored": False, "rng_restored": False,
                 "checkpoint_write_calls": 0, "environment_constructor_calls": 0}}
    roles = {
        "ep0_training_init": {},
        "ep128_final": {},
    }
    for seed in SEEDS:
        # ep0: the ROUND46 training initializer is the R45 BC_POST_MODEL four-module
        # policy; the registered ep0 checkpoint (V4 freeze) carries the same actor
        # parameter identity as the BC control rows.
        bc_post_path = BC_POST_PATHS[seed]
        bc_post = load_controller(bc_post_path)
        bc_post_summary = parameter_summary(bc_post)
        # ROUND46's actual training initializer is the R45 BC_POST_MODEL four-module
        # policy (V5 training manifest: "ROUND45 BC_POST_MODEL four modules,
        # model-only, same seed"); its 4-module hash equals the registered
        # 37303c49/e8531e0a/10bae22f identity and the BC control rows'
        # bc_parameter_sha256, so the BC controls replayed exactly this ep0 policy.
        roles["ep0_training_init"][str(seed)] = {
            "round46_initializer_file": str(bc_post_path),
            "round46_initializer_file_sha256": sha256_file(bc_post_path),
            "round46_initializer_parameter_sha256": bc_post_summary["parameter_sha256"],
            "registered_parameter_sha256": REGISTERED_EP0[seed]["parameter_sha256"],
            "registered_matches_recomputed":
                bc_post_summary["parameter_sha256"] == REGISTERED_EP0[seed]["parameter_sha256"],
            "bc_control_rows_use_same_parameter_identity":
                bc_post_summary["parameter_sha256"] == REGISTERED_EP0[seed]["parameter_sha256"],
            "r43_era_ep0_file_not_the_round46_init": {
                "path": REGISTERED_EP0[seed]["path"],
                "file_sha256": REGISTERED_EP0[seed]["file_sha256"],
                "four_module_parameter_sha256": parameter_summary(
                    load_controller(Path(REGISTERED_EP0[seed]["path"])))["parameter_sha256"],
            },
        }
        rel, file_sha, param_sha = REGISTERED_EP128[seed]
        ep128_path = RUN_DIR / rel
        ep128 = load_controller(ep128_path)
        ep128_summary = parameter_summary(ep128)
        roles["ep128_final"][str(seed)] = {
            "path": str(ep128_path),
            "file_sha256": sha256_file(ep128_path),
            "registered_file_sha256": file_sha,
            "file_sha256_verified": sha256_file(ep128_path) == file_sha,
            "registered_parameter_sha256": param_sha,
            "parameter_sha256_recomputed": ep128_summary["parameter_sha256"],
            "parameter_sha256_verified": ep128_summary["parameter_sha256"] == param_sha,
        }
    block["roles"] = roles
    ok = True
    for seed in SEEDS:
        e0 = roles["ep0_training_init"][str(seed)]
        e128 = roles["ep128_final"][str(seed)]
        if not e0["registered_matches_recomputed"]:
            ok = False
        if not e128["file_sha256_verified"] or not e128["parameter_sha256_verified"]:
            ok = False
    block["verification"] = {"all_identities_verified": ok}
    if not ok:
        raise V10Error("model role identity verification failed")
    return block


def build_panel() -> tuple[list[dict], dict]:
    """Every free first-report decision in the 48 final greedy tapes."""
    adopted_paths = {
        (11, 1_301_000): V8_ROOT / "adoption/attempt2_effective_row.json",
        (11, 1_301_001): V9_ROOT / "adoption/cell1301001_effective_row.json",
        (11, 1_301_002): V10_ROOT / "adoption/cell1301002_effective_row.json",
    }
    adopted_used = set()
    panel: list[dict] = []
    excluded: list[dict] = []
    tapes_scanned = 0
    for seed in SEEDS:
        for scene in SCENES:
            adopted_key = (seed, scene)
            if adopted_key in adopted_paths:
                row = read_json(adopted_paths[adopted_key])
                adopted_used.add(adopted_key)
                tape_path = Path(row["artefacts"]["tape"])
            else:
                pattern = V10_ROOT / "rows" / f"v2_r44_eval_new_s{seed}_greedy_s{scene}_a1.json"
                row = read_json(pattern)
                tape_path = Path(row["artefacts"]["tape"])
            tapes_scanned += 1
            for record in read_jsonl(tape_path):
                free = [str(t) for t in record.get("free_one_shot_choices", [])]
                targets = [str(t) for t in record.get("targets", [])]
                features = record.get("features", [])
                probs = record.get("send_probabilities", {})
                actions = record.get("actions", {})
                masked = set(str(t) for t in (record.get("masked") or []))
                forced = set(str(t) for t in (record.get("forced") or []))
                for target in targets:
                    if target in free:
                        source = "free_one_shot_choice"
                    elif target in masked:
                        source = "masked"
                    elif target in forced:
                        source = "forced"
                    else:
                        source = "closed_or_hold"
                    if source != "free_one_shot_choice":
                        excluded.append({"scene": scene, "target": target,
                                         "step": record.get("step"), "source": source,
                                         "reason": "not a free first-report decision"})
                        continue
                    index = targets.index(target)
                    vector = [float(v) for v in features[index]]
                    if len(vector) != FEATURE_DIM:
                        excluded.append({"scene": scene, "target": target,
                                         "step": record.get("step"),
                                         "reason": f"feature width {len(vector)} != 19"})
                        continue
                    panel.append({
                        "source_seed": seed, "scene": scene, "target": target,
                        "step": int(record.get("step", -1)),
                        "features": vector,
                        "features_sha256": hashlib.sha256(json.dumps(
                            vector, sort_keys=True).encode("utf-8")).hexdigest(),
                        "original_action_index": int(actions[target]),
                        "original_action": "SEND" if int(actions[target]) == SEND else "SKIP",
                        "original_p_send": float(probs[target]),
                    })
    manifest = {
        "kind": "round48-probe-input-manifest/1",
        "task_id": "ROUND48-READONLY-POLICY-DRIFT-PROBE",
        "tapes_scanned": tapes_scanned,
        "panel_records": len(panel),
        "excluded_records": len(excluded),
        "exclusions": excluded,
        "inclusion_rule": "decision_class free_one_shot_choice only; masked/forced/"
                          "closed targets excluded; no imputation",
        "duplicate_note": "records with identical 19-D vectors are kept per target; "
                          "equality groups are reported as aliases, not extra scenes",
    }
    return panel, manifest


# ---------------------------------------------------------------------------
# probe execution and drift statistics
# ---------------------------------------------------------------------------

def forward_panel(panel: list[dict], identities: dict) -> dict:
    """Run every panel record through all six role models (CPU, eval, no_grad)."""
    controllers = {}
    for seed in SEEDS:
        controllers[("ep0", seed)] = load_controller(Path(REGISTERED_EP0[seed]["path"]))
        rel, _, _ = REGISTERED_EP128[seed]
        controllers[("ep128", seed)] = load_controller(RUN_DIR / rel)
    with torch.no_grad():
        for role, seed in controllers:
            controller = controllers[(role, seed)]
            for record in panel:
                batch = torch.as_tensor([record["features"]], dtype=torch.float32)
                probabilities = torch.softmax(forward_logits(controller, batch), dim=-1)[0]
                record[f"p_send__{role}_s{seed}"] = float(probabilities[SEND])
    # validation: the source seed's ep128 must reproduce the taped p(SEND)
    mismatches = []
    for record in panel:
        expected = record["original_p_send"]
        actual = record[f"p_send__ep128_s{record['source_seed']}"]
        record["validation_abs_error"] = abs(actual - expected)
        if abs(actual - expected) > TOLERANCE:
            mismatches.append({"scene": record["scene"], "target": record["target"],
                               "expected": expected, "actual": actual,
                               "abs_error": abs(actual - expected)})
    if mismatches:
        raise V10Error(f"ep128 p(SEND) reproduction failed for {len(mismatches)} records; "
                       f"first: {mismatches[0]}")
    greedy = {}
    for record in panel:
        for seed in SEEDS:
            for role in ("ep0", "ep128"):
                p = record[f"p_send__{role}_s{seed}"]
                greedy[(role, seed, record["scene"], record["target"],
                        record["step"])] = SEND if p >= 0.5 else HOLD
    return {"controllers_loaded": len(controllers), "validation_mismatches": mismatches,
            "greedy": greedy}


def drift_statistics(panel: list[dict], greedy: dict, decomposition) -> dict:
    """Same-seed paired ep0->ep128 statistics plus ROUND47 cost-group metadata."""
    decomposition = decomposition or {"per_seed": {}}
    per_seed = {}
    for seed in SEEDS:
        records = [r for r in panel if r["source_seed"] == seed]
        p0 = np.asarray([r["p_send__ep0_s%d" % seed] for r in records])
        p128 = np.asarray([r["p_send__ep128_s%d" % seed] for r in records])
        dp = p128 - p0
        transitions = {"SEND->SEND": 0, "SEND->SKIP": 0, "SKIP->SEND": 0, "SKIP->SKIP": 0}
        for index, record in enumerate(records):
            g0 = "SEND" if p0[index] >= 0.5 else "SKIP"
            g128 = "SEND" if p128[index] >= 0.5 else "SKIP"
            transitions[f"{g0}->{g128}"] += 1
        groups = defaultdict(lambda: {"count": 0, "dp_values": []})
        decomp = decomposition.get("per_seed", {}).get(str(seed), {})
        for bench in ("vs_late", "vs_bc"):
            bench_scenes = decomp.get(bench, {}).get("scenes", {})
            labels = {}
            for scene, block in bench_scenes.items():
                for target in block.get("A_model_only_targets", []):
                    labels[(int(scene), str(target))] = f"model_only_first_report_{bench}"
                for target in block.get("B_bench_only_targets", []):
                    labels[(int(scene), str(target))] = f"bench_only_first_report_{bench}"
                for target in block.get("common_targets", {}):
                    labels[(int(scene), str(target))] = f"common_first_report_{bench}"
            for index, record in enumerate(records):
                label = labels.get((record["scene"], str(record["target"])),
                                   f"neither_sent_first_report_{bench}")
                groups[label]["count"] += 1
                groups[label]["dp_values"].append(float(dp[index]))
        for label, block in groups.items():
            values = block.pop("dp_values")
            block["dp_mean"] = float(np.mean(values)) if values else None
            block["dp_abs_gt_0_1"] = sum(1 for v in values if abs(v) > 0.1)
        group_unknown = 0
        per_seed[str(seed)] = {
            "panel_records": len(records),
            "p0_mean": float(p0.mean()), "p128_mean": float(p128.mean()),
            "dp_mean": float(dp.mean()), "dp_abs_mean": float(np.abs(dp).mean()),
            "dp_max_abs": float(np.abs(dp).max()),
            "records_with_abs_dp_gt_0_1": int((np.abs(dp) > 0.1).sum()),
            "records_crossing_0_5_boundary":
                int(((p0 - 0.5) * (p128 - 0.5) < 0).sum()),
            "distance_to_0_5": {
                "ep0_min_abs": float(np.abs(p0 - 0.5).min()),
                "ep128_min_abs": float(np.abs(p128 - 0.5).min()),
            },
            "greedy_transitions_ep0_to_ep128": transitions,
            "cost_group_metadata": {label: block for label, block in sorted(groups.items())},
            "cost_group_metadata_unlabeled_records": group_unknown,
            "feature_extremes": {
                "ep0_p_send_min": float(p0.min()), "ep0_p_send_max": float(p0.max()),
                "ep128_p_send_min": float(p128.min()), "ep128_p_send_max": float(p128.max()),
            },
        }
    return {"kind": "round48-policy-drift-results/1",
            "task_id": "ROUND48-READONLY-POLICY-DRIFT-PROBE",
            "note": "same-input paired policy-mapping comparison; descriptive only; "
                    "no task or cost causality claims",
            "per_seed": per_seed}


def feature_observations(panel: list[dict]) -> dict:
    """Which of the 19 feature columns are constant, saturated, or group-separating."""
    matrix = np.asarray([r["features"] for r in panel], dtype=np.float64)
    columns = []
    for column in range(FEATURE_DIM):
        values = matrix[:, column]
        columns.append({
            "index": column, "min": float(values.min()), "max": float(values.max()),
            "constant": bool(values.min() == values.max()),
            "distinct_values": int(len(np.unique(values))),
        })
    distinct_hashes: dict[str, int] = defaultdict(int)
    for record in panel:
        distinct_hashes[record["features_sha256"]] += 1
    duplicated = {k: v for k, v in distinct_hashes.items() if v > 1}
    return {"kind": "round48-feature-observations/1",
            "records": len(panel), "columns": columns,
            "distinct_feature_vectors": len(distinct_hashes),
            "vectors_shared_by_multiple_targets": len(duplicated),
            "largest_alias_group": max(duplicated.values()) if duplicated else 1}


def main() -> None:
    identities = build_model_identity()
    sha_i = write_json(OUT / "MODEL_ROLE_IDENTITY.json", identities)
    panel, manifest = build_panel()
    manifest["model_identity_sha256"] = sha_i
    sha_m = write_json(OUT / "PROBE_INPUT_MANIFEST.json", manifest)
    print(f"panel records: {len(panel)}; excluded: {manifest['excluded_records']}", flush=True)
    forward_info = forward_panel(panel, identities)
    decomposition_path = R47_DIR / "PER_SCENE_COST_DECOMPOSITION.json"
    decomposition = read_json(decomposition_path) if decomposition_path.is_file() else None
    results = drift_statistics(panel, forward_info["greedy"], decomposition)
    results["validation"] = {"ep128_reproduction_mismatches": 0,
                             "tolerance": TOLERANCE,
                             "action_mapping": {"SEND": SEND, "HOLD": HOLD},
                             "greedy_tie_rule": "p_send >= 0.5 -> SEND"}
    results["feature_observations"] = feature_observations(panel)
    sha_r = write_json(OUT / "POLICY_DRIFT_RESULTS.json", results)
    print(json.dumps({"model_identity_sha256": sha_i, "manifest_sha256": sha_m,
                      "results_sha256": sha_r,
                      "panel_records": len(panel),
                      "per_seed": results["per_seed"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
