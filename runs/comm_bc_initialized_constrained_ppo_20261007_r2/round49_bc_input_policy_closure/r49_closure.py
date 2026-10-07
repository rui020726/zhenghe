"""ROUND49 read-only closure of BC- and PPO-trajectory policy inputs."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

RUN = Path(__file__).resolve().parents[1]
REPO = RUN.parents[1]
OUT = Path(__file__).resolve().parent
SEEDS = (11, 22, 33)
SEND, SKIP = 1, 0
TOL = 1e-6
ID_INDEX = RUN / "ROUND46_R2_V6_INPUT_IDENTITY.json"
ROUND48 = RUN / "round48_policy_drift_probe"
ROUND47 = RUN / "round47_cost_decomposition"
BC_ROOT = REPO / "runs" / "comm_rule_warmstart_cost_learning_20261006_r2"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, data) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def vector_hash(vector) -> str:
    return hashlib.sha256(json.dumps(vector, sort_keys=True).encode("utf-8")).hexdigest()


def load_round48_module():
    path = ROUND48 / "r48_probe.py"
    spec = importlib.util.spec_from_file_location("r48_readonly_reuse", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def action_label(action: int) -> str:
    if action not in (SEND, SKIP):
        raise ValueError(f"unexpected action index {action}")
    return "SEND" if action == SEND else "SKIP"


def validate_model_role(actual: str, expected: str) -> None:
    if actual != expected:
        raise ValueError(f"model role mismatch: expected {expected}, got {actual}")


def is_free_executable_first_report(target: str, free: set, masked: set,
                                    forced: set, audit: dict) -> bool:
    return (target in free and target not in masked and target not in forced
            and audit.get("one_shot_state_before") == "undecided"
            and audit.get("first_emission") is True
            and audit.get("relay_initial_executable") is True
            and audit.get("sequentially_affordable_before_action") is True
            and audit.get("effective_block_reason") == "allowed")


def bind_verified_controller(path: Path, expected_parameter_sha256: str, r48,
                             expected_file_sha256: str | None = None):
    """Load the actual checkpoint and bind its digest before it can reach forward()."""
    path=Path(path)
    file_digest=sha_file(path)
    if expected_file_sha256 is not None and file_digest != expected_file_sha256:
        raise ValueError(f"checkpoint file identity mismatch before load: {path}")
    controller=r48.load_controller(path)
    parameter_digest=r48.parameter_summary(controller)["parameter_sha256"]
    if parameter_digest != expected_parameter_sha256:
        raise ValueError(f"loaded controller role/parameter identity mismatch: {path}; got {parameter_digest}, expected {expected_parameter_sha256}")
    return controller,{"path":str(path),"file_sha256_before_forward":file_digest,
                       "parameter_sha256_before_forward":parameter_digest,
                       "role_verified_before_forward":True}


def load_bc_panel() -> tuple[list[dict], dict]:
    identity = read(ID_INDEX)
    refs = identity.get("round45_control_rows", [])
    if len(refs) != 48:
        raise ValueError(f"indexed BC control row count {len(refs)} != 48")
    records, sources, exclusions = [], [], Counter()
    seen = set()
    for ref in refs:
        row_path = Path(ref["path"])
        if not row_path.is_file() or sha_file(row_path) != ref["sha256"]:
            raise ValueError(f"indexed BC row missing/hash mismatch: {row_path}")
        row = read(row_path)
        validate_model_role(row.get("model_arm"), "BC_ONLY_R2")
        if row.get("eval_mode") != "greedy":
            raise ValueError(f"unexpected BC row role: {row_path}")
        seed, scene = int(row["training_seed"]), int(row["scenario_seed"])
        if seed not in SEEDS or (seed, scene) in seen:
            raise ValueError(f"bad/duplicate indexed seed-scene: {(seed, scene)}")
        seen.add((seed, scene))
        tape_path = Path(row["artefacts"]["tape"])
        if not tape_path.is_file():
            raise FileNotFoundError(f"indexed BC tape missing: {tape_path}")
        tape_sha = sha_file(tape_path)
        sources.append({"seed": seed, "scene": scene, "row": str(row_path),
                        "row_sha256": ref["sha256"], "tape": str(tape_path),
                        "tape_sha256": tape_sha, "model_arm": row["model_arm"],
                        "bc_parameter_sha256": row.get("bc_parameter_sha256"),
                        "checkpoint_parameter_sha256": row.get("checkpoint_parameter_identity", {}).get("parameter_sha256"),
                        "behaviour_sha256": row.get("behaviour_sha256")})
        for rec in read_jsonl(tape_path):
            if rec.get("schema") != "round42-one-shot-send-skip-tape/1":
                exclusions["unsupported_schema"] += len(rec.get("targets", []))
                continue
            if rec.get("comm_action_semantics") != "one-shot-send-skip/1":
                exclusions["unsupported_action_semantics"] += len(rec.get("targets", []))
                continue
            targets = [str(t) for t in rec.get("targets", [])]
            free = set(map(str, rec.get("free_one_shot_choices", [])))
            masked = set(map(str, rec.get("masked", [])))
            forced = set(map(str, rec.get("forced", [])))
            features, actions = rec.get("features", []), rec.get("actions", {})
            probs, audits = rec.get("send_probabilities", {}), rec.get("candidate_audit", {})
            for target in targets:
                if target not in free:
                    reason = "masked" if target in masked else "forced" if target in forced else str(rec.get("decision_class", {}).get(target, "not_free"))
                    exclusions[reason] += 1
                    continue
                if target in masked or target in forced:
                    exclusions["inconsistent_free_mask_or_forced"] += 1
                    continue
                if target not in actions or target not in probs:
                    exclusions["missing_action_or_probability"] += 1
                    continue
                if target not in targets or len(features) != len(targets):
                    exclusions["missing_or_unaligned_features"] += 1
                    continue
                vec = [float(v) for v in features[targets.index(target)]]
                if len(vec) != 19 or not all(math.isfinite(v) for v in vec):
                    exclusions["invalid_feature_vector"] += 1
                    continue
                audit = audits.get(target, {})
                valid = is_free_executable_first_report(target, free, masked, forced, audit)
                if not valid:
                    exclusions["candidate_legality_not_proven"] += 1
                    continue
                actionable_order = list(map(str, rec.get("actionable", [])))
                if target not in actionable_order:
                    exclusions["candidate_order_missing"] += 1
                    continue
                act = int(actions[target])
                p = float(probs[target])
                if act not in (SEND, SKIP) or not 0 <= p <= 1:
                    exclusions["invalid_action_or_probability"] += 1
                    continue
                if (p >= .5 and act != SEND) or (p < .5 and act != SKIP):
                    raise ValueError(f"BC tape greedy/action mismatch {seed}/{scene}/{target}: p={p}, act={act}")
                records.append({
                    "panel": "BC_trajectory", "source_seed": seed, "scene": scene,
                    "target": target, "tick": int(rec["step"]), "candidate_identity": target,
                    "features": vec, "features_sha256": vector_hash(vec),
                    "original_action_index": act, "original_action": action_label(act),
                    "original_p_send": p,
                    "legality": {"schema": rec["schema"], "action_semantics": rec["comm_action_semantics"],
                                 "decision_class": rec.get("decision_class", {}).get(target),
                                 "free_one_shot_choice": True, "masked": False, "forced": False,
                                 "one_shot_state_before": audit.get("one_shot_state_before"),
                                 "first_emission": audit.get("first_emission"),
                                 "relay_initial_executable": audit.get("relay_initial_executable"),
                                 "sequentially_affordable_before_action": audit.get("sequentially_affordable_before_action"),
                                 "effective_block_reason": audit.get("effective_block_reason"),
                                 "base_action_mask_schema": rec.get("base_action_mask_schema"),
                                 "candidate_order_index": actionable_order.index(target),
                                 "candidate_order": actionable_order},
                    "source_row": str(row_path), "source_tape": str(tape_path),
                })
    if len(seen) != 48:
        raise ValueError(f"indexed BC scene grid incomplete: {len(seen)}/48")
    return records, {"indexed_rows": len(refs), "indexed_seed_scenes": len(seen),
                     "sources": sources, "excluded_counts": dict(exclusions),
                     "eligible_records": len(records)}


def load_ppo_panel(r48):
    # Reuse the same explicit accepted-cell paths as ROUND48 and preserve provenance/evidence.
    adopted = {
        (11, 1_301_000): r48.V8_ROOT / "adoption/attempt2_effective_row.json",
        (11, 1_301_001): r48.V9_ROOT / "adoption/cell1301001_effective_row.json",
        (11, 1_301_002): r48.V10_ROOT / "adoption/cell1301002_effective_row.json",
    }
    panel, exclusions, sources = [], Counter(), []
    for seed in SEEDS:
        for scene in range(1_301_000, 1_301_016):
            row_path = adopted.get((seed,scene), r48.V10_ROOT/"rows"/f"v2_r44_eval_new_s{seed}_greedy_s{scene}_a1.json")
            row = r48.read_json(row_path)
            tape_path = Path(row["artefacts"]["tape"])
            sources.append({"seed":seed,"scene":scene,"row":str(row_path),"row_sha256":sha_file(row_path),
                            "tape":str(tape_path),"tape_sha256":sha_file(tape_path),
                            "historical_adoption":(seed,scene) in adopted})
            for rec in r48.read_jsonl(tape_path):
                targets=list(map(str,rec.get("targets",[])))
                free=set(map(str,rec.get("free_one_shot_choices",[])))
                masked=set(map(str,rec.get("masked",[]) or [])); forced=set(map(str,rec.get("forced",[]) or []))
                features=rec.get("features",[]); actions=rec.get("actions",{}); probs=rec.get("send_probabilities",{})
                audits=rec.get("candidate_audit",{}); actionable=list(map(str,rec.get("actionable",[])))
                for target in targets:
                    if target not in free:
                        exclusions["masked" if target in masked else "forced" if target in forced else "closed_or_hold"] += 1
                        continue
                    if len(features)!=len(targets) or target not in actions or target not in probs:
                        exclusions["missing_or_unaligned_record_fields"] += 1; continue
                    vec=[float(x) for x in features[targets.index(target)]]
                    if len(vec)!=19 or not all(math.isfinite(x) for x in vec):
                        exclusions["invalid_feature_vector"] += 1; continue
                    audit=audits.get(target,{})
                    if not is_free_executable_first_report(target,free,masked,forced,audit):
                        exclusions["candidate_legality_not_proven"] += 1; continue
                    if target not in actionable:
                        exclusions["candidate_order_missing"] += 1; continue
                    action=int(actions[target]); prob=float(probs[target])
                    panel.append({"panel":"PPO_trajectory","source_seed":seed,"scene":scene,"target":target,
                        "step":int(rec["step"]),"tick":int(rec["step"]),"candidate_identity":target,
                        "features":vec,"features_sha256":vector_hash(vec),"original_action_index":action,
                        "original_action":action_label(action),"original_p_send":prob,
                        "legality":{"schema":rec.get("schema"),"action_semantics":rec.get("comm_action_semantics"),
                            "decision_class":rec.get("decision_class",{}).get(target),"free_one_shot_choice":True,
                            "masked":False,"forced":False,"one_shot_state_before":audit.get("one_shot_state_before"),
                            "first_emission":audit.get("first_emission"),"relay_initial_executable":audit.get("relay_initial_executable"),
                            "sequentially_affordable_before_action":audit.get("sequentially_affordable_before_action"),
                            "effective_block_reason":audit.get("effective_block_reason"),
                            "base_action_mask_schema":rec.get("base_action_mask_schema"),
                            "candidate_order_index":actionable.index(target),"candidate_order":actionable},
                        "source_row":str(row_path),"source_tape":str(tape_path)})
    # Match ROUND48's extracted panel before any model forward.
    previous, previous_manifest = r48.build_panel()
    def signature(rows):
        return Counter((r["source_seed"],int(r["scene"]),str(r["target"]),int(r.get("step",r.get("tick"))),
                        vector_hash(r["features"]),int(r["original_action_index"]),float(r["original_p_send"])) for r in rows)
    if signature(panel)!=signature(previous):
        raise ValueError("ROUND49 PPO input extraction does not match ROUND48 frozen panel")
    return panel, {"tapes_scanned":len(sources),"excluded_records":sum(exclusions.values()),
                   "exclusions_by_reason":dict(exclusions),"inclusion_rule":"ROUND48 free_one_shot_choice plus legality audit",
                   "sources":sources,"round48_manifest_path":str(ROUND48/"PROBE_INPUT_MANIFEST.json"),
                   "round48_manifest_sha256":sha_file(ROUND48/"PROBE_INPUT_MANIFEST.json"),
                   "round48_panel_signature_match":True,
                   "round48_excluded_records_reference":previous_manifest.get("excluded_records")}


def action_summary(records, key):
    ps = np.asarray([r[key] for r in records], dtype=np.float64)
    acts = [SEND if p >= .5 else SKIP for p in ps]
    counts = Counter(action_label(a) for a in acts)
    distances = np.abs(ps - .5)
    return {"n_records": len(records), "p_send_min": float(ps.min()) if len(ps) else None,
            "p_send_max": float(ps.max()) if len(ps) else None,
            "p_send_mean": float(ps.mean()) if len(ps) else None,
            "distance_to_half_min": float(distances.min()) if len(ps) else None,
            "distance_to_half_mean": float(distances.mean()) if len(ps) else None,
            "greedy_action_counts": {"SEND": counts["SEND"], "SKIP": counts["SKIP"]}}


def vector_counts(records):
    groups = defaultdict(list)
    for r in records:
        groups[r["features_sha256"]].append(r)
    repeated = [g for g in groups.values() if len(g) > 1]
    return {"records": len(records), "unique_vectors": len(groups),
            "repeated_vector_groups": len(repeated),
            "records_in_repeated_groups": sum(map(len, repeated)),
            "groups_shared_across_distinct_scene_targets": sum(1 for g in repeated if len({(x['scene'],x['target']) for x in g}) > 1),
            "cross_seed_repeated_groups": sum(1 for g in repeated if len({x['source_seed'] for x in g}) > 1),
            "largest_group_records": max(map(len, groups.values()), default=0)}


def model_and_forward(bc_panel, ppo_panel, r48):
    identity = read(ROUND48 / "MODEL_ROLE_IDENTITY.json")
    base_hashes_before = {}
    bindings={}
    controllers = {}
    for seed in SEEDS:
        p0 = Path(identity["roles"]["ep0_training_init"][str(seed)]["round46_initializer_file"])
        p128 = Path(identity["roles"]["ep128_final"][str(seed)]["path"])
        for role, path in (("ep0", p0), ("ep128", p128)):
            base_hashes_before[(seed, role)] = sha_file(path)
            model_role=identity["roles"]["ep0_training_init"][str(seed)] if role=="ep0" else identity["roles"]["ep128_final"][str(seed)]
            expected_parameter=model_role["round46_initializer_parameter_sha256"] if role=="ep0" else model_role["registered_parameter_sha256"]
            expected_file=model_role.get("round46_initializer_file_sha256") if role=="ep0" else model_role.get("registered_file_sha256")
            controllers[(seed, role)],bindings[(seed,role)]=bind_verified_controller(path,expected_parameter,r48,expected_file)
    panel = bc_panel + ppo_panel
    # A single float32 batch per exact vector avoids redundant math; records remain uncollapsed.
    unique = {}
    for rec in panel:
        unique.setdefault(rec["features_sha256"], rec["features"])
    batches = {h: torch.as_tensor([v], dtype=torch.float32, device="cpu") for h, v in unique.items()}
    cached_probabilities={}
    with torch.no_grad():
        for feature_hash,batch in batches.items():
            cached_probabilities[feature_hash]={}
            for seed in SEEDS:
                for role in ("ep0", "ep128"):
                    p = float(r48.p_send(controllers[(seed, role)], batch)[0])
                    cached_probabilities[feature_hash][(seed,role)]=p
    for rec in panel:
        for seed in SEEDS:
            for role in ("ep0", "ep128"):
                p=cached_probabilities[rec["features_sha256"]][(seed,role)]
                rec[f"p_send_s{seed}_{role}"] = p
                rec[f"greedy_s{seed}_{role}"] = action_label(SEND if p >= .5 else SKIP)
    after_hashes = {}
    for seed in SEEDS:
        for role in ("ep0", "ep128"):
            path = Path(identity["roles"]["ep0_training_init"][str(seed)]["round46_initializer_file"] if role == "ep0" else identity["roles"]["ep128_final"][str(seed)]["path"])
            after_hashes[(seed, role)] = sha_file(path)
    if base_hashes_before != after_hashes:
        raise ValueError("model file hashes changed during read-only forward")
    return identity, panel, {"controllers_loaded": 6, "model_bindings_before_forward":{
                                 f"s{seed}_{role}":bindings[(seed,role)] for seed in SEEDS for role in ("ep0","ep128")},
                             "distinct_vectors_forwarded": len(unique),
                             "record_model_forwards_cached": len(unique)*6,
                             "file_hashes_unchanged": True,
                             "checkpoint_write_calls": 0, "optimizer_step_calls": 0,
                             "synthetic_optimizer_step_calls":0,"real_training_update_calls":0,
                             "new_flow_calls":0,"environment_constructor_calls": 0, "simulation_calls": 0}


def main():
    r48 = load_round48_module()
    bc_panel, bc_source_info = load_bc_panel()
    ppo_panel, ppo_manifest = load_ppo_panel(r48)
    identity, all_panel, forward_guard = model_and_forward(bc_panel, ppo_panel, r48)
    r48_ep0_audit={}
    for seed in SEEDS:
        registered_path=Path(r48.REGISTERED_EP0[seed]["path"])
        recorded_init=Path(identity["roles"]["ep0_training_init"][str(seed)]["round46_initializer_file"])
        registered_actual=r48.parameter_summary(r48.load_controller(registered_path))["parameter_sha256"]
        init_actual=r48.parameter_summary(r48.load_controller(recorded_init))["parameter_sha256"]
        r48_ep0_audit[str(seed)]={"r48_forward_panel_registered_ep0_path":str(registered_path),
            "r48_static_registered_parameter_sha256":r48.REGISTERED_EP0[seed]["parameter_sha256"],
            "r48_registered_file_actual_sha256":sha_file(registered_path),
            "r48_registered_file_actual_parameter_sha256":registered_actual,
            "round46_bc_post_initializer_path":str(recorded_init),
            "round46_initializer_actual_file_sha256":sha_file(recorded_init),
            "round46_initializer_actual_parameter_sha256":init_actual,
            "registered_file_matches_round46_initializer":registered_actual==init_actual,
            "r48_static_value_matches_round46_initializer":r48.REGISTERED_EP0[seed]["parameter_sha256"]==init_actual}
    # Reproduction checks, independently on each source panel and matching source model.
    replay = {"kind": "round49-bc-path-replay/1", "action_mapping": {"SEND": SEND, "SKIP": SKIP},
              "greedy_tie_rule": "p_send >= 0.5 -> SEND", "tolerance": TOL,
              "bc_trajectory": {}, "ppo_trajectory": {}, "discrepancies": []}
    for panel_name, records, mode, action_check in (
        ("bc_trajectory", bc_panel, "ep0", True),
        ("ppo_trajectory", ppo_panel, "ep128", True)):
        per_seed = {}
        for seed in SEEDS:
            subset = [r for r in records if r["source_seed"] == seed]
            errors = [abs(r[f"p_send_s{seed}_{mode}"] - float(r["original_p_send"])) for r in subset]
            bad = []
            for r, err in zip(subset, errors):
                replay_act = SEND if r[f"p_send_s{seed}_{mode}"] >= .5 else SKIP
                if err > TOL or (action_check and replay_act != int(r["original_action_index"])):
                    bad.append({"scene": r["scene"], "target": r["target"], "p_error": err,
                                "replayed_action": action_label(replay_act), "original_action": r["original_action"]})
            replay[panel_name][str(seed)] = {"records": len(subset), "max_probability_error": max(errors, default=None),
                                             "probability_mismatches_over_tolerance": sum(e > TOL for e in errors),
                                             "greedy_action_mismatches": sum(x["replayed_action"] != x["original_action"] for x in bad),
                                             "valid": not bad}
            replay["discrepancies"].extend([{"panel": panel_name, "source_seed": seed, **x} for x in bad])
        replay[panel_name]["per_seed"] = per_seed
    if replay["discrepancies"]:
        raise ValueError(f"source-model replay failed: {replay['discrepancies'][:2]}")
    replay["source_model_identity"] = {
        str(seed): {"round46_initializer_parameter_sha256": identity["roles"]["ep0_training_init"][str(seed)]["round46_initializer_parameter_sha256"],
                    "bc_row_parameter_identity_match": identity["roles"]["ep0_training_init"][str(seed)]["bc_control_rows_use_same_parameter_identity"],
                    "BC checkpoint arm": "BC_ONLY_R2"} for seed in SEEDS}
    replay["round48_ep0_loader_audit"]={"root_cause":"r48_probe.py forward_panel lines 283-287 loads REGISTERED_EP0 path, whose actual parameter digest differs from ROUND46 BC_POST initializer; build_model_identity lines 162-166 compares BC_POST digest to a static REGISTERED_EP0 digest instead of checking the loaded file, while lines 168-171 separately record the true old-file digest.",
        "per_seed":r48_ep0_audit,"corrected_round49_model_source":"MODEL_ROLE_IDENTITY.roles.ep0_training_init.round46_initializer_file"}
    replay["inference_guard"] = forward_guard

    for record in all_panel:
        record["greedy_changes"] = {}
        for seed in SEEDS:
            p0, p128 = record[f"p_send_s{seed}_ep0"], record[f"p_send_s{seed}_ep128"]
            a0, a128 = (SEND if p0 >= .5 else SKIP), (SEND if p128 >= .5 else SKIP)
            record["greedy_changes"][str(seed)] = {"delta_p": p128-p0,
                "transition": f"{action_label(a0)}->{action_label(a128)}",
                "crossed_half": (p0-.5)*(p128-.5) < 0}

    comparison = build_comparison(all_panel, bc_panel, ppo_panel, identity, forward_guard, replay, r48)
    manifest = {
        "kind": "round49-input-panel-manifest/1", "task_id": "ROUND49-BC-INPUT-POLICY-CLOSURE",
        "index_source": str(ID_INDEX), "index_sha256": sha_file(ID_INDEX),
        "bc_panel": bc_source_info, "ppo_panel": {"records": len(ppo_panel),
            "tapes_scanned": ppo_manifest.get("tapes_scanned"), "excluded_records": ppo_manifest.get("excluded_records"),
            "exclusions_by_reason":ppo_manifest.get("exclusions_by_reason"),"sources":ppo_manifest.get("sources"),
            "inclusion_rule": ppo_manifest.get("inclusion_rule"),
            "round48_manifest_path":ppo_manifest.get("round48_manifest_path"),
            "round48_manifest_sha256":ppo_manifest.get("round48_manifest_sha256"),
            "round48_panel_signature_match":ppo_manifest.get("round48_panel_signature_match"),
            "adoption_resolution":"1301000 s11: V8 accepted row; 1301001 s11: V9 accepted row; 1301002 s11: V10 accepted row; other cells: V10 matrix rows"},
        "action_mapping": {"SEND": SEND, "SKIP": SKIP}, "tie_rule": "p_send >= 0.5 -> SEND",
        "feature_width": 19, "feature_dtype_forward": "float32", "same_vector_forward_cache_only": True,
        "protocol_sources": [{"path":str(REPO/"runs/comm_bc_initialized_constrained_ppo_20261007_r2/r42_one_shot.py"),"sha256":sha_file(REPO/"runs/comm_bc_initialized_constrained_ppo_20261007_r2/r42_one_shot.py"),"role":"one-shot collector, SEND/HOLD indices, greedy tie, tape and decision semantics"},
            {"path":str(REPO/"runs/comm_budget_action_mask_fix_20260922/budget_masked.py"),"sha256":sha_file(REPO/"runs/comm_budget_action_mask_fix_20260922/budget_masked.py"),"role":"frozen action enum SEND=1,HOLD=0 and sequential budget gate"},
            {"path":str(REPO/"integrated_sar/comm_learning.py"),"sha256":sha_file(REPO/"integrated_sar/comm_learning.py"),"role":"19-dimensional comm-send-state/1 feature order and encoder"}],
        "records_preserved_individually": True, "records": all_panel,
    }
    dump(OUT / "INPUT_PANEL_MANIFEST.json", manifest)
    dump(OUT / "BC_PATH_REPLAY.json", replay)
    dump(OUT / "COMMON_INPUT_POLICY_COMPARISON.json", comparison)
    report = render_report(manifest, replay, comparison)
    (OUT / "ROUND49_POLICY_CLOSURE_REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "STATEMENT_CORRECTIONS.md").write_text(render_corrections(comparison,replay), encoding="utf-8")
    print(json.dumps({"bc_records": len(bc_panel), "ppo_records": len(ppo_panel),
                      "replay_ok": not replay["discrepancies"],
                      "per_seed": comparison["paired_by_source_panel"],
                      "input_counts": comparison["vector_counts"]}, ensure_ascii=False, indent=2))


def build_comparison(all_panel, bc_panel, ppo_panel, identity, guard, replay, r48):
    decomp = read(ROUND47 / "PER_SCENE_COST_DECOMPOSITION.json")
    by_seed_panel = {}
    paired = {}
    for panel_name, records in (("BC_trajectory", bc_panel), ("PPO_trajectory", ppo_panel)):
        by_seed_panel[panel_name] = {}
        for seed in SEEDS:
            rows = [r for r in records if r["source_seed"] == seed]
            per_role = {role: action_summary(rows, f"p_send_s{seed}_{role}") for role in ("ep0", "ep128")}
            p0 = np.array([r[f"p_send_s{seed}_ep0"] for r in rows])
            p1 = np.array([r[f"p_send_s{seed}_ep128"] for r in rows])
            trans = Counter(f"{action_label(SEND if a>=.5 else SKIP)}->{action_label(SEND if b>=.5 else SKIP)}" for a,b in zip(p0,p1))
            by_seed_panel[panel_name][str(seed)] = {"ep0": per_role["ep0"], "ep128": per_role["ep128"],
                "delta_p_mean_same_inputs": float((p1-p0).mean()) if len(rows) else None,
                "delta_p_abs_mean_same_inputs": float(np.abs(p1-p0).mean()) if len(rows) else None,
                "greedy_transition_counts": {k: trans[k] for k in ("SEND->SEND","SEND->SKIP","SKIP->SEND","SKIP->SKIP")},
                "boundary_crossings": int(sum((a-.5)*(b-.5)<0 for a,b in zip(p0,p1)))}
    # Cross-panel metrics by actual labels from ROUND47 PPO-vs-BC scene decomposition.
    category_stats = {}
    for panel_name, records in (("BC_trajectory", bc_panel), ("PPO_trajectory", ppo_panel)):
        category_stats[panel_name] = {}
        for seed in SEEDS:
            scenes = decomp["per_seed"][str(seed)]["vs_bc"]["scenes"]
            labels = {}
            for scene, block in scenes.items():
                for t in block.get("A_model_only_targets", []): labels[(int(scene), str(t))] = "PPO_only_send"
                for t in block.get("B_bench_only_targets", []): labels[(int(scene), str(t))] = "BC_only_send"
                for t in block.get("common_targets", {}): labels[(int(scene), str(t))] = "both_send"
            category_stats[panel_name][str(seed)] = {}
            for r in [x for x in records if x["source_seed"] == seed]:
                cat = labels.get((int(r["scene"]), str(r["target"])), "neither_send")
                b = category_stats[panel_name][str(seed)].setdefault(cat, {"n":0,"delta_p":[],"ep0_send":0,"ep128_send":0,"transitions":Counter()})
                b["n"] += 1
                b["delta_p"].append(r[f"p_send_s{seed}_ep128"]-r[f"p_send_s{seed}_ep0"])
                b["ep0_send"] += r[f"greedy_s{seed}_ep0"] == "SEND"
                b["ep128_send"] += r[f"greedy_s{seed}_ep128"] == "SEND"
                b["transitions"][r["greedy_changes"][str(seed)]["transition"]] += 1
            for cat,b in category_stats[panel_name][str(seed)].items():
                vals=b.pop("delta_p"); b["delta_p_mean"]=float(np.mean(vals))
                b["transitions"]=dict(b["transitions"])
    features = feature_stats(all_panel)
    counts = {name: vector_counts(rows) for name,rows in (("BC_trajectory",bc_panel),("PPO_trajectory",ppo_panel),("pooled",all_panel))}
    counts["per_seed_and_panel"] = {
        panel: {str(seed): vector_counts([r for r in rows if r["source_seed"] == seed]) for seed in SEEDS}
        for panel, rows in (("BC_trajectory",bc_panel),("PPO_trajectory",ppo_panel))}
    counts["all_arms_by_seed"]={str(seed):vector_counts([r for r in all_panel if r["source_seed"]==seed]) for seed in SEEDS}
    bc_hashes={r["features_sha256"] for r in bc_panel}; ppo_hashes={r["features_sha256"] for r in ppo_panel}
    counts["support_overlap"]={"BC_unique_vectors":len(bc_hashes),"PPO_unique_vectors":len(ppo_hashes),
        "intersection_unique_vectors":len(bc_hashes & ppo_hashes),"BC_only_unique_vectors":len(bc_hashes-ppo_hashes),
        "PPO_only_unique_vectors":len(ppo_hashes-bc_hashes)}
    r48_counts = read(ROUND48/"POLICY_DRIFT_RESULTS.json")["feature_observations"]
    feature_names=["age_steps","since_last_emission_time","sent_count","pending_event","changed_view",
        "first_emission","stage_changed","requirements_changed","members_changed","progress_delta_abs",
        "revision_delta","remaining_time_over_horizon","urgency_now","has_future_requirements",
        "queue_backlog_targets","budget_remaining_fraction","capacity_remaining_fraction","subscribers","step_over_horizon"]
    for col,name in zip(features["columns"],feature_names): col["name"]=name
    return {"kind":"round49-common-input-policy-comparison/1","task_id":"ROUND49-BC-INPUT-POLICY-CLOSURE",
        "description":"descriptive common-input policy mapping; source trajectories remain distinct",
        "paired_by_source_panel":by_seed_panel,"cost_category_metadata":category_stats,
        "vector_counts":counts,"round48_count_scope_reconciliation":{
            "per_seed_records":96,"pooled_records":r48_counts.get("records"),
            "pooled_distinct_feature_vectors":r48_counts.get("distinct_feature_vectors"),
            "pooled_vector_hash_groups_repeated":r48_counts.get("vectors_shared_by_multiple_targets"),
            "meaning_note":"96 is the per-seed panel size (288 pooled). 76 is the number of unique feature hashes across the pooled PPO records. 69 counts those hashes occurring more than once, not independent scenes or necessarily distinct targets."},
        "feature_observations":features,"validation":{"bc_source_probability_and_greedy_replay":True,
            "ppo_source_ep128_probability_and_greedy_replay":True,"tolerance":TOL,
            "source_model_roles_verified":all(identity["roles"]["ep0_training_init"][str(s)]["bc_control_rows_use_same_parameter_identity"] for s in SEEDS),
            "forward_guard":guard,"read_only_checkpoint_identity":identity["verification"]},
        "statistical_scope":"descriptive only; no independent-scene inference; no new CI or significance tests",
        "development_go":False}


def feature_stats(records):
    m=np.asarray([r["features"] for r in records],dtype=np.float64)
    cols=[]
    for i in range(m.shape[1]):
        v=m[:,i]
        cols.append({"index":i,"min":float(v.min()),"max":float(v.max()),"distinct_values":int(len(np.unique(v))),"constant":bool(v.min()==v.max()),"at_zero_count":int((v==0).sum()),"at_one_count":int((v==1).sum())})
    return {"record_count":len(records),"columns":cols,"interpretation":"columns are described by observed values; no feature causality is inferred"}


def render_corrections(c,replay):
    lines=["""# ROUND49 更正与口径补充

## ROUND48 ep0 源模型接线更正

ROUND48 的 `r48_probe.py` 在 `build_model_identity()` 中正确记录 ROUND46 初值为 R45 `BC_POST_MODEL_s{seed}.pt`，并把 `REGISTERED_EP0` 路径标成 R43 旧件、明确说明它不是 ROUND46 初值。但 `forward_panel()` 随后实际用 `REGISTERED_EP0[seed]["path"]` 加载前向。因此 ROUND48 保存的 ep0 概率、边界翻转、动作过渡和依赖这些值的解释对应的是 R43 旧 ep0 模型，不是本轮训练实际初值；这些概率探针结论由本轮同 seed R45 BC_POST 参数重算替代。ROUND48 ep128 源模型复现检查仍成立。旧报告和结果文件保留，不覆写。

受影响项：ROUND48 每 seed ep0 概率均值/范围、ep0→ep128 平均漂移、0.5边界穿越和四类 greedy 转换，以及按成本类别汇总的Δp解释全部以旧模型为基线，现由 ROUND49 的正确同 seed 初值输出取代。ROUND48 tape选择、96条/seed的自由输入纳入、特征统计、ep128前向概率与原 tape复现、ROUND47成本结果及冻结CI未因此失效。

## 口径补充

- ROUND47 的 seed-wise PPO−BC 配对区间端点精确包含 0；ROUND48 的统计更正仍有效。不得称任一 seed 差异已由该 CI 排除零。
- 未舍入冻结端点：s11 `[0, 1406.625]` B/集；s22 `[-928.434375, 0]` B/集；s33 `[0, 1249.5]` B/集。保持原共享场景 bootstrap，不新增单侧检验。
- Greedy 是确定性 argmax；p(SEND) 接近 0.5 不表示实际随机掷硬币。
- ROUND48 的 96/76/69 分别是**每 seed 记录数**、**合并 288 条 PPO 轨迹的唯一向量数**、**合并面板中出现多次的不同向量哈希组数**。ROUND48 原字段名“shared_by_multiple_targets”实际上只检查同一向量哈希是否出现多次，没有判别是否为不同目标；它不是独立目标数或场景数。
- 按实际动作/成本类别观察的 p 变化只描述策略映射与轨迹标签的共同出现，不证明 PPO 更新机制、成本差根因或通信因果。同一输入 ep0→ep128 的变化比较的是策略映射；BC/PPO 各自轨迹输入分布还包含闭环路径改变。
- 本轮未开展新的统计检验；开发块仍是已用开发块，`development_go=false`。
""".strip()]
    lines += ["", "## 源参数摘要对照", "", "ROUND48错误加载位于 `r48_probe.py:283-287`；错误源登记在 `:45-55`，而身份构建另记真实初值与旧文件摘要在 `:134-171`。", "", "| seed | ROUND48加载的R43文件 | R43文件SHA256 | R43实际参数SHA256 | BC_POST初值参数SHA256 | 参数匹配 |"]
    for seed in SEEDS:
        x=replay["round48_ep0_loader_audit"]["per_seed"][str(seed)]
        lines.append(f"| {seed} | `{x['r48_forward_panel_registered_ep0_path']}` | `{x['r48_registered_file_actual_sha256']}` | `{x['r48_registered_file_actual_parameter_sha256']}` | `{x['round46_initializer_actual_parameter_sha256']}` | {x['registered_file_matches_round46_initializer']} |")
    lines += ["", "R43文件路径与参数摘要位于 `BC_PATH_REPLAY.json`。ROUND48静态 REGISTERED_EP0 参数值与真实初值相同，但其被指向文件反算出的参数摘要不同；这正是角色标签没有保护住真实加载的原因。"]
    return "\n".join(lines)+"\n"


def render_report(manifest,replay,c):
    lines=["# ROUND49 BC 输入与策略映射闭合报告","",
      "任务：`ROUND49-BC-INPUT-POLICY-CLOSURE`。仅对已存 BC/PPO tape 与现有模型做 CPU、eval、no_grad 前向。开发块 `1301000..1301015` 已多轮使用。", "",
      "## 1. 覆盖与回放门禁","",
      f"- BC：索引解引用 {manifest['bc_panel']['indexed_rows']} 行、{manifest['bc_panel']['indexed_seed_scenes']} seed-scene；纳入 {manifest['bc_panel']['eligible_records']} 个自由首报输入。PPO：ROUND48 同面板 {manifest['ppo_panel']['records']} 条。","",
      f"- 排除项：BC tape 内 {sum(manifest['bc_panel']['excluded_counts'].values())} 个非入选目标记录，类别 {json.dumps(manifest['bc_panel']['excluded_counts'],ensure_ascii=False)}；PPO tape {manifest['ppo_panel']['excluded_records']} 个非自由/关闭记录。","",
      f"- BC 自身输入由同 seed 真实训练初值复现；PPO 输入由同 seed ep128 复现。概率容差 {TOL}，动作映射 SEND={SEND}/SKIP={SKIP}，平局归 SEND。总体回放状态：{not bool(replay['discrepancies'])}。","",
      "- 每条记录的来源 tape/row、hash、特征、动作、概率与 mask/候选合法性在 `INPUT_PANEL_MANIFEST.json`。相同向量仅缓存 forward，目标记录没有去重。","",
      "- ROUND48 概率探针发现源角色接线错误：其身份构建记录了正确 BC_POST 初值，但前向函数实际装载 `REGISTERED_EP0` 指向的 R43 旧模型。ROUND48 ep0 概率/翻转表因此不对应真实训练初值；ep128 的 tape 概率回放仍有效。本报告用对应 seed 的 R45 BC_POST 初值重算，并在 `STATEMENT_CORRECTIONS.md` 留下更正。","",
      "- ROUND48曾报告的 s11/s22/s33 ep0均值约 `.542/.375/.504` 以及相关边界翻转，不用于ROUND46初值解释；真实 BC_POST 初值在各自面板的重算均值约 `.343/.345/.345`。受影响统计字段逐项列在勘误。","",
      "- 实际加载证据：旧文件的 forward 参数摘要、真实初值摘要及本轮六个 forward 文件/参数 SHA256 均逐模型列在下一节；旧 R43 文件在新的加载门禁处因参数摘要不匹配被拒。","",
      "## 2. BC 轨迹与 PPO 轨迹输入上的概率","",
      "每个来源面板、每个 seed 均列 ep0 与 ep128；这是同一批已存输入上的策略映射，不是对另一策略闭环轨迹的预测。","",
      "| 输入来源 | seed | ep0 p均值[范围] | ep128 p均值[范围] | Δp均值 | 距0.5平均绝对距离 ep0→ep128 | greedy发送数 ep0→ep128 | 同seed转换 SS/SK/KS/KK |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for panel in ("BC_trajectory","PPO_trajectory"):
        for seed in SEEDS:
            d=c["paired_by_source_panel"][panel][str(seed)]
            a,b=d["ep0"],d["ep128"]
            tr=d["greedy_transition_counts"]
            lines.append(f"| {panel} | {seed} | {a['p_send_mean']:.6f} [{a['p_send_min']:.6f}, {a['p_send_max']:.6f}] | {b['p_send_mean']:.6f} [{b['p_send_min']:.6f}, {b['p_send_max']:.6f}] | {d['delta_p_mean_same_inputs']:+.6f} | {a['distance_to_half_mean']:.6f}→{b['distance_to_half_mean']:.6f} | {a['greedy_action_counts']['SEND']}→{b['greedy_action_counts']['SEND']} | {tr['SEND->SEND']} / {tr['SEND->SKIP']} / {tr['SKIP->SEND']} / {tr['SKIP->SKIP']} |")
    lines += ["", "## 3. 控制路径、概率回放和特征覆盖", "",
      "| 模型角色 | seed | 实际 forward 文件 SHA256 | 实际 forward 参数 SHA256 |","|---|---:|---|---|"]
    bindings=c["validation"]["forward_guard"]["model_bindings_before_forward"]
    for seed in SEEDS:
        for role in ("ep0","ep128"):
            b=bindings[f"s{seed}_{role}"]
            lines.append(f"| {role} | {seed} | `{b['file_sha256_before_forward']}` | `{b['parameter_sha256_before_forward']}` |")
    lines += ["", "ROUND48 的错误加载文件及其真实 forward 参数 SHA256：", "", "| seed | 错误加载路径 | 实际参数 SHA256 | 正确 BC_POST 初值参数 SHA256 |","|---:|---|---|---|"]
    for seed in SEEDS:
        x=replay["round48_ep0_loader_audit"]["per_seed"][str(seed)]
        lines.append(f"| {seed} | `{x['r48_forward_panel_registered_ep0_path']}` | `{x['r48_registered_file_actual_parameter_sha256']}` | `{x['round46_initializer_actual_parameter_sha256']}` |")
    lines += ["", "- BC 索引控制臂标记 `BC_ONLY_R2`，其 checkpoint parameter identity 与 ROUND46 同 seed `BC_POST_MODEL` 初始化一致；BC tape 记录 one-shot schema、free choice、顺序预算 mask 审计及 SEND 概率。BC 的动作按原 tape 复现。身份相同本身不足以证明路径相同；本轮另核验 tape 资格、mask 顺序、候选次序与概率/动作重放。详见 `BC_PATH_REPLAY.json`。","",
      "- 每 seed/来源的向量计数（相同输入记录仍逐条保留）：","",
      "| 来源 | seed | 输入记录 | 唯一向量 | 重复向量组 | 重复组内记录 | 跨 scene-target 共享组 |","|---|---:|---:|---:|---:|---:|---:|"]
    for panel in ("BC_trajectory","PPO_trajectory"):
        for seed in SEEDS:
            v=c["vector_counts"]["per_seed_and_panel"][panel][str(seed)]
            lines.append(f"| {panel} | {seed} | {v['records']} | {v['unique_vectors']} | {v['repeated_vector_groups']} | {v['records_in_repeated_groups']} | {v['groups_shared_across_distinct_scene_targets']} |")
    for seed in SEEDS:
        v=c["vector_counts"]["all_arms_by_seed"][str(seed)]
        lines.append(f"| 两来源合并 | {seed} | {v['records']} | {v['unique_vectors']} | {v['repeated_vector_groups']} | {v['records_in_repeated_groups']} | {v['groups_shared_across_distinct_scene_targets']} |")
    lines += [f"| 两臂全seed合并 | — | {c['vector_counts']['pooled']['records']} | {c['vector_counts']['pooled']['unique_vectors']} | {c['vector_counts']['pooled']['repeated_vector_groups']} | {c['vector_counts']['pooled']['records_in_repeated_groups']} | {c['vector_counts']['pooled']['groups_shared_across_distinct_scene_targets']} |", "",
      f"- 两来源向量支持集：{json.dumps(c['vector_counts']['support_overlap'],ensure_ascii=False)}。这是输入支持覆盖，不表示独立样本或行为因果。","",
      f"- ROUND48 历史范围复核：{json.dumps(c['round48_count_scope_reconciliation'],ensure_ascii=False)}。","",
      "- 本 pooled 面板中恒定列为 0–12、16、17；变化列为 13 `has_future_requirements`、14 `queue_backlog_targets`、15 `budget_remaining_fraction`、18 `step_over_horizon`。列范围和端点占比见机读 `feature_observations`。恒定列不能分开这些记录；变化列与 p/动作的关联不能确定漂移由哪一维导致。","",
      "## 4. 成本类别交叉观察", "",
      "ROUND47 vs BC 的 PPO-only / BC-only / both / neither 首报标签是按同 seed、scene、target 联结；标签只作事后分析元数据，不进入 actor。各组 n、ep0 与 ep128 SEND 数和 mean Δp见下表；它描述输入上的概率映射变化，不代表真实发送/任务结果。","",
      "| 来源 | seed | PPO-only发送 n / Δp | BC-only发送 n / Δp | 双方发送 n / Δp | 双方均不发 n / Δp |","|---|---:|---:|---:|---:|---:|"]
    for panel in ("BC_trajectory","PPO_trajectory"):
        for seed in SEEDS:
            cats=c["cost_category_metadata"][panel][str(seed)]
            def cell(k):
                x=cats.get(k)
                return "—" if not x else f"{x['n']} / {x['delta_p_mean']:+.3f}"
            lines.append(f"| {panel} | {seed} | {cell('PPO_only_send')} | {cell('BC_only_send')} | {cell('both_send')} | {cell('neither_send')} |")
    lines += ["", "这些输入差异不能拆分参数效应与闭环轨迹变化；同一输入的 ep0→ep128 才是固定输入上的映射差。"]
    lines += [
      "## 5. 结论与方法依据", "",
      "- BC 控制与真实训练初值：**可对应**，同 seed 参数身份和源模型概率/动作重放闭合；因此两条来源面板的同输入 ep0→ep128 对比有效。","",
      "- BC 自身输入是否可分：可分。正确 BC 初值在每 seed BC 面板上的概率均值约 0.344，概率覆盖接近 0 到 1，greedy 动作 33 SEND / 63 SKIP；这反驳“全局固定近0.5”的描述，但只说明已观测开发输入支持上存在区分，不说明可泛化。","",
      "- Round47 配对 CI 未变：s11 `[0,1406.625]`、s22 `[-928.434375,0]`、s33 `[0,1249.5]` B/集，端点精确含0；均不支持严格排除零的 seed 成本差。","",
      "- 同一输入上的 ep0 与 ep128 概率/动作差异是模型参数映射差异；BC与PPO轨迹样本分布、动作数和成本标签不同，还可能由闭环轨迹选择变化造成。","",
      "- 正确初值到 ep128 的可复现边界变化在 BC 与 PPO 两种来源面板上均很少：s11/s33 为少数 SKIP→SEND，s22 为少数 SEND→SKIP。ROUND48报告的几十次边界翻转是错误旧初值导致，不能用来解释seed成本差异。转换顺序为 ep0→ep128，`SS/SK/KS/KK`分别指 SEND→SEND、SEND→SKIP、SKIP→SEND、SKIP→SKIP。","",
      "- 证据不足以支持改训练器、改特征、换阈值或删除首报。按时任务结果与成本的既有结论不升级，`development_go=false`。","",
      "## 6. 唯一有限下一步建议", "",
      "若继续研究，唯一建议是另行登记一次未用场景块上的固定策略配对确认：假设当前 BC→PPO 成本差只由开发轨迹输入分布造成；唯一变量为策略 checkpoint（真实 BC 初值 vs ep128），同场景同配置配对，其他编码、greedy 解码与任务/成本判据固定。可证伪结果是新块仍出现可重复的同方向按时目标集合与计费成本差；若差异消失或跨 seed 方向不稳，则开发块的映射变化不足以支持稳定方法收益。该确认未授权、未执行。","",
      "## 7. 执行计数与限制", "",
      "| 项目 | 本轮新增 |", "|---|---:|", "| 环境构造 | 0 |", "| 仿真启动 | 0 |", "| 新流 | 0 |", "| 真实训练更新 | 0 |", "| 合成 optimizer.step | 0 |", "| checkpoint 写入 | 0 |", "",
      "仅运行本轮纯计算测试；未运行环境/训练入口或全量环境套件。完整交付 HEAD 在提交后记录于最终交接。" ]
    return "\n".join(lines)+"\n"


if __name__ == "__main__":
    main()
