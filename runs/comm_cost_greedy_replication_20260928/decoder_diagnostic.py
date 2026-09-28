"""ROUND25 §2 零启动诊断：sampled/greedy 解码、观测、顺序 mask、概率与费用路径（0 新环境启动）。

只用 ROUND24 的落盘件（ep128 checkpoint、eval tape、transcript、rows）与冻结代码身份：
1. 同一 checkpoint 身份：两个模式都用 seed 的 ep128 里程碑；从保存的动作前特征重算
   p(SEND) 与 joint log_prob，与 tape 记录比对（≤1e-6）。
2. 解码核验：greedy 行动作必须等于 argmax（p(SEND)>=p(HOLD) 时 SEND，平局归 SEND——
   旧冻结解码的确切 tie-break）；sampled 行不改写概率，只记录动作与其似然。
3. 顺序 mask 核验：free/masked 与 send_executable_initial/block_reason_initial/
   candidate_shadow 的一致性；按 tape 目标顺序用 shadow 信用重放，复现每个候选的
   budget_after；masked 候选不进入 likelihood/entropy 统计。
4. 自由候选统计：p(SEND)、二元熵、距 0.5 边界的 margin，按首发/更新/重试与 mode 分层
   （emission class 由 transcript 的 send/refused 历史推导，只读）。
5. 同输入前缀对照：每场景两模式的**第一条** tape（动作前状态确实相同）逐行对照；
   其后状态分叉，不做逐行比较；整集分布分开描述。
6. 费用路径：逐条 send 的 cost_bytes == payload_bytes × subscribers；Σ == spent_bytes；
   拒付单列且计 0；shadow 预算不为负。
7. 回读 ROUND24 逐场景原始表：seed33 greedy 对 immediate / late 的 dY/dC、16/16 任务平局、
   两个“更差”场景的具体指标（任务/字节/联合占优分开）。

输出 `DECODER_DIAGNOSTIC.json` / `DECODER_DIAGNOSTIC.md`（本目录）。不修改任何阈值、
不训练、不启动环境；发现身份/观测/解码/mask/计费错误即置 passed=false 并停止后续评估。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R24 = ROOT / "runs" / "comm_cost_v2_rerun_20260928"

sys.path.insert(0, str(R24))
TOL = 1e-6


def sha256(path: Path) -> str:
    import hashlib

    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> int:
    import v2_common as common
    import v2_trainer as trainer
    from onpolicy import joint_log_prob
    from v2_controller import fresh_controller

    manifest = json.loads((R24 / "CONFIG_AND_MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    freeze = {"freeze_sha256": manifest["code_identity"]["freeze_sha256"],
              "config_hash": manifest["configuration_hash"],
              "branch_start": manifest["branch_start"],
              "head_at_freeze": manifest["head_at_freeze"]}
    index = json.loads((R24 / "CHECKPOINT_INDEX.json").read_text(encoding="utf-8"))

    payload: dict[str, Any] = {
        "round": "25", "section": "2 decoder diagnostic (zero environment launches)",
        "frozen_code_identity": manifest["code_identity"]["freeze_sha256"],
        "checkpoint_files": {seed: index[seed]["milestones"]["128"] for seed in ("11", "22", "33")},
        "decode_rule": "greedy: SEND iff p(SEND) >= p(HOLD) (ties to SEND, the archived "
                       "collector's exact branch); sampled: torch.multinomial over the same "
                       "probabilities; masked candidates never reach either decoder and carry "
                       "no likelihood",
    }

    problems: list[str] = []
    per_seed: dict[str, Any] = {}

    for seed in (11, 22, 33):
        controller, _optimizer, state = trainer.load_checkpoint(
            R24 / "checkpoints" / f"v2_train_s{seed}_ep128.pt", freeze=freeze)
        assert int(state["episodes_done"]) == 128
        digest_now = sha256(R24 / "checkpoints" / f"v2_train_s{seed}_ep128.pt")
        if digest_now != index[str(seed)]["milestones"]["128"]["sha256"]:
            problems.append(f"seed {seed}: ep128 checkpoint file changed since the index was written")
        fresh_like = fresh_controller(seed)   # only to count modules; not a weight comparison

        modes = {}
        for block in ("final_greedy", "final_sampled"):
            rows_stats = {
                "scenes": 0, "candidates": 0, "free": 0, "masked_budget": 0, "masked_capacity": 0,
                "probability_max_abs_gap": 0.0, "log_prob_max_abs_gap": 0.0,
                "greedy_argmax_violations": 0, "free_set_violations": 0,
                "shadow_walk_violations": 0, "masked_likelihood_leaks": 0,
                "schema_violations": 0,
                "free_p_send": [], "free_entropy": [], "free_margin": [],
                "margin_bands": {"<=0.01": 0, "<=0.05": 0, "<=0.10": 0},
                "masked_reason_crosstab": {},
                "argmax_send_share": 0,
                "emission_class": {"first": [], "refresh": [], "retry": []},
                "episodes": [],
            }
            for scenario in common.DEVELOPMENT_SCENARIOS:
                tape_path = R24 / "logs" / f"v2_eval_s{seed}_{block}_s{scenario}_a0_tape.jsonl"
                row_path = R24 / "rows" / f"v2_eval_s{seed}_{block}_s{scenario}_a0.json"
                ct_path = R24 / "logs" / f"v2_eval_s{seed}_{block}_s{scenario}_a0_ct.jsonl"
                if not tape_path.exists() or not row_path.exists():
                    problems.append(f"seed {seed} {block} s{scenario}: missing tape or row")
                    continue
                tapes = load_jsonl(tape_path)
                row = json.loads(row_path.read_text(encoding="utf-8"))
                transcript = load_jsonl(ct_path) if ct_path.exists() else []
                sends = [r for r in transcript if r.get("kind") == "send"]
                refused = [r for r in transcript if r.get("kind") == "refused"]

                # emission history per target (from the transcript, read-only)
                emitted_before: dict[str, list[int]] = {}
                refused_before: dict[str, list[int]] = {}
                for record in sends:
                    emitted_before.setdefault(record["target"], []).append(int(record["step"]))
                for record in refused:
                    refused_before.setdefault(record["target"], []).append(int(record["step"]))

                episode_free_p: list[float] = []
                episode_argmax_send = 0
                episode_emissions = 0
                for tape in tapes:
                    step = int(tape["step"])
                    if tape.get("schema") != "onpolicy-tape/2":
                        rows_stats["schema_violations"] += 1
                    if tape.get("action_mask_schema") != "budget-sequential-mask/1":
                        rows_stats["schema_violations"] += 1
                    matrix = torch.as_tensor(np.asarray(tape["features"], dtype=np.float32))
                    with torch.no_grad():
                        logits, _values = controller.forward(matrix)
                    probabilities = torch.softmax(logits, dim=-1)
                    log_probabilities = torch.log_softmax(logits, dim=-1)
                    targets = list(tape["targets"])
                    index_of = {target: position for position, target in enumerate(targets)}
                    # 1. per-candidate probability recomputation
                    for target, recorded in tape["send_probabilities"].items():
                        position = index_of[target]
                        gap = abs(float(probabilities[position, 1]) - float(recorded))
                        rows_stats["probability_max_abs_gap"] = max(
                            rows_stats["probability_max_abs_gap"], gap)
                    # 2. joint log_prob over the actionable (free) set only
                    picked = torch.tensor([tape["actions"][t] for t in targets], dtype=torch.long)
                    usable = torch.tensor([1.0 if t in tape["free"] else 0.0 for t in targets],
                                          dtype=torch.float32)
                    recomputed = float(joint_log_prob(logits, picked, usable))
                    rows_stats["log_prob_max_abs_gap"] = max(
                        rows_stats["log_prob_max_abs_gap"], abs(recomputed - float(tape["log_prob"])))
                    # masked candidates must not carry likelihood: their action is HOLD and
                    # they are excluded from `free`, so a leak would show as a nonzero term.
                    for target in tape["masked"]:
                        if target in tape["free"]:
                            rows_stats["masked_likelihood_leaks"] += 1
                    # 3. free/masked consistency: `free` is the set the *sequential* shadow
                    #    honours (honours_this_candidate), not the set initially executable --
                    #    an affordable candidate can become masked after earlier candidates in
                    #    the same step spent the credit, which is the whole point of the
                    #    sequential mask. The initial flags are recorded but are the
                    #    observation-side metadata, not the effective decision.
                    for target in targets:
                        shadow = tape["candidate_shadow"].get(target) or {}
                        in_free = target in tape["free"]
                        if in_free != bool(shadow.get("honours_this_candidate")):
                            rows_stats["free_set_violations"] += 1
                        if in_free and not bool(tape["send_executable_initial"].get(target)):
                            rows_stats["free_set_violations"] += 1
                    for target, why in tape["masked_reasons"].items():
                        if why == "budget_exhausted":
                            rows_stats["masked_budget"] += 1
                        elif why == "capacity_step":
                            rows_stats["masked_capacity"] += 1
                        # descriptive cross-tab: effective (sequential) reason vs the
                        # observation-side initial reason. A candidate can be initially
                        # unaffordable (the executor's own budget already ran out) or become
                        # unaffordable only after earlier candidates in the same step spent
                        # the credit; both are legal and are recorded, not judged.
                        key = f"{why}|{tape['send_block_reason_initial'].get(target)}"
                        rows_stats["masked_reason_crosstab"][key] = \
                            rows_stats["masked_reason_crosstab"].get(key, 0) + 1
                    # 4. greedy decode check
                    for target in tape["free"]:
                        position = index_of[target]
                        p_send = float(probabilities[position, 1])
                        entropy = float(-(p_send * np.log2(p_send + 1e-12)
                                          + (1 - p_send) * np.log2(1 - p_send + 1e-12)))
                        margin = abs(p_send - 0.5)
                        rows_stats["free_p_send"].append(p_send)
                        rows_stats["free_entropy"].append(entropy)
                        rows_stats["free_margin"].append(margin)
                        rows_stats["margin_bands"]["<=0.01"] += int(margin <= 0.01)
                        rows_stats["margin_bands"]["<=0.05"] += int(margin <= 0.05)
                        rows_stats["margin_bands"]["<=0.10"] += int(margin <= 0.10)
                        episode_free_p.append(p_send)
                        if p_send >= 0.5:
                            episode_argmax_send += 1
                        if block == "final_greedy":
                            expected = 1 if p_send >= 0.5 else 0
                            if int(tape["actions"][target]) != expected:
                                rows_stats["greedy_argmax_violations"] += 1
                        # emission class from the transcript history before this step
                        if any(earlier < step for earlier in emitted_before.get(target, [])):
                            rows_stats["emission_class"]["refresh"].append(p_send)
                        elif any(earlier < step for earlier in refused_before.get(target, [])):
                            rows_stats["emission_class"]["retry"].append(p_send)
                        else:
                            rows_stats["emission_class"]["first"].append(p_send)
                    # 5. shadow walk replay (sequential budget mask)
                    budget = float(tape["shadow_budget_before"])
                    for target in tape["targets"]:
                        shadow = tape["candidate_shadow"].get(target)
                        if shadow is None:
                            continue
                        if int(tape["actions"].get(target, 0)) == 1 and target in tape["free"]:
                            budget -= int(tape["attempt_charge_bytes"][target])
                        if abs(budget - float(shadow["budget_after"])) > TOL:
                            rows_stats["shadow_walk_violations"] += 1
                    episode_emissions += sum(1 for t in targets
                                             if tape["requested"].get(t) and t in tape["free"]
                                             and int(tape["actions"][t]) == 1)
                # 6. cost path closure for this scene
                transcript_cost = sum(int(r["cost_bytes"]) for r in sends)
                per_send_mismatch = sum(
                    1 for r in sends
                    if int(r["cost_bytes"]) != int(r["payload_bytes"]) * int(r["subscribers"]))
                if transcript_cost != int(row["spent_bytes"]):
                    problems.append(f"seed {seed} {block} s{scenario}: transcript cost "
                                    f"{transcript_cost} != row {row['spent_bytes']}")
                if per_send_mismatch:
                    problems.append(f"seed {seed} {block} s{scenario}: {per_send_mismatch} sends "
                                    "with cost_bytes != payload_bytes x subscribers")
                refused_billed = sum(int(r.get("cost_bytes", 0) or 0) for r in refused)
                if refused_billed:
                    problems.append(f"seed {seed} {block} s{scenario}: refusals billed")
                min_shadow = min((float(t["shadow_budget_after"]) for t in tapes), default=0.0)
                if min_shadow < 0:
                    problems.append(f"seed {seed} {block} s{scenario}: shadow budget negative")
                rows_stats["scenes"] += 1
                rows_stats["candidates"] += len(tape["targets"]) if tapes else 0
                rows_stats["free"] += len(rows_stats["free_p_send"]) and 0  # placeholder
                rows_stats["episodes"].append({
                    "scenario": scenario, "emissions": row["emissions"],
                    "spent_bytes": row["spent_bytes"], "on_time": row["on_time"],
                    "policy_requested_sends": episode_emissions,
                    "mean_free_p_send": round(float(np.mean(episode_free_p)), 6)
                    if episode_free_p else None,
                    "free_argmax_send_share": round(episode_argmax_send
                                                    / max(1, len(episode_free_p)), 4)})
            rows_stats["free"] = len(rows_stats["free_p_send"])
            rows_stats["mean_free_p_send"] = round(float(np.mean(rows_stats["free_p_send"])), 6) \
                if rows_stats["free_p_send"] else None
            rows_stats["mean_free_entropy_bits"] = round(float(np.mean(rows_stats["free_entropy"])), 6) \
                if rows_stats["free_entropy"] else None
            rows_stats["margin_band_shares"] = {
                band: round(rows_stats["margin_bands"][band] / max(1, rows_stats["free"]), 4)
                for band in rows_stats["margin_bands"]}
            rows_stats["free_argmax_send_share"] = round(
                sum(1 for p in rows_stats["free_p_send"] if p >= 0.5)
                / max(1, rows_stats["free"]), 4)
            for klass, values in rows_stats["emission_class"].items():
                rows_stats["emission_class"][klass] = {
                    "n": len(values),
                    "mean_p_send": round(float(np.mean(values)), 6) if values else None,
                    "share_p_send_ge_0.5": round(sum(1 for v in values if v >= 0.5)
                                                 / max(1, len(values)), 4)}
            modes[block] = rows_stats
        per_seed[str(seed)] = {"ep128_sha256": digest_now, "modes": modes}

    payload["per_seed"] = per_seed

    # ---------------------------------------------------------------- same-input prefix compare (seed 33)
    controller, _optimizer, _state = trainer.load_checkpoint(
        R24 / "checkpoints" / "v2_train_s33_ep128.pt", freeze=freeze)
    prefix = []
    first_decision_diverged = 0
    for scenario in common.DEVELOPMENT_SCENARIOS:
        greedy_tape = load_jsonl(R24 / "logs" /
                                 f"v2_eval_s33_final_greedy_s{scenario}_a0_tape.jsonl")
        sampled_tape = load_jsonl(R24 / "logs" /
                                  f"v2_eval_s33_final_sampled_s{scenario}_a0_tape.jsonl")
        if not greedy_tape or not sampled_tape:
            continue
        g, s = greedy_tape[0], sampled_tape[0]
        same_features = g["features"] == s["features"]
        same_targets = g["targets"] == s["targets"]
        actions_differ = g["actions"] != s["actions"]
        if actions_differ:
            first_decision_diverged += 1
        matrix_g = torch.as_tensor(np.asarray(g["features"], dtype=np.float32))
        matrix_s = torch.as_tensor(np.asarray(s["features"], dtype=np.float32))
        with torch.no_grad():
            logits_g, _ = controller.forward(matrix_g)
            logits_s, _ = controller.forward(matrix_s)
        same_logits = bool(torch.equal(logits_g, logits_s))
        prefix.append({
            "scenario": scenario, "step": int(g["step"]),
            "same_pre_action_features": same_features, "same_candidate_set": same_targets,
            "same_forward_logits": same_logits,
            "greedy_actions": g["actions"], "sampled_actions": s["actions"],
            "first_decision_actions_differ": actions_differ,
            "recorded_probabilities_equal": g["send_probabilities"] == s["send_probabilities"]})
    payload["same_input_prefix_compare_seed33"] = {
        "scenes": len(prefix),
        "scenes_with_identical_first_decision_input": sum(
            1 for entry in prefix if entry["same_pre_action_features"]
            and entry["same_candidate_set"] and entry["same_forward_logits"]),
        "scenes_where_first_decision_actions_differ": first_decision_diverged,
        "per_scene": prefix,
        "note": "only the first tape row of an episode is a valid same-input comparison (same "
                "scenario, same initial state, no action taken yet); after any differing action "
                "the closed-loop states diverge, so no further line-by-line comparison is made",
    }

    # ---------------------------------------------------------------- ROUND24 per-scene re-read (seed 33)
    def row_of(block: str, scenario: int) -> dict[str, Any]:
        return json.loads((R24 / "rows" /
                           f"v2_eval_s33_{block}_s{scenario}_a0.json").read_text(encoding="utf-8"))

    references = {scenario: json.loads(
        (R24 / "rows" / f"v2_reference_s{scenario}_a1.json").read_text(encoding="utf-8"))
        for scenario in common.DEVELOPMENT_SCENARIOS}
    late = {scenario: json.loads(
        (R24 / "rows" / f"v2_late_discovery_gate_s{scenario}_a1.json").read_text(encoding="utf-8"))
        for scenario in common.DEVELOPMENT_SCENARIOS}
    per_scene = []
    for scenario in common.DEVELOPMENT_SCENARIOS:
        greedy = row_of("final_greedy", scenario)
        per_scene.append({
            "scenario": scenario,
            "greedy": {"on_time": greedy["on_time"], "spent_bytes": greedy["spent_bytes"]},
            "immediate": {"on_time": references[scenario]["on_time"],
                          "spent_bytes": references[scenario]["spent_bytes"]},
            "late": {"on_time": late[scenario]["on_time"],
                     "spent_bytes": late[scenario]["spent_bytes"]},
            "dY_vs_immediate": greedy["on_time"] - references[scenario]["on_time"],
            "dC_vs_immediate": greedy["spent_bytes"] - references[scenario]["spent_bytes"],
            "dY_vs_late": greedy["on_time"] - late[scenario]["on_time"],
            "dC_vs_late": greedy["spent_bytes"] - late[scenario]["spent_bytes"]})
    task_ties_immediate = sum(1 for e in per_scene if e["dY_vs_immediate"] == 0)
    task_ties_late = sum(1 for e in per_scene if e["dY_vs_late"] == 0)
    worse_vs_late = [e for e in per_scene if e["dC_vs_late"] > 0]
    payload["round24_seed33_reread"] = {
        "scenes": len(per_scene),
        "task_tie_scenes_vs_immediate": task_ties_immediate,
        "task_tie_scenes_vs_late": task_ties_late,
        "task_regressions_vs_immediate": [e["scenario"] for e in per_scene
                                          if e["dY_vs_immediate"] < 0],
        "task_regressions_vs_late": [e["scenario"] for e in per_scene if e["dY_vs_late"] < 0],
        "scenes_worse_than_late": [
            {"scenario": e["scenario"], "dY_vs_late": e["dY_vs_late"], "dC_vs_late": e["dC_vs_late"],
             "worse_metric": "bytes only" if e["dY_vs_late"] == 0 and e["dC_vs_late"] > 0
                             else ("task and bytes" if e["dY_vs_late"] < 0 and e["dC_vs_late"] > 0
                                   else "task only")}
            for e in worse_vs_late],
        "per_scene": per_scene,
    }

    for seed, entry in per_seed.items():
        for block, stats in entry["modes"].items():
            if stats["probability_max_abs_gap"] > TOL:
                problems.append(f"seed {seed} {block}: probability recomputation gap "
                                f"{stats['probability_max_abs_gap']:.2e} > {TOL}")
            if stats["log_prob_max_abs_gap"] > TOL:
                problems.append(f"seed {seed} {block}: log_prob recomputation gap "
                                f"{stats['log_prob_max_abs_gap']:.2e} > {TOL}")
            if stats["greedy_argmax_violations"]:
                problems.append(f"seed {seed} {block}: {stats['greedy_argmax_violations']} greedy "
                                "actions are not the argmax of the recomputed probabilities")
            if stats["free_set_violations"]:
                problems.append(f"seed {seed} {block}: {stats['free_set_violations']} free/masked "
                                "set inconsistencies")
            if stats["shadow_walk_violations"]:
                problems.append(f"seed {seed} {block}: {stats['shadow_walk_violations']} shadow "
                                "walk mismatches")
            if stats["masked_likelihood_leaks"]:
                problems.append(f"seed {seed} {block}: masked candidates inside the free set")
            if stats["schema_violations"]:
                problems.append(f"seed {seed} {block}: tape schema violations")
    if payload["same_input_prefix_compare_seed33"]["scenes_with_identical_first_decision_input"] \
            != payload["same_input_prefix_compare_seed33"]["scenes"]:
        problems.append("seed 33: some scenes' first-decision inputs are not identical across modes")

    payload["problems"] = problems
    payload["identity_or_correctness_blockers"] = problems
    payload["passed"] = not problems
    (HERE / "DECODER_DIAGNOSTIC.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "problems": problems,
                      "seed33_prefix": {k: v for k, v in
                                        payload["same_input_prefix_compare_seed33"].items()
                                        if k != "per_scene"},
                      "seed33_reread": {k: v for k, v in payload["round24_seed33_reread"].items()
                                        if k != "per_scene"}}, ensure_ascii=False, indent=1)[:2000])
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
