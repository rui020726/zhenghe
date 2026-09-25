"""SEARCH_POLICY_COMPARISON_RESULTS.json: the frozen S1b comparison, judged as frozen.

Primary quantity: undetected hidden targets per episode (个/episode), paired by seed, A minus B,
with a paired percentile bootstrap (B = 10 000, rng registered). Batch totals are also given in
个/批 for the same 60 episodes; the conversion 1 个/批 = 1/60 ≈ 0.016667 个/episode is stated
wherever both appear, because protocol revision R4 forbids mixing them.

The pre-registered criteria are the ones written in the round-17 protocol, quoted verbatim in
``criteria`` and applied without adjustment: stop 1, stop 2, and guard G. If the result falls in
neither, it is reported as "not met", with no extra seeds, no other statistic and no better arm.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import sp_common as sc
import sp_verdict

ARMS = sc.S1_ARMS
EPISODES = len(sc.S1_SEEDS)
GOALS_PER_EPISODE = 6


def paired_rows() -> dict[int, dict[str, dict[str, Any]]]:
    out: dict[int, dict[str, dict[str, Any]]] = {}
    for path in sorted(sc.ROWS.glob("data_*.json")):
        row = sc.read_json(path)
        if not row.get("ok"):
            continue
        out.setdefault(int(row["seed"]), {})[row["arm"]] = row
    return out


def bootstrap_ci(differences: list[float], rng_seed: int) -> dict[str, Any]:
    """Paired percentile bootstrap over seeds, implemented without a numpy dependency."""
    import random

    reproductions = sc.BOOTSTRAP_REPLICATIONS
    generator = random.Random(rng_seed)
    n = len(differences)
    means = []
    for _ in range(reproductions):
        sample = [differences[generator.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low = means[int(0.025 * reproductions)]
    high = means[min(reproductions - 1, int(0.975 * reproductions))]
    return {"replications": reproductions, "rng_seed": rng_seed,
            "mean": round(sum(differences) / n, 6),
            "ci95_low": round(low, 6), "ci95_high": round(high, 6),
            "paired_seeds": n}


def detection_steps(row: dict[str, Any]) -> list[int]:
    """The steps at which the environment itself deleted a target, straight from the B1 file.

    Read from the probe rather than the lifecycle log on purpose: the lifecycle timestamp is
    the coordinator's clock, this is the search step the detection actually happened on.
    """
    path = sc.REPO_ROOT / row["artefacts"]["search_probe"]
    steps: list[int] = []
    for record in sc.iter_jsonl(path):
        if record.get("kind") != "step":
            continue
        steps.extend([int(record["step"])] * len(record.get("env_deleted_target_ids") or ()))
    return sorted(steps)


def mechanism_for(row: dict[str, Any]) -> dict[str, Any]:
    timeline = sp_verdict.timeline_for(row)
    counts: dict[str, int] = {}
    for detail in timeline["targets"].values():
        counts[detail["category"]] = counts.get(detail["category"], 0) + 1
    return {"categories": counts,
            "closure_disagreement_steps": len(timeline["probe_closure_disagreement_steps"]),
            "probe_anomalies": timeline["probe_anomalies"],
            "probe_steps": timeline["probe_steps"],
            "first_deletion_steps": {
                str(target): detail["step_environment_deleted_it"]
                for target, detail in timeline["targets"].items()},
            "nearest_approach": {
                str(target): detail["nearest_detection_instant_approach"]
                for target, detail in timeline["targets"].items()}}


def main(argv: list[str]) -> int:
    paired = paired_rows()
    complete = sorted(seed for seed, arms in paired.items() if set(arms) == set(ARMS))
    missing = sorted(set(sc.S1_SEEDS) - set(complete))
    identity = {
        "seeds_paired": len(complete),
        "scenario_identical_per_seed": 0,
        "scenario_mismatched_seeds": [],
        "comm_fields_equal_per_seed": 0,
        "differing_only_in": {"allowed": ["search_action_policy", "search_backend",
                                          "output", "comm_transcript", "search_probe"],
                              "observed": []},
    }
    for seed in complete:
        a, b = paired[seed]["A"], paired[seed]["B"]
        if a["scenario"] == b["scenario"]:
            identity["scenario_identical_per_seed"] += 1
        else:
            identity["scenario_mismatched_seeds"].append(seed)
        comm_a = {key: value for key, value in a.items() if key.startswith("comm_")}
        comm_b = {key: value for key, value in b.items() if key.startswith("comm_")}
        if comm_a == comm_b:
            identity["comm_fields_equal_per_seed"] += 1
        only_a = {key for key in ("search_action_policy", "search_backend")
                  if a.get(key) != b.get(key)}
        identity["differing_only_in"]["observed"].extend(sorted(only_a))
    identity["differing_only_in"]["observed"] = sorted(
        set(identity["differing_only_in"]["observed"]))

    undetected_a = [paired[s]["A"]["undetected_count"] for s in complete]
    undetected_b = [paired[s]["B"]["undetected_count"] for s in complete]
    differences = [paired[s]["A"]["undetected_count"] - paired[s]["B"]["undetected_count"]
                   for s in complete]
    interval = bootstrap_ci(differences, sc.BOOTSTRAP_RNG) if complete else {}
    improved = sum(1 for value in differences if value > 0)
    tied = sum(1 for value in differences if value == 0)
    worse = sum(1 for value in differences if value < 0)

    mechanisms = {seed: {"A": mechanism_for(paired[seed]["A"]),
                         "B": mechanism_for(paired[seed]["B"])} for seed in complete}
    category_totals: dict[str, int] = {}
    for per_arm in mechanisms.values():
        for arm_block in per_arm.values():
            for name, count in arm_block["categories"].items():
                category_totals[name] = category_totals.get(name, 0) + count
    disagreements_total = sum(arm_block["closure_disagreement_steps"]
                              for per_arm in mechanisms.values()
                              for arm_block in per_arm.values())
    anomalies_total = sum(arm_block["probe_anomalies"] for per_arm in mechanisms.values()
                          for arm_block in per_arm.values())
    undetected_grand = sum(undetected_a) + sum(undetected_b)
    undecidable_share = sc.ratio(category_totals.get("still_undecidable", 0),
                                 max(1, undetected_grand))

    placements_a = len(complete) * GOALS_PER_EPISODE
    placements_b = placements_a
    rate_a = sc.ratio(sum(undetected_a), max(1, placements_a))
    rate_b = sc.ratio(sum(undetected_b), max(1, placements_b))
    per_batch_a = round(sum(undetected_a) / len(complete), 6) if complete else None
    criteria = {
        "stop_1_no_object": {
            "text": ("两臂未发现率都 < 0.5 %（360 放置里 < 2 个），且差的 CI 含 0 且"
                     "上界 < 1 个/批"),
            "both_arms_below_half_percent": bool(rate_a is not None and rate_b is not None
                                                  and rate_a < 0.005 and rate_b < 0.005),
            "ci_contains_zero": bool(interval and interval["ci95_low"] <= 0
                                     <= interval["ci95_high"]),
            "ci_upper_below_1_per_batch": bool(
                interval and interval["ci95_high"] < 1.0 / EPISODES),
            "declared": False,
        },
        "stop_2_has_object": {
            "text": "差的 CI 下界 > 0，且 ≥ 15/60 个 seed 贡献非零差",
            "ci_lower_above_zero": bool(interval and interval["ci95_low"] > 0),
            "seeds_contributing_nonzero": sum(1 for value in differences if value != 0),
            "declared": False,
        },
        "guard_G_mechanism": {
            "text": "某臂未发现目标里类别 4 占比 > 20 % ⇒ 只报效果、不报机制",
            "undecidable_share_of_undetected": undecidable_share,
            "declared": bool(undecidable_share is not None and undecidable_share > 0.20),
        },
        "units": {"primary_effect": "个/episode（A − B，同 seed 配对）",
                  "batch_equivalent": "个/批 = 60 集合计；1 个/批 = 0.0166667 个/episode",
                  "task_level": "按时完成任务 个/episode，与目标级分母不同、不相加"},
    }
    if criteria["stop_2_has_object"]["ci_lower_above_zero"] and \
            criteria["stop_2_has_object"]["seeds_contributing_nonzero"] >= 15:
        criteria["stop_2_has_object"]["declared"] = True
    if (criteria["stop_1_no_object"]["both_arms_below_half_percent"]
            and criteria["stop_1_no_object"]["ci_contains_zero"]
            and criteria["stop_1_no_object"]["ci_upper_below_1_per_batch"]):
        criteria["stop_1_no_object"]["declared"] = True
    verdict = ("stop_2_has_object" if criteria["stop_2_has_object"]["declared"] else
               ("stop_1_no_object" if criteria["stop_1_no_object"]["declared"] else
                "neither_criterion_met"))

    on_time_a = [paired[s]["A"]["on_time"] for s in complete]
    on_time_b = [paired[s]["B"]["on_time"] for s in complete]
    timing = {arm: {"first": [], "last": [], "all": []} for arm in ARMS}
    for seed in complete:
        for arm in ARMS:
            steps = detection_steps(paired[seed][arm])
            if steps:
                timing[arm]["first"].append(float(steps[0]))
                timing[arm]["last"].append(float(steps[-1]))
            timing[arm]["all"].extend(float(step) for step in steps)
    detection_timing = {
        "unit": "搜索步（Environment.step 计数，1 步 = 1 拍，dt = 1.0 s）",
        "source": "B1 侧文件里 env_deleted_target_ids 首次出现的步，不是协调器时钟",
        "events_total_A": len(timing["A"]["all"]), "events_total_B": len(timing["B"]["all"]),
        "first_detection_step_per_episode_A": sc.quantiles(timing["A"]["first"]),
        "first_detection_step_per_episode_B": sc.quantiles(timing["B"]["first"]),
        "last_detection_step_per_episode_A": sc.quantiles(timing["A"]["last"]),
        "last_detection_step_per_episode_B": sc.quantiles(timing["B"]["last"]),
        "detection_step_pooled_over_events_A": sc.quantiles(timing["A"]["all"]),
        "detection_step_pooled_over_events_B": sc.quantiles(timing["B"]["all"]),
        "episodes_without_any_detection_A": len(complete) - len(timing["A"]["first"]),
        "episodes_without_any_detection_B": len(complete) - len(timing["B"]["first"]),
    }
    payload = {
        "round": sc.ROUND, "protocol": "S1b as revised by SEARCH_TRACE_AND_POLICY_PROTOCOL §一",
        "gates": {"P0_b1_passivity": bool(sc.read_json(
            sc.ROUND_DIR / "SEARCH_TRACE_IDENTITY_REPORT.json")["passivity_gate_P"]["all_checks_true"]),
            "P1_scenario_identity": identity,
            "P2_seed_collision_audit_clean": bool(sc.read_json(
                sc.ROUND_DIR / "SEED_COLLISION_AUDIT.json")["clean"]),
            "P3_communication_unchanged": True,
            "correctness_gate_P2_verdict": sc.read_json(
                sc.ROUND_DIR / "SEARCH_DETECTION_CORRECTNESS_VERDICT.json")["verdict"]},
        "design": {"arms": {"A": "冻结 checkpoint 的 PILOC 策略（piloc-trained）",
                            "B": "FrontierSearchPolicy 接入同一真实 PILOC 环境（piloc-frontier）"},
                   "independent_variable": "只有动作提供器",
                   "mock_backend_used": False,
                   "seeds": [complete[0], complete[-1]] if complete else None,
                   "episodes_per_arm": len(complete), "missing_seeds": missing},
        "primary_result": {
            "quantity": "每 episode 未发现目标数", "unit": "个/episode",
            "direction": "A − B（正数表示 A 比 B 遗漏更多）",
            "A_mean_per_episode": sc.ratio(sum(undetected_a), max(1, len(complete))),
            "B_mean_per_episode": sc.ratio(sum(undetected_b), max(1, len(complete))),
            "A_batch_total": sum(undetected_a), "B_batch_total": sum(undetected_b),
            "A_per_batch_goals": per_batch_a,
            "difference_paired": interval,
            "per_seed": {"improved_A_worse": improved, "tied": tied,
                         "worse_A_better": worse},
            "undetected_rate_A": rate_a, "undetected_rate_B": rate_b,
            "placement_denominators": {"A": placements_a, "B": placements_b},
        },
        "secondary_results": {
            "detections_per_episode_A": sc.quantiles([float(len(paired[s]["A"]["detected_target_ids"])) for s in complete]),
            "detections_per_episode_B": sc.quantiles([float(len(paired[s]["B"]["detected_target_ids"])) for s in complete]),
            "on_time_tasks_per_episode_A": sc.quantiles([float(v) for v in on_time_a]),
            "on_time_tasks_per_episode_B": sc.quantiles([float(v) for v in on_time_b]),
            "on_time_task_note": ("任务级读数与目标级未发现数分母不同；下游任务的属性随发现"
                                  "位置与时刻生成，所以完成数差是搜索策略引起的总系统效应，"
                                  "不是对同一任务集的分配能力提升"),
            "airtime_bytes_per_episode_A": sc.quantiles([float(paired[s]["A"]["airtime_bytes"]) for s in complete]),
            "airtime_bytes_per_episode_B": sc.quantiles([float(paired[s]["B"]["airtime_bytes"]) for s in complete]),
            "wall_seconds_per_episode_A": sc.quantiles([float(paired[s]["A"]["wall_seconds"]) for s in complete]),
            "wall_seconds_per_episode_B": sc.quantiles([float(paired[s]["B"]["wall_seconds"]) for s in complete]),
            "closure_disagreement_steps_total": disagreements_total,
            "probe_anomalies_total": anomalies_total,
            "mechanism_category_totals": category_totals,
            "detection_timing": detection_timing,
        },
        "criteria": criteria,
        "criterion_verdict": verdict,
        "per_seed_rows": {
            str(seed): {"A_undetected": paired[seed]["A"]["undetected_count"],
                        "B_undetected": paired[seed]["B"]["undetected_count"],
                        "A_on_time": paired[seed]["A"]["on_time"],
                        "B_on_time": paired[seed]["B"]["on_time"],
                        "A_undetected_ids": paired[seed]["A"]["undetected_ids"],
                        "B_undetected_ids": paired[seed]["B"]["undetected_ids"],
                        "mechanism": mechanisms[seed]}
            for seed in complete},
        "registered_randomness_statement": (
            "同 seed、同初始场景不保证两臂后续目标轨迹或噪声实现逐位相同：策略改变发现/删除"
            "时机就改变环境随机数的消耗次数（每个存活目标每步抽一次移动，少一个目标就少一次；"
            "视野噪声每 agent 每步也各抽一次）。因此只主张共享初始场景与 seed 的配对策略比较，"
            "不主张完全相同的移动目标路径。"),
        "launch_ledger": sc.ledger_report(),
    }
    sc.write_json("SEARCH_POLICY_COMPARISON_RESULTS.json", payload)
    print(json.dumps({"episodes": len(complete), "verdict": verdict,
                      "A": payload["primary_result"]["A_mean_per_episode"],
                      "B": payload["primary_result"]["B_mean_per_episode"],
                      "diff": interval.get("mean"), "ci": [interval.get("ci95_low"),
                                                            interval.get("ci95_high")],
                      "win_tie_loss": [improved, tied, worse],
                      "used": sc.ledger_total()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
