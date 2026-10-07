"""ROUND47 read-only analysis: accounting recheck, BC->PPO comparison, per-scene
cost decomposition. Zero environments, zero training, pure file reading + math."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RUN_DIR = Path(__file__).resolve().parents[1]
REPO = RUN_DIR.parents[1]
V6_ROOT = RUN_DIR / "v6"
V8_ROOT = RUN_DIR / "v8_eval_booking_close"
V9_ROOT = RUN_DIR / "v9_eval_runtime_close"
V10_ROOT = RUN_DIR / "v10_eval_final_close"
R45_DIR = REPO / "runs" / "comm_rule_warmstart_cost_learning_20261006_r2"
R44_DIR = REPO / "runs" / "comm_dual_response_ablation_20261004"
R42_DIR = REPO / "runs" / "comm_one_shot_gate_resume_20261003_v2"
OUT = Path(__file__).resolve().parent

SEEDS = (11, 22, 33)
SCENES = tuple(range(1_301_000, 1_301_016))
BASE_NORMAL = 386
BASE_STEPS = 386_000
LATE_FROZEN_MEAN = 4838.25
BOOTSTRAP_SEED = 20261007
BOOTSTRAP_REPLICATES = 10_000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_json(path: Path, value) -> str:
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    Path(path).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def on_time_ids(row) -> set[str]:
    return {str(e["task_id"]) for e in row.get("task_reward_events", [])
            if e.get("credited") is True and e.get("on_time") is True}


def send_costs(row_path: Path) -> dict[str, dict[str, float]]:
    """Per-target billed cost from the communication transcript send records."""
    per_target: dict[str, dict[str, float]] = {}
    duplicate_first = 0
    send_records = 0
    for record in read_jsonl(row_path):
        if record.get("kind") != "send":
            continue
        send_records += 1
        target = str(record.get("target"))
        entry = per_target.setdefault(target, {
            "cost_bytes": 0, "attempts": 0, "packets_lost": 0,
            "packets_delivered": 0, "send_records": 0, "first_records": 0,
            "subscribers": 0, "payload_bytes": 0, "first_step": None,
        })
        entry["cost_bytes"] += int(record.get("cost_bytes", 0) or 0)
        entry["attempts"] += int(record.get("attempts", 0) or 0)
        entry["packets_lost"] += int(record.get("packets_lost", 0) or 0)
        entry["packets_delivered"] += int(record.get("packets_delivered", 0) or 0)
        entry["send_records"] += 1
        entry["first_records"] += 1 if record.get("first") is True else 0
        entry["subscribers"] = int(record.get("subscribers", 0) or 0)
        entry["payload_bytes"] = int(record.get("payload_bytes", 0) or 0)
        step = int(record.get("step", -1) or -1)
        if entry["first_step"] is None or step < entry["first_step"]:
            entry["first_step"] = step
    for entry in per_target.values():
        if entry["first_records"] > 1:
            duplicate_first += 1
    return {"per_target": per_target, "send_records": send_records,
            "targets_with_multiple_first_records": duplicate_first}


def row_bytes_close(row, transcript_path: Path) -> tuple[bool, int, int]:
    billed = sum(int(r.get("cost_bytes", 0) or 0) for r in read_jsonl(transcript_path)
                 if r.get("kind") == "send")
    stream_total = sum(int(v.get("airtime_bytes", 0) or 0)
                       for v in row.get("cost_stream", {}).values())
    return billed == stream_total == int(row.get("spent_bytes", -1)), billed, stream_total


# ---------------------------------------------------------------------------
# Part A: accounting recheck
# ---------------------------------------------------------------------------
def part_accounting() -> dict:
    claim_roots = {
        "v10_final_close": V10_ROOT,
        "v9_runtime_close_1301002": V9_ROOT,
        "v8_booking_close_1301001": V8_ROOT,
        "v7_eval_only_attempt2": RUN_DIR / "v7_eval_only",
    }
    per_claim = []
    totals = {"claims": 0, "constructors": 0, "run_returns": 0, "steps": 0,
              "complete": 0, "failed_technical": 0, "other": 0}
    for root_name, root in claim_roots.items():
        for path in sorted((root / "launch_claims").glob("*/*.json")):
            claim = read_json(path)
            status = claim.get("status")
            steps = int(claim.get("completed_steps", 0) or 0)
            constructors = int(claim.get("environment_constructor_calls", 0) or 0)
            returns = int(claim.get("run_return_calls", 0) or 0)
            totals["claims"] += 1
            totals["constructors"] += constructors
            totals["run_returns"] += returns
            totals["steps"] += steps
            totals[{"complete": "complete", "failed_technical": "failed_technical"}
                   .get(status, "other")] += 1
            per_claim.append({
                "root": root_name, "path": str(path),
                "logical_job": claim.get("logical_job"), "attempt": claim.get("attempt"),
                "category": claim.get("category"), "slot": claim.get("launch_slot"),
                "status": status, "completed_steps": steps,
                "environment_constructor_calls": constructors,
                "run_return_calls": returns,
                "sha256": sha256_file(path),
            })
    base_ledger_path = RUN_DIR / "runtime" / "ROUND46_EXECUTION_LEDGER.json"
    base_ledger = read_json(base_ledger_path) if base_ledger_path.is_file() else {}
    base_counts = base_ledger.get("counts", {})
    v10_ledger = read_json(V10_ROOT / "runtime" / "ROUND46_R2_V10_EXECUTION_LEDGER.json")
    v10_counts = v10_ledger["counts"]
    expected_steps = BASE_STEPS + totals["steps"]
    expected_constructors = BASE_NORMAL + totals["constructors"]
    recheck = {
        "kind": "round47-accounting-recheck/1",
        "task_id": "ROUND47-READONLY-COST-DECOMPOSITION",
        "rebuilt_from": {
            "base": "prior read-only audit: 386 normal claims, 386000 steps (immutable); "
                    "original execution ledger at " + str(base_ledger_path) +
                    f" (raw counts: {json.dumps(base_counts, sort_keys=True)})",
            "post_training_claims": per_claim,
        },
        "rebuilt_totals": {
            "total_claims": BASE_NORMAL + totals["claims"],
            "total_environment_constructor_calls": expected_constructors,
            "total_run_return_calls": BASE_NORMAL + totals["run_returns"],
            "total_completed_steps": expected_steps,
            "technical_failed_claims_total": 1 + totals["failed_technical"],
            "adopted_cells": 3,
            "new_claim_roots_scanned": list(claim_roots),
        "note": "each recovery round left its claim in its own root: attempt2 in V7, 1301001 in V8, 1301002 in V9, the 93 final cells in V10",
        },
        "v10_reported": {
            "report_says_completed_steps": 482000,
            "ledger_completed_steps": v10_counts.get("completed_steps"),
            "ledger_environment_constructor_calls": v10_counts.get("environment_constructor_calls"),
            "ledger_technical_failed_claims": v10_counts.get("technical_failed_claims"),
        },
        "handover_chat_claim_389000": {
            "where": "assistant handover summary text for ROUND46-R2-V10 only",
            "nature": "arithmetic slip in prose: counted base 386000 + 3 adopted cells "
                      "(3000) but omitted the 93 new V10 cells (93000)",
            "machine_report_and_ledger_already_say": 482000,
            "evidence_closed": True,
        },
        "findings": [
            {"field": "completed_steps", "v10_ledger": v10_counts.get("completed_steps"),
             "rebuilt": expected_steps, "verdict": "correct" if v10_counts.get("completed_steps") == expected_steps else "MISMATCH"},
            {"field": "environment_constructor_calls", "v10_ledger": v10_counts.get("environment_constructor_calls"),
             "rebuilt": expected_constructors,
             "verdict": "correct" if v10_counts.get("environment_constructor_calls") == expected_constructors else "undercount",
             "explanation": "the ledger formula counts only V10-root claims plus a hardcoded +2; the attempt2 claim lives in the V7 root and its constructor is omitted"},
            {"field": "technical_failed_claims", "v10_ledger": v10_counts.get("technical_failed_claims"),
             "rebuilt": 1 + totals["failed_technical"],
             "verdict": "correct" if v10_counts.get("technical_failed_claims") == 1 + totals["failed_technical"] else "undercount",
             "explanation": "true failed_technical claims = attempt1 (inside the 386 base) + attempt2 + 1301001 + 1301002 = 4; the ledger hardcoded +2"},
        ],
        "per_claim_count": len(per_claim),
    }
    return recheck


# ---------------------------------------------------------------------------
# Part B: BC -> PPO comparison
# ---------------------------------------------------------------------------
def load_baseline():
    manifest = read_json(R44_DIR / "ROUND44_BASELINE_REUSE.json")
    references = [i for i in manifest["round42_immediate_references"]
                  if 1_301_000 <= int(i["scene"]) < 1_301_016]
    late_items = [i for i in manifest["round42_fixed_rules"]
                  if i.get("arm") == "late_discovery_gate"]
    immediate = {}
    late = {}
    for item in references:
        row = read_json(Path(item["path"]))
        assert int(row["scenario_seed"]) == int(item["scene"])
        immediate[int(item["scene"])] = row
    for item in late_items:
        row = read_json(Path(item["path"]))
        late[int(item["scene"])] = row
    assert set(immediate) == set(SCENES) and set(late) == set(SCENES)
    return immediate, late


def load_bc_rows() -> dict[int, dict[int, dict]]:
    rows: dict[int, dict[int, dict]] = {}
    identities: dict[int, set] = {}
    for seed in SEEDS:
        rows[seed] = {}
        identities[seed] = set()
        for scene in SCENES:
            pattern = R45_DIR / "rows" / f"v2_r45_bc_greedy_s{seed}_scene{scene}_s{scene}_a1.json"
            row = read_json(pattern)
            assert int(row["training_seed"]) == seed
            assert row.get("model_arm") == "BC_ONLY_R2"
            assert row.get("r45_evaluation_role") == "BC_ONLY_GREEDY_CONTROL"
            assert row.get("eval_mode") == "greedy"
            identities[seed].add((row.get("bc_checkpoint_sha256"),
                                  row.get("bc_parameter_sha256")))
            rows[seed][scene] = row
        assert len(identities[seed]) == 1, f"seed {seed} BC identity is not unique"
    return rows, identities


def load_ppo_rows() -> dict[int, dict[int, dict]]:
    adopted = {
        (11, "greedy", 1_301_000): V8_ROOT / "adoption/attempt2_effective_row.json",
        (11, "greedy", 1_301_001): V9_ROOT / "adoption/cell1301001_effective_row.json",
        (11, "greedy", 1_301_002): V10_ROOT / "adoption/cell1301002_effective_row.json",
    }
    greedy_rows: dict[int, dict[int, dict]] = {seed: {} for seed in SEEDS}
    sampled_rows: dict[int, dict[int, dict]] = {seed: {} for seed in SEEDS}
    for path in sorted((V10_ROOT / "rows").glob("*.json")):
        row = read_json(path)
        seed = int(row["training_seed"]); scene = int(row["scenario_seed"])
        mode = str(row["eval_mode"])
        if mode == "greedy":
            greedy_rows[seed][scene] = row
        elif mode == "sampled":
            sampled_rows[seed][scene] = row
    for (seed, mode, scene), path in adopted.items():
        row = read_json(path)
        assert mode == "greedy"
        greedy_rows[seed][scene] = row
    for seed in SEEDS:
        assert set(greedy_rows[seed]) == set(SCENES), (seed, sorted(set(SCENES) - set(greedy_rows[seed])))
        assert set(sampled_rows[seed]) == set(SCENES)
    return {"greedy": greedy_rows, "sampled": sampled_rows}


def paired_bootstrap(diffs: list[float]) -> dict:
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indexes = rng.integers(0, 16, size=(BOOTSTRAP_REPLICATES, 16), dtype=np.int64)
    values = np.asarray(diffs, dtype=np.float64)
    means = values[indexes].mean(axis=1)
    return {
        "point_mean": float(values.mean()),
        "bootstrap_mean": float(means.mean()),
        "ci_low_linear": float(np.percentile(means, 2.5, method="linear")),
        "ci_high_linear": float(np.percentile(means, 97.5, method="linear")),
        "replicates": BOOTSTRAP_REPLICATES,
        "index_seed": BOOTSTRAP_SEED,
        "index_shape": [BOOTSTRAP_REPLICATES, 16],
    }


def part_comparison() -> dict:
    immediate, late = load_baseline()
    bc_rows, bc_identities = load_bc_rows()
    ppo_rows = load_ppo_rows()["greedy"]
    per_seed = {}
    bc_identity_block = {
        "control_rows": 48,
        "decoding": "greedy",
        "per_seed_bc_checkpoint_identity_unique": True,
        "per_seed": {str(seed): sorted(identities)[0]
                     for seed, identities in bc_identities.items()},
        "training_version_of_round46": "constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/dual-eta-lambda-0p5/1/rule-bc-late147-init/1/published-completion-event-at-relay-tick/2",
        "note": "the rule-bc-late147-init token in the frozen ROUND46 training version is the BC initialization these controls replay; per-seed checkpoint hashes are stable across all 16 scenes",
    }
    for seed in SEEDS:
        scenes = SCENES
        block = {
            "ppo": {"on_time": [], "bytes": [], "exact_immediate": [], "exact_bc": [],
                    "scenes": {}},
            "bc": {"on_time": [], "bytes": [], "exact_immediate": [], "scenes": {}},
            "immediate": {"on_time": [], "bytes": []},
            "late": {"on_time": [], "bytes": []},
        }
        for scene in scenes:
            ppo = ppo_rows[seed][scene]
            bc = bc_rows[seed][scene]
            imm = immediate[scene]
            lt = late[scene]
            ppo_ids = on_time_ids(ppo)
            bc_ids = on_time_ids(bc)
            imm_ids = on_time_ids(imm)
            lt_ids = on_time_ids(lt)
            block["ppo"]["on_time"].append(int(ppo["on_time"]))
            block["ppo"]["bytes"].append(int(ppo["spent_bytes"]))
            block["ppo"]["exact_immediate"].append(ppo_ids == imm_ids)
            block["ppo"]["exact_bc"].append(ppo_ids == bc_ids)
            block["ppo"]["scenes"][str(scene)] = {
                "on_time": int(ppo["on_time"]), "spent_bytes": int(ppo["spent_bytes"]),
                "on_time_task_ids": sorted(ppo_ids)}
            block["bc"]["on_time"].append(int(bc["on_time"]))
            block["bc"]["bytes"].append(int(bc["spent_bytes"]))
            block["bc"]["exact_immediate"].append(bc_ids == imm_ids)
            block["bc"]["scenes"][str(scene)] = {
                "on_time": int(bc["on_time"]), "spent_bytes": int(bc["spent_bytes"]),
                "on_time_task_ids": sorted(bc_ids)}
            block["immediate"]["on_time"].append(int(imm["on_time"]))
            block["immediate"]["bytes"].append(int(imm["spent_bytes"]))
            block["late"]["on_time"].append(int(lt["on_time"]))
            block["late"]["bytes"].append(int(lt["spent_bytes"]))
        def mean(values):
            return sum(values) / len(values)
        ppo_minus_late = [p - l for p, l in zip(block["ppo"]["bytes"], block["late"]["bytes"])]
        ppo_minus_bc = [p - b for p, b in zip(block["ppo"]["bytes"], block["bc"]["bytes"])]
        bc_minus_late = [b - l for b, l in zip(block["bc"]["bytes"], block["late"]["bytes"])]
        per_seed[str(seed)] = {
            "ppo_greedy": {
                "mean_on_time": mean(block["ppo"]["on_time"]),
                "mean_spent_bytes": mean(block["ppo"]["bytes"]),
                "total_spent_bytes": sum(block["ppo"]["bytes"]),
                "scenes_exact_immediate_16_of_16": sum(block["ppo"]["exact_immediate"]) == 16,
                "scenes_exact_bc_16_of_16": sum(block["ppo"]["exact_bc"]) == 16,
                "scenes_exact_immediate_count": sum(block["ppo"]["exact_immediate"]),
                "scenes_exact_bc_count": sum(block["ppo"]["exact_bc"]),
            },
            "bc_greedy": {
                "mean_on_time": mean(block["bc"]["on_time"]),
                "mean_spent_bytes": mean(block["bc"]["bytes"]),
                "total_spent_bytes": sum(block["bc"]["bytes"]),
                "scenes_exact_immediate_count": sum(block["bc"]["exact_immediate"]),
            },
            "immediate_reference": {"mean_on_time": mean(block["immediate"]["on_time"]),
                                    "mean_spent_bytes": mean(block["immediate"]["bytes"])},
            "late_rule": {"mean_on_time": mean(block["late"]["on_time"]),
                          "mean_spent_bytes": mean(block["late"]["bytes"])},
            "paired_bytes_ppo_minus_late": {
                "per_scene": dict(zip(map(str, scenes), ppo_minus_late)),
                "bootstrap": paired_bootstrap(ppo_minus_late),
                "late_frozen_mean": LATE_FROZEN_MEAN,
                "late_recomputed_mean": mean(block["late"]["bytes"]),
            },
            "paired_bytes_ppo_minus_bc": {
                "per_scene": dict(zip(map(str, scenes), ppo_minus_bc)),
                "bootstrap": paired_bootstrap(ppo_minus_bc),
            },
            "paired_bytes_bc_minus_late": {
                "bootstrap": paired_bootstrap(bc_minus_late),
            },
            "answers": {
                "ppo_preserves_exact_target_sets_vs_immediate":
                    sum(block["ppo"]["exact_immediate"]) == 16,
                "ppo_bytes_vs_bc_direction": ("decreased" if mean(block["ppo"]["bytes"]) < mean(block["bc"]["bytes"])
                                              else "increased" if mean(block["ppo"]["bytes"]) > mean(block["bc"]["bytes"])
                                              else "unchanged"),
                "ppo_minus_bc_mean_bytes": mean(block["ppo"]["bytes"]) - mean(block["bc"]["bytes"]),
                "bc_covers_the_immediate_reduction": mean(block["bc"]["bytes"]) <= mean(block["late"]["bytes"]),
            },
        }
    return {"kind": "round47-bc-to-ppo-comparison/1",
            "task_id": "ROUND47-READONLY-COST-DECOMPOSITION",
            "bc_identity": bc_identity_block,
            "per_seed": per_seed,
            "inputs": {
                "immediate_and_late": str(R44_DIR / "ROUND44_BASELINE_REUSE.json"),
                "bc_rows_dir": str(R45_DIR / "rows"),
                "ppo_rows": "V10 analysis matrix (3 adopted rows via registered receipts + 93 V10 rows)",
            }}


# ---------------------------------------------------------------------------
# Part C: per-scene, per-target decomposition
# ---------------------------------------------------------------------------
def part_decomposition() -> dict:
    immediate, late = load_baseline()
    bc_rows = load_bc_rows()[0]
    ppo_rows = load_ppo_rows()["greedy"]
    out = {"kind": "round47-per-scene-cost-decomposition/1",
           "task_id": "ROUND47-READONLY-COST-DECOMPOSITION",
           "per_seed": {}}
    for seed in SEEDS:
        seed_block = {"vs_late": {"scenes": {}}, "vs_bc": {"scenes": {}},
                      "one_shot_semantics": {"violations": []}}
        for scene in SCENES:
            ppo = ppo_rows[seed][scene]
            ppo_costs = send_costs(Path(ppo["artefacts"]["transcript"]))
            for bench_name, bench_row in (("vs_late", late[scene]), ("vs_bc", bc_rows[seed][scene])):
                bench_costs = send_costs(Path(bench_row["artefacts"]["transcript"]))
                ppo_t = ppo_costs["per_target"]
                bench_t = bench_costs["per_target"]
                a_targets = sorted(set(ppo_t) - set(bench_t))
                b_targets = sorted(set(bench_t) - set(ppo_t))
                c_targets = sorted(set(ppo_t) & set(bench_t))
                a_cost = sum(ppo_t[t]["cost_bytes"] for t in a_targets)
                b_cost = sum(bench_t[t]["cost_bytes"] for t in b_targets)
                c_cost = sum(ppo_t[t]["cost_bytes"] - bench_t[t]["cost_bytes"] for t in c_targets)
                delta = int(ppo["spent_bytes"]) - int(bench_row["spent_bytes"])
                common_detail = {}
                for t in c_targets:
                    p_entry, b_entry = ppo_t[t], bench_t[t]
                    common_detail[t] = {
                        "ppo_cost_bytes": p_entry["cost_bytes"],
                        "bench_cost_bytes": b_entry["cost_bytes"],
                        "cost_diff": p_entry["cost_bytes"] - b_entry["cost_bytes"],
                        "ppo_attempts": p_entry["attempts"],
                        "bench_attempts": b_entry["attempts"],
                        "ppo_subscribers": p_entry["subscribers"],
                        "bench_subscribers": b_entry["subscribers"],
                        "ppo_payload_bytes": p_entry["payload_bytes"],
                        "bench_payload_bytes": b_entry["payload_bytes"],
                        "ppo_first_step": p_entry["first_step"],
                        "bench_first_step": b_entry["first_step"],
                    }
                scene_block = {
                    "ppo_spent_bytes": int(ppo["spent_bytes"]),
                    "bench_spent_bytes": int(bench_row["spent_bytes"]),
                    "delta_bytes": delta,
                    "A_model_only_targets": a_targets,
                    "A_model_only_cost_bytes": a_cost,
                    "B_bench_only_targets": b_targets,
                    "B_bench_only_cost_bytes": b_cost,
                    "C_common_target_cost_diff_bytes": c_cost,
                    "identity_holds": delta == a_cost - b_cost + c_cost,
                    "common_targets": common_detail,
                }
                seed_block[bench_name]["scenes"][str(scene)] = scene_block
            for bench_name in ("vs_late", "vs_bc"):
                pass
            for source_name, costs in (("ppo", ppo_costs),):
                if costs["targets_with_multiple_first_records"]:
                    seed_block["one_shot_semantics"]["violations"].append({
                        "scene": scene, "source": source_name,
                        "targets_with_multiple_first_records": costs["targets_with_multiple_first_records"]})
        summary = {}
        for bench_name in ("vs_late", "vs_bc"):
            scenes_block = seed_block[bench_name]["scenes"]
            totals = {
                "delta_bytes_total": sum(s["delta_bytes"] for s in scenes_block.values()),
                "A_total": sum(s["A_model_only_cost_bytes"] for s in scenes_block.values()),
                "B_total": sum(s["B_bench_only_cost_bytes"] for s in scenes_block.values()),
                "C_total": sum(s["C_common_target_cost_diff_bytes"] for s in scenes_block.values()),
                "identity_all_scenes": all(s["identity_holds"] for s in scenes_block.values()),
            }
            totals["residual"] = totals["delta_bytes_total"] - (totals["A_total"] - totals["B_total"] + totals["C_total"])
            top_scenes = sorted(scenes_block.items(), key=lambda kv: abs(kv[1]["delta_bytes"]), reverse=True)[:3]
            summary[bench_name] = {**totals,
                                   "top_scenes_by_abs_delta": [{"scene": k, "delta_bytes": v["delta_bytes"],
                                                                "A": v["A_model_only_cost_bytes"],
                                                                "B": v["B_bench_only_cost_bytes"],
                                                                "C": v["C_common_target_cost_diff_bytes"]}
                                                               for k, v in top_scenes]}
        seed_block["summary"] = summary
        out["per_seed"][str(seed)] = seed_block
    return out


def main() -> None:
    accounting = part_accounting()
    sha_a = write_json(OUT / "ACCOUNTING_RECHECK.json", accounting)
    comparison = part_comparison()
    sha_b = write_json(OUT / "BC_TO_PPO_COMPARISON.json", comparison)
    decomposition = part_decomposition()
    sha_c = write_json(OUT / "PER_SCENE_COST_DECOMPOSITION.json", decomposition)
    inputs = {
        "kind": "round47-input-source-list/1",
        "inputs": [
            {"role": "v10 ledger", "path": str(V10_ROOT / "runtime/ROUND46_R2_V10_EXECUTION_LEDGER.json")},
            {"role": "v10 report", "path": str(V10_ROOT / "ROUND46_R2_V10_REPORT.md")},
            {"role": "v10 machine verdict", "path": str(V10_ROOT / "ROUND46_R2_V10_MACHINE_VERDICT.json")},
            {"role": "ppo rows (93)", "path": str(V10_ROOT / "rows")},
            {"role": "adopted rows", "paths": [str(V8_ROOT / "adoption/attempt2_effective_row.json"),
                                               str(V9_ROOT / "adoption/cell1301001_effective_row.json"),
                                               str(V10_ROOT / "adoption/cell1301002_effective_row.json")]},
            {"role": "immediate/late manifest", "path": str(R44_DIR / "ROUND44_BASELINE_REUSE.json")},
            {"role": "bc rows (48)", "path": str(R45_DIR / "rows")},
            {"role": "bc comparison aggregates", "path": str(R45_DIR / "ROUND45_R2_CONTROL_RAW_ROW_RECOMPUTATION_R3_FINAL3.json")},
        ],
        "deliverable_sha256": {
            "ACCOUNTING_RECHECK.json": sha_a,
            "BC_TO_PPO_COMPARISON.json": sha_b,
            "PER_SCENE_COST_DECOMPOSITION.json": sha_c,
        },
    }
    write_json(OUT / "INPUT_SOURCES.json", inputs)
    print(json.dumps({
        "accounting": {"rebuilt_steps": accounting["rebuilt_totals"]["total_completed_steps"],
                       "rebuilt_constructors": accounting["rebuilt_totals"]["total_environment_constructor_calls"],
                       "findings": accounting["findings"]},
        "per_seed": {seed: {
            "ppo_bytes": block["ppo_greedy"]["mean_spent_bytes"],
            "bc_bytes": block["bc_greedy"]["mean_spent_bytes"],
            "late_bytes": block["late_rule"]["mean_spent_bytes"],
            "ppo_minus_bc": block["answers"]["ppo_minus_bc_mean_bytes"],
            "ppo_exact_immediate": block["answers"]["ppo_preserves_exact_target_sets_vs_immediate"],
            "ppo_minus_late_ci": block["paired_bytes_ppo_minus_late"]["bootstrap"],
        } for seed, block in comparison["per_seed"].items()},
        "decomposition_identity": {seed: block["summary"]["vs_late"]["identity_all_scenes"]
                                   for seed, block in decomposition["per_seed"].items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
