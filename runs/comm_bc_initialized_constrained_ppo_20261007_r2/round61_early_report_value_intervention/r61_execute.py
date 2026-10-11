"""ROUND61 executor: paired FORCE_SEND / FORCE_SKIP intervention on the frozen R60 seed33 ep128
greedy model, through the R61 resource guard. L (original greedy) and L0 (interposer installed,
override disabled) gate first; then 8 cases x {FORCE_SEND, FORCE_SKIP} = 16 cells. 18 normal
launches + up to 2 reserve, hard cap 20. Resumable: completed cells are reloaded from their
persisted summary, cells with a complete attempt but no summary are recovered from their logs,
and a technical-failed cell is reserve-retried. No training, no optimizer/PPO/checkpoint/RNG."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "round58_gen2"))
sys.path.insert(0, str(HERE.parent / "round58_isolation_stop_audit"))
sys.path.insert(0, str(HERE.parent / "round60_partitioned_actor_head_ablation"))
import r60_execute
import r60_partitioned as r60
from r61_guard import R61Guard
from r61_intervene import make_intervene_collector

SEED = 33
MODE = "greedy"
FOCUS_SCENE = 1700027
NORMAL_CAP, RESERVE_CAP, HARD_CAP = 18, 2, 20
MODEL_PARAM_SHA = "bce71a0e9810122c146950f12cf818a9a359406312d96dfc9c7a457ebb654e2a"


def _slug(logical):
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", logical)


def on_time_ids(lifecycle):
    ids = set()
    p = Path(lifecycle)
    if not p.is_file():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("event_type") == "final" and e.get("target_id") and e["target_id"] != "__run__":
            if e.get("status") == "completed":
                ids.add(e["target_id"])
    return ids


def setup(r61_raw: Path):
    cfg = r60_execute.configure_and_inject(Path(r61_raw))
    r60_ep128 = HERE.parent / "round60_partitioned_actor_head_ablation" / "raw" / "checkpoints" / "v2_train_s33_ep128.pt"
    controller, state, identity = r60.load_checkpoint_read_only(r60_ep128, freeze=r60_execute.R60_FREEZE)
    ps = controller.parameter_summary()
    if ps["parameter_sha256"] != MODEL_PARAM_SHA:
        raise RuntimeError(f"frozen model param identity mismatch: {ps['parameter_sha256']}")
    if identity.get("optimizer_restored") is not False or identity.get("rng_restored") is not False:
        raise RuntimeError("frozen model load restored optimizer/RNG")
    from r42_one_shot import OneShotSendSkipCollector
    cfg["controller"] = controller
    cfg["intervene_cls"] = make_intervene_collector(OneShotSendSkipCollector)
    cfg["plain_cls"] = OneShotSendSkipCollector
    cfg["cells_dir"] = Path(r61_raw) / "cells"
    cfg["cells_dir"].mkdir(parents=True, exist_ok=True)
    cfg["model_identity"] = {"parameter_sha256": ps["parameter_sha256"], "structure": ps.get("structure"),
                             "episodes_done": state.get("episodes_done"), "load_mode": "read_only"}
    return cfg


def _persist(cfg, logical, result):
    (cfg["cells_dir"] / f"{_slug(logical)}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def run_cell(cfg, guard, *, logical, scene, collector, note, category="normal"):
    runner = cfg["runner"]; r42_run = cfg["r42_run"]
    slug = _slug(logical)
    claim_path = cfg["raw_root"] / "launch_claims" / slug / "attempt_0001.json"
    with guard.launch(logical, category=category) as L:
        claim_path.parent.mkdir(parents=True, exist_ok=True)
        claim_path.write_text(json.dumps({"logical_job": logical, "slot": L.slot, "note": note,
                                          "scene": int(scene), "category": category,
                                          "model": cfg["model_identity"]["parameter_sha256"]},
                                         ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        r42_run._ACTIVE_CLAIM = claim_path
        r42_run._ACTIVE_PROGRESS = r42_run._new_run_progress()
        L.builder_called()
        try:
            row = runner._run_episode(slug, "eval", int(scene), policy="learned",
                                      attach=collector.policy(), controller=cfg["controller"],
                                      collector=collector,
                                      action_rng_seed=runner.eval_action_seed(SEED, int(scene), MODE),
                                      tag=slug, checkpoint_load_identity=cfg["model_identity"])
        finally:
            prog = dict(r42_run._ACTIVE_PROGRESS)
            r42_run._ACTIVE_CLAIM = None
            r42_run._ACTIVE_PROGRESS = None
        L.env_constructed(); L.run_called()
        L.run_returned(steps=int(prog.get("completed_steps", row.get("steps", -1))), ok=True)
        iv = getattr(collector, "iv_record", None)
        qualified = bool(int(row.get("steps", 0)) == 1000 and int(prog.get("completed_steps", 0)) == 1000)
        L.result(qualified=qualified, detail=f"on_time={row['on_time']} bytes={row['spent_bytes']} iv={'yes' if iv else 'no'}")
        L.finish()
    result = {"logical": logical, "scene": int(scene), "note": note, "category": category,
              "on_time": int(row["on_time"]), "spent_bytes": int(row["spent_bytes"]),
              "steps": int(row.get("steps", -1)),
              "emission_sequence": row.get("emission_sequence"),
              "on_time_ids": sorted(on_time_ids(row.get("artefacts", {}).get("lifecycle")) or []),
              "iv_record": iv, "lifecycle": row.get("artefacts", {}).get("lifecycle"),
              "transcript": row.get("artefacts", {}).get("transcript")}
    _persist(cfg, logical, result)
    return result


def recover_from_logs(cfg, logical, scene):
    """Recover a completed cell's result from its persisted lifecycle + transcript (no re-run)."""
    slug = _slug(logical)
    logs = cfg["raw_root"] / "logs"
    rp = sorted(logs.glob(f"*{slug}*_rp.jsonl"))
    ct = sorted(logs.glob(f"*{slug}*_ct.jsonl"))
    if not rp:
        raise RuntimeError(f"no lifecycle log to recover {logical}")
    ids = on_time_ids(rp[-1]) or set()
    spent = None
    for line in rp[-1].read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e.get("target_id") == "__run__" and e.get("event_type") == "final":
            spent = int(e["detail"]["metrics"]["communication"]["budget"]["spent_bytes"])
    seq = []
    if ct:
        for line in ct[-1].read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e.get("kind") == "send":
                seq.append([int(e["step"]), str(e["target"]), int(e["revision"]),
                            int(e["cost_bytes"]), int(e["subscribers"]), bool(e["first"])])
    result = {"logical": logical, "scene": int(scene), "note": "recovered_from_logs",
              "category": "normal", "on_time": len(ids), "spent_bytes": int(spent),
              "steps": 1000, "emission_sequence": seq, "on_time_ids": sorted(ids),
              "iv_record": None, "lifecycle": str(rp[-1]), "transcript": str(ct[-1]) if ct else None}
    _persist(cfg, logical, result)
    return result


def get_cell(cfg, guard, *, logical, scene, make_collector, note):
    sp = cfg["cells_dir"] / f"{_slug(logical)}.json"
    if sp.exists():
        return json.loads(sp.read_text(encoding="utf-8"))
    term = guard.terminal(logical)
    if term == "complete":
        return recover_from_logs(cfg, logical, scene)
    category = "reserve" if term == "failed" else "normal"
    return run_cell(cfg, guard, logical=logical, scene=scene, collector=make_collector(),
                    note=note, category=category)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--phase", choices=("gate", "force", "all"), required=True)
    args = ap.parse_args(argv)
    r61_raw = args.raw_root.resolve()
    freeze = json.loads((HERE / "ROUND61_CASE_FREEZE.json").read_text(encoding="utf-8"))
    cases = freeze["cases"]
    cfg = setup(r61_raw)
    from budget_masked import SEND, HOLD
    guard = R61Guard(r61_raw, normal_cap=NORMAL_CAP, reserve_cap=RESERVE_CAP, hard_cap=HARD_CAP)
    results = {}
    gate_path = r61_raw / "ROUND61_GATE.json"

    def make_collector(case, forced, enabled):
        if forced is None:
            return cfg["plain_cls"](cfg["controller"], greedy=True)
        return cfg["intervene_cls"](cfg["controller"], greedy=True, target=case["target"],
                                    forced_action=forced, reason=f"round61-{case['role']}-{case['scene']}",
                                    override_enabled=enabled)

    if args.phase in ("gate", "all"):
        focus = cases[0]
        L = get_cell(cfg, guard, logical=f"r61_L_focus_s{FOCUS_SCENE}", scene=FOCUS_SCENE,
                     make_collector=lambda: make_collector(focus, None, False), note="original greedy path L")
        L0 = get_cell(cfg, guard, logical=f"r61_L0_focus_s{FOCUS_SCENE}", scene=FOCUS_SCENE,
                      make_collector=lambda: make_collector(focus, HOLD, False), note="interposer installed, override disabled L0")
        r60_focus = json.loads((HERE.parent / "round60_partitioned_actor_head_ablation" / "raw" / "rows" /
                                f"v2_r60_eval_new_s{SEED}_{MODE}_s{FOCUS_SCENE}_a1.json").read_text(encoding="utf-8"))
        gate = {"L": {k: L.get(k) for k in ("on_time", "spent_bytes", "emission_sequence", "on_time_ids")},
                "L0": {k: L0.get(k) for k in ("on_time", "spent_bytes", "emission_sequence", "on_time_ids")},
                "r60_focus": {"on_time": r60_focus["on_time"], "spent_bytes": r60_focus["spent_bytes"],
                              "emission_sequence": r60_focus["emission_sequence"]},
                "model_param_sha": cfg["model_identity"]["parameter_sha256"]}
        gate["L_equals_L0"] = (L["on_time"] == L0["on_time"] and L["spent_bytes"] == L0["spent_bytes"]
                               and L["emission_sequence"] == L0["emission_sequence"])
        gate["L_matches_r60_focus"] = (L["on_time"] == r60_focus["on_time"]
                                       and L["spent_bytes"] == r60_focus["spent_bytes"]
                                       and L["emission_sequence"] == r60_focus["emission_sequence"])
        gate["status"] = "passed" if (gate["L_equals_L0"] and gate["L_matches_r60_focus"]) else "failed"
        gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        results["gate"] = gate
        print(json.dumps({"gate": gate["status"], "L_equals_L0": gate["L_equals_L0"],
                          "L_matches_r60_focus": gate["L_matches_r60_focus"],
                          "L_on_time": L["on_time"], "L_bytes": L["spent_bytes"]}))
        if gate["status"] != "passed":
            return 2

    if args.phase in ("force", "all"):
        if not gate_path.is_file() or json.loads(gate_path.read_text(encoding="utf-8"))["status"] != "passed":
            print("gate not passed; run --phase gate first"); return 2
        for case in cases:
            for arm, forced in (("FORCE_SEND", SEND), ("FORCE_SKIP", HOLD)):
                tshort = case["target"].replace("piloc-e1-", "")
                logical = f"r61_{arm}_s{case['scene']}_{tshort}"
                row = get_cell(cfg, guard, logical=logical, scene=case["scene"],
                               make_collector=lambda c=case, f=forced: make_collector(c, f, True),
                               note=f"{arm} on {case['target']}")
                results[f"{arm}:{case['scene']}:{case['target']}"] = {
                    "on_time": row["on_time"], "spent_bytes": row["spent_bytes"],
                    "on_time_ids": row["on_time_ids"], "emission_sequence": row["emission_sequence"],
                    "iv_record": row["iv_record"]}
                print(json.dumps({"arm": arm, "scene": case["scene"], "target": case["target"],
                                  "role": case["role"], "on_time": row["on_time"], "bytes": row["spent_bytes"],
                                  "iv_acted": bool(row["iv_record"])}))
    (r61_raw / "ROUND61_CELL_RESULTS.json").write_text(json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    snap = guard.snapshot()
    (r61_raw / "ROUND61_GUARD_SNAPSHOT.json").write_text(json.dumps(snap, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"guard": snap}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
