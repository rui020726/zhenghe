#!/usr/bin/env python
"""ROUND71 approved fixed trial: the ONE runner (gates -> offline-fit -> train -> eval -> stats).

Every phase stands up the SAME live ROUND71 runtime through the readiness binding (the real
ROUND68 executor with only quota/version/checkpoint rebound), behind the archived trial
licence and an explicit model-read approval.  Phases are resumable: a rolling checkpoint, the
committed batch records and the durable claim/slot/permit regime decide where a seed
continues, so an interrupted run never re-runs a committed update.

Quota (hard, no refund): 582 normal (384 training + 192 model eval + 6 focal) + 18 reserve,
hard cap 600 environment constructions; offline trunk input rows <= 72,000 (debited before
the forward); synthetic optimizer steps <= 16 (gates only).  Historical R60/R66/R68
constructions are NOT ROUND71 consumption.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN_ROOT = REPO / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2"
READINESS_DIR = REPO / "analysis" / "round71-critic-calibration-readiness"
for _d in (READINESS_DIR, str(RUN_ROOT / "round68_late_initial_confidence_ablation"), str(HERE)):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import r71_trial_binding as trial                                       # noqa: E402
import r71_trial_gates as gates                                         # noqa: E402
import r71_trial_execute as execute                                     # noqa: E402
import r71_offline_fit as fit                                           # noqa: E402
import r71_runtime_binding as readiness                                  # noqa: E402

import numpy as np  # noqa: E402


def _json_safe(obj):
    """Recursively convert numpy types to JSON-native Python for derived receipts only."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


SEEDS = execute.SEEDS
TASK_ID = trial.R71_TRIAL_TASK_ID
BRANCH = trial.R71_TRIAL_BRANCH
NORMAL_CAP = execute.NORMAL_CAP
RESERVE_CAP = execute.RESERVE_CAP
HARD_CAP = execute.HARD_CAP
HISTORICAL_PROBE = execute.HISTORICAL_PROBE

_read_json = execute._read_json
_write_json = execute._write_json


class R71Ledger:
    """The ROUND71 claim/slot/quota ledger under the ROUND71 raw root (never the R68 one)."""

    def __init__(self, raw: Path) -> None:
        self.raw = Path(raw)
        self.claims_dir = self.raw / "launch_claims"
        self.slots_dir = self.raw / "launch_slots"
        self.runtime_dir = self.raw / "runtime"
        for d in (self.claims_dir, self.slots_dir / "normal", self.slots_dir / "reserve",
                  self.runtime_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.runtime_dir / "ROUND71_LEDGER.json"

    def _slug(self, logical: str) -> str:
        return re.sub(r"[^0-9A-Za-z_.-]+", "_", logical)

    def is_complete(self, logical: str) -> bool:
        directory = self.claims_dir / self._slug(logical)
        if not directory.is_dir():
            return False
        return any(_read_json(p).get("status") == "complete"
                   for p in sorted(directory.glob("attempt_*.json")))

    def counts(self) -> dict:
        normal = reserve = complete = failed = 0
        for path in sorted(self.claims_dir.glob("*/*.json")):
            try:
                claim = _read_json(path)
            except Exception:
                continue
            if claim.get("category") == "normal":
                normal += 1
            elif claim.get("category") == "reserve":
                reserve += 1
            if claim.get("status") == "complete":
                complete += 1
            elif claim.get("status") == "failed_technical":
                failed += 1
        booked = normal + reserve
        actual = booked + HISTORICAL_PROBE
        return {"normal_claims": normal, "reserve_claims": reserve, "complete": complete,
                "failed_technical": failed, "booked_attempts": booked,
                "cumulative_attempts": booked,
                "cumulative_attempts_including_historical_probe": actual,
                "historical_non_plan_probe": HISTORICAL_PROBE,
                "actual_environment_constructions": actual,
                "normal_cap": NORMAL_CAP, "reserve_cap": RESERVE_CAP, "hard_cap": HARD_CAP,
                "normal_headroom": NORMAL_CAP - normal,
                "reserve_headroom": RESERVE_CAP - reserve,
                "hard_headroom": HARD_CAP - actual,
                "hard_cap_exceeded_by": max(0, actual - HARD_CAP)}

    def write_ledger(self) -> dict:
        counts = self.counts()
        ledger = {"kind": "round71-ledger/1", "task_id": TASK_ID, "branch": BRANCH,
                  "run_id": trial.TRIAL_RUN_ID,
                  "training_version": readiness.R71_TRAINING_VERSION,
                  "approval_id": trial.TRIAL_APPROVAL_ID,
                  "quota_identity": readiness.R71_QUOTA_IDENTITY,
                  "construction_permit_regime": execute._atomic_run.__module__ + "._r68_atomic_run",
                  "counts": counts, "updated_unix": time.time()}
        _write_json(self.ledger_path, ledger)
        return ledger


# ------------------------------------------------------------------ phases ---------------
def phase_gates(raw_root: Path) -> int:
    summary = gates.run_all_gates(raw_root)
    print(json.dumps({k: v for k, v in summary.items() if k != "gates"}, indent=1))
    return 0 if summary["status"] == "passed" else 2


def phase_offline_fit(cfg, raw_root: Path, seeds) -> dict:
    gate = trial.trial_gate()
    run_identity = trial.trial_run_identity()
    ledger = gates.forward_ledger(raw_root)
    out = []
    for seed in seeds:
        paths = execute.init_paths(cfg, seed)
        print(f"R71 offline-fit seed {seed} ...", flush=True)
        result = fit.offline_fit_seed(
            seed=int(seed), gate=gate, model_read_approval=trial.MODEL_READ_APPROVAL,
            run_identity=run_identity, ledger=ledger, init_path=paths["init"],
            identity_path=paths["identity"])
        receipt = {k: v for k, v in result.items() if k not in ("identity",)}
        receipt["init_parameter_sha256"] = result["identity"]["init_parameter_sha256"]
        receipt["calibration_module_sha"] = result["identity"]["calibration"]
        _write_json(raw_root / "runtime" / f"r71_offline_fit_s{int(seed)}.json", _json_safe(receipt))
        out.append(receipt)
        print(f"R71 offline-fit seed {seed}: rows={result['ledger']['rows']} "
              f"joint_rows={result['ledger']['joint_rows']} "
              f"changed={result['install']['changed_modules']}", flush=True)
    return {"status": "offline_fit_complete", "seeds": [int(s) for s in seeds],
            "ledger": {"rows": ledger.rows, "calls": ledger.calls,
                       "joint_rows": ledger.joint_rows, "cap": ledger.cap,
                       "remaining": ledger.remaining()}, "results": out}


def phase_train(cfg, raw_root: Path, seeds, *, first_gate_only: bool) -> dict:
    ledger = R71Ledger(raw_root)
    order = [int(x) for x in cfg["common"].scenario_order()]
    if len(order) != execute.EPISODES_PER_SEED:
        raise RuntimeError(f"training order length {len(order)} != {execute.EPISODES_PER_SEED}")
    plan_path = raw_root / "ROUND71_FIXED_PLAN.json"
    if not plan_path.exists():
        plan = execute.build_plan(order)
        _write_json(plan_path, plan)
    else:
        plan = _read_json(plan_path)
    references = execute.reuse_r66_references(cfg, order)
    missing = sorted(set(order) - set(references))
    if missing:
        raise RuntimeError(f"reused ROUND66 references incomplete: {missing}")
    results = []
    for seed in seeds:
        print(f"R71 training seed {seed} ...", flush=True)
        result = execute.train_seed(cfg, ledger, plan, int(seed), order, references,
                                    stop_after_first_gate=first_gate_only)
        print(json.dumps(result, indent=1), flush=True)
        results.append(result)
        ledger.write_ledger()
    ledger.write_ledger()
    return {"status": "train_complete", "seeds": [int(s) for s in seeds], "results": results,
            "first_gate_only": bool(first_gate_only)}


def phase_ledger(raw_root: Path) -> dict:
    ledger = R71Ledger(raw_root).write_ledger()
    forward = gates.forward_ledger(raw_root)
    synth = gates.synthetic_guard(raw_root)
    return {"ledger": ledger["counts"],
            "offline_forward_ledger": {"rows": forward.rows, "calls": forward.calls,
                                       "joint_rows": forward.joint_rows, "cap": forward.cap,
                                       "remaining": forward.remaining()},
            "synthetic_optimizer_steps": {"used": synth.used(), "cap": synth.cap}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROUND71 approved fixed trial runner")
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--phase", choices=("gates", "offline-fit", "train", "eval", "stats",
                                        "ledger"), required=True)
    ap.add_argument("--seed", type=int, action="append", choices=(11, 22, 33), default=None,
                    help="restrict to one or more frozen seeds (repeatable)")
    ap.add_argument("--first-gate-only", action="store_true",
                    help="run each seed's first 4 episodes + first update + real gate, then "
                         "stop (episodes count toward 128)")
    args = ap.parse_args(argv)
    raw_root = Path(args.raw_root).resolve()
    if args.phase == "gates":
        return phase_gates(raw_root)
    if args.phase in ("eval", "stats"):
        import r71_trial_eval as trial_eval
        return trial_eval.main(["--raw-root", str(raw_root), "--phase", args.phase])
    if args.phase == "ledger":
        print(json.dumps(phase_ledger(raw_root), indent=1))
        return 0
    gate = trial.trial_gate()
    run_identity = trial.trial_run_identity()
    cfg = execute.configure_r71_trial(raw_root, gate, run_identity,
                                      model_read_approval=trial.MODEL_READ_APPROVAL)
    seeds = tuple(args.seed) if args.seed else SEEDS
    if args.phase == "offline-fit":
        result = phase_offline_fit(cfg, raw_root, seeds)
        print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=1))
        return 0
    if args.phase == "train":
        result = phase_train(cfg, raw_root, seeds, first_gate_only=args.first_gate_only)
        print(json.dumps(result, indent=1))
        return 0
    raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    raise SystemExit(main())