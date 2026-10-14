#!/usr/bin/env python
"""ROUND73 approved fixed trial: the ONE runner (gates -> init -> train -> eval -> stats).

Every phase stands up the SAME live ROUND73 runtime through the ROUND73 binding (the real
ROUND68 executor with quota/version/checkpoint rebound to ROUND73 and the pure-cost actor
objective installed), behind the archived trial licence and an explicit model-read approval.
Phases are resumable: a rolling checkpoint, the committed batch records and the durable
claim/slot/permit regime decide where a seed continues, so an interrupted run never re-runs a
committed update.

Quota (hard, no refund): 579 normal (384 training + 192 model eval + 3 focal) + 21 reserve,
hard cap 600 environment constructions; offline trunk input rows <= 6,000 (ROUND73 runs no
offline fit, so the planned use is 0); synthetic optimizer steps <= 8 (gates only).
Historical R58/R60/R66/R68/R71 constructions are NOT ROUND73 consumption.
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

import r73_trial_binding as trial                                       # noqa: E402
import r73_checkpoint as ckpt                                           # noqa: E402
import r73_trial_execute as execute                                     # noqa: E402
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
TASK_ID = trial.R73_TRIAL_TASK_ID
BRANCH = trial.R73_TRIAL_BRANCH
NORMAL_CAP = execute.NORMAL_CAP
RESERVE_CAP = execute.RESERVE_CAP
HARD_CAP = execute.HARD_CAP
HISTORICAL_PROBE = execute.HISTORICAL_PROBE

_read_json = execute._read_json
_write_json = execute._write_json


class R73Ledger:
    """The ROUND73 claim/slot/quota ledger under the ROUND73 raw root (never the R71 one)."""

    def __init__(self, raw: Path) -> None:
        self.raw = Path(raw)
        self.claims_dir = self.raw / "launch_claims"
        self.slots_dir = self.raw / "launch_slots"
        self.runtime_dir = self.raw / "runtime"
        for d in (self.claims_dir, self.slots_dir / "normal", self.slots_dir / "reserve",
                  self.runtime_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.runtime_dir / "ROUND73_LEDGER.json"

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
                "hard_headroom": HARD_CAP - actual}

    def write_ledger(self) -> dict:
        counts = self.counts()
        if counts["actual_environment_constructions"] > HARD_CAP:
            raise RuntimeError(f"ROUND73 ledger: {counts['actual_environment_constructions']} "
                               f"constructions exceed the hard cap {HARD_CAP}")
        ledger = {"kind": "round73-ledger/1", "task_id": TASK_ID, "branch": BRANCH,
                  "run_id": trial.TRIAL_RUN_ID,
                  "training_version": ckpt.R73_TRAINING_VERSION,
                  "actor_objective": ckpt.R73_ACTOR_OBJECTIVE,
                  "consumption": {
                      "real_environment_constructions":
                          counts["actual_environment_constructions"],
                      "normal_claims": counts["normal_claims"],
                      "reserve_claims": counts["reserve_claims"],
                      "complete_claims": counts["complete"],
                      "failed_technical_claims": counts["failed_technical"],
                      "historical_non_plan_probe": HISTORICAL_PROBE},
                  "caps": {"hard_cap": HARD_CAP, "normal_cap": NORMAL_CAP,
                           "reserve_cap": RESERVE_CAP},
                  "headroom": {"hard": counts["hard_headroom"],
                               "normal": counts["normal_headroom"],
                               "reserve": counts["reserve_headroom"]},
                  "note": ("no R58/R60/R66/R68/R71 consumption is carried into ROUND73; ROUND73 "
                           "is a fresh independent budget and borrows no earlier remainder")}
        _write_json(self.ledger_path, ledger)
        return ledger


# The gate suites.  The first two are pure computation (no guard charge at all); the third
# drives the LIVE trainer.ppo_update path and therefore spends the round's whole synthetic
# optimizer-step budget, which is why it runs at most once against the real ledgers.
ZERO_ENV_SUITES = ("test_r73_pure_cost_gates.py", "test_r73_binding_gates.py")
QUALIFICATION_SUITE = "test_r73_qualification_gates.py"
QUALIFICATION_RUN_RECORD = ("runtime/synthetic_gates/"
                            "R73_QUALIFICATION_SUITE_RUN.json")


def _run_gate_script(script: str) -> dict:
    """Run one gate suite in its own interpreter and capture its verdict tail."""
    import subprocess
    proc = subprocess.run([sys.executable, str(HERE / script)], capture_output=True, text=True)
    combined = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()
    tail = [line for line in combined
            if line.startswith(("OK", "FAILED", "Ran ", "ERROR"))
            or "module loaded" in line][-4:]
    return {"returncode": int(proc.returncode), "tail": tail,
            "stderr_tail": (proc.stderr or "")[-2000:]}


def _qualification_run_record(raw_root: Path) -> Path:
    return Path(raw_root) / QUALIFICATION_RUN_RECORD


def _qualification_prior_run(raw_root: Path) -> dict:
    """Whether the REAL-path qualification suite already ran green on THIS raw root.

    That suite charges the round's ENTIRE 8-charge synthetic optimizer-step budget: one charge
    per live ``trainer.ppo_update`` call, booked BEFORE the call and never refunded.  It may
    therefore run against the real persisted ledgers exactly once.  Re-running it is not merely
    wasteful - the guard refuses the first charge and the gate would fail on a budget that is
    already spent.  The durable proof is the run record this phase writes after a green run,
    cross-checked against the persisted guard state; nothing here is hard-coded.
    """
    import r73_trial_gates as gates
    record_path = _qualification_run_record(raw_root)
    synth = gates.synthetic_guard(raw_root)
    guard = {"synthetic_optimizer_steps_used": int(synth.used()),
             "synthetic_optimizer_steps_cap": int(synth.cap)}
    if not record_path.is_file():
        return {"already_run": False,
                "reason": "no durable qualification run record for this raw root yet",
                "guard": guard}
    record = _read_json(record_path)
    if int(record.get("returncode", -1)) != 0:
        return {"already_run": False,
                "reason": "the recorded qualification run did not pass",
                "record": record, "guard": guard}
    if int(synth.used()) < int(synth.cap):
        return {"already_run": False,
                "reason": (f"the persisted synthetic guard reads {synth.used()} of "
                           f"{synth.cap} charges, so the recorded run is not the run that "
                           f"spent the budget"),
                "record": record, "guard": guard}
    return {"already_run": True,
            "reason": ("a green qualification run is durably recorded and this round's "
                       "synthetic optimizer-step budget is spent, so the suite is reused "
                       "rather than re-run"),
            "record": record, "guard": guard}


def _persisted_gate_consumption(raw_root: Path) -> dict:
    """Read this round's OWN persisted ledgers.  No number here is hard-filled."""
    import r73_trial_gates as gates
    forward = gates.forward_ledger(raw_root)
    synth = gates.synthetic_guard(raw_root)
    counts = R73Ledger(raw_root).counts()
    runtime = Path(raw_root) / "runtime"

    def _journal_lines(name: str) -> int:
        path = runtime / name
        if not path.is_file():
            return 0
        return len([ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()])

    return {
        "environment_constructions": int(counts["actual_environment_constructions"]),
        "environment_constructions_note": ("normal + reserve launch claims booked under THIS "
                                           "round's claim root plus the historical non-plan "
                                           "probe, read from the claim files themselves"),
        "model_forwards": int(forward.rows),
        "offline_trunk_forward_rows": {"rows": int(forward.rows), "calls": int(forward.calls),
                                       "joint_rows": int(forward.joint_rows),
                                       "cap": int(forward.cap),
                                       "remaining": int(forward.remaining())},
        "optimizer_steps": int(synth.used()),
        "optimizer_steps_note": ("synthetic gate steps only; a real training optimizer step "
                                 "is never charged to this guard"),
        "ppo_updates": _journal_lines("ppo_update_calls.jsonl"),
        "ppo_updates_note": ("committed real PPO update calls appended to this round's PPO "
                             "journal; synthetic gate lines are quarantined out of it"),
        "checkpoint_writes": _journal_lines("checkpoint_writes.jsonl"),
    }


def phase_gates(raw_root: Path) -> dict:
    """The zero-environment gate suite: objective, binding, and the REAL-path qualification.

    The qualification suite is idempotent against the real ledgers: once it has run green and
    spent the round's synthetic optimizer-step budget, a later ``--phase gates`` reuses that
    durable record instead of re-charging a budget that is already gone.
    """
    out = {}
    for script in ZERO_ENV_SUITES:
        result = _run_gate_script(script)
        out[script] = {"returncode": result["returncode"], "tail": result["tail"]}
        if result["returncode"] != 0:
            raise RuntimeError(f"ROUND73 gate {script} failed: {result['stderr_tail']}")
    prior = _qualification_prior_run(raw_root)
    if prior["already_run"]:
        out[QUALIFICATION_SUITE] = {
            "returncode": 0,
            "tail": ["reused the durable green qualification run; this round's synthetic "
                     "optimizer-step budget is already spent"],
            "reused_prior_run": True,
            "prior_run": prior,
        }
    else:
        result = _run_gate_script(QUALIFICATION_SUITE)
        out[QUALIFICATION_SUITE] = {"returncode": result["returncode"],
                                     "tail": result["tail"],
                                     "reused_prior_run": False}
        if result["returncode"] != 0:
            raise RuntimeError(
                f"ROUND73 gate {QUALIFICATION_SUITE} failed: {result['stderr_tail']}"
                f" | persisted guard {prior['guard']} - charges are booked before execution "
                f"and never refunded, so a failure here still spent them")
        _write_json(_qualification_run_record(raw_root), {
            "kind": "round73-qualification-suite-run/1", "task_id": TASK_ID,
            "branch": BRANCH, "script": QUALIFICATION_SUITE,
            "returncode": 0, "tail": result["tail"],
            "consumption": _persisted_gate_consumption(raw_root)})
        prior = _qualification_prior_run(raw_root)
    receipt = {"kind": "round73-prelaunch-gates/1", "task_id": TASK_ID, "branch": BRANCH,
               "run_id": trial.TRIAL_RUN_ID,
               "training_version": ckpt.R73_TRAINING_VERSION,
               "actor_objective": ckpt.R73_ACTOR_OBJECTIVE,
               "consumption": _persisted_gate_consumption(raw_root),
               "suites": out, "qualification": prior, "status": "passed"}
    _write_json(raw_root / "ROUND73_PRELAUNCH_GATES.json", receipt)
    print(json.dumps(receipt, indent=1))
    return receipt


def phase_train_cli(cfg, raw_root: Path, args) -> int:
    seeds = tuple(args.seed) if args.seed else SEEDS
    result = phase_train(cfg, raw_root, seeds, first_gate_only=args.first_gate_only)
    print(json.dumps(result, indent=1))
    return 0


def phase_train(cfg, raw_root: Path, seeds, *, first_gate_only: bool) -> dict:
    ledger = R73Ledger(raw_root)
    order = [int(x) for x in cfg["common"].scenario_order()]
    if len(order) != execute.EPISODES_PER_SEED:
        raise RuntimeError(f"training order length {len(order)} != {execute.EPISODES_PER_SEED}")
    plan_path = raw_root / "ROUND73_FIXED_PLAN.json"
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
        print(f"R73 training seed {seed} ...", flush=True)
        result = execute.train_seed(cfg, ledger, plan, int(seed), order, references,
                                    stop_after_first_gate=first_gate_only)
        print(json.dumps(result, indent=1), flush=True)
        results.append(result)
        ledger.write_ledger()
    ledger.write_ledger()
    return {"status": "train_complete", "seeds": [int(s) for s in seeds], "results": results,
            "first_gate_only": bool(first_gate_only)}


def phase_ledger(raw_root: Path) -> dict:
    ledger = R73Ledger(raw_root).write_ledger()
    import r73_trial_gates as gates
    forward = gates.forward_ledger(raw_root)
    synth = gates.synthetic_guard(raw_root)
    return {"ledger": ledger["consumption"],
            "caps": ledger["caps"], "headroom": ledger["headroom"],
            "offline_forward_ledger": {"rows": forward.rows, "calls": forward.calls,
                                       "joint_rows": forward.joint_rows, "cap": forward.cap,
                                       "remaining": forward.remaining()},
            "synthetic_optimizer_steps": {"used": synth.used(), "cap": synth.cap}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROUND73 approved fixed trial runner")
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--phase", choices=("gates", "acceptance", "init", "train", "eval", "stats",
                                        "ledger"), required=True)
    ap.add_argument("--seed", type=int, action="append", choices=(11, 22, 33), default=None,
                    help="restrict to one or more frozen seeds (repeatable)")
    ap.add_argument("--first-gate-only", action="store_true",
                    help="run each seed's first 4 episodes + first update + real gate, then "
                         "stop (episodes count toward 128)")
    args = ap.parse_args(argv)
    raw_root = Path(args.raw_root).resolve()
    if args.phase == "gates":
        return 0 if phase_gates(raw_root)["status"] == "passed" else 1
    if args.phase in ("eval", "stats"):
        import r73_trial_eval as trial_eval
        return trial_eval.main(["--raw-root", str(raw_root), "--phase", args.phase])
    if args.phase == "ledger":
        print(json.dumps(phase_ledger(raw_root), indent=1))
        return 0
    if args.phase == "acceptance":
        import r73_trial_execute as ex
        cfg = ex.configure_r73_trial(raw_root, trial.trial_gate(), trial.trial_run_identity(),
                                     model_read_approval=trial.MODEL_READ_APPROVAL)
        receipt = ex.zero_env_cli_acceptance(cfg)
        _write_json(raw_root / "runtime" / "ROUND73_ZERO_ENV_ACCEPTANCE.json", receipt)
        print(json.dumps(receipt, indent=1))
        return 0 if receipt["status"] == "passed" else 1
    if args.phase == "init":
        import r73_trial_execute as ex
        cfg = ex.configure_r73_trial(raw_root, trial.trial_gate(), trial.trial_run_identity(),
                                     model_read_approval=trial.MODEL_READ_APPROVAL)
        receipt = ex.stage_init(cfg)
        _write_json(raw_root / "runtime" / "ROUND73_INIT_STAGING.json", receipt)
        print(json.dumps({k: v for k, v in receipt.items() if k != "seeds"}, indent=1))
        return 0
    if args.phase == "train":
        import r73_trial_execute as ex
        cfg = ex.configure_r73_trial(raw_root, trial.trial_gate(), trial.trial_run_identity(),
                                     model_read_approval=trial.MODEL_READ_APPROVAL)
        return phase_train_cli(cfg, raw_root, args)
    raise SystemExit(f"unknown phase {args.phase}")


if __name__ == "__main__":
    raise SystemExit(main())