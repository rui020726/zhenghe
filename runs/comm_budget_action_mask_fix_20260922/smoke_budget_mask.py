"""Behaviour smoke for the budget-masked sampler: eight existing streams, zero optimizer steps.

Nothing here trains. Each launch replays one already recorded development stream with a fixed
initial controller, policy-owned timing, the locked 29 300 B credit and the original channel
settings, and reports what the mask decided against what the transport executed. The number that
matters is the closure -- requested sends equal executed sends, with no budget or capacity refusal
left for the relay to answer -- because that is the property the defect broke.

Task completions are recorded for run-integrity reasons only. This smoke cannot rank policies, and
it is not compared against the models trained on the unfixed action space.

    python smoke_budget_mask.py --mode check
    python smoke_budget_mask.py --mode run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _directory in (
    str(ROOT),
    str(ROOT / "runs" / "comm_budget_pilot_20260919"),
    str(ROOT / "runs" / "comm_stability_20260919"),
    str(ROOT / "runs" / "comm_tight_budget_20260919"),
    str(ROOT / "runs" / "comm_protocol_v2_20260920"),
    str(HERE),
):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import train_pilot  # noqa: E402
import train_protocol  # noqa: E402
import train_budget_masked as entry  # noqa: E402
from tight_common import weight_fingerprint  # noqa: E402

SMOKE_JSON = HERE / "COMM_BUDGET_ACTION_MASK_SMOKE.json"
LAUNCH_JSON = HERE / "LAUNCH_LEDGER.json"
EPISODES = HERE / "episodes"

#: The same eight development streams the send-actionability audit measured, so the actionable
#: counts can be read beside that table instead of being a new, incomparable sample.
STREAMS = (700010, 700018, 700020, 700022, 700031, 700033, 700042, 700046)
CAP = 8
BUDGET_BYTES = 29_300
INIT_SEED = 20260922


class LaunchLedger:
    """The eight-launch cap is enforced here, and a technical failure still spends a launch."""

    def __init__(self) -> None:
        self.payload = (
            json.loads(LAUNCH_JSON.read_text(encoding="utf-8"))
            if LAUNCH_JSON.exists()
            else {"cap": CAP, "starts": 0, "entries": []}
        )

    def reserve(self, kind: str, seed: int) -> None:
        if int(self.payload["starts"]) >= int(self.payload["cap"]):
            raise RuntimeError(f"the {CAP}-launch smoke budget is spent")
        self.payload["starts"] = int(self.payload["starts"]) + 1

    def record(self, kind: str, seed: int, ok: bool, detail: str) -> None:
        self.payload["entries"].append(
            {"index": len(self.payload["entries"]) + 1, "kind": kind, "seed": int(seed),
             "ok": bool(ok), "detail": detail}
        )
        entry_path = LAUNCH_JSON
        entry_path.write_text(json.dumps(self.payload, ensure_ascii=False, indent=1), encoding="utf-8")


def controller_for(seed: int) -> Any:
    """One deterministic initial controller per stream: same recipe, no trained weights."""
    torch.manual_seed(INIT_SEED)
    np.random.seed(INIT_SEED)
    return train_pilot.Controller(seed)


def run_one(seed: int) -> dict[str, Any]:
    directory = EPISODES / f"s{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    controller = controller_for(seed)
    started = time.time()
    episode = entry.run_episode(
        controller,
        seed,
        budget_bytes=BUDGET_BYTES,
        directory=directory,
        label=f"mask_smoke_s{seed}",
    )
    collector = episode["collector"]
    row = episode["row"]
    records = collector.records
    tape = entry.tape_invariants(records)
    alignment = entry.drawn_versus_logged(controller, records)
    diagnostics = entry.episode_diagnostics(collector, row)
    ledger = row["ledger"]
    initial_executable = sum(
        1 for record in records for flag in record["send_executable_initial"].values() if flag
    )
    sampled_send = sum(
        1 for record in records for target in record["actionable"] if record["actions"][target] == 1
    )
    requested = sum(1 for record in records for on in record["requested"].values() if on)
    return {
        "seed": int(seed),
        "seconds": round(time.time() - started, 2),
        "weight_fingerprint": weight_fingerprint(controller),
        "optimizer_steps": 0,
        "tapes": len(records),
        "candidates": int(collector.candidates),
        "initial_executable_candidates": int(initial_executable),
        "actionable_candidates": int(diagnostics["actionable_candidates"]),
        "masked_by_credit": int(diagnostics["masked_by_credit"]),
        "masked_by_capacity": int(diagnostics["masked_by_capacity"]),
        "sampled_send": int(sampled_send),
        "requested_by_policy": int(requested),
        "executed_policy_emissions": int(ledger["emissions_policy"]),
        "executed_emissions": int(ledger["emissions"]),
        "forced_emissions": int(ledger["emissions_forced_maxwait"]),
        "policy_budget_refusals": int(ledger["budget_refused_sends"]),
        "policy_capacity_refusals": int(ledger["capacity_refused_sends"]),
        "terminal_suppressed": int(ledger["emissions_suppressed_terminal"]),
        "credit_spent_bytes": int(row["spent_bytes"]),
        "airtime_bytes": int(ledger["airtime_bytes"]),
        "packets_lost": int(ledger["packets_lost"]),
        "closure": {
            "requested_equals_executed_policy_emissions": requested == int(ledger["emissions_policy"]),
            "no_policy_refusals": not (
                int(ledger["budget_refused_sends"]) or int(ledger["capacity_refused_sends"])
            ),
            "actionable_plus_forced_plus_masked_equals_candidates": tape["passed"],
        },
        "tape_invariants": tape,
        "likelihood_alignment": alignment,
        "diagnostics": diagnostics,
        "task_outcome_for_integrity_only": {
            "on_time": int(row["task"]["on_time"]),
            "targets": int(row["task"]["targets"]),
            "steps_run": train_protocol.actual_steps(row["artefacts"]["lifecycle_log"]),
        },
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "run"), default="check")
    args = parser.parse_args(argv)
    plan = {
        "streams": list(STREAMS),
        "episodes_per_stream": 1,
        "launch_cap": CAP,
        "budget_bytes": BUDGET_BYTES,
        "timing_mode": "policy_owned_timing",
        "trainer_tag": entry.TRAINER_TAG,
        "action_mask_schema": entry.ACTION_MASK_SCHEMA,
        "tape_schema": entry.TAPE_SCHEMA,
        "initial_weight_seed": INIT_SEED,
        "optimizer_steps": 0,
    }
    if args.mode == "check":
        print(json.dumps({**plan, "launches_used_so_far": int(LaunchLedger().payload["starts"])},
                         ensure_ascii=False, indent=1))
        return 0
    ledger = LaunchLedger()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed in STREAMS:
        if int(ledger.payload["starts"]) >= CAP:
            break
        ledger.reserve("smoke", seed)
        try:
            row = run_one(seed)
            rows.append(row)
            ledger.record("smoke", seed, True, f"tapes={row['tapes']} candidates={row['candidates']}")
        except Exception as error:  # a technical failure still spends one of the eight launches
            failures.append({"seed": int(seed), "error": f"{type(error).__name__}: {error}"})
            ledger.record("smoke", seed, False, f"{type(error).__name__}: {error}")
    payload = {
        "schema": "comm-budget-action-mask-smoke/1",
        "plan": plan,
        "launches": json.loads(LAUNCH_JSON.read_text(encoding="utf-8"))["starts"],
        "episodes_run": len(rows),
        "technical_failures": failures,
        "rows": rows,
        "closure": {
            "all_requested_equal_executed": all(
                row["closure"]["requested_equals_executed_policy_emissions"] for row in rows
            ),
            "all_policy_refusals_zero": all(row["closure"]["no_policy_refusals"] for row in rows),
            "all_tape_invariants_pass": all(row["tape_invariants"]["passed"] for row in rows),
            "max_log_prob_gap": max(
                (row["likelihood_alignment"].get("max_abs_gap", 0.0) for row in rows), default=0.0
            ),
            "optimizer_steps_total": sum(int(row["optimizer_steps"]) for row in rows),
        },
        "not_measured": [
            "any performance ranking: no completions are compared across arms",
            "trained behaviour: this smoke runs fixed initial weights for one episode per stream",
            "H=20 spillover, which stays frozen",
        ],
    }
    (HERE / "COMM_BUDGET_ACTION_MASK_SMOKE.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({**payload["closure"], "episodes_run": payload["episodes_run"],
                      "launches": payload["launches"],
                      "failures": len(failures)}, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
