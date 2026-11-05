"""P4: short learnability test at a genuinely binding budget (N=12, fixed 58,600 B).

Why N=12 and not N=6
--------------------
At N=6 the frozen 58,600 B covers 23 of 24 targets, so "broadcast everything" is affordable
and `normal_immediate` is already at the ceiling.  P2 measured exactly that: 23 emissions in
every one of the 8 worlds.  A policy cannot learn target *selection* in a regime where
selecting is unnecessary.

At N=12 a send costs 12 subscribers' worth of payload, so the same 58,600 B buys 11-12 sends
against 24 targets.  P2's own `normal_immediate` arm at N=12 measured 11-12 emissions and
8/8 worlds at the budget cap.  That is the pre-existing, same-caliber, same-revision strong
rule baseline this run is scored against -- no new baseline replay is needed.

The P2 comparability caveat is inherited and disclosed: at N=12 the roster is 12 robots
(`rescue_per_species = 4`), so labour moves with fan-out.  On-time differences therefore
confound "better target selection" with "more robots to execute".  The pass/fail statement
this run may make is therefore about the *cost-vs-on-time trade* against the strong rule at
identical N, not about selection quality in isolation.

What is switched on
-------------------
  CREDIT_MODE      causal_task_id   credit re-keyed onto the decision that sent the target
  ADVANTAGE_MODE   causal_step      A = r - V(s); no multi-step GAE discount
  CAUSAL_GAMMA     0.0              the registered bandit form
  SCALE_INVARIANT  norm_only        #17 -> n/6 and #14 -> backlog/24 on the way in
  N                12               comm_subscribers="all", rescue_per_species=4

The feature transform is installed on `controller.inner.trunk`, so sampling, the cost
critic and the PPO update all see the same rewritten row.  `p4_unitcheck.py` proves the
importance ratio at update 0 is 1 under this seam; with the earlier `forward`-shadow
placement it was 1.589 and every update would have been silently clipped.

One arm, one process, one tag.  A second attempt must use a new --tag: a world directory is
never reopened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
STAGE = BASE.parent
ROOT = STAGE / "recovery_12" / "revision_01"
CHECKPOINT = (STAGE / "recovery_09b" / "revision_01" / "execution" / "checkpoints"
              / "T2_s91002" / "milestone_5000.pth")
IDENTITY = (STAGE / "recovery_09b" / "revision_01" / "protocol"
            / "RECOVERY_EXPERIMENT_IDENTITY.json")

ARM, SEED, START_EPISODE = "T2", 91002, 5000
SCENES = tuple(range(2301901, 2301909))

#: N=12 seam, identical to P2/P3 so the strong-rule baseline stays strictly isomorphic.
N_CONFIG = {"comm_subscribers": "all", "rescue_per_species": 4,
            "expectation_overrides": {"rescue_robots": 12}}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--tag", default="p4_tight_n12")
    ap.add_argument("--save-every", type=int, default=100,
                    help="checkpoint interval inside the run; 0 saves only the final one")
    args = ap.parse_args()

    sys.dont_write_bytecode = True
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    from stage5b_comm import constants as C
    from stage5b_comm import episode, trainer, world
    from stage5b_comm.paths import Layout

    # Every switch is set before the first world is built: configuration_fields reads
    # world.SCALE_* at world-construction time and the trainer reads C.* at update time.
    world.SCALE_SUBSCRIBERS = N_CONFIG["comm_subscribers"]
    world.SCALE_RESCUE_PER_SPECIES = N_CONFIG["rescue_per_species"]
    world.SCALE_EXPECTATION_OVERRIDES = dict(N_CONFIG["expectation_overrides"])
    C.SCALE_INVARIANT = "norm_only"
    C.CREDIT_MODE = "causal_task_id"
    C.ADVANTAGE_MODE = "causal_step"
    C.CAUSAL_GAMMA = 0.0
    C.CREDIT_RULE = "first_accepted_send"

    out = root / "execution" / args.tag / ARM
    if (out / "TRAIN_SUMMARY.json").exists():
        raise SystemExit("%s already holds a completed run; use a new --tag" % out)
    out.mkdir(parents=True, exist_ok=True)
    layout = Layout(root)
    layout.execution = out
    layout.world_logs = out / "world_logs"
    layout.rows = out / "rows"
    layout.ledger = out / "ledger"
    layout.receipts = out / "receipts"
    for path in (layout.world_logs, layout.rows, layout.ledger, layout.receipts):
        path.mkdir(parents=True, exist_ok=True)
    episode.ensure_import_paths(layout)
    episode.native_import_preflight(layout)

    identity_sha = sha256(IDENTITY)
    if not CHECKPOINT.exists():
        raise SystemExit("checkpoint missing: %s" % CHECKPOINT)
    controller, optimizer, state = trainer.load_checkpoint(
        CHECKPOINT, arm=ARM, identity_sha256=identity_sha,
        expected_seed=SEED, expected_episode=START_EPISODE)
    state = dict(state)
    start_params = controller.parameter_summary()["parameter_sha256"]

    updates, episodes_log, checkpoints = [], [], []
    update_index = START_EPISODE // C.EPISODES_PER_UPDATE
    pending = []
    started = time.time()
    indices = list(range(START_EPISODE, START_EPISODE + args.episodes))
    try:
        for position, index in enumerate(indices):
            scene = C.scene_for(SEED, index)
            row = episode.run_training_world(layout, ARM, SEED, index, scene, controller)
            pending.append(row)
            audit = row.get("scale_invariant") or {}
            episodes_log.append({
                "episode_index": int(index), "scene": int(scene),
                "on_time": int(row["on_time"]), "spent_bytes": int(row["spent_bytes"]),
                "emissions": int(row["emissions"]),
                "generated_goals": int(row.get("generated_goals", 0)),
                "subscribers_live": row.get("subscribers_live"),
                "masked_by_credit_total": row.get("masked_by_credit_total"),
                "first_masked_credit_tick": row.get("first_masked_credit_tick"),
                "causal_moved": (row.get("causal_credit_audit") or {}).get("moved"),
                "causal_unmatched": (row.get("causal_credit_audit") or {}).get("unmatched"),
                "scale_mode": audit.get("mode"),
                "scale_matrix_calls": audit.get("matrix_calls"),
                "seconds": row["seconds"],
            })
            if len(pending) < C.EPISODES_PER_UPDATE:
                continue
            update = trainer.ppo_update(controller, optimizer, pending,
                                        arm=ARM, update_index=update_index)
            update["update_index"] = update_index
            update["toward_episode"] = int(index)
            updates.append(update)
            update_index += 1
            pending = []
            done = position + 1
            print("[%s] ep %d/%d  on_time=%d/%d bytes=%d sends=%d  surrogate=%.4f "
                  "v_task=%.4f kl=%.5f"
                  % (ARM, done, args.episodes, row["on_time"], row.get("generated_goals", 0),
                     row["spent_bytes"], row["emissions"],
                     update.get("policy_surrogate", float("nan")),
                     update.get("value_loss_task", float("nan")),
                     update.get("approx_kl", float("nan"))), flush=True)
            if args.save_every and done % args.save_every == 0 and done < args.episodes:
                final_so_far = START_EPISODE + done
                target = out / ("milestone_%d.pth" % final_so_far)
                if not target.exists():
                    state["episode"] = final_so_far
                    state["arm"] = ARM
                    state["training_seed"] = SEED
                    trainer.save_checkpoint(target, controller, optimizer, state,
                                            identity_sha256=identity_sha)
                    checkpoints.append({"episode": final_so_far, "path": str(target),
                                        "sha256": sha256(target)})
                    print("  checkpoint", target.name, flush=True)
    finally:
        elapsed = round(time.time() - started, 1)
        summary = {
            "schema": "stage5b-p4-tight-budget-arm/1", "arm": ARM, "seed": SEED,
            "n_subscribers": 12, "budget_bytes": C.BUDGET_BYTES,
            "seam": dict(N_CONFIG),
            "credit_mode": C.CREDIT_MODE, "advantage_mode": C.ADVANTAGE_MODE,
            "credit_rule": C.CREDIT_RULE, "causal_gamma": C.CAUSAL_GAMMA,
            "scale_invariant": C.SCALE_INVARIANT,
            "cost_pacing": C.COST_PACING, "normalize_advantage": C.NORMALIZE_ADVANTAGE,
            "start_checkpoint": str(CHECKPOINT), "start_checkpoint_sha256": sha256(CHECKPOINT),
            "identity_sha256": identity_sha,
            "start_episode": START_EPISODE,
            "completed_episodes": len(episodes_log),
            "end_episode": START_EPISODE + len(episodes_log),
            "updates": len(updates), "wall_seconds": elapsed,
            "start_parameter_sha256": start_params,
            "end_parameter_sha256": controller.parameter_summary()["parameter_sha256"],
            "checkpoints": checkpoints,
            "episodes": episodes_log, "update_history": updates,
        }
        (out / "TRAIN_SUMMARY.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n")
        final = START_EPISODE + len(episodes_log)
        if episodes_log and not (out / ("milestone_%d.pth" % final)).exists():
            state["episode"] = final
            state["arm"] = ARM
            state["training_seed"] = SEED
            trainer.save_checkpoint(out / ("milestone_%d.pth" % final),
                                    controller, optimizer, state,
                                    identity_sha256=identity_sha)
        print(json.dumps({k: v for k, v in summary.items()
                          if k not in ("episodes", "update_history", "checkpoints")},
                         ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
