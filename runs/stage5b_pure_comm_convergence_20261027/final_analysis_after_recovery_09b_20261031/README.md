# recovery_09b: post-training analysis (2026-10-31)

## Training outcome

- Root: `recovery_09b/revision_01` (descendant of the completed R8 parent).
- Terminal receipt: `execution/receipts/recovery-train_terminal_recovery-train-17212-1793324221731820600.json`
  - status `FINITE_RECOVERY_STAGE_COMPLETE`, returncode 0
  - 18,840/18,840 Worlds; 0 unused slots; 0 unclaimed slots; six runs `ok`
  - task-certified runs: T1/91001, T2/91002, T2/91003 (3/6); `convergence_complete=false`,
    `goal_complete=false` (as recorded by the runner; stage-cap exhaustion is not convergence)
- Final checkpoints: T1/91001 `milestone_5004.pth` (cursor 5,004 / 5,000 completed);
  the other five `milestone_5000.pth`.

## Loss / optimisation curves

Script: `plot_loss_curves.py` (offline; reads only completed telemetry, no project imports).
Outputs: `loss_curves.png`, `loss_curves.svg`, `loss_updates.csv` (4,710 update rows),
`summary.json` (input paths, byte sizes and SHA-256). Thin lines are raw updates; thick lines
are a 25-update trailing mean. Diagnostic panels: policy surrogate, value loss (task),
value loss (cost), policy entropy, approx KL, gradient-norm mean.

Final 25-update means:

| Run | Updates | Episodes | Surrogate | V(task) | V(cost) | Entropy |
|---|---:|---|---:|---:|---:|---:|
| T1/91001 | 710 | 2164–5000 | −0.0317 | 0.1882 | 0.00558 | 0.0719 |
| T1/91002 | 710 | 2160–4996 | −0.0348 | 0.1758 | 0.00535 | 0.0830 |
| T1/91003 | 800 | 1800–4996 | −0.0313 | 0.1671 | 0.00435 | 0.0742 |
| T2/91001 | 800 | 1800–4996 | −0.0189 | 0.1844 | 0.00423 | 0.0646 |
| T2/91002 | 800 | 1800–4996 | −0.0176 | 0.1394 | 0.00577 | 0.0812 |
| T2/91003 | 890 | 1440–4996 | −0.0044 | 0.1201 | 0.00431 | 0.0786 |

Reading note: value losses trend down over the stage and the surrogate settles near zero with
a small negative offset; entropy stays roughly flat (0.06–0.08). These plots are descriptive
and do not evaluate the registered convergence rule.

## Final DEV_REUSED evaluation

Directory: `../final_eval_after_recovery_09b_20261031/execution`.
Harness: `run_final_eval.py` (adapted from the 2026-10-28 intermediate harness; the only
changes are final checkpoint pins, R9b identity pin, labels and the recorded authorization).
48 Worlds = 6 models × 8 DEV scenes (2301901–2301908), eval world seed 20261026, greedy actor,
1,000 steps, 24 tasks, 58,600-byte budget. 48/48 completed, 0 failures, 48 model loads.

| Model | Checkpoint | Tasks | On-time | Success rate |
|---|---|---:|---:|---:|
| T1/91001 | 5004 | 192 | 57 | 29.69% |
| T1/91002 | 5000 | 192 | 82 | 42.71% |
| T1/91003 | 5000 | 192 | 65 | 33.85% |
| T2/91001 | 5000 | 192 | 79 | 41.15% |
| T2/91002 | 5000 | 192 | 104 | 54.17% |
| T2/91003 | 5000 | 192 | 97 | 50.52% |
| **All** | — | **1,152** | **484** | **42.01%** |

Comparison with the 2026-10-28 intermediate run (different checkpoints: 1440/1440/1080/1080/1080/720;
same 48 Worlds): overall 394/1,152 = 34.20% → 484/1,152 = 42.01%. Per model:
T1/91001 1.04% → 29.69%, T1/91002 1.56% → 42.71%, T1/91003 44.79% → 33.85%,
T2/91001 49.48% → 41.15%, T2/91002 55.73% → 54.17%, T2/91003 52.60% → 50.52%.
The overall rise is driven by the two T1 seeds that were near zero at the intermediate point;
the other four seeds are flat-to-lower. With 8 scenes per model (192 tasks) no confidence
interval was computed and no paired significance is claimed here.

Scope caveats, carried from the harness: label `FINAL_DEV_REUSED`, `not_formal_validation=true`,
`holdout_sealed=true`, `goal_complete=false`, original NO_GO preserved. This is not formal
validation on the sealed holdout and it does not overturn the prior NO_GO.
