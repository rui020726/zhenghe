# ROUND74 - PHASE A Resource Wiring Acceptance

- Branch: `exp/round74-constrained-early-head-adaptation`
- HEAD: `accc47bab5b5c9aaa15b10e06919f29b0240e03e`
- Scope: resource wiring + zero-environment acceptance ONLY. This is NOT an experiment, NOT training, NOT evaluation.

## What PHASE A is / is not
PHASE A installs the R74 cumulative-resource guards and proves, with zero scientific side effects,
that the meters work. It does NOT load the task model, run forward/backward, run PPO, or build a
real environment. PHASE B (real CLI/controller/collector/PPO, 384 train + 192 eval + 3 focal) is
NOT authorized here and awaits a separate decision-end continuation.

## Guard design (r74_guard.py)
Single durable append-only total ledger (`raw/runtime/r74_total_ledger.jsonl`, fsync). Every guard
charges BEFORE the physical work, never refunds on failure/interruption/quarantine/retry, and
persists on disk (not reset by restart, subdir change, or backup copy). Independent units:
- env-construction slot: normal 579 + reserve 21, hard 600
- PHYSICAL optimizer.step: normal 12 + reserve 4, hard 16 (denominated in real torch optimizer.step calls, the R73 defect fixed; a PPO call is a separate record-only unit)
- offline trunk TARGET forward rows: normal 3000 + reserve 3000, hard 6000, per-batch <= 1024; rows = actual input vectors whose last dim == 19 (never n_records*epochs); JOINT records in a separate counter
- PPO update call / JOINT record: record-only, separate units

## Spy suite (test_r74_guard.py -> run_tests) : 17/17 PASS, ALL_PASS
Pure synthetic + throwaway fixtures under `_guard_test/` (self-cleaned). Never touches the canonical
ledger and creates no machine licence. Covers: physical-step caps (12 normal ok, 13th refused,
reserve to hard 16, 17th refused), charge-before-work no-refund, restart persistence, PPO-call !=
physical-step independence, offline target-row metering + per-batch 1024 + joint-separate, single
total ledger records all charges, env-slot caps wired.

## Real physical consumption: exactly 2 toy Adam.step
Constant scalar `p=0.0`, explicit gradient `1.0` (no forward/model/env), Adam lr=0.1. Each step
charged BEFORE opt.step(); an independent sub-limit of 2 (own counter, distinct kind
`r74-phase-a-toy-sublimit/1`, mirrored to the single total ledger) is charged first.
- call 1: 0.0 -> -0.09999999 (sublimit used 1, global used 1)
- call 2: -0.09999999 -> -0.19999999 (sublimit used 2, global used 2)
- 3rd attempt: refused by the sub-limit at used_normal=2/2 BEFORE the global charge and BEFORE
  opt.step(). Param unchanged (-0.19999999), Adam state rows unchanged (1). Global hard-16 stayed at 2.

## Actual consumption (authoritative, canonical durable counters)
- physical_optimizer_steps_real = 2
- environment_constructions = 0
- offline_forward_target_rows = 0
- ppo_update_calls = 0
- joint_records = 0
Zero scientific side effects: env/forward/ppo/joint all 0. Read-only CPU forward count this phase = 0.

## Code freeze (sha256)
- r74_guard.py 9a6b06374fc5b56b6d8077dc6aa332444533198aca0967cb36b867795af614d7
- test_r74_guard.py e522d66bffa190a5da1a0f0d4212e8033b1d850ab4011625569667b9b3fe66a0
- r74_phase_a.py 7452687d6eda4b7d576d6dc7d6e6501592ab86cbfbea137d986a139b52e6d2d0

## Protected (not modified / not staged)
- runs/comm_bc_initialized_constrained_ppo_20261007_r2/ROUND46_CURRENT_TASK.json (cf3ad198...)
- runs/comm_bc_initialized_constrained_ppo_20261007_r2/ROUND46_R2_V6_STOP_STATE.json (93a2f69f...)

Machine-readable: ROUND74_PHASE_A_ACCEPTANCE.json, ROUND74_FREEZE.json.