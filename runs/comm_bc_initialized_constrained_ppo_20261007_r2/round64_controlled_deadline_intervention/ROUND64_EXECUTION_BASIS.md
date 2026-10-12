# ROUND64 Execution Basis

## Authorization and Boundary

This package executes the user's fixed ROUND64 local deadline-by-action experiment. It authorizes at most 10 planned episodes plus 2 technical-reserve attempts (12 total slots), all serially through `r64_guard.py`. A failed, interrupted, or disqualified attempt consumes its slot. No further cases or retries are authorized beyond that cap.

No training, optimizer update, PPO replay, training-state/RNG restore, or checkpoint write is permitted. Only read-only loading of the fixed model and normal environment episodes are allowed. No push, merge, deployment, or follow-on experiment.

## Frozen Inputs

- Branch: `exp/round64-controlled-deadline-intervention`
- Pre-freeze base HEAD: `252f2fc4d6101693ef363c9969090eb1ba0bb6a1`
- Model: R60 compliant seed 33, greedy, ep128; exact path, file SHA256 and parameter SHA256 are in `ROUND64_FREEZE.json`.
- Scene: `1700027`; focus target: `piloc-e1-target-5`.
- World: inherited R60 frozen configuration; horizon 1000, six goals, 29,300-byte communication budget, ACK disabled, same search/allocation checkpoints and greedy tie rule.
- Deadline intervention: after the original profile is generated, add only the frozen delta to the focus target's `deadline_offset`; profile generation and its RNG consumption remain unchanged. Deltas are `-30, 0, +30, +60`.
- Action intervention: at the target's first legal, payable free one-shot decision, after sequential budget masking and before emission/latching, force SEND or SKIP once; continue with frozen greedy policy thereafter. No free send, bypass, retry, or altered candidate ordering.
- Main output root: `raw/`; append-only attempts and slots live beneath this root.

## Fixed Run Order

1. `L`: delta 0, unmodified greedy path.
2. `L0`: delta 0, intervention installed but override disabled.
3. For each delta in `-30, 0, +30, +60`, run `FORCE_SEND`, then `FORCE_SKIP`.

The first two runs must pass exact behavior/tape/lifecycle comparison against each other and the R60 focus baseline before any matrix episode is allowed. The remaining eight are fixed; no result-dependent selection is permitted. This yields 10 normal attempts and at most 2 reserve retries.

## Accounting and Qualification

The persistent slot is atomically consumed before the actual environment builder call. Builder invocation/return, `run()` entry/return/failure, steps, and result qualification are separate counters. Equal outputs do not deduplicate executions; copying artifacts does not create an execution. Qualification does not refund a slot.

L/L0 are a validation gate, not independent scientific samples. Each forced pair is interpreted as a paired intervention in one fixed scene realization, not as identical downstream packet loss. The report will separate target outcome, full on-time target set, total tasks, costs, and communication/task lifecycle. The four deadlines may be compared descriptively; no inference of general necessity, safe deletion, noninferiority, or deployment is authorized.

The feature audit preserves all 19 inputs and may describe observed distinctions or exact-vector conflicts only. It does not fit a policy or establish feature sufficiency.

## Pre-Run Evidence

Before any real episode: zero-environment preflight, source binding, read-only model identity, guard tests, and exact run order are frozen and locally committed. The final preflight and machine-readable identities are recorded in `ROUND64_FREEZE.json` and `ROUND64_PREFLIGHT_10.json`.

