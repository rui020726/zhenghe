# ROUND64 Execution Basis

## Authorization and Boundary

This package executes the user's fixed ROUND64 local deadline-by-action experiment. Initial `/2` L consumed normal slot 1 and completed 1,000 steps, then failed the read-only identity label check. `/3` corrected the label but exposed the inherited row persistence contract, consuming reserve slot 1 on another L attempt. `/4` persisted the returned row under the frozen raw root. L and L0 then qualified, but the strict gate recorded `failed` because the run-level final lifecycle record embeds each run's own transcript output path. `/5` added exact path-only reconciliation; `/6` limits it to the run-level final record, leaving six target-level final records unchanged. All original attempts and the strict failed gate remain preserved. No scientific or world setting changed. The cumulative ledger accepts only `/2` through `/6`. Both reserve slots are already consumed; 8 normal slots remain for the fixed matrix.

No training, optimizer update, PPO replay, training-state/RNG restore, or checkpoint write is permitted. Only read-only loading of the fixed model and normal environment episodes are allowed. No push, merge, deployment, or follow-on experiment.

## Frozen Inputs

- Branch: `exp/round64-controlled-deadline-intervention`
- Pre-freeze base HEAD: `252f2fc4d6101693ef363c9969090eb1ba0bb6a1`
- Initial freeze commit: `4f49e78` (`/2`); identity repair: `e7bae95` (`/3`); row persistence: `581c7c8` (`/4`); current explicit gate metadata reconciliation: `round64-controlled-deadline-intervention/6`.
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

The first two runs must match on all frozen scientific fields, tape, transcript, and lifecycle after excluding only `run_config` plus the exact embedded `final.detail.metrics.comm_transcript` output path; the path values are separately preserved. They must also match the R60 focus baseline on common scientific fields. The initial strict gate did not perform the embedded-path exclusion and remains preserved as `failed`; a separate versioned reconciliation must pass before any matrix episode is allowed. The remaining eight are fixed; no result-dependent selection is permitted. This yields 10 normal attempts and at most 2 reserve retries.

## Accounting and Qualification

The persistent slot is atomically consumed before the actual environment builder call. Builder invocation/return, `run()` entry/return/failure, steps, and result qualification are separate counters. Equal outputs do not deduplicate executions; copying artifacts does not create an execution. Qualification does not refund a slot.

L/L0 are a validation gate, not independent scientific samples. Each forced pair is interpreted as a paired intervention in one fixed scene realization, not as identical downstream packet loss. The report will separate target outcome, full on-time target set, total tasks, costs, and communication/task lifecycle. The four deadlines may be compared descriptively; no inference of general necessity, safe deletion, noninferiority, or deployment is authorized.

The feature audit preserves all 19 inputs and may describe observed distinctions or exact-vector conflicts only. It does not fit a policy or establish feature sufficiency.

## Pre-Run Evidence

Before the next real episode: zero-environment preflight, source binding, read-only model identity, guard tests, and exact run order for `/3` are frozen and locally committed. The preflight and machine-readable identities are recorded in the versioned entries in `ROUND64_FREEZE.json`.

