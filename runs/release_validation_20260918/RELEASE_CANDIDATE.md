# Release candidate — coalition redundant-member release

Branch `release/coalition-controller-validation`. Delivery head at the time of writing is
this commit; the measured batches are pinned by the shas named below and in
`RELEASE_MANIFEST.json`. Defaults are **unchanged this round**: both switches stay off.

## 1. What the candidate is

The same binding coalition planner as the stable baseline `868d68a`, plus two coupled
mechanisms:

1. **Redundant-member release** (`--release-redundant-members`). A robot that is working may
   be re-decided in the same tick when, on its own belief, taking it off its stage would
   leave that stage's requirement covered no worse and the plan has a *different* task for
   it. Two conservative refusals come with it: a belief that carries no gross stage
   requirement cannot decide, and a robot missing from the roster its own requirement came
   with is reading two different moments. Both are counted by reason and published per
   decision.
2. **Own-command suppression** (`--suppress-recommand-own-task`). When a release-woke robot
   re-picks the task it is already executing, no command is issued. Re-issuing one restarts
   the environment's work clock and disarms the working timeout.

The shorthand `--coordinator coalition_redundant_release` turns both halves on together and
is what every candidate number here was produced with.

## 2. Release criteria, one by one

**发布阻塞：阶段信息陈旧导致错误释放，且环境旧任务成员残留可能掩盖覆盖损失。**

| Criterion | Status | Evidence |
|---|---|---|
| Information sources fit the existing architecture | **met, after a fix** | every field was already a legal `CoordinatorInput` input; no new permission taken. The verdict itself was reading the wrong denominator and now reads the gross stage requirement (`INFORMATION_PROVENANCE_AUDIT.md`) |
| Joint release does not wrongly remove a needed capability | **NOT MET — open failure** | on 180 stress episodes, **19 of 1 498 acted-on releases removed a capability the stage really needed**. This is counted as a coverage-protection failure, not as accepted residue. The existing metrics cannot clear it: the environment leaves a re-tasked robot on the old task's roster, so the setbacks these releases should cause may not appear as coverage resets at all (§2 of `MEMBERSHIP_CONSISTENCY_AUDIT.md`) |
| Stale-command refusal leaves state consistent and recoverable | **met** | 24 refusals across 800 held-out logs, all the same-tick completion race; 0 executed, 0 repeats, recovery in 1 tick in 24/24 (`STALE_REVISION_RECOVERY_AUDIT.md`) |
| No illegal action executed | **met** | 0 of 989 319 executed assignments across both arms of the independent block; `executed_override_unjustified` 0 everywhere |
| No unresolved livelock or coverage break | **NOT MET** | the release gate covers only the *destination* task's revision, never the precondition "it is safe to leave the source task" (`SOURCE_STAGE_RELEASE_AUDIT.md`). Measured resets are not evidence of absence while the stale-roster question is open |
| Stress shows no clear unexplained regression | **held** | condition B: +0.0243 paired, Bonferroni CI [+0.0019, +0.0579]; one new failure (33011/target-7). Whether the gain is attributable to the controller cannot be settled until the environment's member semantics are separated out |
| Mileage cost reported honestly | **met** | travel +2.3 % in the condition with a significant gain, flat elsewhere; decision p99 0.73 → 1.09 ms; 3.8–6.6 release judgements per seed that the plan declined to act on |

## 3. Recommendation

**No default-on recommendation is made.** The candidate is blocked on two open items: the
19 stale-source-stage releases, and the unresolved status of the environment's stale roster
entries. Both have to be settled — and the controller re-measured against an environment
whose member semantics are known — before any statement about making this a default.

What is established, and stays established: the planner-side release does clear late and
never-finished targets under load on unseen seeds, the refusals of stale commands are
correct and recoverable, and no illegal action executes. What is **not** established is
that it does so without removing capability a stage was using, because the measurement that
would show it is confounded by the environment's own bookkeeping.

Nothing in this branch changes a default, and this round must not be merged as a default
version.

## 4. Explicit enable command

```bash
D:/project_env/python.exe -m integrated_sar.joint_driver \
  --coordinator coalition_redundant_release --sticky-commitments \
  --residual-scale 0.25 --coordination-interval 5 --allocation-interval 5 \
  --command-ttl 10 --comm-delivery-probability 1.0 --comm-delay-ticks 0
```

To make it the default instead of per-invocation, flip two field defaults in
`integrated_sar/joint_driver.py` — `release_redundant_members` and
`suppress_recommand_own_task` — to `True`. Measured support for doing so without worrying
about the second one on its own: the suppression-only arm is behaviourally identical to the
baseline on the development block (81 completed / 8 expired / 1 reset, exactly equal),
because it only ever acts on a decision a release woke.

## 5. Rollback

```bash
# one invocation
D:/project_env/python.exe -m integrated_sar.joint_driver \
  --coordinator coalition_planner --sticky-commitments --residual-scale 0.25
```

```bash
# or pin the whole tree to the stable baseline
git checkout fix/coalition-commitment-continuity      # 868d68a
```

The second command is the exact arm every baseline number in this branch was measured
against, so rollback needs no re-validation.

## 6. Flag pairing, and what is now refused

Turning the release on **without** the suppression is the combination that churns the work
clock: 75 duplicate own-task commands on the 15 development seeds, each one restarting a
robot's working timer. It is a legitimate measurement — it is the ablation's C arm, and it
is how the churn was found — so it stays reachable, but it must now be declared:

```text
--release-redundant-members without --suppress-recommand-own-task
  -> ConfigurationError naming --coordinator coalition_redundant_release
--release-redundant-members --allow-release-without-suppression
  -> accepted, for the ablation that measures the churn on purpose
```

`--coordinator coalition_redundant_release` is always paired and needs no escape. Tests:
`ReleasePairingGuardTests`.

## 7. What in this tree is support, and what is research-only

Nothing was deleted or refactored; this is a map.

**Support configuration** (used by the candidate and the baseline):
`coalition_planner` + `sticky_commitments`, the two switches above, `residual_scale`,
`command_ttl`, the coordination/allocation intervals, `--deadline-tightness` (a TaskEnv
argument plumbed through, default equal to TaskEnv's own), the allocation trace at schema
`allocation-decision-trace/3`, and `run_signature`.

**Research-only, default off, unrelated to this candidate:**

| Path | Reachable through | Status |
|---|---|---|
| Assembly scheduling | `--coordinator coalition_assembly_scheduler` → `integrated_sar/assembly.py`, `Coordinator.assembly_scheduling` | prototype; its own gate report concluded it did not reach the control path. Tests exist (`test_assembly_scheduler.py`); it is not part of this recommendation |
| `--assembly-*` weight/horizon | constants inside the coordinator (`assembly_weight=0.15`, `sync_horizon=50.0`) | not exposed on the CLI; only the policy above turns the term on |
| Deadline-aware oracle | `--coordinator deadline_aware` | diagnostic arm, deliberately not belief-restricted; exists to answer "was ordering the binding constraint" |
| Advisory coalition | `--coordinator coalition` | superseded by the planner arm, kept so earlier numbers stay reproducible |
| `--allow-release-without-suppression` | CLI | explicit ablation escape, never a production setting |

## 8. Evidence attribution

| Block | Seeds | Role | Code |
|---|---|---|---|
| Closed-loop gate | 5 | development | v1 `6915d63`, v2 `fb1c85b` |
| Mechanism ablation A/B/C/D | 15 (5 nested in the gate, all inside the legacy 100) | development | v1 `1f4a598`, v2 `fb1c85b` |
| Paired expansion | 100 + 60, both previously read blocks | development | `6915d63` |
| Independent replay | 400 unseen | **old-candidate evidence** — it measured the residual-based verdict, and its standing under the member-consistency question is not yet assessed | `1f4a598` |
| Live-system loop | 60 unseen, real PILOC search | live-system | `1f4a598` |
| Stress | 180 unseen (3 × 60, disjoint) | robustness for the **v2** verdict | `3fcef63` |
| Provenance and safety scans | reuse of the 15 development and 180 stress seeds, instrumented, verified trajectory-identical | analysis | `fb1c85b`, `3fcef63` |

Across all blocks: **800 unique seeds, 835 paired scenarios, 1 700 replay episodes**. The
independent evidence is 460 unique seeds (400 + 60); the development blocks share 160 unique
seeds among themselves and must not be summed as if independent. `EVIDENCE_ACCOUNTING.json`
regenerates this from the manifests.

## 9. Scope, cost and limits

* Scope: the belief-gated coalition planner on this scenario family, six rescue robots,
  replayed detection streams, perfect and delayed communication as configured. The
  live-system loop showed the mechanism does not feed back into discovery (invariant 10),
  so search coverage is not a claim here.
* Cost: ~2 % more rescue travel where it pays; p99 decision latency roughly 1.5×, still
  ~1 ms; a few release judgements per seed that the plan declines to act on.
* Limits: the 19 stale-source-stage releases are an open coverage-protection failure, not a
  tolerated residue; the environment's stale roster entries confound every "no coverage
  loss" reading of the candidate, including the 400-seed block's; one new failure stands
  under load; and the two ceiling-bound stress conditions are inconclusive. The
  coverage-preserving verdict's own evidence is 180 stress seeds measured under conditions
  this round chose, and the 400-seed block belongs to the verdict it replaced.

## 10. Changes to merge, and what is not merged

Production diff from the stable baseline is confined to `integrated_sar/`: `contracts.py`
(belief field), `coordinator.py` (verdict, refusals, counters), `adapters/task03.py`
(suppression reasons, belief field), `joint_driver.py` (two switches, policy shorthand,
deadline band, pairing guard), `task03_runtime.py` (band forwarding), `allocation_trace.py`
(arm naming, rule marker, suppression reason), `run_signature.py` (schema `/3`), plus three
test modules. Nothing was merged, no default changed, and no force-push was used.
`RELEASE_MANIFEST.json` carries the shas, model digests, seed-manifest digests and raw
artefact fold hashes.
