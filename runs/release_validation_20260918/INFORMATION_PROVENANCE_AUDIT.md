# Information provenance audit — the fields the redundant-release verdict reads

Branch `release/coalition-controller-validation`, brief section 2. Evidence:
`provenance_scan.py` → `PROVENANCE_SCAN.json` (15 development seeds, 3 556 examined release
candidates, instrumented recorder that changed no verdict and reproduced the shipped arm
outcome per seed, target by target), `release_test_variants.py` →
`RELEASE_TEST_VARIANTS.json`, `stage_drift_probe.py`.

**Bottom line.** Every field the verdict uses was already a legal input, and two of the
freshness questions came back clean. One did not: the coverage test was reading a
requirement vector that already had the candidate subtracted from it, so it could not see a
sole provider at all. That is a defect in the safety test itself, fixed in `fb1c85b`. No new
information permission was requested or taken.

## 1. Field by field

| Field | Produced at | Propagation chain | Visible to | Timestamp / update semantics |
|---|---|---|---|---|
| `AgentSnapshot.status` | `old_task_env.py:1065` sets `working` only on arrival after `verify_task_on_arrival` passed; `:991/:1021` `moving`; `:152-159` `_release_agent` → `idle` | `env.agent_dic` → `task03.py:459` `agent_snapshots()` → `CoordinatorInput.agents` | the centralized coordinator, for every robot (pre-existing) | live, per tick, no staleness |
| `AgentSnapshot.current_task` | `apply_action` `:1021` (target task), cleared to `-1` by `_release_agent` `:156` | same | same | live; maps internal → external at `:437` |
| `AgentSnapshot.abilities` | species profile at scenario build (`:261`) | same | same | fixed for the episode |
| `AgentSnapshot.position` | movement integration `:1031-1037` | `env.agent_dic[..]["location"]` → `task03.py:453` | same | live |
| believed stage index | registry/env `task["current_stage_index"]` | carried on the message → `_apply_message` `task03.py:411` → `learn_task_view` `:323` → `_task_beliefs` `:490` | the owning robot only, via its own knowledge | frozen at delivery; `updated_at` = message `sent_at` |
| believed stage requirement (gross) | `task["stages"][i]["requirements"]` | `view["requirements"]` (`task03.py:418`) → `learn_task_view:320` → `knowledge["task_requirements"]` | the owning robot | frozen at delivery, same moment as the residual |
| believed remaining requirement (residual) | `task["status"]`, which `task_update:1114` computes as `max(requirements − get_abilities(members), 0)` — **members includes this robot** | `view["status"]` (`task03.py:417`) → same chain | the owning robot | frozen at delivery |
| `revision` | registry counter, bumped on every accepted detection/progress event | message → `learn_task_view:337-340` | the owning robot | ordering is `(updated_at, revision)`; a lower rank never overwrites (`:406-409`, `old_task_env.py:524-536`) |
| "does member *m* still cover" | **not a field.** Built in `coordinator.py:620-625` from `status` + `current_task` + `abilities` of the live snapshots, and `ONSITE_STATUSES = {"working"}` (`:30`) | inside one `advise()` call | the coordinator | the decision moment itself; a travelling teammate is deliberately not counted |

Gossip moves beliefs between robots (`_share_task_belief`, `:406-435`) from the tick-start
snapshot (`_copy_knowledge`, `:383-404`), so a fact still reaches a robot only by a channel
that already existed, and never more than one hop per tick.

## 2. The five verification questions

**2.1 Was the information already a legal `CoordinatorInput`?** Yes. All seven items above
were already inside it. The team-plan view (`committed`) is not new either: it exists in the
stable baseline `868d68a` at `coordinator.py:264-267` and already feeds `_pair_biases` and
`_renew`. The only *newly read* item is `knowledge["task_requirements"]`, which the same
message that delivers the residual already delivers, and which the belief store has always
held — it simply had never been surfaced on `AgentTaskBelief`. Surfacing it widens no
visibility boundary; reading the registry would have, and `test_no_registry_truth_is_needed_
to_decide_a_release` still passes with the release active.

**2.2 Does `working` really mean "arrived, and still on that task"?** Yes, measured, not
argued. Across 3 556 examined candidates: `working_but_not_a_registry_member` = **0**,
`working_but_not_at_the_task_location` = **0**. That is the expected shape from the source:
the only writer of `working` is the arrival branch, which appends the robot to
`task["members"]` in the same statement, and every path that removes a member
(`:831` timeout, `:1133` finish, `:1151` stage transition) calls `_release_agent`, which
clears `status` and `current_task` together.

**2.3 Do the fields come from one consistent decision moment?** Partly, and this is where
the defect lived. Two distinct inconsistencies were measured:
* **Stage lag**: `believed_stage_differs_from_truth` in 118 of 3 556 candidates (3.3 %). A
  robot that stays on site never re-verifies, so its belief freezes at arrival until
  gossip brings something newer.
* **Self-subtraction** (the serious one): the residual the verdict reads is computed *with
  this robot among the members*, so it is not an independent estimate of what the stage
  needs — it is a measurement that contains the thing being measured. In **3 111 of 3 556**
  candidates (87.5 %) the believed residual was all-zero.

**2.4 Did anything cross a communication or visibility boundary?** No. Same sources, same
channels, same per-robot knowledge. The offline comparison against registry truth in this
scan is analysis-only: it reads `env.task_dic` inside a recorder that returns the original
verdict unchanged, and the identity gate confirms the trajectories are the shipped ones.

**2.5 Was a release refused when the stage and the member snapshot disagreed?** **No — not
before this round**, and that is the finding. Concretely, with the verdict as shipped in
`7e5ba3b`:

| Measured on the 15 development seeds | count |
|---|---|
| candidates examined | 3 556 |
| judged releasable | 3 388 |
| …of which the belief said the current stage needed nothing | 3 111 |
| judged releasable **and actually re-decided that tick** | 205 |
| …where removing that robot would have lowered the coverage that really existed | **33** |
| …where the robot was not even on the roster its own requirement came from | 7 |

Recomputing the same recorded decisions with the coverage-preserving test: 1 852 of 3 388
judgements survive, 131 of the 205 executed releases survive, and **32 of the 33
coverage-lowering releases are blocked**.

Why no failure showed up in the outcome data: a robot re-tasked out of a task it was
covering is never removed from that task's registry `members` — `_release_agent` clears the
robot, not the roster entry, and only timeout/finish/stage-transition edit the list. So the
stage kept counting an absent robot and its `time_left` kept running down. The 400-seed
block's `coverage_resets = 0` was therefore real but overdetermined: the outcome was saved
by an environment artefact, not by the decision rule. Shipping a safety property that holds
only because of a phantom roster entry is not acceptable for a default-on recommendation.

## 3. What changed

`fb1c85b` replaces the slot-claim question ("can this robot still claim a unit of the
residual?") with the coverage question ("is the believed stage requirement covered exactly
as well without this robot as with it?"), evaluated on the gross figure from the same
belief, plus two conservative refusals with counters published per decision:

* `no_stage_requirement` — the belief carries no gross figure, so it cannot decide;
* `absent_from_believed_roster` — the requirement and the on-site pool came from different
  moments.

Tests pin the sole-provider case, both refusals, that the second of two over-provisioned
members is still released, and that refusals are counted per decision and cleared on reset.

**Cost of the change, on the same development seeds** (`DEV_RERUN.json`): the five-seed
closed loop is unchanged (baseline 21 completed / 8 expired / 1 reset / 1.2 lost work;
candidate 30 / 0 / 0 / 0, no duplicate commands, no rejections, no regressions), and the
fifteen-seed block still moves 81 → 90 completions with 0 expired and 0 resets. Releases
drop 205 → 125 and accepted reassignments 86 → 51, so the mechanism now does about 40 %
fewer re-decisions for the same measured benefit; `redundant_working_ticks` rises
3 213 → 3 463 in the release-only arm, which is the honest price of refusing to move a
robot the belief cannot prove is spare.

## 4. Does the information base for earlier claims still hold?

The 400-seed independent block and the 60-seed live-system block were produced by the
residual-based verdict. Their numbers are not retracted and their protocols are untouched,
but they are measurements **of that verdict**, and this round's stress block is the first
evidence about the coverage-preserving one. They must not be pooled: same switch, same arm
label, different rule. The `release_verdict` key added to the trace header exists so an
artefact says which rule produced it without needing its commit.
