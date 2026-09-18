# Member-consistency audit — the controller's roster, the environment's roster

Branch `fix/task-membership-consistency`, delivered head `bed60f9` + `98a34b3` + `7bbc3ac` +
`d194c31`. Evidence: `membership_min_repro.py`, `ghost_scan.py` → `GHOST_SCAN.json`,
`release_harm_scan.py` → `HARM_SCAN.json`,
`integrated_sar/tests/test_task_membership_consistency.py`.

**Verdict: the ghost contribution is real. The inference is not withdrawn.** A robot that has
been re-tasked away stays on the task it left, keeps that task covered, keeps its work clock
running, and is credited to two tasks at once until something else removes it. This is
environment bookkeeping, older than any controller in this branch, and it is what made the
19 harmful releases invisible.

## 1. The chain, with the state each step writes

| Step | Code | What changes |
|---|---|---|
| command created | `adapters/task03.py:663` `allocate` | an `AssignmentCommand` with the destination's registry revision; the `assignment` log row is written **here**, at creation |
| gate acceptance | `assignment_gate.py:54-79` | refused commands never reach the executor; accepted ones are passed as a list |
| `apply_action` | `old_task_env.py:988` | `status="moving"`, `target_*`, `current_task=task_id`, `need_decision=False`; **the previous task's `members` list is not touched** |
| removal from the old task | — | no such step exists |
| joining the new task | `old_task_env.py:1064` | on arrival only, after `verify_task_on_arrival`: `task_dic[tid]["members"].append(aid)`, then `status="working"` at `:1065` |
| movement | `:1031-1037` | position integration; nothing roster-related |
| arrival | `:1041-1068` | the only writer of `working` |
| coverage | `task_update:1114` | `stage["status"] = max(requirements − get_abilities(task["members"]), 0)` |
| work progress | `:1119-1121` | if that status is all-zero **and `members` is non-empty**, `stage["time_left"] -= dt` |
| `get_abilities` | `:1178-1180` | sums `agent_dic[m]["abilities"]` for every `m` in the roster — **no presence, status or distance filter** |
| the only removals | `:831` timeout, `:1133` finish, `:1151` stage-transition keep-list | each removes or replaces the roster and calls `_release_agent` |

## 2. The five distinct facts, and which ones the environment uses

For one robot and one task these are five different questions:

| Fact | Source | Used by the coverage arithmetic? |
|---|---|---|
| listed on the task's roster | `task["members"]` | **yes — the only one** |
| currently assigned to it | `agent["current_task"]` | no |
| physically on site | `agent["location"]` vs `task["location"]` | no |
| state allows serving | `agent["status"]` | no |
| really counted as capability | `get_abilities(members)` | this *is* the count, and it equals the roster |

The controller is stricter than the environment here: the release pool already restricts
itself to `status == "working"` (`coordinator.py:30` `ONSITE_STATUSES`) plus a matching
`current_task`. That mismatch is why the two agree on outcomes only by accident: when the
controller moves a robot, the controller stops counting it immediately and the environment
keeps counting it forever.

Measured on 3 556 release-candidate examinations over the development seeds, the
snapshot-side facts hold exactly: **0** cases of a `working` robot that was not in the
registry roster, **0** of a `working` robot not standing at its task's position. The failure
is not that `working` lies; it is that the roster outlives `working`.

## 3. Minimal reproduction — two tasks, two robots, no coordinator

`membership_min_repro.py`. Robot 0 is the only one able to serve either stage. It is sent to
A, arrives, covers A, then is re-tasked to B — one `apply_action`, exactly what an accepted
cross-task command does.

| Observation | roster as-is | roster consistency on |
|---|---|---|
| A's roster still lists the departed robot | **True** | False |
| ticks where A's clock ran down with nobody able to cover it | **20** | **0** |
| robots on two rosters at once | **{robot 0: [A, B]}** | none |
| A finished | True | False — correctly, it has nobody |
| A finished with nobody ever on site | **True** | False |
| A's closing released the robot that was working at B | **True** | False |
| B's roster afterwards | `[0]` with B on-site empty | `[]` |

The last two rows are the sharpest part of the defect: closing a task releases everyone on
its roster, so a stale entry lets a task that finished **steal a robot from the task it is
actually working on**, and that second task then inherits a stale entry of its own. The
fault propagates rather than staying put.

## 4. The 19 harmful releases, measured against this

Replaying the 17 stress episodes that contained them, watching every release the coordinator
acted on (`GHOST_SCAN.json`, candidate arm on the unfixed environment):

| | |
|---|---|
| releases watched | 232 |
| …that produced at least one ghost tick | **185 (80 %)** |
| ghost ticks total | 339, max 14 for a single release |
| releases where the robot was on two rosters at once | **87** |
| how the residue ended | 177 by the source task finishing, 55 by roster removal |
| longest single residue | 34 ticks |
| source tasks that finished while still crediting an absent robot | **177** |

So the residue is not a historical footnote: it is what the old task's completion is
computed from. And 55 of 232 endings were a *later* timeout or stage transition, meaning the
roster entry usually survives the release entirely.

## 5. What was fixed, and what that changes

`98a34b3`, default off behind `--task-membership-consistency` (`RunConfiguration.
task_membership_consistency`):

* an accepted reassignment — to another task or to the depot — detaches the robot from every
  roster listing it and recomputes those stages' status, in deterministic (sorted) order;
* re-selecting the task a working robot is already on is a true no-op: no new navigation
  target, no work-clock reset, no roster change;
* deadline sampling, stage durations, the random working timeout and the coverage-interrupt
  reset rule are untouched.

The ten tests in `test_task_membership_consistency.py` assert consequences rather than list
contents: the stage's computed coverage goes back to uncovered, the clock returns to full and
then makes no further progress, the abandoned task does not finish, one robot never appears on
two rosters, a no-op does not restart the work clock, a timeout still releases from the
current task only, reset leaves nothing behind, and a stage transition still keeps only
overlapping members.

Cell C below shows the baseline arm is **bit-for-bit unaffected** by this switch, because
with no releases there is nothing to detach.

## 6. Still open

The guard fixes the bookkeeping, not the belief. A release judged from a belief that lags
inside the *same* stage would still be visible now — which is why the controller-side guard
and the source-stage gate exist alongside it (`SOURCE_STAGE_RELEASE_AUDIT.md`).
