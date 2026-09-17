# What the environment actually does to a stage clock, tick by tick

Analysis only. No file under `task_allocation/` or `integrated_sar/` was modified to
produce this note; every claim below is a citation into the code at
`fix/coalition-commitment-continuity@868d68a`.

The six remaining binding-planner failures all die within a fraction of a tick of
finishing (`DURATION_RELEVANCE.md`: 0.24-0.71 steps of covered work owed). Before
attributing that to the planner it is worth being exact about the clock the planner is
racing, because two of the candidate causes named in the brief -- *deadline/tick boundary
semantics* and *complete coverage interrupted* -- are decided entirely here.

## 1. The single-step order

`TaskEnv.step(dt)` (`task_allocation/old_task_env.py:1026-1092`) runs in this order:

| # | line | what happens |
|---|---|---|
| 0 | 1027 | `dt = float(dt) if dt is not None else self.dt` |
| 1 | 1031-1037 | **movement**: every agent with `status == "moving"` and a `target_pos` advances `min(velocity * dt, dist)` toward it |
| 2 | 1039 | `self.current_time += dt` |
| 3 | 1041-1068 | **arrival**: agents now within `EPS_ARRIVE` of their target are resolved; a task arrival is validated and then appended to `task["members"]` and set to `status="working"` |
| 4 | 1070-1072 | communication graph, tidal burst, discovered burst |
| 5 | 1073 | **`task_update()`** -- coverage, stage clock, stage transition, deadline expiry |
| 6 | 1076-1078 | gossip every `gossip_interval` steps |
| 7 | 1080-1081 | `_check_working_timeouts()`, `_cleanup_cooldowns()` |
| 8 | 1083-1092 | utilisation counters; `self.finished = False` |

Two orderings in there matter and are easy to get backwards:

* **Movement precedes the clock advance, and arrival is resolved after it.** So the
  distance an agent closes in tick `T` is measured against its position at `T-1`, and the
  arrival is recorded at `current_time = T`. There is no half-tick arrival.
* **`task_update()` runs after arrivals.** A robot that lands on the target during tick
  `T` is in `members` before the tick-`T` coverage test, so it does contribute its
  abilities to that tick's stage work. Arrival is not delayed by one tick.

## 2. `task_update()` order

`task_update()` (`:1094-1160`) has two loops and the order is not incidental:

**Loop A -- deadline expiry (`:1099-1108`), first.**
```
if self.current_time > task["deadline"]:
    if self.hard_expire:  task["finished"] = True; task["expired"] = True
    else:                 task["expired"] = True
```
Soft expiry (`hard_expire=False`, the production default at `:58`) marks `expired` only.
It does **not** delete the task, does not release `members`, and does not stop the clock.
The task keeps being worked.

**Loop B -- work (`:1110-1154`), second.**
```
stage["status"] = max(stage["requirements"] - get_abilities(task["members"]), 0)
if (stage["status"] <= 0).all() and len(task["members"]) > 0:
    stage["time_left"] -= self.dt
    if stage["time_left"] <= 0:  ... stage transition / task completion ...
else:
    stage["time_left"] = stage["time"]      # <-- full reset
```

### 2.1 Does `time_left` reset when coverage is incomplete? **Yes, to the full duration.**

`:1152-1154`. The `else` branch fires whenever the stage requirement is not fully covered
**or** there are no members at all, and it assigns `stage["time"]`, not a partial value.
Every step of progress accumulated on that stage is discarded. It is a reset, not a decay,
and it happens on the same tick coverage breaks -- a stage at 95 % whose last required
member leaves loses the whole 95 %, not 5 %.

Note the condition is evaluated on **abilities present**, not on headcount: `get_abilities`
(`:1178-1180`) sums the `abilities` vectors of `members`. A stage needing `[1,0,1,0,1]`
with two robots that both carry only dimension 0 stays at `time_left = stage["time"]`
forever, however many robots are standing there.

Also note it re-assigns every tick coverage is broken, so `time_left` is *pinned* at the
full duration for the whole gap. Work is thrown away once per break, not repeatedly.

### 2.2 Does an assigned-but-not-arrived agent count as coverage? **No.**

`members` is only appended at `:1064`, inside the arrival block, and only after
`verify_task_on_arrival` returns valid. An agent with `status == "moving"` and
`target_id == task` is not a member. Travel is therefore pure dead time: it neither
advances the stage clock nor protects it. A robot that is one tick away when the last
member leaves does not stop the reset.

### 2.3 Do extra agents change work speed? **No.**

`:1121` is a flat `stage["time_left"] -= self.dt`. The decrement does not scale with the
number of members, with surplus ability, or with anything else. Full coverage removes
exactly `dt` per tick regardless of whether one robot or five are on site. Over-supply is
wasted for speed; it only buys robustness against a member leaving (and the
`keep_members` filter at a stage transition, `:1145-1148`).

### 2.4 Is stage duration fixed? **Yes, and equal across a task's stages.**

`generate_task` (`:161-183`) sets `total_time = base_time + complexity * scale + noise`,
then `stage_time = max(2.0, total_time / num_stages)` and writes the *same*
`"time": float(stage_time)` into every stage. `stage["time"]` is never mutated afterwards;
only `time_left` moves. So a task's stages are equal-length by construction, and the
duration is a static property of the scenario -- consistent with
`DURATION_RELEVANCE.md`'s finding that the message carries the exact clock.

### 2.5 Does the deadline use `>` or `>=`? **Strict `>`.**

`:1102` is `if self.current_time > task["deadline"]`. At `current_time == deadline` the
task is **not** expired and the tick's work is applied. The task is first marked expired
on the tick where `current_time` strictly exceeds the deadline.

Combined with loop-A-before-loop-B this produces one genuine boundary artifact, recorded
here as a finding rather than fixed:

> **Off-by-one candidate (report only, not repaired).** On the tick that both crosses the
> deadline and finishes the final stage, loop A sets `task["expired"] = True` first and
> loop B then sets `task["finished"] = True`. The task ends the tick with *both* flags.
> The adapter classifies by `expired` first
> (`integrated_sar/adapters/task03.py:1258`:
> `status = TaskStatus.EXPIRED if task.get("expired", False) else TaskStatus.COMPLETED`),
> so a task that **completed on the tick its deadline was crossed is reported as
> expired**. Minimal repro, impact and suggested direction are in
> `DEADLINE_TICK_TESTS.md`; the environment is not touched.

### 2.6 Is `dt` always `self.dt`? **In production yes, by construction -- but the two are
decoupled in the code.**

`step` uses the *argument* `dt` for movement (`:1035`) and for the clock (`:1039`), while
`task_update` uses `self.dt` for the stage decrement (`:1121`). A caller passing
`dt != self.dt` silently desynchronises the work clock from the wall clock: the world
advances by one amount and the stage clock by another.

In every recorded run the two agree. `build_task_env`
(`integrated_sar/task03_runtime.py:105-135`) constructs `TaskEnv(..., dt=dt)` from the same
`configuration.dt` that `Task03Adapter.step` forwards to `env.step(dt)`
(`integrated_sar/adapters/task03.py:1139`), and `joint_driver` rejects any mismatch
(`integrated_sar/joint_driver.py:160-163`: a non-1.0 `dt` with a PILOC checkpoint is a
configuration error). So for the arms under analysis `dt == self.dt == 1.0` and this is
not a live cause. It is recorded because the second `dt` path is what a future
variable-`dt` change would trip over.

## 3. What this fixes as a prior for the six failures

Three consequences, each of which narrows the search:

1. **A coverage break is catastrophic, not costly.** Because the reset is to the full
   duration, the cost of losing coverage is the entire elapsed work on that stage. This is
   why `DURATION_RELEVANCE.md`'s progress-monotonicity check mattered: it confirmed that
   on the five covered targets the within-stage fraction never moved backwards, so the
   refill at `:1154` never fired on a stage they were working. Their loss is late start
   and slow finish, not thrown-away work.
2. **Travel cannot be hidden.** Since only arrived robots count, and the deadline check
   runs before the work check, a late arrival costs exactly its travel steps with no
   partial credit.
3. **The final tick is already credited.** Soft expiry keeps the clock running past the
   deadline, so the work of the tick that crosses the deadline is applied. A
   "final-tick credit" counterfactual is therefore expected to find little -- which is
   exactly why it is still worth running as counterfactual E rather than assuming it: the
   only thing it can change is the *reported status* of a task that finished on the
   crossing tick, per the off-by-one above.

## Reproduce

The citations above are stable line numbers at `868d68a`. The executable form of every
claim is in `DEADLINE_TICK_TESTS.md` / `integrated_sar/tests/test_deadline_tick_semantics.py`.
