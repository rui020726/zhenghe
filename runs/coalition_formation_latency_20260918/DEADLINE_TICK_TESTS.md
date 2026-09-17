# Deadline / tick micro tests: what was asked, what the environment does, what differs

18 tests in `integrated_sar/tests/test_deadline_tick_semantics.py`, all passing against
`old_task_env.py` at `868d68a`. They drive the real `TaskEnv` directly
(`task_update()` / `step()`), not the adapter, so a failure points at the environment.

No production file was modified. The one genuine defect found is **reported, not fixed**
(see "Off-by-one" below); the brief forbids repairing the environment.

## The ten requested cases

| # | requested | test | actual behaviour |
|---|---|---|---|
| 1 | `time_left = 0.2`, `dt = 1`, full coalition on site | `test_partial_final_tick_still_completes_the_stage` | decrement overshoots to `-0.8`; stage **finishes**; no partial credit, nothing carried forward |
| 2 | `time_left = 1.0`, `dt = 1` | `test_exact_one_tick_stage_finishes_on_that_tick` | lands exactly on `0.0`; `<= 0` fires; stage finishes |
| 3 | tick *ends* exactly on the deadline | `test_deadline_is_not_crossed_by_equality` | at `current_time == deadline` the task is **not** expired; first expires one tick later |
| 4 | tick *begins* exactly on the deadline | same test | same rule: the flag is driven by `current_time > deadline`, evaluated after the clock advances |
| 5 | full coverage formed one tick before the deadline | `test_a_stage_finishing_on_the_crossing_tick_is_reported_expired` | the work is applied, but see the off-by-one |
| 6 | coverage held to 95 %, then broken for one tick | `test_breaking_coverage_at_95_percent_restores_the_full_duration` | `time_left` is **reset to the full duration**; all elapsed work discarded |
| 7 | does `time_left` restore the full duration after a break? | same test, plus `test_the_reset_pins_at_the_full_duration_across_a_long_gap` | yes, to `stage["time"]`, and re-assigned every tick of the gap |
| 8 | stage completion and expiry on the same tick | `test_a_stage_finishing_on_the_crossing_tick_is_reported_expired` | both flags end set; see below |
| 9 | multi-stage transition and deadline on the same tick | `test_transition_and_deadline_on_the_same_tick` | task advances to stage 1 **and** is marked expired; not finished |
| 10 | `env.step(dt)` with a `dt` different from `self.dt` | `test_task_update_uses_self_dt_not_the_step_argument` | world advances by the argument, stage clock by `self.dt` — the two decouple |

## Actual vs expected: two divergences

### Divergence 1 (real defect, report only): a stage that finishes on the crossing tick is reported expired

`task_update()` runs the expiry loop (`:1099-1108`) **before** the work loop
(`:1110-1154`). On a tick where `current_time > deadline` and the stage clock also reaches
zero, the first loop sets `expired = True` and the second sets `finished = True`. The task
ends with both.

`Task03Adapter._read_progress` (`integrated_sar/adapters/task03.py:1257-1260`) then reads:

```python
if bool(task.get("finished", False)):
    status = TaskStatus.EXPIRED if bool(task.get("expired", False)) else TaskStatus.COMPLETED
```

`expired` wins. A target that **completed** on the tick its deadline was crossed is
recorded as `EXPIRED` and `failure_reason = "deadline_expired"`, and the expiry
attribution counts it as an expiry.

**Minimal repro** — `test_a_stage_finishing_on_the_crossing_tick_is_reported_expired`:
a 3-step single-stage task, `deadline = 2.0`, one capable robot on site from tick 0. After
three ticks `current_time == 3.0`, `expired is True`, `finished is True`.

**Impact.** Bounded by how often a target finishes on exactly the crossing tick. It cannot
explain the six failures under study: their final progress is 96.4-98.0 % whole-task, i.e.
they did **not** finish, so `finished` is False for all six and the adapter's branch is not
taken. It matters for the opposite reason — it can only ever *inflate* the expired count by
reclassifying a genuine completion, and only on the exact crossing tick.

**Suggested direction (not applied).** Evaluate expiry after the work loop, or exclude
`finished` tasks from the expiry loop, or have `_read_progress` prefer `finished`. Any of
the three changes recorded outcomes for any run containing such a tick, so it needs its own
ablation and re-evaluation rather than a drive-by fix.

### Divergence 2 (latent, not live): `dt` vs `self.dt`

`step(dt)` advances the clock and moves robots by the argument `dt` (`:1035`, `:1039`)
while `task_update()` drains the stage clock by `self.dt` (`:1121`). In every recorded run
the two are equal — `build_task_env` passes the same `configuration.dt` into the
constructor and into `step`, and `joint_driver` rejects a non-1.0 `dt` alongside a PILOC
checkpoint (`joint_driver.py:160-163`). So this is not a live cause for the arms under
analysis; the test pins the coupling so a future variable-`dt` change cannot silently
desynchronise the work clock from the wall clock.

## The three other answers, as tests

* **Assigned is not on site.** `test_an_en_route_robot_does_not_count_as_coverage` — a
  robot one tick away when the last member leaves does not protect the clock; the stage
  still resets to full.
* **Extra agents do not speed the stage up.** `test_extra_agents_do_not_speed_the_stage_up`
  — the decrement is a flat `dt`, so an over-covering second robot buys redundancy, not speed.
* **Partial ability does not start the clock.**
  `test_partial_ability_does_not_start_the_clock` — two robots that between them cannot
  meet the stage requirement leave `time_left` pinned at the full duration indefinitely,
  and `feasible_assignment` stays False.

## Reproduce

```
D:/project_env/python.exe -m unittest integrated_sar.tests.test_deadline_tick_semantics -v
```

Ran 18 tests, OK. Full suite: `Ran 453 tests, OK (skipped=2)` (435 baseline + 18 new).
