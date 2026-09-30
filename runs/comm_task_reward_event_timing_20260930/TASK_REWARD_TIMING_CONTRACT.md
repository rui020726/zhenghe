# Task reward timing contract

## Version

- Training version: `constrained-cost-v2/published-completion-event-at-relay-tick/2`
- Reward capture version: `published-completion-event-at-relay-tick/2`
- Existing GAE/reward-bundling schema remains `post-action-most-recent-decision/1`; its assignment rule is unchanged.

## Runtime order

One orchestrator step uses these time bases:

| Order | Operation | Time fields and ownership |
|---:|---|---|
| 1 | `Task03Adapter.step` calls `CommRelay.tick`; the policy is sampled inside the relay tick and the cost tap records the counter delta. | Relay tick becomes `t`. A decision tape row, when emitted, is tagged `t`. |
| 2 | The adapter calls the raw task environment's `step(dt)`. | Physical environment time advances to the end of tick `t`. |
| 3 | After raw `env.step` returns, the adapter constructs `ProgressEvent(observed_at=env.current_time)`. | The event timestamp is the post-step environment time. |
| 4 | The orchestrator publishes that event synchronously. The registry is the first `ProgressEvent` subscriber; the reward observer is registered later. | Registry commits the accepted transition before reward collection. `clock.now` still holds the pre-step time during dispatch. |
| 5 | The orchestrator advances `clock.now` to the post-step decision time. | This happens after event publication; the observer records but does not change it. |
| 6 | A later strategy sample starts. | The reward has already been recorded before the next action sample. |

The `clock.now` value at callback time is intentionally the pre-step time, one frozen `dt` earlier than the event's `observed_at` in this run configuration. That is the existing scheduler order. The collector does not change the clock or reinterpret event timestamps. `relay._tick` determines the task stream key; `ProgressEvent.observed_at` determines the event time and deadline check.

## Reward eligibility and action attribution

A reward unit is recorded only when all of these are true:

1. the event status is `COMPLETED`;
2. the registry accepted that event and its committed snapshot is `COMPLETED`;
3. neither event nor snapshot is expired;
4. `event.observed_at <= snapshot.deadline`;
5. that target has not already been credited by this episode's observer.

For an eligible event at relay tick `t`, the collector increments `task_stream[t]` immediately during publication. `v2_trainer._bundle` then uses the existing rule: the latest tape decision at or before `t` owns the reward. A same-tick tape row owns it; if there is no row at `t`, the latest earlier decision owns it. If no decision exists yet, the reward remains an unassigned prefix. Tick zero is retained by `_bundle` for this prefix. The tail after the final decision remains assigned to that final decision.

| Hand-calculated case | Event tick | Decision rows | Expected result |
|---|---:|---|---|
| Same-tick completion | 1 | `[1]` | Reward on action row 1 |
| A new action follows next tick | 1 | `[1, 2]` | Reward stays on row 1; row 2 receives zero |
| No action row at the next tick | 2 | `[1]` | Stream key remains event tick 2; the bundler assigns it to row 1 |
| Completion on final execution tick | 1 | `[1]` | Reward is captured immediately; no later step is needed |
| Before first action | 0 | `[1]` | One unassigned prefix unit; row 1 receives zero |
| Completion after deadline | 2 | `[1]`, deadline 1 | No on-time reward |

Multiple targets completed on one tick each add one unit. Repeated dispatch or duplicate event IDs do not credit the same target twice. Partial stage progress is not task completion and creates no reward.

## Implementation boundary

`v2_run._install_taps` leaves the relay cost wrapper in place and no longer wraps or delays `environment.step`. It adds `PublishedProgressRewardCollector` to the existing `ProgressEvent` bus after the orchestrator has subscribed the registry. It reads the accepted registry result, current relay tick, and current clock; it does not publish events, mutate registry data, call randomness, or step the environment. The event observer is unsubscribed when the episode runner exits. The tap also records a small event ledger in the episode row with event time, relay tick, clock value, registry acceptance, deadline, and whether the unit was credited.

The cost stream path, cost units, reward value, objective, masks, observations, network, GAE parameters, and policy are unchanged. The implementation does not move timestamps or globally adjust the clock.

## Verification scope

`test_reward_timing_regression.py` uses the production event bus, task registry, `_install_taps`, and `_bundle`, with only simple fake holders for relay, clock, and raw step. The independent expected event table is declared in the test source. The tests do not construct `JointRun`, SAR simulation objects, or invoke a simulation runner. Passivity checks cover Python RNG state, event history, accepted/rejected registry counts, absence of emitted messages in the fixture, and unchanged cost counter deltas.

These are synthetic and wiring tests only. They do not establish passive behavior in a real simulation; that would require a separately approved smoke run.
