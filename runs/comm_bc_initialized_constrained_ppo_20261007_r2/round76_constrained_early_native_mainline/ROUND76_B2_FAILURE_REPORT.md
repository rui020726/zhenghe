# ROUND76 B2 technical qualification failure

## Status

The single authorized B2 parent launch was consumed. The child process failed in
`pre_model_input_counts` before runtime configuration, guard/ledger creation, NN
construction, optimizer construction, A1, A2, A3, checkpoint save/load, or any
scientific comparison. No retry was attempted and no balance was borrowed.

## Failure

```text
r76_episode_protocol.EpisodeProtocolError:
tape record 9 one-shot state contains an invalid target or state
```

The failing input path was the frozen episode tape listed in the issued contract.
This is an input-protocol technical failure, not a scientific qualification pass or
failure, and it does not produce any R76 search/rescue strategy conclusion.

## Failure attribution evidence

The validator excerpt below is the unchanged source at
`r76_episode_protocol.py:118..150`. Its one-shot-state check requires every key to
belong to the record's target list:

```python
                for value in block_reason.values()):
        raise EpisodeProtocolError(f"{label} block-reason-initial is not valid for every target")
    decision_class = tape["decision_class"]
    if not isinstance(decision_class, dict) or set(decision_class) != set(targets):
        raise EpisodeProtocolError(f"{label} decision class does not cover every target")
    expected_classes = ({target: "transport_forced" for target in tape["forced"]} |
                        {target: "free_one_shot_choice" for target in tape["actionable"]} |
                        {target: "existing_budget_or_capacity_mask" for target in tape["masked"]} |
                        {target: "fixed_hold_one_shot_closed"
                         for target in tape["fixed_hold_one_shot_closed"]})
    if any(decision_class[target] != expected_classes[target] for target in targets):
        raise EpisodeProtocolError(f"{label} decision class disagrees with candidate classes")
    for name in ("shadow_budget_before", "shadow_budget_after"):
        if _finite_number(tape[name], f"{label}.{name}") < 0:
            raise EpisodeProtocolError(f"{label}.{name} is negative")
    for name in ("shadow_capacity_before", "shadow_capacity_after"):
        value = tape[name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise EpisodeProtocolError(f"{label}.{name} is not a valid capacity")
    one_shot_state = tape["one_shot_state_after_step"]
    allowed_states = {"undecided", "send_now_chosen", "skip_episode_chosen",
                      "closed_by_prior_real_emission"}
    if not isinstance(one_shot_state, dict) or \
            any(target not in targets or value not in allowed_states
                for target, value in one_shot_state.items()):
        raise EpisodeProtocolError(f"{label} one-shot state contains an invalid target or state")
    _finite_number(tape["log_prob"], f"{label}.log_prob")
    _finite_number(tape["value"], f"{label}.value")
    _finite_number(tape["cost_value"], f"{label}.cost_value")
    return tape
```

The raw e0 tape is
`round71_critic_initialization_calibration/raw/logs/v2_r71_train_s11_e0_s1300027_a2_tape.jsonl`
with SHA256
`7d2fa6b70fbcc3e66d7c013f0e96a163fadc58f1b0e3279cd96f2a5172c253ae` and 66 lines.
Record index 9 is the tenth line, read verbatim with
`Get-Content -TotalCount 10 | Select-Object -Last 1`. It has
`"targets": ["piloc-e1-target-1", "piloc-e1-target-2", "piloc-e1-target-4", "piloc-e1-target-5"]`
but `"one_shot_state_after_step"` also contains `"piloc-e1-target-3": "skip_episode_chosen"`.
The validator excerpt above rejects that extra key. The complete raw line is:

```json
{"actionable": [], "actions": {"piloc-e1-target-1": 0, "piloc-e1-target-2": 0, "piloc-e1-target-4": 0, "piloc-e1-target-5": 0}, "attempt_charge_bytes": {"piloc-e1-target-1": 2532, "piloc-e1-target-2": 2496, "piloc-e1-target-4": 2490, "piloc-e1-target-5": 2826}, "base_action_mask_schema": "budget-sequential-mask/1", "candidate_audit": {"piloc-e1-target-1": {"effective_action": 0, "first_emission": true, "fixed_reason": "one-shot-choice-already-made", "one_shot_state_before": "skip_episode_chosen", "shadow_budget_after": 29300.0, "shadow_budget_before": 29300.0, "shadow_capacity_after": 0, "shadow_capacity_before": 0, "shadow_charge_applied": false}, "piloc-e1-target-2": {"effective_action": 0, "first_emission": true, "fixed_reason": "one-shot-choice-already-made", "one_shot_state_before": "skip_episode_chosen", "shadow_budget_after": 29300.0, "shadow_budget_before": 29300.0, "shadow_capacity_after": 0, "shadow_capacity_before": 0, "shadow_charge_applied": false}, "piloc-e1-target-4": {"effective_action": 0, "first_emission": true, "fixed_reason": "one-shot-choice-already-made", "one_shot_state_before": "skip_episode_chosen", "shadow_budget_after": 29300.0, "shadow_budget_before": 29300.0, "shadow_capacity_after": 0, "shadow_capacity_before": 0, "shadow_charge_applied": false}, "piloc-e1-target-5": {"effective_action": 0, "first_emission": true, "fixed_reason": "one-shot-choice-already-made", "one_shot_state_before": "skip_episode_chosen", "shadow_budget_after": 29300.0, "shadow_budget_before": 29300.0, "shadow_capacity_after": 0, "shadow_capacity_before": 0, "shadow_charge_applied": false}}, "comm_action_eligibility": "first-report-only/1", "comm_action_semantics": "one-shot-send-skip/1", "cost_value": 0.12617552280426025, "decision_class": {"piloc-e1-target-1": "fixed_hold_one_shot_closed", "piloc-e1-target-2": "fixed_hold_one_shot_closed", "piloc-e1-target-4": "fixed_hold_one_shot_closed", "piloc-e1-target-5": "fixed_hold_one_shot_closed"}, "effective_actions": {"piloc-e1-target-1": 0, "piloc-e1-target-2": 0, "piloc-e1-target-4": 0, "piloc-e1-target-5": 0}, "features": [[9.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 1.0, 4.0, 1.0, 1.0, 6.0, 0.01], [9.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 1.0, 4.0, 1.0, 1.0, 6.0, 0.01], [5.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 1.0, 4.0, 1.0, 1.0, 6.0, 0.01], [9.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.27420997228729566, 9.0, 1.0, 0.0, 1.0, 4.0, 1.0, 1.0, 6.0, 0.01]], "fixed_hold_one_shot_closed": ["piloc-e1-target-1", "piloc-e1-target-2", "piloc-e1-target-4", "piloc-e1-target-5"], "fixed_hold_reasons": {"piloc-e1-target-1": "one-shot-choice-already-made", "piloc-e1-target-2": "one-shot-choice-already-made", "piloc-e1-target-4": "one-shot-choice-already-made", "piloc-e1-target-5": "one-shot-choice-already-made"}, "fixed_hold_refresh": [], "forced": [], "free": [], "free_first_reports": [], "free_one_shot_choices": [], "log_prob": 0.0, "masked": [], "masked_reasons": {}, "one_shot_state_after_step": {"piloc-e1-target-1": "skip_episode_chosen", "piloc-e1-target-2": "skip_episode_chosen", "piloc-e1-target-3": "skip_episode_chosen", "piloc-e1-target-4": "skip_episode_chosen", "piloc-e1-target-5": "skip_episode_chosen"}, "relay_observation_remaining_bytes": 29300.0, "requested": {"piloc-e1-target-1": false, "piloc-e1-target-2": false, "piloc-e1-target-4": false, "piloc-e1-target-5": false}, "schema": "round42-one-shot-send-skip-tape/1", "send_block_reason_initial": {"piloc-e1-target-1": "allowed", "piloc-e1-target-2": "allowed", "piloc-e1-target-4": "allowed", "piloc-e1-target-5": "allowed"}, "send_executable_initial": {"piloc-e1-target-1": true, "piloc-e1-target-2": true, "piloc-e1-target-4": true, "piloc-e1-target-5": true}, "send_probabilities": {}, "sender_state": {"backlog_targets": 4, "budget_bytes_per_episode": 29300, "candidates": 4, "capacity_remaining": null, "remaining_budget_bytes": 29300.0, "step": 10, "subscribers": 6}, "shadow_budget_after": 29300.0, "shadow_budget_before": 29300.0, "shadow_capacity_after": null, "shadow_capacity_before": null, "step": 10, "targets": ["piloc-e1-target-1", "piloc-e1-target-2", "piloc-e1-target-4", "piloc-e1-target-5"], "value": 5.515643119812012}
```

## Execution identity

- Branch: `exp/round76-constrained-early-native-mainline`
- Frozen SourceHead and final HEAD: `c9eed7f18cd19defabd28288a0ebef4bd85c9bef`
- Parent PID: `69840`
- Child PID: `63832`
- OS exit: `1`
- Parent capture: `raw/qualification/R76_B2_PARENT_CAPTURE_20261016`

## Actual consumption

- B2 parent launches: `1/1`
- CPU NN Python process attempts: `1/1`
- Optimizer steps: `0/17`
- INIT/checkpoint loads: `0/2`
- A1/A2/A3 checkpoint saves: `0/3`
- Environment constructions: `0`
- B1 remains complete at `2/2` Python processes and `6/6` fixed-case executions.

This consumption distinction follows the original stderr and call order: the sole
B2 parent launch and its one qualification Python child were consumed. The child
stopped in `pre_model_input_counts`, before `configure_qualification_runtime`,
model construction, native optimizer construction, environment construction, A1,
A2, A3, checkpoint save/load, or any comparison. Therefore actual NN, optimizer,
and environment consumption is zero. Complete `M/N/T` counts are missing rather
than zero, A1/A2/A3 are not executed, and the absent qualification ledger is not
backfilled as a zero ledger.

The child did not create `r76_total_ledger.jsonl`, counter JSON files, or `.pt`
checkpoints. A1, A2, A3, comparison, and success-gate phases were not executed and
are not reported as zero-valued scientific results.

## Evidence hashes

- Contract SHA256: `6db44b625894c4bcd1d7a6ba926c3f9235e98e7c834d77a70dae3593d8ccd484`
- Permit SHA256: `40b16faf2f3d43ef3a99de3c9bb0729a982c1de78b24e1be5991976eb3cd7599`
- Prelaunch claim SHA256: `910168d8c73675006f158b7880b2984359f22e7e2168708f38e978a2f95505b3`
- Parent receipt SHA256: `4fd6803b42efbe24abbb241440ca195a29efd33ca9d27ae854a214b1d9fc12ad`
- Child stdout SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Child stderr SHA256: `b9d1e9633cec90b9eee60d3d27e5eb77275fd26d0305a164e76fb2c8e95d5cb5`

## Preservation

The old R75 ledger, ROUND46 protected files, B1/P1/P2 evidence, contract, permit,
and parent capture remain unchanged. The 12 frozen source files retained their
P2-listed SHA256 identity. No scientific source, checkpoint, model, or raw log was
rewritten in closure.
