# ROUND42-R stop diagnosis

## Stop

The executor stopped during the seed 11 first-batch gate at the check in `r42_analysis.py` that compares the keys of `record["frozen_checkpoints"]` with the frozen PILOC/CTDE path set. The exception was:

```text
RuntimeError: seed 11: frozen PILOC/CTDE checkpoint path set changed
```

The executor performed no automatic retry or resume after this gate failure.

## Checkpoint path evidence

The batch record serialized the two relative keys with Windows separators:

```text
runs\cur_v2rep05_s2\best\shared_mappo_model_21_3100.pth
runs\joint_eval_20260916\ctde_final_7_1.pth
```

The gate constructed its expected keys with `Path.as_posix()`, which uses forward slashes. Replacing the two backslashes in each recorded key with forward slashes makes the sets identical. Read-only SHA256 checks of both files matched the batch record and the frozen configuration:

| Frozen asset | Batch record | Current file | Frozen config |
|---|---|---|---|
| `runs/cur_v2rep05_s2/best/shared_mappo_model_21_3100.pth` | `d57aed25b3d46dffb2bd269ea87140953828ba7f51a960bb01995ec054dfca33` | same | same |
| `runs/joint_eval_20260916/ctde_final_7_1.pth` | `c7c256da0fc20451c1104996ce494148f567e4f83acf265d9baae8f0f053500d` | same | same |

This identifies a Windows path-separator mismatch in the gate's set comparison. It does not indicate that the checkpoint files or their hashes differ. The gate nevertheless returned failure, so its overall status remains failed; no manual pass has been substituted.

## Work completed before the stop

- The reserve retry for `immediate_reference_scene1300000` passed its formal gate: reserve attempt 2, 1000 steps, one constructor and one underlying `run()` call, task count and reward both 6, and airtime tap and spend both 14,916 bytes.
- All 64 immediate reference cells completed and closed.
- Seed 11 episodes 0–3 completed from the frozen ep0 initialization.
- One PPO update completed: 1 update call and 12 Adam steps. Four checkpoint write calls were recorded. The first-batch gate stopped after earlier likelihood, objective, dual, gradient and parameter checks and at the final frozen checkpoint path-set assertion.
- No seed 11 later episodes, seed 22/33 training, model evaluation or rule evaluation ran.

## Ledger and interpretation

The isolated ledger records 68 normal claims (including the original failed normal slot), 1 reserve claim, 69 cumulative constructors and `JointRun` constructions, 68 underlying `run()` calls, 68,000 completed steps, 68 completed logical jobs, 1 PPO update and 12 Adam steps. The single failed claim is the original ROUND42 read-only-method assignment failure. There were no new technical-failure claims.

The 64 immediate references are closed. The four seed 11 rollouts and their recorded update remain preserved execution artifacts, but the first-batch gate did not pass. They are not reported as validated training results, and no continuation or checkpoint restore was attempted. Further execution would require a separately reviewed path-gate repair and an explicit recovery decision consistent with the frozen execution identity and consumed quota.

The original ROUND42 failure claim, stop outputs, empty log and ledger remain unchanged. The retry gate and logical-job ledger preserve the link from the failed normal attempt to reserve attempt 2.
