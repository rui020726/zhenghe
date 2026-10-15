# ROUND75 S1C comparator repair close-out

Request ID: R75_S1C_COMPARATOR_REPAIR_20261015. Branch: exp/round75-native-ppo qualification. Start HEAD: 12e68e6eed33b07b64b2d0e9756d5c612f214e5a.

## Result

STATUS: EXECUTED_AND_FAILED_AT_PHASE_C_WITH_PARTIAL_POST_FAILURE_SOURCE_REVISION. The one authorized real CPU NumPy/Torch comparator process ran and exited 1 with 14 cases: 11 passed and 3 failed. No second real backend process was run. The failure exposed that real torch.Tensor has no tobytes, so the then-current _raw_equal fell through to its unknown-object branch. After the failure, a limited source/test revision was made before the stop instruction, but it is explicitly NOT a repaired or backend-verified version: current _raw_equal still enters the tensor branch only through hasattr(tobytes), while current _first_raw_difference has partial NumPy-view compatibility. A new pure torch-like counterexample was added; the resulting 19-case pure suite exited 1 with that one failure.

## Actual code changes

Current source SHAs: runner f588c790a4538a16a01daecbc470b1a4506bcae4c9ff83e035bacc1fa335f020; wrapper bfe999ad3df01329d4cbca0051060f949ac8d747009983d00067df9ae8d1ddff; pure test 35feac1f6eed0ede00ab238925463161d2a7b5f0efd741a4942e9b9cc0e82d7e.

Phase C frozen source SHAs were: runner 58de83bb263ae79f45dc52dc6b5d16458dd6a5bfed0075a92041392d9d5f9c25; wrapper bfe999ad3df01329d4cbca0051060f949ac8d747009983d00067df9ae8d1ddff; test bf15c43b5cb0fcacdf4b3952661cb238afff3af202d6f9df753cbe05ab80b4f2. Thus the wrapper was unchanged after Phase C; runner and test changed afterward and are unverified on real NumPy/Torch.

Core changes and line anchors:

1. Runner comparator: _raw_equal at r75_s1_runner.py:110, _first_raw_difference at :146, and _rng_payload_differences at :171. They now avoid the previous live == expected fallback for tensor-like payloads, return Python bools, compare tuple/list/dict structure, and report byte/dtype/shape differences without boolean ambiguity. The Phase C process tested the pre-fail version at the frozen SHA above; the post-fail _first_raw_difference NumPy-view revision is not real-backend verified.

2. Wrapper ledger reconciliation: ledger_snapshot at r75_s1_outer_exec_v2.py:115 now reads each guard JSON and the total ledger, sums charged n by category, checks used_after, guard top-level cumulative values, guard event lists, refusal counts and event counts. It excludes ledger-opened from the guard/ledger event count and turns missing or contradictory schema into unknown plus mismatch. verify_pre_charge blocks on guard_ledger_reconciliation.ok not true.

3. Runner stage durability: _save_stage_snapshot at r75_s1_runner.py:1010 and _completed_stage_artifacts at :1052. run_s1 calls A2 stage save at :1303 before A3 controller construction, and A3 start-gate save at :1352 before the A3 paid update. The atomic payload carries the record, counters, unread, code identity and full state; failures are handled without a successful target or temp file. Failure receipts now include completed_stage_artifacts, citing and re-hashing A1/A2/A3-start artifacts already in the attempt record.

## Test and process record

Before Phase C, the new pure stdlib suite had 18 cases. Its first run had 2 failures and 1 error; these exposed real unit-ledger filtering/count defects and a missing stage-snapshot unread identity field. After fixes it ran 18/18 OK. A focused zero-fixture run exposed that the fixture itself omitted the other three guards; after making the fixture complete, the 18-case suite ran OK again. These runs were pure stdlib and did not import NumPy/Torch or canonical production material.

After Phase C failure and before the stop instruction, one pure test and a partial comparator source revision were added. The suite then ran 19 cases with 18 passes and 1 failure: test_torch_like_payload_without_tobytes_is_compared_through_numpy_view. That failure is retained. No further implementation, test or real-process work was attempted.

The one Phase C process imported numpy 2.2.6, torch 2.9.0+cu128 and r75_s1_runner only. It loaded no forbidden module, did not call set_state/manual_seed/random.seed/torch.load, and performed no model load/construction, forward/backward, optimizer/PPO or environment work. Its three case failures were torch_same_clone, mixed_payload_same and rng_differences_all_same; all three trace to the same torch Tensor/tobytes defect. RNG identity before and after was unchanged, but this claim is limited: it hashes the Python repr, the NumPy key-array from get_state()[1] only, and the torch state bytes. It does not prove the complete NumPy RNG state unchanged.

Phase C originals: raw/s1c_probe/phase_c_manifest.json SHA 04f5172c9ee32750d221e57f25608edd4c20df93b348a395347a19735d129425; phase_c_runner.py SHA 77c5ed39bfa4cbbecf1571837c9c668235cffa1f2ed44c3830e4bef0104549a5; phase_c_result.json SHA 717e0f361e4cc4383e90384baf3aa1457b3e6dc9f041b818095504317bbb9bd4; phase_c_stdout.txt SHA 717e0f361e4cc4383e90384baf3aa1457b3e6dc9f041b818095504317bbb9bd4. The pre-run manifest retains process_consumed=false as a frozen pre-run state; the result files and this report are the authoritative consumed/failed record.

## Consumption and invariants

This S1C repair task added: one real CPU NumPy/Torch process; no real S1 run; no model load/construction; no forward/backward; no real or synthetic optimizer/PPO update; no environment; no sampling; no torch.load; no checkpoint write; no RNG restore; no production-ledger charge; no new claim. Pure stdlib Python invocations, including compile checks and test executions, were 12. Technical failures recorded by this task: three pure-suite failures/errors in the first run, one Phase C real-process failure, and one post-failure pure-suite failure. No failure was refunded, reclassified or retried.

The prior S1B consumption remains unchanged: physical_optimizer_step 5, offline_trunk_target_row 840 (70 trunk calls), joint_record 1, environment_construction 0. Current ledger remains 88 events, 19980 bytes, SHA fff8dc1f3d9222ed280a5dc9139f3f82de4c5537ccbc69894d340b44e605475d. The A1 bundle remains 54708 bytes, SHA 23f34508357d0bfaa7b51a6f0ec34dc6462dfab603f402477c813f6e60596de0. Current attempt/runtime file hashes and mtimes are registered in the verdict; all listed S1B attempt and ledger files have mtimes before this S1C task. The ROUND46 protected files re-hash to cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b and 93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18.

## Boundary

This work does not qualify A3 restore, does not authorize or execute S1, and does not support any policy effect claim. The post-failure source is not a completed comparator repair and must not be described as backend-verified or all-pass. Future repair is a separate decision.
