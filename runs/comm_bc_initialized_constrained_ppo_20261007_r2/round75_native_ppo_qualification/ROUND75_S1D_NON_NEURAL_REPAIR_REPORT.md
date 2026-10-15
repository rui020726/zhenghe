# ROUND75 S1D non-neural repair close-out

Request ID: R75_S1D_NON_NEURAL_REPAIR_20261015. Branch: exp/round75-native-ppo-qualification. Start HEAD: b0e04baf764b77987094dbe4bc49b0218066e07f; parent: 328225980bfa6f0483f4914b5c7be3aa7da246dd.

## Result

STATUS: PHASE_B_FAILED_AT_WEIGHTS_ONLY_FIXTURE_LOAD_AND_STOPPED. The one authorized real CPU NumPy/Torch process ran and exited 1. The first 14 comparator cases were constructed and executed before case 15, but the process failed before writing phase_b_result.json, so no pass/fail claim is made for those cases and no before/after full RNG result is available. The stage save succeeded, then the authorized torch.load(path, map_location=cpu, weights_only=True) failed because the self-created fixture contains a NumPy key array and PyTorch weights-only mode rejects numpy._core.multiarray._reconstruct by default. No repair, retry, extra process or extra test was attempted after this failure.

## Repairs retained

Current source SHAs: runner 872f8bcd6c6c7d92c699e7887cf81d39fbf1f6218b31bf713160d907a148eae7; wrapper bfe999ad3df01329d4cbca0051060f949ac8d747009983d00067df9ae8d1ddff; S1D probe ee4ebb410ff0ed88886587f32133238f57f17cb1d484c7b88191dad3bd6b4224; S1D test e291a3f6ab7c0ba1cff89ab2699f020cbbb0399130aadcfbd6d317f04036e35a. These equal the B frozen source identities.

Core anchors in r75_s1_runner.py: complete RNG fingerprints at _numpy_rng_state_fingerprint:108 and _rng_identity_full:130; comparator at _raw_equal:152, _first_raw_difference:192 and _rng_payload_differences:222; execution identity at _execution_identity:1055; stage save at _save_stage_snapshot:1069; run_s1 stage calls at :1372 and :1421. The comparator now recognizes tensor-like objects with either tobytes or a numpy view, requires the same concrete type, dtype, shape and full bytes, and returns Python bools. _first_raw_difference uses the same payload detection.

Stage save now uses the real torch.save ABI, save_fn(payload, temporary), not save_fn(temporary, payload). Its payload records plan_frozen_head and plan_frozen_head_is_not_execution_head separately from execution_identity from git rev-parse, plus code_identity and data_identity. run_s1 preserves the intended order: A2 update, A2 stage save, A3 construction, restore/start gate, A3-start stage save, then A3 paid update.

S1D probe helpers at r75_s1d_non_neural_probe.py: capture_process:31 retains actual stdout/stderr bytes and hashes; consume_once:42 creates a separate O_EXCL consumption record without rewriting the pre-run manifest; _build_cases:71 and _stage_case:136 implement the frozen B matrix.

## A phase

The S1D pure stdlib suite has 16 top-level cases and finished 16/16 OK, exit 0. Its development history is retained: run 1 had 1 failure and 4 errors, run 2 had 2 failures, run 3 had 1 failure, and run 4 passed. The defects found were a temporarily missing _rng_payload_differences during the rewrite, a missing subprocess import, a Windows rather than Posix Path assertion, fake Python RNG state, and an imprecise branch-command matcher. Fixes were confined to implementation or the new S1D test as appropriate. NumPy/Torch were not imported by A tests.

## B phase and consumption

The B manifest froze the source SHAs above, the 15-case matrix, allowed/forbidden sets and B-before hashes for the central runtime nine files plus the nine S1B attempt originals. An initial Node spawnSync launch failed with EPERM before starting Python; its zero-byte capture and parent record are retained. The actual Python process was then launched once with PowerShell Start-Process and redirected to phase_b_stdout.bin and phase_b_stderr.bin. It exited 1 with 0 stdout bytes and 2014 stderr bytes. The failure is UnpicklingError from torch.load weights_only=True: unsupported global numpy._core.multiarray._reconstruct.

Actual consumption: one real Python CPU process; NumPy/Torch imports occurred; torch.save succeeded once on the self-created 19625-byte fixture; torch.load was attempted once and failed; successful RNG-fixture reads were 0; scientific/model/checkpoint loads were 0. There were no model load/construction, forward/backward, optimizer/PPO, environment, CUDA, set_state/seed/sampling, S1 or central-ledger charge events. Pure stdlib Python invocations including compile checks numbered 8. The prior Node launch attempt is an additional failed launcher attempt, not a Python process.

Probe originals: phase_b_manifest.json SHA 88aced73697d8478a1751fdaa46a3661ad4f63b83cc072771cd77993b726aac1; phase_b_consumed.json SHA 484816591055fceae69659b8d9d17fe2f53e9dec61e1b4cbc2d271b607ec1198; actual stderr SHA aafbbd64bc7170f7ee5cf0fb299f081294f2073b92278a40453abbbe84a7fc4e; fixture SHA 675297f549808b2a2ecbc088ae9b6662a46f31441e218f62be3ef6b179c53b74. The manifest retains process_consumed=false as a pre-run field; actual consumption is the separate O_EXCL record and this report.

## Invariants and boundary

All 18 B-before central runtime and S1B attempt files remained byte-identical after B. The ledger remains 19980 bytes, SHA fff8dc1f3d9222ed280a5dc9139f3f82de4c5537ccbc69894d340b44e605475d. A1 bundle remains 54708 bytes, SHA 23f34508357d0bfaa7b51a6f0ec34dc6462dfab603f402477c813f6e60596de0. ROUND46 protected files remain cf3ad1984d0d8ec526d53a57fab98bd43963d5eec5de7f45aec23d3396b2b39b and 93a2f69ff0c6cd2f8e153d4a165ee39d618f2f326343039b59f03c1588d03e18.

The comparator and stage-save wiring are not backend-qualified. S1B A3 restore qualification remains unproven. This task does not authorize S1, additional physical steps, technical reserve use, automatic rerun of the original qualification matrix, or any policy/deployment claim. The weights-only fixture incompatibility is a separate future repair decision.

## Evidence-boundary erratum

A correction record was appended at raw/s1d_probe/phase_b_parent_capture_actual_correction.json with SHA 17fccfda0a3909573a8bbfdb4c48548e2ddbf59f5999e5e102eea352f40ccf9e. The original Node EPERM record remains phase_b_parent_capture.json; it is a failed launcher attempt with no evidence that Python started, and it is not the actual child parent capture. The actual PowerShell Start-Process capture is phase_b_parent_capture_actual.json.

The actual process evidence is limited to: one CPU Python process, exit 1, one completed save of the self-created RNG fixture, and one failed torch.load attempt at case 15. Scientific checkpoint/model loads remained zero by actual call boundary. The child PID, exact process start/end timestamps and native OS byte-level stdio fidelity are not proven. The redirected files are evidence only by existence, length and SHA: stdout 0 bytes and stderr 2014 bytes with SHA aafbbd64bc7170f7ee5cf0fb299f081294f2073b92278a40453abbbe84a7fc4e.

Because main wrote phase_b_result.json only after case 15, the first 14 comparator cases are not individually durably recorded and no 14-pass claim is made. Complete before/after RNG identity is not proven. The pre-run manifest process_consumed=false is not evidence of non-consumption.

Current core source excerpts and identities remain: r75_s1_runner.py SHA 872f8bcd6c6c7d92c699e7887cf81d39fbf1f6218b31bf713160d907a148eae7, r75_s1d_non_neural_probe.py SHA ee4ebb410ff0ed88886587f32133238f57f17cb1d484c7b88191dad3bd6b4224, test_r75_s1d_non_neural_repair.py SHA e291a3f6ab7c0ba1cff89ab2699f020cbbb0399130aadcfbd6d317f04036e35a, r75_s1_outer_exec_v2.py SHA bfe999ad3df01329d4cbca0051060f949ac8d747009983d00067df9ae8d1ddff. No source, test, case or process was changed or rerun for this erratum.
