# ROUND75 S1F Git identity repair — Phase A

Request ID: `R75_S1F_GIT_IDENTITY_REPAIR_DRAFT_20261015`
Start HEAD: `70339f7efd57be07946cf8665815534675787517`
Start parent: `d57e93bbd1990eb5c162adb3f276435a0f35b8fb`
Branch: `exp/round75-native-ppo-qualification`
Output root: `runs/comm_bc_initialized_constrained_ppo_20261007_r2/round75_native_ppo_qualification/raw/s1f_probe/`

## Scope and result

A changed only the permitted branch-reading parameter in the shared scientific runner. The old line was:

```python
branch, branch_error = git_value("--abbrev-ref", "HEAD")
```

The final line is:

```python
branch, branch_error = git_value("rev-parse", "--abbrev-ref", "HEAD")
```

An unintended EOF newline added by the first patch was removed before final verification, so the final runner diff contains only the one semantic line. The S1F probe and parent were copied from the verified S1E entries with only request ID, child filename and `raw/s1f_probe` paths changed. Their 15-case logic, assertions, counters, masks and comparison behavior were not changed. S1E originals and output were not modified.

Phase A did not import real NumPy/Torch, start a CPU probe, model, optimizer, environment, CUDA, or B. It did not write fixture or B outputs. No scientific quota or ledger charge occurred.

## Frozen work identities

| file | bytes | SHA256 |
| --- | ---: | --- |
| `r75_s1_runner.py` (final A) | 98735 | `2bb718f7500afa316aee701e0a23c2d56eb89833a9d03d548d686026ca39fb66` |
| `r75_s1f_fixture_receipt_repair.py` | 22405 | `8bf6fec7c7af584dfa6ac29da711cd72e98c573bac75328a127f43c615294de5` |
| `r75_s1f_parent_capture.ps1` | 8179 | `690a2aef6fccda3dd0fe11382fe1bc43ae3eb0d286c447f5e8dbe54eb05aae37` |
| `test_r75_s1f_git_identity_repair.py` | 4954 | `ab8b58e4938e5632cd42edad7f23163c6e96ced8a8ef6fb5570fe62f76617e3c` |

Unchanged before/after: `r75_s1e_fixture_receipt_repair.py` `21f0105295fe5880565316e625f06d58e2757a38131aa7f5517fdc9c7159af40`; `r75_s1e_parent_capture.ps1` `50f555ef717e82f0b00ab0fb4dad0ea72771546724b5881835b5aa75bec49471`; `r75_s1_outer_exec_v2.py` `bfe999ad3df01329d4cbca0051060f949ac8d747009983d00067df9ae8d1ddff`; `r75_frozen.py` `83813d6e06971a1e1fc1e25d9e9f94b0fa72310e300e037c0fa73d539c540b80`; `r75_guard.py` `4d730f71fb34ad411043c65141b6bcbb3f7d2f3c4668a6b53054e6400e35d55f`; `r75_replay_adapter.py` `0034fa40fbb55362646c84e8ef987e1aa01d72c619bd1c82a244dc88a0224dc2`; central ledger `fff8dc1f3d9222ed280a5dc9139f3f82de4c5537ccbc69894d340b44e605475d`.

## Narrow tests

Exactly 3 distinct top-level pure-stdlib/fake cases were added and no old test was rerun. They use `subprocess.run` fakes around the real `_execution_identity`; they do not replace the helper itself. Case 1 proves the valid `rev-parse --abbrev-ref HEAD` protocol and rejects the old `--abbrev-ref HEAD` call. Case 2 proves all three Git failures return `unknown` and preserve errors. Case 3 proves the default stage helper binds the real `_execution_identity` result through a fake transport and a fake default `torch.save`; it does not import real Torch/NumPy.

Attempt 1 failed with one test-infrastructure error because the default OS temporary directory was not writable in this sandbox. The error and exit were retained. The test was then changed only to put its scratch directory under the repo output root. Attempts 2 and 3 both ran 3 tests and passed with exit 0; attempt 3 revalidated the final runner bytes after removing the accidental EOF newline.

| attempt | result | exit | log SHA256 |
| ---: | --- | ---: | --- |
| 1 | 3 tests, 2 passed, 1 sandbox temp-directory error | 1 | `62dae89eba1b1f1916ad5e2bdb139fb84f4af61fa9ce7c8b29aeb281caac50db` |
| 2 | 3 tests, all passed | 0 | `ab85af13eb78708482d7a3cc44c3581617328000ddc006e0b6f3f452e75dd52d` |
| 3 | 3 tests, all passed against final bytes | 0 | `6969868684b859bd5b8eef6f8d5b779d046bfa46c91c69c468627105347626bd` |

Final attempt 3 output:

```text
test_default_stage_helper_uses_real_identity_wiring ... ok
test_execution_identity_git_failure_returns_unknown_and_errors ... ok
test_execution_identity_uses_valid_git_protocol_and_real_branch ... ok
Ran 3 tests in 0.029s
OK
```

## A consumption

Pure test Python processes: 3 of 3. Top-level test cases: 3. Real S1F B processes: 0. Parent launch attempts: 0. Real Torch/NumPy imports: 0. Fixture save/load/read: 0. Scientific checkpoint/model loads, forward/backward, optimizer/PPO, environments and CUDA: 0. Ledger charges: 0. Historical S1E counts and failures remain unchanged, including attempt 8's 25 distinct top-level cases, 12 S1E pure test processes and 4 PowerShell selftest processes.

## Boundary

This is only the A source freeze. It does not execute S1F B and does not prove the real Torch weights-only fixture flow. Decision-side review must inspect these exact frozen sources before the separately authorized one-process B continuation.
