# ROUND42-R attempt 2: isolated resume adapter

This directory is a separate execution identity. Attempt 1 remains stopped and its original
start marker, traceback, gate result, exact-recovery result, progress, ledger, and report are
preserved at runs/comm_one_shot_gate_resume_20261003. A byte-identical evidence snapshot
also exists under source_snapshot/previous_resume_attempt01 in this directory.

Attempt 1 passed the complete seed 11 batch-0 read-only gate and exact episode-4 recovery
check, then failed in the frozen ROUND40 train_seed resume branch with
UnboundLocalError for parameter_identity. The function imported that name only inside its
fresh-start branch. The failure happened before a claim: zero attempt-1 claims, environment
constructions, starts, completed steps, PPO/Adam updates, checkpoint writes, or new rows.
The cumulative ledger therefore remains 68 normal slots + one earlier reserve attempt = 69.

Attempt 2 uses r40_train_resume_adapter.py, a new local copy of the frozen ROUND40 trainer.
The only source change moves the parameter_identity import to the start of train_seed and
leaves fresh_controller imported in the fresh-start branch. The frozen external ROUND40
source and all scientific or training dependencies remain untouched. AST comparison removes
only that moved import and then verifies the complete original and adapter module trees match.

Pre-freeze checks passed:

- 9 real gate-path identity regression cases, including the former slash mismatch and the
  required rejection cases.
- 2 resume-adapter AST regression checks.
- Python bytecode compilation for the versioned resume, gate, adapter, and adapter test.
- The 69 claims, 69 slots, 68 rows, 6 checkpoints, 2 progress records, and checkpoint-write
  journal copied into this isolated attempt match the old run byte-for-byte.
- All 8 preserved attempt-1 stop/gate/recovery evidence files match byte-for-byte.

These checks used zero environments, zero optimizer steps, and zero checkpoint writes.
Attempt 2 must rerun the actual complete gate and exact recovery verification before the first
new environment claim. The fixed plan remains 604 logical jobs, within the original 700-start
cumulative hard cap. No push, deployment, or additional experiment is authorized.
