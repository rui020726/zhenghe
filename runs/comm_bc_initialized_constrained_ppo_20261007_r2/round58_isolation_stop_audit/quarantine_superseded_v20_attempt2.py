import hashlib, json, shutil
from pathlib import Path

RUN_ROOT = Path(r"D:\code\整合\zhenghe-round42\runs\comm_bc_initialized_constrained_ppo_20261007_r2")
RESUME = RUN_ROOT / "round58_gen3_resume_v18" / "raw"
AUDIT = RUN_ROOT / "round58_isolation_stop_audit"
QUARANTINE = AUDIT / "round58_resume_v18_superseded_identity_quarantine_20261009T2150Z"

files = [
    RESUME / "checkpoints" / "v2_train_s11.pt",
    RESUME / "checkpoints" / "v2_train_s11.pt.r58.json",
    RESUME / "checkpoints" / "v2_train_s11_before_b1.pt",
    RESUME / "checkpoints" / "v2_train_s11_before_b1.pt.r58.json",
    RESUME / "runtime" / "checkpoint_writes.jsonl",
    RESUME / "runtime" / "ROUND53_EXECUTION_LEDGER.json",
]
manifest = []
for src in files:
    if not src.is_file():
        continue
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    dest = QUARANTINE / src.relative_to(RESUME)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    manifest.append({"original_path": str(src), "quarantined_path": str(dest), "sha256": digest})

payload = {
    "kind": "round58-superseded-identity-quarantine/1",
    "status": "quarantined_not_deleted",
    "reason": ("Aborted continuation attempts (worktree guard, then superseded freeze-argument "
               "selector) wrote hash-identical resume-boundary checkpoints of the restored "
               "episode4/update1 state (sha256 21fda8957471a21d6ae07bba2e9874d0dcfe35b5a4fca182221eeb6b86aa5e65) "
               "before the selector-freeze fix; moved aside hash-closed per the frozen quarantine "
               "discipline. No claims, no environment, no optimizer work occurred."),
    "consumed_claims": 0, "environment_constructions": 0, "optimizer_steps": 0,
    "ppo_updates": 0, "evaluation_cells": 0,
    "quarantined_at_local": "2026-10-09T21:50:00+08:00",
    "files": manifest,
}
(QUARANTINE / "MANIFEST.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
print("quarantined:", len(manifest))
