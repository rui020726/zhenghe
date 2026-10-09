import hashlib, json, shutil
from pathlib import Path
RUN_ROOT = Path(r"D:\code\整合\zhenghe-round42\runs\comm_bc_initialized_constrained_ppo_20261007_r2")
RESUME = RUN_ROOT / "round58_gen3_resume_v18" / "raw"
AUDIT = RUN_ROOT / "round58_isolation_stop_audit"
QUARANTINE = AUDIT / "round58_resume_v18_superseded_identity_quarantine_20261009T2230Z"
files = [
    RESUME / "checkpoints" / "v2_train_s11.pt", RESUME / "checkpoints" / "v2_train_s11.pt.r58.json",
    RESUME / "checkpoints" / "v2_train_s11_before_b1.pt", RESUME / "checkpoints" / "v2_train_s11_before_b1.pt.r58.json",
    RESUME / "runtime" / "checkpoint_writes.jsonl", RESUME / "runtime" / "ROUND53_EXECUTION_LEDGER.json",
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
    "kind": "round58-superseded-identity-quarantine/1", "status": "quarantined_not_deleted",
    "reason": ("Third aborted continuation (attempt3 episode completed but artifact-receipt "
               "bookkeeping crashed on the historical RUN_DIR binding; the interrupted-bookkeeping "
               "correction seam and recovery-gate allowance were then versioned). The resume-boundary "
               "re-writes of the restored episode4/update1 state are quarantined hash-closed; the "
               "attempt3 claim, row, tape and companions stay in place in gen3/resume roots."),
    "consumed_claims": 0, "environment_constructions": 0, "optimizer_steps": 0,
    "ppo_updates": 0, "evaluation_cells": 0,
    "attempt3_row_persisted_under": "round58_gen3_resume_v18/raw/rows/v2_r44_train_s11_e4_s1300003_a3.json",
    "quarantined_at_local": "2026-10-09T22:30:00+08:00", "files": manifest,
}
(QUARANTINE / "MANIFEST.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
print("quarantined:", len(manifest))
