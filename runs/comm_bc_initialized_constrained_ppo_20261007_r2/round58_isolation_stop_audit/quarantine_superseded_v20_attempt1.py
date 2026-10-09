import hashlib, json, shutil
from pathlib import Path

RUN_ROOT = Path(r"D:\code\整合\zhenghe-round42\runs\comm_bc_initialized_constrained_ppo_20261007_r2")
RESUME = RUN_ROOT / "round58_gen3_resume_v18" / "raw"
AUDIT = RUN_ROOT / "round58_isolation_stop_audit"
QUARANTINE = AUDIT / "round58_resume_v18_superseded_identity_quarantine_20261009T2110Z"

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
        raise SystemExit(f"expected artifact missing: {src}")
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    dest = QUARANTINE / src.relative_to(RESUME)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    if hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
        raise SystemExit(f"quarantine hash mismatch: {src}")
    manifest.append({"original_path": str(src), "quarantined_path": str(dest),
                     "sha256": digest})

payload = {
    "kind": "round58-superseded-identity-quarantine/1",
    "status": "quarantined_not_deleted",
    "reason": ("The first authorized V20 continuation attempt wrote two hash-identical "
               "resume-boundary checkpoints (rolling re-save and pre-batch-1 snapshot of the "
               "restored episode4/update1 state, sha256 d726531cd43438b6ec8767d9ab6c13668899fddacdb00832269e094819305978) "
               "under a private repair identity that was subsequently extended (R45 one-retry "
               "gate seam). The current contract correctly refuses candidate sidecars attested "
               "by a superseded identity; per the frozen quarantine discipline the boundary "
               "artifacts are moved aside hash-closed, never deleted, and the qualified gen3 "
               "sources remain the only resume inputs."),
    "consumed_claims": 0, "environment_constructions": 0, "optimizer_steps": 0,
    "ppo_updates": 0, "evaluation_cells": 0,
    "attempt3_allocated_by_first_attempt": False,
    "quarantined_at_local": "2026-10-09T21:10:00+08:00",
    "files": manifest,
}
(QUARANTINE / "MANIFEST.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
remaining = sorted(p.name for p in RESUME.iterdir() if p.is_file())
print("quarantined:", len(manifest))
print("resume root files now:", remaining)
