import hashlib, json, shutil, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

RUN_ROOT = Path(r"D:\code\整合\zhenghe-round42\runs\comm_bc_initialized_constrained_ppo_20261007_r2")
RESUME = RUN_ROOT / "round58_gen3_resume_v18" / "raw"
AUDIT = RUN_ROOT / "round58_isolation_stop_audit"
label = sys.argv[1] if len(sys.argv) > 1 else datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%SZ")
QUARANTINE = AUDIT / f"round58_resume_v18_superseded_identity_quarantine_{label}"

files = [
    RESUME / "checkpoints" / "v2_train_s11.pt", RESUME / "checkpoints" / "v2_train_s11.pt.r58.json",
    RESUME / "checkpoints" / "v2_train_s22.pt", RESUME / "checkpoints" / "v2_train_s22.pt.r58.json",
    RESUME / "checkpoints" / "v2_train_s33.pt", RESUME / "checkpoints" / "v2_train_s33.pt.r58.json",
]
files += [p for p in sorted((RESUME / "checkpoints").glob("v2_train_s*_before_b*.pt*"))]
files += [RESUME / "runtime" / "checkpoint_writes.jsonl",
          RESUME / "runtime" / "ROUND53_EXECUTION_LEDGER.json"]
manifest = []
for src in dict.fromkeys(files):
    if not src.is_file():
        continue
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    dest = QUARANTINE / src.relative_to(RESUME)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    manifest.append({"original_path": str(src), "quarantined_path": str(dest), "sha256": digest})
payload = {
    "kind": "round58-superseded-identity-quarantine/1", "status": "quarantined_not_deleted",
    "reason": ("An aborted continuation attempt wrote resume-boundary re-serializations of the "
               "restored episode4/update1 state under a repair identity that was subsequently "
               "versioned; moved aside hash-closed per the frozen quarantine discipline so the "
               "qualified gen3 sources remain the only resume inputs."),
    "consumed_claims": 0, "environment_constructions": 0, "optimizer_steps": 0,
    "ppo_updates": 0, "evaluation_cells": 0,
    "quarantined_at_local": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "files": manifest,
}
(QUARANTINE / "MANIFEST.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
print("quarantined:", len(manifest), "->", QUARANTINE.name)
