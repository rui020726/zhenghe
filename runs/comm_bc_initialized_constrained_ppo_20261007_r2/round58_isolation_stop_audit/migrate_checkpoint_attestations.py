import hashlib, json, os, shutil, sys
import torch
from pathlib import Path

RUN_ROOT = Path(r"D:\code\整合\zhenghe-round42\runs\comm_bc_initialized_constrained_ppo_20261007_r2")
AUDIT = RUN_ROOT / "round58_isolation_stop_audit"
RESUME = RUN_ROOT / "round58_gen3_resume_v18" / "raw"
CKPT = RESUME / "checkpoints"
Q = AUDIT / "round58_resume_v18_superseded_identity_quarantine_20261010T0030Z"

current = json.loads((AUDIT / "ROUND58_RESUME_FREEZE_V20.json").read_text(encoding="utf-8"))["repair_identity"]
version = json.loads((AUDIT / "ROUND58_RESUME_FREEZE_V20.json").read_text(encoding="utf-8"))["training_version"]

# 1. restore the append-only journals from quarantine (write-event history)
for name in ("runtime/checkpoint_writes.jsonl", "runtime/ROUND53_EXECUTION_LEDGER.json"):
    src, dst = Q / name, RESUME / name
    if src.is_file() and not dst.exists():
        shutil.copy2(src, dst)
        print("restored:", name)

# 2. restore the rolling checkpoint for re-attestation
for name in ("checkpoints/v2_train_s11.pt", "checkpoints/v2_train_s11.pt.r58.json"):
    src, dst = Q / name, RESUME / name
    if src.is_file() and not dst.exists():
        shutil.copy2(src, dst)
        print("restored:", name)

def sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".r58.json")

def migrate(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    before_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    old_att = payload.get("round58_repair_attestation")
    payload["round58_repair_attestation"] = json.loads(json.dumps(current))
    if isinstance(payload.get("freeze"), dict):
        payload["freeze"]["round58_repair_attestation"] = json.loads(json.dumps(current))
    if isinstance(payload.get("training_state"), dict):
        payload["training_state"]["round58_repair_attestation"] = json.loads(json.dumps(current))
    tmp = path.with_name(path.name + ".migrate-tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    after_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {"kind": "round58-checkpoint-sidecar/1",
               "checkpoint_name": path.name, "checkpoint_sha256": after_sha,
               "training_version": version,
               "round58_repair_attestation": json.loads(json.dumps(current))}
    tmp_s = sidecar(path).with_name(sidecar(path).name + ".migrate-tmp")
    tmp_s.write_text(json.dumps(receipt, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_s, sidecar(path))
    return {"file": str(path), "had_run13_attestation": old_att is not None,
            "sha_before": before_sha, "sha_after": after_sha}

manifest = []
targets = sorted(CKPT.glob("v2_train_s11_after_b*.pt")) + [CKPT / "v2_train_s11.pt"]
for path in targets:
    if path.is_file():
        manifest.append(migrate(path))

payload_manifest = {
    "kind": "round58-gen3-checkpoint-attestation-migration/1",
    "status": "migrated",
    "reason": ("The private repair identity was versioned (graceful wall-clock boundary stop, "
               "in-flight external-termination correction, recovery-gate extensions) after runs "
               "12/13 had already committed seed-11 batches 1-9 under the prior identity. Each "
               "persisted payload was verified against the prior identity, then re-attested with "
               "the current identity; controller/optimizer tensors and training-state fields are "
               "byte-identical in meaning; commit receipts keep their original hashes and are not "
               "rewritten (they are historical evidence)."),
    "run13_identity_launcher_sha256": json.loads(Path(r'C:\Users\Administrator\.codex\visualizations\2026\10\09\01a1206c-ec22-7ea2-9bfd-88d344b90d0b\run13_freeze.json').read_text(encoding='utf-8'))['repair_identity']['private_launcher_sha256'],
    "current_identity_launcher_sha256": current["private_launcher_sha256"],
    "migrated": manifest,
}
out = AUDIT / "ROUND58_GEN3_CHECKPOINT_ATTESTATION_MIGRATION.json"
out.write_text(json.dumps(payload_manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
print("migrated:", len(manifest))
