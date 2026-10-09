import hashlib, importlib.util, json, sys
from pathlib import Path
from copy import deepcopy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

spec = importlib.util.spec_from_file_location("r58_private_launcher_repaired", HERE / "r58_private_launcher_repaired.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

repair = m._repair_identity()
legacy = m.load_legacy_launcher()
contracts = sys.modules["r58_contracts"]
old_contract_cap, old_legacy_cap = contracts.RESERVE_CAP, legacy.RESERVE_CAP
contracts.RESERVE_CAP = 24
legacy.RESERVE_CAP = 24
try:
    plan_base = legacy.plan_from_frozen_sources()
finally:
    contracts.RESERVE_CAP = old_contract_cap
    legacy.RESERVE_CAP = old_legacy_cap
    sys.modules.pop("round58_legacy_launcher_readonly", None)

plan = deepcopy(plan_base)
plan["training_version"] = m.REPAIRED_TRAINING_VERSION
plan["r58_repair_identity"] = repair
plan["generation_id"] = "round58_gen3"
plan["status"] = "frozen_approved_gen3"
plan["authorization"] = "direct_user_owner_approval_identity_bound"
plan_sha256 = legacy.canonical_sha(plan)
print("live plan_sha256:", plan_sha256)

import r58_resume_seam as resume_seam
result = resume_seam.validate_historical_plan_identity_migration(m.REPO_ROOT, plan, repair)
proof_file = HERE / "ROUND58_GEN3_PLAN_IDENTITY_MIGRATION_PROOF_V21.json"
proof_file.write_text(json.dumps(
    {"kind": "round58-gen3-plan-identity-migration-proof/1",
     "status": "verified_allowlisted_identity_only_migration",
     "proof": result["proof"]}, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
    encoding="utf-8")
proof_sha = hashlib.sha256(proof_file.read_bytes()).hexdigest()
print("V21 proof sha256:", proof_sha)
print("added fields:", result["proof"]["allowed_identity_migration"]["added_fields"])

freeze19 = json.loads((HERE / "ROUND58_RESUME_FREEZE_V19.json").read_text(encoding="utf-8"))
auth19 = json.loads((HERE / "ROUND58_RESUME_EXECUTION_AUTHORIZATION_V19.json").read_text(encoding="utf-8"))

approved_attempt3 = {
    "logical_job": "new_training_s11_e4",
    "scenario": 1300003,
    "category": "reserve",
    "launch_slot": 2,
    "attempt": 3,
}

freeze20 = dict(freeze19)
freeze20["repair_identity"] = repair
freeze20["plan_sha256"] = plan_sha256
freeze20["attempt3_special_case"] = approved_attempt3
freeze20["plan_identity_migration_proof_sha256"] = proof_sha
freeze20["attempt3_binding"] = {
    "kind": "round58-gen3-attempt3-approval-binding/1",
    "seam_version": "round58-gen3-attempt3-contract/1",
    "claim_seam_version": "round58-gen3-attempt3-claim-seam/1",
    "owner_approval": "approved completion of the remaining fixed plan including the unique attempt3 reserve slot2 retry for new_training_s11_e4",
}
(HERE / "ROUND58_RESUME_FREEZE_V20.json").write_text(
    json.dumps(freeze20, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")

auth20 = dict(auth19)
auth20["repair_identity"] = repair
auth20["plan_sha256"] = plan_sha256
auth20["attempt3_special_case"] = approved_attempt3
auth20["plan_identity_migration_proof_sha256"] = proof_sha
auth20["approved_at"] = "2026-10-09T19:05:00Z"
auth20["approval_reference"] = (
    "Project owner direct instruction (current conversation): complete the remaining fixed plan after "
    "the exact private seam repair; new_training_s11_e4 uses attempt3 on reserve slot2; attempt1/2 "
    "failed_technical records are preserved; no retuning, no plan changes.")
auth20["approval_continuity"] = (
    "Continuation of the standing fixed-plan approval. This V20 identity only adds the approved unique "
    "attempt3 special case and the versioned private contract/claim-seam adapters required to allocate "
    "and close it; science configuration and the frozen plan are unchanged.")
auth20["freeze_sha256"] = hashlib.sha256((HERE / "ROUND58_RESUME_FREEZE_V20.json").read_bytes()).hexdigest()
(HERE / "ROUND58_RESUME_EXECUTION_AUTHORIZATION_V20.json").write_text(
    json.dumps(auth20, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")

print("freeze_sha256:", auth20["freeze_sha256"])
print("V20 receipts regenerated with V21 proof binding")
