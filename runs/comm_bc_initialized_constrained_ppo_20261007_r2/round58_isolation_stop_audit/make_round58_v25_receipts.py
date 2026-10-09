"""Append-only V25 private identity receipts for the approved fixed-plan continuation.

This preparatory script never starts a provider, optimizer or checkpoint writer.
It deliberately refuses to overwrite any V25 output; old V20 originals remain.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("r58_private_launcher_repaired",
                                              HERE / "r58_private_launcher_repaired.py")
launcher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launcher
spec.loader.exec_module(launcher)
import r58_resume_seam as resume_seam


def write_new(path: Path, value: dict) -> str:
    encoded = (json.dumps(value, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    names = ("ROUND58_GEN3_PLAN_IDENTITY_MIGRATION_PROOF_V25.json",
             "ROUND58_RESUME_FREEZE_V25.json",
             "ROUND58_RESUME_EXECUTION_AUTHORIZATION_V25.json")
    if any((HERE / name).exists() for name in names):
        raise FileExistsError("V25 identity generation is append-only; inspect existing receipt")
    repair = launcher._repair_identity()
    sources = resume_seam.validate_resume_sources(launcher.REPO_ROOT)
    legacy = launcher.load_legacy_launcher()
    contracts = sys.modules["r58_contracts"]
    old_contract, old_legacy = contracts.RESERVE_CAP, legacy.RESERVE_CAP
    contracts.RESERVE_CAP = legacy.RESERVE_CAP = 24
    try:
        plan = deepcopy(legacy.plan_from_frozen_sources())
    finally:
        contracts.RESERVE_CAP, legacy.RESERVE_CAP = old_contract, old_legacy
        sys.modules.pop("round58_legacy_launcher_readonly", None)
    plan["training_version"] = launcher.REPAIRED_TRAINING_VERSION
    plan["r58_repair_identity"] = repair
    plan["generation_id"] = "round58_gen3"
    plan["status"] = "frozen_approved_gen3"
    plan["authorization"] = "direct_user_owner_approval_identity_bound"
    plan_sha = legacy.canonical_sha(plan)
    migration = resume_seam.validate_historical_plan_identity_migration(
        launcher.REPO_ROOT, plan, repair, source_proof=sources)
    proof_sha = write_new(HERE / names[0], {
        "kind": "round58-gen3-plan-identity-migration-proof/1",
        "status": "verified_allowlisted_identity_only_migration",
        "proof": migration["proof"],
    })
    freeze = json.loads((HERE / "ROUND58_RESUME_FREEZE_V20.json").read_text(encoding="utf-8"))
    freeze.update({"repair_identity": repair, "plan_sha256": plan_sha,
                   "checkpoint_root": str(sources["target_checkpoint_root"]),
                   "plan_identity_migration_proof_sha256": proof_sha})
    freeze_sha = write_new(HERE / names[1], freeze)
    authorization = json.loads((HERE / "ROUND58_RESUME_EXECUTION_AUTHORIZATION_V20.json")
                               .read_text(encoding="utf-8"))
    authorization.update({
        "repair_identity": repair, "plan_sha256": plan_sha,
        "checkpoint_root": str(sources["target_checkpoint_root"]),
        "plan_identity_migration_proof_sha256": proof_sha,
        "freeze_sha256": freeze_sha,
        "approval_reference": (
            "Project owner continuation of the already approved ROUND58 fixed plan: "
            "recover exact committed seed11 batch9 and completed uncommitted episodes40-42 "
            "without new environment calls, dispose interrupted episode43 with preserved "
            "original evidence, then finish remaining fixed tasks within cumulative cap1064."),
        "approval_continuity": (
            "Private source/route version change only; no change to beta1.00, science "
            "configuration, scenario order, task matrix or approved cumulative quota."),
    })
    auth_sha = write_new(HERE / names[2], authorization)
    print(json.dumps({"plan_sha256": plan_sha, "proof_sha256": proof_sha,
                      "freeze_sha256": freeze_sha, "authorization_sha256": auth_sha},
                     indent=2))


if __name__ == "__main__":
    main()



\r\n