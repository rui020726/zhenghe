"""Narrow R58 gen3 continuation seam; no trainer/math implementation lives here."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import re
import subprocess
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

MAP_REL = Path("runs/comm_bc_initialized_constrained_ppo_20261007_r2/round58_isolation_stop_audit/ROUND58_GEN3_APPROVED_RESUME_SOURCES_V11.json")
MAP_SHA256 = "ee9399e7fe893d409414ac83889161a740802c79be7c627e037bdb3c5aea41be"
R42_RUN_SHA256 = "9c5feacb2ffcb2bb930d2b6022e2fca5210408765dc664859d400e6f2b5f8c2e"
HISTORICAL_CHECKPOINT_JOURNAL_SHA256 = "2a43339ac45c2cfd13f84e744b72af0632024f44c3eedaa449bf8c8c7cab0165"
QUALIFIED = "qualified_for_later_resume_review"
PLAN_MIGRATION_VERSION = "round58-gen3-historical-plan-identity-migration/1"
HISTORICAL_PLAN_SHA256 = "af33de50bdda045f0e8391b3373b7188edfe83d80f145ab300cb823c01bd965c"
HISTORICAL_FREEZE_SHA256 = "253f0217e0a5d82f92d75d84db2843f087cf0d96d4219fe77a7106e9c69c9bf5"
HISTORICAL_AUTHORIZATION_SHA256 = "24223fc38f8f2d3adb145dc8f289a03d37c96628d6322c3976bd14d7aec825fa"
HISTORICAL_PLAN_GIT_PATH = "runs/comm_bc_initialized_constrained_ppo_20261007_r2/round58_isolation_stop_audit/ROUND58_GEN3_FIXED_PLAN.json"
HISTORICAL_PRIVATE_LAUNCHER_COMMIT = "c153dfd001043a7d3365364489ba588fb29adb41"
HISTORICAL_CHECKPOINT_CONTRACT_COMMIT = "b38b8688dee7ffc2626aa2064c8af74af75e216b"
HISTORICAL_PRIVATE_LAUNCHER_SHA256 = "56e0b12819b5193437d0ef9b19ca2c6df2040f088f716df5dc1a3874065340c4"
HISTORICAL_CHECKPOINT_CONTRACT_SHA256 = "c23dc0c89d8a6d423591cc684e47cf5a25130a279710243fc60dedf7c39ec95b"
MIGRATION_CHANGED_FIELDS = {
    "private_launcher_sha256", "checkpoint_contract_sha256",
}
MIGRATION_ADDED_FIELDS = {
    "ppo_counter_bridge_version", "identity_ledger_schema", "identity_ledger_sha256",
    "resume_seam_sha256", "approved_resume_map_sha256",
    "historical_checkpoint_journal_sha256",
    "train_pilot_source",
}
TRAINING_VERSION = (
    "constrained-cost-v2/one-shot-send-skip/1/full-trace-gae-lambda-1/1/"
    "dual-eta-lambda-0p5/1/rule-bc-late147-init/1/"
    "published-completion-event-at-relay-tick/2+r53-critic-representation-stop-gradient/1+"
    "r56-soft-early-report-prior-beta0p10/1+r58-prior-strength-beta1p00/1+"
    "r58-repaired-private-generation3/1+"
    "r58-cumulative-quota-history16-normal1024-reserve24-hard1064/2"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def map_path(repo_root: Path) -> Path:
    return Path(repo_root) / MAP_REL


def validate_resume_sources(repo_root: Path, *, manifest_path: Path | None = None,
                            expected_map_sha256: str = MAP_SHA256) -> dict[str, Any]:
    """Hash-close exactly the three qualified gen3 source checkpoints and receipts."""
    root = Path(os.path.abspath(os.fspath(repo_root)))
    path = Path(manifest_path) if manifest_path is not None else map_path(root)
    if sha256(path) != expected_map_sha256:
        raise PermissionError("approved resume-source map hash mismatch")
    manifest = read_json(path)
    if (manifest.get("kind") != "round58-gen3-approved-resume-source-map/1"
            or manifest.get("status") != "source_qualified_not_execution_authorization"
            or manifest.get("generation") != "round58_gen3"
            or manifest.get("training_version") != TRAINING_VERSION):
        raise PermissionError("approved resume-source map identity mismatch")
    audit_ref = manifest["qualification_audit"]
    verdict_ref = manifest["qualification_verdict"]
    audit_path = root / audit_ref["path"]
    verdict_path = root / verdict_ref["path"]
    if sha256(audit_path) != audit_ref["sha256"] or sha256(verdict_path) != verdict_ref["sha256"]:
        raise PermissionError("source qualification evidence hash mismatch")
    audit, verdict = read_json(audit_path), read_json(verdict_path)
    if (audit.get("status") != "all_three_actual_readonly_gates_passed"
            or verdict.get("status") != "readonly_gate_passed_all_three_qualified_for_later_resume_review"):
        raise PermissionError("source qualification evidence is not a three-seed pass")
    entries = manifest.get("seeds")
    if not isinstance(entries, list) or [int(e.get("seed", -1)) for e in entries] != [11, 22, 33]:
        raise PermissionError("source map must enumerate exactly unique seeds 11, 22, 33")
    checkpoint_root = root / manifest["source_root"]
    expected_cursor = manifest["expected_cursor"]
    verified: dict[int, dict[str, Any]] = {}
    for entry in entries:
        seed = int(entry["seed"])
        if entry.get("qualification") != QUALIFIED:
            raise PermissionError(f"seed {seed}: missing qualified source record")
        cp = checkpoint_root / entry["checkpoint"]
        sidecar = Path(str(cp) + ".r58.json")
        batch = checkpoint_root / entry["batch_record"]
        commit = checkpoint_root / entry["commit_receipt"]
        for file, key in ((cp, "checkpoint_sha256"), (sidecar, "sidecar_sha256"),
                          (batch, "batch_record_sha256"), (commit, "commit_sha256")):
            if not file.is_file() or sha256(file) != entry[key]:
                raise PermissionError(f"seed {seed}: approved historical file hash mismatch: {file.name}")
        sc, br, cr = read_json(sidecar), read_json(batch), read_json(commit)
        if (sc.get("kind") != "round58-checkpoint-sidecar/1"
                or sc.get("checkpoint_name") != cp.name
                or sc.get("checkpoint_sha256") != entry["checkpoint_sha256"]
                or sc.get("training_version") != TRAINING_VERSION
                or not isinstance(sc.get("round58_repair_attestation"), dict)):
            raise PermissionError(f"seed {seed}: original sidecar identity mismatch")
        if (int(br.get("training_seed", -1)) != seed or int(br.get("batch", -1)) != 0
                or br.get("episodes") != [0, 1, 2, 3]
                or int(br.get("optimizer_steps_after", -1)) != int(expected_cursor["optimizer_steps"])
                or br.get("training_version") != TRAINING_VERSION):
            raise PermissionError(f"seed {seed}: batch record is outside the qualified cursor")
        if (cr.get("kind") != "round40-batch-checkpoint-commit/1"
                or int(cr.get("training_seed", -1)) != seed or int(cr.get("batch", -1)) != 0
                or cr.get("batch_record") != batch.name
                or cr.get("batch_record_sha256") != entry["batch_record_sha256"]
                or cr.get("rolling_checkpoint") != cp.name
                or cr.get("rolling_checkpoint_sha256") != entry["checkpoint_sha256"]
                or cr.get("postupdate_checkpoint_sha256") != entry["checkpoint_sha256"]
                or int(cr.get("episodes_done", -1)) != int(expected_cursor["episodes_done"])
                or int(cr.get("updates", -1)) != int(expected_cursor["updates"])
                or int(cr.get("optimizer_steps", -1)) != int(expected_cursor["optimizer_steps"])):
            raise PermissionError(f"seed {seed}: durable batch commit does not close source")
        gate = next((x for x in audit.get("seeds", []) if int(x.get("seed", -1)) == seed), None)
        qualified = next((x for x in verdict.get("per_seed", []) if int(x.get("seed", -1)) == seed), None)
        if (gate is None or gate.get("status") != "passed"
                or qualified is None or qualified.get("qualification") != QUALIFIED
                or qualified.get("checkpoint_hashes_unchanged") is not True):
            raise PermissionError(f"seed {seed}: per-seed qualification evidence missing or failed")
        gate_cp_hashes = {c.get("sha256") for c in gate.get("serialized_identity_chain", {}).get("checkpoint_identity", [])}
        if entry["checkpoint_sha256"] not in gate_cp_hashes:
            raise PermissionError(f"seed {seed}: approved checkpoint absent from gate evidence")
        verified[seed] = {
            **entry, "path": cp.resolve(), "sidecar_path": sidecar.resolve(),
            "original_identity": copy.deepcopy(sc["round58_repair_attestation"]),
            "original_sidecar": sc,
        }
    checkpoint_journal = validate_historical_checkpoint_journal(root)
    return {"manifest": manifest, "manifest_sha256": expected_map_sha256,
            "sources": verified, "historical_ppo_carryover": len(verified),
            "historical_optimizer_steps": sum(8 for _ in verified),
            "historical_checkpoint_journal": checkpoint_journal,
            "target_checkpoint_root": (root / manifest["target_checkpoint_root"]).resolve()}


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _git_blob_sha(repo_root: Path, commit: str, relative_path: str) -> str:
    result = subprocess.run(["git", "show", f"{commit}:{relative_path}"], cwd=repo_root,
                            capture_output=True, check=False)
    if result.returncode:
        raise PermissionError(f"historical source blob unavailable: {commit}:{relative_path}")
    return hashlib.sha256(result.stdout).hexdigest()


def validate_historical_plan_identity_migration(
        repo_root: Path, current_plan: dict[str, Any], current_repair_identity: dict[str, Any],
        *, source_proof: dict[str, Any] | None = None) -> dict[str, Any]:
    """Prove the one approved identity-only migration without changing historical files.

    The fixed-plan comparison remains exact after an in-memory, allow-listed view
    substitutes only the new private execution attestation. Scientific plan fields,
    source-content identities, and the original source attestations are independently
    checked before that view can be installed.
    """
    root = Path(os.path.abspath(os.fspath(repo_root)))
    run_root = root / "runs/comm_bc_initialized_constrained_ppo_20261007_r2"
    audit = run_root / "round58_isolation_stop_audit"
    raw_plan_path = run_root / "round58_gen3/raw/ROUND53_FIXED_PLAN.json"
    frozen_plan_path = audit / "ROUND58_GEN3_FIXED_PLAN.json"
    historical_freeze_path = audit / "ROUND58_GEN3_FREEZE.json"
    historical_authorization_path = audit / "ROUND58_GEN3_EXECUTION_AUTHORIZATION.json"
    for path, expected in ((raw_plan_path, HISTORICAL_PLAN_SHA256),
                           (frozen_plan_path, HISTORICAL_PLAN_SHA256),
                           (historical_freeze_path, HISTORICAL_FREEZE_SHA256),
                           (historical_authorization_path, HISTORICAL_AUTHORIZATION_SHA256)):
        if not path.is_file() or sha256(path) != expected:
            raise PermissionError(f"historical plan identity evidence changed: {path.name}")
    raw_bytes = raw_plan_path.read_bytes()
    historical_plan = read_json(raw_plan_path)
    old_freeze = read_json(historical_freeze_path)
    old_authorization = read_json(historical_authorization_path)
    historical_identity = historical_plan.get("r58_repair_identity")
    if not isinstance(historical_identity, dict):
        raise PermissionError("historical fixed plan lacks its original R58 repair identity")
    if (old_freeze.get("kind") != "round58-repaired-execution-freeze/1"
            or old_freeze.get("status") != "frozen"
            or old_freeze.get("plan_artifact_sha256") != HISTORICAL_PLAN_SHA256
            or old_freeze.get("repair_identity") != historical_identity
            or old_authorization.get("status") != "explicitly_authorized"
            or old_authorization.get("freeze_sha256") != HISTORICAL_FREEZE_SHA256
            or old_authorization.get("plan_sha256") != old_freeze.get("plan_sha256")
            or old_authorization.get("repair_identity") != historical_identity
            or old_authorization.get("generation_id") != "round58_gen3"):
        raise PermissionError("historical plan/freeze/authorization chain does not close")

    verified_sources = source_proof or validate_resume_sources(root)
    if set(verified_sources.get("sources", {})) != {11, 22, 33}:
        raise PermissionError("historical identity migration requires exactly the three qualified seed sources")
    for seed, source in verified_sources["sources"].items():
        if source.get("original_identity") != historical_identity:
            raise PermissionError(f"seed {seed}: original sidecar attestation differs from historical plan identity")

    # The exact prior private code blobs must exist in Git history; matching only a
    # JSON hash field would not authenticate the old implementation identity.
    historical_launcher_path = "runs/comm_bc_initialized_constrained_ppo_20261007_r2/round58_isolation_stop_audit/r58_private_launcher_repaired.py"
    historical_contract_path = "runs/comm_bc_initialized_constrained_ppo_20261007_r2/round58_isolation_stop_audit/r58_checkpoint_contract_repaired.py"
    launcher_blob = _git_blob_sha(root, HISTORICAL_PRIVATE_LAUNCHER_COMMIT, historical_launcher_path)
    contract_blob = _git_blob_sha(root, HISTORICAL_CHECKPOINT_CONTRACT_COMMIT, historical_contract_path)
    if (launcher_blob != HISTORICAL_PRIVATE_LAUNCHER_SHA256
            or contract_blob != HISTORICAL_CHECKPOINT_CONTRACT_SHA256
            or historical_identity.get("private_launcher_sha256") != launcher_blob
            or historical_identity.get("checkpoint_contract_sha256") != contract_blob):
        raise PermissionError("historical private source code identity is not authenticated by its Git blobs")
    ancestry = subprocess.run(["git", "merge-base", "--is-ancestor",
                               HISTORICAL_PRIVATE_LAUNCHER_COMMIT, "HEAD"], cwd=root,
                              capture_output=True, check=False)
    if ancestry.returncode:
        raise PermissionError("historical private launcher identity commit is not an ancestor of current HEAD")
    ancestry = subprocess.run(["git", "merge-base", "--is-ancestor",
                               HISTORICAL_CHECKPOINT_CONTRACT_COMMIT, "HEAD"], cwd=root,
                              capture_output=True, check=False)
    if ancestry.returncode:
        raise PermissionError("historical checkpoint-contract identity commit is not an ancestor of current HEAD")

    old_source_id = historical_plan.get("source_content_identities")
    new_source_id = current_plan.get("source_content_identities")
    if old_source_id != new_source_id:
        raise PermissionError("source_content_identities differ; existing independent source verification cannot be reused")
    ignored_by_legacy = {"source_content_identities"}
    old_science = {k: v for k, v in historical_plan.items()
                   if k not in ignored_by_legacy | {"r58_repair_identity"}}
    new_science = {k: v for k, v in current_plan.items()
                   if k not in ignored_by_legacy | {"r58_repair_identity"}}
    if old_science != new_science:
        changed = sorted(k for k in set(old_science) | set(new_science)
                         if old_science.get(k) != new_science.get(k))
        raise PermissionError(f"non-attestation frozen plan fields changed: {changed}")

    old_keys, new_keys = set(historical_identity), set(current_repair_identity)
    added, removed = new_keys - old_keys, old_keys - new_keys
    changed = {k for k in old_keys & new_keys
               if historical_identity[k] != current_repair_identity[k]}
    if added != MIGRATION_ADDED_FIELDS or removed or changed != MIGRATION_CHANGED_FIELDS:
        raise PermissionError(
            "R58 repair-identity migration contains a non-allow-listed change "
            f"(added={sorted(added)}, removed={sorted(removed)}, changed={sorted(changed)})")
    for key in MIGRATION_ADDED_FIELDS:
        if not current_repair_identity.get(key):
            raise PermissionError(f"current repair identity lacks migrated field {key}")

    audit_dir = Path(__file__).resolve().parent
    frozen_train_pilot = (root.parent / "zhenghe" / "runs" /
                          "comm_budget_pilot_20260919" / "train_pilot.py").resolve()
    if not frozen_train_pilot.is_file():
        raise PermissionError("approved canonical train_pilot source is missing")
    train_pilot_sha = sha256(frozen_train_pilot)
    if train_pilot_sha != "87dcf2e5251c99fc533ba52480a43da89902a3a0c371f5b9cfec6240d202ae81":
        raise PermissionError("approved canonical train_pilot source hash changed")
    expected_current_metadata = {
        "private_launcher_sha256": sha256(audit_dir / "r58_private_launcher_repaired.py"),
        "checkpoint_contract_sha256": sha256(audit_dir / "r58_checkpoint_contract_repaired.py"),
        "ppo_counter_bridge_version": "round58-ppo-counter-bridge/2",
        "identity_ledger_schema": "round58-gen3-identity-close/2",
        "identity_ledger_sha256": sha256(audit_dir / "r58_identity_ledger_close_v2.py"),
        "resume_seam_sha256": sha256(Path(__file__).resolve()),
        "approved_resume_map_sha256": verified_sources["manifest_sha256"],
        "historical_checkpoint_journal_sha256":
            verified_sources["historical_checkpoint_journal"]["sha256"],
        "train_pilot_source": {
            "path": str(frozen_train_pilot), "sha256": train_pilot_sha,
            "frozen_manifest_sha256": train_pilot_sha,
            "binding_version": "round58-frozen-train-pilot-restore-import/1",
        },
    }
    if any(current_repair_identity.get(k) != v for k, v in expected_current_metadata.items()):
        raise PermissionError("current repair identity metadata differs from its live code/source bindings")
    if historical_identity.get("private_launcher_sha256") != HISTORICAL_PRIVATE_LAUNCHER_SHA256 \
            or historical_identity.get("checkpoint_contract_sha256") != HISTORICAL_CHECKPOINT_CONTRACT_SHA256:
        raise PermissionError("historical repair identity contains a non-authenticated private source hash")

    # Bind unchanged scientific/critic adapter sources too; the current adapter,
    # trainer and common source must still match the historical identity.
    candidate_identity = read_json(audit / "ROUND58_CANDIDATE_INPUT_IDENTITY.json")
    runtime_hashes = candidate_identity.get("runtime_source_hashes", {})
    private_hashes = candidate_identity.get("private_entry_hashes", {})
    checks = {
        "adapter_sha256": sha256(audit / "r58_gate_prior_repaired.py"),
        "trainer_sha256": runtime_hashes.get("v2_trainer.py", {}).get("sha256"),
        "common_sha256": runtime_hashes.get("v2_common.py", {}).get("sha256"),
    }
    if any(historical_identity.get(k) != v for k, v in checks.items()):
        raise PermissionError("historical source hashes for adapter/trainer/common do not match current frozen files")
    if (private_hashes.get("r58_gate_prior_repaired.py") != historical_identity.get("adapter_sha256")
            or private_hashes.get("r58_checkpoint_contract_repaired.py") != HISTORICAL_CHECKPOINT_CONTRACT_SHA256):
        raise PermissionError("candidate input identity does not corroborate historical private source hashes")

    allowed_view = copy.deepcopy(historical_plan)
    allowed_view["r58_repair_identity"] = copy.deepcopy(current_repair_identity)
    if ({k: v for k, v in allowed_view.items() if k != "source_content_identities"}
            != {k: v for k, v in current_plan.items() if k != "source_content_identities"}):
        raise PermissionError("allow-listed historical plan view does not exactly match current plan")

    body = {
        "kind": PLAN_MIGRATION_VERSION,
        "historical_plan_path": str(raw_plan_path.relative_to(root)).replace("\\", "/"),
        "historical_plan_sha256": HISTORICAL_PLAN_SHA256,
        "historical_plan_canonical_sha256": _canonical_sha(historical_plan),
        "historical_freeze_sha256": HISTORICAL_FREEZE_SHA256,
        "historical_authorization_sha256": HISTORICAL_AUTHORIZATION_SHA256,
        "historical_repair_identity_sha256": _canonical_sha(historical_identity),
        "current_plan_canonical_sha256": _canonical_sha(current_plan),
        "current_repair_identity_sha256": _canonical_sha(current_repair_identity),
        "source_content_identities_sha256": _canonical_sha(old_source_id),
        "source_map_sha256": verified_sources["manifest_sha256"],
        "qualified_seed_source_attestations": {
            str(seed): {"checkpoint_sha256": source["checkpoint_sha256"],
                        "sidecar_sha256": source["sidecar_sha256"],
                        "batch_record_sha256": source["batch_record_sha256"],
                        "commit_sha256": source["commit_sha256"],
                        "original_attestation_sha256": _canonical_sha(source["original_identity"])}
            for seed, source in sorted(verified_sources["sources"].items())},
        "historical_git_blobs": {
            "private_launcher": {"commit": HISTORICAL_PRIVATE_LAUNCHER_COMMIT,
                                 "sha256": launcher_blob},
            "checkpoint_contract": {"commit": HISTORICAL_CHECKPOINT_CONTRACT_COMMIT,
                                    "sha256": contract_blob}},
        "allowed_identity_migration": {
            "changed_fields": sorted(MIGRATION_CHANGED_FIELDS),
            "added_fields": sorted(MIGRATION_ADDED_FIELDS),
            "removed_fields": [],
            "scientific_plan_field_changes": [],
            "source_content_identity_changed": False},
        "actual_legacy_comparison_view": "historical raw plan with only r58_repair_identity replaced in memory by the current validated identity; original on-disk plan unchanged",
    }
    return {"proof": body, "historical_plan": historical_plan,
            "allowed_comparison_view": allowed_view,
            "source_proof": verified_sources,
            "historical_plan_bytes_sha256": hashlib.sha256(raw_bytes).hexdigest()}


def validate_committed_plan_migration_proof(repo_root: Path, proof_path: Path,
                                            current_plan: dict[str, Any],
                                            current_repair_identity: dict[str, Any],
                                            *, expected_sha256: str) -> dict[str, Any]:
    if sha256(proof_path) != expected_sha256:
        raise PermissionError("plan identity migration proof file hash differs from freeze/receipt")
    stored = read_json(proof_path)
    if (stored.get("kind") != "round58-gen3-plan-identity-migration-proof/1"
            or stored.get("status") != "verified_allowlisted_identity_only_migration"):
        raise PermissionError("plan identity migration proof status/schema mismatch")
    result = validate_historical_plan_identity_migration(
        repo_root, current_plan, current_repair_identity)
    if stored.get("proof") != result["proof"]:
        raise PermissionError("committed plan identity migration proof does not match live historical/current identities")
    result["proof_file_sha256"] = expected_sha256
    return result


def install_historical_plan_comparison_view(legacy: Any, *, repo_root: Path,
                                           current_plan: dict[str, Any],
                                           current_repair_identity: dict[str, Any],
                                           proof_path: Path,
                                           expected_proof_sha256: str) -> dict[str, Any]:
    """Wrap only legacy's persisted-plan read; preserve its original exact equality check."""
    verified = validate_committed_plan_migration_proof(
        repo_root, proof_path, current_plan, current_repair_identity,
        expected_sha256=expected_proof_sha256)
    original_read_json = legacy.read_json
    historical_path = (Path(repo_root) / verified["proof"]["historical_plan_path"]).resolve()
    view = copy.deepcopy(verified["allowed_comparison_view"])

    def migrated_read_json(path):
        if Path(path).resolve() == historical_path:
            original = original_read_json(path)
            if _canonical_sha(original) != verified["proof"]["historical_plan_canonical_sha256"]:
                raise PermissionError("historical fixed plan changed before legacy comparison")
            return copy.deepcopy(view)
        return original_read_json(path)

    legacy.read_json = migrated_read_json
    return {"kind": PLAN_MIGRATION_VERSION, "proof_sha256": expected_proof_sha256,
            "historical_plan_path": str(historical_path),
            "historical_plan_canonical_sha256": verified["proof"]["historical_plan_canonical_sha256"],
            "current_plan_canonical_sha256": verified["proof"]["current_plan_canonical_sha256"],
            "only_persisted_plan_read_is_normalized": True,
            "normalization_field": "r58_repair_identity",
            "legacy_exact_plan_equality_remains_enabled": True,
            "undo": lambda: setattr(legacy, "read_json", original_read_json)}


def validate_historical_checkpoint_journal(repo_root: Path) -> dict[str, Any]:
    """Validate the immutable 12 actual gen3 checkpoint-write events as a counter baseline."""
    root = Path(os.path.abspath(os.fspath(repo_root)))
    raw = root / "runs/comm_bc_initialized_constrained_ppo_20261007_r2/round58_gen3/raw"
    path = raw / "runtime" / "checkpoint_writes.jsonl"
    if sha256(path) != HISTORICAL_CHECKPOINT_JOURNAL_SHA256:
        raise PermissionError("historical gen3 checkpoint journal hash mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != 12:
        raise PermissionError(f"historical gen3 checkpoint journal expected 12 rows, got {len(rows)}")
    counts = {11: 0, 22: 0, 33: 0}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        value = str(row.get("path_at_write", ""))
        checkpoint_path = None
        for seed in counts:
            if re.search(rf"v2_train_s{seed}(?:_|\.pt)", value):
                counts[seed] += 1
                break
        else:
            raise PermissionError("historical checkpoint journal has an unknown seed/path")
        # The event path may include a temporary suffix; the committed checkpoint stem is
        # recoverable at the first .pt boundary and must remain inside gen3/raw/checkpoints.
        match = re.search(r"([^\\/]+?\.pt)(?:\.|$)", value)
        if match is None:
            raise PermissionError("historical checkpoint event path lacks a .pt checkpoint name")
        checkpoint_name = match.group(1)
        normalized_event_path = value.replace("/", "\\").casefold()
        required_suffix = f"\\round58_gen3\\raw\\checkpoints\\{checkpoint_name}".casefold()
        event_base = re.sub(r"\.\d+\.(?:milestone|snapshot)-tmp$", "", normalized_event_path)
        if not event_base.endswith(required_suffix):
            raise PermissionError("historical checkpoint event escaped the registered gen3/raw/checkpoints suffix")
        checkpoint_path = raw / "checkpoints" / checkpoint_name
        digest = str(row.get("sha256_at_write", ""))
        if len(digest) != 64 or not os.path.isfile(os.fspath(checkpoint_path)) \
                or sha256(checkpoint_path) != digest:
            raise PermissionError("historical checkpoint event does not match retained checkpoint bytes")
        if str(row.get("training_version", "")) != TRAINING_VERSION:
            raise PermissionError("historical checkpoint event has a different training version")
        key = (str(checkpoint_path).casefold(), digest)
        if key in seen:
            raise PermissionError("historical checkpoint journal repeats a path/hash event")
        seen.add(key)
        if int(row.get("episodes_done", -1)) not in (0, 4) or int(row.get("updates", -1)) not in (0, 1):
            raise PermissionError("historical checkpoint event has an unexpected training cursor")
    if counts != {11: 4, 22: 4, 33: 4}:
        raise PermissionError(f"historical checkpoint event distribution is not 4 per seed: {counts}")
    return {"path": str(path), "sha256": HISTORICAL_CHECKPOINT_JOURNAL_SHA256,
            "rows": len(rows), "per_seed_rows": counts,
            "evidence": "retained direct checkpoint-write journal, each event byte-closed to checkpoint"}


def _batch_records(run_root: Path, extra_root: Path) -> list[Path]:
    found: dict[str, Path] = {}
    for directory in (Path(run_root) / "checkpoints", Path(extra_root)):
        if directory.is_dir():
            for p in directory.glob("r40_batch_s*_b*_attempt*.json"):
                found[str(p.resolve()).casefold()] = p
    return sorted(found.values(), key=lambda p: str(p).casefold())


def carryover_count_from_batches(batch_records: list[Path], proof: dict[str, Any]) -> int:
    approved = proof["sources"]
    by_key: dict[tuple[int, int], Path] = {}
    for path in batch_records:
        row = read_json(path)
        key = (int(row.get("training_seed", -1)), int(row.get("batch", -1)))
        if key in by_key:
            raise RuntimeError(f"duplicate batch cursor in configure roots: {key}")
        by_key[key] = path
    for seed, source in approved.items():
        key = (int(seed), 0)
        if key not in by_key:
            raise RuntimeError(f"qualified batch carryover missing from live batch roots: {key}")
        old = read_json(source["path"].parent / source["batch_record"])
        got = read_json(by_key[key])
        if sha256(by_key[key]) != source["batch_record_sha256"] or got != old:
            raise RuntimeError(f"qualified batch carryover changed or copied ambiguously: {key}")
    return len(approved)


def install_r42_configure_compat(run_module: Any, repo_root: Path, *, extra_batch_root: Path,
                                 proof: dict[str, Any], resume_runtime_dir: Path) -> dict[str, Any]:
    """Bind actual R42 configure/ledger to immutable history plus direct gen3-resume events."""
    source_path = Path(run_module.__file__).resolve()
    if sha256(source_path) != R42_RUN_SHA256:
        raise PermissionError("R42 configure compatibility refused: frozen r42_run source hash mismatch")
    configured_snapshot = run_module.ledger_snapshot
    closure = dict(zip(configured_snapshot.__code__.co_freevars,
                       configured_snapshot.__closure__ or (), strict=True))
    base_cell = closure.get("base_ledger_snapshot")
    base_snapshot = None if base_cell is None else base_cell.cell_contents
    if (not callable(base_snapshot)
            or Path(base_snapshot.__code__.co_filename).resolve() != source_path
            or base_snapshot.__name__ != "ledger_snapshot"):
        raise RuntimeError("configured R42 ledger wrapper did not close over the hash-verified base ledger")
    ledger_source = textwrap.dedent(inspect.getsource(base_snapshot))
    ledger_anchor = '    batch_records = list((RUN_DIR / "checkpoints").glob("r40_batch_s*_b*_attempt*.json"))'
    old_counter_lines = '''    optimizer_steps_actual = int(_OPTIMIZER_STEP_CALLS)
    ppo_update_calls = int(_PPO_UPDATE_CALLS)'''
    old_journal_lines = '''        "checkpoint_write_journal_entries": sum(
            1 for _line in (RUN_DIR / "runtime" / "checkpoint_writes.jsonl").open(
                "r", encoding="utf-8")
        ) if (RUN_DIR / "runtime" / "checkpoint_writes.jsonl").is_file() else 0,'''
    if (ledger_source.count(ledger_anchor) != 1 or ledger_source.count(old_counter_lines) != 1
            or ledger_source.count(old_journal_lines) != 1):
        raise RuntimeError("R42 ledger source differs from the reviewed exact compatibility anchors")
    resume_runtime_dir = Path(resume_runtime_dir).resolve()
    optimizer_journal = resume_runtime_dir / "optimizer_step_calls.jsonl"
    ppo_journal = resume_runtime_dir / "ppo_update_calls.jsonl"
    checkpoint_journal = resume_runtime_dir / "checkpoint_writes.jsonl"
    run_module._R58_EXTRA_BATCH_ROOT = str(Path(extra_batch_root).resolve())
    run_module._R58_RESUME_RUNTIME_DIR = str(resume_runtime_dir)
    run_module._R58_OPTIMIZER_JOURNAL = str(optimizer_journal)
    run_module._R58_PPO_JOURNAL = str(ppo_journal)
    run_module._R58_CHECKPOINT_JOURNAL = str(checkpoint_journal)
    run_module._r58_batch_records = lambda base: _batch_records(
        Path(run_module.RUN_DIR), Path(run_module._R58_EXTRA_BATCH_ROOT))
    run_module._r58_history_ppo_carryover = int(proof["historical_ppo_carryover"])
    run_module._r58_history_batch_verifier = lambda rows: carryover_count_from_batches(rows, proof)
    ledger_source = ledger_source.replace(ledger_anchor,
        '    batch_records = _r58_batch_records(RUN_DIR / "checkpoints")', 1)
    new_counter_lines = '''    _optimizer_rows = _r58_read_direct_journal(Path(_R58_OPTIMIZER_JOURNAL), start=25)
    _ppo_rows = _r58_read_direct_journal(Path(_R58_PPO_JOURNAL), start=4)
    _r58_history_batch_verifier(batch_records)
    optimizer_steps_actual = 24 + len(_optimizer_rows)
    ppo_update_calls = _r58_history_ppo_carryover + len(_ppo_rows)
    if int(_OPTIMIZER_STEP_CALLS) != optimizer_steps_actual:
        raise RuntimeError("ROUND58 live Adam hook counter differs from direct journal plus history")
    if int(_PPO_UPDATE_CALLS) != ppo_update_calls:
        raise RuntimeError("ROUND58 live PPO counter differs from direct journal plus carryover")'''
    ledger_source = ledger_source.replace(old_counter_lines, new_counter_lines, 1)
    new_journal_lines = '''        "checkpoint_write_journal_entries": _r58_count_lines(
            Path(RUN_DIR) / "runtime" / "checkpoint_writes.jsonl")
            + _r58_count_lines(Path(_R58_CHECKPOINT_JOURNAL)),'''
    ledger_source = ledger_source.replace(old_journal_lines, new_journal_lines, 1)
    # Compile the exact frozen ledger function in its original module namespace.
    namespace = run_module.__dict__
    namespace["_r58_read_direct_journal"] = _read_direct_journal
    namespace["_r58_count_lines"] = _count_lines
    exec(compile(ledger_source, str(source_path), "exec"), namespace)
    ledger_patched = namespace["ledger_snapshot"]
    return {"kind": "round58-r42-carryover-configure-adapter/2",
            "source_sha256": R42_RUN_SHA256,
            "ledger_identity": ledger_patched,
            "optimizer_journal": str(optimizer_journal),
            "ppo_journal": str(ppo_journal),
            "checkpoint_journal": str(checkpoint_journal),
            "history_ppo_carryover": proof["historical_ppo_carryover"],
            "raw_journal_rows_synthesized": 0}


def _read_direct_journal(path: Path, *, start: int) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    indexes = [int(row.get("call_index", -1)) for row in rows]
    if indexes != list(range(start, start + len(rows))):
        raise RuntimeError(f"direct resume journal index gap/collision: {path}")
    if any(row.get("call_status") != "step_invoked" for row in rows):
        raise RuntimeError(f"direct resume journal contains a non-invocation: {path}")
    return rows


def _count_lines(path: Path) -> int:
    return (sum(1 for line in Path(path).open("r", encoding="utf-8") if line.strip())
            if Path(path).is_file() else 0)


def install_persistent_call_journals(runtime: dict[str, Any], *, resume_runtime_dir: Path,
                                     historical_optimizer_rows: int = 24,
                                     historical_ppo_calls: int = 3) -> dict[str, Any]:
    """Journal future calls at their actual outer call sites; historical values stay derived."""
    import torch
    trainer, run = runtime["trainer"], runtime["r42_run"]
    root = Path(resume_runtime_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    opt_path, ppo_path = root / "optimizer_step_calls.jsonl", root / "ppo_update_calls.jsonl"
    checkpoint_path = root / "checkpoint_writes.jsonl"
    # Existing files may only be a previously submitted exact continuation. Configure/ledger
    # validate their monotone indexes; no historical rows are copied into these new journals.
    opt_rows = _read_direct_journal(opt_path, start=historical_optimizer_rows + 1)
    ppo_rows = _read_direct_journal(ppo_path, start=historical_ppo_calls + 1)
    checkpoint_baseline = runtime.get("r58_resume_source_proof", {}).get(
        "historical_checkpoint_journal")
    if (not isinstance(checkpoint_baseline, dict)
            or checkpoint_baseline.get("sha256") != HISTORICAL_CHECKPOINT_JOURNAL_SHA256
            or int(checkpoint_baseline.get("rows", -1)) != 12):
        raise RuntimeError("resume checkpoint counter lacks the verified historical journal baseline")
    version_counter = runtime["r42_versions"]
    current_checkpoint_count = int(getattr(version_counter, "CHECKPOINT_WRITE_CALLS", -1))
    if current_checkpoint_count == 0:
        version_counter.CHECKPOINT_WRITE_CALLS = 12
    elif current_checkpoint_count != 12:
        raise RuntimeError("configured checkpoint counter conflicts with the verified 12-event history")
    if _count_lines(checkpoint_path) != int(version_counter.CHECKPOINT_WRITE_CALLS) - 12:
        raise RuntimeError("resume checkpoint journal and live checkpoint counter do not close")
    if int(getattr(run, "_OPTIMIZER_STEP_CALLS", -1)) not in (0, historical_optimizer_rows + len(opt_rows)):
        raise RuntimeError("R42 optimizer counter was not at the approved historical baseline")
    if int(getattr(run, "_PPO_UPDATE_CALLS", -1)) not in (0, historical_ppo_calls + len(ppo_rows)):
        raise RuntimeError("R42 PPO counter was not at the approved derived baseline")
    run._OPTIMIZER_STEP_CALLS = historical_optimizer_rows + len(opt_rows)
    run._PPO_UPDATE_CALLS = historical_ppo_calls + len(ppo_rows)
    lock = threading.Lock()
    original_step = torch.optim.Adam.step
    original_ppo = trainer.ppo_update

    def append_call(path: Path, index: int, *, kind: str, optimizer: Any = None):
        row = {"call_index": int(index), "call_status": "step_invoked",
               "counter_hook_version": "round58-resume-call-journal/1",
               "call_kind": kind, "unix_time": time.time()}
        if optimizer is not None:
            row["optimizer_class"] = type(optimizer).__name__
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            import os
            os.fsync(handle.fileno())

    def counted_step(optimizer, *args, **kwargs):
        with lock:
            index = int(run._OPTIMIZER_STEP_CALLS) + 1
            append_call(opt_path, index, kind="adam_step", optimizer=optimizer)
            return original_step(optimizer, *args, **kwargs)

    def counted_ppo(*args, **kwargs):
        with lock:
            index = int(run._PPO_UPDATE_CALLS) + 1
            append_call(ppo_path, index, kind="ppo_update")
            run._PPO_UPDATE_CALLS = index
        return original_ppo(*args, **kwargs)

    torch.optim.Adam.step = counted_step
    trainer.ppo_update = counted_ppo

    def undo():
        current_step = torch.optim.Adam.step
        current_ppo = trainer.ppo_update
        if current_step is counted_step and current_ppo is counted_ppo:
            torch.optim.Adam.step = original_step
            trainer.ppo_update = original_ppo
            return
        # The frozen configure chain may already have unwound exactly these two
        # wrappers before the private route closes. Accept only the exact
        # immediate underlying objects captured at installation; any third
        # function still signals hook replacement/corruption.
        if current_step is original_step and current_ppo is original_ppo:
            return
        raise RuntimeError(
            "live training call hooks changed before teardown: "
            f"adam_expected_wrapper={current_step is counted_step}, "
            f"adam_already_unwrapped={current_step is original_step}, "
            f"ppo_expected_wrapper={current_ppo is counted_ppo}, "
            f"ppo_already_unwrapped={current_ppo is original_ppo}"
        )

    return {"kind": "round58-resume-call-journals/1", "status": "installed",
            "optimizer_baseline": historical_optimizer_rows,
            "ppo_baseline_derived": historical_ppo_calls,
            "optimizer_rows_existing": len(opt_rows), "ppo_rows_existing": len(ppo_rows),
            "optimizer_path": str(opt_path), "ppo_path": str(ppo_path),
            "checkpoint_path": str(checkpoint_path), "undo": undo,
            "wrapper_step": counted_step, "wrapper_ppo": counted_ppo,
            "wrapped_step": original_step, "wrapped_ppo": original_ppo}


def install_attested_gate(runtime: dict[str, Any], *, frozen: dict[str, Any],
                          identity: dict[str, Any], training_version: str,
                          expected_analysis_sha256: str) -> dict[str, Any]:
    """Wire the actual R45 _run_first_batch_gate -> R42 first_batch_gate call chain."""
    analysis = runtime.get("r42_analysis")
    module = runtime.get("module")
    if analysis is None or module is None:
        raise RuntimeError("actual configured gate modules are missing")
    gate = getattr(analysis, "first_batch_gate", None)
    source = Path(getattr(analysis, "__file__", "")).resolve()
    if not callable(gate) or sha256(source) != expected_analysis_sha256:
        raise PermissionError("actual first-batch gate source identity mismatch")
    outer = getattr(module, "_run_first_batch_gate", None)
    if not callable(outer):
        raise RuntimeError("actual private execution module has no _run_first_batch_gate")
    if getattr(outer, "__module__", None) != "round46_r45_runtime_adapter":
        raise RuntimeError("actual outer first-batch gate is not the configured R45 adapter")
    old_gate = gate

    def attested_gate(seed: int, *, runtime: dict[str, Any], frozen: dict[str, Any]):
        base = runtime.get("round42_freeze")
        if not isinstance(base, dict):
            raise RuntimeError("actual gate runtime lacks the registered science freeze")
        version = frozen.get("round42", {}).get("training_version")
        if version != training_version:
            raise PermissionError("actual gate expected training version differs from grant")
        expected = copy.deepcopy(base)
        expected["training_version"] = training_version
        expected["round58_repair_attestation"] = copy.deepcopy(identity)
        local_runtime = dict(runtime)
        local_runtime["round42_freeze"] = expected
        return old_gate(seed, runtime=local_runtime, frozen=frozen)

    attested_gate.__name__ = "r58_attested_actual_first_batch_gate"
    attested_gate.__wrapped__ = old_gate
    analysis.first_batch_gate = attested_gate
    def undo() -> None:
        if analysis.first_batch_gate is not attested_gate:
            raise RuntimeError("actual first-batch gate binding changed before undo")
        analysis.first_batch_gate = old_gate
    return {"kind": "round58-attested-actual-gate-binding/1",
            "outer_gate": outer, "actual_analysis_gate": old_gate,
            "wrapped_analysis_gate": attested_gate, "analysis_path": str(source),
            "analysis_sha256": expected_analysis_sha256, "undo": undo}


def validate_historical_payload(payload: Any, source: dict[str, Any], *,
                                base_freeze: dict[str, Any]) -> None:
    from r58_identity_ledger_close_v2 import identity_differences
    expected = {"training_version": TRAINING_VERSION,
                "action_semantics": "one-shot-send-skip/1",
                "base_freeze": base_freeze,
                "repair_identity": source["original_identity"]}
    differences = identity_differences(payload, training_version=expected["training_version"],
        action_semantics=expected["action_semantics"], base_freeze=expected["base_freeze"],
        repair_identity=expected["repair_identity"])
    if differences:
        raise PermissionError("historical checkpoint identity mismatch: " +
                              ", ".join(row["field"] for row in differences))
    state = payload.get("training_state", {})
    cursor = source.get("cursor") or {"episodes_done": 4, "updates": 1,
                                        "optimizer_steps": 8, "lambda": 1.0}
    for key in ("episodes_done", "updates", "optimizer_steps"):
        if int(state.get(key, -1)) != int(cursor[key]):
            raise PermissionError(f"historical checkpoint cursor mismatch: {key}")
    if float(state.get("lambda", float("nan"))) != float(cursor["lambda"]):
        raise PermissionError("historical checkpoint cursor mismatch: lambda")
    if not isinstance(state.get("lambda_history"), list) or not isinstance(payload.get("rng"), dict):
        raise PermissionError("historical checkpoint lacks lambda history or RNG state")
