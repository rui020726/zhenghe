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

MAP_REL = Path("runs/comm_bc_initialized_constrained_ppo_20261007_r2/round58_isolation_stop_audit/ROUND58_GEN3_APPROVED_RESUME_SOURCES_V33.json")
MAP_SHA256 = "b5bd3a2a4065f6b10ae5b0291fd42e4623fdaa76cbcccb4eaae93e5bc3ba43a2"
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
    "train_pilot_source", "stage_b_sp_common_source",
    "resume_runtime_aliases",
    "attempt3_contract_sha256", "attempt3_claim_seam_sha256",
    "prefix_recovery_sha256", "committed_prefix_bridge_sha256",
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
    discarded_v26 = (root / "runs/comm_bc_initialized_constrained_ppo_20261007_r2"
                     / "round58_gen3_resume_v26/raw/runtime/checkpoint_writes.jsonl")
    discarded_v26_sha = "d8bd1fe2ea804ac6fb654bcd89791800604025c809bbb2ace0b58857a913cef6"
    if sha256(discarded_v26) != discarded_v26_sha or _count_lines(discarded_v26) != 2:
        raise PermissionError("failed V26 identity-migration checkpoint events changed")
    b20 = validate_committed_seed11_batch20(
        root, validate_committed_seed11_batch14(root, validate_committed_seed11_batch9(root)))
    continuation, seed22_continuation = validate_committed_v29_endpoints(root, b20, verified[22])
    return {"manifest": manifest, "manifest_sha256": expected_map_sha256,
            "sources": verified, "historical_ppo_carryover": len(verified),
            "historical_optimizer_steps": sum(8 for _ in verified),
            "historical_checkpoint_journal": checkpoint_journal,
            "discarded_v26_identity_writes": {"path": str(discarded_v26.resolve()),
                                              "sha256": discarded_v26_sha, "rows": 2,
                                              "scientific_chain": False},
            "committed_continuation": continuation,
            "additional_continuations": {22: seed22_continuation},
            "target_checkpoint_root": (root / manifest["target_checkpoint_root"]).resolve()}


def _historical_crlf_source(repo_root: Path, commit: str, name: str,
                            expected_sha: str) -> dict[str, str]:
    relative = ("runs/comm_bc_initialized_constrained_ppo_20261007_r2/"
                "round58_isolation_stop_audit/" + name)
    raw = subprocess.run(["git", "show", f"{commit}:{relative}"], cwd=repo_root,
                         capture_output=True, check=True).stdout
    if b"\r\n" in raw or not raw.endswith(b"\n"):
        raise PermissionError(f"historical Git blob has unexpected line endings: {name}")
    transformed = raw.replace(b"\n", b"\r\n")
    digest = hashlib.sha256(transformed).hexdigest()
    if digest != expected_sha:
        raise PermissionError(f"historical CRLF source does not match attested bytes: {name}")
    return {"git_commit": commit, "git_blob_sha256": hashlib.sha256(raw).hexdigest(),
            "crlf_sha256": digest, "conversion": "LF-to-CRLF only"}


def validate_committed_seed11_batch9(repo_root: Path) -> dict[str, Any]:
    """Hash-bind the b9 source, its original private code and prior direct calls."""
    root = Path(repo_root).resolve()
    r2 = root / "runs" / "comm_bc_initialized_constrained_ppo_20261007_r2"
    old = r2 / "round58_gen3_resume_v18" / "raw"
    cp = old / "checkpoints"
    audit = r2 / "round58_isolation_stop_audit"
    original = (audit / "round58_resume_v18_superseded_identity_quarantine_20261010T0030Z"
                / "checkpoints" / "v2_train_s11.pt")
    original_sha = "92ad17ca9dfd08e4e1d5a606e76e202aa043ac0d2e8a550cc27e303f7d6533e2"
    commit_path = cp / "r40_batch_commit_s11_b9.json"
    commit = read_json(commit_path)
    if (sha256(original) != original_sha
            or sha256(commit_path) != "ee6887d037db6e87241bb39e6478150abced8e0e53d7421b63b79647afb7d835"
            or commit.get("kind") != "round40-batch-checkpoint-commit/1"
            or (commit.get("training_seed"), commit.get("batch"), commit.get("episodes_done"),
                commit.get("updates"), commit.get("optimizer_steps")) != (11, 9, 40, 10, 60)
            or commit.get("rolling_checkpoint_sha256") != original_sha
            or commit.get("postupdate_checkpoint_sha256") != original_sha):
        raise PermissionError("seed11 batch9 original commit/payload do not close")
    if sha256(cp / commit["batch_record"]) != commit["batch_record_sha256"]:
        raise PermissionError("seed11 batch9 record does not close")
    migration = read_json(audit / "ROUND58_GEN3_CHECKPOINT_ATTESTATION_MIGRATION.json")
    links = [item for item in migration["migrated"]
             if Path(item["file"]).name == "v2_train_s11.pt"]
    current = original_sha
    if len(links) != 3:
        raise PermissionError("seed11 rolling migration chain must have three links")
    for link in links:
        if (Path(link["file"]).resolve() != (cp / "v2_train_s11.pt").resolve()
                or link["sha_before"] != current):
            raise PermissionError("seed11 rolling migration chain gap")
        current = link["sha_after"]
    migrated = cp / "v2_train_s11.pt"
    if (sha256(migrated) != current or
            read_json(Path(str(migrated) + ".r58.json"))["checkpoint_sha256"] != current):
        raise PermissionError("seed11 migrated rolling sidecar/bytes do not close")
    journals = {
        "optimizer_step_calls.jsonl": ("48df4533df9340dae6b3f5a388bad671811061d3c38eada51012b9f336bfa20b", 52),
        "ppo_update_calls.jsonl": ("30d3e96b3864de6881452d8c28e0e8062ff365d039100978af7395483a21daeb", 9),
        "checkpoint_writes.jsonl": ("e0b2aa5903baa1f5251f6fe9f4dd0b7a412d97b08cc5cc19903983930c5c9916", 29),
    }
    for name, (digest, count) in journals.items():
        path = old / "runtime" / name
        if (sha256(path) != digest or
                sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())
                != count):
            raise PermissionError(f"seed11 prior direct journal changed: {name}")
    for batch in range(1, 10):
        receipt = read_json(cp / f"r40_batch_commit_s11_b{batch}.json")
        if (receipt.get("batch") != batch or receipt.get("training_seed") != 11
                or receipt.get("episodes_done") != 4 * (batch + 1)
                or receipt.get("updates") != batch + 1
                or sha256(cp / receipt["batch_record"]) != receipt["batch_record_sha256"]):
            raise PermissionError(f"seed11 prior batch{batch} record/commit differs")
    launcher_source = _historical_crlf_source(
        root, "fbe7c8d264e9d0aa64ea268b8acac56e42a68a71",
        "r58_private_launcher_repaired.py",
        "942618e95eae5b501f0d79ef996b983b70d87aaa2462ef32d4ff16a9cd7066a9")
    seam_source = _historical_crlf_source(
        root, "43a85a750d04fccaf7943cb404b34bd10f882671", "r58_resume_seam.py",
        "aa073e218c673fcc2087aa80afdd3256060daae83929fda88ac58ab1ffa4225c")
    import torch
    payload = torch.load(original, map_location="cpu", weights_only=False)
    attestation = payload.get("round58_repair_attestation")
    if (not isinstance(attestation, dict)
            or payload.get("training_state", {}).get("round58_repair_attestation") != attestation
            or payload.get("freeze", {}).get("round58_repair_attestation") != attestation
            or attestation.get("private_launcher_sha256") != launcher_source["crlf_sha256"]
            or attestation.get("resume_seam_sha256") != seam_source["crlf_sha256"]
            or attestation.get("adapter_sha256") != "7081557e6197aceac17fb453c03e1b5fd539f55ec60a7b3eae92d63f7e697688"
            or attestation.get("trainer_sha256") != "0550b99eb1f10a45d57757caedb4ce365fe858d72e171537e38dc450fb49a841"
            or attestation.get("common_sha256") != "66d019a344d9cfef3c67bd8b722f8cc27144b9f71ddc8e5496451929d12f746a"):
        raise PermissionError("seed11 original batch9 payload has an unbound source identity")
    return {"path": original.resolve(), "checkpoint_sha256": original_sha,
            "commit_path": commit_path.resolve(), "commit_sha256": sha256(commit_path),
            "batch_record_sha256": commit["batch_record_sha256"],
            "migrated_checkpoint_sha256": current, "prior_resume_raw": old.resolve(),
            "historical_ppo_calls": 12, "historical_optimizer_calls": 76,
            "historical_checkpoint_writes": 41, "original_identity": attestation,
            "source_identity": {
                "private_launcher": launcher_source, "resume_seam": seam_source},
            "prior_direct_journals": {name: {"sha256": digest, "rows": count}
                                      for name, (digest, count) in journals.items()},
            "cursor": {"episodes_done": 40, "updates": 10, "optimizer_steps": 60}}


def validate_committed_seed11_batch14(repo_root: Path, parent: dict[str, Any]) -> dict[str, Any]:
    """Bind the next five real V25 commits and their original direct journals."""
    import torch
    root = Path(repo_root).resolve()
    run_root = root / "runs/comm_bc_initialized_constrained_ppo_20261007_r2"
    audit = run_root / "round58_isolation_stop_audit"
    raw = run_root / "round58_gen3_resume_v22/raw"
    cp = raw / "checkpoints"
    source = cp / "v2_train_s11.pt"
    source_sha = "4c4a8c7115a0bcc08cf58d5912d97bd0605d4bf38a3654b0768c57482e82694f"
    commit_path = cp / "r40_batch_commit_s11_b14.json"
    commit_sha = "c18ae6bdd702e8ce139c44a6bf44ad50b7ea8100a4dcc452452f4398f0e0b7f5"
    frozen_path = audit / "ROUND58_RESUME_FREEZE_V25.json"
    if (sha256(source) != source_sha or sha256(commit_path) != commit_sha
            or sha256(frozen_path) != "443018e791d6c6c0f3a36caa349637dcae6dda6ba32af9286fb6f6fcd12fad14"):
        raise PermissionError("V25 batch14 source, commit or freeze bytes changed")
    freeze = read_json(frozen_path)
    attestation = freeze["repair_identity"]
    sidecar_path = Path(str(source) + ".r58.json")
    sidecar = read_json(sidecar_path)
    commit = read_json(commit_path)
    if (sidecar.get("checkpoint_sha256") != source_sha
            or sidecar.get("round58_repair_attestation") != attestation
            or commit.get("rolling_checkpoint_sha256") != source_sha
            or (commit.get("training_seed"), commit.get("batch"),
                commit.get("episodes_done"), commit.get("updates"),
                commit.get("optimizer_steps")) != (11, 14, 60, 15, 88)
            or sha256(cp / commit["batch_record"]) != commit["batch_record_sha256"]):
        raise PermissionError("V25 batch14 sidecar/record/commit do not close")
    for batch in range(10, 15):
        receipt = read_json(cp / f"r40_batch_commit_s11_b{batch}.json")
        if (int(receipt.get("training_seed", -1)) != 11
                or int(receipt.get("batch", -1)) != batch
                or int(receipt.get("episodes_done", -1)) != 4 * (batch + 1)
                or sha256(cp / receipt["batch_record"]) != receipt["batch_record_sha256"]):
            raise PermissionError(f"V25 batch{batch} durable input differs")
    journal_sources = {
        "optimizer_step_calls.jsonl": ("c61faf8166e424a61fb2aaed54cc5c89a80d1d15baf4d90ed7a81403d8aa9513", 28),
        "ppo_update_calls.jsonl": ("a3bdbf49caeb1b0e70496180251238e7143841b7c3bbc40f182c910792ecfd58", 5),
        "checkpoint_writes.jsonl": ("8b9e1a06de6544d69a661d01e540e7e32690ac6475a79fc726d352ae3011fcf5", 17),
    }
    for name, (digest, rows) in journal_sources.items():
        path = raw / "runtime" / name
        if sha256(path) != digest or _count_lines(path) != rows:
            raise PermissionError(f"V25 direct journal differs: {name}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    historical = torch.load(parent["path"], map_location="cpu", weights_only=False)
    state = payload.get("training_state", {})
    science_freeze = lambda p: {k: v for k, v in p["freeze"].items()
                                if k != "round58_repair_attestation"}
    if (payload.get("round58_repair_attestation") != attestation
            or state.get("round58_repair_attestation") != attestation
            or payload["freeze"].get("round58_repair_attestation") != attestation
            or science_freeze(payload) != science_freeze(historical)
            or state.get("round58_resume_provenance", {}).get("source_checkpoint_sha256")
               != parent["checkpoint_sha256"]
            or (state.get("episodes_done"), state.get("updates"), state.get("optimizer_steps"))
               != (60, 15, 88)):
        raise PermissionError("V25 payload does not bind approved b9 ancestry/science/cursor")
    return {"path": source.resolve(), "checkpoint_sha256": source_sha,
            "sidecar_path": sidecar_path.resolve(), "sidecar_sha256": sha256(sidecar_path),
            "commit_path": commit_path.resolve(), "commit_sha256": commit_sha,
            "batch_record_sha256": commit["batch_record_sha256"],
            "original_identity": attestation,
            "prior_resume_raw": parent["prior_resume_raw"],
            "prior_batch_roots": [parent["prior_resume_raw"], raw.resolve()],
            "parent_b9": {"checkpoint_sha256": parent["checkpoint_sha256"],
                          "commit_sha256": parent["commit_sha256"]},
            "prior_direct_journals": parent["prior_direct_journals"],
            "v25_direct_journals": {name: {"sha256": digest, "rows": rows}
                                    for name, (digest, rows) in journal_sources.items()},
            "historical_ppo_calls": 17, "historical_optimizer_calls": 104,
            "historical_checkpoint_writes": 58,
            "cursor": {"episodes_done": 60, "updates": 15, "optimizer_steps": 88}}


def validate_committed_seed11_batch20(repo_root: Path, parent: dict[str, Any]) -> dict[str, Any]:
    """Bind the six V27 committed batches, excluding its later failed restart."""
    import torch
    root = Path(repo_root).resolve()
    run_root = root / "runs/comm_bc_initialized_constrained_ppo_20261007_r2"
    audit = run_root / "round58_isolation_stop_audit"
    raw = run_root / "round58_gen3_resume_v27/raw"
    cp = raw / "checkpoints"
    source = cp / "v2_train_s11_after_b20.pt"
    source_sha = "bb14d99d82409ce7fe35c912e5208dc5a1f23e3f893d3ee5f87ec89fd6722489"
    commit_path = cp / "r40_batch_commit_s11_b20.json"
    commit_sha = "3f2810a67e789bf188f43d9d9e725b013b3e4011d5156cc0f7a33ba26fce98ce"
    frozen_path = audit / "ROUND58_RESUME_FREEZE_V27.json"
    if (sha256(source) != source_sha or sha256(commit_path) != commit_sha
            or sha256(frozen_path) != "1c55a8a03191060ca5e21afaa1d63cfab6d4672a0d53266349a6c6224c3d7a98"):
        raise PermissionError("V27 b20 source, commit or frozen identity changed")
    attestation = read_json(frozen_path)["repair_identity"]
    sidecar_path = Path(str(source) + ".r58.json")
    sidecar = read_json(sidecar_path)
    commit = read_json(commit_path)
    if (sidecar.get("checkpoint_sha256") != source_sha
            or sidecar.get("round58_repair_attestation") != attestation
            or commit.get("postupdate_checkpoint_sha256") != source_sha
            or (commit.get("training_seed"), commit.get("batch"),
                commit.get("episodes_done"), commit.get("updates"),
                commit.get("optimizer_steps")) != (11, 20, 84, 21, 116)
            or sha256(cp / commit["batch_record"]) != commit["batch_record_sha256"]):
        raise PermissionError("V27 b20 sidecar/record/commit do not close")
    for batch in range(15, 21):
        receipt = read_json(cp / f"r40_batch_commit_s11_b{batch}.json")
        if (int(receipt.get("training_seed", -1)) != 11
                or int(receipt.get("batch", -1)) != batch
                or int(receipt.get("episodes_done", -1)) != 4 * (batch + 1)
                or sha256(cp / receipt["batch_record"]) != receipt["batch_record_sha256"]
                or sha256(cp / receipt["postupdate_checkpoint"])
                   != receipt["postupdate_checkpoint_sha256"]):
            raise PermissionError(f"V27 b{batch} durable input differs")
    journal_sources = {
        "optimizer_step_calls.jsonl": ("f0322ec4f9c9cdef0a6eb2ec26b3f8c4000a30a30eed9a4ecfe7a86cf816e43e", 28),
        "ppo_update_calls.jsonl": ("dac80680768be7b595602bccc13099c60f0c9a25633766a7516128132bb7d7f3", 6),
        "checkpoint_writes.jsonl": ("59c32ee4353b82c09a9d01e7913bc712692a77f54b73040747034d069cfb57b5", 21),
    }
    for name, (digest, rows) in journal_sources.items():
        path = raw / "runtime" / name
        if sha256(path) != digest or _count_lines(path) != rows:
            raise PermissionError(f"V27 direct journal differs: {name}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    prior = torch.load(parent["path"], map_location="cpu", weights_only=False)
    state = payload.get("training_state", {})
    science = lambda p: {k: v for k, v in p["freeze"].items()
                         if k != "round58_repair_attestation"}
    if (payload.get("round58_repair_attestation") != attestation
            or state.get("round58_repair_attestation") != attestation
            or payload["freeze"].get("round58_repair_attestation") != attestation
            or science(payload) != science(prior)
            or state.get("round58_resume_provenance", {}).get("source_checkpoint_sha256")
               != parent["checkpoint_sha256"]
            or (state.get("episodes_done"), state.get("updates"), state.get("optimizer_steps"))
               != (84, 21, 116)):
        raise PermissionError("V27 b20 payload ancestry/science/cursor differs")
    return {**parent, "path": source.resolve(), "checkpoint_sha256": source_sha,
            "sidecar_path": sidecar_path.resolve(), "sidecar_sha256": sha256(sidecar_path),
            "commit_path": commit_path.resolve(), "commit_sha256": commit_sha,
            "batch_record_sha256": commit["batch_record_sha256"],
            "original_identity": attestation,
            "prior_batch_roots": [*parent["prior_batch_roots"], raw.resolve()],
            "parent_b14": {"checkpoint_sha256": parent["checkpoint_sha256"],
                           "commit_sha256": parent["commit_sha256"]},
            "v27_direct_journals": {name: {"sha256": digest, "rows": rows}
                                    for name, (digest, rows) in journal_sources.items()},
            "historical_ppo_calls": parent["historical_ppo_calls"] + 6,
            "historical_optimizer_calls": parent["historical_optimizer_calls"] + 28,
            "historical_checkpoint_writes": parent["historical_checkpoint_writes"] + 21,
            "cursor": {"episodes_done": 84, "updates": 21, "optimizer_steps": 116}}


def validate_committed_v29_endpoints(repo_root: Path, parent: dict[str, Any],
                                    seed22_original: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash-close V29 b21–31, s22 b1–4 and all direct calls; exclude failed restart."""
    import torch
    root = Path(repo_root).resolve()
    raw = root / "runs/comm_bc_initialized_constrained_ppo_20261007_r2/round58_gen3_resume_v29/raw"
    cp = raw / "checkpoints"
    freeze_path = raw.parents[1] / "round58_isolation_stop_audit/ROUND58_RESUME_FREEZE_V29.json"
    if sha256(freeze_path) != "27bcf7f96f6c47d9a91fb59106442a4d23ddd4622d62cf541046c15ef9128b91":
        raise PermissionError("V29 execution freeze changed")
    identity = read_json(freeze_path)["repair_identity"]
    endpoints = {
        11: (31, 128, 32, 184,
             "e0f5c6466731b0feac2dcddd21425e97f82dd6431c5613afd523268745efe3d9",
             "8a85ab8441c3c3f1fed83d2547e22b710ca988097e786f8fc5f64d8f7b312cd6",
             "8a3d8092ca0fdf05771aa9b16a683452f0eea369b792c27f9e4b236752cf7617",
             "83ae07b92212905069852e4b7a6a814e4e889e89d13d40f4a1371009d1d0f2de"),
        22: (4, 20, 5, 32,
             "8f87d932df4cd2c6fe4ec60ae11bc3b3835112fbe24eaf76e7c96e6dcc9884f9",
             "4948fbbee3dd8c925d2aca8bf29a23c0bc6ab6226356d20f04d344a3c3ad7782",
             "1570a037ae3bdbca6db8a4b4364e24590bc7483655ae9f03fcb6e06dca4a5db0",
             "cd256e6dba634b9f6908023e4dce83b6a31b3f8532ae0e518179f8005b503625"),
    }
    results = {}
    for seed, (batch, episodes, updates, steps, checkpoint_sha, commit_sha,
               sidecar_sha, record_sha) in endpoints.items():
        source = cp / f"v2_train_s{seed}_after_b{batch}.pt"
        sidecar = Path(str(source) + ".r58.json")
        commit_path = cp / f"r40_batch_commit_s{seed}_b{batch}.json"
        if (sha256(source) != checkpoint_sha or sha256(sidecar) != sidecar_sha
                or sha256(commit_path) != commit_sha):
            raise PermissionError(f"V29 s{seed} committed endpoint bytes changed")
        sc, commit = read_json(sidecar), read_json(commit_path)
        if (sc.get("round58_repair_attestation") != identity
                or sc.get("checkpoint_sha256") != checkpoint_sha
                or commit.get("postupdate_checkpoint_sha256") != checkpoint_sha
                or commit.get("batch_record_sha256") != record_sha
                or (commit.get("training_seed"), commit.get("batch"),
                    commit.get("episodes_done"), commit.get("updates"),
                    commit.get("optimizer_steps")) != (seed, batch, episodes, updates, steps)
                or sha256(cp / commit["batch_record"]) != record_sha):
            raise PermissionError(f"V29 s{seed} endpoint sidecar/record/commit mismatch")
        for b in range(21 if seed == 11 else 1, batch + 1):
            receipt = read_json(cp / f"r40_batch_commit_s{seed}_b{b}.json")
            if (int(receipt.get("training_seed", -1)) != seed
                    or int(receipt.get("batch", -1)) != b
                    or int(receipt.get("episodes_done", -1)) != 4 * (b + 1)
                    or sha256(cp / receipt["batch_record"]) != receipt["batch_record_sha256"]
                    or sha256(cp / receipt["postupdate_checkpoint"])
                       != receipt["postupdate_checkpoint_sha256"]):
                raise PermissionError(f"V29 s{seed} b{b} durable chain differs")
        payload = torch.load(source, map_location="cpu", weights_only=False)
        prior = torch.load(parent["path"] if seed == 11 else seed22_original["path"],
                           map_location="cpu", weights_only=False)
        state = payload.get("training_state", {})
        science = lambda p: {k: v for k, v in p["freeze"].items()
                             if k != "round58_repair_attestation"}
        prior_sha = parent["checkpoint_sha256"] if seed == 11 else seed22_original["checkpoint_sha256"]
        if (payload.get("round58_repair_attestation") != identity
                or state.get("round58_repair_attestation") != identity
                or science(payload) != science(prior)
                or state.get("round58_resume_provenance", {}).get("source_checkpoint_sha256") != prior_sha
                or (state.get("episodes_done"), state.get("updates"), state.get("optimizer_steps"))
                   != (episodes, updates, steps)):
            raise PermissionError(f"V29 s{seed} payload ancestry/science/cursor differs")
        results[seed] = {"path": source.resolve(), "checkpoint_sha256": checkpoint_sha,
                         "sidecar_path": sidecar.resolve(), "sidecar_sha256": sidecar_sha,
                         "commit_path": commit_path.resolve(), "commit_sha256": commit_sha,
                         "batch_record_sha256": record_sha, "original_identity": identity,
                         "cursor": {"episodes_done": episodes, "updates": updates,
                                    "optimizer_steps": steps}}
    journals = {
        "optimizer_step_calls.jsonl": ("782315de7ff8c7099292686fc62083ab99fd5c24d3dd03ce64b8696fc6181140", 92),
        "ppo_update_calls.jsonl": ("30f7175bb8221c6d03be9a783c6fce4c5ac46c0b057d1bd1c7acdea5d6626141", 15),
        "checkpoint_writes.jsonl": ("11a8d70bd423bcf6614e9290a2c9063deb37fbcbb7ddd059a484b3b31b2dd177", 49),
    }
    for name, (digest, rows) in journals.items():
        path = raw / "runtime" / name
        if sha256(path) != digest or _count_lines(path) != rows:
            raise PermissionError(f"V29 direct call journal changed: {name}")
    latest11 = {**parent, **results[11],
                "prior_batch_roots": [*parent["prior_batch_roots"], raw.resolve()],
                "v29_direct_journals": {name: {"sha256": digest, "rows": rows}
                                        for name, (digest, rows) in journals.items()},
                "historical_ppo_calls": parent["historical_ppo_calls"] + 15,
                "historical_optimizer_calls": parent["historical_optimizer_calls"] + 92,
                "historical_checkpoint_writes": parent["historical_checkpoint_writes"] + 49}
    return latest11, results[22]


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


def _batch_records(run_root: Path, extra_root: Path,
                   prior_roots: list[Path] | None = None) -> list[Path]:
    found: dict[str, Path] = {}
    directories = [Path(run_root) / "checkpoints", Path(extra_root)]
    directories[1:1] = [Path(path) for path in (prior_roots or [])]
    for directory in directories:
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
    continuation = proof.get("committed_continuation")
    if continuation is not None:
        prior = Path(continuation["prior_resume_raw"]) / "checkpoints"
        for batch in range(1, 10):
            key = (11, batch)
            expected = prior / f"r40_batch_s11_b{batch}_attempt01.json"
            if (key not in by_key or by_key[key].resolve() != expected.resolve()
                    or sha256(by_key[key]) != sha256(expected)):
                raise RuntimeError(f"committed continuation batch missing/ambiguous: {key}")
        v25 = Path(continuation["prior_batch_roots"][1]) / "checkpoints"
        for batch in range(10, 15):
            key = (11, batch)
            expected = v25 / f"r40_batch_s11_b{batch}_attempt01.json"
            if (key not in by_key or by_key[key].resolve() != expected.resolve()
                    or sha256(by_key[key]) != sha256(expected)):
                raise RuntimeError(f"V25 committed continuation batch missing/ambiguous: {key}")
        v27 = Path(continuation["prior_batch_roots"][2]) / "checkpoints"
        for batch in range(15, 21):
            key = (11, batch)
            expected = v27 / f"r40_batch_s11_b{batch}_attempt01.json"
            if (key not in by_key or by_key[key].resolve() != expected.resolve()
                    or sha256(by_key[key]) != sha256(expected)):
                raise RuntimeError(f"V27 committed continuation batch missing/ambiguous: {key}")
        v29 = Path(continuation["prior_batch_roots"][3]) / "checkpoints"
        for seed, start, end in ((11, 21, 32), (22, 1, 5)):
            for batch in range(start, end):
                key = (seed, batch)
                expected = v29 / f"r40_batch_s{seed}_b{batch}_attempt01.json"
                if (key not in by_key or by_key[key].resolve() != expected.resolve()
                        or sha256(by_key[key]) != sha256(expected)):
                    raise RuntimeError(f"V29 committed continuation batch missing/ambiguous: {key}")
    expected_total = len(approved) + (35 if continuation is not None else 0)
    if len(by_key) < expected_total:
        raise RuntimeError("carryover batch count is below the verified committed history")
    return expected_total


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
    prior_batch_roots = ([Path(raw) / "checkpoints" for raw in
                          proof["committed_continuation"]["prior_batch_roots"]]
                         if "committed_continuation" in proof else [])
    run_module._r58_batch_records = lambda base: _batch_records(
        Path(run_module.RUN_DIR), Path(run_module._R58_EXTRA_BATCH_ROOT), prior_batch_roots)
    continuation = proof.get("committed_continuation")
    history_ppo = (int(continuation["historical_ppo_calls"])
                   if continuation else int(proof["historical_ppo_carryover"]))
    history_optimizer = (int(continuation["historical_optimizer_calls"])
                         if continuation else int(proof["historical_optimizer_steps"]))
    run_module._r58_history_ppo_carryover = history_ppo
    run_module._r58_history_batch_verifier = lambda rows: carryover_count_from_batches(rows, proof)
    ledger_source = ledger_source.replace(ledger_anchor,
        '    batch_records = _r58_batch_records(RUN_DIR / "checkpoints")', 1)
    new_counter_lines = f'''    _optimizer_rows = _r58_read_direct_journal(Path(_R58_OPTIMIZER_JOURNAL), start={history_optimizer + 1})
    _ppo_rows = _r58_read_direct_journal(Path(_R58_PPO_JOURNAL), start={history_ppo + 1})
    _r58_history_batch_verifier(batch_records)
    optimizer_steps_actual = {history_optimizer} + len(_optimizer_rows)
    ppo_update_calls = _r58_history_ppo_carryover + len(_ppo_rows)
    if int(_OPTIMIZER_STEP_CALLS) != optimizer_steps_actual:
        raise RuntimeError("ROUND58 live Adam hook counter differs from direct journal plus history")
    if int(_PPO_UPDATE_CALLS) != ppo_update_calls:
        raise RuntimeError("ROUND58 live PPO counter differs from direct journal plus carryover")'''
    ledger_source = ledger_source.replace(old_counter_lines, new_counter_lines, 1)
    history_checkpoint = (int(continuation["historical_checkpoint_writes"])
                          if continuation else 12)
    new_journal_lines = f'''        "checkpoint_write_journal_entries": {history_checkpoint}
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
            "history_ppo_carryover": history_ppo,
            "history_optimizer_carryover": history_optimizer,
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
    if not Path(path).is_file():
        return 0
    with Path(path).open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _journal_rows_are_resume_boundary_writes(checkpoint_path: Path) -> bool:
    """Every existing resume-journal row must be a genuine continuation write of
    this generation: a parsed round58 write record at or after the restored
    episode4/update1 cursor, whose payload hash matches a checkpoint file that
    still exists in the generation root (or that a later row supersedes)."""
    checkpoint_root = (Path(checkpoint_path).resolve().parent.parent / "checkpoints").resolve()
    try:
        rows = [json.loads(line) for line in
                Path(checkpoint_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    if not rows:
        return False
    migrated_before: set[str] = set()
    migration_manifest = Path(checkpoint_path).resolve().parents[3] / \
        "round58_isolation_stop_audit" / "ROUND58_GEN3_CHECKPOINT_ATTESTATION_MIGRATION.json"
    if migration_manifest.is_file():
        try:
            migrated_before = {
                item.get("sha_before") for item in
                json.loads(migration_manifest.read_text(encoding="utf-8")).get("migrated", [])
                if isinstance(item.get("sha_before"), str)}
        except (OSError, json.JSONDecodeError):
            migrated_before = set()
    existing_shas: dict[str, bool] = {}
    for row in rows:
        if (int(row.get("episodes_done", -1)) < 4 or int(row.get("updates", -1)) < 1
                or not isinstance(row.get("sha256_at_write"), str)):
            return False
        sha = row["sha256_at_write"]
        if sha not in existing_shas:
            existing_shas[sha] = sha in migrated_before or any(
                (checkpoint_root / name).is_file() and sha256(checkpoint_root / name) == sha
                for name in {p.name for p in checkpoint_root.glob("*.pt")} | {"v2_train_s11.pt", "v2_train_s22.pt", "v2_train_s33.pt"})
    last_sha = rows[-1].get("sha256_at_write")
    if not existing_shas.get(last_sha, False):
        return False
    return True


def install_persistent_call_journals(runtime: dict[str, Any], *, resume_runtime_dir: Path,
                                     historical_optimizer_rows: int = 104,
                                     historical_ppo_calls: int = 17) -> dict[str, Any]:
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
    continuation = runtime.get("r58_resume_source_proof", {}).get("committed_continuation")
    historical_checkpoint_rows = (int(continuation["historical_checkpoint_writes"])
                                  if continuation is not None else 12)
    discarded_v26 = runtime.get("r58_resume_source_proof", {}).get(
        "discarded_v26_identity_writes")
    if (not isinstance(checkpoint_baseline, dict)
            or checkpoint_baseline.get("sha256") != HISTORICAL_CHECKPOINT_JOURNAL_SHA256
            or int(checkpoint_baseline.get("rows", -1)) != 12
            or historical_checkpoint_rows != 58 + 21 + 49
            or not isinstance(discarded_v26, dict)
            or discarded_v26.get("sha256") != "d8bd1fe2ea804ac6fb654bcd89791800604025c809bbb2ace0b58857a913cef6"
            or int(discarded_v26.get("rows", -1)) != 2):
        raise RuntimeError("resume checkpoint counter lacks the verified historical journal baseline")
    version_counter = runtime["r42_versions"]
    current_checkpoint_count = int(getattr(version_counter, "CHECKPOINT_WRITE_CALLS", -1))
    journal_row_count = _count_lines(checkpoint_path)
    if current_checkpoint_count == 0:
        version_counter.CHECKPOINT_WRITE_CALLS = historical_checkpoint_rows
    elif current_checkpoint_count != historical_checkpoint_rows and not (
            current_checkpoint_count == journal_row_count
            and journal_row_count > 0
            and _journal_rows_are_resume_boundary_writes(checkpoint_path)):
        raise RuntimeError(
            "configured checkpoint counter conflicts with the verified 12-event history: "
            f"counter={current_checkpoint_count}, journal_rows={journal_row_count}, "
            f"journal={checkpoint_path}")
    if current_checkpoint_count not in (0, historical_checkpoint_rows) and current_checkpoint_count == journal_row_count:
        # The fresh resume journal namespace counts its own rows; rebase the live
        # counter onto the verified 12-event historical baseline so the closed
        # invariant stays "counter - 12 == new journal rows".
        version_counter.CHECKPOINT_WRITE_CALLS = historical_checkpoint_rows + journal_row_count
    if _count_lines(checkpoint_path) != int(version_counter.CHECKPOINT_WRITE_CALLS) - historical_checkpoint_rows:
        raise RuntimeError("resume checkpoint journal and live checkpoint counter do not close")
    if int(getattr(run, "_OPTIMIZER_STEP_CALLS", -1)) not in (
            0, historical_optimizer_rows + len(opt_rows), len(opt_rows)):
        # len(opt_rows) is the r42 configure count over the resume-scope journal;
        # historical_optimizer_rows + len(opt_rows) is the global closed count.
        raise RuntimeError("R42 optimizer counter was not at the approved historical baseline")
    if int(getattr(run, "_PPO_UPDATE_CALLS", -1)) not in (
            0, historical_ppo_calls + len(ppo_rows), len(ppo_rows)):
        raise RuntimeError("R42 PPO counter was not at the approved derived baseline")
    run._OPTIMIZER_STEP_CALLS = historical_optimizer_rows + len(opt_rows)
    run._PPO_UPDATE_CALLS = historical_ppo_calls + len(ppo_rows)
    lock = threading.Lock()
    original_step = torch.optim.Adam.step
    original_ppo = trainer.ppo_update
    configured_r42_runtime = getattr(run, "_RUNTIME", None)
    frozen_adam_restore = (configured_r42_runtime.get("restore_adam_step")
                           if isinstance(configured_r42_runtime, dict) else None)

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
        # The frozen ROUND43 call counter already appends exactly one durable
        # journal row per Adam step to the same journal file; appending here
        # double-counted every step.  Delegate unchanged.
        return original_step(optimizer, *args, **kwargs)

    def counted_ppo(*args, **kwargs):
        with lock:
            index = int(run._PPO_UPDATE_CALLS) + 1
            append_call(ppo_path, index, kind="ppo_update")
            run._PPO_UPDATE_CALLS = index
        return original_ppo(*args, **kwargs)

    def wrapper_chain_reaches(candidate, target) -> bool:
        seen = set()
        current = candidate
        while callable(current) and id(current) not in seen:
            if current is target:
                return True
            seen.add(id(current))
            current = getattr(current, "__wrapped__", None)
        return False

    torch.optim.Adam.step = counted_step
    trainer.ppo_update = counted_ppo

    def undo():
        current_step = torch.optim.Adam.step
        current_ppo = trainer.ppo_update
        if current_step is counted_step and current_ppo is counted_ppo:
            torch.optim.Adam.step = original_step
            trainer.ppo_update = original_ppo
            return
        # PyTorch may place its documented optimizer wrapper around a class-level
        # step hook. Accept only a __wrapped__ chain that reaches this exact
        # invocation counter and only while the paired PPO hook is still exact.
        if wrapper_chain_reaches(current_step, counted_step) and current_ppo is counted_ppo:
            torch.optim.Adam.step = original_step
            trainer.ppo_update = original_ppo
            return
        # The frozen R42 execution wrapper restores its own Adam hook when its
        # authorized call returns. In a controlled pre-episode stop this can
        # happen before the outer private route tears down its PPO journal.
        # Accept only that exact mixed state; never accept an unknown wrapper.
        if current_step is frozen_adam_restore and current_ppo is counted_ppo:
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
            "frozen_r42_adam_restore_captured": frozen_adam_restore is not None,
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

