from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import r58_private_launcher_repaired as launcher

spec = importlib.util.spec_from_file_location("r58_resume_seam_migration_tested", HERE / "r58_resume_seam.py")
assert spec and spec.loader
seam = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seam)


def current_plan_and_identity():
    legacy = launcher.load_legacy_launcher()
    import r58_contracts
    old_caps = (r58_contracts.RESERVE_CAP, legacy.RESERVE_CAP)
    r58_contracts.RESERVE_CAP = legacy.RESERVE_CAP = 24
    try:
        plan = legacy.plan_from_frozen_sources()
    finally:
        r58_contracts.RESERVE_CAP, legacy.RESERVE_CAP = old_caps
        sys.modules.pop(legacy.__name__, None)
    identity = launcher._repair_identity()
    plan.update({"training_version": launcher.REPAIRED_TRAINING_VERSION,
                 "r58_repair_identity": identity, "generation_id": "round58_gen3",
                 "status": "frozen_approved_gen3",
                 "authorization": "direct_user_owner_approval_identity_bound"})
    return plan, identity


class HistoricalPlanMigrationTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.identity = current_plan_and_identity()
        self.raw_path = (launcher.PRIVATE_RAW / "ROUND53_FIXED_PLAN.json").resolve()
        self.historical = json.loads(self.raw_path.read_text(encoding="utf-8"))

    def test_actual_old_comparison_reproduces_failure_on_saved_plan(self):
        old = {k: v for k, v in self.historical.items() if k != "source_content_identities"}
        current = {k: v for k, v in self.plan.items() if k != "source_content_identities"}
        import r58_private_launcher_repaired as p
        legacy = p.load_legacy_launcher()
        self.assertNotEqual(legacy.canonical_sha(old), legacy.canonical_sha(current))
        self.assertEqual(set(k for k in old if old.get(k) != current.get(k)), {"r58_repair_identity"})

    def test_unique_approved_migration_is_proven_from_plan_freeze_auth_and_all_three_sidecars(self):
        result = seam.validate_historical_plan_identity_migration(REPO, self.plan, self.identity)
        self.assertEqual(result["proof"]["kind"], seam.PLAN_MIGRATION_VERSION)
        self.assertEqual(result["proof"]["qualified_seed_source_attestations"].keys(), {"11", "22", "33"})
        self.assertEqual(result["proof"]["allowed_identity_migration"]["scientific_plan_field_changes"], [])
        self.assertFalse(result["proof"]["allowed_identity_migration"]["source_content_identity_changed"])
        self.assertEqual(result["allowed_comparison_view"]["r58_repair_identity"], self.identity)
        self.assertEqual(hashlib.sha256(self.raw_path.read_bytes()).hexdigest(), seam.HISTORICAL_PLAN_SHA256)

    def test_scientific_parameter_or_order_change_is_rejected(self):
        for key, mutate in (
                ("training_episode_count", lambda p: p.__setitem__("training_episode_count", 383)),
                ("training_order", lambda p: p.__setitem__("training_order", list(reversed(p["training_order"]))))):
            altered = copy.deepcopy(self.plan)
            mutate(altered)
            with self.subTest(key=key), self.assertRaisesRegex(PermissionError, "non-attestation frozen plan"):
                seam.validate_historical_plan_identity_migration(REPO, altered, self.identity)

    def test_non_allowlisted_current_identity_change_is_rejected(self):
        altered = copy.deepcopy(self.identity)
        altered["critic_route_version"] = "tampered"
        with self.assertRaisesRegex(PermissionError, "non-allow-listed change"):
            seam.validate_historical_plan_identity_migration(REPO, self.plan, altered)

    def test_migrated_source_map_or_journal_hash_must_match_live_source(self):
        for field in ("approved_resume_map_sha256", "historical_checkpoint_journal_sha256"):
            altered = copy.deepcopy(self.identity)
            altered[field] = "0" * 64
            with self.subTest(field=field), self.assertRaisesRegex(PermissionError, "live code/source bindings"):
                seam.validate_historical_plan_identity_migration(REPO, self.plan, altered)

    def test_seed_original_attestation_mismatch_is_rejected(self):
        sources = seam.validate_resume_sources(REPO)
        sources["sources"][11]["original_identity"]["adapter_sha256"] = "0" * 64
        with self.assertRaisesRegex(PermissionError, "original sidecar attestation"):
            seam.validate_historical_plan_identity_migration(
                REPO, self.plan, self.identity, source_proof=sources)

    def test_legacy_reader_gets_only_an_in_memory_allowlisted_plan_view(self):
        result = seam.validate_historical_plan_identity_migration(REPO, self.plan, self.identity)
        payload = {"kind": "round58-gen3-plan-identity-migration-proof/1",
                   "status": "verified_allowlisted_identity_only_migration",
                   "proof": result["proof"]}
        # Keep the fixture under the authorized repository root. The desktop
        # sandbox's redirected AppData temp directory may deny both writes and
        # TemporaryDirectory's cleanup, leaving an avoidable external artifact.
        with tempfile.TemporaryDirectory(prefix="r58_plan_migration_", dir=HERE) as tmp:
            proof_path = Path(tmp) / "proof.json"
            proof_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            digest = hashlib.sha256(proof_path.read_bytes()).hexdigest()
            class LegacyReader:
                def read_json(self, path):
                    return json.loads(Path(path).read_text(encoding="utf-8"))
            legacy = LegacyReader()
            original_bytes = self.raw_path.read_bytes()
            installed = seam.install_historical_plan_comparison_view(
                legacy, repo_root=REPO, current_plan=self.plan,
                current_repair_identity=self.identity, proof_path=proof_path,
                expected_proof_sha256=digest)
            view = legacy.read_json(self.raw_path)
            self.assertEqual(view["r58_repair_identity"], self.identity)
            self.assertEqual({k: v for k, v in view.items() if k != "source_content_identities"},
                             {k: v for k, v in self.plan.items() if k != "source_content_identities"})
            self.assertEqual(self.raw_path.read_bytes(), original_bytes)
            installed["undo"]()
            self.assertTrue(installed["legacy_exact_plan_equality_remains_enabled"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
