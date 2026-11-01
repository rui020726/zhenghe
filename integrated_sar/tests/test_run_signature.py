"""Run signatures and the preflight: no artefact is read as an arm it did not run.

The phase before this one produced files named for the sticky arm that had been produced
without the flag, and nothing noticed, because nothing recorded what a run *was*. These
tests pin the two mechanisms that make that impossible to repeat:

* every artefact a run writes carries a signature -- arm, flag, commit, checkpoints, seed,
  stream, configuration hash, schema -- and is complete only when it reached its final
  record;
* the preflight refuses a batch unless every expected file is present, finished, signed by
  the arm it is filed under, replayed the same detections as its siblings, and does not
  share a configuration hash with the arm it differs from by one flag.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from integrated_sar import run_signature as signatures
from integrated_sar.data_splits import SPLITS, arm_paths, split_of
from integrated_sar.run_preflight import (
    ArmReport,
    ArmSpec,
    check_fingerprints,
    preflight,
)


def lifecycle_log(
    path: Path,
    *,
    signature: dict | None,
    complete: bool = True,
    detections: int = 2,
) -> None:
    """A minimal but structurally honest lifecycle log."""
    lines: list[str] = []
    header: dict = {"target_id": "__run__", "event_type": "run_config", "timestamp": 0.0}
    if signature is not None:
        header["detail"] = {"run_signature": signature}
    lines.append(json.dumps(header))
    for index in range(detections):
        lines.append(
            json.dumps(
                {
                    "target_id": f"target-{index}",
                    "event_type": "detection",
                    "timestamp": 1.0 + index,
                    "source_event_id": f"event-{index}",
                    "detail": {"grid_position": [index, index]},
                }
            )
        )
    if complete:
        lines.append(
            json.dumps(
                {
                    "target_id": "__run__",
                    "event_type": "final",
                    "timestamp": 100.0,
                    "status": "run_finished",
                    "detail": {"step": 100, "metrics": {}},
                }
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def trace_file(path: Path, *, signature: dict | None, complete: bool = True) -> None:
    header: dict = {"event_type": "trace_config", "format": signatures.TRACE_SCHEMA, "meta": {}}
    if signature is not None:
        header["meta"] = {"run_signature": signature}
    lines = [json.dumps(header)]
    if complete:
        lines.append(
            json.dumps(
                {
                    "event_type": "trace_final",
                    "format": signatures.TRACE_SCHEMA,
                    "records": 0,
                }
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def a_signature(*, arm: str = "coalplan0.25", seed: int = 20000, sticky: bool = False) -> dict:
    return {
        "format": signatures.SIGNATURE_FORMAT,
        "arm": arm,
        "sticky_commitments": sticky,
        "code_commit": "0" * 40,
        "checkpoints": {"piloc": None, "ctde": None},
        "seed": seed,
        "stream": {"path": f"runs/streams/s{seed}.json", "sha256": "a" * 64},
        "config_hash": "c0nfig",
        "trace_schema": signatures.TRACE_SCHEMA,
    }


class SignatureIdentityTests(unittest.TestCase):
    def test_the_same_configuration_builds_the_same_signature(self) -> None:
        from integrated_sar.joint_driver import RunConfiguration

        first = signatures.build(RunConfiguration(seed=1, steps=10), arm="nocoord")
        second = signatures.build(RunConfiguration(seed=1, steps=10), arm="nocoord")
        self.assertEqual(signatures.comparable(first), signatures.comparable(second))

    def test_the_output_paths_are_not_part_of_the_identity(self) -> None:
        from integrated_sar.joint_driver import RunConfiguration

        left = signatures.build(RunConfiguration(seed=1, output="runs/a/rp.jsonl"), arm="nocoord")
        right = signatures.build(RunConfiguration(seed=1, output="runs/b/rp.jsonl"), arm="nocoord")
        self.assertEqual(left["config_hash"], right["config_hash"])

    def test_the_sticky_flag_is_part_of_the_identity(self) -> None:
        from integrated_sar.joint_driver import RunConfiguration

        advisory = signatures.build(RunConfiguration(seed=1), arm="coalplan0.25")
        binding = signatures.build(
            RunConfiguration(seed=1, sticky_commitments=True), arm="coalplan0.25sticky"
        )
        self.assertNotEqual(advisory["config_hash"], binding["config_hash"])
        self.assertFalse(advisory["sticky_commitments"])
        self.assertTrue(binding["sticky_commitments"])

    def test_a_missing_commit_is_a_marker_not_a_match(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            commit = signatures.code_commit(Path(folder))
        self.assertEqual(commit, "<no-git>")


class ArtefactInspectionTests(unittest.TestCase):
    def test_a_signed_and_finished_log_is_usable(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rp_arm_s1.jsonl"
            lifecycle_log(path, signature=a_signature(seed=1))
            artefact = signatures.inspect(path, signatures.LIFECYCLE)
        self.assertTrue(artefact.usable)
        self.assertEqual(artefact.problems, ())

    def test_a_log_without_a_final_record_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rp_arm_s1.jsonl"
            lifecycle_log(path, signature=a_signature(seed=1), complete=False)
            artefact = signatures.inspect(path, signatures.LIFECYCLE)
        self.assertFalse(artefact.usable)
        self.assertIn("incomplete", artefact.problems)

    def test_an_unsigned_log_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rp_arm_s1.jsonl"
            lifecycle_log(path, signature=None)
            artefact = signatures.inspect(path, signatures.LIFECYCLE)
        self.assertFalse(artefact.usable)
        self.assertIn("unsigned", artefact.problems)

    def test_a_missing_file_is_missing_not_empty(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            artefact = signatures.inspect(Path(folder) / "absent.jsonl", signatures.LIFECYCLE)
        self.assertFalse(artefact.exists)
        self.assertEqual(artefact.problems, ("missing",))

    def test_a_trace_is_complete_only_at_its_trace_final_line(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tr_arm_s1.jsonl"
            trace_file(path, signature=a_signature(seed=1))
            self.assertTrue(signatures.inspect(path, signatures.TRACE).usable)
            trace_file(path, signature=a_signature(seed=1), complete=False)
            self.assertFalse(signatures.inspect(path, signatures.TRACE).usable)

    def test_detection_fingerprints_agree_for_the_same_stream(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            left = Path(folder) / "left.jsonl"
            right = Path(folder) / "right.jsonl"
            lifecycle_log(left, signature=a_signature(seed=1), detections=3)
            lifecycle_log(right, signature=a_signature(seed=1), detections=3)
            self.assertEqual(
                signatures.detection_fingerprint(left),
                signatures.detection_fingerprint(right),
            )

    def test_a_dropped_detection_changes_the_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            left = Path(folder) / "left.jsonl"
            right = Path(folder) / "right.jsonl"
            lifecycle_log(left, signature=a_signature(seed=1), detections=3)
            lifecycle_log(right, signature=a_signature(seed=1), detections=2)
            self.assertNotEqual(
                signatures.detection_fingerprint(left),
                signatures.detection_fingerprint(right),
            )

    def test_a_legitimately_empty_run_digests_the_empty_sequence(self) -> None:
        """A seed the frozen search policy never fires on still pairs across arms."""
        with tempfile.TemporaryDirectory() as folder:
            left = Path(folder) / "left.jsonl"
            right = Path(folder) / "right.jsonl"
            lifecycle_log(left, signature=a_signature(seed=1), detections=0)
            lifecycle_log(right, signature=a_signature(seed=1), detections=0)
            digest = signatures.detection_fingerprint(left)
            self.assertIsNotNone(digest)
            self.assertEqual(digest, signatures.detection_fingerprint(right))
        self.assertIsNone(signatures.detection_fingerprint(Path("no/such/file.jsonl")))


class PreflightTests(unittest.TestCase):
    """The batch is checked, not assumed: any defect forbids the summary."""

    SPLIT = "analysis_fresh"

    def _specs(self) -> tuple[ArmSpec, ...]:
        seeds = split_of(self.SPLIT).seeds
        return (
            ArmSpec(
                split=self.SPLIT,
                arm="coalplan0.25",
                sticky=False,
                expected_config_hashes={seed: f"hash-{seed}-advisory" for seed in seeds},
            ),
            ArmSpec(
                split=self.SPLIT,
                arm="coalplan0.25sticky",
                sticky=True,
                expected_config_hashes={seed: f"hash-{seed}-sticky" for seed in seeds},
            ),
        )

    def _signature(
        self,
        spec: ArmSpec,
        repo: Path,
        seed: int,
        *,
        config_hash: str | None = None,
        sticky: bool | None = None,
    ) -> dict:
        from integrated_sar.data_splits import stream_path

        return {
            "format": signatures.SIGNATURE_FORMAT,
            "arm": spec.arm,
            "sticky_commitments": spec.sticky if sticky is None else sticky,
            "code_commit": "0" * 40,
            "checkpoints": {"piloc": None, "ctde": None},
            "seed": seed,
            "stream": {"path": str(stream_path(repo, spec.split, seed)), "sha256": None},
            "config_hash": (
                spec.expected_config_hashes.get(seed) if config_hash is None else config_hash
            ),
            "trace_schema": signatures.TRACE_SCHEMA,
        }

    def _layout(self, repo: Path, specs: tuple[ArmSpec, ...]) -> None:
        for spec in specs:
            for seed in spec.expected_config_hashes:
                log, trace = arm_paths(repo, spec.split, spec.arm, seed)
                signature = self._signature(spec, repo, seed)
                lifecycle_log(log, signature=signature, detections=1)
                trace_file(trace, signature=signature)

    def _run(self, repo: Path, specs: tuple[ArmSpec, ...], *, commit: str = "0" * 40) -> dict:
        with mock.patch.object(signatures, "code_commit", return_value=commit):
            with mock.patch("integrated_sar.run_preflight._digest", return_value=None):
                return preflight(repo, specs, commit=commit)

    def test_a_complete_batch_passes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            specs = self._specs()
            self._layout(repo, specs)
            payload = self._run(repo, specs)
        self.assertTrue(payload["ok"], payload["problems"][:5])
        self.assertTrue(all(report["seeds_complete"] == 60 for report in payload["arms"]))

    def test_a_missing_seed_fails(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            specs = self._specs()
            self._layout(repo, specs)
            log, _ = arm_paths(repo, self.SPLIT, "coalplan0.25", 21001)
            log.unlink()
            payload = self._run(repo, specs)
        self.assertFalse(payload["ok"])
        self.assertTrue(any("missing" in problem for problem in payload["problems"]))

    def test_an_incomplete_log_fails(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            specs = self._specs()
            self._layout(repo, specs)
            log, _ = arm_paths(repo, self.SPLIT, "coalplan0.25", 21000)
            lifecycle_log(log, signature=self._signature(specs[0], repo, 21000), complete=False)
            payload = self._run(repo, specs)
        self.assertFalse(payload["ok"])
        self.assertTrue(any("incomplete" in problem for problem in payload["problems"]))

    def test_a_sticky_label_on_an_advisory_run_fails(self) -> None:
        """The exact failure that produced this whole check."""
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            specs = self._specs()
            self._layout(repo, specs)
            log, trace = arm_paths(repo, self.SPLIT, "coalplan0.25sticky", 21000)
            # Both files were produced without the flag, though they are filed under the
            # binding arm: the signature is what says so.
            lifecycle_log(
                log, signature=self._signature(specs[1], repo, 21000, sticky=False), detections=1
            )
            trace_file(trace, signature=self._signature(specs[1], repo, 21000, sticky=False))
            payload = self._run(repo, specs)
        self.assertFalse(payload["ok"])
        self.assertTrue(
            any("sticky_commitments" in problem for problem in payload["problems"]),
            payload["problems"][:5],
        )

    def test_a_wrong_commit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            specs = self._specs()
            self._layout(repo, specs)
            payload = self._run(repo, specs, commit="1" * 40)
        self.assertFalse(payload["ok"])
        self.assertTrue(any("code_commit" in problem for problem in payload["problems"]))

    def test_a_mismatched_config_hash_fails(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            specs = self._specs()
            self._layout(repo, specs)
            log, trace = arm_paths(repo, self.SPLIT, "coalplan0.25", 21000)
            swapped = self._signature(specs[0], repo, 21000, config_hash="another-invocation")
            lifecycle_log(log, signature=swapped, detections=1)
            trace_file(trace, signature=swapped)
            payload = self._run(repo, specs)
        self.assertFalse(payload["ok"])
        self.assertTrue(any("config_hash" in problem for problem in payload["problems"]))

    def test_arms_that_differ_only_by_the_flag_must_not_share_a_config_hash(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            repo = Path(folder)
            specs = self._specs()
            self._layout(repo, specs)
            advisory_hash = specs[0].expected_config_hashes[21000]
            log, trace = arm_paths(repo, self.SPLIT, "coalplan0.25sticky", 21000)
            shared = self._signature(specs[1], repo, 21000, config_hash=advisory_hash)
            lifecycle_log(log, signature=shared, detections=1)
            trace_file(trace, signature=shared)
            payload = self._run(repo, specs)
        self.assertFalse(payload["ok"])
        self.assertTrue(
            any("share a config hash" in problem for problem in payload["problems"]),
            payload["problems"][:5],
        )

    def test_fingerprints_that_disagree_across_arms_fail(self) -> None:
        self.assertEqual(check_fingerprints([]), [])
        problems = check_fingerprints(_divergent_reports())
        self.assertEqual(len(problems), 1)
        self.assertIn("disagree", problems[0])

    def test_every_seed_of_every_split_is_declared(self) -> None:
        """The registry is the source of the counts, so a batch cannot shrink it."""
        for name, split in SPLITS.items():
            self.assertTrue(split.seeds)
            self.assertEqual(len(split.seeds), len(set(split.seeds)))
            self.assertEqual(split_of(name).name, name)


def _divergent_reports() -> list[ArmReport]:
    seeds = split_of("analysis_fresh").seeds
    left = ArmReport(spec=ArmSpec(split="analysis_fresh", arm="coalplan0.25"))
    right = ArmReport(
        spec=ArmSpec(split="analysis_fresh", arm="coalplan0.25sticky", sticky=True)
    )
    for report in (left, right):
        report.seeds_expected = len(seeds)
        report.seeds_complete = len(seeds)
        report.detection_fingerprints.update({seed: "one" for seed in seeds})
    # One seed where the two arms did not process the same detections: that alone forbids
    # comparing the pair on that seed.
    right.detection_fingerprints[seeds[0]] = "two"
    return [left, right]


if __name__ == "__main__":
    unittest.main()
