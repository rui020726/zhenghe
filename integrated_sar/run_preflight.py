"""Check, before anything is summarised, that the batch on disk is the batch that was asked for.

Every number this project reports is an aggregate over files, and every file was written by
a batch script that decided for itself what to run and where to put it. That is enough rope
to hang the result twice over:

* a batch whose command fails but whose shell keeps going leaves the *previous* run's files
  in place, under the new arm's name -- which is how an advisory run came to be reported as
  a binding one. Existence of a path was the whole resume check, and existence was a lie.
* a batch that silently skips on an existing path reports fewer seeds than it claims, and the
  aggregate divides by whatever it found.

So this module answers one question -- *is the set of files on disk exactly the set the
experiment says it is, each of them finished, each of them the arm it is filed under?* -- and
the driver is forbidden from writing a summary unless the answer is yes. It fails on the
first sighting of anything it cannot vouch for, and it never repairs, deletes or skips: a
mismatch is a finding to be explained, not a file to be worked around.

The comparison is per artefact against the signature the run itself wrote, checked against
the split's own facts (which seeds, which stream, which checkpoint) and against the current
code commit. Nothing here trusts a filename.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import run_signature as signatures
from .data_splits import (
    CTDE_CHECKPOINT,
    PILOC_CHECKPOINT,
    arm_dir,
    arm_paths,
    split_of,
    stream_path,
)


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """One arm of one split, and the identity every file of it must carry.

    ``expected_config_hashes`` is what the driver's own command builder produces for this
    arm, seed by seed: the file's recorded configuration hash must equal the hash of the
    configuration that command would build now. That is what catches a file produced by a
    different invocation -- a different policy, a missing flag -- even when every other
    field happens to agree.
    """

    split: str
    arm: str
    sticky: bool = False
    policy: str = "coalition_planner"
    expected_config_hashes: Mapping[int, str] = field(default_factory=dict)
    # What the arm's own configuration says the checkpoints should digest to: ``None`` for
    # a checkpoint the arm does not load at all. Empty means "derive from the split's
    # frozen checkpoints", which is only right for arms that load both of them.
    expected_checkpoints: Mapping[str, Any] = field(default_factory=dict)
    # A prefix of the split's seeds, for a staged run. Empty means every seed of the split.
    seed_list: tuple[int, ...] = ()

    def seeds(self) -> tuple[int, ...]:
        return self.seed_list or split_of(self.split).seeds


@dataclass
class ArmReport:
    spec: ArmSpec
    seeds_expected: int = 0
    seeds_present: int = 0
    seeds_complete: int = 0
    problems: list[str] = field(default_factory=list)
    detection_fingerprints: dict[int, str | None] = field(default_factory=dict)
    config_hashes: dict[int, str] = field(default_factory=dict)
    extras: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.spec.split,
            "arm": self.spec.arm,
            "sticky_commitments": self.spec.sticky,
            "policy": self.spec.policy,
            "seeds_expected": self.seeds_expected,
            "seeds_present": self.seeds_present,
            "seeds_complete": self.seeds_complete,
            "config_hashes": sorted(set(self.config_hashes.values())),
            "problems": list(self.problems),
        }


def _digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def check_arm(
    repo: Path,
    spec: ArmSpec,
    *,
    commit: str,
    checkpoint_digest: str | None,
) -> ArmReport:
    """Every file of one arm, checked against the split, the run's own signature and HEAD."""
    report = ArmReport(spec=spec)
    split = split_of(spec.split)
    seeds = spec.seeds()
    report.seeds_expected = len(seeds)
    seen: dict[int, set[str]] = {}

    for seed in seeds:
        log, trace = arm_paths(repo, spec.split, spec.arm, seed)
        seen.setdefault(seed, set())
        usable = True
        for path, kind in ((log, signatures.LIFECYCLE), (trace, signatures.TRACE)):
            seen[seed].add(path.name)
            artefact = signatures.inspect(path, kind)
            if not artefact.usable:
                usable = False
                report.problems.append(
                    f"seed {seed}: {path.name} {','.join(artefact.problems) or 'unusable'}"
                )
                continue
            signature = artefact.signature or {}
            expected_checkpoints = (
                dict(spec.expected_checkpoints)
                if spec.expected_checkpoints
                else {"piloc": _digest(PILOC_CHECKPOINT), "ctde": checkpoint_digest}
            )
            fields = signatures.difference(
                {
                    "arm": spec.arm,
                    "sticky_commitments": spec.sticky,
                    "code_commit": commit,
                    "checkpoints": expected_checkpoints,
                    "seed": seed,
                    "stream": {
                        "path": str(stream_path(repo, spec.split, seed)),
                        "sha256": _digest(stream_path(repo, spec.split, seed)),
                    },
                    "config_hash": spec.expected_config_hashes.get(seed),
                    "trace_schema": signatures.TRACE_SCHEMA,
                },
                signature,
            )
            if fields:
                report.problems.append(
                    f"seed {seed}: {path.name} signature differs in {sorted(fields)}"
                )
            report.config_hashes[seed] = str(signature.get("config_hash"))
        if not spec.expected_config_hashes:
            report.problems.append(
                f"no expected config hash was supplied for {spec.split}/{spec.arm}"
            )
        if log.exists():
            report.seeds_present += 1
        if usable:
            report.seeds_complete += 1
        report.detection_fingerprints[seed] = signatures.detection_fingerprint(log)

    # Anything else filed under this arm, or under a sibling arm's directory, is reported:
    # a stray file is how a mislabelled batch announces itself.
    directory = arm_dir(repo, spec.split, spec.arm)
    if directory.exists():
        known = {name for names in seen.values() for name in names}
        report.extras = sorted(
            path.name for path in directory.iterdir() if path.name not in known
        )
        if report.extras:
            report.problems.append(f"unexpected files in {directory.name}: {report.extras}")
    return report


def check_fingerprints(
    reports: Sequence[ArmReport], *, splits: Iterable[str] | None = None
) -> list[str]:
    """The paired detection fingerprints: one seed, one stream, one detection sequence.

    Arms of the same split replay the same recorded stream, so for a given seed their
    detection digests must be equal. A disagreement means at least one arm did not process
    the stream it claims, and no per-arm comparison on that seed means anything.
    """
    by_split: dict[str, list[ArmReport]] = {}
    for report in reports:
        if splits is not None and report.spec.split not in set(splits):
            continue
        by_split.setdefault(report.spec.split, []).append(report)
    problems: list[str] = []
    for split_name, group in sorted(by_split.items()):
        seeds: set[int] = set()
        for report in group:
            seeds.update(report.spec.seeds())
        for seed in sorted(seeds):
            found = {
                report.spec.arm: report.detection_fingerprints.get(seed)
                for report in group
            }
            distinct = {value for value in found.values() if value is not None}
            missing = sorted(arm for arm, value in found.items() if value is None)
            if missing:
                problems.append(
                    f"{split_name} seed {seed}: no detection fingerprint for {missing}"
                )
            if len(distinct) > 1:
                problems.append(
                    f"{split_name} seed {seed}: detection fingerprints disagree across arms"
                )
    return problems


def check_separation(reports: Sequence[ArmReport]) -> list[str]:
    """Two arms of one policy that differ only by the sticky flag must differ in config.

    The binding and the advisory planner differ by exactly one field of the configuration,
    so their recorded configuration hashes must differ for every seed. Where they do not,
    the two directories hold two names for one run -- the exact failure this whole check
    exists to make impossible to report.
    """
    groups: dict[tuple[str, str], dict[bool, dict[int, str]]] = {}
    for report in reports:
        key = (report.spec.split, report.spec.policy)
        groups.setdefault(key, {})[report.spec.sticky] = dict(report.config_hashes)
    problems: list[str] = []
    for (split_name, policy), arms in sorted(groups.items()):
        if True not in arms or False not in arms:
            continue
        for seed, advisory_hash in sorted(arms[False].items()):
            binding_hash = arms[True].get(seed)
            if binding_hash is not None and binding_hash == advisory_hash:
                problems.append(
                    f"{split_name} seed {seed}: {policy} advisory and binding arms "
                    "share a config hash"
                )
    return problems


def preflight(
    repo: Path,
    specs: Sequence[ArmSpec],
    *,
    commit: str,
) -> dict[str, Any]:
    """The whole check, as one serialisable verdict. ``ok`` false forbids a summary."""
    checkpoint_digest = _digest(CTDE_CHECKPOINT)
    reports = [
        check_arm(
            repo,
            spec,
            commit=commit,
            checkpoint_digest=checkpoint_digest,
        )
        for spec in specs
    ]
    problems = [problem for report in reports for problem in report.problems]
    problems.extend(check_fingerprints(reports))
    problems.extend(check_separation(reports))
    return {
        "format": "run-preflight/1",
        "commit": commit,
        "checkpoints": {"ctde_sha256": checkpoint_digest},
        "arms": [report.as_dict() for report in reports],
        "problems": problems,
        "ok": not problems,
    }
