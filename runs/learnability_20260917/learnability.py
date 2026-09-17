"""Commitment-continuity experiment driver: run, verify, preflight, then only then conclude.

Three things went wrong in the phase before this one, and every subcommand here exists
because of one of them:

* the batch script built the sticky command, then returned before appending the flag, so
  the runs it launched under sticky names were advisory runs. ``run`` now builds the argv
  first and appends the flag to it, and every artefact carries a :mod:`run_signature` that
  says which arm it actually was.
* resume skipped on the existence of a path, so a failed run left the previous run's file
  in place and was counted as a success. Resume now reuses a file only when its signature
  matches the run that would be launched now *and* the file reached its final record; on
  any mismatch the old file is quarantined and the seed re-run. Nothing is silently kept.
* aggregation read whatever was on disk. ``preflight`` now checks the whole batch -- per
  arm, per seed, both artefacts, signatures, detection fingerprints, and that the advisory
  and binding arms do not share a configuration hash -- and ``verdict`` refuses to write a
  summary until it passes.

Arms write into per-arm directories under the split's run directory, so no two arms can
share a file name even by accident.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from integrated_sar.data_splits import (  # noqa: E402
    CTDE_CHECKPOINT,
    PILOC_CHECKPOINT,
    RECORDING_PROTOCOL,
    SPLITS,
    arm_paths,
    run_dir,
    split_of,
    stream_path,
)
from integrated_sar.run_preflight import ArmSpec, preflight  # noqa: E402
from integrated_sar import run_signature as signatures  # noqa: E402

STEPS = 1000
PLANNER_SCALE = 0.25
# (label, policy, binding). The third entry turns the commitment into something the
# allocator honours; the first two stay advisory so the comparison is against the arm the
# learnability gate measured.
ARMS = (
    ("coalplan0.25", "coalition_planner", False),
    ("coalplan0.25sticky", "coalition_planner", True),
    ("daw0.25", "deadline_aware", False),
)
# Attribution classes, in the order the verdict reports them.
FAILURE_CLASSES = (
    "candidate_ordering_error",
    "task_contention_error",
    "missing_or_late_belief",
    "travel_or_deadline_impossible",
    "reservation_or_execution_break",
    "capacity_conflict",
)


def manifest() -> dict[str, object]:
    """The split registry, with the digests that make a split reproducible."""
    import hashlib

    def digest(path: Path) -> str | None:
        if not path.exists():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "format": "learnability-manifest/2",
        "code_commit": signatures.code_commit(REPO),
        "steps": STEPS,
        "checkpoints": {
            "piloc": {"path": str(PILOC_CHECKPOINT), "sha256": digest(PILOC_CHECKPOINT)},
            "ctde": {"path": str(CTDE_CHECKPOINT), "sha256": digest(CTDE_CHECKPOINT)},
        },
        "splits": {
            name: {
                "purpose": split.purpose,
                "frozen": split.frozen,
                "seeds": list(split.seeds),
                "stream_dir": str(split.stream_dir(REPO)),
                "arm_dir": str(run_dir(REPO, name) / "arms"),
                "recorded_with": RECORDING_PROTOCOL,
            }
            for name, split in SPLITS.items()
        },
        "arms": {label: policy for label, policy, _ in ARMS},
    }


def command(seed: int, label: str, policy: str, split_name: str, binding: bool = False) -> list[str]:
    """The joint_driver argv for one arm on one recorded stream.

    The argv is assembled before anything is appended to it: an early return here is how
    the binding flag was once lost while the output file name kept claiming it.
    """
    log, trace = arm_paths(REPO, split_name, label, seed)
    argv = [
        sys.executable or "python",
        "-m",
        "integrated_sar.joint_driver",
        "--task-checkpoint",
        str(CTDE_CHECKPOINT),
        "--actor-kv-mode",
        "tokens",
        "--steps",
        str(STEPS),
        "--seed",
        str(seed),
        "--replay-stream",
        str(stream_path(REPO, split_name, seed)),
        "--quiet",
        "--output",
        str(log),
        "--residual-scale",
        f"{PLANNER_SCALE:g}",
        "--coordinator",
        policy,
        "--decision-trace",
        str(trace),
    ]
    if binding:
        argv.append("--sticky-commitments")
    return argv


def expected_config_hashes(split_name: str, seeds: tuple[int, ...] | None = None) -> dict[tuple[str, str], dict[int, str]]:
    """The configuration hash each arm's command builds now, seed by seed.

    Read from the argv the driver itself would launch, so the expectation and the run come
    from one place and cannot disagree.
    """
    from integrated_sar.joint_driver import build_parser, configuration_from_args
    from integrated_sar.run_signature import config_hash
    seeds = split_of(split_name).seeds if seeds is None else seeds
    parser = build_parser()
    out: dict[tuple[str, str], dict[int, str]] = {}
    for label, policy, binding in ARMS:
        hashes: dict[int, str] = {}
        for seed in seeds:
            args = parser.parse_args(command(seed, label, policy, split_name, binding)[3:])
            hashes[seed] = config_hash(configuration_from_args(args))
        out[(split_name, label)] = hashes
    return out


def specs_for(split_name: str, limit: int | None = None) -> list[ArmSpec]:
    from integrated_sar.joint_driver import build_parser, configuration_from_args
    from integrated_sar.run_signature import sha256_file

    seeds = split_of(split_name).seeds
    if limit is not None:
        seeds = seeds[:limit]
    hashes = expected_config_hashes(split_name, seeds)
    specs: list[ArmSpec] = []
    for label, policy, binding in ARMS:
        # The checkpoints an arm actually loads: the replay arms load only the CTDE
        # network, so their recorded piloc digest is None and must be compared to None.
        args = build_parser().parse_args(
            command(seeds[0], label, policy, split_name, binding)[3:]
        )
        configuration = configuration_from_args(args)
        specs.append(
            ArmSpec(
                split=split_name,
                arm=label,
                sticky=binding,
                policy=policy,
                expected_config_hashes=hashes[(split_name, label)],
                expected_checkpoints={
                    "piloc": sha256_file(configuration.piloc_checkpoint),
                    "ctde": sha256_file(configuration.task_checkpoint),
                },
                seed_list=seeds,
            )
        )
    return specs


def batch_commit() -> str:
    """The commit the batch was produced at, read off the artefacts themselves.

    Comparing against ``HEAD`` would make every preflight fail the moment the working tree
    was committed, which is backwards: the artefacts are evidence about the tree they were
    produced in. So the commit is read from the runs and then checked to exist in the
    repository, and every file of every arm is required to agree with it.
    """
    commits: set[str] = set()
    for split_name in SPLITS:
        for label, _, _ in ARMS:
            for seed in split_of(split_name).seeds:
                log, trace = arm_paths(REPO, split_name, label, seed)
                for path, kind in ((log, signatures.LIFECYCLE), (trace, signatures.TRACE)):
                    artefact = signatures.inspect(path, kind)
                    if artefact.signature:
                        commits.add(str(artefact.signature.get("code_commit")))
    if not commits:
        return signatures.code_commit(REPO)
    if len(commits) > 1:
        raise SystemExit(
            f"[preflight] artefacts disagree about the code commit: {sorted(commits)}"
        )
    commit = commits.pop()
    check = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=REPO,
        capture_output=True,
    )
    if check.returncode != 0:
        raise SystemExit(f"[preflight] recorded commit {commit} is not in this repository")
    return commit


def quarantine(path: Path, root: Path) -> Path | None:
    """Move a rejected artefact aside, never delete it.

    Another process may still hold the file for a moment, so the move is retried briefly;
    a file that has already been moved by someone else is not an error.
    """
    import time

    if not path.exists():
        return None
    target = root / path.relative_to(REPO)
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            path.replace(target)
            print(f"  quarantined {path} -> {target}")
            return target
        except PermissionError:
            if not path.exists():
                return None
            time.sleep(0.5 * (attempt + 1))
    print(f"  WARNING: could not quarantine {path}; it stays in place and is re-run")
    return None


def _run_batch(jobs: list[list[str]], args: argparse.Namespace) -> int:
    """Run the list exactly as given: callers apply the shard filter themselves, because
    the resume scan that decides which jobs to run must be sharded the same way."""
    for position, argv in enumerate(jobs, start=1):
        print(f"[{position}/{len(jobs)}] {' '.join(argv[-4:])}", flush=True)
        result = subprocess.run(argv, cwd=REPO)
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode})", flush=True)
            return result.returncode
    return 0


def shard(jobs: list, spec: str | None) -> list:
    if not spec:
        return jobs
    index, count = (int(part) for part in spec.split("/"))
    return [job for position, job in enumerate(jobs) if position % count == index]


def resume_decision(seed: int, label: str, split_name: str, binding: bool, argv: list[str]) -> tuple[str, list[str]]:
    """Reuse, quarantine-and-rerun, or run: never skip on existence alone.

    A file is reusable only when it is complete *and* its signature matches the run that
    would be launched now -- every field, including the code commit and the configuration
    hash. Anything else is moved into a quarantine directory and the seed is run again, and
    the reason is printed rather than swallowed.
    """
    log, trace = arm_paths(REPO, split_name, label, seed)
    reasons: list[str] = []
    for path, kind in ((log, signatures.LIFECYCLE), (trace, signatures.TRACE)):
        artefact = signatures.inspect(path, kind)
        if artefact.usable:
            continue
        reasons.append(f"{path.name}: {','.join(artefact.problems) or 'unusable'}")
    if not reasons:
        from integrated_sar.joint_driver import build_parser, configuration_from_args
        from integrated_sar.run_signature import build as build_signature

        parser = build_parser()
        configuration = configuration_from_args(parser.parse_args(argv[3:]))
        expected = build_signature(
            configuration,
            arm=label,
            repo=REPO,
        )
        for path, kind in ((log, signatures.LIFECYCLE), (trace, signatures.TRACE)):
            artefact = signatures.inspect(path, kind)
            fields = signatures.difference(expected, artefact.signature or {})
            if fields:
                reasons.append(f"{path.name}: signature differs in {sorted(fields)}")
    if not reasons:
        return "reuse", reasons
    for path in (log, trace):
        quarantine(path, HERE / "quarantine_rejected")
    return "run", reasons


def do_record(args: argparse.Namespace) -> int:
    split = split_of(args.split)
    run_dir(REPO, args.split).mkdir(parents=True, exist_ok=True)
    split.stream_dir(REPO).mkdir(parents=True, exist_ok=True)
    jobs = [
        record_command(seed, args.split)
        for seed in split.seeds
        if not stream_path(REPO, args.split, seed).exists()
    ]
    print(f"{len(jobs)} streams to record for split {args.split!r}")
    return _run_batch(shard(jobs, args.shard), args)


def record_command(seed: int, split_name: str) -> list[str]:
    """One live PILOC pass; its detections are the stream every arm of the split replays."""
    return [
        sys.executable or "python",
        "-m",
        "integrated_sar.joint_driver",
        "--piloc-checkpoint",
        str(PILOC_CHECKPOINT),
        "--task-checkpoint",
        str(CTDE_CHECKPOINT),
        "--actor-kv-mode",
        "tokens",
        "--steps",
        str(STEPS),
        "--seed",
        str(seed),
        "--no-coordinator",
        "--quiet",
        "--record-stream",
        str(stream_path(REPO, split_name, seed)),
        "--output",
        str(run_dir(REPO, split_name) / f"rec_s{seed}.jsonl"),
    ]


def do_run(args: argparse.Namespace) -> int:
    split = split_of(args.split)
    seeds = split.seeds if not args.limit else split.seeds[: args.limit]
    for label, _, _ in ARMS:
        arm_paths(REPO, args.split, label, seeds[0])[0].parent.mkdir(
            parents=True, exist_ok=True
        )
    missing = [seed for seed in seeds if not stream_path(REPO, args.split, seed).exists()]
    if missing:
        print(f"{len(missing)} streams missing; run the record pass first")
        return 1
    # The shard filter is applied BEFORE the resume scan: the scan has side effects
    # (quarantining rejected artefacts), so every seed must be scanned by exactly one
    # process, or two shards will race on the same files.
    candidates = [
        (seed, label, policy, binding, command(seed, label, policy, args.split, binding))
        for label, policy, binding in ARMS
        for seed in seeds
    ]
    jobs: list[list[str]] = []
    reused = 0
    for seed, label, _, binding, argv in shard(candidates, args.shard):
        verdict, reasons = resume_decision(seed, label, args.split, binding, argv)
        if verdict == "reuse":
            reused += 1
            continue
        if reasons and args.verbose:
            for reason in reasons:
                print(f"  rerun {label} s{seed}: {reason}")
        jobs.append(argv)
    print(f"{len(jobs)} arm runs for split {args.split!r} ({reused} artefact pairs reused)")
    return _run_batch(jobs, args)


def do_preflight(args: argparse.Namespace) -> int:
    payload = preflight(REPO, specs_for(args.split, args.limit), commit=batch_commit())
    suffix = f"_{args.limit}" if getattr(args, "limit", None) else ""
    out = HERE / f"preflight_{args.split}{suffix}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for report in payload["arms"]:
        print(
            f"  {report['split']}/{report['arm']:<22} present {report['seeds_present']:>3}"
            f"/{report['seeds_expected']} complete {report['seeds_complete']:>3}"
            f" problems {len(report['problems'])}"
        )
    if payload["ok"]:
        print(f"preflight ok: {out}")
        return 0
    print(f"preflight FAILED ({len(payload['problems'])} problems): {out}")
    for problem in payload["problems"][:20]:
        print(f"  - {problem}")
    return 1


def do_verify(args: argparse.Namespace) -> int:
    """Run one seed both ways and audit the two traces against each other.

    This is the hand check the phase asks for before any batch is trusted: for every
    decision that was made under a commitment, print what the allocator wanted, what the
    commitment imposed, what the mask left, and what was actually commanded -- then assert
    the binding arm never executed an override it had no reason for.
    """
    from integrated_sar.allocation_trace import read_trace

    seed = args.seed
    outcomes = {}
    for label, policy, binding in ARMS[:2]:
        argv = command(seed, label, policy, "analysis_legacy", binding)
        result = subprocess.run(argv, cwd=REPO)
        if result.returncode != 0:
            return result.returncode
        log, trace = arm_paths(REPO, "analysis_legacy", label, seed)
        _, decisions = read_trace(trace)
        rows = []
        unjustified = 0
        prevented = 0
        for decision in decisions:
            if not decision.get("commitment_present"):
                continue
            row = {
                "step": decision.get("step"),
                "agent": decision.get("agent_id"),
                "held": bool(decision.get("commitment_held")),
                "release": decision.get("commitment_release_reason"),
                "validity": decision.get("commitment_validity"),
                "raw_choice": decision.get("raw_choice"),
                "forced_choice": decision.get("forced_choice"),
                "post_mask_choice": decision.get("post_mask_choice"),
                "actual_assignment": decision.get("actual_assignment"),
                "committed_task": decision.get("commitment_task"),
                "executed": decision.get("reservation_executed"),
                "reservation_id": decision.get("reservation_id"),
                "advice_revision": decision.get("advice_revision"),
            }
            rows.append(row)
            if row["held"] and not row["executed"] and not row["release"]:
                unjustified += 1
            if row["held"] and row["executed"] and row["forced_choice"] is not None:
                prevented += 1
        outcomes[label] = {
            "decisions_under_a_commitment": len(rows),
            "counterfactual_override_blocked": prevented,
            "executed_override_unjustified": unjustified,
            "rows": rows,
        }
    payload = {"seed": seed, "arms": outcomes}
    out = HERE / f"verify_s{seed}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for label, block in outcomes.items():
        print(
            f"  {label:<22} decisions under a commitment {block['decisions_under_a_commitment']:>4}"
            f" blocked {block['counterfactual_override_blocked']:>3}"
            f" unjustified {block['executed_override_unjustified']}"
        )
    if outcomes["coalplan0.25sticky"]["executed_override_unjustified"]:
        print("  sticky arm executed an override it had no reason for -- trace printed above")
        for row in outcomes["coalplan0.25sticky"]["rows"]:
            if row["held"] and not row["executed"] and not row["release"]:
                print(f"    {row}")
        return 1
    print(f"verify ok: {out}")
    return 0


def do_manifest(args: argparse.Namespace) -> int:
    payload = manifest()
    out = HERE / "manifest.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    for name, split in payload["splits"].items():  # type: ignore[union-attr]
        print(f"  {name:16} {len(split['seeds']):>4} seeds  {split['purpose']}")
    return 0


def do_verdict(args: argparse.Namespace) -> int:
    from integrated_sar.learnability import (
        build_verdict,
        commitment_metrics,
        compare_sticky,
        expired_target_rows,
        split_diagnostics,
    )

    gate_problem = False
    for split_name in args.splits:
        if preflight(REPO, specs_for(split_name), commit=batch_commit())["ok"]:
            continue
        gate_problem = True
        print(f"[verdict] preflight failed for {split_name}; refusing to summarise")
    if gate_problem:
        return 1

    verdict = build_verdict(REPO, splits=args.splits)
    sticky = {}
    for split_name in args.splits:
        advisory_rows = expired_target_rows(REPO, split_name, arm="coalplan0.25")
        binding_rows = expired_target_rows(REPO, split_name, arm="coalplan0.25sticky")
        kept = {(int(row["seed"]), str(row["target"])) for row in binding_rows}
        survivors = [
            row for row in advisory_rows if (int(row["seed"]), str(row["target"])) not in kept
        ]
        sticky[split_name] = {
            "comparison": compare_sticky(REPO, split_name),
            "commitments": commitment_metrics(REPO, split_name, arm="coalplan0.25sticky"),
            "advisory_expired_rows": advisory_rows,
            "binding_expired_rows": binding_rows,
            "survivors": survivors,
        }
    payload = {
        "format": "commitment-continuity/2",
        "learnability": verdict,
        "sticky": sticky,
        "split_diagnostics": split_diagnostics(REPO),
    }
    out = HERE / "commitment_continuity.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    write_continuity(payload, HERE / "COMMITMENT_CONTINUITY.md")
    print(f"wrote {HERE / 'COMMITMENT_CONTINUITY.md'}")
    learnability_out = HERE / "learnability_verdict.json"
    learnability_out.write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {learnability_out}")
    return 0


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_continuity(payload: dict, path: Path) -> None:
    """The document, generated from the numbers rather than retyped beside them."""
    lines: list[str] = []
    add = lines.append
    add("# Commitment continuity: advisory plan versus binding plan")
    add("")
    add(
        "Arms share the same recorded detection stream per seed; the only difference is "
        "whether a standing coalition commitment binds the allocator. Every artefact "
        "carries a run signature, resume is fail-closed, and the preflight over the whole "
        "batch had to pass before this page was written."
    )
    add("")
    for name, block in payload["sticky"].items():
        comparison = block["comparison"]
        commitments = block["commitments"]
        add(f"## {name}")
        add("")
        add(f"- seeds compared: {_fmt(comparison['seeds'])}")
        add(
            f"- expired: advisory {_fmt(comparison['advisory_expired'])} -> binding "
            f"{_fmt(comparison['binding_expired'])}; paired "
            f"{_fmt(comparison['paired_expired']['mean'])} "
            f"(95% CI {_fmt(comparison['paired_expired']['ci95'])})"
        )
        add(
            f"- completion: paired {_fmt(comparison['paired_completion']['mean'])} "
            f"(95% CI {_fmt(comparison['paired_completion']['ci95'])})"
        )
        add(
            f"- rejected assignments {_fmt(comparison['rejected_assignments'])}, "
            f"discarded work mean on completed targets {_fmt(comparison['discarded_work_mean'])} "
            f"(on expired {_fmt(comparison['discarded_work_expired_mean'])}), "
            f"first-work latency median {_fmt(comparison['first_work_latency_median'])}, "
            f"deadlock {comparison['deadlock']}"
        )
        add(
            f"- decisions under a commitment {_fmt(commitments['decisions_with_a_commitment'])}, "
            f"held {_fmt(commitments['commitments_held'])} "
            f"(share {_fmt(commitments['commitment_honour_share'])})"
        )
        add(
            f"- counterfactual overrides blocked {_fmt(commitments['counterfactual_override_blocked'])}; "
            f"executed overrides without a release reason "
            f"**{_fmt(commitments['executed_override_unjustified'])}**"
        )
        add(f"- release reasons: `{commitments['release_reasons']}`")
        add(
            f"- longest idle hold {_fmt(commitments['longest_idle_hold'])} decision cycles "
            f"(deadlock threshold 20)"
        )
        add("")
        from integrated_sar.learnability import class_counts

        advisory_rows = block["advisory_expired_rows"]
        binding_rows = block["binding_expired_rows"]
        survivors = block["survivors"]
        add("### Expiry attribution")
        add("")
        add(
            f"- advisory arm expired {len(advisory_rows)}: "
            f"`{class_counts(advisory_rows)}`"
        )
        add(
            f"- binding arm expired {len(binding_rows)}: "
            f"`{class_counts(binding_rows)}`"
        )
        add(
            f"- survivors ({len(survivors)}): expired under the advisory arm, no longer "
            f"expired under the binding arm"
        )
        for row in survivors:
            add(
                f"  - seed {row['seed']} {row['target']}: primary `{row['class_reason']}`"
                + (f", secondary `{row.get('secondary_reasons')}`" if row.get("secondary_reasons") else "")
            )
        add("")
    diagnostics = payload["split_diagnostics"]
    add("## Split diagnostics")
    add("")
    for name, block in diagnostics["blocks"].items():
        fleet = block["fleet"]
        add(f"### {name}")
        add("")
        add(f"- purpose: {block['purpose']}")
        add(f"- code commit: `{block['code_commit']}`")
        add(f"- piloc sha256: `{_fmt(block['checkpoints']['piloc_sha256'])}`")
        add(f"- ctde sha256: `{_fmt(block['checkpoints']['ctde_sha256'])}`")
        add(
            f"- expired attributed {block['expired_attributed']} over "
            f"{block['seeds_with_runs']} seeds; classes `{block['primary_classes']}`"
        )
        add(f"- secondary classes: `{block['secondary_classes']}`")
        add(
            f"- delivery budget median {_fmt(block['delivery_budget_median'])}, "
            f"assignment latency median {_fmt(block['assignment_latency_median'])}, "
            f"detected-at median {_fmt(block['detected_at_median'])}"
        )
        add(
            f"- fleet: idle at detection median "
            f"{_fmt(fleet['idle_agents_at_detection_median'])}, concurrent active tasks "
            f"{_fmt(fleet['active_tasks_at_detection_median'])}, detection to first "
            f"needs_decision {_fmt(fleet['detection_to_first_decision_median'])}, earliest "
            f"agent availability {_fmt(fleet['earliest_agent_availability_median'])}, "
            f"cycle offset {_fmt(fleet['detection_cycle_offset_median'])}, wait to next "
            f"cycle {_fmt(fleet['detection_wait_to_next_cycle_median'])}"
        )
        add(f"- communication: `{fleet['comm']}`")
        add("")
    add(
        "Stage duration is deliberately not implemented here: the belief carries no work "
        "half of the budget, so that ablation stays a separate, independently seeded step."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)

    maker = sub.add_parser("manifest", help="write manifest.json")
    maker.set_defaults(func=do_manifest)

    recorder = sub.add_parser("record", help="record the fresh analysis streams")
    recorder.add_argument("--split", default="analysis_fresh")
    recorder.add_argument("--shard", default=None, help="i/n")
    recorder.set_defaults(func=do_record)

    runner = sub.add_parser("run", help="run the planner, sticky and oracle arms on a split")
    runner.add_argument("--split", default="analysis_fresh")
    runner.add_argument("--shard", default=None, help="i/n")
    runner.add_argument("--limit", type=int, default=None, help="run only the first N seeds")
    runner.add_argument("--verbose", action="store_true", help="print why a seed was re-run")
    runner.set_defaults(func=do_run)

    checker = sub.add_parser("preflight", help="check the batch before anything summarises it")
    checker.add_argument("--split", default="analysis_legacy")
    checker.add_argument("--limit", type=int, default=None, help="check only the first N seeds")
    checker.set_defaults(func=do_preflight)

    auditor = sub.add_parser(
        "verify", help="run one seed advisory and binding, then audit the two traces"
    )
    auditor.add_argument("--seed", type=int, default=20019)
    auditor.set_defaults(func=do_verify)

    reporter = sub.add_parser("verdict", help="attribute, compare and decide -- preflight first")
    reporter.add_argument("--splits", nargs="*", default=["analysis_legacy", "analysis_fresh"])
    reporter.set_defaults(func=do_verdict)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
