"""Fixtures for the post-mask retraining round: one setting, one freeze, no new numbers.

Everything measurable is read from the rounds that fixed it. The credit is the locked 25 % tier,
the streams are the blocks round 2 recorded, the sampler and the PPO loop are the ones that passed
M0-M3 at ``e692607``, the weight fingerprint and the paired bootstrap are imported rather than
re-typed, and the trainer identity is read out of the collector module itself so a checkpoint
cannot claim a schema the code no longer produces.

What this file adds is only the round's own business: the frozen setting table, the launch ledger
with its 1,500-episode ceiling, the stream-digest re-verification, the *derived* executed-module
list (the thing last round's manifest got wrong), and the cleanliness rule that refuses to train
when the working tree differs from the registered code freeze.

    python retrain_common.py            # print the setting, the graph and the freeze state
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MASK_FIX_ROUND = ROOT / "runs" / "comm_budget_action_mask_fix_20260922"

for _directory in (
    str(ROOT),
    str(ROOT / "runs" / "comm_budget_pilot_20260919"),
    str(ROOT / "runs" / "comm_stability_20260919"),
    str(ROOT / "runs" / "comm_tight_budget_20260919"),
    str(ROOT / "runs" / "comm_protocol_v2_20260920"),
    str(MASK_FIX_ROUND),
    str(HERE),
):
    if _directory not in sys.path:
        sys.path.insert(0, _directory)

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import protocol_common as fixtures  # noqa: E402  (sys.path is set up above)

TRAIN_SEEDS = tuple(fixtures.TRAIN_SEEDS)
VALIDATION_SEEDS = tuple(fixtures.VALIDATION_SEEDS)
STEPS_PER_EPISODE = int(fixtures.STREAM_STEPS)
weight_fingerprint = fixtures.weight_fingerprint
tensors = fixtures.tensors
write_json = fixtures.write_json
sha256 = fixtures.sha256
metrics = fixtures.metrics

#: The only setting this round is authorised to train in. Each value is read from the round that
#: fixed it, and ``setting()`` refuses to run if any of them has moved.
PRIMARY_TIER = fixtures.PRIMARY_TIER
BUDGET_BYTES = int(fixtures.tier_bytes(PRIMARY_TIER))
TIMING_MODE = str(fixtures.POLICY_MODE)
INFORMATION = "isolated"
DELIVERY_PROBABILITY = 1.0
ACK = False

#: Training seeds, ceiling and cadence -- 总纲 v3.1 §4, fixed before the first episode.
TRAINING_SEEDS = (1, 2, 3)
STEPS_PER_RUN = 400_000
TOTAL_STEP_CAP = 1_200_000
EPISODES_PER_UPDATE = int(fixtures.train_pilot.EPISODES_PER_UPDATE)
CHECKPOINT_STEPS = (0, 100_000, 200_000, 300_000, 400_000)
#: 12 probes + 1,200 training episodes + evaluation/rule margin. A technical failure spends the
#: ceiling too: an arm that cannot finish does not get unlimited retries.
LAUNCH_CAP = 1_500
PROBE_EPISODES = 12
CURVE_STREAM_COUNT = 8
CURVE_STREAMS = VALIDATION_SEEDS[:CURVE_STREAM_COUNT]

#: Statistics, declared the way §7 fixes them. The method is the pilot's; only the draw seed is
#: this round's, and it is applied by ``paired`` below rather than by a second implementation.
BOOTSTRAP_REPLICATIONS = 10_000
BOOTSTRAP_SEED = 20260922
NON_INFERIORITY_MARGIN = -0.05

MAIN_RULE = "send_once"
SEALED_BLOCK = "800000..800399"
CONSUMED_BLOCK = "900000..900199"
FIX_HEAD = "e692607212fcc67fc82d045a8a53d44c79bde4d7"
RESULTS_HEAD = "7f9929c83cd80185830ca1ece4ec6899629605cf"
ARCHIVED_SEND_ONCE_DEV = {
    "document": "runs/comm_protocol_v2_20260920/COMM_PROTOCOL_V2_RESULTS.json",
    "arm": "send_once",
    "tier": "25pct",
    "episodes": 24,
    "mean_on_time_per_episode": 5.79167,
    "mean_airtime_bytes": 14916.0,
    "mean_emissions": 6.0,
    "detected_total": 144,
    "expired_total": 5,
}

TRAIN_OUT = HERE / "train"
EPISODE_OUT = HERE / "episodes"
LAUNCH_LEDGER = HERE / "LAUNCH_LEDGER.json"
FREEZE_FILE = HERE / "CODE_FREEZE.json"
PREFLIGHT_FILE = HERE / "PREFLIGHT.json"


def json_path(name: str) -> Path:
    return HERE / name


def read_json(name: str) -> Any:
    return json.loads(json_path(name).read_text(encoding="utf-8"))


def git(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def git_out(*arguments: str) -> str:
    return git(*arguments).stdout.strip()


# ------------------------------------------------------------------ the frozen setting
def setting() -> dict[str, Any]:
    """The setting as a table, checked rather than asserted in prose.

    ``budget_bytes`` is compared against the locked calibration and against the number the brief
    names, because a tier that silently re-calibrates would change the experiment's question.
    """
    from integrated_sar.comm_learning import FEATURE_LAYOUT, FEATURE_NAMES

    configuration_module = importlib.import_module("integrated_sar.joint_driver")
    configuration = configuration_module.RunConfiguration()
    budget = int(fixtures.tier_bytes(PRIMARY_TIER))
    if budget != 29_300 or budget != int(fixtures.locked_budget("25pct")):
        raise RuntimeError(f"the primary tier is no longer 29,300 B: it reads {budget}")
    if BUDGET_BYTES != budget:
        raise RuntimeError(f"BUDGET_BYTES {BUDGET_BYTES} disagrees with the locked tier {budget}")
    if len(FEATURE_NAMES) != 19 or FEATURE_LAYOUT != "comm-send-state/1":
        raise RuntimeError(
            f"the sender's observation is {len(FEATURE_NAMES)}-dim {FEATURE_LAYOUT}; this round may "
            "not change the network input"
        )
    from budget_masked import ACTION_MASK_SCHEMA, TAPE_SCHEMA, TRAINER_TAG

    return {
        "budget_bytes": budget,
        "budget_tier": PRIMARY_TIER,
        "timing_mode": TIMING_MODE,
        "information": INFORMATION,
        "delivery_probability": DELIVERY_PROBABILITY,
        "ack": ACK,
        "feature_layout": FEATURE_LAYOUT,
        "feature_dimension": len(FEATURE_NAMES),
        "trainer_tag": TRAINER_TAG,
        "action_mask_schema": ACTION_MASK_SCHEMA,
        "tape_schema": TAPE_SCHEMA,
        "episodes_per_update": EPISODES_PER_UPDATE,
        "steps_per_episode": STEPS_PER_EPISODE,
        "steps_per_run": STEPS_PER_RUN,
        "total_step_cap": TOTAL_STEP_CAP,
        "training_seeds": list(TRAINING_SEEDS),
        "checkpoints": list(CHECKPOINT_STEPS),
        "train_streams": [int(seed) for seed in TRAIN_SEEDS],
        "dev_streams": [int(seed) for seed in VALIDATION_SEEDS],
        "curve_streams": [int(seed) for seed in CURVE_STREAMS],
        "main_rule": MAIN_RULE,
        "bootstrap": {"replications": BOOTSTRAP_REPLICATIONS, "rng": BOOTSTRAP_SEED},
        "non_inferiority_margin_targets_per_episode": NON_INFERIORITY_MARGIN,
        "launch_cap": LAUNCH_CAP,
        "probe_episodes": PROBE_EPISODES,
        "network": {
            "hidden": int(fixtures.train_pilot.HIDDEN),
            "gamma": float(fixtures.train_pilot.GAMMA),
            "gae_lambda": float(fixtures.train_pilot.GAE_LAMBDA),
            "clip": float(fixtures.train_pilot.CLIP),
            "value_coefficient": float(fixtures.train_pilot.VALUE_COEFFICIENT),
            "entropy_bonus": float(fixtures.train_pilot.ENTROPY_BONUS),
            "learning_rate": float(fixtures.train_pilot.LEARNING_RATE),
            "max_grad_norm": float(fixtures.train_pilot.MAX_GRAD_NORM),
            "epochs_per_batch": int(fixtures.train_pilot.EPOCHS_PER_BATCH),
            "minibatch_steps": int(fixtures.train_pilot.MINIBATCH_STEPS),
        },
        "production_defaults": {
            "comm_ack": bool(configuration.comm_ack),
            "comm_timing_mode": str(configuration.comm_timing_mode),
            "comm_policy": str(configuration.comm_policy),
            "comm_budget_bytes": configuration.comm_budget_bytes,
            "comm_delivery_probability": configuration.comm_delivery_probability,
        },
    }


# ------------------------------------------------------------------ the data, re-verified
def stream_registry() -> dict[str, str]:
    """The digests recorded when the streams were first cut (round 2's manifest)."""
    manifest = json.loads(
        (
            ROOT / "runs" / "comm_budget_pilot_20260919" / "COMM_ROUND_MANIFEST.json"
        ).read_text(encoding="utf-8")
    )
    return dict(manifest["seeds"]["stream_sha256"])


def verify_streams() -> dict[str, Any]:
    """Same number must still mean same content, or this round is a different dataset.

    Checked over all 72 streams this round touches, not a sample: a silently re-recorded stream
    would move both the training distribution and the dev comparison, and nothing else here would
    notice.
    """
    registry = stream_registry()
    checked: dict[str, Any] = {}
    mismatches: dict[str, Any] = {}
    missing: list[str] = []
    for seed in (*TRAIN_SEEDS, *VALIDATION_SEEDS):
        name = f"s{seed}"
        path = fixtures.pilot.stream_path(seed)
        expected = registry.get(name)
        if expected is None:
            missing.append(name)
            continue
        if not path.exists():
            missing.append(f"{name} (file absent)")
            continue
        actual = sha256(path)
        checked[name] = actual
        if actual != expected:
            mismatches[name] = {"registered": expected, "on_disk": actual}
    return {
        "registry": "runs/comm_budget_pilot_20260919/COMM_ROUND_MANIFEST.json seeds.stream_sha256",
        "streams_checked": len(checked),
        "expected": len(TRAIN_SEEDS) + len(VALIDATION_SEEDS),
        "absent": missing,
        "mismatches": mismatches,
        "training_block": [int(seed) for seed in TRAIN_SEEDS],
        "development_block": [int(seed) for seed in VALIDATION_SEEDS],
        "sealed_block_untouched": SEALED_BLOCK,
        "passed": not missing and not mismatches and len(checked) == 72,
    }


# ------------------------------------------------------------------ what actually executed
#: The entry points of the execution graph. The module list is *derived* from these by importing
#: them, which is the correction to last round's hand-typed allow-list.
ENTRY_MODULES = (
    "train_retrain",
    "train_budget_masked",
    "budget_masked",
    "train_protocol",
    "train_onpolicy",
    "onpolicy",
    "protocol_env",
    "pilot_env",
    "train_pilot",
    "protocol_common",
    "tight_common",
    "stability_common",
    "pilot_common",
)

#: The same graph as it stood at the previous round's delivery head, used by the provenance
#: correction -- which must describe the code that ran then, before this round's harness existed.
MASK_FIX_ENTRIES = (
    "smoke_budget_mask",
    "train_budget_masked",
    "budget_masked",
    "train_protocol",
    "train_onpolicy",
    "onpolicy",
    "protocol_env",
    "pilot_env",
    "train_pilot",
    "protocol_common",
    "tight_common",
    "stability_common",
    "pilot_common",
)

#: Directories whose files are code this round can legitimately execute.
CODE_ROOTS = ("integrated_sar", "runs", "piloc", "task_allocation")


def executed_modules(entries: tuple[str, ...] = ENTRY_MODULES) -> list[str]:
    """Every repo-local module the training entry really imports, by walking ``sys.modules``.

    Last round's manifest listed ``integrated_sar/comm_transcript.py`` because a person typed a
    plausible path: the file has never existed in Git or on disk, and nothing imports it (the class
    it names lives in ``integrated_sar/comm.py``). Deriving the list from the import graph makes
    that class of mistake structurally impossible -- a phantom cannot appear in a list built by
    asking Python what it loaded.
    """
    for name in entries:
        importlib.import_module(name)
    found: set[str] = set()
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if not file:
            continue
        # ``torch.ops`` and friends carry a bare relative ``__file__``, which would resolve into
        # whatever directory happens to be current. An absolute path that exists is the claim.
        candidate = Path(file)
        if not candidate.is_absolute() or not candidate.exists():
            continue
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if "__pycache__" in str(resolved):
            continue
        try:
            relative = resolved.relative_to(ROOT)
        except ValueError:
            continue
        if relative.parts[0] not in CODE_ROOTS:
            continue
        if relative.suffix != ".py":
            continue
        text = relative.as_posix()
        # Another round's artefacts and documents are not this round's executed code.
        if text.startswith("runs/") and not text.startswith(("runs/comm_budget_masked_retrain",)):
            if "/tests/" in text:
                continue
        found.add(text)
    return sorted(found)


def round_scripts() -> list[str]:
    """This directory's own committed drivers, so a freeze covers them too."""
    return sorted(
        (HERE / name).relative_to(HERE).as_posix()
        for name in sorted(os.listdir(HERE))
        if name.endswith(".py") and not name.startswith("__")
    )


def blob_identity(paths: list[str]) -> dict[str, dict[str, Any]]:
    """Git blob ids at HEAD and in the working tree, with "the path is not in HEAD" as its own answer.

    ``git rev-parse HEAD:<missing path>`` does *not* fail -- it echoes the argument back on stdout.
    Last round's builder truncated that echo into ``head_blob: "HEAD:integrated_"`` and then
    reported the path as a dirty module, so an absent file was indistinguishable from an edited
    one. Absence is now resolved with ``git cat-file -e`` and recorded as ``in_head: false``.
    """
    report: dict[str, dict[str, Any]] = {}
    for path in paths:
        in_head = git("cat-file", "-e", f"HEAD:{path}").returncode == 0
        head = git_out("rev-parse", f"HEAD:{path}") if in_head else ""
        working = git_out("hash-object", path) if Path(ROOT / path).exists() else ""
        report[path] = {
            "in_head": in_head,
            "on_disk": Path(ROOT / path).exists(),
            "head_blob": head[:16] or None,
            "working_blob": working[:16] or None,
            "matches_head": bool(head) and head == working,
        }
    return report


def working_tree_diff(reference: str, paths: list[str]) -> dict[str, Any]:
    """``git diff --quiet <reference> -- <paths>``, reported rather than swallowed."""
    changed = git_out("diff", "--name-only", reference, "HEAD", "--", *paths).splitlines()
    uncommitted = git_out("diff", "--name-only", reference, "--", *paths).splitlines()
    return {
        "reference": reference,
        "committed_difference": sorted(set(changed)),
        "working_tree_difference": sorted(set(uncommitted)),
        "clean": not changed and not uncommitted,
    }


def untracked_code(paths: list[str]) -> list[str]:
    """Graph members that Git does not track -- a freeze cannot cover what it has no object for.

    Scoped to the given paths and this round's own directory on purpose. The repo carries other
    sessions' untracked files (an unfinished script under ``comm_multistep_stage1_20260922``), and
    a check that treated every untracked ``.py`` anywhere as this round's fault would never pass
    and would say nothing about this round's code.
    """
    listing = set(
        git_out("ls-files", "--others", "--exclude-standard", "--", *CODE_ROOTS).splitlines()
    )
    local = HERE.relative_to(ROOT).as_posix()
    suspects = set(paths) | {
        f"{local}/{name}" for name in sorted(os.listdir(HERE)) if name.endswith(".py")
    }
    return sorted(path for path in suspects if path in listing)


def freeze() -> dict[str, Any]:
    """The registered freeze, or an unregistered stub.

    ``registered`` is derived from the presence of a SHA rather than trusted from the file: a
    freeze that names a commit *is* registered whatever its own flag says, and a flag without a
    SHA is nothing to check against. Reading the two independently is how a first registration
    that recorded a SHA but omitted the flag reported itself as unregistered.
    """
    if not FREEZE_FILE.exists():
        return {"registered": False, "sha": None}
    payload = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    payload["registered"] = bool(payload.get("sha"))
    return payload


def freeze_sha() -> str:
    payload = freeze()
    if not payload.get("registered"):
        raise RuntimeError(
            "no code freeze is registered; provenance §2.5 forbids training against an "
            "unregistered execution path"
        )
    return str(payload["sha"])


def freeze_cleanliness(paths: list[str] | None = None) -> dict[str, Any]:
    """The rule that gates every launch: what runs must be what was frozen.

    Checked against the freeze SHA for the *derived* module list, not for a hand-written one, so
    adding an import to the trainer is automatically inside the check instead of outside it.
    """
    payload = freeze()
    sha = str(payload.get("sha") or "")
    if not sha:
        return {"registered": False, "clean": False, "reason": "no freeze registered"}
    modules = paths if paths is not None else executed_modules()
    diff = working_tree_diff(sha, modules)
    orphan = untracked_code(modules)
    frozenblobs = payload.get("modules") or {}
    now = blob_identity(modules)
    moved = sorted(
        path
        for path, ids in now.items()
        if path in frozenblobs and ids.get("working_blob") != frozenblobs[path].get("working_blob")
    )
    added = sorted(path for path in now if path not in frozenblobs)
    return {
        "registered": True,
        "sha": sha,
        "modules_in_graph": len(modules),
        "committed_difference": diff["committed_difference"],
        "working_tree_difference": diff["working_tree_difference"],
        "untracked_code": orphan,
        "modules_not_in_frozen_graph": added,
        "modules_whose_blob_moved": moved,
        "clean": diff["clean"] and not orphan and not moved and not added,
    }


# ------------------------------------------------------------------ launches
class LaunchLedger:
    """One counter for every episode this round starts, with the ceiling enforced at reserve time.

    Training, probing and evaluation all draw from the same 1,500, and a launch that raised is
    still recorded: the ceiling exists to bound the round's cost, not to bound its successes.

    Each process owns one file. The three training arms run in parallel and the payload is a
    read-modify-write JSON document, so a single shared ledger would have each process overwrite
    the others' counts -- an accounting defect in the one artefact whose job is accounting. The cap
    is still global because ``reserve`` reads every ledger file, not just its own.
    """

    PATTERN = "LAUNCH_LEDGER*.json"

    def __init__(self, name: str = "LAUNCH_LEDGER.json") -> None:
        self.path = HERE / name
        self.payload = (
            json.loads(self.path.read_text(encoding="utf-8"))
            if self.path.exists()
            else {"cap": LAUNCH_CAP, "starts": 0, "by_kind": {}, "entries": [], "failures": 0}
        )
        if int(self.payload["cap"]) != LAUNCH_CAP:
            raise RuntimeError(
                f"the ledger was opened with cap {self.payload['cap']}, this round's ceiling is "
                f"{LAUNCH_CAP}"
            )

    @property
    def starts(self) -> int:
        return int(self.payload["starts"])

    def total_starts(self) -> int:
        return ledger_totals()["starts"]

    def remaining(self) -> int:
        return LAUNCH_CAP - self.total_starts()

    def reserve(self, kind: str, seed: int, *, arm: str = "") -> None:
        total = self.total_starts()
        if total >= LAUNCH_CAP:
            raise RuntimeError(
                f"the {LAUNCH_CAP}-episode launch ceiling is spent ({total} started across "
                f"{self.PATTERN}); refusing to start {kind} s{seed} rather than extending the round"
            )
        self.payload["starts"] = self.starts + 1
        self.payload["by_kind"][kind] = int(self.payload["by_kind"].get(kind, 0)) + 1
        self._write()

    def record(self, kind: str, seed: int, ok: bool, detail: str, *, arm: str = "") -> None:
        self.payload["entries"].append(
            {
                "index": len(self.payload["entries"]) + 1,
                "kind": kind,
                "arm": arm,
                "seed": int(seed),
                "ok": bool(ok),
                "detail": detail,
            }
        )
        self.payload["failures"] = sum(
            1 for entry in self.payload["entries"] if not entry["ok"]
        )
        self._write()

    def _write(self) -> None:
        write_json(self.path, self.payload)


def ledger_totals() -> dict[str, Any]:
    """Every ledger file added up: the round's real cost, across processes and reruns."""
    files = sorted(HERE.glob(LaunchLedger.PATTERN))
    totals: dict[str, Any] = {
        "cap": LAUNCH_CAP,
        "starts": 0,
        "failures": 0,
        "by_kind": {},
        "files": [],
        "entries": [],
    }
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        totals["starts"] += int(payload.get("starts", 0))
        totals["failures"] += int(payload.get("failures", 0))
        for kind, count in (payload.get("by_kind") or {}).items():
            totals["by_kind"][kind] = int(totals["by_kind"].get(kind, 0)) + int(count)
        totals["files"].append(
            {"file": path.name, "starts": int(payload.get("starts", 0))}
        )
        totals["entries"].extend(payload.get("entries") or [])
    totals["remaining"] = LAUNCH_CAP - totals["starts"]
    totals["within_cap"] = totals["starts"] <= LAUNCH_CAP
    return totals


# ------------------------------------------------------------------ statistics
def paired(
    rows: list[dict[str, Any]], *, baseline: str, arm: str, key: str
) -> dict[str, Any]:
    """One paired bootstrap for one arm against one baseline, drawn with this round's RNG.

    Two facts matter here and both were learned the expensive way. The interval must come from a
    call whose arm set is exactly ``{baseline, arm}``: the pilot's routine consumes one draw per
    arm in dict order, so a shared row set would make one comparison's numbers depend on how many
    other arms happened to be in the file. And the method stays the archived one -- only the seed
    is this round's, and it is written back into the result so a reader can tell the two apart.
    """
    import evaluate_pilot

    subset = [row for row in rows if str(row["arm"]) in {baseline, arm}]
    previous_seed = evaluate_pilot.BOOTSTRAP_SEED
    evaluate_pilot.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    try:
        result = evaluate_pilot.paired(
            subset, baseline=baseline, extract=lambda row, name=key: float(row[name])
        )
    finally:
        evaluate_pilot.BOOTSTRAP_SEED = previous_seed
    arms = result.get("arms") or {}
    if arm not in arms:
        raise RuntimeError(
            f"no paired interval for {arm} against {baseline} on {key}: {result.get('shared_seeds')} "
            "shared seeds; a missing comparison must not read as a zero"
        )
    return {
        **arms[arm],
        "baseline": baseline,
        "arm": arm,
        "metric": key,
        "unit": "个/episode" if key == "on_time_targets_per_episode" else "B/episode",
        "arms_in_draw": len({str(row["arm"]) for row in subset}),
        "method": "evaluate_pilot.paired (seed-level paired percentile bootstrap)",
        "replications": int(evaluate_pilot.BOOTSTRAP_REPLICATIONS),
        "rng": BOOTSTRAP_SEED,
    }


# ------------------------------------------------------------------ hygiene
def sealed_block_scan() -> dict[str, Any]:
    """The final test block must still not exist on disk in any form."""
    low, high = (int(item) for item in SEALED_BLOCK.split(".."))
    hits: list[str] = []
    for path in (ROOT / "runs").rglob("*"):
        if not path.is_file():
            continue
        digits = "".join(character for character in path.name if character.isdigit())
        try:
            seed = int(digits[:6]) if digits else None
        except ValueError:
            continue
        if seed is not None and low <= seed <= high:
            hits.append(path.relative_to(ROOT).as_posix())
    return {
        "block": SEALED_BLOCK,
        "artefacts_found": sorted(hits)[:10],
        "count": len(hits),
        "clean": not hits,
    }


def production_diff(reference: str = FIX_HEAD) -> dict[str, Any]:
    """This round is a training round: shipped code must not move at all.

    The reference is the mask-fix head rather than the round's own first commit, because the claim
    worth checking is "the model that produced these numbers ran the code that passed M0-M3".
    """
    changed = git_out("diff", "--name-only", reference, "HEAD", "--", "integrated_sar").splitlines()
    runtime = git_out(
        "diff", "--name-only", reference, "HEAD", "--", "piloc", "task_allocation"
    ).splitlines()
    return {
        "reference": reference,
        "integrated_sar_changed": sorted(set(changed)),
        "runtime_modules_changed": sorted(set(runtime)),
        "empty": not changed and not runtime,
    }


def child_environment() -> dict[str, str]:
    """Force UTF-8 on a child: the repository path itself is not representable in the ANSI codepage.

    The first version of this runner asked for ``encoding="utf-8"`` and nothing else. A suite that
    printed one ResourceWarning containing ``D:\\code\\整合\\...`` wrote it in cp936, the parent's
    UTF-8 reader thread died on the ``整`` byte, and the subprocess returned ``stdout=None`` -- which
    read back as "Ran 0 tests, not passed". A gate that fails because it could not hear the answer
    has to say so out loud rather than report a zero, so the child is told to write UTF-8 and an
    unreadable child raises instead of returning an empty report.
    """
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    return environment


def read_child(result: subprocess.CompletedProcess) -> str:
    """The child's combined output, or a failure. Never a silent empty string."""
    if result.stdout is None and result.stderr is None:
        raise RuntimeError(
            f"a child process returned no decodable output at all (returncode "
            f"{result.returncode}); an unreadable test report is not a passing one and not a zero "
            "either"
        )
    return (result.stderr or "") + (result.stdout or "")


def integration_smoke() -> dict[str, Any]:
    """``python -m integrated_sar``, AGENTS' minimum verification, judged on what it printed."""
    result = subprocess.run(
        (sys.executable, "-m", "integrated_sar"),
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_environment(),
    )
    output = read_child(result)
    lines = [line for line in output.splitlines() if line.strip()]
    return {
        "directory": ".",
        "pattern": "python -m integrated_sar",
        "ran": 1,
        "passed": result.returncode == 0 and "completed" in output,
        "skipped": 0,
        "failures_or_errors": result.returncode != 0 or "completed" not in output,
        "returncode": result.returncode,
        "tail": lines[-1] if lines else "",
    }


def run_tests(directory: Path, pattern: str) -> dict[str, Any]:
    """Run one discovery and parse what it reported; a suite is evidence, not a quotation."""
    result = subprocess.run(
        (sys.executable, "-m", "unittest", "discover", "-s", str(directory), "-p", pattern),
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_environment(),
    )
    output = read_child(result)
    ran = _first(output, r"^Ran (\d+) tests")
    skipped = _first(output, r"^OK \(skipped=(\d+)")
    report = {
        "directory": Path(directory).relative_to(ROOT).as_posix(),
        "pattern": pattern,
        "ran": int(ran) if ran else 0,
        "passed": bool(_first(output, r"^OK\b")),
        "skipped": int(skipped) if skipped else 0,
        "failures_or_errors": bool(_first(output, r"^(FAILED|ERROR)")),
        "returncode": result.returncode,
        "tail": output.strip().splitlines()[-1] if output.strip() else "",
    }
    if report["ran"] == 0:
        raise RuntimeError(
            f"discovery in {report['directory']} for {pattern} collected nothing: "
            f"{report['tail'][:400]!r} -- an empty suite is not a pass and must not be reported as "
            "a gate result"
        )
    return report


def _first(text: str, pattern: str) -> str | None:
    import re

    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None
    return match.group(1) if match.groups() else match.group(0)


def main(argv: list[str]) -> int:
    payload = {
        "document": "retrain fixtures",
        "setting": setting(),
        "streams": verify_streams(),
        "executed_modules": executed_modules(),
        "round_scripts": round_scripts(),
        "freeze": freeze(),
        "cleanliness": freeze_cleanliness() if freeze().get("registered") else {"registered": False},
        "sealed_block": sealed_block_scan(),
        "production_diff": production_diff(),
        "launches": ledger_totals(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
