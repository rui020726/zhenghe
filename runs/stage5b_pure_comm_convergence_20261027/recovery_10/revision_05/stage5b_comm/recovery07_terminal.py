"""Strict adapter for recovery_07's writer-native terminal evidence."""
from __future__ import annotations

import re
from pathlib import Path

from . import recovery_runtime as RT

SCHEMA = "recovery-07-monitor-terminal-receipt/1"
ADAPTER = "RECOVERY_07_MONITOR_TERMINAL/1"
ROSTER = ("T1/91001", "T1/91002", "T1/91003", "T2/91001", "T2/91002", "T2/91003")


def _pin(reader, path, root, supplied=None):
    path = Path(path).resolve()
    if supplied is not None:
        RT.need(isinstance(supplied, dict), "recovery_07 evidence pin is malformed")
        reader.raw(path, {key: supplied[key] for key in RT.PIN_KEYS}, root=root)
    else:
        reader.raw(path, root=root)
    return dict(reader.pins[str(path)])


def _writer_run(label):
    arm, seed = label.split("/")
    return "%s_s%s" % (arm, seed)


WRAPPER_ADAPTER = "RECOVERY_08_MONITOR_WRAPPER/1"


def _integral(value, label):
    if type(value) is int and not isinstance(value, bool):
        return value
    if type(value) is float and value.is_integer():
        return int(value)
    raise RT.RecoveryRuntimeError(label + " is not an exact integer")


def normalize_wrapper(root, run_dir, reader):
    """Validate R8's two-field monitor wrapper against its actual terminal files."""
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    monitor_path = run_dir / "monitor_terminal_receipt.json"
    monitor = reader.json(monitor_path, root=root)
    RT.need(set(monitor) == {"observer", "state"},
            "recovery_08 monitor wrapper keys are noncanonical")
    observer = monitor.get("observer") or {}
    RT.need(observer.get("kind") == "TERMINAL"
            and type(observer.get("observed_utc")) is str
            and type(observer.get("poll_count")) is int
            and observer["poll_count"] > 0
            and isinstance(observer.get("poll"), dict),
            "recovery_08 monitor observer is incomplete")
    state = RT.strict_json_bytes(str(monitor.get("state", "")).encode("utf-8"),
                                 "recovery_08 monitor state")
    RT.need(type(state.get("utc")) is str
            and state.get("claims") == state.get("results") == 720
            and state.get("weights") == 6,
            "recovery_08 monitor totals do not close")
    terminal = RT.strict_json_bytes(str(state.get("terminal", "")).encode("utf-8"),
                                    "recovery_08 monitor terminal")
    RT.need(terminal.get("phase") == "recovery-train"
            and type(terminal.get("pid")) is int
            and terminal.get("exit_code") == 0,
            "recovery_08 monitor terminal child exit is not zero")

    outer_path = run_dir / "outer_poll_terminal.json"
    outer_pin = _pin(reader, outer_path, root)
    outer = reader.json(outer_path, {key: outer_pin[key] for key in RT.PIN_KEYS}, root=root)
    RT.need(outer == observer and outer.get("kind") == "TERMINAL"
            and outer.get("poll", {}).get("exit_code") == 0,
            "recovery_08 monitor observer differs from the saved outer terminal poll")
    outer_output = outer["poll"].get("output")
    RT.need(type(outer_output) is str
            and ("CHILD_PID=%d" % terminal["pid"]) in outer_output
            and "CHILD_EXIT=0" in outer_output,
            "recovery_08 outer poll does not close the observed child")

    child_json_path = run_dir / "child_exit.json"
    child_json_pin = _pin(reader, child_json_path, root)
    child_json = reader.json(child_json_path,
                             {key: child_json_pin[key] for key in RT.PIN_KEYS}, root=root)
    RT.need(child_json.get("phase") == "recovery-train"
            and child_json.get("pid") == terminal["pid"]
            and child_json.get("exit_code") == 0,
            "recovery_08 child exit record disagrees with the monitor")
    child_text_path = run_dir / "child_exit.txt"
    child_text_pin = _pin(reader, child_text_path, root)
    child_text = reader.raw(child_text_path,
                            {key: child_text_pin[key] for key in RT.PIN_KEYS}, root=root)
    RT.need(child_text.decode("ascii").strip() == "0",
            "recovery_08 child exit is not zero")

    started_path = run_dir / "started.json"
    started_pin = _pin(reader, started_path, root)
    started = reader.json(started_path,
                          {key: started_pin[key] for key in RT.PIN_KEYS}, root=root)
    RT.need(started.get("phase") == "recovery-train"
            and started.get("pid") == terminal["pid"]
            and Path(started.get("cwd", "")).resolve() == root
            and started.get("workers") == 3,
            "recovery_08 started record does not match the terminal process")

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    stdout_pin = _pin(reader, stdout_path, root)
    stderr_pin = _pin(reader, stderr_path, root)
    RT.need(stdout_pin["bytes"] == state.get("stdout_bytes")
            and stderr_pin["bytes"] == state.get("stderr_bytes"),
            "recovery_08 monitor stream byte counts are stale")
    stdout = reader.raw(stdout_path,
                        {key: stdout_pin[key] for key in RT.PIN_KEYS}, root=root)
    cli_lines = [line for line in stdout.decode("utf-8").replace("\r", "").splitlines()
                 if line.startswith('{"phase": "recovery-train"')]
    RT.need(len(cli_lines) == 1, "recovery_08 needs one final CLI receipt line")
    final = RT.strict_json_bytes(cli_lines[0].encode("utf-8"),
                                 "recovery_08 final CLI receipt")
    receipt_id = final.get("receipt_id")
    RT.need(final.get("phase") == "recovery-train" and final.get("returncode") == 0
            and final.get("workers") == 3 and type(receipt_id) is str
            and receipt_id.startswith("recovery-train-"),
            "recovery_08 final CLI receipt is invalid")
    receipt_path = root / "execution/receipts" / (
        "recovery-train_terminal_%s.json" % receipt_id)
    receipt_pin = _pin(reader, receipt_path, root)

    runs = state.get("runs")
    RT.need(type(runs) is list and len(runs) == len(ROSTER),
            "recovery_08 monitor run roster is incomplete")
    by_label = {}
    for item in runs:
        RT.need(isinstance(item, dict) and set(item) == {
            "run", "results", "last_episode"},
            "recovery_08 monitor run entry is malformed")
        label = item.get("run")
        RT.need(label in ROSTER and label not in by_label,
                "recovery_08 monitor run label is unknown/duplicated")
        RT.need(item.get("results") == 120,
                "recovery_08 monitor run did not close 120 results")
        by_label[label] = item
    RT.need(set(by_label) == set(ROSTER),
            "recovery_08 monitor run roster does not close")

    checkpoints, certificates = [], []
    for label in ROSTER:
        last = _integral(by_label[label]["last_episode"],
                         "recovery_08 last episode")
        episode = last + 1
        writer = _writer_run(label)
        cp_path = root / "execution/checkpoints" / writer / ("milestone_%d.pth" % episode)
        cp_pin = _pin(reader, cp_path, root)
        checkpoints.append({**cp_pin, "run": writer, "episode": episode})
        cert_path = root / "execution/recovery/certificates" / (
            "CERTIFICATE_%s_s%s.json" % tuple(label.split("/")))
        cert_pin = _pin(reader, cert_path, root)
        cert = reader.json(cert_path,
                           {key: cert_pin[key] for key in RT.PIN_KEYS}, root=root)
        RT.need(cert.get("status") in ("CONVERGED", "NOT_YET_CONVERGED")
                and cert.get("terminal_episode") == episode,
                "recovery_08 monitor certificate disagrees with its checkpoint")
        certificates.append({**cert_pin, "run": writer})

    evidence = [outer_pin, child_json_pin, child_text_pin, started_pin,
                stdout_pin, stderr_pin, receipt_pin]
    return {
        "schema": "stage5b-recovery-stage-terminal-summary/1",
        "root": str(root), "stage_complete": True, "converged": False,
        "result_count": 720, "update_journal_count": 180,
        "result_status_counts": {"ok": 720},
        "claims": 720, "results": 720, "update_journals": 180,
        "child_exit": {**child_text_pin, "value": "0"},
        "outer_terminal": {
            **outer_pin, "exit_code": outer["poll"]["exit_code"],
            "output": outer_output,
            "wall_time_seconds": outer["poll"].get("wall_time_seconds"),
        },
        "started": started_pin,
        "stop_or_terminal_named_files": evidence,
        "checkpoints": checkpoints, "certificates": certificates,
        "adapter": WRAPPER_ADAPTER,
    }, outer_path.name, monitor_path


def normalize(root, run_dir, reader):
    """Translate only the exact R7 terminal schema into the shared terminal checks."""
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    monitor_path = run_dir / "monitor_terminal_receipt.json"
    monitor = reader.json(monitor_path, root=root)
    if set(monitor) == {"observer", "state"} and monitor.get("schema") is None:
        return normalize_wrapper(root, run_dir, reader)
    RT.need(monitor.get("schema") == SCHEMA
            and monitor.get("terminal_state") == "CHILD_AND_OUTER_EXIT_ZERO"
            and monitor.get("stage_assessment") == "TERMINAL_SUCCESS_STAGE_COMPLETE_NOT_CONVERGED"
            and monitor.get("six_certificates_all_not_yet_converged") is True
            and monitor.get("eval_started") is False
            and monitor.get("validation_holdout_sealed") is True,
            "recovery_07 monitor terminal status is incomplete/noncanonical")
    results = monitor.get("results") or {}
    claims = monitor.get("claims") or {}
    updates = monitor.get("updates") or {}
    RT.need(results.get("count") == results.get("cap") == 720
            and claims.get("count") == 720 and updates.get("count") == 180,
            "recovery_07 monitor totals do not close at 720/720/180")
    runs = monitor.get("runs")
    RT.need(type(runs) is dict and tuple(sorted(runs)) == tuple(sorted(ROSTER)),
            "recovery_07 monitor run roster is incomplete")

    evidence = {}
    raw_evidence = monitor.get("evidence_files")
    RT.need(type(raw_evidence) is list and raw_evidence, "recovery_07 evidence inventory missing")
    for item in raw_evidence:
        pin = _pin(reader, item.get("path", ""), root, item)
        name = Path(pin["path"]).name
        RT.need(name not in evidence, "recovery_07 evidence basename is duplicated")
        evidence[name] = pin
    for required in ("outer_poll_208.json", "child.exit.txt", "started.json"):
        RT.need(required in evidence, "recovery_07 evidence missing " + required)

    outer_pin = evidence["outer_poll_208.json"]
    outer_path = Path(outer_pin["path"]).resolve()
    outer = reader.json(outer_path, {key: outer_pin[key] for key in RT.PIN_KEYS}, root=root)
    RT.need(type(outer.get("exit_code")) is int and outer["exit_code"] == 0
            and outer.get("output") == "", "recovery_07 final outer poll is not a clean exit zero")

    child_pin = evidence["child.exit.txt"]
    child_path = Path(child_pin["path"]).resolve()
    child = reader.raw(child_path, {key: child_pin[key] for key in RT.PIN_KEYS}, root=root)
    RT.need(child.decode("ascii").strip() == "0", "recovery_07 child exit is not zero")

    started_pin = evidence["started.json"]
    started_path = Path(started_pin["path"]).resolve()
    RT.need(started_path == run_dir / "started.json",
            "recovery_07 started record is not at its writer path")

    final = monitor.get("final_cli_receipt") or {}
    receipt_id = final.get("receipt_id")
    RT.need(final.get("phase") == "recovery-train" and final.get("returncode") == 0
            and final.get("workers") == 3 and type(receipt_id) is str
            and receipt_id.startswith("recovery-train-"),
            "recovery_07 final CLI receipt marker is invalid")
    receipt_path = root / "execution/receipts" / ("recovery-train_terminal_%s.json" % receipt_id)
    receipt = reader.json(receipt_path, root=root)
    RT.need(receipt.get("schema") == "stage5b-phase-receipt/1"
            and receipt.get("phase") == "recovery-train"
            and receipt.get("stage") == "terminal"
            and receipt.get("receipt_id") == receipt_id
            and receipt.get("returncode") == 0,
            "recovery_07 terminal receipt is not successful training")
    detail = receipt.get("detail") or {}
    RT.need(detail.get("status") == "FINITE_RECOVERY_STAGE_COMPLETE"
            and detail.get("worlds") == 720
            and detail.get("convergence_complete") is False
            and detail.get("slot_counts") == {
                "claimed": 720, "ok": 720, "planned": 720,
                "remaining": 0, "results": 720}
            and len(detail.get("results") or []) == 6,
            "recovery_07 terminal detail does not close")
    detail_by_run = {}
    for result in detail["results"]:
        slot = str(result.get("slot", ""))
        match = re.fullmatch(r"recovery:(T[12]):(9100[1-3])", slot)
        RT.need(match is not None, "recovery_07 terminal result slot is invalid")
        label = "%s/%s" % (match.group(1), match.group(2))
        RT.need(label not in detail_by_run, "recovery_07 terminal result is duplicated")
        detail_by_run[label] = result
    RT.need(tuple(sorted(detail_by_run)) == tuple(sorted(ROSTER)),
            "recovery_07 terminal result roster is incomplete")

    checkpoints, certificates = [], []
    for label in ROSTER:
        item = runs[label]
        observed = detail_by_run[label]
        run_result = item.get("results") or {}
        first, last = run_result.get("first_episode"), run_result.get("last_episode")
        RT.need(type(first) is int and type(last) is int and last - first + 1 == 120
                and observed.get("episodes") == 120
                and observed.get("terminal_episode") == last + 1
                and observed.get("status") == "ok"
                and observed.get("converged") is False
                and item.get("claims") == 120 and item.get("updates") == 30,
                "recovery_07 run totals/cursor do not close: " + label)
        writer = _writer_run(label)
        cp = item.get("checkpoint") or {}
        cp_path = Path(cp.get("path", "")).resolve()
        RT.need(cp_path == root / "execution/checkpoints" / writer / ("milestone_%d.pth" % (last + 1)),
                "recovery_07 checkpoint writer path/cursor mismatch: " + label)
        cp_pin = _pin(reader, cp_path, root, cp)
        checkpoints.append({**cp_pin, "run": writer, "episode": last + 1})
        cert = item.get("certificate") or {}
        cert_path = Path(cert.get("path", "")).resolve()
        RT.need(cert_path == root / "execution/recovery/certificates" /
                ("CERTIFICATE_%s_s%s.json" % tuple(label.split("/"))),
                "recovery_07 certificate writer path mismatch: " + label)
        RT.need(cert.get("status") == "NOT_YET_CONVERGED"
                and cert.get("terminal_episode") == last + 1,
                "recovery_07 certificate is not the observed nonconverged endpoint")
        cert_pin = _pin(reader, cert_path, root, cert)
        certificates.append({**cert_pin, "run": writer})

    receipt_pin = _pin(reader, receipt_path, root)
    stop_pins = list(evidence.values()) + [receipt_pin]
    summary = {
        "schema": "stage5b-recovery-stage-terminal-summary/1",
        "root": str(root), "stage_complete": True, "converged": False,
        "result_count": 720, "update_journal_count": 180,
        "result_status_counts": {"ok": 720},
        "claims": 720, "results": 720, "update_journals": 180,
        "child_exit": {**child_pin, "value": "0"},
        "outer_terminal": {**outer_pin, "exit_code": 0, "output": outer.get("output"),
                           "wall_time_seconds": outer.get("wall_time_seconds")},
        "started": started_pin,
        "stop_or_terminal_named_files": stop_pins,
        "checkpoints": checkpoints, "certificates": certificates,
        "adapter": ADAPTER,
    }
    return summary, outer_path.name, monitor_path