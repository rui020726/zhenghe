"""ROUND66 detached phase launcher: runs one R66 phase module as an auditable, heartbeat-reporting process.

Usage:
    _r66_launch.py <phase> [seed] [gate] [--module execute|eval_fixed]

<phase> is forwarded verbatim to the chosen module's --phase argument, so the same launcher drives
r66_execute.py (acceptance/train/eval/stats/all) and the private corrected evaluation adapter
r66_eval_fixed.py (plan/eval/stats). --module defaults to "execute", so every historical invocation
keeps exactly the same child command; only the log file name is now module-prefixed so two phases
cannot overwrite each other's log.

Records pid/command/log/exit code under raw/runtime/launch_records.jsonl and prints a heartbeat line
every 60 seconds so a supervising session can tell the run is alive without touching the science.
"""
import os, sys, json, time, subprocess, threading
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
R66 = None
for root, dirs, files in os.walk(r"D:\code"):
    dirs[:] = [d for d in dirs if d != ".git"]
    if os.path.basename(root) == "round66_bc_early_path_preserving_late_adaptation":
        R66 = root; break
PY = r"D:\project_env\python.exe"
MODULES = {"execute": "r66_execute.py", "eval_fixed": "r66_eval_fixed.py"}
argv = list(sys.argv[1:])
module_key = "execute"
if "--module" in argv:
    i = argv.index("--module")
    if i + 1 >= len(argv) or argv[i + 1] not in MODULES:
        raise SystemExit("--module must be one of " + ", ".join(sorted(MODULES)))
    module_key = argv[i + 1]
    del argv[i:i + 2]
if not argv:
    raise SystemExit(__doc__)
phase = argv[0]
seed = argv[1] if len(argv) > 1 and argv[1] not in ("", "-") else None
gate_only = len(argv) > 2 and argv[2] == "gate"
script = os.path.join(R66, MODULES[module_key])
cmd = [PY, "-u", script, "--raw-root", os.path.join(R66, "raw"), "--phase", phase]
if seed:
    cmd += ["--seed", seed]
if gate_only:
    cmd += ["--first-gate-only"]
tag = module_key + "_" + phase + (f"_s{seed}" if seed else "") + ("_gate" if gate_only else "")
log = os.path.join(R66, f"_r66_{tag}.log")
rt = os.path.join(R66, "raw", "runtime")
os.makedirs(rt, exist_ok=True)
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
rec = {"kind": "round66-launch-record/1", "launcher_pid": os.getpid(), "module": module_key,
       "script": script, "cmd": cmd, "log": log, "started_unix": time.time(),
       "phase": phase, "seed": seed, "first_gate_only": gate_only, "cwd": R66}
with open(os.path.join(rt, "launch_records.jsonl"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec) + "\n")
stop = threading.Event()
def heartbeat():
    while not stop.is_set():
        time.sleep(60)
        if stop.is_set():
            return
        try:
            n_slots = len(os.listdir(os.path.join(R66, "raw", "launch_slots", "normal")))
        except OSError:
            n_slots = -1
        print(f"HEARTBEAT unix={time.time():.0f} phase={phase} module={module_key} slots={n_slots}", flush=True)
hb = threading.Thread(target=heartbeat, daemon=True)
hb.start()
with open(log, "w", encoding="utf-8") as out:
    out.write("LAUNCH " + json.dumps(rec) + "\n")
    out.flush()
    rc = subprocess.call(cmd, cwd=R66, stdout=out, stderr=subprocess.STDOUT, env=env)
stop.set()
with open(os.path.join(rt, "launch_records.jsonl"), "a", encoding="utf-8") as fh:
    fh.write(json.dumps({**rec, "exit_code": rc, "finished_unix": time.time()}) + "\n")
print(f"LAUNCHER_DONE phase={phase} module={module_key} exit_code={rc}", flush=True)
sys.exit(rc)
