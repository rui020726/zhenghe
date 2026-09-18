"""Record the stress streams, one condition configuration per scenario seed.

One live PILOC pass per seed with the coordinator off, writing the ordered DetectionEvent
stream, exactly as the 400-seed block recorded its own. Each condition records under its
own band and goal count, and both replay arms of that condition read the same file per
seed, so the experiment measures allocation given identical search.

    D:/project_env/python.exe runs/release_validation_20260918/stress_record.py
    ... stress_record.py --condition A_tighter_deadline --worker 0 --workers 8
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
try:  # applied before any checkpoint load
    import torch

    torch.set_num_threads(1)
except Exception:  # torch absent only in dependency-free environments, where it is moot
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402
from stress_replay import CONDITIONS, STEPS  # noqa: E402

PIL_CHECKPOINT = REPO / "runs" / "cur_v2rep05_s2" / "best" / "shared_mappo_model_21_3100.pth"
CTDE_CHECKPOINT = REPO / "runs" / "joint_eval_20260916" / "ctde_final_7_1.pth"
WORKERS = 8


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seeds_for(condition: str) -> list[int]:
    payload = json.loads((HERE / "STRESS_SEEDS.json").read_text(encoding="utf-8"))
    return [int(seed) for seed in payload["conditions"][condition]]


def record_one(condition: str, seed: int) -> dict:
    directory = HERE / "stress_streams" / condition
    directory.mkdir(parents=True, exist_ok=True)
    (HERE / "stress_record_logs").mkdir(parents=True, exist_ok=True)
    stream = directory / f"s{seed}.json"
    scenario = CONDITIONS[condition]
    configuration = RunConfiguration(
        seed=seed,
        steps=STEPS,
        device="cpu",
        output=str(HERE / "stress_record_logs" / f"{condition}_s{seed}.jsonl"),
        piloc_checkpoint=str(PIL_CHECKPOINT),
        task_checkpoint=str(CTDE_CHECKPOINT),
        mock=False,
        use_coordinator=False,
        search_agents=2,
        num_goals=scenario["num_goals"],
        deadline_tightness=tuple(scenario["deadline_tightness"]),
        grid_size=60,
        obstacle_density=0.15,
        detection_radius=5,
        dt=1.0,
        record_stream=str(stream),
        quiet=True,
    )
    build_joint_run(configuration).run()
    payload = json.loads(stream.read_text(encoding="utf-8"))
    return {
        "seed": seed,
        "condition": condition,
        "stream": str(stream.relative_to(REPO)).replace("\\", "/"),
        "sha256": sha256(stream),
        "frames": len(payload.get("frames", [])),
        "detections": sum(len(frame.get("detections", [])) for frame in payload.get("frames", [])),
        "map_sha256": payload.get("map_sha256"),
    }


def index_path(condition: str, worker: int | None) -> Path:
    name = f"index_w{worker}.json" if worker is not None else "index.json"
    return HERE / "stress_streams" / condition / name


def aggregate(condition: str) -> int:
    seeds = seeds_for(condition)
    index: dict[str, dict] = {}
    for path in sorted((HERE / "stress_streams" / condition).glob("index_w*.json")):
        index.update(json.loads(path.read_text(encoding="utf-8")))
    missing = [seed for seed in seeds if str(seed) not in index]
    zero = [seed for seed in seeds if index.get(str(seed), {}).get("detections") == 0]
    print(
        f"[{condition}] recorded {len(index)}/{len(seeds)}, missing {len(missing)}, "
        f"zero-detection {len(zero)} {zero[:6]}"
    )
    if missing:
        return 1
    index_path(condition, None).write_text(
        json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return 0


def main(argv: list[str]) -> int:
    condition = argv[argv.index("--condition") + 1] if "--condition" in argv else None
    if condition is None:
        codes = []
        for name in CONDITIONS:
            codes.append(
                subprocess.call(
                    [sys.executable, str(Path(__file__).resolve()), "--condition", name] + argv[1:],
                    cwd=str(REPO),
                )
            )
        return 1 if any(codes) else 0

    if "--aggregate" in argv:
        return aggregate(condition)

    seeds = seeds_for(condition)
    if "--worker" in argv:
        index = int(argv[argv.index("--worker") + 1])
        total = int(argv[argv.index("--workers") + 1])
        own_path = index_path(condition, index)
        own = json.loads(own_path.read_text(encoding="utf-8")) if own_path.exists() else {}
        for position, seed in enumerate(seeds):
            if position % total != index or str(seed) in own:
                continue
            own[str(seed)] = record_one(condition, seed)
            part = own_path.with_suffix(".json.part")
            part.write_text(json.dumps(own, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            os.replace(part, own_path)
            print(f"[{condition} worker {index}] s{seed}: {own[str(seed)]['detections']} detections", flush=True)
        return 0

    processes = [
        subprocess.Popen(
            [
                sys.executable, str(Path(__file__).resolve()), "--condition", condition,
                "--worker", str(i), "--workers", str(WORKERS),
            ],
            cwd=str(REPO),
        )
        for i in range(WORKERS)
    ]
    codes = [process.wait() for process in processes]
    if any(codes):
        print(f"worker failures: {codes}")
        return 1
    return aggregate(condition)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
