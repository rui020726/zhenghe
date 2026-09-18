"""Record the held-out detection streams (brief section four, step one).

One live PILOC pass per seed at 1000 steps with the coordinator off, writing the ordered
DetectionEvent stream -- the same recording protocol the 20260917 held-out block used,
pinned here by checkpoint digest and configuration hash. Both replay arms then read the
same file per seed, so the experiment measures allocation given identical search.

    D:/project_env/python.exe runs/coalition_validation_20260918/heldout_record.py
    ... heldout_record.py --worker 2 --workers 8
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from integrated_sar.joint_driver import RunConfiguration, build_joint_run  # noqa: E402

STREAMS = HERE / "streams"
INDEX = HERE / "stream_index.json"
PIL_CHECKPOINT = REPO / "runs" / "cur_v2rep05_s2" / "best" / "shared_mappo_model_21_3100.pth"
CTDE_CHECKPOINT = REPO / "runs" / "joint_eval_20260916" / "ctde_final_7_1.pth"
STEPS = 1000
WORKERS = 8


def heldout_seeds() -> list[int]:
    return [int(seed) for seed in json.loads(
        (HERE / "HELDOUT_SEEDS.json").read_text(encoding="utf-8")
    )["seeds"]]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_one(seed: int) -> dict:
    STREAMS.mkdir(parents=True, exist_ok=True)
    (HERE / "record_logs").mkdir(parents=True, exist_ok=True)
    stream = STREAMS / f"s{seed}.json"
    configuration = RunConfiguration(
        seed=seed,
        steps=STEPS,
        device="cpu",
        output=str(HERE / "record_logs" / f"rec_s{seed}.jsonl"),
        piloc_checkpoint=str(PIL_CHECKPOINT),
        task_checkpoint=str(CTDE_CHECKPOINT),
        mock=False,
        use_coordinator=False,
        search_agents=2,
        num_goals=6,
        grid_size=60,
        obstacle_density=0.15,
        detection_radius=5,
        dt=1.0,
        record_stream=str(stream),
        quiet=True,
    )
    run = build_joint_run(configuration)
    run.run()
    payload = json.loads(stream.read_text(encoding="utf-8"))
    return {
        "seed": seed,
        "stream": str(stream.relative_to(REPO)).replace("\\", "/"),
        "sha256": sha256(stream),
        "frames": len(payload.get("frames", [])),
        "detections": sum(len(frame.get("detections", [])) for frame in payload.get("frames", [])),
        "map_sha256": payload.get("map_sha256"),
    }


def main(argv: list[str]) -> int:
    seeds = heldout_seeds()
    if "--worker" in argv:
        index = int(argv[argv.index("--worker") + 1])
        total = int(argv[argv.index("--workers") + 1])
        own_path = HERE / f"stream_index_w{index}.json"
        own = json.loads(own_path.read_text(encoding="utf-8")) if own_path.exists() else {}
        for position, seed in enumerate(seeds):
            if position % total != index or str(seed) in own:
                continue
            own[str(seed)] = record_one(seed)
            # Each worker owns one index file, so no two writers touch the same bytes;
            # the aggregate step merges them.
            part = own_path.with_suffix(".json.part")
            part.write_text(json.dumps(own, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            part.rename(own_path)
            print(f"[worker {index}] s{seed}: {own[str(seed)]['detections']} detections", flush=True)
        return 0

    if "--aggregate" not in argv:
        processes = [
            subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker", str(i), "--workers", str(WORKERS)],
                cwd=str(REPO),
            )
            for i in range(WORKERS)
        ]
        codes = [p.wait() for p in processes]
        if any(codes):
            print(f"worker failures: {codes}")
            return 1

    index: dict[str, dict] = {}
    for path in sorted(HERE.glob("stream_index_w*.json")):
        index.update(json.loads(path.read_text(encoding="utf-8")))
    missing = [seed for seed in seeds if str(seed) not in index]
    print(f"recorded {len(index)}/{len(seeds)}, missing {len(missing)}")
    if missing:
        return 1
    collisions = len(index) - len({row["stream"] for row in index.values()})
    print(f"stream path collisions: {collisions}")
    zero = [seed for seed in seeds if index[str(seed)]["detections"] == 0]
    print(f"streams with zero detections: {len(zero)} {zero[:10]}")
    print(f"piloc checkpoint sha256: {sha256(PIL_CHECKPOINT)}")
    print(f"ctde checkpoint sha256: {sha256(CTDE_CHECKPOINT)}")
    (HERE / "stream_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
