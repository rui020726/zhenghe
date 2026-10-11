"""Read-only extraction of R60 seed33 greedy early first-report candidates from the saved eval
tapes (no environment, no model forward). A candidate is a real discovered target whose baseline
first-report was a free, legal, payable one-shot choice with ready_step = round(f18*1000)-round(f0)
<= 147."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
R60 = HERE.parent / "round60_partitioned_actor_head_ablation" / "raw"
LOGS = R60 / "logs"
SCENES = list(range(1700000, 1700032))

def ready(f):
    return int(round(float(f[18]) * 1000.0) - round(float(f[0])))

cands = []
for scene in SCENES:
    tapes = sorted(LOGS.glob(f"v2_r60_eval_new_s33_greedy_s{scene}_a*_tape.jsonl"))
    if not tapes:
        raise SystemExit(f"no seed33 greedy eval tape for scene {scene}")
    tape_path = tapes[0]
    seen_targets = set()
    for line in tape_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        tape = json.loads(line)
        targets = tape["targets"]
        feats = tape["features"]
        actions = tape["actions"]
        free = set(tape.get("free_one_shot_choices", []))
        for i, t in enumerate(targets):
            if t in seen_targets or t not in free:
                continue
            f = feats[i]
            if float(f[5]) != 1.0:  # first_emission only
                continue
            r = ready(f)
            if r > 147:
                continue
            seen_targets.add(t)
            cands.append({"scene": scene, "target": t, "ready": r,
                          "action": "SEND" if int(actions[t]) == 1 else "SKIP",
                          "features": [float(x) for x in f],
                          "tape": str(tape_path), "decision_step": int(tape["step"])})

print("total early (ready<=147) free first-report candidates:", len(cands))
scenes_with = sorted({c["scene"] for c in cands})
print("distinct scenes:", len(scenes_with))
focus = [c for c in cands if c["scene"] == 1700027 and c["target"] == "piloc-e1-target-5"]
print("focus (1700027/target-5):", [(c["ready"], c["action"], c["decision_step"]) for c in focus])
sends = [c for c in cands if c["action"] == "SEND"]
skips = [c for c in cands if c["action"] == "SKIP"]
print("SEND candidates:", len(sends), "across scenes", len({c['scene'] for c in sends}))
print("SKIP candidates:", len(skips), "across scenes", len({c['scene'] for c in skips}))
# per-scene candidate counts
from collections import Counter
print("per-scene candidate counts:", dict(sorted(Counter(c['scene'] for c in cands).items())))
Path(HERE / "_candidates.json").write_text(json.dumps(cands, ensure_ascii=False), encoding="utf-8")
print("wrote _candidates.json")
