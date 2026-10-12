"""ROUND65 input sources + identity chain (read-only). Emits INPUT_SOURCES.json."""
from __future__ import annotations
import argparse, json, sys, hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "round60_partitioned_actor_head_ablation"))
import r65_audit as A
import r60_partitioned as r60

RUN = HERE.parent
R60 = RUN / "round60_partitioned_actor_head_ablation"
R64 = RUN / "round64_controlled_deadline_intervention"
CK = R60 / "raw" / "checkpoints"
SEEDS = [11, 22, 33]
MILES = [("ep0", "before_b0"), ("ep64", "after_b15"), ("ep128", "ep128")]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(HERE))
    out = Path(ap.parse_args().out)

    focal32, focal_bytes, prov = A.parse_focal()
    index, batches, problems = A.build_index()

    binding = r60.configure_source_imports()
    import v2_controller as vc, v2_trainer as vt, v2_common as vcm
    canonical = {
        "v2_controller_file": getattr(vc, "__file__", None),
        "v2_trainer_file": getattr(vt, "__file__", None),
        "v2_common_file": getattr(vcm, "__file__", None),
        "v2_trainer_sha256": r60.EXPECTED_TRAINER_SHA256,
        "v2_common_sha256": r60.EXPECTED_COMMON_SHA256,
        "v2_controller_sha256": r60.EXPECTED_CONTROLLER_SHA256,
        "canon_dir": str(r60.CANON),
    }

    chain = []
    for seed in SEEDS:
        idn = json.load(open(CK / f"r60_ep0_identity_s{seed}.json", encoding="utf-8"))
        for milestone, tag in MILES:
            path = CK / f"v2_train_s{seed}_{tag}.pt"
            if milestone == "ep0":
                exp, src = idn["parameter_sha256"], "r60_ep0_identity (R58-ep0-derived NEW init)"
            elif milestone == "ep64":
                exp, src = batches[(seed, 15)]["parameters_after_sha"], "batch b15 parameters_after_sha"
            else:
                exp, src = batches[(seed, 31)]["parameters_after_sha"], "batch b31 parameters_after_sha"
            chain.append({"seed": seed, "milestone": milestone, "file": path.name,
                          "file_sha256": sha(path) if path.exists() else None,
                          "expected_param_sha256": exp, "expected_source": src})

    srcs = {
        "kind": "round65-input-sources/1",
        "repo_head_at_audit": None,  # filled by caller/git, not guessed here
        "compliant_source_index": {
            "description": "96 attempt01 batch records are the compliant source of truth; "
                           "episodes/scenes come from the records, never from globbing filenames. "
                           "R60 isolated non-compliant constructions are excluded.",
            "n_batches": len(batches), "n_episodes": len(index), "index_problems": problems,
            "per_seed_episodes": {str(s): sum(1 for (sd, _) in index if sd == s) for s in SEEDS},
        },
        "focal": prov,
        "canonical_modules": canonical,
        "checkpoint_identity_chain": chain,
        "key_source_artifacts": {
            "r64_focal_tape": {"path": str(A.FOCAL_TAPE), "sha256": sha(A.FOCAL_TAPE)},
            "r60_batch_record_example_s33_b31": {
                "path": str(CK / "r40_batch_s33_b31_attempt01.json"),
                "sha256": sha(CK / "r40_batch_s33_b31_attempt01.json")},
            "r60_ep0_identity_s33": {"path": str(CK / "r60_ep0_identity_s33.json"),
                                     "sha256": sha(CK / "r60_ep0_identity_s33.json")},
        },
        "notes": [
            "Raw tapes, rows, model .pt weights and original logs are NOT added to Git; only "
            "summaries, key examples and path SHAs are recorded.",
            "The focal 19-D float32 is parsed from the R64 L tape and validated against the "
            "r65_common constant; the parsed value is authoritative and matching aborts on drift.",
        ],
    }
    json.dump(srcs, open(out / "INPUT_SOURCES.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote INPUT_SOURCES.json; index problems:", problems,
          "| chain entries:", len(chain))


if __name__ == "__main__":
    main()
