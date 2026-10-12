#!/usr/bin/env python
"""ROUND68 read-only checkpoint index.

Loads each seed's fixed ep128 checkpoint READ-ONLY (no optimizer restore, no training-RNG restore,
no training-cursor restore, no environment construction) and records its identity: file SHA,
parameter SHA, per-module SHAs, the frozen-mask block, the six-parameter optimizer ownership, the
Adam state, and the training-state cursor. Nothing is written back and no training source is
modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r68_execute as base   # noqa: E402
import r68_frozen as r68     # noqa: E402
import r66_frozen as r66f    # noqa: E402

SEEDS = (11, 22, 33)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def index_seed(seed: int, ck: Path, freeze: dict) -> dict:
    path = ck / f"v2_train_s{seed}_ep128.pt"
    controller, state, ident = r68.load_checkpoint_read_only(path, freeze=freeze)
    summary = controller.parameter_summary()
    # _module_sha takes the CONTROLLER-side names. The cost critic is registered as `cost_head`
    # (controller.cost_head) while the serialised state_dict calls the same object
    # `value_cost_head`; the task value head is controller.inner.value_head but is addressed as
    # `value_task_head`. Both spellings are recorded so the naming difference is explicit rather
    # than papered over, and a genuine failure is recorded as the real exception, never as 0.
    modules = {}
    for name in ("trunk", "early_head", "late_head", "value_task_head", "cost_head"):
        try:
            modules[name] = r68._module_sha(controller, name)
        except Exception as exc:
            modules[name] = {"unavailable": type(exc).__name__, "detail": str(exc)}
    frozen_params, trainable_params = r66f.partitioned_parameters(controller)
    frozen_ident = r66f._tensor_identity(frozen_params)
    trainable_ident = r66f._tensor_identity(trainable_params)

    # Adam state and param-group ownership are read from a separate read-only CPU load of the
    # stored optimizer payload; the read-only loader above deliberately never restores them.
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    opt = payload.get("optimizer", {})
    param_groups = opt.get("param_groups", [])
    state_map = opt.get("state", {})
    adam_entries = {}
    for group in param_groups:
        for pid in group.get("params", []):
            st = state_map.get(str(pid))
            if st is None:
                st = state_map.get(pid)
            st = st or {}
            adam_entries[str(pid)] = {
                "step": int(st.get("step", 0)) if st.get("step") is not None else None,
                "has_exp_avg": "exp_avg" in st,
                "has_exp_avg_sq": "exp_avg_sq" in st,
            }
    mask = ident.get("frozen_mask", {}) if isinstance(ident, dict) else {}
    ts = state
    lam = ts.get("lambda_history") or []
    lam_vals = [float(e["lambda_next"]) for e in lam
                if isinstance(e, dict) and e.get("lambda_next") is not None]
    return {
        "seed": seed,
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "file_bytes": path.stat().st_size,
        "checkpoint_identity": {"load_mode": ident.get("load_mode"),
                                "optimizer_restored": ident.get("optimizer_restored"),
                                "rng_restored": ident.get("rng_restored"),
                                "training_version": ident.get("training_version"),
                                "reward_timing_version": ident.get("reward_timing_version"),
                                "structure": ident.get("structure"),
                                "late_init_alpha": ident.get("late_init_alpha")},
        "parameter_sha256": summary["parameter_sha256"],
        "parameter_object_count": summary.get("tensor_count"),
        "modules": modules,
        "module_naming_note": {
            "controller_side_names_used": ["trunk", "early_head", "late_head",
                                           "value_task_head", "cost_head"],
            "cost_critic_controller_path": "controller.cost_head",
            "cost_critic_state_dict_key": "value_cost_head",
            "task_value_head_controller_path": "controller.inner.value_head",
            "task_value_head__module_sha_name": "value_task_head",
            "note": ("the cost critic is addressed as `cost_head` by r68_frozen._module_sha but "
                     "serialised as `value_cost_head` in the checkpoint state_dict; they are the "
                     "same object. An earlier draft of this index used `value_cost_head` as the "
                     "_module_sha name and recorded the cost critic as unavailable; that was a "
                     "name error on the reader side, not a missing parameter, and is corrected "
                     "here rather than filled with 0."),
        },
        "state_dict_keys": sorted(controller.state_dict().keys()),
        "frozen_mask": {
            "frozen_modules": mask.get("frozen_modules"),
            "trainable_modules": mask.get("trainable_modules"),
            "frozen_sha256": mask.get("frozen_sha256"),
            "frozen_scalar_count": mask.get("frozen_scalar_count"),
            "frozen_parameter_object_count": mask.get("frozen_parameter_object_count"),
            "late_init_alpha": mask.get("late_init_alpha"),
            "late_head_scaled_sha256": mask.get("late_head_scaled_sha256"),
        },
        "frozen_identity": {"sha256": frozen_ident["sha256"], "tensor_count": frozen_ident["tensor_count"],
                            "scalar_count": frozen_ident.get("scalar_count")},
        "trainable_identity": {"sha256": trainable_ident["sha256"],
                               "tensor_count": trainable_ident["tensor_count"],
                               "scalar_count": trainable_ident.get("scalar_count")},
        "optimizer_param_groups": payload.get("optimizer_param_groups"),
        "optimizer_owns_param_count": sum(len(g.get("params", [])) for g in param_groups),
        "adam_entries": adam_entries,
        "training_state": {
            "episodes_done": ts.get("episodes_done"),
            "updates": ts.get("updates"),
            "batches": ts.get("batches"),
            "training_version": ts.get("training_version"),
            "late_init_alpha": ts.get("late_init_alpha"),
            "lambda_current": ts.get("lambda"),
            "lambda_history_len": len(lam),
            "lambda_history_final": (lam_vals[-1] if lam_vals else None),
            "lambda_history_monotone_nondecreasing": (
                all(lam_vals[i] <= lam_vals[i + 1] for i in range(len(lam_vals) - 1))
                if len(lam_vals) > 1 else None),
            "lambda_history_first3": lam[:3],
            "optimizer_steps": ts.get("optimizer_steps"),
            "training_seed": ts.get("training_seed"),
        },
        "has_rng_block": isinstance(payload.get("rng"), dict),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ROUND68 read-only checkpoint index")
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    raw = Path(args.raw_root).resolve()
    ck = raw / "checkpoints"
    base.configure_and_inject(raw)
    out = {"kind": "round68-checkpoint-index/1",
           "task_id": base.TASK_ID,
           "branch": "exp/round68-late-initial-confidence-ablation",
           "training_version": r68.R68_TRAINING_VERSION,
           "environment_constructions": 0, "optimizer_steps": 0, "ppo_calls": 0,
           "checkpoint_writes": 0, "model_forwards": 0,
           "seeds": {}}
    for seed in SEEDS:
        out["seeds"][str(seed)] = index_seed(seed, ck, base.R68_FREEZE)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    tmp.replace(args.out)
    for seed, e in out["seeds"].items():
        print(f"s{seed}: param={e['parameter_sha256'][:12]} file={e['file_sha256'][:12]} "
              f"opt_params={e['optimizer_owns_param_count']} "
              f"frozen_objs={e['frozen_identity']['tensor_count']} "
              f"trainable_objs={e['trainable_identity']['tensor_count']} "
              f"lambda_final={e['training_state']['lambda_history_final']} "
              f"episodes={e['training_state']['episodes_done']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())