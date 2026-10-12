import hashlib, json
import torch

RUN = r"D:\code\整合\zhenghe-round42\runs\comm_bc_initialized_constrained_ppo_20261007_r2"
R60_CK = RUN + r"\round60_partitioned_actor_head_ablation\raw\checkpoints"
R66_CK = RUN + r"\round66_bc_early_path_preserving_late_adaptation\raw\checkpoints"
OUT   = RUN + r"\round66_bc_early_path_preserving_late_adaptation\raw\runtime\ROUND66_REVIEW_PARAM_DIFF.json"

MODULES = ["trunk", "early_head", "late_head", "value_task_head", "value_cost_head"]
FROZEN = ["trunk", "early_head"]
TRAINABLE = ["late_head", "value_task_head", "value_cost_head"]

R60_EP0_SHA = {
    11: "f267558d0cc8933987c09576b8d6d6ad512dbc3c73653d059104705c1faca437",
    22: "249f769770fd0bffcc8df345539088890f9323f1338a224e3c8195d35adc8555",
    33: "5f47f28b2678a0091e2370345b622aad7bdf1d8064b19d3ed57282bc8f005e93",
}
R66_EP128_SHA = {
    11: "d5c003c60b67716f4d25a4cdfa170f43c6fda3265a782a8e63b30ca2293da3d7",
    22: "77c2606d559455f0892e22989f94d1509dd2fd01855cc03106d32aed601656cf",
    33: "8c61ea7b82baf1833bd1afa4a9850ab14bae88d2e668d240dad2ce1626a9f06a",
}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_controller(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["controller"], payload

def module_diff(a_state, b_state):
    ka, kb = set(a_state), set(b_state)
    keys_match = (ka == kb)
    keys = sorted(ka & kb)
    n = 0; identical = keys_match; max_abs = 0.0; sq = 0.0
    for k in keys:
        ta = a_state[k].detach().cpu()
        tb = b_state[k].detach().cpu()
        same = (ta.shape == tb.shape) and (ta.dtype == tb.dtype) and torch.equal(ta, tb)
        identical = identical and same
        if ta.shape == tb.shape:
            d = ta.to(torch.float64) - tb.to(torch.float64)
            max_abs = max(max_abs, float(d.abs().max()) if d.numel() else 0.0)
            sq += float((d * d).sum())
        n += 1
    return {"n_tensors": n, "keys_match": bool(keys_match), "bit_identical": bool(identical),
            "max_abs_diff": max_abs, "l2_diff_norm": sq ** 0.5}

result = {"kind": "round66-review-param-diff/1",
          "comparison": "R60 ep0 init (BC) -> R66 ep128 (NEW); model parameters only, read-only",
          "note": "torch.load read-only, no writes. frozen=trunk+early_head; trainable=late_head+task_critic+cost_critic."}
seeds = {}
for seed in (11, 22, 33):
    ep0_path = f"{R60_CK}\\v2_train_s{seed}_before_b0.pt"
    ep128_path = f"{R66_CK}\\v2_train_s{seed}_ep128.pt"
    ep0_sha = sha256_file(ep0_path)
    ep128_sha = sha256_file(ep128_path)
    c0, p0 = load_controller(ep0_path)
    c128, p128 = load_controller(ep128_path)
    per = {m: module_diff(c0[m], c128[m]) for m in MODULES}
    def agg(mods):
        sq = 0.0; ident = True; n = 0
        for m in mods:
            r = per[m]; sq += r["l2_diff_norm"] ** 2; ident = ident and r["bit_identical"]; n += r["n_tensors"]
        return {"modules": mods, "n_tensors": n, "all_bit_identical": bool(ident), "l2_diff_norm": sq ** 0.5}
    seeds[str(seed)] = {
        "ep0_file_sha256": ep0_sha, "ep0_file_sha_matches_expected": ep0_sha == R60_EP0_SHA[seed],
        "ep0_episodes_done": int((p0.get("training_state") or {}).get("episodes_done", -1)),
        "ep128_file_sha256": ep128_sha, "ep128_file_sha_matches_index": ep128_sha == R66_EP128_SHA[seed],
        "ep128_episodes_done": int((p128.get("training_state") or {}).get("episodes_done", -1)),
        "ep128_structure": p128.get("structure"),
        "per_module": per,
        "frozen_trunk_plus_early": agg(FROZEN),
        "trainable_late_plus_critics": agg(TRAINABLE),
    }
result["seeds"] = seeds
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=1)
print(json.dumps(result, indent=1))
