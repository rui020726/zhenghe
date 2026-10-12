import hashlib, json

BASE = r"D:\code\整合\zhenghe-round42\runs\comm_bc_initialized_constrained_ppo_20261007_r2\round66_bc_early_path_preserving_late_adaptation"
RI = BASE + r"\raw\runtime\INDEPENDENT_REVIEW_INPUTS.json"
ST = BASE + r"\raw\ROUND66_CORRECTED_STATISTICS.json"
OUT = BASE + r"\raw\runtime\ROUND66_REVIEW_NUMERIC_PROJECTION.json"

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

ri = json.load(open(RI, encoding="utf-8"))
st = json.load(open(ST, encoding="utf-8"))

def slim(arm):
    return {"Y": arm["on_time_vec"], "C": arm["bytes_vec"], "M": arm["target_mask_vec"],
            "Y_total": arm["on_time_total"], "C_total": arm["bytes_total"]}

arms = {}
for k, v in ri["model_arms"].items():
    arms[k] = slim(v)
for k, v in ri.get("rule_arms", {}).items():
    arms[k] = slim(v)

proj = {
    "kind": "round66-review-numeric-projection/1",
    "source_files": {
        "INDEPENDENT_REVIEW_INPUTS.json": sha256_file(RI),
        "ROUND66_CORRECTED_STATISTICS.json": sha256_file(ST),
    },
    "audited_code_head": ri.get("audited_code_head"),
    "primary_comparison": st.get("primary_comparison"),
    "secondary_comparison": st.get("secondary_comparison"),
    "greedy_is_primary": st.get("greedy_is_primary"),
    "scene_order": ri["scene_order"],
    "focal_scene": ri["focal_scene"],
    "bootstrap": st.get("bootstrap"),
    "mask_semantics": ri["source_summary"]["mask_semantics"],
    "legend": "Y=on_time per scene; C=spent_bytes per scene; M=target bitmask (bit i set iff piloc-e1-target-i on_time); arms: NEW=R66 ep128, OLD=R60 ep128, BC=R60 ep0 init; rule arms greedy-only",
    "arms": arms,
    "focal": ri["focal"],
    "closure": ri["closure_check"],
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(proj, f, indent=1)
print(json.dumps(proj, separators=(",", ":")))
