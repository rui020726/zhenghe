"""批次链只读复核：补偿冻结 harness 中 `controller_module_digests_before` 的已知病态。

背景（如实登记）：冻结版 `v2_train.py` 的 `module_digests_before` 在每个训练进程起点计算一次，
因此续跑进程写出的批记录里，该字段对 batch 1..N-1 是进程起点值而非各批更新前值。它是纯记账
字段：`ppo_update` 不读它、任何训练量都不依赖它；`controller_parameters_before/after`
（每批完整的 parameter identity）与 `module_gradient_l1_*`、`lambda`/`g_hat`、
`likelihood_alignment` 均为逐批正确值。`v2_check.batch_audit` 的传递一致性检查因此对
seed 11 报了一条问题（batch 1 的 before 摘要等于 batch 0 的 after 摘要）。

本脚本不改任何已冻结 harness 文件（冻结身份锁定训练所用代码），只用落盘批记录复核真正承重的
链条：逐批参数链、cost_head 每批权重摘要的唯一性、cost_head 每批预裁剪梯度非零、对偶轨迹、
likelihood、冻结权重哈希。输出写 `BATCH_CHAIN_REVIEW_s<seed>.json`。

    /d/project_env/python.exe batch_chain_review.py --seed 11
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="read-only per-batch chain review")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)

    records = []
    for batch in range(32):
        path = HERE / "checkpoints" / f"v2_batch_s{args.seed}_b{batch}.json"
        if not path.exists():
            raise SystemExit(f"missing batch record: {path}")
        records.append(json.loads(path.read_text(encoding="utf-8")))
    if [record["batch"] for record in records] != list(range(32)):
        raise SystemExit("batch records are not the contiguous 0..31 set")

    chain_links = [
        {"batch": index,
         "before_sha": records[index]["controller_parameters_before"]["parameter_sha256"][:16],
         "previous_after_sha": records[index - 1]["controller_parameters_after"]["parameter_sha256"][:16],
         "consistent": records[index]["controller_parameters_before"]["parameter_sha256"]
         == records[index - 1]["controller_parameters_after"]["parameter_sha256"]}
        for index in range(1, 32)]
    cost_after = [record["controller_module_digests_after"]["cost_head"] for record in records]
    gradients = [record["update"]["module_gradient_l1_min"]["cost_head"] for record in records]
    lambdas = [record["dual"]["lambda_next"] for record in records]
    alignment = [record["update"]["likelihood_alignment"]["max_abs_gap"] for record in records]
    frozen = [record["frozen_checkpoints"] for record in records]
    optimizer_steps = [record["update"]["optimizer_steps"] for record in records]

    checks = {
        "batch_records_contiguous": True,
        "parameter_chain_consistent": all(link["consistent"] for link in chain_links),
        "cost_head_weights_change_every_batch": len(set(cost_after)) == len(cost_after),
        "cost_head_gradient_nonzero_every_batch": all(value > 0.0 for value in gradients),
        "all_module_gradients_recorded": all(
            set(record["update"]["module_gradient_l1_min"]) ==
            {"trunk", "action_head", "task_value_head", "cost_head"} for record in records),
        "lambda_recorded_every_batch": all(isinstance(value, float) for value in lambdas),
        "likelihood_within_tolerance": all(value <= 1e-6 for value in alignment),
        "frozen_checkpoint_hashes_constant": all(entry == frozen[0] for entry in frozen),
        "optimizer_steps_positive": all(value > 0 for value in optimizer_steps),
    }
    payload = {
        "training_seed": args.seed,
        "review": "read-only per-batch chain review compensating the registered stale "
                  "module_digests_before bookkeeping field",
        "batches": len(records),
        "episodes_done": records[-1]["update"] and None,
        "parameter_chain_links": chain_links,
        "cost_head_after_digests": cost_after,
        "cost_head_preclip_grad_l1_min": gradients,
        "lambda_trajectory": lambdas,
        "likelihood_max_abs_gap": alignment,
        "checks": checks,
        "known_instrumentation_defect": {
            "field": "controller_module_digests_before",
            "behaviour": "frozen v2_train.py computes it once per training process start, so a "
                         "resumed process writes the process-start value into batches 1..N-1",
            "impact": "bookkeeping only: ppo_update never reads it and no training quantity "
                      "depends on it; the per-batch controller_parameters_before/after chain, "
                      "module gradients, lambda/g and likelihood records are per-batch correct",
            "why_not_fixed_mid_run": "the frozen identity locks the code training ran under; "
                                     "re-editing the harness after checkpoints exist would mix "
                                     "code identities, so the compensating review is separate"},
        "passed": all(checks.values()),
    }
    payload["episodes_done"] = None
    del payload["episodes_done"]
    path = HERE / f"BATCH_CHAIN_REVIEW_s{args.seed}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"training_seed": args.seed, "batches": len(records),
                      "checks": checks, "passed": payload["passed"]}, ensure_ascii=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
