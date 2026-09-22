"""Close last round's provenance before this round spends a checkpoint on it.

Two things were wrong in ``COMM_BUDGET_ACTION_MASK_MANIFEST.json``, and neither of them touched a
number -- which is exactly why they survived a round whose gates all passed.

The builder's executed-module allow-list was typed by hand, and one of the six paths it named,
``integrated_sar/comm_transcript.py``, has never existed in the repository: the file is not on disk,
no object for it exists anywhere in Git's history, and nothing imports it -- the class it is named
after lives in ``integrated_sar/comm.py``. The reason it read as *dirty* rather than as absent is a
second, sharper bug: ``git rev-parse HEAD:<missing path>`` does not fail, it echoes the argument
back on stdout, and the builder truncated that echo into a fourteen-character fake blob id. A
phantom and an edited file were therefore indistinguishable in the manifest.

The head problem is the same category of mistake in the other direction. The M0-M3 evidence is
recorded against ``7f9929c``, and the branch was then extended by ``e692607``, which adds a
read-only checkpoint test and regenerates the result documents. Saying "delivered at e692607" about
numbers measured at 7f9929c is harmless only while nobody checks; this script checks, and records
which files differ between the two heads and that no executed module is among them.

Nothing here re-runs an episode or rewrites an archived measurement: the old results are quoted
read-only, with their blob ids, so a reader can confirm they were not touched.

    python provenance_correction.py
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import retrain_common as common

PHANTOM = "integrated_sar/comm_transcript.py"
OLD_DIR = common.MASK_FIX_ROUND
OLD_MANIFEST = OLD_DIR / "COMM_BUDGET_ACTION_MASK_MANIFEST.json"
OLD_RESULTS = OLD_DIR / "COMM_BUDGET_ACTION_MASK_RESULTS.json"
OLD_PROTOCOL = OLD_DIR / "COMM_BUDGET_ACTION_MASK_PROTOCOL.md"
OLD_SMOKE = OLD_DIR / "COMM_BUDGET_ACTION_MASK_SMOKE.json"

__all__ = ["main", "phantom_evidence", "heads_evidence", "correction"]


# ------------------------------------------------------------------ the phantom path
def phantom_evidence() -> dict[str, Any]:
    """Prove the absence four ways, because "I looked and it wasn't there" is not a record."""
    archived = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    modules = archived["executed_code"]["modules"]
    # The live reproduction of the git behaviour that made the phantom look like a defect.
    rev_parse = common.git("rev-parse", f"HEAD:{PHANTOM}")
    cat_file = common.git("cat-file", "-e", f"HEAD:{PHANTOM}")
    history = common.git_out("log", "--all", "--oneline", "--diff-filter=A", "--", PHANTOM)
    graph = common.executed_modules(common.MASK_FIX_ENTRIES)
    from integrated_sar.comm import CommTranscript

    return {
        "path_named_in_manifest": PHANTOM,
        "manifest_entry": modules.get(PHANTOM),
        "exists_on_disk": (common.ROOT / PHANTOM).exists(),
        "ever_added_in_git_history": bool(history.strip()),
        "git_log_added_commits": history or "(no commit ever added this path)",
        "in_derived_import_graph": PHANTOM in graph,
        "class_actually_lives_in": f"{CommTranscript.__module__}.py",
        "class_defined_at": Path(
            importlib.import_module(CommTranscript.__module__).__file__
        )
        .resolve()
        .relative_to(common.ROOT)
        .as_posix(),
        "mechanism": {
            "git_rev_parse_stdout": rev_parse.stdout.strip(),
            "git_rev_parse_returncode": rev_parse.returncode,
            "git_cat_file_returncode": cat_file.returncode,
            "note": (
                "rev-parse echoes an unresolvable argument to stdout instead of failing, so the "
                "old builder's head_blob became a truncated copy of the path and matches_head "
                "flipped to false; absence was reported as a dirty module"
            ),
        },
        "dirty_modules_recorded_then": archived["executed_code"]["dirty_modules"],
        "real_modules_in_that_list": sorted(
            path for path, ids in modules.items() if ids.get("matches_head")
        ),
        "conclusion": (
            "the only entry in dirty_modules was a path that does not exist and was never "
            "executed; no executed module was dirty, so M0-M3's measurements stand, but the "
            "manifest's identity column did not"
        ),
    }


# ------------------------------------------------------------------ the two heads
def heads_evidence() -> dict[str, Any]:
    """Separate "the code that ran" from "the commit that delivered the branch"."""
    graph = common.executed_modules(common.MASK_FIX_ENTRIES)
    archived_manifest = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    archived_results = json.loads(OLD_RESULTS.read_text(encoding="utf-8"))
    between = common.git_out("diff", "--name-only", common.RESULTS_HEAD, common.FIX_HEAD).splitlines()
    identity = common.working_tree_diff(common.RESULTS_HEAD, graph)
    show = {
        head: common.git_out("show", "--no-patch", "--format=%H|%ad|%s", "--date=iso", head)
        for head in (common.RESULTS_HEAD, common.FIX_HEAD)
    }
    return {
        "results_head": {
            "sha": common.RESULTS_HEAD,
            "commit": show[common.RESULTS_HEAD],
            "recorded_as_manifest_executed_head": archived_manifest["executed_code"]["head"],
            "recorded_as_results_head": archived_results["head"],
        },
        "final_head": {
            "sha": common.FIX_HEAD,
            "commit": show[common.FIX_HEAD],
            "files_changed_relative_to_results_head": sorted(between),
            "adds_executed_modules": sorted(
                path for path in between if path in graph
            ),
        },
        "executed_graph_at_both_heads": identity,
        "statement": (
            "M0-M3 were measured with the executed graph of 7f9929c; e692607 changed only a test "
            "file and the regenerated result documents, and the derived graph is identical at "
            f"both heads (clean={identity['clean']})"
        ),
        "why_it_matters": (
            "a reader who re-runs the gates at e692607 gets the same code under test, but a "
            "reader who cites 'delivered at e692607' as the SHA of the measurement would be "
            "quoting a commit that only documents it"
        ),
    }


# ------------------------------------------------------------------ the builder fix
def builder_fix() -> dict[str, Any]:
    """What the corrected builder now does, verified by importing it rather than by grep.

    A source-text search for the phantom path is useless here: the correction that removed it from
    the allow-list also names it, in the docstring, to explain what was removed.
    """
    archived = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    import mask_manifest as old_builder

    graph = common.executed_modules(common.MASK_FIX_ENTRIES)
    declared = list(old_builder.EXECUTED_MODULES)
    derived_from_builder = old_builder.executed_modules()
    return {
        "file": (OLD_DIR / "mask_manifest.py").relative_to(common.ROOT).as_posix(),
        "phantom_in_declared_contract": PHANTOM in declared,
        "phantom_in_builder_derived_graph": PHANTOM in derived_from_builder,
        "declared_contract_modules": declared,
        "derives_from_import_graph": callable(getattr(old_builder, "executed_modules", None)),
        "resolves_absence_before_hashing": "in_head" in old_builder.blob_identity([PHANTOM])[PHANTOM],
        "phantom_reported_by_fixed_builder": old_builder.blob_identity([PHANTOM])[PHANTOM],
        "derived_graph_size": len(graph),
        "derived_graph": graph,
        "archived_numbers_left_untouched": {
            "totals": archived_results_totals(),
            "smoke_closure": json.loads(OLD_SMOKE.read_text(encoding="utf-8"))["closure"],
        },
        "old_manifest_blob": common.blob_identity([OLD_MANIFEST.relative_to(common.ROOT).as_posix()]),
    }


def archived_results_totals() -> dict[str, Any]:
    archived = json.loads(OLD_RESULTS.read_text(encoding="utf-8"))
    return {
        "candidates": int(archived["totals"]["candidates"]),
        "initial_executable_candidates": int(archived["totals"]["initial_executable_candidates"]),
        "actionable_candidates": int(archived["totals"]["actionable_candidates"]),
        "requested_by_policy": int(archived["totals"]["requested_by_policy"]),
        "executed_policy_emissions": int(archived["totals"]["executed_policy_emissions"]),
        "policy_refusals": int(
            archived["totals"]["policy_budget_refusals"]
            + archived["totals"]["policy_capacity_refusals"]
        ),
        "max_abs_log_prob_gap": archived["gates"]["M2_ppo_probability_agreement"]["evidence"][
            "max_abs_log_prob_gap"
        ],
        "all_gates_passed": bool(archived["all_gates_passed"]),
    }


def correction() -> dict[str, Any]:
    phantom = phantom_evidence()
    heads = heads_evidence()
    builder = builder_fix()
    untouched = [
        path
        for path in (
            OLD_MANIFEST,
            OLD_RESULTS,
            OLD_RESULTS.with_suffix(".md"),
            OLD_SMOKE,
            OLD_PROTOCOL,
        )
    ]
    return {
        "document": "COMM_BUDGET_MASK_PROVENANCE_CORRECTION",
        "date": "2026-09-22",
        "branch": common.git_out("rev-parse", "--abbrev-ref", "HEAD"),
        "round": "comm_budget_masked_retrain_20260922",
        "brief": "COMM_BUDGET_MASKED_RETRAIN_NEXT_ROUND.md",
        "section": "2 (close last round's provenance before training)",
        "phantom_module": phantom,
        "heads": heads,
        "manifest_builder_fix": builder,
        "archived_files_untouched": {
            path.relative_to(common.ROOT).as_posix(): common.sha256(path)[:16]
            for path in untouched
            if path.exists()
        },
        "episodes_started_by_this_correction": 0,
        "what_does_not_change": {
            "M0_to_M3": "all four gates were decided by re-run suites and a re-computed closure; "
            "neither defect touched a measured quantity",
            "smoke_numbers": builder["archived_numbers_left_untouched"]["totals"],
            "trainer_identity": common.setting(),
        },
        "what_this_round_freezes": {
            "rule": (
                "the executed graph is derived by importing the training entry, every member's "
                "blob is recorded against a registered code-freeze SHA, and any launch checks "
                "that the working tree and HEAD still match it"
            ),
            "fails_closed_when": [
                "no freeze is registered",
                "a graph member differs from the freeze in HEAD or in the working tree",
                "a graph member or a script in this directory is untracked",
                "a module appears in the graph that the freeze never recorded",
            ],
        },
    }


def render(data: dict[str, Any]) -> str:
    phantom = data["phantom_module"]
    heads = data["heads"]
    builder = data["manifest_builder_fix"]
    totals = builder["archived_numbers_left_untouched"]["totals"]
    out: list[str] = []
    add = out.append
    add("# 上一轮 manifest 的 provenance 更正（不新增 episode）")
    add("")
    add(
        f"日期：{data['date']}。本轮 `{data['round']}`，分支 `{data['branch']}`，执行指令 "
        f"`{data['brief']}` §2。上一轮：`fix/comm-budget-action-mask`，产物目录 "
        "`runs/comm_budget_action_mask_fix_20260922/`。"
    )
    add("")
    add(
        "**结论先说**：两处都是**身份口径**缺陷，不触及任何被测数量；M0–M3 的判定不因此改变。"
        "但正式训练会把权重与结论挂在一个 SHA 上，所以先把「哪段代码产生了这些数字」说清楚。"
    )
    add("")
    add("## 一、幽灵模块：`integrated_sar/comm_transcript.py`")
    add("")
    add(
        f"- 盘上存在：`{phantom['exists_on_disk']}`；Git 历史上任何一次提交加入过它："
        f"`{phantom['ever_added_in_git_history']}`（`{phantom['git_log_added_commits']}`）；"
        f"在派生 import 图里：`{phantom['in_derived_import_graph']}`。"
    )
    add(
        f"- 它所指的那个类真实定义在 `{phantom['class_defined_at']}`"
        f"（模块 `{phantom['class_actually_lives_in']}`），由 `joint_driver.py` 从 `.comm` 导入，"
        "从来不存在独立模块文件。"
    )
    add(
        "- 为什么它显示成 `dirty_modules` 而不是 `absent`："
        f"`git rev-parse HEAD:<不存在路径>` **不报错**，它把参数原样写到 stdout"
        f"（实测 stdout=`{phantom['mechanism']['git_rev_parse_stdout']!r}`，"
        f"returncode={phantom['mechanism']['git_rev_parse_returncode']}；"
        f"`git cat-file -e` 才返回非零 "
        f"{phantom['mechanism']['git_cat_file_returncode']}）。旧 builder 取前 16 字符当 blob id，"
        "于是幽灵条目得到一个假 blob 和 `matches_head: false`："
    )
    add("")
    add("```json")
    add(json.dumps(phantom["manifest_entry"], ensure_ascii=False, indent=1))
    add("```")
    add("")
    add(
        f"- 旧 manifest 的 `dirty_modules` = `{phantom['dirty_modules_recorded_then']}`，"
        f"其中真正存在且与 HEAD 一致的模块 {len(phantom['real_modules_in_that_list'])} 个："
        f"`{phantom['real_modules_in_that_list']}`。"
    )
    add(f"- {phantom['conclusion']}")
    add("")
    add("## 二、两个头：证据头 `7f9929c` 与交付头 `e692607`")
    add("")
    add(f"- 证据头：`{heads['results_head']['commit']}`；"
        f"旧 manifest 的 `executed_code.head` = `{heads['results_head']['recorded_as_manifest_executed_head'][:9]}`，"
        f"旧 RESULTS 的 `head` = `{heads['results_head']['recorded_as_results_head'][:9]}`。")
    add(f"- 交付头：`{heads['final_head']['commit']}`。")
    add(
        f"- 两者之间改动过的文件：`{heads['final_head']['files_changed_relative_to_results_head']}`；"
        f"其中属于执行图的：`{heads['final_head']['adds_executed_modules']}`（空即表示交付头没有动过任何执行代码）。"
    )
    graph_diff = heads["executed_graph_at_both_heads"]
    add(
        f"- 派生执行图（{len(builder['derived_graph'])} 个模块）在两头的差异："
        f"已提交 `{graph_diff['committed_difference']}`、工作区 "
        f"`{graph_diff['working_tree_difference']}` → clean=`{graph_diff['clean']}`。"
    )
    add(f"- {heads['statement']}")
    add(f"- {heads['why_it_matters']}")
    add("")
    add("## 三、builder 怎么修，旧数字怎么留")
    add("")
    add(
        f"- 修的是 `{builder['file']}`：模块清单改为**由 import 派生**"
        f"（`derives_from_import_graph={builder['derives_from_import_graph']}`），"
        "并且先用 `git cat-file -e` 判定「在不在 HEAD」再去 hash"
        f"（`resolves_absence_before_hashing={builder['resolves_absence_before_hashing']}`）。"
    )
    add(
        f"- 修好后重建的 builder 仍把幽灵算进清单吗？"
        f"声明式契约里 `{builder['phantom_in_declared_contract']}`、派生图里 "
        f"`{builder['phantom_in_builder_derived_graph']}`；builder 现在对该路径的回答是"
        f"`{json.dumps(builder['phantom_reported_by_fixed_builder'], ensure_ascii=False)}`"
        "——即 `in_head: false` 的缺失项，而不是脏模块。"
    )
    add("")
    add("**旧 smoke 数字原样保留（未重跑、未改写）：**")
    add("")
    add("| 量 | 值 |")
    add("|---|---:|")
    for key, value in totals.items():
        add(f"| `{key}` | {value if not isinstance(value, float) else f'{value:g}'} |")
    add("")
    closure = builder["archived_numbers_left_untouched"]["smoke_closure"]
    add(
        f"- 旧 smoke 闭合：requested==executed `{closure['all_requested_equal_executed']}`、"
        f"policy 拒付 0 `{closure['all_policy_refusals_zero']}`、log-prob 最大差 "
        f"`{closure['max_log_prob_gap']}`、optimizer.step `{closure['optimizer_steps_total']}`。"
    )
    add("- 本轮更正**启动的 episode 数**："
        f"`{data['episodes_started_by_this_correction']}`（纯身份核算，不产生新测量）。")
    add("")
    add("## 四、本轮据此建立的冻结规则")
    add("")
    add(f"- 做法：{data['what_this_round_freezes']['rule']}")
    for item in data["what_this_round_freezes"]["fails_closed_when"]:
        add(f"- 失败即停：{item}")
    add("")
    add("## 五、旧文件未被改动（sha256 前 16 位）")
    add("")
    for path, digest in data["archived_files_untouched"].items():
        add(f"- `{path}` — `{digest}`")
    add("")
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    data = correction()
    common.write_json(common.json_path("COMM_BUDGET_MASK_PROVENANCE_CORRECTION.json"), data)
    common.json_path("COMM_BUDGET_MASK_PROVENANCE_CORRECTION.md").write_text(
        render(data), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "phantom_exists_on_disk": data["phantom_module"]["exists_on_disk"],
                "phantom_in_graph": data["phantom_module"]["in_derived_import_graph"],
                "dirty_then": data["phantom_module"]["dirty_modules_recorded_then"],
                "executed_graph_clean_between_heads": data["heads"][
                    "executed_graph_at_both_heads"
                ]["clean"],
                "graph_modules": data["manifest_builder_fix"]["derived_graph_size"],
                "episodes_started": data["episodes_started_by_this_correction"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
