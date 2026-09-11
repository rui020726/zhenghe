# 协作与审核流程

## 分支

- `main`：保持可测试，不直接开发。
- `feat/<name>`：新功能。
- `fix/<name>`：缺陷修复。
- `exp/<name>`：实验或消融。
- `docs/<name>`：文档。

每次工作前：

```bash
git switch main
git pull --ff-only
git switch -c feat/example
```

## 提交

使用小而完整的提交，示例：

```text
feat: emit stable PILOC detection events
fix: mask all-padding neighbor attention
test: cover task revision rejection
docs: update joint training status
```

不要把格式化整个仓库和功能修改混在同一提交中。

## Pull Request

PR 说明至少包含：

- 目标和设计选择；
- 修改的模块；
- observation/action/reward 维度是否变化；
- 执行过的测试和结果；
- checkpoint 是否兼容；
- 已知风险和后续工作。

合并前至少运行：

```bash
python -m unittest discover -s integrated_sar/tests -v
python -m integrated_sar
```

修改模型或环境时，还应加入针对该行为的最小确定性测试。

## 大文件

普通 Git 不接收 checkpoint、日志、动画、数据集和生成文档。确需版本化的权重使用 Git LFS
或 GitHub Release，并在 README 中记录文件名、哈希和对应代码提交。

## 交给 Codex/其他 agent 审核

推送功能分支并创建 PR，然后提供：

```text
请审核 <branch> 相对 main，重点检查 <模块/风险>，先不要合并。
```

审核应优先报告正确性、数据泄漏、训练/执行不一致、shape/mask、信用分配和缺失测试，
而不是只做风格评论。
