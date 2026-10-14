# ROUND75 S0 验收增量报告：决策端五缺陷修复（零环境）

- 分支：`exp/round75-native-ppo-qualification`
- 父提交（S0 交付，原件保留）：`0f8574accae16676a75a248da180be9d49e9fa04`
- 本文件写入时 HEAD：`0f8574accae16676a75a248da180be9d49e9fa04`（本次验收提交的完整 SHA 在最终答复中给出，文件内不猜补）
- 运行目录：`runs/comm_bc_initialized_constrained_ppo_20261007_r2/round75_native_ppo_qualification/`
- 结论：**S0 验收轮完成，五缺陷全部在源码层关闭，各有"改前必失败/改后通过"的反例；全程零环境。S1 付费执行仍未授权，本轮不执行、不预热、不预留。**

## 一、本轮只做一件事

`0f8574a` 交付的 S0 接线在决策端复盘中暴露出五个真实缺陷。本轮不重写科学逻辑，只修这五个缺陷，并为每个缺陷补一个纯 stdlib 反例测试。S0 的报告、verdict、plan、freeze 与 wiring diff 全部原样保留，本报告与 `ROUND75_S0_ACCEPTANCE_VERDICT.json`、`ROUND75_S0_ACCEPTANCE_IDENTITY.json` 是追加件。

零环境计数：模型加载/构造 0、forward 0、backward 0、optimizer 更新（真实/合成）0、PPO 更新/重放 0、环境构造 0、采样 0、checkpoint 写入 0、训练恢复 0、RNG 恢复 0、torch import 0。中央账本计费事件 0。

## 二、五缺陷与源码证据

### 缺陷 1 生产根未固定 + 用可伪造 bool 认身份

`r75_guard.py` 现在把唯一持久根钉死，生产接口只接受它：

```python
R75_CANONICAL_ROOT = HERE / "raw" / "runtime"        # r75_guard.py:73
...
if mode == "production":
    if root is not None and Path(root).resolve() != canonical.resolve():
        raise R75LedgerError("R75 production resources are PINNED to the canonical durable root ...")
```

`fixture(root)` 是独立非生产接口，规范根及其后代一律拒绝（`r75_guard.py:252-260`）；`simulated()` 直接抛错（`:284-288`）。真正挡住伪造对象的是类身份与源码 pin，而不是布尔属性：

```python
if type(resources) is not R75Resources:              # r75_guard.py:440
    raise R75LedgerError("... a forged production object is refused by source")
...
actual = sha256_file(Path(__file__).resolve())       # r75_guard.py:453
if actual != expected_guard_source_sha256:
    raise R75LedgerError("... a modified guard is refused by the native path")
```

`r75_frozen.py:85` 钉住 `EXPECTED_R75_GUARD_SOURCE_SHA256 = "4d730f71..."`，`:195-196` 在每次 native 更新入口把它传进去——所以改弱的 guard 不能靠改个对象名混进真实 NN 路径。反例 8 个全过：非规范根拒绝、只接受钉住的规范根、夹具拒绝规范根及后代、夹具是独立接口、拒绝夹具与伪造对象、检查 guard 源码 pin、`simulated` 拒绝、`r75_frozen` 钉的确实是盘上哈希。

### 缺陷 2 计数器缺 technical 类上限、无状态校验、对账不阻断

R74 已验证件是**实时导入**，不是重写弱化版：`_ProdCounter`/`count_target_vectors`/`LedgerStateError` 来自 `r74_guard_prod.py`（`623b9ec6...`），`R74TotalLedger`/`R74QuotaExhausted` 来自 `r74_guard.py`（`9a6b0637...`），`count_at_entry` 来自 `r43_call_counter_hooks.py`（`02f80072...`）；`r75_guard.py:92-104` 在导入时校验这三件的字节身份，漂移即 fail-closed。R75 只在上层加自己的上限与种类：

```python
class R75PhysicalStepGuard(_ProdCounter):            # r75_guard.py:114
    super().__init__(path, mode=mode, ledger=ledger, normal_cap=12, reserve_cap=4,
                     hard_cap=16, kind="r75-physical-optimizer-step-guard/1", ...)
```

offline 行 5000+1000/6000 且单次 ≤1024（`:121-137`，超限在任何工作之前拒绝且不消费）；joint record 单列；环境构造 0/0/0。对账从"只报告"变成"先阻断"：`verify_or_fail()` 在生产模式 `used != ledger_max` 即抛错并明确"已消费的许可保留、不退、不恢复、不改写"（`:351-368`），`consistency_check()` 保留为不阻断孪生（`:370-376`）。残留锁不恢复：每把计数器锁都是逐次计费时的 O_EXCL，没有实例持有的锁可"恢复"（`:290-291`）。technical 扣费必须预先登记，普通耗尽不会自动升级成储备（`:300-307`）。

反例 13 个全过，其中 `test_corrupt_durable_states_fail_closed` 用 5 个 subtest 覆盖缺 `kind`、错上限、负 used、超 normal、超 hard；另有"有计费却缺计数器""计数器领先账本""备份值偏低"三种损坏各一例，全部 fail-closed。

### 缺陷 3 计量不在真实入口

删掉两处调用方聚合扣费，改成在**控制器自己调用的那个模块**上装表：

```python
def _forward(self, *args, **kwargs):                 # r75_guard.py:190
    self.calls_attempted += 1
    batch = args[0] if args else kwargs.get("input", kwargs.get("batch"))
    rows = count_target_vectors(batch)               # 非 19 维在扣费与前向之前就抛错
    try:
        self.guard.charge_trunk_forward(batch, f"{self.what_prefix}_call{self.calls_attempted}")
    except R75QuotaExhausted:
        self.calls_refused += 1
        raise
    self.calls_charged += 1
    ...
    try:
        output = self._original_forward(*args, **kwargs)
    except BaseException:
        self.calls_failed += 1                       # 已扣费、绝不计为完成、绝不动
        raise
    self.calls_completed += 1
```

`r75_frozen.py:278-279` 在 `r66.likelihood_alignment` 之前 `ensure_trunk_meter`，因此对齐趟、每个 epoch 和任何其他合法 trunk 调用都按**实际输入形状**即时预扣；attempted / charged(licensed) / completed 与行数分列，半路失败不会把预扣伪装成一次已完成前向。重复安装拒绝，卸载显式（`:327-348`）。反例 6 个全过，含 AST 级"调用方聚合扣费已不存在"。

### 缺陷 4 可训练未显式

`r66.fresh_partitioned_controller` 与其 `_PartitionedInner` 里 `requires_grad` 出现次数为 0，规范控制器链同样为 0（静态证据见 identity 件）——也就是说四个可训练模块此前靠 `nn.Module` 默认值活着。现在先显式置 True 再冻结 trunk：

```python
def freeze_trunk_only(controller):                   # r75_frozen.py:145
    frozen, trainable = partition(controller)
    for parameter in trainable:
        parameter.requires_grad_(True)               # 显式，不再依赖默认
    for parameter in frozen:
        parameter.requires_grad_(False)
    mask = assert_mask(controller)
```

反例 2 个全过：源码里 True 必须出现在 False 之前、`assert_mask` 双向校验；r66 构造函数必须保持无 flag（否则这个显式化就失去意义）。

### 缺陷 5 joint record 不记账

每次 native 读同一批已记录记录，都要在**读之前**扣 1 条 joint record，并把 unique 记录数、暴露次数、读次数一起返回：

```python
resources.charge_joint_record(1, what=f"r75_native_update_{update_index}_batch_read")   # r75_frozen.py:253
...
batch_read = {"update_index": int(update_index), "native_batch_reads": 1,
              "unique_input_records": int(unique_input_records),
              "record_exposures": int(unique_input_records),
              "joint_records_charged": 1, ...}
```

同一批记录再读一次是**重复暴露，不是新环境 episode**，环境护栏保持 0/0/0。反例用真实适配器跑：66 条记录读两次 = 132 exposures、2 条 joint record、env 0。

## 三、测试实跑结果

新验收套件 `test_r75_s0_acceptance.py`（纯 stdlib，不 import torch），交付时复跑：

```text
Ran 31 tests in 0.958s

OK
EXITCODE=0
```

分类：ProductionPinning 8、CounterHardening 13、TrunkEntryMeter 6、FreezeFlag 2、BatchReadAccounting 2。生产模式用例用进程内 patch `guard_mod.R75_CANONICAL_ROOT` 指向一次性根目录，出货源码不变；夹具永不交给真实 NN 路径。

**旧套件披露（不当成验收证据）**：`test_r75_s0_spy.py` 在缺陷 1 修好后只跑过一次并记录：11 ok（5 Adapter + 6 WiringSource）/ 9 GuardTests ERROR，exit 1。9 个错误是**设计性失败**：`GuardTests._open()` 用 `R75Resources.production(self.root)`，而 `self.root = $R/_s0_selftest/ledger` 正是被规范根固定拒绝的非规范路径，且旧套件还在用修订前属性名（`synth_steps`、旧四个计数器文件）。交付时未再跑旧套件凑数；S0 报告里"20/20"指的是修订前代码，作为历史 S0 收据保留，不是对当前代码的陈述。

## 四、规范账本只读证据

`raw/runtime` 在新 schema 下仍是零值身份：4 个零计数器（`13b9d36d...` / `314fbb13...` / `1d4022fc...` / `be58a93d...`），总账本 `r75_total_ledger.jsonl`（`1f5929a4...`）共 12 事件、**0 个 charged 事件**，无锁残留。12 条事件全部是 bootstrap/登记类追加（8 条 `ledger-opened` + 4 条 `legacy-s0-zero-superseded`），不是计费。4 个 S0 遗留零计数器字节未变，bootstrap 只按哈希登记、拒绝改写。R60 历史 608 次环境构造不搬入本账本、不构成余量；R61/R74 历史同样不搬入。

## 五、计划修订（必须写准的那几个数）

`ROUND75_S0_PLAN.json` 的 `candidate_accounting.joint_records = 0` **作废**。按缺陷 5，每次 native 读扣 1 条 joint record。对冻结的 66 记录 / 168 候选行批次：

| 量 | 值 | 由来 |
| --- | --- | --- |
| 每次更新物理 optimizer 步 | 4 | 4 epoch × ceil(66/256)=1 个 minibatch |
| trunk 入口调用次数 | 70 | 对齐趟 66 次（每样本一次，合计 168 行）+ 4 epoch × 1 次（每次 168 行） |
| offline trunk 目标行 | 840 | 168 + 4×168 = 840，与保守算术 `update_accounting(66,168)`=168×5 一致 |
| 最大单次行计费 | 168 | ≤1024 单次上限 |
| 每次更新 joint record | 1 | 缺陷 5 |
| 每次更新环境构造 | 0 | 0/0/0 契约 |

候选 A1（≤1 步 / 0 行）+ A2（4 步 / 840 行 / 1 joint）+ A3（4 步 / 840 行 / 1 joint）⇒ **≤9 物理步（上限 12 normal / 16 hard）、≤1680 行（5000/6000）、最大单次 168、joint 2、env 0**。该上界只对冻结的 66 记录批次成立，换批次会改变 `ceil(n/256)` 与行数。

另加 S1 存档要求：八个可训练对象逐一登记 Adam 状态（`exp_avg`/`exp_avg_sq`）有无，缺失就记缺失、不得补造非空；不改写旧 `log_prob`；对齐边界保持 `log_prob_tolerance = 1e-6`。

## 六、未覆盖范围

torch 导入、模型加载与构造、R58 ep0 的 state-dict 键映射；`likelihood_alignment` 的真实结果（2.3e-13 是既往记录的 R71 对齐差，不是本轮新测）；任何真实或合成 optimizer 步、PPO 更新与重放；真实 collector、真实 episode、闭环仿真与任何评估块；完整多批训练与任何科研 checkpoint；任何关于通信策略效果、成本或任务结局的科学结论。

## 七、交付物

代码与测试：`r75_guard.py`（`4d730f71...`）、`r75_frozen.py`（`a17482bf...`）、`test_r75_s0_acceptance.py`（`88fc8ea2...`，31/31 exit 0）；本轮文档：`ROUND75_S0_ACCEPTANCE_VERDICT.json`、`ROUND75_S0_ACCEPTANCE_IDENTITY.json`、本报告，以及追加 `committed_acceptance_revision` 的 `ROUND75_GIT_EXTERNAL_MANIFEST.json`。S0 原件（`ROUND75_S0_*`、`ROUND75_NATIVE_WIRING_*`、`ROUND75_FREEZE.json`、`test_r75_s0_spy.py`、`r75_replay_adapter.py`）保持原样。账本原件仍在 `raw/runtime/`，不提交权重、行、tape、日志或账本状态。

保护：两份 ROUND46 脏文件交付时复哈希一致、未动未暂存；R59 分析日志保持未跟踪、永不暂存；`__pycache__` 已清理。不推送、不合并、不部署。S1 仍未授权。