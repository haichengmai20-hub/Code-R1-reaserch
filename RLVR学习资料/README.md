# Code-R1 RLVR 学习资料首页

先读 [00 全程走读](00_一条记录从原始Task到训练.md)，用一条真实记录看清从原始 Task 到训练的全过程；再按 01—04 逐站深入。全套资料只有一道贯穿题：LeetCode 53「最大子数组和」，所有机制例子都用它；「两数之和」只在 04 用来做跨题迁移练习。每阶段先理解一个机制，再观察运行结果、改变一个条件、独立解释差异。遇到疑问，按当前问题跳到 [主框架](RLVR工业流程.md#mvp-overview) 和源码，不要求首次通读全部参考。

主线只跟踪 Task、Tests（源码保存为 GT）、候选代码、Verifier。原始与入池任务是同一题的不同表示，metadata 是属性；Oracle 是求解/核对答案的方法，不另要求开发一个对象体系。名称简化仍保留每条 input/expected 的来源与验证证据。

**要记住的抽象机制：** 先明确任务要求，用有依据的检查验收程序行为，再把反馈转成策略优化信号；每次改变数据、判定或策略，都要知道哪条依赖因此改变。学习时能用人话解释，深挖时能指出源码中的字段、计算与反例。

```text
题意/接口 → 合法输入与判定依据 → Tests 保存为 GT
                                          ↓
题目 → 候选代码 → Verifier → reward → 同题 advantage
                                          ↓
                 token logprob/mask → loss → 更新/同步
                                          ↓
                     固定协议评测 → 归因与重新验收
```

## 四阶段路线与交接

| 阶段入口 | 这一步解决什么 | 同一实验留下什么 |
|---|---|---|
| [00 全程走读](00_一条记录从原始Task到训练.md) | 一条记录从原始 Task 到训练的七站全景 | 每站的口述版、学习进度与资料修改记录 |
| [01 数据与 GT](01_数据理解与GT构建.md) | 同一道题怎样成为可追溯的任务与测试 | 自己新增的配对、审核理由、新 GT 与读回 |
| [02 Verifier 与任务池](02_Verifier与任务池验收.md) | 一个回答怎样变成 reward，这个分能不能信 | 两个候选从格式到 reward 的追踪、一次单条件变化、8 条记录的故障定位与小池读回 |
| [03 Rollout 与 GRPO](03_Rollout与GRPO机制.md) | 分数怎样经分组、logprob 和 loss 改变参数 | uid/mask、解析梯度与有限差分、clip/KL/长度解释 |
| [04 更新与独立评测](04_真实训练与独立评测.md) | 怎样证明更新、同步及效果分别成立 | 两数之和迁移与综合验收；真实运行证据按资源补齐 |
| [05 实跑推演](05_实跑推演_Qwen3-8B.md) | 没有 GPU 时，以 Qwen3-8B-Base（预训练模型）起步的两轮 GRPO 推演为例，把机器、数据、框架、配置、曲线、评测、归因七环想一遍 | 自己的七环决策日志、事故定位、配对比较结论；产物在 `examples/training/runs/qwen3-8b-grpo-20260928/` |

每阶段仍为三项作业。把实际结果和解释写进各自练习目录的一份 `notes.md` 即可，不要求另建一套题卡、Oracle 系统或报告平台。全量审查与规模化工程按瓶颈深入。

## 开始前的五项自测

先口答，再看阶段给出的解释；答不出就按该阶段的补课位置处理。

| 自测 | 合格回答要点 | 补课位置 |
|---|---|---|
| 模型输入、程序输入和判题输入分别是什么？ | 题目→代码；nums→整数；候选+GT→判定/reward | [01](01_数据理解与GT构建.md#prerequisites-1) |
| JSONL 与 Parquet 为什么不能随意替换文件名？ | 存储格式和加载器不同；读回字段与类型，再接消费者 | [01](01_数据理解与GT构建.md#prerequisites-1) |
| 退出码 0 证明了什么？ | 进程报告成功，未必完成测试；还需可信结果与完成证据 | [02](02_Verifier与任务池验收.md#interview-2) |
| shape=(B,R) 和 mask 是什么？ | B条回答×R个response位置；mask标有效位置，需说明构造规则 | [03](03_Rollout与GRPO机制.md#prerequisites-3) |
| 离散 reward 为什么能训练模型？ | 固定advantage加权当前token的logprob，梯度经模型forward产生 | [03](03_Rollout与GRPO机制.md#gradient-bridge) |

不需要先学完全部 PyTorch/Ray/FSDP。必学是当前链路的输入输出、对齐、求导边界与证据；周边系统按 [04 的深入入口](04_真实训练与独立评测.md#deeper-topics) 查阅。

## 开始前选择可完成的范围

- **机制学习通关：** 标准库与 CPU，完成学生自己的测试变更、跨题迁移、未知类别诊断，以及梯度/分组/指标解释。解析求导与两标量更新不是 torch 或 LLM 训练。
- **真实模型采样：** 在阶段三实采样前确定 checkpoint、模型资源、采样参数、Linux 执行 backend、预算与停止条件。未知代码不能交给本课普通 subprocess。
- **真实更新：** 在已验收执行环境及合适 GPU 上完成阶段四的参数更新、权重同步、保存/恢复与固定评测。无资源标待实跑；不把 CPU 报告写成真实训练或泛化提升。

只学不复现能走到哪、实跑分几档、面试能答到哪，见[路线评估：仅学习与实跑](路线评估_仅学习与实跑.md)。Task 选取与构造、GT 构造、expected / oracle / 测试扩展、Sandbox 四个专题分散在各篇和补强项里，位置索引见[专题索引：任务构造与沙箱](专题索引_任务构造与沙箱.md)。

**效果判断单独验收。** baseline与更新checkpoint按同一预登记协议比较，报告按题配对、重复采样波动、失败与成本。一次闭环跑通不要求性能必须提高；提出能力提升结论必须有独立评测证据。三种完成范围不是三套新对象。

<a id="run-setup"></a>

## 可运行的共同演示

所有命令都在仓库根目录的 bash 里执行。需要 Python 3.10+、标准库和 CPU，不调用模型。macOS 自带的 `python3` 是 3.9，跑不了 `run_demo.py`，本机用 uv 安装的 3.12：

```bash
cd /Users/s11437/Code-R1-reaserch
PY=/Users/s11437/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12   # 换成任何 Python 3.10+ 都可以
git submodule update --init LeetCodeDataset                                               # 首次运行前初始化数据子模块
$PY -B RLVR学习资料/examples/maximum-subarray/run_demo.py all
```

下文各篇的命令都沿用这里的 `$PY`。作者在 Windows、Python 3.14.6 上验证过这些脚本；2026-10-13 在 macOS、Python 3.12.15 复跑通过。

`prepare / verify / score / advantage / evaluate` 也可逐步运行，各自补齐依赖。固定演示资产说明见 [examples/maximum-subarray](examples/maximum-subarray/README.md)，结果见 [report.json](examples/maximum-subarray/outputs/report.json)。演示输出可以随时重新生成。旧版资料用过的五道题只保存在[历史归档](examples/history-20261003/README.md)，不属于当前贯穿题。

当前演示区分 8 条接受记录与 7 条故意错配/非法输入负控制。原 scorer 的提前退出高分仍作为缺口保留；独立审核没有把它当正确程序。数据审核、判分协议、执行安全与模型效果分别记录。

## 学生练习入口与阅读层级

固定演示的 `outputs/` 不是学生输入。实际练习通过 [practice.py](examples/maximum-subarray/practice.py) 创建独立目录、消费你编辑的测试数据，并保留每次预测与结果：

```bash
$PY -B RLVR学习资料/examples/maximum-subarray/practice.py init --run data-01 --mode seed
```

随后按 [01 的作业](01_数据理解与GT构建.md#assignment-1-2) 修改 `practice_runs/data-01/cases.json` 和 `prediction.json`，再 `check / score`。完整命令、退出码及产物见 [操作说明](examples/maximum-subarray/EXERCISES.md)。迁移与未知类别诊断先保存预测，再展开结果；[答案与评分依据](examples/maximum-subarray/ANSWERS.md) 放在资产目录，提交后自检。

每个阶段的“先记机制”是第一遍必读；“面试深挖”折叠栏按需要展开。面试回答用四句话组织：**机制是什么 → 本版本怎样实现 → 一个反例/取舍 → 当前证据能证明到哪里**。禁止把教学toy、配置或作者报告讲成自己的真实训练经历。

建议复习顺序：当天不看笔记画出主链并口答三个输入输出；隔日从随机一个字段或故障出发追到消费者；完成四阶段后再用三分钟复述与两个现场深挖题复核。想不起名词先回到它解决的问题，不能只背答案表。每次留下错误判断、修正依据和一次独立变式。

**反向补充源码：** 资料讲到了、源码没有、而且设计靠谱的机制，会写成代码补进源码。原版的预处理和判分文件不改，只在判分路由和训练入口各加一个分支；补强版是并列的新模块，每一项都有 CPU 可复现的验证，见[反向补充源码](examples/source-supplements/README.md)。

真实训练准备见 [小池、batch预检与Linux命令](examples/training/README.md)。本机已验证标准库准备与batch计算；Parquet导出依赖目标环境的pyarrow，Linux/CUDA训练按资源实跑后验收，不提前盖章。

## 综合通关与返工

[作业4.3](04_真实训练与独立评测.md#assignment-4-3)汇总四阶段。自动程序检查配对、数字、分类和材料完整性；源码解释、因果判断与是否独立完成按评分表人工核对。运行成功不自动等于学会。

- 自己新增的合法小输入有独立答案依据，进入新GT；坏配对只隔离坏行。
- 两数之和识别并修正输出顺序的判定缺口，保留原版与新判定版本。
- 未知类别诊断允许“没有错误”，也允许多个责任层；先证据后归因。
- 能追token对齐、冻结量、clip梯度、EOS和micro batch长度权重，并手算一个变式。
- 能区分reward、correctness、pass@k，以及机制、运行、效果三种证据。

上述关键项有误就返回对应作业修正；不是继续增加数据或训练步数。最终评测材料一旦用于选择或修复方案，改记开发用途。

## 按问题跳到主框架

[Task 清洗](RLVR工业流程.md#task-preparation) · [Tests 逐条审核](RLVR工业流程.md#tests) · [GT 封装](RLVR工业流程.md#gt-packaging) · [Verifier](RLVR工业流程.md#verification) · [执行边界](RLVR工业流程.md#execution-boundary) · [奖励](RLVR工业流程.md#reward-mapping) · [GRPO](RLVR工业流程.md#grpo-advantage) · [更新](RLVR工业流程.md#policy-update) · [评测](RLVR工业流程.md#evaluation) · [归因](RLVR工业流程.md#layer-attribution) · [证据边界](RLVR工业流程.md#historical-evidence)

顶部全局互链、作业内锚点和底部上一/下一仍保留；阶段到框架按当前问题跳转，框架到阶段链接对应练习。四篇课程各自组织作业，框架集中保存机制、来源与案例，避免反复复制整套定义或题面。
