# CLAUDE.md — Code-R1-reaserch（RLVR 学习仓库）

写给在本仓库工作的 Claude 会话。目标读者是协助用户学习 RLVR 并维护 `RLVR学习资料/` 的助手。

## 1. 这个仓库是什么

- `ganler/code-r1` 的 fork（基线 commit `443f8da`，2026-10-13 时上游 HEAD 仍是它）。code-r1 = `volcengine/verl` 训练框架 + 代码 RL 的数据预处理（`examples/data_preprocess/coder1.py`）+ 代码判分器（`verl/utils/reward_score/coder1/`）+ firejail 沙箱。
- 远程：`https://github.com/haichengmai20-hub/Code-R1-reaserch.git`，分支 `main`。
- 子模块 `LeetCodeDataset/`（`newfacade/LeetCodeDataset`，固定 `f3519e6`）。学习脚本和 23 处文档链接都依赖它。
- 2026-10-10 的 commit `a4002bd` 加入了 `RLVR学习资料/`：中文四阶段 RLVR 课程 + 可运行的 CPU 示例。**这是本仓库的工作重点；verl / code-r1 源码是学习对象，不是开发对象。**
- 两个 README：`README.md` 是 code-r1 的（作者结果、firejail、数据集），`README.md.verl` 是 verl 原版的。

## 2. 工作目标与协作约定

### 资料的优先级原则（用户 2026-10-13 明确，冲突时高优先级胜出）

1. **懂机制**：让用户真正理解工业级 code RLVR 流程每个环节为什么这样做、改一个条件会怎样。
2. **能过面试**：每个机制能用四句结构口述（机制 → 实现 → 反例/取舍 → 证据范围），被追问时能定位到具体环节。
3. **易读易记、便于口述**：人话优先，结构稳定；一个机制配一句定义、一个数字例子、一个反例；便于复述与回答问题。
4. **最后才是与 code-r1 源码的贴合度和一致性**：源码是支撑证据和查阅入口，不是资料的主体。允许为可读性做简化与近似，标一句「简化」即可；行号漂移、hash 锁定、作者产物都不应阻碍对资料的改进。

实际含义：写法上先给结论和人话，再给源码定位；删减重复的证据边界警告，集中保留一处；面试深挖表是一等公民；「与源码不完全一致」不是拒绝一处好解释的理由。底线只有一条：推断写推断，实跑写实跑，不伪造运行证据。

### 目标与约定

目标：通过 Code-R1 源码搞懂 RLVR 的完整闭环，即
`Task/Tests → GT → 候选代码 → Verifier/Sandbox → reward → 同题 advantage → token logprob/mask → loss → 参数更新 → 权重同步 → 固定协议评测 → 分层归因`，并在学习过程中持续完善 `RLVR学习资料/`（修错、补平台差异、补源码证据、调整作业）。

协作约定：

- 用中文交流。源码定位写「文件 + 函数（+ 行号）」，行号以本仓库固定基线为准。
- 回答机制问题按课程的四句结构：**机制是什么 → 本版本怎样实现 → 一个反例/取舍 → 当前证据能证明到哪里**。
- 严格区分三种证据：**CPU 受控演示**（固定候选、标准库）/ **真实采样或训练**（需要 Linux GPU）/ **效果评测**（固定协议、baseline 配对）。2026-10-16 起另有第四类 **推演**（05 与 `runs/qwen3-8b-grpo-20260928/`）：形态对齐真实产物，但数字来自参数模型，只能用于练决策，不能当实跑证据；用户要求推演做得像真的，已做到形态和数量级对齐，但来源说明不删。配置、注释、作者 README 的表格都不是实跑证据；不伪造运行结果，没跑就写"待实跑"。
- **反向补充源码**（用户 2026-10-14 明确；**2026-11-08 规则更新**）：学习资料覆盖的机制多于源码、并且资料的设计质量高、安排合理时，把它实现成代码补进源码。2026-11-08 起用户明确：**与用户讨论定案的方案直接落实到现版源码，叙述不再区分原版/现版**——新定案可直接改 `coder1_plus` 及其数据侧（含必要时改上游文件），资料以现版行为为主语（例：「约束给了」而不是「原版没给、补强版给了」）；历史包袱只保留两处兼容：推演生成器钉 `CODER1_PLUS_REWARD=legacy`（runs 产物与 git 基线一致性）、`run_demo.py` 的源码哈希锁定。清单、状态和验证结果见 `RLVR学习资料/examples/source-supplements/README.md`；讨论定案的核对基准是 `RLVR学习资料/设计定案表.md`（改 md 前后拿它逐条核对）。**上游推演看原始版本（D23，2026-11-13 用户定案）**：从上游缺陷推下游 bad case 时，只读原始上游 commit（基线 `443f8da`）与原始数据快照，**不要用已落盘的 coder1_plus 修复版做发现**——修复会把缺陷修没、新增字段会误导；修复版只用于「验证修复」。原始副本不进仓，需要时从 GitHub 拉独立目录（如 `D:\RL及harness\Code-R1-original`），全程只读。推演产物七字段见 `RLVR学习资料/bad_case调查方法论.md` §二。
- 用户是学习者，Claude 默认是导师兼资料维护者：作业里的预测、手算、`notes.md` 先由用户独立完成，提交后再核对与讲解；用户明确要答案时直接给。
- 阅读顺序：`RLVR学习资料/README.md` → `00`（主线全景）→ `01` → `02` → `03` → `04`；`RLVR工业流程.md` 按机制查阅，不必通读。每阶段三项作业：完整示范 / 改变一个条件 / 独立检查。
- **每学一篇 0x 的固定流程**（用户 2026-10-14 明确）：先给三段讲解：纵览（整个训练流程，以及 01–04 各在哪）、0x 总览（在流程里做什么、长什么样）、0x 文字版讲解。写进资料之前，逐条对照源码和资料的**最新状态**核对数字与说法，能跑的重跑，旧的、错的不记，顺带修掉发现的旧说法。纵览写进 00 开头的 `overview-00` 小节（已有就按最新状态更新），总览和讲解写进 0x 开头的 `overview-N`、`guide-N` 小节；改动记进 00 的修改记录。
- 术语固定：Task / Tests / 候选代码 / Verifier 四个主线名；GT 是 accepted Tests 的评分接口表示（`json.dumps({"functional": ...})`），不是参考代码；Oracle 只是求解/核对答案的方法；Reference Code ≠ KL 用的 Reference Policy；`uid` 是批内同题分组身份，≠ `task_id`。

## 3. 本机环境现状（2026-10-13 实测，macOS arm64）

| 项目 | 状态 |
|---|---|
| 系统 `python3` | 3.9.6；没有 `python` 命令；没有 pytest |
| 学习脚本所需 Python | `run_demo.py` 需要 ≥3.10（`Path.write_text(newline=)`）；`practice.py`/`mechanism_lab.py`/两套测试在 3.9 也能跑 |
| 已安装的 Python | uv 装了 3.12 和 3.14。日常用 3.12：`/Users/s11437/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12`；3.14 在 `.../cpython-3.14-macos-aarch64-none/bin/python3.14`，只用来对照注解延迟求值等版本差异 |
| 子模块 | 已 `git submodule update --init LeetCodeDataset`，23 MB，commit `f3519e6`，等于 `run_demo.py` 的 `DATA_REV` |
| `outputs/` | 被 `.gitignore` 的 `**/outputs` 忽略，**永远不会入库**；文档里 22 个指向 `examples/maximum-subarray/outputs/*.json` 的链接要先本地跑 `run_demo.py all` 才有效 |
| GPU / firejail / verl 依赖 | 都没有。本机只能做 CPU 机制主线；真实采样、训练、同步、效果评测都要 Linux + GPU |
| 装包限制（2026-10-15 实测） | uv 管理的 3.12 标记为外部管理（PEP 668），`pip install` 直接拒绝，要补依赖须先建虚拟环境（`uv venv` 或 `python3.12 -m venv`）。PyPI 上 vLLM 0.6.3 与 0.7.3、flash-attn 都没有 macOS arm64 的 wheel；torch 2.4.0、tensordict 0.5.0、ray 2.40.0、pandas、pyarrow、datasets、transformers 有。训练代码在模块开头导入 flash_attn、加载模型写死 `flash_attention_2`、设备网格写死 cuda、分布式后端写死 nccl，所以本机装什么都跑不了训练；flash-attn 2 只支持 Ampere 及更新的 NVIDIA 卡 |
| 文档命令风格 | 2026-10-13 起全部统一为 bash：仓库根目录执行，`$PY -B RLVR学习资料/...`；`$PY` 的定义只在 `README.md` 的 `run-setup` 锚点处给一次。`examples/training/README.md` 里给 Linux GPU 环境的两段 bash 仍用 `python`，其中 `python -m verl.trainer.main_ppo` 会被 `prepare_training.py` 解析，不能改。只有 history 归档保留原 Windows 写法 |

已核验的运行结果（Python 3.12.15）：`run_demo.py all` 退出 0、约 1.6 s；`test_learning_tools.py` 19/19；`test_training_prepare.py` 6/6；reward `[1.1,1.1,0.1,0.1,-1.1,1.1]`、advantage `±0.86602`、infra 回填 `[0.5,0.5,0.5,-1.5]`、pass@k `{1:0.5, 2:0.8333, 3:1, 4:1}` 与文档一致。`run_demo.py` 固定了 verl scorer、`coder1.py`、`core_algos.py` 和原始数据行的 sha256，当前全部匹配。

## 4. 常用命令（在仓库根目录执行）

```bash
PY=/Users/s11437/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12
cd /Users/s11437/Code-R1-reaserch

# 固定演示（写入被 ignore 的 outputs/，可反复重跑）
$PY -B RLVR学习资料/examples/maximum-subarray/run_demo.py all          # 或 prepare|verify|score|advantage|evaluate

# 学生练习：每次提交生成新的 practice_runs/<run>/attempt-NNN/，旧 attempt 不覆盖；退出码 0 匹配 / 1 预测不匹配 / 2 材料或装配错误
$PY -B RLVR学习资料/examples/maximum-subarray/practice.py init --run data-01 --mode seed     # 01/02：自己的测试进 GT
$PY -B RLVR学习资料/examples/maximum-subarray/practice.py check --run data-01
$PY -B RLVR学习资料/examples/maximum-subarray/practice.py score --run data-01 [--bad-format]
$PY -B RLVR学习资料/examples/maximum-subarray/practice.py blind-init --run blind-01 --seed 53  # 02：未知类别故障定位
$PY -B RLVR学习资料/examples/maximum-subarray/practice.py blind-grade --run blind-01
$PY -B RLVR学习资料/examples/maximum-subarray/practice.py transfer-init --run transfer-01     # 04：两数之和跨题迁移
$PY -B RLVR学习资料/examples/maximum-subarray/practice.py transfer-grade --run transfer-01
$PY -B RLVR学习资料/examples/maximum-subarray/mechanism_lab.py --run mechanism-01           # 03：两标量 logits 的 PPO-clip 梯度 toy

# 训练准备（CPU 静态检查；--export-parquet 需要 pyarrow，本机没有）
$PY -B RLVR学习资料/examples/training/prepare_training.py --run prepare-01
$PY -B RLVR学习资料/examples/training/prepare_training.py --run batch-two-ranks --world-size 2 --q 4 --g 4 --mini 4 --micro 2 --tp 1

# 反向补充源码：单元测试与两种 GT 契约对照
$PY -B RLVR学习资料/examples/source-supplements/test_source_supplements.py
$PY -B RLVR学习资料/examples/source-supplements/verify_gt_contract.py
$PY -B RLVR学习资料/examples/source-supplements/verify_test_quality.py   # 补强项 2、3；--all 跑全量约 7 分钟
$PY -B RLVR学习资料/examples/source-supplements/verify_judge_and_selection.py   # 补强项 4 到 7
$PY -B examples/data_preprocess/coder1_plus.py --root_dir ./data   # 补强版数据，JSONL；--parquet 需要 pandas
$PY -B examples/data_preprocess/coder1_plus_reference.py coverage --out_dir data/reference_coverage   # 补强项 11：D12 丢弃清单 + 难度对照（全量约 35 分钟，断点续跑；产物在 gitignore 的 data/ 下）
$PY -B examples/data_preprocess/coder1_plus_reference.py verify --out_dir data/reference_coverage      # 补强项 11：门②参考解行为层验收 → untrusted 清单（读 coverage 缓存，纯 CPU）
$PY -B examples/data_preprocess/coder1_plus_reference.py fetch --problems maximum-subarray   # 补强项 11：取参考解 + 行为层验收
# 补强项 6 整体运行：要 torch，先建 venv（uv 的 Python 禁止直接 pip）：$PY -m venv ~/.venvs/code-r1-cpu && ~/.venvs/code-r1-cpu/bin/pip install "numpy<2" pandas requests "torch==2.4.0" "tensordict==0.5.0" "ray==2.40.0"
~/.venvs/code-r1-cpu/bin/python -B RLVR学习资料/examples/source-supplements/verify_reward_manager_cpu.py

# 实跑推演（Qwen3-8B-Base 两轮 GRPO 的全部产物，确定性生成；用 venv 的 python 才画 PNG 曲线）
$PY -B RLVR学习资料/examples/training/scenario_qwen3_8b.py
$PY -B RLVR学习资料/examples/training/scenario_yourplan.py   # 按用户在 05 前四环的决定生成「你的 run」（runs/qwen3-8b-grpo-yourplan-20261018）
$PY -B RLVR学习资料/examples/training/check_scenario_docs.py   # 叙述里的数字是否与产物一致；改了生成器假设后必跑

# 改过脚本或资料后必跑
$PY -B RLVR学习资料/examples/maximum-subarray/test_learning_tools.py
$PY -B RLVR学习资料/examples/training/test_training_prepare.py
```

## 5. `RLVR学习资料/` 的结构与约定

| 路径 | 职责 |
|---|---|
| `README.md` | 首页：四阶段路线、五项自测、三种完成范围、综合通关条件 |
| `00_一条记录从原始Task到训练.md` | 2026-10-13 新增的直观走读：用 LeetCode 53 一条真实记录走完 ①原始记录→②样本行→③采样→④判分→⑤advantage→⑥loss/更新→⑦同步/验证/保存，附 60 秒口述版与面试追问；已在首页路线和各篇导航里。开头的 `overview-00` 是纵览（七站进出与所在篇、四篇各答什么），文末有 `reproduce-00`（本机复现）和 `progress-00`（学习进度与修改记录）。七站各有锚点 `station-1` 到 `station-7`（2026-10-19 加）。七站与阶段的对应：①② 是 01，④ 是 02，③⑤⑥ 是 03，⑦ 是 04。01 的流水线 2026-11-09 起是三段（取料 / 把关 / 封装；把关含校验、内部审查、外部审查、过滤四道关，旧九步编号已并入），圈号只指七站 |
| `路线评估_仅学习与实跑.md` | 2026-10-15 新增的维护文档：仅学习 vs 实跑三档各要补什么、能学到什么、面试能答到哪；仅学习要补的八项资料清单及状态（RL 公式卡、算法家族对比、训练动态、训练系统、手写题、问题库、话术、03/04 重排）；用户 2026-10-15 已答四问：业务算法岗（后训练与后训练数据构造）、RL 与 LLM 训练都有基础、实跑预算 300 元以内、面试两到三个月后；文档据此定了十二周安排、必做单卡 smoke、新增高优先级的第 9 项「后训练数据构造的横向视角」。以后每补一项资料，要回这份文档改状态列。锚点 `verdict / track-compare / track-study / track-run / interview-map / honest-answers / open-questions / maintenance-log` |
| `讨论结论库.md` | 2026-11-10 建库：讨论产出的**无损母本**（表格/plain 图/本质句原样收录，每条带定案关联与收录状态）；后续并入学习资料从此库取材，搬完标「已收录 → 指针」。与设计定案表分工：定案表记压缩结论（核对基准），本库记完整形态输出（可直搬母本）。只增不改历史条目 |
| `测试扩容方案_v1.md` | 2026-11-09 收录：每题奖励测试的覆盖/判定质量/成本三组指标、覆盖清单先行、分级预算（40-80/200/1000 候选）、带预算的扩容闭环、独立审计分阶段、成本入验收、阶段 A（7 题试点）→B（100 题）→C（分批）→D（训练复审）。开头有核对与落地状态表：判分归因修正已落地（大输入答错=正确性失败 0.1，`verify_perf_round.py` 18/18）；「13 条放过错误初始化」核对为与现状不符（现行 13 条已抓住，被放过的是 3-seed 时期）；阶段 A 待实施，实施位置按方案 §9 职责表。定案登记 D13 |
| `05_实跑推演_Qwen3-8B.md` | 2026-10-16 新增：Qwen3-8B-Base（预训练模型，没经过 SFT 或 RL，R1-Zero 式起点；用户 2026-10-16 明确；架构参数与 Qwen3-8B 相同）两轮 GRPO 的实跑推演，七环（机器、数据、框架搭建、参数配置、曲线解读、评测、归因优化）各有「你来决定」和折叠的推演答案；锚点 `overview-5 / guide-5 / stage-machine / stage-data / stage-framework / stage-config / stage-curves / stage-eval / stage-attribution / homework-5 / interview-5 / provenance-5 / record-5`。按用户决定生成的「你的 run」在 `runs/qwen3-8b-grpo-yourplan-20261018/`（`scenario_yourplan.py`；2026-10-19 去掉 TACO 后改为两轮：your-01 诊断 → your-02 按诊断改三处续训 → held-out 验证；有 rollout 落盘、wandb、checkpoint 清单，控制台无逐样本判分块），决策日志 `runs/qwen3-8b-grpo-20260928/notes/your_decisions.md`（第八节是去 TACO 后的主线），第二轮记录 `notes/your-02_plan.md`。**它是推演不是实测**：产物目录 `examples/training/runs/qwen3-8b-grpo-20260928/`（启动脚本、展开配置、控制台日志、metrics.jsonl、wandb 目录、checkpoint 清单、评测明细、曲线图、决策日志、事故报告），由 `examples/training/scenario_qwen3_8b.py` 生成（确定性，不依赖 PYTHONHASHSEED），`PROVENANCE.md` 逐项写明哪些是真实输出（第 1 步 256 条判分日志、parquet sha256、去污染结果、判分探针）、哪些是合成。2026-10-16 经 4 个评审视角复核后修正：v0.4.1 的指标名（`actor/entropy`、`training/*`、`timing_s/dump_rollout_generations`、`perf/*memory*`）、prime 管理器的打印（`[Error] Task failed`、`[Shutdown]`）、`train_batch_size >= ppo_mini_batch_size` 与 `max_num_batched_tokens >= prompt+response` 两条断言、run-02 需要 4 个 epoch、第一步 lr 在 v0.4.1 下不是 0、grad_norm 不存在 1/16 缩放、推演日期设为 09-27 到 10-09，机器按用户要求改为个人研究量级的 4×A800；起点改为 Qwen3-8B-Base 后两轮与 your-01 重生成（n=16、320 步的 Code-R1 配方在 4×A800 上 76.1 小时、¥3,043 一轮；run-02 100.6 小时、¥4,023；your-01 200 步 21.2 小时、¥847，去 TACO 后 your-02 续训 26.1 小时、¥1,043；Base 的 reward 从负值起步，前 64 步增益约八成是格式，采样 pass@8 也上升，不要再写「基线强、只锐化」）。改假设重算后要跑 `check_scenario_docs.py` 并手工同步叙述；不要把它写成实跑 |
| `bad_case调查方法论.md` | 2026-11-12 新增：bad case 双向调查方法（D20）——上游→下游推演四步（核心机械动作＝列消费者）、可观测出口四类×分级三级、口述纪律（发现顺序≠叙事顺序、开口必须真实下游观测、反事实收第二层）与统一模板，附上标 105 完整演练；卷首与 bad_case登记表的两条铁律互指。2026-11-13 补**步骤 0：固定原始源材料**（D23，推演只读原始上游，修复版仅验证）与推演产物七字段 |
| `Code-R1_面试口述稿.md` | 2026-11-13 新增：8–10 分钟面试口述稿，十段（开场/采集/清洗/模型输入/GT/Verifier/GRPO/评测/归因/收尾），每段=完整 plain 图截取+口述+过渡句；含 30 秒精简版与「可能被打断的追问」表；已同步飞书知识库 RVLR 节点 |
| `bad_case登记表.md` | 2026-11-08 新增：训练归因视角的已知判分异常库（BC-01~25，数据/判分器/管理器/沙箱/策略配置五层+爬虫/命名空间两类，每条五要素：症状签名/根因/验证/口径依赖/状态）+ 症状速查表 + 预检与维护规则。归因铁律：reward 异常先用参考解复现——参考解同分即环境实锤。2026-11-14 补「原始版本锚点」表（D23）并新增 BC-25 长度上标唯一口径。锚点 `layer-data / layer-judge / layer-manager / layer-sandbox / layer-policy / symptom-index` |
| `专题索引_任务构造与沙箱.md` | 2026-10-19 新增的位置索引：Task 选取/筛选/构造、GT 构造、expected/oracle/测试扩展、Sandbox 四个专题各在 00/01/02/主框架/补强项 README/源码的哪里（先读 → 深读 → 补强版 → 代码与工具 → 本课的数字），锚点 `topic-task / topic-gt / topic-oracle / topic-sandbox / maintenance-index`。只做索引不另写内容；改动相关资料时回来改链接和维护记录 |
| `RLVR工业流程.md` | 主框架（1059 行）：按机制查阅。锚点 `mvp-overview / running-example / task-preparation / tests / gt-packaging / verification / execution-boundary / reward-mapping / grpo-advantage / gradient-bridge / policy-update / evaluation / answer-validation / layer-attribution / historical-evidence`；2026-10-19 另加小节锚点 `a1-preparation / a2-filtering / a4-answers / a-reference-answers / verifier-internals`（A7 用原有的 `verification`） |
| `01_数据理解与GT构建.md` … `04_真实训练与独立评测.md` | 四个阶段，每阶段三项作业 + 折叠的「面试深挖」表；锚点 `assignment-N-M`、`interview-N`、`prerequisites-N`、`graduation`、`deeper-topics`；01 另有 `overview-1`（总览）、`guide-1`（文字版导读）、`pipeline-1`、`internal-external-review`、`metadata-fields`、`actor-prompt`、`tools-1`、`oral-1`、`source-reading-1`、`record-1`。01 已于 2026-10-14 重排为「总览、文字版导读、概念四节、动手前与三项作业、60 秒口述与面试深挖、按需读源码、完成记录」；02 已于 2026-10-15 按同样结构重排，锚点 `overview-2`、`guide-2`、`prerequisites-2`（三个角色）、`pipeline-2`（判分七步）、`reward-levels-2`、`execution-gaps-2`、`task-pool-2`、`tools-2`、`oral-2`、`source-reading-2`、`record-2`，原有 `assignment-2-1/2/3`、`interview-2` 保留；03、04 已于 2026-10-19 按同样结构重排：03 的锚点 `overview-3`、`guide-3`、`prerequisites-3`、`sampling-3`、`advantage-3`、`gradient-bridge`、`runtime-facts-3`（mini-batch 语义、ratio 恒 1、梯度缩放、首步 lr=0 与 v0.4.1 差别）、`tools-3`、`oral-3`、`rollout-real-3`、`source-reading-3`、`record-3`；04 的锚点 `overview-4`、`guide-4`、`evidence-4`、`metrics-4`、`protocol-4`、`update-4`、`attribution-4`、`tools-4`、`oral-4`、`source-reading-4`、`record-4`，原有 `assignment-N-M`、`interview-N`、`graduation`、`deeper-topics` 保留；04 把 05 的第六、七环当沙盘引用。主框架另有 `test-strength`（A5 错解表），首页有 `run-setup`（环境设置） |
| 六篇的「传统 CS 知识补充」章 | 2026-11-04 新增：每篇末尾锚点 `cs-00`、`cs-1`～`cs-5`，分组「定义+人话」表 + 统一生活场景 + ★易混标注 + 串讲图 + 读图自测；同一术语只在首次出现的篇详写、后续篇链接；定义是通识说法不挂源码版本，CS 章引用的数字均取自各篇正文已记录内容（05 的 CS 章全部标推演口径） |
| `examples/maximum-subarray/` | 贯穿题 LeetCode 53（v2 RL 第 143 行，`Solution().maxSubArray`）。`run_demo.py` 固定 8 接受/7 隔离演示；`practice.py` 学生练习；`mechanism_lab.py` 梯度 toy；`EXERCISES.md` 操作与退出码；`ANSWERS.md` 提交后才看的答案与评分表；`validation_summary.json`/`bundle_validation.json`/`math_validation_*.json` 是作者的工具验证记录 |
| `examples/maximum-subarray/practice_runs/validation-*` | 作者的工具测试产物（已入库）。**不是学生成绩**，不要删、不要当作自己的 attempt |
| `examples/training/显存帐_Qwen3-8B.md` | 2026-10-16 应用户要求建立的显存帐：Qwen3-8B-Base（架构常数与 Qwen3-8B 相同）的常数、公式、各设置（卡数、vLLM 占比、micro、回答上限、梯度检查点、offload）的峰值显存与速度、质量影响；用户的方案是 4×A800 80GB。每改一个设置加一行，真机实测后把推算换成实测 |
| `examples/training/` | `prepare_training.py`：4 train / 2 dev 的真实来源小池（JSONL，`train_ready=false`）+ batch 算术 + 复用源码 `_validate_config`；`README.md` 含 Linux 单卡 smoke/恢复命令（本机未跑）；`runs/validation-*` 作者产物 |
| `examples/source-supplements/` | 反向补充源码的登记处（README 的补强项各有锚点 `supplement-1` 到 `supplement-10`；训练 README 的 Linux 验收一节是 `linux-acceptance`；01 另有 `schema-1`、`error-layers-1`）：补强项清单与状态、每项的问题、改法、代价和验证结果；`supplement_env.py` 在无 torch 的 CPU 环境加载真实判分包，`verify_gt_contract.py` 做 2090 题两种契约对照，`test_source_supplements.py` 是单元测试，`verify_perf_round.py` 验证补强项 10 的标定链与性能轮四对照 |
| `examples/history-20261003/` | 旧五题核验的只读归档。**不要运行 `audit.py`**（硬编码 `D:/` 路径，会覆盖已入库的 `audit.json`） |

资料内的证据标签：`[Code-R1 实现]` 当前源码行为、`[通用设计]` 扩展目标；本机实跑的结果写明日期，如「2026-10-13 实测」。旧的 `[2026-10-03 历史核验]` 标签已随旧五题例子一起移出正文，旧核验只在主框架第七章以「历史归档」一段指向 `examples/history-20261003/`。

相对链接都指向显式 `<a id="...">` 锚点或真实文件：作者 `bundle_validation.json` 记录 360 个，2026-10-15 复查 508 个、0 错误。可能失效的只有两类：未初始化的子模块（本机已初始化），以及没先跑 `run_demo.py all` 时的 `outputs/`。

## 6. 源码地图（2026-10-13 逐条核对，基线 `443f8da`）

这一节是查阅工具，服务于第 2 节的优先级：回答时先讲机制，需要证据时再来这里找「文件 + 函数 + 行号」。不要求用户背诵。

**数据 → GT**（`examples/data_preprocess/coder1.py`）
- 活跃入口 `dataset_makes=[leetcode2k, taco]`（`:359`）；`codecontests()` 未调用且有 bug；`kodcode.py` 单独存在但两份启动脚本都不读它。
- `leetcode2k()`（`:275-341`）：合并 v2 RL + v2 SFT，排除 v2 test 的 `question_id`，按 ID 保留首次；prompt = 固定一句话 + `meta.query`；**GT = `json.dumps({"functional": test + "\n\ncheck(" + entry_point + ")"})`**，不拼 imports；`extra_info.reference = completion`；不执行验证。
- `taco()`（`:59-205`）：`likaixin/TACO-verified`，跳过 geeksforgeeks/leetcode 来源、短题面、含图片题；functional 用 `_deep_eq`（浮点 1e-5）；stdio 经 `minimize_stdio`（≤8 条、跳过 >4 KB stdin、`zip` 会静默截断）；**验证用 `solutions[-1]`，保存的 reference 是 `solutions[0]`**；`:201-203` 给 test 行设 `split="test"` 的循环写在临时字典上，无效。
- 行 schema：`data_source="code"`（LeetCode/TACO 都是）、`prompt=[system, user]`、`ability="coding"`、`reward_model={"style":"rule","ground_truth":<json str>}`、`extra_info={split,index,prompt,reference,dataset}`。
- **LeetCode 的 `test` 全部就是题面样例**：1890 题每题断言数都等于 `examples` 数，中位数 2、最多 5、93% 不超过 3 条。数据集 README 说明样例解析成测试，并与 doocs/leetcode 题解交叉核对；Code-R1 README 说的 generally reliable 指答案对，不是覆盖够。
- 输出目录 `data/code-r1-{N}k-leetcode2k-taco/`，而 `main_grpo.sh` 读 `data/code-r1-12k/`：路径不自动匹配。
- **GT 不拼原始 `prompt` 前缀**：1890 道 LeetCode 训练题中 121 道树和链表题的 `test` 调用只在前缀里定义的 `tree_node`/`list_node`/`is_same_tree`/`is_same_list`，在 Code-R1 下必然 `NameError`，正确解也只拿 0.1（2026-10-13 用 `create-binary-tree-from-descriptions` 对照实跑：Code-R1 拼法失败、原生拼法通过；v2 test 0/200 受影响）。已写入 01 的 `internal-external-review` 小节。

**Verifier / reward**
- 路由：`verl/utils/reward_score/__init__.py:33-35` 把 `"code"` 交给 `coder1.compute_score`。
- `verl/utils/reward_score/coder1/__init__.py`：`validate_response_structure`（`:37-39`，strip 后必须以 `<think>` 开头、`</answer>` 结尾）；`extract_code_from_string`（`:57-60`，取**最后一个** `<answer>` 内**所有**代码块拼接；原生 `leetcote/data.py:extract_completion` 取最长块，两者不同）；`_compute_score`（`:63-127`）：格式/无代码 **-1.1**，格式合格但执行失败 **0.1**，执行成功 **1.1**；functional 把 `solution + "\n" + gt["functional"]` 当一个程序跑，成功 = 退出码 0；stdio 比较 `output.strip() == stdout.strip()`，首个失败即返回；`extra_info.reference` 不参与判分。
- 执行后端：`CODER1_EXEC` 环境变量，默认 `firejail`（`:14-29`），可选 `docker / ces / kira`；只有 firejail 支持 `pytest=`。`firejail_exec.py:38-49` 限额 `--rlimit-nproc=32 --rlimit-nofile=32 --rlimit-fsize=2m --rlimit-as=4096m --timeout=00:00:30`，成功仅看 `returncode==0`（`:95-96`）→ `sys.exit(0)` 提前退出得 1.1 的缺口由此而来；docker 后端不断网；kira 不看退出码只看 stderr。
- `verl/workers/reward_manager/prime.py`：线程池 `cpu_count()`，单样本超时 600 s（`:64`），超时/异常回填 **0.0**（`:81-83`），全局异常则整批 0.0（`:135-140`）；只解码 response（`skip_special_tokens=True`）；**标量写在最后一个有效 response token**（`:144`）。`single_compute_score` 的参数名 `reference` 实际装 ground_truth。`naive.py` 解码 prompt+response，会让 coder1 的 `<think>` 开头检查永远失败（推断，未跑）。
- verl 自带第二套判分器 `verl/utils/reward_score/prime_code/`：APPS 风格，先整体跑、失败再逐条跑，`continuous=True` 时返回前 10 条的通过率；进程内 `reliability_guard` 加 `signal.alarm`，不是隔离沙箱。它只接 data_source 为 taco、codecontests、apps、codeforces 的数据；Code-R1 的数据是 `code`，不走它。
- **补强版 `coder1_plus`（2026-10-14 新增，原版预处理和判分文件不改）**：数据侧 `examples/data_preprocess/coder1_plus.py`，纯标准库生成 JSONL，加 `--parquet` 写 parquet；prompt、筛选、`index` 与原版相同，GT 多一个 `prelude` 字段存原始前缀。判分侧 `verl/utils/reward_score/coder1_plus/`，按「前缀、候选代码、测试」执行；没有 `prelude` 的 GT 原样交给原版。对原版文件只有两处改动，都只增不改：路由 `verl/utils/reward_score/__init__.py` 的 `code_plus` 分支，训练入口 `verl/trainer/main_ppo.py` 的 `prime_plus` 奖励管理器分支。实测（2026-10-14 复核）：依赖前缀辅助函数的 121 题从 0.1 全部变成 1.1；补强版下训练集 1877/1890、验证集 199/200 通过，剩下 14 题全部缺第三方库（13 题 `sortedcontainers`、1 题 `numpy`）；`word-ladder-ii` 原本随哈希种子时过时不过，补强项 4 改成忽略外层顺序比较后稳定通过；题面写「in any order」的有 68 题，测试却比较列表顺序，由补强项 4 的 `compare=unordered_outer` 处理。
- **补强项 2、3（2026-10-14）**：`examples/data_preprocess/coder1_plus_tests.py` 提供变异审计 `audit_test_strength` 和逐条审核 `audit_new_cases`、`extend_test`；`coder1_plus_oracles.py` 收录 7 道题的独立 oracle、合法输入生成器与校验器；`coder1_plus.py --augment_cases` 只追加 accepted 用例并把 `test_version` 改成 `leetcode-examples+augN`。全量审计（补强项 4 上线后复跑）：2014 题、17316 个变异体，样例测试杀死 81%；杀死率高不等于测试够，LeetCode 53 杀死 9/9 仍放过 wrong_empty。
- **补强项 4 到 7（2026-10-14）**：`coder1_plus/protocol.py` 受保护判题协议，期望答案不进候选进程，子进程逐条报告输出、判分方比较，2090 题的断言只有三种写法、全部覆盖，GT 的 `compare=unordered_outer` 忽略外层顺序，`word-ladder-ii` 手工登记；`coder1_plus/sandbox.py` 结构化执行层，状态分类、耗时、超时杀进程组，后端由 `CODER1_PLUS_SANDBOX` 选，macOS 用 sandbox-exec 禁网、限写、禁读仓库，Linux 用 firejail；macOS 上 `preexec_fn` 设 RLIMIT_CPU 会立刻 SIGXCPU，所以只在 Linux 设。`verl/workers/reward_manager/infra_policy.py` 加 `prime_plus.py`：判题故障先重试，仍失败**单条掩码**（2026-11-09 D15：故障样本分数=组内有效均值，advantage 恰 0，不浪费其余 rollout；全故障组作废；`CODER1_PLUS_INFRA_POLICY=mask_sample|neutralize_group|legacy_zero`，默认 mask_sample）。`examples/data_preprocess/coder1_plus_selection.py`：评测集去重与按通过率筛题，与 v1 test 去重找到 61 对同题号、3 对措辞重叠。2026-10-15 `verify_reward_manager_cpu.py` 在 macOS CPU、torch 2.4.0 上整体跑通 prime_plus（三种组合全过）；firejail 分支和接入真实训练循环待 Linux 实跑，难度筛选需要 GPU 采样。
- **补强项 9（2026-11-06）**：`coder1_plus` 奖励分值字典序修正——正确性 1.0/0、格式 ±0.1，严格标签缺失时退回整段响应、抽取其中全部完整代码块拼接照样执行；四档 1.1/0.9/0.1/−0.1，任何对答严格高于任何错答（修掉原版格式边际 1.2 > 正确边际 1.0 的倒挂）。`CODER1_PLUS_REWARD=legacy` 复现原版口径；无 `prelude` 的 GT 仍委托原版、分值不变；两个推演生成器已钉 legacy（重算与 git 基线逐行一致）。CPU 验证：单元测试、`verify_judge_and_selection.py` 的 `reward_scheme` 三检查、`verify_reward_manager_cpu.py` 的 `[1.1, 0.1, 0.1, 0.9]`。
- **补强项 10（2026-11-07；归因 2026-11-09 修正）**：性能轮——每题时间复杂度约束接入判分器。数据侧 `examples/data_preprocess/coder1_plus_perf.py` 离线一次性标定（Constraints 解析规模上界，上标丢失 `105`→10^5、`10**5`、`5*104` 三种写法都认；参考解实测墙钟×3 得 `time_limit_ms`，下限 500ms），`coder1_plus.py --perf_index` 把标定并进数据行 `extra_info.perf`（含确定性种子与期望输出）；判分侧 `coder1_plus/perf.py` 在正确性轮通过后加跑大输入（种子确定性重建，`sandbox.run_python` 每题 timeout）。**归因（2026-11-09）**：超时记性能瑕疵扣 0.2（瑕疵并罚：1.1/0.9/0.7）；**大输入输出不符/出错记正确性失败 0.1 档（`failed_perf_mismatch`）**——大输入是题意合法输入，答错即错解，留在 0.9 会让崩溃错答与慢而对打平、破坏「任何对答严格高于任何错答」。错答不触发；`legacy` 整体跳过；infra 故障走补强项 6 的单条掩码。空间复杂度只做 `as=4096m` 限额护栏不设奖励档。设计依据：CodeContests 每题 `time_limit`/`memory_limit_bytes` 数据字段（contest_problem.proto）、Codeforces Polygon 预期解×系数标定+慢解负对照。`sandbox.py` 同步补 Windows local 后端兼容（显式选择才可用，auto 仍不静默降级）。验证：`verify_perf_round.py` 18/18（2026-11-09 增「大输入答错 → 0.1」对照，macOS 复验；原 15 项 2026-11-07 Windows+local 通过），既有套件 27/27。第一版只支持恰一个 `List[int]` 参数的题，树/链表/stdio 题与全量 1890 题标定待做。**D1 管线落地（2026-11-08）**：`coder1_plus.py --append_constraints` 从 `question_title` 解析 Constraints 规范化后追加进 user prompt（1846/1890 成功，默认关闭保持与原版逐字相同的单变量对照）；prompt 三条边界等讨论定案汇总在 `RLVR学习资料/设计定案表.md`（D1–D10，改 md 前后的核对基准）。
- **补强项 11（2026-11-09）**：社区参考解获取与 D12 过滤（定案见 `RLVR学习资料/设计定案表.md`）。数据侧 `examples/data_preprocess/coder1_plus_reference.py`：`coverage` 从 doocs/leetcode 目录分片 + 逐题 README 判「有无 Python3 解」，产出 `drop_reference_missing.json`（JSON list，给 `coder1_plus.py --drop_task_ids`）与难度分布对照（丢弃非随机要计量）；`coder1_plus.py --drop_task_ids` 同日起支持多个 JSON 文件合并（nargs='+'，D12/D14 两份清单一起传）；`fetch` 取参考解并按原生拼法在补强沙箱做行为层验收（冻结 completion 对照组；无 torch 环境经 supplement_env 包桩加载沙箱）。全请求限速 1s、成功才缓存、断点续跑；题号 ≥100000 的 lcci 题记 `lcci_out_of_doocs_range` 丢弃。2026-11-09 实测：1868 道算法题 doocs 文件夹级全覆盖、Python 解抽样 32/40≈80%；4 题 6 解行为层验收全过，3/4 题「方法一」与冻结 completion 相似度 ≥0.99（上游出处即 doocs）。`verify` 子命令（D14 门②，2026-11-09）：对 coverage 缓存的每题做参考解行为层验收（any 语义，记录哪份方法通过），产出 `drop_reference_untrusted.json` 与 `verify_report.json`；对照组（仓库 completion）也失败判 `env_suspect` 不自动丢（BC-24 铁律：参考解同分即环境实锤）。单元测试 27/27（新增 4 条）。定案 D14（入池质量门①②③④）见 `RLVR学习资料/设计定案表.md`；门③④（覆盖度）接测试扩容方案（D13），清单同走 drop/keep 机制。学习侧探针 `RLVR学习资料/examples/upstream_probe/reference_probe.py`，快照在 `upstream_probe/doocs_snapshot/`。
- **三条讨论定案已收录资料（2026-11-07）**：prompt 装什么的三条边界（描述给全/约束该给/样例永不给——test=样例即判分依据泄露）写进 01 的 `actor-prompt`、主框架 `task-preparation`、专题索引；判分器一对多架构（通用判分器+题目特异性走数据字段，CodeContests/Polygon 同构）写进 02 的 `reward-levels-2`；02 新增 CS 补充章 `cs-2`、其余五篇各建 `cs-00/cs-1/cs-3/cs-4/cs-5`（按 Shopping-Agent 01 第十一段方法，术语首次出现详写、后续互链）。
- `coder1` 没有任何自动化测试；`tests/sandbox/` 只测 prime_code/prime_math；`scripts/test_sandbox.py`、`scripts/stress_exec.py` 是手工冒烟。

**训练主线**（`main_grpo.sh` → `verl/trainer/main_ppo.py`）
- `main_grpo.sh`：`adv_estimator=grpo`、`train_batch_size=16` 题、`rollout.n=16`、`ppo_mini_batch_size=GPUS*8*4`、`micro=8/GPU`、`use_kl_loss=True kl_loss_coef=0.001 kl_loss_type=low_var_kl`、`lr=5e-7`、`reward_manager=prime`、`max_prompt/response=2048/4096`、`save_freq=64 test_freq=16`；未覆盖的默认值：`entropy_coeff=0.001`、`clip_ratio=0.2`、`grad_clip=1.0`、`ppo_epochs=1`、`temperature=1.0`、`rollout.tensor_model_parallel_size=2`。
- `ray_trainer.py::fit()`（`:862-1040`）顺序：`_load_checkpoint` → 训练前 `_validate` → `generate_sequences` → 每题一个 `uid` + `repeat(n, interleave=True)` + `union` → `_balance_batch`（总是执行，按 token 数重排）→ 重算 `old_log_probs` → `ref_log_prob` → `reward_fn` 得 `token_level_scores` → `use_kl_loss=True` 时 `token_level_rewards = token_level_scores`（`:978-984`，`apply_kl_penalty` 与 `kl_ctrl.kl_coef` 无效）→ `compute_advantage`（driver 上）→ `update_actor` → 更新**之后**按 `test_freq` 验证 → 按 `save_freq` 保存。
- `core_algos.py::compute_grpo_outcome_advantage`（`:111-154`）：score = token reward 求和，按 `uid` 分组；`torch.std` 样本标准差（N-1）；单例组 mean 0 / std 1；`(s-mean)/(std+1e-6)`，广播到 response 并乘 `eos_mask`。`compute_policy_loss`（`:272-303`）：`ratio=exp(new-old)`，`max(-A r, -A clamp(r,1±0.2))`，`masked_mean`；`kl_penalty`（`:351-383`）支持 `kl/abs/mse/low_var_kl`，`full` 抛 `NotImplementedError`。
- `verl/workers/actor/dp_actor.py`：`_forward_micro_batch` 先 `logits /= temperature` 再 log-softmax，普通路径截 `[:, -R-1:-1]`，remove-padding 路径先 roll 再还原；`compute_log_prob` 在 `eval()` + `no_grad`；`update_policy` 每 mini-batch 一次 optimizer step，每 micro 内 token 均值再除以 `gradient_accumulation`（`:260, :312`），`_optimizer_step` 做梯度范数裁剪。
- `verl/workers/fsdp_workers.py:119-120`：`ppo_mini_batch_size *= rollout.n; //= world_size//sp`（mini 的单位是**题数**）。ref 模型固定 CPU offload + bf16，不受 `ref.fsdp_config.param_offload` 影响。
- 权重同步 `verl/workers/sharding_manager/fsdp_vllm.py`：每次 `generate_sequences`（训练与验证）都进入 `__enter__`：取 actor `state_dict()`，旧 vLLM 走 `sync_model_weights`，新 vLLM 走 `wake_up + load_dtensor_weights`；`__exit__` offload/sleep 并恢复训练 RNG。rollout 文件按 vLLM 版本选择（`<=0.6.3` 用 `vllm_rollout.py`，否则 `vllm_rollout_spmd.py`）；README 装 0.7.3 而 `setup.py` 钉 `<=0.6.3`，两处不一致。
- rollout 返回的 `DataProto` 只有 token 序列，**不保留 finish/stop reason**；`get_eos_mask` 含首个 EOS，无 EOS 全 1。
- checkpoint（`fsdp_checkpoint_manager.py`）保存模型分片 + optimizer + `lr_scheduler` + RNG（cpu/cuda/numpy/random）+ `data.pt`（dataloader）；**不保存** sharding manager 里的 vLLM 生成 RNG；ref 模型从 `model.path` 重载。
- 评测：`_validate()`（`:609-690`）`do_sample=False` → vLLM `temperature=0, n=1` 的贪心单回答；`val/test_score/{data_source}` = 每条 reward 求和后 `np.mean`，是 raw reward 均值不是通过率，LeetCode/TACO 合并在 `code` 下。原生 `LeetCodeDataset/leetcote/evaluate.py::estimate_pass_at_k` 读 release 题库、要求覆盖全部题，不读本课 GT。

## 7. 已核对、但资料尚未收录的源码事实（候选补充，并入前请用户复核）

来自 2026-10-13 的源码阅读，均**未实跑**：

1. `main_grpo.sh` 的 mini-batch 语义错配：脚本按「回答数」算 `GLOBAL_BATCH_SIZE=256`，verl 按「题数」再 ×16 → 每 GPU mini=512，而每 GPU 只有 32 条回答 → 每步只有 1 个 mini-batch，`gradient_accumulation=512//8=64` 但只有 4 个 micro → 梯度被缩到 micro 均值的 1/16（因子 = `train_batch_size/ppo_mini_batch_size`）。AdamW 基本抵消常数缩放，但 `grad_clip=1.0` 实际放宽 16 倍，日志 `actor/grad_norm` 偏小 16 倍。课程 `examples/training/README.md` 的小配置（Q=4,G=4,M=4,world=2）恰好一致，所以作者没有碰到这个问题。
2. 第一步 optimizer 的 lr=0：`lr_warmup_steps_ratio=0` → `lr_lambda=min(1, step/max(1,0))` 在 step 0 为 0（`torch_functional.py:455-464`）；`actor/lr` 在 `scheduler.step()` 之后记录，显示的是下一步的 lr。
3. 实际只执行 `total_training_steps-1` 次更新（计数从 1 起，`>=` 即停，`ray_trainer.py:891, :1027-1029`）。
4. `ppo_epochs=1` 且单 mini-batch 时，update 用的权重就是算 `old_log_probs` 的权重 → ratio≈1、`pg_clipfrac≈0`，PPO clip 实际不起作用（课程 03 的 clip 示例是机制教学，不是本脚本的运行态）。
5. `entropy_coeff=0.001` 默认开启，`main_grpo.sh` 未关闭。
6. loss 聚合是每个 micro 内 token 均值再累积；advantage 为 0 的组仍占分母。
7. `_balance_batch` 重排后按 `uid` 值分组仍正确，但 `actor/*` 指标只来自 rank 0（`concat` 保留第一个 `meta_info`）；`actor/kl_loss` 每 micro 覆盖、`actor/grad_norm` 只记最后一个 mini-batch。
8. `update_policy` 不 shuffle mini-batch（`actor.shuffle` 未使用）；`_optimizer_step` 不检查非有限梯度。
9. `practice.py:328` 对所有子命令（含不需要数据内容的 `blind-init/blind-grade`）都先做源码/数据指纹检查，子模块缺失时它们也会退出 2。
10. `tests/sandbox/test_sandbox.py::test_parallelism` 在本 fork 的 `prime.py` 签名下会因 `zip(..., None)` 报 `TypeError`（2026-10-15 已复现）。
11. （2026-10-15 可跑性评估）单卡时 `fsdp_workers.py` 把 `load_format` 改成 `dummy_hf`，权重同步走 `hf` 格式；vLLM≥0.7 的 `fsdp_vllm.py::__enter__` 只实现 `dtensor`，遇到 `hf` 直接 `NotImplementedError`，所以「单卡 + README 装的 vLLM 0.7.3」第一次生成就失败，单卡要用 0.6.3（源码推断）。已写进 `examples/training/README.md` 的「实跑前必查」。
12. `vllm_rollout/__init__.py:28` 用字符串比较 `package_version <= '0.6.3'`，`third_party/vllm` 只认精确版本：`0.6.3.post1`（仓库 `docker/Dockerfile.ngc.vllm` 装的就是它）会被分到 spmd 分支再报版本不支持；没装 vLLM 时报 TypeError。
13. Hydra 覆盖值含非 ASCII 字符（如路径里的「学习资料」）不加单引号会 `LexerNoViableAltException`（2026-10-15 在 venv 里实测）；训练 README 的 smoke 命令已改成加引号。
14. ces、kira 后端在服务不在时 `while True` 每 3 秒重试、永不返回；管理器 600 秒后记 0.0，但线程池退出时会一直等这些线程，训练会卡住（源码推断）。docker 后端：stdin 很大时引用未定义的 `stdin_path`，把代码包在单引号里交给 `sh -c`，代码含单引号就坏；不关网络，stdout 与 stderr 混在一起。
15. `coder1_plus/sandbox.py` 在线程池里用 `preexec_fn` 设资源上限，Python 文档说明多线程下 `preexec_fn` 不安全，理论上可能死锁（推断，未观察到）。装依赖的 venv 必须在仓库外：sandbox-exec 禁止读仓库，解释器在仓库里会被拒绝执行，全部样本变成判题故障。
16. 仓库自带 tests、examples 里会坏的：`tests/e2e/run_ray_trainer_rmpad.sh:11` 和 `examples/rloo_trainer/run_qwen2-7b.sh:22` 覆盖了配置里没有的键（推断 Hydra 拒绝）；`examples/slurm/ray_on_slurm.slurm:25` 的数组写法把所有主机名塞进一个元素（已复现）；`tests/e2e/arithmetic_sequence/rl/README.md` 的命令指向不存在的模块；`main_eval.py` 只支持 `lighteval/MATH`，代码题的 pass@k 评测要自己写；`tests/ray` 的资源池默认要 GPU，Mac 上会卡住。
17. （2026-10-15 复核员实测）装了 CPU 依赖的 venv 里还能多跑：`tests/ray` 中只有 CPU 逻辑的 4 个测试，设 `RAY_OVERRIDE_RESOURCES='{"CPU":48,"GPU":8}'` 声明逻辑资源后全部通过；`tests/gpu_utility/test_torch_functional.py::test_lr_scheduler` 在仓库外放一个 `flash_attn/bert_padding.py` 桩后通过，它断言 warmup 第 0 步 lr=0；LeetCodeDataset 自带的 `leetcote` 评测（eval_lct）纯 CPU，v2 test 参考解 pass@1=0.995，可作代码题 pass@k 评测的现成办法，但它在输入文件旁写结果、要求覆盖整个 split、用原生拼法（带前缀）；作者在 HF 发布的 `ganler/code-r1-12k` 文件名是 `data/train-00000-of-00001.parquet`（12458 行）和 `test-…`（712 行），要改名才能给 `main_grpo.sh` 或 `--merge_taco_from` 用。

## 8. 修改 `RLVR学习资料/` 的规则（服从第 2 节的优先级）

- **贯穿题只有一道：LeetCode 53「最大子数组和」**（v2 RL 第 143 行，`Solution().maxSubArray`）。所有机制例子、错解、反例都用它；不要再引入服务器题或旧五题。例外只有三个，且都是设计使然：04 的「两数之和」跨题迁移练习、`examples/training/` 的多题小池、说明全库问题时引用的统计（如 121 道树和链表题）。
- **全局视角**：看或改任何一篇（00、01–04、主框架、首页、examples 文档）时，同时检查与其他各篇是否一致：术语、字段计数、候选名单与分数、关键数字（8 接受 / 7 隔离、`[1.1,1.1,0.1,0.1]`、±0.866、pass@k、1890、121、61、3835）、命令写法和互链。发现冲突就地统一，并在 00 的修改记录表记一行。主框架负责机制，01–04 负责各阶段展开，00 负责主线，同一内容只在一处详写、其余处链接过去。
- **机制与可读性优先**：允许简化、合并、删掉重复的边界警告，简化处标一句即可。证据标签（`[Code-R1 实现]` / `[通用设计]` / `[2026-10-03 历史核验]`）保留，但不必每句都挂。推断写推断，实跑写日期/平台/版本。
- **面试导向的结构**：每个机制保留「一句定义 + 一个数字例子 + 一个反例 + 一个源码定位」；各阶段的「面试深挖」表和 README 的五项自测优先维护；新增内容先问「口述时怎么讲」。
- **源码一致性放最后**：资料文字可以与源码细节有出入，只要标注。`run_demo.py` 锁定了 `verl/`、`examples/data_preprocess/coder1.py`、`LeetCodeDataset/` 的 sha256，这是演示脚本的约束，不是资料内容的约束；不要为了资料去改这些源码。若确需改源码，同步更新 `SOURCE_HASHES`/`DATA_REV`/`RAW_LINE_SHA`。
- `examples/history-20261003/` 只读；`practice_runs/validation-*` 与 `training/runs/validation-*` 是作者产物，不删不改；学生 run 用新名字（`data-01`、`blind-01`、`transfer-01`、`mechanism-01` …）。
- `outputs/` 永远不入库；不要为了修链接去取消 ignore，而是在文档里说明「先跑 `run_demo.py all`」。
- 新增或改动链接只用显式 `<a id="...">` 锚点；改完 md 后做一次相对链接 + 锚点复查（作者基线：11 个 markdown、360 个本地链接、0 错误；2026-10-14 复查：461 个本地链接、0 错误；2026-10-15 重排 02 后复查：508 个、0 错误）。复查脚本要按 CommonMark 识别代码围栏：00 第 ④ 站的示例回答里有一行以 ```` ```</answer> ```` 开头，它在代码块内部、不是闭合围栏，按「行首三个反引号就切换」去数会把后文的锚点和链接全部漏掉。
- 命令统一写成 bash：仓库根目录执行，`$PY -B RLVR学习资料/...`，首次出现时链接首页的 `run-setup`；不要再引入 PowerShell、`D:\` 路径或裸 `python`（Linux GPU 环境的训练命令除外）。
- 改了任何 `.py` 后跑第 4 节的两套测试；改了数值口径要同步 `ANSWERS.md` 与 `report.json` 的对照说明。
- 补强代码：原版文件不改；新代码进 `coder1_plus` 的数据侧或判分侧。每补一项都在模块 docstring 和 `examples/source-supplements/README.md` 登记，附 CPU 可复现的验证；改完依次跑 `test_source_supplements.py`、`verify_gt_contract.py`、`verify_test_quality.py`、`verify_judge_and_selection.py`、原有两套测试和 `run_demo.py all`。
- 提交前 `git status` 确认只有预期文件；commit message 用中文或英文均可，末尾加 `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`。只在用户要求时 commit/push。

## 9. 建议的下一步（截至 2026-10-13）

1. 已完成（2026-10-13）：命令统一为 bash + `$PY`，贯穿题统一为 LeetCode 53，00 加入首页路线和各篇导航。
2. 从阶段 01 开始：`practice.py init --run data-01 --mode seed` → 用户先填预测 → `check` → 追加自己的合法小输入与一条错配 → 再 `check/score`；把结果和解释写进 `practice_runs/data-01/notes.md`。
3. 进入 03/04 时，只把第 7 节中有助于理解或面试的条目（如 mini-batch 语义错配、clip 实际不生效、首步 lr=0）复核后并入「面试深挖」表或 `RLVR工业流程.md` 对应小节；指标日志类的小 quirk 不必收。
5. 反向补充源码：7 项已全部实现并在 CPU 上验证，详情见 `examples/source-supplements/README.md`。仍待 Linux 或 GPU 的部分：firejail 分支、`prime_plus` 接入真实训练循环、`--merge_taco_from`、难度筛选所需的真实采样通过率。`prime_plus` 的 CPU 整体运行和 parquet 导出已在 2026-10-15 用 venv 实跑。学 02、03 时把补强项 4 到 6 当学习材料讲「原版与补强版」。
4. 真实采样/训练需要 Linux GPU 环境；在此之前只记录 `examples/training/README.md` 的协议与命令，不补造数字。
6. （2026-10-16）05 的七环已与用户走完（决策日志 `your_decisions.md`，your-01 产物，your-02 计划）。起点是 Qwen3-8B-Base（用户 2026-10-16 明确，不是后训练版 Qwen3-8B）：两轮参考推演与 your-01 已按 Base 重生成，叙述里不再有「基线强、只锐化不扩展」的旧说法；Base 的 tokenizer 是否自带与 Qwen3-8B 相同的聊天模板，按「自带，smoke 时核对」写。2026-10-19 用户决定去掉 TACO、以 your 的配置做完整的「诊断 → 修 → 验证」：your-02 已推演（yourplan 目录），补强项 8 因无 TACO 暂不需要。接下来：03/04 重构；用户自己的 01/02 练习（`data-01` 的 prediction.json 仍为空）；clip-higher 已按用户授权去掉（2026-10-19，没有被诊断出的问题不预防性地改）。

## 10. 长度上标 bad case 的唯一正确口径（2026-11-13 用户定稿，优先级高于本文件其他相关表述）

> **幂指数上标在测试样例扩容解析和 verifier 解析中统一被解析成 `105`**（`10⁶`→`106`，其他幂指数同理）。题面样例（官网直接得到）正常通过；perf 测试样例按错误解析出的上界 105 生成、不被 validate 拦；correctness 轮随机扩容中长度 106~1000 的用例被 validate 拦截丢弃。另因标定下限 `MIN_PERF_SIZE=1000`，解析出 105 的题性能轮整题跳过（909 题）。结果：所有轮次停在 105 规模、`10⁵` 从未出现——实际没有 10⁵ 规模的性能考核，correctness 只剩 ≤105 的随机用例。

配套固定事实（与上句同源，改动须显式修订）：

- **correctness 轮组成**：① LeetCode 官网题面样例（原样，条数=断言数）+ ② 扩展用例，长度 1~1000（数值边界 + 随机，随机必过 validate）。
- **perf 轮**：每题 1 条大输入，长度 = 解析出的规模上界（如 10⁵），种子固定、判分时确定性重建；超时 0.9、大输入答错 0.1。
- **离线/在线**：「轮」是判分内部两个阶段，不是离线/在线各一轮；离线一次性产出用例（含 expected）、每题一个 `time_limit_ms`、大输入种子；判分时零标定、同题各步不漂移。
- **标定成本**：`time_limit_ms` = 参考解墙钟 ×3（放大系数，不是跑三遍），下限 500ms；每题 1 次参考解执行；不做每用例一个钟（correctness 轮不判快慢，判别力与 perf 轮重复）。
- **verifier 三门**：validate 门读约束判输入合法性；correctness 打分与 perf 打分都不读约束。
- **伤害窗口**：correctness 随机扩容 106~1000 被拦（用例级）；perf 样例 105 不被拦但标定 <1000 → 性能轮整题跳过（909 题，题级）；信号 = validate invalid 计数异常 + accepted 长度天花板贴 105 + 10⁵ 规模从未出现。
- **修法**：修正幂指数解析（`10[1-9]`→10 的幂）+ 保持单点解析（解析一次写进数据行，扩容/validate/标定共读）。
- **已废弃说法**：「扩容幂识别正确、validate 幂识别失败」的双解析器分裂——不采用；本口径是唯一正确版本。
- 完整定案：`RLVR学习资料/设计定案表.md` **D22**；两轮成本与三套钟：`讨论结论库.md` 第 11 条；方法论：`bad_case调查方法论.md` §五。
