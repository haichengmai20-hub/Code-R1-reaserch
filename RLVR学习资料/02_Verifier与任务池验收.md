# 阶段二：用同一 Task 学习 Verifier 与任务池

**资料导航：** [资料首页](README.md) · [工业主框架](RLVR工业流程.md) · [01 数据与 GT](01_数据理解与GT构建.md) · **02 Verifier 与任务池（当前）** · [03 Rollout 与 GRPO](03_Rollout与GRPO机制.md) · [04 训练与评测](04_真实训练与独立评测.md)

**本阶段学习内容：** 继续使用最大子数组和的已接受 Tests/GT，复现原 scorer 的提取、执行和 reward；增加一个同题 STDIO 对照，再解释怎样将同一 Task 放入池。输入输出答案审核与后端判分可信分开验收。

**输入交接：** 固定示范对照用 [prepared_task](examples/maximum-subarray/outputs/prepared_task.json)、[cases](examples/maximum-subarray/outputs/cases.json)、[report](examples/maximum-subarray/outputs/report.json)。学生变更继续使用01的 `RLVR学习资料/examples/maximum-subarray/practice_runs/data-01/`，以最新attempt的accepted Tests、新GT和test_version为准，不退回固定8/7。GT是Tests的接口表示，metadata说明来源与用途；无需先重建工业执行系统。

**主框架：** 查[联合验收](RLVR工业流程.md#verification)、[执行边界](RLVR工业流程.md#execution-boundary)、[实际奖励](RLVR工业流程.md#reward-mapping)、[故障归因](RLVR工业流程.md#layer-attribution)。

**先记机制：** Tests告诉判题器要验收什么，执行后端运行程序，评分规则把观察映射成reward。数据答案、实际执行、比较政策和奖励数字各有证据；高分或退出码0不能替代它们。后端隔离与判题完成协议解决不同问题。

## 主框架、项目实现与作业的对应关系

| 当前问题 | Code-R1 入口 | 作业与证据 |
|---|---|---|
| [代码、GT 与 reward](RLVR工业流程.md#reward-mapping)怎样关联？ | [reward 路由](../verl/utils/reward_score/__init__.py)、[scorer](../verl/utils/reward_score/coder1/__init__.py) 的 `_compute_score()` | [2.1](#assignment-2-1)：正常、错误、格式与提前退出 |
| [为什么不能只看退出码](RLVR工业流程.md#execution-boundary)？ | [firejail_exec.py](../verl/utils/reward_score/coder1/firejail_exec.py) 与 [PrimeRewardManager](../verl/workers/reward_manager/prime.py) | [2.2](#assignment-2-2)：改变一个条件、划清证据边界 |
| [Task Pool 怎样读回](RLVR工业流程.md#gt-packaging)？ | [coder1.py](../examples/data_preprocess/coder1.py) 的封装/导出与 [main_grpo.sh](../main_grpo.sh) 的路径 | [2.3](#assignment-2-3)：同一行记录与原生评测契约 |

## 阅读入口与顺序

先追 `data_source=code` 到 scorer，读 `validate_response_structure()`、`extract_code_from_string()` 与 functional/STDIO 分支。再读 `PrimeRewardManager` 的异常回填与末 token 赋值；原生 [data.py](../LeetCodeDataset/leetcote/data.py) 的最长 Python 代码块提取和 [execution.py](../LeetCodeDataset/leetcote/execution.py) 的 imports/时间限制是另一条路径。

实际选择执行 backend 后，才看 [Firejail](../verl/utils/reward_score/coder1/firejail_exec.py)、[Docker](../verl/utils/reward_score/coder1/docker_exec.py)、[CES](../verl/utils/reward_score/coder1/ces_exec.py)、[Kira](../verl/utils/reward_score/coder1/kira_exec.py) 的相应限额与结果。不同文件名或容器名称不自动证明可靠与安全。

## 本阶段作业总览

| 作业 | 必做 | 扩展/实跑条件 |
|---|---|---|
| [2.1](#assignment-2-1) | 固定受控候选的评分与 STDIO 对照 | 更多语言、真实服务并发 |
| [2.2](#assignment-2-2) | 一条件变化与原实现缺口审计 | 受保护完成协议、未知代码隔离和超时清理 |
| [2.3](#assignment-2-3) | 无提示错配定位与小池读回 | 真实 Parquet 导出、全量数据与正式训练 |

一份阶段说明加共同报告即可。系统修复按需要另做；指出缺口不等于将后端认证为可信。

<a id="assignment-2-1"></a>

## 作业 2.1：完整示范——原 scorer 怎样给分

**目的：** 把格式、测试结果、执行状态和最终数字对到具体代码。

```powershell
python RLVR学习资料/examples/maximum-subarray/run_demo.py score
```

脚本 AST 载入锁定源码的 parser 与 `_compute_score()`，仅将 Linux/远程后端替换为固定已审程序的本地 subprocess。它没有启动原 Linux backend，也不接任意新模型代码。

| 固定候选 | 原 scorer reward | 独立 audit 观察 Case 数 | 有限 Tests 的 correctness |
|---|---:|---:|---|
| `reference`、`correct_prefix` | 各 1.1 | 各 8 | 通过 |
| `wrong_empty`、`wrong_sum` | 各 0.1 | 各 8 | 失败 |
| `bad_format` | -1.1 | 0 | 失败 |
| `early_exit` | 1.1 | 0 | 失败 |

原 functional GT 遇到失败断言会早停；上表的 `audit_completed_case_count` 来自另一次逐条输出复查，不是原 scorer 的受保护完成计数。报告的 `source_first_failed_case_id_from_traceback` 显示错误允许空数组首先失败在 `diag-all-negative`、错误总和首先失败在 `seed-1`。reward/log 与独立 audit 分开解释，不把观察次数串成同一次执行。

默认三档：格式/代码缺失为 -1.1；格式合格但执行失败为 0.1；执行报告成功为 1.1。管理器等待超时或异常还可能回填 0.0；该兜底不等于新的正确性标签。`reward>0` 不是通过，`reward==1.1` 在有绕过缺口时也不足以证明通过。

**STDIO 小对照：** 同一题另做教学适配：stdin 是一行 JSON 整数数组，expected stdout 是最大和整数；逐条绑定 stdin/expected/actual。原 scorer 需要执行成功且 `actual.strip()==expected.strip()`。正确适配得 1.1、错误总和得 0.1。原语料仍为 function，这不是从 TACO 下载的新真实样本。

**交付与验收：** 选两个不同结果，逐项指出响应格式、提取代码、使用的 GT、失败 CaseID、原分数与独立 audit 的含义。解释 `extra_info.reference` 为什么不用于文本相似度打分；RewardManager 参数名 reference 在该调用链实际装 GT。

<a id="assignment-2-2"></a>

## 作业 2.2：条件变化——判分与执行边界

**目的：** 一个条件改变怎样影响反馈，哪些结论当前实现无法提供。

继续用01的 `data-01`。先在 `prediction.json` 填 `wrong_empty_reward` 与理由，运行自己的accepted Tests：

```powershell
python -B RLVR学习资料/examples/maximum-subarray/practice.py score --run data-01
```

只改变一个条件：补一条自己手算的全负输入，观察 `wrong_empty`；或冻结Tests，去掉回答标签/代码块，更新预测后使用：

```powershell
python -B RLVR学习资料/examples/maximum-subarray/practice.py score --run data-01 --bad-format
```

每次预测和结果在独立attempt中保存。错误expected在数据审核时隔离，不硬塞进新GT；“坏GT让正确程序失败”是固定示范verify里的受控变异回归，两种实验不能混记。结果会区分原reward、另次audit、实际GT版本与预测是否匹配。命令退出1表示预测未匹配，需要解释/返工；退出2表示输入、格式或材料完整性检查未通过。

`sys.exit(0)` 在 check 前退出却获 1.1，是原协议的假阳性。只换 Linux/Docker 不自动修复它；普通完成字符串也可能伪造。这个负控制验证目前的缺口，没有建立可保护任意候选的完成协议。

**必做完成标准：** 已检查程序的受控机制复现，以及现有 scorer/backend 缺口审计。可以据此进入阶段三的离线组统计。**未知代码实跑条件：** 固定实际 Linux runtime、隔离、资源限额、可信完成证据、超时清理和有限重试，须在目标环境验收；不满足就记待完成，不要求先修全部工业系统才可继续学习。

执行预算教学设定 `B=8,G=4,M=3`：32 个候选、最多96次测试，不必启动96个实例；同一实例批量或首次失败早停会改变成本。它是预算计算，不是实测吞吐。性能上限应匹配固定环境与题意规模，不通过任意缩短超时制造错解。

**交付与验收：** 一次单条件变化的预测/结果/原因，以及原版能够和不能证明的事项。实跑扩展再交 backend 的版本、限额、并发、超时和隔离证据。0.0 或 0.1 要结合原因日志解释，不能由一个浮点数反推出完整 infra 分类。

<a id="assignment-2-3"></a>

## 作业 2.3：独立检查——小池、入口与 GT 配对

**目的：** 不依赖已给答案定位一份记录中的合同错配，确保下一阶段用的仍是同一 Task。

这次由工具生成不标故障类别的材料，而不是自己植入已知错误：

```powershell
python -B RLVR学习资料/examples/maximum-subarray/practice.py blind-init --run blind-01 --seed 53
```

从项目根打开 `RLVR学习资料/examples/maximum-subarray/practice_runs/blind-01/packets.json`。每份记录可以正确，也可以有一个或多个责任层；从Task接口、输入答案、GT对应、完成/实际结果、指标分母逐层查。只编辑同目录prediction.json：每个record给类别列表与整体理由，不改题包；然后提交：

```powershell
python -B RLVR学习资料/examples/maximum-subarray/practice.py blind-grade --run blind-01
```

工具保存提交前预测，校验题包与初始seed一致，再显示诊断。自动只核对类别，理由与最小区分检查按[人工评分](examples/maximum-subarray/ANSWERS.md#blind-rubric)复核。源码和答案可读，这不是防作弊系统；真正验收用另一seed或由同伴换一份未展开的记录，并口答证据。题包是教学构造，不是已实测的infra故障。

查看自己的最新attempt里的data_pool.jsonl，逐项核对 `data_source/prompt/ability/reward_model.ground_truth/extra_info`；GT解码与该attempt的prepared_task相同，每条accepted配对与assert/case_index对应，该次quarantined/pending没有混入。数量以本次结果为准，固定8/7只作示范对照；导出成功不证明答案质量。

**以后需要真实小池时：** 诊断题继续用于调试，train/开发验证/最终测试另按锁定协议选现有来源记录。复用已有GT要审核题意、入口、seed与来源；新增/修改expected要重做该条核验。TACO先按train映射再split，循环修改迭代行不能证明Dataset已更新，应读回文件核对metadata。原输出目录名与main_grpo的DATASET路径也须核对。

本课 JSONL 只演示 Code-R1 row 形状和读回，trainer 实际使用 Parquet。完成转换、参数路径与环境才是实训输入；不直接把本课 JSONL 文件名传给训练入口。policy 难度估计放在下一阶段，需要可信评分与特定 checkpoint 的真实新采样。

**交付与验收：** 独立错配定位说明，以及同一Task/Tests/GT在两次读回中的对应。无法确认的expected或接口先隔离；能确认的正确Case继续保留。若真实模型运行条件未具备，明确区分教学主线完成与实跑待完成。

<a id="interview-2"></a>

<details>
<summary>面试深挖：判分、比较与资源边界</summary>

| 追问 | 答案要点与反例 | 查哪里 |
|---|---|---|
| reward>0或1.1就是通过吗？ | 0.1只是格式合格但执行失败；提前exit可1.1。另次audit的完成数不能冒充原scorer同次执行完成数 | [scorer._compute_score](../verl/utils/reward_score/coder1/__init__.py) |
| Counter缺失是算法错吗？ | 先查提交契约。原生执行拼imports，RL代码+GT不自动补；算法、依赖和接口错误分别归因 | [原生执行](../LeetCodeDataset/leetcote/execution.py)，[scorer](../verl/utils/reward_score/coder1/__init__.py) |
| 两套提取器相同吗？ | 原生选最长Python代码块，RL取最后answer中的代码块并拼接。草稿块抛异常时，选块策略可改变结果 | [extract_completion](../LeetCodeDataset/leetcote/data.py)，[extract_code_from_string](../verl/utils/reward_score/coder1/__init__.py) |
| STDIO的strip能处理任意空白吗？ | 只去两端空白；内部一个与两个空格仍不同。functional比较由assert/辅助判定决定；多解不能任意固定排列 | [remote_check_stdio](../verl/utils/reward_score/coder1/__init__.py)，[两数之和练习](examples/maximum-subarray/EXERCISES.md#transfer-exercise) |
| manager超时证明程序已杀掉吗？ | wait_for取消等待不证明底层线程/进程终止。源码kill-all是注释，None可回填0；需查后端状态和清理 | [PrimeRewardManager](../verl/workers/reward_manager/prime.py)，[Docker finally](../verl/utils/reward_score/coder1/docker_exec.py) |
| 测试越多越可靠吗？ | 需要能检出已确认错误；zip数量不等可能截断，短测试裁剪可能漏复杂度缺陷。时间限制应绑定环境与规模 | [minimize_stdio](../examples/data_preprocess/coder1.py)，[分层归因](RLVR工业流程.md#layer-attribution) |

要证明判题可信，给正确/错误/绕过/故障各自的观察和范围，不靠backend名称盖章。当前固定候选练习不认证未知代码环境。

</details>

---

**继续阅读：** [返回资料首页](README.md) · [Verifier与执行边界](RLVR工业流程.md#verification) · [上一阶段：01 数据与GT](01_数据理解与GT构建.md) · [下一阶段：03 Rollout与GRPO](03_Rollout与GRPO机制.md)
