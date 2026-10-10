# Code RLVR 工业级完整流程：数据、Verifier、Sandbox、训练与评测

**适用范围：** 单题代码生成、函数实现、竞赛编程，不涉及多轮 Coding Agent。

**四项基本原则：**

1. Reference Code 先检查可执行性，再用于生成 Expected Output，最后对输出进行独立可信性验证。
2. Test Input Construction 不等于完整 Test Suite Construction。只有 Input 和对应的 Expected Output 经过验证后，才形成 Verified Test Suite。
3. Sandbox 负责执行，Verifier 负责正确性判定，Oracle 提供判定依据，Reward Mapping 负责生成训练信号。
4. Verifier 不应使用 Reference Code 生成的答案，再通过同一份 Reference Code 自我验证其正确性。

---

# 一、先厘清六个核心对象

```text
Task / Problem Specification
│
├── Ground Truth（GT，可执行正确性标准）
│   │
│   └── Verified Test Suite
│       ├── Test Case 1
│       │   ├── Test Input
│       │   └── Expected Output
│       ├── Test Case 2
│       │   ├── Test Input
│       │   └── Expected Output
│       └── ...
│
├── Oracle（正确性依据）
│   ├── 已有的可信 Expected Output
│   ├── Reference Code（可选）
│   └── 独立规则 / 可执行断言
│
└── Verifier（验证程序）
    ├── 调用 Sandbox 执行 Actor Code
    ├── 获取 Actual Output
    ├── 读取 GT
    ├── 比较 Actual 与 Expected Output
    └── 输出 Pass / Fail / Error
                 ↓
           Reward Mapping
                 ↓
               Reward
```

| 对象 | 含义 | 是否必需 |
|---|---|---|
| Test Case | 一条输入及对应的预期结果或断言 | 通常需要 |
| Expected Output | 某个输入的正确结果 | 输出比较模式下需要 |
| GT | Verified Test Suite 表达的可执行正确性标准 | 必需 |
| Oracle | 构造或确定正确结果的依据或机制 | 必需 |
| Reference Code | 用于生成或核验 Expected Output 的正确实现 | 可选 |
| Verifier | 执行验证并给出判定结果的程序 | 必需 |

这份母版默认采用确定性输出的代码任务：

\[
GT_{\mathrm{operational}}=\{(x_j,y_j^*)\}_{j=1}^{M}
\]

其中，\(x_j\) 是 Test Input，\(y_j^*\) 是 Expected Output，\(M\) 是测试数量。

GT 是完整 Problem Specification 在有限测试输入上的可执行表达，不等于完整的正确性定义。

Reference Code 是构造 GT 的可选工具，Verifier 则消费 GT，对 Candidate Code 进行正确性判定。

对于允许多种正确输出的任务，可以采用合法输出集合或可执行断言，而不是固定的 Expected Output。

# 二、完整工业流程

## 阶段 A：离线构造 RLVR Task Pool

```text
Raw Tasks
    ├─ 公开代码题库
    ├─ 企业内部题库
    └─ Teacher 合成任务
    ↓
① Task Preparation
    ├─ Schema 统一
    ├─ 必填字段 / 格式检查
    └─ 题意与约束完整性检查
    ↓
② Dedup + Leakage Control
    ├─ Exact / Semantic Dedup
    └─ Train / Eval 按题目家族隔离
    ↓
③ Test Input Construction
    ├─ 复用已有 Test Inputs
    ├─ Teacher 生成新输入
    └─ 随机 / 边界 / 对抗测试扩充
    ↓
④ Oracle Construction
    │
    ├─ 已有可信 Expected Output
    │   └─ 直接复用
    │
    └─ 没有可信 Expected Output
        ├─ 获取 / 构造 Reference Code
        ├─ 检查可执行性及已知样例
        └─ 执行新输入 → Expected Output
    ↓
⑤ Test + Oracle Validation
    ├─ Test Input 是否合法
    ├─ Expected Output 是否可信
    ├─ 正确解是否通过
    ├─ 错误解是否能被识别
    └─ 测试结果是否稳定
    ↓
⑥ Verified Test Suite（可执行 GT）
    └─ Test Case = Input + Expected Output
    ↓
⑦ Verifier + Sandbox 验收
    ├─ 正确性判定
    ├─ 隔离 / Timeout / 资源限制
    └─ 异常处理及重复执行
    ↓
⑧ Difficulty Filtering
    └─ 当前 Policy 多次 Rollout
    ↓
RLVR Task Pool
```

### 关键依赖：Reference Code 为什么需要检查两次？

实际上不是重复检查，而是两个不同的检查对象。

**第一次：Reference Code 初步验证**

在 Oracle Construction 内执行，确认它能够运行，并通过已有的可信样例，避免用明显错误的程序生成 Expected Output。

**第二次：Expected Output 独立验证**

在 Test + Oracle Validation 中进行，检查新生成的 Expected Output 是否真正符合题目要求，而不是再次检查 Reference Code 能否运行。

```text
Reference Code
     ↓
检查可执行性 + 已知样例
     ↓
运行 Teacher 生成的 Test Inputs
     ↓
得到 Expected Outputs
     ↓
独立 Oracle Validation
     ↓
Verified Test Suite
```

这两次检查分别回答：

- Reference Code 是否具备生成候选结果的基本条件？
- 生成出来的候选结果是否有足够证据可以成为 GT？

独立验证的具体方法见第六部分。

### 工业依据

以下是公开的技术方法，不代表相关厂商完整的内部 Pipeline 已经公开。

- [Google DeepMind CodeContests](https://github.com/google-deepmind/code_contests)：公开数据包含成对的输入输出测试，以及正确和错误的人类程序，可用于构造和审查 Verifier。
- [Qwen3-Coder](https://qwenlm.github.io/blog/qwen3-coder/)：官方介绍了通过自动扩展不同代码任务的测试用例，构建大规模 Code RL 训练数据。
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)：公开论文介绍了代码任务使用编译器和预定义测试集产生基于规则的正确性 Reward。

Train/Eval 应在任务级隔离，并在测试扩充后再次检查泄漏。Difficulty Filtering 只针对训练池，不随意修改已经锁定的评测集。

难度估计需要记录所用的 Policy Checkpoint；随着 Policy 更新，可以按需刷新训练任务的难度标签。

对于 GRPO，优先选择具有适当成功率和 Reward 方差的任务。全对或全错的采样组没有组内相对优势信号，但不意味着必须永久删除这些任务。

## 阶段 B：在线 RLVR 训练

这里衔接 Ray + verl + vLLM/SGLang + GRPO。

```text
             RLVR Task Pool
                   ↓
             Task Sampling
                   ↓
         Rollout Policy（θ_old）
                   │
              每题采样 G 次
                   ↓
             Candidate Codes
                   │
                   ▼
            Sandbox Workers
            ├─ 编译 / 解释执行
            ├─ 注入 Test Inputs
            ├─ 获取 Actual Outputs
            └─ 检查 Timeout / Runtime
                   │
                   ▼
                 Verifier
                   │
        Actual vs Expected Output
                   │
                   ▼
          Binary Reward（默认 0 / 1）
                   │
                   ▼
             Group Advantage
                   │
                   ▼
       old / new / ref logprobs
                   │
                   ▼
             GRPO / PPO Loss
                   │
                   ▼
              Actor Update
                   │
                   ▼
               Weight Sync
                   │
                   └────→ 下一轮 Rollout
```

训练中的三个关键对象：

| 对象 | 来源 | 用途 |
|---|---|---|
| Candidate Code | Actor 生成 | 待验证的候选程序 |
| Actual Output | Sandbox 执行 Candidate Code | Verifier 的待检查结果 |
| Expected Output | 离线构造的 GT | Verifier 的比较依据 |

对于 GRPO，假设每个任务采样 \(G\) 个 Candidate Codes，对应 Reward 为 \(r_1,\ldots,r_G\)，常用的组内优势估计为：

\[
A_i=\frac{r_i-\bar r}{\sigma_r+\epsilon}
\]

**默认使用 Binary Reward。** 二值奖励下，优势尺度由组内成功与失败的数量决定。全对或全错的 Group 没有组内相对优势信号。

Partial Reward 可以提供更稠密的训练信号，但可能增加针对测试集优化的风险，作为可选实验保留。

Logprob 的计算与用途：

| Logprob | 计算时机 | 作用 |
|---|---|---|
| Old | Rollout 时记录，或由冻结的 Old Policy 重算 | Importance Sampling |
| New | Actor 训练时 | 梯度更新 |
| Ref | 启用 Reference KL 时 | 约束策略偏移 |

训练 Loss 作用于参与策略优化的 Policy-generated Tokens。若模型还生成了推理过程，这些 token 也可以参与训练；Prompt、Test Inputs 和环境反馈不参与 Loss。

SFT Cold Start 可以放在 RL 之前，但不是所有 Code RLVR 都需要单独进行一次 SFT。

# 三、Sandbox 与 Verifier 的工程关系

Sandbox 和 Verifier 应当解耦，但由 Verifier 或其上层任务调度器统一编排。

```text
Ray / Task Scheduler
        │
        ▼
Sandbox Worker
├─ Runtime Image
├─ Candidate Code
├─ Test Inputs
├─ CPU / RAM / Timeout
└─ Execution Logs
        │
        ▼
Verifier
├─ Expected Outputs / Assertions
├─ Comparator
├─ Pass / Fail / Error
└─ Reward Mapping
        │
        ▼
Reward Worker → Trainer
```

工业环境可以使用以下工程组件：

| 模块 | 实现方式 |
|---|---|
| Runtime | Docker Image，固定编译器及依赖 |
| Isolation | gVisor / MicroVM |
| Resource Control | cgroups、Timeout |
| Scheduling | Ray Workers / 任务队列 |
| Logging | stdout、stderr、exit code、执行时间 |

其中，[gVisor](https://gvisor.dev/) 提供了用于执行不可信程序的隔离机制，可通过 `runsc` 与容器生态集成。普通 Docker 容器不应直接等同于充分的安全隔离。

需要保证：

- Candidate Code 无法读取隐藏的 Expected Outputs 或篡改评分器。
- 不同任务的执行环境相互隔离，限制网络、文件权限和资源。
- 基础设施异常应执行有限次数的重试；超过上限后隔离，并记录异常，不能直接转化为模型负奖励。

对于单题代码 RLVR，通常使用 CPU Sandbox 执行代码，GPU 则主要用于模型 Rollout 和训练。

### Sandbox 总执行预算

设每批任务数为 \(B\)，每题采样 \(G\) 个候选程序，每个候选程序执行 \(M\) 条测试，则最大计划测试执行量为：

\[
N_{\mathrm{test}}=BGM
\]

这些测试不一定需要启动相同数量的 Sandbox 实例，可以在一个隔离实例内批量执行。

Binary Reward 可以采用 Early Exit，在首次明确失败后停止剩余测试，以减少执行成本。但需要完整 Partial Pass Rate 或逐条测试诊断时，应运行完整测试集或采用单独的诊断流程。

# 四、Evaluation 与 Bad Case 回流

```text
Updated Actor
      ↓
Held-out Tasks + Held-out Tests
      ↓
Evaluation
├─ Success Rate / Pass@k
├─ Compile / Runtime Error Rate
├─ Token / Latency / Cost
└─ Train Reward vs Held-out Performance
      ↓
Bad Case Mining
      ↓
Failure Attribution
├─ Data Failure
├─ Model / Training Failure
├─ Verifier Failure
└─ Sandbox Failure
      ↓
针对性修复
      ↓
重新验收 + 回归评测
      ↓
下一轮训练
```

这里需要特别关注 Train Reward 与 Held-out Pass Rate 的差异。

例如，Train Reward 不断提高，但独立评测没有改善，可能是训练测试覆盖不足、Reward Hacking、过拟合或任务分布偏移，不能直接认定模型能力已经提高。

为了支持后续归因，每条执行记录至少保留：

```text
task_id / run_id
policy_checkpoint
test_suite_version
verifier_version
sandbox_image_digest / worker_id
candidate_code
generation_finish_reason
execution_status / test_results
reward
execution_logs
```

Comparator、Output Normalization 等判定规则属于 Verifier Version；测试或 Expected Output 的变化属于 Test Suite Version。

跨版本比较 Reward 时，需要确认判定依据一致。必要时，应使用相同版本的测试和 Verifier 重新评估。

算法错误类别可以在 Bad Case 分析阶段通过人工审核或 LLM 辅助标注产生，不要求每次 Rollout 都完成语义归因。

具体归因方法在第六部分展开。

# 五、Verifier 的构造与内部执行流程

需要区分两件事：

- **离线构造 Verifier**：让它具备验证不同代码任务的能力。
- **在线执行 Verifier**：每次 Actor Rollout 后，自动验证 Candidate Code 并计算 Reward。

## 1. 离线构造

不需要为每一道题重新开发整个 Verifier。工业上可以使用通用的执行与评分框架，为不同 Task 配置对应的 GT 和执行参数。

```text
通用 Verifier Engine
     │
     ├─ Code Parser / Extractor
     ├─ Language Runtime
     ├─ Sandbox Executor
     ├─ Output Comparator
     ├─ Error Handler
     └─ Reward Mapper
              │
              ▼
       加载 Task Config
       ├─ Language
       ├─ Input / Output Format
       ├─ Verified Test Suite
       ├─ Comparator Rule
       └─ Resource Limits
              │
              ▼
         Verifier Audit
       ├─ 可信正确程序应当通过
       ├─ 已确认错误程序应当失败
       ├─ 重复执行结果稳定
       └─ 异常能够正确分类
              │
              ▼
         Verifier Ready
```

其中，GT 来自第二部分已经完成的 Verified Test Suite。

Verifier Audit 可以使用具有可信标签的正确和错误程序，也可以通过 Mutation Testing（变异测试）构造错误程序。

但 Mutation 不保证一定引入实际错误。需要找到能够证明变异体行为错误的输入，才能将其作为已确认的 Negative Sample。

## 2. 在线执行

```text
Candidate Code + Task Config
              ↓
① Code Extraction
   提取代码、检查提交格式
              ↓
② Sandbox Execution
   编译 → 运行 Test Inputs
              ↓
③ Execution Result
   Actual Output + Exit Status
              ↓
④ Error Classification
   ├─ Compile Error
   ├─ Runtime Error
   ├─ Timeout
   └─ Infrastructure Error
              ↓
⑤ Output Normalization
   按 Task Config 归一化输出
              ↓
⑥ Output Comparison
   Actual vs Expected / Assertion
              ↓
⑦ Test Aggregation
   汇总所有测试的判定结果
              ↓
⑧ Reward Mapping
   Pass / Fail → Reward
              ↓
Reward + Detailed Execution Record
```

Output Normalization 必须遵循版本化的 Task Config 和 Comparator Rule，不能为了提高通过率随意放宽比较标准。

例如，只有题目明确允许浮点误差时，才应该设置浮点容差。

### Reward 如何计算？

默认使用 Binary Reward：

\[
R_i=\prod_{j=1}^{M}I(\operatorname{Pass}_{ij})
\]

候选程序 \(i\) 必须通过全部 \(M\) 个测试才能得到 1，否则为 0。

也可以使用 Partial Reward：

\[
R_i=\frac{1}{M}\sum_{j=1}^{M}I(\operatorname{Pass}_{ij})
\]

两者的区别是，Binary Reward 强调最终正确性，Partial Reward 提供更密集的学习信号，但可能增加测试集过拟合的风险。

母版默认使用 Binary Reward；Partial Pass Rate 作为可选训练信号和诊断指标。

### 异常应该如何处理？

下表采用默认 Binary Reward 口径。

| 执行状态 | 处理方式 |
|---|---|
| PASS | Reward = 1 |
| WRONG_ANSWER | Reward = 0 |
| COMPILE_ERROR | Reward = 0 |
| RUNTIME_ERROR | Reward = 0 |
| 程序超出任务时间限制 | 按任务规则判为失败 |
| TRUNCATED | 单独记录；不完整的提交按失败处理 |
| INFRA_ERROR | 有上限地重试，持续失败则隔离 |

如果生成结果被截断，但代码已经完整且符合预先规定的提交格式，可以按固定规则继续验证。不能根据最终测试结果临时调整截断处理规则。

对于 INFRA_ERROR，应配置 `max_retries`。达到上限仍然失败的样本不进入正常 Reward 训练；如果错误率超过预设阈值，则暂停相关 Worker 或任务队列并排查。

# 六、边界问题与 Failure Attribution

## 1. Teacher 生成输入后，如何确认 Ref Code 的输出真的是 Expected Output？

这是数据构造中最重要的可信性问题。

Reference Code 能运行、能够生成结果，不等于它一定正确。多个模型给出相同结果，也不构成严格的正确性证明。

```text
Teacher 生成 Test Input
          ↓
Reference Code A
          ↓
得到 Candidate Expected Output
          ↓
独立验证
├─ 官方 Expected Output
├─ 独立 Reference Code B
├─ 数学性质 / 不变量
└─ 必要时人工审核
          ↓
Accepted / Quarantined / Rejected
          ↓
Verified Test Suite
```

这里存在三种常见情况。

**情况 A：有官方 Expected Output**

直接以官方结果作为验证依据。例如已有题库提供的正式测试，可以直接使用，无须重复生成。

**情况 B：没有官方输出，但有独立的正确实现**

可以使用不同算法实现的 Reference Code 交叉验证，并利用已知样例、边界条件和问题不变量辅助校验。

对于输入规模较小的问题，还可以使用低效但容易审查的暴力算法作为独立 Oracle。

例如，对于排序任务，除了比较两个独立实现的结果，还可以检查：

- 输出是否有序。
- 输出与输入的元素及出现次数是否一致。

这些性质独立于某一份 Reference Code，因此比简单重复执行同一份程序更有价值。

**情况 C：没有可靠 Reference Code，也没有独立 Oracle**

Teacher 可以提供候选答案，多 Solver 可以辅助交叉检查，但不能据此保证 GT 正确。

这类测试需要额外审核；如果无法建立足够可信的 Oracle，应隔离或删除，不直接进入高可信的 RLVR Task Pool。

对于存在随机性、时间依赖或非确定性执行结果的任务，需要固定可控的随机因素，或者使用断言式 Oracle；不能直接套用确定性的 `(Input, Expected Output)` 比较方式。

## 2. 如何区分数据、Verifier、Sandbox 和训练问题？

不能仅根据 Reward 高低归因，需要结合执行链路和统计诊断。

| Failure | 典型现象 | 如何验证 | 回流位置 |
|---|---|---|---|
| Prompt Pool | 题意歧义、约束缺失、分布不合理 | 题目审核、分布统计 | Task Preparation |
| Test Cases / Oracle | 错误程序通过、正确程序失败 | 独立测试、交叉验证 | Test / Oracle Construction |
| Verifier | 比较规则错误、输出解析错误 | 固定输入输出做单元测试 | Comparator / Reward Mapper |
| Sandbox | 正确程序在特定环境运行失败 | 独立环境复现、日志检查 | Runtime / Resource Config |
| Model | 代码逻辑错误、无法正确实现算法 | 人工分析、错误类别统计 | SFT / RL 数据构造 |
| Training | Reward 异常、KL 失控、独立性能下降 | 训练日志、固定集评测、消融实验 | Loss / Reward / Sampling |

### 一个完整的归因例子

假设 Actor 生成的程序获得 Reward = 1，但在独立测试中失败。

```text
Train Reward = 1
Held-out Test = Fail
          ↓
① Execution Diagnosis
   ├─ Prompt：题意与约束是否完整？
   ├─ Test：是否缺少边界测试？
   ├─ Oracle：Expected Output 是否正确？
   ├─ Verifier：Comparator 是否错误？
   └─ Sandbox：执行是否稳定？
          │
          ▼
② Statistical Diagnosis
   ├─ Train / Eval 题型与难度分布
   ├─ Test Coverage / Pass-Fail Matrix
   └─ Reward / KL / Length 分布
          │
          ▼
③ Model / Training Diagnosis
   ├─ 模型能力不足？
   ├─ 是否过拟合训练测试？
   ├─ 是否存在 Reward Hacking？
   └─ 是否发生策略退化？
          ↓
Failure Attribution
```

两类典型问题需要区分：

- **Test Coverage 不足：** 同一道题的 Candidate Code 通过原始测试，但增加边界测试后失败。
- **Distribution Shift：** 模型在训练题型上表现正常，但在新的算法类型、任务难度或输入规模下表现下降。

两者可能同时发生，因此需要结合测试覆盖分析、分布统计和对照实验，不能仅凭一次执行失败进行互斥归因。

完成归因后，不应该把所有 Bad Cases 都直接加入 SFT。

- 数据错误先修复数据和 GT。
- Verifier 错误先修复评分机制。
- Sandbox 错误先修复执行环境。
- 只有确认属于模型能力或训练问题后，才考虑构造新的 SFT 数据、调整 Reward 或继续 RL。

修复后，需要重新执行受影响的样本，必要时重算历史 Reward，并在锁定的 Held-out Benchmark 上进行回归评测。

## 3. 三个必须明确的理论边界

**边界一：GT 是有限测试集表达的可执行正确性标准。**

\[
GT_{\mathrm{operational}}=\{(x_j,y_j^*)\}_{j=1}^{M}
\]

它由 Problem Specification 派生，不等于完整的 Problem Specification，也无法保证覆盖所有输入。

**边界二：Reference Code 是可选的 Oracle 构造工具。**

它可以离线生成 Expected Output，但不是每个 Test Case 都需要一份 Reference Code，也不需要在每次在线 Rollout 时执行。

**边界三：通过全部 Test Cases 不代表程序在所有输入上正确。**

\[
\operatorname{Pass}(\text{Test Suite})
\not\Rightarrow
\text{Universal Correctness}
\]

因此需要通过边界测试、随机测试、独立 Oracle 和 Held-out Evaluation 提高验证可信性。

---

# 最终：Code RLVR 端到端闭环

```text
Raw Tasks
    ↓
Task Preparation
    ↓
Dedup + Leakage Control
    ↓
Test Input Construction
    ↓
Oracle Construction
    ↓
Test + Oracle Validation
    ↓
Verified Test Suite（GT）
    ↓
Verifier + Sandbox 验收
    ↓
Difficulty Filtering
    ↓
RLVR Task Pool
    ↓
Actor Rollout
    ↓
Sandbox Execution
    ↓
Verifier → Reward
    ↓
Advantage → GRPO / PPO Loss
    ↓
Actor Update → Weight Sync
    ↓
Held-out Evaluation
    ↓
Bad Case Attribution
    ↓
Data / Verifier / Sandbox / Training 修复
    ↓
重新验收 → 下一轮 RL
```

整个系统最重要的是两条主线：

**数据主线：** `Task → Test Input → Oracle → Expected Output → Verified GT`。

**训练主线：** `Actor → Candidate Code → Sandbox → Verifier → Reward → Policy Update → Evaluation`。

第一条保证 Reward 的可信性，第二条利用可信 Reward 优化策略。Bad Case Attribution 则把两条主线连接成持续迭代的工业闭环。
