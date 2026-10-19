# Code RLVR 学习主框架：最小闭环与工业扩展

**资料导航：** [资料首页](README.md) · **工业主框架（当前）** · [00 全程走读](00_一条记录从原始Task到训练.md) · [01 数据与 GT](01_数据理解与GT构建.md) · [02 Verifier 与任务池](02_Verifier与任务池验收.md) · [03 Rollout 与 GRPO](03_Rollout与GRPO机制.md) · [04 训练与评测](04_真实训练与独立评测.md)

**适用范围：** 单题代码生成、函数实现、竞赛编程。学习从 [资料首页](README.md) 和 01—04 阶段进入，本文按机制查阅。全文唯一的贯穿题是真实 Hot 100「最大子数组和」，即 LeetCode 53，所有机制例子都用它。这道题从原始记录走到训练的全过程见 [00 全程走读](00_一条记录从原始Task到训练.md)；旧版资料用过的五道题只在第七章作为历史归档提及。

**审查结论：** 原文的“离线数据构造 → 在线验证与训练 → 独立评测与归因”主线合理，可以作为学习与方案设计的强参考。要用来逐项解释 Code-R1 源码，必须区分通用设计、当前源码行为与实际执行证据；要用于生产训练，还需要对目标执行后端验收。本文按本地源码补齐这些边界，不把一张流程图等同于已经完成的工业系统。

**核对基线：** 最近修订 2026-10-13，全文贯穿题统一为 LeetCode 53；Code-R1 本地提交 `443f8da8058d07051793eff875da117534fd1ba1`，remote 为 `ganler/code-r1`；LeetCodeDataset 本地提交 `f3519e6b7f42265dd4e4d35d4812e8c29b08111a`，remote 为 `newfacade/LeetCodeDataset`。`[Code-R1 实现]` 说明当前源码，`[通用设计]` 说明扩展目标；本机实际跑出的结果写明日期，例如「2026-10-13 实测」。LeetCode 53 演示的结果见 [运行报告](examples/maximum-subarray/outputs/report.json)。配置与源码不单独证明训练已执行或效果已提高。

## 阅读前先知道：原文哪些地方需要补强

| 审查点 | 判断及修订口径 | 对学习的影响 |
|---|---|---|
| 总体流程 | 主线成立；图中的箭头表示产物依赖，不意味着所有开发只能串行进行 | 先定题意、接口、比较规则和运行契约；通用 Verifier 可以与测试构造同步推进 |
| “GT = 测试集” | 对本母版的有限测试模式成立；在 Code-R1 中还要封装为 JSON 字符串里的可执行 `functional` 或输入输出对 | GT 不是参考代码，也不是模型应逐字模仿的答案 |
| 元数据 | 原文缺少获取、映射与版本说明 | 元数据说明题目身份和协议；它不直接证明题目或答案正确 |
| imports 与入口 | 两套评测路径不同：原生评测器拼接 imports，RL 评分器不自动补 LeetCode imports | 同一份裸 `completion` 可能在一边通过、另一边 `NameError` |
| 原始奖励 | 母版默认二值奖励属于设计；Code-R1 实际有 `-1.1 / 0.1 / 1.1`，管理器异常还可能回填 `0.0` | 学习 GRPO 前先核对实际 reward，不能直接套二值成功率 |
| Train/Eval 隔离 | 必须绑定数据版本和评测协议；v1 的 175 道 test 题里有 61 道在 v2 成了训练题 | “标过 test 的题永远不能训练”过强；如果继续以 v1 test 评测，则必须从 v2 训练来源排除这些题 |
| Verifier 可信性 | Sandbox 隔离与判题可信性需要分别验收 | 进程退出码为 0 不证明测试完成；本次受控探针已复现提前退出得高分的缺口 |
| Actor 可见信息 | `meta.query` 不含题面中的 Constraints 和样例；v1、v2 共 3835 条记录全都如此 | 构造 prompt 时应审核规模与约束是否传给模型，否则不能只在隐藏大输入上要求它满足未说明的规模 |
| 难度过滤 | 是可选的训练采样策略，不是数据正确性的必需定义；当前预处理器没有完整的 Policy Rollout 难度筛选 | 必须先让奖励可靠，再用特定 checkpoint 估计难度 |
| “独立验证” | 不同算法的对照提高可信度；不同文件、不同模型回答、重复运行不天然独立 | 记录算法与证据来源，避免两个实现共享同一个题意误解 |

附件中的 Code-R1 审查已经正确提醒：LeetCode 分支主要封装已有测试；TACO 验证用 `solutions[-1]`，保存的 reference 却是 `solutions[0]`。本次补充还要求区分数据版本、执行前缀和测试是否实际完成。这里的“强参考”指可追溯的学习与工程参考；某份代码能否成为可信 Oracle，仍要单独建立证据。

**怎么读：** 第一章给出 MVP 和贯穿题；第二章说明 Task 清洗、逐条测试审核与入池；第三至六章按需查执行、训练、评测与归因；第七章汇总证据边界，并给出旧版历史归档的位置。首次学习按四阶段推进，在需要解释某个机制时跳到本文对应小节。2026-10-10 的课程修订增加学生练习与机制验收，源码基线仍为上述提交。

**四项基本原则：**

1. 简化名称仍要保留逐条证据：输入是否合法、期望输出从哪里来、核对过什么、为什么接受或隔离。
2. 每条测试将 input 和 expected output 配对保存。发现错误先定位该条；题意或接口本身不清时才升级为 Task 层问题。
3. GT 是评分器消费的测试表示；源码字段名叫 ground_truth，不会自动使其中的答案正确。
4. Verifier 调用 Sandbox 执行并判分。测试数据可信与执行/评分可信分别验收；原版受控复现和可信模型训练有不同的证据条件。

---

<a id="mvp-overview"></a>

# 一、先贯通一个最小闭环

主线只跟踪四个名字：Task、Tests、候选代码、Verifier。

| 主线名称 | 本课怎样理解 |
|---|---|
| Task（任务） | 同一道题的题意、接口、测试与 metadata；原始记录和入池记录是它的不同处理状态/表示 |
| Tests（测试） | 一组具体检查。在 Code-R1 的评分接口中，它们连同入口或比较规则保存为 GT |
| 候选代码 | 这道题的待测实现；模型实采样与手工受控候选分别标注 |
| Verifier（判题程序） | 读取 GT、调用执行后端并给出结果和 reward 的公共程序 |

测试内部，input 是调用参数，expected output 是该参数的预期结果；一条配对检查叫 Test Case，多条叫 Test Suite。它们是测试的组成与集合关系，封装为 GT 时没有重新制造一套答案。**Oracle 只表示求解或核对正确答案的方法；Ref Code 是可选工具。** 主文使用“测试”“求解”“核对”讲过程，读源码时保留实际字段和函数名。

metadata 是 Task 的身份与履历，不另造一种“meta task”。原始 LeetCode JSONL 已经有 `meta`；增加标签本身不证明任务可以训练。Reference Code 也不等于训练时提供 KL 约束的 Reference Policy。

这四个名字帮助理解数据与判题，不要求把训练中所有角色压成四个对象。进入阶段三后仍需区分 old/new/ref 策略的用途、有效 token 与求导边界。每个术语在它解决的具体问题处介绍，避免首次学习先背整套名词。

```text
整理 Task → 准备并逐条审核 Tests → 封装 GT 与入池
                                         ↓
                 生成候选代码 → Verifier → reward
                                         ↓
                    advantage → 更新 → 独立评测
```

通用 Verifier 可以与数据工作同步准备，正式模型实验前再联合验收。学习主线先复现原版机制并识别缺口；测试覆盖扩展、语义去重、受保护判题协议和规模化并发审计按具体需求增加。受控演示仅执行已检查的固定程序，不能代替未知模型代码的隔离环境。

<a id="running-example"></a>

## 1. 贯穿题：Hot 100 最大子数组和

题目来自 [LeetCode 53 官方题面](https://leetcode.com/problems/maximum-subarray/description/) 与 [Hot 100](https://leetcode.cn/studyplan/top-100-liked/)，本地原记录在 [v2 RL JSONL](../LeetCodeDataset/data/LeetCodeDataset-v2-rl-problems.jsonl)，以 `task_id=maximum-subarray` 查找，核对时位于第 143 行。入口为 `Solution().maxSubArray`；合法输入是长度 1—100000 的整数数组，元素范围 -10000—10000，返回非空连续子数组的最大和。

原记录有三条 seed：

| input | expected output |
|---|---:|
| `nums=[-2,1,-3,4,-1,2,1,-5,4]` | 6 |
| `nums=[1]` | 1 |
| `nums=[5,4,-1,7,8]` | 23 |

它们的结果都为正数。一个错误地允许空子数组的程序可能通过三条 seed，却在 `nums=[-3,-2,-5]` 上返回 0。按题意枚举六个非空连续子数组，和为 `-3,-5,-10,-2,-7,-5`，因此这份输入的正确输出是 -2。这是具体输入上的完整枚举依据，不靠“参考程序说了算”。

在项目根目录运行 [受控演示](examples/maximum-subarray/run_demo.py)：

```bash
$PY -B RLVR学习资料/examples/maximum-subarray/run_demo.py all
```

同一资产依次展示原记录、逐条审核、GT、原 scorer、组内优势与指标；具体结果见 [报告](examples/maximum-subarray/outputs/report.json)。原记录的 `meta.split=rl` 保留为来源事实，本课使用 `diagnostic`；熟悉题或手工候选上的结果不作为未见题泛化成绩。

## 2. 接受什么，固定什么

先明确题意、接口、合法输入和比较规则。每条输入求得候选答案后核对，再把已接受的测试封装到 GT。固定测试、Verifier、运行环境和奖励版本后才能比较模型结果；出错可以升版本、重验，但不在某个候选失败后临时放宽规则。

$$
GT_{\mathrm{operational}}=\{(x_j,y_j^*)\}_{j=1}^{M}
$$

这里 $x_j$ 是某条已接受测试的输入，$y_j^*$ 是与它配对的期望结果，$M$ 是实际收录的测试数。它是有限检查的操作标准，不是全部输入上的正确性证明。多种合法输出任务需要语义断言或合法性判定；不能强行将它们压成任意一个固定字符串。

对应练习：[阶段 01 作业 1.1](01_数据理解与GT构建.md#assignment-1-1)、[逐条测试审核](01_数据理解与GT构建.md#assignment-1-2)。

<a id="task-preparation"></a>

# 二、Task 清洗、测试审核与入池

## 阶段 A：离线构造 RLVR Task Pool

```text
原始 Task
    ↓
整理任务：字段/题意/接口/约束/来源、去重与实验划分
    ↓
准备 Tests：复用或补输入 → 获得候选答案 → 逐条核对
    ↓
只收录已接受的测试，按评分接口封装 GT
    ↓
保存 Task 与 metadata，读回 Task Pool
    ↓
联合验收后，按实验条件进入受控演示或真实采样/训练
```

通用 Verifier 与环境可以同步准备；正式计算训练奖励前固定数据与判分协议。下文 A1—A8 是这条短主线的细节与扩展，不要求首次学习先建设八个独立系统。必做先贯通一题与已检查的候选；语义去重、系统随机/性能测试、可靠后端和 policy 难度筛选按需要增加。输入来源有三类：题库已有、人工/程序扩充、Teacher 合成；后两类不能因生成成功就自动接受。

## 阶段 A 的逐步解释：输入、规则、例子与完成条件

<a id="a1-preparation"></a>

### A1. 元数据获取与 Task Preparation：先把题目读成能执行的任务

**人话：** 先弄清这张题卡从哪来、要求写哪个函数、合法输入是什么，再统一字段名称。获取元数据主要是读取已有字段并记录来源；不需要让 LLM 猜一遍已经写在 JSONL 里的 ID 和难度。

**[Code-R1 实现]** 原生 `leetcote/data.py` 的 `read_jsonl()` 逐行 `json.loads()`；`read_problems()` 按 `task_id` 建字典。预处理器的 `leetcode2k()` 通过 Hugging Face Datasets 加载本地 **v2** JSONL。文件名、加载器的 `"train"` 容器名称、原始 `meta.split`、最终 `extra_info.split` 是不同层的概念，必须记录各自含义。

贯穿题的清洗顺序是：读取原记录并保留源版本 → 检查 task_id、题面、`maxSubArray(nums)` 接口与返回整数的规则 → 核对非空和数值范围 → 生成包含这些约束的 actor prompt → 按题身份去重和划分 → 审核已有与新增测试 → 编码为 Code-R1 样本。代码块、转义和空白只做明确的格式适配，不能全局替换后改变题意或 Python 程序。

LeetCode 53 这条记录里最关键的五个原始字段：

| 原始字段 | 人话与用途 | LeetCode 53 的实际值 |
|---|---|---|
| `task_id` | 稳定标识题目 | `maximum-subarray` |
| `meta.question_id` | 来源题库的数字 ID，用于去重和 train/test 排除 | `53` |
| `meta.query` | 给模型的短题面，含函数签名 | 题面第一段加签名，不含约束和样例 |
| `test` 与 `entry_point` | 判题依据与调用入口 | 3 条 assert；`Solution().maxSubArray` |
| `completion` | 参考实现候选 | Kadane 实现；先审再用，不自动成为 Oracle |

原始记录的 14 个字段、Code-R1 训练行的 10 个字段、演示扩展的 20 个字段，各自的含义和消费者见 [01 的 metadata 字段字典](01_数据理解与GT构建.md#metadata-fields)。

**[通用设计]** MVP 的 metadata 只需说明身份、来源、用途和所用测试。Code-R1 实际样本通过 `extra_info` 保存辅助信息；演示新增的溯源字段属于本课扩展，原预处理器没有。原始 `rl` 表示来源划分，本课 `diagnostic` 表示用途；记录来源标签不等于决定本次 train/eval 协议。原始题面、难度和测试保存在 [原始记录](examples/maximum-subarray/outputs/raw_task.json)，随时可追溯。

对应练习：[清洗与 metadata](01_数据理解与GT构建.md#assignment-1-1)。

**例子：** LeetCode 53 的 `meta.query` 只有题面第一段和签名，没有约束也没有样例。完整题面在 `meta.question_title`，但约束的上标在抓取时丢了，字面是 `1 <= nums.length <= 105`，真实含义是 10^5。构造 prompt 时应按官方题面补写已审核的约束并记录来源，不能全局替换数字；同时保持隐藏测试和答案不进入 actor 输入。演示版追加的约束原文见 [01 的 actor prompt](01_数据理解与GT构建.md#actor-prompt)。

**完成条件：** 严格 JSONL 能读；必填字段、签名、返回类型和合法输入规则一致；来源和适配可追溯。`read_problems()` 的字典会覆盖重复 `task_id`，因此要在建字典前统计重复。

<a id="a2-filtering"></a>

### A2. 过滤、去重与划分：决定这道题能否进入本次实验

**人话：** 过滤不是“把看起来简单的题删掉”。先排除格式不完整、题意不可执行、答案不可核验的题，再防止同一道题出现在训练与锁定的评测中。来源难度过滤和 policy 难度过滤是后续的不同决策。

**[Code-R1 实现]** `leetcode2k()` 合并 v2 RL 与 v2 SFT 文件，排除 **v2 test 的 `question_id`**，再按 ID 保留第一次出现的题。它没有证明跨所有版本、所有 benchmark 的语义去重已经完成。

实际上 v2 RL 与 v2 SFT 各 945 题、互不重复，也都不与 v2 test 相交，所以这一步一行没删，得到 1890 题。

**版本协议决定一道题能不能训练。** 同一道题在不同版本里的身份可能不同：v1 的 175 道 test 题里，有 61 道在 v2 变成了 rl 或 sft 训练题。若锁定 **v1 test** 作为评测，用 v2 来源训练时必须排除这些题及相关题目家族；若按 Code-R1 的 **v2 test** 协议，它们可以训练，但训练后不能再拿它们在 v1 test 上的成绩充当未见题性能。

LeetCode 53 在 v1 属于 train 第 36 行，在 v2 属于 rl 第 143 行，哪个版本的 test 都没有它，按两种协议都是合法训练题。但本课已经用它演示、调试和扩充测试，所以把它记为 **诊断题**，即 `split=diagnostic`；最终盲测要另选没参与调试的题。

**[Code-R1 实现]** TACO 的活跃路径从 `likaixin/TACO-verified` 读取数据，跳过 `geeksforgeeks` 和 `leetcode` 来源、过短题面及包含图片提示的题目；再检查函数型或 STDIO 测试。名称含 verified 不意味着无需二次审查。函数分支里虽然还有 `source == "leetcode"` 的处理，但当前上游过滤已跳过这一来源，不应据这段分支描述活跃路径。

**完成条件：** 有保留、隔离、拒绝原因；按固定来源身份和题目家族查交集；报告实际实验划分。“来源划分”在本演示对应 `extra_info.original_corpus_split`，“本次用途”对应 `extra_info.split`。不另增加 `source_split / experiment_split` 两套重复字段，不覆盖原字段来抹去历史来源。

<a id="tests"></a>

### A3. 准备测试：输入、期望输出逐条配对

输入从原测试/样例或合法输入生成器取得。expected output 从可信既有结果、题意手算/穷举或经过审核的实现取得。生成器和 Teacher 给出的输出先作为候选；输入合法和答案正确是两项检查，任何一项不通过都不能进入已接受 Tests。

贯穿演示给每条测试一个 `case_id`，将 `input`、`expected`（预期输出）、来源与输入摘要 hash 保存在同一条记录中。Case 是一条检查，不是一个新的 Task；hash 仅用于对齐/复查，不证明答案正确。异步执行按 case_id 对回相同输入，不依赖返回顺序；从 TACO 的平行 inputs/outputs 数组适配时先检查长度相等，再逐项配对，不能让 `zip()` 静默截断或把不同输入的答案串位。

| 当前记录 | 检查 | 处理 |
|---|---|---|
| 原 seed：混合数组 → 6 | 来源测试与原样例一致，再用小规模穷举和已审核实现核对 | 本次输入上证据一致后接受 |
| 新输入 `[-3,-2,-5]` → -2 | 按题意完整枚举六个非空子数组，再与 reference 对照 | 接受并记录新增来源 |
| 相同新输入错配为 6 | 6 是另一份输入的结果；本输入穷举/reference 得到 -2 | 隔离错行，保留其他已接受测试 |
| `[1]` 同时对应 1、2 或 23 | 先记录冲突，再按该输入重新求解，不能多数投票或只按 ID 覆盖 | 查清后保留 1，对错行记录原因 |
| 空数组、非整数、越界元素 | 不满足当前任务约束 | 隔离输入；不拿它指控合法输入上的候选算法 |

一条错误的 expected output 会让正确程序失败，所以必须在组合 GT 前审核，而不是见到所有候选失败才猜模型差。若题意、函数入口或数据来源整体不一致，则隔离 Task 并查源；若只是一条错配，则定位该条并重验受影响的 GT 版本。新增、纠错与删除都记录原因，不静默改写来源输出。

**输出正确性怎样成立：** 对小输入，枚举全部合法子数组并取最大和直接对应题意，手算可以核查实现是否漏项。参考实现与这个独立方法一致是额外证据。对大输入，需要已审核的高效实现、可信正式结果或充分的数学判定依据；只有两个相似程序一致、结果落在合理范围、或模型多次回答相同，不足以单独认证答案。有限对照也不证明 reference 在所有输入上正确。

测试还应能识别确认错误的程序。三条 seed 的答案全是正数，没抓住允许空子数组的错误，补全负例后应能抓住它；这提高给定错误模式的检出能力，不保证覆盖全部错误程序。

逐条状态、原配对和审核结果见 [cases.json](examples/maximum-subarray/outputs/cases.json)。对应练习：[作业 1.2](01_数据理解与GT构建.md#assignment-1-2)。

**人话：** 设计哪些输入最能检查错误。随机输入发现意料之外的问题，边界输入检查规则的临界点，针对性输入检查已知错误模式。LeetCode 53 的演示三类都有：

| 类型 | 输入 | 想检查什么 |
|---|---|---|
| 针对性 | `[-3,-2,-5]`、`[-7]` | 全负数组：抓「允许空子数组、返回 0」的错解 |
| 针对性 | `[2,-5,3,4,-1]` | 前面累积为负时要丢掉重来，最优段 `[3,4]` 不从开头开始 |
| 边界 | `[0,0]`、`[10000,-10000,10000]` | 全零；元素取到约束的上下限 |
| 穷举 | 长度 1 到 4、元素 -2 到 2 的全部 780 个数组 | 发现没预料到的分歧，与暴力枚举逐个对照 |

新输入先记为 **候选 Test Input**；即使 Teacher 或参考程序同时给了答案，也只是候选答案，需要下一步核对。

**[Code-R1 实现]** LeetCode 分支复用现有 `test`，没有自动生成这些新增输入。TACO 的 `minimize_stdio()` 筛掉 `sys.getsizeof(stdin) > 4*1024` 的输入，并优先保留较短输入，默认最多 8 对。这是执行成本取舍，可能丢失复杂度测试，不能视为覆盖增强；它也没有限制 LeetCode functional 分支只能有 8 条断言。

**完成条件：** 每条输入满足规格，并标注为什么添加、打算识别什么错误。生成器必须满足题目约束：长度 1 到 10^5，元素 -10^4 到 10^4。空数组、字符串或 bool 元素、越界值都不是合法输入；演示把它们隔离，不拿来指控候选算法。

<a id="a4-answers"></a>

### A4. 求解与核对答案：方法可换，接受依据要保留

**人话：** 程序吐出一个值，只说明它算出了这个值。文献中的 Oracle 是“确定正确结果的方法/依据”，这里不要求开发一个统一的 Oracle 模块；用手算、暴力枚举或独立高效实现核对，并保留适用范围即可。

LeetCode 53 可以独立写一个便于审核的暴力 Oracle：

```python
def max_subarray_oracle(nums):
    # 枚举全部非空连续子数组，直接对应题意
    return max(sum(nums[i:j + 1]) for i in range(len(nums)) for j in range(i, len(nums)))
```

对新增输入 `[-3,-2,-5]`，六个非空连续子数组的和是 `-3、-5、-10、-2、-7、-5`，最大值 -2。手算与暴力实现都得到 -2，再对照 Kadane 参考解的结果。这段暴力代码是 O(n³)，适合小输入审核，不适合承担 10^5 规模的数据生成。

**[Code-R1 实现]** LeetCode 把原有测试结果当作已有判定依据，不在这一步重新求解所有答案。TACO 则执行 `solutions[-1]` 检查既有输出，不是由这份解重新生成所有 expected outputs；保存的 `extra_info.reference` 是 `solutions[0]`，二者不要混同。

**完成条件：** 输出有来源与适用范围；分歧先隔离，检查题意、输入、oracle 和 reference，不能自动相信较快实现或简单多数投票。性质检查也要区分充分与必要条件：答案一定不小于最大单个元素、不大于全部正数之和，数组没有正数时答案就是最大元素。这是必要条件，落在这个区间里不能证明答案正确。源码补强版的 oracle 库收录了 7 道题的独立 oracle、合法输入生成器和校验器，见[反向补充源码](examples/source-supplements/README.md)。

<a id="test-strength"></a>

### A5. 测试验收与扩展：答案可信，错解可检出

**人话：** 检查“正确程序能过”只完成了一半；还要拿已确认错误的程序验证这组测试能把它抓出来。每个变异体先找到有效反例，再把它标为错误程序。

LeetCode 53 的几种典型错解，以及能抓住它的输入。2026-10-13 用暴力 Oracle 逐条核对：

| 错误模式 | 违反的规则 | 原 3 条 seed 能抓住吗 | 能抓住它的输入 |
|---|---|---|---|
| 允许空子数组，初值取 0，即演示的 `wrong_empty` | 子数组必须非空 | 不能，3 条答案都是正数 | `[-3,-2,-5]` 应得 -2，错解得 0 |
| 直接返回总和，即演示的 `wrong_sum` | 要选一段，不是全部 | 能，seed-1 应得 6，错解得 1 | seed-1 |
| 只返回最大单个元素 | 最优段可以有多个元素 | 能，seed-1 应得 6，错解得 4 | seed-1、seed-3 |
| 前缀和做法漏掉空前缀 | 子数组可以从第一个元素开始 | 能，seed-3 应得 23，错解得 18 | seed-2、seed-3 |
| O(n²) 暴力枚举 | 结果正确，但 n 到 10^5 会超时 | 不能，小输入全对 | 只有大输入加时间上限才能区分；原版 prompt 没给约束，属于题目没说清 |

第一行最值得记：原始 3 条 seed 抓不住「允许空子数组」，所以在原始 Code-R1 数据上这个错解会拿 1.1；补上全负输入后才能抓住。测试数量不等于检出能力，要看覆盖了哪些错误模式。

**Code-R1 的测试有多少。** LeetCodeDataset 的测试是把题面样例解析成断言，再和 doocs/leetcode 题解交叉核对得到的。Code-R1 用到的 1890 题，每题的断言数都等于题面样例数，中位数 2 条、最多 5 条，93% 不超过 3 条。答案可信，覆盖却很薄，上表第一行的错解在整个 LeetCode 部分都可能拿满分。工业界判断测试够不够，看的是能不能抓住确认过的错解：用标好对错的提交衡量测试的准确率，设最少条数的门槛，再专门补边界、对抗和大规模输入。

**补强版的实测。** 源码补强版对全部题目做了变异审计：参考解的单点小改动共 17316 个，样例测试杀死 81%，每题杀死率不到一半的有 99 道。读这个数有两个边界：存活的可能只是等价改动；杀死率高也不等于测试够，LeetCode 53 的 9 个变异体全被杀死，wrong_empty 照样拿满分。补测试按「校验器、参考解、独立 oracle」三道关逐条审核，一致才进 GT；7 道示范题补完后，剩下的存活变异体与 oracle 对拍 200 次全部一致，wrong_empty 也被抓住。详见[反向补充源码](examples/source-supplements/README.md)。

参考解 `f = max(f, 0) + x` 也值得会讲：`f` 是以当前元素结尾的最大子段和，前面累积为负就丢掉重来；`ans` 记录所有 `f` 里的最大值。这是 Kadane 算法的不变量，它解释了为什么扫一遍就够。

**完成条件：** 输入合法、期望结果可信、重复执行稳定；分别报告错误程序检出比例及样本构造方法。该比例衡量给定错误样本集的检出能力，不是全部错误程序的总体检出率。

<a id="gt-packaging"></a>

### A6. 将已接受的 Tests 封装为 GT

**人话：** 原先是人能读懂的题目与测试，现在把同一道题的检查封装成引擎实际使用的判定数据。序列化不是正确性审核；它只改变存储形式。

**[Code-R1 实现]** `leetcode2k()` 的实际核心是：

```python
functional_tests = example["test"] + "\n\ncheck(" + example["entry_point"].strip() + ")"
ground_truth = json.dumps({"functional": functional_tests})
```

`functional` 包含 `def check(candidate)` 中的 3 条 assert，以及最终 `check(Solution().maxSubArray)` 调用。expected output 已嵌在 `== 6` 等断言右边，并没有丢失，也不要求再单独存一个同名字段。`json.dumps()` 外层是字符串，解码后才得到含 `functional` 的字典。

演示版的构造器只读 `accepted` 的 case，并将各自 input 和 expected output 编成自己的 assert；每条 assert 保留 case_id 的失败提示。隔离记录仍在审核报告中，却不混入 GT。JSON 解码、重新执行正确程序及确认错误程序、比较接受的 case 数与 GT 检查数，是封装后的回归；它检查传输/装配没有弄错，不能替代上一节的答案审核。

查看 [prepared_task.json](examples/maximum-subarray/outputs/prepared_task.json) 与 [data_pool.jsonl](examples/maximum-subarray/outputs/data_pool.jsonl)：题意给 actor，Tests/入口给 Verifier，metadata 解释来源和用途。Task Pool 只是这些记录的集合，没有另一种“带 meta 的题目”类型。GT 可以给训练评分接口消费，但本课记录是诊断用途；数据核验完成并不自动证明原评分协议和目标环境已可信。

对应练习：[封装与读回](01_数据理解与GT构建.md#assignment-1-3)、[池与接口](02_Verifier与任务池验收.md#assignment-2-3)。

实际样本保存在 `reward_model.ground_truth`，actor 看到的是聊天格式 `prompt`；`extra_info.reference` 是辅助信息，当前 `_compute_score()` 不使用它做代码文本相似度打分。SFT 学习某份目标回答；RLVR 在采样后按行为验收，可以接受与 reference 写法不同的正确解。

**完成条件：** JSON 往返不改变断言；入口存在；至少一条有效检查；可信正确程序能按既定 imports 契约运行；测试依赖的辅助函数在判分时都在。Code-R1 不拼原始 `prompt` 前缀，121 道树和链表题因此必然失败，见 [01 的内部审查例子](01_数据理解与GT构建.md#internal-external-review)；补强版已修复，见[反向补充源码](examples/source-supplements/README.md)。不要因为有 `check()` 函数就接受空测试，也不要把 reference 或隐藏 GT 拼进 actor prompt。

<a id="verification"></a>

### A7. Verifier + Sandbox：机制复现与可信运行分别验收

**人话：** 同一套题拿给好程序、坏程序、死循环、格式错误和提前退出程序，确认各自得到预定结果，再检查故障能不能正确归因。

**[Code-R1 实现]** 代码提取、functional 调用与奖励映射集中在 `verl/utils/reward_score/coder1/__init__.py`；执行后端由 `CODER1_EXEC` 选择。这里常常不是“Sandbox 返回 Actual Output，另一个程序再比较”：functional 模式把候选解与 assert 放在同一程序中执行，`AssertionError` 通过执行失败传回评分器。STDIO 模式才在上层比较 stdout 与 expected 字符串。

**学习主线完成条件：** 能用已检查的固定程序复现格式、测试与 reward 路径，逐项指出已知缺口，并解释它们影响什么。**可信运行条件：** 隔离、测试完成证据、超时清理和故障处理要在目标后端验收；未知模型代码实跑不能由本课 subprocess 代替。发现原版缺口后，不要求学习者先重建工业引擎才能完成源码机制学习；正式能力结论仍需要补齐相应条件。具体缺口见第三、第五章与[历史记录](#historical-evidence)。

### A8. Difficulty Filtering 与 Task Pool：奖励可靠之后，再决定训练取样

**人话：** 原始 `Medium` 表示题库对一般刷题者的分级；当前模型是否能做，要拿固定 checkpoint 和采样参数试。难度不是题目永久不变的属性。

按每题有效生成数量 `G` 和通过数量 `c` 估计成功率 `c/G`，同时记录格式失败、运行失败、infra error 与组内 reward 方差。例子：4 次有效提交中 2 次通过，估计成功率为 1/2；这不是 1 次运行即可确认的稳定难度。小样本的不确定性和采样温度都应保留。

**[Code-R1 实现]** 活跃预处理器合并、打乱数据并导出 `train.parquet` / `test.parquet`；没有在导出前完成上述 policy 难度估计。实际目录名按 `round(len(train_dataset)/1000)` 和数据来源生成，而 `main_grpo.sh` 使用 `DATASET=code-r1-12k`，学习时要核对实际文件路径，不能假设运行预处理后自动匹配脚本。

**完成条件：** 训练和评测文件与协议一致；训练难度筛选可选并带 checkpoint；未验收和 infra 故障样本先隔离。源码补强版的选择工具实现了按实测通过率筛题与评测集去重，见[反向补充源码](examples/source-supplements/README.md)。全对或全错不是永久无用，只有在当前组归一化的 outcome 信号上缺少相对差异；KL 等其他梯度项可能仍存在。

<a id="a-reference-answers"></a>

### 条件分支：采用参考程序求新增输入答案时，分别检查什么？

只有选择“由参考程序求新增答案”这条路径时，才需要下面两层检查。已有可信结果可以复用，手算或完整穷举也可以直接建立配对依据；Teacher 和 Ref Code 不是每题必须开发的模块。

**第一次：Reference Code 初步验证**

先确认程序能够运行并通过已有可信样例，避免用明显错误的程序生成候选 expected。通过样例仍不证明全部输入正确。

**第二次：Expected Output 独立验证**

再检查具体 input/expected 是否符合题意，记录适用范围与接受依据。它不同于重复检查程序能否运行。

```text
Reference Code
     ↓
检查可执行性 + 已知样例
     ↓
运行新增的合法输入
     ↓
得到 Expected Outputs
     ↓
逐条核对答案与配对
     ↓
已接受 Tests
```

这两次检查分别回答：

- Reference Code 是否具备生成候选结果的基本条件？
- 生成出来的候选结果是否有足够证据可以成为 GT？

三种答案来源与核验方法集中见[答案核验](#answer-validation)，不在主线再创建一个 Oracle 对象。

### 工业依据

以下是公开的技术方法，不代表相关厂商完整的内部 Pipeline 已经公开。

- [Google DeepMind CodeContests](https://github.com/google-deepmind/code_contests)：公开数据包含成对的输入输出测试，以及正确和错误的人类程序，可用于构造和审查 Verifier。
- [Qwen3-Coder](https://qwenlm.github.io/blog/qwen3-coder/)：官方介绍了通过自动扩展不同代码任务的测试用例，构建大规模 Code RL 训练数据。
- [DeepSeek-R1 v1，2.2.2 节](https://arxiv.org/html/2501.12948v1#S2.SS2.SSS2)：公开论文介绍了代码任务使用编译器和预定义测试集产生基于规则的正确性 Reward。

Train/Eval 应在任务级隔离，并在测试扩充后再次检查泄漏。Difficulty Filtering 只针对训练池，不随意修改已经锁定的评测集。

难度估计需要记录所用的 Policy Checkpoint；随着 Policy 更新，可以按需刷新训练任务的难度标签。

对于 GRPO，优先选择具有适当成功率和 Reward 方差的任务。全对或全错的采样组没有组内相对优势信号，但不意味着必须永久删除这些任务。

<a id="grpo-advantage"></a>

## 阶段 B：在线 RLVR 训练

这里衔接 Ray + verl + rollout 引擎 + GRPO。下图是通用机制概览；Code-R1 的具体调用顺序与实际 reward 另列。CPU 演示只复现已检查程序的评分、组统计和数值，不调用模型或更新参数。

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
          Reward（通用示意 0 / 1）
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

对于 GRPO，假设每个任务采样 $G$ 个 Candidate Codes，对应 Reward 为 $r_1,\ldots,r_G$，常用的组内优势估计为：

$$
A_i=\frac{r_i-\bar r}{\sigma_r+\epsilon}
$$

**这里是通用0/1教学设计；当前Code-R1使用实际三档分数与异常回填。** 二值例子下优势尺度由组内成功/失败数量决定；多候选全同分组缺少相对outcome信号，不代表其他loss项无梯度。

Partial Reward 可以提供更稠密的训练信号，但可能增加针对测试集优化的风险，作为可选实验保留。

Logprob 的计算与用途：

| Logprob | 计算时机 | 作用 |
|---|---|---|
| Old | Rollout 时记录，或由冻结的 Old Policy 重算 | Importance Sampling |
| New | Actor 训练时 | 梯度更新 |
| Ref | 启用 Reference KL 时 | 约束策略偏移 |

直接策略 Loss 统计有效 response token；若回答含推理与代码，两者都可能参与。Prompt 位置不计直接策略 Loss，但作为条件参与 forward，梯度可经相关计算影响共享参数。此单题协议中的隐藏 Tests 与外部执行反馈不作为生成 action token，不能将“没有直接位置 loss”理解成“条件计算路径没有梯度”。

SFT Cold Start 可以放在 RL 之前，但不是所有 Code RLVR 都需要单独进行一次 SFT。

## 阶段 B 的人话与 Code-R1 对照：奖励如何变成训练

**人话：** GT 是判卷依据；Actor 每次自己写答案；Verifier 判卷；GRPO 比较同一道题这组回答谁更好，再提高较好回答的采样概率。它不会把执行日志重新当作当前回答的 action token，也不按 reference code 的文本一致率训练。

**[Code-R1 实现]** `main_grpo.sh` 配置每题生成 `16` 个回答，使用 vLLM、GRPO 和 PrimeRewardManager。这是脚本配置，不表示本机已经完成该训练。`ray_trainer.py::fit()` 的关键顺序是生成回答、为原始题目分配 `uid` 并重复对齐、重算 `old_log_probs`、计算 reference logprob、评分、求 advantage、更新 actor。图里的旧/新/ref logprob 展示的是用途，具体调用时序以源码为准。

当前源码的运行关系可记为：`生成 → uid/repeat 对齐与 balance → old/ref logprob → reward → advantage → actor update`。贯穿演示的四个算法候选 `reference / correct_prefix / wrong_empty / wrong_sum` 由教学代码固定提供，得到 `[1.1,1.1,0.1,0.1]`；它们并非模型新 rollout。对它们求 advantage 可以先看清信号怎样传播，之后才在明确模型、Linux 执行环境和预算的条件下进行真实采样。

以 **教学计算** 举例：把这四个候选按分数排好，格式都合格，2 个测试失败、2 个通过，分数是 `[0.1, 0.1, 1.1, 1.1]`，均值 0.6；按当前 `compute_grpo_outcome_advantage()` 使用的 `torch.std` 样本标准差，标准差约为 0.57735，忽略很小的 epsilon 后，advantage 约为 `[-0.86603, -0.86603, +0.86603, +0.86603]`。这些是手算数值，不是历史训练日志。

这个例子若改成二值 `[0,0,1,1]`，组标准化后近似相同，因为这里仅作了正比例缩放与平移；但实际 Code-R1 还有格式失败 `-1.1`、管理器异常回填 `0.0`，不能因此把所有情形统称为二值奖励。尤其 `reward > 0` 不是通过条件：测试失败也有 `0.1`。

`PrimeRewardManager` 把每个回答的标量 reward 写到**最后一个有效 response token**，形成 token reward 张量；GRPO 先沿 response 维求和，按同题 `uid` 分组归一化，再把同一个 outcome advantage 扩到该回答的有效 token。序列因 token 均衡可能重排，所以不能简单按相邻 16 行认定同组。当前源码对只有 1 个回答的组还有单独处理；学习标准组相对信号时应使用 `G >= 2`。

Logprob 是模型对已生成 token 给出的对数概率。Old 是更新前策略的概率基准，本仓库在 rollout 后、actor 更新前重算；New 是训练 forward 的当前概率；Ref 是固定参考模型的概率，供 KL 项使用。`compute_policy_loss()` 计算 `exp(new_logprob-old_logprob)` 并裁剪，不能把原始 logprob 直接相除。这里的 Reference Policy 不是解题 Reference Code。

**完成条件：** 能追踪一题多回答的 `uid`、有效 token mask、终止位置 reward、old/new/ref 的来源和更新顺序。训练生成若包含推理与答案，当前 response mask 可以使两者的有效 token 都参与优化；只检查最终代码的测试，不等于只训练代码 token。Weight Sync 在下一轮生成前通过 FSDP-vLLM sharding manager 使推理引擎使用当前参数；不要仅凭流程图推定具体 wake/sleep 顺序。

**一个必须追到底的异常例子：** 原管理器可能将单行评分异常回填为 `0.0`，这条值仍会进入当前 GRPO 组统计。若同题三个格式合格但测试失败的提交是 `0.1`，第四行评分基础设施异常回填 0，组分数 `[0.1,0.1,0.1,0]` 的均值为 0.075、样本标准差为 0.05；忽略 epsilon，优势为 `[0.5,0.5,0.5,-1.5]`。相对信号被基础设施故障改变，不能只在日志里备注“有异常”。这是按源码公式的教学推演，不是实测故障或性能结论。处理重试、隔离或补采时要预先约定组完整性与分母，不能事后丢掉失败样本来提高分数。

对应练习：[受控组的完整示范](03_Rollout与GRPO机制.md#assignment-3-1)、[条件变化](03_Rollout与GRPO机制.md#assignment-3-2)、[独立检查](03_Rollout与GRPO机制.md#assignment-3-3)。

<a id="gradient-bridge"></a>

### 从分数到参数：必须补齐的求导链路

logit 是模型对词表各 token 给出的未归一化分数。给定正温度 $T$，先做 softmax，再取实际生成 token $y_t$ 的对数概率：

$$
\ell_t=\log\pi_\theta(y_t\mid q,y_{<t})=z_{y_t}/T-\log\sum_v\exp(z_v/T).
$$

这里的符号对应模型概率，不是测试 expected output。因果模型前一位置预测下一 token；本版本普通路径取 `logits[:, -R-1:-1, :]` 与 `responses` 对齐。old/ref 重算和 new forward 使用相应配置的温度，不能只在生成时改温度而忽略概率重算。

old logprob、ref logprob 和组 advantage 是固定数值；new logprob 来自当前带梯度 forward。判题程序不必可微，梯度来自以固定 advantage 加权的模型 logprob。不是经 subprocess 反传，也不是拿 Reference Code 做逐词监督。

$$
r_t=\exp(\ell_t-\ell_t^{old}),\qquad
L_t=\max(-A r_t,-A\,\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)).
$$

未裁断区域有 $\partial L_t/\partial\ell_t=-A r_t$。A>0且r>上界，或A<0且r<下界时，该PG项可变成常数、梯度为0；其他token、KL/entropy与共享参数仍可改变其概率。这是目标裁剪，不是梯度范数裁剪，也不保证所有更新后概率比都落在区间内。new和old数值相等时r=1，固定old仍允许非零梯度；把old也连到同一个new计算图会错误抵消信号。

**长度与累积：** 源码在每个micro batch内对有效token执行 `masked_mean`，再按固定gradient accumulation或动态行数比例缩放并backward。这不同于逐回答等权，也不一定等于整个mini batch统一按所有token平均。两回答逐token loss为+1/-1、长度为2/6，同一micro内均值为-0.5；拆为两个等权micro则为0。这是教学计算，配置变化也可能改变权重关系。

**mask边界：** `get_eos_mask()`包含首个EOS；没有EOS时全1。pad与EOS不同、缺少真实长度时，不能靠该函数无条件屏蔽pad。mask不能证明自然停止；当前rollout返回字段没有完整保留停止原因，须另记引擎信息。参见[源码细节与练习](03_Rollout与GRPO机制.md#gradient-bridge)、[两标量梯度实验](examples/maximum-subarray/mechanism_lab.py)。

<a id="policy-update"></a>

### 从数值练习进入实际更新

先确定真实推理资源与判题条件，再制定带梯度训练的 batch、KL、保存和恢复配置。`update_actor()` 的实际参数变化、下一轮 vLLM 使用的新 policy 版本、保存/恢复状态及固定开发验证，分别需要运行证据。手工 logprob、受控程序或 loss 数值都不能代替真实参数更新；一批次跑通也不代表泛化提高。

对应练习：[阶段 04 的更新、同步与恢复](04_真实训练与独立评测.md#assignment-4-2)。

<a id="execution-boundary"></a>

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

## 用仓库理解 Sandbox 与 Verifier 的具体边界

**人话：** Sandbox 像限制权限、时间和资源的考场；Verifier 像判卷机。考场不越权，不能证明学生确实答完了卷子；判卷机正确比较答案，也不能证明程序的执行权限安全。两项要分别检查。

| 路径 | 实际怎样运行 | 容易混淆的地方 |
|---|---|---|
| LeetCodeDataset 原生评测 | `problem["prompt"] + completion + test + check(entry_point)`，imports 自动在前面 | 它接受去掉 imports 的 completion；`reliability_guard()` 源码明确说不是安全 Sandbox |
| Code-R1 functional 评分 | 提取候选代码，拼接 GT 的 `functional`，交给所选 executor | 不自动注入上述 LeetCode prelude；prompt 要求自包含代码 |
| Code-R1 STDIO 评分 | 分别执行输入，比较 `output.strip()` 和 `stdout.strip()` | strip 是具体比较政策，不等于所有题都允许任意空白归一化 |
| Code-R1 Firejail 后端 | 设置私有环境、进程数、文件数、内存和时间限制；返回退出状态与日志 | 非零退出混合了断言、运行及其他失败，未天然提供细粒度归因 |

**例子，2026-10-13 实测：** LeetCode 53 的参考解签名里有 `List[int]`。拼上原始 prelude 能通过；不拼前缀、作为自包含 RL 提交时，Python 3.12 报 `NameError: name 'List' is not defined`，Python 3.14 却能通过，因为 3.14 把注解改成了延迟求值。这是提交契约或环境适配问题，不能认定算法错误；判分结果还取决于训练镜像的 Python 版本。树和链表题更严重：测试本身调用前缀里的辅助函数，见 [01 的内部审查例子](01_数据理解与GT构建.md#internal-external-review)。

**提前退出的缺口。** LeetCode 53 演示用原始 parser 与 `_compute_score()`，只把执行后端换成受控的本地子进程：格式合格、第二行就 `sys.exit(0)` 的 `early_exit` 得到 1.1。测试代码排在候选代码后面，候选提前以退出码 0 结束，后面的 assert 一条都没执行，评分器只看到执行成功；另做的逐条审计显示它完成的测试数是 0。这验证了评分协议缺口，**没有实跑 Firejail 或 Docker**；当前 Firejail 源码同样只按 `returncode == 0` 判成功，正式后端验收需检查这种情况。这个演示是固定程序的独立检查，不是受保护的通用协议；本课复现原版机制、指出缺口之后可以继续离线学习，正式执行未知代码和提出可信能力结论仍需目标环境与判分协议的证据。源码补强版已经实现受保护的判题协议和结构化沙箱，把这个缺口堵上，见[反向补充源码](examples/source-supplements/README.md)。

**[通用设计]** 高可信验证应由受保护的测试控制方确认测试开始、实际执行数量、完成状态与结果，再给 reward；缺失执行结果不能默认成功。把可伪造的完成字符串与 candidate 放在同一可读写空间，不足以解决问题。通用母版中“candidate 无法读取或篡改隐藏判定依据”是验收目标，当前 functional 拼接实现不能被描述成已经充分满足这一目标。

**本机边界：** 资料最初在 Windows 上编写，现在在 macOS 上复现。`main_grpo.sh` 和 Firejail 只能在 Linux 上运行，原生 `execution.py` 的时间限制使用 `signal.setitimer/SIGALRM`。本课的小规模审核只在本机受控执行已检查的代码，没有启动整个原生评测器，也没有搭建 Linux GPU 训练环境。部署执行应在与训练一致的 Linux 环境验收，不能用本机的子进程代替安全隔离结论。

需要保证：

- Candidate Code 无法读取隐藏的 Expected Outputs 或篡改评分器。
- 不同任务的执行环境相互隔离，限制网络、文件权限和资源。
- 基础设施异常应执行有限次数的重试；超过上限后隔离，并记录异常，不能直接转化为模型负奖励。

对于单题代码 RLVR，通常使用 CPU Sandbox 执行代码，GPU 则主要用于模型 Rollout 和训练。

### Sandbox 总执行预算

设每批任务数为 $B$，每题采样 $G$ 个候选程序，每个候选程序执行 $M$ 条测试，则最大计划测试执行量为：

$$
N_{\mathrm{test}}=BGM
$$

这些测试不一定需要启动相同数量的 Sandbox 实例，可以在一个隔离实例内批量执行。

Binary Reward 可以采用 Early Exit，在首次明确失败后停止剩余测试，以减少执行成本。但需要完整 Partial Pass Rate 或逐条测试诊断时，应运行完整测试集或采用单独的诊断流程。

<a id="evaluation"></a>

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

## 用人话理解独立评测与回流

**人话：** 训练卷分数高，不能自动说明换一套题也会做。诊断集用来找错和修方法；开发验证集用来选方案；最终测试集用来报告效果。这些用途应在实验前分开。已经针对最终测试错误改过方法，就要承认它被用于开发，后续另留未参与选择的评测。

**[Code-R1 实现]** `LeetCodeDataset/leetcote/evaluate.py` 从每道题的候选数 `n`、通过数 `c` 计算 pass@k，再对题目取均值：

实际必须分清三条通路：

| 通路 | 候选与判定依据 | 得到什么 |
|---|---|---|
| Code-R1 trainer `_validate()` | `do_sample=False`；vLLM 贪心温度 0、每题 `n=1`；调用原 reward | `val/test_score/{data_source}` 是 raw reward 平均值。LeetCode/TACO 都标 `code`，会合并来源 |
| 原生 LeetCode `eval_lct` | 按 version/split 找 release problem file，要求覆盖所传题集；使用原生提取、imports 和 check 契约 | 原测试下的逐题通过与 pass@k；不直接消费本课 Parquet/JSONL 中新增的 GT，也不能直接评混合 TACO 池 |
| 本课小池演示 | 固定受控候选、当前接受 Tests 与同 scorer；另检查测试完成和实际结果 | mean reward、correctness 与有限候选池 pass@k 分列，不能叫模型采样或未见题成绩 |

因此原始 `0.1` 可能代表格式合格但正确性失败；平均 reward 不等于通过率。原生评测器的公式可以用于聚合，但直接运行旧 release CLI 可能又读回旧测试。小池应使用明确的当前 GT/执行政策取得每题 `(n,c)`，或对纯 LeetCode 子集构造专用 problem file 再调用低层接口；必须固定提取、imports、比较与异常政策。

$$
\widehat{\mathrm{pass}}@k=1-\frac{\binom{n-c}{k}}{\binom{n}{k}},\qquad n\geq k.
$$

这里 pass@k 指同一题抽取 k 个候选时至少有一个通过的估计；不是 k 条 test case 的通过比例。只有每题都有至少 k 个候选时，该实现才报告该 k。例子：同题 `n=4,c=2`，pass@1 为 1/2，pass@2 为 5/6，pass@4 为 1。这个估计受生成协议、采样相关性和测试可靠性约束，不能仅凭数字认定普遍正确。

题目平均与样本平均也不同：A 题 1/1 通过、B 题 0/9 通过时，按题平均 pass@1 为 1/2，按全部候选计成功率为 1/10。报告必须写清分母、每题候选数、数据与测试版本、模型 checkpoint、温度和 token 限制。

格式失败、代码缺失、截断等策略失败仍计入原生成候选分母；基础设施故障依预登记政策重试/标无效，并报告原生成数、有效执行数和拒绝数。不能事后只挑格式正确或执行成功的回答计算“成功率”。本课四算法候选组与包括格式/提前退出负控制的六候选合集分别报告，不能将前者当作过滤失败后的真实模型成绩。

对应练习：[指标与 baseline](04_真实训练与独立评测.md#assignment-4-1)、[独立评测与归因](04_真实训练与独立评测.md#assignment-4-3)。

**回流例子：** `wrong_empty` 通过了 LeetCode 53 原始的 3 条 seed，却在新增的全负输入 `[-3,-2,-5]` 上返回 0。算法 bug 属于候选；原测试没抓住它，又揭示了测试覆盖缺口，两件事同时成立。回流时要把全负输入补进测试并升测试版本，不能只修这一份回答就宣称旧测试已经可靠，也不能把这道诊断题上的改进当成未见题泛化。

**完成条件：** 用固定协议比较 baseline 与新 checkpoint；单独记录 format/runtime/infra 故障和成本；对修复所用题集与最终报告题集说明隔离边界。来自最终测试的信息若参与数据修复或调参，必须记录它的开发用途，不能无说明地回流后继续当盲测。

具体归因方法在第六部分展开。

<a id="verifier-internals"></a>

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
        读取同一 Task 的判定/执行配置
       ├─ Language
       ├─ Input / Output Format
        ├─ 已接受 Tests（保存为 GT）
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

其中，GT 是第二部分已接受 Tests 的接口表示；图中的配置项属于同一 Task 和实验环境，不要求另建一个 Task Config 主对象。

这里“GT 已完成”表示**本次联合验收所用的版本已接受**；并不要求通用 Verifier 必须等所有题的 GT 完成后才开始开发。先用一题的可信小测试验证引擎，再增加任务配置并逐批验收即可。引擎是公共能力，GT 是每题的判定数据，任务配置连接二者。

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

下面是通用0/1奖励设计示例，不是本项目原scorer的默认值：

$$
R_i=\prod_{j=1}^{M}I(\operatorname{Pass}_{ij})
$$

候选程序 $i$ 必须通过全部 $M$ 个测试才能得到 1，否则为 0。

也可以使用 Partial Reward：

$$
R_i=\frac{1}{M}\sum_{j=1}^{M}I(\operatorname{Pass}_{ij})
$$

两者的区别是，Binary Reward 强调最终正确性，Partial Reward 提供更密集的学习信号，但可能增加测试集过拟合的风险。

通用设计可以用0/1；Partial Pass Rate是可选训练信号与诊断指标。Code-R1的当前functional/STDIO都非逐条partial reward，实际映射见下一节。

<a id="reward-mapping"></a>

### Code-R1 实际奖励和原生评测器的区别

**[Code-R1 实现]** `SYSTEM_PROMPT` 要求 `<think>...</think><answer>...</answer>`；评分器先检查该结构，再提取最后一个 answer 中的 Markdown 代码块。原生 `extract_completion()` 则从 Python 代码块中取最长块，两者不能互换描述。

| 原始 `_compute_score()` 情况 | 默认分数 | 人话 |
|---|---:|---|
| 格式不合格或没有可提取代码 | -1.1 | 没有按提交协议交出有效代码 |
| 格式合格，测试执行失败 | 0.1 | 格式分保留，正确性未通过 |
| 格式合格，执行报告成功 | 1.1 | 格式分与正确性分之和；还需保证测试真的完成 |
| RewardManager 的评分异常或等待超时 | 可能回填 0.0 | 管理器兜底值；不等于原始 correctness reward |

functional 分支是整体通过/失败，没有逐断言 partial reward；STDIO 分支遇到任一不通过就返回 `0.1`。`PrimeRewardManager::single_compute_score()` 的参数名 `reference` 在这条调用链中装的是 **ground_truth**，不是 `extra_info.reference`；名字不能代替值来源追踪。

**[通用设计]** 建议返回结构化状态和测试完成证据，先区分 candidate 失败与 infra error，再映射 reward。当前接口主要是 bool 与输出日志，源代码也会将部分异常填 0；因此母版的“infra error 有限重试后隔离”是应补齐的设计，不能标成现有完整实现。用 `asyncio.wait_for` 停止等待，也不自动证明底层 executor 线程及它启动的进程已经终止。

**验收例子：** LeetCode 53 准备一组固定候选：正确解 `reference` 与 `correct_prefix`、允许空子数组的 `wrong_empty`、返回总和的 `wrong_sum`、没有 think/answer 标签的 `bad_format`、提前退出的 `early_exit`；建议再加一份漏写 `from typing import List` 的提交。检查格式分、正确性分、实际执行和错误分类分别来自哪一层，而不是只看最后一个浮点数。生产级验收还应加入死循环、资源耗尽、服务中断、输出比较与结果伪造等目标后端测试。

演示复用锁定源码的 parser 与 `_compute_score()`，仅将后端换为已检查程序的本地 subprocess。STDIO 是同题的教学适配，原始语料仍为 function；stdin、expected stdout、actual stdout 也逐条绑定，并按原源码 `.strip()` 规则比较，不能说这是原始 TACO 样本或新建的生产 Sandbox。

对应练习：[评分真值表与 STDIO 对照](02_Verifier与任务池验收.md#assignment-2-1)、[条件变化与边界](02_Verifier与任务池验收.md#assignment-2-2)。

### 异常应该如何处理？

下表是通用0/1异常策略设计；Code-R1实际score与manager回填另列，不能将本表认作当前完整故障治理。

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

对于 INFRA_ERROR，应配置 `max_retries`。达到上限仍然失败的样本不进入正常 Reward 训练；如果错误率超过预设阈值，则暂停相关 Worker 或任务队列并排查。源码补强版的奖励管理器 `prime_plus` 按这个思路实现：判题故障先重试，仍失败的整组作废，并输出故障统计，见[反向补充源码](examples/source-supplements/README.md)。

<a id="failure-attribution"></a>

# 六、边界问题与 Failure Attribution

<a id="answer-validation"></a>

## 1. Teacher 生成输入后，如何确认 Ref Code 的输出真的是 Expected Output？

这是“采用 Teacher/参考程序生成新增答案”路径的可信性问题。原 LeetCode 封装既有 Tests、手算和穷举等路径不必经过 Teacher；此处集中保存核验方法。

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

<a id="layer-attribution"></a>

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

$$
GT_{\mathrm{operational}}=\{(x_j,y_j^*)\}_{j=1}^{M}
$$

它由 Problem Specification 派生，不等于完整的 Problem Specification，也无法保证覆盖所有输入。

**边界二：Reference Code 是可选的 Oracle 构造工具。**

它可以离线生成 Expected Output，但不是每个 Test Case 都需要一份 Reference Code，也不需要在每次在线 Rollout 时执行。

**边界三：通过全部 Test Cases 不代表程序在所有输入上正确。**

$$
\operatorname{Pass}(\text{Test Suite})
\not\Rightarrow
\text{Universal Correctness}
$$

因此需要通过边界测试、随机测试、独立 Oracle 和 Held-out Evaluation 提高验证可信性。

## 4. 同一道题怎样避免把不同失败混为一谈

| 现象 | 先检查什么 | 应得出的有限结论 |
|---|---|---|
| `wrong_empty` 通过原 3 条 seed，却在全负输入上失败 | 原测试覆盖了哪些错误模式 | 算法 bug 属于候选，原测试没抓住是覆盖缺口；两者同时成立 |
| 一条测试把 `[-3,-2,-5]` 配成 6，正确解反而失败 | 对该输入独立枚举 | 测试数据错误：隔离这一行、保留同输入的正确行，不能据此惩罚正确程序 |
| 参考解不拼前缀时报 `NameError: List` | 原生 prelude 与 RL 自包含协议；运行环境的 Python 版本 | 运行契约问题，不是算法错；Python 3.14 下同一份代码能通过 |
| 树和链表题的回答全组 0.1 | 测试依赖的辅助函数判分时是否都在 | Code-R1 不拼前缀造成的执行契约问题，不是模型不会做 |
| `early_exit` 得 1.1 | 测试是否真的执行完 | 判分协议缺口，不是模型能力 |
| O(n²) 暴力在大输入上超时 | 题面是否给了约束，时间上限是否绑定固定环境 | 原版 prompt 没给约束，先算题目没说清，不直接判模型错 |
| 用 v2 训练过的题在 v1 test 上得高分 | 跨版本题目身份和使用协议 | 结果可能反映已见题能力，不是未见题性能的证据 |

确定性 correctness 测试与性能测试也要分开。时间上限取决于固定环境、输入规模、硬件和并发，不能任意缩短超时来制造错误解。暴力 Oracle 在小输入上是可信工具，在最大规模上慢并不否定小输入结果。

---

<a id="historical-evidence"></a>

# 七、证据边界、历史归档与来源

## 1. 本课实际执行过什么

下表结果均于 2026-10-13 在 macOS、Python 3.12 实跑得到；前三行由 `run_demo.py all` 生成，后两行是单独的对照。

| 检查 | 结果 | 能说明什么 |
|---|---|---|
| 本地数据文件记录数 | v1 train=1570、v1 test=175；v2 rl=945、v2 sft=945、v2 test=200 | 文件行数，不是最终过滤后的 12K 训练池数量 |
| LeetCode 53 逐条测试审核 | 8 条接受、7 条隔离；780 个小数组与暴力枚举全部一致 | 有限小域的对照，不是全输入正确性证明 |
| 原 scorer 加受控本地子进程 | `bad_format`=-1.1，两个错解=0.1，两个正确解=1.1，`early_exit`=1.1 | 奖励映射与提前退出缺口已复现；不是生产执行后端验收 |
| 不拼前缀运行参考解 | Python 3.12 报 `NameError: List`，Python 3.14 通过 | 运行前缀和 Python 版本都会改变判分 |
| 树和链表题的前缀依赖 | 1890 道 LeetCode 训练题中 121 道必然失败；v2 test 0/200 | GT 不拼前缀造成的系统性假阴性 |
| 反向补充：GT 执行契约，2026-10-14 | 补强版下 121 题全部从 0.1 变成 1.1；剩下 14 题因缺第三方库失败，1 题随哈希种子时过时不过 | 前缀依赖已修复；其余两类分别属于运行环境和 verifier，见[反向补充源码](examples/source-supplements/README.md) |
| 反向补充：测试强度审计，2026-10-14 | 2014 题、17316 个变异体，样例测试杀死 81%；按测试条数从 1 条的 80% 升到 5 条的 87% | 样例测试对参考解附近的错误检出率尚可，但不能说明对远处错误够用 |
| 反向补充：测试补充与逐条审核，2026-10-14 | 7 道示范题补完后剩余存活变异体对拍全部一致；LeetCode 53 的 wrong_empty 从 1.1 变成 0.1 | 用独立 oracle 核对后补的测试确实提高了检出能力；疑似等价只靠对拍，不是证明 |
| 反向补充：受保护判题协议与沙箱，2026-10-14 | 提前退出、os._exit、伪造完成帧、读取数据文件都判 0.1；沙箱禁网、禁写工作目录外、禁读仓库，超时后子进程被回收；word-ladder-ii 改用忽略外层顺序的比较后 8 次全过 | 判题结论由判分方得出；macOS 的 sandbox-exec 只用于本机开发，训练仍用 firejail |
| 反向补充：判题故障策略与数据选择，2026-10-14 | 同组一条判题故障时，原版 advantage 为 +0.5、+0.5、+0.5、-1.5，补强版整组为 0；与 v1 test 去重找到 61 对同题号和 3 对措辞重叠 | 故障策略的纯逻辑已测，奖励管理器整体待 Linux 验证；难度筛选需要真实采样的通过率 |

运行入口与完整结果见 [LeetCode 53 演示资产](examples/maximum-subarray/README.md) 和它生成的 [运行报告](examples/maximum-subarray/outputs/report.json)。

## 2. 没有越过的证据边界

本课没有做完整 Oracle 形式证明、最大规模性能测试、Firejail 或 Docker 的安全与并发验收、真实模型 rollout、GPU 参数更新或作者训练效果复现。文中的扩展设计不代表相应 pipeline 已实现。源码证据只覆盖页首锁定的提交；一手论文和项目文章只支持其公开陈述，不支持把本框架认定为某家公司的完整内部流程。

LeetCode 53 已被用于讲解、调试和扩充测试，所以它在本课只作诊断题，演示结果不当作未见 benchmark。要跑真实实验，先决定用 v1、v2 还是其他独立协议，再核对跨来源题目家族与最终测试边界。

## 3. 旧版资料的历史归档

资料的早期版本曾用另外五道 LeetCode 题做过一次小规模核验。那次核验的脚本、报告和当时的文档按原字节保存在 [只读历史归档](examples/history-20261003/README.md)，不属于当前贯穿题，也不再作为正文例子；当前正文的所有机制例子都改用 LeetCode 53。不要在归档目录里直接运行旧脚本，它会覆盖旧报告。

## 4. 直接来源与版本定位

- 本地 [LeetCode v1 test](../LeetCodeDataset/data/LeetCodeDataset-v1-test-problems.jsonl) 与 [v2 RL](../LeetCodeDataset/data/LeetCodeDataset-v2-rl-problems.jsonl) / [v2 SFT](../LeetCodeDataset/data/LeetCodeDataset-v2-sft-problems.jsonl) / [v2 test](../LeetCodeDataset/data/LeetCodeDataset-v2-test-problems.jsonl)：字段、版本位置与运行实例。
- 本地 [coder1.py](../examples/data_preprocess/coder1.py)：`minimize_stdio()`、`taco()`、`leetcode2k()` 和 Parquet 导出；核对版本为页首 Code-R1 commit，不以在线 main 的未来内容代替本地证据。
- 本地 [Code-R1 scorer](../verl/utils/reward_score/coder1/__init__.py)、[Firejail executor](../verl/utils/reward_score/coder1/firejail_exec.py)、[PrimeRewardManager](../verl/workers/reward_manager/prime.py)：格式、GT 消费、执行成功定义、分数和异常兜底。
- 本地 [原生执行器](../LeetCodeDataset/leetcote/execution.py)、[原生评测器](../LeetCodeDataset/leetcote/evaluate.py)：imports 前缀、时间限制、候选统计与 pass@k。
- 本地 [ray_trainer.py](../verl/trainer/ppo/ray_trainer.py)、[core_algos.py](../verl/trainer/ppo/core_algos.py)、[FSDP-vLLM manager](../verl/workers/sharding_manager/fsdp_vllm.py)：奖励张量、uid 分组、样本标准差、概率比裁剪与权重同步。
- [LeetCode 53 官方题面](https://leetcode.com/problems/maximum-subarray/description/)：题意与约束的外部依据；演示追加的约束据此补写。
- [DeepSeek-R1 v1](https://arxiv.org/html/2501.12948v1#S2.SS2.SSS2)、[Qwen3-Coder 官方文章](https://qwenlm.github.io/blog/qwen3-coder/)：分别支持规则正确性奖励与测试扩展的公开方法，不用于替代本仓库的实际执行证据。

# 最终：Code RLVR 端到端闭环

```text
整理 Task（题意、接口、来源、划分）
    ↓
准备 Tests（合法输入、配对答案、逐条核验）
    ↓
接受测试 → GT 封装、metadata 与 Task Pool 读回
    ↓
联合验收（学习复现与可信模型实验分别记录）
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

**数据主线：** `整理 Task → 配对并逐条核验 Tests → 封装 GT 与入池`。

**训练主线：** `Actor → Candidate Code → Sandbox → Verifier → Reward → Policy Update → Evaluation`。

第一条为 Reward 提供可审计的正确性依据，第二条利用通过验收的 Reward 优化策略。Bad Case Attribution 把两条主线连接起来；测试覆盖、执行可信性和最终独立评测共同限定结论能推广多远。
