# 反向补充源码：原版与补强版对照

[资料首页](../../README.md) · [00 全程走读](../../00_一条记录从原始Task到训练.md) · [01 数据与 GT](../../01_数据理解与GT构建.md) · [主框架](../../RLVR工业流程.md)

**规则：** 学习资料讲到了、Code-R1 源码没有、而且资料的设计靠谱的地方，就写成代码补进源码。原版文件只改两处、各加 4 行：判分路由里加 `code_plus` 分支，训练入口里加 `prime_plus` 奖励管理器分支。其余都是平行的新模块，用新的 `data_source=code_plus` 切换。这样原版的缺口还能拿来教学，`run_demo.py` 对原版文件的 hash 锁也不受影响。每一项先以 CPU 能验证为完成标准，接入真实训练的验证留到 Linux GPU 环境。

## 补强项一览

| 编号 | 补什么 | 资料依据 | 源码位置 | 状态 |
|---|---|---|---|---|
| 1 | GT 执行契约：数据带运行前缀，判分时先拼前缀 | [01 的内部审查例子](../../01_数据理解与GT构建.md#internal-external-review) | `examples/data_preprocess/coder1_plus.py`、`verl/utils/reward_score/coder1_plus/`、路由新增 `code_plus` 分支 | 2026-10-14 完成，CPU 已验证 |
| 2 | 测试强度审计：对参考解做小改动造出错解，看测试能不能抓住，标出测试太弱的题 | [主框架 A5](../../RLVR工业流程.md#test-strength) | `examples/data_preprocess/coder1_plus_tests.py` | 2026-10-14 完成，CPU 已验证 |
| 3 | 测试补充与逐条审核：从只支持 LeetCode 53 推广到一般函数题，逐条接受、隔离、升版本 | [01 作业 1.2](../../01_数据理解与GT构建.md#assignment-1-2)、[主框架 A3](../../RLVR工业流程.md#tests) | `coder1_plus_tests.py`、`coder1_plus_oracles.py`、`coder1_plus.py --augment_cases` | 2026-10-14 完成，7 道示范题已验证 |
| 4 | 受保护的判题协议：逐条执行、结构化结果、提前退出判失败；允许任意顺序的题改用顺序无关的比较 | [02](../../02_Verifier与任务池验收.md)、[主框架第三章](../../RLVR工业流程.md#execution-boundary) | `verl/utils/reward_score/coder1_plus/protocol.py` | 2026-10-14 完成，CPU 已验证 |
| 5 | 沙箱：执行结果分成编译错误、运行错误、超时、基础设施错误等状态；记录耗时；超时后确保子进程被清理；本机开发另配一个用 macOS sandbox-exec 禁网、限写的隔离后端，Linux 训练仍用 firejail。范围已经用户确认 | [主框架第三章](../../RLVR工业流程.md#execution-boundary)、[第五章的异常处理](../../RLVR工业流程.md#reward-mapping) | `verl/utils/reward_score/coder1_plus/sandbox.py` | 2026-10-14 完成，macOS 已验证，firejail 分支待 Linux 实跑 |
| 6 | 基础设施异常不再记 0 分，改成结构化状态加有限次重试，必要时整组作废 | [03 作业 3.2](../../03_Rollout与GRPO机制.md#assignment-3-2) | `verl/workers/reward_manager/infra_policy.py`、`prime_plus.py`、训练入口的 `prime_plus` 分支 | 2026-10-14 完成；2026-10-15 奖励管理器在 macOS CPU 上用 torch 整体跑通；firejail 分支与接入真实训练待 Linux 实跑 |
| 7 | 和评测集去重；按当前模型的通过率过滤难度 | [01 全景](../../01_数据理解与GT构建.md#pipeline-1)、主框架 A8 | `examples/data_preprocess/coder1_plus_selection.py`、`coder1_plus.py --drop_task_ids/--keep_task_ids` | 2026-10-14 完成：去重用 v1 test 演示；难度筛选的通过率要靠 GPU 真实采样 |

<a id="supplement-1"></a>

## 补强项 1：GT 执行契约

**问题。** LeetCodeDataset 的树和链表题，测试会调用 `tree_node`、`is_same_tree`、`list_node`、`is_same_list`。这些辅助函数只定义在原始记录的 `prompt` 前缀里。原生评测器先拼前缀，所以能跑；原版 Code-R1 的 GT 只取 `test`，判分时是「候选代码加测试」，这些题无论模型写得多对都会报 `NameError`，只拿 0.1。

**改法。** 数据侧把原始前缀存进 GT 的 `prelude` 字段；判分侧按「前缀、候选代码、测试」的顺序执行，和原生评测器一致。prompt、题目筛选、去重和 `index` 的含义都与原版相同，只改执行契约这一个变量。没有 `prelude` 的 GT，例如 TACO 和 KodCode 的题，原样交给原版判分。

**代价。** 前缀里有 `from typing import *` 等导入，模型漏写 imports 不会再被判错。这与原生评测器和 LeetCode 官方判题的行为一致；想继续惩罚漏写 imports，就要另行设计。

**验证结果。** 2026-10-14 在 macOS、Python 3.12 上实跑。每道题的候选回答是「前缀里的 imports 和 TreeNode、ListNode 类，加上数据集参考解」，也就是一个写全导入、但不知道数据集私有测试辅助函数的认真回答。同一份回答分别按原版契约和补强版契约判分：

| 题集 | 题数 | 测试依赖前缀辅助函数 | 原版契约得 1.1 | 补强版契约得 1.1 |
|---|---:|---:|---:|---:|
| LeetCode 训练集 | 1890 | 121 | 1756 | 1877 |
| v2 test | 200 | 0 | 199 | 199 |

依赖前缀辅助函数的 121 题，原版全部 0.1，补强版全部 1.1。表里补强版的 1877 是补强项 4 上线后复跑的数。补强项 1 刚完成时是 1876，仍失败 15 题，原因都很明确，不是这次改动造成的；补强项 4 上线后 `word-ladder-ii` 稳定通过，只剩 14 题：

| 类别 | 题数 | 原因 | 责任层与后续 |
|---|---:|---|---|
| 运行环境依赖 | 14 | 参考解用了 `sortedcontainers` 或 `numpy`，本机没装 | 运行环境：训练镜像要不要装这些库需要验收，LeetCode 官方环境提供 `sortedcontainers` |
| 顺序相关的不稳定测试 | 1 | `word-ladder-ii` 允许任意顺序返回，测试却比较列表；输出顺序依赖 set 的迭代顺序，6 个哈希种子下过 3 次 | verifier：补强项 4 上线后改用忽略外层顺序的比较，现在稳定通过 |

另外，题面写明「in any order」的有 68 题，训练集里 65 题。参考解恰好按样例顺序输出所以能过，但模型给出另一种合法顺序时会被判错。补强项 4 给这些题的 GT 加上 `compare=unordered_outer`。

原先还没堵上的提前退出缺口，已由补强项 4 解决：单元测试里那条用例的期望值从 1.1 改成了 0.1。

<a id="supplement-2"></a>

## 补强项 2：测试强度审计

**问题。** 原版没有任何手段判断测试够不够，而 LeetCode 部分每道题只有题面样例，中位数 2 条。

**做法。** 对参考解做单点小改动，生成变异体：比较符互换，比如 `<` 与 `<=`；加减互换；`max` 与 `min` 互换；`and` 与 `or` 互换；小常数加一或减一；去掉 `not`。每题最多 10 个，种子固定，结果可以复现。现有测试能判失败的比例叫杀死率，是测试强度的一个代理指标。

**结果。** 2026-10-14 对全部 2090 题实跑，每个变异体限时 5 秒：

| 指标 | 结果 |
|---|---|
| 参与审计的题 | 2014 道；另有 14 道参考解自己没通过，都缺第三方库，62 道没有可变异的位置 |
| 变异体与被杀死的数量 | 17316 个，杀死 14049 个，合计 81% |
| 每题杀死率分布 | 100% 的 691 道，75% 到 99% 的 760 道，50% 到 74% 的 464 道，不到 50% 的 99 道 |
| 按测试条数看平均杀死率 | 1 条 80%，2 条 82%，3 条 83%，4 条 87%，5 条 87% |

两次全量实跑只差 1 个变异体：个别变异体的运行时间贴近 5 秒超时线，结果会有这一点波动。上表是补强项 4 上线后同日复跑的结果：`word-ladder-ii` 的参考解在忽略外层顺序的比较下通过了基线，进入审计，多出 10 个变异体、杀死其中 7 个，所以从最早的 2013 道、17306 个变成现在的数。

**这个数要怎么读。** 有两个边界必须一起讲：
- **存活不等于测试弱。** 存活的可能是等价改动。比如 shuffle-the-array 把 `nums[i + n]` 改成 `nums[i - n]`，数组长度正好是 2n，Python 的负下标让两者取到同一个元素。
- **杀死率高也不等于测试够。** 变异体只覆盖参考解附近的错误。LeetCode 53 的 9 个变异体全被样例杀死，但换了一种写法的错解 wrong_empty 仍然拿满分。

所以杀死率只能用来找可疑的题，要和第 3 项的补测试、独立 oracle 对拍一起用。

<a id="supplement-3"></a>

## 补强项 3：测试补充与逐条审核

**做法。** 每道题提供三样东西：合法输入生成器、约束校验器、按题意直接计算的 oracle，写法要和参考解不同。候选输入依次经过三道关：

| 结果 | 条件 | 能否进 GT |
|---|---|---|
| invalid | 不满足题目约束 | 否 |
| quarantined | 参考解报错，或参考解与 oracle 不一致 | 否，要查是参考解错还是 oracle 错 |
| pending | 没有独立 oracle，答案只来自参考解 | 否，等待可信依据 |
| accepted | 参考解与 oracle 一致 | 是 |

接受的用例追加进 GT 的断言，断言消息带 `case_id`，测试版本变成 `leetcode-examples+augN`。补完再做一次变异审计；仍然存活的变异体，用 200 个随机合法输入和 oracle 对拍。找到反例就把反例按同样三道关审核后补进去，这叫变异体引导补测试；对拍全部一致的，视为疑似等价改动。

**结果。** 7 道示范题，每题先随机补 10 条已接受的用例，再做一轮变异体引导补测试：

| 题 | 测试条数 | 杀死率：补前、随机补后、引导补后 | 最后剩下的存活变异体 |
|---|---|---|---|
| maximum-subarray | 3 变 13 | 9/9、9/9、9/9 | 无。离参考解很远的 wrong_empty 从 1.1 变成 0.1 |
| increasing-triplet-subsequence | 3 变 14 | 0/2、1/2、2/2 | 无。引导补测试找到反例 `[1, 0, -2, -2, 2, 2]` |
| valid-palindrome | 2 变 12 | 1/10、9/10、9/10 | 1 个疑似等价：`i < j` 改成 `i <= j`，只是多比较中间字符和它自己 |
| number-of-senior-citizens | 2 变 12 | 0/1、1/1、1/1 | 无。补进了恰好 60 岁的用例 |
| find-minimum-in-rotated-sorted-array | 3 变 13 | 2/10、5/10、5/10 | 5 个疑似等价 |
| third-maximum-number | 3 变 13 | 1/4、1/4、1/4 | 3 个疑似等价：重复值已被提前跳过，`>` 和 `>=` 没有区别 |
| shuffle-the-array | 3 变 13 | 0/1、0/1、0/1 | 1 个疑似等价：负下标让 `i - n` 和 `i + n` 取到同一个元素 |

「疑似等价」只是 200 次对拍全部一致，不是证明。maximum-subarray 还故意加了两条非法输入，空数组和含 10001 的数组，都被校验器拦下。

**接进训练数据。** 验证脚本把接受的用例写到 `outputs/accepted_cases.json`；`coder1_plus.py --augment_cases` 读这个文件，只追加状态为 accepted 的用例。

**还缺什么。** oracle 要逐题手写，目前只收录 7 题。工业界的做法是让 LLM 写输入生成器和校验器，再用多份可信解或人写的 oracle 核对，代表工作有 CodeContests+ 和 HardTests。

<a id="supplement-4"></a>

## 补强项 4：受保护的判题协议

**问题。** 原版把候选代码和测试断言拼成一个程序，按退出码判通过。于是提前退出能拿 1.1；期望答案写在同一个进程里，候选代码理论上能读到；只有整体通过或失败，不知道哪一条错；允许任意顺序的题仍按列表顺序比较。

**做法。** 判分方和执行方分开。判分方先把每条断言拆成「调用表达式」和「期望答案」；LeetCodeDataset 的断言只有三种写法：直接比值、`is_same_tree`、`is_same_list`，2090 题全部覆盖。子进程只拿到调用表达式，执行候选代码后逐条调用，把输出的 repr 带着一个随机标记打印出来，最后报告完成。判分方自己比较：缺少完成事件、缺少某一条结果、某一条报错或答案不同，都判失败。树和链表先转成列表再比较。GT 带 `compare=unordered_outer` 时，列表答案忽略外层顺序。

**为什么挡得住。** 期望答案根本不在候选进程里。候选代码即使从全局变量里找到上报函数，伪造了全部「通过」帧和完成事件，也不知道正确答案是什么；提前退出则表现为结果缺失。

**结果。** 2026-10-14 在 macOS、Python 3.12、sandbox-exec 后端实测，题目是 LeetCode 53：

| 候选 | 原版得分 | 补强版得分 | 补强版看到的现象 |
|---|---:|---:|---|
| 参考解 | 0.1 | 1.1 | 三条都通过。原版低分是因为参考解没写 `typing` 导入、原版不拼前缀 |
| 返回总和 | 0.1 | 0.1 | 第一条答案错，后两条通过 |
| `sys.exit(0)` 提前退出 | **1.1** | 0.1 | 候选代码抛出 SystemExit，被记为加载失败，三条结果缺失 |
| `os._exit(0)` 提前退出 | **1.1** | 0.1 | 没有完成事件，三条结果缺失 |
| 伪造通过帧与完成事件 | 0.1 | 0.1 | 伪造的答案全部不对 |
| 读取仓库里的数据文件 | 0.1 | 0.1 | 沙箱拒绝读取，候选代码加载失败 |

全量对照里，补强版的 LeetCode 训练题从 1876 道通过变成 1877 道，多出的一道是 `word-ladder-ii`：8 次重复判分全部通过。剩下 14 道失败全部是缺第三方库。

**边界。** 期望答案的比较沿用原断言的 `==` 语义；只忽略外层顺序，内层顺序仍要一致。分组、内层也可任意排列的题，需要逐题写专门的判定器，类似 04 两数之和的迁移练习。`word-ladder-ii` 是按哈希种子实验发现、手工登记的，题面没写「in any order」的同类题还可能漏掉。

<a id="supplement-5"></a>

## 补强项 5：结构化执行层与隔离后端

**问题。** 原版 firejail 后端只返回退出码是否为 0 和一段混在一起的输出；奖励管理器那一层的超时只是不再等待，不保证子进程被清理；也区分不了候选的错误和判题环境的故障。

**做法。** 执行结果分成 ok、compile_error、runtime_error、memory_error、timeout、killed、infra_error 几种状态，并记录耗时；超时后杀掉整个进程组。后端由环境变量 `CODER1_PLUS_SANDBOX` 选择：Linux 用 firejail，参数与原版相同；macOS 用系统自带的 sandbox-exec，禁网、只允许写临时工作目录、禁止读取本仓库；都不可用时返回 infra_error，不会退回到不隔离的执行。文件大小上限 2 MB，打开文件数上限 64。

**结果。** 2026-10-14 在 macOS 14.2 实测：

| 探针 | 结果 |
|---|---|
| 正常程序、抛异常、语法错误 | 分别判为 ok、runtime_error、compile_error |
| 死循环，超时 2 秒 | 2.01 秒判为 timeout |
| 候选代码派生一个睡眠 30 秒的子进程 | 超时后子进程已被回收 |
| 连网、写工作目录以外的文件、读仓库里的数据文件 | 全部被拒绝 |
| 指定一个不存在的后端 | 判为 infra_error |

**一个实测的坑。** 在这台 macOS 上通过 `preexec_fn` 设置 RLIMIT_CPU 后，无论上限设多少秒，子进程都会立刻收到 SIGXCPU。所以 CPU 时间上限只在 Linux 上设置，macOS 靠墙钟超时加整组清理兜底。

**边界。** sandbox-exec 已被苹果标为不推荐使用，只适合本机开发验证。firejail 分支沿用原版参数，本机无法实跑，需要在 Linux 训练环境验收。工业界会用 gVisor、Firecracker、nsjail 这类更强的隔离，配合大规模并发池。

<a id="supplement-6"></a>

## 补强项 6：判题故障的处理策略

**问题。** 原版奖励管理器遇到评分超时或异常，把这一条直接记成 0.0，0.0 又进入同组的均值和标准差，三条错误回答会因此拿到正的 advantage。

**做法。** 每条样本的评分带状态：正常得分，或判题故障。判题故障先有限次重试；仍失败的，按 uid 把整组分数设成同一个值，组内 advantage 全为 0，这组不提供 outcome 信号，KL 与熵项照常。同时输出故障条数、作废的组数和样本数，故障率高时应停下排查。训练配置写 `reward_model.reward_manager=prime_plus` 启用。

**结果。** 同组一条判题故障的对照：

| 处理方式 | 这一组的 advantage |
|---|---|
| 原版，故障记 0.0 | +0.5、+0.5、+0.5、-1.5 |
| 补强版，整组作废 | 0、0、0、0 |

另一组不受影响，仍是 ±0.866。重试逻辑也测过：前一次出错、第二次成功的，返回第二次的分数；持续出错的，按设定次数重试后标为判题故障。

**整体运行，2026-10-15 实测。** 奖励管理器依赖 torch 和 verl 主包，所以另写了 `verify_reward_manager_cpu.py`，在装了 CPU 依赖的虚拟环境里跑：macOS、Python 3.12、torch 2.4.0 CPU 版，不要 GPU。LeetCode 53 的 4 条固定回答放进同一个 uid 组：

| 组合 | 参考解 | 返回总和 | 提前退出 | 缺格式 | 说明 |
|---|---:|---:|---:|---:|---|
| 原版管理器，原版 GT，本机没有 firejail | 0.0 | 0.0 | 0.0 | -1.1 | 3 条执行错误被吞掉，记成 0.0 |
| 补强版管理器，补强版 GT，sandbox-exec | 1.1 | 0.1 | 0.1 | -1.1 | 判题故障 0 条 |
| 补强版管理器，原版 GT，本机没有 firejail | 0.0 | 0.0 | 0.0 | 0.0 | 3 条判题故障，重试后整组作废 |

第一行就是缺口在真实张量上的样子：机器上没有 firejail，原版不报错，只是把这几条记成 0.0 送进组内统计。

**边界。** 仍待 Linux 的是 firejail 分支，以及接入真实训练循环：真实 tokenizer、Ray worker、`reward_model.reward_manager=prime_plus` 的启动路径。整组作废会让这组题这一步没有 outcome 信号，故障多时等于少训了这些题。

## 按学习进度的扫描记录

**02（2026-10-15）。** 02 讲到的判分机制，原版缺的部分都已由补强项 4 到 6 覆盖：受保护判题协议、结构化执行层、判题故障整组作废。另有两项暂不补：

- 多解题的专用判定器。补强项 4 只忽略外层顺序；分组或内层也可任意排列的题要逐题写判定器，04 的两数之和迁移练习覆盖这个思路，按题补比做通用框架合适。
- 标准输入输出题接入受保护协议。原版的标准输入输出分支本来就在判分方比较 stdout，提前退出会因为没有输出而判失败；补强版的数据只改 LeetCode 函数题，TACO 行原样交给原版判分。

<a id="supplement-7"></a>

## 补强项 7：去污染与难度筛选

**去污染的做法。** 训练题和评测题之间查两种依据：题号或 task_id 相同；或者只比较自然语言描述，训练题面的 8 元词组里至少一半出现在同一道评测题里。比较前先去掉代码块和所有题共用的固定句子，否则链表题的模板注释会造成大量误报。

**去污染的结果。** 2026-10-14 用 LeetCode v1 test 当评测集演示：

| 依据 | 找到的对数 |
|---|---:|
| 题号相同 | 61 |
| 措辞重叠，阈值 0.5 | 3 |

措辞重叠里最典型的是 check-if-strings-can-be-made-equal-with-operations 的 i 和 ii 两版：题号不同，题面几乎一样。只按题号去重，发现不了这种同一题族。阈值降到 0.3 时，误报会大量增加，主要来自子数组定义这类公共句子，所以疑似对需要人工复核。

**难度筛选。** 输入是某个固定 checkpoint 对每道题的真实采样结果，即采样数和通过数；只保留通过率严格介于 0 和 1 之间、且采样数够多的题，因为全对和全错的题在 GRPO 里没有组内 outcome 信号。通过率必须来自真实采样，换 checkpoint 就要重测；本机没有 GPU，只用构造的数据测了筛选逻辑。

**接进训练数据。** 去污染得到要删的题，交给 `coder1_plus.py --drop_task_ids`；难度筛选得到要留的题，交给 `--keep_task_ids`。两者只作用于训练集，评测集不动。

<a id="supplement-8"></a>

## 补强项 8（候选，未实现）：标准输入输出题的受保护协议

**缘起。** 05 第三环里学习者决定把 TACO 并进训练集，并「先把受保护协议扩到标准输入输出题，再开训」。原版对这类题的判法是整段 stdout 与期望比对、看退出码，没有前缀、没有逐条报告，也分不开候选错误与判题故障，所以 prime_plus 的整组作废拿不到它的故障信号。

**要补的。** 一个进程喂多条输入、逐条报告结果（通过 / 错误 / 超时 / 故障）、候选错误与判题故障分开记、接 prime_plus 的整组作废；提前退出沿用原版的「stdout 为空即失败」。CPU 可验证：固定候选（对、错、提前退出、超时）加一个人为的故障（沙箱起不来）。

**状态。** 只登记，没有代码；实现与否等学习者决定。2026-10-19 学习者决定训练方案先去掉 TACO（本机没有数据），这一项暂不需要；TACO 并回来时再做。实现时与补强项 4、6 同一目录，原版判分文件不改。

## 怎样运行

在仓库根目录的 bash 里执行，`$PY` 的设置见[资料首页](../../README.md#run-setup)：

```bash
$PY -B RLVR学习资料/examples/source-supplements/test_source_supplements.py   # 22 个单元测试，约 2 秒
$PY -B RLVR学习资料/examples/source-supplements/verify_gt_contract.py       # 补强项 1：2090 题两种契约对照，约 20 秒
$PY -B RLVR学习资料/examples/source-supplements/verify_test_quality.py      # 补强项 2、3：抽 200 题审计加 7 题闭环，约 1 分钟
$PY -B RLVR学习资料/examples/source-supplements/verify_test_quality.py --all  # 全部 2090 题审计，约 7 分钟
$PY -B RLVR学习资料/examples/source-supplements/verify_judge_and_selection.py   # 补强项 4 到 7：沙箱探针、攻击候选、故障策略、去污染，约 10 秒
$PY -B examples/data_preprocess/coder1_plus.py --root_dir ./data            # 生成补强版 JSONL；加 --parquet 写训练用 parquet
$PY -B examples/data_preprocess/coder1_plus.py --root_dir ./data --augment_cases RLVR学习资料/examples/source-supplements/outputs/accepted_cases.json
```

`--augment_cases` 读取的 `accepted_cases.json` 由 `verify_test_quality.py` 生成，先跑它。

奖励管理器的整体检查要导入 torch，需要单独的虚拟环境。uv 管理的 Python 不允许直接 pip 安装，所以先建 venv，放在仓库外面，免得被 git 当成新文件：

```bash
$PY -m venv ~/.venvs/code-r1-cpu
~/.venvs/code-r1-cpu/bin/pip install "numpy<2" pandas requests "torch==2.4.0" "tensordict==0.5.0" "ray==2.40.0"
~/.venvs/code-r1-cpu/bin/python -B RLVR学习资料/examples/source-supplements/verify_reward_manager_cpu.py   # 补强项 6 整体运行，约 10 秒
```

验证报告写到 `outputs/` 目录：`gt_contract_report.json`、`test_strength_report.json`、`augmentation_report.json`、`accepted_cases.json` 和 `judge_and_selection_report.json`。这个目录被 `.gitignore` 忽略，随时可以重新生成。

原版判分器的本地验证用普通子进程代替 firejail；补强版判分器用自己的沙箱后端，在 macOS 上是 sandbox-exec。`supplement_env.py` 只为判分子包放了最小的桩模块，不导入会拉起 torch 的 verl 主包。这里只执行数据集自带的参考解和本课固定的候选，不执行未知的模型代码。

训练时用补强版数据：先用 `--parquet` 生成 parquet，把 `main_grpo.sh` 的数据路径指过去；`data_source=code_plus` 的行会自动路由到补强版判分器。需要 TACO 时加 `--merge_taco_from` 指向原版输出目录，原样复用其中的 TACO 行。`--parquet` 需要 pandas 和 pyarrow，2026-10-15 在上面的 venv 里实跑，生成了 train、test 两个 parquet；`--merge_taco_from` 要先有原版的 TACO 输出，原版预处理 TACO 时要联网下载数据，还要用 firejail 执行参考解，所以留到 Linux 环境。
