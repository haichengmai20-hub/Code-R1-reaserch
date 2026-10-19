# 阶段二：Verifier 怎样给分，这个分能不能信

**资料导航：** [资料首页](README.md) · [工业主框架](RLVR工业流程.md) · [00 全程走读](00_一条记录从原始Task到训练.md) · [01 数据与 GT](01_数据理解与GT构建.md) · **02 Verifier 与任务池（当前）** · [03 Rollout 与 GRPO](03_Rollout与GRPO机制.md) · [04 训练与评测](04_真实训练与独立评测.md)

**先记机制：** Tests 说明要验什么，执行后端负责把程序跑起来，判分器把看到的结果换成 reward。三件事各要各的证据：答案对不对、程序是不是真的跑完了测试、比较规则合不合理。退出码 0 和 reward 1.1 都不能单独证明「通过」。

<a id="overview-2"></a>

## 总览：02 在流程里做什么、长什么样

02 对应 [00 全程走读](00_一条记录从原始Task到训练.md#overview-00)七站里的第 ④ 站。它完全离线，只用 CPU，判的是课程固定的 6 个候选程序，不跑模型。它要回答的问题是：一个回答怎样变成 reward，这个 reward 能不能信。答案落在三个角色上：

- 判分器：检查格式、抽出代码、拼上测试、看执行结果，给出 -1.1、0.1 或 1.1。
- 执行后端：把拼好的程序跑起来，只回报成功与否和输出。
- 奖励管理器：一批回答并发判分，超时兜底，把分数写到回答的最后一个 token 上。

全篇分三段：

| 段 | 包含的小节 | 学完能回答什么 |
|---|---|---|
| 概念 | [三个角色](#prerequisites-2)；[判分七步](#pipeline-2)；[reward 三档](#reward-levels-2)；[执行边界](#execution-gaps-2)；[任务池](#task-pool-2) | 谁输入什么、输出什么；一个回答经过哪几步变成分数；每个分数证明了什么；原版有哪些缺口、补强版怎么堵；数据行怎样读回、出错怎样归层 |
| 动手 | [动手前的工具说明](#tools-2)；作业 [2.1](#assignment-2-1)、[2.2](#assignment-2-2)、[2.3](#assignment-2-3) | 6 个固定候选怎样拿到各自的分；改一个条件后分数怎么变；不看答案能不能把故障定位到层 |
| 自测 | [60 秒口述](#oral-2)；[面试深挖](#interview-2)；[按需读源码](#source-reading-2)；[完成记录](#record-2) | 能不能脱稿讲清楚，能不能扛住追问 |

概念五节末尾各有一句「口述」，读完要能脱稿说出来。机制细节查主框架的 [Verifier 验收](RLVR工业流程.md#verification)、[执行边界](RLVR工业流程.md#execution-boundary)、[实际奖励](RLVR工业流程.md#reward-mapping)和[故障归因](RLVR工业流程.md#layer-attribution)。

概念和作业的对应：

- 2.1 覆盖判分七步和 reward 三档：用固定示范看 6 个候选怎样拿到各自的分。
- 2.2 覆盖执行边界：在自己的 data-01 上改一个条件，看分数怎么变，再看提前退出的漏洞和补强版。
- 2.3 覆盖任务池：不看答案给 8 条记录定位故障层，再读回自己的小池。

02 的「原版 vs 补强版」用补强项 4 到 6：受保护的判题协议、结构化的执行层、判题故障整组作废。原版的判分文件不改，补强版是并列的新模块，用 `data_source=code_plus` 切换，细节见[反向补充源码](examples/source-supplements/README.md)。

01 交来的是同一道题、同一组测试和同一份 GT；你自己的练习以 data-01 最新一次 attempt 接受的测试为准，不退回示范的 8 条。所以做 2.2 之前，先把 01 的[作业 1.2](01_数据理解与GT构建.md#assignment-1-2)做完。02 交给 03 的，是验收过的判分器和每个候选的 reward，03 拿它们算组内 advantage。

<a id="guide-2"></a>

## 文字版导读：先把全篇顺一遍

第一次读，先把这一节读完，再按链接进细节；复习时只读这一节。

**第一节：先分清三个角色。** 判分这一站有三个角色：

- 判分器：输入是回答文本和 GT，输出是 reward 和一段日志。
- 执行后端：输入是一段拼好的程序，输出是成功与否和 stdout、stderr。
- 奖励管理器：输入是一批 256 个回答，输出是每个回答一个分数，写在最后一个有效 token 上。

执行后端像考场，限制权限、时间和资源；判分器像判卷机。考场守规矩，不证明卷子答完了；判卷机比对正确，也不证明考场安全。两件事要分别验收。详见[三个角色](#prerequisites-2)。

**第二节：判分七步。** 一个回答变成 reward，要走七步：

1. 格式门：去掉首尾空白后，必须以 `<think>` 开头、以 `</answer>` 结尾，否则 -1.1。
2. 抽代码：取最后一个 `<answer>` 里的所有代码块，拼成一段；没有代码也是 -1.1。
3. 拼程序：函数题把候选代码和 GT 里的断言拼成一个程序。
4. 执行：交给执行后端，默认是 firejail，限时 30 秒。
5. 判定：函数题只看退出码是不是 0；标准输入输出题比较去掉首尾空白的 stdout。
6. 映射 reward：失败 0.1，成功 1.1。
7. 管理器收尾：单个回答最多等 600 秒，超时或出错记 0.0，分数写到最后一个有效 token。

要记住的判断：函数题里没有「比较输出」这一步，断言和候选代码在同一个程序里，断言失败只通过退出码非 0 传回来。详见[判分七步](#pipeline-2)。

**第三节：reward 三档，每个数字证明了什么。**

- -1.1：没按协议交卷。它不说明算法对错。
- 0.1：格式合格，但执行失败。它不区分算法错、漏写导入、超时和后端故障。
- 1.1：格式合格，程序以退出码 0 结束。它不证明断言跑完了，也不证明测试够。
- 0.0：不是判分器给的，是奖励管理器超时或出错时的兜底值，却会进入组内统计。

课程的 6 个固定候选正好覆盖这几种情况：两个正确解 1.1，两个错解 0.1，格式错 -1.1，提前退出的 `early_exit` 也是 1.1。详见 [reward 三档](#reward-levels-2)。

**第四节：执行边界，原版的四个缺口。**

1. 提前退出拿满分：候选代码在断言之前 `sys.exit(0)`，退出码是 0，判 1.1。四个后端都按退出码判成功，换后端堵不住。
2. 超时不等于清理：管理器只是不再等，后台线程还在跑；源码里杀掉所有进程的那段代码被注释掉了。
3. 判题故障记 0.0：0.0 混进组内均值和标准差。同组三条错误回答各得 0.1、一条故障记 0.0 时，三条错误回答的 advantage 都是 +0.5。
4. 期望答案和候选代码在同一个进程里：候选代码理论上能读到断言。

补强版的做法：期望答案不进候选进程，子进程逐条报告输出，由判分方比较，缺结果就算失败；执行层区分超时、报错、判题故障，超时后杀掉整组进程；判题故障先重试，仍失败就把这一组作废，advantage 全为 0。详见[执行边界](#execution-gaps-2)。

**第五节：任务池，同一行数据怎样读回、怎样查错。** 判分器只认数据行里的 `data_source` 和 `ground_truth`，所以入池前要读回核对：GT 里的断言和接受的记录一一对应，隔离的记录没混进去，测试版本跟着 GT 变。训练读的是 parquet，本课的 JSONL 只演示形状。出了错按五层归因：Task、Tests、GT、Verifier、Evaluation，一条记录可能没错，也可能同时错在几层。详见[任务池](#task-pool-2)。

**第六节：三项作业和自测。** 先分清三套工具：`run_demo.py score` 是固定示范，只看不改；`practice.py score` 在你的 data-01 上判分；`practice.py blind-init` 和 `blind-grade` 是不标答案的故障定位练习。

- **2.1 看原 scorer 怎样给分。** 跑示范，挑两个结果不同的候选，从格式、代码、GT、第一个失败的断言一直追到 reward。
- **2.2 改一个条件。** 在 data-01 上先预测再判分：补一条全负输入，或者去掉回答的格式标签，看哪个候选的分数变了、为什么。
- **2.3 不看答案定位故障。** 给 8 条记录各自判断错在哪一层，再读回自己的小池。

学完脱稿讲一遍 [60 秒口述](#oral-2)，再从[面试深挖](#interview-2)里随机挑一个追问，按四句来回答：机制是什么、这个版本怎么实现、一个反例、证据能证明到哪一步。

读完导读，按顺序读下面的概念五节，再从[作业 2.1](#assignment-2-1)开始动手。

<a id="prerequisites-2"></a>

## 先分清三个角色

| 谁 | 输入 | 输出 | 源码 |
|---|---|---|---|
| 判分器 | 回答文本，加 GT 字符串 | reward（-1.1、0.1 或 1.1）和一段日志 | [coder1 的 _compute_score](../verl/utils/reward_score/coder1/__init__.py) |
| 执行后端 | 一段拼好的 Python 程序；标准输入输出题另加一条 stdin | 成功与否，加 stdout、stderr | [firejail_exec.py](../verl/utils/reward_score/coder1/firejail_exec.py)，由环境变量 `CODER1_EXEC` 选择 |
| 奖励管理器 | 一批回答，Code-R1 每步 16 题 × 16 个 = 256 个 | 每个回答一个分数，写在最后一个有效 response token 上 | [PrimeRewardManager](../verl/workers/reward_manager/prime.py) |

数据行里的 `data_source="code"` 决定交给哪个判分器：[路由](../verl/utils/reward_score/__init__.py)把它交给 `coder1.compute_score`，补强版的数据写 `code_plus`，交给 `coder1_plus`。

执行后端像考场，限制权限、时间和资源；判分器像判卷机。考场守规矩，不证明学生答完了卷子；判卷机比对正确，也不证明考场安全。所以「沙箱安全」和「判分可信」要分别验收。

**口述：** 判分是三个角色合作：执行后端负责跑，判分器负责判，奖励管理器负责调度和兜底。哪一层出了问题，分数都会变，但修法完全不同。

<a id="pipeline-2"></a>

## 全景：一个回答怎样变成 reward

```text
1 格式门       去掉首尾空白后以 <think> 开头、以 </answer> 结尾；否则 -1.1
2 抽代码       最后一个 <answer> 里的所有代码块拼成一段；没有代码也是 -1.1
3 拼程序       函数题：候选代码 + "\n" + GT 的 functional，也就是 check 断言加 check(入口)
4 执行         交给执行后端；firejail 限时 30 秒、内存 4 GB、禁止创建 socket
5 判定         函数题：退出码 0 算成功；标准输入输出题：stdout 去掉首尾空白后与期望相等
6 映射 reward  失败 0.1，成功 1.1
7 管理器收尾   单个回答最多等 600 秒，超时或出错记 0.0；分数写到最后一个有效 token
```

| 步骤 | 工业上要做到 | Code-R1 原版实际 | 补强版 `coder1_plus` |
|---|---|---|---|
| 1 格式门 | 提交协议写进 prompt，格式失败单独计数 | 正则从开头匹配，开头多一句寒暄就是 -1.1 | 同原版 |
| 2 抽代码 | 规则固定，并与评测时一致 | 最后一个 `<answer>` 里所有代码块拼接；LeetCodeDataset 原生评测器取最长的 python 块，两者不同 | 同原版 |
| 3 拼程序 | 期望答案不进候选进程 | 候选代码和断言拼成一个程序 | 判分方把断言拆成「调用」和「期望答案」，子进程只拿到调用 |
| 4 执行 | 隔离、限额、超时后清理整组进程 | firejail：进程数 32、打开文件数 32、单文件 2 MB、内存 4 GB、30 秒、禁止 socket | 状态分成正常、编译错、运行错、内存超限、超时、被杀、判题故障，记录耗时，超时杀掉整个进程组；macOS 用 sandbox-exec |
| 5 判定 | 有测试完成的证据，逐条给结果 | 函数题退出码 0 就算过；标准输入输出题比较去掉首尾空白的 stdout | 子进程逐条报告输出，判分方比较；缺结果、缺完成事件都算失败 |
| 6 映射 reward | 区分候选失败和判题故障 | 三档分数 | 三档分数，判题故障单独标记 |
| 7 管理器 | 判题故障有限重试，仍失败就隔离 | 超时或出错记 0.0，照常进入组内统计 | 先重试；仍失败就把这一组的分数设成同一个值，advantage 全为 0 |

三个容易错的细节：

- **函数题没有「比较输出」这一步。** 断言和候选代码在同一个程序里，断言一失败程序就带着 `AssertionError` 退出，后面的断言不再执行。判分器只看到退出码非 0。
- **标准输入输出题不是按顺序判的。** 原版把多条用例并发跑，按完成先后检查，遇到第一个失败就返回 0.1。比较只去掉首尾空白，`"1 2"` 和 `"1  2"` 仍算不同。
- **两套代码提取规则不同。** 回答里先放一段会抛异常的草稿代码、后放正确代码时，原版把两段拼在一起跑，得 0.1；原生评测器只取最长的那段，就能通过。2026-10-15 用原版判分器实跑确认。

**口述：** 判分七步：格式门、抽代码、拼程序、执行、判定、映射 reward、管理器收尾。函数题把断言拼在代码后面一起跑，只看退出码；高分还要问断言真的跑完了吗。

<a id="reward-levels-2"></a>

## reward 三档：每个数字证明了什么

| 分数 | 什么时候出现 | 证明了什么 | 没有证明什么 |
|---:|---|---|---|
| -1.1 | 格式不合格，或者抽不到代码 | 没按提交协议交卷 | 不说明算法对错 |
| 0.1 | 格式合格，执行失败 | 格式分保留；有断言失败、报错或超时 | 不区分算法错、漏写导入、超时和后端故障 |
| 1.1 | 格式合格，程序以退出码 0 结束 | 格式分 0.1 加正确分 1.0 | 不证明断言真的跑完了，也不证明测试够 |
| 0.0 | 奖励管理器等待超时或出错 | 这次判分没有完成 | 不是正确性标签，却照样进入组内均值和标准差 |

课程的 6 个固定候选，在示范的 8 条接受测试上，2026-10-15 实测结果：

| 候选 | 做了什么 | reward | 逐条审计完成几条 | 说明 |
|---|---|---:|---:|---|
| `reference` | 正确的 Kadane 写法，自带 `from typing import List` | 1.1 | 8 | 全部通过 |
| `correct_prefix` | 正确的前缀和写法 | 1.1 | 8 | 全部通过 |
| `wrong_empty` | 允许空子数组，全负数组返回 0 | 0.1 | 8 | 第一个失败的断言是 `diag-all-negative`，输入 `[-3,-2,-5]` |
| `wrong_sum` | 直接返回整个数组的和 | 0.1 | 8 | 第一个失败的是 `seed-1`：应得 6，返回 1 |
| `bad_format` | 只给代码块，没有 think 和 answer 标签 | -1.1 | 0 | 日志是 `Bad format detected!` |
| `early_exit` | 代码第二行就 `sys.exit(0)` | **1.1** | 0 | 一条断言都没执行，退出码却是 0 |

「逐条审计」是演示脚本另跑的一次检查：把 8 个输入逐条喂给候选、记下返回值，和原 scorer 的那次执行无关。它告诉你 `early_exit` 一条都没完成，原 scorer 自己看不到这一点。

两个要分开的「骗过判分器」：

- `wrong_empty` 是**测试不够**。原始数据只有 3 条 seed，答案都是正数，它在 seed 上全对，会拿 1.1；补一条全负输入才能抓住。这归 Tests 层，修法是补测试，见[主框架的错解表](RLVR工业流程.md#test-strength)。
- `early_exit` 是**测试根本没跑**。测试再多也没用，修法是让判分方拿到「每条测试都执行了」的证据。这归 Verifier 层。

还有一种 0.1 不是算法错：漏写 `from typing import List` 时，签名里的 `List[int]` 在 Python 3.12 上报 `NameError`，只拿 0.1；Python 3.14 改成延迟求值注解，同一份代码就能通过。这是提交契约和运行环境的问题，见[主框架的执行边界](RLVR工业流程.md#execution-boundary)。

**口述：** -1.1 是没按协议交卷，0.1 是交了但没过，1.1 是程序以 0 退出，0.0 是管理器兜底。0.1 不是通过，1.1 也不一定是通过：可能测试不够，也可能测试根本没跑。

<a id="execution-gaps-2"></a>

## 执行边界：原版的四个缺口与补强版

**缺口 1：提前退出拿满分。** 候选代码在断言之前 `sys.exit(0)` 或 `os._exit(0)`，程序以 0 结束，判 1.1。换后端堵不住：firejail、docker、CES 都按退出码判成功，kira 只看 stderr 是否为空。普通的「完成」字符串也能被候选代码自己打印出来。

**缺口 2：超时不等于清理。** 奖励管理器用 `asyncio.wait_for` 等每个回答，600 秒没结果就不再等，记 0.0。可是后台线程还在跑；`with ThreadPoolExecutor` 退出时还会等这些线程结束。源码里「杀掉所有进程」的那段写在注释里，没有生效。真正兜底的是后端自己的限时：firejail 的 `--timeout` 30 秒，docker 在 `finally` 里强制删除容器。

**缺口 3：判题故障记 0.0。** 0.0 和正常分数一起算组内均值和标准差。同组四条回答：三条错误回答各 0.1，一条判题故障记 0.0，算出的 advantage 是 +0.5、+0.5、+0.5、-1.5，三条错误回答反而被奖励。原版只有 CES 后端会对疑似服务器超时重试。

**缺口 4：期望答案和候选代码在同一个进程里。** 断言就写在候选代码后面，候选代码理论上能读到期望答案。比较规则也是写死的：题面说「任意顺序」，测试仍按列表顺序比较。

**补强版怎么做。** 以 LeetCode 53 为例：

1. 判分方把 3 条断言拆成「调用」和「期望答案」，例如调用 `candidate(nums=[-2,1,-3,4,-1,2,1,-5,4])`，期望 6。
2. 子进程只拿到候选代码和 3 条调用，逐条执行，把每条输出带着一个随机标记打印出来，最后打印「完成」。
3. 判分方逐条比较输出和期望答案。缺任何一条结果、缺完成事件、某条报错或答案不同，都判失败。

所以 `sys.exit(0)` 表现为 3 条结果缺失，判 0.1；候选代码即使伪造出「通过」和「完成」，也不知道正确答案是什么。

| 候选 | 原版 | 补强版 | 补强版看到的现象 |
|---|---:|---:|---|
| 数据集参考解，不带 `typing` 导入 | 0.1 | 1.1 | 补强版先拼原始前缀，3 条都通过 |
| 返回整个数组的和 | 0.1 | 0.1 | 第一条答案错 |
| `sys.exit(0)` 提前退出 | **1.1** | 0.1 | 候选代码加载失败，3 条结果缺失 |
| `os._exit(0)` 提前退出 | **1.1** | 0.1 | 没有完成事件，3 条结果缺失 |
| 伪造通过帧和完成事件 | 0.1 | 0.1 | 伪造的答案全不对 |
| 读取仓库里的数据文件 | 0.1 | 0.1 | 沙箱拒绝读取 |

这张表里的「参考解」是数据集原样的代码，不带 `typing` 导入，所以原版只给 0.1；上一节演示里的 `reference` 自带导入，原版给 1.1。两者不是同一份程序。

执行层和故障处理也补了：

- 执行结果分成正常、编译错、运行错、内存超限、超时、被杀、判题故障几种状态，记录耗时，超时后杀掉整个进程组。macOS 用系统自带的 sandbox-exec，禁止联网、只允许写临时目录、禁止读本仓库；后端不可用时报判题故障，不会退回到不隔离的执行。
- 判题故障先有限次重试；仍失败的，把同一 uid 的整组分数设成同一个值，这一组的 advantage 全为 0，另一组不受影响，仍是 ±0.866。

以上结果在 2026-10-14 实测，2026-10-15 重跑一致。sandbox-exec 只适合本机开发；firejail 分支和补强版奖励管理器的整体运行要到 Linux 训练环境验收。

**执行量估算。** 每批 B 道题、每题 G 个回答、每个回答 M 条测试，计划执行的测试数是 B×G×M，公式见[主框架](RLVR工业流程.md#execution-boundary)。Code-R1 的函数题把所有断言放进一个程序，所以每步是 16 × 16 = 256 次程序执行，不随测试条数增加；标准输入输出题每个回答最多跑 8 次。

**口述：** 原版有四个缺口：提前退出拿满分、超时不清理、故障记 0 混进组内统计、期望答案和候选同进程。补强版把判分方和执行方拆开，期望答案不进候选进程，缺结果就算失败；故障先重试，仍失败整组作废。

<a id="task-pool-2"></a>

## 任务池：同一行数据怎样读回、怎样查错

判分器只认数据行里的两样东西：`data_source` 决定用哪个判分器，`reward_model.ground_truth` 决定怎么判。一行的完整字段见 [01 的字段字典](01_数据理解与GT构建.md#metadata-fields)。

入池前读回核对三件事：

- GT 解码后，每条断言都对应一条接受的记录，`case_id` 和 `extra_info.case_index` 一致。
- 隔离和待审核的记录没有混进 GT。
- 测试版本跟着 GT 变：补一条测试，`test_version` 就换一个。

导出成功只证明格式能读回，不证明答案对。

训练入口还有三处要核对：

- **格式。** 训练加载器用 `pd.read_parquet` 读 parquet，见 [rl_dataset.py](../verl/utils/dataset/rl_dataset.py)。本课的 JSONL 只演示行的形状，不能直接交给训练入口。
- **路径。** 预处理写到 `data/code-r1-{N}k-leetcode2k-taco/`，[main_grpo.sh](../main_grpo.sh) 读的是 `data/code-r1-12k/`，两者不会自动对上。
- **TACO 的划分和测试裁剪。** [coder1.py](../examples/data_preprocess/coder1.py) 的 `taco()` 先把所有行按 train 写好再切分，之后那个循环改的是临时字典，所以 test 行的 `extra_info.split` 仍然写着 train。标准输入输出题还经过 `minimize_stdio`：按输入大小排序只留最小的 8 条，跳过超过 4 KB 的输入，输入输出条数不等时 `zip` 会悄悄截断。大输入全被裁掉，复杂度问题就测不出来。

出了错，按五层归因。下表是作业 2.3 题包里的字段：

| 层 | 题包里看哪些字段 | 怎么查 |
|---|---|---|
| Task | `entry_point` | 是不是 `Solution().maxSubArray` |
| Tests | `input`、`expected` | 输入是否合法；expected 是否等于完整枚举的结果 |
| GT | `gt_expected` | 是否和 expected 一致 |
| Verifier | `scorer_reward`、`observed_cases`、`required_cases`、`claims_correctness`、`actual_values` | 声称正确时，结果是否齐全、reward 是否 1.1、实际输出是否等于 expected |
| Evaluation | `n`、`c`、`k`、`reported_pass_at_k` | 是否满足 1 ≤ k ≤ n；报告的 pass@k 是否等于公式结果，公式见[主框架的评测](RLVR工业流程.md#evaluation) |

一条记录可能没错，记 `none`；也可能同时错在几层。expected 本身错了，先算 Tests 的错，不能拿它指控候选。

**口述：** 入池前读回三查：断言和接受记录一一对应，坏记录没混进来，测试版本跟着变。训练读 parquet，路径要对上。出了错按 Task、Tests、GT、Verifier、Evaluation 五层归因，一条记录可以有多层错。

<a id="tools-2"></a>

## 动手前：三套工具

| 工具 | 用途 | 输出在哪 | 能不能改 |
|---|---|---|---|
| `run_demo.py score` | 固定示范：原 scorer 给 6 个候选判分，加一个同题的标准输入输出对照 | `examples/maximum-subarray/outputs/report.json`，不入库，随时可重新生成 | 只看不改 |
| `practice.py score` | 在你的 data-01 上判分；加 `--bad-format` 就去掉回答的格式标签 | `practice_runs/data-01/attempt-NNN/`，每次一个新目录 | 只改你自己的 `cases.json` 和 `prediction.json` |
| `practice.py blind-init`、`blind-grade` | 不标答案的故障定位：生成 8 条记录，你判断各错在哪一层 | `practice_runs/blind-01/` | 只改 `prediction.json`，不改题包 |

- 命令都在仓库根目录执行，`$PY` 的设置见[资料首页](README.md#run-setup)。
- 演示脚本只把执行后端换成本机子进程，判分逻辑用的是原版源码；它只运行课程审过的固定程序，不接任意代码，也不提供系统隔离。
- `practice.py` 的退出码：0 表示机械检查通过、预测匹配；1 表示预测不匹配，结果已保存，要解释原因；2 表示材料或格式有问题，先修再跑。三种都不能单独证明学会。
- 文字说明写进 `practice_runs/data-01/notes.md` 和 `practice_runs/blind-01/notes.md`。操作细节见[操作说明](examples/maximum-subarray/EXERCISES.md#data-exercise)。

<a id="assignment-2-1"></a>

## 作业 2.1：完整示范——原 scorer 怎样给分

**目的：** 把格式、代码、GT、执行结果和最终分数对到具体代码上。对应[判分七步](#pipeline-2)和 [reward 三档](#reward-levels-2)。

```bash
$PY -B RLVR学习资料/examples/maximum-subarray/run_demo.py score
```

**操作：**

1. 打开[报告](examples/maximum-subarray/outputs/report.json)的 `score.candidates`，对照 [reward 三档](#reward-levels-2)一节的候选表，确认 6 个候选的分数。
2. 挑两个结果不同的候选，例如 `wrong_empty` 和 `early_exit`，逐项写出：回答的格式、抽出的代码、用的 GT、`source_first_failed_case_id_from_traceback` 里第一个失败的断言、原 reward、`audit_completed_case_count`。
3. 看 `score.stdio_adapter`：同一题改成标准输入输出形式，stdin 是一行 JSON 数组，期望输出是最大和。正确写法得 1.1，返回总和的错解得 0.1。这是教学改写，原始数据仍是函数题。
4. 读 [PrimeRewardManager](../verl/workers/reward_manager/prime.py) 的 `single_compute_score`：参数名叫 `reference`，实际传进来的是 GT。`extra_info.reference` 里的参考解不参与判分，也不做文本相似度比较。

**交付：** 在 `notes.md` 里写两个候选的完整追踪，并说明原 reward 和逐条审计是两次不同的执行。

**检查是否学会：** 先口答，再展开参考要点。

1. `wrong_empty` 在原始 3 条 seed 下和 `early_exit` 一样拿 1.1，它们骗过判分器的方式有什么不同？
2. 0.1 能说明什么，不能说明什么？
3. 为什么把 firejail 换成 docker，也堵不住 `early_exit`？

<details>
<summary>参考要点：先口答再展开</summary>

1. `wrong_empty` 的断言都执行了，只是 3 条 seed 的答案全是正数，测不出「允许空子数组」，这是 Tests 不够，补一条全负输入就能抓住。`early_exit` 是断言根本没执行，程序在断言之前以 0 退出，测试再多也没用，要靠判分方拿到「每条都执行了」的证据，这是 Verifier 的问题。
2. 0.1 只说明格式合格、程序没有以 0 退出。可能是算法错，也可能是漏写导入、超时、后端出错。要看日志才能分清，不能直接当成「模型写错了」。
3. 四个后端都按退出码判成功，kira 只看 stderr 是否为空。换后端只换了「考场」，判分规则没变；要堵住，得让期望答案离开候选进程，由判分方逐条确认结果。

</details>

<a id="assignment-2-2"></a>

## 作业 2.2：条件变化——改一个条件，reward 怎么变

**目的：** 在自己的数据上只改一个条件，先预测再运行，看哪个候选的分数变了、为什么。对应[执行边界](#execution-gaps-2)。

**操作：**

1. 继续用 01 的 data-01。在 `prediction.json` 里填 `wrong_empty_reward` 和理由，然后判分：

   ```bash
   $PY -B RLVR学习资料/examples/maximum-subarray/practice.py score --run data-01
   ```

2. 只改一个条件，二选一：
   - 在 `cases.json` 末尾追加一条自己手算的全负输入，用新的 `case_id`；同步更新 `accepted` 和 `wrong_empty_reward` 的预测，再跑上面的命令。
   - 不动 `cases.json`，更新 `wrong_empty_reward` 的预测，然后去掉回答的格式标签：

     ```bash
     $PY -B RLVR学习资料/examples/maximum-subarray/practice.py score --run data-01 --bad-format
     ```

3. 对比前后两次 attempt 的 `report.json`：每个候选的 reward、`test_version`、四个候选的教学 advantage。
4. 读[执行边界](#execution-gaps-2)里补强版的做法，解释 `early_exit` 在补强版里为什么是 0.1。

每次判分都生成新的 attempt，旧的保留。配错的 expected 在 01 的数据审核里就会被隔离，不会进 GT，所以这里改的是测试覆盖和回答格式，不是去造一条坏 GT。

**交付：** 在 `notes.md` 里写清一次单条件变化：改了什么、预测是什么、结果是什么、为什么；再写原版能证明什么、不能证明什么。

**检查是否学会：**

1. 补一条全负输入后，哪个候选的分数会变，`test_version` 为什么也变？
2. 加 `--bad-format` 后，为什么 6 个候选的分数一样？
3. 一组 4 条回答里，一条遇到判题故障记 0.0，对其余三条错误回答有什么影响？补强版怎么处理？

<details>
<summary>参考要点：提交之后再展开</summary>

1. `wrong_empty` 从 1.1 降到 0.1，其余不变；四个候选的教学 advantage 从 +0.5、+0.5、+0.5、-1.5 变成 ±0.866。GT 里多了一条断言，测试版本换成新的，以后的结果要能说明是用哪版测试判的。
2. 格式门在最前面，去掉标签后 6 个回答都过不了格式门，全是 -1.1，后面的代码和测试根本没有用到。
3. 0.0 混进组内均值和标准差。三条错误回答各 0.1 时，它们的 advantage 变成 +0.5，错误回答反而被奖励。补强版把判题故障单独标记，先重试，仍失败就把这一组的 advantage 全设为 0，这一步不让这组题产生 outcome 信号。

</details>

**完成标准：** 必做的是用课程的固定程序复现机制，并说清原版判分和后端的缺口；做到这一步就能进入 03。要在真实环境里执行未知的模型代码，还得在目标 Linux 环境里验收隔离、限额、测试完成证据、超时清理和有限重试；条件不具备就记为待完成，不影响继续学习。

<a id="assignment-2-3"></a>

## 作业 2.3：独立检查——不看答案定位故障，再读回小池

**目的：** 不靠已知答案，把一条记录的问题定位到具体层；再确认下一阶段用的仍是同一道题、同一组测试和同一份 GT。对应[任务池](#task-pool-2)。

**操作：**

1. 生成不标答案的题包：

   ```bash
   $PY -B RLVR学习资料/examples/maximum-subarray/practice.py blind-init --run blind-01 --seed 53
   ```

2. 打开 `practice_runs/blind-01/packets.json`，共 8 条记录。按[任务池](#task-pool-2)一节的五层表逐条检查，不改题包。每条记录可能没错，也可能错在几层，不知道有几处错时不要硬凑。
3. 先在 `notes.md` 里写每条的依据，以及能区分两种解释的最小检查；再在 `prediction.json` 里给每条记录填一个类别列表，干净的写 `["none"]`，并写整体理由。然后提交：

   ```bash
   $PY -B RLVR学习资料/examples/maximum-subarray/practice.py blind-grade --run blind-01
   ```

4. 打开 data-01 最新一次 attempt 的 `data_pool.jsonl`，逐项核对 `data_source`、`prompt`、`ability`、`reward_model.ground_truth`、`extra_info`；GT 解码后和同一次 attempt 的 `prepared_task.json` 一致，每条接受的记录都对应一条断言，隔离和待审核的没有混进去。

工具会先保存你的预测，核对题包和初始化时的 seed 一致，再给出诊断。自动只核对类别，理由要按[评分依据](examples/maximum-subarray/ANSWERS.md#blind-rubric)人工复核。源码和答案都能读到，它不是防作弊系统；正式自测换一个 seed，或请同伴换一份没展开过的记录，再口答依据。题包是教学构造，没有真的注入服务故障。

**交付：** 一份故障定位说明，加一次读回追踪：同一份测试和 GT 在 `prepared_task.json` 和 `data_pool.jsonl` 里的对应。

**检查是否学会：**

1. 一条记录的 `expected` 错了，候选的实际输出又和 `expected` 不同，应该怪谁？
2. reward 是 1.1、`claims_correctness` 为真，但 `observed_cases` 少于 `required_cases`，是哪一层的问题？
3. 导出的 `data_pool.jsonl` 能读回，能说明这批数据可以训练吗？

<details>
<summary>参考要点：提交之后再展开</summary>

1. 先算 Tests 的错：expected 本身不对，就不能拿它判断候选是否正确。先按完整枚举修好 expected，再看候选。
2. Verifier：声称正确却没有全部结果，说明「完成」没有证据。reward 1.1 不够，`early_exit` 就是这种情况。
3. 不能。读回只证明格式和对应关系没错；答案对不对靠数据审核，训练还要 parquet、对上训练脚本的路径，并确认这批题用于训练而不是诊断。

</details>

**以后需要真实小池时：** 诊断用的题继续只做调试；训练、开发验证、最终测试按锁定的协议从现有来源另选记录。复用已有 GT 要核对题意、入口、seed 和来源；新增或修改 expected 要重新逐条核验。多来源小池的准备见[训练准备](examples/training/README.md#pool-preparation)。按当前 policy 估难度要用可信判分和固定 checkpoint 真实采样，放在 03 及以后。

<a id="oral-2"></a>

## 60 秒口述

判分这一站有三个角色：执行后端负责把程序跑起来，判分器负责给分，奖励管理器负责并发调度和兜底。一个回答要过七步：格式门、抽代码、拼程序、执行、判定、映射 reward、管理器收尾。函数题把断言拼在候选代码后面一起跑，只看退出码；格式错 -1.1，执行失败 0.1，退出码 0 是 1.1，管理器超时或出错记 0.0。三个数字都不能直接当「通过」：`wrong_empty` 在 3 条正数答案的 seed 上拿 1.1，是测试不够；`early_exit` 在断言前退出也拿 1.1，是测试根本没跑，换后端也堵不住。原版还有两个缺口：超时只是不再等，不清理进程；判题故障记 0.0 混进组内统计，会奖励错误回答。补强版把判分方和执行方拆开，期望答案不进候选进程，缺结果就算失败；故障先重试，仍失败就整组作废。入池前读回核对断言和接受记录一一对应，出了错按 Task、Tests、GT、Verifier、Evaluation 五层归因。

<a id="interview-2"></a>

<details>
<summary>面试深挖：判分、执行边界与归因</summary>

回答按四句组织：机制是什么、本版本怎样实现、一个反例、证据能证明到哪里。

| 追问 | 能抗住的回答与反例 | 源码证据 |
|---|---|---|
| reward 是 1.1 就是通过了吗？ | 不是。1.1 只说明格式合格、程序以 0 退出。`early_exit` 在断言前 `sys.exit(0)` 也拿 1.1；`wrong_empty` 在 3 条正数答案的 seed 上也拿 1.1。前者是测试没跑，后者是测试不够。补强版的受保护协议下，提前退出表现为结果缺失，判 0.1 | [_compute_score](../verl/utils/reward_score/coder1/__init__.py)，[反向补充源码](examples/source-supplements/README.md) |
| Sandbox 和 Verifier 为什么要分开验收？ | 沙箱管「跑得安全」：权限、网络、资源、超时；判分器管「判得对」：测试是否执行完、比较规则是否合理。沙箱再严，`sys.exit(0)` 照样骗过只看退出码的判分器；判分规则再对，沙箱不禁网也不安全 | [主框架的执行边界](RLVR工业流程.md#execution-boundary)，[firejail_exec.py](../verl/utils/reward_score/coder1/firejail_exec.py) |
| 函数题怎么判对错，比较输出吗？ | 不比较。候选代码后面接着 `check` 断言和 `check(入口)`，作为一个程序执行，断言失败通过退出码非 0 传回。所以只有整体通过或失败，第一条失败后后面的断言不再执行 | [_compute_score](../verl/utils/reward_score/coder1/__init__.py)，[主框架的 Verifier 验收](RLVR工业流程.md#verification) |
| 漏写 `from typing import List` 报 NameError，算算法错吗？ | 先查提交契约。原生评测器先拼 imports，RL 的拼法不补。同一份 LeetCode 53 写法在 Python 3.12 报错、3.14 通过。算法、依赖和环境的错误要分开归因 | [原生 execution.py](../LeetCodeDataset/leetcote/execution.py)，[主框架的执行边界](RLVR工业流程.md#execution-boundary) |
| 两套代码提取规则一样吗？ | 不一样。原生评测器取最长的 python 代码块，RL 取最后一个 answer 里的所有代码块拼起来。回答里先放一段会抛异常的草稿时，原版判 0.1，原生评测器能通过 | [extract_completion](../LeetCodeDataset/leetcote/data.py)，[extract_code_from_string](../verl/utils/reward_score/coder1/__init__.py) |
| 标准输入输出题的 strip 能处理任意空白吗？ | 不能，只去首尾空白，内部一个空格和两个空格仍算不同。原版还把多条用例并发跑，遇到第一个失败就返回。多解的题不能硬按一种排列比较 | [_compute_score](../verl/utils/reward_score/coder1/__init__.py)，[两数之和练习](examples/maximum-subarray/EXERCISES.md#transfer-exercise) |
| 管理器超时，程序就被杀掉了吗？ | 不一定。`wait_for` 只是不再等，后台线程还在跑，线程池退出时还要等它；源码里杀进程那段是注释。兜底靠后端自己的限时。补强版沙箱超时后杀掉整个进程组，实测子进程被回收 | [PrimeRewardManager](../verl/workers/reward_manager/prime.py)，[docker_exec.py](../verl/utils/reward_score/coder1/docker_exec.py)，[反向补充源码](examples/source-supplements/README.md) |
| 判题故障为什么不能直接记 0？ | 0 会进入组内均值和标准差。三条错误回答各 0.1、一条故障记 0.0，三条错误回答的 advantage 都是 +0.5。应该先有限重试，仍失败就隔离；补强版把这一组的 advantage 全设为 0，并输出故障统计 | [infra_policy.py](../verl/workers/reward_manager/infra_policy.py)，[主框架的实际奖励](RLVR工业流程.md#reward-mapping) |
| 怎样设计一个挡得住提前退出和伪造的判题协议？ | 判分方和执行方分开：期望答案不进候选进程，子进程只拿到调用，逐条报告输出，判分方比较；缺结果、缺完成事件都算失败。伪造「通过」也不知道正确答案。代价是要能把每条断言拆成调用和期望答案 | [protocol.py](../verl/utils/reward_score/coder1_plus/protocol.py)，[反向补充源码](examples/source-supplements/README.md) |
| 测试越多越可靠吗？ | 要看能不能抓住确认过的错解。TACO 的 `minimize_stdio` 只留最小的 8 条、跳过大输入，条数不等时 `zip` 还会截断，复杂度问题测不出来。限时要和运行环境、题目规模一起定 | [minimize_stdio](../examples/data_preprocess/coder1.py)，[主框架的故障归因](RLVR工业流程.md#layer-attribution) |

要证明判分可信，就分别给出正确、错误、绕过、故障四类候选各自的观察结果和适用范围，不靠后端的名字盖章。课程的固定候选练习不等于认证了执行未知代码的环境。

</details>

<a id="source-reading-2"></a>

## 按需读源码

需要拿源码作证时按这个顺序读，第一遍不必通读：

| 顺序 | 直接阅读的材料 | 阅读时要回答的问题 |
|---|---|---|
| 1 | [reward 路由](../verl/utils/reward_score/__init__.py) | `data_source` 怎样决定用哪个判分器？`code` 和 `code_plus` 各去哪？ |
| 2 | [coder1 判分器](../verl/utils/reward_score/coder1/__init__.py) 的 `validate_response_structure`、`extract_code_from_string`、`_compute_score` | 格式门、抽代码、函数题和标准输入输出题两条分支各怎样给分？ |
| 3 | [firejail_exec.py](../verl/utils/reward_score/coder1/firejail_exec.py)，再按需看 [docker](../verl/utils/reward_score/coder1/docker_exec.py)、[CES](../verl/utils/reward_score/coder1/ces_exec.py)、[kira](../verl/utils/reward_score/coder1/kira_exec.py) | 各后端的限额是什么？成功的标准是什么？超时后谁来清理？ |
| 4 | [PrimeRewardManager](../verl/workers/reward_manager/prime.py) | 并发多少？等多久？超时和异常记几分？分数写到哪个 token？ |
| 5 | 原生 [data.py](../LeetCodeDataset/leetcote/data.py) 和 [execution.py](../LeetCodeDataset/leetcote/execution.py) | 原生评测器怎样抽代码、怎样拼 imports？它的限制算不算安全沙箱？ |
| 6 | 补强版 [protocol.py](../verl/utils/reward_score/coder1_plus/protocol.py)、[sandbox.py](../verl/utils/reward_score/coder1_plus/sandbox.py)、[infra_policy.py](../verl/workers/reward_manager/infra_policy.py)、[prime_plus.py](../verl/workers/reward_manager/prime_plus.py) | 期望答案在哪个进程？哪些状态算判题故障？整组作废怎么做？ |
| 7 | [coder1.py](../examples/data_preprocess/coder1.py) 的 `taco()`、`minimize_stdio`，以及 [main_grpo.sh](../main_grpo.sh) 的 `DATASET` | TACO 的测试怎样裁剪、划分怎样写？预处理的输出目录和训练脚本读的路径对得上吗？ |

<a id="record-2"></a>

## 本阶段完成记录

学完这一篇，应该能说清四件事：一个回答怎样变成 reward、每个分数证明了什么、原版判分有哪些缺口、数据行出错时怎样归层。

| 项目 | 完成证据 | 尚未完成时的处理 |
|---|---|---|
| 判分机制 | 两个候选从格式到 reward 的完整追踪；原 reward 和逐条审计的区别 | 回到[判分七步](#pipeline-2)，逐步对源码 |
| 单条件变化 | 前后两次 attempt 的预测、结果和原因 | 保留旧 attempt，写明为什么改了判断 |
| 执行边界 | 原版四个缺口各自的影响，补强版对应的做法和边界 | 未知代码的实跑条件记为待完成 |
| 故障定位与读回 | 8 条记录的分层依据；同一份 GT 在两次读回里的对应 | 无法确认的 expected 或接口先隔离，能确认的正确记录保留 |

本阶段只运行课程审过的固定程序，不要求 GPU。进入 03 时写清：用的是哪一版测试、哪些候选的分数已经核对、哪些执行条件还没在目标环境验收。

---

**继续阅读：** [返回资料首页](README.md) · [工业主框架的 Verifier 验收](RLVR工业流程.md#verification) · [上一阶段：01 数据与 GT](01_数据理解与GT构建.md) · [下一阶段：03 Rollout 与 GRPO](03_Rollout与GRPO机制.md)
