# 00 · 直观走读：一条记录怎样从原始 Task 走到训练

**资料导航：** [资料首页](README.md) · [工业主框架](RLVR工业流程.md) · **00 全程走读（当前）** · [01 数据与 GT](01_数据理解与GT构建.md) · [02 Verifier 与任务池](02_Verifier与任务池验收.md) · [03 Rollout 与 GRPO](03_Rollout与GRPO机制.md) · [04 训练与评测](04_真实训练与独立评测.md)

**用途：** 进入 01–04 之前先看全景。这里只跟一条真实记录走完全程：LeetCode 53「最大子数组和」，本地 `LeetCodeDataset/data/LeetCodeDataset-v2-rl-problems.jsonl` 第 143 行。每一站只回答三件事：进来的是什么、出去的是什么、口述时怎么讲。源码位置放在每站末尾，只作查阅。

<a id="overview-00"></a>

## 纵览：一步训练做什么，七站分在哪几篇

Code-R1 每一步训练做这些事：

- 取 16 道题，每题让模型写 16 个回答，逐个放进沙箱判分。
- 同一道题的 16 个分数互相比较，得到每个回答的相对分，也就是 advantage。
- 用相对分加权模型生成每个 token 的概率，反传并更新参数。
- 把新权重装回采样引擎，开始下一步。

本文把这一步拆成七站，用 LeetCode 53 这一条记录走完。每站进来什么、出去什么、在哪篇细学：

| 站 | 进来 | 出去 | 在哪篇细学 |
|---|---|---|---|
| ① 原始记录 | LeetCodeDataset v2 RL 第 143 行 | 14 个字段，真正进训练的只有题面、测试、入口、题号 | 01 |
| ② 训练样本行 | 原始记录 | parquet 一行：prompt、GT 字符串、extra_info | 01 |
| ③ 采样 | 16 道题的 prompt | 每题 16 个回答，同一道题共享一个 uid | 03 |
| ④ 判分 | 一个回答加 GT | reward：-1.1、0.1 或 1.1 | 02 |
| ⑤ advantage | 同 uid 的 16 个 reward | 组内标准化的相对分，铺到回答的每个 token | 03 |
| ⑥ loss 与更新 | advantage，加 old、ref、new 三份 logprob | PPO-clip loss，反传后 AdamW 更新参数 | 03 学机制，04 实跑 |
| ⑦ 同步、验证、保存 | 新权重 | 装回 vLLM；每 16 步贪心验证一次，每 64 步存一次档 | 04 |

四篇各回答一个面试大问题，前一篇的产出就是后一篇的输入：

| 篇 | 回答的问题 | 产出 |
|---|---|---|
| [01 数据与 GT](01_数据理解与GT构建.md#overview-1) | reward 的证据从哪来 | 同一道题、同一组测试、同一份 GT |
| [02 Verifier 与任务池](02_Verifier与任务池验收.md#overview-2) | reward 可不可信 | 验收过的判分器，以及每个候选的 reward |
| [03 Rollout 与 GRPO](03_Rollout与GRPO机制.md#overview-3) | reward 怎样变成梯度 | reward、advantage、logprob 三者的对应关系 |
| [04 训练与评测](04_真实训练与独立评测.md#overview-4) | 怎样证明训练真的有效 | baseline、真实更新与恢复、固定协议评测、故障归因 |

另外三份资料：

- 本文是主线：先看全景，学习进度和资料修改记录在[文末](#progress-00)。
- [主框架](RLVR工业流程.md)按机制查，不用通读。
- [反向补充源码](examples/source-supplements/README.md)登记原版缺、补强版补上的源码改动，从 01 开始陆续引用。

本机能做到哪一步：①②④⑤ 用 `run_demo.py` 在 CPU 上能跑通，⑥ 有一个两标量的梯度 toy，命令见[本机复现](#reproduce-00)；③⑦ 和 ⑥ 的真实更新要 Linux GPU，本机只能读源码。

## 一张图记住全程

```text
① 原始记录（JSONL 一行）
        │  leetcode2k()：题面 + 系统提示 → prompt；test + check(入口) → GT（JSON 字符串）
        ▼
② 训练样本行（parquet：prompt / reward_model.ground_truth / extra_info）
        │  RLHFDataset：套聊天模板 → token id，左填充到 2048
        ▼
③ 一步训练取 16 题，vLLM 给每题采 16 个回答（温度 1.0，最长 4096 token）
        │  这条记录的 16 个回答共享一个 uid
        ▼
④ 每个回答：格式检查 → 抽代码 → 拼上 GT → 沙箱执行 → reward ∈ {-1.1, 0.1, 1.1}（管理器异常回填 0.0）
        │
        ▼
⑤ reward 放在回答最后一个有效 token；同 uid 的 16 个回答做 (r - mean) / (std + 1e-6) → advantage，广播到该回答所有 token
        │
        ▼
⑥ 重算 old logprob、ref logprob（无梯度）；actor forward 得 new logprob（有梯度）
   ratio = exp(new - old)；PPO clip 0.2 + 0.001·low_var_KL − 0.001·entropy → loss → backward → AdamW（lr 5e-7）
        │
        ▼
⑦ 新权重同步进 vLLM → 下一步采样；每 16 步贪心验证；每 64 步保存 checkpoint
```

一句话版本：**题目和断言是数据，模型写代码是行为，沙箱跑断言是判卷，同题 16 份卷子互相比出相对分，相对分去加权模型自己生成每个 token 的概率，反传更新，再把新模型装回采样引擎。**

<a id="station-1"></a>

## 第 ① 站 原始记录：一行 JSONL 里有什么

这一行有 8 个顶层键，`meta` 里再有 7 个子字段。口述时先记 5 个：

| 字段 | 这条记录的真实内容（已截短） | 它是什么 |
|---|---|---|
| `task_id` | `maximum-subarray` | 题目身份 |
| `meta.query` | `Given an integer array nums, find the subarray with the largest sum, and return its sum.` + `Please complete the following python code precisely:` + 签名 `class Solution: def maxSubArray(self, nums: List[int]) -> int:` | **给模型看的题** |
| `entry_point` | `Solution().maxSubArray` | 判题时调用谁 |
| `test` | `def check(candidate):` 下 3 条 `assert candidate(nums=...) == 6 / 1 / 23` | **判题依据**，后面会变成 GT |
| `completion` | Kadane 算法的参考实现（7 行） | 参考解，**不参与判分** |

另外三个容易误会的字段：`prompt` 是一串 `from typing import *` 之类的 imports 前缀，树和链表题还带辅助函数定义，不是题目；`meta.question_title` 是完整题面（含 Constraints 段），但预处理只用了更短的 `meta.query`，所以约束没有传给模型；`meta.response` 是一段历史模型回答，不是本轮 rollout。

一个数据质量细节：原文的约束在文件里字面写成 `1 <= nums.length <= 105` 和 `-104 <= nums[i] <= 104`，上标在抓取时丢了，真实含义是 10^5 和 10^4。这种文本错误不能全局替换数字去「修」，只能逐题审阅；它也说明「字段存在」不等于「内容已审核」。

三条断言的输入输出：`[-2,1,-3,4,-1,2,1,-5,4] → 6`、`[1] → 1`、`[5,4,-1,7,8] → 23`。注意它们的答案全是正数，这个事实在第 ④ 站会变成一个坑。

**口述：** 原始记录里「给模型看的题」和「判题用的断言」是两个字段，参考解只是旁证；题面里的约束不一定传给模型。真正进入训练的只有题面、测试和入口，外加用于去重的题号。

源码：读取方式见 `LeetCodeDataset/leetcote/data.py` 的 `read_jsonl()`。全部 14 个字段的含义见 [01 的 metadata 字段字典](01_数据理解与GT构建.md#metadata-fields)。

<a id="station-2"></a>

## 第 ② 站 变成训练样本行：`leetcode2k()` 做了三件事

预处理器 `examples/data_preprocess/coder1.py` 把这一行变成 Code-R1 的样本行（下面是按源码字符串操作复原的真实结果）：

```json
{
  "data_source": "code",
  "prompt": [
    {"role": "system", "content": "You are a helpful programming assistant. ... The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively."},
    {"role": "user",   "content": "Please solve the programming task below using a self-contained code snippet in a markdown code block.\n\nGiven an integer array nums, find the subarray with the largest sum, and return its sum.\n \nPlease complete the following python code precisely:\n```python\nclass Solution:\n    def maxSubArray(self, nums: List[int]) -> int:\n```"}
  ],
  "ability": "coding",
  "reward_model": {"style": "rule", "ground_truth": "{\"functional\": \"def check(candidate):\\n    assert candidate(nums = [-2,1,-3,4,-1,2,1,-5,4]) == 6\\n    assert candidate(nums = [1]) == 1\\n    assert candidate(nums = [5,4,-1,7,8]) == 23\\n\\n\\ncheck(Solution().maxSubArray)\"}"},
  "extra_info": {"split": "train", "index": 142, "reference": "class Solution: ...", "prompt": "...", "dataset": "LeetCodeDataset"}
}
```

三件事：

1. **造 prompt**：系统提示要求 `<think>…</think><answer>…</answer>`；用户消息 = 固定一句话「用自包含代码块解题」+ `meta.query`。不拼 imports，所以模型必须自己写 `from typing import List`。完整文本和「放什么、不放什么」见 [01 的 actor prompt](01_数据理解与GT构建.md#actor-prompt)。
2. **造 GT**：`test` 原文 + 两个换行 + `check(Solution().maxSubArray)`，整体 `json.dumps` 成一个字符串放进 `reward_model.ground_truth`。GT 就是「断言 + 调用入口」，不是参考代码。GT 也不拼原始 `prompt` 前缀：本题没影响，但 1890 道 LeetCode 训练题里有 121 道树和链表题，测试调用前缀里的 `tree_node`、`is_same_tree` 等辅助函数，在 Code-R1 里必然 `NameError`，正确解也只拿 0.1，见 [01 的内部审查例子](01_数据理解与GT构建.md#internal-external-review)。源码补强版 `coder1_plus` 已修好这一点，见[反向补充源码](examples/source-supplements/README.md)。
3. **记履历**：`extra_info` 存 split、index、参考解、纯文本 prompt、来源名。`index` 是过滤去重后训练集里的位置，不是原文件行号；这条记录恰好也是 142，因为 v2 RL 与 v2 SFT 本来就是 945 + 945 道互不重复的题，且都不与 v2 test 的 `question_id` 相交，过滤和去重一行都没删掉。判分时只会读 `extra_info.prompt` 用来打日志。

同时做了去重和划分：合并 v2 RL 与 v2 SFT，排除 v2 test 的 `question_id`，同 ID 只留第一次出现，得到 1890 道题（README 说的「2K LeetCode」）；与 TACO 的约 10K 条合并、打乱后写成 parquet，目录名形如 `data/code-r1-12k-leetcode2k-taco/`，而 `main_grpo.sh` 读的是 `data/code-r1-12k/`，用前要对齐路径。

**口述：** 预处理把「题面」交给 prompt、把「断言 + 入口」打包成 GT 字符串、把「参考解」放进履历；三者消费者不同，不能混用。

源码：`coder1.py` 的 `process_fn`（`:305-335`），`dataset_makes=[leetcode2k, taco]`（`:359`）。九步数据准备流水线见 [01 的全景](01_数据理解与GT构建.md#pipeline-1)。

<a id="station-3"></a>

## 第 ③ 站 采样：这条记录在一步训练里变成 16 个回答

`main_grpo.sh` 的设定：每步取 `train_batch_size=16` 道题，vLLM 对每题生成 `rollout.n=16` 个回答，温度 1.0，回答最长 4096 token。一步共 256 个回答，这条记录贡献其中 16 个，它们共享同一个 `uid`。

进入模型前，`RLHFDataset` 把两条聊天消息套上 Qwen2.5 的 ChatML 模板，大致是：

```text
<|im_start|>system
You are a helpful programming assistant. ...<|im_end|>
<|im_start|>user
Please solve the programming task below ... class Solution: ...<|im_end|>
<|im_start|>assistant
```

模型从 `assistant` 之后开始生成。一个格式合格的回答长这样（课程演示用的固定候选，不是真实采样）：

```text
<think>Fixed teaching candidate; no model generation.</think><answer>```python
class Solution:
    def maxSubArray(self, nums):
        ...
```</answer>
```

采样后只保留 token 序列；回答是自然停止还是被 4096 截断，当前版本不单独记录，只能从有没有 EOS 推断。

**口述：** 一题多答是 GRPO 的前提；同题回答靠 `uid` 认亲，不靠在 batch 里的相邻位置，因为后面会按 token 数重排。

源码：`verl/trainer/ppo/ray_trainer.py` 的 `fit()`（`:914-942`：生成、分配 uid、`repeat(16)`、`_balance_batch`）；`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`。

<a id="station-4"></a>

## 第 ④ 站 判分：一个回答怎样变成 -1.1 / 0.1 / 1.1

`PrimeRewardManager` 用线程池对 256 个回答逐个调用 `coder1._compute_score`，每个回答走四步：

| 步骤 | 做什么 | 失败时 |
|---|---|---|
| 1 格式检查 | strip 后必须以 `<think>` 开头、`</answer>` 结尾 | 返回 **-1.1** |
| 2 抽代码 | 取最后一个 `<answer>` 里的所有代码块，拼起来 | 没代码块也是 **-1.1** |
| 3 拼程序 | `候选代码 + "\n" + GT 的 functional`，即代码后面跟着 `check(...)` 的断言 | |
| 4 沙箱执行 | firejail 里 `python3 -c 程序`，30 秒、4 GB 内存、禁网；退出码 0 算成功 | 非 0 返回 **0.1**；成功返回 **1.1** |

0.1 = 格式分，1.1 = 格式分 0.1 + 正确分 1.0。管理器层面若单个回答超 600 秒或抛异常，回填 **0.0**。

课程有 6 个固定候选，判分结果如下。课程的 GT 在 3 条 seed 之外补了 5 条诊断断言：

| 候选 | 做了什么 | reward | 判分日志里的证据 |
|---|---|---:|---|
| `reference` | 正确的 Kadane 写法 | 1.1 | `Test Execution Passed!` |
| `correct_prefix` | 正确的前缀和写法 | 1.1 | `Test Execution Passed!` |
| `wrong_empty` | 允许空子数组，全负数组返回 0 | 0.1 | `AssertionError: diag-all-negative`（输入 `[-3,-2,-5]`，期望 -2） |
| `wrong_sum` | 直接返回整个数组的和 | 0.1 | `AssertionError: seed-1`：应得 6，返回 1 |
| `bad_format` | 只给代码块，没有 think/answer 标签 | -1.1 | `Bad format detected!` |
| `early_exit` | 代码第二行 `sys.exit(0)` | **1.1** | 没有任何断言执行，但退出码是 0 |

两个值得口述的坑：

- **原始 GT 抓不住 `wrong_empty`。** 3 条 seed 的答案都是正数（6、1、23），它都返回正确值，只有补了全负输入 `[-3,-2,-5] → -2` 才失败。所以在原始 Code-R1 数据上，这个错解会拿 1.1。这就是 README 说的「reward 假阳性会迷惑模型」。整个 LeetCode 部分都是这样：1890 题的测试全是题面样例，中位数只有 2 条。源码补强版用独立 oracle 核对后给这道题补了 10 条测试，wrong_empty 就被判成 0.1。
- **退出码 0 不等于跑完了断言。** `early_exit` 在 `check()` 之前就退出，判分器只看到成功。要堵这个洞需要「测试完成证据」，不是换个沙箱就能解决。源码补强版的受保护判题协议已经堵上：期望答案不进候选进程，结果缺失就判失败。

**口述：** 判分 = 格式门 + 把断言拼到代码后面跑一遍 + 看退出码；三档分数里 0.1 不是通过；高分还要问「断言真的跑完了吗、断言本身够不够」。

源码：`verl/utils/reward_score/coder1/__init__.py` 的 `validate_response_structure`（`:37`）、`extract_code_from_string`（`:57`）、`_compute_score`（`:63-127`）；`firejail_exec.py:38-49, :95-96`；`verl/workers/reward_manager/prime.py:64, :81-83`。

<a id="station-5"></a>

## 第 ⑤ 站 reward 变成 advantage：同题相比，再铺到每个 token

先放位置：每个回答的 reward 是一个标量，写在该回答**最后一个有效 token**上，其余位置为 0。用课程里的教学长度画一个回答（长度 7，最大长度 8）：

```text
token 位置        0    1    2    3    4    5    6    7
eos_mask          1    1    1    1    1    1    1    0
token_level_reward 0    0    0    0    0    0   1.1   0
```

再分组归一化。用 6 个候选里的 4 个算法候选 `reference`、`correct_prefix`、`wrong_empty`、`wrong_sum` 代替真实的 16 个回答：

```text
同 uid 的 reward      [1.1, 1.1, 0.1, 0.1]
均值                  0.6
样本标准差（除 N-1）  0.57735
advantage            [+0.866, +0.866, -0.866, -0.866]
```

然后把 +0.866 复制到第一个回答的全部 7 个有效 token 上，-0.866 复制到失败回答的全部 token 上。推理部分和代码部分拿到的是同一个数：这是粗粒度的 outcome 信号，不区分哪一步推理对了。

三个边界：

- 16 个回答全对或全错，std = 0，advantage 全 0，这题这一步没有相对信号（KL 和 entropy 项仍可能有梯度）。
- 一次基础设施异常回填 0.0 会改变全组：`[0.1,0.1,0.1,0.0]` → `[+0.5,+0.5,+0.5,-1.5]`，三个同样失败的回答突然得到正信号。源码补强版的奖励管理器先重试，仍失败就把整组作废，组内 advantage 全为 0。
- 组内只有 1 个回答时，源码用 mean 0、std 1。

**口述：** GRPO 不要 critic，用同题其他回答当 baseline；reward 先放末 token，求和后按 uid 分组标准化，再广播到整个回答。

源码：`prime.py:144`（末 token 赋值）；`verl/trainer/ppo/core_algos.py` 的 `compute_grpo_outcome_advantage`（`:111-154`）；`ray_trainer.py:978-991`（`use_kl_loss=True` 时 `token_level_rewards = token_level_scores`，再算 advantage）。

<a id="station-6"></a>

## 第 ⑥ 站 loss 与更新：advantage 怎样变成参数变化

需要三份 logprob，都是「模型对自己已生成 token 的对数概率」：

| 名字 | 谁算 | 什么时候 | 有梯度吗 |
|---|---|---|---|
| old | 当前 actor，`eval()` + `no_grad` | 采样之后、更新之前重算，不用 vLLM 的值 | 否，常量 |
| ref | 冻结的参考模型（初始权重，bf16，CPU offload） | 同上 | 否，常量 |
| new | 当前 actor，`train()` | 更新时 forward | **是** |

每个有效 token 上：

```text
ratio = exp(new - old)
pg    = max( -A·ratio , -A·clip(ratio, 0.8, 1.2) )
loss  = mean_over_tokens(pg) - 0.001·entropy + 0.001·low_var_kl(new, ref)
```

然后 `loss.backward()`，`clip_grad_norm_(1.0)`，AdamW 以 lr 5e-7 走一步。判题程序不可微也没关系：梯度只经过「固定 advantage × 当前模型 logprob」这条路回到参数。

用 `mechanism_lab.py` 的两标量 toy 看方向：两个候选 token 的 logits `[0, ln3]`，概率 `[0.25, 0.75]`，模型生成了第二个，A = +0.5，old = new 所以 ratio = 1，loss = -0.5。对 logits 的梯度是 `[+0.125, -0.125]`，学习率 0.1 走一步后第二个 token 的概率从 0.75 升到 0.7547。把 A 改成负数，方向反过来。

两个常见误区：

- loss 数值等于 0 不代表没梯度；ratio = 1 时导数仍是 -A。
- clip 裁的是目标函数的有利方向，不是梯度范数；A>0 且 ratio>1.2 时这个 token 的 PG 梯度为 0。

在 `main_grpo.sh` 的配置下（`ppo_epochs=1`，每步每卡只有一个 mini-batch），更新用的权重就是算 old 的权重，ratio 基本恒为 1，clip 几乎不触发；这一点是源码推断，训练日志里的 `actor/pg_clipfrac` 可以验证。

**口述：** 固定 old、ref、advantage，只对 new 求导；PPO 的 ratio 和 clip 是为了复用旧样本时不走太远；KL 项把模型拴在参考模型附近。

源码：`verl/workers/actor/dp_actor.py` 的 `compute_log_prob`（`:168-224`，no_grad）与 `update_policy`（`:226-327`）；`core_algos.py` 的 `compute_policy_loss`（`:272-303`）、`kl_penalty`（`:351-383`）；`verl/workers/fsdp_workers.py:119-120`（mini-batch 按题数乘 16 再除以卡数）。

<a id="station-7"></a>

## 第 ⑦ 站 同步、验证、保存：训练「最后」留下什么

- **权重同步**：下一步生成前，sharding manager 取 actor 的 `state_dict()` 装进 vLLM（新版本走 `wake_up` + 加载权重，旧版本走 `sync_model_weights`），生成完再卸载。每次生成都同步一次，包括验证。
- **验证**：每 16 步跑一次 `_validate()`，对验证集每题**贪心生成 1 个回答**，用同一个判分器打分；`val/test_score/code` = 这些 reward 的均值。它是 raw reward 均值，不是通过率：三个格式合格但全错的回答平均也是 0.1。
- **保存**：每 64 步在 `models/code-r1-12k-grpo/global_step_N/` 写 actor 分片、optimizer、lr scheduler、RNG，以及 dataloader 状态 `data.pt`。训练结束时的「最后」就是最后一个 `global_step_N/actor`，可以用 `scripts/model_merger.py` 合并成 HF 格式。
- **日志**：wandb 里看 `critic/score/mean`（本步 reward 均值）、`actor/pg_loss`、`actor/kl_loss`、`actor/entropy_loss`、`response_length/mean`、`val/test_score/code`。

**口述：** 更新完要证明三件事分开成立：参数真的变了、下一轮采样用的是新参数、保存的状态能恢复；效果好不好另用固定协议评。

源码：`verl/workers/sharding_manager/fsdp_vllm.py` 的 `__enter__/__exit__`；`ray_trainer.py` 的 `_validate()`（`:609-690`）、`_save_checkpoint()`（`:759-790`）。

## 60 秒口述版

一条 LeetCode 记录里，题面给模型，三条断言打包成 GT 字符串，参考解只做旁证。训练每步取 16 题，vLLM 给每题采 16 个回答。每个回答先过格式门，再把代码和断言拼成一个程序丢进 firejail 跑，退出码 0 得 1.1，失败 0.1，格式错 -1.1。标量 reward 写在回答末 token，同题 16 个回答做均值方差标准化得到 advantage，广播到整个回答。然后固定 old 和 ref 的 logprob，用当前模型重新 forward 得 new logprob，算 ratio、PPO clip、加一点 KL 和熵，反传、AdamW 一步。新权重装回 vLLM 进入下一步；每 16 步贪心验证，每 64 步存 checkpoint。整个闭环里最脆弱的是判分：断言不够、退出码骗过判分器，都会把错误行为奖励进模型。

## 面试追问六条

| 追问 | 一句话答 |
|---|---|
| GT 是参考代码吗 | 不是。GT 是 `check(candidate)` 断言加入口调用的 JSON 字符串；参考解在 `extra_info` 里，判分不读它 |
| 为什么失败是 0.1 不是 0 | 0.1 是格式分；格式错才是 -1.1；0.0 是管理器超时或异常的兜底值，三者含义不同 |
| 退出码 0 就是通过吗 | 不是。`sys.exit(0)` 在断言前退出也得 1.1；原始三条断言的答案全是正数，抓不住「允许空子数组」的错解 |
| 16 个回答怎么变成信号 | 同 uid 组内 (r − mean)/(std + 1e-6)，广播到回答所有 token；全同分组信号为 0 |
| 梯度从哪里来 | 不经过沙箱；固定 advantage 加权当前模型的 token logprob，反传到参数；old 和 ref 是无梯度常量 |
| `val/test_score/code` 是通过率吗 | 不是，是贪心单回答的 raw reward 均值，LeetCode 和 TACO 合在一起 |

<a id="reproduce-00"></a>

## 在本机亲手复现每一站

环境与[资料首页](README.md#run-setup)相同：

```bash
PY=/Users/s11437/.local/share/uv/python/cpython-3.12-macos-aarch64-none/bin/python3.12
cd /Users/s11437/Code-R1-reaserch

# ① 看原始记录
sed -n '143p' LeetCodeDataset/data/LeetCodeDataset-v2-rl-problems.jsonl | $PY -m json.tool | head -40

# ②④⑤ 用固定候选跑完整 CPU 演示（prepare → verify → score → advantage → evaluate）
$PY -B RLVR学习资料/examples/maximum-subarray/run_demo.py all
$PY -m json.tool RLVR学习资料/examples/maximum-subarray/outputs/prepared_task.json | head -60   # 样本行
$PY -m json.tool RLVR学习资料/examples/maximum-subarray/outputs/report.json | less               # 判分日志、advantage 矩阵

# ⑥ 两标量梯度 toy
$PY -B RLVR学习资料/examples/maximum-subarray/mechanism_lab.py
```

③ 和 ⑦ 需要 Linux GPU 环境才能真实执行；本机只能读源码。注意演示的 GT 比原始 GT 多 5 条诊断断言，`run_demo.py` 的 prompt 也补了约束说明，这些是课程适配，不是原始预处理的行为。

<a id="progress-00"></a>

## 学习路线与进度（以本文为主线）

| 站 | 对应阶段与作业 | 练习入口 | 状态 |
|---|---|---|---|
| ① 原始记录 | [01 作业 1.1](01_数据理解与GT构建.md#assignment-1-1) | `run_demo.py prepare`，读 `outputs/raw_task.json` | 2026-10-14 纵览已记入本文开头，01 的总览和文字版导读已记入 01 开头；待口答 01 的「检查是否学会」 |
| ② 样本行与 GT | [01 作业 1.2](01_数据理解与GT构建.md#assignment-1-2)、[1.3](01_数据理解与GT构建.md#assignment-1-3) | `practice.py init --run data-01 --mode seed` 已创建；填 `prediction.json` 后 `check`，再加自己的配对 | 进行中：01 已重排为「概念、动手、自测」三段；等待填写 data-01 预测 |
| ③ 采样 | [03 前置自测](03_Rollout与GRPO机制.md#prerequisites-3) 与 3.1 的时序部分 | 读 `ray_trainer.py::fit()`；本机不能真实采样 | 未开始 |
| ④ 判分 | [02 作业 2.1–2.3](02_Verifier与任务池验收.md#assignment-2-1) | `run_demo.py score`；`practice.py score` 用 data-01；`blind-init/blind-grade` 用 blind-01 | 2026-10-15 02 已按 01 的结构重排，总览和文字版导读已写入；待做 2.1–2.3，2.2 要先完成 01 的作业 1.2 |
| ⑤ advantage | [03 作业 3.1、3.2](03_Rollout与GRPO机制.md#assignment-3-1) | `run_demo.py advantage` | 未开始 |
| ⑥ loss 与更新 | [03 作业 3.3](03_Rollout与GRPO机制.md#assignment-3-3) 与 [梯度链路](03_Rollout与GRPO机制.md#gradient-bridge) | `mechanism_lab.py --run mechanism-01` | 未开始 |
| ⑦ 同步、验证、保存 | [04 作业 4.1–4.3](04_真实训练与独立评测.md#assignment-4-1) | `run_demo.py evaluate`、`practice.py transfer-init/transfer-grade` | 未开始 |

学习中对本文或 01–04 的修改记在这里（日期 · 改了什么 · 为什么）：

| 日期 | 修改 | 原因 |
|---|---|---|
| 2026-10-13 | ① 补约束文本上标丢失的数据质量点；② 修正 `index` 的含义、写明 1890 题与一行未被过滤的事实、修正 parquet 目录名与训练脚本路径不一致 | 进入阶段 01 时复查原始数据与 `coder1.py` 发现 |
| 2026-10-13 | 01 开头新增「全景：九步流水线」「metadata 字段字典」「actor prompt」三节，面试深挖加一行；00 第 ①② 站加链接，补 GT 不拼前缀导致 121 道树和链表题必然失败 | 学 01 时发现原 01 按作业组织、缺流程全景、字段解释用的是另一道题、没有展示 actor prompt 原文；核对字段时发现前缀依赖，并用一题做了 Code-R1 拼法与原生拼法的对照实跑 |
| 2026-10-13 | 全仓贯穿题统一为 LeetCode 53：主框架 A1–A6 与第三到第七章的服务器题、旧五题例子全部换成 LeetCode 53，第七章改为证据边界与历史归档；A5 新增经实跑核对的错解表；全部命令统一为 bash 加 `$PY`，环境设置集中在首页；00 加入首页路线和各篇导航；修正 00 第 ④⑤ 站候选名单前后不一致、01 与 00 的字段计数口径 | 用户要求全仓只保留一道贯穿题，并在学习每一篇时用全局视角保持前后统一 |
| 2026-10-14 | 01 重排为三段：先概念四节，即三个输入输出、全景、字段字典、actor prompt；再「动手前」工具说明和三项作业；最后 60 秒口述、面试深挖、按需读源码、完成记录。合并了重复的四个导航小节，作业改成编号步骤，压缩重复的证据警告；新增 1.1 口答题的参考要点和两条面试追问。02–04 仍是旧布局，学到时按同样结构重排 | 学 01 时发现导航小节重复、最基础的三个输入输出排在第六屏、作业说明冗长，不利于阅读和口述 |
| 2026-10-14 | 新增「反向补充源码」：原版不动，新增补强版 `coder1_plus`，先修 GT 执行契约，121 道树和链表题全部恢复；01、00、主框架、首页同步引用；补上「LeetCode 测试全是题面样例」这一事实；候选清单列出后续六项，含沙箱 | 用户新增学习规则：资料覆盖多于源码且设计靠谱时，要反向补充源码 |
| 2026-10-14 | 用户确认补强方案后，完成 01 阶段的补强项 2、3：变异审计工具与全量审计结果，独立 oracle 核对的补测试工具与 7 道示范题闭环；01、00、主框架同步结果与边界 | 按学习进度，01 阶段的补强项要在学 01 时完成 |
| 2026-10-14 | 补完补强项 4 到 7：受保护判题协议、结构化沙箱与 macOS 隔离后端、判题故障整组作废的奖励管理器、评测集去重与按通过率筛题；00 第 ④⑤ 站、01、02、03、主框架同步指向补强版 | 用户目标要求把沙箱和 verifier 也构造出来，这几项提前到 01 阶段完成，学 02、03 时当学习材料用 |
| 2026-10-14 | 00 开头的阶段对应补上第 ③ 站：采样属于 03。01 的九步改用 1 到 9 编号，作业对应和正文直接写步骤名 | 重新带学 01 时发现：00 开头漏了 ③；01 开头说「对应第 ① ② 站」，几行后的 ② 却指「抽取 metadata」，两套圈号容易混 |
| 2026-10-14 | 00 开头新增「纵览」：一步训练做什么、七站的进出与所在篇、四篇各答什么问题、本机能跑到哪一步；原来开头那行「与四阶段的对应」并入纵览。01 开头新增「总览」和「文字版导读」：原来的「这一篇解决什么」「怎么学」并入总览，全景里的「九步与作业的对应」移到总览 | 用户要求把带学时的纵览记进 00、01 总览和文字版讲解记进 01，只记按最新源码和资料核对过的版本；以后每学一篇 0x 都按这个流程 |
| 2026-10-14 | 写入前逐条复核，并修正旧说法：「三条正数 seed」改成「三条 seed 的答案都是正数」（00、01、主框架）；「原版文件不动」改成「原版预处理和判分文件不改，只在判分路由和训练入口各加一个分支」（首页、01）；全量变异审计复跑后，2013 题、17306 个变异体更新为 2014 题、17316 个，补强版契约下训练集通过数从 1876 更新为 1877（01、主框架、反向补充源码 README）。复核依据：重跑 `run_demo.py all`、2090 题 GT 契约对照、全量变异审计、typing 在 3.12 与 3.14 的对照和数据统计 | 种子输入里有负数，只有答案是正数；补强版实际在两个原版文件里各加了 4 行；补强项 4 上线后 `word-ladder-ii` 稳定通过，进入了审计 |
| 2026-10-15 | 02 按 01 的结构重排：开头新增总览 `overview-2` 和文字版导读 `guide-2`；概念五节为三个角色、判分七步、reward 三档、执行边界的四个缺口与补强版、任务池与五层归因；三项作业改成编号步骤，各加「检查是否学会」与折叠的参考要点；面试深挖从 6 行扩到 10 行；新增 60 秒口述、按需读源码、完成记录。删掉「另次 audit」「合同错配」「受控变异回归」「指标分母」等说法；执行量例子从教学设定 B=8、G=4、M=3 换成 Code-R1 的实际情况：函数题每步 256 次程序执行。00 纵览里 02 的链接改指 `overview-2`，首页 02 一行改写 | 开始学 02；02 还是旧布局，句子密、缩写多，不便口述 |
| 2026-10-15 | 写入前逐条复核 02 的说法：重跑 `run_demo.py score`、`practice.py score` 与 `--bad-format`、补全负输入、blind 题包（只用 seed 54、55）、`verify_judge_and_selection.py`；用原版判分器实跑草稿代码块、开头寒暄和 strip 比较；读四个执行后端和奖励管理器源码。新写进资料的事实：四个后端都按退出码判成功，kira 只看 stderr；docker 后端不关网络，stdout 与 stderr 混在一起，`finally` 里强制删容器；管理器超时后线程池退出时仍会等线程；标准输入输出题并发执行、按完成先后判；`minimize_stdio` 按输入大小只留最小的 8 条；TACO 的 split 循环遍历整个数据集、改的是临时字典；两处「参考解」不是同一份程序 | 用户要求写进资料的内容都按源码和资料的最新状态核对过 |
| 2026-10-15 | 可跑性评估：在仓库副本里重跑资料写到的全部 CPU 命令，全部能跑；新增 `source-supplements/verify_reward_manager_cpu.py`，在装了 torch 2.4.0 CPU 版的 venv 里整体跑通原版与补强版奖励管理器，supplements README 的补强项 6 改为「CPU 整体已验证，firejail 分支与接入训练待 Linux」，并写明 venv 命令和 `--parquet` 已实跑 | 用户要求评估源码和作业哪些能在 CPU 跑、哪些不能、要补什么；补强项 6 原来写「整体待 Linux 实跑」，现在有了 CPU 证据 |
| 2026-10-15 | `examples/training/README.md` 的单卡 smoke 命令：路径值加单引号；新增「实跑前必查」五条（中文路径、单卡用 vLLM 0.6.3、先下载模型、firejail 探针、1 步时 lr=0）；写明 `--cfg job --resolve` 在 Mac 的 venv 里就能做 | 可跑性评估时在 venv 里实测：路径含「学习资料」不加引号，Hydra 解析阶段就报错；加引号后整条命令的配置能展开 |
| 2026-10-15 | 新增《路线评估_仅学习与实跑》：可行性判断、两条路线对比、仅学习要补的八项、实跑三档、面试覆盖表、诚实话术、待确认四问；首页「选择可完成的范围」加链接 | 用户问只学不复现能否过面试、要补什么，并要求维护一份分路线的文档 |
| 2026-10-15 | 《路线评估》按用户回答更新：岗位是业务算法（后训练与数据构造）、有 RL 与 LLM 训练基础、预算 300 元以内、两到三个月；新增十二周安排、300 元预算用法、第 9 项数据构造横向视角 | 用户回答了路线评估的四个待确认问题 |
| 2026-10-16 | 新增《05 实跑推演_Qwen3-8B》与产物目录 `examples/training/runs/qwen3-8b-grpo-20260928/`（生成器 `scenario_qwen3_8b.py`）：Qwen3-8B 两轮 GRPO，七环各有「你来决定」；首页路线表加 05 一行；路线评估加指向 | 用户目标改为：模拟一次 Qwen3-8B 实跑并让其思考全流程；产物按真实形态做，来源说明保留 |
| 2026-10-16 | 推演经 4 个视角评审（数字一致性、verl v0.4.1 保真、Qwen3 与物理合理性、教学与标注）后重做：生成器改为确定性种子、整数计数、v0.4.1 字段与打印、run-02 的 epochs 与 chunked prefill 断言、去掉错误的「lr=0」「grad_norm 缩 1/16」说法；05 的导读和第七环答案折叠、总览去掉事后评价、`clip_ratio` 改称截断率；产物目录改名 `qwen3-8b-grpo-20260928`，推演日期设在 10-07 到 10-11；新增 `check_scenario_docs.py` 核对叙述与产物 | 评审查出叙述与产物数字对不上（根源是 hash() 随机化）等 35 条必须修的问题 |
| 2026-10-16 | 05 带学开始：第一环用户选 4×A800、vLLM 睡眠、checkpoint 放根盘全留、按 4×A800 估预算；新建《显存帐_Qwen3-8B》和 `notes/your_decisions.md`，05 第一环加链接 | 用户要求逐环做决定，并要一份专门看显存的帐 |
| 2026-10-16 | 推演参考的机器从 8×H100 改为 4×A800（目录改名 `qwen3-8b-grpo-20260928`，日期 09-27 到 10-09，run-01 90.1 小时 ¥3,605、run-02 124.5 小时 ¥4,980）；05、README、决策日志、事故报告、显存帐同步 | 用户要求资源与费用保持个人研究量级 |
| 2026-10-16 | 按用户前四环的决定生成「你的 run」your-01（`scenario_yourplan.py` → `runs/qwen3-8b-grpo-yourplan-20261018/`）：4×A800、12.4k 行、6144 / n=8 / 200 步、mini-batch 4、prime_plus；42.7 小时 ¥1,709，reward 0.211 → 0.548，dev 贪心 0.630 → 0.680；存盘后沙箱超时作废组的周期性事故留给第五环自己找 | 05 第五到七环要有用户自己的曲线可读 |
| 2026-10-16 | 05 第五到七环走完：曲线四问由导师给机制与证据链（记在 `your_decisions.md`）；评测补 LCB held-out（your-01 贪心 dev +5.0、held-out +4.2，pass@8 不动）；第七环定 your-02（tmpfs、筛题 + clip_ratio_high 0.28、续训 200 步、切 300 题验证），计划与 `config/your-02.sh` 在 yourplan 目录；补强项 8 候选登记进 supplements README；检查器扩到 94 个数字 | 用户要求全流程自己思考，曲线由导师解释 |
| 2026-10-16 | （上面四条 2026-10-16 记录里的时长、费用与评测数字——90.1 / 124.5 / 42.7 小时、dev +5.0 等——已全部作废）推演起点改为 Qwen3-8B-Base（预训练模型，没经过 SFT 或 RL，架构与 Qwen3-8B 相同；R1-Zero 式，`<think>…</think><answer>…</answer>` 格式靠系统提示从头学）：两轮参考推演与 your-01 重生成，05、产物目录叙述、决策日志、显存帐、首页、路线评估、CLAUDE.md 同步。关键数字：run-01 76.1 小时 ¥3,043，reward −0.409 → 0.559，验证 −0.224 → 0.595；run-02 100.6 小时 ¥4,023，reward −0.174 → 0.709，验证 −0.096 → 0.656；your-01 24.5 小时 ¥979，reward −0.341 → 0.397，验证 −0.166 → 0.559。离线评测基线 dev 贪心 0.380、held-out 0.134；run-01 / run-02 / your-01 的 dev 贪心 0.615 / 0.635 / 0.575，held-out 0.246 / 0.317 / 0.261，dev 采样 pass@8 0.768 → 0.890 / 0.907 / 0.870。前 64 步增益约八成是格式（run-01 85%、run-02 78%、your-01 84%），采样 pass@1 与 pass@8 都大幅上升，「基线强、只锐化不扩展」「增益在噪声内」的旧说法作废；仍是推演，不是实测 | 用户 2026-10-16 明确基础模型就是 Qwen3-8B-Base；旧叙述按后训练版 Qwen3-8B 写，「基线过高」与用户的起点矛盾 |
| 2026-10-19 | Base 重写后的三视角审查修复：`enable_thinking` 口径统一（默认不预填、传 False 会预填空 think 块而撞格式门）、决策日志里 8 卡残留算术改为 4 卡（每卡 64 条、16 个 micro）、grad_norm 区间统一、run-02 长度谷底改为第 40 步 1,286、EvalPlus 常数改为 Base 量级（0.671 / 0.681 起，训练后 +2–3 个点）并重生成、显存帐例子与 KV 标签改、筛题预算与数据行数统一、你的 run 的日期与目录描述改；检查器 94 个数字、链接审计 631 条全部通过 | 工作流的审查员发现，核实后逐条修 |
| 2026-10-19 | 用户决定去掉 TACO、用 your 的配置做完整一轮：`scenario_yourplan.py` 重写为两轮（your-01 第 1–200 步诊断；your-02 从第 200 步续训到 400，tmpfs + 筛题 + clip-higher + 换验证集），数据与参考 run-01 同一份 LeetCode 1889 题，rollout 落盘里的代码经真实判分代码验证；your-01 21.2 小时 ¥847、held-out 0.134 → 0.261，your-02 26.1 小时 ¥1,043、held-out 0.317（对 your-01 净胜 8 题）；yourplan 目录的 README / PROVENANCE / 协议 / 报告 / 事故报告 / 归因 / your-02 记录全部重写，your_decisions 五到七环按 LeetCode 版重写并加第八节 | 用户：「假如 your 的其他配置比 run 更合理，先去 TACO 版本，做到完整的诊断 → 修 → 验证」 |
| 2026-10-19 | your-02 去掉计划里的 `clip_ratio_high` 0.28 并重生成：它对着的「pass@8 不动」在 Base 起点下不存在，没有消融预算时不做预防性改动；your-02 只剩 tmpfs、筛题、换验证集三处，熵 0.29 → 0.28 留作观察项，其余数字不变；README / PROVENANCE / 报告 / 计划记录 / your_decisions / 05 / CLAUDE.md 同步 | 用户：「假如去更合理的话去掉然后跑」 |
| 2026-10-19 | 03 按 01、02 的结构重排：总览 `overview-3`、导读 `guide-3`，概念五节（前置自测、采样与分组、reward 到 advantage、从 logits 到更新、本版本的运行态），动手前工具、三项作业、60 秒口述、面试深挖（加「这套脚本里 clip 真的起作用吗」「熵系数和回答长度」两问）、有资源后、按需读源码、完成记录；把 CLAUDE.md 第 7 节的 mini-batch 语义错配、ratio 恒 1、梯度缩放、首步 lr=0 并入「运行态」一节并注明 v0.4.1 的差别；外部锚点不变 | 用户要求 03/04 重排并给出总览与文字版介绍 |
| 2026-10-19 | 04 按同样结构重排：总览 `overview-4`、导读 `guide-4`，概念五节（四种证据、评测口径、协议先锁、一步真实更新、故障归因），把 05 的第六、七环当沙盘引用，补 v0.4.1 指标名、单卡 vLLM 0.6.3、续训去 `data.pt`、去污染与 held-out；三项作业只留操作与交付；综合通关、口述、面试深挖（加 held-out、配对净胜两问）、按需读源码、完成记录；外部锚点不变。01、02 的布局复核无需改动 | 同上 |
| 2026-10-19 | 新增《专题索引_任务构造与沙箱》：Task 选取/筛选/构造、GT 构造、expected/oracle/测试扩展、Sandbox 四个专题各在哪篇哪节、哪个补强项、哪个源码文件，按先读 → 深读 → 补强版 → 代码与工具 → 本课的数字列出；首页和 CLAUDE.md 登记 | 用户要求用一个文件维护这四个专题的位置 |

---

**继续阅读：** [返回资料首页](README.md) · [下一阶段：01 数据与 GT](01_数据理解与GT构建.md)
