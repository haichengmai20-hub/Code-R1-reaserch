# 阶段三：reward 怎样变成梯度——采样、分组与 GRPO 更新

**资料导航：** [资料首页](README.md) · [工业主框架](RLVR工业流程.md) · [00 全程走读](00_一条记录从原始Task到训练.md) · [01 数据与 GT](01_数据理解与GT构建.md) · [02 Verifier 与任务池](02_Verifier与任务池验收.md) · **03 Rollout 与 GRPO（当前）** · [04 训练与评测](04_真实训练与独立评测.md)

**先记机制：** reward 是对一个回答的行为反馈，advantage 是同一道题的回答互相比出来的相对分。相对分定下来之后就是常数，真正求导的是当前模型生成每个 token 的概率：相对分加权 logprob，反传回参数。判题程序不需要可微。同题分组、token 对齐、求导边界、平均方式，任何一处错配，优化的就是另一件事。

<a id="overview-3"></a>

## 总览：03 在流程里做什么、长什么样

03 对应 [00 全程走读](00_一条记录从原始Task到训练.md#overview-00)七站里的 ③ 采样、⑤ advantage 和 ⑥ loss 与更新（⑥ 的真实运行归 04）。它要回答的问题是：reward 怎样变成梯度。答案分三步：

- 采样：一步取 16 道题，vLLM 给每题写 16 个回答，同一道题的 16 个共享一个 `uid`；回答按 token 数重排，所以认亲只能靠 uid。
- 分组：reward 放在回答的最后一个有效 token 上，同 uid 的 16 个分数做 (r − mean) / (std + 1e-6)，得到 advantage，再铺到这个回答的每个 token。
- 更新：固定 old、ref、advantage 三个常量，只对当前模型的 new logprob 求导；ratio = exp(new − old) 过 PPO clip 0.2，加 0.001 的 KL 和熵项，反传后 AdamW 以 lr 5e-7 走一步。

全篇分三段：

| 段 | 包含的小节 | 学完能回答什么 |
|---|---|---|
| 概念 | [前置自测](#prerequisites-3)；[采样与分组](#sampling-3)；[reward 到 advantage](#advantage-3)；[从 logits 到更新](#gradient-bridge)；[本版本的运行态](#runtime-facts-3) | shape 和位置对齐是什么；一题多答怎样成组、为什么不能按相邻行猜；分数怎样变成相对分、全同分组怎么办；三份 logprob 各从哪来、谁有梯度、clip 和 KL 各管什么；`main_grpo.sh` 这套配置实际跑出来是什么样 |
| 动手 | [动手前的工具说明](#tools-3)；作业 [3.1](#assignment-3-1)、[3.2](#assignment-3-2)、[3.3](#assignment-3-3) | 固定 4 个候选怎样算出 ±0.866；一次判题故障怎样把全组信号改掉；换一组数字能不能自己算出梯度方向和 clip 分支 |
| 自测 | [60 秒口述](#oral-3)；[面试深挖](#interview-3)；[有资源后](#rollout-real-3)；[按需读源码](#source-reading-3)；[完成记录](#record-3) | 能不能脱稿讲清楚，能不能扛住追问；真要采样前先定什么 |

概念各节末尾各有一句「口述」，读完要能脱稿说出来。机制细节查主框架的 [GRPO 优势](RLVR工业流程.md#grpo-advantage)、[梯度桥](RLVR工业流程.md#gradient-bridge)和[策略更新](RLVR工业流程.md#policy-update)。

概念和作业的对应：

- 3.1 覆盖采样与分组、reward 到 advantage、从 logits 到更新：用固定示范把 4 个候选从判定追到 ±0.866，再对齐 `fit()` 的真实时序和一个 token 的 clip 算术。
- 3.2 覆盖分组的边界：只改一次判题故障的处理条件，看全组 advantage 怎样从全 0 变成 `[+0.5, +0.5, +0.5, −1.5]`。
- 3.3 覆盖对齐与求导边界：混排、改长度、换 clip 的另一侧、自己提一个反例，再用梯度 toy 自检。

03 的「原版 vs 补强版」用补强项 6：原版的奖励管理器把判题故障记成 0.0 混进组内统计，补强版 `prime_plus` 先重试、仍失败就把整组分数设成同一个值，advantage 全 0。它在 3.2 里出现，细节见[反向补充源码](examples/source-supplements/README.md)。

02 交来的是验收过的判分器和每个候选的 reward；03 交给 04 的，是 reward、advantage、logprob 三者的对应关系，以及一条能逐函数指认的更新链路。真正的参数更新、权重同步和效果评测在 04 验收。

<a id="guide-3"></a>

## 文字版导读：先把全篇顺一遍

第一次读，先把这一节读完，再按链接进细节；复习时只读这一节。

**第一节：前置自测。** 只要能读 shape、概率和「把某些数当常量」就能开始。`(B, R)` 是 B 条回答、每条 R 个 response 位置；logits 多一维词表 V。一个容易错的位置：prompt 和 response 拼接后，预测第一个 response token 的是 prompt 的最后一个位置，不是 token 自己的位置。源码普通路径截 `[:, -R-1:-1]`，remove-padding 路径先左移再还原。详见[前置自测](#prerequisites-3)。

**第二节：采样与分组。** `main_grpo.sh` 每步取 16 道题，vLLM 对每题生成 16 个回答，温度 1.0、回答最长 4096 token，一步 256 个回答。同一道题的 16 个共享一个 `uid`，这是批内的组身份，和稳定的 `task_id` 用途不同。生成后 `_balance_batch` 按 token 数把 256 行重排，所以相邻 16 行不是一道题；GT、回答、mask、reward 必须跟着 uid 一起重排。rollout 返回的只有 token 序列，没有停止原因：mask 全 1 可能是正常停止，也可能是被截断。详见[采样与分组](#sampling-3)。

**第三节：reward 到 advantage。** 奖励管理器把标量写在回答的最后一个有效 token 上，其他位置是 0。GRPO 先对每条回答的 token reward 求和（就是那个标量），按 uid 分组，减组均值、除以样本标准差（N − 1）加 1e-6，再把这个数铺到回答的所有有效 token。课程的 4 个候选：`[0.1, 0.1, 1.1, 1.1]`，均值 0.6，样本标准差 0.57735，advantage `[−0.866, −0.866, +0.866, +0.866]`。三个边界：全同分组 std=0，advantage 全 0，这题这一步没有相对信号；组里只有 1 个回答时源码用 mean 0、std 1；原版管理器把判题故障记 0.0，`[0.1, 0.1, 0.1, 0.0]` 会变成 `[+0.5, +0.5, +0.5, −1.5]`——三条错解拿到正信号。详见[reward 到 advantage](#advantage-3)。

**第四节：从 logits 到更新。** 一次更新用到三份 logprob：old 是采样后、更新前用当前 actor 重算的（`eval()` + `no_grad`），ref 是参考模型算的（`no_grad`），new 是训练 forward 算的，只有 new 带梯度。ratio = exp(new − old)；loss 取 `max(−A·ratio, −A·clip(ratio, 0.8, 1.2))`，在每个 micro batch 内按 token 平均，再加 0.001 × low_var_kl(new, ref) 和 −0.001 × entropy；反传、梯度范数裁到 1.0、AdamW 以 5e-7 走一步。一个 token 的算术：old −2、new −1.8，ratio = e^0.2 = 1.2214，裁剪值 1.2；A=1 时取 −1.2，A=−1 时取 +1.2214。clip 裁的是目标函数的有利方向，不是梯度范数：A>0 且 ratio 超过 1.2，这个 token 的 PG 梯度为 0，但其他 token、KL 和熵项仍能改参数。温度也进了重算：logits 先除以 temperature 再 log-softmax。详见[从 logits 到更新](#gradient-bridge)。

**第五节：本版本的运行态。** 读源码要分清「机制上会怎样」和「这套配置实际怎样」。`main_grpo.sh` 把 mini-batch 按回答数写成 256，而 worker 把它当题数再乘 16，于是每步每卡只有一个 mini-batch、一次更新：更新用的权重就是算 old 的权重，ratio 恒 1，`actor/pg_clipfrac` 和 `actor/ppo_kl` 恒 0，clip 在这套配置下不起作用；梯度被多除了 16 倍，`grad_clip=1.0` 实际放宽 16 倍；本 fork 第一步的学习率是 0。这些是源码推断，[05 的曲线解读](05_实跑推演_Qwen3-8B.md#stage-curves)用「两个恒零指标」讲它，并说明 verl v0.4.1 已经改掉了其中两条。详见[本版本的运行态](#runtime-facts-3)。

**第六节：三项作业和自测。** 两套工具：`run_demo.py score / advantage` 是固定示范，只看不改；`mechanism_lab.py --run mechanism-01` 是两标量 logits 的梯度 toy，用解析导数和中心有限差分互相核对，没有 torch。

- **3.1 完整示范与源码对齐。** 从 4 个候选的判定追到 ±0.866，画末 token 和 mask，对齐 `fit()` 的时序，算一个 token 的 clip。
- **3.2 infra 回填怎样改变整个组。** 只改一个条件，算出 `[+0.5, +0.5, +0.5, −1.5]`，写出自己的故障政策，对照补强版。
- **3.3 独立检查。** 混排和改长度、clip 的另一侧、自己提反例，再做一组机制变式。

学完脱稿讲一遍 [60 秒口述](#oral-3)，再从[面试深挖](#interview-3)里随机挑一个追问，按四句来回答：机制是什么、这个版本怎么实现、一个反例、证据能证明到哪一步。真要采样前先看[有资源后](#rollout-real-3)。

读完导读，按顺序读下面的概念五节，再从[作业 3.1](#assignment-3-1)开始动手。

<a id="prerequisites-3"></a>

## 前置自测：shape、概率与求导

`(B, R)` 表示 B 条回答、每条 R 个 response 位置；词表 logits 还有 V 这一维，形状为 `(B, R, V)`。mask=1 的位置才进入本次直接策略 loss。能读 sum、mean、exp、log 和「把某些数视为常量」即可先做纸面与标准库实验；PyTorch 的 `no_grad / backward` 对应何处随后追源码。

口答：prompt 与 response 拼接后，哪个位置预测第一个 response token？在因果模型中是 prompt 最后一个位置。若用 token 自己所在位置的 logits 计算自己的概率，就错位了。普通路径截取 `[:, -R-1:-1]`；remove-padding 路径先左移 labels，再还原并截取 response。见 [dp_actor._forward_micro_batch](../verl/workers/actor/dp_actor.py)。

**口述：** 概率来自前一个位置的 logits；shape 里 R 只数 response，prompt 不在 mask 里。

<a id="sampling-3"></a>

## 采样与分组：一题 16 答，靠 uid 认亲

| 环节 | 本版本怎么做 | 要记住的判断 |
|---|---|---|
| 取题 | `train_batch_size=16` 道题一步 | 题是 parquet 的一行，prompt 已套聊天模板、左填充到 2048 |
| 生成 | vLLM 对每题 `rollout.n=16` 个回答，温度 1.0、top_p 1.0，最长 4096 token | 温度是采样分布的一部分，重算 logprob 时也要除同一个温度 |
| 认亲 | 每题分配一个 `uid`，`repeat(16, interleave=True)` 把题行复制 16 份与回答对齐 | `uid` 是本批的组身份，`task_id` 是稳定的数据身份；跨 policy、跨批不能混组 |
| 重排 | `_balance_batch` 总是执行，按 token 数把 256 行重排到各卡 | 相邻 16 行不是一道题；GT、回答、mask、reward 要和 uid 一起搬 |
| 停止 | 返回的 `DataProto` 只有 token 序列，`get_eos_mask` 含首个 EOS，无 EOS 全 1 | mask 全 1 不证明自然停止，截断也是全 1；停止原因要另从引擎记 |

Rollout 是由某个 checkpoint 按给定参数生成新回答；logprob 是模型对 token 给出的对数概率；advantage 是同题组内的相对训练信号；mask 指出哪些位置有效。这些定义都要在项目字段和函数里找到对应值：生成和认亲在 [ray_trainer.py](../verl/trainer/ppo/ray_trainer.py) 的 `fit()`，生成器在 [vllm_rollout_spmd.py](../verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py)。

**口述：** 一题多答是 GRPO 的前提；同题回答靠 uid 认亲，不靠相邻位置，因为后面会按 token 数重排；mask 全 1 不等于正常停止。

<a id="advantage-3"></a>

## 从 reward 到 advantage：同题相比，再铺到每个 token

奖励管理器把标量 reward 写在回答的最后一个有效 response token 上，其余位置是 0。GRPO 的 outcome advantage 分四步：

1. 每条回答的 token reward 求和——就是那个标量。
2. 按 uid 分组，组内求均值和**样本标准差**（`torch.std`，分母 N − 1）。
3. `(score − mean) / (std + 1e-6)`。
4. 广播到该回答所有位置，再乘 mask。

课程的 4 个固定候选：两份正确实现得 1.1，两份格式合格但测试失败的得 0.1。排成 `[0.1, 0.1, 1.1, 1.1]`：均值 0.6，样本标准差 0.57735，advantage `[−0.86603, −0.86603, +0.86603, +0.86603]`。换成总体标准差数字就不对，不能沿用。

三个边界：

- 全同分组：16 个回答全对或全错，std = 0，advantage 全 0，这题这一步没有相对信号；KL 和熵项仍可能有梯度。
- 单例组：组里只有 1 个回答时，源码用 mean 0、std 1，和多候选全同分的处理不同。
- 判题故障：原版管理器超时或异常回填 0.0，`[0.1, 0.1, 0.1, 0.0]` 的均值 0.075、样本标准差 0.05，advantage `[+0.5, +0.5, +0.5, −1.5]`——三条仍然错误的回答得到正信号。补强版 `prime_plus` 先重试，仍失败就把整组分数设成同一个值，advantage 全 0（[补强项 6](examples/source-supplements/README.md)）。

模型生成中的格式失败是策略失败，正常进组；判题故障按预先登记的规则另处置，不能结果出来后删掉失败来抬高数字。

**口述：** GRPO 不要 critic，用同题其他回答当 baseline；reward 先放末 token，求和后按 uid 分组标准化，再广播到整个回答；全同分组没有相对信号，判题故障记 0.0 会把错解推成正信号。

<a id="gradient-bridge"></a>

## 从 logits 到更新：一条可核对的链路

先看[主框架的公式与求导边界](RLVR工业流程.md#gradient-bridge)，再运行：

```bash
$PY -B RLVR学习资料/examples/maximum-subarray/mechanism_lab.py --run mechanism-01
```

这是两标量 logits 的实际 toy SGD，使用解析导数和中心有限差分；没有 torch、tokenizer 或 LLM。报告在 `practice_runs/mechanism-01/attempt-xxx/mechanism_report.json`。它补齐「数字为何产生这个方向」的机制，不充当本项目真实 autograd 或 GPU 更新的证据。

完整示范：logits 为 `[0, ln 3]`、T=1、选第二个 token，概率 0.75，old logprob 固定为 ln(0.75)，A=0.5。ratio=1，loss=−0.5；对两个 logit 的解析梯度为 `[+0.125, −0.125]`，与有限差分核对。学习率 0.1 的一步 SGD 降低第一个、提高第二个 logit，选中 token 的概率上升。

| 量 | 在本次更新中怎样得到 | 是否对它求当前策略梯度 |
|---|---|---|
| old logprob | rollout 后、update 前用当前 actor 重算，`compute_log_prob` 用 `eval()` + `no_grad`，不用 vLLM 的值 | 固定基准；下批重新得到 |
| ref logprob | Reference Policy 计算，`no_grad`；本版本 ref 固定 CPU offload + bf16 | 固定约束基准，不是 Reference Code |
| advantage | 对 score 求和、按 uid 分组、样本标准差归一化，`no_grad` | 固定权重，含 epsilon 与 mask |
| new logprob | 当前 actor forward，取已生成 token 的概率 | 梯度经模型计算图回到参数 |

new 和 old 数值相同不代表无梯度；如果让 old 也连接到同一 new 计算图，差值恒 0，梯度被错误抵消。离散 reward 进入固定权重，实际求导的是模型概率，而非执行程序或测试整数答案。

一步的 loss 写出来：

```text
ratio = exp(new − old)                        # 逐 token
pg    = max( −A·ratio , −A·clip(ratio, 0.8, 1.2) )
loss  = mean_over_tokens(pg) − 0.001·entropy + 0.001·low_var_kl(new, ref)
```

然后 `loss.backward()`，`clip_grad_norm_(1.0)`，AdamW 以 lr 5e-7 走一步。`main_grpo.sh` 开着 `use_kl_loss`，所以 KL 在 actor loss 里，`token_level_rewards` 直接取 `token_level_scores`；关掉它时 trainer 的 `apply_kl_penalty()` 会先从 reward 扣 KL 再算 advantage，组内信号和长度影响都不一样。low_var_kl 是 `exp(ref − new) − (ref − new) − 1` 再 clamp，不是精确全词表 KL；`full` 在本版本未实现。

改变一个条件：把 A 改为负，检查梯度方向；或把 T 改为 2，比较选中概率约 0.633975。当前源码重算也除以 temperature；不能把 T=0 代入 softmax 训练公式，greedy 是生成器的另一分支。若另外开启 top-p/top-k 或 processor，还需核对重算概率与真实采样分布是否一致。

**clip 不是梯度范数裁剪：** A>0 且 ratio 高于上界，或 A<0 且低于下界时，所选 PG 分支成为常数、梯度为 0；相反方向仍需纠正。共享参数、其他 token、KL/entropy 仍能改变概率。报告给正负 A 同 ratio 的对照；独立练习再检查另一侧边界。

**长度与 mask：** toy 里两回答逐 token loss 为 +1/−1，长度为 2/6，同一 micro 内 masked_mean=−0.5；先逐回答平均，或拆成两个等权 micro 累积，则为 0。本源码先在每个 micro 内按 token 平均，再按固定累积或动态行数比例缩放。不能只说「每条回答权重相同」或「改 micro 只影响显存」。EOS 示范包含首个 EOS；无 EOS 全 1，pad 与 EOS 不同且缺真实长度时也可能全 1。要结合停止信息，不能由 mask 宣称正常完成。

**口述：** 固定 old、ref、advantage，只对 new 求导；PPO 的 ratio 和 clip 是为了复用旧样本时不走太远；KL 项把模型拴在参考模型附近；平均方式决定长回答和短回答谁的权重大。

<a id="runtime-facts-3"></a>

## 本版本的运行态：机制上会怎样，这套配置实际怎样

下面几条是 2026-10-13 读源码得到的推断（基线 `443f8da`），没有实跑；它们解释了为什么真实日志里有些指标会「恒零」：

| 事实 | 来源 | 后果 |
|---|---|---|
| mini-batch 的单位是题数：worker 把 `ppo_mini_batch_size` 乘 `rollout.n` 再除以卡数 | [fsdp_workers.py](../verl/workers/fsdp_workers.py) | `main_grpo.sh` 按回答数写 256，worker 当题数再乘 16，于是每步每卡只有一个 mini-batch、一次 optimizer step |
| 单 mini-batch 且 `ppo_epochs=1` 时，更新用的权重就是算 old 的权重 | [dp_actor.update_policy](../verl/workers/actor/dp_actor.py) | ratio 恒 1，`actor/pg_clipfrac` 和 `actor/ppo_kl` 恒 0，clip 在这套配置下不起作用；要让它起作用，把 mini-batch 改小（[05 第四环](05_实跑推演_Qwen3-8B.md#stage-config)用 4 题） |
| 梯度累积按 mini / micro = 64 算，但每卡只有 4 个 micro | 同上 | 梯度被多除 16 倍，`grad_clip=1.0` 实际放宽 16 倍，日志里 `actor/grad_norm` 偏小 16 倍；AdamW 基本抵消常数缩放 |
| 本 fork 第一步的学习率是 0 | [torch_functional.py](../verl/utils/torch_functional.py) 的 warmup lambda | 只跑 1 步参数不变；`actor/lr` 记的是下一步的值 |
| 实际只执行 `total_training_steps − 1` 次更新 | [ray_trainer.py](../verl/trainer/ppo/ray_trainer.py) 的计数 | 算预算时少一步 |

verl v0.4.1 已经改掉了其中两条：断言 `train_batch_size >= ppo_mini_batch_size`，第一步 lr 不再是 0，也没有 1/16 的梯度缩放；[05](05_实跑推演_Qwen3-8B.md#stage-curves) 的推演按 v0.4.1 写，所以它的 run-01 只有「两个恒零指标」这一条。面试里讲这些要先说是哪个版本。

**口述：** 读配置要算到运行态：这套脚本每步只更新一次，ratio 恒 1、clip 不触发；要让 PPO 的那套机制真的起作用，mini-batch 得小于一步的题数。

<a id="tools-3"></a>

## 动手前：两套工具

| 工具 | 用途 | 输出在哪 | 能不能改 |
|---|---|---|---|
| `run_demo.py score / advantage` | 固定示范：LeetCode 53 的 6 个固定候选判分，4 个算法候选成组算 advantage | `examples/maximum-subarray/outputs/report.json`，不入库，随时重跑 | 只看不改 |
| `mechanism_lab.py --run mechanism-01` | 两标量 logits 的梯度 toy：解析导数、有限差分、一步 SGD、温度、正负 A 的 clip 对照、micro 平均、EOS | `practice_runs/mechanism-01/attempt-NNN/mechanism_report.json` | 换 logits、选中 token、A 做自己的变式 |

- 命令都在仓库根目录执行，`$PY` 的设置见[资料首页](README.md#run-setup)。`run_demo.py` 的子命令会自动补齐依赖，也可一次 `all`。
- 示范用的是标准库、CPU 和已检查的固定候选，复用源码的 parser 和 `_compute_score()`，只把执行入口换成受控的本地 subprocess。它演示机制，没有从 checkpoint 采样：固定候选的分数、pass@k 和教学 token 长度不是模型 rollout、模型难度或能力提升。它也不是能安全执行任意新候选的沙箱。
- 三项作业的文字说明都写进 `practice_runs/mechanism-01/notes.md`，一个文件就够；3.1 和 3.2 共用同一份 `report.json`。

<a id="assignment-3-1"></a>

## 作业 3.1：完整示范与源码对齐

**目的：** 沿同一个最大子数组和 Task，看清候选判定怎样接到训练信号。

在仓库根目录执行：

```bash
$PY -B RLVR学习资料/examples/maximum-subarray/run_demo.py score
$PY -B RLVR学习资料/examples/maximum-subarray/run_demo.py advantage
```

输入、Tests、包装后的 Task 与小池分别落在同目录 `outputs/raw_task.json`、`cases.json`、`prepared_task.json`、`data_pool.jsonl`，结果集中在 `outputs/report.json`，都来自 LeetCode 53 这一道题。JSONL 用于教学读回和 Code-R1 schema 示范；真实 trainer 仍需 Parquet 转换、正确数据路径与运行环境，不能直接把这个 JSONL 传给 `main_grpo.sh`。

**完整示范：**

1. 在报告中找到两份正确实现 `reference`、`correct_prefix` 与错误实现 `wrong_empty`、`wrong_sum` 的判定，再沿 `prepared_task.json` 找到同一组 Tests。全负数组 `[-3,-2,-5]` 的正确答案为 −2，能检出「允许空子数组而返回 0」；局部最优子数组与整体和不同的输入能检出「直接返回总和」。先看测试实际执行和通过状态，再解释 reward，不能仅按 `reward>0` 计通过。
2. 正常组中两份正确实现得 1.1、两份格式合格但测试失败的实现得 0.1。把四行重排成 `[0.1, 0.1, 1.1, 1.1]`，均值为 0.6，样本标准差约 0.57735；忽略很小的 epsilon 后，advantage 为 `[−0.86603, −0.86603, +0.86603, +0.86603]`。候选顺序以实际报告为准。
3. 用教学长度 `[4, 6, 3, 5]`、最大长度 6 画出 `(4, 6)` 的 token reward 与 response mask；这是独立手算设定，运行报告的长度以报告为准，均不是 tokenizer 测量。Prime 把标量放最后一个有效 response token，GRPO 求和、按 uid 归一化、广播。已正确构造 mask 时 padding 贡献 0；真实 EOS/pad/截断边界按[采样与分组](#sampling-3)检查，prompt 不在 response mask 中。
4. 在 `fit()` 对齐本版本真实实现时序：生成 → uid 与重复对齐 → 均衡重排 → old logprob 重算 → 按需计算 ref logprob → reward/KL 分支 → advantage → actor update。
5. 用一个 token 示范 clip：`old_logprob=-2`、`new_logprob=-1.8`、`cliprange=0.2`，概率比为 `exp(new − old)=1.22140`，裁剪值为 1.2。源码取 `max(−A·r, −A·clip(r))`；A=1 时取 −1.2，A=−1 时取 +1.22140。不能直接相除 logprob，也不能以为正负 advantage 的裁剪选择相同。

**交付与验收：** 将报告路径、候选判定、标量/末 token/mask 图与源码时序写入同一份 `notes.md`。如果回答包含推理与答案，当前有效 response token 都可能承接 outcome advantage；不能因 Tests 只检查最终代码就声称只训练代码 token。报告中 `bad_format` 得 −1.1，`early_exit` 虽得 1.1，却 `audit_completed_case_count=0`、`correctness_pass=false`；它们不能作为正确性通过，后者直接说明高 reward 不保证完成 Tests。手算或函数单独执行成功不称为已完成 RL 参数更新。

<a id="assignment-3-2"></a>

## 作业 3.2：infra 回填怎样改变整个组

**目的：** 只改变一次基础设施失败的处理条件，检查错误 reward 怎样影响同组其他候选。

先设四个格式合格候选都因逻辑错误未通过 Tests，raw reward 为 `[0.1, 0.1, 0.1, 0.1]`，组内 outcome advantage 全为 0。保持 Task、Tests、代码与 policy 不变，只将其中一次执行改为 infra error，并假设异常处理将它回填为 0：

```text
原始逻辑失败组     [0.1, 0.1, 0.1, 0.1]
一次 infra 回填后  [0.1, 0.1, 0.1, 0.0]
```

这是一项故障处理的数值模拟，不能说已在真实后端注入并测得故障。核对 `parallel_compute_score_async()` 的异常回填，再计算：均值 0.075、样本标准差 0.05，忽略 epsilon 的 advantage 是 `[+0.5, +0.5, +0.5, −1.5]`。

**条件变化后的检查：** 为什么三个仍然测试失败的候选现在得到正 advantage？为什么不能解释成模型突然会解题？如果恢复原判定，哪些值应恢复？再写出你在真实实验前采用的 infra 政策，例如整组标记无效并重新采样，或停止该批次排查；说明代价与对训练分布的影响。源码回填行为与你设计的可靠工程策略分别记录，不能把当前源码写成已有完整故障治理。对照补强版 `prime_plus`：先有限次重试，仍失败就把整组分数设成同一个值，组内 advantage 全为 0；[05 的 your-01](examples/training/runs/qwen3-8b-grpo-yourplan-20261018/notes/incident_report.md) 推演了它在一次周期性判题故障里怎样把损失限制在作废的组内。

同时核对两个边界：同组全同 reward 缺少组内相对 outcome 信号，不意味着任务永远无用或所有其他梯度项为零；单例组在本版本源码使用均值 0、标准差 1，与多候选全同分组的处理不同。写明组大小、epsilon 与样本标准差。

**交付与验收：** 将一条件对照、计算与处理政策加入同一份 `notes.md`。能讲清「infra 状态 → 回填 reward → 全组均值/标准差 → 每条 advantage」的因果链。模型生成中的格式失败是策略失败，应进入采样分母；infra 按预先登记的规则另处置，不能结果出来后删去失败以抬高 pass@k。

<a id="assignment-3-3"></a>

## 作业 3.3：独立检查分组、mask 与策略损失

**目的：** 在变化条件下独立定位错误，检查是否只会复述示范答案。

独立完成下列任务，不按 3.1 的步骤抄结果；保留自己的计算与证据，再用源码和演示报告自检。

- 将最大子数组和组与另一个 Task 组混排，并改变部分回答长度。设计会暴露「按相邻四行分组」的检查，给出每条候选恢复到正确 GT、uid、mask 和 policy 版本所需的信息；证明重排不改变同一候选的组内结果。
- 对一个有效 token 使用 `old_logprob=-2`、`new_logprob=-2.3`、`cliprange=0.2`，分别选正负 advantage，独立算概率比、裁剪值与源码最终使用的 policy loss 项。加入 padding 后说明哪些位置参与平均。
- 自己提出一个合法的最大子数组和输入与一段新的错误实现，使失败条件与 `wrong_empty`、`wrong_sum` 有区别。先推导正确答案，再说明怎样用 Tests 检出它，以及为什么正确答案仍需核对 scorer；不要把给出的反例原样作为独立发现。未知代码先纸面推演，实际执行须使用隔离后端，不能直接接入受控演示。
- 比较 `[0, 0, 1, 1]` 与 3.1 的归一化结果，解释近似相同的条件，以及为什么这不能推出项目 reward 总是二值。指出格式失败、未完整执行 Tests 却得高 reward 等状态怎样分别记录。

另交自己的机制变式：换一组 logits/选中 token/A，手算概率、冻结量和梯度方向，再用[机制函数](examples/maximum-subarray/mechanism_lab.py)自检；比较正负 A 在高/低 ratio 的四种 clip 情形。写出两种长度分配与 micro 分组的 loss 差异，以及 EOS/no-EOS 的 mask。评分与参考推导见[机制答案](examples/maximum-subarray/ANSWERS.md#mechanism-rubric)，不要将现成报告当独立答案。

**交付与验收：** 仍使用同一份 `notes.md`。独立找到至少一个能区分分组、mask、clip 或判定错误的条件，以字段、计算或反例说明影响。源码定位写「文件 + 函数」；注释、配置与模拟输出不作为实际推理或梯度证据。

<a id="oral-3"></a>

## 60 秒口述

一步训练取 16 道题，vLLM 给每题写 16 个回答，同一道题的回答共享一个 uid；回答会按 token 数重排，所以认亲只能靠 uid。判分器给每个回答一个标量，放在它最后一个有效 token 上；同 uid 的 16 个分数减均值、除以样本标准差，就是 advantage，再铺到这个回答的每个 token——4 个候选 `[0.1, 0.1, 1.1, 1.1]` 得到 ±0.866，全同分的组得到全 0，判题故障记成 0.0 会把三条错解推成 +0.5。更新时固定 old、ref、advantage 三个常量，只对当前模型的 new logprob 求导：ratio = exp(new − old) 过 0.2 的 clip，加 0.001 的 KL 和熵项，反传、裁梯度、AdamW 走一步。判题程序不可微没关系，梯度只经过「固定 advantage × 当前概率」这条路。这套脚本每步只更新一次，ratio 恒 1、clip 不触发，要让它起作用得把 mini-batch 改小。

<a id="interview-3"></a>

<details>
<summary>面试深挖：概率、梯度、KL、mask 与分组</summary>

回答按四句组织：机制是什么、本版本怎样实现、一个反例、证据能证明到哪里。

| 追问 | 回答要点/能区分的反例 | 本版本来源 |
|---|---|---|
| loss 等于 0 就无更新吗？ | 数值不等于导数。old 固定时 ratio=1 仍有 −A 的 logprob 导数；组内 loss 数值相消也不保证不同 token 的参数梯度相消 | [dp_actor](../verl/workers/actor/dp_actor.py)，[policy loss](../verl/trainer/ppo/core_algos.py) |
| 为什么用概率比？为什么 ref 不是 old？ | 旧样本上 ratio 表达当前概率相对采样/更新前的变化，构造 token 级 surrogate 与 clip；不等于全部序列分布被精确重加权。old 随批次重算，ref 用于约束偏移 | [compute_policy_loss](../verl/trainer/ppo/core_algos.py)，[fit](../verl/trainer/ppo/ray_trainer.py) |
| GRPO 为什么可以不用 critic？ | 用同题组分数构造相对 baseline；省 critic 不保证低方差，G 小、全同分或错误 reward 会影响信号。本版本 singleton 单独处理 | [compute_grpo_outcome_advantage](../verl/trainer/ppo/core_algos.py) |
| PPO clip 保证概率比始终在区间吗？ | 不保证；它裁目标的有利方向，PG 梯度可 0，但共享参数/其他项可改概率。参数梯度范数裁剪在 `_optimizer_step` 是另一步 | [compute_policy_loss](../verl/trainer/ppo/core_algos.py)，[actor](../verl/workers/actor/dp_actor.py) |
| 这套脚本里 clip 真的起作用吗？ | 不起：mini-batch 按题数再乘 16 后每步只有一个 mini-batch，更新用的权重就是算 old 的权重，ratio 恒 1、clipfrac 恒 0。要起作用就把 mini-batch 改小；05 的 run-02 用 4 题，clipfrac 2–6% | [本版本的运行态](#runtime-facts-3)，[05 第五环](05_实跑推演_Qwen3-8B.md#stage-curves) |
| actor KL 与 reward KL 等价吗？ | 前者用 new/ref 加 loss，outcome advantage 来自 raw score；后者用 old/ref 逐 token 扣 reward，再求和并组归一化，改变组信号和长度影响 | [apply_kl_penalty/fit](../verl/trainer/ppo/ray_trainer.py)，[kl_penalty](../verl/trainer/ppo/core_algos.py) |
| 单 token logp − logq 为负是 KL 错吗？ | 有符号单点可负；在适当分布下期望才对应 KL。low_var 用 exp(ref − new) − (ref − new) − 1 并 clamp，不是精确全词表 KL；有限旧策略样本是 surrogate | [kl_penalty](../verl/trainer/ppo/core_algos.py) |
| 全同 reward 完全没梯度吗？ | 多候选组 outcome advantage 为 0，但 KL/entropy 仍可能有梯度；singleton 在本版本另用 mean 0 / std 1 | [GRPO advantage](../verl/trainer/ppo/core_algos.py)，[actor loss](../verl/workers/actor/dp_actor.py) |
| 相邻 G 行就是一题吗？ | token 均衡重排后不是；按 uid，GT/response/mask/reward 必须一起重排。task_id 是稳定数据身份，uid 是本批组身份，跨 policy/批不能混组 | [fit/_balance_batch](../verl/trainer/ppo/ray_trainer.py)，[DataProto.reorder](../verl/protocol.py) |
| 只检查代码就只训练代码 token 吗？ | 当前 outcome advantage 广播到有效 response，包括推理。它是粗粒度 credit assignment，不是逐步推理正确性监督 | [Prime](../verl/workers/reward_manager/prime.py)，[GRPO](../verl/trainer/ppo/core_algos.py) |
| 全 1 mask 证明正常停止吗？ | 无 EOS 也全 1，可能长度截断。当前 rollout 未完整保留 stop reason，需要另记引擎信息；pad/EOS/真实长度一起查 | [get_eos_mask](../verl/utils/torch_functional.py)，[vLLM rollout](../verl/workers/rollout/vllm_rollout/vllm_rollout.py) |
| 熵系数和回答长度有什么关系？ | 熵项鼓励分布不塌；Base 起点的熵从 1.1 掉下来大半是格式锁定，不是崩塌；判断多样性看采样 pass@8，不看熵本身 | [05 第五环](05_实跑推演_Qwen3-8B.md#stage-curves) |

能回答「本版本怎么做」和「为什么这个反例改变结论」，比只背 GRPO 公式更重要。

概率核对还需固定 token/prefix、温度、processor、policy 版本和模型模式。当前 old 计算用 eval，update 用 train；有随机层的模型即使参数相同也未必数值相同，须结合实际模型配置验证，不把 toy 的 ratio=1 当无条件运行事实。`full` KL 在本版本未实现，不能据配置名字声称已计算完整词表 KL。

</details>

<a id="rollout-real-3"></a>

## 有资源后：真实 rollout 与当前 policy 难度

真实采样可接在机制作业之后。**模型选择、Linux 执行后端和采样预算须在生成未知候选之前明确，不能等第四阶段才决定。** 先确认 checkpoint/tokenizer、模型许可、显存与推理 backend；验收隔离执行环境；固定每题采样数、温度、seed、token 上限、超时、并发与总预算。CPU 受控演示已能完成机制主线，不要求先购买资源或完成训练。没有 GPU 时，[05 的第一、四环](05_实跑推演_Qwen3-8B.md#stage-machine)和[显存帐](examples/training/显存帐_Qwen3-8B.md)把这些决定推演了一遍。

资源具备后，从第二阶段小池取少量诊断 Task，例如每题 4 个候选，另记主脚本原配置。保存新回答、提取代码、task_id、uid、checkpoint、判定/runtime 版本、状态与 reward；停止原因需另从引擎记录，当前 DataProto 返回字段未完整保留，不能由 mask 反推。历史 `meta.response` 不能代替本轮生成。具体数据与 batch 准备见 [Linux 小配置](examples/training/README.md#batch-contract)。

按题报告通过数/全部生成候选数、格式失败、逻辑/运行失败、infra 与组内 reward 方差。4 个生成候选中 2 个正确，成功率为 1/2；格式错误仍占分母，0.1 的格式奖励不是正确性通过。infra 按预先锁定政策报告，包含重试与成本。`Medium` 是 Task 的原始 metadata；当前成功率取决于 checkpoint、采样参数和判定版本，小组估计有不确定性，换 checkpoint/温度后不是同一难度结果。

对照[预处理器 coder1.py](../examples/data_preprocess/coder1.py)：它没有执行完整 policy rollout 难度过滤。补强项 7 的 `coder1_plus_selection.py difficulty` 按真实采样通过率筛题，通过率必须来自固定 checkpoint 的采样，换 checkpoint 要重测；[05 的 your-02](examples/training/runs/qwen3-8b-grpo-yourplan-20261018/notes/your-02_plan.md) 推演了一次筛题，并发现模型变强后全对题又会回来，筛题是周期性的事。评分不可靠或 infra 占主导时，返回第二阶段排查，不用这些数字解释模型能力。

<a id="source-reading-3"></a>

## 按需读源码

需要拿源码作证时按这个顺序读，第一遍不必通读：

| 顺序 | 直接阅读的材料 | 阅读时要回答的问题 |
|---|---|---|
| 1 | [run_demo.py](examples/maximum-subarray/run_demo.py)、[coder1 scorer](../verl/utils/reward_score/coder1/__init__.py) 的 `_compute_score()` | 固定候选怎样接原判分？高 reward 是否保证完整执行了 Tests？ |
| 2 | [PrimeRewardManager](../verl/workers/reward_manager/prime.py) 的 `__call__()`、`parallel_compute_score_async()` | reward 张量的 shape、有效长度与赋值位置是什么？异常回填怎样改变组统计？ |
| 3 | [core_algos.py](../verl/trainer/ppo/core_algos.py) 的 `compute_grpo_outcome_advantage()`、`compute_policy_loss()`、`kl_penalty()` | 怎样求和、分组、求样本标准差、广播并加 mask？clip 怎样用 old/new logprob？low_var_kl 是什么式子？ |
| 4 | [main_ppo.py](../verl/trainer/main_ppo.py)、[ray_trainer.py](../verl/trainer/ppo/ray_trainer.py) 的 `fit()` | 生成、uid、重复对齐、均衡重排、old/ref logprob、reward、advantage 与 update 如何衔接？ |
| 5 | [dp_actor.py](../verl/workers/actor/dp_actor.py) 的 `_forward_micro_batch()`、`compute_log_prob()`、`update_policy()` | 温度在哪里除？位置怎样截？micro 内怎样平均、累积怎样缩放？ |
| 6 | [main_grpo.sh](../main_grpo.sh)、[ppo_trainer.yaml](../verl/trainer/config/ppo_trainer.yaml)、[fsdp_workers.py](../verl/workers/fsdp_workers.py) | 若做真实 rollout，模型/backend/每题采样数、长度、batch 与 KL 从哪里配置？mini-batch 归一化怎样算？ |

行号以本仓库固定基线 `443f8da` 为准，[CLAUDE.md](../CLAUDE.md) 第 6 节有逐函数的行号。

<a id="record-3"></a>

## 本阶段完成记录

学完这一篇，应该能指出每条训练箭头的实际函数和字段：谁生成、谁认亲、谁判分、谁标准化、谁求导。

| 项目 | 完成证据 | 证据不足时的标记 |
|---|---|---|
| CPU 机制主线 | 固定候选报告、末 token/mask、样本标准差与源码时序 | 仅阅读/手算；演示待执行 |
| 条件变化与独立迁移 | infra 因果练习、重排/clip 与自己提出的反例 | 区分纸面推演与实际执行 |
| 机制变式 | mechanism-01 的报告加自己的一组 logits/A/clip 变式 | 只跑了示范，没有变式 |
| 可选真实 rollout 与难度 | checkpoint、实际新生成、判定版本与逐题分母 | 模型/后端/预算待明确；真实采样待完成 |

CPU 演示、真实采样、训练更新按证据分别记录，不把固定候选的通过率写成模型性能。

| 日期 | 记录 |
|---|---|
| 2026-10-19 | 按 01、02 的结构重排：开头新增总览 `overview-3` 和文字版导读 `guide-3`；概念五节为前置自测、采样与分组、reward 到 advantage、从 logits 到更新、本版本的运行态（mini-batch 语义、ratio 恒 1、梯度缩放、首步 lr=0，并注明 v0.4.1 的差别）；三项作业、面试深挖、有资源后、按需读源码、完成记录按统一顺序排列；外部链接的锚点 `prerequisites-3`、`gradient-bridge`、`assignment-3-*`、`interview-3` 不变 |

---

**继续阅读：** [返回资料首页](README.md) · [工业主框架](RLVR工业流程.md) · [上一阶段：02 Verifier 与任务池](02_Verifier与任务池验收.md) · [下一阶段：04 训练与评测](04_真实训练与独立评测.md)
