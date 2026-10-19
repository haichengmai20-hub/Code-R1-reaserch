# 05 · 实跑推演：Qwen3-8B-Base 在 Code-R1 流水线上的两轮 GRPO

**资料导航：** [资料首页](README.md) · [00 全程走读](00_一条记录从原始Task到训练.md) · [01](01_数据理解与GT构建.md) · [02](02_Verifier与任务池验收.md) · [03](03_Rollout与GRPO机制.md) · [04](04_真实训练与独立评测.md) · **05 实跑推演（当前）** · [路线评估](路线评估_仅学习与实跑.md)

**这一篇解决什么：** 本机没有 GPU，但面试会问「你怎么从零把一次训练跑起来、曲线怎么看、结果怎么评、问题怎么归因」。这一篇用一次完整的推演补上：以 Qwen3-8B-Base（预训练模型，没经过 SFT 或 RL，R1-Zero 式的起点；架构与 Qwen3-8B 相同）为模型，把机器、数据、框架搭建、参数配置、训练曲线解读、结果测评、归因与优化七个环节全部走一遍，每一环先由你做决定，再对照推演里的决定和事后评价。运行产物在 [examples/training/runs/qwen3-8b-grpo-20260928/](examples/training/runs/qwen3-8b-grpo-20260928/README.md)，按真实运行会留下的文件做：启动脚本、展开后的配置、控制台日志、每步指标、wandb 目录、checkpoint 清单、评测明细、曲线图。

**证据声明：** 这是推演。配置键、断言、日志字段、打印语句、文件布局、数量级都对齐 verl v0.4.1 和本仓库的判分代码，第 1 步的 256 条判分日志是用真实判分代码执行真实参考解和变异体得到的；但没有真的租卡训练，曲线由一个显式的参数模型生成（[scenario_qwen3_8b.py](examples/training/scenario_qwen3_8b.py)），哪些是真实输出、哪些是合成的，逐项列在 [PROVENANCE.md](examples/training/runs/qwen3-8b-grpo-20260928/PROVENANCE.md)。面试时的说法是「我做过一次完整的实跑推演」，并能讲清每一环为什么这样定。

**怎么用：** 每一环都有「你来决定」。先合上推演的答案，把你的决定写进自己的决策日志（模板见[作业](#homework-5)），再展开对照。想先自己决策，就跳过下面的总览和导读，直接从[一、机器](#stage-machine)开始读；总览里的数字是现场数据，导读里有全部结论，做完七环再展开。看懂一环的标准是：能说出决定、依据、反例，以及事后哪里错了。

<a id="overview-5"></a>

## 总览：两轮训练一页看完

| | run-01 | run-02 |
|---|---|---|
| 时间 | 2026-09-28 08:30 起，76.1 小时 | 2026-10-03 07:02 起，100.6 小时 |
| 机器 | 4×A800 80GB 单机，128 逻辑核，512 GiB 内存，根盘 300 GB，数据盘 3.5 TB | 同 |
| 模型 | Qwen/Qwen3-8B-Base：预训练模型，没经过 SFT 或 RL，`<think>…</think><answer>…</answer>` 的格式靠系统提示和 reward 从头学 | 同，从头训，不接 run-01 |
| 框架 | verl v0.4.1 + 移植的 coder1 / coder1_plus 判分 + prime_plus 管理器 | 同 |
| 数据 | code-r1-plus-2k-leetcode2k 去污染版，1889 题 | 用 run-01 第 192 步 checkpoint 采样筛题后 1536 题 |
| 关键配置 | 16 题 × 16 回答，回答上限 4096，lr 5e-7，KL 0.001，mini-batch 16 题，v0.4.1 原版 prime 管理器，checkpoint 放根盘 | 回答上限 8192，mini-batch 4 题，prime_plus，checkpoint 放数据盘，其余同 |
| 训练 reward | −0.409 → 0.559 | −0.174 → 0.709 |
| 验证 val-core（贪心，受训练回答上限影响） | -0.224 → 0.595 | -0.096 → 0.656 |
| dev 贪心 pass@1 | 基线 0.380 → 0.615 | 0.635 |
| held-out 贪心 pass@1 | 0.134 → 0.246（净胜 16 题） | 0.317（净胜 26 题；对 run-01 净胜 10 题） |
| 异常 | 有一次，见第五环 | 无；prime_plus 作废了 2 组 |
| 费用（¥10/卡时） | ¥3,043（$423） | ¥4,023（$559） |

两轮加评测和环境，约 ¥7,500（3,043 + 4,023 + 评测与环境约 400）：Code-R1 的配方（n=16、320 步）放到个人能租到的 4×A800 上，Base 起点回答短也要 3 天多一轮、两轮 7 天多。这不在你 300 元的预算内，所以它是推演；你自己的方案（n=8、200 步，去 TACO 后两轮）训练约 ¥1,900（your-01 ¥847 + your-02 ¥1,043），见第四环和 [notes/your_decisions.md](examples/training/runs/qwen3-8b-grpo-20260928/notes/your_decisions.md)。真要花钱，仍按[路线评估](路线评估_仅学习与实跑.md#track-run)的一档单卡 smoke 做。

七环与产物的对应：

| 环 | 产物 | 对应资料 |
|---|---|---|
| 一 机器 | `env/versions.txt`、`env/nvidia-smi.txt`、`notes/decision_log.md` 第一节 | [训练准备](examples/training/README.md) |
| 二 数据 | `data/data_manifest.txt`、`data/decontamination_v2test.json` | [01](01_数据理解与GT构建.md)、[补强项 1、7](examples/source-supplements/README.md) |
| 三 框架搭建 | `notes/framework_port.md`、`env/probe_scorer.log`、`env/pip_freeze.txt` | [02](02_Verifier与任务池验收.md)、[可跑性评估](examples/training/README.md) |
| 四 参数配置 | `config/run-01.sh`、`config/run-01.resolved.yaml` | [03](03_Rollout与GRPO机制.md)、[04](04_真实训练与独立评测.md) |
| 五 曲线解读 | `logs/run-01/metrics.jsonl`、`curves/*.png`、`logs/run-01/incident_step129-135.log`、`notes/incident_report.md` | [03](03_Rollout与GRPO机制.md) |
| 六 结果测评 | `eval/protocol.md`、`eval/tables.md`、`eval/report.md` | [04](04_真实训练与独立评测.md) |
| 七 归因与优化 | `attribution/reward_decomposition.md`、`notes/decision_log.md` 第七节、`config/run-02.sh` | [主框架的故障归因](RLVR工业流程.md#layer-attribution) |

<a id="guide-5"></a>

<details>
<summary>文字版导读：七环各在想什么（含全部结论，建议做完七环决定后再展开）</summary>

**一、机器。** 先算显存再选卡：actor 的 fp32 参数、梯度、Adam 状态约每参数 16 字节，8.2B 参数就是 131 GB，4 卡分摊后每卡 33 GB；vLLM 采样时还要每卡 40 GB，0.8.5 以后训练时能让它睡眠；激活开梯度检查点。4×A800 80GB 放得下，前提是采样引擎训练时睡眠；代价是只有 2 个采样副本、算力约 H100 的 1/3，每步估 17 分钟、实际 14 分钟（Base 的回答短，生成只占每步两成）。判题要 CPU 核。磁盘最容易被忽略：一份 checkpoint 98 GB，根盘 300 GB 放两份就满。推演在这里犯了错。

**二、数据。** 用补强版的 2k LeetCode：GT 带运行前缀，121 道树和链表题才能判对。去污染两步：题号对不上 v2 test 是构造保证的，措辞重叠查出 1 对，去掉。没有按难度筛，也没有增强测试：前者让第 192 步采样时仍有 19% 的题全对或全错（198 + 155 题），后者让 dev 早早饱和、分不出两轮。

**三、框架搭建。** 本仓库的 fork 装不了 Qwen3：transformers 4.47 不认识 `qwen3`（Base 与 Qwen3-8B 同架构，一样要 ≥ 4.51），vLLM 0.6.3 和 0.7.3 也不支持。选择迁到 verl v0.4.1，把判分代码移过去，训练入口不用改。启动前四项检查：配置展开、自带的小模型 smoke、firejail 探针（没装 firejail 时分数会悄悄全变 0）、Qwen3-8B-Base 专项（聊天模板是否与 Qwen3-8B 同一套且不预填、`<think>` 标签能不能过解码、Base 会不会循环到上限）。

**四、参数配置。** 大多数照 Code-R1：16 × 16、lr 5e-7、KL 0.001 进 loss、熵 0.001。两处是有意为之又被证明错的：回答上限 4096（Base 开局 23% 的回答循环到上限被截断，后期仍有 6% 撞上限），mini-batch 16 题（每步只 1 次更新，clip 不起作用）。一处是没想到的：checkpoint 放根盘。

**五、曲线解读。** 四段：reward 从 −0.409 起步，前 64 步陡升，拆开看 85% 是格式（标签缺失 30% → 1%、循环到上限 23% → 7%）；64 到 128 步稳升，靠正确率；129 到 134 步掉到 0，是判题故障被记成 0.0，不是模型坏了，看 `rewards/min` 变 0.0、`timing_s/reward` 掉到几秒、日志里的 `No space left on device`；192 步后增速放缓，格式部分已榨干、正确率慢慢涨，全同组不到一成，不是饱和。Base 起点的三条副曲线：回答长度先降后升（1,787 → 994 → 1,517），熵从 1.10 起、前 60 步快降是格式锁定，KL 走到 0.065。两个恒为 0 的指标（`pg_clipfrac`、`ppo_kl`）有源码解释。

**六、结果测评。** 协议训练前冻结：dev 200 题选 checkpoint，held-out 142 题只在最终 checkpoint 上评一次，贪心和 3 个 seed 的采样都报，按题配对。Base 基线低（dev 0.380、held-out 0.134，两成回答格式错），run-01 在 dev 上 +23.5 个点、held-out +11.3，配对净胜 16 题、只 2 题变差，远超 seed 噪声；采样 pass@8 也涨（held-out 0.366 → 0.582），既学会了格式和停机，会做的题也多了。两轮在 dev 上几乎分不出（+2.0、净胜 4 题），held-out 和 pass@8 分得出（+7.0、净胜 10 题；0.582 → 0.695）。训练中的验证数字受回答上限影响，和离线协议不可比。

**七、归因与优化。** 五个结论各对应一处改动：格式占大头（85%：标签缺失 30% → 1%、循环到上限 23% → 7%） → 上限 8192；全同组没有梯度（第 192 步采样 8 条/题，全对 198 题、全错 155 题） → 用 checkpoint 采样筛题；故障记 0.0 → prime_plus 整组作废加 checkpoint 换盘；clip 没起作用 → mini-batch 4 题；dev 饱和、分不出两轮 → 只信 held-out 和 pass@8。run-02 的 held-out +7.0 个点、净胜 10 题、只 1 题变差，超过 3 倍标准差。还没做的：增强测试、再筛题、困难题动得少。

</details>

<a id="stage-machine"></a>

## 一、机器：先算再租

**现场。** 个人研究能租到的有 4×A800 80GB、4×A100 80GB、单张 A100，8×H100 要排队且一天就过千元；模型 Qwen3-8B-Base（预训练模型，没经过 SFT 或 RL），架构参数与 Qwen3-8B 完全相同：8.19B 参数，36 层，GQA 32/8 头，词表 151,936，原生上下文 32k；`<think>` / `</think>` 是词表里的普通 token（151667 / 151668）。

**你来决定。** 写下来再往下看：

1. 需要几张什么卡？列出每卡显存的算式。
2. 训练时 vLLM 的显存怎么办？
3. 判题需要什么？checkpoint 多大、放哪？
4. 预算多少，按什么单价？

<details>
<summary>推演里的决定与事后评价</summary>

| 项 | 决定 | 依据 | 事后 |
|---|---|---|---|
| 卡 | 4×A800 80GB | actor 每参数 16 B × 8.19B = 131 GB，4 卡各 32.8 GB；vLLM `gpu_memory_utilization=0.5` 占 40 GB，0.8.5 的 sleep mode 在训练阶段释放，不睡眠就放不下；激活开梯度检查点。8×H100 快 4 倍，但个人研究租不起也说不过去 | 对；日志 `perf/max_memory_allocated_gb` 约 56–57 GB，与算式相符；Base 回答短，生成只占每步 19%，更新占 51% |
| TP 与副本 | TP=2（张量并行：一个 vLLM 引擎跨 2 张卡），4 卡 = 2 个引擎副本 | 8B bf16 单卡放得下，TP=2 是为 KV cache 留空间 | 可以；TP=1 四个副本会更快，但每卡 KV 更少，128 条一波就排不下 |
| CPU | 128 逻辑核 | 判题并发 | 够；v0.4.1 的 prime 固定 64 个判题进程，`timing_s/reward` 约 25 秒 |
| 磁盘 | 根盘 300 GB，数据盘 3.5 TB | 数据和 rollout 放数据盘 | **错**：checkpoint 放了根盘，一份 98 GB，第 128 步写满 |
| 预算 | ¥40/小时，run-01 预算 4 天 | 每步估 17 分钟 × 320 步 | run-01 实际 76.1 小时、¥3,043（$423）：每步均 856 秒，Base 回答短，比估的快 |

</details>

**显存帐：** 不同卡数、vLLM 占比、micro batch、回答上限、梯度检查点各占多少显存、换来什么速度和质量，集中记在[显存帐_Qwen3-8B.md](examples/training/显存帐_Qwen3-8B.md)，持续维护；你自己的七环决定记在 [notes/your_decisions.md](examples/training/runs/qwen3-8b-grpo-20260928/notes/your_decisions.md)。

**口述：** 选卡先算三样：训练态每参数 16 字节、采样引擎的显存、激活；再算 CPU 核数够不够并发判题；最后算 checkpoint 大小和存放位置。推演里错在最后一项。

<a id="stage-data"></a>

## 二、数据：拿什么训、怎么证明没污染

**现场。** 本仓库有 2k LeetCode（原版和补强版两种 GT）、作者的 12k（含 TACO，需要联网和 firejail 验证）；评测集有 v2 test 200 题、LiveCodeBench、EvalPlus。

**你来决定。**

1. 用原版 GT 还是补强版 GT？为什么？
2. 要不要并 TACO？
3. 去污染查什么、怎么查？held-out 怎么定？
4. 要不要按难度筛、要不要增强测试？现在不做的话，什么时候做？

<details>
<summary>推演里的决定与事后评价</summary>

- 补强版 GT：带运行前缀，121 道树和链表题才能判对；`data_source=code_plus` 走受保护协议。
- 不并 TACO：先把一套数据跑通；TACO 的标准输入输出题要另一套比较规则。
- 去污染：题号与 v2 test 无交集是构造保证；措辞用 8 元词组、阈值 0.5 查出 1 对（`longest-increasing-subsequence`），去掉，训练集 1889 题；held-out 用 LiveCodeBench 里发布日期晚于模型数据截止的 142 题，训练前锁定。真实输出在 [data/decontamination_v2test.json](examples/training/runs/qwen3-8b-grpo-20260928/data/decontamination_v2test.json)，去污染后的 parquet 也是本机真实生成的（sha256 在 [data_manifest.txt](examples/training/runs/qwen3-8b-grpo-20260928/data/data_manifest.txt)）。
- 没筛难度、没增强测试：想先看基线分布。事后看，前者让第 192 步采样时 19% 的题（全对 198、全错 155）没有 outcome 信号，后者让 dev 早早饱和、两轮分不出。

</details>

**口述：** 数据环节三件事：GT 要能判对（前缀）、训练集和评测集要隔离（题号加措辞加日期）、测试要够强。推演只做好了前两件。

<a id="stage-framework"></a>

## 三、框架搭建：模型装不进去怎么办

**现场。** 本仓库 fork 是 verl 0.2.0.dev，`setup.py` 钉 vLLM ≤ 0.6.3，作者装 0.7.3，transformers 4.47。Qwen3（含 Qwen3-8B-Base，同一个 `qwen3` 架构）要 transformers ≥ 4.51.0（旧版报 `ValueError: … model type `qwen3` but Transformers does not recognize this architecture`），vLLM 0.8.4 起注册、官方推荐 ≥ 0.8.5。模型放在节点的 `/root/models/Qwen3-8B-Base`。

**你来决定。**

1. 三个选项：换模型、硬升级 fork、迁到新版 verl。选哪个，代价是什么？
2. 迁过去要搬哪些东西？训练入口要不要改？
3. 正式启动前做哪几项检查？每项失败时会看到什么？
4. Qwen3-8B-Base 没经过 SFT，`<think>…</think><answer>…</answer>` 这套格式它从哪学？聊天模板的 `enable_thinking` 对 Base 还有没有意义？

<details>
<summary>推演里的决定与事后评价</summary>

迁到 verl v0.4.1，移植 5 处（两个判分目录、路由的两个分支、prime_plus 与 infra_policy、数据脚本），入口不改；prime_plus 要适配 v0.4.1 的管理器协议（`reward_fn_key`、`return_dict`），细节在 [notes/framework_port.md](examples/training/runs/qwen3-8b-grpo-20260928/notes/framework_port.md)。四项检查：

| 检查 | 做法 | 失败时的样子 |
|---|---|---|
| 配置展开 | `--cfg job --resolve`，存成 `config/run-01.resolved.yaml` | 键名错会在解析阶段退出；路径含中文要加引号；v0.4.1 还会断言 `train_batch_size ≥ ppo_mini_batch_size`、开 chunked prefill 时 `max_num_batched_tokens ≥ prompt + response` |
| 框架 smoke | verl 自带 gsm8k 单卡 1 步 | Ray、FSDP、vLLM 任一起不来 |
| 判分探针 | firejail 下给四个固定候选判分，期望 1.1 / 0.1 / 1.1 / −1.1（本机真实输出 [env/probe_scorer.log](examples/training/runs/qwen3-8b-grpo-20260928/env/probe_scorer.log)；节点上用 firejail 跑同样的探针） | 没装 firejail 时原版管理器不报错，格式合格的回答全部 0.0 |
| Qwen3-8B-Base 专项 | 聊天模板按与 Qwen3-8B 同一套算（tokenizer 自带，smoke 时核对）：`enable_thinking` 只是模板开关，默认不预填 assistant 段，`<think>` 得模型自己写。Base 没有「思考模式」这个模型属性，格式只能靠系统提示加 reward 从头学；它也不会自己写空 think 块，所以不存在模型习惯和格式门打架，只要别把 `enable_thinking=False` 传进模板（那会在 prompt 里预填空 think 块，response 不再以 `<think>` 开头）。标签是普通 added token，`skip_special_tokens=True` 解码后仍在；5 条样例 900–4,096 token，最长一条循环到 4096 上限被截断 | 标签被吃掉就全是 −1.1；不写标签、循环到上限也都是 −1.1，第 1 步一半回答如此 |

第四项看到了循环到上限的风险，没处理，留给 run-01；标签缺失的比例要到第 1 步的 256 条判分才看清（30%）。

</details>

**口述：** 搭框架先问模型和框架版本能不能配上，再问判分和数据代码怎么跟过去，最后用四项检查证明「能起、能判、格式对得上」，而不是直接开训。

<a id="stage-config"></a>

## 四、参数配置：哪些照抄，哪些要想

**现场。** Code-R1 的配方：16 × 16，lr 5e-7，low_var_kl 0.001 进 loss，熵 0.001，PPO 裁剪 `actor.clip_ratio` 0.2（和第五环的截断率不是一回事），回答上限 4096，每 16 步验证、每 64 步保存。完整脚本 [config/run-01.sh](examples/training/runs/qwen3-8b-grpo-20260928/config/run-01.sh)，展开后 [config/run-01.resolved.yaml](examples/training/runs/qwen3-8b-grpo-20260928/config/run-01.resolved.yaml)。

**你来决定。** 对每个参数写「照抄 / 改 / 不确定」和一句理由：

1. 回答上限 4096 对一个会循环到上限的 Base 够吗？
2. `ppo_mini_batch_size` 填多少？v0.4.1 不允许它大于 `train_batch_size`，那每步几次更新？
3. 训练温度 1.0，验证贪心；Base 贪心容易循环，验证数字还能看吗？
4. 奖励管理器用 prime 还是 prime_plus？
5. checkpoint 存几份、放哪？

<details>
<summary>推演里的决定与事后评价</summary>

| 参数 | run-01 | 依据 | 事后 |
|---|---|---|---|
| 回答上限 | 4096 | 照 Code-R1，先看截断率 | **错**：开局 23% 循环到上限，第 128 步后仍有 6% 撞上限、不再下降 |
| mini-batch | 16 题 | v0.4.1 断言不能大于 train_batch_size，16 是上限；每卡 16 × 16 / 4 = 64 条，正好一个 mini-batch | 每步只 1 次更新，`pg_clipfrac` 和 `ppo_kl` 恒 0；梯度累积 64 / 4 = 16 个 micro 全在，`grad_norm` 起点 0.83、前 64 步降到 0.55 左右、之后 0.30–0.60，是真实量级 |
| 温度 | 训练 1.0，验证贪心 | 训练要探索；贪心可复现 | 可以；Base 贪心容易循环，第 0 步验证 47% 的贪心回答标签缺失或撞到 4096，所以验证只看趋势，结论用离线协议，贪心和采样都报 |
| lr 与 warmup | 5e-7，warmup 0 | 照 Code-R1 | 可以；v0.4.1 下第一步 lr 就是 5e-7，320 步 320 次更新（旧版 fork 才有第一步 lr=0 的坑） |
| KL、熵 | 0.001、0.001 | 照 Code-R1（v0.4.1 的熵默认是 0） | KL 走到 0.065：Base 要新学格式、停机和思考结构，策略必须离 ref 远，系数 0.001 下它只是保险；熵 1.10 → 0.32，前 60 步的快降是格式锁定，之后平稳 |
| 管理器 | prime | v0.4.1 原版 | **错**：故障记 0.0 |
| 保存 | 每 64 步、不限份数、根盘 | 全留 | **错** |

</details>

**口述：** 配置分三类：照抄的要知道它在新模型上的后果（4096 对会循环的 Base 太短），要改的要知道源码里的语义和断言（mini-batch 的单位是题数、会乘回答数、不能大于 batch），不确定的先跑小步看指标。

<a id="stage-curves"></a>

## 五、训练曲线解读：四段，两个恒零，一次事故

**现场。** [logs/run-01/metrics.jsonl](examples/training/runs/qwen3-8b-grpo-20260928/logs/run-01/metrics.jsonl) 每步一行，字段名是 verl 的；[curves/](examples/training/runs/qwen3-8b-grpo-20260928/curves/reward.png) 有 10 张图。下表是 run-01 的关键步；「截断率」是 `response_length/clip_ratio`，即回答撞到长度上限的比例：

| step | rewards/mean | rewards/min | 回答长度 | 截断率 | kl_loss | entropy | pg_clipfrac | grad_norm | timing gen / reward / step（秒） |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | −0.409 | −1.1 | 1787 | 0.229 | 0.001 | 1.103 | 0.0 | 0.830 | 223 / 28 / 916 |
| 16 | 0.053 | −1.1 | 1229 | 0.142 | 0.009 | 0.832 | 0.0 | 0.766 | 141 / 27 / 1091 |
| 32 | 0.270 | −1.1 | 994 | 0.096 | 0.017 | 0.643 | 0.0 | 0.557 | 132 / 28 / 1015 |
| 64 | 0.393 | −1.1 | 1005 | 0.068 | 0.029 | 0.441 | 0.0 | 0.541 | 129 / 26 / 1188 |
| 128 | 0.480 | −1.1 | 1189 | 0.062 | 0.049 | 0.340 | 0.0 | 0.476 | 147 / 29 / 1219 |
| 129 | 0.397 | 0.0 | 1194 | 0.066 | 0.050 | 0.359 | 0.0 | 0.411 | 147 / 24 / 839 |
| 131 | 0.091 | 0.0 | 1222 | 0.059 | 0.061 | 0.335 | 0.0 | 0.434 | 135 / 7 / 779 |
| 132 | 0.000 | 0.0 | 1184 | 0.063 | 0.059 | 0.329 | 0.0 | 0.457 | 144 / 3 / 803 |
| 135 | 0.453 | −1.1 | 1256 | 0.057 | 0.060 | 0.362 | 0.0 | 0.432 | 158 / 26 / 853 |
| 192 | 0.523 | −1.1 | 1327 | 0.058 | 0.063 | 0.319 | 0.0 | 0.435 | 169 / 29 / 1237 |
| 320 | 0.559 | −1.1 | 1517 | 0.056 | 0.065 | 0.322 | 0.0 | 0.454 | 186 / 25 / 1256 |

step 为 16 的倍数时 `timing_s/step` 多出验证的时间，64 的倍数再多出保存。验证 `val-core/code_plus/reward/mean@1`：-0.224（第 0 步：通过 48 题、格式错 94 题，即 47% 的贪心回答标签缺失或循环到 4096）→ 0.115（16）→ 0.266（32）→ 0.418（64）→ 0.495（128）→ 0.495（144，事故后持平）→ 0.545（192）→ 0.585（256）→ 0.595（320）。

verl 的指标里没有「标签缺失」这一项，推演把它记在 `scenario_internal.jsonl`（真实训练里要自己从 rollout 落盘统计）：格式失败率 53.6%（其中循环到上限 23.1%、标签缺失 30.4%）→ 第 64 步 7.9%（6.7% / 1.2%）→ 第 128 步起 7.0%（6.0% / 1.0%）不再降；格式合格者的正确率 0.286 → 0.418（64）→ 0.500（128）→ 0.584（320）。

**你来决定。** 看表不看答案：

1. 前 64 步 reward 从 −0.409 涨到 0.393，是模型变会解题了吗？用哪几列判断？
2. `pg_clipfrac` 和 `ppo_kl` 恒为 0，是 bug 还是配置的必然？
3. 第 129–134 步发生了什么？列出三条你会先看的证据。
4. 第 129 到 144 步 KL 从 0.049 跳到 0.062、第 144 步验证持平在 0.495，要不要回滚到第 128 步？
5. 192 步后增速明显放缓，是简单题饱和了吗？

<details>
<summary>推演里的解读</summary>

1. 大半不是。截断率 0.229 → 0.068、格式失败率 53.6% → 7.9%（标签缺失 30.4% → 1.2%，循环到上限 23.1% → 6.7%）、长度 1,787 → 1,005，说明模型先学会写标签、再学会在 4096 内停下；拆分表显示 0.802 的增量里 0.678 是格式（85%），格式合格者的正确率只从 0.286 到 0.418。
2. 配置的必然：单 mini-batch 且 ppo_epochs=1，更新用的权重就是算 old_log_prob 的权重，ratio 恒 1。要让 clip 起作用，把 mini-batch 改小。`grad_norm` 没有被缩放，是真实量级：起点 0.83，前 64 步降到 0.55 左右，之后 0.30–0.60；run-02 的 mini-batch 小 4 倍、每次更新只看 64 条，起点 0.80、终点 0.24，量级相近。
3. `rewards/min` 从 −1.1 变 0.0（0.0 只有管理器兜底才会出现）；`timing_s/reward` 从 25 秒掉到 2–7 秒（判题根本没跑）；控制台 1,139 行 `[Error] Task failed: [Errno 28] No space left on device`；第 131 步的 rollout 落盘里 256 条有 205 条 score 是 0.0。[事故报告](examples/training/runs/qwen3-8b-grpo-20260928/notes/incident_report.md)。
4. 没回滚：第 129–131 步被 0.0 污染（零分条数 51 / 115 / 205），第 132–134 步整组同分没有 outcome 梯度；第 144 步验证持平 0.495，第 160 步 0.510 回到趋势线。代价是约 1.5 小时和一段偏离。回滚更干净，但要重做 6 步并手动改 resume。
5. 不是饱和，是格式部分榨干了：192 → 320 的增量 +0.035 全来自正确率（格式 0.000），格式合格者的正确率 0.546 → 0.584 只能慢慢涨；全对组 4.4% → 6.0%、全错组 2.0% → 1.4%，全同组合计不到一成，这些组 advantage 全 0 但不是主因；`entropy` 0.32 平稳、长度回升到 1,517，没有坍缩或爆炸。

</details>

**口述：** 看曲线先拆 reward：格式、正确率、故障三部分分开；再看两个该为 0 的指标；事故先看 `rewards/min`、`timing_s/reward` 和日志，不看模型。

<a id="stage-eval"></a>

## 六、结果测评：先定协议，再看数字

**现场。** [eval/protocol.md](examples/training/runs/qwen3-8b-grpo-20260928/eval/protocol.md) 训练前冻结；结果表由生成器写出 [eval/tables.md](examples/training/runs/qwen3-8b-grpo-20260928/eval/tables.md)，结论在 [eval/report.md](examples/training/runs/qwen3-8b-grpo-20260928/eval/report.md)。

| | 基线 Qwen3-8B-Base | run-01 | run-02 |
|---|---:|---:|---:|
| dev 200 题贪心 pass@1（格式错比例） | 0.380（18.5%） | 0.615（2.5%） | 0.635（2.0%） |
| dev 采样 pass@1（3 seed） | 0.352 ± 0.008 | 0.587 ± 0.008 | 0.640 ± 0.012 |
| dev 采样 pass@8 | 0.768 | 0.890 | 0.907 |
| held-out 142 题贪心 pass@1（格式错比例） | 0.134（20.4%） | 0.246（2.1%） | 0.317（2.1%） |
| held-out 采样 pass@1（3 seed） | 0.102 ± 0.005 | 0.231 ± 0.015 | 0.291 ± 0.014 |
| held-out 采样 pass@8 | 0.366 | 0.582 | 0.695 |
| held-out 配对（赢 / 输 / 平，对基线） | — | 18 / 2 / 122 | 26 / 0 / 116 |
| HumanEval+ / MBPP+（补全式，无格式门） | 0.671 / 0.681 | 0.689 / 0.694 | 0.701 / 0.706 |

**你来决定。**

1. run-01 在 dev 上 +23.5 个点、净胜 47 题，能说它更会写代码了吗？
2. 训练里的验证 0.595（run-01）和 0.656（run-02）能直接比吗？
3. held-out 的 +11.3 个点算不算提升？用什么判断？run-02 比 run-01 多出的 +7.0 呢？
4. 为什么 dev 的增益比 held-out 大这么多？run-02 在 dev 上为什么和 run-01 几乎分不出？

<details>
<summary>推演里的结论</summary>

1. 能说一部分，但不能只凭 dev：Base 基线 18.5% 的格式错和循环被修掉，占了增益的一大块；采样 pass@8 从 0.768 到 0.890，说明会做的题也多了，不只是贪心变稳。dev 的 GT 是题面样例，通过样例不等于正确；dev 还用来选 checkpoint。
2. 不能：验证用训练的回答上限，4096 下基线 -0.224、8192 下基线 -0.096，同一个模型两个数，差的是循环到上限的那部分。离线协议一律 8192。
3. 算：采样标准差 0.5–1.5 个点，3 倍约 4.5 个点；run-01 对基线净胜 16 题、只 2 题变差，远超噪声。run-02 对 run-01 的 +7.0 超过 3 倍标准差、净胜 10 题、1 题变差，也能说有提升，但 142 题的区间仍宽。
4. 三个原因：dev 测试弱，样例能过就算过；dev 和训练集同源（LeetCode）而 held-out 混了 AtCoder、Codeforces；Base 起点下两轮先学会的是格式和停机，这部分在同源的 dev 上兑现得最多。run-02 在 dev 上 +2.0 个点、净胜 4 题（12 赢 8 输），已接近这套样例测试能区分的上限；held-out +7.0、净胜 10 题（11 赢 1 输）、pass@8 0.582 → 0.695 才看得出差别。

</details>

**口述：** 评测三件事：协议先冻结；贪心和采样都报并带 seed 方差；dev 选模型、held-out 下结论。训练里的验证数字只看趋势。

<a id="stage-attribution"></a>

## 七、归因与优化：从结论到第二轮

**现场。** [attribution/reward_decomposition.md](examples/training/runs/qwen3-8b-grpo-20260928/attribution/reward_decomposition.md) 和 [notes/decision_log.md](examples/training/runs/qwen3-8b-grpo-20260928/notes/decision_log.md) 第七节。

**你来决定。** 先写你的五条归因，每条三栏：结论、证据、要改什么；再展开对照。

<details>
<summary>推演里的归因与改动</summary>

| 结论 | 证据 | 改动 | run-02 的结果 |
|---|---|---|---|
| 训练 reward 大半是格式分 | 拆分表 0.678 / 0.802（85%）；标签缺失 30% → 1%、循环到上限 23% → 7% | 回答上限 8192，`max_num_batched_tokens` 同步提到 12288（v0.4.1 断言它不能小于 prompt + response） | 开局格式失败 37%（其中截断 7%，对 run-01 的 54% / 23%），标签还是要从头学；正确率 0.278 → 0.653 |
| 全同组没有梯度 | 第 192 步 checkpoint 对每题采 8 条：全对 198 题、全错 155 题，合计 19%；训练里全对组 0.4% → 6.0%、全错组 13.3% → 1.4% | 去掉这 353 题，剩 1536 题；筛题后一轮只有 96 步，epochs 提到 4 | 筛的是第 192 步眼里的全同题，从头训的 Base 开局全错组仍 14%；到 320 步全对组 10.1%、全错组 0.8%，比 run-01 的 6.0% / 1.4% 更饱和，筛题要随 checkpoint 刷新 |
| 判题故障污染组内统计 | 事故 | prime_plus 整组作废；checkpoint 换到数据盘、只留 3 份 | 第 77、212、288 步三次故障记录，作废 0 / 1 / 1 组（各 16 条），曲线无痕 |
| clip 没起作用 | clipfrac 恒 0 | mini-batch 4 题，每步 4 次更新 | clipfrac 2–6%，ppo_kl 约 6e-4 |
| dev 饱和、偏乐观 | dev +23.5、held-out +11.3；dev 的 GT 只是样例 | 只信 held-out，贪心和 pass@8 都报 | held-out +18.3 个点（对基线净胜 26 题、0 题变差），比 run-01 多 +7.0、净胜 10 题；dev 上只多 +2.0、净胜 4 题，分不出；pass@8 0.582 → 0.695 |

run-02 的代价：每步 1,131 秒（gen 285 秒），100.6 小时，比 run-01 多 24.5 小时；回答长度 2,652 → 1,427（第 32 步）→ 谷底第 40 步 1,286 → 2,425，截断率 6.7% → 1.4%，没有归零。

</details>

**还没做、下一轮该做的：** 增强测试（[补强项 3](examples/source-supplements/README.md)）让 dev 不再只看样例；每 64 步用当前 checkpoint 重新筛题；困难题动得少（held-out hard 0.053 → 0.053 → 0.132，只有 38 题，run-01 一题没多对），考虑难度课程或更多采样；回答长度仍在涨，考虑超长惩罚。

**口述：** 归因就是把「reward 涨了」拆成格式、正确率、故障、全同组四件事，每件配一个指标和一个改动，第二轮只改能解释的，改完用 held-out 和配对比较验收。

<a id="homework-5"></a>

## 作业：把推演变成你自己的

已经有一位学习者按下面的方式走过一遍：七环决定在 [notes/your_decisions.md](examples/training/runs/qwen3-8b-grpo-20260928/notes/your_decisions.md)，按其决定推演的两轮在 [runs/qwen3-8b-grpo-yourplan-20261018/](examples/training/runs/qwen3-8b-grpo-yourplan-20261018/README.md)：your-01 用来做第五到七环的诊断，your-02 按诊断改三处续训、用 held-out 验证（[your-02_plan.md](examples/training/runs/qwen3-8b-grpo-yourplan-20261018/notes/your-02_plan.md) 是计划与执行记录；2026-10-19 去掉了 TACO，数据与参考 run-01 相同）。你可以照着做，也可以只借它的表头。

1. 复制 [notes/decision_log.md](examples/training/runs/qwen3-8b-grpo-20260928/notes/decision_log.md) 的表头，自己填七环的「决定与依据」，再展开本文各环的折叠部分对照，记下哪里和推演不同、谁更有道理。
2. 打开 [logs/run-01/metrics.jsonl](examples/training/runs/qwen3-8b-grpo-20260928/logs/run-01/metrics.jsonl)，不看事故报告，只用指标找出事故的起止步和证据，再看 [incident_step129-135.log](examples/training/runs/qwen3-8b-grpo-20260928/logs/run-01/incident_step129-135.log) 验证。
3. 用 [eval/summary.json](examples/training/runs/qwen3-8b-grpo-20260928/eval/summary.json) 和 `eval/lcb_greedy/*.jsonl` 自己算一遍 held-out 的配对净胜和 seed 标准差，判断 run-01、run-02 各能下什么结论；核对 `win − loss` 等于两边通过题数之差。
4. 改 [scenario_qwen3_8b.py](examples/training/scenario_qwen3_8b.py) 开头的一个假设（比如 run-01 也用 8192），重算，看哪些数字跟着变、哪些不变，写一句为什么；再跑 [check_scenario_docs.py](examples/training/check_scenario_docs.py) 看叙述里哪些数字要同步。

**检查是否学会：** 不看资料，口答：机器怎么算；为什么要迁框架；Base 起点格式从哪学、前 64 步为什么 85% 是格式；4096 为什么错；两个恒零指标；事故三证据；dev 和 held-out 为什么不一样；run-02 五处改动各对应什么证据。

<a id="interview-5"></a>

<details>
<summary>面试深挖：实跑类追问</summary>

回答按四句组织：机制是什么、这次怎么做的、一个反例、证据能证明到哪里。开头先说清「这是推演」。

| 追问 | 能抗住的回答 | 证据 |
|---|---|---|
| 8B 模型 GRPO 要多少显存？ | 训练态每参数 16 字节，8.2B 是 131 GB，分到 4 卡 33 GB，更新阶段峰值约 57 GB；采样引擎 40 GB 另算，训练时必须睡眠；激活开检查点。8 卡快 4 倍，个人研究用不起 | 第一环 |
| 训练前怎么确认判分是对的？ | 探针：已知正确、错误、提前退出、格式错四个候选，期望 1.1 / 0.1 / 1.1 / −1.1；没装沙箱时原版不报错、分数全 0 | `env/probe_scorer.log` |
| 曲线一开始涨得很快，是好事吗？ | 先拆：格式、正确率、故障。Base 起点的推演里前 64 步 85% 是格式（标签缺失 30% → 1%、循环到上限 23% → 7%）；看截断率、长度和标签错率就能判断 | 拆分表 |
| reward 突然掉到 0 怎么查？ | `rewards/min` 变 0.0、`timing_s/reward` 掉到几秒、日志里的 `[Error] … No space left on device`；0.0 只有管理器兜底才会出现；再决定回滚还是继续 | 事故报告 |
| pg_clipfrac 一直是 0 正常吗？ | 单 mini-batch 且 ppo_epochs=1 时必然为 0，不是 bug；要 clip 起作用就把 mini-batch 改小；v0.4.1 不允许 mini-batch 大于 batch | 第五环 |
| 训练 reward 涨了，模型就更强了吗？ | 不一定：dev 测试弱、同源；held-out 配对净胜和 seed 方差才算数。推演里 run-01 对 Base 基线 dev +23.5、held-out +11.3、净胜 16 题，远超噪声；但 run-02 比 run-01 在 dev 上只多 +2.0，要到 held-out 和 pass@8 才分得出 | 评测报告 |
| 第二轮改了五处，怎么知道哪处起作用？ | 推演里没做消融，只能说合起来有效；要分开就一次改一处，成本是 5 倍。能单独验证的是格式部分（拆分表）和 clip（clipfrac） | 第七环 |
| Base 模型做 RLVR 有什么特别？ | 格式从头学：第 1 步一半回答格式错（三成没写标签、两成多循环到上限），前 64 步 85% 的增益是格式，看截断率和标签错率就能拆；回答长度先降后升（循环先被罚掉，再学会多想）；熵从 1.1 起，前 60 步快降是格式锁定不是坍缩，判断看 pass@8 有没有涨；KL 走得远（0.065 对系数 0.001）是正常的；聊天模板按默认不预填、`<think>` 模型自己写，模板是否自带 smoke 时核对；上限仍要给够，chunked prefill 的批大小跟着调 | 第三、四、五环 |

</details>

<a id="provenance-5"></a>

## 推演是怎么生成的、边界在哪

- 曲线：[scenario_qwen3_8b.py](examples/training/scenario_qwen3_8b.py) 里的闭式函数加固定种子噪声，开头列了全部假设；批均值按整数计数算，所以落在 256 条样本能取到的格点上；改假设重算，数据文件一起变，叙述里的数字用 [check_scenario_docs.py](examples/training/check_scenario_docs.py) 核对后手工同步。
- 真实的部分：第 1 步 256 条判分日志由真实判分代码执行参考解和变异体得到，加起来的均值就是指标行里的 −0.409；数据 parquet（原始版和去污染版）和去污染结果是真实输出；判分探针是真实输出；配置键名和断言对照 verl v0.4.1 的 `ppo_trainer.yaml` 与 `ray_trainer.py`，指标名对照 `metric_utils.py`。
- 合成的部分：思考文本、评测明细、wandb 文件、checkpoint 清单、版本号、费用。逐项见 [PROVENANCE.md](examples/training/runs/qwen3-8b-grpo-20260928/PROVENANCE.md)。
- 推演没法回答的：真实的生成吞吐、真实的曲线形状、Qwen3-8B-Base 在这套数据上的真实基线（格式错和循环各占多少、标签多少步学会）、Base 的 tokenizer 是否自带与 Qwen3-8B 相同的聊天模板（实跑前核对）。拿到 GPU 后，先跑[路线评估](路线评估_仅学习与实跑.md#track-run)的一档单卡 smoke，再决定要不要把这份推演的配置真跑一遍。

<a id="record-5"></a>

## 本篇完成记录

| 项目 | 完成证据 |
|---|---|
| 七环决策 | 自己的决策日志，与推演对照后的差异说明 |
| 曲线解读 | 不看报告找出事故起止步与三条证据 |
| 评测判断 | 自己算的配对净胜与 seed 标准差，以及两轮各能下的结论 |
| 口述 | 面试深挖表任选三问，四句答法 |
| 2026-10-16 | 学习者走完七环：前四环决定 → 生成 your-01 → 曲线四问（导师给机制与证据链）→ 评测三决定（报贪心净胜并注明限制、交第 200 步、补 held-out）→ 归因四决定（tmpfs、筛题 + clip-higher、续训 200 步、切 300 题验证） |
| 2026-10-19 | 学习者决定去掉 TACO，用 your 的配置做完整一轮：your-01（LeetCode 1889 题、6144 / n=8 / 200 步）诊断四处 → your-02 续训 200 步修三处（tmpfs、筛题、换验证集；计划里的 clip-higher 经学习者授权去掉）→ held-out 对 your-01 净胜 8 题（+5.6 个点）、事故修好（存盘后判题 25 秒、作废 0 组）、熵 0.29 → 0.28 没再塌；记录在 yourplan 目录与 your_decisions 第八节 |

---

**继续阅读：** [返回资料首页](README.md) · [路线评估](路线评估_仅学习与实跑.md) · [运行记录](examples/training/runs/qwen3-8b-grpo-20260928/README.md) · [你的 run](examples/training/runs/qwen3-8b-grpo-yourplan-20261018/README.md)
