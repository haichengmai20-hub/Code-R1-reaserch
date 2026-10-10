# 阶段三：沿工业框架学习 Code-R1 的 Rollout 与 GRPO

**资料导航：** [资料首页](README.md) · [工业主框架](RLVR工业流程.md) · [01 数据与 GT](01_数据理解与GT构建.md) · [02 Verifier 与任务池](02_Verifier与任务池验收.md) · **03 Rollout 与 GRPO（当前）** · [04 训练与评测](04_真实训练与独立评测.md)

**阶段位置：** 四个阶段中的第三阶段。

**本阶段学习内容：** 沿最大子数组和这道真实 Hot100 Task，从候选代码、Tests 和 Verifier 追到 reward、同题分组、advantage、mask、old/new/ref logprob 与策略损失。先用 CPU 受控候选完成机制主线，再按资源进入真实模型 rollout。

**主框架：** 先看[最小闭环](RLVR工业流程.md#mvp-overview)，再读[判定与执行](RLVR工业流程.md#verification)、[奖励映射](RLVR工业流程.md#reward-mapping)、[GRPO 优势](RLVR工业流程.md#grpo-advantage)与[策略更新](RLVR工业流程.md#policy-update)。可靠工程扩展按实际瓶颈补充，不作为先完成工业系统才能学习本阶段的门槛。真实参数更新和同步的运行验收放在第四阶段。

**输入：** 前两阶段准备的同一个 Task、Tests（源码装入 `reward_model.ground_truth` 的 GT）、候选代码与已检查的 Verifier。task_id、来源、难度与版本等 metadata 是同一 Task 的属性。**输出：** 同一演示的报告与一份阶段说明，能对齐候选、判定、reward 与组内计算。

**先记机制：** reward是行为反馈，advantage是同题相对信号；固定该信号后，对当前模型生成token的logprob构造loss并求导。判题程序不需要可微。同题分组、token对齐、求导边界和平均方式任何一处错配，优化的就可能是另一件事。

正确实现可用作 Oracle，即求解或核验答案的方法；它不替代 Tests，也不是 KL 所用的 Reference Policy。基础对象保持为 Task、Tests、候选代码与 Verifier。

## 主框架、项目实现与作业的对应关系

| 工业流程中的环节 | 在 Code-R1 具体查什么 | 对应作业及证据 |
|---|---|---|
| [判定与执行](RLVR工业流程.md#verification) → [奖励映射](RLVR工业流程.md#reward-mapping) | [coder1 scorer](../verl/utils/reward_score/coder1/__init__.py) 的 `_compute_score()`；[PrimeRewardManager](../verl/workers/reward_manager/prime.py) 的 `__call__()` | [3.1](#assignment-3-1)：候选、Tests、执行状态与标量分数 |
| [GRPO 优势](RLVR工业流程.md#grpo-advantage) | [ray_trainer.py](../verl/trainer/ppo/ray_trainer.py) 的 `compute_advantage()`；[core_algos.py](../verl/trainer/ppo/core_algos.py) 的 `compute_grpo_outcome_advantage()` | [3.1](#assignment-3-1)/[3.2](#assignment-3-2)：样本标准差、uid、末 token 与 mask |
| [策略更新](RLVR工业流程.md#policy-update) | `RayPPOTrainer.fit()` 的生成、重复对齐与重排；`compute_policy_loss()` 与 [DataParallelPPOActor.update_policy()](../verl/workers/actor/dp_actor.py) | [3.1](#assignment-3-1)/[3.3](#assignment-3-3)：old/new/ref、clip 与调用时序 |
| [故障归因](RLVR工业流程.md#layer-attribution) | `parallel_compute_score_async()` 的异常回填与全组归一化 | [3.2](#assignment-3-2)：只改变 infra 处理条件的因果检查 |
| [任务准备](RLVR工业流程.md#task-preparation)与[评测](RLVR工业流程.md#evaluation)：当前 policy 难度 | [main_grpo.sh](../main_grpo.sh) 的采样配置、`fit()` 的新生成与逐题分母 | [3.3](#assignment-3-3)：独立检查；真实 rollout 扩展另记录 checkpoint |

本阶段要能沿一条 Code-R1 样本追完整数据流。框架给出模块用途，源码决定本版本具体时序；不要把示意图的顺序机械当作实现顺序。

## 学到什么才算完成

你应能解释“一道题多次生成”如何形成一个组，正确性评分怎样作用于模型 token，以及为什么解题用正确代码与 Reference Policy 不能混为一谈。前者是求解/核验答案的方法，后者是用于 KL 约束的语言模型。

[最大子数组和演示](examples/maximum-subarray/run_demo.py)使用标准库、CPU 与已检查的固定候选，复用源码里的 parser 和 `_compute_score()`，只替换执行入口为受控本地 subprocess。它演示机制，没有从 checkpoint 采样；固定候选分数、pass@k 和教学 token 长度不能称为模型 rollout、模型难度或能力提升。

这个受控程序没有构成安全执行任意新候选的 sandbox；未知模型代码须进入有隔离能力且已验收的执行后端。正确 GT 也不能单独证明 scorer、比较规则与执行状态可信，第二阶段的反例检查继续适用。

Rollout 是由某个 checkpoint 按给定参数生成新回答；logprob 是模型对 token 给出的对数概率；advantage 是同题组内的相对训练信号；mask 指出哪些位置有效。理解这些定义之后，必须在项目字段和函数中找到对应值。

<a id="prerequisites-3"></a>

## 前置自测：shape、概率与求导

`(B,R)`表示B条回答、每条R个response位置；词表logits还有V这一维，形状为 `(B,R,V)`。mask=1的位置才进入本次直接策略loss。能读sum、mean、exp、log和“把某些数视为常量”即可先做纸面与标准库实验；PyTorch的 `no_grad/backward` 对应何处随后追源码。

口答：prompt与response拼接后，哪个位置预测第一个response token？在因果模型中是prompt最后一个位置。若用token自己所在位置的logits计算自己的概率，就错位了。普通路径截取 `-R-1:-1`；remove-padding路径先左移labels，再还原并截取response。见 [dp_actor._forward_micro_batch](../verl/workers/actor/dp_actor.py)。

<a id="gradient-bridge"></a>

## 从logits到更新：一条可核对的链路

先看[主框架的公式与求导边界](RLVR工业流程.md#gradient-bridge)，再运行：

```powershell
python -B RLVR学习资料/examples/maximum-subarray/mechanism_lab.py --run mechanism-01
```

这是两标量logits的实际toy SGD，使用解析导数和中心有限差分；没有torch、tokenizer或LLM。报告在 `practice_runs/mechanism-01/attempt-xxx/mechanism_report.json`。它补齐“数字为何产生这个方向”的机制，不充当本项目真实autograd或GPU更新证据。

完整示范：logits为 `[0, ln(3)]`、T=1、选第二个token，概率0.75，old logprob固定为ln(0.75)，A=0.5。ratio=1，loss=-0.5；对两个logit的解析梯度为 `[+0.125,-0.125]`，与有限差分核对。学习率0.1的一步SGD降低第一个、提高第二个logit，选中token概率上升。

| 量 | 在本次更新中怎样得到 | 是否对它求当前策略梯度 |
|---|---|---|
| old logprob | rollout后、update前重算，`compute_log_prob`用no_grad | 固定基准；下批重新得到 |
| ref logprob | Reference Policy计算，no_grad | 固定约束基准，不是Ref Code |
| advantage | 对score求和、按uid分组、样本标准差归一化，no_grad | 固定权重，含epsilon与mask |
| new logprob | 当前actor forward，取已生成token的概率 | 梯度经模型计算图回到参数 |

new和old数值相同不代表无梯度；如果让old也连接到同一new计算图，差值恒0，梯度被错误抵消。离散reward进入固定权重，实际求导的是模型概率，而非执行程序或测试整数答案。

改变一个条件：把A改为负，检查梯度方向；或把T改为2，比较选中概率约0.633975。当前源码重算也除以temperature；不能把T=0代入softmax训练公式，greedy是生成器的另一分支。若另外开启top-p/top-k或processor，还需核对重算概率与真实采样分布是否一致。

**clip不是梯度范数裁剪：** A>0且ratio高于上界，或A<0且低于下界时，所选PG分支可成为常数、梯度为0；相反方向仍需纠正。共享参数、其他token、KL/entropy仍能改变概率。报告给正负A同ratio对照；独立练习再检查另一侧边界。

**长度与mask：** toy里两回答逐token loss为+1/-1，长度为2/6，同一micro内masked_mean=-0.5；先逐回答平均，或拆成两个等权micro累积，则为0。本源码先在每个micro内按token平均，再按固定累积或动态行数比例缩放。不能只说“每条回答权重相同”或“改micro只影响显存”。EOS示范包含首个EOS；无EOS全1，pad与EOS不同且缺真实长度时也可能全1。要结合停止信息，不能由mask宣称正常完成。

## 阅读入口与顺序

| 顺序 | 直接阅读的材料 | 要回答的问题 |
|---|---|---|
| 1 | [run_demo.py](examples/maximum-subarray/run_demo.py)、[coder1 scorer](../verl/utils/reward_score/coder1/__init__.py) 的 `_compute_score()` | 固定候选怎样接原判分？高 reward 是否保证完整执行了 Tests？ |
| 2 | [PrimeRewardManager](../verl/workers/reward_manager/prime.py) 的 `__call__()`、`parallel_compute_score_async()` | reward 张量的 shape、有效长度与赋值位置是什么？异常回填怎样改变组统计？ |
| 3 | [core_algos.py](../verl/trainer/ppo/core_algos.py) 的 `compute_grpo_outcome_advantage()`、`compute_policy_loss()` | 怎样求和、分组、求样本标准差、广播并加 mask？clip 怎样用 old/new logprob？ |
| 4 | [main_ppo.py](../verl/trainer/main_ppo.py)、[ray_trainer.py](../verl/trainer/ppo/ray_trainer.py) 的 `fit()` | 生成、uid、重复对齐、均衡重排、old/ref logprob、reward、advantage 与 update 如何衔接？ |
| 5 | [main_grpo.sh](../main_grpo.sh)、[ppo_trainer.yaml](../verl/trainer/config/ppo_trainer.yaml)、[FSDP workers](../verl/workers/fsdp_workers.py) | 若做真实 rollout，模型/backend/每题采样数、长度、batch 与 KL 从哪里配置？原脚本的 16 次采样是否在你环境实际执行？第四阶段再核验同步 |

## 本阶段作业总览

| 作业重点 | 对应位置 |
|---|---|
| 跟着完整示范完成机制主线 | [作业 3.1：完整示范与源码对齐](#assignment-3-1) |
| 只改变一个条件，解释全组变化 | [作业 3.2：infra 回填怎样改变整个组](#assignment-3-2) |
| 无步骤提示，独立完成迁移检查 | [作业 3.3：独立检查分组、mask 与策略损失](#assignment-3-3) |

三项作业共用一份阶段说明和同一 [report.json](examples/maximum-subarray/outputs/report.json)，不要求为了各个字段另拆许多交付文件。

<a id="assignment-3-1"></a>

## 作业 3.1：完整示范与源码对齐

**目的：** 沿同一个最大子数组和 Task，看清候选判定怎样接到训练信号。

在项目根目录 `D:/RL及harness/code-r1` 执行：

```powershell
python .\RLVR学习资料\examples\maximum-subarray\run_demo.py score
python .\RLVR学习资料\examples\maximum-subarray\run_demo.py advantage
```

子命令会自动补齐依赖；也可一次执行 `python .\RLVR学习资料\examples\maximum-subarray\run_demo.py all`。输入、Tests、包装后的 Task 与小池分别落在同目录 `outputs/raw_task.json`、`cases.json`、`prepared_task.json`、`data_pool.jsonl`，结果集中在 `outputs/report.json`。它们来自本次最大子数组和 Task，与旧五题、8,088 组历史核验无关。JSONL 用于教学读回和 Code-R1 schema 示范；真实 trainer 仍需 Parquet 转换、正确数据路径与运行环境，不能直接将这个 JSONL 传给 `main_grpo.sh`。

**完整示范：**

1. 在报告中找到两份正确实现 `reference`、`correct_prefix` 与错误实现 `wrong_empty`、`wrong_sum` 的判定，再沿 `prepared_task.json` 找到同一组 Tests。全负数组 `[-3,-2,-5]` 的正确答案为 -2，能检出“允许空子数组而返回 0”；局部最优子数组与整体和不同的输入能检出“直接返回总和”。先看测试实际执行和通过状态，再解释 reward，不能仅按 `reward>0` 计通过。
2. 正常组中两份正确实现得 1.1、两份格式合格但测试失败的实现得 0.1。把四行重排成 `[0.1,0.1,1.1,1.1]`，均值为 0.6，样本标准差约 0.57735；忽略很小的 epsilon 后，advantage 为 `[-0.86603,-0.86603,+0.86603,+0.86603]`。这是固定候选判分与公式计算，候选顺序以实际报告为准。`torch.std` 在本项目使用样本标准差，不能换成总体标准差而继续沿用数字。
3. 用教学长度 `[4,6,3,5]`、最大长度6画出 `(4,6)` 的token reward与response mask；这是独立手算设定，运行报告的长度以报告为准，均不是tokenizer测量。Prime把scalar放最后一个有效response token，GRPO求和、按uid归一化、广播。已正确构造mask时padding贡献0；真实EOS/pad/截断边界按上面的源码检查，prompt不在response mask中。
4. 在 `fit()` 对齐本版本真实实现时序：生成 → uid 与重复对齐 → 均衡重排 → old logprob 重算 → 按需计算 ref logprob → reward/KL 分支 → advantage → actor update。`uid` 是批次内同题候选的组身份，与稳定 task_id 用途不同；均衡重排会打乱行，不能靠相邻四行猜组。
5. 用一个 token 示范 clip：`old_logprob=-2`、`new_logprob=-1.8`、`cliprange=0.2`，概率比为 `exp(new-old)=1.22140`，裁剪值为 1.2。源码取 `max(-A*r, -A*clip(r))`；A=1 时取 -1.2，A=-1 时取 +1.22140。不能直接相除 logprob，也不能以为正负 advantage 的裁剪选择相同。

old 是本次 actor 更新前策略对这些 token 的 logprob；new 来自训练 forward；ref 来自 KL 所用的 Reference Policy。主脚本启用 actor KL loss 时，`token_level_rewards` 直接取 `token_level_scores`，KL 在 actor loss 处理；未启用该开关时，trainer 的 `apply_kl_penalty()` 可先从 reward 扣 KL 再算 advantage。演示是没有模型 KL 的 outcome 数值，不自动代表任意训练配置。

**交付与验收：** 将报告路径、候选判定、scalar/末 token/mask 图与源码时序写入同一阶段说明。如果回答包含推理与答案，当前有效 response token 都可能承接 outcome advantage；不能因 Tests 只检查最终代码就声称只训练代码 token。报告中 `bad_format` 得 -1.1，`early_exit` 虽得 1.1，却 `audit_completed_case_count=0`、`correctness_pass=false`；它们不能作为正确性通过，后者直接说明高 reward 不保证完成 Tests。手算或函数单独执行成功不称为已完成 RL 参数更新。

<a id="assignment-3-2"></a>

## 作业 3.2：infra 回填怎样改变整个组

**目的：** 只改变一次基础设施失败的处理条件，检查错误 reward 怎样影响同组其他候选。

先设四个格式合格候选都因逻辑错误未通过 Tests，raw reward 为 `[0.1,0.1,0.1,0.1]`，组内 outcome advantage 全为 0。保持 Task、Tests、代码与 policy 不变，只将其中一次执行改为 infra error，并假设异常处理将它回填为 0：

```text
原始逻辑失败组     [0.1, 0.1, 0.1, 0.1]
一次 infra 回填后  [0.1, 0.1, 0.1, 0.0]
```

这是一项故障处理的数值模拟，不能说已在真实后端注入并测得故障。核对 `parallel_compute_score_async()` 的异常回填，再计算：均值 0.075、样本标准差 0.05，忽略 epsilon 的 advantage 是 `[+0.5,+0.5,+0.5,-1.5]`。

**条件变化后的检查：** 为什么三个仍然测试失败的候选现在得到正 advantage？为什么不能解释成模型突然会解题？如果恢复原判定，哪些值应恢复？再写出你在真实实验前采用的 infra 政策，例如整组标记无效并重新采样，或停止该批次排查；说明代价与对训练分布的影响。源码回填行为与你设计的可靠工程策略分别记录，不能把当前源码写成已有完整故障治理。

同时核对两个边界：同组全同 reward 缺少组内相对 outcome 信号，不意味着任务永远无用或所有其他梯度项为零；单例组在本版本源码使用均值 0、标准差 1，与多候选全同分组的处理不同。写明组大小、epsilon 与样本标准差。

**交付与验收：** 将一条件对照、计算与处理政策加入同一说明。能讲清“infra 状态 → 回填 reward → 全组均值/标准差 → 每条 advantage”的因果链。模型生成中的格式失败是策略失败，应进入采样分母；infra 按预先登记的规则另处置，不能结果出来后删去失败以抬高 pass@k。

<a id="assignment-3-3"></a>

## 作业 3.3：独立检查分组、mask 与策略损失

**目的：** 在变化条件下独立定位错误，检查是否只会复述示范答案。

独立完成下列任务，不按 3.1 的步骤抄结果；保留自己的计算与证据，再用源码和演示报告自检。

- 将最大子数组和组与另一个 Task 组混排，并改变部分回答长度。设计会暴露“按相邻四行分组”的检查，给出每条候选恢复到正确 GT、uid、mask 和 policy 版本所需的信息；证明重排不改变同一候选的组内结果。
- 对一个有效 token 使用 `old_logprob=-2`、`new_logprob=-2.3`、`cliprange=0.2`，分别选正负 advantage，独立算概率比、裁剪值与源码最终使用的 policy loss 项。加入 padding 后说明哪些位置参与平均。
- 自己提出一个合法的最大子数组和输入与一段新的错误实现，使失败条件与 `wrong_empty`、`wrong_sum` 有区别。先推导正确答案，再说明怎样用 Tests 检出它，以及为什么正确答案仍需核对 scorer；不要把给出的反例原样作为独立发现。未知代码先纸面推演，实际执行须使用隔离后端，不能直接接入受控演示。
- 比较 `[0,0,1,1]` 与 3.1 的归一化结果，解释近似相同的条件，以及为什么这不能推出项目 reward 总是二值。指出格式失败、未完整执行 Tests 却得高 reward 等状态怎样分别记录。

**交付与验收：** 仍使用同一阶段说明。独立找到至少一个能区分分组、mask、clip 或判定错误的条件，以字段、计算或反例说明影响。源码定位写“文件 + 函数”；注释、配置与模拟输出不作为实际推理或梯度证据。

另交自己的机制变式：换一组logits/选中token/A，手算概率、冻结量和梯度方向，再用[机制函数](examples/maximum-subarray/mechanism_lab.py)自检；比较正负A在高/低ratio的四种clip情形。写出两种长度分配与micro分组的loss差异，以及EOS/no-EOS的mask。评分与参考推导见[机制答案](examples/maximum-subarray/ANSWERS.md#mechanism-rubric)，不要将现成报告当独立答案。

<a id="interview-3"></a>

<details>
<summary>面试深挖：概率、梯度、KL、mask与分组</summary>

| 追问 | 回答要点/能区分的反例 | 本版本来源 |
|---|---|---|
| loss等于0就无更新吗？ | 数值不等于导数。old固定时ratio=1仍有- A 的logprob导数；组内loss数值相消也不保证不同token的参数梯度相消 | [dp_actor](../verl/workers/actor/dp_actor.py)，[policy loss](../verl/trainer/ppo/core_algos.py) |
| 为什么用概率比？为什么ref不是old？ | 旧样本上ratio表达当前概率相对采样/更新前的变化，构造token级surrogate与clip；不等于全部序列分布被精确重加权。old随批次重算，ref用于约束偏移 | [compute_policy_loss](../verl/trainer/ppo/core_algos.py)，[fit](../verl/trainer/ppo/ray_trainer.py) |
| GRPO为什么可以不用critic？ | 用同题组分数构造相对baseline；省critic不保证低方差，G小、全同分或错误reward会影响信号。本版本singleton单独处理 | [compute_grpo_outcome_advantage](../verl/trainer/ppo/core_algos.py) |
| PPO clip保证概率比始终在区间吗？ | 不保证；它裁目标的有利方向，PG梯度可0，但共享参数/其他项可改概率。参数梯度范数裁剪在_optimizer_step是另一步 | [compute_policy_loss](../verl/trainer/ppo/core_algos.py)，[actor](../verl/workers/actor/dp_actor.py) |
| actor KL与reward KL等价吗？ | 前者用new/ref加loss，outcome advantage来自raw score；后者用old/ref逐token扣reward，再求和并组归一化，改变组信号和长度影响 | [apply_kl_penalty/fit](../verl/trainer/ppo/ray_trainer.py)，[kl_penalty](../verl/trainer/ppo/core_algos.py) |
| 单token logp-logq为负是KL错吗？ | 有符号单点可负；在适当分布下期望才对应KL。low_var用exp(ref-new)-(ref-new)-1并clamp，不是精确全词表KL；有限旧策略样本是surrogate | [kl_penalty](../verl/trainer/ppo/core_algos.py) |
| 全同reward完全没梯度吗？ | 多候选组outcome advantage为0，但KL/entropy仍可能有梯度；singleton在本版本另用mean0/std1 | [GRPO advantage](../verl/trainer/ppo/core_algos.py)，[actor loss](../verl/workers/actor/dp_actor.py) |
| 相邻G行就是一题吗？ | token均衡重排后不是；按uid，GT/response/mask/reward必须一起重排。task_id是稳定数据身份，uid是本批组身份，跨policy/批不能混组 | [fit/_balance_batch](../verl/trainer/ppo/ray_trainer.py)，[DataProto.reorder](../verl/protocol.py) |
| 只检查代码就只训练代码token吗？ | 当前outcome advantage广播到有效response，包括推理。它是粗粒度credit assignment，不是逐步推理正确性监督 | [Prime](../verl/workers/reward_manager/prime.py)，[GRPO](../verl/trainer/ppo/core_algos.py) |
| 全1 mask证明正常停止吗？ | 无EOS也全1，可能长度截断。当前rollout未完整保留stop reason，需要另记引擎信息；pad/EOS/真实长度一起查 | [get_eos_mask](../verl/utils/torch_functional.py)，[vLLM rollout](../verl/workers/rollout/vllm_rollout/vllm_rollout.py) |

能回答“本版本怎么做”和“为什么这个反例改变结论”，比只背GRPO公式更重要。

概率核对还需固定token/prefix、温度、processor、policy版本和模型模式。当前old计算用eval，update用train；有随机层的模型即使参数相同也未必数值相同，须结合实际模型配置验证，不把toy的ratio=1当无条件运行事实。`full` KL在本版本未实现，不能据配置名字声称已计算完整词表KL。

</details>

### 有资源后：真实 rollout 与当前 policy 难度

真实采样可接在机制作业之后。**模型选择、Linux 执行后端和采样预算须在生成未知候选之前明确，不能等第四阶段才决定。** 先确认 checkpoint/tokenizer、模型许可、显存与推理 backend；验收隔离执行环境；固定每题采样数、温度、seed、token 上限、超时、并发与总预算。CPU 受控演示已能完成机制主线，不要求先购买资源或完成训练。

资源具备后，从第二阶段小池取少量诊断Task，例如每题4个候选，另记主脚本原配置。保存新回答、提取代码、task_id、uid、checkpoint、判定/runtime版本、状态与reward；停止原因需另从引擎记录，当前DataProto返回字段未完整保留，不能由mask反推。历史meta.response不能代替本轮生成。具体数据与batch准备见[Linux小配置](examples/training/README.md#batch-contract)。

按题报告通过数/全部生成候选数、格式失败、逻辑/运行失败、infra 与组内 reward 方差。4 个生成候选中 2 个正确，成功率为 1/2；格式错误仍占分母，0.1 的格式奖励不是正确性通过。infra 按预先锁定政策报告，包含重试与成本。`Medium` 是 Task 的原始 metadata；当前成功率取决于 checkpoint、采样参数和判定版本，小组估计有不确定性，换 checkpoint/温度后不是同一难度结果。

对照 [预处理器 coder1.py](../examples/data_preprocess/coder1.py)：它没有执行完整 policy rollout 难度过滤。新统计记录为本 policy 与协议下的实际测量；评分不可靠或 infra 占主导时，返回第二阶段排查，不用这些数字解释模型能力。

## 本阶段完成记录

| 项目 | 完成证据 | 证据不足时的标记 |
|---|---|---|
| CPU 机制主线 | 固定候选报告、末 token/mask、样本标准差与源码时序 | 仅阅读/手算；演示待执行 |
| 条件变化与独立迁移 | infra 因果练习、重排/clip 与自己提出的反例 | 区分纸面推演与实际执行 |
| 可选真实 rollout 与难度 | checkpoint、实际新生成、判定版本与逐题分母 | 模型/后端/预算待明确；真实采样待完成 |

进入第四阶段前，应能指出每条训练箭头的实际函数和字段。CPU 演示、真实采样、训练更新按证据分别记录，不把固定候选的通过率写成模型性能。

---

**继续阅读：** [返回资料首页](README.md) · [工业主框架](RLVR工业流程.md) · [上一阶段：02 Verifier 与任务池](02_Verifier与任务池验收.md) · [下一阶段：04 训练与评测](04_真实训练与独立评测.md)
