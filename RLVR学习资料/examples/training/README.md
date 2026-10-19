# 从CPU准备到Code-R1真实小实验

[资料首页](../../README.md) · [01数据](../../01_数据理解与GT构建.md) · [03采样与机制](../../03_Rollout与GRPO机制.md) · [04运行与验收](../../04_真实训练与独立评测.md)

本页是一条具体操作桥接。标准库的数据/schema与batch准备已在本机核验；Linux/CUDA、pyarrow导出、模型采样、更新和恢复按目标环境实跑后记录。给出的配置是该源码版本的小算例，不承诺某张卡一定能跑，也不复现作者8×H200效果。

<a id="pool-preparation"></a>

## 1. CPU：真实不同任务的小池与读回

在仓库根目录运行：

```bash
$PY -B RLVR学习资料/examples/training/prepare_training.py --run prepare-01
```

新 `runs/prepare-01/` 含train.jsonl、dev.jsonl、manifest.json。train四题为climbing-stairs、house-robber、single-number、maximum-product-subarray；dev两题为valid-parentheses、longest-substring-without-repeating-characters。它们来自实际v2 RL/SFT记录，按原合并顺序首次保留，检查没有v2 test question_id交集；最大子数组和及两数之和的诊断用途排除。

每行保留原test并追加check(entry)，GT为JSON字符串；prompt使用完整原题面（含公开样例和约束）加签名。这是有记录的教学适配，原coder1只用meta.query。隐藏test正文与reference不进入actor prompt。prompt中公开样例不是隐藏测试泄漏。

manifest记录源文件/行号/hash、原split、本次train/dev、seed数量、源码版本与未完成事项。状态是prepared_not_admitted、train_ready=false：程序检了schema、ID和读取，不重新求解六题全部expected，不做语义家族或最大规模认证。源题面中的指数/格式仍需逐题审阅，不能全局改数字“修复”。先按01方法核对题意、接口和seed，再开展目标环境验收。

目标Linux环境已有项目依赖与pyarrow后，在同一源码基线执行：

```bash
python RLVR学习资料/examples/training/prepare_training.py --run parquet-01 --export-parquet
```

脚本写Parquet后按嵌套值读回并比较原row。缺pyarrow明确返回2；本机没有安装该库，因此本机证据只覆盖JSONL，不称Parquet已现场导出。不要把JSONL文件名传给trainer，也不要仅改扩展名。

<a id="batch-contract"></a>

## 2. 先算实际batch，再选运行配置

```bash
$PY -B RLVR学习资料/examples/training/prepare_training.py --run batch-two-ranks --world-size 2 --q 4 --g 4 --mini 4 --micro 2 --tp 1
```

这是CPU静态算术，没有启动两个GPU。Q是题行数，G是每题候选数，M是输入actor mini值。本版本普通1维FSDP、SP=1下：

| 场景 | 全局回答Q×G | 每rank行数 | worker的local mini=M×G/world | 每GPU micro | 固定累积 |
|---|---:|---:|---:|---:|---:|
| world=1，Q=4，G=4，M=4，TP=1 | 16 | 16 | 16 | 2 | 8 |
| world=2，其余相同 | 16 | 8 | 8 | 2 | 4 |

程序拒绝world不能被TP整除、Q×G不能被world整除、mini/micro不完整等情况。它只支持本小例SP=1；不同sharding布局另验。源码会乘G再归一化，所以输入M不自动等于回答数。源脚本算术断言、静态可整除和显存可用是不同证据。

<a id="linux-acceptance"></a>

## 3. Linux环境与判题两条验收边界

按[项目README](../../../README.md)的环境要求核对Python、torch、CUDA、vLLM和flash-attn等，不混用当前新版verl配置。vLLM 只装 0.6.3 或 0.7.3：项目 README 装 0.7.3，但它在单卡上第一次生成就会失败，单卡请用 0.6.3（配 torch 2.4.0），0.7.3 至少要 2 卡（配 torch 2.5.1），原因见下面的「实跑前必查」。记录实际版本、模型/checkpoint/tokenizer与许可、本地模型路径、显存、上下文长度、CPU执行并发和预算。prepare工具只观察当前host/包是否存在，不安装或下载。

未知候选开始执行前检查目标backend的权限/文件/网络隔离、时间/内存限制、输出预算与清理；先用已审正确、错误、死循环等程序做目标环境探针。原Firejail的退出码判分有已演示的early-exit缺口；换Docker不自动修复。

- **原版机制smoke：** 在执行隔离验收后，复现原链路并记录其已知假阳性；从独立audit判定有限case，不能把raw reward当可信正确性。它用于观察运行机制。
- **可信效果实验：** 还必须修复/验收完成与结果协议、输入答案、比较规则、故障治理，冻结实际版本。原版early-exit探针失败时，这条验收未完成。

train_ready是本包的信息字段，原trainer不会自动以它阻止启动。源GT和“prepared”状态不构成运行许可或质量认证；根据以上具体检查决定实验范围，并在记录里写清未完成边界。

## 4. 本版本单rank一步命令示例

先完成数据/环境检查、固定协议baseline，确认存在实际Parquet和模型目录。本段在Linux Bash执行，macOS 只运行前面的准备命令。路径变量按自己的目标环境填写，命令未在本机执行：

```bash
set -euo pipefail
export LEARN_MODEL_PATH=/absolute/local/model-checkpoint
export LEARN_POOL_DIR=/absolute/code-r1/RLVR学习资料/examples/training/runs/parquet-01
export LEARN_RUN_DIR=/absolute/code-r1/RLVR学习资料/examples/training/runs/model-smoke-01
export CODER1_EXEC=firejail
test -d "$LEARN_MODEL_PATH"
test -f "$LEARN_POOL_DIR/train.parquet"
test -f "$LEARN_POOL_DIR/dev.parquet"
learn_smoke() {
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  "data.train_files='$LEARN_POOL_DIR/train.parquet'" \
  "data.val_files='$LEARN_POOL_DIR/dev.parquet'" \
  data.train_batch_size=4 data.max_prompt_length=1024 data.max_response_length=1024 \
  "actor_rollout_ref.model.path='$LEARN_MODEL_PATH'" \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  trainer.nnodes=1 trainer.n_gpus_per_node=1 \
  "trainer.total_epochs=${LEARN_EPOCHS:-1}" \
  "trainer.total_training_steps=${LEARN_TOTAL_STEPS:-1}" \
  "trainer.resume_mode='${LEARN_RESUME_MODE:-disable}'" trainer.resume_from_path=False \
  trainer.save_freq=1 trainer.test_freq=1 \
  'trainer.logger=[console]' \
  "trainer.default_local_dir='$LEARN_RUN_DIR'" \
  reward_model.reward_manager=prime "$@"
}
learn_smoke
```

**实跑前必查（2026-10-15 核对）：**

- 路径值都加了单引号。本课目录名「RLVR学习资料」含中文，不加引号时 Hydra 在解析阶段就报 `LexerNoViableAltException`，GPU 机器上也一样（本机实测）。
- 单卡请装 vLLM 0.6.3。单卡时权重同步走 `hf` 格式，0.7 以上的分支只实现了 `dtensor`，第一次生成就抛 `NotImplementedError`（[fsdp_workers.py](../../../verl/workers/fsdp_workers.py) 的单卡分支和 [fsdp_vllm.py](../../../verl/workers/sharding_manager/fsdp_vllm.py) 的 `__enter__`，源码推断，未实跑）。也不要装 `0.6.3.post1`：rollout 按字符串比较版本，会被分到 0.7 的分支后报「版本不支持」。
- 模型要先下载到本地目录再填 `LEARN_MODEL_PATH`，仓库没有这一步。
- 先确认 firejail 能用。没装时原版判分器不报错，格式合格的回答全部记 0.0，训练照跑却没有正确性信号；开训前用一个正确解和一个错解做探针，分数不能全是 0（本机用 [verify_reward_manager_cpu.py](../source-supplements/verify_reward_manager_cpu.py) 看到过这个现象）。
- 只跑 1 步时，第一次更新的学习率是 0，参数不会变；接着跑下面的恢复命令再更新一次，才看得到参数变化。

保存配置展开结果与stdout/stderr。可先在同样参数后加 `--cfg job --resolve` 检查Hydra配置；它不是训练，在 Mac 上也能做：在仓库外建 venv，装 torch 2.4.0、tensordict 0.5.0、ray 2.40.0、hydra-core 1.3.2、transformers 4.47.1、torchdata 0.8.0、codetiming、pandas 后，用 venv 的 python 执行（2026-10-15 实测通过）。若prompt超过上限、model不支持、backend故障或内存不足，停在具体失败处改资源/配置并重新核对；不盲目重跑、不随意降低超时。

训练入口默认在更新前执行_validate，得到greedy一回答的raw reward baseline；这个值不能替代04定义的独立多候选correctness/pass@k baseline。默认日志/DataProto不完整保留候选、stop reason、所有policy版本，实跑应另记录这些字段或在自己的实验副本加观察，不声称源代码已提供全部trace。

## 5. 更新、同步与恢复具体看什么

| 检查 | 应关联的实跑证据 | 反例与停止条件 |
|---|---|---|
| 数据加载/生成 | Task/GT版本、实际Q×G、候选与UID、checkpoint/tokenizer | 数据路径或题目错配先修；全组同分意味着可能无outcome信号 |
| 参数更新 | 同一参数前后差异、grad/lr、optimizer step与run/step | loss或grad非零不证明更新，lr=0可参数不变 |
| 下一轮同步 | actor state与vLLM加载内容/版本、manager时序、固定输入数值 | 随机文本变了不证明同步；发现旧权重停更新链排查 |
| 保存 | global_step_x/actor各rank model/optim/extra shards、trainer data.pt与step | 只有目录名不证明内容完整 |
| 恢复 | 实际读入路径、step/optimizer/RNG/dataloader、开发验证 | 缺dataloader会从头开始；不保证生成引擎位级延续 |

恢复时先看 [_load_checkpoint](../../../verl/trainer/ppo/ray_trainer.py)：本版本auto找default_local_dir最新global_step；显式模式将 `trainer.resume_mode` 设为实际含global_step_的目录路径，并保持 `trainer.resume_from_path=False`。设为True且默认目录有latest时会走另一分支，显式目录可能不覆盖latest。不要套另一个版本的resume_path字面值。将总步数/epochs增加到能继续一批，再核对step与状态；代码支持路径不证明恢复已执行。

在上面定义函数的同一Bash会话，核对实际checkpoint位置后继续一步：

```bash
export LEARN_RESUME_MODE="$LEARN_RUN_DIR/global_step_1"
export LEARN_EPOCHS=2
export LEARN_TOTAL_STEPS=2
test -d "$LEARN_RESUME_MODE/actor"
test -f "$LEARN_RESUME_MODE/data.pt"
learn_smoke
```

如果实际保存的step不是1，使用真实目录，不伪造文件名。先比对恢复日志/状态与新增step，再看固定开发验证。该恢复命令同样待目标环境实跑。

参数差异与数值观察应写入自己的实验副本/报告；本包不替你生成模型日志。先验收单步，再扩大；可信效果路线按04冻结baseline、题集、判分与采样，报告配对、重复波动、成本和停止条件。

## 当前可核验与条件项

标准库准备、真实来源/ID、JSONL读回、具体batch算术与拒绝对照可在本机复做。prepare还从本页读取单rank标量配置，执行原版 `_validate_config` 的纯方法进行micro新旧字段等检查；没有导入trainer或初始化模型/CUDA，也不代替完整Hydra解析：

```bash
$PY -B RLVR学习资料/examples/training/test_training_prepare.py
```

Parquet分支的读回断言在代码中，现场结果待有pyarrow环境；本机未安装、未导出。Linux/CUDA和未知代码后端、真实baseline/update/sync/resume、泛化结论均不是本包已执行事实。它们有具体操作与证据标准，按所选学习范围完成。
