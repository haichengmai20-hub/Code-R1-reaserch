# 框架搭建：为什么不能直接用本仓库的 fork，移植了什么

## 卡在哪

本仓库是 verl 0.2.0.dev 的 fork，`setup.py` 钉 `vllm<=0.6.3`，作者 README 装 0.7.3，transformers 4.47。Qwen3-8B-Base 与 Qwen3-8B 同架构，模型类型都是 `qwen3`，从 transformers 4.51.0 才有，vLLM 从 0.8.4 起注册、官方推荐 ≥ 0.8.5；旧版加载报 `ValueError: The checkpoint you are trying to load has model type `qwen3` but Transformers does not recognize this architecture…`。本 fork 的 rollout 按版本号分派，只认 0.6.3 和 0.7.x 两条路径，0.8.5 进不去。

三个选项：

| 选项 | 代价 | 决定 |
|---|---|---|
| 换回 Qwen2.5-7B-Instruct，本 fork 原样跑 | 零改动；但不是目标模型 | 否 |
| 在本 fork 上硬升 transformers 4.51 + vLLM 0.8.5 | 权重同步走 `vllm_rollout_spmd` 的内部路径，0.8 的引擎结构变了，要改源码且无人验证 | 否 |
| 迁到上游 verl v0.4.1（官方支持 Qwen3、vLLM 0.8.5），把本仓库的判分和数据代码移过去 | 移植 5 个文件；配置键大体相同 | **是** |

## 移植清单

| 本仓库 | 放到 v0.4.1 的位置 | 改动 |
|---|---|---|
| `verl/utils/reward_score/coder1/` | 同路径 | 不改 |
| `verl/utils/reward_score/coder1_plus/` | 同路径 | 不改 |
| `verl/utils/reward_score/__init__.py` 的 `code` / `code_plus` 分支 | v0.4.1 的 `default_compute_score` | 加两个 `elif`，签名相同 |
| `verl/workers/reward_manager/prime_plus.py`、`infra_policy.py` | 同路径 | 要适配 v0.4.1 的管理器协议：`__init__` 增加 `reward_fn_key="data_source", **kwargs`；`__call__(self, data, return_dict=False)`，`return_dict=True` 时返回 `{"reward_tensor": …}`（v0.4.1 的验证和训练都这样调用）；`_default_compute_score` 改成 `default_compute_score`；再加 `@register("prime_plus")` |
| `examples/data_preprocess/coder1_plus*.py` | 同路径 | 不改 |

训练入口不用改：v0.4.1 按 `reward_model.reward_manager` 的名字查注册表。run-01 用的是 v0.4.1 自带的 `prime` 管理器：64 个判题进程、异常时只打印一行 `[Error] Task failed: …` 并记 0.0、每批结束打印 `[Shutdown] 64 subprocess(es) terminated.`。

另外两条 v0.4.1 与本 fork 不同、会直接影响配置的事实：`_validate_config` 断言 `data.train_batch_size >= actor.ppo_mini_batch_size`，所以 Code-R1 原脚本里 mini-batch 写 256（GPUS×8×4）的写法在 v0.4.1 启动即报错，只能写 ≤ 16；开 chunked prefill 时断言 `max_num_batched_tokens >= max_prompt_length + max_response_length`，回答上限改 8192 时必须把它提到 ≥ 10240。

## 第一天的四项检查（都在正式启动前）

1. `--cfg job --resolve` 展开配置，核对 `max_response_length`、`ppo_mini_batch_size`、`reward_manager`、`default_local_dir`（存成 `config/run-01.resolved.yaml`）。
2. 跑 verl 自带的 gsm8k 单卡 smoke（Qwen2.5-0.5B，4 题 × 4 回答，1 步），确认 Ray、FSDP、vLLM 同步在这台机器上能起。顺带单独起一次 vLLM（`VLLM_LOGGING_LEVEL=INFO`，训练时 verl 把它压到 WARN，日志里看不到）读显存剖析：TP=2 时每卡权重 7.63 GiB，可用 39.82 GiB（80 GB × 0.5），KV cache 约 30.4 GiB ≈ 44 万 token，6,144 token 的请求并发约 72 路；这是第一环显存算式的对照数（smoke 时从 vLLM 日志读的，推演数字）。
3. firejail 探针：`CODER1_EXEC=firejail` 下给 LeetCode 53 的四个固定候选判分，期望 1.1 / 0.1 / 1.1 / −1.1；没装 firejail 时原版管理器不会报错，分数会悄悄全变 0.0。结果在 `../env/probe_scorer.log`。
4. Qwen3-8B-Base 专项：
   - 聊天模板：Base 的 tokenizer 自带与 Qwen3-8B 同一套 ChatML 模板（含 `enable_thinking` 开关；smoke 时核对 `tokenizer_config.json` 里的 `chat_template`，实跑前再核对一次），assistant 段不预填任何东西，`<think>` 得模型自己写。这和 Code-R1 的格式门（回答必须以 `<think>` 开头）吻合，但 Base 没学过这个格式：5 条样例里 3 条以 `<think>` 开头，1 条直接写代码，1 条用 markdown 标题分段。`enable_thinking` 对 Base 只是模板行为：保持默认（不传或 True）时 assistant 段不预填，`<think>` 由模型自己写；传 `enable_thinking=False` 会在 prompt 里预填空的 think 块，response 不再以 `<think>` 开头、被格式门判 −1.1，所以不能传 False。Base 自己不会写空 think 块，所以不存在「模型习惯和格式门打架」这一种冲突；模板是否自带、关掉是否预填，smoke 时核对 `tokenizer_config.json` 的 `chat_template`。格式要靠系统提示和训练 reward 从头学。
   - `<think>`（151667）、`</think>`（151668）在 Base 的词表里就是普通的 added token，`skip_special_tokens=True` 解码后仍在，格式门能看到它们；`<|im_end|>`（151645）是模板的回合结束符，解码时被去掉，不影响 `</answer>` 结尾的检查。Base 的 `generation_config.json` 里 eos 是否包含 `<|im_end|>` 要在 smoke 时核对（Qwen 系列 Base 的 eos 常是 `<|endoftext|>` 151643）：不包含就把它加进 rollout 的停止符，否则回答写完 `</answer>` 后不停、一路写到上限。本次按「`<|im_end|>` 是停止符」写。
   - Base 会循环到上限：5 条样例 900–4,096 token，最长那条把同一段推理反复重写、顶到 4096 被截断。这是 run-01 把上限设 4096 的风险点，当时记为「先跑看截断率」。
