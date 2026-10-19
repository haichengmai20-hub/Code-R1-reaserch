# your-02：计划与执行记录（第七环的决定，2026-10-16 定、10-19 去 TACO、去 clip-higher 后按 LeetCode 版执行）

决定记录在 [your_decisions.md](../../qwen3-8b-grpo-20260928/notes/your_decisions.md) 第七、八环。启动脚本 [config/your-02.sh](../config/your-02.sh) 由 your-01 改出，改动只有下面四处加一处预算（计划里的 clip-higher 执行前去掉，原因在表里）；执行后的数字在 [README.md](../README.md) 的「诊断 → 修 → 验证」表和 [eval/report.md](../eval/report.md)。

## 改什么、为什么、验证到了什么

| 改动 | 对着 your-01 的哪条证据 | 计划时的预期 | 执行后看到的 |
|---|---|---|---|
| 判题临时目录 `/data/tmp/judge` → `/dev/shm/judge`（tmpfs） | 第 65 / 129 / 193 步存盘后沙箱启动超时，作废 17 组；`timing_s/reward` 64–69 秒 | 存盘后一步不再有 `[prime_plus] infra error` | 第 257 / 321 / 385 步 `timing_s/reward` 25.0 / 23.5 / 27.5 秒，`neutralized_groups` 全程 0（[after_save_step256-258.log](../logs/your-02/after_save_step256-258.log)） |
| 筛题：用 your-01 第 200 步对 1889 题采样 8 条/题，只留通过率在 1/8–7/8 之间的题 | 第 200 步每步 9.6% 的组全对、5.7% 全错，advantage 为 0 | 每步有效组数上升；训练 reward 的绝对值会跳一下（题变了） | 去掉全对 158、全错 124，剩 1607；第 201 步全对 5.0%、全错 3.6%；reward 几乎没跳（0.569 → 0.553），因为去掉的两类题数量相当；到第 400 步全对又回到 10.4%——模型变强后筛题要重做 |
| ~~`clip_ratio_high` 0.2 → 0.28~~（计划有，执行前去掉） | 熵 1.14 → 0.30，但前 60 步是格式锁定、之后缓降，pass@8 还在涨——没有被诊断出的问题，只是预防 | 原预期熵止跌、clipfrac 降 | 去掉的理由：两轮没有消融预算，没有证据的预防性改动会让归因更乱。执行结果：熵 0.29 → 0.28（第 300 步附近最低 0.26）没再塌，`pg_clipfrac` 均值 3.5% 不变，pass@8 0.870 → 0.898（dev）、0.596 → 0.693（LCB）。掉到 0.2 以下或 pass@8 掉头再上它 |
| 从 your-01 第 200 步续训 200 步（`resume_mode=auto`，`total_training_steps=400`） | 筛题用的就是第 200 步的能力 | 第 201 步有台阶，之后比较斜率不比较绝对值 | 控制台 `Load from checkpoint folder: /data/ckpt/your-02/global_step_200`，`global_steps` 从 200 起；台阶很小（见上）；201–400 步 reward +0.111，全部来自正确率（格式已学完） |
| 验证集换成筛后切出的 200 题（val200），从训练集里拿掉 | 验证集即评测集，用它选 checkpoint 又用它报结果 | dev 和 LCB 只在第 400 步评一次，报告以 LCB 为准 | val200 从 0.634 涨到 0.736（最高 0.741 在第 368 步）；held-out 上 your-02 对 your-01 净胜 8 题（9 赢 1 输，+5.6 个点） |
| 预算 | — | 筛题采样约 1 小时 ¥40（1889 题 × 8 条 ≈ 23M token，vLLM 独立服务 4 卡 TP=1 约 10k tok/s），训练 200 步约 26 小时 ¥1,040（回答变长，每步 469 s），评测 ¥60；合计约 ¥1,150 | 训练 26.1 小时、¥1,043 |

## 筛题怎么做（用 verl 自己的验证通道，不另写采样程序）

1. 采样：`trainer.val_only=True`，`data.val_files=train.parquet`（1889 题全量，不用子集），`actor_rollout_ref.rollout.val_kwargs.n=8`、`do_sample=True`、`temperature=1.0`，`trainer.validation_data_dir=/data/val_dump/your-01-step200`，从 `/data/ckpt/your-01/global_step_200` 加载。4 卡 TP=2 两个引擎、每引擎 64 路并发时约 1,500 tok/s，1889 × 8 × 1,500 token ≈ 23M token 要 4 小时，太慢；改用 vLLM 独立服务（4 卡各一个 TP=1 引擎、显存 0.9、并发 256）吞吐约 10k tok/s，40 分钟，加上起服务、落盘和统计按 1 小时留。
2. 统计：把落盘的每题 8 条判分结果汇成 `{task_id: {"n": 8, "c": 通过数}}`（二十行脚本），交给 `examples/data_preprocess/coder1_plus_selection.py difficulty --stats pass_stats.json --out keep.json`（默认 `0 < c/n < 1`，正是「去掉全对全错」）。
3. 落地：`coder1_plus.py --keep_task_ids keep.json --parquet` 写 `code-r1-plus-2k-leetcode2k-filtered-your01/train.parquet`；再从 keep 里随机切 200 题写 `val200.parquet`，训练集去掉它们（1407 行，87 批一个 epoch）。账记在 [data/filter_manifest.json](../data/filter_manifest.json)。

## 不改的，和为什么

- n 仍是 8：筛题后每组里有对有错的概率上去了，不必靠 n=16 补信号；每步时间不变。
- lr、KL 系数、mini-batch 4、6144 上限不动：第五环没有任何一条证据指向它们（KL 0.08 是 Base 起点正常的距离，第 200 步截断率 3%）。
- 起点不换：仍是 Qwen3-8B-Base 训出的 your-01 第 200 步，这一轮的归因都建立在它上面。
- clip-higher 不加：见上表；熵是观察项，不是这一轮的改动。

## 续训的两个坑（实跑前核对）

- v0.4.1 的 `_load_checkpoint` 会顺带恢复 `data.pt` 里的 dataloader 状态；数据集换了就要去掉它，脚本里已处理。
- `total_training_steps=400` 配 `resume_mode=auto`：global_steps 从 200 起，再走 200 步；lr 是常数调度，不受影响。如果 `latest_checkpointed_iteration.txt` 没写对，训练会从第 1 步重来并覆盖目录。

## 没做的消融

三处改动加续训一起上，your-02 对 your-01 的 +5.6 个点里各占多少分不开；能单独验证的只有事故（存盘后判题时间、作废组数）。要分开就一次改一处，代价是 3 轮、约 ¥3,000。
