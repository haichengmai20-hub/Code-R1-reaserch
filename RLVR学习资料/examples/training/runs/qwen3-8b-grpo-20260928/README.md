# qwen3-8b-grpo-20260928：Qwen3-8B-Base 两轮 GRPO 的运行记录

> 性质说明见 [PROVENANCE.md](PROVENANCE.md)：这是推演产物，按真实运行记录的写法记，方便当作实跑来读。

## 一页摘要

| | run-01 | run-02 |
|---|---|---|
| 时间 | 2026-09-28 08:30 → 10-01 12:34，76.1 h | 2026-10-03 07:02 → 10-07 11:37，100.6 h |
| 机器 | 4×A800 80GB 单机，128 逻辑核，512 GiB 内存，根盘 300 GB，数据盘 3.5 TB | 同 |
| 模型 | Qwen/Qwen3-8B-Base（预训练模型，没经过 SFT 或 RL，架构参数与 Qwen3-8B 相同；`<think>…</think><answer>…</answer>` 的格式只靠 Code-R1 的系统提示从头学） | 同，从头训，不接 run-01 |
| 数据 | code-r1-plus-2k-leetcode2k 去污染版，1889 题 | 用 run-01 第 192 步 checkpoint 采样 8 条/题，去掉全对 198 题、全错 155 题，剩 1536 题 |
| 关键配置 | 16 题 × 16 回答，回答上限 4096，lr 5e-7，KL 0.001，mini-batch 16 题（每步 1 次更新），v0.4.1 原版 prime 管理器，checkpoint 放根盘 | 回答上限 8192（`max_num_batched_tokens` 同步提到 12288），mini-batch 4 题（每步 4 次更新），prime_plus 管理器，checkpoint 放数据盘、只留 3 份，epochs 提到 4（筛题后一轮只有 96 步），其余同 |
| 每步耗时 | 平均 856 s：生成 162 s（19%）、更新 438 s（51%） | 平均 1,131 s：生成 285 s、更新 582 s |
| 训练 reward | -0.409 → 0.559（第 1 步格式失败 54%：标签缺失 30%、循环到 4096 被截断 23%） | -0.174 → 0.709 |
| 验证 val-core（贪心，受训练回答上限影响） | -0.224 → 0.595（通过 48 → 111 / 200） | -0.096 → 0.656（52 → 116） |
| dev 200 题贪心 pass@1（离线协议，8192） | 基线 0.380 → 0.615（格式错 18.5% → 2.5%） | 0.635（对 run-01 净胜 4 题，分不出） |
| dev 采样 pass@1 / pass@8（3 seed） | 0.352 → 0.587 / 0.768 → 0.890 | 0.640 / 0.907 |
| held-out LCB 142 题贪心 pass@1 | 0.134 → 0.246（配对净胜 16 题，远超 seed 噪声） | 0.317（净胜 26 题；对 run-01 净胜 10 题、只 1 题变差） |
| held-out 采样 pass@1 / pass@8 | 0.102 → 0.231 / 0.366 → 0.582 | 0.291 / 0.695 |
| 事故 | 第 129–134 步根盘写满，判题被记成 0.0 | 无；prime_plus 作废了 2 组 |
| 费用（¥10/卡时估） | ¥3,043（$423） | ¥4,023（$559） |

## 时间线

- **09-27** 环境：Qwen3 在本仓库 fork 上装不了（transformers 4.47 不认识 `qwen3`），迁到 verl v0.4.1 并移植判分代码，记录在 [notes/framework_port.md](notes/framework_port.md)。生成数据 parquet、去污染（去掉 1 题）、写评测协议 [eval/protocol.md](eval/protocol.md)。
- **09-28 08:21** 节点就绪，`nvidia-smi` 存档；判分探针四个候选分数 1.1 / 0.1 / 1.1 / −1.1（[env/probe_scorer.log](env/probe_scorer.log) 是本机真实输出，节点上 firejail 跑同样探针），确认判分不是全 0。
- **09-28 08:30** run-01 启动（[config/run-01.sh](config/run-01.sh)）。第 0 步验证 -0.224：200 题贪心只过 48 题，94 题（47%）标签缺失或循环到 4096——Base 没学过 `<think>` 格式。第 1 步 reward -0.409，格式失败 54%（标签缺失 30%、截断 23%），回答均值 1,787 token。
- **09-28 23:18** 第 64 步：reward 0.393，格式失败率从 54% 降到 8%（标签缺失 30% → 1%，截断 23% → 7%），回答从 1,787 token 缩到 1,005（第 30–64 步在 900–1,000 的谷底，最低 893，循环被罚掉）。存第一份 checkpoint，98 GB。
- **09-29 14:00** 第 128 步存第二份 checkpoint，根盘 99%。
- **09-29 14:00–15:20** 第 129–134 步：`[Error] Task failed: [Errno 28] No space left on device`，判题故障被 prime 记成 0.0，reward 曲线掉到 0。**15:21** 清盘，第 135 步（15:34 结束）恢复。处理见 [notes/incident_report.md](notes/incident_report.md)。
- **10-01 12:34** run-01 结束，320 步。离线评测 dev 0.615、LCB 0.246（基线 0.380 / 0.134）。
- **10-02** 归因（[attribution/reward_decomposition.md](attribution/reward_decomposition.md)）：第 1–64 步 reward 增量 0.802 里 0.678（85%）来自学会格式，之后的增益全靠正确率、越来越慢；held-out 净胜 16 题，远超噪声；用第 192 步 checkpoint 对 1889 题各采 8 条，198 题全对、155 题全错，这些题没有组内信号。决定改五处，见 [notes/decision_log.md](notes/decision_log.md) 第七环；按采样通过率筛题。
- **10-03 07:02** run-02 启动，同样从 Base 从头训。**10-07 11:37** 结束。离线评测 dev 0.635、LCB 0.317；对 run-01 在 dev 上分不出（净胜 4 题），在 held-out 上净胜 10 题、pass@8 0.582 → 0.695。
- **10-09** 写 [eval/report.md](eval/report.md)。

## 目录

| 路径 | 看什么 |
|---|---|
| `config/` | 两轮的启动脚本和展开后的配置 |
| `env/` | 节点、版本、`pip freeze` 节选、判分探针 |
| `data/` | 数据清单（sha256）、去污染结果 |
| `logs/run-0*/` | `metrics.jsonl` 每步指标（verl v0.4.1 字段名）、`validation.jsonl`、`grpo.log.gz` 控制台全文、`grpo.head.log` 开头到第 1 步结束、`incident_step129-135.log` 事故窗口、`scenario_internal.jsonl` 内部量 |
| `rollouts/` | `rollout_data_dir` 落盘的样本：run-01 第 131 步（事故中）、run-02 第 320 步 |
| `wandb/` | 离线 run 目录：`config.yaml`、`wandb-summary.json`、`wandb-metadata.json`、`history.export.jsonl` |
| `checkpoints/` | 两轮的 checkpoint 文件清单和 `latest_checkpointed_iteration.txt` |
| `eval/` | 协议、每题明细、汇总、`tables.md`（生成的结果表）、配对比较、报告 |
| `curves/` | 10 张曲线图 |
| `attribution/` | reward 拆分、全同组统计 |
| `notes/` | 决策日志、事故报告、框架移植、你的决策日志（`your_decisions.md`，带学时逐环续写） |

读法：先看 `curves/reward.png`（从负值起步）、`curves/response_length.png`（先降后升）和 `curves/val_core.png`，再对照 `logs/run-01/metrics.jsonl` 第 129–135 步，再看 `eval/report.md`。Base 起点下前 64 步的增益大头是格式，拆分看 `logs/run-0*/scenario_internal.jsonl` 的格式失败率和截断率。学习材料在 [05_实跑推演_Qwen3-8B.md](../../../../05_实跑推演_Qwen3-8B.md)。
