# 这个目录里的文件是怎么来的

这是**推演**（沙盘），2026-10-16 做、2026-10-19 去掉 TACO 后重做：把你在 [05](../../../../05_实跑推演_Qwen3-8B.md) 七环里做的决定（记在参考目录的 [notes/your_decisions.md](../qwen3-8b-grpo-20260928/notes/your_decisions.md)）当作两轮真实训练的配置，按真实运行会留下的产物做出来：your-01 用来诊断，your-02 按诊断改三处（tmpfs、筛题、换验证集），用 held-out 验证；计划里的 clip-higher 执行前去掉了。起点是 Qwen3-8B-Base。推演里的日期设在 2026-10-17 到 10-21（10-17 写协议，10-18 08:00 your-01 启动，10-19 评测与筛题，10-20 14:00 your-02 启动，10-21 结束并评测）。

与参考目录 [qwen3-8b-grpo-20260928](../qwen3-8b-grpo-20260928/) 的关系：同一套合成器（`../../scenario_qwen3_8b.py` 的参数模型、真实判分和评测模型），换成你的假设（6144 / n=8 / 200 步 / mini-batch 4 / prime_plus / 熵 0 / checkpoint 在数据盘；第二轮 tmpfs / 筛题 / 换验证集）。数据与参考 run-01 同一份 LeetCode 去污染版 parquet，所以 rollout 落盘里的题目和候选代码能用真实判分代码做。

| 文件 | 性质 | 怎么来的 |
|---|---|---|
| `config/your-0*.sh` | 推演，未在任何机器上执行 | 由参考 run-01 的覆盖项改出；your-02 多 `resume_mode=auto` 和复制 checkpoint 的几行 |
| `config/your-0*.resolved.yaml` | 推演 | v0.4.1 默认值加覆盖项合成，不是 `--cfg job` 的真实输出 |
| `data/filter_manifest.json` | 推演 | 第七环筛题的账：用 your-01 第 200 步的全对 / 全错比例推出去掉的题数、剩余题数、验证集切分 |
| `logs/your-0*/metrics.jsonl`、`validation.jsonl` | 推演 | `../../scenario_yourplan.py` 的参数模型生成，字段名与 verl v0.4.1 一致；批均值按整数计数算（128 条、200 题） |
| `logs/your-0*/grpo.log.gz`、`grpo.head.log`、`incident_step65-66.log`、`after_save_step256-258.log` | 推演 | 完整控制台（your-01 983 行、your-02 907 行）：打印语句按 verl v0.4.1 和 prime_plus，含进度条、验证块、存盘行、续训的 `Load from checkpoint folder` 行、沙箱超时与作废汇总；没有逐样本判分块（prime_plus 不打印，与参考 run-02 相同）。head 是到第二步的节选 |
| `logs/your-0*/scenario_internal.jsonl` | 推演 | verl 不记的内部量：格式失败率、正确率、故障重试与作废组数、全同组比例 |
| `rollouts/your-0*/*.jsonl.gz` | 混合 | 字段按 v0.4.1 的 `rollout_data_dir` 落盘格式；prompt 是真实题目的真实聊天模板；回答的代码是参考解（通过）或**经本机真实判分代码验证会失败的变异体**（不通过），每条的分数与该步指标行的整数计数一致；作废的组整组 0.0；思考文本是模板合成 |
| `wandb/offline-run-*/files/*` | 推演 | 按 wandb 离线目录格式写出；没有 `.wandb` 二进制；`history.export.jsonl` 相当于用 wandb API 导出的历史 |
| `checkpoints/your-0*/MANIFEST.txt` | 推演 | 文件名按 verl FSDP checkpoint 布局，大小按 Qwen3-8B-Base 的参数量算（8.19B）；your-01 64 / 128 / 192 / 200 四份全留，your-02 另含复制来的起点（无 `data.pt`） |
| `eval/dev_*/*.jsonl`、`eval/lcb_*/*.jsonl`、`eval/summary.json`、`eval/tables.md` | 推演 | 与参考同一评测模型：每题一个潜在难度，三个模型共用；dev 的 task_id 是 v2 test 的真实题号；LCB 142 题题单与参考目录相同（题号按平台格式合成） |
| `curves/*.png` | 推演 | 从两轮的 metrics.jsonl 画在同一条轴上；your-02 的验证曲线用的是另一个集合（切出的 200 题），不能和 your-01 的 v2 test 曲线直接比高低 |
| `attribution/reward_decomposition_*.json` | 推演 | 用内部量按 1→64→128→200 和 201→256→320→400 做格式 / 正确率分解 |
| `summary.json` | 推演 | 两轮的时长、费用、关键指标、事故步、筛题账、评测汇总 |
| `README.md`、`eval/protocol.md`、`eval/report.md`、`notes/*`、`attribution/*.md` | 推演叙述 | 按上面的数字写 |

重算：`$PY -B RLVR学习资料/examples/training/scenario_yourplan.py`（仓库根目录；要 PNG 就用装了 matplotlib 的 venv；会在本机执行真实判分，约两三分钟）。改脚本开头的假设后数据文件一起变，叙述文件要手动对数，然后跑 `check_scenario_docs.py`。
