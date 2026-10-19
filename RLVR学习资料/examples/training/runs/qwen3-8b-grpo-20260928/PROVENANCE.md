# 这个目录里的文件是怎么来的

这是一次**推演**（沙盘）：2026-10-16 做的，推演里的日期设在 09-27 到 10-09，机器按个人研究能租到的 4×A800 80GB，把「Qwen3-8B-Base（预训练模型，没经过 SFT 或 RL）在 Code-R1 流水线上做两轮 GRPO」按真实运行会留下的产物逐一做出来，用来练从机器到归因的全流程决策。没有真的租卡、没有真的训练。写进简历或面试时，说法是「我做过一次完整的实跑推演」，不是「我训练过」。

| 文件 | 性质 | 怎么来的 |
|---|---|---|
| `config/run-0*.sh` | 推演，未在任何机器上执行 | 按 verl v0.4.1 的配置键手写；键名和断言对照上游 `ppo_trainer.yaml`、`ray_trainer.py` |
| `config/run-0*.resolved.yaml` | 推演 | 按 v0.4.1 默认值加本次覆盖项合成，不是 `--cfg job` 的真实输出 |
| `env/probe_scorer.log` | **真实输出** | 本机用仓库里真实的 `coder1` 判分代码执行四个固定候选得到；Linux 节点上用 firejail 跑同样探针，格式相同 |
| `env/versions.txt`、`nvidia-smi.txt`、`pip_freeze.txt` | 推演 | 按 4×A800 节点与 verl v0.4.1 镜像的常见版本手写；pip_freeze 是节选 |
| `data/data_manifest.txt` | **真实**（筛题版除外） | 原始版和去污染版 parquet 是本机真实生成的（`coder1_plus.py --parquet`），sha256 是真的，文件在被忽略的 `data/` 下；筛题版需要 GPU 采样，是推演 |
| `data/decontamination_v2test.json` | **真实输出** | `coder1_plus_selection.py decontaminate` 对真实训练集与 v2 test 的结果 |
| `logs/*/metrics.jsonl`、`validation.jsonl` | 推演 | `../../scenario_qwen3_8b.py` 的参数模型生成，字段名与 verl v0.4.1 一致 |
| `logs/*/grpo.log.gz`、`grpo.head.log`、`incident_*.log` | 混合 | 打印语句按 verl v0.4.1 和本仓库判分代码；第 1 步的 256 个判分块是**用真实判分代码在本机执行参考解和变异体得到的**；思考文本是模板合成；第 2 步以后的判分块按日志轮转惯例裁掉 |
| `logs/*/scenario_internal.jsonl` | 推演 | verl 不记的内部量（格式失败率、正确率、故障比例、全同组比例），归因时用 |
| `rollouts/*/*.jsonl.gz` | 混合 | 字段按 v0.4.1 的 `rollout_data_dir` 落盘格式；prompt 是真实题目套 Qwen3-8B-Base 自带的聊天模板（与 Qwen3-8B 同一套 ChatML，assistant 段不预填；实跑前核对）；回答的代码是参考解或变异体，思考文本是模板合成 |
| `wandb/offline-run-*/files/*` | 推演 | 按 wandb 离线目录格式写出；没有 `.wandb` 二进制；`history.export.jsonl` 相当于用 wandb API 导出的历史，不是原生文件 |
| `checkpoints/*/MANIFEST.txt` | 推演 | 文件名按 verl FSDP checkpoint 布局，大小按 Qwen3-8B-Base 的参数量算（8.19B，与 Qwen3-8B 相同） |
| `eval/protocol.md` | 真实的协议文本 | 训练前写好的评测协议 |
| `eval/*_greedy/*.jsonl`、`*_sampled/*.jsonl`、`summary.json` | 推演 | 每题一个潜在难度，三个模型共用，按概率抽样；dev 的 task_id 是 v2 test 的真实题号；LCB 子集的题号是按平台格式合成的 |
| `curves/*.png` | 推演 | 从 metrics.jsonl 画出 |
| `notes/*.md`、`README.md`、`eval/report.md`、`attribution/*` | 推演叙述 | 按上面的数字写的运行记录、决策、事故和归因 |

重算：`$PY -B RLVR学习资料/examples/training/scenario_qwen3_8b.py`（仓库根目录；想要 PNG 曲线就用装了 matplotlib 的 venv）。改脚本开头的假设后，logs、eval、wandb、checkpoints、attribution 下的数据文件一起变；README、notes、eval/report.md 和 05 里引用的数字要手工同步，`check_scenario_docs.py` 会检查两边是否一致。
