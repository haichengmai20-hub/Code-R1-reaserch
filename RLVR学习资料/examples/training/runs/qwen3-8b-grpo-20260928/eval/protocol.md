# 评测协议（2026-09-27 21:10 预登记，训练开始前冻结）

## 题集

| 集合 | 来源 | 题数 | 用途 | 训练中能看吗 |
|---|---|---|---|---|
| dev | LeetCodeDataset v2 test（补强版 GT，带运行前缀） | 200 | 每 16 步贪心验证；离线评测；选 checkpoint | 能，只看汇总 |
| held-out | LiveCodeBench 中 contest_date ≥ 2025-05-01 的题（晚于 Qwen3 系列的预训练数据截止），平台 leetcode / atcoder / codeforces | 142 | 最终报告；训练中不跑，不看 | 不能 |
| extra | HumanEval+、MBPP+（EvalPlus 0.3.1，贪心） | 164 / 378 | 与 Code-R1 作者表格同口径对照 | 不能 |

去污染：训练集与 dev 按题号无交集（构造时排除）；8 元词组重叠 1 对，已从训练集去掉（见 ../data/data_manifest.txt）；held-out 按日期隔离。

## 解码

| 口径 | 参数 | 产出 |
|---|---|---|
| 贪心 pass@1 | temperature 0，max_tokens 8192，聊天模板用 Qwen3-8B-Base 自带的那套（与训练相同；`enable_thinking` 保持模板默认，assistant 段不预填，`<think>` 由模型自己写） | 每题一条，通过/失败/格式错 |
| 采样 pass@1、pass@8 | temperature 0.6，top_p 0.95，top_k 20，min_p 0，n=8，seed ∈ {1, 2, 3}，max_tokens 8192 | 每题每 seed 8 条；pass@1 取 8 条平均，pass@8 取任一通过 |

max_tokens 8192 对三个模型相同，与训练用的 4096 / 8192 无关；训练中的验证用 4096（run-01）或 8192（run-02），所以 val-core 的数字不与本协议直接比较。三个模型（Qwen3-8B-Base 基线、run-01、run-02）用同一套系统提示和聊天模板；Base 没学过 `<think>…</think><answer>…</answer>`，格式错按下面的规则单独计数，不并进失败。

## 判分

- 补强版受保护协议（`coder1_plus`），firejail 后端，30 秒限时；格式门与训练一致：回答去掉首尾空白后以 `<think>` 开头、`</answer>` 结尾，答案代码取最后一个 `<answer>` 里的全部代码块。
- 通过 = 全部断言通过；格式错单独计数，不算进失败里再分析。
- 判题故障（infra）重试 2 次，仍失败的题记「未判」，报告里单列。

## 比较

- 同一题集、同一 checkpoint、同一解码参数下，基线与训练后按题配对：赢 / 输 / 平。
- 采样口径报告 3 个 seed 的均值和样本标准差；提升小于 2 倍标准差不下结论。
- checkpoint 只用 dev 选；held-out 只在最终报告跑一次。

## 结果文件

- `dev_greedy/{base,run-01,run-02}.jsonl`、`lcb_greedy/{base,run-01,run-02}.jsonl`：每题一行（贪心）
- `dev_sampled/{base,run-01,run-02}.jsonl`、`lcb_sampled/{base,run-01,run-02}.jsonl`：每题一行，3 个 seed 的通过条数
- `lcb_postcutoff_142.problems.jsonl`：held-out 题单（平台、题号、日期、难度）
- `summary.json`（含 EvalPlus 汇总）、`paired_comparisons.csv`、`report.md`
