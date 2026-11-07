# Code-R1-reaserch 项目 Context（2026-11-01核验）

## 结论

这个项目已经可以开始“学习与源码研究”：10 篇主学习文档、贯穿样例、源码补强、数据清单、CPU 契约检查和两套 Qwen3-8B 实跑推演产物都在仓库内，并且 Git 对象完整、本地 Markdown 链接无缺失、LeetCodeDataset 子模块停在记录的 pinned commit。

不能把它当成“本机已可真实训练”。当前 Windows 环境缺少 torch/vLLM/Ray/pyarrow 等训练依赖；本机 GPU 是 6 GB 的 RTX 3060 Laptop，低于项目学习 smoke 路线建议的 24 GB；受保护判分沙箱需要 Linux firejail 或 macOS sandbox-exec，Windows 下没有后端。仓库里的 Qwen3-8B 两轮“训练记录”是明确标注的推演/沙盘，不是真实 GPU 训练证据。

## 项目身份与版本

- 主仓库路径：`D:/RL及harness/Code-R1-reaserch`。
- 当前分支：`main`，HEAD 是 `8ca9c92927ef75416b328dbe03a6024686aab93d`；`origin/main` 在 `a4002bdeadc39b04adde2c4c371c8d55fc4a871e`，因此本地 ahead 1。
- 原版 Code-R1 基线是 `443f8da8058d07051793eff875da117534fd1ba1`。学习包在此之上新增/扩展；除 `RLVR学习资料` 外，源码侧只新增 `coder1_plus`/`prime_plus` 相关文件，并在两个原版文件中加入分支：`verl/utils/reward_score/__init__.py` 增加 `code_plus` 路由，`verl/trainer/main_ppo.py` 增加 `prime_plus` 奖励管理器路由。
- `LeetCodeDataset` 是 pinned 子模块，父仓库记录 `f3519e6b7f42265dd4e4d35d4812e8c29b08111a`，子模块当前 HEAD 相同且工作区干净。
- `git fsck --no-dangling` 通过。当前 6 个已跟踪脚本只有 executable bit 变化，没有内容变化；另有未跟踪 `.claude/settings.local.json` 和一个 63,498,240 字节的 `Code-R1-reaserch.zip`。该 zip 有 ZIP 头但缺少标准 EOCD 结束记录，标准 `zipfile` 无法打开，不能当备份依赖；学习主链不依赖它。

## 学习入口与路线

主线只保留一条贯穿题：LeetCode 53 maximum-subarray。按顺序读：

1. [00_一条记录从原始Task到训练.md](00_一条记录从原始Task到训练.md)：一行 Task 到一步训练的七站全景。
2. [01_数据理解与GT构建.md](01_数据理解与GT构建.md)：Task、Tests/GT、metadata、逐条证据与练习。
3. [02_Verifier与任务池验收.md](02_Verifier与任务池验收.md)：格式/正确性/反作弊分数、沙箱边界、任务池读回。
4. [03_Rollout与GRPO机制.md](03_Rollout与GRPO机制.md)：同题多回答、uid、advantage、mask、logprob、loss。
5. [04_真实训练与独立评测.md](04_真实训练与独立评测.md)：更新、同步、checkpoint、恢复、独立 baseline 与归因。
6. [05_实跑推演_Qwen3-8B.md](05_实跑推演_Qwen3-8B.md)：按七环做 Qwen3-8B-Base 的决策沙盘，只作推演，不作真实训练证据。
7. [RLVR工业流程.md](RLVR工业流程.md) 和 [专题索引_任务构造与沙箱.md](专题索引_任务构造与沙箱.md)：主框架与专题跳转。
8. [路线评估_仅学习与实跑.md](路线评估_仅学习与实跑.md)：仅学习、单卡 smoke、小规模训练、完整复现的取舍。注意其中“本机没有 GPU”是旧环境结论；本次核验到本机有 6 GB GPU，但仍低于项目建议的 smoke 资源。

## 源码阅读地图

| 环节 | 先看 | 说明 |
|---|---|---|
| 原始数据构造 | `examples/data_preprocess/coder1.py` | `leetcode2k()` 与原版 prompt/GT 生成。 |
| 补强数据构造 | `examples/data_preprocess/coder1_plus.py`、`coder1_plus_tests.py`、`coder1_plus_oracles.py`、`coder1_plus_selection.py` | prelude/GT、测试审核与补充、去重和难度筛选。 |
| 原版判分 | `verl/utils/reward_score/coder1/` | 原版 `code` 判分与执行路径。 |
| 补强判分 | `verl/utils/reward_score/coder1_plus/` | 结构化协议、沙箱状态、GT 前缀契约。 |
| 奖励管理 | `verl/workers/reward_manager/prime_plus.py`、`infra_policy.py` | 基础设施异常重试、按 uid 作废组，避免故障被写成真实 0 分。 |
| GRPO 入口 | `main_grpo.sh`、`verl/trainer/main_ppo.py`、`verl/trainer/ppo/ray_trainer.py` | 原版训练入口、奖励管理器选择、训练循环。 |
| GRPO 数学 | `verl/trainer/ppo/core_algos.py` | `compute_grpo_outcome_advantage` 与 `compute_policy_loss`。 |
| 更新/同步 | `verl/workers/fsdp_workers.py`、`verl/workers/sharding_manager/` | actor 更新、权重同步与放置方式的源码依据。 |

## 数据与产物现状

- 仓库共有 658 个 tracked 文件，其中 `RLVR学习资料` 260 个。
- 学习资料下 37 个 Markdown 的本地链接全部存在。
- LeetCodeDataset 原始 JSONL 行数：v1 train 1,570、v1 test 175、v2 SFT 945、v2 RL 945、v2 test 200。
- `data/code-r1-plus-2k-leetcode2k` 与 `data/dedup/...` 存在于本机并被 ignore；`qwen3-8b-grpo-20260928/data/data_manifest.txt` 中 6 个 sha256 全部与当前文件一致。
- run-01 使用的 dedup train.parquet 存在；run-02 使用的筛题版 parquet 是推演条件，本机没有，也不应伪称存在。
- 两套训练目录均有 `PROVENANCE.md`：`qwen3-8b-grpo-20260928` 与 `qwen3-8b-grpo-yourplan-20261018` 都是推演/沙盘。日志、曲线、wandb 导出、checkpoint manifest、eval 明细和报告可用来学习证据形态与归因，不能证明模型被训练过。
- 原仓库 README 的作者模型指标和 Hugging Face 结果是作者报告；本仓库没有本地复现证据，不能升级为本人实验结果。

## 本次可运行验证

在当前 Windows / Python 3.14.6 环境执行：

- `RLVR学习资料/examples/training/test_training_prepare.py`：6/6 通过。
- `RLVR学习资料/examples/maximum-subarray/test_learning_tools.py`：19/19 通过。
- `RLVR学习资料/examples/maximum-subarray/run_demo.py all`：prepare、verify、score、advantage、evaluate 全链通过；8 个 accepted cases，7 个控制样本逐条隔离。
- `RLVR学习资料/examples/training/check_scenario_docs.py`：137 个推演数字全部对上。
- `RLVR学习资料/examples/source-supplements/test_source_supplements.py`：22 项中 15 通过、3 项按设计跳过、4 项错误；错误均为当前主机没有 sandbox-exec/firejail 后端，判分器按设计拒绝未隔离执行。这说明材料存在、代码路径可达，但受保护判分契约需要在 Linux/macOS 目标环境验收。
- `pytest` 未安装不影响上述检查，测试文件自身可用 `python -B ...` 走 unittest。

## 当前环境边界

- GPU：NVIDIA GeForce RTX 3060 Laptop，6,144 MiB，Driver 592.82。
- Python 3.14.6 中 `torch`、`vllm`、`transformers`、`ray`、`pyarrow`、`datasets`、`numpy`、`pandas`、`matplotlib` 均未安装。
- 原版训练要求 Linux/CUDA、vLLM、flash-attn、Ray、firejail；仓库 `requirements.txt` 写 `vllm<=0.6.3`，根 README 的 0.7.3 说明与单卡约束存在版本差异，实跑前必须按目标拓扑重新锁定版本。
- 真实 smoke 的最小证据链应包含：目标环境版本记录、firejail 正/错解判分探针、模型本地路径、至少一次参数更新、checkpoint 保存与恢复、固定题集 baseline/after 的多候选评测。当前本机一项都没有执行。
- 学习期间不要把 6 GB 本机 GPU 说成可完成 Qwen3-8B 训练；它最多只能作为硬件存在的事实，不足以支撑资源结论。

## 建议开始方式

1. 从 `README.md` 的五项自测开始，不懂再进 `00`，不要先通读全部源码。
2. 每阶段先跑对应 demo/练习，再读源码；把作业证据写在各阶段指定 `notes/attempt-*` 目录，不另建空泛笔记体系。
3. Windows 本机先做文档、数据、机制、`maximum-subarray` 和训练准备静态检查；涉及 sandbox 的 4 个错误留到 WSL/Linux 环境复验。
4. 若后续目标是真实 smoke，先单独建 Linux/CUDA 环境，锁定 0.5B/1.5B 模型和 vLLM/torch 版本，确认 24 GB 以上 GPU 或租用资源；不要在当前环境直接安装或开训。
5. 学习与实跑分开记账：学习完成不等于模型训练过；推演数字一致只证明沙盘自洽，不证明训练效果。