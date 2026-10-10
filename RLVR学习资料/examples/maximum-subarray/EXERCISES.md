# 学生操作说明：从改变数据到观察机制

[资料首页](../../README.md) · [01](../../01_数据理解与GT构建.md) · [02](../../02_Verifier与任务池验收.md) · [03](../../03_Rollout与GRPO机制.md) · [04](../../04_真实训练与独立评测.md) · [提交后看答案](ANSWERS.md)

所有命令在 `D:/RL及harness/code-r1` 根目录执行，Python3.10+、标准库、CPU。学生入口是 `practice.py`；原 `run_demo.py` 继续提供锁定8/7演示。不要修改固定outputs来做变更实验，原命令会重建它们。

只运行脚本内已审固定候选；修改的是JSON测试数据或预测，没有接受任意候选代码、任意GT或shell命令的CLI。未知模型候选的真实执行进入已验收后端。原source/dataset不由这些命令改写。

<a id="data-exercise"></a>

## 01/02：自己的测试实际进入GT和评分

```powershell
Set-Location -LiteralPath 'D:\RL及harness\code-r1'
python -B RLVR学习资料/examples/maximum-subarray/practice.py init --run data-01 --mode seed
```

它创建项目根下 `RLVR学习资料/examples/maximum-subarray/practice_runs/data-01/cases.json` 和同目录prediction.json。下文短路径相对于本示例目录，命令仍从项目根执行。mode=seed从源记录提取3条seed；mode=expanded加入5条已审诊断，方便验证第9条能加入。已有run不覆盖，另选名字重做。

在cases列表追加自己的记录。下面只示范格式，不把这条当独立发现：

```json
{"case_id":"my-case","input":{"nums":[-4,-2]},"expected":-2,"origin":"enumeration in notes.md"}
```

选择自己的小数组，先枚举全部非空连续子数组，写出expected和合法性。再追加同输入配错误expected的另一行，使用不同case_id，观察正确行保留。`input_sha256`可省略，程序计算；如提供则必须匹配。不要加可执行代码或修改nums接口。

填写prediction：accepted/quarantined/pending是整数，wrong_empty_reward是你的分数预测，why是理由。首次check只要求前三个数量与why；score还要求分数预测。

```powershell
python -B RLVR学习资料/examples/maximum-subarray/practice.py check --run data-01
python -B RLVR学习资料/examples/maximum-subarray/practice.py score --run data-01
```

每次产生新 `attempt-001/002/...`。本次提交数据与预测先保存，再生成审核、GT和结果；前一次不覆盖。编辑原cases/prediction后再运行会消费新值。报告中 `data_prediction_matches / reward_prediction_matches` 说明预测是否匹配，不说明文字理由是否正确。

| 本次attempt文件 | 你需要追踪什么 |
|---|---|
| submitted_cases.json / submitted_prediction.json | 先预测了什么，实际提交哪份输入 |
| cases_audit.json | accepted/quarantined/pending，独立计算与原因 |
| prepared_task.json / data_pool.jsonl | 新GT、入口、case_index、test_version和读回 |
| report.json | 本次数量、源reward、另次audit、预测匹配与教学advantage |

Tests有错先隔离坏行；没有accepted时不生成空GT、不评分。单条超过64元素的合法输入pending，表示超出本练习暴力预算；合法域仍按题目约束检查。总记录最多64条。重复case_id或提供错误hash属于装配错误，运行停止，先修正再继续。

只改变格式时，冻结cases，更新wrong_empty_reward预测后运行：

```powershell
python -B RLVR学习资料/examples/maximum-subarray/practice.py score --run data-01 --bad-format
```

这个参数移除回答标签，不修改算法和Tests。比较两个attempt，不混入另一份GT或policy。固定候选不是模型采样，四候选advantage仅为公式对照。

<a id="blind-exercise"></a>

## 02：未知类别故障定位

```powershell
python -B RLVR学习资料/examples/maximum-subarray/practice.py blind-init --run blind-01 --seed 53
```

查看packets.json，不修改它。不同seed会改变输入、配对值与记录顺序。每份记录可以没有错误，或含多个责任层；不知道数量时不要强行每条找一处。逐项检查题目接口、合法输入与答案、GT配对、完成与actual、(n,c,k)和指标。

只填prediction.json的layers与why。类别为 `Task / Tests / GT / Verifier / Evaluation / none`；每个record用列表，例如单项或多个，干净记录用 `["none"]`。先在notes.md写出依据和最小区分检查，再提交：

```powershell
python -B RLVR学习资料/examples/maximum-subarray/practice.py blind-grade --run blind-01
```

程序按初始化manifest中的seed和规范化hash核对题包，防止意外删改、类型变化或串位；每次保存manifest与提交快照再给类别结果。这些是可编辑本地文件，不是安全签名或防作弊系统；改manifest/seed/源码后不能据程序通过称完成原题。自动只核对结构与类别，why需人工复核。正式自测先独立答，复测使用新run/seed或同伴未展开记录。题包是教学构造，没有真的注入服务故障。

<a id="transfer-exercise"></a>

## 04：两数之和跨题迁移

```powershell
python -B RLVR学习资料/examples/maximum-subarray/practice.py transfer-init --run transfer-01
```

material.json提供v2 RL第556行的原题、接口、三条seed与三个固定已审程序。原材料保持只读；想改输入应另作数据练习，不让评分静默换回原题。独立回答：唯一索引对是否意味着返回顺序唯一？完整合法性需要检查哪些条件？

prediction.json对A/B/C分别填legacy_pass和semantic_pass布尔值，以及why。legacy是原列表相等，semantic是从题意推出的合法性；先自己推导，提交后才展开新GT。

```powershell
python -B RLVR学习资料/examples/maximum-subarray/practice.py transfer-grade --run transfer-01
```

程序核对实际读入材料的规范化hash，保留类型差异，再使用固定源码执行候选。报告给实际返回、两种比较及原scorer两版GT分数；新 `semantic_ground_truth.json` 与 `prepared_task.json` 显示同一Task换判定规则，test_version为two-sum-semantic-diagnostic-v1。原语料与原scorer没有被修改。

这一迁移改善有限seed的判定规则，不证明全部域正确、模型泛化或后端安全。用自己的反例补充重复索引、越界、bool索引和错误和数等边界，纸面检查语义条件；不要只sort/set就宣布验收完整。

<a id="mechanism-exercise"></a>

## 03：概率、梯度、clip和mask

```powershell
python -B RLVR学习资料/examples/maximum-subarray/mechanism_lab.py --run mechanism-01
```

报告包含解析梯度、有限差分、两标量SGD前后值、温度、正负A的clip对照、micro平均和EOS。它没有torch、真实token或模型更新。先理解完整示范，再自己换logits/选中token/A，手算后调用文件内公开函数自检；代码变化只用于自己审阅的toy，不接模型候选。

独立变式交付在notes.md：固定old/ref/advantage的理由，new如何求导，clip有利方向的四种情形，两种micro分组为何不同，无EOS为什么不能认定自然停止。见[03机制](../../03_Rollout与GRPO机制.md#gradient-bridge)与[评分依据](ANSWERS.md#mechanism-rubric)。

## 退出码、复测与范围

0=本命令的机械检查/预测匹配；1=预测不匹配，报告已保存；2=材料、参数或装配检查失败。三者都不能单独证明学习完成。改错后保留旧attempt，写明为什么改变判断。

作者回归使用新的 `validation-*` run保存工具验证，学生用data-01等新名字开始。作者正确预测证明工具流程可用，不替代学生完成。最终按[4.3综合评分表](../../04_真实训练与独立评测.md#graduation)核对解释和操作。
