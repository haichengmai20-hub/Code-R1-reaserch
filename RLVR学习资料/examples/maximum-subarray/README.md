# 最大子数组和：受控 RLVR 诊断演示

[资料首页](../../README.md) · [共同题目](../../RLVR工业流程.md#running-example) · [01 数据与GT](../../01_数据理解与GT构建.md) · [02 Verifier](../../02_Verifier与任务池验收.md) · [03 GRPO](../../03_Rollout与GRPO机制.md) · [04 评测](../../04_真实训练与独立评测.md)

**阅读职责：** 本页说明固定演示的来源与结果；学生实际变更按[操作说明](EXERCISES.md)，提交后按[答案/评分](ANSWERS.md)自检。新 [practice.py](practice.py) 不使用演示固定8/7限制，不覆盖本页outputs；[mechanism_lab.py](mechanism_lab.py)补概率与梯度toy。先按阶段调用函数，不必先通读整个演示实现。

在仓库根目录运行，需要 Python 3.10+，仅用标准库和 CPU；`$PY` 的设置见[资料首页](../../README.md#run-setup)：

```bash
$PY -B RLVR学习资料/examples/maximum-subarray/run_demo.py all
```

也支持 `prepare`、`verify`、`score`、`advantage`、`evaluate`。每个命令重新生成其所需的全部前置阶段；例如 `score` 会先准备并核验数据，`evaluate` 执行完整流水线。输出固定在本目录的 `outputs/`，后一次运行会更新同名演示产物。

题目锁定为本地 `LeetCodeDataset/data/LeetCodeDataset-v2-rl-problems.jsonl` 第 **143** 行，`task_id=maximum-subarray`，题号 **53**，入口 `Solution().maxSubArray`。Code-R1 来源 revision 为 `443f8da8058d07051793eff875da117534fd1ba1`，dataset 来源 revision 为 `f3519e6b7f42265dd4e4d35d4812e8c29b08111a`。脚本核验原始记录、已审参考答案和相关源码的锁定 hash，漂移即停止。原始语料的 split 为 `rl`；本次产物的 split 为 `diagnostic`，不冒充官方训练集或留出测试集。

actor prompt 保留 `meta.query` 全部原文，并追加已审核的英文合同：非空 `List[int]`、长度 1..100000、元素 -10000..10000、返回非空连续子数组最大和的 int、禁止空子数组。`extra_info.actor_prompt_adaptation` 保存来源字段 `meta.query / meta.question_title`、题目官方 URL、追加原文和适用范围；这次追加是演示适配，原 Code-R1 preprocess 没有这项处理。`report.prepare.actor_prompt_validation` 核验原 query 保留、所有约束存在、隐藏 functional GT/原测试/reference 正文没有进入 actor prompt、原记录 hash 不变；raw snapshot 原文与原始行字节不因这次适配改变。

`extra_info.source_file / source_line_1_based / source_question_id / source_difficulty` 对应原始文件、行 143、题号 53、难度 Medium；`index=142` 仍为原文件的零基位置，不能当作去重后训练池 index。`report.prepare.source_metadata_readback_validation` 在实际 JSONL 读回后逐项对照 raw snapshot 与 raw Task。

每条记录绑定自己的 `case_id / input / expected / origin / input_sha256`。GT 只从已接受记录生成：

| case_id | nums | expected | 来源 |
| --- | --- | ---: | --- |
| seed-1 | `[-2,1,-3,4,-1,2,1,-5,4]` | 6 | 原始 assert |
| seed-2 | `[1]` | 1 | 原始 assert |
| seed-3 | `[5,4,-1,7,8]` | 23 | 原始 assert |
| diag-all-negative | `[-3,-2,-5]` | -2 | 新增诊断 |
| diag-restart | `[2,-5,3,4,-1]` | 7 | 新增诊断 |
| diag-zero | `[0,0]` | 0 | 新增诊断 |
| diag-single-negative | `[-7]` | -7 | 新增诊断 |
| diag-boundary | `[10000,-10000,10000]` | 10000 | 新增诊断 |

**7 条负控制逐条隔离**：全负 input 错配 expected=6、`[1]` 错配 23、空数组、字符串元素、bool 元素、越界 10001、`[1]` 冲突 expected=2。错配记录保留具体原因和其对应 input；正确的 `[1]→1`、`[-3,-2,-5]→-2` 继续被接受。

核验包含题意对应的**非空连续子数组穷举**、独立 Kadane、已审原始 reference，以及新增 prefix 实现；长度 1..4、元素 -2..2 的小域共有 **780** 条。全负例的 6 个非空子数组和为 `[-3,-5,-10,-2,-7,-5]`，最大值 -2，可手算复核。报告还验证正确/错误程序、故意改坏 GT、空 GT 和隔离记录进入 GT 的拒绝保护。重复 case_id、input/hash 不符、seed 与 examples 数量不等会在参考代码执行前拒绝，并各有机械对齐负回归；这些检查只校验数据。

| 输出文件 | 内容 |
| --- | --- |
| [raw_task.json](outputs/raw_task.json) | 原始记录、原始 JSONL 行文本及换行、来源和 hash |
| [cases.json](outputs/cases.json) | 8 条接受记录、7 条隔离记录、逐条独立核验与全负手算 |
| [prepared_task.json](outputs/prepared_task.json) | Code-R1 形状的演示 row；actor prompt 含有记录来源的约束适配；ground_truth 为 JSON 字符串；附 case 索引 |
| [data_pool.jsonl](outputs/data_pool.jsonl) | 同 row 的 JSONL 模拟落盘与读回，不代表原 Parquet 预处理已运行 |
| [report.json](outputs/report.json) | 回归、原 scorer 分数/日志、另一次独立 audit 的实际输出与完成记录、STDIO、advantage、评测及源码保护证据 |

AST 载入原 `validate_response_structure`、答案/代码提取函数及 `_compute_score` 的函数体，避免模块导入时加载 numpy、选择 Linux/远程执行器。只有执行后端改为固定候选白名单的本地 Python subprocess；不接收任意用户代码，不安装依赖，不联网，不调用模型或 GPU。原源码和原 dataset 在运行前后按字节 hash 复核，输出路径限制在本演示目录。

`reward / scorer_success_signal / scorer_log` 来自原 `_compute_score` 的一次运行。`audit_completed_case_count / audit_completion_status / audit_observations` 来自另一次 `collect_outputs` 独立执行；`correctness_pass` 只表示该独立 audit 在 8 条有限 case 上的结果。**原 scorer 不报告受保护的测试完成计数**；audit 的 8 条观察不能当作原 scorer 实际执行了 8 条断言。

| 候选 | 原 scorer reward | 原 scorer traceback 中首个失败 case_id | 另一次 audit 观察 case 数 | 独立 audit 的有限 case correctness |
| --- | ---: | --- | ---: | --- |
| reference | 1.1 | 无 | 8 | true |
| correct_prefix | 1.1 | 无 | 8 | true |
| wrong_empty | 0.1 | diag-all-negative | 8 | false |
| wrong_sum | 0.1 | seed-1 | 8 | false |
| bad_format | -1.1 | 格式拒绝，未执行 | 0 | false |
| early_exit | 1.1 | 无失败 traceback | 0 | false |

失败 case_id 由原 scorer 日志的 traceback 提取，字段为 `source_first_failed_case_id_from_traceback`；不据此推算完成数。`wrong_empty` 在原 scorer 的第 4 个断言失败，`wrong_sum` 在第 1 个断言失败；表中的 audit 8 条来自另一次不逐条 assert 早退的输出收集。

`early_exit` 执行 `sys.exit(0)`，退出发生在 `check()` 前；原 scorer 因退出码 0 给 **1.1**，另一次独立 audit 未产生完成载荷，观察数 **0、correctness=false**。这是已展示的协议缺口；本演示没有建立可信的测试完成协议或未知代码安全边界。row 明确 `train_ready=false`，原因分别为诊断用途、原 scorer 存在测试绕过、未建立未知模型代码的安全执行边界。

STDIO 部分是**同题教学适配**：stdin 为一行 JSON 数组，expected stdout 为结果整数，复用原 scorer 的执行成功与 `.strip()` 后文本相等判断；每条 stdin、expected、actual 成对记录。原始 LeetCode 记录仍是 function contract，这不是原始 TACO STDIO 数据。

正常 4 候选 reward `[1.1,1.1,0.1,0.1]` 的均值约 0.6，样本标准差约 0.577350；按原公式和 `epsilon=1e-6`，advantage 约 `[0.866024,0.866024,-0.866024,-0.866024]`。终止位置、mask 和 uid 重排是教学构造，并非 tokenizer 输出。基础设施故障回填 `[0.1,0.1,0.1,0]` 是**反事实教学模拟**，advantage 约 `[0.499990,0.499990,0.499990,-1.499970]`，忽略 epsilon 时即 `[0.5,0.5,0.5,-1.5]`；未实际制造或采到超时。

所选 GT 下 4 个算法候选正确率 **2/4=50%**，meanReward **约 0.6**；从此有限受控池无放回均匀选取时，`pass@1=0.5`、`pass@2=5/6`、`pass@3=pass@4=1`。这不是模型采样、真实泛化或训练效果。报告也单独汇总包含 2 个协议负控制的 6 候选池，correctness 与 scorer 成功信号分别计算。

作者在 Windows、Python 3.14.6 验证，2026-10-13 在 macOS、Python 3.12.15 复跑通过。标准库数值翻译未运行 torch，也不保证与 float32 位级一致。780 个小数组和 8 个诊断 case 不构成对全部合法输入的证明；本地 subprocess 无操作系统安全隔离，不能扩展为未知代码执行器。

## 作者验证证据

[端到端记录](validation_summary.json)保存正确流程与预期拒绝路径；[文档检查](bundle_validation.json)记录链接、锚点、编码与语法；[公式检查](math_validation_result.json)只证明KaTeX解析，未验证整个IDE渲染。作者validation-*练习是工具测试，不计学生通关，也不是模型成绩。两套行为测试入口为 [test_learning_tools.py](test_learning_tools.py) 与 [训练准备检查](../training/test_training_prepare.py)。
