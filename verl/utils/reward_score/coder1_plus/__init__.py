"""Code-R1 判分器的补强版（coder1_plus）。

原版 ``coder1`` 一个字都不改。本模块只补学习资料（RLVR学习资料/）指出、原版缺少的环节，
每补一项都在下面登记，并在 RLVR学习资料/examples/source-supplements/ 里给出 CPU 可复现的验证。

1. GT 执行契约（2026-10-14）
   LeetCodeDataset 的测试依赖原始 ``prompt`` 前缀里的辅助函数 ``tree_node``、``is_same_tree``、
   ``list_node``、``is_same_list``。原版 GT 只取 ``test``，1890 道 LeetCode 训练题里有 121 道树和链表题
   因此必然 NameError。补强版的 GT 多带 ``prelude`` 字段，执行时先运行前缀。
4. 受保护的判题协议（2026-10-14，见 protocol.py）
   期望答案不进入候选代码所在的进程；子进程逐条报告输出，判分方自己比较。提前退出、伪造完成标记都判失败；
   每条用例都有结论；GT 的 ``compare`` 为 ``unordered_outer`` 时，列表答案忽略外层顺序。
5. 结构化执行层与隔离后端（2026-10-14，见 sandbox.py）
   执行结果分成 ok、compile_error、runtime_error、memory_error、timeout、killed、infra_error；超时后清理
   整个进程组；Linux 用 firejail，macOS 用 sandbox-exec，由 CODER1_PLUS_SANDBOX 选择。
9. 奖励分值的字典序修正（2026-11-06）
   原版口径下格式边际（0.1 对 −1.1，差 1.2）大于正确边际（1.1 对 0.1，差 1.0）：格式合格但错误的回答
   压过格式不合格但正确的回答，而且后者在格式门处直接 −1.1，代码根本不执行。补强版改为：
   正确性 +1.0 / 其余 0，格式 ±0.1 做次级信号；严格标签缺失时退回整段响应、抽取其中全部完整代码块拼接照样执行。
   四档：对+标签 1.1、对+无标签 0.9、错+标签 0.1、错/无代码+无标签 −0.1。任何含正确解的回答严格高于
   任何不含的。CODER1_PLUS_REWARD=legacy 可复现原版口径（−1.1/0.1/1.1）。没有 ``prelude`` 的 GT 仍委托
   原版判分，分值不变。
10. 性能轮（2026-11-07，见 perf.py；归因 2026-11-09 随测试扩容方案修正）
    原版 reward 只测输出，O(n^2) 与 O(n) 的正确回答同分，RL 学不会「写快」。补强版给带
    ``extra_info.perf`` 标定值的数据行加一个性能轮：正确性轮通过后，用记录的种子确定性重建大输入
    （规模来自 Constraints 解析，上标丢失按 10 的幂识别），timeout 用「参考解实测 × 3」的每题标定值。
    归因：单纯超时记性能瑕疵，扣 0.2 与「对但无标签」并列 0.9 档（对+快+标签 1.1、对+一项瑕疵
    0.9、对+两项 0.7）；大输入输出不符或出错记正确性失败（0.1 档）——大输入是题意合法输入，
    答错就是错解，留在 0.9 会让崩溃的错答与慢而对打平。标定是一次性离线产物（数据侧
    ``coder1_plus_perf.py``），训练时零标定开销、同题各 step 判分标准不漂移；legacy 口径整体跳过。
    设计依据：CodeContests 每题带 time_limit 数据字段；Codeforces Polygon 用预期解实测 × 系数
    标定 TL、用故意慢的解做负对照验证。

没有 ``prelude`` 字段的 GT，例如 TACO 的函数题和标准输入输出题、KodCode 的 pytest 题，
原样交给原版 ``coder1._compute_score``，行为不变。断言写法超出协议支持范围时，退回第 1 项的拼接契约。

接口：
- ``compute_score``：路由调用，返回分数；遇到基础设施错误时抛出 ``InfraError``，原版奖励管理器会按
  原行为记 0 分，补强版奖励管理器 ``prime_plus`` 则会重试并按整组处理，见补强项 6。
- ``compute_score_detailed``：返回 (分数, 结构化信息)，信息里有 status、protocol、执行状态和逐条结论。

数据侧见 ``examples/data_preprocess/coder1_plus.py``；路由见 ``verl/utils/reward_score/__init__.py``
的 ``code_plus`` 分支。
"""
import json
import os
import secrets
import time

from .. import coder1 as _orig
from . import perf, protocol, sandbox

GT_CONTRACT = "prelude+solution+functional/v1"
DEFAULT_TIMEOUT = 30


class InfraError(RuntimeError):
    """判题基础设施故障：沙箱不可用、启动失败等。不是候选代码的错。"""


def build_program(prelude: str, solution_code: str, functional: str) -> str:
    """第 1 项的拼接契约：前缀、候选代码、测试。协议无法覆盖时才用它。"""
    return prelude.rstrip() + "\n\n" + solution_code + "\n\n" + functional


def _load_gt(ground_truth):
    return json.loads(ground_truth) if isinstance(ground_truth, str) else dict(ground_truth)


def compute_score_detailed(solution_str, ground_truth, extra_info=None, format_reward=0.1, answer_reward=1.,
                           timeout=DEFAULT_TIMEOUT):
    gt = _load_gt(ground_truth)
    if "prelude" not in gt:
        original_gt = ground_truth if isinstance(ground_truth, str) else json.dumps(ground_truth)
        score, log = _orig._compute_score(solution_str, original_gt, extra_info=extra_info,
                                          format_reward=format_reward, answer_reward=answer_reward)
        status = "format_error" if score < 0 else "passed" if score >= format_reward + answer_reward else "failed"
        return score, {"status": status, "protocol": "original", "log": log}
    if "functional" not in gt:
        raise ValueError(f"coder1_plus: a ground truth with 'prelude' must also carry 'functional': {list(gt)}")

    # 第 9 项：代码抽取与格式分值。strict = 标签合格且 <answer> 里有代码；
    # fallback = 标签不合格时退回整段响应、抽取其中全部完整代码块拼接（原版 extract 对无标签串本来就取全文全部代码块）。
    strict = _orig.validate_response_structure(solution_str)
    solution_code = _orig.extract_code_from_string(solution_str)
    extraction = "strict" if strict and solution_code else ("fallback" if solution_code else "none")
    if os.environ.get("CODER1_PLUS_REWARD", "aligned") == "legacy":
        if extraction != "strict":
            return -answer_reward - format_reward, {"status": "format_error", "protocol": "format_gate",
                                                    "extraction": extraction,
                                                    "log": "Bad format detected!\n" + solution_str}
        fmt = format_reward
    else:
        fmt = format_reward if extraction == "strict" else -format_reward
        if extraction == "none":
            return fmt, {"status": "format_error", "protocol": "format_gate", "extraction": "none",
                         "log": "Bad format detected (no extractable code)!\n" + solution_str}

    started = time.time()
    try:
        entry, cases = protocol.parse_cases(gt["functional"])
    except protocol.UnsupportedTests as reason:
        result = sandbox.run_python(build_program(gt["prelude"], solution_code, gt["functional"]), timeout=timeout)
        info = {"protocol": "legacy", "legacy_reason": str(reason), "extraction": extraction,
                "exec_status": result.status, "duration": round(time.time() - started, 3)}
        if result.status == "infra_error":
            return None, {**info, "status": "infra_error", "log": result.stderr}
        passed = result.ok
    else:
        nonce = secrets.token_hex(16)
        result = sandbox.run_python(protocol.build_harness(gt["prelude"], solution_code, entry, cases, nonce),
                                    timeout=timeout)
        info = {"protocol": "protected", "extraction": extraction, "exec_status": result.status,
                "duration": round(time.time() - started, 3)}
        if result.status == "infra_error":
            return None, {**info, "status": "infra_error", "log": result.stderr}
        verdict = protocol.judge(result.stdout, cases, nonce, gt.get("compare", "exact"))
        info["judge"] = verdict
        passed = verdict["passed"]

    lines = [f"protocol={info['protocol']} exec_status={result.status} duration={info['duration']}s"]
    if "judge" in info:
        lines.append(f"events={info['judge']['events']} passed_cases={info['judge']['n_pass']}/{info['judge']['n_cases']}")
        if info["judge"]["solution_error"]:
            lines.append(f"solution_error: {info['judge']['solution_error']}")
        lines.extend(f"  {item['label']}: {item['verdict']}" for item in info["judge"]["cases"])
    if result.stderr.strip():
        lines.append(result.stderr.strip()[-_orig._MAX_CHAR_DISPLAY:])
    info["log"] = "\n".join(lines)
    info["status"] = "passed" if passed else "failed"

    # 第 10 项：性能轮。正确性通过且数据行 extra_info.perf 带标定值时加跑一轮大输入。
    # 归因（2026-11-09 随测试扩容方案修正）：单纯超时（slow）是性能瑕疵，扣 2×format_reward，
    # 与「对但无标签」并列 0.9 档；大输入上输出不符或出错（mismatch/error）是正确性失败——
    # 大输入是题意合法输入，答错就是错解，回 0.1 档（与正确性轮失败同档；LeetCode 的 RE 同判
    # 不通过）。若把 mismatch 留在 0.9，崩溃的错答会和慢而对 0.9 打平，破坏「任何对答严格
    # 高于任何错答」。legacy 口径整体跳过（推演生成器钉 legacy，行为与 git 基线一致）。
    perf_field = extra_info.get("perf") if isinstance(extra_info, dict) else None
    if (passed and isinstance(perf_field, dict) and "time_limit_ms" in perf_field
            and os.environ.get("CODER1_PLUS_REWARD", "aligned") != "legacy"):
        perf_out = perf.run_perf_round(gt, solution_code, perf_field, sandbox=sandbox)
        info["perf"] = perf_out
        if perf_out["status"] == "infra":
            return None, {**info, "status": "infra_error", "log": info["log"] + "\nperf infra_error"}
        if perf_out["status"] in ("mismatch", "error"):
            info["status"] = "failed_perf_mismatch"  # 合法大输入上答错/崩溃 = 正确性失败
            return fmt, info
        if perf_out["status"] == "slow":
            info["status"] = "passed_perf_gap"
            return (answer_reward + fmt - 2 * format_reward), info

    return (answer_reward + fmt if passed else fmt), info


def _compute_score(solution_str, ground_truth, extra_info, format_reward=0.1, answer_reward=1.):
    """与原版相同的返回约定：(分数, 日志)。基础设施故障抛出 InfraError。"""
    score, info = compute_score_detailed(solution_str, ground_truth, extra_info, format_reward, answer_reward)
    if info["status"] == "infra_error":
        raise InfraError(info.get("log", "sandbox infrastructure error"))
    return score, info["log"]


def compute_score(solution_str, ground_truth, extra_info, format_reward=0.1, answer_reward=1.):
    # 数据加载器传来的 extra_info 可能是 numpy 的 object 数组，与原版一样先取出字典。
    if not isinstance(extra_info, dict) and hasattr(extra_info, "item"):
        extra_info = extra_info.item()
    score, reward_log = _compute_score(solution_str, ground_truth, extra_info=extra_info,
                                       format_reward=format_reward, answer_reward=answer_reward)
    marker = "✅" if score == (format_reward + answer_reward) else "❌"
    print(marker * 16 + "Reward Calculation (coder1_plus)" + marker * 16 + "\n" + reward_log + "\n" + marker * 16 +
          f"Final Reward = {score}" + marker * 16 + "\n\n")
    return score
