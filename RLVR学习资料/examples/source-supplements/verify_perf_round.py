"""补强项 10 的验证：标定链 + 性能轮四对照（Polygon 式正负对照）。

跑法（仓库根目录；本机 Windows 设 local 后端，Linux/macOS 用默认沙箱）：

    PYTHONUTF8=1 CODER1_PLUS_SANDBOX=local python -B "RLVR学习资料/examples/source-supplements/verify_perf_round.py"

覆盖：
1. 标定链：贯穿题 maximum-subarray 的 calibrate_one 产出完整 perf 字段（规模 10^5 来自
   '105' 的上标丢失解析，time_limit_ms ≥ 500，expected 非空）。
2. 正对照：Kadane 参考解在性能轮限时内通过 → 1.1（对+快+标签）。
3. 负对照（Polygon 慢解标签的对应物）：O(n^2) 暴力小输入全对、大输入超时 → 0.9
   （对但慢），info.perf.status == "slow"。
4. 双瑕疵：O(n^2) 暴力 + 无严格标签（fallback 提取）→ 0.7（对+慢+无标签）。
5. 大输入答错归正确性失败（2026-11-09 归因修正）：小输入全对、大输入返回错值的构造解
   → 0.1（不是 0.9），info.perf.status == "mismatch"、整体状态 failed_perf_mismatch。
6. 错答不触发性能轮：wrong_sum → 0.1，info 无 perf 键（性能轮只对正确性通过者加跑）。
7. legacy 口径：CODER1_PLUS_REWARD=legacy 下性能轮整体跳过，暴力有标签 → 1.1
   （原版口径不变，推演生成器钉 legacy 的行为不受影响）。
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("CODER1_PLUS_SANDBOX", "local")

import supplement_env as env  # noqa: E402

REPO = env.REPO
BRUTE_FORCE = """class Solution:
    def maxSubArray(self, nums):
        best = None
        for i in range(len(nums)):
            total = 0
            for j in range(i, len(nums)):
                total += nums[j]
                if best is None or total > best:
                    best = total
        return best
"""
WRONG_SUM = "class Solution:\n    def maxSubArray(self, nums):\n        return sum(nums)\n"
# 构造的「规模相关错误」：小输入全对，大输入（len ≥ 1000）返回 0——测的是归因机制：
# 合法大输入上答错必须是正确性失败（0.1），不能算性能瑕疵（0.9）
SCALE_WRONG = """class Solution:
    def maxSubArray(self, nums):
        if len(nums) >= 1000:
            return 0
        ans = f = nums[0]
        for x in nums[1:]:
            f = max(f, 0) + x
            ans = max(ans, f)
        return ans
"""


def load_task(task_id):
    for name in ("LeetCodeDataset-v2-rl-problems.jsonl", "LeetCodeDataset-v2-sft-problems.jsonl"):
        with open(REPO / "LeetCodeDataset" / "data" / name, encoding="utf-8") as handle:
            for line in handle:
                if line.strip() and json.loads(line)["task_id"] == task_id:
                    return json.loads(line)
    raise SystemExit(f"task not found: {task_id}")


def make_gt_and_extra(record, perf):
    gt = json.dumps({"prelude": record["prompt"],
                     "functional": record["test"] + f"\n\ncheck({record['entry_point'].strip()})"},
                    ensure_ascii=False)
    extra = {"prompt": record["meta"]["query"], "reference": record["completion"]}
    if perf is not None:
        extra["perf"] = perf
    return gt, extra


def raw_block(code):
    return "```python\n" + code.rstrip() + "\n```"


def check(label, got, want):
    if isinstance(want, float) and isinstance(got, float):
        ok = abs(got - want) < 1e-9      # 0.1+1.0-0.2 的浮点表示差异不当作失败
    else:
        ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got={got!r} want={want!r}")
    return ok


def main():
    print("== 1. 标定链（maximum-subarray）==")
    record = load_task("maximum-subarray")
    perf_path = REPO / "verl" / "utils" / "reward_score" / "coder1_plus" / "perf.py"
    spec = importlib.util.spec_from_file_location("perf_core", perf_path)
    perf_core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(perf_core)
    perf, reason = perf_core.calibrate_one(record)
    assert perf is not None, f"calibration failed: {reason}"
    results = [check("spec.size == 100000 ('105' 上标丢失 -> 10^5)", perf["spec"]["size"], 100000),
               check("time_limit_ms >= 500", perf["time_limit_ms"] >= 500, True),
               check("expected 非空", bool(perf["expected"]), True),
               check("entry 齐全", perf["entry"], "Solution().maxSubArray")]
    print(f"  标定产物：time_limit_ms={perf['time_limit_ms']} "
          f"ref_wall={perf['calibrated']['reference_wall_s']}s expected={perf['expected'][:24]}")

    _, _, PLUS = env.load_reward_score()
    gt, extra = make_gt_and_extra(record, perf)

    def score(response, legacy=False):
        old = os.environ.get("CODER1_PLUS_REWARD")
        if legacy:
            os.environ["CODER1_PLUS_REWARD"] = "legacy"
        else:
            os.environ.pop("CODER1_PLUS_REWARD", None)
        try:
            value, info = PLUS.compute_score_detailed(response, gt, extra_info=extra)
        finally:
            if old is None:
                os.environ.pop("CODER1_PLUS_REWARD", None)
            else:
                os.environ["CODER1_PLUS_REWARD"] = old
        return value, info

    print("\n== 2. 正对照：Kadane 参考解（对+快+标签）-> 1.1 ==")
    value, info = score(env.wrap_response(record["completion"]))
    results.append(check("score", value, 1.1))
    results.append(check("perf.status", info.get("perf", {}).get("status"), "pass"))

    print("\n== 3. 负对照：O(n^2) 暴力（小输入全对、大输入超时）-> 0.9 ==")
    value, info = score(env.wrap_response(BRUTE_FORCE))
    results.append(check("score", value, 0.9))
    results.append(check("perf.status == slow", info.get("perf", {}).get("status"), "slow"))
    results.append(check("整体状态 = passed_perf_gap", info.get("status"), "passed_perf_gap"))

    print("\n== 4. 双瑕疵：暴力 + 无严格标签（fallback 提取）-> 0.7 ==")
    value, info = score(raw_block(BRUTE_FORCE))
    results.append(check("score", value, 0.7))
    results.append(check("extraction == fallback", info.get("extraction"), "fallback"))

    print("\n== 5. 大输入答错归正确性失败（归因修正）：构造解 -> 0.1 ==")
    value, info = score(env.wrap_response(SCALE_WRONG))
    results.append(check("score", value, 0.1))
    results.append(check("perf.status == mismatch", info.get("perf", {}).get("status"), "mismatch"))
    results.append(check("整体状态 = failed_perf_mismatch", info.get("status"), "failed_perf_mismatch"))

    print("\n== 6. 错答不触发性能轮：wrong_sum -> 0.1 ==")
    value, info = score(env.wrap_response(WRONG_SUM))
    results.append(check("score", value, 0.1))
    results.append(check("无 perf 键（性能轮未触发）", "perf" in info, False))

    print("\n== 7. legacy 口径：暴力有标签 -> 1.1，性能轮跳过 ==")
    value, info = score(env.wrap_response(BRUTE_FORCE), legacy=True)
    results.append(check("score（原版口径不变）", value, 1.1))
    results.append(check("无 perf 键（legacy 跳过）", "perf" in info, False))

    print(f"\n{'=' * 46}\n{'全部通过' if all(results) else '存在失败'}：{sum(results)}/{len(results)}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
