"""C1（升级方案第 7 节）：三种判分路径的量化审计——正确解被拒率 / 已知错解误过率 / 变异杀死率 / 执行成本。

7 道题来自 ``coder1_plus_oracles``（独立 oracle + 合法输入生成器）；候选代码全部由本脚本自建（可信代码），
因此在无沙箱平台用 ``CODER1_PLUS_SANDBOX=local`` 显式执行（sandbox.py 声明的用途：只执行可信代码）。
原版路径的执行器由 ``supplement_env.local_exec`` 替换，与 firejail 后端返回格式一致。

三种路径用同一组候选对照：
- original：GT 不带 prelude，走原版 ``coder1._compute_score``（拼接 + 整体退出码）；
- legacy：functional 里塞一条协议不支持的语句（print），触发 UnsupportedTests 退回拼接契约；
- protected：prelude + 受协议支持的断言（期望答案不进候选进程，逐条判定）。

用法（仓库根目录）：
    python -B RLVR学习资料/examples/source-supplements/verify_verifier_metrics.py
结果写 ``verifier-metrics-20261101/``（metrics.json + report.md）并打印摘要。数字是本机实测。
"""
from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "examples" / "data_preprocess"))

import os

os.environ.setdefault("CODER1_PLUS_SANDBOX", "local")  # 候选全部本脚本自建（可信）；local 后端不隔离

from supplement_env import load_reward_score, wrap_response  # noqa: E402
from coder1_plus_oracles import ORACLES  # noqa: E402
from coder1_plus_tests import generate_mutants  # noqa: E402

N_CASES = 6      # 每题从 oracle 生成的用例数
N_MUTANTS = 6    # 每题保留的变异体数（generate_mutants 同 seed 确定性）

METHOD = {"maximum-subarray": "maxSubArray", "increasing-triplet-subsequence": "increasingTriplet",
          "valid-palindrome": "isPalindrome", "number-of-senior-citizens": "countSeniors",
          "find-minimum-in-rotated-sorted-array": "findMin", "third-maximum-number": "thirdMax",
          "shuffle-the-array": "shuffle"}

CORRECT = {
    "maximum-subarray": "class Solution:\n    def maxSubArray(self, nums):\n        best = cur = nums[0]\n"
                        "        for value in nums[1:]:\n            cur = max(value, cur + value)\n            best = max(best, cur)\n        return best",
    "increasing-triplet-subsequence": "class Solution:\n    def increasingTriplet(self, nums):\n"
                                      "        first = second = float('inf')\n        for value in nums:\n"
                                      "            if value <= first:\n                first = value\n"
                                      "            elif value <= second:\n                second = value\n"
                                      "            else:\n                return True\n        return False",
    "valid-palindrome": "class Solution:\n    def isPalindrome(self, s):\n        kept = [ch.lower() for ch in s if ch.isalnum()]\n"
                        "        left, right = 0, len(kept) - 1\n        while left < right:\n"
                        "            if kept[left] != kept[right]:\n                return False\n"
                        "            left += 1\n            right -= 1\n        return True",
    "number-of-senior-citizens": "class Solution:\n    def countSeniors(self, details):\n"
                                 "        return sum(1 for item in details if item[11:13] > '60')",
    "find-minimum-in-rotated-sorted-array": "class Solution:\n    def findMin(self, nums):\n"
                                            "        left, right = 0, len(nums) - 1\n"
                                            "        while left < right:\n            mid = (left + right) // 2\n"
                                            "            if nums[mid] > nums[right]:\n                left = mid + 1\n"
                                            "            else:\n                right = mid\n        return nums[left]",
    "third-maximum-number": "class Solution:\n    def thirdMax(self, nums):\n"
                            "        distinct = sorted(set(nums), reverse=True)\n"
                            "        return distinct[2] if len(distinct) >= 3 else distinct[0]",
    "shuffle-the-array": "class Solution:\n    def shuffle(self, nums, n):\n"
                         "        out = []\n        for i in range(n):\n            out.append(nums[i])\n"
                         "            out.append(nums[n + i])\n        return out",
}

WRONG = {
    "maximum-subarray": [
        ("wrong_max_element", "class Solution:\n    def maxSubArray(self, nums):\n        return max(nums)"),
        ("wrong_zero_start", "class Solution:\n    def maxSubArray(self, nums):\n        best = cur = 0\n"
                             "        for value in nums:\n            cur = max(value, cur + value)\n            best = max(best, cur)\n        return best"),
    ],
    "increasing-triplet-subsequence": [
        ("wrong_non_strict", "class Solution:\n    def increasingTriplet(self, nums):\n"
                             "        first = second = float('inf')\n        for value in nums:\n"
                             "            if value < first:\n                first = value\n"
                             "            elif value < second:\n                second = value\n"
                             "            else:\n                return True\n        return False"),
    ],
    "valid-palindrome": [("wrong_case_sensitive", "class Solution:\n    def isPalindrome(self, s):\n"
                         "        kept = [ch for ch in s if ch.isalnum()]\n        return kept == kept[::-1]")],
    "number-of-senior-citizens": [("wrong_inclusive_60", "class Solution:\n    def countSeniors(self, details):\n"
                                   "        return sum(1 for item in details if item[11:13] >= '60')")],
    "find-minimum-in-rotated-sorted-array": [("wrong_first_half",
                                              "class Solution:\n    def findMin(self, nums):\n"
                                              "        return min(nums[:len(nums) // 2 + 1])")],
    "third-maximum-number": [("wrong_no_dedup", "class Solution:\n    def thirdMax(self, nums):\n"
                              "        ordered = sorted(nums, reverse=True)\n"
                              "        return ordered[2] if len(ordered) >= 3 else ordered[0]")],
    "shuffle-the-array": [("wrong_swapped_pairs", "class Solution:\n    def shuffle(self, nums):\n"
                           "        n = len(nums) // 2\n        out = []\n        for i in range(n):\n"
                           "            out.append(nums[n + i])\n            out.append(nums[i])\n        return out")],
}

# 第 15 个候选：作弊解——提前 sys.exit(0)。原版/legacy 按整体退出码判分必然误过（原版已知漏洞），
# protected 协议要求完成事件，提前退出表现为结果缺失，判失败。这是三条路径差异的机制性对照。
CHEAT = ("cheat_exit_0", "import sys\n\nclass Solution:\n    def maxSubArray(self, nums):\n"
         "        sys.exit(0)  # 提前伪造成功退出")


def build_functional(slug: str, rng: random.Random) -> tuple[str, str, str]:
    """返回 (protected_functional, original_functional, legacy_functional)。用例由独立 oracle 生成。"""
    spec = ORACLES[slug]
    asserts, seen = [], set()
    guard = 0
    while len(asserts) < N_CASES and guard < 200:
        guard += 1
        kwargs = spec["generate"](rng)
        key = json.dumps(kwargs, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        expected = spec["oracle"](**kwargs)
        args = ", ".join(f"{name}={value!r}" for name, value in kwargs.items())
        asserts.append(f"    assert candidate({args}) == {expected!r}, 'case-{len(asserts) + 1}'")
    body = "\n".join(asserts)
    entry = f"Solution().{METHOD[slug]}"
    functional = f"def check(candidate):\n{body}\n\ncheck({entry})"
    legacy = functional.replace("def check(candidate):\n", "def check(candidate):\n    print('legacy audit')\n", 1)
    return functional, functional, legacy


def main() -> int:
    router, coder1, coder1_plus = load_reward_score()
    assert callable(getattr(coder1_plus, "compute_score_detailed"))
    rng = random.Random(53)
    rows, started = [], time.time()
    for slug in ORACLES:
        functional, original_functional, legacy_functional = build_functional(slug, rng)
        ground_truths = {
            "original": json.dumps({"functional": original_functional}),
            "legacy": json.dumps({"prelude": "", "functional": legacy_functional}),
            "protected": json.dumps({"prelude": "", "functional": functional, "compare": "exact"}),
        }
        candidates = [("correct", CORRECT[slug])] + WRONG.get(slug, [])
        if slug == "maximum-subarray":
            candidates.append(CHEAT)
        candidates += [("mutant:" + description, code)
                       for description, code in generate_mutants(CORRECT[slug], N_MUTANTS, seed=53)]
        for kind, code in candidates:
            response = wrap_response(code)
            for path, gt in ground_truths.items():
                begin = time.time()
                try:
                    score, info = coder1_plus.compute_score_detailed(response, gt, extra_info={}, timeout=15)
                    passed = score == 1.1
                    status = info.get("status")
                except Exception as error:  # 判分器异常如实记录，不计成通过
                    passed, status, score = False, f"exception:{type(error).__name__}", None
                rows.append({"slug": slug, "path": path, "kind": kind, "passed": passed,
                             "score": score, "status": status, "duration_s": round(time.time() - begin, 3)})
        print(f"[{slug}] {len(candidates)} candidates x 3 paths done", flush=True)

    def rate(items, predicate) -> float:
        return round(sum(1 for r in items if predicate(r)) / len(items), 4) if items else None

    metrics = {}
    for path in ("original", "legacy", "protected"):
        subset = [r for r in rows if r["path"] == path]
        correct = [r for r in subset if r["kind"] == "correct"]
        wrong = [r for r in subset if r["kind"].startswith("wrong_")]
        cheats = [r for r in subset if r["kind"] == "cheat_exit_0"]
        mutants = [r for r in subset if r["kind"].startswith("mutant:")]
        metrics[path] = {
            "correct_pass_rate": rate(correct, lambda r: r["passed"]),
            "correct_rejected_rate": rate(correct, lambda r: not r["passed"]),
            "wrong_pass_rate": rate(wrong, lambda r: r["passed"]),
            "cheat_pass": [r["passed"] for r in cheats],
            "mutant_kill_rate": rate(mutants, lambda r: not r["passed"]),
            "avg_duration_s": round(statistics.fmean(r["duration_s"] for r in subset), 3) if subset else None,
            "n": len(subset),
        }
    summary = {"date": "2026-11-01", "platform": sys.platform, "python": sys.version.split()[0],
               "sandbox": os.environ.get("CODER1_PLUS_SANDBOX"), "problems": len(ORACLES),
               "total_s": round(time.time() - started, 1), "metrics": metrics, "rows": rows}

    out_dir = HERE / "verifier-metrics-20261101"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 三种判分路径的量化审计（2026-11-01 本机实测）", "",
             f"7 题 × 每题 1 正确解 + 已知错解 + {N_MUTANTS} 变异体（+1 作弊解），同组候选过三条路径。", "",
             "| 路径 | 正确解通过率 | 已知错解误过率 | 作弊解(sys.exit(0)) | 变异杀死率 | 平均耗时/次 |",
             "|---|---|---|---|---|---|"]
    for path, item in metrics.items():
        cheat = "—" if not item["cheat_pass"] else ("误过" if any(item["cheat_pass"]) else "拦截")
        lines.append(f"| {path} | {item['correct_pass_rate']} | {item['wrong_pass_rate']} | {cheat} | "
                     f"{item['mutant_kill_rate']} | {item['avg_duration_s']}s |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
