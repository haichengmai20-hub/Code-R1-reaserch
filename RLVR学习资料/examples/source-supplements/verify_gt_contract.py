"""验证补强项 1「GT 执行契约」：同一份正确的参考解，原版契约与补强版契约各判一次。

每道题的候选回答 = 前缀里的 imports 与 TreeNode、ListNode 类 + 数据集参考解，
也就是一个写全了导入、但不知道数据集私有测试辅助函数的认真回答。
原版 GT 只取 test；补强版 GT 带 prelude，判分时先拼前缀。

在仓库根目录运行（需要 Python 3.10+，只用标准库）：
    $PY -B RLVR学习资料/examples/source-supplements/verify_gt_contract.py
报告写到同目录的 outputs/gt_contract_report.json（被 .gitignore 忽略，可随时重新生成）。
"""
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import supplement_env as env  # noqa: E402

HERE = Path(__file__).resolve().parent


def score_one(coder1, coder1_plus, row):
    gt = json.loads(row["reward_model"]["ground_truth"])
    candidate = env.self_contained_candidate(gt["prelude"], row["extra_info"]["reference"])
    response = env.wrap_response(candidate)
    original_gt = json.dumps({"functional": gt["functional"]})  # 原版 leetcode2k() 的 GT
    extra = row["extra_info"]
    original, original_log = coder1._compute_score(response, original_gt, extra_info=extra)
    plus, plus_log = coder1_plus._compute_score(response, row["reward_model"]["ground_truth"], extra_info=extra)
    uses_helpers = any(re.search(rf"\b{name}\b", gt["functional"]) for name in env.TEST_HELPERS)
    error, category = None, None
    if plus != 1.1:
        lines = [line.strip() for line in plus_log.splitlines() if line.strip()]
        error = (next((line for line in lines if line.startswith("solution_error:")), None)
                 or next((line for line in lines if any(key in line for key in ("wrong_answer", "runtime_error",
                                                                                "missing", "unparseable"))), None)
                 or next((line for line in reversed(lines) if "Error" in line or "Timeout" in line), None)
                 or (lines[-1] if lines else ""))
        category = classify_failure(coder1_plus, gt, candidate, error)
    return {"task_id": extra["task_id"], "split": extra["split"], "uses_prefix_helpers": uses_helpers,
            "original": original, "plus": plus, "plus_error": error, "failure_category": category}


def classify_failure(coder1_plus, gt, candidate, error):
    """把补强版契约下仍失败的题分成三类，每一类都对应不同的责任层。"""
    if error and "ModuleNotFoundError" in error:
        return "environment_dependency"  # 参考解用了第三方库，运行环境没装
    program = coder1_plus.build_program(gt["prelude"], candidate, gt["functional"])
    outcomes = {env.local_exec(program, hash_seed=seed)[0] for seed in range(1, 7)}
    if outcomes == {True, False}:
        return "order_dependent_flaky"  # 结果随 set 迭代顺序变化：题目允许任意顺序，测试却比较列表
    return "unexplained"


def main():
    _, coder1, coder1_plus = env.load_reward_score()
    builder = env.load_builder()
    train, test = builder.build_leetcode_rows()
    rows = train + test
    started = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda row: score_one(coder1, coder1_plus, row), rows))
    elapsed = time.time() - started

    summary = {}
    for split in ("train", "test"):
        part = [r for r in results if r["split"] == split]
        helper = [r for r in part if r["uses_prefix_helpers"]]
        summary[split] = {
            "problems": len(part),
            "tests_use_prefix_helpers": len(helper),
            "original_contract_rewards": dict(Counter(r["original"] for r in part)),
            "plus_contract_rewards": dict(Counter(r["plus"] for r in part)),
            "helper_problems_original": dict(Counter(r["original"] for r in helper)),
            "helper_problems_plus": dict(Counter(r["plus"] for r in helper)),
            "flipped_0.1_to_1.1": sum(1 for r in part if r["original"] == 0.1 and r["plus"] == 1.1),
            "flipped_1.1_to_lower": sum(1 for r in part if r["original"] == 1.1 and r["plus"] != 1.1),
        }
    failures = [r for r in results if r["plus"] != 1.1]
    helper = [r for r in results if r["uses_prefix_helpers"]]
    checks = {
        "every_prefix_helper_problem_fails_under_original": all(r["original"] == 0.1 for r in helper),
        "every_prefix_helper_problem_passes_under_plus": all(r["plus"] == 1.1 for r in helper),
        "no_unexplained_plus_failure": not any(r["failure_category"] == "unexplained" for r in failures),
    }
    report = {
        "what": "Same correct reference candidate scored under the original GT contract (candidate + test) "
                "and coder1_plus (prelude contract, protected judging protocol, sandbox backend).",
        "executor": "original contract: local subprocess (not a sandbox); coder1_plus: its own sandbox backend "
                    "(sandbox-exec on macOS); only dataset reference solutions are run",
        "python": sys.version.split()[0],
        "elapsed_seconds": round(elapsed, 1),
        "summary": summary,
        "plus_failure_categories": dict(Counter(r["failure_category"] for r in failures)),
        "checks": checks,
        "plus_contract_failures": failures,
    }
    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "gt_contract_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("python", "elapsed_seconds", "summary",
                                                   "plus_failure_categories", "checks")},
                     ensure_ascii=False, indent=2))
    for failure in failures:
        print(f"plus failure [{failure['failure_category']}]: {failure['task_id']} ({failure['split']}): {failure['plus_error']}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
