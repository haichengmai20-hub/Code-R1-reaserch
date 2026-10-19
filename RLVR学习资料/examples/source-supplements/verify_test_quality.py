"""验证补强项 2「测试强度审计」与补强项 3「测试补充与逐条审核」。

第一部分：对 LeetCode 题做变异审计，统计现有测试能杀死多少变异体。默认随机抽 200 题，约 1 分钟；
         加 --all 跑全部 2090 题，约 7 分钟。
第二部分：对 coder1_plus_oracles.py 里的 7 道示范题做闭环：
         ① 补测试前审计；② 随机生成候选输入，逐条审核，接受的用例取前 10 条并进 GT，再审计；
         ③ 对仍存活的变异体用随机合法输入和 oracle 对拍：找到反例的，把反例经同样的审核后补进 GT，
            这叫变异体引导补测试；对拍全部一致的，视为疑似等价变异体；④ 最终再审计一次。

在仓库根目录运行（Python 3.10+，只用标准库）：
    $PY -B RLVR学习资料/examples/source-supplements/verify_test_quality.py
    $PY -B RLVR学习资料/examples/source-supplements/verify_test_quality.py --all
报告写到同目录的 outputs/，其中 accepted_cases.json 可直接交给 coder1_plus.py 的 --augment_cases。
"""
import argparse
import collections
import copy
import importlib.util
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import supplement_env as env  # noqa: E402

HERE = Path(__file__).resolve().parent
TOOLS_PATH = env.REPO / "examples" / "data_preprocess" / "coder1_plus_tests.py"
ORACLES_PATH = env.REPO / "examples" / "data_preprocess" / "coder1_plus_oracles.py"
MUTANT_TIMEOUT = 5
CANDIDATES_PER_TASK = 60
ADDED_CASES_PER_TASK = 10
DIFF_INPUTS = 200


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def executor(code):
    return env.local_exec(code, timeout=MUTANT_TIMEOUT)


def strength_summary(records):
    audited = [r for r in records if r["baseline_ok"] and r["mutants"]]
    rates = [r["kill_rate"] for r in audited]
    buckets = collections.Counter("100%" if x == 1 else "75-99%" if x >= .75 else "50-74%" if x >= .5 else "<50%"
                                  for x in rates)
    by_tests = collections.defaultdict(list)
    for r in audited:
        by_tests[r["n_tests"]].append(r["kill_rate"])
    return {
        "problems": len(records),
        "baseline_failed": sum(1 for r in records if not r["baseline_ok"]),
        "no_mutation_sites": sum(1 for r in records if r["baseline_ok"] and not r["mutants"]),
        "audited": len(audited),
        "mutants": sum(r["mutants"] for r in audited),
        "killed": sum(r["killed"] for r in audited),
        "pooled_kill_rate": round(sum(r["killed"] for r in audited) / max(1, sum(r["mutants"] for r in audited)), 3),
        "mean_problem_kill_rate": round(statistics.mean(rates), 3) if rates else None,
        "kill_rate_buckets": dict(buckets),
        "mean_kill_rate_by_number_of_tests": {n: {"problems": len(v), "mean": round(statistics.mean(v), 3)}
                                              for n, v in sorted(by_tests.items())},
    }


def differential_check(tools, row, mutant_code, entry, rng):
    """用随机合法输入对拍变异体与 oracle；返回 (不一致次数, 第一个反例)。"""
    inputs = []
    while len(inputs) < DIFF_INPUTS:
        candidate = entry["generate"](rng)
        if entry["validate"](**copy.deepcopy(candidate)):
            inputs.append(candidate)
    mutant_row = copy.deepcopy(row)
    mutant_row["extra_info"]["reference"] = mutant_code
    outputs = tools.reference_outputs(mutant_row, inputs, executor)
    mismatches, first = 0, None
    for kwargs, result in zip(inputs, outputs):
        expected = json.loads(json.dumps(entry["oracle"](**copy.deepcopy(kwargs))))
        if not result.get("ok") or result.get("out") != expected:
            mismatches += 1
            first = first or {"input": kwargs, "mutant": result, "oracle": expected}
    return mismatches, first


def _rebuild(builder, accepted_cases):
    train, test = builder.build_leetcode_rows(accepted_cases)
    return {row["extra_info"]["task_id"]: row for row in train + test}


def augmentation_demo(tools, oracles, builder):
    train, test = builder.build_leetcode_rows()
    by_task = {row["extra_info"]["task_id"]: row for row in train + test}
    accepted_cases, report = {}, {}
    for task_id, entry in oracles.ORACLES.items():
        rng = random.Random(f"candidates:{task_id}")
        before = tools.audit_test_strength(by_task[task_id], executor)
        candidates = [entry["generate"](rng) for _ in range(CANDIDATES_PER_TASK)]
        if task_id == "maximum-subarray":
            candidates += [{"nums": []}, {"nums": [10001]}]  # 两条故意非法的输入，演示校验器
        records = tools.audit_new_cases(by_task[task_id], candidates, executor, oracle=entry["oracle"],
                                        validator=entry["validate"])
        accepted = [r for r in records if r["status"] == "accepted"][:ADDED_CASES_PER_TASK]
        accepted_cases[task_id] = accepted
        report[task_id] = {
            "tests_before": before["n_tests"],
            "kill_before": f"{before['killed']}/{before['mutants']}",
            "survivors_before": [s["change"] for s in before["survivors"]],
            "candidate_status": dict(collections.Counter(r["status"] for r in records)),
            "added_cases": len(accepted),
        }
    augmented = _rebuild(builder, accepted_cases)
    for task_id, entry in oracles.ORACLES.items():
        after_random = tools.audit_test_strength(augmented[task_id], executor)
        guided = []
        for survivor in after_random["survivors"]:
            mismatches, example = differential_check(tools, augmented[task_id], survivor["code"], entry,
                                                     random.Random(f"diff:{task_id}:{survivor['change']}"))
            if example is not None:
                records = tools.audit_new_cases(by_task[task_id], [example["input"]], executor,
                                                oracle=entry["oracle"], validator=entry["validate"],
                                                origin=f"mutation-guided: {survivor['change']}")
                added = [r for r in records if r["status"] == "accepted"
                         and r["case_id"] not in {c["case_id"] for c in accepted_cases[task_id]}]
                accepted_cases[task_id].extend(added)
                guided.append({"change": survivor["change"], "counterexample": example["input"],
                               "added_case_ids": [r["case_id"] for r in added]})
        report[task_id].update({"kill_after_random_cases": f"{after_random['killed']}/{after_random['mutants']}",
                                "mutation_guided_cases": guided})
    augmented = _rebuild(builder, accepted_cases)
    for task_id, entry in oracles.ORACLES.items():
        final = tools.audit_test_strength(augmented[task_id], executor)
        survivors = []
        for survivor in final["survivors"]:
            mismatches, example = differential_check(tools, augmented[task_id], survivor["code"], entry,
                                                     random.Random(f"final:{task_id}:{survivor['change']}"))
            survivors.append({"change": survivor["change"], "differential_mismatches": mismatches,
                              "verdict": "likely_equivalent" if mismatches == 0 else "real_bug_missed",
                              "counterexample": example})
        report[task_id].update({
            "tests_after": final["n_tests"],
            "kill_after": f"{final['killed']}/{final['mutants']}",
            "test_version": augmented[task_id]["extra_info"]["test_version"],
            "survivors_after": survivors,
        })
    # 离参考解很远的错解：允许空子数组。原测试放过它，补测试后应被抓住。
    _, coder1, coder1_plus = env.load_reward_score()
    wrong_empty = ("class Solution:\n    def maxSubArray(self, nums):\n        best = cur = 0\n"
                   "        for x in nums:\n            cur = max(0, cur + x)\n            best = max(best, cur)\n"
                   "        return best\n")
    response = env.wrap_response(wrong_empty)
    original_row, augmented_row = by_task["maximum-subarray"], augmented["maximum-subarray"]
    far_error = {
        "original_tests": coder1_plus._compute_score(response, original_row["reward_model"]["ground_truth"],
                                                     extra_info=original_row["extra_info"])[0],
        "augmented_tests": coder1_plus._compute_score(response, augmented_row["reward_model"]["ground_truth"],
                                                      extra_info=augmented_row["extra_info"])[0],
    }
    return report, accepted_cases, far_error


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="audit all 2090 problems instead of a 200-problem sample")
    parser.add_argument("--skip_strength", action="store_true", help="only run the augmentation demo")
    args = parser.parse_args(argv)

    env.load_reward_score()  # 换上本地执行器
    tools = _load(TOOLS_PATH, "coder1_plus_tests")
    oracles = _load(ORACLES_PATH, "coder1_plus_oracles")
    builder = env.load_builder()
    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    started = time.time()
    checks = {}

    if not args.skip_strength:
        train, test = builder.build_leetcode_rows()
        rows = train + test
        if not args.all:
            rows = random.Random(0).sample(rows, 200)
        with ThreadPoolExecutor(max_workers=8) as pool:
            records = list(pool.map(lambda row: tools.audit_test_strength(row, executor), rows))
        summary = strength_summary(records)
        (out_dir / "test_strength_report.json").write_text(
            json.dumps({"scope": "all" if args.all else "sample-200", "summary": summary, "problems": records},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"strength_audit": summary}, ensure_ascii=False, indent=2))
        checks["strength_audit_ran"] = summary["audited"] > 0

    report, accepted_cases, far_error = augmentation_demo(tools, oracles, builder)
    (out_dir / "accepted_cases.json").write_text(json.dumps(accepted_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "augmentation_report.json").write_text(
        json.dumps({"tasks": report, "far_error_wrong_empty": far_error}, ensure_ascii=False, indent=2), encoding="utf-8")
    for task_id, item in report.items():
        verdicts = collections.Counter(s["verdict"] for s in item["survivors_after"])
        print(f"{task_id:38s} tests {item['tests_before']}->{item['tests_after']}  kill {item['kill_before']}"
              f" -> {item['kill_after_random_cases']} (random) -> {item['kill_after']} (guided)"
              f"  guided cases {sum(len(g['added_case_ids']) for g in item['mutation_guided_cases'])}"
              f"  survivors {dict(verdicts)}")
    print("far error wrong_empty on maximum-subarray:", far_error)

    checks["every_task_got_accepted_cases"] = all(item["added_cases"] > 0 for item in report.values())
    checks["no_surviving_real_bug"] = all(s["verdict"] == "likely_equivalent"
                                          for item in report.values() for s in item["survivors_after"])
    checks["far_error_now_caught"] = far_error == {"original_tests": 1.1, "augmented_tests": 0.1}
    checks["invalid_inputs_rejected"] = report["maximum-subarray"]["candidate_status"].get("invalid", 0) >= 2
    print("checks:", checks, f"elapsed {time.time() - started:.0f}s")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
