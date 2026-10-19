"""验证补强项 4 到 7：受保护判题协议、沙箱执行层、判题故障策略、去污染与难度筛选。

在仓库根目录运行（Python 3.10+，只用标准库；沙箱部分需要 macOS 的 sandbox-exec 或 Linux 的 firejail）：
    $PY -B RLVR学习资料/examples/source-supplements/verify_judge_and_selection.py
报告写到同目录的 outputs/judge_and_selection_report.json。
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import supplement_env as env  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_FILE = env.REPO / "LeetCodeDataset" / "data" / "LeetCodeDataset-v2-rl-problems.jsonl"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ADVERSARIAL = {
    "reference": None,  # 用数据集参考解
    "wrong_sum": "class Solution:\n    def maxSubArray(self, nums):\n        return sum(nums)\n",
    "early_exit_sys": "import sys\nsys.exit(0)\n",
    "early_exit_os": "import os\nos._exit(0)\n",
    "forged_frames": ("import os\nemit = [v for k, v in globals().items() if k.endswith('_emit')][0]\n"
                      "for i in range(3):\n    emit({'case': i, 'status': 'ok', 'repr': '0'})\n"
                      "emit({'event': 'done'})\nos._exit(0)\n"),
    "read_dataset_file": (f"data = open({str(DATA_FILE)!r}).read()\n"
                          "class Solution:\n    def maxSubArray(self, nums):\n        return 0\n"),
}


def sandbox_probes(sandbox):
    backend = sandbox.choose_backend()
    if backend in ("unavailable", "local"):
        return {"backend": backend, "skipped": "no isolating backend on this machine"}
    probe_path = os.path.expanduser("~/.coder1_plus_sandbox_probe")
    programs = {
        "ok": ("print('hello')", 10),
        "runtime_error": ("raise ValueError('x')", 10),
        "compile_error": ("def f(:\n    pass", 10),
        "timeout": ("while True: pass", 2),
        "network": ("import socket\ns = socket.socket(); s.settimeout(2)\ntry:\n    s.connect(('1.1.1.1', 53)); print('CONNECTED')\n"
                    "except OSError as e:\n    print('blocked', type(e).__name__)", 10),
        "write_outside_workdir": (f"try:\n    open({probe_path!r}, 'w').write('x'); print('WROTE')\n"
                                  "except OSError as e:\n    print('blocked', type(e).__name__)", 10),
        "read_repository": (f"try:\n    open({str(DATA_FILE)!r}).read(); print('READ')\n"
                            "except OSError as e:\n    print('blocked', type(e).__name__)", 10),
        "spawned_child_cleanup": ("import subprocess, sys, time\np = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
                                  "print('child', p.pid, flush=True)\ntime.sleep(30)", 2),
    }
    results = {"backend": backend}
    for name, (code, timeout) in programs.items():
        result = sandbox.run_python(code, timeout=timeout)
        item = {"status": result.status, "stdout": result.stdout.strip()[:80], "duration": round(result.duration, 2)}
        if name == "spawned_child_cleanup" and result.stdout.split():
            time.sleep(0.5)
            try:
                os.kill(int(result.stdout.split()[1]), 0)
                item["child_alive_after_timeout"] = True
            except ProcessLookupError:
                item["child_alive_after_timeout"] = False
        results[name] = item
    results["probe_file_left_behind"] = os.path.exists(probe_path)
    results["missing_backend"] = sandbox.run_python("print(1)", backend="firejail" if backend != "firejail" else "sandbox-exec").status
    return results


def main():
    _, _, coder1_plus = env.load_reward_score()
    sandbox = coder1_plus.sandbox
    policy = _load(env.REPO / "verl" / "workers" / "reward_manager" / "infra_policy.py", "infra_policy")
    selection = _load(env.REPO / "examples" / "data_preprocess" / "coder1_plus_selection.py", "coder1_plus_selection")
    builder = env.load_builder()
    train, test = builder.build_leetcode_rows()
    by_task = {row["extra_info"]["task_id"]: row for row in train + test}
    report, checks = {}, {}

    # 补强项 5：沙箱
    probes = sandbox_probes(sandbox)
    report["sandbox"] = probes
    if "skipped" not in probes:
        checks["sandbox_statuses"] = [probes[k]["status"] for k in ("ok", "runtime_error", "compile_error", "timeout")] == \
            ["ok", "runtime_error", "compile_error", "timeout"]
        checks["sandbox_isolation"] = all(probes[k]["stdout"].startswith("blocked")
                                          for k in ("network", "write_outside_workdir", "read_repository"))
        checks["sandbox_cleanup"] = (probes["spawned_child_cleanup"].get("child_alive_after_timeout") is False
                                     and not probes["probe_file_left_behind"])
        checks["sandbox_missing_backend_is_infra_error"] = probes["missing_backend"] == "infra_error"

    # 补强项 4：受保护判题协议
    row = by_task["maximum-subarray"]
    legacy_gt = json.dumps({"functional": json.loads(row["reward_model"]["ground_truth"])["functional"]})
    adversarial = {}
    for name, code in ADVERSARIAL.items():
        response = env.wrap_response(code or row["extra_info"]["reference"])
        original, _ = env.load_reward_score()[1]._compute_score(response, legacy_gt, extra_info=row["extra_info"])
        plus, info = coder1_plus.compute_score_detailed(response, row["reward_model"]["ground_truth"], row["extra_info"])
        adversarial[name] = {"original_coder1": original, "coder1_plus": plus, "exec_status": info.get("exec_status"),
                             "events": info.get("judge", {}).get("events"),
                             "case_verdicts": [c["verdict"] for c in info.get("judge", {}).get("cases", [])]}
    report["adversarial_candidates"] = adversarial
    expected_plus = {"reference": 1.1, "wrong_sum": 0.1, "early_exit_sys": 0.1, "early_exit_os": 0.1,
                     "forged_frames": 0.1, "read_dataset_file": 0.1}
    checks["protocol_adversarial"] = all(adversarial[k]["coder1_plus"] == v for k, v in expected_plus.items())
    checks["original_gap_reproduced"] = adversarial["early_exit_sys"]["original_coder1"] == 1.1
    ladder = by_task["word-ladder-ii"]
    stable = [coder1_plus.compute_score_detailed(env.wrap_response(ladder["extra_info"]["reference"]),
                                                 ladder["reward_model"]["ground_truth"], ladder["extra_info"])[0]
              for _ in range(8)]
    report["word_ladder_ii_unordered_compare"] = stable
    checks["any_order_stable"] = all(score == 1.1 for score in stable)

    # 补强项 6：判题故障策略
    uids = ["q1"] * 4 + ["q2"] * 4
    scores = [0.1, 0.1, 0.1, None, 1.1, 1.1, 0.1, 0.1]
    statuses = ["scored"] * 3 + ["infra_error"] + ["scored"] * 4
    legacy, _ = policy.resolve_group_failures(scores, statuses, uids, policy="legacy_zero")
    neutral, stats = policy.resolve_group_failures(scores, statuses, uids)
    report["infra_policy"] = {
        "legacy_advantages": [round(a, 3) for a in policy.grpo_advantages(legacy, uids)],
        "neutralized_advantages": [round(a, 3) for a in policy.grpo_advantages(neutral, uids)],
        "stats": stats,
    }
    checks["infra_group_neutralized"] = report["infra_policy"]["neutralized_advantages"][:4] == [0.0] * 4

    # 补强项 7：去污染与难度筛选
    v1_test = [json.loads(line) for line in (env.REPO / "LeetCodeDataset/data/LeetCodeDataset-v1-test-problems.jsonl")
               .read_text(encoding="utf-8").splitlines()]
    eval_items = [{"id": r["task_id"], "text": r["meta"]["query"], "question_id": r["meta"]["question_id"],
                   "task_id": r["task_id"]} for r in v1_test]
    flagged = selection.find_contamination(train, eval_items)
    report["decontamination_vs_v1_test"] = {
        "pairs": len(flagged),
        "train_tasks": len({f["task_id"] for f in flagged}),
        "same_question_id": sum(1 for f in flagged if f["reason"] == "same_question_id"),
        "ngram_only": [f for f in flagged if f["reason"] == "ngram_overlap"],
    }
    checks["decontamination_finds_id_overlap"] = report["decontamination_vs_v1_test"]["same_question_id"] == 61
    keep, dropped = selection.select_by_pass_rate({"easy": {"n": 16, "c": 16}, "hard": {"n": 16, "c": 0},
                                                   "useful": {"n": 16, "c": 5}, "too_few": {"n": 2, "c": 1}})
    report["difficulty_selection_example"] = {"keep": keep, "dropped": dropped}
    checks["difficulty_keeps_mixed_tasks_only"] = keep == ["useful"]

    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "judge_and_selection_report.json").write_text(json.dumps({"report": report, "checks": checks},
                                                                        ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sandbox_backend": probes.get("backend"), "adversarial": {k: (v["original_coder1"], v["coder1_plus"])
                                                                              for k, v in adversarial.items()},
                      "infra_policy": report["infra_policy"], "decontamination": {
                          k: v for k, v in report["decontamination_vs_v1_test"].items() if k != "ngram_only"},
                      "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
