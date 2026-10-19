"""反向补充源码的单元测试：只用标准库与 CPU，几秒内跑完。

在仓库根目录运行：
    $PY -B RLVR学习资料/examples/source-supplements/test_source_supplements.py
"""
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import supplement_env as env  # noqa: E402

ROUTER, CODER1, PLUS = env.load_reward_score()
BUILDER = env.load_builder()
TRAIN, TEST = BUILDER.build_leetcode_rows()
BY_TASK = {row["extra_info"]["task_id"]: row for row in TRAIN + TEST}
TOOLS = sys.modules["coder1_plus_tests"]  # 由 coder1_plus.py 导入，补强项 2 和 3 的工具
PROTOCOL, SANDBOX = PLUS.protocol, PLUS.sandbox  # 补强项 4、5
HAS_SANDBOX = SANDBOX.choose_backend() in ("sandbox-exec", "firejail")


def _load_path(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


POLICY = _load_path(env.REPO / "verl/workers/reward_manager/infra_policy.py", "infra_policy")  # 补强项 6
SELECTION = _load_path(env.REPO / "examples/data_preprocess/coder1_plus_selection.py", "coder1_plus_selection")  # 补强项 7

WRONG_SUM = "class Solution:\n    def maxSubArray(self, nums):\n        return sum(nums)\n"
EARLY_EXIT = "import sys\nsys.exit(0)\n"


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def reference_candidate(task_id):
    row = BY_TASK[task_id]
    gt = json.loads(row["reward_model"]["ground_truth"])
    return row, gt, env.self_contained_candidate(gt["prelude"], row["extra_info"]["reference"])


class GroundTruthContract(unittest.TestCase):

    def test_router_scores_tree_problem_only_under_plus_contract(self):
        row, gt, candidate = reference_candidate("create-binary-tree-from-descriptions")
        response = env.wrap_response(candidate)
        plus = quiet(ROUTER._default_compute_score, "code_plus", response, row["reward_model"]["ground_truth"],
                     extra_info=row["extra_info"])
        original = quiet(ROUTER._default_compute_score, "code", response, json.dumps({"functional": gt["functional"]}),
                         extra_info=row["extra_info"])
        self.assertEqual(plus, 1.1)
        self.assertEqual(original, 0.1, "original contract cannot see tree_node / is_same_tree")

    def test_ground_truth_without_prelude_is_delegated_to_original(self):
        row, gt, candidate = reference_candidate("maximum-subarray")
        legacy_gt = json.dumps({"functional": gt["functional"]})
        for code, expected in ((candidate, 1.1), (WRONG_SUM, 0.1)):
            response = env.wrap_response(code)
            plus, _ = PLUS._compute_score(response, legacy_gt, extra_info=row["extra_info"])
            original, _ = CODER1._compute_score(response, legacy_gt, extra_info=row["extra_info"])
            self.assertEqual(plus, original)
            self.assertEqual(plus, expected)

    def test_format_gate_and_wrong_answer_unchanged(self):
        row, _, candidate = reference_candidate("maximum-subarray")
        gt = row["reward_model"]["ground_truth"]
        bad_format, _ = PLUS._compute_score("```python\n" + candidate + "\n```", gt, extra_info=row["extra_info"])
        wrong, _ = PLUS._compute_score(env.wrap_response(WRONG_SUM), gt, extra_info=row["extra_info"])
        self.assertEqual(bad_format, -1.1)
        self.assertEqual(wrong, 0.1)

    def test_early_exit_is_now_rejected(self):
        # 原版只看退出码，提前退出拿 1.1；补强项 4 的受保护协议发现结果缺失，判 0.1。
        row = BY_TASK["maximum-subarray"]
        legacy_gt = json.dumps({"functional": json.loads(row["reward_model"]["ground_truth"])["functional"]})
        original, _ = CODER1._compute_score(env.wrap_response(EARLY_EXIT), legacy_gt, extra_info=row["extra_info"])
        score, _ = PLUS._compute_score(env.wrap_response(EARLY_EXIT), row["reward_model"]["ground_truth"],
                                       extra_info=row["extra_info"])
        self.assertEqual((original, score), (1.1, 0.1))

    def test_prelude_runs_before_solution_and_tests(self):
        program = PLUS.build_program("PRELUDE = 1", "SOLUTION = 2", "TESTS = 3")
        self.assertLess(program.index("PRELUDE"), program.index("SOLUTION"))
        self.assertLess(program.index("SOLUTION"), program.index("TESTS"))


class PreprocessParity(unittest.TestCase):

    def test_selection_and_index_match_original_leetcode2k(self):
        self.assertEqual((len(TRAIN), len(TEST)), (1890, 200))
        row = BY_TASK["maximum-subarray"]
        self.assertEqual(row["extra_info"]["index"], 142)
        self.assertEqual(row["extra_info"]["source_line_1_based"], 143)
        self.assertEqual(row["extra_info"]["split"], "train")

    def test_prompt_is_byte_identical_to_original_format(self):
        row = BY_TASK["maximum-subarray"]
        raw = json.loads((env.REPO / "LeetCodeDataset/data/LeetCodeDataset-v2-rl-problems.jsonl")
                         .read_text(encoding="utf-8").splitlines()[142])
        self.assertEqual(row["prompt"][0]["content"], BUILDER.original_system_prompt())
        self.assertEqual(row["prompt"][1]["content"], BUILDER.USER_PROMPT_PREFIX + raw["meta"]["query"].strip())
        self.assertEqual(row["extra_info"]["prompt"], row["prompt"][1]["content"])

    def test_ground_truth_keeps_original_functional_and_adds_prelude(self):
        row = BY_TASK["maximum-subarray"]
        raw = json.loads((env.REPO / "LeetCodeDataset/data/LeetCodeDataset-v2-rl-problems.jsonl")
                         .read_text(encoding="utf-8").splitlines()[142])
        gt = json.loads(row["reward_model"]["ground_truth"])
        self.assertEqual(gt["functional"], f"{raw['test']}\n\ncheck({raw['entry_point'].strip()})")
        self.assertEqual(gt["prelude"], raw["prompt"])
        self.assertEqual(row["data_source"], "code_plus")
        self.assertTrue({"split", "index", "reference", "prompt", "dataset"} <= set(row["extra_info"]))


WRONG_EMPTY = ("class Solution:\n    def maxSubArray(self, nums):\n        best = cur = 0\n"
               "        for x in nums:\n            cur = max(0, cur + x)\n            best = max(best, cur)\n"
               "        return best\n")


def _max_subarray_oracle(nums):
    return max(sum(nums[i:j + 1]) for i in range(len(nums)) for j in range(i, len(nums)))


def _local(code):
    return env.local_exec(code, timeout=5)


class TestStrengthAudit(unittest.TestCase):

    def test_mutants_are_single_change_and_deterministic(self):
        reference = BY_TASK["maximum-subarray"]["extra_info"]["reference"]
        first = TOOLS.generate_mutants(reference, max_mutants=5, seed=0)
        self.assertEqual(first, TOOLS.generate_mutants(reference, max_mutants=5, seed=0))
        original = TOOLS.ast.unparse(TOOLS.ast.parse(reference))
        for change, code in first:
            self.assertNotEqual(code, original, change)
            differing = [a for a, b in zip(code.splitlines(), original.splitlines()) if a != b]
            self.assertEqual(len(differing), 1, change)

    def test_audit_reports_kill_rate_for_leetcode_53(self):
        record = TOOLS.audit_test_strength(BY_TASK["maximum-subarray"], _local)
        self.assertTrue(record["baseline_ok"])
        self.assertEqual((record["killed"], record["mutants"]), (9, 9))

    def test_audit_finds_weak_tests(self):
        record = TOOLS.audit_test_strength(BY_TASK["number-of-senior-citizens"], _local)
        self.assertEqual([s["change"] for s in record["survivors"]], ["line 3: > -> >="])


class TestAugmentation(unittest.TestCase):

    def test_case_statuses_follow_the_evidence(self):
        row = BY_TASK["maximum-subarray"]
        candidates = [{"nums": [-7]}, {"nums": [-7]}, {"nums": []}, {"nums": [2, -1, 2]}]
        validate = lambda nums: 1 <= len(nums) <= 10**5 and all(-10**4 <= x <= 10**4 for x in nums)
        records = TOOLS.audit_new_cases(row, candidates, _local, oracle=_max_subarray_oracle, validator=validate)
        self.assertEqual([r["status"] for r in records], ["accepted", "invalid", "accepted"])  # 重复输入只留一条
        wrong_oracle = lambda nums: _max_subarray_oracle(nums) + 1
        self.assertEqual(TOOLS.audit_new_cases(row, [{"nums": [3]}], _local, oracle=wrong_oracle)[0]["status"],
                         "quarantined")
        self.assertEqual(TOOLS.audit_new_cases(row, [{"nums": [3]}], _local)[0]["status"], "pending")

    def test_only_accepted_cases_enter_ground_truth_and_catch_far_error(self):
        row = BY_TASK["maximum-subarray"]
        records = TOOLS.audit_new_cases(row, [{"nums": [-7]}], _local, oracle=_max_subarray_oracle)
        pending = TOOLS.audit_new_cases(row, [{"nums": [-9]}], _local)
        train, _ = BUILDER.build_leetcode_rows({"maximum-subarray": records + pending})
        augmented = next(r for r in train if r["extra_info"]["task_id"] == "maximum-subarray")
        functional = json.loads(augmented["reward_model"]["ground_truth"])["functional"]
        self.assertIn(records[0]["case_id"], functional)
        self.assertNotIn(pending[0]["case_id"], functional)
        self.assertEqual(augmented["extra_info"]["test_version"], "leetcode-examples+aug1")
        response = env.wrap_response(WRONG_EMPTY)
        original_score, _ = PLUS._compute_score(response, row["reward_model"]["ground_truth"], extra_info=row["extra_info"])
        augmented_score, _ = PLUS._compute_score(response, augmented["reward_model"]["ground_truth"],
                                                 extra_info=augmented["extra_info"])
        self.assertEqual((original_score, augmented_score), (1.1, 0.1))


class TestProtectedProtocol(unittest.TestCase):

    def test_parse_cases_covers_value_tree_and_list_asserts(self):
        for task_id, kind in (("maximum-subarray", "value"), ("create-binary-tree-from-descriptions", "tree"),
                              ("reverse-linked-list-ii", "list")):
            gt = json.loads(BY_TASK[task_id]["reward_model"]["ground_truth"])
            entry, cases = PROTOCOL.parse_cases(gt["functional"])
            self.assertTrue(entry.startswith("Solution()."), task_id)
            self.assertEqual({case["kind"] for case in cases}, {kind}, task_id)

    def test_judge_requires_every_result_and_completion(self):
        cases = [{"label": "a", "kind": "value", "expected": [1, 2]}, {"label": "b", "kind": "value", "expected": 3}]
        mark = PROTOCOL.NONCE_PREFIX + "n0 "
        frames = [mark + json.dumps({"event": "start"}), mark + json.dumps({"case": 0, "status": "ok", "repr": "[2, 1]"}),
                  mark + json.dumps({"case": 1, "status": "ok", "repr": "3"})]
        done = mark + json.dumps({"event": "done"})
        self.assertFalse(PROTOCOL.judge("\n".join(frames + [done]), cases, "n0")["passed"])  # 顺序不同
        self.assertTrue(PROTOCOL.judge("\n".join(frames + [done]), cases, "n0", "unordered_outer")["passed"])
        self.assertFalse(PROTOCOL.judge("\n".join(frames), cases, "n0", "unordered_outer")["passed"])  # 没有完成事件
        forged = "\n".join(frames + [done]).replace(PROTOCOL.NONCE_PREFIX + "n0", PROTOCOL.NONCE_PREFIX + "zz")
        self.assertFalse(PROTOCOL.judge(forged, cases, "n0")["passed"])  # 标记不对的行一律不认

    @unittest.skipUnless(HAS_SANDBOX, "needs sandbox-exec or firejail")
    def test_early_exit_and_forgery_fail_under_protocol(self):
        row = BY_TASK["maximum-subarray"]
        forged = ("import os\nemit = [v for k, v in globals().items() if k.endswith('_emit')][0]\n"
                  "for i in range(3):\n    emit({'case': i, 'status': 'ok', 'repr': '0'})\n"
                  "emit({'event': 'done'})\nos._exit(0)\n")
        for code in ("import os\nos._exit(0)\n", forged):
            score, info = PLUS.compute_score_detailed(env.wrap_response(code), row["reward_model"]["ground_truth"],
                                                      row["extra_info"])
            self.assertEqual(score, 0.1, info["log"])


@unittest.skipUnless(HAS_SANDBOX, "needs sandbox-exec or firejail")
class TestSandbox(unittest.TestCase):

    def test_statuses(self):
        for code, status in (("print(1)", "ok"), ("raise ValueError", "runtime_error"), ("def f(:", "compile_error")):
            self.assertEqual(SANDBOX.run_python(code, timeout=10).status, status, code)
        self.assertEqual(SANDBOX.run_python("while True: pass", timeout=1).status, "timeout")

    def test_isolation_blocks_network_and_repository_reads(self):
        code = ("import socket\ntry:\n    socket.socket().connect(('1.1.1.1', 53)); print('net-open')\n"
                "except OSError:\n    print('net-blocked')\n"
                f"try:\n    open({str(env.BUILDER_PATH)!r}).read(); print('repo-open')\n"
                "except OSError:\n    print('repo-blocked')\n")
        self.assertEqual(SANDBOX.run_python(code, timeout=10).stdout.split(), ["net-blocked", "repo-blocked"])


class TestInfraPolicy(unittest.TestCase):

    def test_group_is_neutralized_instead_of_zero_filled(self):
        uids, statuses = ["q"] * 4, ["scored"] * 3 + ["infra_error"]
        legacy, _ = POLICY.resolve_group_failures([0.1, 0.1, 0.1, None], statuses, uids, policy="legacy_zero")
        neutral, stats = POLICY.resolve_group_failures([0.1, 0.1, 0.1, None], statuses, uids)
        self.assertEqual([round(a, 2) for a in POLICY.grpo_advantages(legacy, uids)], [0.5, 0.5, 0.5, -1.5])
        self.assertEqual(POLICY.grpo_advantages(neutral, uids), [0.0] * 4)
        self.assertEqual(stats["neutralized_groups"], 1)

    def test_retries_then_gives_up(self):
        attempts = {"n": 0}

        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise RuntimeError("sandbox hiccup")
            return 1.1, {"status": "passed"}

        self.assertEqual(POLICY.score_with_retries(flaky, max_retries=2), (1.1, "scored", 2))
        self.assertEqual(POLICY.score_with_retries(lambda: (None, {"status": "infra_error"}), 1), (None, "infra_error", 2))


class TestSelection(unittest.TestCase):

    def test_decontamination_finds_same_id_and_reworded_copies(self):
        rows = [{"extra_info": {"task_id": "a", "question_id": 1, "prompt": "Given an array of integers return the sum of all positive values in it please"}},
                {"extra_info": {"task_id": "b", "question_id": 2, "prompt": "Count the islands in a grid of land and water cells"}}]
        evals = [{"id": "x", "question_id": 2, "text": "unrelated"},
                 {"id": "y", "text": "Given an array of integers return the sum of all positive values in it"}]
        reasons = {(f["task_id"], f["reason"]) for f in SELECTION.find_contamination(rows, evals, n=5)}
        self.assertEqual(reasons, {("b", "same_question_id"), ("a", "ngram_overlap")})

    def test_difficulty_keeps_only_tasks_with_mixed_outcomes(self):
        keep, dropped = SELECTION.select_by_pass_rate({"all": {"n": 8, "c": 8}, "none": {"n": 8, "c": 0},
                                                       "mixed": {"n": 8, "c": 3}, "few": {"n": 2, "c": 1}})
        self.assertEqual(keep, ["mixed"])
        self.assertEqual(set(dropped), {"all", "none", "few"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
