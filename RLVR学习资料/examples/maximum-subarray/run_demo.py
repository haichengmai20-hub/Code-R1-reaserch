"""CPU / standard-library diagnostic for one pinned public LeetCode task.

Only reviewed, fixed candidates are executable. This is not an arbitrary-code
runner, a sandbox, a model rollout, or a training run.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Optional, Tuple


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUTPUTS = HERE / "outputs"
DATASET = REPO / "LeetCodeDataset/data/LeetCodeDataset-v2-rl-problems.jsonl"
SCORER = REPO / "verl/utils/reward_score/coder1/__init__.py"
PREPROCESS = REPO / "examples/data_preprocess/coder1.py"
CORE = REPO / "verl/trainer/ppo/core_algos.py"
CODE_REV = "443f8da8058d07051793eff875da117534fd1ba1"
DATA_REV = "f3519e6b7f42265dd4e4d35d4812e8c29b08111a"
TEST_VERSION = "maximum-subarray-diagnostic-v1"
ACTOR_PROMPT_APPEND = """Explicit task constraints and return contract:
- nums is a non-empty List[int].
- 1 <= len(nums) <= 100000.
- For every element x in nums, -10000 <= x <= 10000.
- Return an int: the maximum sum of a non-empty contiguous subarray of nums.
- Empty subarrays are not allowed."""
RAW_LINE_SHA = "cd4ecaa1bf8d73bc9b6b926d6b5c43a58f5eed5386c5a9f95fd0bac9fcf70a79"
REF_SHA = "32ec6c7071ebeae28a05e24a7f5c3a7010f2b92b713987d753ec2c1ba2452ca8"
# Text hashes normalize CRLF to LF; the raw record snapshot keeps its terminator.
SOURCE_HASHES = {
    "scorer": "78186d532c246261542d548a8edfaeacfa06b7e7dfa4f03a21f6270a0fcf05f3",
    "preprocess": "322556c03faf3d4da6cf131889438c26ba48d6ec5076901f093686508e8f910f",
    "core": "74d680b9186690a7788157679725bc351a5436b6419ff8c7a2b570bc4cbf13ee",
}
REFERENCE = """from typing import List
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        ans = f = nums[0]
        for x in nums[1:]:
            f = max(f, 0) + x
            ans = max(ans, f)
        return ans
"""
CORRECT_PREFIX = """class Solution:
    def maxSubArray(self, nums):
        prefix = lowest = 0
        best = nums[0]
        for value in nums:
            prefix += value
            best = max(best, prefix - lowest)
            lowest = min(lowest, prefix)
        return best
"""
WRONG_EMPTY = """class Solution:
    def maxSubArray(self, nums):
        ending = best = 0
        for value in nums:
            ending = max(0, ending + value)
            best = max(best, ending)
        return best
"""
WRONG_SUM = """class Solution:
    def maxSubArray(self, nums):
        return sum(nums)
"""
EARLY_EXIT = "import sys\nsys.exit(0)\n"
STDIO_CORRECT = """import json, sys
nums = json.loads(sys.stdin.read())
prefix = lowest = 0
best = nums[0]
for value in nums:
    prefix += value
    best = max(best, prefix - lowest)
    lowest = min(lowest, prefix)
print(best)
"""
STDIO_WRONG = "import json, sys\nprint(sum(json.loads(sys.stdin.read())))\n"


def digest(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def readonly_source_fingerprints():
    result = {}
    for path in (DATASET, SCORER, PREPROCESS, CORE):
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        result[path.relative_to(REPO).as_posix()] = hasher.hexdigest()
    return result


def dump(name, value, jsonl=False):
    output_root = OUTPUTS.resolve()
    if not output_root.is_relative_to(HERE) or output_root == HERE:
        raise RuntimeError("Output directory resolves outside the authorized demo asset directory.")
    target = OUTPUTS / name
    if not target.resolve().is_relative_to(output_root):
        raise RuntimeError("Output file resolves outside the fixed outputs directory.")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    text = canonical(value) + "\n" if jsonl else json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    target.write_text(text, encoding="utf-8", newline="\n")
    # Read back bytes and JSON, rather than treating write success as validation.
    written = target.read_bytes()
    assert not written.startswith(b"\xef\xbb\xbf")
    assert json.loads(written.decode("utf-8")) == value


def check_pinned_sources():
    checked = {}
    for key, path in (("scorer", SCORER), ("preprocess", PREPROCESS), ("core", CORE)):
        observed = digest(path.read_text(encoding="utf-8"))
        if observed != SOURCE_HASHES[key]:
            raise RuntimeError(f"Reviewed {key} source has changed; stop before executing it.")
        checked[key] = {"path": path.relative_to(REPO).as_posix(), "lf_text_sha256": observed}
    return checked


def load_raw():
    with DATASET.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number == 143:
                break
        else:
            raise RuntimeError("Pinned dataset record at line 143 is missing.")
    raw_bytes = line.rstrip(b"\r\n")
    if digest(raw_bytes) != RAW_LINE_SHA:
        raise RuntimeError("Pinned raw record has changed; no unreviewed reference will execute.")
    raw = json.loads(raw_bytes)
    assert raw["task_id"] == "maximum-subarray"
    assert raw["entry_point"] == "Solution().maxSubArray"
    assert raw["meta"]["split"] == "rl"
    assert digest(raw["completion"]) == REF_SHA
    assert REFERENCE == "from typing import List\n" + raw["completion"]
    snapshot = {
        "source": {"dataset_file": DATASET.relative_to(REPO).as_posix(), "line_1_based": 143,
                   "code_r1_revision": CODE_REV, "dataset_revision": DATA_REV,
                   "original_split": "rl", "raw_record_sha256_without_terminator": RAW_LINE_SHA},
        "original_jsonl_line_utf8": line.decode("utf-8"),
        "raw_record": raw,
    }
    assert json.loads(snapshot["original_jsonl_line_utf8"]) == raw
    return raw, snapshot


def extract_seed_cases(raw):
    tree = ast.parse(raw["test"])
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    fn = tree.body[0]
    assert fn.name == "check" and [a.arg for a in fn.args.args] == ["candidate"]
    cases = []
    for index, statement in enumerate(fn.body, 1):
        assert isinstance(statement, ast.Assert)
        comparison = statement.test
        assert isinstance(comparison, ast.Compare) and len(comparison.ops) == 1
        assert isinstance(comparison.ops[0], ast.Eq)
        call = comparison.left
        assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
        assert call.func.id == "candidate" and not call.args and len(call.keywords) == 1
        assert call.keywords[0].arg == "nums"
        nums = ast.literal_eval(call.keywords[0].value)
        expected = ast.literal_eval(comparison.comparators[0])
        cases.append(make_case(f"seed-{index}", nums, expected, {
            "kind": "original_seed", "dataset_line_1_based": 143,
            "test_line_1_based": statement.lineno, "original_split": "rl",
        }))
    assert len(cases) == 3
    # The examples and executable asserts must describe the same atomic pairs.
    if len(cases) != len(raw["examples"]):
        raise ValueError("Seed/assert and example counts differ; zip would silently truncate.")
    for case, example in zip(cases, raw["examples"]):
        assignment = ast.parse(example["input"]).body[0]
        assert isinstance(assignment, ast.Assign)
        assert ast.literal_eval(assignment.value) == case["input"]["nums"]
        assert ast.literal_eval(example["output"]) == case["expected"]
    return cases


def make_case(case_id, nums, expected, origin):
    value = {"nums": nums}
    return {"case_id": case_id, "input": value, "expected": expected,
            "input_sha256": digest(canonical(value)), "origin": origin}


def diagnostic_cases():
    positives = [
        ("diag-all-negative", [-3, -2, -5], -2),
        ("diag-restart", [2, -5, 3, 4, -1], 7),
        ("diag-zero", [0, 0], 0),
        ("diag-single-negative", [-7], -7),
        ("diag-boundary", [10000, -10000, 10000], 10000),
    ]
    negatives = [
        ("control-wrong-pair", [-3, -2, -5], 6),
        ("control-mechanical-swap", [1], 23),
        ("control-empty", [], 0),
        ("control-illegal-element", [1, "2"], 3),
        ("control-bool-element", [True], 1),
        ("control-out-of-range", [10001], 10001),
        ("control-conflicting-expectation", [1], 2),
    ]
    result = []
    for kind, values in (("verified_diagnostic_proposal", positives), ("intentional_negative_control", negatives)):
        for case_id, nums, expected in values:
            result.append(make_case(case_id, nums, expected, {"kind": kind, "test_version": TEST_VERSION}))
    return result


def legality(nums):
    if not isinstance(nums, list) or not nums:
        return "input_must_be_nonempty_list"
    if len(nums) > 100000:
        return "input_length_exceeds_constraint"
    if any(type(value) is not int for value in nums):
        return "input_elements_must_be_integers_excluding_bool"
    if any(abs(value) > 10000 for value in nums):
        return "input_element_out_of_range"
    return None


def brute_force(nums):
    """Enumerate NONEMPTY contiguous slices; independent specification oracle."""
    best = None
    witness = None
    for left in range(len(nums)):
        subtotal = 0
        for right in range(left, len(nums)):
            subtotal += nums[right]
            if best is None or subtotal > best:
                best = subtotal
                witness = {"start_inclusive": left, "end_exclusive": right + 1,
                           "subarray": nums[left:right + 1]}
    return best, witness


def kadane(nums):
    best = ending = nums[0]
    for value in nums[1:]:
        ending = max(value, ending + value)
        best = max(best, ending)
    return best


def controlled_exec(code, allowed, stdin=None):
    """Local compatibility backend, restricted to exact known code strings.

    -I and a temporary cwd reduce ambient dependencies. They do NOT provide OS
    isolation. No arbitrary path, prompt, response, code, or GT CLI is accepted.
    """
    if code not in allowed:
        raise RuntimeError("Code is outside the reviewed fixed-candidate allowlist.")
    env = {"SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
           "PATH": os.defpath, "PYTHONIOENCODING": "utf-8"}
    with tempfile.TemporaryDirectory(prefix="maximum-subarray-") as directory:
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code], input=stdin,
                                capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=5, cwd=directory, env=env)
    if result.returncode == 0:
        return True, result.stdout
    return False, "Failed to execute program: STDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr.strip()


def collect_outputs(code, nums_list):
    if code not in (REFERENCE, CORRECT_PREFIX, WRONG_EMPTY, WRONG_SUM, EARLY_EXIT):
        raise RuntimeError("Only the reviewed fixed function candidates may be audited.")
    # JSON representation supplies only already constructed literals, never code.
    harness = "\nimport json\n_demo_inputs = json.loads(" + repr(canonical(nums_list)) + ")\n"
    harness += "print(json.dumps([Solution().maxSubArray(nums=list(x)) for x in _demo_inputs]))\n"
    source = code + harness
    succ, output = controlled_exec(source, {source})
    if not succ:
        raise RuntimeError(output)
    # A successful early exit leaves no completion payload and is not correctness.
    return json.loads(output) if output.strip() else None


def verify_cases(proposals):
    # Alignment failures are rejected before any reference execution. These
    # checks only validate case data; they do not expand executable code access.
    seen_ids = set()
    for case in proposals:
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            raise ValueError("Case IDs must be nonempty strings and unique.")
        seen_ids.add(case_id)
        if case["input_sha256"] != digest(canonical(case["input"])):
            raise ValueError(f"Input hash mismatch for case_id={case_id}.")
    legal_inputs = [c["input"]["nums"] for c in proposals if legality(c["input"]["nums"]) is None]
    reference_values = iter(collect_outputs(REFERENCE, legal_inputs))
    accepted, quarantined, seen = [], [], {}
    for proposal in proposals:
        case = dict(proposal)
        reasons = []
        nums = case["input"]["nums"]
        invalid = legality(nums)
        if invalid:
            reasons.append(invalid)
            case["verification"] = {"oracle_status": "not_applicable_to_illegal_input"}
        else:
            brute, witness = brute_force(nums)
            dynamic = kadane(nums)
            reference = next(reference_values)
            assert brute == dynamic == reference, (case["case_id"], nums, brute, dynamic, reference)
            case["verification"] = {"nonempty_bruteforce": brute, "witness": witness,
                                    "independent_kadane": dynamic, "reviewed_reference": reference}
            if type(case["expected"]) is not int or case["expected"] != brute:
                reasons.append("expected_mismatch_with_independent_oracles")
            prior = seen.get(case["input_sha256"])
            if prior is not None and prior["expected"] != case["expected"]:
                reasons.append("conflicting_expectation_for_same_input")
                case["conflict_with_case_id"] = prior["case_id"]
        case["status"] = "quarantined" if reasons else "accepted"
        case["reasons"] = reasons
        if reasons:
            quarantined.append(case)
        else:
            accepted.append(case)
            seen[case["input_sha256"]] = case
    assert len(accepted) == 8 and len(quarantined) == 7
    assert all(c["origin"]["kind"] != "intentional_negative_control" for c in accepted)
    assert all(c["origin"]["kind"] == "intentional_negative_control" for c in quarantined)
    all_negative = [-3, -2, -5]
    hand_check = [{"input": {"nums": all_negative}, "subarray": all_negative[i:j],
                   "sum": sum(all_negative[i:j])}
                  for i in range(3) for j in range(i + 1, 4)]
    assert [r["sum"] for r in hand_check] == [-3, -5, -10, -2, -7, -5]
    return {"test_version": TEST_VERSION, "accepted": accepted, "quarantined": quarantined,
            "summary": {"proposals": 15, "original_seeds": 3, "accepted": 8, "quarantined": 7},
            "hand_check_all_negative": {"input": {"nums": all_negative}, "expected": -2,
                                        "all_nonempty_subarrays": hand_check}}


def generate_functional(cases):
    if not cases or any(c["status"] != "accepted" for c in cases):
        raise ValueError("GT generation requires a nonempty list of accepted atomic cases.")
    lines = ["def check(candidate):"]
    for case in cases:
        assert legality(case["input"]["nums"]) is None
        assert case["input_sha256"] == digest(canonical(case["input"]))
        assert brute_force(case["input"]["nums"])[0] == case["expected"]
        lines.append(f"    assert candidate(nums={case['input']['nums']!r}) == {case['expected']!r}, {case['case_id']!r}")
    lines.extend(["", "check(Solution().maxSubArray)", ""])
    return "\n".join(lines)


def extract_system_prompt():
    tree = ast.parse(PREPROCESS.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SYSTEM_PROMPT" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("SYSTEM_PROMPT assignment missing.")


def prepare():
    sources = check_pinned_sources()
    raw, snapshot = load_raw()
    cases = verify_cases(extract_seed_cases(raw) + diagnostic_cases())
    functional = generate_functional(cases["accepted"])
    prompt = ("Please solve the programming task below using a self-contained code snippet "
              "in a markdown code block.\n\n" + raw["meta"]["query"] + "\n\n" + ACTOR_PROMPT_APPEND)
    actor_prompt_validation = {
        "original_query_preserved_verbatim": raw["meta"]["query"] in prompt,
        "exact_constraint_append_present": prompt.endswith(ACTOR_PROMPT_APPEND),
        "nonempty_list_int_contract_present": "nums is a non-empty List[int]." in prompt,
        "length_constraint_present": "1 <= len(nums) <= 100000." in prompt,
        "element_range_constraint_present": "-10000 <= x <= 10000." in prompt,
        "int_and_nonempty_contiguous_return_contract_present":
            "Return an int: the maximum sum of a non-empty contiguous subarray of nums." in prompt,
        "empty_subarray_explicitly_forbidden": "Empty subarrays are not allowed." in prompt,
        "functional_gt_not_in_actor_prompt": functional not in prompt and "def check(candidate):" not in prompt,
        "original_test_not_in_actor_prompt": raw["test"] not in prompt,
        "reference_body_not_in_actor_prompt": REFERENCE not in prompt and raw["completion"] not in prompt,
        "raw_record_sha256_unchanged":
            digest(snapshot["original_jsonl_line_utf8"].encode("utf-8").rstrip(b"\r\n")) == RAW_LINE_SHA,
    }
    if not all(actor_prompt_validation.values()):
        raise RuntimeError("Actor prompt adaptation failed preservation, contract, or hidden-evidence checks.")
    row = {
        "data_source": "code",
        "prompt": [{"role": "system", "content": extract_system_prompt()}, {"role": "user", "content": prompt}],
        "ability": "coding", "reward_model": {"style": "rule", "ground_truth": canonical({"functional": functional})},
        "extra_info": {
            "split": "diagnostic", "original_corpus_split": "rl", "index": 142,
            "index_semantics": "0-based raw JSONL record position; not the filtered/deduplicated Code-R1 pool index.",
            "source_file": DATASET.relative_to(REPO).as_posix(),
            "source_line_1_based": 143,
            "source_question_id": raw["meta"]["question_id"],
            "source_difficulty": raw["meta"]["difficulty"],
            "task_id": raw["task_id"], "entry_point": raw["entry_point"], "reference": raw["completion"],
            "prompt": prompt, "dataset": "LeetCodeDataset", "test_version": TEST_VERSION,
            "code_r1_revision": CODE_REV, "dataset_revision": DATA_REV,
            "raw_record_sha256": RAW_LINE_SHA, "ground_truth_sha256": digest(functional),
            "case_index": [{k: c[k] for k in ("case_id", "input", "input_sha256", "expected", "origin")}
                           for c in cases["accepted"]],
            "scope": "Controlled diagnostic; not an official train/test artifact or a held-out evaluation.",
            "actor_prompt_adaptation": {
                "source_field": ["meta.query", "meta.question_title"],
                "official_problem_url": "https://leetcode.com/problems/maximum-subarray/",
                "append_text": ACTOR_PROMPT_APPEND,
                "scope": "Reviewed diagnostic adaptation: preserve the complete original meta.query and append explicit constraints/return contract reviewed from the pinned task statement. This append is not present in the original Code-R1 preprocessing implementation; the URL identifies the problem, not a live webpage verification.",
            },
            "data_acceptance": "8 input/expected pairs accepted for this diagnostic; 7 faulty control rows isolated.",
            "train_ready": False,
            "train_ready_reasons": ["Diagnostic exercise, not an approved training dataset.",
                                    "Original scorer has the demonstrated returncode-only test-bypass gap.",
                                    "Local fixed-candidate backend has not established safe execution of unknown model code."],
            "preprocess_mapping": "Original rl file enters the prospective train pool before overlap filtering/deduplication.",
        },
    }
    dump("raw_task.json", snapshot)
    dump("cases.json", cases)
    dump("prepared_task.json", row)
    dump("data_pool.jsonl", row, jsonl=True)
    read_row = json.loads((OUTPUTS / "data_pool.jsonl").read_text(encoding="utf-8"))
    assert read_row == row
    actor_prompt_validation["actor_prompt_readback_matches"] = read_row["prompt"][1]["content"] == prompt
    source_metadata_readback_validation = {
        "source_file_matches_raw_snapshot": read_row["extra_info"]["source_file"] == snapshot["source"]["dataset_file"],
        "source_line_matches_raw_snapshot": read_row["extra_info"]["source_line_1_based"] == snapshot["source"]["line_1_based"],
        "source_question_id_matches_raw_task": read_row["extra_info"]["source_question_id"] == raw["meta"]["question_id"],
        "source_difficulty_matches_raw_task": read_row["extra_info"]["source_difficulty"] == raw["meta"]["difficulty"],
        "index_matches_raw_zero_based_position": read_row["extra_info"]["index"] == snapshot["source"]["line_1_based"] - 1,
    }
    if not all(source_metadata_readback_validation.values()) or not actor_prompt_validation["actor_prompt_readback_matches"]:
        raise RuntimeError("Actor prompt or original Task source attributes failed JSONL readback validation.")
    assert json.loads(read_row["reward_model"]["ground_truth"])["functional"] == functional
    assert all(c["input_sha256"] == digest(canonical(c["input"])) for c in read_row["extra_info"]["case_index"])
    context = {"raw": raw, "cases": cases, "row": read_row, "functional": functional}
    summary = {"case_counts": cases["summary"], "source_files": sources,
               "json_roundtrip": True, "data_pool_readback": True,
               "actor_prompt_validation": actor_prompt_validation,
               "source_metadata_readback_validation": source_metadata_readback_validation,
               "ground_truth_sha256": digest(functional), "ground_truth_case_ids": [c["case_id"] for c in cases["accepted"]]}
    return context, summary


def verify(context):
    proposals = context["cases"]["accepted"] + context["cases"]["quarantined"]
    duplicate_ids = json.loads(canonical(proposals))
    duplicate_ids[1]["case_id"] = duplicate_ids[0]["case_id"]
    mismatched_hash = json.loads(canonical(proposals))
    mismatched_hash[0]["input_sha256"] = "0" * 64
    alignment_regressions = {}
    for name, payload, expected_message in (
        ("duplicate_case_id_rejected", duplicate_ids, "Case IDs must be nonempty strings and unique."),
        ("input_hash_mismatch_rejected", mismatched_hash, "Input hash mismatch for case_id=seed-1."),
    ):
        try:
            verify_cases(payload)
        except ValueError as error:
            assert str(error) == expected_message
            alignment_regressions[name] = True
        else:
            raise AssertionError(f"Missing alignment rejection: {name}")
    missing_example = dict(context["raw"])
    missing_example["examples"] = context["raw"]["examples"][:-1]
    try:
        extract_seed_cases(missing_example)
    except ValueError as error:
        assert str(error) == "Seed/assert and example counts differ; zip would silently truncate."
        alignment_regressions["seed_example_length_mismatch_rejected"] = True
    else:
        raise AssertionError("Missing seed/example length rejection")
    nums_list = [list(values) for size in range(1, 5)
                 for values in itertools.product(range(-2, 3), repeat=size)]
    exhaustive = [brute_force(nums)[0] for nums in nums_list]
    assert exhaustive == [kadane(nums) for nums in nums_list]
    assert exhaustive == collect_outputs(REFERENCE, nums_list)
    assert exhaustive == collect_outputs(CORRECT_PREFIX, nums_list)
    assert len(nums_list) == 780
    gt = context["functional"]
    regressions = {}
    for name, code, expected_success in (("reference", REFERENCE, True), ("correct_prefix", CORRECT_PREFIX, True),
                                          ("wrong_empty", WRONG_EMPTY, False), ("wrong_sum", WRONG_SUM, False)):
        source = code + "\n" + gt
        succ, _ = controlled_exec(source, {source})
        assert succ == expected_success
        regressions[name] = succ
    mutated = gt.replace("== 6, 'seed-1'", "== 999, 'seed-1'", 1)
    assert mutated != gt
    source = REFERENCE + "\n" + mutated
    mutation_succ, _ = controlled_exec(source, {source})
    assert not mutation_succ
    guard_empty = guard_quarantined = False
    try:
        generate_functional([])
    except ValueError:
        guard_empty = True
    try:
        generate_functional(context["cases"]["quarantined"])
    except ValueError:
        guard_quarantined = True
    assert guard_empty and guard_quarantined
    return {"exhaustive_small_domain": {"lengths": [1, 2, 3, 4], "element_domain": [-2, -1, 0, 1, 2],
                                        "count": 780, "bruteforce_kadane_reference_prefix_agree": True},
            "functional_gt_regressions": regressions, "intentionally_mutated_gt_rejected_by_reference": True,
            "empty_gt_guard": guard_empty, "quarantined_gt_guard": guard_quarantined,
            "data_alignment_regressions": alignment_regressions,
            "negative_controls_rejected_individually": [c["case_id"] for c in context["cases"]["quarantined"]]}


def load_original_scorer(allowed):
    """Compile only reviewed pure function definitions; do not import the module.

    This preserves parser and _compute_score logic while bypassing numpy imports,
    CODER1_EXEC selection and Linux/remote executor side effects.
    """
    names = {"remote_check_stdio", "validate_response_structure", "try_extract_solution",
             "extract_code_from_string", "_compute_score"}
    tree = ast.parse(SCORER.read_text(encoding="utf-8"))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == names
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {"re": re, "json": json, "time": time, "Tuple": Tuple, "Optional": Optional,
                 "ThreadPoolExecutor": ThreadPoolExecutor, "as_completed": as_completed,
                 "_MAX_CHAR_DISPLAY": 2048, "_ERROR_MSG_PREFIX": "Failed to execute program: ",
                 "CODE_PATTERN": re.compile(r"```(?:\w+)?\n(.*?)\n```", re.DOTALL),
                 "code_exec": lambda code, stdin=None: controlled_exec(code, allowed, stdin)}
    exec(compile(module, str(SCORER), "exec"), namespace)
    return namespace


def response(code):
    return "<think>Fixed teaching candidate; no model generation.</think><answer>```python\n" + code.rstrip() + "\n```</answer>"


def score(context):
    candidates = {"reference": REFERENCE, "correct_prefix": CORRECT_PREFIX,
                  "wrong_empty": WRONG_EMPTY, "wrong_sum": WRONG_SUM, "early_exit": EARLY_EXIT}
    functional = context["functional"]
    allowed = {code.rstrip() + "\n" + functional for code in candidates.values()}
    allowed.update((STDIO_CORRECT.rstrip(), STDIO_WRONG.rstrip()))
    scorer = load_original_scorer(allowed)
    parsed = scorer["extract_code_from_string"](response(REFERENCE))
    assert parsed == REFERENCE.rstrip()
    results = []
    accepted = context["cases"]["accepted"]
    inputs = [c["input"]["nums"] for c in accepted]
    for name in ("reference", "correct_prefix", "wrong_empty", "wrong_sum", "bad_format", "early_exit"):
        text = response(candidates[name]) if name != "bad_format" else "```python\n" + REFERENCE.rstrip() + "\n```"
        reward, log = scorer["_compute_score"](text, context["row"]["reward_model"]["ground_truth"],
                                               context["row"]["extra_info"])
        # Independent completion audit: return code alone cannot certify assertions.
        values = collect_outputs(candidates[name], inputs) if name != "bad_format" else None
        observations = [] if values is None else [
            {"case_id": c["case_id"], "input": c["input"], "input_sha256": c["input_sha256"],
             "expected": c["expected"], "actual": actual, "equal": actual == c["expected"]}
            for c, actual in zip(accepted, values)
        ]
        complete = values is not None and len(values) == len(accepted)
        correct = complete and all(o["equal"] for o in observations)
        traceback_matches = re.findall(r"AssertionError:\s*([^\r\n]+)", log)
        first_failed_id = traceback_matches[-1].strip() if traceback_matches else None
        if first_failed_id is not None:
            assert first_failed_id in {c["case_id"] for c in accepted}
        results.append({"candidate_id": name, "reward": reward, "scorer_success_signal": reward == 1.1,
                        "scorer_log": log, "source_first_failed_case_id_from_traceback": first_failed_id,
                        "correctness_pass": bool(correct), "audit_completed_case_count": len(observations),
                        "audit_completion_status": "completed" if complete else "no_audit_completion_payload",
                        "audit_observations": observations})
    assert [r["reward"] for r in results] == [1.1, 1.1, 0.1, 0.1, -1.1, 1.1]
    assert [r["correctness_pass"] for r in results] == [True, True, False, False, False, False]
    assert results[2]["source_first_failed_case_id_from_traceback"] == "diag-all-negative"
    assert results[3]["source_first_failed_case_id_from_traceback"] == "seed-1"
    stdio_records = [{"case_id": c["case_id"], "input": c["input"], "expected": c["expected"],
                      "stdin": canonical(c["input"]["nums"]) + "\n", "expected_stdout": str(c["expected"]) + "\n"}
                     for c in accepted]
    stdio_gt = canonical({"inputs": [c["stdin"] for c in stdio_records],
                          "outputs": [c["expected_stdout"] for c in stdio_records]})
    stdio_results = {}
    for name, code, expected_reward in (("correct_prefix", STDIO_CORRECT, 1.1), ("wrong_sum", STDIO_WRONG, 0.1)):
        reward, _ = scorer["_compute_score"](response(code), stdio_gt, context["row"]["extra_info"])
        assert reward == expected_reward
        pairs = []
        for record in stdio_records:
            succ, actual = controlled_exec(code.rstrip(), allowed, record["stdin"])
            pairs.append({**record, "actual_stdout": actual, "execution_success": succ,
                          "equal_after_strip": succ and actual.strip() == record["expected_stdout"].strip()})
        stdio_results[name] = {"reward": reward, "pairs": pairs}
    return {"parser_and_score": "Exact reviewed AST function bodies from pinned Code-R1; original _compute_score reused.",
            "backend_difference": "Fixed-candidate local Python subprocess replaces firejail/remote backend; no OS sandbox claim.",
            "candidates": results,
            "early_exit_protocol_gap": "sys.exit(0) exits before check(): source scorer gives 1.1 from returncode=0; a separate independent audit returns no completion payload and 0 observed cases. No protected completion protocol is established.",
            "stdio_adapter": {"provenance": "Teaching adaptation of this function task; not original TACO or original LeetCode STDIO data.",
                              "comparison": "Original scorer requires execution success and actual_stdout.strip() == expected_stdout.strip().",
                              "results": stdio_results}}


def normalized(scores, epsilon=1e-6):
    mean = statistics.mean(scores) if len(scores) > 1 else 0.0
    std = statistics.stdev(scores) if len(scores) > 1 else 1.0
    return mean, std, [(score - mean) / (std + epsilon) for score in scores]


def advantage(scoring):
    primary = scoring["candidates"][:4]
    rewards = [c["reward"] for c in primary]
    mean, std, scalars = normalized(rewards)
    lengths, width, uid = [7, 5, 6, 4], 8, "maximum-subarray:diagnostic-v1"
    records = []
    for candidate, length, scalar in zip(primary, lengths, scalars):
        mask = [int(position < length) for position in range(width)]
        token_rewards = [candidate["reward"] if position == length - 1 else 0.0 for position in range(width)]
        assert sum(token_rewards) == candidate["reward"]
        records.append({"candidate_id": candidate["candidate_id"], "uid": uid, "reward": candidate["reward"],
                        "valid_response_length": length, "terminal_position_0_based": length - 1,
                        "eos_mask": mask, "token_level_rewards": token_rewards,
                        "group_scalar_advantage": scalar, "token_advantages": [scalar * v for v in mask]})
    order = [2, 0, 3, 1]
    shuffled = [records[i] for i in order]
    grouped = {}
    for record in shuffled:
        grouped.setdefault(record["uid"], []).append(record)
    for group in grouped.values():
        _, _, recomputed = normalized([r["reward"] for r in group])
        assert all(math.isclose(value, record["group_scalar_advantage"], abs_tol=1e-12)
                   for value, record in zip(recomputed, group))
    infrastructure = [0.1, 0.1, 0.1, 0.0]
    infra_mean, infra_std, infra_scalars = normalized(infrastructure)
    assert all(math.isclose(a, b, abs_tol=4e-5) for a, b in zip(infra_scalars, [0.5, 0.5, 0.5, -1.5]))
    assert normalized([0.1, 0.1])[2] == [0.0, 0.0]
    return {"implementation": "Standard-library translation of compute_grpo_outcome_advantage; sample std correction=1, epsilon=1e-6; not a torch execution or bitwise float32 match.",
            "position_and_mask": "Illustrative lengths and masks; not tokenizer output. Source places reward at valid_response_length-1.",
            "rewards": rewards, "mean": mean, "sample_std": std, "scalar_advantages": scalars,
            "rows": records, "uid_reorder": {"order": order, "candidate_ids": [r["candidate_id"] for r in shuffled],
                                             "advantages": [r["group_scalar_advantage"] for r in shuffled],
                                             "grouping_preserved": True},
            "infrastructure_fill_simulation": {"origin": "Teaching counterfactual, not an observed timeout/failure or sampled reward.",
                                               "rewards": infrastructure, "mean": infra_mean, "sample_std": infra_std,
                                               "scalar_advantages": infra_scalars, "rounded_without_epsilon": [0.5, 0.5, 0.5, -1.5]},
            "edge_cases": {"equal_reward_group_advantages": normalized([0.1, 0.1])[2],
                           "singleton_source_convention": normalized([0.4])[2]}}


def evaluate(scoring):
    def summarize(records):
        total = len(records)
        correct = sum(r["correctness_pass"] for r in records)
        return {"n": total, "correct": correct, "correctness_rate": correct / total,
                "mean_reward": statistics.mean(r["reward"] for r in records),
                "scorer_success_signal_rate": sum(r["scorer_success_signal"] for r in records) / total,
                "pass_at_k": {str(k): 1.0 - (math.comb(total - correct, k) / math.comb(total, k)
                                               if k <= total - correct else 0.0) for k in range(1, total + 1)}}
    primary = summarize(scoring["candidates"][:4])
    assert primary["correctness_rate"] == 0.5 and math.isclose(primary["mean_reward"], 0.6)
    assert math.isclose(primary["pass_at_k"]["2"], 5 / 6)
    return {"scope": "One task / selected diagnostic GT / four fixed algorithm candidates. Not sampled model outputs, unseen-task accuracy, or model generalization.",
            "pass_at_k_definition": "1-C(n-c,k)/C(n,k): uniform subset without replacement from this controlled finite candidate pool; not empirical model pass@k.",
            "primary_four_algorithm_candidates": primary,
            "all_six_including_protocol_controls": summarize(scoring["candidates"]),
            "correctness_definition": "Separate independent audit of fixed candidates observes all 8 accepted input/expected pairs, complete and equal. This is finite-case evidence, not a source-scorer completed-test count or a protected completion protocol. Early exit and bad format fail.",
            "excluded_from_generalization_claims": ["same GT used for scorer demonstration and evaluation", "fixed candidate selection", "one public problem", "no held-out model rollouts"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = ["prepare", "verify", "score", "advantage", "evaluate"]
    parser.add_argument("stage", choices=["all"] + stages, help="Run selected stage and all prerequisites; outputs are regenerated.")
    args = parser.parse_args()
    source_fingerprints_before = readonly_source_fingerprints()
    context, prepared = prepare()
    report = {"task_id": "maximum-subarray", "test_version": TEST_VERSION,
              "source_revisions": {"code_r1": CODE_REV, "dataset": DATA_REV},
              "requested_stage": args.stage, "completed_stages": ["prepare"], "prepare": prepared,
              "meta": {
                  "source_scorer_evidence": "Candidate reward, scorer_success_signal and scorer_log come from the original _compute_score run. It reports no protected completed-case count.",
                  "source_traceback_evidence": "source_first_failed_case_id_from_traceback identifies the raised assertion from the source scorer log; it is not a measured completion count.",
                  "independent_audit_evidence": "audit_completed_case_count, audit_completion_status and audit_observations come from a second collect_outputs execution, not from the source scorer's GT execution.",
                  "correctness_pass": "Finite-case result of the separate independent audit over the 8 accepted cases; not proof of a trusted completion protocol or unknown-code safety.",
                  "data_alignment": "Duplicate case IDs, input/hash mismatches and unequal seed/example counts fail before reference execution; checks validate data only.",
              },
              "limits": ["Local fixed-candidate subprocess is not a secure sandbox.",
                         "No installation, network, model, GPU, tokenizer, PPO optimizer or real training is used.",
                         "8 diagnostic cases and 780 small arrays do not prove correctness for every legal input.",
                         "Infrastructure-failure advantages and response lengths/masks are teaching simulations.",
                         "Pass@k is a finite controlled-pool calculation, not a measured model metric.",
                         "Pinned source hashes stop on drift; JSONL is a simulated row readback, not original Parquet preprocessing."]}
    stop = len(stages) - 1 if args.stage == "all" else stages.index(args.stage)
    if stop >= 1:
        report["verify"] = verify(context)
        report["completed_stages"].append("verify")
    if stop >= 2:
        report["score"] = score(context)
        report["completed_stages"].append("score")
    if stop >= 3:
        report["advantage"] = advantage(report["score"])
        report["completed_stages"].append("advantage")
    if stop >= 4:
        report["evaluate"] = evaluate(report["score"])
        report["completed_stages"].append("evaluate")
    assert readonly_source_fingerprints() == source_fingerprints_before
    report["source_protection"] = {"raw_dataset_and_original_sources_byte_hashes_unchanged": True,
                                   "sha256": source_fingerprints_before,
                                   "fixed_output_directory": str(OUTPUTS),
                                   "resolved_output_path_guard": True}
    dump("report.json", report)
    print("completed: " + ", ".join(report["completed_stages"]))
    print("cases: 8 accepted (3 seed + 5 diagnostic); 7 controls quarantined individually")
    if "verify" in report:
        print("independent small-domain agreement: 780 / 780; functional regressions and GT mutation control passed")
    if "score" in report:
        print("rewards: " + repr([r["reward"] for r in report["score"]["candidates"]]))
        print("early_exit: source scorer reward 1.1; separate audit observed cases 0, correctness false")
    if "advantage" in report:
        print("advantages: " + repr(report["advantage"]["scalar_advantages"]))
        print("simulated infra-fill advantages: " + repr(report["advantage"]["infrastructure_fill_simulation"]["scalar_advantages"]))
    if "evaluate" in report:
        print("controlled four-candidate evaluation: " + canonical(report["evaluate"]["primary_four_algorithm_candidates"]))
    print("outputs: " + str(OUTPUTS))


if __name__ == "__main__":
    main()
