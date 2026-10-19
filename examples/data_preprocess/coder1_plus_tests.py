"""Code-R1 测试质量工具：coder1_plus 数据侧的补强项 2 和 3。

原版 Code-R1 没有任何判断测试够不够、或者补充测试的代码。学习资料主框架 A3 到 A5、01 作业 1.2 讲了做法，
这里把它们写成可以用在任意 LeetCode 函数题上的工具。

补强项 2：测试强度审计
    对参考解做单点的小改动，比如把 < 改成 <=、把 max 换成 min、把小常数加一，得到一批变异体，
    再看现有测试能不能把它们判失败。被判失败的比例叫杀死率。
    存活的变异体只是「可疑」：可能测试太弱，也可能这次改动恰好没改变行为，即等价变异体。
    要找到一个输入，让它的输出和可信答案不同，才能确认它是错解。

补强项 3：测试补充与逐条审核
    对一批候选输入，先用校验器检查输入合法，再用参考解算候选答案，用独立的 oracle 核对：
    一致才接受，不一致隔离，没有 oracle 的保持待审核。只有接受的用例才追加进 GT，测试版本随之变化。
    这把 LeetCode 53 演示里的逐条审核推广到一般的函数题。

两项工具都通过参数接收执行器 executor(code) -> (成功与否, 标准输出)，与 coder1.code_exec 的约定相同。
训练环境用 coder1 的执行后端；本机验证见 RLVR学习资料/examples/source-supplements/。
"""

import ast
import copy
import hashlib
import json
import random
import re

# ---------------------------------------------------------------------------
# 与判分侧保持一致的执行契约
# ---------------------------------------------------------------------------


def build_program(prelude: str, code: str, functional: str) -> str:
    """与 verl/utils/reward_score/coder1_plus.build_program 相同：前缀、候选代码、测试。"""
    return prelude.rstrip() + "\n\n" + code + "\n\n" + functional


def entry_point_of(functional: str) -> str:
    match = re.search(r"check\((.+)\)\s*$", functional.strip())
    if not match:
        raise ValueError("cannot find the final check(<entry_point>) call in the ground truth")
    return match.group(1)


def count_asserts(functional: str) -> int:
    return len(re.findall(r"^\s*assert\b", functional, flags=re.M))


# ---------------------------------------------------------------------------
# 补强项 2：测试强度审计
# ---------------------------------------------------------------------------

_CMP_SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_BIN_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add, ast.FloorDiv: ast.Mult}
_BOOL_SWAP = {ast.And: ast.Or, ast.Or: ast.And}
_CALL_SWAP = {"min": "max", "max": "min"}
_OP_TEXT = {
    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!=",
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.FloorDiv: "//", ast.And: "and", ast.Or: "or",
}


def _mutation_sites(tree):
    """按固定顺序列出所有可变异的位置；同一份代码每次得到相同的列表。"""
    sites = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            sites.extend(("compare", node, i) for i, op in enumerate(node.ops) if type(op) in _CMP_SWAP)
        elif isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in _BIN_SWAP:
            sites.append(("arith", node, None))
        elif isinstance(node, ast.BoolOp) and type(node.op) in _BOOL_SWAP:
            sites.append(("bool", node, None))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _CALL_SWAP:
            sites.append(("minmax", node, None))
        elif isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 2:
            sites.extend((("const+1", node, None), ("const-1", node, None)))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            sites.append(("drop-not", node, None))
    return sites


def _apply_mutation(kind, node, extra) -> str:
    """原地改一个位置，返回人能读懂的改动说明。"""
    line = getattr(node, "lineno", "?")
    if kind == "compare":
        old = type(node.ops[extra])
        node.ops[extra] = _CMP_SWAP[old]()
        return f"line {line}: {_OP_TEXT[old]} -> {_OP_TEXT[type(node.ops[extra])]}"
    if kind in ("arith", "bool"):
        table = _BIN_SWAP if kind == "arith" else _BOOL_SWAP
        old = type(node.op)
        node.op = table[old]()
        return f"line {line}: {_OP_TEXT[old]} -> {_OP_TEXT[type(node.op)]}"
    if kind == "minmax":
        old = node.func.id
        node.func.id = _CALL_SWAP[old]
        return f"line {line}: {old}() -> {node.func.id}()"
    if kind in ("const+1", "const-1"):
        old = node.value
        node.value = old + (1 if kind == "const+1" else -1)
        return f"line {line}: {old} -> {node.value}"
    if kind == "drop-not":
        node.operand = ast.UnaryOp(op=ast.Not(), operand=node.operand)
        return f"line {line}: not x -> bool(x)"
    raise ValueError(f"unknown mutation kind: {kind}")


def generate_mutants(code: str, max_mutants: int = 10, seed: int = 0):
    """返回 [(改动说明, 变异体代码)]。每个变异体只改一处；改完文本不变的跳过；同一 seed 结果固定。"""
    original = ast.unparse(ast.parse(code))
    order = list(range(len(_mutation_sites(ast.parse(code)))))
    random.Random(seed).shuffle(order)
    mutants = []
    for index in order:
        tree = ast.parse(code)
        kind, node, extra = _mutation_sites(tree)[index]
        description = _apply_mutation(kind, node, extra)
        mutated = ast.unparse(tree)
        if mutated != original:
            mutants.append((description, mutated))
        if len(mutants) >= max_mutants:
            break
    return mutants


def audit_test_strength(row, executor, max_mutants: int = 10, seed: int = 0):
    """对一行 coder1_plus 数据做变异审计，返回这道题的杀死率与存活的变异体。"""
    gt = json.loads(row["reward_model"]["ground_truth"])
    reference = row["extra_info"]["reference"]
    record = {
        "task_id": row["extra_info"].get("task_id"),
        "split": row["extra_info"].get("split"),
        "n_tests": count_asserts(gt["functional"]),
        "baseline_ok": None,
        "mutants": 0,
        "killed": 0,
        "kill_rate": None,
        "survivors": [],
    }
    record["baseline_ok"] = executor(build_program(gt["prelude"], reference, gt["functional"]))[0]
    if not record["baseline_ok"]:
        return record  # 参考解自己都过不了，审计没有意义；原因见补强项 1 的分类
    for description, mutated in generate_mutants(reference, max_mutants, seed):
        record["mutants"] += 1
        passed, _ = executor(build_program(gt["prelude"], mutated, gt["functional"]))
        if passed:
            record["survivors"].append({"change": description, "code": mutated})
        else:
            record["killed"] += 1
    if record["mutants"]:
        record["kill_rate"] = record["killed"] / record["mutants"]
    return record


# ---------------------------------------------------------------------------
# 补强项 3：测试补充与逐条审核
# ---------------------------------------------------------------------------


def input_sha256(kwargs) -> str:
    return hashlib.sha256(json.dumps(kwargs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _json_normalize(value):
    return json.loads(json.dumps(value))


def reference_outputs(row, inputs, executor):
    """在一个子进程里用参考解算出每个输入的候选答案。每次调用都新建 Solution 实例。"""
    gt = json.loads(row["reward_model"]["ground_truth"])
    entry = entry_point_of(gt["functional"])
    harness = (
        "\n\nimport json as _json\n"
        f"_inputs = _json.loads({json.dumps(json.dumps(inputs))})\n"
        "for _kwargs in _inputs:\n"
        "    try:\n"
        f"        _out = ({entry})(**_kwargs)\n"
        "    except Exception as _error:\n"
        "        print(_json.dumps({'ok': False, 'error': type(_error).__name__}))\n"
        "        continue\n"
        "    try:\n"
        "        print(_json.dumps({'ok': True, 'out': _out}))\n"
        "    except (TypeError, ValueError):\n"
        "        print(_json.dumps({'ok': False, 'error': 'unserializable_output'}))\n")
    ok, output = executor(build_program(gt["prelude"], row["extra_info"]["reference"], harness))
    if not ok:
        raise RuntimeError(f"reference harness failed for {row['extra_info'].get('task_id')}: {output[-500:]}")
    lines = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    if len(lines) != len(inputs):
        raise RuntimeError(f"expected {len(inputs)} reference outputs, got {len(lines)}")
    return lines


def audit_new_cases(row, candidates, executor, oracle=None, validator=None, origin="generated"):
    """逐条审核候选输入，返回 case 记录列表。

    状态含义：invalid 输入不合法；quarantined 参考解报错或与 oracle 不一致；
    pending 没有独立 oracle，候选答案只来自参考解；accepted 参考解与 oracle 一致。
    """
    records, seen = [], set()
    for kwargs in candidates:
        digest = input_sha256(kwargs)
        if digest in seen:
            continue
        seen.add(digest)
        record = {"case_id": f"aug-{digest[:10]}", "input": kwargs, "expected": None, "origin": origin,
                  "input_sha256": digest, "status": None, "evidence": {}}
        if validator is not None and not validator(**copy.deepcopy(kwargs)):
            record["status"] = "invalid"
            record["evidence"]["reason"] = "violates the input constraints"
        records.append(record)

    runnable = [record for record in records if record["status"] is None]
    if runnable:
        for record, result in zip(runnable, reference_outputs(row, [r["input"] for r in runnable], executor)):
            if not result.get("ok"):
                record["status"] = "quarantined"
                record["evidence"]["reference_error"] = result.get("error")
                continue
            record["expected"] = result["out"]
            record["evidence"]["reference"] = result["out"]
            if oracle is None:
                record["status"] = "pending"
                record["evidence"]["reason"] = "no independent oracle; the reference alone is not trusted"
                continue
            independent = _json_normalize(oracle(**copy.deepcopy(record["input"])))
            record["evidence"]["oracle"] = independent
            record["status"] = "accepted" if independent == record["expected"] else "quarantined"
    return records


def extend_test(test: str, cases) -> str:
    """把接受的用例追加进 def check(candidate) 的断言里；断言消息带 case_id，方便定位失败。"""
    lines = [test.rstrip("\n")]
    for case in cases:
        arguments = ", ".join(f"{name}={value!r}" for name, value in case["input"].items())
        lines.append(f"    assert candidate({arguments}) == {case['expected']!r}, {case['case_id']!r}")
    return "\n".join(lines) + "\n"
